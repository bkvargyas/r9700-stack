#!/usr/bin/env python3
"""Folded-exponent MXFP4 survey over a served checkpoint: the guard's d = (row reference exponent - block exponent)
distribution per weight tensor (fold_stats / fold_ok, exactly what the loader logs with R9K_FOLD=1), and, with
--gemm, the output error of the folded prefill / MT kernels against the exact dequantized reference on the real
weights (random bf16 activations), next to the exact kernels' error.

  fold_survey.py --model /models/Qwen3.8-Flash-Next-MXFP4-FP8 --kind mxfp4 [--layers 0,7,23,47] [--gemm]
  fold_survey.py --model /models/Qwen3.8-27B-NVFP4 --kind nvfp4 [--layers 0,10,30,55] [--gemm]
     (nvfp4: the MLPs are converted to MXFP4 the way R9K_NVFP4=mxfp4 serves them, per-rank TP2 shapes:
      gate|up merged N=17408 K=5120, down N=5120 with K=8704 split in two halves)
Run inside the ROCm image with the checkpoints mounted.
"""
import argparse
import dataclasses
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open

from r9700_vllm.kernels import moe as K

LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def mx_deq(p, e):
    p, e = p.cuda(), e.cuda()
    N, Kh = p.shape
    c = torch.stack((p & 15, p >> 4), -1).reshape(N, Kh * 2).long()
    return (LUT.cuda()[c].reshape(N, -1, 32) * torch.exp2(e.float() - 127).unsqueeze(-1)).reshape(N, Kh * 2)


class Ckpt:
    def __init__(self, path):
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            self.map = None
        self._open = {}

    def get(self, name):
        f = self.map[name] if self.map else "model.safetensors"
        if f not in self._open:
            self._open[f] = safe_open(os.path.join(self.path, f), "pt", device="cpu")
        return self._open[f].get_tensor(name)

    def names(self):
        return list(self.map) if self.map else list(safe_open(os.path.join(self.path, "model.safetensors"), "pt").keys())


def gemm_error(name, packed, e8, Ms=(512,), routed=None):
    """packed [N, K/2] u8, e8 [N, K/32] u8 (CPU): folded vs exact kernels vs dequant reference."""
    N, Kd = packed.shape[0], packed.shape[1] * 2
    W = K.prepare_mxfp4_weights(packed.cuda()[None], e8.cuda()[None])
    Wf = dataclasses.replace(W, fold=True)
    wd = mx_deq(packed, e8)
    g = torch.Generator(device="cuda").manual_seed(1)
    out = []
    for M in Ms:
        x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
        q, s = K.quant_rows_fp8(x)
        ref = (q.float() * s[:, None]) @ wd.T
        cfg = K.pick_cfg(N, Kd, 32, M=M, kind="mxfp4")
        res = {}
        if K.is_prefill_cfg(cfg):
            blk = K.prefill_block(cfg[1])
            mpad = (M + blk - 1) // blk * blk
            t = (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // blk, dtype=torch.int32, device="cuda"),
                 torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
            for lab, Wk in (("exact", W), ("fold", Wf)):
                o = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                K.moe_gemm(q, s, Wk, o, *t, M, 1, None, num_experts=1, prefill=cfg[1])
                res[lab] = ((o.float() - ref).norm() / ref.norm()).item()
        MT = 4 if M >= 64 else (2 if M >= 32 else 1)
        mpad = (M + 16 * MT - 1) // (16 * MT) * (16 * MT)
        t = (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // (16 * MT), dtype=torch.int32, device="cuda"),
             torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
        for lab, Wk in (("exactMT", W), ("foldMT", Wf)):
            o = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            K.moe_gemm(q, s, Wk, o, *t, M, 1, None, 2, 4, 2, num_experts=1, MT=MT, ldsa=MT > 1 and Kd % 256 == 0)
            res[lab] = ((o.float() - ref).norm() / ref.norm()).item()
        out.append((M, res))
        print(f"    gemm {name} {N}x{Kd} M={M}: " + " ".join(f"{k} {v:.2e}" for k, v in res.items()), flush=True)
    del W, Wf, wd
    torch.cuda.empty_cache()
    return out


def survey_tensor(name, e8, packed=None, gemm=False, Ms=(512,), stats_only_rows=None):
    """e8 [.., N, K/32] u8 checkpoint order (E leading dim optional)."""
    if e8.dim() == 2:
        e8 = e8[None]
    st = K.fold_stats(K.pack_scales(e8))
    print(f"  {name:60s} {st} -> {'FOLD' if K.fold_ok(st) else 'exact: ' + K.fold_why(st)}", flush=True)
    res = {"stats": dataclasses.asdict(st), "p_inexact": st.p_inexact, "p_flush": st.p_flush, "fold_ok": K.fold_ok(st)}
    if gemm and packed is not None:
        if packed.dim() == 3:               # experts: a few of them, each as a dense GEMM
            res["gemm"] = {int(ex): gemm_error(f"{name}[e{ex}]", packed[ex], e8[ex], Ms) for ex in (0, 1, 2, 3)}
        else:
            res["gemm"] = gemm_error(name, packed, e8[0], Ms)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=("mxfp4", "nvfp4"), required=True)
    ap.add_argument("--layers", default="", help="comma list; default all")
    ap.add_argument("--gemm", action="store_true", help="also run the kernels on the real weights (GPU)")
    ap.add_argument("--gemm-layers", default="", help="layers whose weights go through the kernels (default: first two)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ck = Ckpt(a.model)
    names = ck.names()
    pre = "model.language_model.layers."
    layers = sorted({int(n[len(pre):].split(".")[0]) for n in names if n.startswith(pre)})
    if a.layers:
        layers = [int(v) for v in a.layers.split(",")]
    gl = {int(v) for v in a.gemm_layers.split(",")} if a.gemm_layers else set(layers[:2])
    out = {}
    if a.kind == "mxfp4":
        for L in layers:
            p = f"{pre}{L}.mlp."
            gemm = a.gemm and L in gl
            for t in ("experts.gate_up_proj", "experts.down_proj"):
                if f"{p}{t}_scale" not in names:
                    continue
                e8, pk = ck.get(f"{p}{t}_scale"), ck.get(f"{p}{t}_packed") if gemm else None
                out[f"L{L}.{t}"] = survey_tensor(f"L{L}.{t}", e8, pk, gemm)
            for t in ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj"):
                if f"{p}{t}.weight_scale" not in names:
                    continue
                e8, pk = ck.get(f"{p}{t}.weight_scale"), ck.get(f"{p}{t}.weight_packed") if gemm else None
                out[f"L{L}.{t}"] = survey_tensor(f"L{L}.{t}", e8, pk, gemm)
    else:
        from r9700_vllm.quant.nvfp4 import nvfp4_to_mxfp4
        for L in layers:
            p = f"{pre}{L}.mlp."
            if f"{p}gate_proj.weight_scale" not in names:
                continue            # fp8 MLP layers (56..63 on the 27B)
            gemm = a.gemm and L in gl
            # per-rank TP2: gate|up rows split in half each and merged -> [17408, 5120]; down K split in half
            parts = []
            for t in ("gate_proj", "up_proj"):
                pk, s16, gs = ck.get(f"{p}{t}.weight_packed"), ck.get(f"{p}{t}.weight_scale"), ck.get(f"{p}{t}.weight_global_scale")
                N = pk.shape[0]
                pk, s16 = pk[:N // 2].cuda(), s16[:N // 2].cuda()
                pm, em = nvfp4_to_mxfp4(pk, s16, gs.float().cuda().expand(N // 2).contiguous())
                parts.append((pm.cpu(), em.cpu()))
            pm, em = torch.cat([q for q, _ in parts]), torch.cat([e for _, e in parts])
            out[f"L{L}.gate_up"] = survey_tensor(f"L{L}.gate_up(rank0)", em, pm, gemm)
            pk, s16, gs = ck.get(f"{p}down_proj.weight_packed"), ck.get(f"{p}down_proj.weight_scale"), ck.get(f"{p}down_proj.weight_global_scale")
            N, Kh = pk.shape
            pk, s16 = pk[:, :Kh // 2].cuda().contiguous(), s16[:, :s16.shape[1] // 2].cuda().contiguous()
            pm, em = nvfp4_to_mxfp4(pk, s16, gs.float().cuda().expand(N).contiguous())
            out[f"L{L}.down"] = survey_tensor(f"L{L}.down(rank0 K-half)", em.cpu(), pm.cpu(), gemm)
            torch.cuda.empty_cache()
    tot_in = sum(v["stats"]["hist"][i] for v in out.values() for i in range(9, 17))
    tot_fl = sum(v["stats"]["hist"][i] for v in out.values() for i in range(13, 17))
    tot = sum(v["stats"]["blocks"] for v in out.values())
    n_ok = sum(1 for v in out.values() if v["fold_ok"])
    print(f"SUMMARY {a.model}: {len(out)} tensors, {n_ok} pass the guard; blocks {tot}: "
          f"inexact(d>8) {100 * tot_in / tot:.4f}%  flushed(d>12) {100 * tot_fl / tot:.5f}%; "
          f"worst tensor inexact {100 * max(v['p_inexact'] for v in out.values()):.3f}% "
          f"flush {100 * max(v['p_flush'] for v in out.values()):.4f}%")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
