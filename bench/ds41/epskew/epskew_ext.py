"""Worker extension (stock `--worker-extension-cls r9k_epskew.EpSkew`) for package H3's expert-parallel skew study
(notes/exl3-ep-skew.md): records the routing every EXL3 MoE layer ran with at decode, step by step, into a device ring
buffer, HIP-graph safe. Diagnostic only; nothing in the product imports it. Mounted into the container as
`r9k_epskew` (bench/ds41/epskew/run_skew.sh), reached through the stock dev endpoint POST /collective_rpc
(VLLM_SERVER_DEV_MODE=1); client bench/ds41/epskew/run_skew.py.

Mechanics: importing this module (the worker resolves the extension class at init, before the model is built) wraps
Exl3MoEMethod.apply. Every call with T <= 24 tokens (decode / verify / drafter; prefill chunks are not recorded) writes
its topk ids (global, int16, padded with -1 to 144 = 24 x 6 slots) into ring[layer, ptr[layer] % N] and advances
ptr[layer], all as device ops, so a captured graph replays the recording. Every rank records (symmetric extra work: a
rank-0-only recorder would itself make rank 0 late at every post-MoE all-reduce); rank 0 dumps.
R9K_EPSKEW_RECORD=0 installs nothing (the maps RPC only). R9K_EPSKEW_N (default 4096) steps a layer; 64 layers x N x 144 x 2 bytes (75 MiB at 4096).
"""
from __future__ import annotations

import os
import time

import torch

N = int(os.environ.get("R9K_EPSKEW_N", 4096))
SLOTS = 144
MAXL = 64
_st: dict = {}      # device index -> {"ring", "ptr", "names": {layer_name: idx}}


def _state(dev: torch.device):
    s = _st.get(dev.index)
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("r9k_epskew: first record under capture (the warm-up should have allocated)")
        s = _st[dev.index] = {"ring": torch.full((MAXL, N, SLOTS), -1, dtype=torch.int16, device=dev),
                              "ptr": torch.zeros((MAXL,), dtype=torch.int64, device=dev),
                              "one": torch.ones((1,), dtype=torch.int64, device=dev), "names": {}, "G": {}}
    return s


def _record(layer, topk_ids: torch.Tensor):
    T = topk_ids.shape[0]
    S = topk_ids.numel()
    if T == 0 or T > 24 or S > SLOTS:
        return
    s = _state(topk_ids.device)
    name = getattr(layer, "layer_name", None) or str(id(layer))
    li = s["names"].get(name)
    if li is None:
        if torch.cuda.is_current_stream_capturing():
            return       # never seen before capture: skip rather than allocate
        li = s["names"][name] = len(s["names"])
        s["G"][name] = int(getattr(layer, "global_num_experts", 0) or 0)
        assert li < MAXL
    row = torch.nn.functional.pad(topk_ids.reshape(1, S).to(torch.int16), (0, SLOTS - S), value=-1)
    p = s["ptr"][li:li + 1]
    s["ring"][li].index_copy_(0, torch.remainder(p, N), row)
    p.add_(s["one"])


def _install():
    from r9700_vllm.quant import exl3_moe as M
    cls = M.Exl3MoEMethod
    if getattr(cls, "_r9k_epskew", False):
        return
    orig = cls.apply

    def apply(self, layer, x, topk_weights, topk_ids, *a, **k):
        _record(layer, topk_ids)
        return orig(self, layer, x, topk_weights, topk_ids, *a, **k)

    cls.apply = apply
    cls._r9k_epskew = True


if os.environ.get("R9K_EPSKEW_RECORD", "1") == "1":   # 0: only the RPCs (r9k_ep_maps), no recording
    _install()


class EpSkew:
    """Mixed into the vLLM Worker. Methods are called by name through POST /collective_rpc."""

    def r9k_ep_reset(self):
        torch.cuda.synchronize()
        for s in _st.values():
            s["ring"].fill_(-1)
            s["ptr"].zero_()
        torch.cuda.synchronize()
        return {"rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, "t": time.time()}

    def r9k_ep_dump(self, path: str = "/h3out/ring.pt"):
        torch.cuda.synchronize()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        res = {"rank": rank, "t": time.time()}
        if rank != 0:
            return res
        out = {}
        for dev, s in _st.items():
            n = len(s["names"])
            out = {"ring": s["ring"][:n].cpu(), "ptr": s["ptr"][:n].cpu(), "names": list(s["names"]),
                   "G": [s["G"][k] for k in s["names"]], "N": N}
        if out:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(out, path)
            res.update(path=path, layers=len(out["names"]), ptr=out["ptr"].tolist())
        return res

    def r9k_ep_maps(self):
        """Every EXL3 MoE layer's global -> local map on this rank (checks a placement map took)."""
        mr = self.model_runner
        res = {"rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0, "maps": {}}
        mods = [("model", mr.model)]
        d = getattr(mr, "drafter", None) or getattr(mr, "speculator", None)
        if d is not None and hasattr(d, "model"):
            mods.append(("drafter", d.model))
        for tag, m in mods:
            for name, mod in m.named_modules():
                g = getattr(mod, "exl3_g2l", None)
                if g is not None:
                    res["maps"][f"{tag}:{name}"] = [i for i, v in enumerate(g.tolist()) if v < mod._exl3.E]
        return res
