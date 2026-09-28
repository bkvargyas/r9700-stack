"""Our fused QSA-indexer glue (kernels/r9k_norm_rope.hip, attn/indexer_glue.py) vs the stock functions it replaces:
GemmaRMSNorm (ir.ops.rms_norm native path) and ApplyRotaryEmb.forward_static (neox, bf16 cos/sin).

Single GPU:  python3 tests/test_indexer_glue_r9k.py            (BENCH=0 skips the timing table)

Bit-equality is the bar: the only admitted difference is the fp32 order of the 128-element sum of squares, which can
flip a bf16 rounding on rare rows (counted; must be < 1e-3 of the elements). Shapes: Flash-Next indexer D=128,
rotary_dim 32, 4 q heads at 1 / 5 / 64 / 1000 tokens with a strided q view; norm-only (k path) rows; D=64/256.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.layernorm import GemmaRMSNorm  # noqa: E402
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb  # noqa: E402
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope  # noqa: E402

from r9700_vllm.attn import indexer_glue as G  # noqa: E402

dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0
MAXPOS = 4096


def rnd(*shape, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).to(dev)


def make_norm(D):
    """As served: built under the model dtype (bf16 weight); CustomOp construction needs a config context."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(VllmConfig()):
            n = GemmaRMSNorm(D, eps=1e-6).to(dev)
    finally:
        torch.set_default_dtype(prev)
    assert n.weight.dtype == torch.bfloat16
    with torch.no_grad():
        n.weight.copy_((torch.randn(D, generator=g) * 0.3).to(torch.bfloat16))
    return n


def stock_rope(x, cs, pos, rd):
    """apply_qsa_rope's 1-D branch: forward_static on the first rd dims, cat with the rest."""
    cos, sin = cs[pos].chunk(2, dim=-1)
    rot = ApplyRotaryEmb.forward_static(x[..., :rd], cos, sin, True, False)
    return torch.cat((rot, x[..., rd:]), dim=-1)


def stock_mrope(x, cs, pos, rd, sec, interleaved):
    """apply_qsa_rope's 2-D branch: vLLM's Triton MRoPE kernel on [T, H*D] with cos/sin [3, T, rd/2]."""
    T, H, D = x.shape
    cos, sin = cs[pos].chunk(2, dim=-1)
    out, _ = triton_mrope(x.reshape(T, -1).clone(), x.new_empty((T, D)), cos.contiguous(), sin.contiguous(), list(sec),
                          D, rd, interleaved, True)
    return out.reshape(T, H, D)


def check(name, T, H, D, rd, strided=True, rope=True, mrope=None, same_pos=False, pos_stride=1):
    """mrope: (section, interleaved) -> [3, T] positions through the Triton MRoPE reference. pos_stride > 1 hands the
    kernel a strided column view of a wider int64 buffer (serving's 1-D positions are a [T] view of [T, 3], stride 3;
    for MRoPE the [3, T] rows get token stride pos_stride)."""
    global bad
    norm = make_norm(D)
    wide = rnd(T, (H + 2) * D) if strided else rnd(T, H * D)
    x = wide[:, : H * D].view(T, H, D)
    cs = (torch.rand((MAXPOS, rd), generator=g) * 2 - 1).to(torch.bfloat16).to(dev)
    if mrope is None:
        pos = torch.randint(0, MAXPOS, (T, pos_stride), generator=g).to(dev)[:, 0]
    elif same_pos:
        pos = torch.randint(0, MAXPOS, (1, T), generator=g).expand(3, T).contiguous().to(dev)
    else:
        pos = torch.randint(0, MAXPOS, (3, T, pos_stride), generator=g).to(dev)[:, :, 0]
    assert pos.stride(-1) == pos_stride
    with torch.no_grad():
        s_n = norm.forward_native(x.reshape(-1, D)).reshape(T, H, D)
        o_n = G.gemma_norm(x, norm.weight, norm.variance_epsilon)
        if not rope:
            s, o = s_n, o_n
        elif mrope is None:
            s = stock_rope(s_n, cs, pos, rd)
            o = G.gemma_norm_rope(x, norm.weight, norm.variance_epsilon, cs, pos, rd, 0, 0, 0, False)
        else:
            sec, il = mrope
            s = stock_mrope(s_n, cs, pos, rd, sec, il)
            o = G.gemma_norm_rope(x, norm.weight, norm.variance_epsilon, cs, pos, rd, *sec, il)
    torch.cuda.synchronize()
    nf = (o_n != s_n).sum().item()
    rf = (o != s).sum().item()
    ok = o.shape == s.shape and o.dtype == s.dtype and torch.isfinite(o).all() and nf <= s_n.numel() * 1e-3 \
        and rf <= s.numel() * 1e-3
    bad += not ok
    print(f"  {name:<40} norm flips {nf}/{s_n.numel()}  {'rope' if rope else 'out'} flips {rf}/{s.numel()}  "
          f"{'ok' if ok else 'FAIL'}")


def graph_us(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=st):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    gr.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 5 / reps * 1e6


def bench(T, H=4, D=128, rd=32):
    norm = make_norm(D)
    x = rnd(T, (H + 2) * D)[:, : H * D].view(T, H, D)
    cs = (torch.rand((MAXPOS, rd), generator=g) * 2 - 1).to(torch.bfloat16).to(dev)
    pos = torch.randint(0, MAXPOS, (T,), generator=g).to(dev)
    with torch.no_grad():
        for name, f in (("stock norm+rope", lambda: stock_rope(norm.forward_native(x.reshape(-1, D)).reshape(T, H, D), cs, pos, rd)),
                        ("ours norm+rope", lambda: G.gemma_norm_rope(x, norm.weight, norm.variance_epsilon, cs, pos, rd, 0, 0, 0, False)),
                        ("stock norm", lambda: norm.forward_native(x.reshape(-1, D))),
                        ("ours norm", lambda: G.gemma_norm(x, norm.weight, norm.variance_epsilon))):
            print(f"  tokens {T:<5} {name:<16} {graph_us(f):7.1f} us (graph replay)")


def main():
    if not G.available():
        print("libr9k.so has no r9k_gemma_norm_rope: rebuild kernels/")
        sys.exit(1)
    G.register()
    global bad
    norm = make_norm(128)
    x = rnd(5, 4, 128)
    cs = (torch.rand((MAXPOS, 32), generator=g) * 2 - 1).to(torch.bfloat16).to(dev)
    pos = torch.randint(0, MAXPOS, (5,), generator=g).to(dev)
    with torch.no_grad():
        bad += not torch.equal(torch.ops.r9700.qsa_gemma_norm(x, norm.weight, 1e-6), G.gemma_norm(x, norm.weight, 1e-6))
        bad += not torch.equal(torch.ops.r9700.qsa_gemma_norm_rope(x, norm.weight, 1e-6, cs, pos, 32, 0, 0, 0, False),
                               G.gemma_norm_rope(x, norm.weight, 1e-6, cs, pos, 32, 0, 0, 0, False))
    print(f"  {'torch.ops.r9700.qsa_gemma_norm[_rope] registered and equal to direct calls':<40} "
          f"{'ok' if bad == 0 else 'FAIL'}")
    check("q: 1 token, 4 heads, D=128, rd=32", 1, 4, 128, 32)
    check("q: 5 tokens (strided view)", 5, 4, 128, 32)
    check("q: 64 tokens", 64, 4, 128, 32)
    check("q: 1000 tokens", 1000, 4, 128, 32)
    check("q: 333 tokens, contiguous", 333, 4, 128, 32, strided=False)
    check("k: 700 rows, 1 head, norm only", 700, 1, 128, 32, strided=False, rope=False)
    check("k: 700 rows, 1 head, rope", 700, 1, 128, 32, strided=False)
    check("q: 2048 tokens, 1 head, positions stride 3", 2048, 1, 128, 64, strided=True, pos_stride=3)
    check("q: 64 tokens, positions stride 5", 64, 4, 128, 32, pos_stride=5)
    check("D=64 rd=16, 2 heads", 77, 2, 64, 16)
    check("D=256 rd=64, 3 heads", 50, 3, 256, 64)
    check("full rotary rd=D=128", 40, 4, 128, 128)
    check("mrope [11,11,10] interleaved, 4 heads", 64, 4, 128, 32, mrope=((11, 11, 10), True))
    check("mrope interleaved, equal T/H/W rows", 64, 4, 128, 32, mrope=((11, 11, 10), True), same_pos=True)
    check("mrope [6,5,5] concatenated, 1 head", 300, 1, 128, 32, strided=False, mrope=((6, 5, 5), False))
    check("mrope [6,5,5] concat, pos stride 2", 300, 1, 128, 32, strided=False, mrope=((6, 5, 5), False),
          pos_stride=2)
    check("mrope interleaved rd=64 D=256", 30, 3, 256, 64, mrope=((11, 11, 10), True))
    print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")
    if os.environ.get("BENCH", "1") == "1":
        bench(4)
        bench(4096)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
