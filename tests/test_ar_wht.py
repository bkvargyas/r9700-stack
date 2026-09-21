"""Walsh-Hadamard 6-bit all-reduce payload (kernels/r9k_ar_wht.hip): accuracy, symmetry, throughput.

Single GPU -- it exercises the pack/reduce pair directly, standing in for the two ranks, so no torchrun needed:
    python3 tests/test_ar_wht.py

What matters for an all-reduce payload:
  * the reduce is exactly `dequant(pack(a)) + dequant(pack(b))` rotated back, so BOTH RANKS get the same answer
    even though it is not the exact sum -- checked by reducing (a,b) and (b,a)
  * the error is small relative to the magnitude of the sum, and much smaller than quantising without the
    rotation (that comparison is the whole reason the Hadamard step is there)
  * it must beat shipping bf16 on bytes: 50 bytes per 64 elements = 6.25 bits/elem = 2.56x
"""
import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import moe as KM  # noqa: E402

DT = {torch.bfloat16: 0, torch.float16: 1}


def lib():
    L = KM.lib()
    for n in ("r9k_wht_group", "r9k_wht_bits", "r9k_wht_group_bytes"):
        getattr(L, n).restype = ctypes.c_int
    L.r9k_wht_pack.restype = ctypes.c_int
    L.r9k_wht_pack.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int, ctypes.c_long]
    L.r9k_wht_reduce.restype = ctypes.c_int
    L.r9k_wht_reduce.argtypes = [ctypes.c_long] * 4 + [ctypes.c_int, ctypes.c_long]
    return L


def main():
    L = lib()
    G, BITS, GB = L.r9k_wht_group(), L.r9k_wht_bits(), L.r9k_wht_group_bytes()
    dev = "cuda"
    stream = torch.cuda.current_stream().cuda_stream
    print(f"group {G} elements, {BITS} bits, {GB} bytes/group = {GB*8/G:.2f} bits/elem "
          f"({16/(GB*8/G):.2f}x vs bf16)")
    ok = True

    def pack(x):
        pk = torch.empty(((x.numel() + G - 1) // G) * GB, dtype=torch.uint8, device=dev)
        rc = L.r9k_wht_pack(x.data_ptr(), pk.data_ptr(), x.numel(), DT[x.dtype], stream)
        assert rc == 0, f"pack rc {rc}"
        return pk

    def reduce(pa, pb, numel, dtype):
        out = torch.empty(numel, dtype=dtype, device=dev)
        rc = L.r9k_wht_reduce(pa.data_ptr(), pb.data_ptr(), out.data_ptr(), numel, DT[dtype], stream)
        assert rc == 0, f"reduce rc {rc}"
        return out

    for dtype in (torch.bfloat16, torch.float16):
        for numel, desc in ((G, "one group"), (4096, "small"), (5120 * 8, "decode-ish"), (1 << 20, "prefill-ish")):
            g = torch.Generator(device=dev).manual_seed(numel)
            a = torch.randn(numel, generator=g, device=dev, dtype=torch.float32).to(dtype)
            b = torch.randn(numel, generator=g, device=dev, dtype=torch.float32).to(dtype)
            pa, pb = pack(a), pack(b)
            out = reduce(pa, pb, numel, dtype)
            ref = (a.float() + b.float())

            rel = ((out.float() - ref).norm() / ref.norm()).item()
            # both ranks must land on the same bits: the pair is the same, only the argument order differs
            sym = torch.equal(out, reduce(pb, pa, numel, dtype))
            good = rel < 0.02 and sym
            ok &= good
            print(f"  {str(dtype):>16} {desc:>12} n={numel:>8}: rel {rel:.4f}  ranks agree {sym}"
                  + ("" if good else "   <-- FAIL"))

    # the rotation should beat plain 6-bit group quantisation on heavy-tailed data (that is why it is there)
    g = torch.Generator(device=dev).manual_seed(11)
    x = torch.randn(1 << 18, generator=g, device=dev, dtype=torch.float32)
    x[::997] *= 40.0                                            # outliers, as activations actually have
    a = x.to(torch.bfloat16)
    z = torch.zeros_like(a)
    got = reduce(pack(a), pack(z), a.numel(), torch.bfloat16).float()
    rel_wht = ((got - a.float()).norm() / a.float().norm()).item()
    blk = a.float().reshape(-1, G)                              # same budget, no rotation
    s = blk.abs().amax(1, keepdim=True) / 31.0
    rel_raw = (((blk / s).round().clamp(-31, 31) * s - blk).norm() / blk.norm()).item()
    print(f"  outlier data: rotated {rel_wht:.4f} vs unrotated 6-bit {rel_raw:.4f} "
          f"({rel_raw/max(rel_wht,1e-9):.2f}x better)")
    ok &= rel_wht < rel_raw

    # throughput: pack+reduce must not cost more than the link time it saves
    x = torch.randn(1 << 20, device=dev, dtype=torch.bfloat16)
    px, py = pack(x), pack(x)

    def timeit(fn, n=100):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
        s_.record()
        for _ in range(n):
            fn()
        e_.record()
        torch.cuda.synchronize()
        return s_.elapsed_time(e_) * 1e3 / n

    t_pack = timeit(lambda: pack(x))
    t_red = timeit(lambda: reduce(px, py, x.numel(), torch.bfloat16))
    saved = (x.numel() * 2 - px.numel()) / (16e9 / 8)           # PCIe3 x16 ~16 GB/s each way
    print(f"  {x.numel()*2} B: pack {t_pack:.1f} us + reduce {t_red:.1f} us = {t_pack+t_red:.1f} us; "
          f"bytes saved ~{saved*1e6:.1f} us of link time")

    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
