#!/usr/bin/env python3
"""dqchunk_probe.py: which row counts M give exllamav3's hgemm (hipBLAS, f16 in, f32 out) the same per-row result:
the first 128 rows of an M-row product compared bitwise against the M=128 product, for M = 1..16384, f32 and f16 out,
K x N = 5120 x 2304 (gate / up) and 2304 x 5120 (down)."""
import torch
from r9700_vllm.quant.exl3 import ext
X = ext(); D = torch.device("cuda:0")
torch.manual_seed(0)
for k, n in ((5120, 2304), (2304, 5120)):
    w = (torch.randn((k, n), device=D) * 0.02).half()
    a = torch.randn((16384, k), device=D).half()
    for odt in (torch.float32, torch.float16):
        ref = {}
        line = []
        for M in (1, 8, 16, 64, 128, 129, 256, 512, 768, 1000, 1024, 1025, 1536, 2048, 3000, 4096, 6144, 8192, 12288, 16384):
            o = torch.empty((M, n), dtype=odt, device=D)
            X.hgemm(a[:M], w, o)
            r = o[:min(M, 128)].clone()
            if 128 not in ref and M >= 128:
                pass
            ref[M] = r
        base = ref[128]
        cls = {}
        for M, r in ref.items():
            m = min(M, 128)
            key = None
            for c, rr in cls.items():
                if torch.equal(r.view(torch.int16 if odt == torch.float16 else torch.int32)[:m],
                               rr.view(torch.int16 if odt == torch.float16 else torch.int32)[:m]):
                    key = c; break
            if key is None:
                key = M; cls[M] = r
            line.append(f"{M}->{key}")
        print(f"k{k} n{n} out {str(odt)[6:]}: same-result classes (M -> first M of its class): " + " ".join(line))
