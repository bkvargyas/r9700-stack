"""r9k_moe_sum against vLLM's moe_sum (+ the runner's bf16 add of the shared expert).

Without a shared term the two must agree bit for bit (fp32 sum over top-k, one bf16 rounding). With it, ours
rounds once where stock rounds twice, so results may differ by one bf16 ulp of the final value, never more."""
import os
import sys

import torch
import vllm._custom_ops as ops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import moe as KM                                     # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
fails = 0
for M, topk, N in ((1, 10, 2560), (4, 10, 2560), (37, 10, 2560), (64, 10, 640), (256, 10, 2560), (3, 4, 320)):
    down = (torch.randn(M, topk, N, device=dev) * 0.5).to(torch.bfloat16)
    shared = (torch.randn(M, N, device=dev) * 0.5).to(torch.bfloat16)
    ref = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    ops.moe_sum(down, ref)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    KM.moe_sum(down, None, out)
    torch.cuda.synchronize()
    eq = torch.equal(out, ref)
    # with the shared term: stock = bf16(bf16(sum) + shared); ours = bf16(sum + shared)
    ref2 = ref + shared
    out2 = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    KM.moe_sum(down, shared, out2)
    exact = (down.float().sum(1) + shared.float())
    d_ours = (out2.float() - exact).abs().max().item()
    d_stock = (ref2.float() - exact).abs().max().item()
    ulp = torch.finfo(torch.bfloat16).eps * exact.abs().max().item()
    ok = eq and d_ours <= d_stock + 1e-6 and d_ours <= ulp
    print(f"M={M:3d} topk={topk} N={N}: no-shared bit-equal {eq}; with shared: |ours-exact| {d_ours:.3e} "
          f"|stock-exact| {d_stock:.3e} (1 ulp at max {ulp:.3e}) {'OK' if ok else 'FAIL'}")
    fails += 0 if ok else 1
# strided shared / output rows
M, topk, N = 8, 10, 2560
down = torch.randn(M, topk, N, device=dev).to(torch.bfloat16)
big = torch.randn(M, N + 512, device=dev).to(torch.bfloat16)
out = torch.empty(M, N + 256, dtype=torch.bfloat16, device=dev)
KM.moe_sum(down, big[:, :N], out[:, :N])
ref = torch.empty(M, N, dtype=torch.bfloat16, device=dev); ops.moe_sum(down, ref)
d = (out[:, :N].float() - (ref.float() + big[:, :N].float())).abs().max().item()
ok = d <= torch.finfo(torch.bfloat16).eps * (ref.float().abs() + big[:, :N].float().abs()).max().item()
print(f"strided rows: max diff {d:.3e} {'OK' if ok else 'FAIL'}"); fails += 0 if ok else 1
print("test_moe_sum_r9k:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
