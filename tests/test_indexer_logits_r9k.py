"""Decode-side indexer logits over the tiled paged K cache (attn/indexer_logits.py) vs the torch reference, with the cache
written by vLLM's kpool compress kernel (GLM-5.3-Flash, page 64 pooled entries); the stock stage-1 reader is shown for
contrast (it reads a linear page and is wrong on this layout). Run in the ROCm image with PYTHONPATH=/opt/r9700, one GPU.
The real GLM writer (kpool compress, page 64 pooled entries) -> stock stage1 decode logits vs our tiled-layout
kernel vs the torch reference on the compressed keys the writer returns. next_n 1 and 4; timing."""
import os, sys, time, torch, vllm  # noqa: F401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as S
from vllm.v1.worker.workspace import init_workspace_manager, current_workspace_manager
from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops
from r9700_vllm.attn.indexer_logits import paged_mqa_logits

dev = torch.device("cuda:0"); init_workspace_manager(dev)
H, D, PAGE, KP, K = 32, 128, 64, 4, 512
FP8 = S.FP8_DTYPE
mod = S.paged_mqa_logits_module()
g = torch.Generator(device="cpu").manual_seed(3)
ok_all = True


def overlap(a, b):
    a = a.long(); b = b.long(); return (a[:, :, None] == b[:, None, :]).any(-1).float().mean().item()


def tm(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6


for W, Nc, NN in ((16, 760, 1), (64, 3200, 1), (128, 8000, 1), (512, 32000, 1), (1024, 65000, 1), (64, 3200, 4), (128, 8000, 4)):
    npages = W + 8
    kv = torch.zeros(npages, PAGE, 1, D + 4, dtype=torch.uint8, device=dev)
    tok_k = (torch.randn(Nc * KP, D, generator=g) * 0.5).to(dev).to(torch.bfloat16)      # per-token K
    score = torch.randn(Nc * KP, D, generator=g).to(dev).to(torch.bfloat16)               # per-token gate score
    ape = (torch.randn(KP, D, generator=g) * 0.1).to(dev)
    qf = torch.randn(1, NN, H, D, generator=g) * 0.5
    nstar = Nc * 2 // 3
    tok_k[nstar * KP:(nstar + 1) * KP] = (3.0 * qf[0].mean(dim=(0, 1)).sign()).to(dev).to(torch.bfloat16)   # the needle pool
    q = qf.to(dev).to(FP8); w = torch.rand(NN, H, generator=g).to(dev)
    bt = torch.randperm(npages, generator=g)[:W].to(torch.int32).to(dev)[None, :]
    pools = torch.arange(Nc, device=dev)
    loc = (bt[0, pools // PAGE].long() * PAGE + pools % PAGE)
    ck, cs = kpool_ops.kpool_compress_and_write_cache(kv, tok_k.view(Nc, KP, D), score.view(Nc, KP, D), ape, loc, pool_size=KP, head_dim=D,
                                                      write_mask=None, round_scale=True, return_compressed=True, write_cache=True)
    ref = S.fp8_mqa_logits_torch(q[0], (ck, cs.reshape(-1)), w, torch.zeros(NN, dtype=torch.int32, device=dev),
                                 torch.tensor([Nc - (NN - 1 - i) for i in range(NN)], dtype=torch.int32, device=dev))
    # sanity: the stock gather (SHUFFLE default for page > 1) reads back what the writer wrote
    kq = torch.empty(Nc, D, dtype=FP8, device=dev); ksc = torch.empty(Nc, 4, dtype=torch.uint8, device=dev)
    S.cp_gather_indexer_k_quant_cache_triton(kv, kq, ksc, bt, torch.tensor([0, Nc], dtype=torch.int32, device=dev), torch.zeros(Nc, dtype=torch.int32, device=dev))
    gather_ok = torch.equal(kq.view(torch.uint8), ck.view(torch.uint8)) and torch.allclose(ksc.view(torch.float32).reshape(-1), cs.reshape(-1))
    ctx = torch.tensor([Nc], dtype=torch.int32, device=dev); ML = W * PAGE
    def run_stock():
        return S.rocm_fp8_paged_mqa_logits(q, kv, w, ctx, bt, None, ML)
    def run_ours():
        return paged_mqa_logits(q, kv, w, ctx, bt, ML)
    print(f"W={W:4d} pools={Nc:5d} next_n={NN} needle={nstar}  writer->gather round trip exact: {gather_ok}")
    valid = torch.isfinite(ref)
    for tag, fn in (("stock (stage1)", run_stock), ("ours (tiled)", run_ours)):
        try:
            out = fn()[:, :Nc]
            d = (out - ref).abs()[valid].max().item(); mag = ref[valid].abs().max().item()
            ov = overlap(torch.topk(out.masked_fill(~valid, float("-inf")), K, dim=-1).indices, torch.topk(ref.masked_fill(~valid, float("-inf")), K, dim=-1).indices)
            am = out.masked_fill(~valid, float("-inf")).argmax(-1).tolist()
            beyond_ok = not torch.isfinite(fn()[:, Nc:]).any().item() if ML > Nc else True
            good = d < 2e-2 * mag and ov > 0.97 and all(a == nstar for a in am) and beyond_ok
            if tag.startswith("ours"): ok_all &= good
            print(f"   {tag:16s} {'ok ' if good else 'BAD'}  max|diff| {d:8.3g} (|ref| {mag:.0f})  top-{K} overlap {ov * 100:5.1f}%  argmax {am}  -inf beyond ctx {beyond_ok}  {tm(fn):7.1f} us")
        except Exception as e:
            if tag.startswith("ours"): ok_all = False
            print(f"   {tag:16s} raised {type(e).__name__}: {str(e)[:200]}")
print("test_indexer_logits_r9k: PASS" if ok_all else "test_indexer_logits_r9k: FAIL"); sys.exit(0 if ok_all else 1)
