"""Where do the ~87 us of non-link time go in the compressed all-reduce?

    torchrun --nproc-per-node=2 tuning/ar_profile.py

Phase 1 step 1.1 of notes/replacement-plan.md. Our compressed path runs at ~0.22 us/KB against 0.115 us/KB for
our own exact kernel, so it is overhead-bound rather than link-bound -- but "overhead" covers three different
things and they have different fixes. This separates them:

  * each of the three kernels timed ALONE (pack, push, reduce), so we know the per-kernel cost;
  * the full three-call path, so the difference is launch gaps and fences between them;
  * the EXACT kernel moving the same number of BYTES as the compressed payload. That is the control that
    matters: if exact-at-800KB is much faster than our push-at-800KB, the push kernel itself is the problem;
    if they match, the cost is in the gaps and the fix is fusion or fewer fences.
  * a drain/acq sweep, because a system-scope fence per block per call is not free and we never measured
    whether we need the strongest one.

Do NOT skip to writing kernels off this: the last guess (fuse all three) measured 314 us against 179.
"""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def bench(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / n


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    os.environ.setdefault("R9K_AR_QUANT", "1")

    from r9700_vllm.comm.r9k_ar import R9kAllReduce, _DTYPE
    ar = R9kAllReduce(dist.group.WORLD, dev)
    assert not ar.disabled and ar._wht, "compressed path not enabled (need R9K_AR_QUANT=1)"
    L, st = ar.L, torch.cuda.current_stream().cuda_stream

    def line(label, us, nbytes=None):
        per_kb = f"{us / (nbytes / 1024):7.3f} us/KB" if nbytes else " " * 15
        if rank == 0:
            print(f"  {label:<34} {us:8.1f} us   {per_kb}")

    for numel in (1 << 19, 1 << 20):                       # 1 MB and 2 MB of bf16
        x = torch.randn(numel, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        raw = numel * 2
        ng = numel // ar._qgroup
        pk = ng * ar._qbytes                               # bytes actually on the wire, compressed
        if rank == 0:
            print(f"\n=== {raw} B raw -> {pk} B on the wire ({raw/pk:.2f}x) ===")

        # --- the three kernels, alone
        t_pack = bench(lambda: L.r9k_wht_pack(x.data_ptr(), ar._locpk.data_ptr(), numel, _DTYPE[x.dtype], st))
        line("pack alone", t_pack, raw)
        t_push = bench(lambda: L.r9k_ar_push_2rank(
            ar._peer_scratch, ar._peer_flags, ar._flags, ar._qseq.data_ptr(), ar.slot16,
            ar._locpk.data_ptr(), (pk + 15) // 16 * 16, st, ar.max_nb, 0, ar.drain, ar.acq, ar._slot.data_ptr()))
        line("push alone (exchange only)", t_push, pk)
        t_red = bench(lambda: L.r9k_wht_reduce_at(
            ar._locpk.data_ptr(), ar._scratch, ar._slot.data_ptr(), ar.max_bytes,
            out.data_ptr(), numel, _DTYPE[x.dtype], st))
        line("reduce alone", t_red, raw)

        # --- the whole path, so the delta is launch gaps + fences
        t_all = bench(lambda: ar._all_reduce_wht(x, out))
        line("full compressed path", t_all, raw)
        if rank == 0:
            print(f"  {'  -> sum of parts':<34} {t_pack + t_push + t_red:8.1f} us"
                  f"   gap {t_all - (t_pack + t_push + t_red):+7.1f} us")

        # --- controls
        t_exact = bench(lambda: L.r9k_ar_oneshot_2rank(
            ar._peer_scratch, ar._scratch, ar._peer_flags, ar._flags, ar._seq.data_ptr(), ar.slot16,
            x.data_ptr(), out.data_ptr(), numel, _DTYPE[x.dtype], st, ar.max_nb, 0, ar.drain, ar.acq))
        line("exact kernel, same ELEMENTS", t_exact, raw)
        # same bytes on the wire as the compressed payload: how fast SHOULD an exchange of pk bytes be?
        nb_equiv = (pk // 2) // 64 * 64
        y = torch.randn(nb_equiv, device=dev, dtype=torch.bfloat16)
        oy = torch.empty_like(y)
        t_exact_eq = bench(lambda: L.r9k_ar_oneshot_2rank(
            ar._peer_scratch, ar._scratch, ar._peer_flags, ar._flags, ar._seq.data_ptr(), ar.slot16,
            y.data_ptr(), oy.data_ptr(), nb_equiv, _DTYPE[y.dtype], st, ar.max_nb, 0, ar.drain, ar.acq))
        line("exact kernel, same BYTES on wire", t_exact_eq, nb_equiv * 2)
        if rank == 0:
            print(f"  {'  -> push overhead vs exact':<34} {t_push - t_exact_eq:+8.1f} us")

    # --- is the strongest fence actually needed?
    numel = 1 << 20
    x = torch.randn(numel, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ref = x.float().clone()
    dist.all_reduce(ref)
    if rank == 0:
        print("\n=== fence sweep (2 MB, compressed) -- correctness AND speed ===")
    base_drain, base_acq = ar.drain, ar.acq
    for drain in (3, 2, 1):
        for acq in (0, 1):
            ar.drain, ar.acq = drain, acq
            got = ar._all_reduce_wht(x, torch.empty_like(x))
            torch.cuda.synchronize()
            rel = ((got.float() - ref).norm() / ref.norm()).item()
            other = got.clone()
            dist.broadcast(other, src=0)
            agree = torch.equal(got, other)
            t = bench(lambda: ar._all_reduce_wht(x, out), n=100)
            if rank == 0:
                flag = "ok" if (rel < 0.035 and agree) else "  <-- WRONG"
                print(f"  drain={drain} acq={acq}   {t:7.1f} us   rel {rel:.4f}  agree {agree}  {flag}")
    ar.drain, ar.acq = base_drain, base_acq

    dist.barrier()
    if rank == 0:
        print("\nNote: a fence setting that looks correct here can still be racy under real load -- the test "
              "issues calls back to back with both ranks in lockstep. Treat a win as a hypothesis to verify "
              "end to end with GSM8K, not as a result.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
