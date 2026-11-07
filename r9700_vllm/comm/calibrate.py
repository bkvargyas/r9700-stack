"""Startup calibration of the N-rank collective cutoffs (R9kAllReduceN) against RCCL, on the box that is serving.

Why: the fixed cutoffs (one-shot <= 16 KB, two-shot <= 512 KB, r9k all-gather <= 128 KB) were measured on PLX
boxes, where every pair of cards sits behind one PEX 8747 and a P2P write stays inside the switch. On a box with
every card on its own CPU root port (gpusrv1: EPYC 7532, 4x R9700 at Gen4 x16, KVM + VFIO), the order flips
(tests/test_ar_nrank.py, graph-timed, us; 2026-11-06):

    message   rccl   one-shot  two-shot          PLX box (same test): rccl / one-shot / two-shot
      5 KB    40.1      6.4      19.4                                  59.2 /   6.8 /   9.7
     20 KB    36.2     18.2      24.0                                  72.6 /  13.4 /  11.3
     80 KB    35.7     69.8      92.8                                  71.2 /  37.2 /  24.2
    160 KB    40.4       -      216.4
    320 KB    53.4       -      466.1                                 102.0 / 130.4 /  73.2

so two-shot is the slowest choice at every size there and RCCL wins from ~40 KB, while the fixed cutoffs send
everything up to 512 KB through two-shot: 16 streams ran at 466 tok/s instead of 910, and the 9 r9k all-gathers a
step cost 0.9 ms (vs RCCL). This measures the three paths at the sizes decode produces, in HIP graphs (decode
runs inside graphs, eager timings would mislead), on every rank, and keeps the fastest exact path per size.

Safety:
  * every rank takes the slowest rank's time (MAX over the CPU group), so the cutoffs agree on every rank;
  * a candidate is replayed only once every rank captured it (MIN over a captured flag), so a capture failure
    on one rank cannot leave the others waiting inside a collective;
  * only the routing cutoffs change: pools, kernels and bits are the ones the fixed configuration uses.

Knobs: R9K_ARN_CALIBRATE=0 keeps the fixed cutoffs (PLX boxes, A/B). An explicit R9K_ARN_1S_KB / R9K_ARN_MAX_KB
keeps the all-reduce cutoffs as given; an explicit R9K_AG / R9K_AG_MAX_KB keeps the all-gather as given.
R9K_ARN_CAL_SIZES_KB overrides the all-reduce probe sizes (comma list).
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

INF = float("inf")
AR_SIZES_KB = (5, 10, 20, 40, 60, 80, 120, 160, 240, 320, 480)   # 1..96 tokens x 2560 bf16 (Flash-Next hidden)
AG_SIZES_KB = (2, 8, 32, 64, 128)                                  # per-rank bytes


def enabled() -> bool:
    return os.environ.get("R9K_ARN_CALIBRATE", "1") == "1"


def ar_pinned() -> bool:
    """The caller fixed the all-reduce cutoffs explicitly: leave them alone."""
    return "R9K_ARN_1S_KB" in os.environ or "R9K_ARN_MAX_KB" in os.environ


def ag_pinned() -> bool:
    return "R9K_AG" in os.environ or "R9K_AG_MAX_KB" in os.environ


def _graph(fn, reps: int):
    """Capture `reps` calls of fn in one HIP graph (after a warm-up on a side stream). None if capture fails."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        torch.cuda.synchronize()
        return g
    except Exception as e:                       # noqa: BLE001 - any failure just drops this candidate
        logger.debug("r9700: calibration capture failed: %s", e)
        try:
            torch.cuda.synchronize()
        except Exception:                        # noqa: BLE001
            pass
        return None


def _all_ok(ok: bool, cpu_group) -> bool:
    t = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(t.item())


def _time(fn, cpu_group, reps: int = 20, iters: int = 8) -> float:
    """us per call, graph-timed on this rank; INF on every rank if any rank failed to capture."""
    g = _graph(fn, reps)
    if not _all_ok(g is not None, cpu_group):
        return INF
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * iters)


def _slowest(times: list[float], cpu_group) -> list[float]:
    t = torch.tensor([1e12 if x == INF else x for x in times], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
    return [INF if v >= 1e12 else float(v) for v in t.tolist()]


def _sizes(var: str, default: tuple[int, ...]) -> list[int]:
    v = os.environ.get(var)
    return [int(float(x)) for x in v.split(",")] if v else list(default)


def calibrate_nrank(ar, rccl_all_reduce, rccl_all_gather, cpu_group) -> None:
    """Measure and set ar.max1 / ar.max_bytes (all-reduce) and ar.ag_max / ar._spg (all-gather) in place.

    ar: an installed R9kAllReduceN. rccl_all_reduce(x) / rccl_all_gather(x, dim): the communicator's own RCCL
    paths (what a message falls back to when should() says no)."""
    t0 = time.time()
    dev = ar.device
    report = []

    if not ar_pinned():
        pool1 = ar.slot1 * 16                       # one-shot slot (sized for the calibration ceiling)
        pool2 = ar.max_bytes if ar._sp2 is not None else 0
        sizes = [kb * 1024 for kb in _sizes("R9K_ARN_CAL_SIZES_KB", AR_SIZES_KB) if kb * 1024 <= max(pool1, pool2,
                                                                                                    ar.ar4_min)]
        rows = []
        for nb in sizes:
            x = torch.randn(nb // 2, device=dev).to(torch.bfloat16)
            t_r = _time(lambda: rccl_all_reduce(x.clone()), cpu_group)
            t_1 = _time(lambda: ar.all_reduce(x, mode=1), cpu_group) if nb <= pool1 else INF
            t_2 = _time(lambda: ar.all_reduce(x, mode=2), cpu_group) if 0 < nb <= pool2 else INF
            rows.append((nb, t_r, t_1, t_2))
        flat = _slowest([v for r in rows for v in r[1:]], cpu_group)
        rows = [(r[0], *flat[3 * i:3 * i + 3]) for i, r in enumerate(rows)]
        win = ["rccl" if r[1] <= min(r[2], r[3]) else ("one" if r[2] <= r[3] else "two") for r in rows]
        # one-shot up to the end of its leading run of wins; two-shot over the run of wins right after it;
        # RCCL above (the compressed path only from ar.ar4_min, where it was validated, as before)
        i, max1 = 0, 0
        while i < len(rows) and win[i] == "one":
            max1 = rows[i][0]
            i += 1
        max2 = max1
        while i < len(rows) and win[i] == "two":
            max2 = rows[i][0]
            i += 1
        ar.max1, ar.max_bytes = max1, max2
        report.append("all-reduce: " + ", ".join(
            f"{r[0] >> 10}K {'/'.join('-' if v == INF else f'{v:.0f}' for v in r[1:])}" for r in rows))
        report.append(f"-> one-shot <= {max1 >> 10} KiB, two-shot "
                      f"{'off' if max2 == max1 else f'<= {max2 >> 10} KiB'}, RCCL above"
                      f"{f', compressed from {ar.ar4_min >> 10} KiB' if ar.ar4 is not None else ''}")

    if not ag_pinned() and ar._spg is not None:
        rows = []
        for kb in [k for k in AG_SIZES_KB if k * 1024 <= ar.ag_max]:
            x = torch.randn(4, kb * 1024 // 2 // 4, device=dev).to(torch.bfloat16)
            t_r = _time(lambda: rccl_all_gather(x, -1), cpu_group)
            t_g = _time(lambda: ar.all_gather(x, -1), cpu_group)
            rows.append((kb * 1024, t_r, t_g))
        flat = _slowest([v for r in rows for v in r[1:]], cpu_group)
        rows = [(r[0], *flat[2 * i:2 * i + 2]) for i, r in enumerate(rows)]
        cap = 0
        for nb, t_r, t_g in rows:
            if t_g >= t_r:
                break
            cap = nb
        ar.ag_max = cap
        if cap == 0:
            ar._spg = None                       # should_ag() is False: every all-gather goes to RCCL
        report.append("all-gather (rccl/r9k): " + ", ".join(
            f"{r[0] >> 10}K {'/'.join('-' if v == INF else f'{v:.0f}' for v in r[1:])}" for r in rows))
        report.append(f"-> r9k all-gather {'off' if cap == 0 else f'<= {cap >> 10} KiB'}, RCCL above")

    if report and ar.rank == 0:
        logger.info("r9700: collective calibration (%.1f s, us per call, slowest rank): %s", time.time() - t0,
                    "; ".join(report))
