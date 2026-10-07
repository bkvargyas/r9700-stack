"""The pipelined prefill tail's comm mechanics (comm/pipe.py) without the model: a second ar4 instance on its own
stream, part all-reduces into row slices of one buffer, overlapped with compute on the main stream.
    torchrun --nproc-per-node=4 tests/test_ar_pipe.py
Checks: the part-wise result equals the whole-message ar4 result within the 4-bit wire's error (both against the
fp32 sum), on every rank; then times a layer-tail stand-in (two 21 MB all-reduces around a ~2 ms "MoE" GEMM, with
a 0.4 ms "combine" in between) in stock order vs pipelined with 2 and 4 parts."""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    assert world == 4, "4 ranks"
    from r9700_vllm.comm.r9k_ar4 import R9kAllReduce4
    ar_main = R9kAllReduce4(dist.group.WORLD, dev)
    ar_pipe = R9kAllReduce4(dist.group.WORLD, dev, max_mb=24)
    assert not ar_main.disabled and not ar_pipe.disabled
    cs = torch.cuda.Stream(priority=-1)
    main_s = torch.cuda.current_stream()
    M, HID = 4096, 2560
    fails = 0

    def inp(it):
        g = torch.Generator(device="cpu").manual_seed(1000 * it + rank)
        x = torch.randn(M, HID, generator=g)
        x[torch.rand(M, HID, generator=g) < 0.01] *= 20
        return x.to(torch.bfloat16).to(dev)

    def ref(it):
        acc = torch.zeros(M, HID, dtype=torch.float32, device=dev)
        for q in range(world):
            g = torch.Generator(device="cpu").manual_seed(1000 * it + q)
            x = torch.randn(M, HID, generator=g)
            x[torch.rand(M, HID, generator=g) < 0.01] *= 20
            acc += x.to(torch.bfloat16).to(dev).float()
        return acc

    def pipelined(x, P):
        b = [M * i // P for i in range(P + 1)]
        out = torch.empty_like(x)
        e0 = torch.cuda.Event()
        e0.record(main_s)
        cs.wait_event(e0)
        x.record_stream(cs)
        out.record_stream(cs)
        evs = []
        with torch.cuda.stream(cs):
            for p in range(P):
                ar_pipe.all_reduce(x[b[p]:b[p + 1]], out=out[b[p]:b[p + 1]])
                e = torch.cuda.Event()
                e.record(cs)
                evs.append(e)
        for e in evs:
            main_s.wait_event(e)
        return out

    for it in range(3):
        x = inp(it)
        r = ref(it)
        whole = ar_main.all_reduce(x)
        for P in (2, 4):
            out = pipelined(x, P)
            torch.cuda.synchronize()
            e_whole = ((whole.float() - r).norm() / r.norm()).item()
            e_part = ((out.float() - r).norm() / r.norm()).item()
            same = torch.equal(out, whole)
            ok = e_part < 2 * e_whole + 1e-3
            fails += 0 if ok else 1
            if rank == 0:
                print(f"it {it} P={P}: rel err whole {e_whole:.3e}  parts {e_part:.3e}  bit-equal {same}  "
                      f"{'ok' if ok else '<-- FAIL'}")
    # every rank agrees on the pipelined result
    out = pipelined(inp(0), 2)
    torch.cuda.synchronize()
    gathered = [torch.empty_like(out) for _ in range(world)]
    dist.all_gather(gathered, out)
    agree = all(torch.equal(g, gathered[0]) for g in gathered)
    if rank == 0:
        print(f"ranks agree on the pipelined result: {agree}")
    fails += 0 if agree else 1

    # ---- timing: the layer tail stand-in
    W = (torch.randn(HID, 4096, device=dev) * 0.02).to(torch.bfloat16)      # "MoE": [M, 4096] x [4096, HID] ~ 2 ms? scaled below
    Wmoe = (torch.randn(HID * 4, HID, device=dev) * 0.02).to(torch.bfloat16)
    Wcomb = (torch.randn(HID, HID, device=dev) * 0.02).to(torch.bfloat16)
    attn = inp(5)

    def moe(xr):                     # ~ a few GEMMs' worth of compute on the rows
        y = torch.nn.functional.linear(xr, Wmoe)                # [rows, 4*HID]
        return torch.nn.functional.linear(y, Wmoe.t().contiguous()[:HID].t().contiguous()) if False else \
            torch.nn.functional.linear(y[:, :HID], Wcomb)

    def comb(xr):
        return torch.nn.functional.linear(xr, Wcomb)

    def stock():
        a = ar_main.all_reduce(attn)
        h = comb(a)
        m = moe(h)
        return ar_main.all_reduce(m)

    def piped(P):
        b = [M * i // P for i in range(P + 1)]
        a_r = torch.empty_like(attn)
        m_out = torch.empty_like(attn)
        for t in (attn, a_r, m_out):
            t.record_stream(cs)
        e0 = torch.cuda.Event()
        e0.record(main_s)
        cs.wait_event(e0)
        ev_a = []
        with torch.cuda.stream(cs):
            for p in range(P):
                ar_pipe.all_reduce(attn[b[p]:b[p + 1]], out=a_r[b[p]:b[p + 1]])
                e = torch.cuda.Event()
                e.record(cs)
                ev_a.append(e)
        ev_m = []
        for p in range(P):
            main_s.wait_event(ev_a[p])
            h = comb(a_r[b[p]:b[p + 1]])
            m = moe(h)
            e = torch.cuda.Event()
            e.record(main_s)
            cs.wait_event(e)
            m.record_stream(cs)
            with torch.cuda.stream(cs):
                ar_pipe.all_reduce(m, out=m_out[b[p]:b[p + 1]])
                e2 = torch.cuda.Event()
                e2.record(cs)
                ev_m.append(e2)
        for e2 in ev_m:
            main_s.wait_event(e2)
        return m_out

    def t(fn, reps=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    t_ar = t(lambda: ar_main.all_reduce(attn))
    t_moe = t(lambda: moe(comb(attn)))
    t_stock = t(stock)
    res = {P: t(lambda: piped(P)) for P in (2, 4)}
    if rank == 0:
        print(f"one 21 MB ar4 {t_ar:.2f} ms, the compute stand-in {t_moe:.2f} ms")
        print(f"tail stock (AR + compute + AR) {t_stock:.2f} ms;  pipelined P=2 {res[2]:.2f} ms  P=4 {res[4]:.2f} ms"
              f"  (ideal: compute + one AR = {t_moe + t_ar:.2f})")
        print("test_ar_pipe:", "FAIL" if fails else "PASS")
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
