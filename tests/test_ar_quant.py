"""libr4d all-reduce, exact vs wht6 (R9K_AR_QUANT): accuracy vs the exact sum, rank bit-identity, latency in graphs."""
import os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp


def run(rank, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29561", R9K_AR_QUANT="1")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from r9700_vllm.comm.r4d_ar import R4dAllReduce
    ar = R4dAllReduce(dist.group.WORLD, rank)
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    res = []
    for rows in (8, 16, 64, 512, 4096):
        x = torch.randn(rows, 5120, device="cuda", generator=g).to(torch.bfloat16)
        x[:, 7] *= 30                                  # an outlier channel, as in real hidden states
        xs = [torch.empty_like(x) for _ in range(2)]
        dist.all_gather(xs, x.cpu()) if False else None
        other = x.clone()
        dist.broadcast(other.cpu() if False else other, src=rank) if False else None
        # exact reference: gather both inputs on CPU
        cpu = [torch.empty(rows, 5120, dtype=torch.bfloat16) for _ in range(2)]
        dist.all_gather(cpu, x.cpu())
        ref = (cpu[0].float() + cpu[1].float())
        y = ar.all_reduce(x); torch.cuda.synchronize()
        yc = y.cpu()
        err = ((yc.float() - ref).norm() / ref.norm()).item()
        peers = [torch.empty_like(yc) for _ in range(2)]
        dist.all_gather(peers, yc)
        same = torch.equal(peers[0], peers[1])
        gr = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): ar.all_reduce(x)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        with torch.cuda.graph(gr):
            for _ in range(20): ar.all_reduce(x)
        dist.barrier(); gr.replay(); torch.cuda.synchronize(); dist.barrier()
        t = time.perf_counter()
        for _ in range(5): gr.replay()
        torch.cuda.synchronize(); us = (time.perf_counter() - t) / 100 * 1e6
        res.append((rows, rows * 10, err, same, us))
    if rank == 0:
        q.put(res)
    dist.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=run, args=(r, q)) for r in range(2)]
    [p.start() for p in ps]; res = q.get(); [p.join() for p in ps]
    ok = True
    for rows, kb, err, same, us in res:
        quant = kb >= 128
        good = same and err < (3e-2 if quant else 1e-2)   # 6.25-bit payload: ~2.3% on Gaussian data
        ok &= good
        print(f"  {rows:5d} x 5120 ({kb:6d} KB) {'wht6 ' if quant else 'exact'}: rel err {err:.2e}  ranks identical {same}  "
              f"{us:8.1f} us  {'ok' if good else '<-- FAIL'}")
    print("ALL OK" if ok else "FAILURES"); sys.exit(0 if ok else 1)
