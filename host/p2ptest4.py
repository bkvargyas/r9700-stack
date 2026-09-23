# N-GPU P2P check: pairwise peer-copy bandwidth matrix (with data check) + RCCL all-reduce over every GPU.
import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp

def copy_matrix(n, mb=256, it=10):
    print(f"peer copy GB/s ({mb}MB, row=src col=dst), peer access:")
    bufs = [torch.arange(mb * 1024 * 1024 // 4, dtype=torch.int32, device=f"cuda:{i}") for i in range(n)]
    for s in range(n):
        row = []
        for d in range(n):
            if s == d: row.append("   -  "); continue
            dst = torch.empty_like(bufs[s], device=f"cuda:{d}")
            dst.copy_(bufs[s]); torch.cuda.synchronize(d)
            assert torch.equal(dst.cpu()[:1 << 20], bufs[s].cpu()[:1 << 20]), f"bad copy {s}->{d}"
            t = time.time()
            for _ in range(it): dst.copy_(bufs[s])
            torch.cuda.synchronize(s); torch.cuda.synchronize(d)
            row.append(f"{mb / 1024 * it / (time.time() - t):6.2f}{'' if torch.cuda.can_device_access_peer(s, d) else '*'}")
        print(f"  {s}: " + " ".join(row), flush=True)

def run(rank, n):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29532")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=n)
    want = n * (n + 1) / 2
    for mb in (1, 16, 256):
        x = torch.full((mb * 1024 * 1024 // 2,), float(rank + 1), dtype=torch.bfloat16, device="cuda")
        dist.all_reduce(x); torch.cuda.synchronize()
        assert torch.all(x == want), f"bad result {x[:4]}"
        t = time.time(); it = 20
        for _ in range(it): dist.all_reduce(x)
        torch.cuda.synchronize(); dt = (time.time() - t) / it
        if rank == 0:  # ring bus bandwidth = algbw * 2(n-1)/n
            print(f"allreduce x{n} {mb}MB ok  {dt*1e3:.3f} ms  busbw {mb/1024/dt*2*(n-1)/n:.2f} GB/s", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    n = torch.cuda.device_count()
    print("gpus", n, "nccl", torch.cuda.nccl.version(), flush=True)
    copy_matrix(n)
    mp.spawn(run, args=(n,), nprocs=n)
