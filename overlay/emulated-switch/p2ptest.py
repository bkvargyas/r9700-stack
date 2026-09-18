import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp

def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    if rank == 0:
        print("nccl version", torch.cuda.nccl.version(), "peer access", torch.cuda.can_device_access_peer(0, 1), flush=True)
    for mb in (1, 16, 256):
        n = mb * 1024 * 1024 // 2
        x = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device="cuda")
        dist.all_reduce(x); torch.cuda.synchronize()
        assert torch.all(x == 3), f"bad result {x[:4]}"
        t = time.time(); it = 20
        for _ in range(it): dist.all_reduce(x)
        torch.cuda.synchronize(); dt = (time.time() - t) / it
        if rank == 0:
            print(f"allreduce {mb}MB ok  {dt*1e3:.3f} ms  busbw {mb/1024/dt:.2f} GB/s", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
