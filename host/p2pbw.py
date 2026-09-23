import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp
# Longer run than p2ptest.py so link state can be sampled mid-transfer; plus 1-way copy bandwidth.
def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29532")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    n = 256 * 1024 * 1024 // 2
    x = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device="cuda")
    dist.all_reduce(x); torch.cuda.synchronize()
    t = time.time(); it = 200
    for _ in range(it): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.time() - t) / it
    if rank == 0: print(f"allreduce 256MB x{it}: busbw {0.25/dt:.2f} GB/s", flush=True)
    # one-way send/recv 0 -> 1
    dist.barrier(); torch.cuda.synchronize(); t = time.time()
    for _ in range(100):
        if rank == 0: dist.send(x, 1)
        else: dist.recv(x, 0)
    torch.cuda.synchronize(); dt = (time.time() - t) / 100
    if rank == 0: print(f"send 0->1 256MB: {0.25/dt:.2f} GB/s", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
    # direct peer copy (no RCCL): device-to-device memcpy
    a = torch.empty(256*1024*1024, dtype=torch.uint8, device="cuda:0"); b = torch.empty_like(a, device="cuda:1")
    for src, dst, name in ((a, b, "0->1"), (b, a, "1->0")):
        dst.copy_(src); torch.cuda.synchronize(); t = time.time()
        for _ in range(50): dst.copy_(src)
        torch.cuda.synchronize(); print(f"memcpy {name} 256MB: {0.25/((time.time()-t)/50):.2f} GB/s", flush=True)
