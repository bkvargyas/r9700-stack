import time, torch
# Concurrent 0->1 and 1->0 peer copies on separate streams, vs each alone; long enough to sample link state.
a0 = torch.empty(256*1024*1024, dtype=torch.uint8, device="cuda:0"); a1 = torch.empty_like(a0, device="cuda:1")
b0 = torch.empty_like(a0); b1 = torch.empty_like(a1)
s0 = torch.cuda.Stream("cuda:0"); s1 = torch.cuda.Stream("cuda:1")
def go(dirs, it=150):
    torch.cuda.synchronize("cuda:0"); torch.cuda.synchronize("cuda:1"); t = time.time()
    for _ in range(it):
        if "01" in dirs:
            with torch.cuda.stream(s0): a1.copy_(a0, non_blocking=True)
        if "10" in dirs:
            with torch.cuda.stream(s1): b0.copy_(b1, non_blocking=True)
    torch.cuda.synchronize("cuda:0"); torch.cuda.synchronize("cuda:1")
    return 0.25 * it / (time.time() - t)
a0.copy_(torch.randint(0, 256, a0.shape, dtype=torch.uint8, device="cuda:0"))
a1.copy_(torch.randint(0, 256, a1.shape, dtype=torch.uint8, device="cuda:1"))
b1.copy_(a1); b0.zero_()
go(["01", "10"], 10)
for d in (["01"], ["10"], ["01", "10"]):
    bw = go(d)
    print(f"{'+'.join(d):6s} per-direction {bw:.2f} GB/s  total {bw*len(d):.2f} GB/s", flush=True)
ok01 = torch.equal(a1.cpu(), a0.cpu()); ok10 = torch.equal(b0.cpu(), b1.cpu())
print(f"data check 0->1 {'OK' if ok01 else 'CORRUPT'}  1->0 {'OK' if ok10 else 'CORRUPT'}", flush=True)
