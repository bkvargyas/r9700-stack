"""r9k PLE int6 gather+dequant vs a torch reference of the fused int6 row format (gfx1201)."""
import sys
import torch
from r9700_vllm.kernels.ple import gather_int6, int6_row_bytes

def ref_dequant(fused, head_dim):
    pb = head_dim * 6 // 8
    lead = fused.shape[:-1]
    packed = fused[..., :pb].reshape(*lead, -1, 3).to(torch.int32)
    word = packed[..., 0] | (packed[..., 1] << 8) | (packed[..., 2] << 16)
    codes = torch.stack([(word >> (6 * i)) & 0x3F for i in range(4)], dim=-1)
    codes = (codes - 32).reshape(*lead, head_dim // 32, 32)
    scale = fused[..., pb:].contiguous().view(torch.float16)
    return (codes.float() * scale.float().unsqueeze(-1)).reshape(*lead, head_dim)

def make_rows(rows, head_dim, g):
    codes = torch.randint(1, 64, (rows, head_dim), generator=g, dtype=torch.int32)   # stored codes (value+32)
    w = codes.reshape(rows, -1, 4)
    word = w[..., 0] | (w[..., 1] << 6) | (w[..., 2] << 12) | (w[..., 3] << 18)
    packed = torch.stack([(word >> (8 * i)) & 0xFF for i in range(3)], dim=-1).reshape(rows, -1).to(torch.uint8)
    scale = (torch.rand(rows, head_dim // 32, generator=g) * 0.02 + 1e-3).to(torch.float16)
    return torch.cat([packed, scale.view(torch.uint8).reshape(rows, -1)], dim=1).contiguous()

g = torch.Generator().manual_seed(0)
ok = True
for head_dim, rows, n in [(160, 5000, 37), (160, 100000, 4096), (256, 1000, 5)]:
    table = make_rows(rows, head_dim, g)
    assert table.shape[1] == int6_row_bytes(head_dim)
    ids = torch.randint(0, rows, (n, 16), generator=g)
    exp = ref_dequant(table[ids], head_dim)
    for where in ("device", "uva"):
        if where == "device":
            t = table.cuda()
        else:
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
            host = table.pin_memory()
            t = get_accelerator_view_from_cpu_tensor(host)
        got = gather_int6(t, ids.cuda(), head_dim).float().cpu()
        rel = ((got - exp).norm() / exp.norm()).item()
        good = rel < 5e-3
        ok &= good
        print(f"  head_dim={head_dim} rows={rows} ids={tuple(ids.shape)} {where:6s} rel {rel:.2e} {'ok' if good else '<-- FAIL'}")
print("ALL OK" if ok else "FAILURES"); sys.exit(0 if ok else 1)
