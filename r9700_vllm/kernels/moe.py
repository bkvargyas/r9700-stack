"""Python bindings for libr9k's grouped MXFP4 x FP8 MoE GEMM (gfx1201).

Weights come from a compressed-tensors ``mxfp4-pack-quantized`` checkpoint: per expert
``weight_packed [N, K/2] uint8`` (element 2i in the low nibble, 2i+1 in the high nibble, e2m1 codes) and
``weight_scale [N, K/32] uint8`` (E8M0). ``prepare_mxfp4_weights`` converts them once at load to the layout
the kernel reads; see kernels/r9k_moe_mxfp4a8.hip for the format.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch

GROUP = 32
MOE_BLOCK = 16

_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = os.environ.get("R9K_LIB") or os.path.join(os.path.dirname(__file__), "libr9k.so")
        L = ctypes.CDLL(path)
        L.r9k_moe_mxfp4a8.restype = ctypes.c_int
        L.r9k_moe_mxfp4a8.argtypes = [ctypes.c_long] * 12 + [ctypes.c_int] * 9 + [ctypes.c_long]
        L.r9k_quant_rows_fp8.restype = ctypes.c_int
        L.r9k_quant_rows_fp8.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long]
        L.r9k_silu_mul_quant_fp8.restype = ctypes.c_int
        L.r9k_silu_mul_quant_fp8.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 2 + [ctypes.c_long]
        assert L.r9k_moe_block() == MOE_BLOCK
        _LIB = L
    return _LIB


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


@dataclass
class Mxfp4Experts:
    wq: torch.Tensor    # [E, N/16 * K/16 * 32] int32, fragment order
    wsr: torch.Tensor   # [E, K/32 + 1, N] uint8: E8M0 exponents K-block-major, last row = per-row reference
    N: int
    K: int

    @property
    def estride(self) -> int:
        return (self.K // GROUP + 1) * self.N


def permute_fragments(packed: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] uint8 checkpoint order -> [E, N/16*K/16*32] int32 fragment order.

    Slot l of tile (nt, ks) holds W[nt*16 + (l & 15)][ks*16 + 8*(l >> 4) .. +8] (4 packed bytes);
    identical to libr4d mxfp4_layout.permute_w, vectorized over experts.
    """
    E, N, Kh = packed.shape
    K = Kh * 2
    nt, ks = N // 16, K // 16
    w = packed.reshape(E, nt, 16, ks, 2, 4)            # [E, nt, r, ks, h, 4 bytes]
    w = w.permute(0, 1, 3, 4, 2, 5).contiguous()       # [E, nt, ks, h, r, 4] -> lane = h*16 + r
    return w.reshape(E, -1).view(torch.int32)


def prepare_mxfp4_weights(packed: torch.Tensor, scale: torch.Tensor) -> Mxfp4Experts:
    E, N, Kh = packed.shape
    K = Kh * 2
    assert N % 16 == 0 and K % GROUP == 0, (N, K)
    assert scale.shape == (E, N, K // GROUP), (scale.shape, (E, N, K // GROUP))
    return Mxfp4Experts(permute_fragments(packed.view(torch.uint8)), pack_scales(scale), N, K)


def pack_scales(scale: torch.Tensor) -> torch.Tensor:
    """[E, N, K/32] E8M0 -> [E, K/32 + 1, N]: transposed block exponents plus the per-row max as the last row."""
    scale = scale.view(torch.uint8)
    E, N, nb = scale.shape
    out = torch.empty((E, nb + 1, N), dtype=torch.uint8, device=scale.device)
    out[:, :nb].copy_(scale.transpose(1, 2))
    out[:, nb] = scale.max(dim=2).values
    return out


def quant_rows_fp8(x: torch.Tensor):
    """bf16 [M, K] -> (e4m3fn [M, K], fp32 [M]) with a per-row dynamic scale."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((M,), dtype=torch.float32, device=x.device)
    rc = lib().r9k_quant_rows_fp8(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, x.stride(0), _stream())
    if rc:
        raise RuntimeError(f"r9k_quant_rows_fp8 failed ({rc})")
    return q, s


def silu_mul_quant_fp8(gu: torch.Tensor):
    """bf16 [M, 2I] (gate | up) -> (e4m3fn [M, I], fp32 [M]): silu(gate) * up with a per-row dynamic scale."""
    assert gu.dtype == torch.bfloat16 and gu.is_contiguous() and gu.shape[1] % 2 == 0
    M, I2 = gu.shape
    q = torch.empty((M, I2 // 2), dtype=torch.float8_e4m3fn, device=gu.device)
    s = torch.empty((M,), dtype=torch.float32, device=gu.device)
    rc = lib().r9k_silu_mul_quant_fp8(gu.data_ptr(), q.data_ptr(), s.data_ptr(), M, I2 // 2, _stream())
    if rc:
        raise RuntimeError(f"r9k_silu_mul_quant_fp8 failed ({rc})")
    return q, s


def moe_gemm(a_q: torch.Tensor, a_s: torch.Tensor, w: Mxfp4Experts, out: torch.Tensor,
             sorted_ids: torch.Tensor, expert_ids: torch.Tensor, ntpp: torch.Tensor, numel: int,
             a_row_div: int, topk_w: torch.Tensor | None = None, WV: int = 2, SK: int = 4, NPW: int = 2,
             num_experts: int | None = None, MT: int = 1):
    """out[r, :] = dequant(a_q[r // a_row_div]) @ W[expert(r)]^T (* topk_w[r]) for every routed flat row r.

    sorted_ids/expert_ids must come from moe_align_block_size(block_size=16*MT). MT M-tiles per workgroup share
    each weight fragment (use MT>1 when experts see many rows: prefill / wide batches).
    The grid covers at most ceil(numel/BLK) + min(numel, E) blocks -- the most moe_align_block_size can fill
    with numel routed rows over E experts -- instead of the buffer's worst-case capacity; host-known, so the
    launch stays cudagraph-safe."""
    assert out.dtype == torch.bfloat16 and out.shape[1] == w.N and a_q.shape[1] == w.K
    E = num_experts if num_experts is not None else w.wq.shape[0]
    blk = MOE_BLOCK * MT
    max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
    rc = lib().r9k_moe_mxfp4a8(
        a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.wsr.data_ptr(),
        w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
        sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
        topk_w.data_ptr() if topk_w is not None else 0,
        w.estride, w.estride, max_blocks, numel, a_row_div, w.K, w.N, WV, SK, NPW, MT, _stream())
    if rc:
        raise RuntimeError(f"r9k_moe_mxfp4a8 failed ({rc}) N={w.N} K={w.K} WV={WV} SK={SK} NPW={NPW} MT={MT}")
    return out


def pick_cfg(N: int, K: int) -> tuple[int, int, int]:
    """A legal (WV, SK, NPW) for any N % 16 == 0, K % 32 == 0: deep split-K for long K (weight streaming with few
    N tiles per wave), shallow for short K. Mirrors the tuned Flash-Next defaults (2,4,2) @K=2560, (4,2,1) @K=320."""
    for WV, SK, NPW in ((2, 4, 2), (4, 2, 1), (2, 5, 2), (4, 1, 1)):
        if K % (SK * GROUP) == 0 and (K >= 1024 or SK <= 2):
            return WV, SK, NPW
    return 4, 1, 1


def pick_mt(numel: int, num_experts: int) -> int:
    """M tiles per routing block from the host-known row count: rows per touched expert >= 32 -> 4, >= 16 -> 2."""
    per = numel / max(1, min(num_experts, numel))
    return 4 if per >= 32 else (2 if per >= 16 else 1)


def align_block_size_ref(topk_ids: torch.Tensor, num_experts: int, block: int = MOE_BLOCK):
    """Pure-torch equivalent of vLLM's moe_align_block_size (for tests): pads each expert's rows to a
    multiple of `block`, pad value = numel. Returns (sorted_ids, expert_ids, ntpp) sized to capacity."""
    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()
    cap = numel + num_experts * (block - 1)
    cap = (cap + block - 1) // block * block
    sorted_ids = torch.full((cap,), numel, dtype=torch.int32)
    expert_ids = torch.full((cap // block,), -1, dtype=torch.int32)
    pos = 0
    for e in range(num_experts):
        rows = (flat == e).nonzero().flatten()
        if rows.numel() == 0:
            continue
        n = (rows.numel() + block - 1) // block * block
        sorted_ids[pos:pos + rows.numel()] = rows.to(torch.int32)
        expert_ids[pos // block:(pos + n) // block] = e
        pos += n
    return sorted_ids, expert_ids, torch.tensor([pos], dtype=torch.int32)
