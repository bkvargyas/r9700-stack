"""DeepSeek-V4.1 manifold hyper-connections at decode widths (kernels/r9k_mhc.hip) in place of stock's TileLang seam.

On RDNA4 stock's ``MHCPreDelayedOp.forward_hip`` runs each seam (two a layer) as three TileLang launches -- mhc_post,
hc_prenorm_gemm (the [M, 20480] x [20480, 24] f32 GEMV on two workgroups at M=1) and mhc_pre_big_fuse -- 27 us at
M=1, 39 us at M=24 in graph replay on an R9700. Here a seam is one launch (the post folded into the pre, the GEMV
split over 80 chunks, a ticketed fixed-order reduction, gates and sinkhorn in the last workgroup): 8.0 us at M=1,
10.1 at M=6, 14.2 at M=24; the standalone ``MHCPostOp`` one elementwise launch (3.7 us vs 9.8). Rows up to r9k_mhc_max_tokens() (32; R9K_DSV41_MHC_MAX_M lowers it); wider
calls, a fused RMSNorm weight (gfx950's seam only), odd shapes and non-contiguous inputs run stock's ops per call.

Numerics against model_executor/kernels/mhc/torch.py: the new residual and the collapsed layer input are bit-exact
(torch's rounding points and summation orders on this GPU); the post / comb / pre gates are f32 from a 20480-long
dot product in a different order than torch's matmul (as stock's TileLang GEMM) with the sinkhorn's divisions by a
refined reciprocal: ~1e-5 relative to torch. Deterministic and
batch-invariant: a token's results do not depend on the other rows of the call.

Registered as torch.ops.r9700.dsv41_mhc_pre / dsv41_mhc_post_pre / dsv41_mhc_post (the callers sit inside the
compiled model graph); bound over the CustomOps' forward_hip by models/deepseek_v41_rdna4.py (R9K_DSV41_MHC=r9k|stock).
"""
from __future__ import annotations

import ctypes
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger("vllm." + __name__)

_L = None
_DONE = False
_SCRATCH: dict = {}
HC, NMIX = 4, 24


def lib():
    global _L
    if _L is None:
        from .kernels import moe as KM
        L = KM.lib()
        L.r9k_mhc_seam.restype = ctypes.c_int
        L.r9k_mhc_seam.argtypes = [ctypes.c_long] * 16 + [ctypes.c_int] * 4 + [ctypes.c_float] * 4 + [ctypes.c_long]
        L.r9k_mhc_post.restype = ctypes.c_int
        L.r9k_mhc_post.argtypes = [ctypes.c_long] * 5 + [ctypes.c_int] * 2 + [ctypes.c_long]
        L.r9k_mhc_max_tokens.restype = ctypes.c_int
        L.r9k_mhc_scratch_floats.restype = ctypes.c_long
        L.r9k_mhc_scratch_floats.argtypes = [ctypes.c_int]
        L.r9k_mhc_scratch_ctrs.restype = ctypes.c_int
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_mhc_seam")
    except Exception:
        return False


def max_m() -> int:
    return min(int(os.environ.get("R9K_DSV41_MHC_MAX_M", "32")), lib().r9k_mhc_max_tokens())


_MAXM = None


def _maxm() -> int:
    global _MAXM
    if _MAXM is None:
        _MAXM = max_m()
    return _MAXM


def _scratch(device: torch.device, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(device, H) chunk partials + per-token-group arrival counters; the kernel leaves the counters at zero, so
    one set serves every call on the stream (decode is single-stream)."""
    key = (device.index, H)
    s = _SCRATCH.get(key)
    if s is None:
        L = lib()
        s = (torch.empty(L.r9k_mhc_scratch_floats(H), dtype=torch.float32, device=device),
             torch.zeros(L.r9k_mhc_scratch_ctrs(), dtype=torch.int32, device=device))
        _SCRATCH[key] = s
    return s


def _f32c(t: torch.Tensor | None) -> bool:
    return t is None or (t.dtype == torch.float32 and t.is_contiguous())


def _fits(residual, fn, hc_scale, hc_base, pre_mix, x, norm_weight, sublayer_out, post_layer_mix,
          comb_res_mix) -> bool:
    if norm_weight is not None or residual.dim() != 3 or residual.dtype != torch.bfloat16 \
            or not residual.is_contiguous():
        return False
    T, hc, H = residual.shape
    if not (0 < T <= _maxm()) or hc != HC or H % 512:
        return False
    if not (_f32c(fn) and _f32c(hc_scale) and _f32c(hc_base) and _f32c(pre_mix)):
        return False
    K = H if x is not None else HC * H
    if fn.shape != (NMIX, K) or hc_scale.numel() != 3 or hc_base.numel() != NMIX:
        return False
    if pre_mix is not None and pre_mix.shape != (T, HC):
        return False
    if x is not None and (sublayer_out is not None or x.shape != (T, H) or x.dtype != torch.bfloat16
                          or not x.is_contiguous()):
        return False
    if sublayer_out is not None:
        if sublayer_out.shape != (T, H) or sublayer_out.dtype != torch.bfloat16 or not sublayer_out.is_contiguous():
            return False
        if not (_f32c(post_layer_mix) and _f32c(comb_res_mix)) or post_layer_mix.numel() != T * HC \
                or comb_res_mix.numel() != T * HC * HC:
            return False
    return True


def _seam(residual, sublayer_out, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
          hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, x):
    T, _, H = residual.shape
    dev = residual.device
    res_out = torch.empty_like(residual) if sublayer_out is not None else None
    post = torch.empty((T, HC, 1), dtype=torch.float32, device=dev)
    comb = torch.empty((T, HC, HC), dtype=torch.float32, device=dev)
    li = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    nxt = torch.empty((T, HC), dtype=torch.float32, device=dev)
    part, ctr = _scratch(dev, H)

    def p(t):
        return 0 if t is None else t.data_ptr()

    rc = lib().r9k_mhc_seam(residual.data_ptr(), p(sublayer_out), p(post_layer_mix), p(comb_res_mix), p(x),
                            fn.data_ptr(), hc_scale.data_ptr(), hc_base.data_ptr(), p(pre_mix), p(res_out),
                            post.data_ptr(), comb.data_ptr(), li.data_ptr(), nxt.data_ptr(), part.data_ptr(),
                            ctr.data_ptr(), T, H, int(fn.shape[1]), int(sinkhorn_repeat), float(rms_eps),
                            float(hc_pre_eps), float(hc_sinkhorn_eps), float(hc_post_mult_value),
                            torch.cuda.current_stream(dev).cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_mhc_seam failed ({rc}) T={T} H={H} K={fn.shape[1]}")
    return res_out, post, comb, li, nxt


def _stock_post(x, residual, post_layer_mix, comb_res_mix):
    from vllm.model_executor.layers import mhc as M
    if M.HAS_TILELANG_MHC:
        return torch.ops.vllm.mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
    import importlib
    return importlib.import_module("vllm.model_executor.kernels.mhc.torch").mhc_post_torch(
        x, residual, post_layer_mix, comb_res_mix)


def _stock_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
               sinkhorn_repeat, pre_mix, x, norm_weight, norm_eps):
    from vllm.model_executor.layers import mhc as M
    if M.HAS_TILELANG_MHC:
        return torch.ops.vllm.mhc_pre_delayed_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                                                       hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                                                       pre_mix, x, norm_weight, norm_eps)
    import importlib
    T = importlib.import_module("vllm.model_executor.kernels.mhc.torch")
    post, comb, li, nxt = T.mhc_pre_delayed_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                                                  hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                                                  pre_mix=pre_mix, x=x)
    li = M._apply_mhc_norm(li, norm_weight, norm_eps)
    return post, comb, li.clone() if li.data_ptr() == residual.data_ptr() else li, nxt


def mhc_pre(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
            rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
            sinkhorn_repeat: int, pre_mix: torch.Tensor | None = None, x: torch.Tensor | None = None,
            norm_weight: torch.Tensor | None = None, norm_eps: float = 1e-6
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """mhc_pre_delayed without a post: (post [T,4,1], comb [T,4,4], layer_input [T,H], next_pre [T,4])."""
    if not _fits(residual, fn, hc_scale, hc_base, pre_mix, x, norm_weight, None, None, None):
        return _stock_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                          hc_post_mult_value, sinkhorn_repeat, pre_mix, x, norm_weight, norm_eps)
    _, post, comb, li, nxt = _seam(residual, None, None, None, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, x)
    return post, comb, li, nxt


def mhc_post_pre(residual: torch.Tensor, sublayer_out: torch.Tensor, post_layer_mix: torch.Tensor,
                 comb_res_mix: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                 rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
                 sinkhorn_repeat: int, pre_mix: torch.Tensor | None = None, norm_weight: torch.Tensor | None = None,
                 norm_eps: float = 1e-6
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The seam: mhc_post into the streams, then mhc_pre_delayed on the result:
    (residual' [T,4,H], post, comb, layer_input, next_pre)."""
    if not _fits(residual, fn, hc_scale, hc_base, pre_mix, None, norm_weight, sublayer_out, post_layer_mix,
                 comb_res_mix):
        r = _stock_post(sublayer_out, residual, post_layer_mix, comb_res_mix)
        return (r, *_stock_pre(r, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                               sinkhorn_repeat, pre_mix, None, norm_weight, norm_eps))
    return _seam(residual, sublayer_out, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                 hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix, None)


def mhc_post(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor,
             comb_res_mix: torch.Tensor) -> torch.Tensor:
    """out_j = sum_i comb[i, j] * residual_i + post_j * x, bf16 (MHCPostOp)."""
    ok = residual.dim() == 3 and residual.dtype == torch.bfloat16 and residual.is_contiguous() \
        and 0 < residual.shape[0] <= _maxm() and residual.shape[1] == HC and residual.shape[2] % 512 == 0 \
        and x.shape == (residual.shape[0], residual.shape[2]) and x.dtype == torch.bfloat16 and x.is_contiguous() \
        and _f32c(post_layer_mix) and _f32c(comb_res_mix) and post_layer_mix.numel() == residual.shape[0] * HC \
        and comb_res_mix.numel() == residual.shape[0] * HC * HC
    if not ok:
        return _stock_post(x, residual, post_layer_mix, comb_res_mix)
    T, _, H = residual.shape
    out = torch.empty_like(residual)
    rc = lib().r9k_mhc_post(residual.data_ptr(), x.data_ptr(), post_layer_mix.data_ptr(), comb_res_mix.data_ptr(),
                            out.data_ptr(), T, H, torch.cuda.current_stream(residual.device).cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_mhc_post failed ({rc}) T={T} H={H}")
    return out


def _pre_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
              sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6):
    T, hc, H = residual.shape
    f = dict(dtype=torch.float32, device=residual.device)
    return (torch.empty((T, hc, 1), **f), torch.empty((T, hc, hc), **f),
            torch.empty((T, H), dtype=torch.bfloat16, device=residual.device), torch.empty((T, hc), **f))


def _post_pre_fake(residual, sublayer_out, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
                   hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None,
                   norm_weight=None, norm_eps=1e-6):
    return (torch.empty_like(residual), *_pre_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                                                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat))


def _post_fake(x, residual, post_layer_mix, comb_res_mix):
    return torch.empty_like(residual)


def register() -> None:
    global _DONE
    if _DONE:
        return
    from .ops import _LIB
    direct_register_custom_op("dsv41_mhc_pre", mhc_pre, mutates_args=[], fake_impl=_pre_fake, target_lib=_LIB)
    direct_register_custom_op("dsv41_mhc_post_pre", mhc_post_pre, mutates_args=[], fake_impl=_post_pre_fake,
                              target_lib=_LIB)
    direct_register_custom_op("dsv41_mhc_post", mhc_post, mutates_args=[], fake_impl=_post_fake, target_lib=_LIB)
    _DONE = True


def pre_delayed_forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                            hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None,
                            norm_eps=1e-6, sublayer_out=None, post_layer_mix=None, comb_res_mix=None):
    """MHCPreDelayedOp.forward_hip on RDNA4: returns residual, post_mix, comb_mix, layer_input, next_pre_mix."""
    if sublayer_out is None:
        return residual, *torch.ops.r9700.dsv41_mhc_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                                                         hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                                                         pre_mix, x, norm_weight, norm_eps)
    assert x is None and post_layer_mix is not None and comb_res_mix is not None
    return torch.ops.r9700.dsv41_mhc_post_pre(residual, sublayer_out, post_layer_mix, comb_res_mix, fn, hc_scale,
                                              hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                                              sinkhorn_repeat, pre_mix, norm_weight, norm_eps)


def post_forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
    return torch.ops.r9700.dsv41_mhc_post(x, residual, post_layer_mix, comb_res_mix)
