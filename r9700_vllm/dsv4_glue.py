"""DeepSeek-V4.1 decode-width glue on our kernels (kernels/r9k_dsv4_glue.hip): RMSNorm + cast (+ rotation) pre-pass and
the one-launch router. The hooks that use them live in models/deepseek_v41_rdna4.py; numbers in notes/exl3-glue.md.

* ``norm_pre(segs, M, eps, mode)``: up to two row segments in one launch, each [M, n] bf16 (or the f32 C buffer of an
  EXL3 GEMV, rounded to bf16 first) -> RMSNorm (optional) -> bf16 rows and/or f16 rows (the EXL3 GEMV input), the f16
  rows optionally rotated with ``suh`` exactly as exllamav3's GEMV rotates its input (for a GEMV that takes rotated
  input; package D's interface: ``prepass(x bf16 [M, K], weight or None, suh f16 [K]) -> f16 [M, K] rotated``).
* ``route(x, gate_w, bias, ...)``: [ffn_norm +] gate GEMV + sqrt-softplus top-k + renormalize + routed scaling ->
  (topk_weights f32 [M, k], topk_ids int32 [M, k][, normalized x bf16]).

``mode`` = the stock norm's rounding: 0 = vLLM IR native (bf16(bf16(x * rrms) * w)), 1 = one cast of x * rrms * w
(stock fused_q_kv_rmsnorm / csrc). NORM_MODE (R9K_DSV41_NORM_MODE) is what the stock RMSNorm module runs on this pin
(tests/test_dsv4_glue.py prints which mode matches it bit for bit).
Every call is graph-safe: no host syncs; the router's scratch (logits + one arrival counter) is per device and the
counter is left at zero by the kernel.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_L = None
_SCR: dict = {}
MAX_M = 32
# split-K GEMV scratch: (K / 512) slices x MAX_M rows x 48 groups x 64 slots at K <= 6144
SK_PART_FLOATS = 12 * MAX_M * 48 * 64
NORM_MODE = int(os.environ.get("R9K_DSV41_NORM_MODE", "1"))
# R9K_DSV41_SHARED_FUSE=1 (default): the router launch also runs the shared expert's gate_up + activation
SHARED_FUSE = os.environ.get("R9K_DSV41_SHARED_FUSE", "1") == "1"
# Row cutoffs (measured on an R9700, tests/test_dsv4_glue.py timing): above them the hooks run stock's ops.
# ffn front (norm + router + shared gate_up + act): 21 / 25 / 31 / 34 us at 1 / 4 / 6 / 8 rows vs stock's 39; 49 at 12.
ROUTE_MAX_M = int(os.environ.get("R9K_DSV41_ROUTE_MAX_M", "8"))
# attention input (attn_norm + kv_score + weights_proj): 13-24 us at 1-6 rows vs stock's 27-29; level at 8.
ATTN_IN_MAX_M = int(os.environ.get("R9K_DSV41_ATTN_IN_MAX_M", "6"))


def lib():
    global _L
    if _L is None:
        path = os.environ.get("R9K_LIB") or os.path.join(os.path.dirname(__file__), "kernels", "libr9k.so")
        L = ctypes.CDLL(path)
        if not hasattr(L, "r9k_dsv4_norm_pre"):
            raise RuntimeError(f"{path} has no r9k_dsv4_norm_pre (rebuild kernels/)")
        seg = [ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_int, ctypes.c_long,
               ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        L.r9k_dsv4_norm_pre.restype = ctypes.c_int
        L.r9k_dsv4_norm_pre.argtypes = seg + seg + [ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_long]
        L.r9k_dsv4_markov_step.restype = ctypes.c_int
        L.r9k_dsv4_markov_step.argtypes = [ctypes.c_long] * 15 + [ctypes.c_int] * 4 + [ctypes.c_long]
        L.r9k_dsv4_markov_last.restype = ctypes.c_int
        L.r9k_dsv4_markov_last.argtypes = [ctypes.c_long] * 4 + [ctypes.c_int] * 3 + [ctypes.c_long]
        L.r9k_dsv4_markov_blocks.restype = ctypes.c_int
        if hasattr(L, "r9k_dsv4_markov_conf"):
            L.r9k_dsv4_markov_conf.restype = ctypes.c_int
            L.r9k_dsv4_markov_conf.argtypes = ([ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_long, ctypes.c_int]
                                               + [ctypes.c_long] * 8 + [ctypes.c_int] * 2 + [ctypes.c_long])
        L.r9k_dsv4_markov_blocks.argtypes = [ctypes.c_int]
        L.r9k_dsv4_sk.restype = ctypes.c_int
        L.r9k_dsv4_sk.argtypes = ([ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_float, ctypes.c_int,
                                   ctypes.c_long, ctypes.c_int, ctypes.c_int]
                                  + [ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_long] * 3
                                  + [ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_long,
                                     ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_float, ctypes.c_long,
                                     ctypes.c_long, ctypes.c_long, ctypes.c_long])
        L.r9k_dsv4_glue_max_m.restype = ctypes.c_int
        _L = L
    return _L


def available() -> bool:
    try:
        lib()
        return True
    except Exception:
        return False


def _p(t):
    return 0 if t is None else t.data_ptr()


@dataclass
class Seg:
    """One row segment of norm_pre. inp: [M, n] bf16 or f32 with unit column stride (any row stride)."""
    inp: torch.Tensor
    weight: torch.Tensor | None = None      # bf16 [n]
    norm: bool = True
    out_bf: torch.Tensor | None = None      # bf16 [M, n], unit column stride
    out_h: torch.Tensor | None = None       # f16 [M, n], unit column stride
    suh: torch.Tensor | None = None         # f16 [n]: out_h rotated

    def args(self):
        x = self.inp
        n = x.shape[-1]
        return (x.data_ptr(), x.stride(0), int(x.dtype == torch.float32), n, _p(self.weight), int(self.norm),
                _p(self.out_bf), 0 if self.out_bf is None else self.out_bf.stride(0),
                _p(self.out_h), 0 if self.out_h is None else self.out_h.stride(0), _p(self.suh))


_NONE = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def norm_pre(segs: list[Seg], M: int, eps: float, mode: int | None = None) -> None:
    assert 1 <= len(segs) <= 2 and M >= 1
    for s in segs:
        x = s.inp
        assert x.dim() == 2 and x.shape[0] >= M and x.stride(1) == 1 and x.dtype in (torch.bfloat16, torch.float32)
        assert x.shape[1] % 128 == 0
        assert s.out_bf is None or (s.out_bf.dtype == torch.bfloat16 and s.out_bf.stride(-1) == 1)
        assert s.out_h is None or (s.out_h.dtype == torch.float16 and s.out_h.stride(-1) == 1)
        assert s.weight is None or (s.weight.dtype == torch.bfloat16 and s.weight.is_contiguous())
        assert s.suh is None or (s.suh.dtype == torch.float16 and s.suh.is_contiguous() and s.out_h is not None)
    a0 = segs[0].args()
    a1 = segs[1].args() if len(segs) > 1 else _NONE
    dev = segs[0].inp.device
    rc = lib().r9k_dsv4_norm_pre(*a0, *a1, M, float(eps), NORM_MODE if mode is None else mode,
                                 torch.cuda.current_stream(dev).cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_dsv4_norm_pre failed ({rc}) M={M} n={[s.inp.shape[-1] for s in segs]}")


def prepass(x: torch.Tensor, weight: torch.Tensor | None, suh: torch.Tensor | None, eps: float = 1e-6,
            out: torch.Tensor | None = None, mode: int | None = None) -> torch.Tensor:
    """x bf16 [M, K] -> f16 [M, K]: RMSNorm with ``weight`` (None: no norm, just the cast), then -- ``suh`` given --
    exllamav3's input rotation (had_in_128: x * suh in f16, 128-point Hadamard, / sqrt(128), f16). One launch."""
    M, K = x.shape
    if out is None:
        out = torch.empty((M, K), dtype=torch.float16, device=x.device)
    norm_pre([Seg(x, weight, weight is not None, None, out, suh)], M, eps, mode)
    return out


def norm_gemv(x: torch.Tensor, norm_w: torch.Tensor, eps: float, mats, mode: int | None = None):
    """RMSNorm rows of x bf16 [M, K] (M <= 32) and up to two GEMVs on the normalized rows in one launch:
    mats = [(W bf16 [rows, K] contiguous, out_f32: bool), ...] -> (xn bf16 [M, K], [out [M, rows] f32 or bf16])."""
    M, K = x.shape
    assert 1 <= M <= MAX_M and x.dtype == torch.bfloat16 and x.stride(1) == 1 and 1 <= len(mats) <= 2
    dev = x.device
    xn = torch.empty((M, K), dtype=torch.bfloat16, device=dev)
    outs, args = [], []
    for w, f32 in mats:
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.shape[1] == K
        o = torch.empty((M, w.shape[0]), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
        outs.append(o)
        args += [w.data_ptr(), w.shape[0], 2 if f32 else 3, o.data_ptr()]
    while len(args) < 12:
        args += [0, 0, 0, 0]
    _, _, part, gctr = _scratch(dev)
    rc = lib().r9k_dsv4_sk(x.data_ptr(), x.stride(0), norm_w.data_ptr(), float(eps), NORM_MODE if mode is None else mode,
                           xn.data_ptr(), K, M, *args, 0, 0, 0, 0, 0, 0, 1, 1, 1.0, 0, 0, 0, 0, 0.0, 0,
                           part.data_ptr(), gctr.data_ptr(), torch.cuda.current_stream(dev).cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_dsv4_sk (norm_gemv) failed ({rc}) M={M} K={K}")
    return xn, outs


def _scratch(dev: torch.device):
    s = _SCR.get(dev.index)
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("dsv4_glue: router scratch must exist before graph capture (call warm())")
        s = _SCR[dev.index] = (torch.empty(MAX_M * 512, dtype=torch.float32, device=dev),
                               torch.zeros(1, dtype=torch.int32, device=dev),
                               torch.empty(SK_PART_FLOATS, dtype=torch.float32, device=dev),
                               torch.zeros(48, dtype=torch.int32, device=dev))
    return s


def warm(dev: torch.device) -> None:
    _scratch(dev)


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, topk: int, renorm: bool, rsf: float,
          bias_vl: torch.Tensor | None = None, input_ids: torch.Tensor | None = None, sentinel_lo: int = 0,
          is_padding: torch.Tensor | None = None, norm_w: torch.Tensor | None = None, eps: float = 1e-6,
          mode: int | None = None, logits_out: bool = False,
          shared_w: torch.Tensor | None = None, limit: float = float("inf")):
    """x bf16 [M, K] (unit column stride; raw rows when norm_w is given), gate_w bf16 [E, K], bias f32 [E];
    shared_w (optional): the shared expert's gate_up weight bf16 [2 I, K] -> its activation (silu-and-mul with the
    swiglu clamp ``limit``) computed in the same launch.
    Returns (topk_weights f32 [M, topk], topk_ids int32 [M, topk], xn bf16 [M, K] or None, logits f32 [M, E] or None,
    act bf16 [M, I] or None)."""
    M, K = x.shape
    E = gate_w.shape[0]
    assert 1 <= M <= MAX_M and x.dtype == torch.bfloat16 and x.stride(1) == 1 and gate_w.is_contiguous()
    assert gate_w.dtype == torch.bfloat16 and gate_w.shape[1] == K and bias.dtype == torch.float32
    dev = x.device
    logits, ctr, part, gctr = _scratch(dev)
    w_out = torch.empty((M, topk), dtype=torch.float32, device=dev)
    id_out = torch.empty((M, topk), dtype=torch.int32, device=dev)
    xn = torch.empty((M, K), dtype=torch.bfloat16, device=dev) if norm_w is not None else None
    ids64 = 0
    if input_ids is not None:
        assert input_ids.is_contiguous() and input_ids.dtype in (torch.int32, torch.int64)
        ids64 = int(input_ids.dtype == torch.int64)
    if is_padding is not None:
        assert is_padding.dtype == torch.bool and is_padding.is_contiguous()
    act, I = None, 0
    if shared_w is not None:
        assert shared_w.dtype == torch.bfloat16 and shared_w.is_contiguous() and shared_w.shape[1] == K
        I = shared_w.shape[0] // 2
        act = torch.empty((M, I), dtype=torch.bfloat16, device=dev)
    rc = lib().r9k_dsv4_sk(x.data_ptr(), x.stride(0), _p(norm_w), float(eps), NORM_MODE if mode is None else mode,
                           _p(xn), K, M, gate_w.data_ptr(), E, 0, 0, _p(shared_w), 2 * I, 1, 0, 0, 0, 0, 0,
                           bias.data_ptr(), _p(bias_vl), _p(input_ids), ids64, int(sentinel_lo), _p(is_padding), topk,
                           int(renorm), float(rsf), logits.data_ptr(), w_out.data_ptr(), id_out.data_ptr(),
                           ctr.data_ptr(), float(limit), _p(act), part.data_ptr(), gctr.data_ptr(),
                           torch.cuda.current_stream(dev).cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_dsv4_sk (route) failed ({rc}) M={M} E={E} K={K} topk={topk}")
    return w_out, id_out, xn, (logits[:M * E].view(M, E).clone() if logits_out else None), act


# ================================================================================================================ hooks
# Bound by models/deepseek_v41_rdna4.py (each behind its own knob). They wrap stock modules per instance at
# construction (the classes are patched before the model is built), so a layer whose shapes do not fit keeps stock's
# path, and any call outside the decode widths (prefill) runs stock's ops unchanged.

def _padding_mask(M: int):
    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router as R
    return R._get_padding_mask(M)


class _RouteHook:
    """R9K_DSV41_ROUTE=r9k: ffn_norm + gate GEMV + sqrt-softplus top-k + renormalize + scaling in one launch
    (r9k_dsv4_sk). ffn_norm's forward returns the normalized rows the kernel wrote and leaves the routing on this
    object; the runner's gate call then returns a placeholder (no launch) and the router's _compute_routing returns
    the stashed (weights, ids). Matching is by the normalized tensor (data_ptr + shape), so any other caller of the
    gate / router runs stock's ops."""

    def __init__(self, layer):
        moe = layer.ffn
        runner = moe.experts
        self.norm = layer.ffn_norm
        self.gate = moe.gate
        self.router = getattr(runner, "router", None)
        self.layer = layer
        self.stash = None
        self.ids = None
        self.shared = getattr(moe, "shared_experts", None) if SHARED_FUSE else None
        self.act = None
        self.limit = float("inf")

    def ok(self) -> bool:
        r, g = self.router, self.gate
        if r is None or type(r).__name__ != "FusedTopKBiasRouter":
            return False
        if getattr(r, "scoring_func", None) != "sqrtsoftplus" or getattr(r, "_hash_indices_table", None) is not None:
            return False
        if getattr(r, "num_fused_shared_experts", 0) or r.e_score_correction_bias is None:
            return False
        if getattr(g, "tid2eid", None) is not None or getattr(g, "bias", None) is not None:
            return False
        w = getattr(g, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[0] % 32 or w.shape[0] > 512:
            return False
        if w.shape[1] % 128 or self.norm.weight.dtype != torch.bfloat16 or r.top_k > 8:
            return False
        return getattr(self.norm, "variance_size_override", None) is None

    def shared_ok(self) -> bool:
        """The shared expert (DeepseekV4MLP) folds in when its gate_up is a plain bf16 [2 I, K] weight (exl3 dense /
        unquantized), its activation silu-and-mul (with or without the swiglu clamp, alpha 1, beta 0)."""
        sh = self.shared
        if sh is None or not hasattr(sh, "gate_up_proj") or not hasattr(sh, "down_proj"):
            return False
        if getattr(sh, "_gateup_scale", None) is not None:
            return False
        gu = sh.gate_up_proj
        w = getattr(gu, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] != self.gate.weight.shape[1]:
            return False
        if w.shape[0] % 2 or getattr(gu, "bias", None) is not None or not w.is_contiguous():
            return False
        act = sh.act_fn
        name = type(act).__name__
        if name == "SiluAndMulWithClamp":
            if getattr(act, "alpha", 1.0) != 1.0 or getattr(act, "beta", 0.0) != 0.0:
                return False
            self.limit = float(act.swiglu_limit)
        elif name == "SiluAndMul":
            self.limit = float("inf")
        else:
            return False
        return True

    def norm_forward(self, x, residual=None):
        if residual is None and x.dim() == 2 and 0 < x.shape[0] <= min(MAX_M, ROUTE_MAX_M) \
                and x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 8 == 0 and self.ok():
            r = self.router
            M = x.shape[0]
            ids = self.ids
            if ids is not None and (ids.dim() != 1 or ids.shape[0] < M or not ids.is_contiguous()):
                ids = None
            bvl = r.bias_vl.data if getattr(r, "bias_vl", None) is not None else None
            lo = int(getattr(r, "image_sentinel_lo", 0) or 0)
            if bvl is not None and lo > 0 and ids is None:
                return self._orig_norm(x)            # vision routing needs the token ids
            if self.shared_on is None and not torch.cuda.is_current_stream_capturing():
                self.shared_on = self.shared_ok()       # weights exist only after loading
            sw = self.shared.gate_up_proj.weight if self.shared_on else None
            tw, ti, xn, _, act = route(x, self.gate.weight, r.e_score_correction_bias.data, r.top_k,
                                       bool(r.renormalize), float(r.routed_scaling_factor), bias_vl=bvl,
                                       input_ids=ids[:M] if ids is not None else None, sentinel_lo=lo,
                                       is_padding=_padding_mask(M), norm_w=self.norm.weight,
                                       eps=self.norm.variance_epsilon, shared_w=sw, limit=self.limit)
            self.stash = (xn.data_ptr(), tuple(xn.shape), tw, ti)
            self.act = (xn.data_ptr(), tuple(xn.shape), act) if act is not None else None
            return xn
        self.stash = self.act = None
        return self._orig_norm(x) if residual is None else self._orig_norm(x, residual)

    def _match(self, x) -> bool:
        s = self.stash
        return s is not None and x.data_ptr() == s[0] and tuple(x.shape) == s[1]

    def gate_forward(self, x):
        if self._match(x):
            out = torch.empty((x.shape[0], self.gate.weight.shape[0]), dtype=torch.float32, device=x.device)
            return out if not self.gate.return_bias else (out, None)
        return self._orig_gate(x)

    def compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        if self._match(hidden_states):
            _, _, tw, ti = self.stash
            self.stash = None
            if indices_type is not None and indices_type != torch.int32:
                ti = ti.to(indices_type)
            return tw, ti
        return self._orig_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)

    def shared_forward(self, x):
        a = self.act
        if a is not None and x.data_ptr() == a[0] and tuple(x.shape) == a[1]:
            self.act = None
            y, _ = self.shared.down_proj(a[2])
            return y
        return self._orig_shared(x)

    def bind(self) -> bool:
        if self.router is None or not hasattr(self.router, "_compute_routing"):
            return False
        self.shared_on = None if self.shared is not None else False
        if self.shared is not None:
            self._orig_shared = self.shared.forward
            self.shared.forward = self.shared_forward
        self._orig_norm = self.norm.forward
        self._orig_gate = self.gate.forward
        self._orig_routing = self.router._compute_routing
        self.norm.forward = self.norm_forward
        self.gate.forward = self.gate_forward
        self.router._compute_routing = self.compute_routing
        return True


def install_route() -> bool:
    """Patch DeepseekV4DecoderLayer (deepseek_v41/amd/model.py: target and DSpark drafter layers) so every layer built
    afterwards binds a _RouteHook, and so its forward records the step's input_ids for the router (bias_vl rows)."""
    from vllm.models.deepseek_v41.amd import model as Mod
    if not available():
        return False
    cls = Mod.DeepseekV4DecoderLayer
    if getattr(cls.forward, "__r9k_patch__", None) != "dsv41_route":
        fwd = cls.forward

        def forward(self, x, positions, input_ids, *args, **kwargs):
            h = self.__dict__.get("_r9k_route")
            if h is not None:
                h.ids = input_ids
            return fwd(self, x, positions, input_ids, *args, **kwargs)

        forward.__r9k_patch__ = "dsv41_route"
        forward.__r9k_orig__ = fwd
        cls.forward = forward

    def bind(layer):
        h = _RouteHook(layer)
        if h.bind():
            layer.__dict__["_r9k_route"] = h

    _add_layer_binder("route", bind)
    return True


class _QbStack:
    """R9K_DSV41_QB_STACK=1: the attention's wq_b and the indexer's wq_b read the same normalized qr [M, 1280]; on
    indexer layers both run as one r9k_exl3_gemv launch (package D's kernel, quant/exl3_gemv.py: per-part inputs and
    outputs) when the attention projects q, and the indexer's own wq_b call takes the stashed result. Same kernel, same
    per-part arithmetic, so the outputs are bit-identical to two separate calls."""

    def __init__(self, attn):
        self.attn = attn
        self.plan = None
        self.n0 = self.n1 = 0
        self.stash = None
        self.tried = False

    def _build(self):
        self.tried = True
        from .quant import exl3 as E
        a, ix = self.attn.wq_b, self.attn.indexer.wq_b
        if getattr(E, "GEMV", None) != "r9k" or getattr(a, "exl3_gemv", None) is None or getattr(ix, "exl3_gemv", None) is None:
            return
        if getattr(a, "exl3_dense", True) or getattr(ix, "exl3_dense", True) or a.exl3_k != ix.exl3_k:
            return
        if a.exl3_gemv.dev != ix.exl3_gemv.dev:
            return
        import dataclasses
        from .quant import exl3_gemv as XG
        parts = list(a.exl3_parts) + [dataclasses.replace(p, dst=p.dst + a.exl3_n) for p in ix.exl3_parts]
        plan = XG.Exl3GemvPlan(parts, a.exl3_k)
        plan.reserve(E.GEMV_M)
        self.plan, self.n0, self.n1, self.na = plan, a.exl3_n, ix.exl3_n, len(a.exl3_parts)
        self.gemv, self.max_m = XG.r9k_exl3_gemv, E.GEMV_M

    def project(self, qr):
        """q [M, n0] from the attention's wq_b; the indexer's [M, n1] stashed. None when the stack does not apply."""
        if not self.tried:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._build()
        if self.plan is None or not isinstance(qr, torch.Tensor) or qr.dim() != 2 or qr.dtype != torch.bfloat16 \
                or not 0 < qr.shape[0] <= self.max_m or qr.stride(1) != 1 or qr.stride(0) % 4 or qr.data_ptr() % 16:
            self.stash = None
            return None
        M = qr.shape[0]
        q = torch.empty((M, self.n0), dtype=qr.dtype, device=qr.device)
        iq = torch.empty((M, self.n1), dtype=qr.dtype, device=qr.device)
        na, np_ = self.na, len(self.plan.parts)
        outs = []
        for j, p in enumerate(self.plan.parts):
            outs.append(q[:, p.dst:p.dst + p.n] if j < na else iq[:, p.dst - self.n0:p.dst - self.n0 + p.n])
        self.gemv(self.plan, [qr] * np_, outs)
        self.stash = (qr.data_ptr(), tuple(qr.shape), iq)
        return q

    def take(self, qr):
        s = self.stash
        if s is not None and qr.data_ptr() == s[0] and tuple(qr.shape) == s[1]:
            self.stash = None
            return s[2]
        return None


def install_qb_stack() -> bool:
    from vllm.models.deepseek_v41 import attention as A
    from vllm.models.deepseek_v41.amd import rocm as R
    acls, icls = R.DeepseekV41ROCMAiterMLAAttention, A.DeepseekV4Indexer
    if getattr(acls._wq_b_proj, "__r9k_patch__", None) == "dsv41_qb_stack":
        return True
    a_orig, i_orig = acls._wq_b_proj, icls._wq_b_proj

    def attn_wq_b_proj(self, qr, qr_scale=None):
        if qr_scale is None and getattr(self, "indexer", None) is not None:
            st = self.__dict__.get("_r9k_qb")
            if st is None:
                st = self.__dict__["_r9k_qb"] = _QbStack(self)
                self.indexer.__dict__["_r9k_qb"] = st
            q = st.project(qr)
            if q is not None:
                return q
        return a_orig(self, qr, qr_scale)

    def idx_wq_b_proj(self, qr, qr_scale=None):
        st = self.__dict__.get("_r9k_qb")
        if st is not None and qr_scale is None and isinstance(qr, torch.Tensor):
            iq = st.take(qr)
            if iq is not None:
                return iq
        return i_orig(self, qr, qr_scale)

    for cls, fn, orig in ((acls, attn_wq_b_proj, a_orig), (icls, idx_wq_b_proj, i_orig)):
        fn.__r9k_patch__ = "dsv41_qb_stack"
        fn.__r9k_orig__ = orig
        cls._wq_b_proj = fn
    return True


class _AttnInHook:
    """R9K_DSV41_ATTN_IN=r9k: on compressor / indexer layers, attn_norm + the compressor's kv_score GEMV
    (f32 [M, 512 | 1024]) + the indexer's weights_proj (bf16 [M, 32]) in one launch (r9k_dsv4_sk); the
    attention's _run_parallel_input_projections then takes the stashed outputs and runs only fused_wqa_wkv.
    Layers with neither keep stock's attn_norm (nothing to fold)."""

    def __init__(self, layer):
        self.norm = layer.attn_norm
        self.attn = layer.attn
        self.stash = None

    def mats(self):
        a = self.attn
        out = []
        comp, idx = getattr(a, "compressor", None), getattr(a, "indexer", None)
        if comp is not None:
            w = getattr(getattr(comp, "fused_wkv_wgate", None), "weight", None)
            if w is None or w.dtype != torch.bfloat16 or not w.is_contiguous():
                return None
            out.append(("kv", w, True))
        if idx is not None:
            wp = getattr(idx, "weights_proj", None)
            w = getattr(wp, "weight", None)
            if w is None or w.dtype != torch.bfloat16 or not w.is_contiguous() or getattr(wp, "bias", None) is not None:
                return None
            out.append(("w", w, False))
        return out or None

    def norm_forward(self, x, residual=None):
        if residual is None and x.dim() == 2 and 0 < x.shape[0] <= min(MAX_M, ATTN_IN_MAX_M) \
                and x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 8 == 0 \
                and self.norm.weight.dtype == torch.bfloat16 \
                and getattr(self.norm, "variance_size_override", None) is None:
            ms = self.mats()
            if ms is not None and all(w.shape[1] == x.shape[1] for _, w, _ in ms):
                xn, outs = norm_gemv(x, self.norm.weight, self.norm.variance_epsilon, [(w, f) for _, w, f in ms])
                got = {k: o for (k, _, _), o in zip(ms, outs)}
                self.stash = (xn.data_ptr(), tuple(xn.shape), got.get("kv"), got.get("w"))
                return xn
        self.stash = None
        return self._orig_norm(x) if residual is None else self._orig_norm(x, residual)

    def take(self, h):
        s = self.stash
        if s is not None and isinstance(h, torch.Tensor) and h.data_ptr() == s[0] and tuple(h.shape) == s[1]:
            self.stash = None
            return s[2], s[3]
        return None

    def bind(self) -> bool:
        if getattr(self.attn, "compressor", None) is None and getattr(self.attn, "indexer", None) is None:
            return False
        self._orig_norm = self.norm.forward
        self.norm.forward = self.norm_forward
        self.attn.__dict__["_r9k_attn_in"] = self
        return True


def install_attn_in() -> bool:
    from vllm.models.deepseek_v41.amd import model as Mod
    from vllm.models.deepseek_v41.amd import rocm as R
    if not available():
        return False
    acls = R.DeepseekV41ROCMAiterMLAAttention
    if getattr(acls._run_parallel_input_projections, "__r9k_patch__", None) != "dsv41_attn_in":
        orig = acls._run_parallel_input_projections

        def _run_parallel_input_projections(self, hidden_states):
            h = self.__dict__.get("_r9k_attn_in")
            got = h.take(hidden_states) if h is not None else None
            if got is not None:
                return self._fused_wqa_wkv_gemm(hidden_states), got[0], got[1]
            return orig(self, hidden_states)

        _run_parallel_input_projections.__r9k_patch__ = "dsv41_attn_in"
        _run_parallel_input_projections.__r9k_orig__ = orig
        acls._run_parallel_input_projections = _run_parallel_input_projections
    _add_layer_binder("attn_in", lambda layer: _AttnInHook(layer).bind())
    return True


_BINDERS: dict = {}


def _add_layer_binder(name, fn) -> None:
    """Per-layer binders run after DeepseekV4DecoderLayer.__init__ (one class patch shared by the hooks)."""
    from vllm.models.deepseek_v41.amd import model as Mod
    _BINDERS[name] = fn
    cls = Mod.DeepseekV4DecoderLayer
    if getattr(cls.__init__, "__r9k_patch__", None) == "dsv41_binders":
        return
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        for n, f in list(_BINDERS.items()):
            try:
                f(self)
            except Exception as e:  # pragma: no cover
                from vllm.logger import init_logger
                init_logger("vllm." + __name__).warning("r9700: dsv41 %s hook not bound (%s)", n, e)

    __init__.__r9k_patch__ = "dsv41_binders"
    __init__.__r9k_orig__ = init
    cls.__init__ = __init__


# ================================================================================================ DSpark Markov head
MARKOV_FP8 = os.environ.get("R9K_DSV41_MARKOV_FP8", "0") == "1"
# Adaptive verification with the checkpoint's confidence head (the DeepSeek-V4.1 default): r9k = one launch
# (r9k_dsv4_markov_conf: f64-summed dot, f32 sigmoid); stock = the model's own compute_confidence (gather + cat + f32 GEMM
# + sigmoid), bit-exact with stock's loop.
MARKOV_CONF = os.environ.get("R9K_DSV41_MARKOV_CONF", "r9k")
_MK: dict = {}


def _mk_scratch(dev, V):
    key = (dev.index, V)
    b = _MK.get(key)
    if b is None:
        nb = lib().r9k_dsv4_markov_blocks(V)
        b = _MK[key] = [(torch.empty(8, nb, dtype=torch.float32, device=dev),
                         torch.empty(8, nb, dtype=torch.int32, device=dev)) for _ in range(2)]
    return b


def fp8_rows(w: torch.Tensor):
    """e4m3 copy of a bf16 [V, R] matrix with one f32 scale a row (amax / 448)."""
    wf = w.float()
    sc = (wf.abs().amax(1) / 448.0).clamp(min=1e-30)
    return (wf / sc[:, None]).to(torch.float8_e4m3fn).contiguous(), sc.contiguous()


def markov_greedy(w1, w2, base, input_ids, anchor, tokens, rows, n_spec, w2_fp8=None, on_logits=None):
    """Greedy DSpark Markov sampling: n_spec steps of r9k_dsv4_markov_step + the last-token launch.
    w1 / w2 bf16 [V, 256]; base bf16 [rows, n_spec, V] (unit vocab stride); input_ids int32; anchor int64 [rows];
    tokens int64 [>= rows, >= n_spec] (written in place). w2_fp8: optional (e4m3 [V, 256], f32 scale [V]).
    on_logits(step, logits bf16 [rows, V]): when given, each step's logits are stored and handed to it."""
    V, R = w2.shape[0], w2.shape[1]
    dev = base.device
    bufs = _mk_scratch(dev, V)
    st = torch.cuda.current_stream(dev).cuda_stream
    L = lib()
    for step in range(n_spec):
        pm, pa = bufs[(step + 1) % 2]
        om, oa = bufs[step % 2]
        lg = torch.empty((rows, V), dtype=torch.bfloat16, device=dev) if on_logits is not None else None
        b = base[:, step]
        rc = L.r9k_dsv4_markov_step(pm.data_ptr(), pa.data_ptr(), om.data_ptr(), oa.data_ptr(), input_ids.data_ptr(),
                                    anchor.data_ptr(), tokens.data_ptr(), tokens.stride(0), w1.data_ptr(),
                                    (w2_fp8[0] if w2_fp8 is not None else w2).data_ptr(),
                                    w2_fp8[1].data_ptr() if w2_fp8 is not None else 0, b.data_ptr(), b.stride(0),
                                    _p(lg), V if lg is None else lg.stride(0), V, R, step, rows, st)
        if rc:
            raise RuntimeError(f"r9k_dsv4_markov_step failed ({rc}) V={V} R={R} rows={rows}")
        if lg is not None:
            on_logits(step, lg)
    om, oa = bufs[(n_spec - 1) % 2]
    rc = L.r9k_dsv4_markov_last(om.data_ptr(), oa.data_ptr(), tokens.data_ptr(), tokens.stride(0), V, n_spec - 1,
                                rows, st)
    if rc:
        raise RuntimeError(f"r9k_dsv4_markov_last failed ({rc})")


def markov_confidence(hidden, w1, proj_w, proj_b, input_ids, anchor, tokens, rows, n_spec, out):
    """DSpark confidence head over the drafts in one launch: out[r, i] (f32, unit step stride) =
    sigmoid([hidden[r * n_spec + i], w1[prev(r, i)]] . proj_w + proj_b), prev(r, 0) = input_ids[anchor[r]],
    prev(r, i) = tokens[r, i - 1]. hidden bf16 [rows * n_spec, H] (unit column stride); w1 bf16 [V, R] or None
    (a head without the Markov input); proj_w f32 [1, H + R]; proj_b f32 [1] or None; tokens int64 (row stride)."""
    H = hidden.shape[-1]
    R = 0 if w1 is None else w1.shape[1]
    st = torch.cuda.current_stream(hidden.device).cuda_stream
    rc = lib().r9k_dsv4_markov_conf(hidden.data_ptr(), hidden.stride(0), H, _p(w1), R, proj_w.data_ptr(), _p(proj_b),
                                    input_ids.data_ptr(), anchor.data_ptr(), tokens.data_ptr(), tokens.stride(0),
                                    out.data_ptr(), out.stride(0), rows, n_spec, st)
    if rc:
        raise RuntimeError(f"r9k_dsv4_markov_conf failed ({rc}) H={H} R={R} rows={rows}")


def _conf_why_not(model, spec, head_hidden):
    """None when the confidence head can run on r9k_dsv4_markov_conf (else the reason)."""
    if not hasattr(lib(), "r9k_dsv4_markov_conf"):
        return "libr9k without r9k_dsv4_markov_conf"
    ch = getattr(model.model, "confidence_head", None)
    if ch is None or not hasattr(ch, "proj"):
        return "no confidence head module"
    p = ch.proj
    w = getattr(p, "weight", None)
    b = getattr(p, "bias", None)
    if p.quant_method.__class__.__name__ != "UnquantizedLinearMethod" or w is None or w.dtype != torch.float32 \
            or w.dim() != 2 or w.shape[0] != 1 or not w.is_contiguous():
        return "confidence head not an unquantized f32 [1, K] linear"
    if b is not None and (b.dtype != torch.float32 or b.numel() != 1):
        return "confidence head bias"
    R = model.model.markov_head.markov_w1.weight.shape[1] if getattr(ch, "with_markov", True) else 0
    if head_hidden.dtype != torch.bfloat16 or head_hidden.dim() != 2 or w.shape[1] != head_hidden.shape[1] + R \
            or head_hidden.stride(1) != 1:
        return "confidence head input shape"
    if not hasattr(spec, "draft_token_confidence_probs"):
        return "no draft_token_confidence_probs"
    return None


def _markov_why_not(model, spec, num_reqs):
    head = getattr(getattr(model, "model", None), "markov_head", None)
    if head is None:
        return "no Markov head"
    if getattr(spec, "draft_logits", None) is not None:
        return "probabilistic drafts"
    if getattr(spec, "draft_watermarker", None) is not None or getattr(spec, "_d2t_scatter_index", None) is not None:
        return "watermarker / reduced vocabulary"
    if getattr(spec, "_draft_topk", None) is not None:
        return "draft top-k"
    if getattr(spec, "use_confidence_head", False) and getattr(model.model, "confidence_head", None) is None:
        return "confidence head expected but missing"
    if hasattr(model, "apply_output_logit_transform"):
        return "an output logit transform"
    if not 1 <= num_reqs <= 8:
        return f"{num_reqs} requests"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16 or w1.shape[1] != 256 or w2.shape[1] != 256:
        return "Markov weights not bf16 x 256"
    if not (w1.is_contiguous() and w2.is_contiguous()):
        return "Markov weight layout"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    lp = getattr(model, "logits_processor", None)
    if lp is None or getattr(lp, "scale", 1.0) != 1.0 or getattr(lp, "soft_cap", None) is not None:
        return "a scaled logits processor"
    return None


_MK_SEEN: set = set()


def _mk_note(key, msg, *args, warn=False):
    if key not in _MK_SEEN:
        _MK_SEEN.add(key)
        (logger.warning if warn else logger.info)(msg, *args)


def markov_sample_sequential(model, spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """DSparkSpeculator._sample_sequential (greedy drafts) on r9k_dsv4_markov_step: n_spec + 1 launches instead of
    ~5 a step (embedding, bias GEMV, add, argmax, copy). Adaptive verification: with the confidence head (DeepSeek-V4.1's
    checkpoint has one; the speculator then has no acceptance estimator) one more launch writes
    draft_token_confidence_probs from the drafts (r9k_dsv4_markov_conf, or stock's compute_confidence with
    R9K_DSV41_MARKOV_CONF=stock); with the acceptance estimator it gets each step's logits. Returns False, having done
    nothing, otherwise."""
    why = _markov_why_not(model, spec, num_reqs)
    conf = bool(getattr(spec, "use_confidence_head", False))
    cwhy = None
    if why is None and conf and MARKOV_CONF == "r9k":
        cwhy = _conf_why_not(model, spec, head_hidden)
    if why is not None:
        _mk_note(("no", why), "r9700: DSpark Markov head on stock's loop (%s)", why, warn=True)
        return False
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base = model.compute_draft_logits(sample_hidden)
    V = model.model.markov_head.markov_w2.weight.shape[0]
    if base.dtype != torch.bfloat16 or base.shape[-1] != V or base.stride(-1) != 1 or not base.is_contiguous():
        base = base.to(torch.bfloat16).contiguous()
        if base.shape[-1] != V:
            _mk_note(("rest",), "r9700: DSpark Markov head: draft logits width %d != %d, stock bias loop", base.shape[-1],
                     V, warn=True)
            return _markov_rest(model, spec, num_reqs, base, sample_hidden)
    base = base.view(num_reqs, n_spec, V)
    head = model.model.markov_head
    w2 = head.markov_w2.weight
    fp8 = None
    if MARKOV_FP8:
        fp8 = head.__dict__.get("_r9k_fp8")
        if fp8 is None and not torch.cuda.is_current_stream_capturing():
            fp8 = head.__dict__["_r9k_fp8"] = fp8_rows(w2)
    on = None
    if getattr(spec, "acceptance_estimator", None) is not None:
        idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)

        def on(step, lg):
            spec._maybe_predict_acceptance(lg, idx_map[:, step], spec._step_cols[step])
    markov_greedy(head.markov_w1.weight, w2, base, spec.input_buffers.input_ids, spec._anchor_idx, spec.draft_tokens,
                  num_reqs, n_spec, fp8, on)
    mode = "estimator gets each step's logits" if on is not None else "no adaptive input"
    if conf:
        if MARKOV_CONF == "r9k" and cwhy is None:
            ch = model.model.confidence_head
            markov_confidence(sample_hidden, head.markov_w1.weight if getattr(ch, "with_markov", True) else None,
                              ch.proj.weight, ch.proj.bias, spec.input_buffers.input_ids, spec._anchor_idx,
                              spec.draft_tokens, num_reqs, n_spec, spec.draft_token_confidence_probs)
            mode = "confidence head on r9k_dsv4_markov_conf"
        else:
            _stock_confidence(model, spec, num_reqs, sample_hidden)
            mode = "confidence head on stock's compute_confidence" + (f" ({cwhy})" if cwhy else "")
    _mk_note(("on", mode), "r9700: DSpark Markov head on r9k_dsv4_markov (adaptive verification: %s)", mode)
    return True


def _stock_confidence(model, spec, num_reqs, sample_hidden):
    """Stock's confidence tail from the drafts: markov_embed(prev) per step (prev = anchor, then the drafts), stacked
    (req, step), compute_confidence -> draft_token_confidence_probs. Same ops on the same tokens as stock's loop."""
    n_spec = spec.num_speculative_steps
    prev0 = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    embeds = [model.markov_embed(prev0)] + [model.markov_embed(spec.draft_tokens[:num_reqs, i])
                                             for i in range(n_spec - 1)]
    confidence = model.compute_confidence(sample_hidden, torch.stack(embeds, dim=1).flatten(0, 1))
    spec.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)


def _markov_rest(model, spec, num_reqs, base_logits, sample_hidden) -> bool:
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    base_logits = base_logits.view(num_reqs, n_spec, -1)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        bias = model.markov_bias(model.markov_embed(prev))
        draft = spec._sample_logits(base_logits[:, i] + bias, idx_map[:, i], sample_pos[:, i], i)
        spec.draft_tokens[:num_reqs, i] = draft
        prev = draft
    if getattr(spec, "use_confidence_head", False):
        _stock_confidence(model, spec, num_reqs, sample_hidden)
    return True


def install_markov() -> bool:
    from vllm.models.deepseek_v41.amd import dspark as Ds
    cls = Ds.DSparkDeepseekV4ForCausalLM
    if getattr(getattr(cls, "sample_sequential", None), "__r9k_patch__", None) == "dsv41_markov":
        return True
    if not available():
        return False
    orig = getattr(cls, "sample_sequential", None)

    def sample_sequential(self, speculator, num_reqs, head_hidden):
        if markov_sample_sequential(self, speculator, num_reqs, head_hidden):
            return True
        return orig(self, speculator, num_reqs, head_hidden) if orig is not None else False

    sample_sequential.__r9k_patch__ = "dsv41_markov"
    sample_sequential.__r9k_orig__ = orig
    cls.sample_sequential = sample_sequential
    return True
