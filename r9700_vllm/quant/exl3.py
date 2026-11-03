"""EXL3 (ExLlamaV3 trellis-quantized) weights on stock vLLM: the ``exl3`` quantization config and the linear method.

Serves published EXL3 packs (first target: LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) through exllamav3's
compiled extension (``exllamav3_ext``, the gfx1201 wheel) -- its exl3_gemm / reconstruct kernels, our glue. Design:
notes/exl3-plan.md. Per quantized linear the checkpoint holds ``<name>.trellis`` int16 [in/16, out/16, 16*K], ``.suh``
f16 [in], ``.svh`` f16 [out] and the codebook marker ``.mul1`` (or ``.mcg``) int32 []. Tiles decode independently;
Hadamard blocks are 128 wide on both sides, so every tensor-parallel split lands on a 128-multiple.

Loading: each LinearBase gets empty placeholder Parameters ``trellis`` / ``suh`` / ``svh`` / the codebook marker whose
own ``weight_loader(param, tensor, shard_id=None)`` slices the tensor to this rank and stashes it on the layer per shard
(a stacked layer's shards are separate matrices with their own ``suh``; a plain column layer may also take equal-width
sub-matrices by index, e.g. ``wo_a``'s eight o-group slices). The placeholders never change shape, so no vLLM loader
shape check applies. ``process_weights_after_loading`` turns the stash into one ``Exl3Part`` per matrix. A layer whose
rank split is not a 128-multiple (shared experts at TP8: 2304/8 = 288) is dequantized at load to a bf16 ``weight`` and
runs the stock unquantized GEMM.

Forward (decode, M <= RECON_M): x -> f16 into a static per-layer buffer, ``exl3_gemm`` per part in 8-row chunks into a
static f32 C (stacked layers whose parts share K: one ``exl3_mgemm`` for all parts), one cast into the caller's dtype
(per part when M > 1 and the layer has several). exllamav3 takes only f16 A and f16/f32 C, so the two casts stay:
4 launches a single-matrix apply. Rows 1..8 are exllamav3's RDNA multi-row GEMV; a 9+ row call would take its
autotuned WMMA trellis GEMM, 1.4-2.6x slower than 8-row GEMV chunks on gfx1201 and host-syncing on its first call at a new M (which kills a HIP graph
capture), so it is never called. Above RECON_M (exllamav3's reconstruct_hgemm, notes/exl3-kernels.md): dequantize the
part to f16 and run hipBLAS in 1024-row chunks (unchunked 4096-row f16 GEMMs fall to 15-31 TFLOPS on some shapes vs
84-92 chunked) -- the plain decode with the two Hadamards on the activations below FUSED_M rows, exllamav3's fused
reconstruct_had_slice (original-basis weight, no activation Hadamards) from FUSED_M, and from FUSED_M_WIDE on outputs of
16384+ columns, where had_r_128 over the wide output is the cost.

Not stream-safe: the RDNA GEMV hands its pointers to the dot kernel through one per-device parameter block, so two EXL3
GEMMs must never be in flight on two streams of one device (the ROCm DeepSeek-V4 attention's execute_in_parallel
branches would do that; the model glue keeps them serial).

Public names other packages rely on: Exl3Config, Exl3LinearMethod, Exl3LMHeadMethod, Exl3Part, ext(), k_of(),
reconstruct_f16(), dequant_f32(), exl3_gemm_f16().
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

logger = init_logger("vllm.r9700_vllm.exl3")

# Rows above which a linear dequantizes and runs an f16 GEMM (exllamav3's AUTO_RECONSTRUCT_THRESHOLD = 144 is a CUDA
# number; on gfx1201 8-row GEMV chunks lose to dequant + GEMM from 24-48 rows: notes/exl3-kernels.md "crossover").
RECON_M = int(os.environ.get("R9K_EXL3_RECON_M", "16"))  # two GEMV chunks still beat the dequant; three do not
# Rows from which the prefill path takes the fused reconstruct (all widths / outputs >= 16384 wide): measured crossover,
# tests/test_exl3_linear.py numbers.
FUSED_M = int(os.environ.get("R9K_EXL3_FUSED_M", "1024"))
FUSED_M_WIDE = int(os.environ.get("R9K_EXL3_FUSED_M_WIDE", "384"))
_GEMV_ROWS = 8          # exl3_gemm takes the RDNA GEMV for 1..8 rows (tags 90-92) and the autotuned GEMM above
# Decode GEMV (package D): R9K_EXL3_GEMV=r9k (default when libr9k has it) runs rows 1..GEMV_M through our multi-matrix
# kernel (quant/exl3_gemv.py: bf16 in / out, rotations in-kernel, one launch a layer); =exl3 keeps exllamav3's exl3_gemm.
GEMV = os.environ.get("R9K_EXL3_GEMV", "r9k")
GEMV_M = int(os.environ.get("R9K_EXL3_GEMV_M", "48"))   # every decode width (4 streams x 6 = 24; 8 x 6 = 48)
_MM_ROWS = 1024         # hipBLAS f16 GEMM row chunk
_FIELDS = ("trellis", "suh", "svh")
_DTYPES = {"trellis": torch.int16, "suh": torch.float16, "svh": torch.float16, "mul1": torch.int32, "mcg": torch.int32}

_EXT = None


def ext():
    """exllamav3's compiled extension, imported on first use (the serving image carries the gfx1201 wheel)."""
    global _EXT
    if _EXT is None:
        import exllamav3_ext  # noqa: F401
        _EXT = exllamav3_ext
    return _EXT


def k_of(trellis: torch.Tensor) -> int:
    """Bits per weight from the trellis tile width (16*K int16 per 16x16 tile; this pack has no half-integer K)."""
    w = trellis.shape[-1]
    assert w % 16 == 0, f"exl3: half-integer bitrate (tile width {w}) not supported"
    return w // 16


_HAD = {}


def _had128(dev: torch.device) -> torch.Tensor:
    """The 128-point Hadamard exllamav3 applies (Sylvester order, 1/sqrt(128); == ext.had_r_128), f32."""
    h = _HAD.get(dev)
    if h is None:
        h = torch.ones(1, 1, device=dev, dtype=torch.float32)   # not the default dtype: vLLM sets it to bf16
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = _HAD[dev] = h * 128 ** -0.5
    return h


def dequant_f32(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: int, mul1: bool,
                mcg: bool = False) -> torch.Tensor:
    """W^T f32 [in, out] = diag(suh) H decode(trellis) H diag(svh), blockwise 128 (LinearEXL3.get_weight_tensor): the
    plain trellis decode is exact in f16 and the rotations run in f32, so the only rounding is the caller's cast."""
    assert K == k_of(trellis)
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    assert k % 128 == 0 and n % 128 == 0, f"exl3: {k}x{n} is not 128-blocked"
    rot = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    ext().reconstruct(rot, trellis, K, mcg, mul1)
    H = _had128(trellis.device)
    w = (H @ rot.float().view(k // 128, 128, n)).view(k, n)
    del rot
    w *= suh.float()[:, None]
    w = (w.view(-1, 128) @ H).view(k, n)
    w *= svh.float()[None, :]
    return w


def reconstruct_f16(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: int, mul1: bool,
                    mcg: bool = False) -> torch.Tensor:
    """Exact dequantization to f16 [in, out] (W^T): dequant_f32 rounded once. Load-time paths (lm_head, dense layers)
    use dequant_f32 directly; the prefill forward uses exllamav3's fused kernel (Exl3Part.dequant)."""
    return dequant_f32(trellis, suh, svh, K, mul1, mcg).half()


def exl3_gemm_f16(a_f16: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: int, mul1: bool,
                  mcg: bool = False, out_f32: bool = False, a_had: torch.Tensor | None = None) -> torch.Tensor:
    """Trellis GEMM through the RDNA GEMV, 8 rows a launch: a_f16 [m, in] contiguous -> [m, out] (f16, or f32 when
    out_f32). ``a_had`` is the kernel's scratch for the rotated input (same shape as a_f16; allocated when not given).
    Meant for decode row counts; above RECON_M rows dequantize instead (Exl3Part.recon_mm)."""
    assert K == k_of(trellis) and a_f16.dtype == torch.float16 and a_f16.is_contiguous()
    m = a_f16.shape[0]
    c = torch.empty((m, trellis.shape[1] * 16), dtype=torch.float32 if out_f32 else torch.float16, device=a_f16.device)
    if a_had is None:
        a_had = torch.empty_like(a_f16)
    for i in range(0, m, _GEMV_ROWS):
        j = min(m, i + _GEMV_ROWS)
        # force_num_sms 0 = device default (-1 would skip the GEMV and the autotuner and set num_sms = -1)
        ext().exl3_gemm(a_f16[i:j], trellis, c[i:j], suh, a_had[i:j], svh, -1, mcg, mul1, 0)
    return c


def _gemv_plan(parts, k: int):
    """The r9k decode-GEMV plan for a layer's parts (quant/exl3_gemv.py), or None when libr9k lacks the kernel, the knob
    is off or a part is outside its K range (2..6). Reserves the kernel's scratch for GEMV_M rows and runs every row count
    once, so nothing allocates under graph capture."""
    if GEMV == "off":
        return None
    try:
        from . import exl3_gemv as G
        if not G.available():
            return None
    except Exception:  # pragma: no cover
        return None
    if any(not 2 <= p.K <= 6 or p.mcg == p.mul1 for p in parts):
        return None
    plan = G.Exl3GemvPlan(parts, k)
    plan.reserve(GEMV_M)
    dev = parts[0].trellis.device
    x = torch.zeros((GEMV_M, k), dtype=torch.bfloat16, device=dev)
    y = torch.empty((GEMV_M, plan.n_total), dtype=torch.bfloat16, device=dev)
    for m in range(1, GEMV_M + 1):
        G.r9k_exl3_gemv(plan, x[:m], y[:m])
    logger.info_once("exl3: decode rows 1..%d on r9k_exl3_gemv (R9K_EXL3_GEMV=%s)", GEMV_M, GEMV)
    return plan


@dataclass
class Exl3Part:
    """One EXL3 matrix of a layer on this rank: output columns [dst, dst + n) of the layer's output."""
    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    K: int
    mcg: bool
    mul1: bool
    dst: int
    n: int

    def gemm(self, a_f16: torch.Tensor, a_had: torch.Tensor) -> torch.Tensor:
        return exl3_gemm_f16(a_f16, self.trellis, self.suh, self.svh, self.K, self.mul1, self.mcg, True, a_had)

    def dequant(self) -> torch.Tensor:
        """f16 [in, out] in one kernel (reconstruct_had_slice: decode + both Hadamards + sign vectors; ~9e-4
        relative, its Hadamards run in f16)."""
        w = torch.empty((self.trellis.shape[0] * 16, self.n), dtype=torch.float16, device=self.trellis.device)
        ext().reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, self.mcg, self.mul1, 0)
        return w

    def recon_mm(self, a_f16: torch.Tensor) -> torch.Tensor:
        """Prefill: a_f16 [m, in] -> f16 [m, out] through a dequantized weight and hipBLAS in 1024-row chunks. Below the
        fused crossover the weight stays in the rotated basis (exact decode) and the activations take the Hadamards
        (had_r_128 in, had_r_128 out: 2e-4 relative); above it the fused kernel's original-basis weight (9e-4)."""
        m, k = a_f16.shape
        fused = m >= FUSED_M or (self.n >= 16384 and m >= FUSED_M_WIDE)
        y = torch.empty((m, self.n), dtype=torch.float16, device=a_f16.device)
        if fused:
            w, xs = self.dequant(), a_f16
        else:
            xs = torch.empty_like(a_f16)
            ext().had_r_128(a_f16, xs, self.suh, None, 1.0)
            w = torch.empty((k, self.n), dtype=torch.float16, device=a_f16.device)
            ext().reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
        for i in range(0, m, _MM_ROWS):
            torch.matmul(xs[i:i + _MM_ROWS], w, out=y[i:i + _MM_ROWS])
        if not fused:
            ext().had_r_128(y, y, None, self.svh, 1.0)
        return y

    def dequant_f32(self) -> torch.Tensor:
        return dequant_f32(self.trellis, self.suh, self.svh, self.K, self.mul1, self.mcg)


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """``quantization_config = {"quant_method": "exl3", ...}`` from the pack's config.json."""

    # ---- attributes the DeepSeek-V4 model code reads off its quant config (package N: report additions here) ----
    weight_block_size = None            # pad_shared_expert / _linear_scale_param_name / wo_a block size probe
    # ---------------------------------------------------------------------------------------------------------------

    def __init__(self, avg_bits: float, head_bits: int, codebook: str, version: str, out_scales: str = "always",
                 k_map: dict | None = None):
        super().__init__()
        self.avg_bits, self.head_bits, self.codebook = avg_bits, head_bits, codebook
        self.version, self.out_scales = version, out_scales
        self.k_map = k_map or {}
        # R9K_EXL3_DENSE=<regex>: matching linears are dequantized to bf16 at load whatever their split
        pat = os.environ.get("R9K_EXL3_DENSE", "")
        self.dense_re = re.compile(pat) if pat else None

    def get_name(self) -> str:  # type: ignore[override]
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(avg_bits=float(config.get("avg_bits", 0)), head_bits=int(config.get("head_bits", 0)),
                   codebook=str(config.get("codebook", "mul1")), version=str(config.get("version", "")),
                   out_scales=str(config.get("out_scales", "always")))

    def maybe_update_config(self, model_name: str, hf_config=None, revision: str | None = None):
        """k_map = exl3_k_map.json's routed-expert bits per layer ({layer: K}), when the model is a local directory."""
        path = os.path.join(model_name, "exl3_k_map.json")
        if os.path.isfile(path):
            with open(path) as f:
                self.k_map = {int(k): int(v) for k, v in json.load(f).get("routed", {}).items()}

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
        if isinstance(layer, ParallelLMHead):
            return Exl3LMHeadMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        if isinstance(layer, RoutedExperts):
            from .exl3_moe import Exl3MoEMethod
            return Exl3MoEMethod(layer.moe_config, self)
        if isinstance(layer, LinearBase):
            return Exl3LinearMethod(self)
        return None


def _placeholders(layer: torch.nn.Module, codebook: str, make_loader) -> None:
    """trellis / suh / svh and the one codebook marker this pack uses (vLLM's loader rejects a parameter that never
    receives a tensor, so the other marker is not registered)."""
    for name in _FIELDS + (codebook,):
        p = Parameter(torch.empty(0, dtype=_DTYPES[name]), requires_grad=False)
        p.weight_loader = make_loader(name)
        layer.register_parameter(name, p)


def _flag_or(field: str, w: torch.Tensor, dev: torch.device, narrow=None):
    """Codebook markers become True; tensors go to the device contiguous, after the optional narrow."""
    if field in ("mul1", "mcg"):
        return True
    if narrow is not None:
        w = w.narrow(*narrow)
    return w.contiguous().to(dev)


class Exl3LinearMethod(LinearMethodBase):
    """LinearBase layers with one EXL3 matrix per shard (vLLM's stacked layers fuse matrices whose ``suh`` differ)."""

    def __init__(self, quant_config: Exl3Config):
        self.quant_config = quant_config

    # ------------------------------------------------------------------------------------------------- geometry
    @staticmethod
    def _tp(layer) -> tuple[int, int]:
        if isinstance(layer, ReplicatedLinear):
            return 0, 1
        return layer.tp_rank, layer.tp_size

    @staticmethod
    def _col_span(layer, shard_id, width: int) -> tuple[int, int, int] | None:
        """Column layers: (lo, hi, dst) -- this rank keeps columns [lo, hi) of the sub-matrix (``width`` wide) and
        writes them at output column dst; None when the rank owns none of it. Merged layers: shard i = logical output
        i, split evenly over the ranks. Plain column layers: shard_id None = the whole matrix, int g = the g-th of
        equal-width sub-matrices stacked along the output (wo_a's o-group slices)."""
        rank, size = Exl3LinearMethod._tp(layer)
        sizes = getattr(layer, "output_sizes", None)
        if sizes is not None:
            assert isinstance(shard_id, int) and 0 <= shard_id < len(sizes), f"exl3 {layer.prefix}: shard {shard_id!r}"
            assert width == sizes[shard_id], f"exl3 {layer.prefix}: shard {shard_id} is {width} wide, layer {sizes}"
            p = sizes[shard_id] // size
            return rank * p, (rank + 1) * p, sum(s // size for s in sizes[:shard_id])
        P = sum(layer.output_partition_sizes)
        g0 = 0 if shard_id is None else int(shard_id) * width
        lo, hi = max(g0, rank * P), min(g0 + width, (rank + 1) * P)
        if lo >= hi:
            return None
        return lo - g0, hi - g0, lo - rank * P

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        row = isinstance(layer, RowParallelLinear)
        k_part = input_size_per_partition
        aligned = k_part % 128 == 0 and all(p % 128 == 0 for p in output_partition_sizes)
        dre = self.quant_config.dense_re
        layer.exl3_dense = not aligned or bool(dre and dre.search(getattr(layer, "prefix", "") or ""))
        layer.exl3_row = row
        layer.exl3_k = k_part
        layer.exl3_n = sum(output_partition_sizes)
        layer.exl3_dtype = params_dtype
        layer.exl3_stash = {}
        layer.exl3_parts = []

        def make_loader(field: str):
            def loader(param: Parameter, w: torch.Tensor, shard_id=None) -> None:
                dev = torch.device("cuda", torch.cuda.current_device())
                st = layer.exl3_stash.setdefault(shard_id, {})
                if row:
                    assert shard_id is None, f"exl3 {layer.prefix}: row-parallel layers take whole matrices"
                    rank = self._tp(layer)[0]
                    lo, hi = rank * k_part, (rank + 1) * k_part
                    st["span"] = (lo, hi, 0)
                    if layer.exl3_dense or field in ("svh", "mul1", "mcg"):
                        st[field] = _flag_or(field, w, dev)
                    else:                                   # input-side tensors: rows [lo, hi) / tile rows lo/16..
                        assert lo % 128 == 0 and hi % 128 == 0
                        st[field] = _flag_or(field, w, dev, (0, lo // 16, k_part // 16) if field == "trellis"
                                             else (0, lo, k_part))
                    return
                if field in ("mul1", "mcg") or field == "suh":
                    st[field] = _flag_or(field, w, dev)
                    return
                width = w.shape[1] * 16 if field == "trellis" else w.shape[0]
                span = self._col_span(layer, shard_id, width)
                if span is None:
                    st["skip"] = True
                    return
                st["span"], st["width"] = span, width
                lo, hi, _ = span
                if layer.exl3_dense:
                    st[field] = _flag_or(field, w, dev)
                else:
                    assert lo % 128 == 0 and hi % 128 == 0, f"exl3 {layer.prefix}: split {lo}:{hi} not 128-aligned"
                    st[field] = _flag_or(field, w, dev, (1, lo // 16, (hi - lo) // 16) if field == "trellis"
                                         else (0, lo, hi - lo))
            return loader

        _placeholders(layer, self.quant_config.codebook, make_loader)

    # -------------------------------------------------------------------------------------------- after loading
    def _dummy_stash(self, layer) -> dict:
        """R9K_EXL3_DUMMY=1 (``--load-format dummy`` bring-up): random K=5 matrices of the layer's real per-rank
        geometry, in the stash format the loaders produce (dense layers: whole matrices + this rank's span, then the
        normal densify). Any int16 trellis decodes, so the forward and its warm-up run on real shapes."""
        dev = torch.device("cuda", torch.cuda.current_device())
        rank, size = self._tp(layer)
        sizes = layer.output_partition_sizes

        def mat(k, n):
            return {"trellis": torch.randint(-32768, 32768, (k // 16, n // 16, 80), dtype=torch.int32,
                                             device=dev).to(torch.int16),
                    "suh": (torch.randint(0, 2, (k,), device=dev) * 2 - 1).half(),
                    "svh": ((torch.rand(n, device=dev, dtype=torch.float32) - 0.5) * 2 / k ** 0.5).half(),
                    self.quant_config.codebook: True}
        k = layer.exl3_k
        if layer.exl3_row:
            whole = layer.exl3_dense
            st = mat(k * size if whole else k, layer.exl3_n)
            st["span"] = (rank * k, (rank + 1) * k, 0) if whole else (0, k, 0)
            return {None: st}
        out, dst = {}, 0
        for i, n in enumerate(sizes):
            whole = layer.exl3_dense and not isinstance(layer, ReplicatedLinear)
            st = mat(k, n * size if whole else n)
            st["span"] = (rank * n, (rank + 1) * n, dst) if whole else (0, n, dst)
            out[i] = st
            dst += n
        return out

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        stash = layer.exl3_stash
        if not stash and (layer.exl3_parts or hasattr(layer, "weight")):
            return                                           # already processed
        prefix = getattr(layer, "prefix", "") or type(layer).__name__
        if not stash and os.environ.get("R9K_EXL3_DUMMY") == "1":
            stash = self._dummy_stash(layer)
            logger.info_once("exl3: R9K_EXL3_DUMMY=1 -- random EXL3 matrices for layers that loaded none")
        parts = []
        for sid, st in stash.items():
            if st.get("skip"):
                continue
            missing = [f for f in ("trellis", "suh", "svh") if f not in st]
            assert not missing, f"exl3 {prefix}: shard {sid!r} has no {missing} (got {sorted(st)})"
            assert not (st.get("mul1") and st.get("mcg")), f"exl3 {prefix}: both codebook markers"
            tr = st["trellis"]
            K = k_of(tr)
            assert st["suh"].numel() == tr.shape[0] * 16 and st["svh"].numel() == tr.shape[1] * 16, \
                f"exl3 {prefix}: suh/svh {st['suh'].shape}/{st['svh'].shape} vs trellis {tuple(tr.shape)}"
            lo, hi, dst = st["span"]
            parts.append((dst, lo, hi, Exl3Part(tr, st["suh"], st["svh"], K, bool(st.get("mcg")), bool(st.get("mul1")),
                                                dst, tr.shape[1] * 16)))
        assert parts, f"exl3 {prefix}: no EXL3 tensors loaded (a bf16 matrix on an exl3 layer is not supported)"
        parts.sort(key=lambda t: t[0])
        layer.exl3_stash = {}
        if layer.exl3_dense:
            self._densify(layer, parts, prefix)
        else:
            off = 0
            for dst, _, _, p in parts:
                assert dst == off and p.trellis.shape[0] * 16 == layer.exl3_k, \
                    f"exl3 {prefix}: parts do not tile the output ({dst} != {off}) or k {p.trellis.shape[0] * 16}"
                off += p.n
            assert off == layer.exl3_n, f"exl3 {prefix}: parts cover {off} of {layer.exl3_n} outputs"
            layer.exl3_parts = [p for *_, p in parts]
            self._warm(layer)

    def _densify(self, layer, parts, prefix: str) -> None:
        """bf16 [n_part, k_part] weight from the whole matrices: dequantize each, keep this rank's rows / columns."""
        w = torch.empty((layer.exl3_n, layer.exl3_k), dtype=layer.exl3_dtype, device=parts[0][3].trellis.device)
        for dst, lo, hi, p in parts:
            wt = p.dequant_f32()                                 # [in, out]
            if layer.exl3_row:
                assert wt.shape[1] == layer.exl3_n
                w.copy_(wt[lo:hi].t())
            else:
                assert wt.shape[0] == layer.exl3_k
                w[dst:dst + hi - lo].copy_(wt[:, lo:hi].t())
            del wt
        layer.weight = Parameter(w, requires_grad=False)
        logger.info_once("exl3: %s dequantized to %s %s at load (split not 128-aligned)", prefix, w.dtype,
                         tuple(w.shape))

    @staticmethod
    def _warm(layer) -> None:
        """Static decode buffers, the multi-matrix tables, and one eager call per row count 1..8 before any capture:
        allocates the GEMV's per-device parameter block (a cudaMalloc, not capturable) and asserts every chunk size
        takes the GEMV (tags 90-92, 93 multi-matrix), never the autotuned GEMM whose first call host-syncs."""
        parts, k, n = layer.exl3_parts, layer.exl3_k, layer.exl3_n
        dev = parts[0].trellis.device
        rows = max(RECON_M, _GEMV_ROWS)
        layer.exl3_abuf = torch.empty((rows, k), dtype=torch.float16, device=dev)
        layer.exl3_ahad = torch.empty(len(parts) * _GEMV_ROWS * k, dtype=torch.float16, device=dev)
        layer.exl3_cbuf = torch.empty(rows * n, dtype=torch.float32, device=dev)
        layer.exl3_mg = None
        p0 = parts[0]
        if len(parts) > 1 and all((p.K, p.mcg, p.mul1) == (p0.K, p0.mcg, p0.mul1) for p in parts):
            # one exl3_mgemm for all parts (one input rotation + one dot launch): part j writes its [m, n_j] block at
            # cbuf[m * dst_j:], chunk c's rows from row 8c; the pointer table holds every (m, chunk) up front
            base, chunks = layer.exl3_cbuf.data_ptr(), -(-rows // _GEMV_ROWS)
            cp = [[[base + 4 * (m * p.dst + c * _GEMV_ROWS * p.n) for p in parts] for c in range(chunks)]
                  for m in range(rows + 1)]
            def ptrs(f):
                return torch.tensor([getattr(p, f).data_ptr() for p in parts], dtype=torch.long, device=dev)
            layer.exl3_mg = {"trellis": ptrs("trellis"), "suh": ptrs("suh"), "svh": ptrs("svh"),
                             "size_n": torch.tensor([p.n for p in parts], dtype=torch.int32, device=dev),
                             "cptrs": torch.tensor(cp, dtype=torch.long, device=dev),
                             "carrier": torch.empty((1, 1, max(p.n for p in parts)), dtype=torch.float32, device=dev)}
        for m in range(1, _GEMV_ROWS + 1):
            for tag in Exl3LinearMethod._decode(layer, layer.exl3_abuf[:m].zero_()):
                assert tag in (90, 91, 92, 93), f"exl3 {layer.prefix}: {m} rows took exl3_gemm path {tag}, not the GEMV"
        layer.exl3_gemv = _gemv_plan(parts, k)

    @staticmethod
    def _gemv_apply(layer, x: torch.Tensor, m: int, k: int, n: int) -> torch.Tensor:
        """Rows 1..GEMV_M: one r9k_exl3_gemv launch a (K, codebook) for every part, bf16 / f16 in, x's dtype out."""
        from .exl3_gemv import r9k_exl3_gemv
        a = x.reshape(m, k)
        if a.dtype not in (torch.bfloat16, torch.float16):
            a = a.to(torch.float16)
        if a.stride(1) != 1 or a.stride(0) % 4 or a.data_ptr() % 16:
            a = a.clone(memory_format=torch.contiguous_format)
        y = torch.empty((m, n), dtype=x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                        else torch.bfloat16, device=x.device)
        return r9k_exl3_gemv(layer.exl3_gemv, a, y)

    @staticmethod
    def _decode(layer, a: torch.Tensor) -> list[int]:
        """a [m, k] f16 (m <= RECON_M) -> layer.exl3_cbuf: part blocks [m, n_j] at m * dst_j (== row layout when there
        is one part or one row), 8-row GEMV chunks."""
        m, k = a.shape
        tags, mg, cbuf, ahad = [], layer.exl3_mg, layer.exl3_cbuf, layer.exl3_ahad
        for c, i in enumerate(range(0, m, _GEMV_ROWS)):
            j = min(m, i + _GEMV_ROWS)
            if mg is not None:
                p0 = layer.exl3_parts[0]
                tags.append(ext().exl3_mgemm(a[i:j].unsqueeze(0), mg["trellis"], mg["carrier"].expand(1, j - i, -1),
                                             mg["suh"], ahad, mg["svh"], None, None, p0.K, -1, p0.mcg, p0.mul1, -1,
                                             -1, 0, 1, mg["size_n"], mg["cptrs"][m, c]))
                continue
            ah = ahad[:(j - i) * k].view(j - i, k)
            for p in layer.exl3_parts:
                cv = cbuf[m * p.dst:m * (p.dst + p.n)].view(m, p.n)
                tags.append(ext().exl3_gemm(a[i:j], p.trellis, cv[i:j], p.suh, ah, p.svh, -1, p.mcg, p.mul1, 0))
        return tags

    # -------------------------------------------------------------------------------------------------- forward
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.exl3_dense:
            return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
        lead, k, n = x.shape[:-1], x.shape[-1], layer.exl3_n
        m = x.numel() // k
        parts = layer.exl3_parts
        plan = getattr(layer, "exl3_gemv", None)
        if plan is not None and m <= GEMV_M and GEMV == "r9k" and plan.takes(m):
            y = self._gemv_apply(layer, x, m, k, n)
        elif m <= RECON_M:
            # decode: bf16 -> f16 straight into the static input (one kernel, any input strides), the GEMVs into the
            # static f32 C, then one cast to the caller's dtype per output block (one in all when m == 1)
            a = layer.exl3_abuf[:m]
            a.view(x.shape).copy_(x)
            self._decode(layer, a)
            cbuf = layer.exl3_cbuf
            if len(parts) == 1 or m == 1:
                y = cbuf[:m * n].view(m, n).to(x.dtype)
            else:
                y = torch.empty((m, n), dtype=x.dtype, device=x.device)
                for p in parts:
                    y[:, p.dst:p.dst + p.n].copy_(cbuf[m * p.dst:m * (p.dst + p.n)].view(m, p.n))
        else:
            a = x.reshape(m, k).to(torch.float16).contiguous()
            if len(parts) == 1:
                y = parts[0].recon_mm(a).to(x.dtype)
            else:
                y = torch.empty((m, n), dtype=x.dtype, device=x.device)
                for p in parts:
                    y[:, p.dst:p.dst + p.n].copy_(p.recon_mm(a))
        if bias is not None:
            y += bias
        return y.view(*lead, n)


class Exl3LMHeadMethod(UnquantizedEmbeddingMethod):
    """ParallelLMHead from an EXL3 head, dequantized at load: this rank's vocab rows [org_start, org_end) come from
    the 128-aligned slab of trellis columns around them (the head cannot stay EXL3 vocab-sharded: 129280/8 = 16160 is
    not a 128-multiple). Leaves the stock bf16 ``weight`` [vocab_per_partition, hidden] (padding rows zero), so the
    stock logits path runs unchanged. A checkpoint carrying a plain ``weight`` loads through the stock loader."""

    def __init__(self, quant_config: Exl3Config):
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        super().create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                               params_dtype, **extra_weight_attrs)
        layer.exl3_stash = {}

        def make_loader(field: str):
            def loader(param: Parameter, w: torch.Tensor, shard_id=None) -> None:
                assert shard_id is None
                dev = torch.device("cuda", torch.cuda.current_device())
                si = layer.shard_indices
                s0, s1 = si.org_vocab_start_index, si.org_vocab_end_index
                b0 = s0 // 128 * 128
                st = layer.exl3_stash
                if field in ("trellis", "svh"):
                    full = w.shape[1] * 16 if field == "trellis" else w.shape[0]
                    assert full == layer.org_vocab_size, f"exl3 lm_head: {field} is {full} != {layer.org_vocab_size}"
                    b1 = min(-(-s1 // 128) * 128, full)
                    st["slab"] = (b0, b1)
                    st[field] = _flag_or(field, w, dev, (1, b0 // 16, (b1 - b0) // 16) if field == "trellis"
                                         else (0, b0, b1 - b0))
                else:
                    st[field] = _flag_or(field, w, dev)
            return loader

        _placeholders(layer, self.quant_config.codebook, make_loader)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        st, layer.exl3_stash = layer.exl3_stash, {}
        if st:
            tr = st["trellis"]
            b0, _ = st["slab"]
            s0, s1 = layer.shard_indices.org_vocab_start_index, layer.shard_indices.org_vocab_end_index
            wt = dequant_f32(tr, st["suh"], st["svh"], k_of(tr), bool(st.get("mul1")), bool(st.get("mcg")))
            layer.weight.data[:s1 - s0].copy_(wt[:, s0 - b0:s1 - b0].t())
            layer.weight.data[s1 - s0:].zero_()
            del wt
            logger.info_once("exl3: lm_head dequantized at load (%d-bit, vocab rows %d:%d a rank)", k_of(tr), s0, s1)
        super().process_weights_after_loading(layer)
