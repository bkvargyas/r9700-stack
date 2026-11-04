"""EXL3 routed experts on stock vLLM's RoutedExperts layer: exllamav3's exl3_moe kernel behind FusedMoEMethodBase.

Expert parallel: each rank owns whole experts (384 / 8 = 48 a layer for DeepSeek-V4.1-Flash), so no expert matrix is ever
split and the pack's per-expert tensors load as they are. Per local expert: w1 (gate) and w3 (up) [5120 -> 2304], w2 (down)
[2304 -> 5120] as trellis/suh/svh + codebook flag, K uniform within a layer (3, or 2 on layers 18-22). Design:
notes/exl3-plan.md.

Loading: the checkpoint's ``experts.{e}.w1|w3|w2.<t>`` arrive through stock vLLM's expert mapping as ``w13_<t>`` /
``w2_<t>`` with (expert_id, shard_id); the placeholder parameters' loader maps the global id through the layer's expert
map and copies the tensor straight into one device buffer per (role, field), [E_local, ...], allocated on the first
expert's arrival (shapes come from the checkpoint, so the parameters themselves stay empty). The kernels only need
addresses: process_weights_after_loading turns each buffer into an int64 pointer table, one entry per local expert.

Forward (as exllamav3's BlockSparseMLP.forward, ROCm defaults): topk ids -> local ids (E = not on this rank, the
sentinel bin). Up to R9K_ROWS (r9k) / DECODE_ROWS tokens the decode tier (see DECODE: r9k = kernels/r9k_moe_exl3.hip,
two launches; mgemv = exllamav3's exl3_mgemm) runs every local (token, pick) slot without sorting. Above it,
expert_count / token_sorted / weight_sorted in exllamav3's layout
on the device for the exl3_moe fused kernel: up to FUSED_ROWS tokens no expert can hold more rows than its temp buffers,
so the call stays device-only (HIP-graph safe); above that the counts come back to the host once (one sync a layer,
prefill only) and experts with more than RECON_ROWS rows take the reconstruct tier (dequantize the expert's three
matrices into one f16 scratch, hipBLAS gemms). Accumulation is deterministic by default: every local assignment owns
an f32 slot, exl3_moe_gather sums each token's slots in top-k order (R9K_EXL3_MOE_ATOMIC=1: the kernel's atomic adds
into the output instead). The output is this rank's partial sum; the MoE runner adds the shared expert (not ours: no
moe_kernel, so the runner runs it) and all-reduces.

Streams: not safe beside other EXL3 work on a second stream of the same device. exl3_mgemm (RDNA GEMV) keeps its
argument pointers in one per-device parameter block, shared with Exl3LinearMethod's exl3_gemm; exl3_moe's group
barriers use exllamav3's per-device lock array (DevCtx), shared with the cooperative exl3_gemm. This method runs on the
caller's stream only; the shared expert must not be an EXL3 linear on vLLM's shared-experts aux stream.
"""
from __future__ import annotations

import os
import re

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

from .exl3 import ext, k_of

logger = init_logger("vllm.r9700_vllm.exl3_moe")

# exllamav3 util/backend.py MOE_FUSED_ROWS on ROCm: the RDNA fused kernel pipelines and tiles rows itself and takes
# experts up to 512 rows (no batched reconstruct tier, no 32/64-row instances there)
FUSED_ROWS = int(os.environ.get("R9K_EXL3_MOE_FUSED_ROWS", 512))
# Above FUSED_ROWS tokens (counts read back anyway): experts with more than RECON_ROWS rows take the reconstruct tier
# (measured on gfx1201, random routing: 8k tokens 27.0 -> 22.9 ms a layer, 16k 50.3 -> 29.5 ms; no gain below 128)
RECON_ROWS = min(int(os.environ.get("R9K_EXL3_MOE_RECON_ROWS", 128)), FUSED_ROWS)
# Reconstruct tier pieces: a heavy expert with more than DQ_ROWS rows goes through dequant + hipBLAS in pieces of exactly
# DQ_ROWS rows (the last one ends at the expert's last row and overlaps the one before it), with the fused tier's state /
# intermediate buffers (idle once its kernel has run; C x FUSED_ROWS = 4,096 rows on gfx1201) as the scratch, so the tier
# allocates nothing a call. Exact: on this pin hipBLAS computes every row of an M >= 4,096-row f16 GEMM of these shapes
# with the same reduction as M = 4,096 (below it the result depends on M: classes 1-127 / 128-4095 / 512, 1536, 3000 ...),
# so 4,096-row pieces reproduce the one-piece product bit for bit, and up to 4,096 rows the call is one piece as before
# (bench/ds41/dqchunk_test.py). Was one piece of up to (k + 1) x NBT rows: the DSpark capture's dummy batch puts 12,288
# rows on one expert and its ~0.6 GiB of transients bounded the KV budget (notes/exl3-serving.md 3b).
DQ_ROWS = int(os.environ.get("R9K_EXL3_MOE_DQ_ROWS", 4096))
ATOMIC = os.environ.get("R9K_EXL3_MOE_ATOMIC", "0") == "1"
# Decode tier for up to DECODE_ROWS tokens (R9K_ROWS for r9k):
#   r9k    kernels/r9k_moe_exl3.hip (moe/exl3_decode.py): two launches a layer, local slots compacted on device,
#          slots sharing an expert decoded together, deterministic; falls back to mgemv when unavailable
#   mgemv  exl3_mgemm over the picked experts (the RDNA multi-matrix GEMV, exl3_mgemv_rdna.cu): gate, up, down as one
#          launch each over every local (token, pick) slot, compacted in-kernel; up to 128 slots (the kernel's
#          compaction capacity), 2.5-3.5x faster than coop / the fused tier at decode sizes on gfx1201
#   coop   exl3_moe_coop (BC_BlockSparseMLP.run_bszN, exllamav3's own decode path), up to MAX_BSZN tokens
#   fused  the exl3_moe fused tier as for prefill
DECODE = os.environ.get("R9K_EXL3_MOE_DECODE", "r9k")
DECODE_ROWS = int(os.environ.get("R9K_EXL3_MOE_MGEMM_M", 16))
R9K_ROWS = int(os.environ.get("R9K_EXL3_MOE_R9K_M", 48))   # 8 streams x a 6-token verify (scratch sized once for it)
_R9K_SLOTS: list = []  # the kernel's slot capacity (T x topk), read once from libr9k
MAX_BSZN = 8          # exllamav3_ext/libtorch/blocksparse_mlp.h: exl3_moe_coop scratch bound
MGEMV_SLOTS = 128     # exl3_mgemv_try_launch: bszm <= 128
MOE_ACT_SILU = 0  # exllamav3_ext/quant/exl3_moe_common.cuh

ROLES = ("gate", "up", "down")             # checkpoint w1, w3, w2
SHARD_ROLE = {"w1": 0, "w3": 1, "w2": 2}
FIELDS = ("trellis", "suh", "svh", "mul1", "mcg")

_shared: dict = {}
_warmed: set = set()


def _shared_buffer(device: torch.device, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """One buffer per (device, name, shape) for every MoE layer of the process (exllamav3's g_tensor_cache)."""
    key = (device.index, name, shape, dtype)
    t = _shared.get(key)
    if t is None:
        t = _shared[key] = torch.empty(shape, dtype=dtype, device=device)
    return t


class _Stash:
    """Per-layer expert tensors: one device buffer per (role, field) holding every local expert, plus load bookkeeping."""

    def __init__(self, layer, E: int, H: int, I: int, device: torch.device):
        self.E, self.H, self.I, self.device = E, H, I, device
        self.dims = ((H, I), (H, I), (I, H))   # (in, out) per role
        self.buf: dict = {}
        self.seen = {(r, f): set() for r in range(3) for f in FIELDS}
        emap = getattr(layer, "expert_map", None)
        self.g2l = None if emap is None else emap.tolist()

    def local(self, expert_id: int) -> int:
        return expert_id if self.g2l is None else self.g2l[expert_id]

    def put(self, role: int, field: str, e: int, w: torch.Tensor):
        k, n = self.dims[role]
        if field in ("mul1", "mcg"):
            self.seen[role, field].add(e)
            return
        if field == "trellis":
            assert w.dim() == 3 and w.shape[0] * 16 == k and w.shape[1] * 16 == n and w.dtype == torch.int16, \
                f"exl3 moe: {ROLES[role]} trellis {tuple(w.shape)} {w.dtype}, expected [{k // 16}, {n // 16}, 16K] int16"
            shape, dtype = tuple(w.shape), torch.int16
        else:
            shape, dtype = ((k,) if field == "suh" else (n,)), torch.float16
            assert tuple(w.shape) == shape, f"exl3 moe: {ROLES[role]} {field} {tuple(w.shape)}, expected {shape}"
        b = self.buf.get((role, field))
        if b is None:
            b = self.buf[role, field] = torch.empty((self.E,) + shape, dtype=dtype, device=self.device)
        assert b.shape[1:] == shape, f"exl3 moe: {ROLES[role]} {field} shape {tuple(w.shape)} differs between experts " \
                                     f"(K must be uniform within a layer)"
        b[e].copy_(w)
        self.seen[role, field].add(e)


class Exl3MoEMethod(FusedMoEMethodBase):

    def __init__(self, moe, quant_config):
        super().__init__(moe)
        self.quant_config = quant_config

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config):
        return hidden_size, intermediate_size_per_partition

    def create_weights(self, layer, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        H, I = hidden_size, intermediate_size_per_partition
        assert self.moe.moe_parallel_config.tp_size == 1 and I == self.moe.intermediate_size, (
            f"exl3 moe: experts are never split (intermediate {self.moe.intermediate_size} -> {I} a rank); "
            f"serve with --enable-expert-parallel")
        assert H % 128 == 0 and I % 128 == 0, f"exl3 moe: hidden {H} / intermediate {I} not 128-multiples"
        assert self.moe.is_act_and_mul, "exl3 moe: gated SiLU experts only"
        device = torch.device("cuda", torch.cuda.current_device())
        layer._exl3 = _Stash(layer, num_experts, H, I, device)
        for stem in ("w13", "w2"):
            for field in FIELDS:
                p = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)
                p.weight_loader = self._make_loader(layer, field)
                layer.register_parameter(f"{stem}_{field}", p)

    @staticmethod
    def _make_loader(layer, field: str):
        def weight_loader(param, loaded_weight: torch.Tensor, weight_name: str, shard_id: str, expert_id: int,
                          return_success: bool = False):
            st = layer._exl3
            e = st.local(expert_id)
            if e < 0:
                return False if return_success else None
            st.put(SHARD_ROLE[shard_id], field, e, loaded_weight)
            return True if return_success else None
        weight_loader.supports_moe_loading = True
        return weight_loader

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer) -> None:
        st: _Stash = layer._exl3
        E, H, I, dev = st.E, st.H, st.I, st.device
        if not any(st.seen[k] for k in st.seen):
            # dummy load (--load-format dummy): zero output scales, so every expert contributes exact zeros
            K = int(self._k_map_bits(layer) or 3)
            logger.warning_once("exl3 moe: no expert tensors loaded (dummy weights?) -- experts output zeros")
            for r, (k, n) in enumerate(st.dims):
                st.buf[r, "trellis"] = torch.zeros((E, k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev)
                st.buf[r, "suh"] = torch.ones((E, k), dtype=torch.float16, device=dev)
                st.buf[r, "svh"] = torch.zeros((E, n), dtype=torch.float16, device=dev)
                st.seen[r, "mul1"] = set(range(E))
        else:
            for r in range(3):
                for f in ("trellis", "suh", "svh"):
                    missing = E - len(st.seen[r, f])
                    assert missing == 0, f"exl3 moe {layer.layer_name}: {missing} of {E} local experts have no " \
                                         f"{ROLES[r]} {f}"
        mul1 = [len(st.seen[r, "mul1"]) == E for r in range(3)]
        mcg = [len(st.seen[r, "mcg"]) == E for r in range(3)]
        assert all(mul1) != all(mcg) and all(m or c for m, c in zip(mul1, mcg)), \
            f"exl3 moe {layer.layer_name}: the kernel needs one codebook (mul1 or mcg) for every expert and role"
        Ks = [k_of(st.buf[r, "trellis"]) for r in range(3)]
        km = self._k_map_bits(layer)
        if km is not None and any(k != km for k in Ks):
            logger.warning("exl3 moe %s: trellis K %s, exl3_k_map says %s", layer.layer_name, Ks, km)

        def ptrs(t: torch.Tensor) -> torch.Tensor:
            step = t[0].numel() * t.element_size()
            return t.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * step

        for r, role in enumerate(ROLES):
            for f in ("trellis", "suh", "svh"):
                layer.register_buffer(f"exl3_{role}_{f}", st.buf[r, f], persistent=False)
                layer.register_buffer(f"exl3_{role}_{f}_ptrs", ptrs(st.buf[r, f]), persistent=False)
        st.buf.clear()
        for stem in ("w13", "w2"):
            for f in FIELDS:
                delattr(layer, f"{stem}_{f}")

        emap = getattr(layer, "expert_map", None)
        G = self.moe.num_experts
        if emap is None:
            g2l = torch.arange(G, dtype=torch.int64, device=dev)
        else:
            emap = emap.to(dev, torch.int64)
            g2l = torch.where(emap >= 0, emap, torch.full_like(emap, E))
        layer.register_buffer("exl3_g2l", g2l, persistent=False)

        C = ext().exl3_moe_max_concurrency(dev.index)
        R = FUSED_ROWS
        self.fused = (_shared_buffer(dev, "moe_state_g", (C, R, H), torch.float16),
                      _shared_buffer(dev, "moe_state_u", (C, R, H), torch.float16),
                      _shared_buffer(dev, "moe_interm_g", (C, R, I), torch.float16),
                      _shared_buffer(dev, "moe_interm_u", (C, R, I), torch.float16))
        self.dq_temp = _shared_buffer(dev, "moe_dq_temp", (H * I,), torch.float16)
        S = MAX_BSZN * self.moe.experts_per_token
        ctr_len = S * (I // 128) + MAX_BSZN * (H // 128) + 2 + (S + 1) + S   # exl3_moe_coop_ctr_len
        ctr = _shared.get((dev.index, "moe_coop_ctr"))
        if ctr is None or ctr.numel() < ctr_len:
            ctr = _shared[dev.index, "moe_coop_ctr"] = torch.zeros((ctr_len,), dtype=torch.int32, device=dev)
        self.coop = (_shared_buffer(dev, "moe_coop_had_g", (S, H), torch.float16),
                     _shared_buffer(dev, "moe_coop_had_u", (S, H), torch.float16),
                     _shared_buffer(dev, "moe_coop_gu_g", (S, I), torch.float16),
                     _shared_buffer(dev, "moe_coop_gu_u", (S, I), torch.float16),
                     _shared_buffer(dev, "moe_coop_act", (S, I), torch.float16),
                     _shared_buffer(dev, "moe_coop_dout", (S, H), torch.float32),
                     ctr,
                     _shared_buffer(dev, "moe_coop_out", (MAX_BSZN, H), torch.float32))
        S = MGEMV_SLOTS
        self.mgemv = (_shared_buffer(dev, "moe_mgemv_g", (S, 1, I), torch.float16),
                      _shared_buffer(dev, "moe_mgemv_u", (S, 1, I), torch.float16),
                      _shared_buffer(dev, "moe_mgemv_a", (S, 1, I), torch.float16),
                      _shared_buffer(dev, "moe_mgemv_d", (S, 1, H), torch.float32),
                      _shared_buffer(dev, "moe_mgemv_had", (S * H,), torch.float16),
                      torch.arange(S, device=dev) // self.moe.experts_per_token)
        self.K = Ks
        self.mul1, self.mcg = all(mul1), all(mcg)
        limit = getattr(layer, "swiglu_limit", None) or self.moe.swiglu_limit
        self.act_limit = float(limit or 0.0)
        self._warm(layer)
        logger.info_once("exl3 moe: %d local experts a layer, K %s, fused rows %d, concurrency %d (%.0f MiB fused "
                         "buffers), act_limit %g, %s accumulation", E, tuple(Ks), R, C, C * R * (H + I) * 4 / 2**20,
                         self.act_limit, "atomic" if ATOMIC else "deterministic")

    def _k_map_bits(self, layer):
        km = getattr(self.quant_config, "k_map", None) or {}
        km = km.get("routed", km)
        m = re.search(r"layers\.(\d+)\.", getattr(layer, "layer_name", "") + ".")
        return (km.get(int(m.group(1))) or km.get(m.group(1))) if m else None

    def _warm(self, layer) -> None:
        """One call per tier and decode size before vLLM captures: the RDNA GEMV allocates its per-device parameter
        block on first use (cudaMalloc, fatal inside a capture) and exl3_moe / exl3_moe_coop set kernel attributes
        and cache occupancy per instance on first launch."""
        key = (layer._exl3.device.index, tuple(self.K), self.mul1)
        if key in _warmed:
            return
        dev, H, topk = layer._exl3.device, layer._exl3.H, self.moe.experts_per_token
        local = (layer.exl3_g2l < layer._exl3.E).nonzero().flatten()[:topk].to(torch.int32)
        rows = self._decode_rows()
        if DECODE == "r9k" and self._r9k_ready(layer):
            from ..moe import exl3_decode as XD
            XD.reserve(dev, rows, topk, H, layer._exl3.I)
        sizes = sorted({t for t in (1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, rows) if 0 < t <= rows} | {rows + 1, 64})
        for T in sizes:
            x = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
            ids = local.expand(T, -1).contiguous()
            w = torch.full((T, topk), 1.0 / topk, dtype=torch.float32, device=dev)
            self.apply(layer, x, w, ids)
        torch.cuda.synchronize(dev)
        _warmed.add(key)

    def _decode_rows(self) -> int:
        return R9K_ROWS if DECODE == "r9k" else DECODE_ROWS

    def _r9k_ready(self, layer) -> bool:
        """The r9k tier takes this layer: libr9k has the kernel, one K for the three roles (2-6), mul1 or mcg."""
        ok = getattr(layer, "_exl3_r9k", None)
        if ok is None:
            from ..moe import exl3_decode as XD
            ok = XD.available() and len(set(self.K)) == 1 and 2 <= self.K[0] <= 6 and (self.mul1 or self.mcg)
            if ok:
                layer._exl3_r9k_tab = XD.Tables(layer)
                if not _R9K_SLOTS:
                    _R9K_SLOTS.append(XD.max_slots())
            elif DECODE == "r9k":
                logger.warning_once("exl3 moe: r9k decode tier unavailable (libr9k without r9k_moe_exl3_decode, "
                                    "or K %s / codebook unsupported): mgemv decode", tuple(self.K))
            layer._exl3_r9k = ok
        return ok

    def _apply_r9k(self, layer, x, topk_weights, topk_ids):
        from ..moe import exl3_decode as XD
        return XD.moe_decode(x, topk_weights, topk_ids, layer.exl3_g2l, layer._exl3.E, layer._exl3_r9k_tab,
                             self.K[0], 2 if self.mul1 else 1, self.act_limit, layer._exl3.I)

    def _run_fused(self, layer, x16, out, count, token_sorted, weight_sorted, num_active, scratch, base, hi):
        L = layer
        ext().exl3_moe(
            x16, out, count, token_sorted, weight_sorted, *self.fused, MOE_ACT_SILU,
            float(self.K[0]), float(self.K[1]), float(self.K[2]),
            L.exl3_gate_trellis_ptrs, L.exl3_gate_suh_ptrs, L.exl3_gate_svh_ptrs,
            L.exl3_up_trellis_ptrs, L.exl3_up_suh_ptrs, L.exl3_up_svh_ptrs,
            L.exl3_down_trellis_ptrs, L.exl3_down_suh_ptrs, L.exl3_down_svh_ptrs,
            self.mcg, self.mul1, self.mcg, self.mul1, self.mcg, self.mul1,
            self.act_limit, num_active, scratch, base, 1, hi, 16)

    def _gemm_dq(self, layer, role: str, r: int, e: int, a: torch.Tensor, out: torch.Tensor, ah: torch.Tensor):
        """out = a @ W_e (original basis) through reconstruct + hipBLAS: had(a * suh) @ W_rot, then had(.) * svh.
        ah: scratch shaped like a."""
        X = ext()
        trellis = getattr(layer, f"exl3_{role}_trellis")[e]
        k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
        X.had_r_128(a, ah, getattr(layer, f"exl3_{role}_suh")[e], None, 1.0)
        w = self.dq_temp[:k * n].view(k, n)
        X.reconstruct(w, trellis, self.K[r], self.mcg, self.mul1)
        X.hgemm(ah, w, out)
        X.had_r_128(out, out, None, getattr(layer, f"exl3_{role}_svh")[e], 1.0)

    def _dq_scratch(self, dev: torch.device, H: int, I: int) -> tuple:
        """(x rows, Hadamard input, gate, up) flat f16 scratch of DQ_ROWS rows: the fused tier's buffers when they hold
        that many rows (they do on gfx1201: concurrency 8 x 512), else one shared set for the process."""
        C, R = self.fused[0].shape[:2]
        if C * R >= DQ_ROWS:
            return tuple(b.view(-1) for b in self.fused)
        return tuple(_shared_buffer(dev, f"moe_dq_{n}", (DQ_ROWS * d,), torch.float16)
                     for n, d in (("x", H), ("had", H), ("g", I), ("u", I)))

    def _expert_dq(self, layer, e: int, x16: torch.Tensor, tok: torch.Tensor, out: torch.Tensor | None,
                   weight: torch.Tensor | None = None, acc: torch.Tensor | None = None):
        """Unweighted expert MLP for a heavy expert (rows > RECON_ROWS) over the rows x16[tok]: out f32 [c, H] gets row
        i's result (deterministic path), or, with out None, acc.index_add_(tok, y * weight) (R9K_EXL3_MOE_ATOMIC).
        Up to DQ_ROWS rows one piece, above that pieces of exactly DQ_ROWS rows, the last one overlapping (see DQ_ROWS);
        scratch from _dq_scratch (this call runs after the fused kernel, on its stream)."""
        X = ext()
        H, I = layer._exl3.H, layer._exl3.I
        sx, sh, sg, su = self._dq_scratch(x16.device, H, I)
        c = tok.numel()
        n = min(c, DQ_ROWS)
        starts = [0] if c <= DQ_ROWS else list(range(0, c - DQ_ROWS, DQ_ROWS)) + [c - DQ_ROWS]
        done = 0
        for r0 in starts:
            t = tok[r0:r0 + n]
            xe = sx[:n * H].view(n, H)
            torch.index_select(x16, 0, t, out=xe)
            g, u = sg[:n * I].view(n, I), su[:n * I].view(n, I)
            self._gemm_dq(layer, "gate", 0, e, xe, g, sh[:n * H].view(n, H))
            self._gemm_dq(layer, "up", 1, e, xe, u, sh[:n * H].view(n, H))
            X.silu_mul(g, u, g, self.act_limit)
            y = out[r0:r0 + n] if out is not None else torch.empty((n, H), dtype=torch.float32, device=x16.device)
            self._gemm_dq(layer, "down", 2, e, g, y, sh[:n * I].view(n, I))
            if out is None:     # the overlapped rows of the last piece are already in acc
                k0 = done - r0
                acc.index_add_(0, t[k0:], y[k0:] * weight[r0 + k0:r0 + n].float().unsqueeze(1))
            done = r0 + n

    def _apply_coop(self, layer, x, topk_weights, topk_ids):
        L, E = layer, layer._exl3.E
        sel = L.exl3_g2l[topk_ids.long()]
        had_g, had_u, gu_g, gu_u, act, dout, ctr, out = self.coop
        ext().exl3_moe_coop(
            x.to(torch.float16).contiguous(), sel, topk_weights.to(torch.float16).contiguous(), 0, E, x.shape[1],
            L.exl3_gate_trellis_ptrs, L.exl3_gate_suh_ptrs, L.exl3_gate_svh_ptrs,
            L.exl3_up_trellis_ptrs, L.exl3_up_suh_ptrs, L.exl3_up_svh_ptrs,
            L.exl3_down_trellis_ptrs, L.exl3_down_suh_ptrs, L.exl3_down_svh_ptrs,
            None, None, None, float(self.K[0]), float(self.K[1]), float(self.K[2]), self.mcg, self.mul1,
            MOE_ACT_SILU, self.act_limit, True, had_g, had_u, gu_g, gu_u, act, dout, ctr, out, None, None)
        return out[:x.shape[0]].to(x.dtype)

    def _apply_mgemv(self, layer, x, topk_weights, topk_ids):
        """Every (token, pick) slot through exl3_mgemm, local experts only: the kernel compacts the in-range slots
        (local id < E) to the front, in slot order, and reads per-slot inputs at their packed row, so the multi-token
        input is gathered into packed order first (one token: the input row is broadcast and the down projection
        reduces the weighted slots itself)."""
        L, E, X = layer, layer._exl3.E, ext()
        T, H = x.shape
        S = topk_ids.numel()
        g, u, a, d, had, slot_tok = self.mgemv
        g, u, a, d = g[:S], u[:S], a[:S], d[:S]
        Kg, Ku, Kd = (float(k) for k in self.K)
        sel = L.exl3_g2l[topk_ids.reshape(-1).long()]
        idx = sel.view(1, S)
        x16 = x.to(torch.float16)
        if T == 1:
            xp = x16.view(1, 1, H)
        else:
            valid = sel < E
            pos = torch.cumsum(valid, 0) - 1
            tok = torch.zeros((S + 1,), dtype=torch.int64, device=x.device)
            tok.scatter_(0, torch.where(valid, pos, S), slot_tok[:S])
            xp = x16.index_select(0, tok[:S]).view(S, 1, H)
        X.exl3_mgemm(xp, L.exl3_gate_trellis_ptrs, g, L.exl3_gate_suh_ptrs, had, L.exl3_gate_svh_ptrs, idx, None, Kg,
                     -1, self.mcg, self.mul1, 0, E, 0)
        X.exl3_mgemm(xp, L.exl3_up_trellis_ptrs, u, L.exl3_up_suh_ptrs, had, L.exl3_up_svh_ptrs, idx, None, Ku,
                     -1, self.mcg, self.mul1, 0, E, 0)
        X.silu_mul(g.view(S, -1), u.view(S, -1), a.view(S, -1), self.act_limit)
        if T == 1:
            d[0].zero_()                  # no local pick: the kernel writes nothing
            X.exl3_mgemm(a, L.exl3_down_trellis_ptrs, d, L.exl3_down_suh_ptrs, had, L.exl3_down_svh_ptrs, idx,
                         topk_weights.to(torch.float16).view(1, S), Kd, -1, self.mcg, self.mul1, 0, E, 0)
            return d[0].to(x.dtype)
        X.exl3_mgemm(a, L.exl3_down_trellis_ptrs, d, L.exl3_down_suh_ptrs, had, L.exl3_down_svh_ptrs, idx, None, Kd,
                     -1, self.mcg, self.mul1, 0, E, 0)
        y = d.view(S, H)[pos.clamp(min=0)] * topk_weights.reshape(S, 1)
        return torch.where(valid.unsqueeze(1), y, 0.0).view(T, -1, H).sum(1).to(x.dtype)

    def apply(self, layer, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, shared_experts=None,
              shared_experts_input: torch.Tensor | None = None):
        # shared_experts: the runner runs them (moe_kernel is None, so mk_can_overlap_shared_experts is False)
        E = layer._exl3.E
        T, H = x.shape
        if DECODE == "r9k" and 0 < T <= R9K_ROWS and self._r9k_ready(layer) \
                and T * topk_ids.shape[1] <= _R9K_SLOTS[0]:
            return self._apply_r9k(layer, x, topk_weights, topk_ids)
        if 0 < T <= DECODE_ROWS:
            if DECODE in ("mgemv", "r9k") and topk_ids.numel() <= MGEMV_SLOTS:
                return self._apply_mgemv(layer, x, topk_weights, topk_ids)
            if DECODE == "coop" and T <= MAX_BSZN:
                return self._apply_coop(layer, x, topk_weights, topk_ids)
        topk = topk_ids.shape[1]
        A = T * topk
        dev = x.device
        x16 = x.to(torch.float16).contiguous()
        local = layer.exl3_g2l[topk_ids.reshape(-1).long()]
        count = torch.zeros((E + 1,), dtype=torch.int64, device=dev).scatter_add_(0, local, torch.ones_like(local))
        order = local.argsort(stable=True)
        token_sorted = order // topk
        weight_sorted = topk_weights.reshape(-1)[order].to(torch.float16)
        out = torch.zeros((T, H), dtype=torch.float32, device=dev)

        heavy, num_active, slots, hi = [], -1, A, FUSED_ROWS
        if T > FUSED_ROWS:
            hi = RECON_ROWS
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"exl3 moe: {T} tokens under graph capture; the fused tier holds {FUSED_ROWS} rows "
                                   f"(cap max_cudagraph_capture_size at {FUSED_ROWS})")
            cnt = count.tolist()
            starts = [0] * (E + 1)
            for i in range(E):
                starts[i + 1] = starts[i] + cnt[i]
            heavy = [(i, starts[i], cnt[i]) for i in range(E) if cnt[i] > hi]
            num_active = sum(1 for i in range(E) if 0 < cnt[i] <= hi)
            slots = starts[E]

        if ATOMIC:
            if num_active:
                self._run_fused(layer, x16, out, count, token_sorted, weight_sorted, num_active, None, None, hi)
            for e, s, c in heavy:
                self._expert_dq(layer, e, x16, token_sorted[s:s + c], None, weight_sorted[s:s + c], out)
        else:
            # slot of a local assignment = its sorted position (the sentinel bin sorts last); kind 1 = fused (slot
            # weighted by the kernel), 2 = reconstruct tier (unweighted, weighted by the gather)
            start = torch.cumsum(count, 0) - count
            kind = (count > 0).long() + (count > hi).long()
            scratch = torch.empty((max(slots, 1), H), dtype=torch.float32, device=dev)
            if num_active:
                self._run_fused(layer, x16, out, count, token_sorted, weight_sorted, num_active, scratch, start, hi)
            for e, s, c in heavy:
                self._expert_dq(layer, e, x16, token_sorted[s:s + c], scratch[s:s + c])
            inv_order = torch.empty_like(order).scatter_(0, order, torch.arange(A, device=dev))
            ext().exl3_moe_gather(out, scratch, local, inv_order, start[:E], start[:E], kind[:E], weight_sorted)
        return out.to(x.dtype)
