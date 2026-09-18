"""Per-layer device-side LRU expert cache for R9700Mxfp4Experts (stock vLLM, gfx1201).

Every routed expert of a layer lives in pinned host memory (UVA): fragment-order weights w13/w2 and packed scales
s13/s2 -- the backing store. A VRAM arena of S slots per layer holds a mutable subset. Each forward, before
the GEMMs, davetha's r4d_lru_manage (vendored, Apache-2.0) refreshes the resident experts routed this step,
picks victims among slots not routed this step (LRU by step stamp, deterministic so TP ranks agree), rewrites
table/map_cold and emits a miss list; r4d_lru_gather then copies the missed experts host -> slot. The GEMMs run
twice: over the arena with moe_align(expert_map=table) and over the host store with
moe_align(expert_map=map_cold) -- the second is empty unless the step read through (too many distinct experts,
or the insert cap was hit). All of it is device-side and pointer-stable, so it is cudagraph-safe.

Config (per rank): R9K_EXPERT_CACHE_GB (0 = off), R9K_LRU_THRESH (0.5), R9K_LRU_MAX_INSERTS (64),
R9K_LRU_GATHER="chunks,lanes" (8,16). Warm start: the checkpoint's model-expertprofile.safetensors
(expert_routing_counts [layers, experts]) if present, else experts 0..S-1.
"""
from __future__ import annotations

import ctypes
import os

import torch

from vllm.logger import init_logger

from ..kernels import moe as K

logger = init_logger(__name__)

_BOUND = False
_PROFILE: torch.Tensor | None = None
_PROFILE_LOADED = False


def _L():
    global _BOUND
    L = K.lib()
    if not _BOUND:
        L.r4d_lru_manage.restype = ctypes.c_int
        L.r4d_lru_manage.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int] + [ctypes.c_void_p] * 8 + [ctypes.c_void_p]
        L.r4d_lru_fused.restype = ctypes.c_int
        L.r4d_lru_fused.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int] + [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + \
            [ctypes.c_void_p] * 6 + [ctypes.c_void_p]
        L.r4d_lru_gather.restype = ctypes.c_int
        L.r4d_lru_gather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long] * 6 + \
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _BOUND = True
    return L


def _uva_empty(shape, dtype) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = torch.empty(shape, dtype=dtype, pin_memory=True)
    v = get_accelerator_view_from_cpu_tensor(host)
    v._r9k_host = host  # keep the pinned allocation alive with the view
    return v


def is_host(t: torch.Tensor) -> bool:
    """True if a (cuda-typed) tensor is a UVA view of pinned host memory made by vLLM's offloader or by us."""
    return bool(getattr(t, "_vllm_is_uva_offloaded", False) or getattr(t, "_r9k_host", None) is not None)


def to_host(t: torch.Tensor) -> torch.Tensor:
    """Copy a device tensor into a fresh pinned-host UVA view (chunked along dim 0)."""
    h = _uva_empty(t.shape, t.dtype)
    step = max(1, (256 << 20) // max(1, t[0].numel() * t.element_size()))
    for i in range(0, t.shape[0], step):
        h[i:i + step].copy_(t[i:i + step])
    return h


def budget_gb() -> float:
    return float(os.environ.get("R9K_EXPERT_CACHE_GB", "0") or 0)


def _profile() -> torch.Tensor | None:
    global _PROFILE, _PROFILE_LOADED
    if _PROFILE_LOADED:
        return _PROFILE
    _PROFILE_LOADED = True
    try:
        from vllm.config import get_current_vllm_config
        from safetensors.torch import load_file
        p = os.path.join(get_current_vllm_config().model_config.model, "model-expertprofile.safetensors")
        if os.path.isfile(p):
            _PROFILE = load_file(p)["expert_routing_counts"].float()
            logger.info_once("r9700: expert routing profile %s %s", p, tuple(_PROFILE.shape))
    except Exception as e:
        logger.warning("r9700: no expert routing profile (%s)", e)
    return _PROFILE


def _num_moe_layers() -> int:
    try:
        from vllm.config import get_current_vllm_config
        hf = get_current_vllm_config().model_config.hf_text_config
        return int(getattr(hf, "num_hidden_layers"))
    except Exception:
        return 48


class LayerCache:
    def __init__(self, layer_idx: int, w13: torch.Tensor, w2: torch.Tensor, s13: torch.Tensor, s2: torch.Tensor,
                 N1: int, K1: int, N2: int, K2: int, slots: int):
        dev = torch.device("cuda", torch.cuda.current_device())
        E = w13.shape[0]
        self.E, self.S, self.layer_idx = E, min(slots, E, 1024), layer_idx
        self.N1, self.K1, self.N2, self.K2 = N1, K1, N2, K2
        # host backing store (all E): weights already UVA (caller guarantees), scales moved to host
        self.h_w13, self.h_w2 = w13, w2
        self.h_s13, self.h_s2 = _uva_empty(s13.shape, torch.uint8), _uva_empty(s2.shape, torch.uint8)
        self.h_s13.copy_(s13)
        self.h_s2.copy_(s2)
        self.h_dummy = _uva_empty((E, 16), torch.uint8)
        S = self.S
        self.a_w13 = torch.empty((S, w13.shape[1]), dtype=w13.dtype, device=dev)
        self.a_w2 = torch.empty((S, w2.shape[1]), dtype=w2.dtype, device=dev)
        self.a_s13 = torch.empty((S, *s13.shape[1:]), dtype=torch.uint8, device=dev)
        self.a_s2 = torch.empty((S, *s2.shape[1:]), dtype=torch.uint8, device=dev)
        self.a_dummy = torch.empty((S, 16), dtype=torch.uint8, device=dev)
        self.bytes = [w13[0].numel() * w13.element_size(), w2[0].numel() * w2.element_size(),
                      s13[0].numel(), s2[0].numel()]
        for b in self.bytes:
            assert b % 16 == 0, self.bytes
        # LRU state
        i32 = dict(dtype=torch.int32, device=dev)
        self.table = torch.full((E,), -1, **i32)
        self.map_cold = torch.arange(E, **i32)
        self.slot_expert = torch.full((S,), -1, **i32)
        self.slot_stamp = torch.zeros((S,), dtype=torch.int64, device=dev)
        self.routed = torch.zeros((E,), dtype=torch.uint8, device=dev)
        self.step = torch.zeros((1,), dtype=torch.int64, device=dev)
        self.max_inserts = min(int(os.environ.get("R9K_LRU_MAX_INSERTS", "64")), S)
        self.max_distinct = int(S * float(os.environ.get("R9K_LRU_THRESH", "0.5")))
        self.miss = torch.full((max(1, self.max_inserts), 2), -1, **i32)
        # rows per step up to which the cold (host read-through) pass cannot have work: distinct <= rows
        self.no_cold_limit = min(self.max_distinct, self.max_inserts) if S > self.max_distinct else 0
        self.n_miss = torch.zeros((1,), **i32)
        self.fused = os.environ.get("R9K_LRU_FUSED", "1") == "1" and E <= 1024
        self._align: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        g = os.environ.get("R9K_LRU_GATHER", "8,16").split(",")
        self.chunks, self.lanes = int(g[0]), int(g[1])
        self._warm_start()

    def _warm_start(self):
        prof = _profile()
        if prof is not None and self.layer_idx < prof.shape[0] and prof.shape[1] == self.E:
            order = torch.argsort(prof[self.layer_idx], descending=True)
        else:
            order = torch.arange(self.E)
        init = order[: self.S].to(torch.int32)
        n = init.numel()
        dev = self.table.device
        slots = torch.arange(n, dtype=torch.int32)
        pairs = torch.stack([init, slots], dim=1).to(dev)
        # one bulk gather through the same kernel, in max_inserts-sized batches
        for i in range(0, n, max(1, self.max_inserts)):
            chunk = pairs[i: i + self.max_inserts]
            self.miss[: chunk.shape[0]].copy_(chunk)
            self.n_miss.fill_(chunk.shape[0])
            self._gather()
        self.n_miss.zero_()
        self.miss.fill_(-1)
        self.table[init.long().to(dev)] = slots.to(dev)
        self.map_cold[init.long().to(dev)] = -1
        self.slot_expert[:n] = init.to(dev)
        # older stamps for colder warm experts so the hottest are evicted last
        self.slot_stamp[:n] = torch.arange(n, 0, -1, dtype=torch.int64, device=dev) - n
        torch.cuda.synchronize()

    def _gather(self):
        st = torch.cuda.current_stream().cuda_stream
        b = self.bytes
        rc = _L().r4d_lru_gather(
            self.a_w13.data_ptr(), self.h_w13.data_ptr(), b[0],
            self.a_w2.data_ptr(), self.h_w2.data_ptr(), b[1],
            self.a_s13.data_ptr(), self.h_s13.data_ptr(), b[2],
            self.a_s2.data_ptr(), self.h_s2.data_ptr(), b[3],
            self.a_dummy.data_ptr(), self.h_dummy.data_ptr(), 16,
            self.a_dummy.data_ptr(), self.h_dummy.data_ptr(), 16,
            self.miss.data_ptr(), self.n_miss.data_ptr(), self.chunks, self.lanes, st)
        if rc:
            raise RuntimeError(f"r4d_lru_gather failed ({rc})")

    def update(self, topk_ids: torch.Tensor) -> None:
        ids = topk_ids.reshape(-1)
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        st = torch.cuda.current_stream().cuda_stream
        rc = _L().r4d_lru_manage(ids.data_ptr(), ids.numel(), self.E, self.S, self.max_distinct, self.max_inserts,
                                 self.table.data_ptr(), self.map_cold.data_ptr(), self.slot_expert.data_ptr(),
                                 self.slot_stamp.data_ptr(), self.routed.data_ptr(), self.step.data_ptr(),
                                 self.miss.data_ptr(), self.n_miss.data_ptr(), st)
        if rc:
            raise RuntimeError(f"r4d_lru_manage failed ({rc})")
        self._gather()

    def _align_bufs(self, mk: int, bs: int):
        """Persistent align outputs for (mk, bs), sized exactly as vLLM's moe_align_block_size would."""
        key = (mk, bs)
        b = self._align.get(key)
        if b is None:
            E = self.E
            L = min(mk * bs, mk + E * (bs - 1)) if mk < E else mk + E * (bs - 1)
            NB = (L + bs - 1) // bs
            dev = self.table.device
            i32 = dict(dtype=torch.int32, device=dev)
            b = (L, NB, torch.empty((L,), **i32), torch.empty((NB,), **i32), torch.empty((1,), **i32),
                 torch.empty((L,), **i32), torch.empty((NB,), **i32), torch.empty((1,), **i32))
            self._align[key] = b
        return b

    def update_fused(self, topk_ids: torch.Tensor, bs: int):
        """LRU manage + both moe_align outputs (hot over slots, cold over host) in one launch, then gather.
        Returns ((sorted, eids, npad) hot, (sorted, eids, npad) cold)."""
        ids = topk_ids.reshape(-1)
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        mk = ids.numel()
        L, NB, sh, eh, nh, sc, ec, nc = self._align_bufs(mk, bs)
        st = torch.cuda.current_stream().cuda_stream
        rc = _L().r4d_lru_fused(ids.data_ptr(), mk, self.E, self.S, self.max_distinct, self.max_inserts,
                                self.table.data_ptr(), self.map_cold.data_ptr(), self.slot_expert.data_ptr(),
                                self.slot_stamp.data_ptr(), self.routed.data_ptr(), self.step.data_ptr(),
                                self.miss.data_ptr(), self.n_miss.data_ptr(), bs, L, NB,
                                sh.data_ptr(), eh.data_ptr(), nh.data_ptr(), sc.data_ptr(), ec.data_ptr(),
                                nc.data_ptr(), st)
        if rc:
            raise RuntimeError(f"r4d_lru_fused failed ({rc})")
        self._gather()
        return (sh, eh, nh), (sc, ec, nc)

    def hot(self):
        return K.Mxfp4Experts(self.a_w13, self.a_s13, self.N1, self.K1), K.Mxfp4Experts(self.a_w2, self.a_s2, self.N2, self.K2)

    def cold(self):
        return K.Mxfp4Experts(self.h_w13, self.h_s13, self.N1, self.K1), K.Mxfp4Experts(self.h_w2, self.h_s2, self.N2, self.K2)


def slots_per_layer(bytes_per_expert: int) -> int:
    gb = budget_gb()
    if gb <= 0:
        return 0
    return int(gb * 2**30 // (_num_moe_layers() * bytes_per_expert))
