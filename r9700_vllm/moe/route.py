"""MoE routing in one launch at decode widths (kernels/r9k_moe_route.hip): softmax top-k, renormalisation and
vLLM's moe_align_block_size tables together.

Stock vLLM routes a Flash-Next decode step with four launches per MoE layer -- ``topk_softmax`` in the router,
then ``moe_align_block_size`` + ``count_and_sort_expert_tokens`` (and their fills) inside the experts' apply --
192 HIP-graph nodes of a few microseconds each per step. Here the router is replaced by ``R9kTopKRouter`` (a
``FusedTopKRouter`` whose ``_compute_routing`` runs ``torch.ops.r9700.moe_route``) and the routing tables it
produces are handed to ``R9700Mxfp4Experts.apply`` through the ``RoutedExperts`` object the two share, so the
experts skip their own align. Rows above ``MAX_ROWS`` (prefill chunks) take the stock path unchanged.

Numerics: fp32 softmax and top-k on the bf16 logits as stock does; ids agree except on near-ties of the
probabilities, weights differ at the fp32 ulp level (row sums in wave-tree order). ``R9K_MOE_ROUTE=stock`` keeps
vLLM's router.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from ..kernels import moe as KM
from ..ops import _LIB

logger = init_logger("vllm." + __name__)

MAX_ROWS = int(os.environ.get("R9K_MOE_ROUTE_MAX_ROWS", "256"))
_L = None
_DONE = False


def lib():
    global _L
    if _L is None:
        L = KM.lib()
        if not hasattr(L, "r9k_moe_route"):
            raise RuntimeError("libr9k.so has no r9k_moe_route (rebuild kernels/)")
        L.r9k_moe_route.restype = ctypes.c_int
        L.r9k_moe_route.argtypes = [ctypes.c_long, ctypes.c_int] + [ctypes.c_long] * 6 + [ctypes.c_int] * 5 \
            + [ctypes.c_long]
        L.r9k_moe_route_max_experts.restype = ctypes.c_int
        L.r9k_moe_route_max_topk.restype = ctypes.c_int
        _L = L
    return _L


def available() -> bool:
    try:
        lib()
        return True
    except Exception:
        return False


def capacity(numel: int, num_experts: int, blk: int) -> int:
    """sorted_ids length for `numel` routed rows, as vLLM's moe_align_block_size sizes it."""
    cap = numel + num_experts * (blk - 1)
    return (cap + blk - 1) // blk * blk


def scratch_ints(E: int) -> int:
    """int32 count of the kernel's scratch (counts[E], cursor[E], arrive): zero it once at allocation."""
    return 2 * E + 1


def moe_route(logits: torch.Tensor, topk: int, renorm: bool, blk: int, sorted_ids: torch.Tensor,
              expert_ids: torch.Tensor, ntpp: torch.Tensor, scratch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """logits [M, E] bf16 -> (weights fp32 [M, topk], ids int32 [M, topk]); fills the first capacity(M*topk)
    entries of sorted_ids / expert_ids and ntpp[0]. scratch: scratch_ints(E) int32, zero before the first call
    (the kernel leaves it zero)."""
    M, E = logits.shape
    w = torch.empty((M, topk), dtype=torch.float32, device=logits.device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=logits.device)
    rc = lib().r9k_moe_route(logits.data_ptr(), logits.stride(0), w.data_ptr(), ids.data_ptr(),
                             sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(), scratch.data_ptr(),
                             M, E, topk, 1 if renorm else 0, blk, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_moe_route failed ({rc}) M={M} E={E} topk={topk} blk={blk}")
    return w, ids


def _moe_route_fake(logits: torch.Tensor, topk: int, renorm: bool, blk: int, sorted_ids: torch.Tensor,
                    expert_ids: torch.Tensor, ntpp: torch.Tensor, scratch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M = logits.shape[0]
    return (logits.new_empty((M, topk), dtype=torch.float32), logits.new_empty((M, topk), dtype=torch.int32))


def register() -> None:
    global _DONE
    if _DONE:
        return
    direct_register_custom_op("moe_route", moe_route, mutates_args=["sorted_ids", "expert_ids", "ntpp", "scratch"],
                              fake_impl=_moe_route_fake, target_lib=_LIB)
    _DONE = True


@dataclass
class Tables:
    """Routing tables for one call, keyed by the topk_ids tensor the router returned with them."""
    ids: torch.Tensor
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    ntpp: torch.Tensor
    blk: int
    numel: int

    def matches(self, topk_ids: torch.Tensor, blk: int, numel: int) -> bool:
        return (self.ids.data_ptr() == topk_ids.data_ptr() and self.ids.shape == topk_ids.shape
                and self.blk == blk and self.numel == numel)


def _fits(logits: torch.Tensor, topk: int, E: int) -> bool:
    L = lib()
    return (logits.dim() == 2 and logits.dtype == torch.bfloat16 and logits.stride(1) == 1
            and 1 <= logits.shape[0] <= MAX_ROWS and logits.shape[1] == E and E <= L.r9k_moe_route_max_experts()
            and 1 <= topk <= min(E, L.r9k_moe_route_max_topk()))


def make_router(stock, routed_experts):
    """An R9kTopKRouter standing in for a stock FusedTopKRouter (softmax scoring, no EPLB)."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    class R9kTopKRouter(FusedTopKRouter):
        def __init__(self):
            super().__init__(top_k=stock.top_k, global_num_experts=stock.global_num_experts,
                             scoring_func=stock.scoring_func, renormalize=stock.renormalize,
                             eplb_state=stock.eplb_state)
            self.capture_fn = getattr(stock, "capture_fn", None)
            self._routing_replay_out = getattr(stock, "_routing_replay_out", None)
            self.routed_experts = routed_experts
            self._bufs: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            # Allocate NOW, at model construction. The first fused call otherwise lands inside vLLM's cudagraph
            # memory-profiling pass, whose allocations are released afterwards: the buffers' memory was handed
            # to other tensors and the next capture pass faulted (GPU page fault at the 24-token graph).
            if torch.cuda.is_available():
                self._buffers(torch.device("cuda", torch.cuda.current_device()), stock.global_num_experts)

        def _buffers(self, device: torch.device, E: int):
            b = self._bufs.get(device)
            if b is None:
                # every decode width up to MAX_ROWS rows: blk is 16 there (pick_mt -> 1 below 16 rows an expert)
                cap = max(capacity(m * self.top_k, E, 16 * KM.pick_mt(m * self.top_k, E))
                          for m in range(1, MAX_ROWS + 1))
                b = (torch.empty(cap, dtype=torch.int32, device=device),
                     torch.empty(cap // 16, dtype=torch.int32, device=device),
                     torch.zeros(1, dtype=torch.int32, device=device),
                     torch.zeros(scratch_ints(E), dtype=torch.int32, device=device))
                self._bufs[device] = b
            return b

        def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
            E = self.global_num_experts
            if indices_type not in (None, torch.int32) or not _fits(router_logits, self.top_k, E):
                self.routed_experts._r9k_tables = None
                return super()._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)
            M = router_logits.shape[0]
            numel = M * self.top_k
            blk = KM.MOE_BLOCK * KM.pick_mt(numel, E)
            sorted_ids, expert_ids, ntpp, scratch = self._buffers(router_logits.device, E)
            w, ids = torch.ops.r9700.moe_route(router_logits, self.top_k, self.renormalize, blk, sorted_ids,
                                               expert_ids, ntpp, scratch)
            self.routed_experts._r9k_tables = Tables(ids, sorted_ids, expert_ids, ntpp, blk, numel)
            return w, ids

    return R9kTopKRouter()


def take(layer, topk_ids: torch.Tensor, blk: int, numel: int):
    """The tables the router left for this call on the RoutedExperts `layer`, or None."""
    t = getattr(layer, "_r9k_tables", None)
    if t is None or not t.matches(topk_ids, blk, numel):
        return None
    return t.sorted_ids, t.expert_ids, t.ntpp


def install(model: torch.nn.Module) -> int:
    """Replace the router of every MoE runner under `model` that routes by plain softmax top-k.
    R9K_MOE_ROUTE=stock skips."""
    if os.environ.get("R9K_MOE_ROUTE", "r9k") != "r9k" or not available():
        return 0
    register()
    n = 0
    for name, mod in model.named_modules():
        router = getattr(mod, "router", None)
        re_ = getattr(mod, "routed_experts", None)
        if router is None or re_ is None or type(router).__name__ != "FusedTopKRouter":
            continue
        if getattr(router, "scoring_func", None) != "softmax" or getattr(router, "eplb_state", None) is not None:
            logger.warning_once("r9700: fused MoE routing skipped for %s (scoring %s, eplb %s)", name,
                                getattr(router, "scoring_func", None), getattr(router, "eplb_state", None))
            continue
        mod.router = make_router(router, re_)
        n += 1
    if n:
        logger.info("r9700: fused MoE routing (softmax top-k + align in one launch, <= %d rows) on %d runners",
                    MAX_ROWS, n)
    return n
