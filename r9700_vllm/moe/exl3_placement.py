"""Load-time expert placement for the EXL3 routed experts under expert parallelism (package H3, notes/exl3-ep-skew.md).

Stock vLLM places experts linearly (rank r owns global experts [48 r, 48 r + 48) of DeepSeek-V4.1-Flash's 384; the
drafter's 128 as 16 a rank). The routing is not uniform, so some ranks carry more distinct picked experts a step than
others and every other rank waits for them at the post-MoE all-reduce. R9K_EXL3_EP_MAP=<file.json> (written by
bench/ds41/epskew/skew_place.py from a recorded routing histogram) gives, per MoE layer, the EP rank that owns each
global expert; this module installs it before the layer's weights are created, as the layer's own expert map
(RoutedExperts._expert_map: global id -> local slot, -1 = elsewhere), so

  * quant/exl3_moe.py's weight loader stores exactly the experts this rank owns (it maps every checkpoint expert
    through the layer's expert map; the pack's per-expert tensors are not touched -- a load-time transform),
  * its process_weights_after_loading derives the tier's global -> local table (exl3_g2l) from the same map,

and nothing else changes: the same experts compute the same slots with the same weights on whichever rank holds them.
(The per-rank partial sums group the picks differently, so the bf16 rounding of y before the all-reduce can differ
from the linear placement in the last bit, as any change of EP layout would.)

Map format: {"version": 1, "ranks": 8, "layers": {"<model layer index>": {"G": 384, "owner": [rank of expert 0, ...]}}}.
The target's layers are 0..39, the drafter's 40..42 (DSpark names them layers.{num_hidden_layers + i}). Layers absent
from the file keep stock's linear placement. A map that does not partition G into equal shares for this EP size is
refused (logged, the layer stays linear). install() is called from r9700_vllm.register() when the knob is set.
"""
from __future__ import annotations

import json
import os
import re

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.r9700_vllm.exl3_placement")

PATH = os.environ.get("R9K_EXL3_EP_MAP", "")
_MAP: dict | None = None
_DONE: list = []


def _load() -> dict:
    global _MAP
    if _MAP is None:
        with open(PATH) as f:
            d = json.load(f)
        _MAP = {int(k): v for k, v in d.get("layers", {}).items()}
    return _MAP


def local_map(layer_name: str, G: int, ep_rank: int, ep_size: int) -> torch.Tensor | None:
    """The expert map (int32 [G], local slot or -1) this rank should use for the layer, or None to keep stock's."""
    m = re.search(r"layers\.(\d+)\.", layer_name + ".")
    if not m:
        return None
    ent = _load().get(int(m.group(1)))
    if ent is None or int(ent.get("G", -1)) != G:
        return None
    owner = [int(x) for x in ent["owner"]]
    share = G // ep_size
    if len(owner) != G or G % ep_size or any(not 0 <= o < ep_size for o in owner) or \
            any(owner.count(r) != share for r in range(ep_size)):
        logger.warning("exl3 placement: map for %s is not a partition of %d experts into %d x %d; linear kept",
                       layer_name, G, ep_size, share)
        return None
    owned = [g for g in range(G) if owner[g] == ep_rank]
    emap = torch.full((G,), -1, dtype=torch.int32)
    emap[torch.tensor(owned)] = torch.arange(len(owned), dtype=torch.int32)
    return emap


def install() -> bool:
    """Wrap Exl3MoEMethod.create_weights to swap in the mapped expert map first. False when the knob is unset."""
    if not PATH:
        return False
    from ..quant import exl3_moe as M
    cls = M.Exl3MoEMethod
    if getattr(cls, "_r9k_ep_map", False):
        return True
    _load()
    orig = cls.create_weights

    def create_weights(self, layer, num_experts, *a, **k):
        pc = self.moe.moe_parallel_config
        old = getattr(layer, "_expert_map", None)
        if getattr(pc, "use_ep", False) and old is not None:
            emap = local_map(getattr(layer, "layer_name", ""), int(self.moe.num_experts), int(pc.ep_rank), int(pc.ep_size))
            if emap is not None:
                assert int((emap >= 0).sum()) == num_experts, (layer.layer_name, num_experts)
                emap = emap.to(old.device)
                layer.register_buffer("_expert_map", emap, persistent=False)
                mgr = getattr(layer, "expert_map_manager", None)
                if mgr is not None and hasattr(mgr, "_expert_map"):
                    mgr._expert_map = emap
                _DONE.append(layer.layer_name)
                if len(_DONE) == 1 or int(pc.ep_rank) == 0:
                    logger.info_once("exl3 placement: R9K_EXL3_EP_MAP=%s installed (rank %d of %d)", PATH,
                                     int(pc.ep_rank), int(pc.ep_size))
        return orig(self, layer, num_experts, *a, **k)

    cls.create_weights = create_weights
    cls._r9k_ep_map = True
    return True


def placed_layers() -> list:
    return list(_DONE)
