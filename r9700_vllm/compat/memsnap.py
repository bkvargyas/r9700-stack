"""R9K_MEMSNAP=1: where the profiling run's peak activation memory goes (one 4096-token chunk at TP4 costs
3.83 GiB a card, which vLLM subtracts from the KV cache). Forward hooks on the model, its decoder layers and
their parts record the allocation delta and the peak above the entry level; the first forward with at least
R9K_MEMSNAP_MIN tokens (default 2048) prints the table on rank 0 and dumps an allocator snapshot
(torch.cuda.memory._dump_snapshot) to R9K_MEMSNAP_PATH (default /root/.cache/vllm/memsnap-rank<r>.pickle, i.e.
the vllmstock-cache mount) for the per-allocation stack traces. Use with EAGER=1: under torch.compile the hooks
would be traced into the graph. Diagnostic only; nothing in the product reads it.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

ENABLED = os.environ.get("R9K_MEMSNAP", "0") == "1"
MIN_TOKENS = int(os.environ.get("R9K_MEMSNAP_MIN", "2048"))
_ROWS: list[tuple[str, float, float]] = []     # (name, peak MiB above entry, delta MiB)
_ST = {"done": False, "depth": 0}
_PARTS = ("self_attn", "linear_attn", "mlp", "attn_hyper_connection", "mlp_hyper_connection", "ple")


def _mib(b: int) -> float:
    return b / 2**20


def _pre(name):
    def hook(mod, args):
        mod._r9k_mem0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    return hook


def _post(name):
    def hook(mod, args, out):
        base = getattr(mod, "_r9k_mem0", 0)
        _ROWS.append((name, _mib(torch.cuda.max_memory_allocated() - base), _mib(torch.cuda.memory_allocated() - base)))
    return hook


def _top_pre(mod, args, kwargs):
    if _ST["done"]:
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    except Exception as e:                                   # noqa: BLE001
        logger.warning("r9700 memsnap: no allocator history (%s)", e)
    _ROWS.clear()
    mod._r9k_mem0 = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def _top_post(mod, args, kwargs, out):
    if _ST["done"]:
        return
    n = 0
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor) and a.dim() >= 1:
            n = max(n, a.shape[0])
    if n < MIN_TOKENS:
        return
    _ST["done"] = True
    base = getattr(mod, "_r9k_mem0", 0)
    peak, delta = _mib(torch.cuda.max_memory_allocated() - base), _mib(torch.cuda.memory_allocated() - base)
    rank = int(os.environ.get("RANK", "0") or 0)
    try:
        from vllm.distributed.parallel_state import get_tp_group
        rank = get_tp_group().rank_in_group
    except Exception:                                        # noqa: BLE001
        pass
    path = os.environ.get("R9K_MEMSNAP_PATH", f"/root/.cache/vllm/memsnap-rank{rank}.pickle")
    try:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as e:                                   # noqa: BLE001
        logger.warning("r9700 memsnap: snapshot failed (%s)", e)
    if rank != 0:
        return
    lines = [f"r9700 memsnap: forward of {n} tokens: peak +{peak:.0f} MiB above entry, +{delta:.0f} MiB retained; "
             f"allocated at entry {_mib(base):.0f} MiB; snapshot {path}"]
    # per layer: the layer's own peak and its parts
    by_layer: dict[str, list[tuple[str, float, float]]] = {}
    for name, p, d in _ROWS:
        by_layer.setdefault(name.rsplit(".", 1)[0] if "." in name else name, []).append((name, p, d))
    layers = [(name, p, d) for name, p, d in _ROWS if "." not in name]
    layers.sort(key=lambda r: -r[1])
    lines.append("  layers by peak (top 6):")
    for name, p, d in layers[:6]:
        parts = sorted((r for r in by_layer.get(name, []) if "." in r[0]), key=lambda r: -r[1])
        lines.append(f"    {name:28s} peak +{p:7.0f} MiB  retained {d:+7.0f}   parts: " +
                     ", ".join(f"{pn.rsplit('.', 1)[1]} +{pp:.0f}/{pd:+.0f}" for pn, pp, pd in parts))
    parts_all = sorted((r for r in _ROWS if "." in r[0]), key=lambda r: -r[1])
    lines.append("  parts by peak (top 8): " + ", ".join(f"{pn} +{pp:.0f}" for pn, pp, pd in parts_all[:8]))
    logger.info("\n".join(lines))


def install(model: torch.nn.Module) -> int:
    if not ENABLED:
        return 0
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Qwen4ExpDecoderLayer":
            continue
        tag = name.rsplit(".", 1)[-1] if name else "layer"
        tag = name.replace("model.", "").replace("language_model.", "")
        mod.register_forward_pre_hook(_pre(tag))
        mod.register_forward_hook(_post(tag))
        for part in _PARTS:
            sub = getattr(mod, part, None)
            if isinstance(sub, torch.nn.Module):
                sub.register_forward_pre_hook(_pre(f"{tag}.{part}"))
                sub.register_forward_hook(_post(f"{tag}.{part}"))
        n += 1
    model.register_forward_pre_hook(_top_pre, with_kwargs=True)
    model.register_forward_hook(_top_post, with_kwargs=True)
    logger.info("r9700 memsnap: hooks on %d decoder layers of %s (EAGER=1 expected)", n, type(model).__name__)
    return n
