"""Worker extension (stock `--worker-extension-cls`) for the serving memory audit: per-rank torch allocator state and an
inventory of every device tensor reachable from the model runner, the drafter and the r9700_vllm plugin modules.
Diagnostic only. The launcher (serve/ds41-exl3.sh MEMPROBE=1) mounts this file into the container as `r9k_memprobe`,
adds `--worker-extension-cls r9k_memprobe.MemProbe` and VLLM_SERVER_DEV_MODE=1 (stock dev endpoint POST /collective_rpc);
bench/ds41/memprobe.py is the client. Nothing in the product imports it.
"""
from __future__ import annotations

import sys

import torch

_MIB = 2**20


def _tensors(obj, depth=0, seen=None):
    """Yield (path, tensor) for tensors held in obj's attributes / lists / dicts (bounded depth)."""
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 3:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield "", obj
        return
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (type, torch.nn.Module)) and not callable(obj):
        items = vars(obj).items()
    else:
        return
    for k, v in items:
        if isinstance(v, (int, float, str, bytes, bool)) or v is None:
            continue
        for p, t in _tensors(v, depth + 1, seen):
            yield f".{k}{p}", t


def _category(path: str) -> str:
    p = path.lower()
    if p.startswith("plugin:"):
        mod = path.split(":", 1)[1].split("/", 1)[0]
        return f"plugin scratch {mod}"
    if p.startswith("vllm:"):
        return f"vllm global {path.split(':', 1)[1].split('/', 1)[0]}"
    drafter = p.startswith("drafter")
    pre = "drafter " if drafter else ""
    if "kv_cache" in p or ".kv_caches" in p:
        return pre + "kv cache"
    if ".experts." in p or p.endswith(".experts") or ".experts" in p and "shared" not in p:
        return pre + "routed experts"
    if "shared_expert" in p or "shared_experts" in p:
        return pre + "shared experts"
    if "engram" not in p and ("embed_tokens" in p or ".embed." in p):
        return pre + "embed"
    if "lm_head" in p or ".head." in p:
        return pre + "lm_head"
    if "engram" in p and "embed_tokens" in p:
        return pre + "engram tables (pinned HOST, not VRAM)"
    if "engram" in p:
        return pre + "engram (device part)"
    if "rotary_emb" in p or "cos_sin" in p:
        return pre + "rope cos/sin caches"
    if "attn" in p or "indexer" in p or "compressor" in p:
        return pre + "attention"
    if "hc_" in p or "hyper" in p:
        return pre + "hyper-connections"
    if "gate" in p:
        return pre + "router"
    return pre + "other"


class MemProbe:
    """Mixed into the vLLM Worker. Methods are called by name through POST /collective_rpc."""

    def r9k_reset_peak(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return {"ok": True}

    def r9k_mem(self, inventory: str = "1"):
        torch.cuda.synchronize()
        st = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        out = {
            "rank": int(getattr(self, "rank", -1)),
            "allocated_mib": st.get("allocated_bytes.all.current", 0) / _MIB,
            "allocated_peak_mib": st.get("allocated_bytes.all.peak", 0) / _MIB,
            "reserved_mib": st.get("reserved_bytes.all.current", 0) / _MIB,
            "reserved_peak_mib": st.get("reserved_bytes.all.peak", 0) / _MIB,
            "inactive_split_mib": st.get("inactive_split_bytes.all.current", 0) / _MIB,
            "num_alloc_retries": st.get("num_alloc_retries", 0),
            "num_ooms": st.get("num_ooms", 0),
            "device_free_mib": free / _MIB,
            "device_used_mib": (total - free) / _MIB,
            "device_total_mib": total / _MIB,
        }
        # reserved memory by allocator pool: (0, 0) is the default pool; the others are HIP-graph private pools
        pools: dict[str, list[float]] = {}
        try:
            for seg in torch.cuda.memory_snapshot():
                pid = str(tuple(seg.get("segment_pool_id", (0, 0))))
                a = pools.setdefault(pid, [0.0, 0.0, 0])
                a[0] += seg["total_size"] / _MIB
                a[1] += seg["allocated_size"] / _MIB
                a[2] += 1
        except Exception as e:  # noqa: BLE001
            pools["error"] = [0, 0, str(e)]
        out["pools"] = {k: {"reserved_mib": v[0], "allocated_mib": v[1], "segments": v[2]} for k, v in pools.items()}
        if inventory == "1":
            out["inventory"] = self._r9k_inventory()
        return out

    def r9k_try_alloc(self, step_mib: str = "64", max_mib: str = "4096"):
        """Raw hipMalloc (outside torch's cache) in step_mib chunks until it fails or max_mib: what the driver will really
        still hand out, vs hipMemGetInfo's free and sysfs. Everything is freed before returning."""
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so")
        torch.cuda.synchronize()
        free0, _ = torch.cuda.mem_get_info()
        ptrs, got, step, cap = [], 0, int(step_mib), int(max_mib)
        err = 0
        while got + step <= cap:
            p = ctypes.c_void_p()
            err = hip.hipMalloc(ctypes.byref(p), ctypes.c_size_t(step * _MIB))
            if err != 0:
                break
            ptrs.append(p); got += step
        free1, _ = torch.cuda.mem_get_info()
        for p in ptrs:
            hip.hipFree(p)
        hip.hipGetLastError()
        return {"rank": int(getattr(self, "rank", -1)), "allocated_mib": got, "last_err": int(err),
                "hip_free_before_mib": free0 / _MIB, "hip_free_at_max_mib": free1 / _MIB}

    def r9k_unreached(self, top: str = "30"):
        """Every live device tensor the gc can see whose storage the inventory walk did not reach, grouped by
        (shape, dtype); plus the summed size. Names where torch objects hide (closures, compiled graphs' static inputs)."""
        import gc
        dev = torch.cuda.current_device()
        inv_keys = set(self._r9k_inventory(keys_only=True))
        groups: dict[str, list[float]] = {}
        seen = set()
        for o in gc.get_objects():
            try:
                if not isinstance(o, torch.Tensor) or o.device.type != "cuda" or o.device.index != dev:
                    continue
                s = o.untyped_storage(); k = s.data_ptr()
            except Exception:  # noqa: BLE001
                continue
            if k in inv_keys or k in seen or s.nbytes() == 0:
                continue
            seen.add(k)
            g = groups.setdefault(f"{tuple(o.shape)} {o.dtype}", [0.0, 0])
            g[0] += s.nbytes() / _MIB; g[1] += 1
        rows = sorted(groups.items(), key=lambda kv: -kv[1][0])
        return {"rank": int(getattr(self, "rank", -1)), "unreached_mib": sum(v[0] for v in groups.values()),
                "count": sum(v[1] for v in groups.values()),
                "top": [[k, round(v[0], 1), v[1]] for k, v in rows[:int(top)]]}

    def _r9k_inventory(self, keys_only=False):
        dev = torch.cuda.current_device()
        seen: dict[int, tuple[str, int]] = {}

        def add(path, t):
            if not isinstance(t, torch.Tensor) or t.device.type != "cuda" or t.device.index != dev:
                return
            try:
                s = t.untyped_storage()
                key, nb = s.data_ptr(), s.nbytes()
            except Exception:  # noqa: BLE001
                return
            if nb == 0 or key in seen:
                return
            seen[key] = (path, nb)

        mr = self.model_runner
        roots = [("model", getattr(mr, "model", None))]
        spec = getattr(mr, "speculator", None) or getattr(mr, "drafter", None)
        if spec is not None:
            roots.append(("drafter", getattr(spec, "model", None)))
        for rname, root in roots:
            if not isinstance(root, torch.nn.Module):
                continue
            for mname, mod in root.named_modules():
                base = f"{rname}.{mname}" if mname else rname
                for k, v in vars(mod).items():
                    if k in ("_modules",):
                        continue
                    for p, t in _tensors(v):
                        add(f"{base}.{k}{p}", t)
        # model-runner-level state (KV caches, input buffers, the drafter's own buffers)
        for k, v in vars(mr).items():
            if isinstance(v, torch.nn.Module):
                continue
            for p, t in _tensors(v):
                add(f"runner.{k}{p}", t)
        if spec is not None:
            for k, v in vars(spec).items():
                if isinstance(v, torch.nn.Module):
                    continue
                for p, t in _tensors(v):
                    add(f"drafter.runner.{k}{p}", t)
        # module globals: the plugin's scratch caches and vLLM's workspace managers
        for name, m in list(sys.modules.items()):
            if m is None:
                continue
            if name.startswith("r9700_vllm"):
                tag = "plugin:" + name.replace("r9700_vllm.", "")
            elif name.startswith("vllm.") and ("workspace" in name or "mla" in name or "indexer" in name
                                                or "deepseek" in name or "sparse" in name or "dspark" in name):
                tag = "vllm:" + name.replace("vllm.", "")
            else:
                continue
            for k, v in list(vars(m).items()):
                if k.startswith("__") or isinstance(v, type(sys)) or isinstance(v, type):
                    continue
                for p, t in _tensors(v):
                    add(f"{tag}/{k}{p}", t)
        if keys_only:
            return list(seen.keys())
        cats: dict[str, float] = {}
        big = []
        for path, nb in seen.values():
            c = _category(path) if not path.startswith("runner.") else (
                "kv cache" if "kv_cache" in path else "runner buffers")
            if path.startswith("drafter.runner."):
                c = "drafter runner buffers"
            cats[c] = cats.get(c, 0.0) + nb / _MIB
            big.append((nb / _MIB, path))
        big.sort(reverse=True)
        big = [b for b in big if "_trellis" not in b[1] and "engram.embed_tokens" not in b[1]]
        return {"categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
                "total_mib": sum(v for k, v in cats.items() if "HOST" not in k),
                "largest": [[round(m, 1), p] for m, p in big[:60]]}
