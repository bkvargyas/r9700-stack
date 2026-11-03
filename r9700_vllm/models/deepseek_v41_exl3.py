"""DeepSeek-V4.1 from an EXL3 pack: the loader glue over stock vLLM's ROCm DeepseekV41 model classes.

Thin subclasses registered through ``ModelRegistry.register_model`` (the mechanism in models/qwen4_exp.py) for the target
(``DeepseekV41ForCausalLM`` -> stock amd/vl_model.py, wrapping amd/model.py) and the DSpark drafter (``DSparkV41DraftModel``
-> stock amd/dspark.py). They behave exactly like stock unless ``vllm_config.quant_config`` is the ``exl3`` config; fp8
serving of the official checkpoint is untouched. The classes are built on first attribute access (PEP 562), so registering
ARCHS imports nothing from vllm.models.deepseek_v41 in processes that never serve it. Layer map: notes/exl3-layer-map.md.

With exl3:
  * construction: ``attn.wo_a`` is built unquantized (bf16_linears): stock ROCm reads its weight directly (_o_proj ->
    rocm_inv_rope_einsum caches a bf16 copy). Everything else that asks the quant config gets quant/exl3.py /
    quant/exl3_moe.py (the shared experts go dense there: 2304 / 8 = 288 is not a 128-multiple). The attention layers lose
    their aux streams (serialize_attention): exllamav3's RDNA GEMV is not stream-safe.
  * load_weights: the pack's names are the HF names with each quantized ``.weight`` replaced by ``.trellis/.suh/.svh/.mul1``.
    For a matrix whose vLLM module runs the EXL3 method they pass through the stock mapper, stacked mapping (fused_wqa_wkv
    <- wq_a 0 / wkv 1) and expert mapping (``experts.{e}.w1|w3|w2.<t>`` -> ``experts.routed_experts.w13_<t>|w2_<t>``)
    untouched. A matrix whose module has a plain bf16 weight -- lm_head, compressor.fused_wkv_wgate and indexer.wk (stock
    builds them without a quant config), wo_a -- is collected, dequantized (quant/exl3.py dequant_f32) and handed to the
    module's stock weight_loader, which does vLLM's own TP slicing (wo_a: the eight o-group slices ``wo_a.slice.{g}``,
    rows g*1024 of the base's [8192, 4096], assembled in order first). From the base checkpoint's shards 47/48 only the
    Engram tables load (``engram.embed.weight|scale``, host memory under ``--engram-config '{"cpu_offload": true}'``); their
    bf16 k/q_weight and fp8 wkv duplicates are dropped.

``python3 -m r9700_vllm.models.deepseek_v41_exl3 pack-tensors.json base-tensors.json [expected-params.json]`` is the name
dry run (gate 6c): every pack tensor and base Engram tensor to the parameter it lands in, or an explicit skip.
"""
from __future__ import annotations

import contextlib
import re
from collections.abc import Iterable, Iterator

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.r9700_vllm.deepseek_v41_exl3")

_MOD = "r9700_vllm.models.deepseek_v41_exl3"
ARCHS: dict[str, str] = {
    "DeepseekV41ForCausalLM": f"{_MOD}:DeepseekV41ForCausalLMExl3",
    "DSparkV41DraftModel": f"{_MOD}:DSparkDeepseekV41Exl3",
}

# one EXL3 matrix part (HF name); routed experts (``experts.{e}.w1|w2|w3``) always go to quant/exl3_moe.py via stock
_QPART = re.compile(r"^(?P<base>.+)\.(?P<part>trellis|suh|svh|mul1|mcg)$")
_EXPERT = re.compile(r"\.experts\.\d+\.w[123]$")
# base-checkpoint copies of tensors the pack also carries (shards 47/48 hold the whole Engram module)
_BASE_ENGRAM_WKV = re.compile(r"^layers\.\d+\.engram\.wkv\.(weight|scale)$")
_ENGRAM_KQ = re.compile(r"^layers\.\d+\.engram\.[kq]_weight$")
# HF sub-module -> (vLLM sub-module, stacked shard): stock's stacked mapping and mapper renames (amd/model.py 1014-1022,
# 1265; amd/dspark.py 432-437)
_RENAME = {"attn.wq_a": ("attn.fused_wqa_wkv", 0), "attn.wkv": ("attn.fused_wqa_wkv", 1),
           "attn.compressor.wkv": ("attn.compressor.fused_wkv_wgate", 0),
           "attn.compressor.wgate": ("attn.compressor.fused_wkv_wgate", 1),
           "ffn.shared_experts.w1": ("ffn.shared_experts.gate_up_proj", 0),
           "ffn.shared_experts.w3": ("ffn.shared_experts.gate_up_proj", 1),
           "ffn.shared_experts.w2": ("ffn.shared_experts.down_proj", None)}
_SLICE = re.compile(r"^attn\.wo_a\.slice\.(\d+)$")
# vLLM-side module prefixes built unquantized under exl3: stock ROCm reads wo_a's weight directly (_o_proj ->
# rocm_inv_rope_einsum caches a bf16 copy), so it must be a plain bf16 linear until _o_proj is routed through the method
BF16_LINEARS = re.compile(r"\.attn\.wo_a$")
_COLS = 16384  # dequant chunk (output columns, a 128-multiple): bounds the f32 temporaries on the GPU (the head is 129280)


def is_exl3(quant_config) -> bool:
    return quant_config is not None and getattr(quant_config, "get_name", lambda: None)() == "exl3"


def _bf16(dtype) -> bool:
    return dtype in (torch.bfloat16, "BF16", "bfloat16")


def vllm_target(base: str, draft: bool) -> tuple[str, int | tuple[str, int] | None] | None:
    """HF matrix name -> (vLLM module path under the model root, shard): root = the VL model for the target
    (``language_model.…``), the DSpark model for the drafter (``model.…``). Shard: stacked index, ("slice", g) for wo_a's
    o-group g, or None."""
    if base == "head" and not draft:
        return "language_model.lm_head", None
    m = re.match(r"^(layers|mtp)\.(\d+)\.(.+)$", base)
    if m is None or (m.group(1) == "mtp") != draft:
        return None
    rest = m.group(3)
    if draft and rest.startswith("main_proj"):
        return "model." + rest, None
    root = f"language_model.model.layers.{m.group(2)}." if not draft else f"model.layers.{m.group(2)}."
    s = _SLICE.match(rest)
    if s:
        return root + "attn.wo_a", ("slice", int(s.group(1)))
    sub, shard = _RENAME.get(rest, (rest, None))
    return root + sub, shard


def classify(name: str, dtype, draft: bool, exl3_module) -> tuple[str, object]:
    """What the exl3 loader does with one checkpoint tensor (HF name). ``exl3_module(path)`` says whether the vLLM module
    runs the EXL3 method (True), something else -- a plain bf16 ``weight`` (False) -- or does not exist (None).
    ("pass", None): on to the stock loader unchanged; ("slice", (path, g)): wo_a o-group g into an EXL3 wo_a;
    ("dequant", base): collected, dequantized to bf16 and handed to the module's stock weight_loader; ("skip", why)."""
    if draft and not name.startswith("mtp."):
        return "skip", "target weight (the drafter loads mtp.* only)"
    if not draft and name.startswith("mtp."):
        return "skip", "DSpark drafter weight (stock target mapper drops mtp.*)"
    if _BASE_ENGRAM_WKV.match(name):
        return "skip", "base fp8 engram.wkv (the pack's EXL3 engram.wkv loads)"
    if _ENGRAM_KQ.match(name) and _bf16(dtype):
        return "skip", "base bf16 engram k/q_weight (the pack's f32 copy loads)"
    m = _QPART.match(name)
    if m is None or _EXPERT.search(m.group("base")):
        return "pass", None
    tgt = vllm_target(m.group("base"), draft)
    if tgt is None:
        return "pass", None
    path, shard = tgt
    q = exl3_module(path)
    if q is None:
        return "pass", None                      # let the stock loader name the problem
    if not q:
        return "dequant", m.group("base")
    if isinstance(shard, tuple):
        return "slice", (path, shard[1])
    return "pass", None


def dequant_bf16(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """W [out, in] bf16 from one matrix's trellis/suh/svh/marker through quant/exl3.py dequant_f32 (W^T f32 [in, out]),
    _COLS output columns at a time; one chunk stays on the GPU, a larger matrix is assembled in host memory."""
    from ..quant.exl3 import dequant_f32, k_of
    dev = torch.device("cuda", torch.cuda.current_device())
    tr, suh, svh = parts["trellis"], parts["suh"].to(dev), parts["svh"]
    K, n, k = k_of(tr), svh.shape[0], suh.shape[0]
    mul1, mcg = "mul1" in parts, "mcg" in parts
    out = None if n <= _COLS else torch.empty(n, k, dtype=torch.bfloat16, pin_memory=True)
    for c0 in range(0, n, _COLS):
        c1 = min(c0 + _COLS, n)
        w = dequant_f32(tr[:, c0 // 16:c1 // 16].contiguous().to(dev), suh, svh[c0:c1].contiguous().to(dev), K, mul1, mcg)
        w = w.t().to(torch.bfloat16)
        if out is None:
            return w.contiguous()
        out[c0:c1].copy_(w)
        del w
    return out


@contextlib.contextmanager
def bf16_linears(quant_config):
    """Construction scope: BF16_LINEARS ask the exl3 config and get UnquantizedLinearMethod. The bound method is shadowed
    on the instance for the duration of the model's __init__ (get_quant_method is only asked at construction)."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    had = "get_quant_method" in vars(quant_config)
    orig = quant_config.get_quant_method

    def get_quant_method(layer, prefix):
        if isinstance(layer, LinearBase) and BF16_LINEARS.search(prefix):
            return UnquantizedLinearMethod()
        return orig(layer, prefix)

    quant_config.get_quant_method = get_quant_method
    try:
        yield
    finally:
        if had:
            quant_config.get_quant_method = orig
        else:
            del quant_config.get_quant_method


def serialize_attention(model: torch.nn.Module) -> int:
    """No aux streams on the attention layers: exllamav3's RDNA GEMV passes its pointers through one per-device
    parameter block, so two EXL3 GEMMs in flight on two streams corrupt each other. With ``aux_stream_list = None`` the
    ROCm V4.1 attention takes its sequential fallback (amd/rocm.py forward: enable_multi_stream_overlap(None) is False;
    _run_parallel_input_projections / _prepare_and_attn hand execute_in_parallel no streams)."""
    n = 0
    for m in model.modules():
        if getattr(m, "aux_stream_list", None) is not None and hasattr(m, "fused_wqa_wkv"):
            m.aux_stream_list = None
            n += 1
    return n


class _Rewriter:
    """The exl3 checkpoint stream for one model root (the VL target or the DSpark drafter)."""

    def __init__(self, root: torch.nn.Module, draft: bool):
        self.root, self.draft = root, draft
        self.pending: dict[str, dict[str, torch.Tensor]] = {}
        self.slices: dict[str, dict[int, torch.Tensor]] = {}
        self.seen: set[str] = set()
        self.direct: set[str] = set()  # parameter names loaded here, relative to root
        self.skipped: dict[str, int] = {}
        self._q: dict[str, bool | None] = {}

    def exl3_module(self, path: str) -> bool | None:
        if path not in self._q:
            from ..quant.exl3 import Exl3LinearMethod, Exl3LMHeadMethod
            try:
                mod = self.root.get_submodule(path)
            except AttributeError:
                self._q[path] = None
            else:
                self._q[path] = isinstance(getattr(mod, "quant_method", None), (Exl3LinearMethod, Exl3LMHeadMethod))
        return self._q[path]

    def __call__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for name, w in weights:
            action, arg = classify(name, w.dtype, self.draft, self.exl3_module)
            if action == "skip":
                self.skipped[arg] = self.skipped.get(arg, 0) + 1
            elif action == "slice":
                path, g = arg
                part = name.rsplit(".", 1)[1]
                param = getattr(self.root.get_submodule(path), part)
                param.weight_loader(param, w, g)
                self.direct.add(f"{path}.{part}")
            elif action == "dequant":
                parts = self.pending.setdefault(arg, {})
                parts[name.rsplit(".", 1)[1]] = w
                if {"trellis", "suh", "svh"} <= parts.keys() and ("mul1" in parts or "mcg" in parts):
                    self.finish(arg, self.pending.pop(arg))
            elif name in self.seen:
                logger.warning_once("r9700 exl3: duplicate checkpoint tensor %s ignored (composed model dir?)", name)
            else:
                self.seen.add(name)
                yield name, w
        if self.pending or self.slices:
            raise RuntimeError("r9700 exl3: incomplete matrices in the checkpoint: "
                               f"{sorted(self.pending)[:8]} {sorted(self.slices)[:8]}")
        for why, n in self.skipped.items():
            logger.info("r9700 exl3: skipped %d tensors: %s", n, why)

    def finish(self, base: str, parts: dict[str, torch.Tensor]) -> None:
        """bf16 W [out, in] to the module's stock weight_loader, which applies vLLM's own TP slicing (ParallelLMHead:
        the vocab shard; stacked layers: shard_id; wo_a: the eight o-group slices assembled to [8192, 4096] first)."""
        path, shard = vllm_target(base, self.draft)
        w = dequant_bf16(parts)
        if isinstance(shard, tuple):
            got = self.slices.setdefault(path, {})
            got[shard[1]] = w
            n = self.root.get_submodule(path.rsplit(".", 1)[0]).n_groups
            if len(got) < n:
                return
            del self.slices[path]
            w, shard = torch.cat([got[g].to(w.device) for g in range(n)]), None
        param = self.root.get_submodule(path).weight
        if shard is None:
            param.weight_loader(param, w)
        else:
            param.weight_loader(param, w, shard)
        self.direct.add(f"{path}.weight")


def trim_rope_caches(model: torch.nn.Module, max_model_len: int, margin: int = 128) -> int:
    """Stock builds both rotary embeddings' ``cos_sin_cache`` for 1,048,576 positions ([1M, 64] f32 = 256 MiB each, two
    objects shared by the layers): slice them to the positions this server can see. Row p is the same whatever the cache
    length, so every position below max_model_len + margin is bit-identical; the launcher's MAXLEN decides. Returns MiB freed
    (package S's audit, 2026-11-03: 448 MiB a rank at 131072, 496 at 32768)."""
    rows = max_model_len + margin
    seen: set[int] = set()
    freed = 0
    for m in model.modules():
        rope = getattr(m, "rotary_emb", None)
        cache = getattr(rope, "cos_sin_cache", None) if rope is not None else None
        if cache is None or id(rope) in seen or cache.shape[0] <= rows:
            continue
        seen.add(id(rope))
        freed += (cache.shape[0] - rows) * cache.shape[1] * cache.element_size() >> 20
        rope.cos_sin_cache = cache[:rows].contiguous()
    if freed:
        torch.cuda.empty_cache()
        logger.info_once("r9700 exl3: RoPE cos/sin caches trimmed to %d positions (%d MiB freed a rank)", rows, freed)
    return freed


def _build():
    from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM
    from vllm.models.deepseek_v41.amd.vl_model import DeepseekV41ForCausalLM

    class DeepseekV41ForCausalLMExl3(DeepseekV41ForCausalLM):
        """Stock ROCm DeepSeek-V4.1 (VL wrapper); EXL3 packs load through _Rewriter."""

        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            self._r9k_exl3 = is_exl3(vllm_config.quant_config)
            self._r9k_max_len = int(vllm_config.model_config.max_model_len)
            if not self._r9k_exl3:
                super().__init__(vllm_config=vllm_config, prefix=prefix)
                return
            with bf16_linears(vllm_config.quant_config):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            n = serialize_attention(self)
            logger.info_once("r9700 exl3: DeepSeek-V4.1 with EXL3 weights; wo_a, head, compressor, indexer.wk "
                             "dequantized to bf16 at load; %d attention layers on one stream", n)

        def load_weights(self, weights):
            if not self._r9k_exl3:
                return super().load_weights(weights)
            rw = _Rewriter(self, draft=False)
            loaded = set(super().load_weights(rw(weights))) | rw.direct
            trim_rope_caches(self, self._r9k_max_len)
            return loaded

    class DSparkDeepseekV41Exl3(DSparkDeepseekV4ForCausalLM):
        """Stock ROCm DSpark drafter (``mtp.{0,1,2}.*``); EXL3 packs load through _Rewriter (embed / lm_head alias the
        target's)."""

        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            self._r9k_exl3 = is_exl3(vllm_config.quant_config)
            self._r9k_max_len = int(vllm_config.model_config.max_model_len)
            if not self._r9k_exl3:
                super().__init__(vllm_config=vllm_config, prefix=prefix)
                return
            with bf16_linears(vllm_config.quant_config):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            serialize_attention(self)

        def load_weights(self, weights):
            if not self._r9k_exl3:
                return super().load_weights(weights)
            rw = _Rewriter(self, draft=True)
            loaded = set(super().load_weights(rw(weights))) | rw.direct
            trim_rope_caches(self, self._r9k_max_len)
            return loaded

    out = {}
    for c in (DeepseekV41ForCausalLMExl3, DSparkDeepseekV41Exl3):
        c.__module__, c.__qualname__ = _MOD, c.__name__  # picklable by reference (module __getattr__ below)
        out[c.__name__] = c
    return out


_CLASSES: dict[str, type] = {}


def __getattr__(name: str):
    if name in ("DeepseekV41ForCausalLMExl3", "DSparkDeepseekV41Exl3"):
        if not _CLASSES:
            _CLASSES.update(_build())
        return _CLASSES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---- name dry run (gate 6c) -------------------------------------------------------------------------------------------
# Pure name rewriting through the stock rules: classify() above, then the VL mapper (amd/vl_model.py
# _make_deepseek_v4_vl_weights_mapper), then DeepseekV4Model.load_weights' stacked and expert mappings (amd/model.py
# 1014-1022 / 1165-1174; the drafter's amd/dspark.py 418-437 + _remap_dspark_name). Needs vLLM importable (the image).

_STACKED = [("gate_up_proj", "w1", 0), ("gate_up_proj", "w3", 1), ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1), ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1)]


def _expert_map(n: int) -> list[tuple[str, str, int, str]]:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    return RoutedExperts.build_expert_params_mapping("w1", "w2", "w3", n, 0, "routed_experts")


def _target_param(name: str, mapper, emap) -> tuple[str, str]:
    """HF name (already through classify) -> (parameter name under the VL model, how)."""
    n = mapper.map_name(name)
    if n is None:
        return "", "skip: dropped by the stock VL mapper"
    if not n.startswith("language_model.model."):
        return n, "direct (VL wrapper / lm_head)"
    if ".experts." in n:
        for param, weight, eid, shard in emap:
            if weight in n:
                return n.replace(weight, param), f"expert {eid} {shard}"
        return "", "UNMAPPED expert name"
    for param, weight, shard in _STACKED:
        if weight in n:
            return n.replace(weight, param), f"stacked shard {shard}"
    return n, "direct"


def _draft_param(name: str, emap) -> tuple[str, str]:
    m = re.match(r"mtp\.(\d+)\.(.*)", name)
    stage, rest = int(m.group(1)), m.group(2)
    if rest.startswith("markov_head.embed."):
        n = "model.markov_head.markov_w1." + rest[len("markov_head.embed."):]
    elif rest.startswith("markov_head.head."):
        n = "model.markov_head.markov_w2." + rest[len("markov_head.head."):]
    elif rest.startswith(("main_proj.", "main_norm.", "norm.", "markov_head.", "confidence_head.")):
        n = f"model.{rest}"
    else:
        n = f"model.layers.{stage}.{rest}"
    n = n.replace(".shared_experts.w2", ".shared_experts.down_proj")
    if ".experts." in n:
        for param, weight, eid, shard in emap:
            if weight in n:
                return n.replace(weight, param), f"expert {eid} {shard}"
        return "", "UNMAPPED expert name"
    if n.startswith("model.layers."):
        for param, weight, shard in _STACKED[:4]:
            if weight in n:
                return n.replace(weight, param), f"stacked shard {shard}"
    if n.endswith(".ffn.gate.bias"):
        n = n.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
    return n, "direct"


def dry_run(pack_json: str, base_json: str, expected_json: str) -> dict:
    """Every pack tensor + the base shards 47/48 -> parameter or skip, for the target and the drafter. ``expected_json``:
    {"target": [...], "draft": [...]} parameter names of the models as built under exl3, and "target_exl3" /
    "draft_exl3": the module paths that run the EXL3 method (what exl3_module() answers at runtime)."""
    import json
    from collections import Counter
    from vllm.models.deepseek_v41.amd.vl_model import _make_deepseek_v4_vl_weights_mapper
    pack = json.load(open(pack_json))["files"]
    base = json.load(open(base_json))["files"]
    exp_all = json.load(open(expected_json))
    stream = [(n, t["dtype"], "pack") for f in sorted(pack) for n, t in pack[f]["tensors"].items()]
    stream += [(n, t["dtype"], "base") for f in sorted(base) if f.startswith(("model-00047-", "model-00048-"))
               for n, t in base[f]["tensors"].items()]
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", "weight_scale_inv")
    emap, dmap = _expert_map(384), _expert_map(128)
    out, report = {"target": {}, "draft": {}}, {}
    for key, side in out.items():
        draft = key == "draft"
        params = set(exp_all[key])
        q = set(exp_all[key + "_exl3"])
        modules = {n.rsplit(".", 1)[0] for n in params}

        def exl3_module(path):
            return True if path in q else (False if path in modules else None)

        for name, dtype, origin in stream:
            k = f"{name} ({origin})"
            action, arg = classify(name, dtype, draft, exl3_module)
            if action == "skip":
                side[k] = ("", "skip: " + arg)
            elif not draft and name.startswith(("vision.", "aligner.")):
                # stock: StageMissingLayer under --language-model-only (weights skipped), else the VL tower loads them
                side[k] = ("", "skip: vision tower, stubbed under --language-model-only")
            elif action == "slice":
                side[k] = (f"{arg[0]}.{name.rsplit('.', 1)[1]}", f"exl3 wo_a slice {arg[1]}")
            elif action == "dequant":
                path, shard = vllm_target(arg, draft)
                how = "dequant -> stock loader" + ("" if shard is None else
                                                   f" (o-groups assembled)" if isinstance(shard, tuple) else
                                                   f" shard {shard}")
                side[k] = (f"{path}.weight", how)
            else:
                side[k] = _draft_param(name, dmap) if draft else _target_param(name, mapper, emap)
        loaded = {p for p, _ in side.values() if p}
        how = Counter((h.split(":")[0] if h.startswith("skip") else " ".join(h.split(" ")[:3]).split(" shard")[0]
                       .split(" (")[0].replace(" expert", "") if h.startswith("dequant") else h.split(" ")[0])
                      for _, h in side.values())
        unmapped = sorted(n for n, (p, h) in side.items() if not p and not h.startswith("skip"))
        report[key] = {"tensors": len(side), "params_reached": len(loaded), "by_action": dict(how),
                       "unmapped": unmapped[:50], "n_unmapped": len(unmapped),
                       "not_in_model": sorted(loaded - params)[:50], "n_not_in_model": len(loaded - params),
                       "never_loaded": sorted(params - loaded)[:80], "n_never_loaded": len(params - loaded)}
    report["_detail"] = out
    return report


if __name__ == "__main__":
    import json
    import sys
    rep = dry_run(*sys.argv[1:4])
    detail = rep.pop("_detail")
    print(json.dumps(rep, indent=1))
    if len(sys.argv) > 4:
        json.dump(detail, open(sys.argv[4], "w"))
