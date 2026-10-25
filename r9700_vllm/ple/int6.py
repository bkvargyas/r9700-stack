"""Qwen4Exp (Flash-Next) PLE n-gram table in fused int6 rows, held in pinned host memory (UVA), on stock vLLM.

Stock vLLM's AMD PLE path wants a bf16 table ([160M rows/rank, 160] = ~51 GB/rank at TP2): it cannot load the
int6 GPTQ checkpoint (``ngram_embedding.shard_N.weight_packed`` u8 [rows, 120] + ``.weight_scale`` f16
[rows, 5]) and could not hold the table in VRAM anyway. This hook:

  * ``R9kInt6PLEEmbedding`` (a ``PLEVocabParallelEmbedding`` subclass, installed by models/qwen4_exp.py only
    while the model is constructed) builds its vocab-parallel metadata on the meta device, then allocates its TP
    slice of the table as fused int6 rows (130 B/row) in pinned host memory behind a UVA view;
  * its quant method's ``embedding(layer, ids)`` is libr9k's gather+dequant kernel (stock masking, zero-fill and
    TP all-reduce in ``VocabParallelEmbedding.forward`` are untouched);
  * it has its own ``load_weights``: stock ``Qwen4ExpNGramEmbedding.load_weights`` hands the (non-``.weight``)
    ``shard_N.weight_packed / weight_scale`` tensors to ``AutoWeightsLoader``, which recurses into the child's
    ``load_weights`` -- they land in the fused host rows (TP overlap computed here; stock's helper moved twice in 2026-10).

Only engages when the checkpoint's PLE shards are int6 (fp16 scales); otherwise stock behaviour.
~20.8 GB of pinned host RAM per rank at TP2 for Qwen3.8-Flash-Next.
"""
from __future__ import annotations

import functools
import json
import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_SHARD_RE = re.compile(r"^shard_(\d+)\.weight_(packed|scale)$")


def checkpoint_ple_format() -> str | None:
    """'int6' if the served checkpoint's PLE shards carry fp16 scales, else None (stock path)."""
    forced = os.environ.get("R9K_PLE_FORMAT")
    if forced:
        return None if forced == "stock" else forced
    try:
        from vllm.config import get_current_vllm_config
        model = get_current_vllm_config().model_config.model
        idx = os.path.join(model, "model.safetensors.index.json")
        if not os.path.isfile(idx):
            return None
        wm = json.load(open(idx))["weight_map"]
        key = next((k for k in wm if k.endswith("ngram_embedding.shard_0.weight_scale")), None)
        if key is None:
            return None
        import struct
        with open(os.path.join(model, wm[key]), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return "int6" if hdr[key]["dtype"] == "F16" else None
    except Exception as e:
        logger.warning("r9700: PLE format probe failed (%s); leaving PLE stock", e)
        return None


class _Int6EmbeddingMethod:
    """Quant method stand-in: VocabParallelEmbedding.forward only calls .embedding(); the 2026-10 vLLM PLE
    embedding also delegates .dequantize() to it (our gather already returns bf16)."""

    def __init__(self, head_dim: int):
        self.head_dim = head_dim

    def create_weights(self, *a, **k):  # never called (weights are made by the embedding itself)
        raise RuntimeError("unused")

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        raise RuntimeError("unused")

    def embedding(self, layer, ids: torch.Tensor) -> torch.Tensor:
        from ..kernels.ple import gather_int6
        return gather_int6(layer.weight, ids, self.head_dim)

    def dequantize(self, layer, embeddings: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        return embeddings if embeddings.dtype == output_dtype else embeddings.to(output_dtype)


@functools.lru_cache(None)
def make_int6_embedding_cls(base):
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # noqa: F401
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    from ..kernels.ple import int6_row_bytes

    # Two vLLM generations: up to 2026-09 the PLE table was a plain PLEVocabParallelEmbedding (weights made by
    # VocabParallelEmbedding.__init__: build on meta, replace); from 2026-10 it is a Qwen4ExpPLE*Embedding whose
    # embedding method asks the layer for its storage (allocate_embedding_weight) -- the hook we need.
    new_api = hasattr(base, "allocate_embedding_weight")

    class R9kInt6PLEEmbedding(base):
        supports_prefetch = False                    # the plain device lookup path; our gather is the lookup

        def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
            if new_api:
                super().__init__(num_embeddings, embedding_dim, *args, **kwargs)   # -> allocate_embedding_weight
            else:
                with torch.device("meta"):
                    super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
                self._alloc_int6(self.num_embeddings_per_partition, embedding_dim)
                del self.weight
                self.weight = torch.nn.Parameter(self._r9k_view, requires_grad=False)
            self.weight._vllm_is_uva_offloaded = True
            try:
                from vllm.config import get_current_vllm_config
                cfg = get_current_vllm_config().model_config.hf_text_config
                self.split_parts = int(getattr(cfg, "split_ngram_parts", 512))
            except Exception:
                self.split_parts = 512
            self.quant_method = _Int6EmbeddingMethod(embedding_dim)
            if new_api:
                self.embedding_method = self.quant_method
            self.params_dtype = torch.bfloat16
            logger.info_once("r9700: PLE int6 table %d rows x %d B = %.1f GiB pinned host (UVA) per rank",
                             self._r9k_host.shape[0], self.row_bytes, self._r9k_host.numel() / 2**30)

        def _alloc_int6(self, rows: int, embedding_dim: int) -> torch.Tensor:
            from ..utils.hostmem import pinned_empty
            self.head_dim = embedding_dim
            self.row_bytes = int6_row_bytes(embedding_dim)
            host = pinned_empty((rows, self.row_bytes), torch.uint8)   # exact size (torch pinning rounds to 2^k)
            assert host.is_pinned(), "r9700: PLE host table not pinned; the UVA view would be a copy"
            self._r9k_host = host
            self._r9k_view = get_accelerator_view_from_cpu_tensor(host)
            return self._r9k_view

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            """2026-10 vLLM: the storage behind `weight` -- the int6 rows in pinned host memory, UVA view."""
            return self._alloc_int6(num_embeddings, embedding_dim)

        def start_prefetch(self, hidden_states, ngram_ids):
            return None

        def load_int6_shard(self, shard_index: int, kind: str, tensor: torch.Tensor, shard_size: int) -> int:
            # the overlap of checkpoint shard [start, start+rows) with this rank's vocabulary range; stock's helper
            # for it moved twice in October 2026 (module function -> PLEShardOverlap._compute_shard_overlap), and
            # it is three lines of arithmetic, so it lives here
            ck_start, ck_end = shard_index * shard_size, shard_index * shard_size + tensor.shape[0]
            tp_start, tp_end = self.shard_indices.org_vocab_start_index, self.shard_indices.org_vocab_end_index
            lo, hi = max(ck_start, tp_start), min(ck_end, tp_end)
            if lo >= hi:
                return 0
            src_start, dst_start, rows = lo - ck_start, lo - tp_start, hi - lo
            src = tensor.narrow(0, src_start, rows).to("cpu")
            dst = self._r9k_host.narrow(0, dst_start, rows)
            packed_bytes = self.head_dim * 6 // 8
            if kind == "packed":
                dst[:, :packed_bytes].copy_(src.view(torch.uint8))
            else:
                dst[:, packed_bytes:].copy_(src.contiguous().view(torch.uint8).view(rows, -1))
            return rows

        def load_weights(self, weights):
            shard_size = (self.org_vocab_size + self.split_parts - 1) // self.split_parts
            loaded = set()
            for name, w in weights:
                m = _SHARD_RE.match(name)
                if m is None:
                    raise ValueError(f"r9700: unexpected PLE int6 tensor {name!r}")
                self.load_int6_shard(int(m.group(1)), m.group(2), w, shard_size)
                loaded.add(name)
            if loaded:
                loaded.add("weight")
            return loaded

    return R9kInt6PLEEmbedding
