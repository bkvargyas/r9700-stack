"""Qwen4Exp (Flash-Next) PLE n-gram table in fused int6 rows, held in pinned host memory (UVA), on stock vLLM.

Stock vLLM's AMD PLE path wants a bf16 table ([160M rows/rank, 160] = ~51 GB/rank at TP2): it cannot load the
int6 GPTQ checkpoint (``ngram_embedding.shard_N.weight_packed`` u8 [rows, 120] + ``.weight_scale`` f16
[rows, 5]) and could not hold the table in VRAM anyway. This hook:

  * replaces ``PLEVocabParallelEmbedding`` inside ``vllm.models.qwen4_exp.amd.ple_layer`` with a subclass that
    builds its vocab-parallel metadata on the meta device, then allocates its TP slice of the table as fused
    int6 rows (130 B/row) in pinned host memory and exposes it to the GPU through a UVA view;
  * swaps the embedding's quant method so ``embedding(layer, ids)`` is libr9k's gather+dequant kernel (stock
    masking, zero-fill and TP all-reduce in ``VocabParallelEmbedding.forward`` are untouched);
  * wraps ``Qwen4ExpNGramEmbedding.load_weights`` to route ``shard_N.weight_packed / weight_scale`` into the
    fused host rows (TP overlap computed with stock ``compute_ple_shard_overlap``).

Only engages when the checkpoint's PLE shards are int6 (fp16 scales); otherwise stock behaviour.
~20.8 GB of pinned host RAM per rank at TP2 for Qwen3.8-Flash-Next.
"""
from __future__ import annotations

import json
import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_PATCHED = False
_SHARD_RE = re.compile(r"^ngram_embedding\.shard_(\d+)\.weight_(packed|scale)$")


def _checkpoint_ple_format() -> str | None:
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
    """Quant method stand-in: VocabParallelEmbedding.forward only calls .embedding()."""

    def __init__(self, head_dim: int):
        self.head_dim = head_dim

    def create_weights(self, *a, **k):  # never called (weights are made by the embedding itself)
        raise RuntimeError("unused")

    def process_weights_after_loading(self, layer):
        pass

    def embedding(self, layer, ids: torch.Tensor) -> torch.Tensor:
        from ..kernels.ple import gather_int6
        return gather_int6(layer.weight, ids, self.head_dim)


def _make_embedding_cls(base):
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # noqa: F401
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    from ..kernels.ple import int6_row_bytes

    class R9kInt6PLEEmbedding(base):
        def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
            with torch.device("meta"):
                super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
            self.head_dim = embedding_dim
            self.row_bytes = int6_row_bytes(embedding_dim)
            rows = self.num_embeddings_per_partition
            from ..utils.hostmem import pinned_empty
            host = pinned_empty((rows, self.row_bytes), torch.uint8)   # exact size (torch pinning rounds to 2^k)
            view = get_accelerator_view_from_cpu_tensor(host)
            del self.weight
            self.weight = torch.nn.Parameter(view, requires_grad=False)
            self.weight._vllm_is_uva_offloaded = True
            self._r9k_host = host
            self.quant_method = _Int6EmbeddingMethod(embedding_dim)
            self.params_dtype = torch.bfloat16
            logger.info_once("r9700: PLE int6 table %d rows x %d B = %.1f GiB pinned host (UVA) per rank",
                             rows, self.row_bytes, rows * self.row_bytes / 2**30)

        def load_int6_shard(self, shard_index: int, kind: str, tensor: torch.Tensor, shard_size: int) -> int:
            from vllm.models.qwen4_exp.common.ple import compute_ple_shard_overlap
            ov = compute_ple_shard_overlap(checkpoint_start=shard_index * shard_size, checkpoint_rows=tensor.shape[0],
                                           tp_start=self.shard_indices.org_vocab_start_index,
                                           tp_end=self.shard_indices.org_vocab_end_index)
            if ov is None:
                return 0
            src = tensor.narrow(0, ov.source_start, ov.row_count).to("cpu")
            dst = self._r9k_host.narrow(0, ov.destination_start, ov.row_count)
            packed_bytes = self.head_dim * 6 // 8
            if kind == "packed":
                dst[:, :packed_bytes].copy_(src.view(torch.uint8))
            else:
                dst[:, packed_bytes:].copy_(src.contiguous().view(torch.uint8).view(ov.row_count, -1))
            return ov.row_count

    return R9kInt6PLEEmbedding


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from vllm.models.qwen4_exp.amd import ple_layer as pl
        from vllm.models.qwen4_exp.common.ple import PLEVocabParallelEmbedding
    except Exception as e:
        logger.warning("r9700: PLE hook not installed (%s)", e)
        return False

    stock_cls = PLEVocabParallelEmbedding
    int6_cls = _make_embedding_cls(stock_cls)

    def factory(*args, **kwargs):
        if _checkpoint_ple_format() == "int6":
            return int6_cls(*args, **kwargs)
        return stock_cls(*args, **kwargs)

    pl.PLEVocabParallelEmbedding = factory

    ng = pl.Qwen4ExpNGramEmbedding
    orig_load = ng.load_weights

    def load_weights(self, weights):
        emb = self.ngram_embedding
        if not isinstance(emb, int6_cls):
            return orig_load(self, weights)
        shard_size = (emb.org_vocab_size + self.split_ngram_parts - 1) // self.split_ngram_parts
        rest, got = [], set()
        for name, w in weights:
            m = _SHARD_RE.match(name)
            if m:
                emb.load_int6_shard(int(m.group(1)), m.group(2), w, shard_size)
                got.add(name)
            else:
                rest.append((name, w))
        loaded = set(orig_load(self, rest))
        if got:
            loaded.add("ngram_embedding.weight")
            loaded.update(got)
        return loaded

    ng.load_weights = load_weights
    _PATCHED = True
    return True
