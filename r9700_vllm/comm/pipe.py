"""Prefill all-reduce pipelining for Qwen4Exp decoder layers (R9K_AR_PIPE=r9k; default stock).

At TP=4 on the Gen3 switch topology a 4096-token chunk spends ~147 of ~600 ms in the two per-layer all-reduces
(attention out-projection, MoE output), each a 21 MB message bound by the single inter-switch uplink. The ops after
the attention core are row-local (hyper-connection combine + mix, the MoE, their all-reduces), so the tail of a
layer is run in row parts: the attention output's parts are all-reduced on a second stream while the main stream
combines / mixes / runs the MoE on the parts already reduced, and each part's MoE output is all-reduced while the
next part computes. Exposed all-reduce time per layer drops from two messages to roughly one (P=2 parts) --
more parts at longer chunks (8k: P=4); below two parts the stock order runs.

Rows per part default 2048 so the routed experts' prefill tiles stay full (a 4096-chunk gives an expert ~80 rows
= two 64-row blocks; 2048-row parts give ~40 = one block each: the same work).

Mechanics: the layer's forward is rebound (Qwen4ExpDecoderLayer.forward, compat/gate.py) so the attention module's
RowParallelLinear no longer reduces (reduce_results=False) and the MoE runner skips its final all-reduce
(moe_config.skip_final_all_reduce); the whole post-attention tail is one opaque custom op (layer_tail) that at
decode widths does the stock sequence with the two all-reduces moved into it (the same kernels; captured by the
cudagraph as before) and at prefill widths runs the pipeline. A second all-reduce instance (ar4 at TP=4, the
2-rank P2P kernel at TP=2; own IPC scratch, 24 MiB messages) serves the part all-reduces on the comm stream, so it
never shares flags with the main-stream instance.
"""
from __future__ import annotations

import os
import types

import torch

from vllm.logger import init_logger

from ..compat import gate

logger = init_logger("vllm." + __name__)

ENABLED = os.environ.get("R9K_AR_PIPE", "stock") == "r9k"
ROWS = int(os.environ.get("R9K_AR_PIPE_ROWS", "2048"))          # rows per part
MAX_PARTS = int(os.environ.get("R9K_AR_PIPE_PARTS", "4"))
PIPE_MB = float(os.environ.get("R9K_AR_PIPE_MB", "24"))          # the comm-stream ar4 instance's largest message
PIPE_BLOCKS = int(os.environ.get("R9K_AR_PIPE_BLOCKS", "128"))   # its push kernels' workgroups (CUs it takes)
PIPE_SDMA = os.environ.get("R9K_AR_PIPE_SDMA", "0") == "1"       # DMA-engine pushes: no CUs held while bytes move

_LAYERS: dict[int, torch.nn.Module] = {}
_ST = types.SimpleNamespace(stream=None, ar=None, tried=False)


def parts(M: int) -> list[int] | None:
    """Row boundaries of the pipeline parts, or None when M is below two parts."""
    P = min(MAX_PARTS, M // ROWS)
    if P < 2:
        return None
    return [M * i // P for i in range(P + 1)]


def comm():
    """The comm-stream ar4 instance (built once, collectively, at install)."""
    if _ST.tried:
        return _ST.ar
    _ST.tried = True
    try:
        from vllm.distributed.parallel_state import get_tp_group
        from .r9k_ar4 import R9kAllReduce4
        tp = get_tp_group()
        if tp.world_size == 4:
            a = R9kAllReduce4(tp.cpu_group, torch.cuda.current_device(), max_mb=PIPE_MB, blocks=PIPE_BLOCKS,
                              sdma=PIPE_SDMA)
        elif tp.world_size == 2 and os.environ.get("R9K_AR_IMPL", "r9k") == "r9k":
            from .r9k_ar import R9kAllReduce                 # the 2-rank P2P all-reduce (wht-compressed >= 128 KB)
            a = R9kAllReduce(tp.cpu_group, torch.cuda.current_device(), max_mb=PIPE_MB)
        else:
            logger.info("r9700: all-reduce pipelining needs TP=4 (ar4) or TP=2 (r9k 2-rank); off")
            return None
        if a.disabled:
            return None
        _ST.ar = a
        _ST.stream = torch.cuda.Stream(priority=-1)
    except Exception as e:                                   # noqa: BLE001
        logger.warning("r9700: all-reduce pipelining off (%s)", e)
        _ST.ar = None
    return _ST.ar


# ------------------------------------------------------------------------------------------------ the tail op
def _tail(hidden_states: torch.Tensor, attn_out: torch.Tensor, injection: torch.Tensor, lid: int
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden [M, DIM], attn_out [M, HID] (TP-partial), injection [M, HC] -> (hidden', mlp_out (reduced), inj')."""
    from vllm.distributed import tensor_model_parallel_all_reduce as ar_main
    from .. import hc as H
    layer = _LAYERS[lid]
    mlp_hc, mlp = layer.mlp_hyper_connection, layer.mlp
    M = attn_out.shape[0]
    ar = _ST.ar
    b = parts(M) if ar is not None else None
    if b is not None and not (ar.should(attn_out[b[0]:b[1]]) and attn_out.is_contiguous() and hidden_states.is_contiguous()
                              and injection.is_contiguous()):
        b = None
    if b is None:                                            # decode and short prefills: the stock order
        attn_r = ar_main(attn_out)
        hs, bi, inj = mlp_hc.combine_and_mix(hidden_states, attn_r, injection)
        return hs, ar_main(mlp(bi)), inj

    cs, main = _ST.stream, torch.cuda.current_stream()
    P = len(b) - 1
    attn_r = torch.empty_like(attn_out)
    mlp_out = torch.empty_like(attn_out)
    hs_out = torch.empty_like(hidden_states)
    xn = torch.empty_like(hidden_states)
    for t in (attn_out, attn_r, mlp_out):
        t.record_stream(cs)
    e0 = torch.cuda.Event()
    e0.record(main)
    cs.wait_event(e0)
    ev_a = []
    with torch.cuda.stream(cs):                              # every attention part first: they are all ready
        for p in range(P):
            r = slice(b[p], b[p + 1])
            ar.all_reduce(attn_out[r], out=attn_r[r])
            e = torch.cuda.Event()
            e.record(cs)
            ev_a.append(e)
    inj_parts, ev_m = [], []
    w, eps, hc = mlp_hc.hc_norm.weight, mlp_hc.config.rms_norm_eps, mlp_hc.hc_count
    for p in range(P):
        r = slice(b[p], b[p + 1])
        main.wait_event(ev_a[p])
        hs_p, xn_p = H.combine_norm_into(hidden_states[r], attn_r[r], injection[r], w, eps, hc, out=hs_out[r], y=xn[r])
        if hs_p.data_ptr() != hs_out[r].data_ptr():          # stock fallback returned fresh tensors
            hs_out[r].copy_(hs_p)
        bi, inj_p = H._mix_tail(mlp_hc, xn_p)
        inj_parts.append(inj_p)
        mo = mlp(bi)                                         # TP-partial: the runner's final all-reduce is skipped
        if not mo.is_contiguous():
            mo = mo.contiguous()
        e = torch.cuda.Event()
        e.record(main)
        cs.wait_event(e)
        mo.record_stream(cs)
        with torch.cuda.stream(cs):
            ar.all_reduce(mo, out=mlp_out[r])
            e2 = torch.cuda.Event()
            e2.record(cs)
            ev_m.append(e2)
    for e2 in ev_m:
        main.wait_event(e2)
    return hs_out, mlp_out, torch.cat(inj_parts, 0)


def _tail_fake(hidden_states, attn_out, injection, lid):
    return hidden_states.new_empty(hidden_states.shape), attn_out.new_empty(attn_out.shape), \
        injection.new_empty(injection.shape)


_OP_DONE = False


def _register() -> None:
    global _OP_DONE
    if _OP_DONE:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    from .. import ops
    ops.register()
    direct_register_custom_op("layer_tail", _tail, mutates_args=[], fake_impl=_tail_fake, target_lib=ops._LIB)
    _OP_DONE = True


# ------------------------------------------------------------------------------------------ the layer forward
def _forward(self, hidden_states, prev_block_output, prev_injection, positions, *, input_ids, query_start_loc,
             ngram_context):
    """Qwen4ExpDecoderLayer.forward (vLLM e97573215) up to the attention core; the tail is the pipelined op."""
    attn_hc = self.attn_hyper_connection
    if self.ple is not None:
        if prev_block_output is not None and prev_injection is not None:
            hidden_states = attn_hc.combine(hidden_states, prev_block_output, prev_injection)
            prev_block_output = prev_injection = None
        if input_ids is None or query_start_loc is None or ngram_context is None:
            raise RuntimeError("PLE inputs were not prepared")
        hidden_states = hidden_states + self.ple(hidden_states, input_ids, query_start_loc, ngram_context)
    if prev_block_output is not None and prev_injection is not None:
        hidden_states, block_input, injection = attn_hc.combine_and_mix(hidden_states, prev_block_output,
                                                                        prev_injection)
    else:
        hidden_states, block_input, injection = attn_hc.mix(hidden_states)
    if self.layer_type == "linear_attention":
        attn_out = self.linear_attn(hidden_states=block_input)
    else:
        attn_out = self.self_attn(hidden_states=block_input, positions=positions)
    return torch.ops.r9700.layer_tail(hidden_states, attn_out, injection, self._r9k_lid)


def _row_linears(mod: torch.nn.Module) -> list:
    return [m for m in mod.modules() if type(m).__name__ == "RowParallelLinear" and getattr(m, "reduce_results", False)
            and getattr(m, "tp_size", 1) > 1]


def install(model: torch.nn.Module) -> int:
    """Rebind the forward of every MoE decoder layer under `model` to the pipelined tail. R9K_AR_PIPE=stock skips."""
    if not ENABLED:
        return 0
    if comm() is None:
        return 0
    from .. import hc as H
    if not H.available():
        logger.warning("r9700: all-reduce pipelining needs the r9k hyper-connection ops; off")
        return 0
    _register()
    n = 0
    for name, layer in model.named_modules():
        if type(layer).__name__ != "Qwen4ExpDecoderLayer" or getattr(layer, "_r9k_lid", None) is not None:
            continue
        mlp = getattr(layer, "mlp", None)
        attn = getattr(layer, "linear_attn", None) if getattr(layer, "layer_type", "") == "linear_attention" \
            else getattr(layer, "self_attn", None)
        mlp_hc = getattr(layer, "mlp_hyper_connection", None)
        runners = [m for m in mlp.modules() if type(m).__name__ == "MoERunner"] if mlp is not None else []
        rows = _row_linears(attn) if attn is not None else []
        ok = (type(mlp).__name__ == "Qwen4ExpSparseMoeBlock" and not getattr(mlp, "replicate_shared_expert", True)
              and not getattr(mlp, "is_sequence_parallel", True) and len(runners) == 1 and len(rows) == 1
              and all(hasattr(layer, a) for a in ("attn_hyper_connection", "ple", "layer_type"))
              and getattr(mlp_hc, "_r9k_mix", False)
              and all(hasattr(mlp_hc, a) for a in ("hc_norm", "config", "hc_count"))
              and hasattr(runners[0], "moe_config") and hasattr(runners[0].moe_config, "skip_final_all_reduce")
              and not runners[0]._fused_output_is_reduced)
        if not ok:
            logger.warning_once("r9700: all-reduce pipelining skipped for %s (layer shape changed)", name)
            continue
        gate.check("layer_tail_pipe")
        rows[0].reduce_results = False                       # the tail op reduces the attention output
        runners[0].moe_config.skip_final_all_reduce = True   # and the MoE output
        for m in mlp.modules():                              # the FusedMoE layer shares the config object; be sure
            cfg = getattr(m, "moe_config", None)
            if cfg is not None and hasattr(cfg, "skip_final_all_reduce"):
                cfg.skip_final_all_reduce = True
        layer._r9k_lid = len(_LAYERS)
        _LAYERS[layer._r9k_lid] = layer
        layer.forward = types.MethodType(_forward, layer)
        n += 1
    if n:
        logger.info("r9700: all-reduce pipelining on %d decoder layers (%d rows a part, up to %d parts, comm "
                    "stream ar4 <= %g MiB)", n, ROWS, MAX_PARTS, PIPE_MB)
    return n
