"""Version gate for the plugin's remaining monkeypatches.

Every patch that reaches into a vLLM internal (rather than an extension point) is listed here with the vLLM
commit(s) it was written and tested against. On any other vLLM it still installs if its target is present
(each patch is written to self-disable once upstream behaves), but logs a warning so a pin bump re-tests it.
tests/test_integration.py asserts each patch's target still exists.
"""
from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

TESTED = {
    # patch name: (vLLM commits tested, upstream issue/PR that would retire it)
    "mtp_allowlist": (("dee37d891", "e97573215"), "vllm-project/vllm#55292"),
    # the modular MoE kernel copies its workspace into the layer output on ROCm unless aiter is on (moe/fold.py)
    "moe_output_alias": (("e97573215",), "vllm: FusedMoEKernelModularImpl._fused_experts output alias on ROCm"),
    # the runner adds the shared expert after the experts return; ours folds it into the top-k sum (moe/fold.py)
    "moe_shared_fold": (("e97573215",), "vllm: MoERunner.forward shared_output + fused_output"),
    # the decoder layer's post-attention tail as one op with the two all-reduces pipelined by row parts
    # (comm/pipe.py): Qwen4ExpDecoderLayer.forward, RowParallelLinear.reduce_results, moe_config.skip_final_all_reduce
    "layer_tail_pipe": (("e97573215",), "vllm: an all-reduce / compute overlap for row-local layer tails"),
}


def vllm_commit() -> str:
    """Short git sha of the running vLLM ('0.3.1.dev85+gdee37d891' -> 'dee37d891'), or its version string."""
    try:
        import vllm
        v = getattr(vllm, "__version__", "") or ""
    except Exception:
        return ""
    return v.split("+g", 1)[1].split(".", 1)[0] if "+g" in v else v


def check(name: str) -> bool:
    tested, retire = TESTED[name]
    cur = vllm_commit()
    if not any(cur and (cur.startswith(t) or t.startswith(cur)) for t in tested):
        logger.warning("r9700: patch %r untested on vLLM %s (tested: %s); retire via %s", name, cur or "?",
                       ", ".join(tested), retire)
    return True
