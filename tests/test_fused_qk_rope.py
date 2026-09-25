"""vLLM's fused Triton QK-RMSNorm + MRoPE + gate kernel vs the eager path Qwen4Exp's QSA layer runs on ROCm.

Single GPU:  python3 tests/test_fused_qk_rope.py

The stock layer enables ``fused_qk_rmsnorm_rope_gate`` only on CUDA and otherwise runs GemmaRMSNorm x2 + the
rotary module + chunk/reshape eagerly (~30 launches per layer at decode). attn/qsa.py switches the layer to the
fused kernel on gfx1201 (R9K_FUSED_QKROPE=0 reverts); this checks the two agree on Flash-Next's exact
configuration: head_dim 256, partial rotary 0.25 (rotary_dim 64), interleaved MRoPE sections [11, 11, 10],
theta 1e7, Gemma norm (1 + w), TP=4 head counts (6 q / 1 kv), 2-D positions (text: T = H = W).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate  # noqa: E402
from vllm.model_executor.layers.layernorm import GemmaRMSNorm  # noqa: E402
from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402

ROPE = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
        "rope_theta": 10000000, "rope_type": "default"}


def run(n, hq, hkv, hd=256, eps=1e-6, seed=0, positions_2d=True, mm=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device("cuda")
    with set_current_vllm_config(VllmConfig()):      # CustomOp construction needs a config context
        rope = get_rope(head_size=hd, max_position=262144, rope_parameters=ROPE).to(dev)
        qn, kn = GemmaRMSNorm(hd, eps=eps).to(dev), GemmaRMSNorm(hd, eps=eps).to(dev)
    with torch.no_grad():
        qn.weight.copy_(torch.randn(hd, generator=g) * 0.3)
        kn.weight.copy_(torch.randn(hd, generator=g) * 0.3)
    q_size, kv_size = hq * hd, hkv * hd
    qkv = (torch.randn((n, 2 * q_size + 2 * kv_size), generator=g) * 2).to(torch.bfloat16).to(dev)
    pos1 = torch.randint(0, 200000, (n,), generator=g).to(dev)
    if positions_2d:
        pos = pos1.unsqueeze(0).expand(3, -1).contiguous()
        if mm:                                       # image-like: distinct H/W streams
            pos = pos.clone(); pos[1] += 7; pos[2] += 13
    else:
        pos = pos1
    # eager (stock ROCm path)
    q_gate, k_raw, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
    k = k_raw
    qg = q_gate.view(n, hq, -1)
    q, gate = torch.chunk(qg, 2, dim=-1)
    q, gate = q.reshape(n, -1), gate.reshape(n, -1)
    q = qn(q.view(-1, hq, hd)).view(-1, q_size)
    k = kn(k.view(-1, hkv, hd)).view(-1, kv_size)
    q_e, k_e = rope(pos, q, k)
    # fused
    q_f, k_f, gate_f = fused_qk_rmsnorm_rope_gate(
        q_gate, k_raw, qn.weight, kn.weight, rope.cos_sin_cache, pos, qn.variance_epsilon, hq, hkv, hd, rope.rotary_dim,
        norm_beta=1.0, mrope_section=rope.mrope_section if positions_2d else None)
    torch.cuda.synchronize()
    def err(a, b):
        a, b = a.float(), b.float()
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()
    eq, ek, egate = err(q_f, q_e), err(k_f, k_e), err(gate_f, gate)
    ok = eq < 2e-2 and ek < 2e-2 and egate == 0.0 and q_f.shape == q_e.shape and k_f.shape == k_e.shape
    print(f"  n={n:4d} hq={hq} hkv={hkv} 2d={positions_2d} mm={mm}: rel err q {eq:.2e} k {ek:.2e} gate {egate:.1e} "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main():
    bad = 0
    bad += not run(1, 6, 1)
    bad += not run(4, 6, 1, seed=1)
    bad += not run(37, 6, 1, seed=2)
    bad += not run(4096, 6, 1, seed=3)
    bad += not run(64, 12, 1, seed=4)               # TP=2
    bad += not run(64, 6, 1, seed=5, mm=True)        # distinct T/H/W streams
    bad += not run(64, 6, 1, seed=6, positions_2d=False)
    print("fused qk-norm/rope/gate:", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
