"""PLE short-conv prefill on our tiled transpose (kernels/r9k_transpose.hip, ple/short_conv.py) vs the stock method.

Single GPU:  python3 tests/test_ple_conv.py            (BENCH=0 skips the timing table)

  * r9k_transpose16 vs x.transpose(1, 2).contiguous() on ragged shapes and strided (sliced) inputs, bf16 and fp16
  * ple/short_conv.prefill_batched vs Qwen4ExpPLELayer._short_conv_dilated_prefill_batched, run unbound on a stand-in
    layer object (the method only reads conv_state_len and short_conv_dilation): output bit-identical, written-back
    conv state bit-identical, on one long prefill, a mixed batch with a decode prefix, empty and NULL-block states
Then times both on the served shape (one 4096-token prefill, 4 x 2560 channels).
"""
from __future__ import annotations

import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer  # noqa: E402
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID  # noqa: E402

from r9700_vllm.ple import short_conv as S  # noqa: E402

dev = torch.device("cuda")
H, KSZ, DIL = 4 * 2560, 4, 3           # Flash-Next: hc 4 x hidden 2560, conv kernel 4, dilation = ngram_size
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def ok(name, cond):
    global bad
    print(f"  {name:<60} {'ok' if cond else 'FAIL'}")
    bad += not cond


def t_transpose():
    for B, R, C, dt in ((1, 4096, H, torch.bfloat16), (3, 1000, 2560, torch.bfloat16), (2, 33, 77, torch.float16),
                        (1, 4096 + 9, 10240 + 24, torch.bfloat16), (5, 1, 1, torch.bfloat16), (1, 64, 64, torch.bfloat16)):
        x = torch.randn((B, R, C), generator=g).to(dt).to(dev)
        ok(f"transpose [{B},{R},{C}] {str(dt)[6:]}", torch.equal(S.transpose12(x), x.transpose(1, 2).contiguous()))
    x = torch.randn((2, 300, 600), generator=g).to(torch.bfloat16).to(dev)
    v = x[:, 7:263, 8:520]                                     # strided rows and batch, contiguous last dim
    ok("transpose strided view", torch.equal(S.transpose12(v), v.transpose(1, 2).contiguous()))


def run_case(name, q_lens, num_decode_tokens, state_slots, has_init, null_slots=(), max_peak=None):
    """Same inputs to both implementations; compare output and conv state."""
    fake = types.SimpleNamespace(conv_state_len=(KSZ - 1) * DIL, short_conv_dilation=DIL)
    P = len(q_lens)
    ntok = sum(q_lens)
    x = torch.randn((ntok, H), generator=g).to(torch.bfloat16).to(dev)
    w = (torch.randn((H, KSZ), generator=g) * 0.3).to(torch.bfloat16).to(dev)
    qsl = torch.tensor([0] + [num_decode_tokens] * 0 + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32)
    qsl = (qsl + num_decode_tokens)                            # non_spec_query_start_loc includes the decode prefix
    qsl = torch.cat([torch.arange(num_decode_tokens, dtype=torch.int32), qsl]) if num_decode_tokens else qsl
    md = types.SimpleNamespace(non_spec_query_start_loc=qsl.to(dev), has_initial_states_p=torch.tensor(has_init).to(dev),
                               max_prefill_query_len=max(q_lens))
    S_ = state_slots
    st0 = torch.randn((S_, H, fake.conv_state_len + 3), generator=g).to(torch.bfloat16).to(dev) if S_ else \
        torch.empty((0, H, fake.conv_state_len + 3), dtype=torch.bfloat16, device=dev)
    idx = torch.randperm(max(S_, P), generator=g)[:P].to(torch.int32) if S_ else torch.zeros(P, dtype=torch.int32)
    for j in null_slots:
        idx[j] = NULL_BLOCK_ID
    idx = idx.to(dev)
    outs, peaks = [], []
    for fn in (Qwen4ExpPLELayer._short_conv_dilated_prefill_batched, S.prefill_batched):
        cs = st0.clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        o = fn(fake, x, md, cs, w, idx, P, num_decode_tokens, ntok)
        torch.cuda.synchronize()
        peaks.append((torch.cuda.max_memory_allocated() - base) / 2**20)
        outs.append((o, cs))
    ok(f"{name}: output", torch.equal(outs[0][0], outs[1][0]))
    ok(f"{name}: conv state", torch.equal(outs[0][1], outs[1][1]))
    if max_peak is not None:
        ok(f"{name}: peak {peaks[1]:.0f} MiB (stock {peaks[0]:.0f}), limit {max_peak}", peaks[1] <= max_peak)
    return x, w, md, st0, idx, fake, P, ntok


def main():
    if not S.available():
        print("libr9k.so has no r9k_transpose16: rebuild kernels/")
        sys.exit(1)
    t_transpose()
    run_case("prefill 1 x 4096, initial state", [4096], 0, 8, [True])
    run_case("prefill 1 x 4096, no initial state", [4096], 0, 8, [False])
    run_case("mixed 3 reqs + 5 decode rows", [300, 1500, 17], 5, 16, [True, False, True])
    run_case("tiny lengths 1..5", [1, 2, 3, 4, 5], 0, 8, [True] * 5)
    run_case("NULL block slot", [700, 300], 0, 8, [True, True], null_slots=(1,))
    run_case("empty state cache", [512, 64], 0, 0, [False, False])
    # prefills of very different lengths in one step: stock packs [prefills, longest, 4*hidden] for all of them
    # (the four-card OOM of 2026-10-02); ours packs by length, same bits, a fraction of the memory
    assert S.length_groups([3800, 40, 40, 40, 40, 40, 40, 40], 5100) == [[0], [1, 2, 3, 4, 5, 6, 7]]
    assert S.length_groups([500] * 8, 5000) == [list(range(8))]
    assert S.length_groups([96, 2000, 500, 1500], 5120) == [[1, 3], [2, 0]]
    run_case("skewed 3800 + 7 x 40", [3800, 40, 40, 40, 40, 40, 40, 40], 0, 16, [True] * 8, max_peak=1100)
    run_case("skewed, unsorted, decode prefix", [17, 2900, 300, 5, 800], 6, 16, [True, False, True, True, False],
             max_peak=1000)
    run_case("skewed with a NULL block", [60, 3000, 900], 0, 8, [True, True, True], null_slots=(0,), max_peak=1000)
    run_case("16 prefills, one long", [3000] + [70] * 15, 0, 32, [True] * 16, max_peak=1000)
    run_case("equal lengths 8 x 500 (one group)", [500] * 8, 0, 16, [True] * 8)
    print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")
    if os.environ.get("BENCH", "1") == "1":
        x, w, md, st0, idx, fake, P, ntok = run_case("bench shape", [4096], 0, 8, [True])
        for fn, lab in ((Qwen4ExpPLELayer._short_conv_dilated_prefill_batched, "stock"), (S.prefill_batched, "ours")):
            cs = st0.clone()
            for _ in range(3):
                fn(fake, x, md, cs, w, idx, P, 0, ntok)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                fn(fake, x, md, cs, w, idx, P, 0, ntok)
            torch.cuda.synchronize()
            print(f"  prefill short conv 1 x 4096 x {H}   {lab:<6} {(time.perf_counter() - t) / 10 * 1e6:8.0f} us")
        x = torch.randn((1, 4096, H), generator=g).to(torch.bfloat16).to(dev)
        for f, lab in ((lambda: x.transpose(1, 2).contiguous(), "torch"), (lambda: S.transpose12(x), "ours")):
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                f()
            torch.cuda.synchronize()
            print(f"  transpose [1, 4096, {H}] bf16            {lab:<6} {(time.perf_counter() - t) / 10 * 1e6:8.0f} us")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
