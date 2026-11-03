#!/usr/bin/env python3
"""mklist.py LIST.txt OUT.txt: the per-layer launch list for notes/exl3-ledger.md from an eager ledger.py --list (one T = 6
verify step, rank 0): the step's pre-forward, decoder layers 0, 1, 2, 3, 24, 37 and 39 in full (the layer classes: first, Engram,
compressor + indexer source, plain, indexer-only, aux-hidden-state, last), the sampler, and the drafter (pre, layer 0, last
layer + heads). Adds a bytes column for the weight read of each GEMV / GEMM (per rank, from the layer map; expert kernels:
the plan's estimate) and the effective GB/s at the eager duration."""
import re, sys
rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    p = [x.strip() for x in line.split("|")]
    h = p[0].split(None, 2)
    rows.append(dict(i=int(h[0]), phase=h[1], cat=h[2] if len(h) > 2 else "", name=p[1], grid=p[2], dur=float(p[3]),
                     op=p[5] if len(p) > 5 else "", dims=p[6] if len(p) > 6 else ""))
MB = 1e6
def wbytes(r, ctx):
    n, g, d = r["name"], r["grid"], r["dims"]
    m = re.match(r"exl3_gemv_mr_dot_(single|multi)<(\d)", n)
    if m:
        K = int(m.group(2)); gx = int(re.match(r"\[(\d+)", g).group(1))
        if m.group(1) == "multi":       # fused_wqa_wkv: 5120 -> 1280 + 512
            return 5120 * 1792 * K / 8, "fused_wqa_wkv 5120->1792"
        tiles = {256: (1280, 4096, "wq_b 1280->4096 (rank's 32 heads)"), 320: (1024, 5120, "wo_b 1024->5120 (row split)"),
                 200: (6144, 3200, "engram.wkv 6144->3200")}
        if ctx == "drafter-pre" and gx == 320:
            return 15360 * 5120 * K / 8, "mtp.0.main_proj 15360->5120 (replicated)"
        if gx in tiles:
            a, b, lab = tiles[gx]; return a * b * K / 8, lab
        return None, ""
    m = re.search(r"\[\[([\d, ]+)\], \[([\d, ]+)\]", d)
    if r["cat"].startswith("hipBLAS") and m:
        b = [int(x) for x in m.group(2).split(",")]
        el = 1
        for x in b:
            el *= x
        lab = {(4096, 1024): "wo_a slice (bf16)", (1, 4096, 1024): "wo_a slice (bf16)", (5120, 576): "shared w13 (bf16, TP)",
               (288, 5120): "shared w2 (bf16, TP)", (5120, 384): "router gate", (5120, 1024): "compressor wkv+wgate (bf16)",
               (5120, 32): "indexer weights_proj", (512, 128): "indexer wk", (5120, 16160): "lm_head (bf16, vocab slice)",
               (576, 5120): "shared w13 (bf16, TP)", (5120, 288): "shared w2 (bf16, TP)", (5120, 128): "drafter router gate",
               }.get(tuple(b), "")
        if "wvSplitK" in n and b and b[0] == 129280:
            lab = "markov_head.head [129280, 256] f16"
        if "wvSplitK" in n and b and b[0] == 16160:
            lab = "drafter lm_head (bf16, vocab slice)"
        return el * 2, lab
    if n.startswith("k_gu"):
        return None, "experts gate+up: distinct local experts x 8.85 MB (K=3; plan: ~4.3 a layer at T=6)"
    if n.startswith("k_dn"):
        return None, "experts down: distinct local experts x 4.42 MB (K=3)"
    if n.startswith("r9k_mhc_seam"):
        return 24 * 20480 * 4, "hc_fn f32 [24, 20480]"
    if n.startswith("r9k_ar8") or n.startswith("r9k_ag") or n.startswith("ncclDev"):
        m = re.search(r"\[\[([\d, ]+)\]", d)
        if m:
            el = 1
            for x in m.group(1).split(","):
                el *= int(x)
            return None, f"message {el * 2 / 1024:.0f} KiB bf16"
    return None, ""
fw = [r for r in rows if r["phase"] == "forward"]; dr = [r for r in rows if r["phase"] == "draft"]; sm = [r for r in rows if r["phase"] == "sample"]
def layers(rr):
    seams = [j for j, r in enumerate(rr) if r["name"].startswith("r9k_mhc_seam_kernel")][0::2]
    segs = [rr[:seams[0]]] + [rr[s:(seams[k + 1] if k + 1 < len(seams) else len(rr))] for k, s in enumerate(seams)]
    return segs
F = layers(fw); D = layers(dr)
out = open(sys.argv[2], "w")
out.write("# Launch list of one T = 6 verify step (single stream, k = 5), rank 0, eager launch E1 (EAGER=1, ADAPTIVE=false: adaptive\n"
          "# verification requires full graphs), v0.4.0 tree. Durations are eager kernel times (no graph replay), so the collective\n"
          "# rows include waiting for the slowest rank (eager launches skew the ranks by milliseconds): read their graph numbers in\n"
          "# notes/exl3-ledger.md. Columns: # | category | kernel | grid block | us | weight bytes read (per rank) | GB/s | what | cpu op | input dims\n"
          f"# source: ~/ds41/exl3/Z/E1/s1-list.txt (all {len(rows)} launches of the step), traces ~/ds41/exl3/Z/E1/s1/\n")
def dump(title, seg, ctx=""):
    out.write(f"\n== {title}: {len(seg)} launches, {sum(r['dur'] for r in seg if r['cat'] != 'collective'):.1f} us without collectives\n")
    for q, r in enumerate(seg):
        b, lab = wbytes(r, ctx)
        bw = f"{b / r['dur'] / 1e3:6.0f}" if b and r["dur"] > 0 else "      "
        out.write(f"{q:3d} {r['cat'][:30]:30s} {r['name'][:58]:58s} {r['grid']:26s} {r['dur']:8.1f} {(f'{b / MB:7.2f} MB') if b else '          '} {bw}  {lab[:46]:46s} {r['op'][:30]:30s} {r['dims'][:90]}\n")
dump("forward: step preamble (input prep, embedding, Engram lookups)", F[0])
for li, title in ((0, "layer 0 (first layer: mhc pre, SWA attention)"), (1, "layer 1 (Engram layer)"),
                  (2, "layer 2 (kv source: compressor + indexer wk + sparse attention)"), (3, "layer 3 (plain layer: SWA / sparse decode, no compressor, no indexer)"),
                  (24, "layer 24 (indexer layer without compressor)"), (37, "layer 37 (aux hidden state for the drafter)"),
                  (39, "layer 39 (last layer: final mhc post + hc head)")):
    dump(f"forward: {title}", F[li + 1])
dump("sample_tokens: target lm_head + logits gather + rejection sampler", sm)
dump("drafter: preamble (main_proj over the three aux hidden states, KV writes for the block, input prep)", D[0], "drafter-pre")
dump("drafter: layer 0", D[1])
dump("drafter: last layer + final norm, drafter lm_head, logits gather, markov head (5 sequential tokens)", D[-1])
