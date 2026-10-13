# Renders docs/r9700-stack-v0.3.3.png, the one-image summary at the top of README.md. Needs matplotlib:
#   docker run --rm -v $PWD:/w --entrypoint bash r9700/vllm:dev -c "pip install -q matplotlib && python3 /w/docs/summary-image.py"
# Update the numbers below from notes/release-v<version>.md when a release changes them.
# r9700-stack v0.3.2 summary image for the README. Dark surface, the reference dataviz palette's dark steps
# (slots 1-3, documented as validated all-pairs in dark mode), text in ink tokens only, thin marks.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

SURF, SURF2, INK, INK2, MUTED, GRID = "#1a1a19", "#232322", "#f0efec", "#c3c2b7", "#8f8e86", "#3a3a38"
S1, S2, S3 = "#3987e5", "#d95926", "#199e70"          # Flash-Next 4 cards, Flash-Next 2 cards (host RAM), 27B 2 cards
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})
W, H = 16, 9
fig = plt.figure(figsize=(W, H), dpi=200)
fig.patch.set_facecolor(SURF)

def text(x, y, s, size, color=INK, weight="normal", ha="left", va="top", **kw):
    fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)

def tile(x, y, w, h, label, value, sub=None):
    fig.patches.append(FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H, boxstyle="round,pad=0,rounding_size=0.008",
                                      transform=fig.transFigure, facecolor=SURF2, edgecolor="none"))
    text(x + 0.22, y + 0.2, label, 10.5, INK2)
    text(x + 0.22, y + 0.5, value, 20, INK, "semibold")
    if sub:
        text(x + 0.22, y + h - 0.42, sub, 9.5, MUTED)

# ---- header
text(0.6, 0.45, "r9700-stack", 30, INK, "bold")
text(3.55, 0.62, "v0.3.3", 15, INK2)
text(0.6, 1.1, "Qwen3.8 on AMD Radeon AI PRO R9700  ·  stock vLLM + stock ROCm  ·  tuned gfx1201 kernels loaded as a plugin, no fork",
     11.5, INK2)

# ---- hero figure (one) + the KPI row for the four-card configuration
text(0.6, 1.75, "Flash-Next, four cards, tensor parallel 4", 10.5, INK2)
text(0.6, 2.0, "225", 52, INK, "semibold")
text(2.35, 2.78, "tok/s single-stream decode", 13, INK2, va="bottom")
text(0.65, 3.0, "14.3 ms a step, 3.1 tokens a step with MTP-3", 10, MUTED, va="top")

tiles = [("Aggregate decode, 16 streams", "944 tok/s", "530 at 8 streams (BetterBench)"),
         ("Prompt processing, 16k prompt", "7,489 tok/s", "7,141 at 8k · 7,233 at 32k"),
         ("KV cache", "518k tokens", "15 full 32k contexts"),
         ("Time to first token", "76 ms", "p50, single stream")]
x0, y0, tw, th, gap = 6.35, 1.62, 2.25, 1.32, 0.15
for i, (l, v, s) in enumerate(tiles):
    tile(x0 + i * (tw + gap), y0, tw, th, l, v, s)

# ---- grouped bars: aggregate decode by streams, three configurations
ax = fig.add_axes([0.6 / W, 1 - 7.35 / H, 6.3 / W, 3.75 / H])
streams = ["1", "8", "16"]
data = [("Flash-Next, 4 cards", S1, [225, 623, 944]),
        ("Flash-Next, 2 cards, experts in host RAM", S2, [143, 257, 245]),
        ("27B-NVFP4, 2 cards", S3, [222, 527, 650])]
import numpy as np
xs = np.arange(len(streams)); bw = 0.13
XMIN, XMAX, YMAX = -0.6, len(streams) - 0.4, 1080
def bar(x, v, col):   # <= 24 px thick, 4 px rounded data-end, square at the baseline (drawn in axes fractions so the corners are round)
    fx, fw, fh = (x - bw / 2 - XMIN) / (XMAX - XMIN), bw / (XMAX - XMIN), v / YMAX
    ax.add_patch(FancyBboxPatch((fx, 0), fw, fh, boxstyle="round,pad=0,rounding_size=0.0075", mutation_aspect=3.75 / 6.3,
                                transform=ax.transAxes, facecolor=col, edgecolor="none", linewidth=0, clip_on=False))
    ax.add_patch(Rectangle((fx, 0), fw, min(fh, 0.04), transform=ax.transAxes, facecolor=col, edgecolor="none", clip_on=False))
for j, (name, col, vals) in enumerate(data):
    for x, v in zip(xs + (j - 1) * (bw + 0.03), vals):
        bar(x, v, col)
        ax.text(x, v + 14, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=INK2)   # every bar is a headline
ax.set_xlim(XMIN, XMAX)
ax.set_xticks(xs); ax.set_xticklabels([f"{s} stream{'s' if s != '1' else ''}" for s in streams], fontsize=9.5)
ax.set_ylim(0, 1080); ax.set_yticks([0, 250, 500, 750, 1000]); ax.set_yticklabels(["0", "250", "500", "750", "1K"], fontsize=8.5)
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
ax.set_title("Aggregate decode, tok/s, by concurrent streams", loc="left", fontsize=11, color=INK, pad=10)
ax.legend(handles=[Line2D([0], [0], marker="s", color=SURF, markerfacecolor=c, markersize=8, linewidth=0, label=n) for n, c, _ in data],
          loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2, handletextpad=0.3, borderaxespad=0.2)

# ---- right column: what is inside
text(7.6, 3.55, "What the kernels do", 11, INK, "semibold")
items = ["Grouped MXFP4 x fp8 MoE GEMMs with routing, activation, quant and top-k sum fused around them",
         "fp8 / NVFP4 / MXFP4 dense GEMMs on WMMA, the activation quant inside the GEMM",
         "Hyper-connection GEMMs on fp8 copies at every width; the bf16 copies freed (KV cache +14%)",
         "One fused Gated DeltaNet decode core per layer; one state page per request under speculation",
         "Paged prefill attention and the query-sparse attention kernels",
         "P2P all-reduce / all-gather over PCIe for 2 and 4 cards, HIP-graph safe; compressed for prefill",
         "MTP and DFlash2 speculative decoding; int6 per-layer embeddings in pinned host memory",
         "Experts streamed from host RAM through a device LRU cache for the two-card Flash-Next"]
for i, it in enumerate(items):
    text(7.6, 3.92 + i * 0.37, "·  " + it, 9.3, INK2)

text(7.6, 6.95, "Every default gated", 11, INK, "semibold")
text(7.6, 7.3, "GSM8K, all 1,319 questions, paired per question against the previous numerics (p = 1.00)  ·  HumanEval 162 / 164\n"
               "0 bad answers in 5,060 strict-sanity checks under overload  ·  0 errors in 1,224 mixed-length soak requests\n"
               "across every configuration  ·  4-rank communication tests bit-exact  ·  kernel unit gates", 9.3, INK2, linespacing=1.6)

# ---- footer: deploy
fig.patches.append(Rectangle((0.6 / W, 1 - 8.0 / H), 14.8 / W, 0.002, transform=fig.transFigure, facecolor=GRID, edgecolor="none"))
text(0.6, 8.15, "Deploy", 11, INK, "semibold")
text(1.6, 8.17, "docker build -t r9700/vllm:dev docker/     then     GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 bash serve/flashnext.sh     or     bash serve/27b.sh",
     9.5, INK2, family="DejaVu Sans Mono")
text(1.6, 8.5, "OpenAI-compatible API on :8080  ·  210 W per card  ·  github.com/bkvargyas/r9700-stack", 9.3, MUTED)
fig.savefig("docs/r9700-stack-v0.3.3.png", dpi=200, facecolor=SURF)
print("saved")
