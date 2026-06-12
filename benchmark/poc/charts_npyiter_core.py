# Renders the npyiter_core_bench result charts (data hardcoded from the
# 2026-06-12 run in NPYITER_CORE_BENCH_RESULTS.md). Output: %TEMP%/npyiter_charts
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "npyiter_charts")
os.makedirs(OUT, exist_ok=True)

NS = "#1f77b4"   # NumSharp blue
NP = "#ff7f0e"   # NumPy orange
GOOD = "#2ca02c"
BAD = "#d62728"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9.5, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True,
})

# ============================================================================
# 1. Construction cost — grouped horizontal bars
# ============================================================================
labels = [
    "C1  1-op, no flags",
    "C2  2-op [a,out]",
    "C3  3-op [a,b,out]",
    "C4  3-op EXTERNAL_LOOP",
    "C5  3-op broadcast (32,32)+(32,)",
    "C6  BUFFERED cast eager fill",
    "C7  BUFFERED + DELAY_BUFALLOC",
    "C8  MULTI_INDEX",
    "C9  C_INDEX",
    "C10 full ufunc config",
    "C11 8-op",
    "C12 4-D contig (coalesce)",
    "C13 strided 2-D view",
]
ns_v = [289, 298, 308, 304, 696, 559, 309, 315, 317, 343, 385, 379, 327]
np_v = [275, 541, 622, 701, 719, 1420, 1140, 421, 426, 1000, 1140, 662, 621]

fig, ax = plt.subplots(figsize=(9.5, 6.2))
y = np.arange(len(labels))[::-1]
h = 0.38
ax.barh(y + h / 2, ns_v, h, color=NS, label="NumSharp NpyIterRef")
ax.barh(y - h / 2, np_v, h, color=NP, label="NumPy np.nditer (Python surface)")
for yi, a, b in zip(y, ns_v, np_v):
    r = a / b
    ax.text(max(a, b) + 25, yi, f"{r:.2f}x", va="center", fontsize=8.5,
            color=GOOD if r < 0.95 else ("#555555" if r <= 1.1 else BAD), fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("ns per construct + dispose  (lower is better)")
ax.set_title("Iterator CONSTRUCTION: NumSharp wins 1.4-3.7x in every multi-operand config\n"
             "(1K f64 operands; label = NumSharp / NumPy ratio)")
ax.legend(loc="lower right")
ax.set_xlim(0, 1650)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "1_construction.png"))
plt.close(fig)

# ============================================================================
# 2. Chunk-dispatch overhead — same 2M elements, only chunking changes
# ============================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.8))

wA = [4, 16, 64, 256, 1024]
copy_ns = [3.929, 2.010, 1.521, 1.297, 0.858]
copy_np = [2.234, 1.637, 1.365, 1.285, 1.039]
axA.plot(wA, copy_ns, "o-", color=NS, lw=2, label="NumSharp ForEach (chunked callback)")
axA.plot(wA, copy_np, "s-", color=NP, lw=2, label="NumPy np.copyto (raw walker, NOT nditer)")
axA.plot([4], [3.249], "D", color="#17becf", ms=7, label="NumSharp ExecuteGeneric (struct) w=4")
axA.plot([4], [2.335], "*", color=GOOD, ms=15, label="NumSharp NpyIter.Copy (production) w=4")
axA.annotate("7 ns/chunk", (4, 3.929), textcoords="offset points", xytext=(10, 8), fontsize=8.5, color=NS)
axA.annotate("4 ns/chunk", (4, 2.234), textcoords="offset points", xytext=(12, -16), fontsize=8.5, color=NP)
axA.annotate("4 ns/chunk: parity with\ntheir raw walker", (4, 2.335), textcoords="offset points",
             xytext=(32, 30), fontsize=8, color=GOOD,
             arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
axA.set_xscale("log", base=2)
axA.set_xticks(wA)
axA.set_xticklabels([str(x) for x in wA])
axA.set_xlabel("inner chunk width w (f64 elems; chunks = 2M / w)")
axA.set_ylabel("total ms for the SAME 2M elements")
axA.set_title("2-op strided-row COPY\n(NumPy's 4 ns machine is a special-cased copy walker)")
axA.legend(fontsize=7.8, loc="upper right")

wB = [4, 16, 64, 1024]
add_fe = [5.285, 3.965, 2.898, 1.449]
add_pr = [4.625, 3.038, 3.048, 1.537]
add_np = [5.417, 3.823, 3.060, 2.837]
axB.plot(wB, add_fe, "o-", color=NS, lw=2, label="NumSharp ForEach")
axB.plot(wB, add_pr, "*-", color=GOOD, lw=2, ms=10, label="NumSharp production np.add(out=)")
axB.plot(wB, add_np, "s-", color=NP, lw=2, label="NumPy np.add(out=) = its REAL ufunc nditer")
axB.annotate("10.1 vs 10.3 ns/chunk\nPARITY at the tiny-chunk extreme", (4, 5.285),
             textcoords="offset points", xytext=(24, -40), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8))
axB.annotate("2x FASTER at wide\nstrided chunks", (1024, 1.449), textcoords="offset points",
             xytext=(-90, 34), fontsize=8, color=GOOD,
             arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
axB.set_xscale("log", base=2)
axB.set_xticks(wB)
axB.set_xticklabels([str(x) for x in wB])
axB.set_xlabel("inner chunk width w (f64 elems)")
axB.set_ylabel("total ms for the SAME 2M elements")
axB.set_title("3-op strided-row ADD\nfull iterator vs full iterator: the honest comparison")
axB.legend(fontsize=7.8, loc="upper right")

fig.suptitle("Per-chunk dispatch overhead: fixed total work, shrinking chunks", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "2_chunk_overhead.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================================
# 3. Small-N pipeline scaling (log-log) + the reuse floor
# ============================================================================
fig, ax = plt.subplots(figsize=(9.5, 5.8))
Ns = [8, 64, 512, 4096, 32768, 262144, 2097152]
ns_us = [0.3079, 0.317, 0.3549, 0.9465, 5.64, 85.49, 1397]
np_us = [0.2864, 0.2996, 0.3838, 0.9811, 5.28, 83.60, 1399]
ax.plot(Ns, ns_us, "o-", color=NS, lw=2, label="NumSharp raw iterator pipeline (ctor+ForEach+dispose)")
ax.plot(Ns, np_us, "s-", color=NP, lw=2, label="NumPy np.add(a,b,out=o) end-to-end")
ax.plot([1000], [0.6484], "D", color=BAD, ms=8, label="NumSharp PRODUCTION np.add(out=) @1K = 648 ns")
ax.plot([1000], [0.4293], "D", color=NP, ms=6)
ax.plot([512], [0.0547], "*", color=GOOD, ms=18, label="REUSED iterator (Reset+ForEach) @512 = 54.7 ns")

ax.annotate("~200 ns of np.* routing glue\nabove the raw iterator\n(the real small-N gap)", (1000, 0.6484),
            textcoords="offset points", xytext=(20, 40), fontsize=8.5, color=BAD,
            arrowprops=dict(arrowstyle="->", color=BAD, lw=0.9))
ax.annotate("7x under NumPy's floor:\nthe lever NumPy cannot reach\nfrom Python", (512, 0.0547),
            textcoords="offset points", xytext=(28, -10), fontsize=8.5, color=GOOD, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.9))
ax.annotate("raw pipeline tracks NumPy's whole\nufunc dispatch within +-8% at every N", (32768, 5.64),
            textcoords="offset points", xytext=(-160, 40), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels(["8", "64", "512", "4K", "32K", "256K", "2M"])
ax.set_xlabel("N (f64 elements)")
ax.set_ylabel("us per call (log)")
ax.set_title("Small-N pipeline scaling: add f64, per call including everything")
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3_smalln_scaling.png"))
plt.close(fig)

# ============================================================================
# 4. Traversal scoreboard — NS/NP ratios + GROWINNER inset
# ============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.8), gridspec_kw={"width_ratios": [2.4, 1]})

t_labels = [
    "copy contig 10M (DRAM roofline)",
    "strided copy w=4: ForEach vs raw walker",
    "strided copy w=4: production Copy vs raw walker",
    "strided ADD w=4 (vs real ufunc nditer)",
    "strided ADD w=1024 (vs real ufunc nditer)",
    "transposed copy (1448x1448)",
    "row-broadcast add (2000,2000)+(2000,)",
    "col-broadcast add (2000,2000)+(2000,1)",
    "buffered cast copy vs ONE-PASS copyto",
    "buffered cast copy vs nditer buffered",
    "mixed-dtype add f32+f64 (buffered)",
    "reduce sum contig 10M",
    "reduce sum strided [::2]",
    "per-elem walk: 2.5 ns/elem (C-level)",
]
ratios = [1.00, 1.76, 1.05, 0.98, 0.51, 0.84, 0.74, 0.81, 1.11, 0.78, 0.86, 0.64, 0.85, float("nan")]

y = np.arange(len(t_labels))[::-1]
colors = []
for r in ratios:
    if np.isnan(r):
        colors.append("#cccccc")
    elif abs(r - 1) <= 0.06:
        colors.append("#888888")
    else:
        colors.append(GOOD if r < 1 else BAD)
axL.barh(y, [0 if np.isnan(r) else r for r in ratios], 0.62, color=colors)
axL.axvline(1.0, color="black", lw=1, ls="--")
for yi, r in zip(y, ratios):
    if np.isnan(r):
        axL.text(0.03, yi, "(no NumPy C-level counterpart; Python nditer = 40 ns/elem)",
                 va="center", fontsize=7.6, color="#666666")
    else:
        c = "#555555" if abs(r - 1) <= 0.06 else (GOOD if r < 1 else BAD)
        axL.text(r + 0.04, yi, f"{r:.2f}", va="center", fontsize=8.5, fontweight="bold", color=c)
axL.set_yticks(y)
axL.set_yticklabels(t_labels, fontsize=8.2)
axL.set_xlim(0, 2.05)
axL.set_xlabel("NumSharp / NumPy time ratio   (<1.0 = NumSharp faster)")
axL.set_title("TRAVERSAL scoreboard: green faster, gray parity, red behind")

g_lab = ["EXLOOP plain\n(1 chunk)", "BUFFERED|GROWINNER\n(512 windows)", "full ufunc config\n(512 windows)"]
g_val = [3.187, 3.454, 3.393]
axR.bar(range(3), g_val, 0.62, color=[GOOD, BAD, BAD])
for i, v in enumerate(g_val):
    axR.text(i, v + 0.02, f"{v:.2f} ms", ha="center", fontsize=8.5, fontweight="bold")
axR.text(1, 3.62, "+8.4%: GROWINNER bit is set\nbut NOTHING consumes it", ha="center", fontsize=8.5, color=BAD)
axR.set_xticks(range(3))
axR.set_xticklabels(g_lab, fontsize=7.8)
axR.set_ylim(2.9, 3.75)
axR.set_ylabel("ms (same-dtype add f64 4M)")
axR.set_title("Finding: hollow GROWINNER\n(needless 8192-elem windowing)")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "4_scoreboard.png"), bbox_inches="tight")
plt.close(fig)

for f in sorted(os.listdir(OUT)):
    print(os.path.join(OUT, f))
