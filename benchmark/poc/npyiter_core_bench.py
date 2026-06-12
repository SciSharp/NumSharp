# =============================================================================
# npyiter_core_bench.py — NumPy side of the NpyIter core benchmark.
# Same aspect ids as npyiter_core_bench.cs (run both, merge by id).
#
# Two honest lenses per section:
#   * np.nditer rows measure NumPy's NpyIter exposed through Python — the same
#     user-facing surface NumSharp's NpyIterRef provides. Construction rows pay
#     one Python object allocation; traversal rows pay per-chunk Python costs
#     where chunk counts are large (annotated).
#   * C-level rows (np.copyto / np.add(out=) / np.sum) are NumPy's OWN internal
#     consumers of its iterator — what the machinery achieves with zero Python
#     in the loop. These are the real targets.
#
# Run: python benchmark/poc/npyiter_core_bench.py
# =============================================================================
import time
import numpy as np

fails = 0


def check(ok, what):
    global fails
    if not ok:
        fails += 1
        print(f"  CORRECTNESS FAIL: {what}")


def best_ms(body, iters, warm, rounds=5):
    for _ in range(warm):
        body()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            body()
        dt = (time.perf_counter() - t0) * 1000.0 / iters
        best = min(best, dt)
    return best


def row(id_, label, ms, note=""):
    if ms >= 1.0:
        val = f"{ms:10.3f} ms"
    elif ms >= 0.001:
        val = f"{ms * 1000:10.2f} us"
    else:
        val = f"{ms * 1e6:10.1f} ns"
    print(f"{id_:<6} {label:<52} {val}  {note}")


print(f"NumPy {np.__version__} NpyIter core bench")
print(f"{'id':<6} {'aspect':<52} {'per call':>13}")
print("-" * 90)

# =============================================================================
# C — CONSTRUCTION (np.nditer object construct only)
# =============================================================================
a = np.arange(1000, dtype=np.float64)
b = np.arange(1000, dtype=np.float64) + 1.0
o = np.empty(1000, dtype=np.float64)
a32 = np.arange(1000, dtype=np.float32)
o64 = np.empty(1000, dtype=np.float64)
g32 = np.arange(1024, dtype=np.float64).reshape(32, 32)
row32 = np.arange(32, dtype=np.float64)
og32 = np.empty((32, 32), dtype=np.float64)
a4d = np.arange(1024, dtype=np.float64).reshape(8, 8, 4, 4)
o4d = np.empty((8, 8, 4, 4), dtype=np.float64)
back2d = np.arange(64 * 8, dtype=np.float64).reshape(64, 8)
sview = back2d[:, :4]
sdst = np.empty((64, 4), dtype=np.float64)

RO_WO = [["readonly"], ["writeonly"]]
RO_RO_WO = [["readonly"], ["readonly"], ["writeonly"]]
RO_RO_WO_ELW = [["readonly", "overlap_assume_elementwise"],
                ["readonly", "overlap_assume_elementwise"],
                ["writeonly", "overlap_assume_elementwise"]]
F64x2 = ["float64", "float64"]

row("C1", "ctor 1-op contig 1K f64, no flags",
    best_ms(lambda: np.nditer(a), 200_000, 25_000))
row("C2", "ctor 2-op [a,out] contig 1K",
    best_ms(lambda: np.nditer((a, o), op_flags=RO_WO), 200_000, 25_000))
row("C3", "ctor 3-op [a,b,out] contig 1K",
    best_ms(lambda: np.nditer((a, b, o), op_flags=RO_RO_WO), 200_000, 25_000))
row("C4", "ctor 3-op EXTERNAL_LOOP",
    best_ms(lambda: np.nditer((a, b, o), flags=["external_loop"], op_flags=RO_RO_WO), 200_000, 25_000))
row("C5", "ctor 3-op broadcast (32,32)+(32,)->(32,32)",
    best_ms(lambda: np.nditer((g32, row32, og32), flags=["external_loop"], op_flags=RO_RO_WO), 100_000, 12_000))
row("C6", "ctor 2-op BUFFERED cast f32->f64 eager (EXL|GROW)",
    best_ms(lambda: np.nditer((a32, o64), flags=["buffered", "external_loop", "growinner"],
                              op_flags=RO_WO, op_dtypes=F64x2, casting="safe"), 50_000, 6_000))
row("C7", "ctor C6 + DELAY_BUFALLOC (defer alloc+fill)",
    best_ms(lambda: np.nditer((a32, o64), flags=["buffered", "external_loop", "growinner", "delay_bufalloc"],
                              op_flags=RO_WO, op_dtypes=F64x2, casting="safe"), 100_000, 12_000))
row("C8", "ctor 1-op MULTI_INDEX (32,32)",
    best_ms(lambda: np.nditer(g32, flags=["multi_index"]), 200_000, 25_000))
row("C9", "ctor 1-op C_INDEX (32,32)",
    best_ms(lambda: np.nditer(g32, flags=["c_index"]), 200_000, 25_000))
row("C10", "ctor 3-op ufunc config (EXL|BUF|GROW|DELAY|CIO|ZS)",
    best_ms(lambda: np.nditer((a, b, o),
                              flags=["external_loop", "buffered", "growinner", "delay_bufalloc",
                                     "copy_if_overlap", "zerosize_ok"],
                              op_flags=RO_RO_WO_ELW), 100_000, 12_000))
ops8 = (a, a, a, a, a, a, a, o)
opf8 = [["readonly"]] * 7 + [["writeonly"]]
row("C11", "ctor 8-op contig 1K",
    best_ms(lambda: np.nditer(ops8, flags=["external_loop"], op_flags=opf8), 100_000, 12_000))
row("C12", "ctor 2-op 4-D contig (8,8,4,4) [coalesce 4 axes]",
    best_ms(lambda: np.nditer((a4d, o4d), flags=["external_loop"], op_flags=RO_WO), 100_000, 12_000))
row("C13", "ctor 2-op strided 2-D view (64,4) of (64,8)",
    best_ms(lambda: np.nditer((sview, sdst), flags=["external_loop"], op_flags=RO_WO), 100_000, 12_000))

np.add(a, b, out=o)
row("H0", "anchor np.add(a,b,out=o) 1K f64 e2e",
    best_ms(lambda: np.add(a, b, out=o), 200_000, 25_000))

# =============================================================================
# T — TRAVERSAL (C-level consumers of NumPy's iterator)
# =============================================================================

# T1: contiguous copy 10M f64
N = 10_000_000
src = np.arange(N, dtype=np.float64)
dst = np.empty(N, dtype=np.float64)
t = best_ms(lambda: np.copyto(dst, src), 25, 8, 7)
check(dst[N - 3] == src[N - 3], "T1 copy")
row("T1", "copy contig f64 10M (np.copyto)", t, f"{80.0 / t:.0f}+{80.0 / t:.0f} GB/s rw")
del src, dst

# T2: strided-row copy, total 2M f64, width sweep
TOTAL = 2_097_152
for w in (4, 16, 64, 256, 1024):
    rows_n = TOTAL // w
    back = np.arange(rows_n * 2 * w, dtype=np.float64).reshape(rows_n, 2 * w)
    sv = back[:, :w]
    d = np.empty((rows_n, w), dtype=np.float64)
    t = best_ms(lambda: np.copyto(d, sv), 12 if w <= 16 else 25, 5, 7)
    check(d[rows_n - 1, w - 1] == sv[rows_n - 1, w - 1], f"T2 w={w}")
    row(f"T2.{w}", f"copy strided rows f64 2M total, inner w={w} ({rows_n} chunks)", t,
        f"{t * 1e6 / rows_n:.0f} ns/chunk")

    if w == 4:
        # T2x: the SAME traversal through Python-level nditer (chunk loop in
        # Python) — what a Python user pays for iterator-driven custom code.
        def nditer_copy():
            with np.nditer([sv, d], flags=["external_loop"],
                           op_flags=[["readonly"], ["writeonly"]], order="K") as it:
                for x, y in it:
                    y[...] = x
        t = best_ms(nditer_copy, 1, 1, 3)
        row("T2x", "  same w=4 via Python nditer chunk loop (context)", t,
            f"{t * 1e6 / rows_n:.0f} ns/chunk")
    del back, sv, d

# T3: transposed copy
n = 1448
A = np.arange(n * n, dtype=np.float64).reshape(n, n)
At = A.T
D = np.empty((n, n), dtype=np.float64)
t = best_ms(lambda: np.copyto(D, At), 12, 5, 7)
check(D[5, 7] == A[7, 5], "T3 transpose copy")
row("T3", "copy transposed (1448,1448) f64 -> contig", t)
del A, At, D

# T4: broadcast add f32 (2000,2000)
a2k = (np.arange(4_000_000, dtype=np.float32) % 977).reshape(2000, 2000)
rowv = np.arange(2000, dtype=np.float32)
colv = np.arange(2000, dtype=np.float32).reshape(2000, 1)
o2k = np.empty((2000, 2000), dtype=np.float32)
t = best_ms(lambda: np.add(a2k, rowv, out=o2k), 25, 8, 7)
check(o2k[3, 17] == a2k[3, 17] + rowv[17], "T4r row bcast")
row("T4r", "add row-bcast (2000,2000)+(2000,) f32", t)
t = best_ms(lambda: np.add(a2k, colv, out=o2k), 25, 8, 7)
check(o2k[3, 17] == a2k[3, 17] + colv[3, 0], "T4c col bcast")
row("T4c", "add col-bcast (2000,2000)+(2000,1) f32", t)
del a2k, rowv, colv, o2k

# T5: cast copy f32 -> f64 4M (np.copyto = iterator + cast transfer fn)
M = 4_194_304
s32 = np.arange(M, dtype=np.float32)
d64 = np.empty(M, dtype=np.float64)
t = best_ms(lambda: np.copyto(d64, s32), 25, 8, 7)
check(d64[123_456] == np.float64(s32[123_456]), "T5 cast copy")
row("T5", "cast copy f32->f64 4M (np.copyto)", t)

# T5i: the same cast THROUGH Python nditer with buffering (NumPy's buffered
#      window machinery exposed; few chunks so Python overhead is small)
def nditer_cast_copy():
    with np.nditer([s32, d64], flags=["buffered", "external_loop", "growinner"],
                   op_flags=[["readonly"], ["writeonly"]],
                   op_dtypes=["float64", "float64"], casting="safe") as it:
        for x, y in it:
            y[...] = x
t = best_ms(nditer_cast_copy, 12, 4, 7)
row("T5i", "  same via nditer buffered+growinner (context)", t)
del s32, d64

# T6: mixed add f32+f64 -> f64 4M (ufunc buffering, C level)
a32m = (np.arange(M, dtype=np.float32) % 977) + 1
b64m = (np.arange(M, dtype=np.float64) % 31.0) + 2.0
o64m = np.empty(M, dtype=np.float64)
t = best_ms(lambda: np.add(a32m, b64m, out=o64m), 25, 8, 7)
check(abs(o64m[777] - (np.float64(a32m[777]) + b64m[777])) < 1e-12, "T6 mixed add")
row("T6", "buffered mixed add f32+f64->f64 4M (np.add)", t)
del a32m, b64m, o64m

# T7: per-element protocol on (1000,1000) f64 through Python nditer.
#     The Python for-loop dominates here; rows are CONTEXT (what Python users
#     pay), the C# numbers are the machinery itself.
g = np.arange(1_000_000, dtype=np.float64).reshape(1000, 1000)
NE = 1_000_000

def walk_base():
    it = np.nditer(g)
    for _ in it:
        pass
t = best_ms(walk_base, 3, 1, 5)
row("T7a", "per-element walk (1000,1000) f64 [python nditer]", t, f"{t * 1e6 / NE:.1f} ns/elem")

def walk_cindex():
    it = np.nditer(g, flags=["c_index"])
    for _ in it:
        pass
t = best_ms(walk_cindex, 3, 1, 5)
row("T7b", "  + C_INDEX", t, f"{t * 1e6 / NE:.1f} ns/elem")

def walk_multi():
    it = np.nditer(g, flags=["multi_index"])
    for _ in it:
        pass
t = best_ms(walk_multi, 3, 1, 5)
row("T7c", "  + MULTI_INDEX", t, f"{t * 1e6 / NE:.1f} ns/elem")
del g

# T8: reduce sum (np.sum = NumPy's own pairwise reduction machinery)
r10 = (np.arange(10_000_000, dtype=np.float64) % 97.0) + 1.0
t = best_ms(lambda: np.sum(r10), 25, 8, 7)
row("T8", "reduce sum f64 10M contig (np.sum)", t)
backr = (np.arange(2_000_000, dtype=np.float64) % 53.0) + 1.0
svr = backr[::2]
t = best_ms(lambda: np.sum(svr), 100, 25, 7)
row("T8s", "reduce sum f64 1M strided a[::2] (np.sum)", t)
del r10, backr, svr

# =============================================================================
# H — SMALL-N PIPELINE SCALING: np.add(a, b, out=o) per call
# =============================================================================
for n_ in (8, 64, 512, 4096, 32_768, 262_144, 2_097_152):
    ah = np.arange(n_, dtype=np.float64)
    bh = np.arange(n_, dtype=np.float64) + 1.0
    oh = np.empty(n_, dtype=np.float64)
    iters = 100_000 if n_ <= 4096 else (2_000 if n_ <= 262_144 else 100)
    t = best_ms(lambda: np.add(ah, bh, out=oh), iters, iters // 8)
    check(oh[n_ - 1] == ah[n_ - 1] + bh[n_ - 1], f"H{n_}")
    row(f"H{n_}", f"np.add(a,b,out=o) f64 N={n_}", t)
    del ah, bh, oh

print("-" * 90)
print("ALL CORRECTNESS CHECKS PASS" if fails == 0 else f"{fails} CORRECTNESS FAILURES")
