# =============================================================================
# npyiter_frontier_bench.py — NumPy side of the frontier bench (same ids as
# npyiter_frontier_bench.cs). Targets the suspected NOT-winning territory:
# axis reductions through the iterator, ALLOCATE outputs, where= masks at
# degenerate run lengths, strided buffered casts, forced-order outputs, 0-d,
# tiny-chunk copyto, multi-input fusion, and the kernel-bound dtype frontier.
#
# Run: python benchmark/poc/npyiter_frontier_bench.py
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
        best = min(best, (time.perf_counter() - t0) * 1000.0 / iters)
    return best


def row(id_, label, ms, note=""):
    if ms >= 1.0:
        val = f"{ms:10.3f} ms"
    elif ms >= 0.001:
        val = f"{ms * 1000:10.2f} us"
    else:
        val = f"{ms * 1e6:10.1f} ns"
    print(f"{id_:<6} {label:<56} {val}  {note}")


print(f"NumPy {np.__version__} NpyIter frontier bench")
print(f"{'id':<6} {'aspect':<56} {'per call':>13}")
print("-" * 95)

# =============================================================================
# R — axis reductions (np.add.reduce IS NumPy's nditer-driven reduction)
# =============================================================================
A = ((np.arange(4_000_000, dtype=np.float64) % 97.0) + 1.0).reshape(2000, 2000)
po = np.empty(2000, dtype=np.float64)

row("R0a", "production np.sum(A, axis=0) f64 (2000,2000)",
    best_ms(lambda: np.sum(A, axis=0), 50, 12, 7))
row("R0b", "production np.sum(A, axis=1) f64 (2000,2000)",
    best_ms(lambda: np.sum(A, axis=1), 50, 12, 7))
row("R1", "np.add.reduce(A, axis=0, out=) [nditer reduction]",
    best_ms(lambda: np.add.reduce(A, axis=0, out=po), 50, 12, 7))
check(abs(po[777] - np.sum(A, axis=0)[777]) < 1e-6, "R1")
row("R2", "np.add.reduce(A, axis=1, out=) [nditer reduction]",
    best_ms(lambda: np.add.reduce(A, axis=1, out=po), 50, 12, 7))
check(abs(po[777] - np.sum(A, axis=1)[777]) < 1e-6, "R2")
del A, po

# =============================================================================
# A — allocating output (ufunc allocates EMPTY through its iterator machinery)
# =============================================================================
M = 4_194_304
a = np.arange(M, dtype=np.float64)
b = np.arange(M, dtype=np.float64) + 1.0
o = np.empty(M, dtype=np.float64)

row("A3", "anchor: np.add(a,b,out=o) f64 4M",
    best_ms(lambda: np.add(a, b, out=o), 25, 8, 7))
row("A1", "np.add(a,b) ALLOCATING f64 4M (empty, not zeros)",
    best_ms(lambda: np.add(a, b), 10, 4, 7))
row("A2", "  (same row — NumPy has one allocating path)",
    best_ms(lambda: np.add(a, b), 10, 4, 7))
del a, b, o

# =============================================================================
# W — where= masked execution
# =============================================================================
a = (np.arange(M, dtype=np.float32) % 977) + 1
b = (np.arange(M, dtype=np.float32) % 31) + 2
o = np.zeros(M, dtype=np.float32) - 1
idx = np.arange(M)
m_all = idx >= 0
m_alt = (idx % 2) == 0
m_blk = (idx % 128) < 64

np.add(a, b, out=o, where=m_all)
t = best_ms(lambda: np.add(a, b, out=o, where=m_all), 25, 8, 7)
check(o[777] == a[777] + b[777], "W1")
row("W1", "np.add(out=, where=ALL-TRUE) f32 4M", t)

o[:] = -1
t = best_ms(lambda: np.add(a, b, out=o, where=m_alt), 5, 2, 5)
check(o[776] == a[776] + b[776] and o[777] == -1, "W2 mask semantics")
row("W2", "np.add(out=, where=ALTERNATING, run=1) f32 4M", t)

t = best_ms(lambda: np.add(a, b, out=o, where=m_blk), 25, 8, 7)
check(o[63] == a[63] + b[63], "W3")
row("W3", "np.add(out=, where=BLOCKY, run=64) f32 4M", t)
del a, b, o, idx, m_all, m_alt, m_blk

# =============================================================================
# B — strided-source cast (NumPy: one-pass strided cast transfer)
# =============================================================================
back = np.arange(M, dtype=np.float32)
sv = back[::2]
d64 = np.empty(M // 2, dtype=np.float64)
t = best_ms(lambda: np.copyto(d64, sv), 25, 8, 7)
check(d64[777] == np.float64(sv[777]), "B1")
row("B1", "cast copy f32[::2]->f64 2M (np.copyto one-pass)", t)
row("B1p", "  (same row — copyto IS the production route)", t)
del back, sv, d64

# =============================================================================
# X — layout frontier
# =============================================================================
n = 1448
aC = ((np.arange(n * n, dtype=np.float64) % 97.0) + 1.0).reshape(n, n)
bC = ((np.arange(n * n, dtype=np.float64) % 31.0) + 2.0).reshape(n, n)
oF = np.empty((n, n), order="F", dtype=np.float64)
t = best_ms(lambda: np.add(aC, bC, out=oF), 12, 4, 7)
check(abs(oF[5, 7] - (aC[5, 7] + bC[5, 7])) < 1e-12, "X1")
row("X1", "np.add C+C -> F-ORDER out (1448,1448) f64", t)
row("X1p", "  (same row — one production path)", t)
del aC, bC, oF

src = np.arange(M, dtype=np.float64)
rev = src[::-1]
dst = np.empty(M, dtype=np.float64)
t = best_ms(lambda: np.copyto(dst, rev), 25, 8, 7)
check(dst[0] == src[M - 1] and dst[M - 1] == src[0], "X2")
row("X2", "copy REVERSED a[::-1] f64 4M -> contig (np.copyto)", t)
del src, rev, dst

# =============================================================================
# O — 0-d scalar ufunc calls
# =============================================================================
s1 = np.array(2.5)
s2 = np.array(1.5)
s3 = np.array(0.0)
np.add(s1, s2, out=s3)
check(float(s3) == 4.0, "O1")
row("O1", "np.add(0-d, 0-d, out=0-d)",
    best_ms(lambda: np.add(s1, s2, out=s3), 200_000, 25_000))
row("O2", "np.add(0-d, 0-d) allocating",
    best_ms(lambda: np.add(s1, s2), 100_000, 12_000))

# =============================================================================
# P — copyto at the tiny-chunk frontier (same as core T2.4, re-anchored)
# =============================================================================
TOTAL = 2_097_152
w = 4
rows_n = TOTAL // w
back = np.arange(rows_n * 2 * w, dtype=np.float64).reshape(rows_n, 2 * w)
sv = back[:, :w]
dst = np.empty((rows_n, w), dtype=np.float64)
t = best_ms(lambda: np.copyto(dst, sv), 12, 5, 7)
check(dst[rows_n - 1, w - 1] == sv[rows_n - 1, w - 1], "P4")
row("P4", "np.copyto strided rows w=4 (524288 chunks)", t, f"{t * 1e6 / rows_n:.0f} ns/chunk")
del back, sv, dst

# =============================================================================
# Y — 7-input sum: best NumPy can do is 6 chained 2-op passes (out= reuse)
# =============================================================================
ins = [(np.arange(M, dtype=np.float64) % (7.0 + i)) + 1.0 for i in range(7)]
acc = np.empty(M, dtype=np.float64)

def chained():
    np.add(ins[0], ins[1], out=acc)
    for i in range(2, 7):
        np.add(acc, ins[i], out=acc)

chained()
expect = sum(x[777] for x in ins)
check(abs(acc[777] - expect) < 1e-9, "Y2")
row("Y2", "chained 6x np.add(out=) sum of 7 arrays f64 4M", best_ms(chained, 25, 8, 7))
del ins, acc

# =============================================================================
# Z — kernel-bound dtype frontier
# =============================================================================
ac = np.arange(M, dtype=np.complex128)
bc = ((np.arange(M, dtype=np.float64) % 7.0) + 1.0).astype(np.complex128)
oc = np.empty(M, dtype=np.complex128)
row("Z1", "np.add complex128 4M (out=)",
    best_ms(lambda: np.add(ac, bc, out=oc), 12, 4, 7))
row("Z2", "np.multiply complex128 4M (out=)",
    best_ms(lambda: np.multiply(ac, bc, out=oc), 12, 4, 7))
del ac, bc, oc

ah = (np.arange(M) % 1000).astype(np.float16)
bh = (np.arange(M) % 31).astype(np.float16)
oh = np.empty(M, dtype=np.float16)
row("Z3", "np.add float16 4M (out=)",
    best_ms(lambda: np.add(ah, bh, out=oh), 12, 4, 7))
del ah, bh, oh

ai = (np.arange(M) % 100).astype(np.int8)
bi = (np.arange(M) % 27).astype(np.int8)
oi = np.empty(M, dtype=np.int8)
row("Z4", "np.add int8 4M (out=)",
    best_ms(lambda: np.add(ai, bi, out=oi), 50, 12, 7))
del ai, bi, oi

print("-" * 95)
print("ALL CORRECTNESS CHECKS PASS" if fails == 0 else f"{fails} CORRECTNESS FAILURES")
