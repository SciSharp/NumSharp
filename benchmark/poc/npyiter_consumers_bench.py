# =============================================================================
# npyiter_consumers_bench.py — NumPy side of the internal-consumers benchmark
# (same ids as npyiter_consumers_bench.cs). Every row exercises a documented
# internal NpyIter consumer (see the .cs header for the source-grounded map),
# including the consumers NumSharp lacks (EI/AT/RA rows = NumPy-only targets).
#
# Run: python benchmark/poc/npyiter_consumers_bench.py
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
    print(f"{id_:<6} {label:<58} {val}  {note}")


M = 4_194_304
print(f"NumPy {np.__version__} — internal NpyIter consumers benchmark")
print(f"{'id':<6} {'aspect':<58} {'per call':>13}")
print("-" * 97)

# UF — execute_ufunc_loop argument matrix
a = (np.arange(M, dtype=np.float64) % 97.0) + 1.0
b = (np.arange(M, dtype=np.float64) % 31.0) + 2.0
o32 = np.empty(M, dtype=np.float32)
ai = np.arange(M, dtype=np.int32)

r1 = np.add(a, b, dtype=np.float32)
check(r1.dtype == np.float32, "UF1")
row("UF1", "np.add(f64, f64, dtype=float32) allocating 4M",
    best_ms(lambda: np.add(a, b, dtype=np.float32), 12, 4, 7))
np.add(a, b, out=o32, casting="same_kind")
row("UF2", "np.add(f64, f64, out=f32) write-cast 4M",
    best_ms(lambda: np.add(a, b, out=o32, casting="same_kind"), 12, 4, 7))
rs = np.sqrt(ai)
check(abs(rs[9] - 3.0) < 1e-12, "UF3")
row("UF3", "np.sqrt(int32) promoting unary 4M (buffered config)",
    best_ms(lambda: np.sqrt(ai), 12, 4, 7))
del o32, ai, r1, rs

# RD — PyUFunc_ReduceWrapper argument matrix
A = ((np.arange(M, dtype=np.float64) % 97.0) + 1.0).reshape(2048, 2048)
af32 = (np.arange(M, dtype=np.float32) % 977) + 1
B = ((np.arange(M, dtype=np.float64) % 53.0) + 1.0).reshape(128, 256, 128)

row("RD1", "np.sum(A) full reduce f64 (2048,2048)",
    best_ms(lambda: np.sum(A), 25, 8, 7))
k = np.sum(A, axis=0, keepdims=True)
check(k.shape == (1, 2048), "RD2")
row("RD2", "np.sum(A, axis=0, keepdims=True)",
    best_ms(lambda: np.sum(A, axis=0, keepdims=True), 50, 12, 7))
s64 = np.sum(af32, dtype=np.float64)
check(s64.dtype == np.float64, "RD3")
row("RD3", "np.sum(f32 4M, dtype=float64) upcast accumulate",
    best_ms(lambda: np.sum(af32, dtype=np.float64), 50, 12, 7))
m1 = np.sum(B, axis=1)
check(m1.shape == (128, 128), "RD4")
row("RD4", "np.sum(B (128,256,128), axis=1) middle axis",
    best_ms(lambda: np.sum(B, axis=1), 50, 12, 7))
row("RD5", "np.amin(A, axis=1)",
    best_ms(lambda: np.amin(A, axis=1), 50, 12, 7))

# NumPy-only argument-surface rows (NumSharp missing)
row("RDt", "np.sum(A, axis=(0,1)) — axis TUPLE [NS missing]",
    best_ms(lambda: np.sum(A, axis=(0, 1)), 25, 8, 7))
mask2d = (np.arange(M).reshape(2048, 2048) % 3) == 0
row("RDw", "np.add.reduce(A, axis=0, where=mask) [NS missing]",
    best_ms(lambda: np.add.reduce(A, axis=0, where=mask2d), 25, 8, 7))
row("RDi", "np.sum(A, initial=5.0) [NS missing]",
    best_ms(lambda: np.sum(A, initial=5.0), 25, 8, 7))
del af32, B, k, s64, m1, mask2d

# AC — PyUFunc_Accumulate
row("AC1", "np.cumsum(a) flat f64 4M",
    best_ms(lambda: np.cumsum(a), 12, 4, 7))
row("AC2", "np.cumsum(A, axis=0) (2048,2048)",
    best_ms(lambda: np.cumsum(A, axis=0), 12, 4, 7))
row("AC3", "np.cumsum(A, axis=1)",
    best_ms(lambda: np.cumsum(A, axis=1), 12, 4, 7))

# WH — PyArray_Where
c = (np.arange(M) % 2) == 0
w = np.where(c, a, b)
check(w[0] == a[0] and w[1] == b[1], "WH1")
row("WH1", "np.where(c, x, y) f64 4M same-shape",
    best_ms(lambda: np.where(c, a, b), 12, 4, 7))
row("WH2", "np.where(c, x, 0.0) scalar branch",
    best_ms(lambda: np.where(c, a, 0.0), 12, 4, 7))
A2 = a.reshape(2048, 2048)
c2d = ((np.arange(M) % 3) == 0).reshape(2048, 2048)
rowv = np.arange(2048, dtype=np.float64)
wb = np.where(c2d, rowv, A2)
check(wb.shape == (2048, 2048), "WH3")
row("WH3", "np.where(c2d, row(2048,), y2d) broadcasting",
    best_ms(lambda: np.where(c2d, rowv, A2), 12, 4, 7))
del w, wb, c2d, rowv

# BM — boolean subscript + nonzero family
mask = (np.arange(M) % 2) == 0
sel = a[mask]
check(sel.size == M // 2 and sel[1] == a[2], "BM1")
row("BM1", "a[mask] boolean READ f64 4M (50% true)",
    best_ms(lambda: a[mask], 12, 4, 7))
aw = a.copy()
aw[mask] = 5.0
check(aw[0] == 5.0 and aw[1] == a[1], "BM2")


def bm2():
    aw[mask] = 5.0


row("BM2", "a[mask] = 5.0 boolean ASSIGN f64 4M",
    best_ms(bm2, 12, 4, 7))
cnt = np.count_nonzero(a)
check(cnt == M, "BM3")
row("BM3", "np.count_nonzero(f64 4M) [non-bool dtype]",
    best_ms(lambda: np.count_nonzero(a), 50, 12, 7))
aw2 = np.argwhere(mask)
check(aw2.shape[0] == M // 2, "BM4")
row("BM4", "np.argwhere(bool 4M, 50% true) -> indices",
    best_ms(lambda: np.argwhere(mask), 12, 4, 7))
del sel, aw, aw2

# FX — fancy indexing (MapIter)
NI = 1_048_576
idx = ((np.arange(NI, dtype=np.int64) * 2654435761) % M).astype(np.int32)
b1m = np.arange(NI, dtype=np.float64)
g = a[idx]
check(g.size == NI and g[3] == a[idx[3]], "FX1")
row("FX1", "a[idx] fancy GATHER 1M random of 4M f64",
    best_ms(lambda: a[idx], 12, 4, 7))
aw = a.copy()


def fx2():
    aw[idx] = b1m


fx2()
row("FX2", "a[idx] = b fancy SCATTER 1M f64",
    best_ms(fx2, 12, 4, 7))
del g, aw

# RV — PyArray_CopyAsFlat consumers
n = 2048
At = A.T
rv = np.ravel(At)
check(rv[1] == A[1, 0], "RV1")
row("RV1", "np.ravel(A.T) forced copy (2048,2048) f64",
    best_ms(lambda: np.ravel(At), 12, 4, 7))
rf = np.ravel(A, order="F")
check(rf[1] == A[1, 0], "RV2")
row("RV2", "np.ravel(A, order='F') strided copy",
    best_ms(lambda: np.ravel(A, order="F"), 12, 4, 7))
row("RV3", "A.flatten() contiguous copy 4M",
    best_ms(lambda: A.flatten(), 12, 4, 7))
c32 = A.astype(np.float32)
check(abs(c32[5, 7] - np.float32(A[5, 7])) < 1e-3, "RV4")
row("RV4", "A.astype(float32) allocating cast 4M",
    best_ms(lambda: A.astype(np.float32), 12, 4, 7))
del At, rv, rf, c32

# MI — compiled_base consumers
flat = ((np.arange(NI, dtype=np.int64) * 2654435761) % (2048 * 2048)).astype(np.int64)
dims = (2048, 2048)
coords = np.unravel_index(flat, dims)
check(coords[0][7] * 2048 + coords[1][7] == flat[7], "MI1")
row("MI1", "np.unravel_index(1M flat, (2048,2048))",
    best_ms(lambda: np.unravel_index(flat, dims), 12, 4, 7))
packed = np.ravel_multi_index(coords, dims)
check(packed[7] == flat[7], "MI2")
row("MI2", "np.ravel_multi_index((i,j), (2048,2048)) 1M",
    best_ms(lambda: np.ravel_multi_index(coords, dims), 12, 4, 7))
del flat, coords, packed

# EI / AT / RA — consumers NumSharp lacks: NumPy-only target numbers
v = np.arange(2048, dtype=np.float64)
row("EI1", "np.einsum('i,i->', a, b) 4M dot [NS missing]",
    best_ms(lambda: np.einsum("i,i->", a, b), 12, 4, 7))
row("EI2", "np.einsum('ij,j->i', A, v) [NS missing]",
    best_ms(lambda: np.einsum("ij,j->i", A, v), 25, 8, 7))

acc = a.copy()


def at1():
    np.add.at(acc, idx, b1m)


at1()
row("AT1", "np.add.at(a, idx, b) 1M scatter-add [NS missing]",
    best_ms(at1, 3, 1, 5))

starts = np.arange(0, M, 64, dtype=np.int64)
row("RA1", "np.add.reduceat(a, starts every 64) [NS missing]",
    best_ms(lambda: np.add.reduceat(a, starts), 12, 4, 7))

print("-" * 97)
print("ALL CORRECTNESS CHECKS PASS" if fails == 0 else f"{fails} CORRECTNESS FAILURES")
