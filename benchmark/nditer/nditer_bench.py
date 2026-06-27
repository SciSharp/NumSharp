# =============================================================================
# nditer_bench.py — NumPy side of THE canonical NDIter benchmark (identical
# ids to nditer_bench.cs). Section-addressable via the NPYITER_SECTION env var
# so the orchestrator (nditer_sheet.py) can run each category in its own
# process. Emits machine-readable "id<TAB>milliseconds" rows on stdout.
#
# Run a section:  NPYITER_SECTION=elementwise python nditer_bench.py
# Run everything: python nditer_bench.py
# =============================================================================
import os
import sys
import time
import numpy as np

SECTION = os.environ.get("NPYITER_SECTION", "all").strip().lower()


def want(s):
    return SECTION in ("all", s)


fails = 0


def check(ok, what):
    global fails
    if not ok:
        fails += 1
        print(f"  CORRECTNESS FAIL: {what}", file=sys.stderr)


def best_ms(body, iters, warm, rounds):
    for _ in range(warm):
        body()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            body()
        best = min(best, (time.perf_counter() - t0) * 1000.0 / iters)
    return best


def row(id_, ms):
    print(f"{id_}\t{ms!r}")


def pick(n):
    if n <= 1:
        return 200_000, 20_000, 5
    if n <= 1_000:
        return 80_000, 10_000, 5
    if n <= 100_000:
        return 2_500, 400, 4
    if n <= 1_000_000:
        return 120, 30, 3
    return 30, 8, 3


def grid(n):
    return {1: (1, 1), 1_000: (25, 40), 100_000: (250, 400),
            1_000_000: (1_000, 1_000), 10_000_000: (2_500, 4_000)}[n]


SIZES = [("1", 1), ("1K", 1_000), ("100K", 100_000), ("1M", 1_000_000), ("10M", 10_000_000)]
RO, WO = ["readonly"], ["writeonly"]
want_ops = any(want(s) for s in ("elementwise", "reductions", "selection", "copycast", "indexmath", "dtypes", "dividends"))

print(f"[nditer_bench.py] section={SECTION} numpy={np.__version__}", file=sys.stderr)

# =====================================================================
# OPERATIONS x SIZE
# =====================================================================
if want_ops:
    for tag, n in SIZES:
        iters, warm, rounds = pick(n)
        R, C = grid(n)
        a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        b = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
        o = np.empty(n, dtype=np.float64)

        if want("elementwise"):
            b1 = np.array([3.0], dtype=np.float64)
            a2 = (np.arange(2 * n, dtype=np.float64) % 53.0) + 1.0
            b2 = (np.arange(2 * n, dtype=np.float64) % 17.0) + 1.0
            sa, sb = a2[::2], b2[::2]
            so = np.empty(n, dtype=np.float64)
            a32 = (np.arange(n, dtype=np.float32) % 977) + 1
            o64 = np.empty(n, dtype=np.float64)
            rev = a[::-1]
            dstRev = np.empty(n, dtype=np.float64)
            af32 = (np.arange(n, dtype=np.float32) % 977) + 1
            np.add(a, b, out=o)
            check(o[n - 1] == a[n - 1] + b[n - 1], f"add@{tag}")
            row(f"add@{tag}", best_ms(lambda: np.add(a, b, out=o), iters, warm, rounds))
            row(f"sqrt@{tag}", best_ms(lambda: np.sqrt(a, out=o), iters, warm, rounds))
            row(f"copy@{tag}", best_ms(lambda: np.positive(a, out=o), iters, warm, rounds))
            np.add(sa, sb, out=so)
            check(so[n - 1] == sa[n - 1] + sb[n - 1], f"sadd@{tag}")
            row(f"sadd@{tag}", best_ms(lambda: np.add(sa, sb, out=so), iters, warm, rounds))
            row(f"bcast@{tag}", best_ms(lambda: np.add(a, b1, out=o), iters, warm, rounds))
            row(f"frev@{tag}", best_ms(lambda: np.positive(rev, out=dstRev), iters, warm, rounds))
            row(f"castbuf@{tag}", best_ms(lambda: np.positive(a32, out=o64), iters, warm, rounds))
            row(f"mixbuf@{tag}", best_ms(lambda: np.add(af32, b, out=o64), iters, warm, rounds))

        if want("reductions"):
            af32 = (np.arange(n, dtype=np.float32) % 977) + 1
            A = ((np.arange(n, dtype=np.float64) % 97.0) + 1.0).reshape(R, C)
            all_false = np.arange(n) == -1
            early_hit = np.arange(n) == min(1000, n - 1)
            row(f"psum@{tag}", best_ms(lambda: np.sum(a), iters, warm, rounds))
            row(f"sumax0@{tag}", best_ms(lambda: np.sum(A, axis=0), iters, warm, rounds))
            row(f"sumax1@{tag}", best_ms(lambda: np.sum(A, axis=1), iters, warm, rounds))
            row(f"sumdt@{tag}", best_ms(lambda: np.sum(af32, dtype=np.float64), iters, warm, rounds))
            row(f"amin@{tag}", best_ms(lambda: np.amin(A, axis=1), iters, warm, rounds))
            row(f"cumsum@{tag}", best_ms(lambda: np.cumsum(a), iters, warm, rounds))
            check(not bool(np.any(all_false)), f"anyff@{tag}")
            row(f"anyff@{tag}", best_ms(lambda: np.any(all_false), iters, warm, rounds))
            check(bool(np.any(early_hit)), f"anyeh@{tag}")
            row(f"anyeh@{tag}", best_ms(lambda: np.any(early_hit), iters, warm, rounds))

        if want("selection"):
            mask = (np.arange(n) % 2) == 0
            aMaskDst = a.copy()
            cond = (np.arange(n) % 2) == 0
            idx = ((np.arange(n, dtype=np.int64) * 2654435761) % n).astype(np.int32)
            idxVals = np.arange(n, dtype=np.float64)
            aScatter = a.copy()
            row(f"where@{tag}", best_ms(lambda: np.where(cond, a, b), iters, warm, rounds))
            row(f"bread@{tag}", best_ms(lambda: a[mask], iters, warm, rounds))

            def bassign():
                aMaskDst[mask] = 5.0
            row(f"bassign@{tag}", best_ms(bassign, iters, warm, rounds))
            row(f"cnz@{tag}", best_ms(lambda: np.count_nonzero(a), iters, warm, rounds))
            row(f"argw@{tag}", best_ms(lambda: np.argwhere(mask), iters, warm, rounds))
            row(f"gather@{tag}", best_ms(lambda: a[idx], iters, warm, rounds))

            def scatter():
                aScatter[idx] = idxVals
            row(f"scatter@{tag}", best_ms(scatter, iters, warm, rounds))

        if want("copycast"):
            A = ((np.arange(n, dtype=np.float64) % 97.0) + 1.0).reshape(R, C)
            At = A.T
            row(f"flatten@{tag}", best_ms(lambda: A.flatten(), iters, warm, rounds))
            row(f"astype@{tag}", best_ms(lambda: A.astype(np.float32), iters, warm, rounds))
            row(f"ravelT@{tag}", best_ms(lambda: np.ravel(At), iters, warm, rounds))
            ipa = a.copy()

            def inplace():
                np.add(ipa, b, out=ipa)
            row(f"inplace@{tag}", best_ms(inplace, iters, warm, rounds))
            ob = np.empty(n, dtype=np.bool_)
            row(f"lessbool@{tag}", best_ms(lambda: np.less(a, b, out=ob), iters, warm, rounds))

        if want("indexmath"):
            flat = ((np.arange(n, dtype=np.int64) * 2654435761) % (R * C)).astype(np.int64)
            dims = (R, C)
            coords = np.unravel_index(flat, dims)
            row(f"unravel@{tag}", best_ms(lambda: np.unravel_index(flat, dims), iters, warm, rounds))
            packed = (coords[0], coords[1])
            row(f"ravelmi@{tag}", best_ms(lambda: np.ravel_multi_index(packed, dims), iters, warm, rounds))

        if want("dtypes"):
            ac = np.arange(n).astype(np.complex128)
            bc = (np.arange(n, dtype=np.float64) % 7.0 + 1.0).astype(np.complex128)
            oc = np.empty(n, dtype=np.complex128)
            ah = (np.arange(n) % 1000).astype(np.float16)
            bh = (np.arange(n) % 31).astype(np.float16)
            oh = np.empty(n, dtype=np.float16)
            ai8 = (np.arange(n) % 100).astype(np.int8)
            bi8 = (np.arange(n) % 27).astype(np.int8)
            oi8 = np.empty(n, dtype=np.int8)
            row(f"cplx@{tag}", best_ms(lambda: np.add(ac, bc, out=oc), iters, warm, rounds))
            row(f"f16@{tag}", best_ms(lambda: np.add(ah, bh, out=oh), iters, warm, rounds))
            row(f"i8@{tag}", best_ms(lambda: np.add(ai8, bi8, out=oi8), iters, warm, rounds))

        if want("dividends"):
            ins = [(np.arange(n, dtype=np.float64) % (7.0 + i)) + 1.0 for i in range(7)]
            acc = np.empty(n, dtype=np.float64)

            def fuse7():
                np.add(ins[0], ins[1], out=acc)
                for i in range(2, 7):
                    np.add(acc, ins[i], out=acc)
            row(f"fuse7@{tag}", best_ms(fuse7, iters, warm, rounds))
            row(f"reuse@{tag}", best_ms(lambda: np.add(a, b, out=o), iters, warm, rounds))
            if n >= 8:
                src = (np.arange(n, dtype=np.float64) % 6.283185) - 3.1415926
                dst = np.empty(n, dtype=np.float64)
                row(f"par8@{tag}", best_ms(lambda: np.sin(src, out=dst), max(10, iters // 20), 4, rounds))

# =====================================================================
# CONSTRUCTION — np.nditer build across flag configs (size-invariant, 1K)
# =====================================================================
if want("construction"):
    a = np.arange(1000, dtype=np.float64)
    b = np.arange(1000, dtype=np.float64) + 1.0
    o = np.empty(1000, dtype=np.float64)
    a32 = np.arange(1000, dtype=np.float32)
    o64 = np.empty(1000, dtype=np.float64)
    g32 = np.arange(1024, dtype=np.float64).reshape(32, 32)
    a4d = np.arange(1024, dtype=np.float64).reshape(8, 8, 4, 4)
    o4d = np.empty((8, 8, 4, 4), dtype=np.float64)
    a8d = np.arange(65536, dtype=np.float64).reshape(4, 4, 4, 4, 4, 4, 4, 4)
    o8d = np.empty((4, 4, 4, 4, 4, 4, 4, 4), dtype=np.float64)
    back2d = np.arange(64 * 8, dtype=np.float64).reshape(64, 8)
    sview = back2d[:, :4]
    sdst = np.empty((64, 4), dtype=np.float64)

    row("ctor.1op", best_ms(lambda: np.nditer(a), 400_000, 50_000, 5))
    row("ctor.3op_exl", best_ms(lambda: np.nditer((a, b, o), flags=["external_loop"],
        op_flags=[RO, RO, WO]), 400_000, 50_000, 5))
    row("ctor.ufunc", best_ms(lambda: np.nditer((a, b, o),
        flags=["external_loop", "buffered", "growinner", "delay_bufalloc", "copy_if_overlap", "zerosize_ok"],
        op_flags=[RO, RO, WO]), 200_000, 25_000, 5))
    row("ctor.bufcast", best_ms(lambda: np.nditer((a32, o64),
        flags=["external_loop", "buffered", "growinner"], op_flags=[RO, WO],
        op_dtypes=["float64", "float64"], casting="safe"), 100_000, 12_000, 5))
    row("ctor.multiindex", best_ms(lambda: np.nditer(g32, flags=["multi_index"]), 400_000, 50_000, 5))
    row("ctor.8op", best_ms(lambda: np.nditer((a, a, a, a, a, a, a, o), flags=["external_loop"],
        op_flags=[RO, RO, RO, RO, RO, RO, RO, WO]), 200_000, 25_000, 5))
    row("ctor.4d", best_ms(lambda: np.nditer((a4d, o4d), flags=["external_loop"], op_flags=[RO, WO]), 200_000, 25_000, 5))
    row("ctor.8d", best_ms(lambda: np.nditer((a8d, o8d), flags=["external_loop"], op_flags=[RO, WO]), 200_000, 25_000, 5))
    row("ctor.strided2d", best_ms(lambda: np.nditer((sview, sdst), flags=["external_loop"], op_flags=[RO, WO]), 200_000, 25_000, 5))

# =====================================================================
# CHUNKWIDTH — strided rows, honest comparator np.positive (real ufunc)
# =====================================================================
if want("chunkwidth"):
    TOTAL = 2_097_152
    for w in (4, 16, 64, 256, 1024):
        rows = TOTAL // w
        back = np.arange(rows * 2 * w, dtype=np.float64).reshape(rows, 2 * w)
        sv = back[:, :w]
        dst = np.empty((rows, w), dtype=np.float64)
        t = best_ms(lambda: np.positive(sv, out=dst), 12 if w <= 16 else 25, 5, 7)
        check(dst[rows - 1, w - 1] == sv[rows - 1, w - 1], f"cw{w}")
        row(f"cw.{w}", t)

# =====================================================================
# PATHOLOGY — regression canaries
# =====================================================================
if want("pathology"):
    a8k = (np.arange(8192, dtype=np.float64) % 97.0) + 1.0
    bc = np.broadcast_to(a8k, (1024, 8192))
    check(abs(float(np.sum(bc)) - 1024.0 * float(np.sum(a8k))) / (1024.0 * float(np.sum(a8k))) < 1e-9, "path.bcast_reduce")
    row("path.bcast_reduce", best_ms(lambda: np.sum(bc), 25, 8, 7))

    M = 4_194_304
    a = np.arange(M, dtype=np.float64)
    b = np.arange(M, dtype=np.float64) + 1.0
    row("path.allocate", best_ms(lambda: np.add(a, b), 10, 4, 7))

    x = (np.arange(M, dtype=np.float64) % 53.0) + 1.0
    xs, xd = x[:-1], x[1:]
    np.add(xs, xs, out=xd)
    row("path.overlap_copy", best_ms(lambda: np.add(xs, xs, out=xd), 12, 4, 7))

    nn = 1448
    aC = ((np.arange(nn * nn, dtype=np.float64) % 97.0) + 1.0).reshape(nn, nn)
    bC = ((np.arange(nn * nn, dtype=np.float64) % 31.0) + 2.0).reshape(nn, nn)
    oF = np.empty((nn, nn), dtype=np.float64).T
    np.add(aC, bC, out=oF)
    check(abs(oF[5, 7] - (aC[5, 7] + bC[5, 7])) < 1e-9, "path.forder_out")
    row("path.forder_out", best_ms(lambda: np.add(aC, bC, out=oF), 12, 4, 7))

    s1 = np.float64(2.5)
    s2 = np.float64(1.5)
    s3 = np.empty((), dtype=np.float64)
    row("path.zerodim", best_ms(lambda: np.add(s1, s2, out=s3), 200_000, 25_000, 5))

print(f"[ok] correctness {'pass' if fails == 0 else str(fails) + ' FAIL'}", file=sys.stderr)
print(f"[section-done] {SECTION}", file=sys.stderr)
