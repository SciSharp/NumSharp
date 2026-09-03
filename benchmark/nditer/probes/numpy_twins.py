"""numpy_twins.py — the NumPy 2.4.2 side of the NDIter probes (same shapes, same timing policy).

    python benchmark/nditer/probes/numpy_twins.py fixed     # twin of fixed_cost_probe.cs
    python benchmark/nditer/probes/numpy_twins.py ab        # twin of ab_ops_probe.cs (id<TAB>ns rows)
    python benchmark/nditer/probes/numpy_twins.py angles    # twin of angles_probe.cs
    python benchmark/nditer/probes/numpy_twins.py narrow [ABCDE]   # twin of narrow_probe.cs (2-D block kernel)
    python benchmark/nditer/probes/numpy_twins.py fancy_where [ABC] # twin of fancy_where_probe.cs (fancy index, where= runs, masked narrow rows; its section D is C#-only)
    python benchmark/nditer/probes/numpy_twins.py neighbours [EFG]  # twin of neighbours_probe.cs (fancy-index neighbours, where= neighbours, Tier 2 claims)
    python benchmark/nditer/probes/numpy_twins.py join before.tsv after.tsv [numpy.tsv]

Run with OPENBLAS_NUM_THREADS=1 so nothing on the NumPy side fans out, and on a hybrid P/E-core host
with NS_PROBE_AFFINITY=<hex mask> (the same mask as the C# probe) so both sides sit on one P-core.
See docs/NDITER_PERF_DISCOVERY.md.
"""
import sys
import time

import numpy as np


def best_ns(body):
    t = time.perf_counter()
    body()
    while time.perf_counter() - t < 0.06:
        body()
    t0 = time.perf_counter()
    pc = 0
    while time.perf_counter() - t0 < 0.02:
        body()
        pc += 1
    per = (time.perf_counter() - t0) / pc
    it = max(1, min(1_000_000, int(round(0.001 / max(per, 1e-9)))))
    rds = max(3, int(0.15 / max(it * per, 1e-9)) + 1)
    best = float("inf")
    for _ in range(rds):
        t0 = time.perf_counter()
        for _ in range(it):
            body()
        best = min(best, (time.perf_counter() - t0) * 1e9 / it)
    return best


def best_ms(body, rounds=7):
    t = time.perf_counter()
    body()
    while time.perf_counter() - t < 0.08:
        body()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        body()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def fixed():
    a = np.arange(1000, dtype=np.float64)
    b = a + 1.0
    o = np.empty(1000)
    RO, WO = ["readonly"], ["writeonly"]
    print("--- A. np.nditer construction (ns) ---")
    print(f"nditer 1op            : {best_ns(lambda: np.nditer(a)):.1f}")
    print(f"nditer 3op exl        : {best_ns(lambda: np.nditer((a, b, o), flags=['external_loop'], op_flags=[RO, RO, WO])):.1f}")
    print("--- B. strided rows positive (2M f64), ms [ns/row] ---")
    for w in (4, 16, 64):
        rows = 2_097_152 // w
        back = np.arange(rows * 2 * w, dtype=np.float64).reshape(rows, 2 * w)
        sv, dst = back[:, :w], np.empty((rows, w))
        t = best_ms(lambda: np.positive(sv, out=dst))
        print(f"w={w:4d} rows={rows:7d}: np.positive(out) {t:.3f} [{t * 1e6 / rows:.1f}]")
    print("--- C. contiguous elementwise 1M / 10M (ms) ---")
    for n in (1_000_000, 10_000_000):
        A = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        B = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
        O = np.empty(n)
        print(f"n={n:9d}: np.add(out) {best_ms(lambda: np.add(A, B, out=O)):.3f}  np.sqrt(out) {best_ms(lambda: np.sqrt(A, out=O)):.3f}  np.positive(out) {best_ms(lambda: np.positive(A, out=O)):.3f}")
    for n in (1, 1000, 100_000):
        a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        b = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
        o = np.empty(n)
        ob = np.empty(n, dtype=bool)
        sa = ((np.arange(2 * n, dtype=np.float64) % 53.0) + 1.0)[::2]
        sb = ((np.arange(2 * n, dtype=np.float64) % 17.0) + 1.0)[::2]
        ipa = a.copy()
        print(f"--- D. production routes n={n} (ns/call) ---")
        print(f"np.add(a,b,out=o)          : {best_ns(lambda: np.add(a, b, out=o)):.1f}")
        print(f"np.multiply(a,b,out=a)     : {best_ns(lambda: np.multiply(ipa, b, out=ipa)):.1f}")
        print(f"np.less(a,b,out=ob)        : {best_ns(lambda: np.less(a, b, out=ob)):.1f}")
        print(f"np.sqrt(a,out=o)           : {best_ns(lambda: np.sqrt(a, out=o)):.1f}")
        print(f"np.negative(a,out=o)       : {best_ns(lambda: np.negative(a, out=o)):.1f}")
        print(f"np.positive(a,out=o)       : {best_ns(lambda: np.positive(a, out=o)):.1f}")
        print(f"np.add(a,b) new            : {best_ns(lambda: np.add(a, b)):.1f}")
        print(f"np.add(sa,sb) strided new  : {best_ns(lambda: np.add(sa, sb)):.1f}")
    print("--- E. allocation ---")
    print(f"np.empty(1000)             : {best_ns(lambda: np.empty(1000)):.1f}")


def ab():
    for n in (1000, 100_000):
        tag = "1K" if n == 1000 else "100K"
        R, C = (25, 40) if n == 1000 else (250, 400)
        a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        b = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
        o = np.empty(n)
        A = ((np.arange(n, dtype=np.float64) % 97.0) + 1.0).reshape(R, C)
        At = A.T
        row = (np.arange(C, dtype=np.float64) % 5.0) + 1.0
        col = ((np.arange(R, dtype=np.float64) % 5.0) + 1.0).reshape(R, 1)
        sa = ((np.arange(2 * n, dtype=np.float64) % 53.0) + 1.0)[::2]
        sb = ((np.arange(2 * n, dtype=np.float64) % 17.0) + 1.0)[::2]
        af32 = (np.arange(n, dtype=np.float32) % 977) + 1
        ai32 = (np.arange(n) % 1000).astype(np.int32)
        mask = (np.arange(n) % 2) == 0
        sparse = (np.arange(n) % 97) == 0
        mask_dst = a.copy()
        anan = a.copy()
        anan[3] = np.nan
        dst_strided = np.empty(2 * n)[::2]
        ob = np.empty(n, dtype=bool)
        ipa = a.copy()
        idx = ((np.arange(n, dtype=np.int64) * 2654435761) % n).astype(np.int32)

        def masked_set():
            mask_dst[mask] = 5.0

        rows = [
            (f"add(A,row) bcast@{tag}", lambda: np.add(A, row)),
            (f"add(A,col) bcast@{tag}", lambda: np.add(A, col)),
            (f"add(sa,sb) strided@{tag}", lambda: np.add(sa, sb)),
            (f"add(At,At) transposed@{tag}", lambda: np.add(At, At)),
            (f"sqrt(sa) strided@{tag}", lambda: np.sqrt(sa)),
            (f"sqrt(ai32) promote@{tag}", lambda: np.sqrt(ai32)),
            (f"less(A,row) bcast@{tag}", lambda: np.less(A, row)),
            (f"less(sa,sb) strided@{tag}", lambda: np.less(sa, sb)),
            (f"left_shift(ai32,1)@{tag}", lambda: np.left_shift(ai32, 1)),
            (f"add(a,b,out)@{tag}", lambda: np.add(a, b, out=o)),
            (f"multiply(a,b,out=a) inplace@{tag}", lambda: np.multiply(ipa, b, out=ipa)),
            (f"add(a,b,out,where)@{tag}", lambda: np.add(a, b, out=o, where=mask)),
            (f"less(a,b,out)@{tag}", lambda: np.less(a, b, out=ob)),
            (f"sqrt(a,out)@{tag}", lambda: np.sqrt(a, out=o)),
            (f"sqrt(af32,out f64) cast@{tag}", lambda: np.sqrt(af32, out=o)),
            (f"copyto(strided,src)@{tag}", lambda: np.copyto(dst_strided, a)),
            (f"At.copy()@{tag}", lambda: At.copy()),
            (f"ravel(At)@{tag}", lambda: np.ravel(At)),
            (f"sa.astype(f32)@{tag}", lambda: sa.astype(np.float32)),
            (f"concatenate([a,b])@{tag}", lambda: np.concatenate([a, b])),
            (f"pad(a,2)@{tag}", lambda: np.pad(a, 2)),
            (f"diff(a)@{tag}", lambda: np.diff(a)),
            (f"a.fill(1.0)@{tag}", lambda: ipa.fill(1.0)),
            (f"sum(A,axis=1)@{tag}", lambda: np.sum(A, axis=1)),
            (f"sum(At,axis=1)@{tag}", lambda: np.sum(At, axis=1)),
            (f"amin(A,axis=1)@{tag}", lambda: np.amin(A, axis=1)),
            (f"sum(af32,dtype=f64)@{tag}", lambda: np.sum(af32, dtype=np.float64)),
            (f"nanmax(anan)@{tag}", lambda: np.nanmax(anan)),
            (f"nanmean(anan)@{tag}", lambda: np.nanmean(anan)),
            (f"nanstd(anan)@{tag}", lambda: np.nanstd(anan)),
            (f"any(sparse)@{tag}", lambda: np.any(sparse)),
            (f"all(mask)@{tag}", lambda: np.all(mask)),
            (f"cumsum(A,axis=1)@{tag}", lambda: np.cumsum(A, axis=1)),
            (f"average(a,weights=b)@{tag}", lambda: np.average(a, weights=b)),
            (f"where(cond,a,b)@{tag}", lambda: np.where(mask, a, b)),
            (f"a[mask]@{tag}", lambda: a[mask]),
            (f"a[mask]=5@{tag}", masked_set),
            (f"count_nonzero(a)@{tag}", lambda: np.count_nonzero(a)),
            (f"argwhere(mask)@{tag}", lambda: np.argwhere(mask)),
            (f"nonzero(sparse)@{tag}", lambda: np.nonzero(sparse)),
            (f"sort(A,axis=1)@{tag}", lambda: np.sort(A, axis=1)),
            (f"argsort(A,axis=1)@{tag}", lambda: np.argsort(A, axis=1)),
            (f"np.nditer class iternext@{tag}", lambda: [None for _ in np.nditer(a)]),
            (f"copy(a) contiguous@{tag}", lambda: a.copy()),
            (f"a[idx] fancy@{tag}", lambda: a[idx]),
        ]
        for name, body in rows:
            print(f"{name}\t{best_ns(body):.1f}")


def angles():
    print("--- (1) narrow strided rows, 2M f64 (ms) ---")
    for w in (4, 16, 64):
        rows = 2_097_152 // w
        back = np.arange(rows * 2 * w, dtype=np.float64).reshape(rows, 2 * w)
        sv, d = back[:, :w], np.empty((rows, w))
        print(f"w={w:4d}: np.positive(out) {best_ms(lambda: np.positive(sv, out=d)):.3f}  np.sqrt(out) {best_ms(lambda: np.sqrt(sv, out=d)):.3f}")
    n = 100_000
    a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
    b = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
    o = np.empty(n)
    alt = (np.arange(n) % 2) == 0
    blocks = ((np.arange(n) // 64) % 2) == 0
    half = np.arange(n) < n // 2
    print("--- (2) where= masked, 100K (us) ---")
    print(f"where=alternating {best_ns(lambda: np.add(a, b, out=o, where=alt)) / 1e3:.1f}  where=64-blocks {best_ns(lambda: np.add(a, b, out=o, where=blocks)) / 1e3:.1f}  where=half {best_ns(lambda: np.add(a, b, out=o, where=half)) / 1e3:.1f}  unmasked {best_ns(lambda: np.add(a, b, out=o)) / 1e3:.1f}")
    idx = ((np.arange(n, dtype=np.int64) * 2654435761) % n).astype(np.int32)
    vals = np.arange(n, dtype=np.float64)
    dst = a.copy()

    def scat():
        dst[idx] = vals

    print("--- (3) fancy index vs take/put, 100K (us) ---")
    print(f"a[idx] {best_ns(lambda: a[idx]) / 1e3:.1f}  np.take {best_ns(lambda: np.take(a, idx)) / 1e3:.1f}  a[idx]=v {best_ns(scat) / 1e3:.1f}  np.put {best_ns(lambda: np.put(dst, idx, vals)) / 1e3:.1f}")
    print("--- (4) allocation floor (ns) ---")
    print(f"np.empty(1000) {best_ns(lambda: np.empty(1000)):.1f}  np.empty(1) {best_ns(lambda: np.empty(1)):.1f}")
    # Interleaved and repeated on purpose (see the C# twin): a one-shot placement number is not evidence.
    print("--- (5) 1M add: page-offset stagger, 5 interleaved reps (ms) ---")
    n = 1_000_000
    A = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
    B = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
    O = np.empty(n)
    m = n - 1024
    print(f"page offsets mod 4096: A {A.ctypes.data % 4096}  B {B.ctypes.data % 4096}  O {O.ctypes.data % 4096}")
    for rep in range(5):
        t0 = best_ms(lambda: np.add(A[:m], B[:m], out=O[:m]), 15)
        t1 = best_ms(lambda: np.add(A[:m], B[16:16 + m], out=O[32:32 + m]), 15)
        t2 = best_ms(lambda: np.add(A[:m], B[8:8 + m], out=O[16:16 + m]), 15)
        print(f"rep{rep}: natural (same) offsets {t0:.3f}   B+128B/O+256B {t1:.3f}   B+64B/O+128B {t2:.3f}")


def best_us(body):
    return best_ns(body) / 1e3


def narrow(sections="ABCDE"):
    """Twin of narrow_probe.cs — the 2-D block kernel outside the first report's regime."""
    if "A" in sections:
        print("=== (A) f64 (rows, w) strided view, out=: positive | add | sqrt ===")
        for total in (100_000, 1_000_000, 2_097_152):
            for w in (1, 2, 3, 4, 8, 16, 64):
                rows = total // w
                back = np.arange(rows * 2 * w, dtype=np.float64).reshape(rows, 2 * w)
                back2 = (np.arange(rows * 2 * w, dtype=np.float64) % 7.0).reshape(rows, 2 * w)
                sv, sv2, dst = back[:, :w], back2[:, :w], np.empty((rows, w))
                tp = best_us(lambda: np.positive(sv, out=dst))
                ta = best_us(lambda: np.add(sv, sv2, out=dst))
                ts = best_us(lambda: np.sqrt(sv, out=dst))
                print(f"N={total:8d} w={w:3d} rows={rows:7d}: positive {tp:9.1f} [{tp * 1e3 / rows:5.2f}]  add {ta:9.1f} [{ta * 1e3 / rows:5.2f}]  sqrt {ts:9.1f} [{ts * 1e3 / rows:5.2f}]   (us [ns/row])")
    if "B" in sections:
        print("=== (B) 1M other dtypes (rows, w) out=: positive | add ===")
        for dt, name, widths in ((np.float32, "f32", (3, 4, 8, 16)), (np.uint8, "u8", (3, 16, 32)), (np.int32, "i32", (3, 4, 8))):
            total = 1_000_000
            for w in widths:
                rows = total // w
                back = (np.arange(rows * 2 * w) % 100).astype(dt).reshape(rows, 2 * w)
                back2 = (np.arange(rows * 2 * w) % 7).astype(dt).reshape(rows, 2 * w)
                sv, sv2, dst = back[:, :w], back2[:, :w], np.empty((rows, w), dtype=dt)
                tp = best_us(lambda: np.positive(sv, out=dst))
                ta = best_us(lambda: np.add(sv, sv2, out=dst))
                print(f"{name:4s} w={w:3d} rows={rows:7d}: positive {tp:9.1f} [{tp * 1e3 / rows:5.2f}]  add {ta:9.1f} [{ta * 1e3 / rows:5.2f}]")
    if "C" in sections:
        print("=== (C) 1M f64 exp(out) | mod(sv, 3.0, out) | contiguous twins ===")
        for w in (4, 16, 64):
            total = 1_000_000
            rows = total // w
            back = ((np.arange(rows * 2 * w, dtype=np.float64) % 13.0) + 0.5).reshape(rows, 2 * w)
            sv, dst = back[:, :w], np.empty((rows, w))
            flat, fdst = np.ascontiguousarray(sv), np.empty((rows, w))
            te, tec = best_us(lambda: np.exp(sv, out=dst)), best_us(lambda: np.exp(flat, out=fdst))
            tm, tmc = best_us(lambda: np.mod(sv, 3.0, out=dst)), best_us(lambda: np.mod(flat, 3.0, out=fdst))
            print(f"w={w:3d} rows={rows:7d}: exp {te:9.1f} [{te * 1e3 / rows:5.2f}] (contig {tec:8.1f})   mod {tm:9.1f} [{tm * 1e3 / rows:5.2f}] (contig {tmc:8.1f})")
    if "D" in sections:
        print("=== (D) 1M f64 broadcast shapes, out=: add(A,col) | multiply(view,2.0) | view*2.0 | add(A,row) | add(view,col) ===")
        for c in (4, 16, 64):
            total = 1_000_000
            r = total // c
            A = (np.arange(total, dtype=np.float64) % 97.0).reshape(r, c)
            col = (np.arange(r, dtype=np.float64) % 5.0).reshape(r, 1)
            row = (np.arange(c, dtype=np.float64) % 5.0).reshape(1, c)
            back = np.arange(r * 2 * c, dtype=np.float64).reshape(r, 2 * c)
            sv, dst = back[:, :c], np.empty((r, c))
            tc = best_us(lambda: np.add(A, col, out=dst))
            tms = best_us(lambda: np.multiply(sv, 2.0, out=dst))
            tos = best_us(lambda: sv * 2.0)
            tr = best_us(lambda: np.add(A, row, out=dst))
            tvc = best_us(lambda: np.add(sv, col, out=dst))
            print(f"c={c:3d} rows={r:7d}: add(A,col) {tc:8.1f} [{tc * 1e3 / r:5.2f}]  multiply(sv,2.0,out) {tms:8.1f} [{tms * 1e3 / r:5.2f}]  sv*2.0 {tos:8.1f}  add(A,row) {tr:8.1f} [{tr * 1e3 / r:5.2f}]  add(sv,col) {tvc:8.1f} [{tvc * 1e3 / r:5.2f}]")
    if "E" in sections:
        print("=== (E) 2M f64 3-D: flattenable x[:, :, :w] vs non-flattenable x[::2, :, :w], out=: positive | add ===")
        for w in (4, 16):
            total = 2_097_152
            d1 = 64
            d0 = total // (w * d1)
            x = np.arange(d0 * d1 * 2 * w, dtype=np.float64).reshape(d0, d1, 2 * w)
            y = (np.arange(d0 * d1 * 2 * w, dtype=np.float64) % 7.0).reshape(d0, d1, 2 * w)
            flat, flat2, dstF = x[:, :, :w], y[:, :, :w], np.empty((d0, d1, w))
            half, half2, dstH = x[::2, :, :w], y[::2, :, :w], np.empty(((d0 + 1) // 2, d1, w))
            rowsF, rowsH = d0 * d1, ((d0 + 1) // 2) * d1
            tpf, taf = best_us(lambda: np.positive(flat, out=dstF)), best_us(lambda: np.add(flat, flat2, out=dstF))
            tph, tah = best_us(lambda: np.positive(half, out=dstH)), best_us(lambda: np.add(half, half2, out=dstH))
            print(f"w={w:3d}: flat  positive {tpf:8.1f} [{tpf * 1e3 / rowsF:5.2f}]  add {taf:8.1f} [{taf * 1e3 / rowsF:5.2f}]   |  ::2  positive {tph:8.1f} [{tph * 1e3 / rowsH:5.2f}]  add {tah:8.1f} [{tah * 1e3 / rowsH:5.2f}]")


# ---------------------------------------------------------------------------
# fancy_where — the NumPy twin of fancy_where_probe.cs (identical keys): (A) the fancy-index
# operator vs np.take/np.put at 1K/100K/10M, (B) where= across mask run lengths, (C) where=
# over narrow strided rows. Masks use `//` (the C# side uses np.floor_divide — `/` on an
# integer NDArray is true division).
# ---------------------------------------------------------------------------
def _fw_row(key, us):
    print(f"{key}\t{us!r}")


def fancy_where_a():
    for n in (1_000, 100_000, 10_000_000):
        tag = {1_000: "1K", 100_000: "100K"}.get(n, "10M")
        a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        ai = np.arange(n, dtype=np.int32)
        idx32 = ((np.arange(n, dtype=np.int64) * 2654435761) % n).astype(np.int32)
        idx64 = idx32.astype(np.int64)
        vals = np.arange(n, dtype=np.float64)
        vals_i = np.arange(n, dtype=np.int32)
        dst = a.copy()
        dst_i = ai.copy()
        _fw_row(f"{tag}|f64|get|a[idx32]", best_us(lambda: a[idx32]))
        _fw_row(f"{tag}|f64|get|a[idx64]", best_us(lambda: a[idx64]))
        _fw_row(f"{tag}|f64|get|take(a,idx32)", best_us(lambda: np.take(a, idx32)))
        _fw_row(f"{tag}|f64|get|take(a,idx64)", best_us(lambda: np.take(a, idx64)))
        _fw_row(f"{tag}|i32|get|a[idx32]", best_us(lambda: ai[idx32]))
        _fw_row(f"{tag}|i32|get|take(a,idx64)", best_us(lambda: np.take(ai, idx64)))

        def s1():
            dst[idx32] = vals

        def s2():
            dst[idx64] = vals

        def s3():
            dst[idx64] = 3.0

        def s4():
            dst_i[idx32] = vals_i

        _fw_row(f"{tag}|f64|set|a[idx32]=v", best_us(s1))
        _fw_row(f"{tag}|f64|set|a[idx64]=v", best_us(s2))
        _fw_row(f"{tag}|f64|set|put(a,idx64,v)", best_us(lambda: np.put(dst, idx64, vals)))
        _fw_row(f"{tag}|f64|set|a[idx64]=scalar", best_us(s3))
        _fw_row(f"{tag}|i32|set|a[idx32]=v", best_us(s4))
        _fw_row(f"{tag}|i32|set|put(a,idx64,v)", best_us(lambda: np.put(dst_i, idx64, vals_i)))
        if n >= 8:
            rows_ = n // 8
            m = a.reshape(rows_, 8)
            ridx = ((np.arange(rows_, dtype=np.int64) * 2654435761) % rows_).astype(np.int64)
            rvals = np.arange(rows_ * 8, dtype=np.float64).reshape(rows_, 8)
            md = m.copy()

            def s5():
                md[ridx] = rvals

            def s6():
                md[ridx] = rvals[0]

            _fw_row(f"{tag}|f64|get2d|m[ridx]", best_us(lambda: m[ridx]))
            _fw_row(f"{tag}|f64|get2d|take(m,ridx,0)", best_us(lambda: np.take(m, ridx, axis=0)))
            _fw_row(f"{tag}|f64|set2d|m[ridx]=v", best_us(s5))
            _fw_row(f"{tag}|f64|set2d|m[ridx]=row", best_us(s6))


def fancy_where_b():
    for n in (100_000, 1_000_000):
        tag = "100K" if n == 100_000 else "1M"
        a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        b = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
        o = np.empty(n)
        ar = np.arange(n)
        for run in (1, 8, 64, 1024):
            mask = (ar // run % 2) == 0
            _fw_row(f"{tag}|add|where=run{run}", best_us(lambda: np.add(a, b, out=o, where=mask)))
            if run == 64:
                _fw_row(f"{tag}|sqrt|where=run{run}", best_us(lambda: np.sqrt(a, out=o, where=mask)))
        half = ar < n // 2
        all_t = ar >= 0
        all_f = ar < 0
        _fw_row(f"{tag}|add|where=half", best_us(lambda: np.add(a, b, out=o, where=half)))
        _fw_row(f"{tag}|add|where=allTrue", best_us(lambda: np.add(a, b, out=o, where=all_t)))
        _fw_row(f"{tag}|add|where=allFalse", best_us(lambda: np.add(a, b, out=o, where=all_f)))
        _fw_row(f"{tag}|add|unmasked", best_us(lambda: np.add(a, b, out=o)))


def fancy_where_c():
    for w in (3, 4, 16):
        total = 1_000_000
        rows_ = total // w
        back = (np.arange(rows_ * 2 * w, dtype=np.float64) % 97.0 + 1.0).reshape(rows_, 2 * w)
        back2 = (np.arange(rows_ * 2 * w, dtype=np.float64) % 31.0 + 2.0).reshape(rows_, 2 * w)
        sv, sv2 = back[:, :w], back2[:, :w]
        o = np.empty((rows_, w))
        ar = np.arange(rows_ * w).reshape(rows_, w)
        blocks = (ar // 64 % 2) == 0
        rowmask = ((np.arange(rows_) % 2) == 0)[:, None]
        colmask = ((np.arange(w) % 2) == 0)[None, :]
        _fw_row(f"1M|w{w}|add|where=blocks64", best_us(lambda: np.add(sv, sv2, out=o, where=blocks)))
        _fw_row(f"1M|w{w}|add|where=rowmask", best_us(lambda: np.add(sv, sv2, out=o, where=rowmask)))
        _fw_row(f"1M|w{w}|add|where=colmask", best_us(lambda: np.add(sv, sv2, out=o, where=colmask)))
        _fw_row(f"1M|w{w}|sqrt|where=blocks64", best_us(lambda: np.sqrt(sv, out=o, where=blocks)))
        _fw_row(f"1M|w{w}|add|unmasked", best_us(lambda: np.add(sv, sv2, out=o)))
        _fw_row(f"1M|w{w}|sqrt|unmasked", best_us(lambda: np.sqrt(sv, out=o)))



def fancy_where(sections="ABC"):
    if "A" in sections:
        fancy_where_a()
    if "B" in sections:
        fancy_where_b()
    if "C" in sections:
        fancy_where_c()


# ---------------------------------------------------------------------------
# neighbours — the NumPy twin of neighbours_probe.cs (identical keys): (E) the fancy-index shapes
# still on the delegate route, (F) where= neighbours, (G) the Tier 2 claims. Masks use `//`.
# ---------------------------------------------------------------------------
def _nb_row(key, us):
    print(f"{key}\t{us!r}")


def neighbours_e():
    for n in (100_000, 1_000_000):
        tag = "100K" if n == 100_000 else "1M"
        side = int(np.sqrt(n))
        m = (np.arange(side * side, dtype=np.float64) % 97.0 + 1.0).reshape(side, side)
        mT = m.T
        a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        ri = (np.arange(n, dtype=np.int64) * 2654435761) % side
        ci = (np.arange(n, dtype=np.int64) * 40503) % side
        rk = (np.arange(side, dtype=np.int64) * 2654435761) % side
        idx = (np.arange(n, dtype=np.int64) * 2654435761) % n
        idx_view = np.concatenate([idx, idx])[::2]
        mask = (np.arange(n) % 2) == 0
        mask_few = (np.arange(n) % 100) == 0
        vals = np.arange(n, dtype=np.float64)
        dst = a.copy()
        dst_view = np.arange(2 * n, dtype=np.float64)[::2]
        col_vals = np.arange(side * side, dtype=np.float64).reshape(side, side)

        def set2():
            m[ri, ci] = vals

        def setcol():
            m[:, rk] = col_vals

        def setmask_s():
            dst[mask] = 3.0

        def setmask_v():
            dst[mask] = vals[mask]

        def setview():
            dst_view[idx] = vals

        _nb_row(f"{tag}|fancy|m[ri,ci] (2 index arrays)", best_us(lambda: m[ri, ci]))
        _nb_row(f"{tag}|fancy|m[:, rk] (column gather)", best_us(lambda: m[:, rk]))
        _nb_row(f"{tag}|fancy|m[rk, :2] (rows + slice)", best_us(lambda: m[rk, :2]))
        _nb_row(f"{tag}|fancy|mT[rk] (F-order source rows)", best_us(lambda: mT[rk]))
        _nb_row(f"{tag}|fancy|a[idx[::2]] (strided index view)", best_us(lambda: a[idx_view]))
        _nb_row(f"{tag}|fancy|m[ri,ci]=v (2-array set)", best_us(set2))
        _nb_row(f"{tag}|fancy|m[:, rk]=v (column set)", best_us(setcol))
        _nb_row(f"{tag}|mask|a[mask] (50%)", best_us(lambda: a[mask]))
        _nb_row(f"{tag}|mask|a[mask] (1%)", best_us(lambda: a[mask_few]))
        _nb_row(f"{tag}|mask|a[mask]=scalar (50%)", best_us(setmask_s))
        _nb_row(f"{tag}|mask|a[mask]=v (50%)", best_us(setmask_v))
        _nb_row(f"{tag}|mask|compress", best_us(lambda: np.compress(mask, a)))
        _nb_row(f"{tag}|mask|count_nonzero", best_us(lambda: np.count_nonzero(mask)))
        _nb_row(f"{tag}|take|axis1 m.take(rk, axis=1)", best_us(lambda: np.take(m, rk, axis=1)))
        _nb_row(f"{tag}|put|put(view[::2], idx, v)", best_us(lambda: np.put(dst_view, idx, vals)))
        _nb_row(f"{tag}|put|view[::2][idx]=v", best_us(setview))


def neighbours_f():
    for n in (100_000, 1_000_000):
        tag = "100K" if n == 100_000 else "1M"
        a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
        b = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
        a32, b32 = a.astype(np.float32), b.astype(np.float32)
        o = np.empty(n)
        ob = np.empty(n, dtype=bool)
        ar = np.arange(n)
        blocks = (ar // 64 % 2) == 0
        _nb_row(f"{tag}|where|add(f32,f32,out=f64,where) cast-out", best_us(lambda: np.add(a32, b32, out=o, where=blocks)))
        _nb_row(f"{tag}|where|add(f32,f32,out=f64) cast-out unmasked", best_us(lambda: np.add(a32, b32, out=o)))
        _nb_row(f"{tag}|where|less(a,b,out,where)", best_us(lambda: np.less(a, b, out=ob, where=blocks)))
        _nb_row(f"{tag}|where|less(a,b,out)", best_us(lambda: np.less(a, b, out=ob)))
        _nb_row(f"{tag}|where|copyto(o,a,where)", best_us(lambda: np.copyto(o, a, where=blocks)))
        _nb_row(f"{tag}|where|np.where(mask,a,b)", best_us(lambda: np.where(blocks, a, b)))
        _nb_row(f"{tag}|where|add(a,b,where) no out (alloc)", best_us(lambda: np.add(a, b, where=blocks)))
        w = 4
        rows_ = n // w
        sv2 = b.reshape(rows_, w)
        o32 = np.empty((rows_, w), dtype=np.float32)
        m2 = blocks.reshape(rows_, w)
        back = (np.arange(rows_ * 2 * w, dtype=np.float64) % 97.0 + 1.0).reshape(rows_, 2 * w)
        svs = back[:, :4]
        o2 = np.empty((rows_, w))
        _nb_row(f"{tag}|where|rows w4 add(view,view,out=f32,where) cast-out", best_us(lambda: np.add(svs, sv2, out=o32, where=m2, casting="unsafe")))
        _nb_row(f"{tag}|where|rows w4 add(view,view,out,where)", best_us(lambda: np.add(svs, sv2, out=o2, where=m2)))


def neighbours_g():
    for n in (100_000, 1_000_000):
        tag = "100K" if n == 100_000 else "1M"
        i32 = (np.arange(n, dtype=np.int32) % 97) + 1
        i64 = i32.astype(np.int64)
        f32 = i32.astype(np.float32)
        f64 = i32.astype(np.float64)
        o64 = np.empty(n)
        _nb_row(f"{tag}|cast|sqrt(i32)", best_us(lambda: np.sqrt(i32)))
        _nb_row(f"{tag}|cast|sqrt(i32,out=f64)", best_us(lambda: np.sqrt(i32, out=o64)))
        _nb_row(f"{tag}|cast|sqrt(i64)", best_us(lambda: np.sqrt(i64)))
        _nb_row(f"{tag}|cast|sqrt(f32)", best_us(lambda: np.sqrt(f32)))
        _nb_row(f"{tag}|cast|sqrt(f64)", best_us(lambda: np.sqrt(f64)))
        _nb_row(f"{tag}|cast|exp(i32)", best_us(lambda: np.exp(i32)))
        _nb_row(f"{tag}|cast|negative(i32,out=f64)", best_us(lambda: np.negative(i32, out=o64)))
        _nb_row(f"{tag}|cast|add(i32,f64)", best_us(lambda: np.add(i32, f64)))
        _nb_row(f"{tag}|cast|astype(i32->f64)", best_us(lambda: i32.astype(np.float64)))
        _nb_row(f"{tag}|reduce|sum(f32,dtype=f64)", best_us(lambda: np.sum(f32, dtype=np.float64)))
        _nb_row(f"{tag}|reduce|sum(f32)", best_us(lambda: np.sum(f32)))
        _nb_row(f"{tag}|reduce|sum(i32) (->i64)", best_us(lambda: np.sum(i32)))
        _nb_row(f"{tag}|reduce|mean(f32)", best_us(lambda: np.mean(f32)))
        _nb_row(f"{tag}|reduce|sum(i32,dtype=f64)", best_us(lambda: np.sum(i32, dtype=np.float64)))
        for w in (3, 4, 8, 16, 64):
            rows_ = n // w
            x = f64[:rows_ * w].reshape(rows_, w)
            _nb_row(f"{tag}|axis|sum(x,axis=1) w{w}", best_us(lambda: np.sum(x, axis=1)))
            _nb_row(f"{tag}|axis|max(x,axis=1) w{w}", best_us(lambda: np.max(x, axis=1)))
            _nb_row(f"{tag}|axis|mean(x,axis=1) w{w}", best_us(lambda: np.mean(x, axis=1)))
            xt = f64[:rows_ * w].reshape(w, rows_)
            _nb_row(f"{tag}|axis|sum(xt,axis=0) ({w},N)", best_us(lambda: np.sum(xt, axis=0)))
    for n in (1, 16, 100, 1000):
        a = np.arange(n, dtype=np.float64) + 1.0
        b = a.copy()
        o = a.copy()
        i32 = np.arange(n, dtype=np.int32)
        _nb_row(f"tiny{n}|add(out)", best_us(lambda: np.add(a, b, out=o)))
        _nb_row(f"tiny{n}|add (alloc)", best_us(lambda: np.add(a, b)))
        _nb_row(f"tiny{n}|sqrt(out)", best_us(lambda: np.sqrt(a, out=o)))
        _nb_row(f"tiny{n}|sqrt(i32) cast", best_us(lambda: np.sqrt(i32)))
        _nb_row(f"tiny{n}|sum", best_us(lambda: np.sum(a)))
        _nb_row(f"tiny{n}|less(out)", best_us(lambda: np.less(a, b)))



def neighbours(sections="EFG"):
    if "E" in sections:
        neighbours_e()
    if "F" in sections:
        neighbours_f()
    if "G" in sections:
        neighbours_g()


def join(before, after, numpy_tsv=None):
    def load(path):
        d = {}
        for line in open(path, encoding="utf-8"):
            if "\t" in line:
                k, v = line.rstrip("\n").split("\t")
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
        return d

    b, a = load(before), load(after)
    npd = load(numpy_tsv) if numpy_tsv else {}
    print(f"{'op':40s} {'before':>10s} {'after':>10s} {'gain':>7s} {'numpy':>10s} {'NPY/NS':>7s}")
    for k in b:
        if k not in a:
            continue
        npv = npd.get(k)
        print(f"{k:40s} {b[k]:10.1f} {a[k]:10.1f} {b[k] / a[k]:6.2f}x {('%10.1f' % npv) if npv else '':>10s} {('%6.2fx' % (npv / a[k])) if npv else '':>7s}")


def pin_affinity():
    """NS_PROBE_AFFINITY=<hex mask> pins the process like the C# probes do (hybrid-core hosts)."""
    import os
    aff = os.environ.get("NS_PROBE_AFFINITY")
    if aff and sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(int(aff, 16)))


if __name__ == "__main__":
    pin_affinity()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    print(f"# numpy {np.__version__}", file=sys.stderr)
    if cmd == "fixed":
        fixed()
    elif cmd == "ab":
        ab()
    elif cmd == "angles":
        angles()
    elif cmd == "narrow":
        narrow(sys.argv[2] if len(sys.argv) > 2 else "ABCDE")
    elif cmd == "fancy_where":
        fancy_where(sys.argv[2] if len(sys.argv) > 2 else "ABC")
    elif cmd == "neighbours":
        neighbours(sys.argv[2] if len(sys.argv) > 2 else "EFG")
    elif cmd == "join":
        join(*sys.argv[2:])
    else:
        print(__doc__)
