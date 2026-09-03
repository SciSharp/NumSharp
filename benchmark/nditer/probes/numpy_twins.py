"""numpy_twins.py — the NumPy 2.4.2 side of the NDIter probes (same shapes, same timing policy).

    python benchmark/nditer/probes/numpy_twins.py fixed     # twin of fixed_cost_probe.cs
    python benchmark/nditer/probes/numpy_twins.py ab        # twin of ab_ops_probe.cs (id<TAB>ns rows)
    python benchmark/nditer/probes/numpy_twins.py angles    # twin of angles_probe.cs
    python benchmark/nditer/probes/numpy_twins.py narrow [ABCDE]   # twin of narrow_probe.cs (2-D block kernel)
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
    elif cmd == "join":
        join(*sys.argv[2:])
    else:
        print(__doc__)
