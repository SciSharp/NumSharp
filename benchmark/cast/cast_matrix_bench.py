import numpy as np, time, sys
# cast_matrix_bench.py — NumPy side. Phase 0 of CAST_BEAT_NUMPY_PLAN.md.
# For every src dtype x layout x dst dtype at 1M, times v.astype(dst, copy=True).
# Output key: 1M|{src}|{layout}|{dst}\t{ms}  (identical keys to the C# side).
# Decimal has no NumPy dtype -> omitted (any pair touching 'dec' is NS-only).
# char -> uint16 (NumSharp Char is a 2-byte unsigned numeric).

def best_ms(f, it, wm, rd):
    # Warmup; its timed tail (first call excluded — cache/alloc-cold) doubles as a per-call pilot.
    f()
    t = time.perf_counter()
    for _ in range(max(2, wm) - 1): f()
    per_call = (time.perf_counter() - t) * 1000.0 / max(1, max(2, wm) - 1)
    # Sample rule: >=50 timed rounds (>=20 when one round exceeds 10 ms); rounds auto-size to ~1 ms.
    it = 1 if per_call > 10.0 else max(1, min(1_000_000, int(round(1.0 / max(per_call, 1e-6)))))
    rd = max(rd, 20 if per_call > 10.0 else 50)
    best = float('inf')
    for _ in range(rd):
        t = time.perf_counter()
        for _ in range(it): f()
        best = min(best, (time.perf_counter() - t) / it)
    return best * 1000.0

R, C = 1000, 1000
it, wm, rd = 20, 5, 3
DTYPES = [("bool", np.bool_), ("u8", np.uint8), ("i8", np.int8),
          ("i16", np.int16), ("u16", np.uint16), ("i32", np.int32),
          ("u32", np.uint32), ("i64", np.int64), ("u64", np.uint64),
          ("char", np.uint16), ("f16", np.float16), ("f32", np.float32),
          ("f64", np.float64), ("c128", np.complex128)]
LAYOUTS = ["C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast"]

def layout(a, l):
    if l == "C": return a
    if l == "F": return np.asfortranarray(a)
    if l == "T": return a.T
    if l == "sliced": return a[1:a.shape[0]-1, 1:a.shape[1]-1]
    if l == "negrow": return a[::-1, :]
    if l == "negcol": return a[:, ::-1]
    if l == "strided": return a[:, ::2]
    if l == "bcast": return np.broadcast_to(a[0:1, :], (a.shape[0], a.shape[1]))
    raise ValueError(l)

out = []
for sn, sdt in DTYPES:
    base = ((np.arange(R * C) % 17) + 1).astype(sdt).reshape(R, C)
    for lay in LAYOUTS:
        v = layout(base, lay)
        for dn, ddt in DTYPES:
            try:
                v.astype(ddt, copy=True)
                ms = best_ms(lambda v=v, ddt=ddt: v.astype(ddt, copy=True), it, wm, rd)
                out.append(f"1M|{sn}|{lay}|{dn}\t{ms:.6g}")
            except Exception as e:
                sys.stderr.write(f"cast {sn}/{lay}/{dn}: {type(e).__name__}: {e}\n")
print("\n".join(out))
sys.stderr.write(f"[cast_matrix_bench.py] {len(out)} rows; numpy {np.__version__}\n")
