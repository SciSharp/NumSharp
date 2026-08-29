import numpy as np, time, sys, os
# cast_matrix_bench.py — NumPy side. Phase 0 of CAST_BEAT_NUMPY_PLAN.md.
# For every src dtype x layout x dst dtype at 1M, times v.astype(dst, copy=True).
# Output key: 1M|{src}|{layout}|{dst}\t{ms}  (identical keys to the C# side).
# Decimal has no NumPy dtype -> omitted (any pair touching 'dec' is NS-only).
# char -> uint16 (NumSharp Char is a 2-byte unsigned numeric).

def best_ms(f, it, wm, rd):
    depth = os.environ.get("NUMSHARP_BENCHMARK_DEPTH", "measure").lower()
    if depth == "pass":
        t = time.perf_counter(); f(); return (time.perf_counter() - t) * 1000.0
    if depth == "light":
        wm = min(wm, 3)
    # Warmup; its timed tail (first call excluded — cache/alloc-cold) doubles as a per-call pilot.
    f()
    t = time.perf_counter()
    for _ in range(max(2, wm) - 1): f()
    per_call = (time.perf_counter() - t) * 1000.0 / max(1, max(2, wm) - 1)
    # Min-time policy: a call >20 ms/call runs EXACTLY 100 times (min over 100); everything else
    # batches ~1 ms windows and accumulates enough to span ~200 ms total (time-bound, not a round count).
    slow = per_call > 20.0
    it = 1 if slow else max(1, min(1_000_000, int(round(1.0 / max(per_call, 1e-6)))))
    budget = 200.0 / (6 if depth == "light" else 1)
    rd = (17 if depth == "light" else 100) if slow else max(1, int(budget / max(it * per_call, 1e-9)) + 1)
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
ALIASES = {"bool":"bool","uint8":"u8","int8":"i8","int16":"i16","uint16":"u16",
           "int32":"i32","uint32":"u32","int64":"i64","uint64":"u64","char":"char",
           "float16":"f16","float32":"f32","float64":"f64","decimal":"dec","complex128":"c128"}
REQUESTED = {ALIASES.get(x.strip().lower(), x.strip().lower())
             for x in os.environ.get("NUMSHARP_BENCHMARK_DTYPES", "").split(",") if x.strip()}
def wanted(*dtypes): return not REQUESTED or any(dtype in REQUESTED for dtype in dtypes)
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
            if not wanted(sn, dn):
                continue
            try:
                v.astype(ddt, copy=True)
                ms = best_ms(lambda v=v, ddt=ddt: v.astype(ddt, copy=True), it, wm, rd)
                out.append(f"1M|{sn}|{lay}|{dn}\t{ms:.6g}")
            except Exception as e:
                sys.stderr.write(f"cast {sn}/{lay}/{dn}: {type(e).__name__}: {e}\n")
print("\n".join(out))
sys.stderr.write(f"[cast_matrix_bench.py] {len(out)} rows; numpy {np.__version__}\n")
