import numpy as np, time, sys
# operand_bench.py — NumPy twin of operand_bench.cs (identical keys).
# Layout classes the op×layout×dtype matrix omits: 1-D contig/strided/reversed,
# scalar operand, mixed operand layouts (C+F, C+T), binary broadcast (row+col),
# column-broadcast unary.


def best(f, it, wm, rd):
    for _ in range(wm):
        f()
    b = float("inf")
    for _ in range(rd):
        t = time.perf_counter()
        for _ in range(it):
            f()
        b = min(b, (time.perf_counter() - t) / it)
    return b * 1000.0


N1, R, C = 1_000_000, 1000, 1000
it, wm, rd = 30, 6, 3
DTYPES = [("f64", np.float64), ("f32", np.float32), ("f16", np.float16),
          ("i32", np.int32), ("i64", np.int64), ("c128", np.complex128)]

out = []
for dn, dt in DTYPES:
    a1 = ((np.arange(N1) % 17) + 1).astype(dt)
    a1s = a1[::2]
    a1r = a1[::-1]
    a2 = ((np.arange(R * C) % 17) + 1).astype(dt).reshape(R, C)
    a2F = np.asfortranarray(a2)
    a2T = a2.T
    row = a2[0:1, :]
    col = a2[:, 0:1]
    colb = np.broadcast_to(col, (R, C))
    sc = dt(2)
    cases = {
        "1d_C": lambda: a1 + a1,
        "1d_strided": lambda: a1s + a1s,
        "1d_rev": lambda: a1r + a1r,
        "scalar_rhs": lambda: a2 + sc,
        "scalar_lhs": lambda: sc + a2,
        "mix_C_F": lambda: a2 + a2F,
        "mix_C_T": lambda: a2 + a2T,
        "bcast_row": lambda: a2 + row,
        "bcast_col": lambda: a2 + col,
        "colbcast_unary": lambda: np.positive(colb),
    }
    for k, fn in cases.items():
        try:
            fn()
            out.append(f"{k}|{dn}\t{best(fn, it, wm, rd):.6g}")
        except Exception as e:
            sys.stderr.write(f"{k}|{dn}: {type(e).__name__}: {e}\n")

print("\n".join(out))
sys.stderr.write(f"[opnd_layout_bench.py] {len(out)} rows; numpy {np.__version__}\n")
