"""numpy_twins.py — the NumPy 2.4.2 side of benchmark/fusion/probes/expr_probe.cs.

    NS_PROBE_AFFINITY=4 OPENBLAS_NUM_THREADS=1 python benchmark/fusion/probes/numpy_twins.py [ACD]

Same deterministic data, same expressions, same timing policy (best-of rounds) as the C#
probe. NumPy has no expression fusion (numexpr is not a pinned dependency), so the C
section is the UNFUSED NumPy baseline the fused/unfused NumSharp numbers sit against;
section A prints the NumPy answers the C# semantics probes are compared with.
"""
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "scripts"))
try:
    from benchmark_host import pin_current_process_from_env  # noqa: E402
    pin_current_process_from_env()
except Exception:  # pragma: no cover - pinning is best-effort
    pass

sections = (sys.argv[1] if len(sys.argv) > 1 else "ACD").upper()


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


def data(n):
    idx = np.arange(n, dtype=np.float64)
    a = (idx % 977.0) / 977.0
    b = ((idx * 7.0) % 991.0) / 991.0 + 0.5
    c = ((idx * 13.0) % 983.0) / 983.0 * 0.5
    return a, b, c


print(f"numpy {np.__version__}")

if "A" in sections:
    print("\n=== A. NumPy answers for the semantics probes ===")
    z = np.array([3 - 4j, 0j, -1 + 1j, complex(np.nan, 0)])
    print(f"  [A3]  np.abs(complex128)        -> {np.abs(z).dtype} {np.abs(z)}")
    print(f"  [A4]  np.sign(complex128)       -> {np.sign(z).dtype} {np.sign(z)}")
    print(f"  [A4d] np.isnan(complex128)      -> {np.isnan(z)}")
    i4 = np.arange(64, dtype=np.int32) - 32
    try:
        np.power(i4, i4)
        print("  [A5]  np.power(i4, i4-with-negatives) -> OK (unexpected)")
    except Exception as e:  # noqa: BLE001
        print(f"  [A5]  np.power(i4, i4-with-negatives) -> {type(e).__name__}: {e}")
    print(f"  [A6]  np.sum(z)={np.sum(z)} prod={np.prod(z)} mean={np.mean(z)}")
    try:
        print(f"  [A6c] np.min(z)={np.min(z)}")
    except Exception as e:  # noqa: BLE001
        print(f"  [A6c] np.min(z) -> {type(e).__name__}: {e}")
    A, B, _ = data(100_003)
    af, bf = A.astype(np.float32), B.astype(np.float32)
    s32 = np.sum(af * bf)
    m32 = np.mean(af * bf)
    print(f"  [A8]  np.sum(af*bf)  = 0x{np.float32(s32).view(np.uint32):08X} ({s32!r})")
    print(f"  [A9]  np.mean(af*bf) = 0x{np.float32(m32).view(np.uint32):08X} ({m32!r})")
    print(f"  [A9b] np.sum(a*b) f64 = {np.sum(A * B)!r}")
    u = np.array([2**63 + 1], dtype=np.uint64)
    s = np.array([2**63 - 1], dtype=np.int64)
    print(f"  [A11] np.greater(u8[2^63+1], i8[2^63-1]) -> {np.greater(u, s)}  (result_type={np.result_type(u, s)})")
    print(f"  [A12] uint64([1]) + (2**64-1) -> {(np.array([1], dtype=np.uint64) + (2**64 - 1)).dtype} {(np.array([1], dtype=np.uint64) + (2**64 - 1))}")
    e = np.zeros((0, 3))
    print(f"  [A13b] np.sum(e*e, axis=0) over (0,3) -> shape {np.sum(e * e, axis=0).shape} {np.sum(e * e, axis=0)}")
    x = np.arange(10, dtype=np.float64)
    np.multiply(x[0:5], 10.0, out=x[5:10])
    print(f"  [A14] out= aliasing strided view -> {x}")
    m = np.arange(6, dtype=np.float64).reshape(2, 3)
    r = m.T * 2.0
    print(f"  [A17] (m.T*2).flags: F={r.flags.f_contiguous} C={r.flags.c_contiguous}")
    zero = np.zeros(64, dtype=np.int32)
    with np.errstate(all="ignore"):
        print(f"  [A15] i4/0 -> {(i4 / zero).dtype}:{(i4 / zero)[1]}  i4%0 -> {(i4 % zero)[1]}  i4//0 -> {(i4 // zero)[1]}")
    bb = np.array([True, False])
    print(f"  [A18] logical_not(bool)={np.logical_not(bb)} isnan(bool)={np.isnan(bb)} abs(bool)={np.abs(bb).dtype}")
    print(f"  [A20] where(i4>0, f4, i4) dtype -> {np.where(i4 > 0, af[:64], i4).dtype}")
    print(f"  [A23] bool+1 -> {(bb + 1).dtype}, bool*2.5 -> {(bb * 2.5).dtype}")
    f2 = A[:64].astype(np.float16)
    print(f"  [A7]  (f2*f2+f2).dtype={ (f2 * f2 + f2).dtype }  exp(f2).dtype={np.exp(f2).dtype}")

if "C" in sections:
    print("\n=== C. NumPy unfused, ms/call (best-of) ===")
    print("case\tN\tnumpy_ms")
    for n in (1_000, 100_000, 4_000_000):
        a, b, c = data(n)
        af, bf = a.astype(np.float32), b.astype(np.float32)
        f2a, f2b = a.astype(np.float16), b.astype(np.float16)
        i4 = (np.arange(n) % 1000).astype(np.int32)
        i8 = np.arange(n) % 1000
        rows, cols = {1_000: (10, 100), 100_000: (100, 1000)}.get(n, (2000, 2000))
        a2, b2 = a.reshape(rows, cols), b.reshape(rows, cols)
        half = n // 2
        aS, bS, cS = a[0:2 * half:2], b[0:2 * half:2], c[0:2 * half:2]
        cases = [
            ("a*b+c f64", lambda: a * b + c),
            ("(a-b)/(a+b) f64", lambda: (a - b) / (a + b)),
            ("sqrt(a*a+b*b) f64", lambda: np.sqrt(a * a + b * b)),
            ("where(a>b,a,b) f64", lambda: np.where(a > b, a, b)),
            ("maximum(a,b) f64", lambda: np.maximum(a, b)),
            ("a>0.5 (bool out)", lambda: a > 0.5),
            ("(a>0.2)&(b<0.8)", lambda: (a > 0.2) & (b < 0.8)),
            ("leaky relu f64", lambda: np.where(a > 0.5, a, a * 0.01)),
            ("exp(a)*b f64", lambda: np.exp(a) * b),
            ("exp(af)*bf f32", lambda: np.exp(af) * bf),
            ("sin(a)*cos(a) f64", lambda: np.sin(a) * np.cos(a)),
            ("i4*2+f8 mixed", lambda: i4 * 2 + c),
            ("i8*i8+1 int64", lambda: i8 * i8 + 1),
            ("f2*f2+f2 half", lambda: f2a * f2b + f2a),
            ("abs(a) f64", lambda: np.abs(a)),
            ("strided a*b+c", lambda: aS * bS + cS),
            ("sum(a*b) f64", lambda: np.sum(a * b)),
            ("sum(af*bf) f32", lambda: np.sum(af * bf)),
            ("mean((a-b)^2) f64", lambda: np.mean((a - b) * (a - b))),
            ("max(a*b) f64", lambda: np.max(a * b)),
            ("sum(a2*b2,ax0)", lambda: np.sum(a2 * b2, axis=0)),
            ("sum(a2*b2,ax1)", lambda: np.sum(a2 * b2, axis=1)),
        ]
        for cid, fn in cases:
            print(f"{cid}\t{n}\t{best_ms(fn):.4f}")

if "D" in sections:
    print("\n=== D. NumPy fixed cost (n=8), ns/call ===")
    a, b, c = data(8)
    o = np.empty_like(a)

    def two_pass():
        np.multiply(a, b, out=o)
        np.add(o, c, out=o)

    print("case\tns")
    print(f"unfused a*b+c\t{best_ns(lambda: a * b + c):.0f}")
    print(f"np.multiply/add out= (2 passes)\t{best_ns(two_pass):.0f}")
    print(f"np.sum(a*b)\t{best_ns(lambda: np.sum(a * b)):.0f}")
    print(f"np.where(a>b,a,b)\t{best_ns(lambda: np.where(a > b, a, b)):.0f}")

print("\ndone.")
