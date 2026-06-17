import numpy as np, time

def bench(f, it):
    for _ in range(3): f()
    ts = []
    for _ in range(it):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts)//2] * 1000.0

def reduce(op, a, ax):
    return {'sum': np.sum, 'prod': np.prod, 'min': np.amin, 'max': np.amax, 'mean': np.mean}[op](a, axis=ax)

sizes = [(316,316,'100K'), (1000,1000,'1M'), (3162,3162,'10M')]

# ---------- Complex128 ----------
for r, c, lbl in sizes:
    n = r*c
    re = np.arange(n, dtype=np.float64).reshape(r,c)*0.0009 + 0.7
    im = np.arange(n, dtype=np.float64).reshape(r,c)*0.0006 + 0.3
    aC = np.ascontiguousarray(re + 1j*im)
    aT = aC.T
    it = 40 if n <= 1_000_000 else 12
    for op in ['sum','prod','min','max','mean']:
        for ax in [0,1]:
            print(f"complex|{op}|axis{ax}|C|{lbl}|{bench(lambda op=op,ax=ax: reduce(op,aC,ax), it):.4f}")
        for ax in [0,1]:
            print(f"complex|{op}|axis{ax}|T|{lbl}|{bench(lambda op=op,ax=ax: reduce(op,aT,ax), it):.4f}")

# ---------- Decimal (NumPy: object dtype Decimal) & Half (float16) ----------
from decimal import Decimal
for r, c, lbl in [s for s in sizes if s[2] != '10M']:
    n = r*c
    baseD = np.arange(n, dtype=np.float64).reshape(r,c)*0.0009 + 0.7
    it = 25 if n <= 1_000_000 else 10
    # NumPy has no native decimal; float128 (longdouble) is the closest high-precision native dtype.
    aDec = baseD.astype(np.longdouble)
    for op in ['sum','prod','min','max','mean']:
        for ax in [0,1]:
            print(f"decimal|{op}|axis{ax}|C|{lbl}|{bench(lambda op=op,ax=ax: reduce(op,aDec,ax), it):.4f}")
    aHalf = baseD.astype(np.float16)
    for op in ['mean','sum']:
        for ax in [0,1]:
            print(f"half|{op}|axis{ax}|C|{lbl}|{bench(lambda op=op,ax=ax: reduce(op,aHalf,ax), it):.4f}")

# ---------- Fused: NumPy baseline is plain a*b then reduce (no numexpr) ----------
for r, c, lbl in sizes:
    n = r*c
    a = np.arange(n, dtype=np.float64).reshape(r,c)*0.0009 + 0.7
    b = np.arange(n, dtype=np.float64).reshape(r,c)*0.0006 + 0.3
    it = 40 if n <= 1_000_000 else 12
    for op in ['sum','mean','max']:
        for ax in [0,1]:
            print(f"fused|{op}|axis{ax}|C|{lbl}|{bench(lambda op=op,ax=ax: reduce(op, a*b, ax), it):.4f}")
