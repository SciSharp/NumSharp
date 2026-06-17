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

sizes = [(316,316,'100K'), (1000,1000,'1M')]
for r, c, lbl in sizes:
    n = r*c
    baseD = (((np.arange(n) % 11).astype(np.float64) - 5.0) * 0.01 + 1.0).reshape(r,c)  # 0.95..1.05
    it = 25 if n <= 1_000_000 else 10
    # NumPy has no decimal; longdouble is the nearest native high-precision dtype.
    aDec = baseD.astype(np.longdouble)
    for op in ['sum','prod','min','max','mean']:
        for ax in [0,1]:
            print(f"decimal|{op}|axis{ax}|C|{lbl}|{bench(lambda op=op,ax=ax: reduce(op,aDec,ax), it):.4f}")
    aHalf = baseD.astype(np.float16)
    for op in ['sum','prod','min','max','mean']:
        for ax in [0,1]:
            print(f"half|{op}|axis{ax}|C|{lbl}|{bench(lambda op=op,ax=ax: reduce(op,aHalf,ax), it):.4f}")

fsizes = [(316,316,'100K'), (1000,1000,'1M'), (3162,3162,'10M')]
for r, c, lbl in fsizes:
    n = r*c
    a = np.arange(n, dtype=np.float64).reshape(r,c)*0.0009 + 0.7
    b = np.arange(n, dtype=np.float64).reshape(r,c)*0.0006 + 0.3
    it = 40 if n <= 1_000_000 else 12
    for op in ['sum','mean','max']:
        for ax in [0,1]:
            print(f"fused|{op}|axis{ax}|C|{lbl}|{bench(lambda op=op,ax=ax: reduce(op, a*b, ax), it):.4f}")
