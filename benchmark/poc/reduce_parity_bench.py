import numpy as np, time, sys

def bench(f, it):
    for _ in range(3): f()
    ts = []
    for _ in range(it):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts)//2] * 1000.0

sizes = [(316,316,'100K'), (1000,1000,'1M'), (3162,3162,'10M')]
for r, c, lbl in sizes:
    base = (np.arange(r*c, dtype=np.float64).reshape(r,c)*0.0009 + 0.7) \
         + 1j*(np.arange(r*c, dtype=np.float64).reshape(r,c)*0.0006 + 0.3)
    aC = np.ascontiguousarray(base)
    aT = np.ascontiguousarray(base).T  # transposed view (non-contig)
    it = 50 if r*c <= 1_000_000 else 15
    for op in ['sum','prod','min','max']:
        for ax in [0,1]:
            f = (lambda op=op, ax=ax: getattr(np, op)(aC, axis=ax))
            print(f"{lbl}|{op}|axis{ax}|C|{bench(f,it):.4f}")
        for ax in [0,1]:
            f = (lambda op=op, ax=ax: getattr(np, op)(aT, axis=ax))
            print(f"{lbl}|{op}|axis{ax}|T|{bench(f,it):.4f}")
