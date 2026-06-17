import numpy as np, time

def bench(f, it):
    for _ in range(3): f()
    ts = []
    for _ in range(it):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts)//2] * 1000.0

dt = {'double': np.float64, 'float': np.float32, 'int64': np.int64}
sizes = [(1000,1000,'1M'), (3162,3162,'10M')]
for name, tc in dt.items():
    for r, c, lbl in sizes:
        n = r*c
        aC = ((np.arange(n, dtype=np.float64).reshape(r,c)*0.0009 + 0.7)).astype(tc)
        aT = aC.T
        it = 40 if n <= 1_000_000 else 15
        for tag, a in [('C', aC), ('T', aT)]:
            for ax in [0,1]:
                print(f"{name}|{tag}|axis{ax}|{lbl}|numpy|{bench(lambda a=a,ax=ax: np.sum(a, axis=ax), it):.4f}")
