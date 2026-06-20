import numpy as np, time, sys

def best_ms(f, iters, warm, rounds):
    for _ in range(warm): f()
    best = float('inf')
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(iters): f()
        best = min(best, (time.perf_counter()-t)/iters)
    return best*1000.0

def pick(n):
    return (200,30,3) if n <= 100_000 else (15,4,2)

SIZES = [("100K",316,316), ("1M",1000,1000)]
DTYPES = [("f64",np.float64),("f32",np.float32),("c128",np.complex128),
          ("dec",np.float64),("f16",np.float16),("i32",np.int32),("i64",np.int64)]  # dec modelled as f64
OPS = {"sum":np.sum,"min":np.amin,"max":np.amax,"prod":np.prod}
LAYOUTS = ["C","F","T","strided","negrow","negcol","sliced","bcast"]

def layout(a, name):
    if name=="C": return a
    if name=="F": return np.asfortranarray(a)
    if name=="T": return a.T
    if name=="strided": return a[:, ::2]
    if name=="negrow": return a[::-1, :]
    if name=="negcol": return a[:, ::-1]
    if name=="sliced": return a[1:a.shape[0]-1, 1:a.shape[1]-1]
    if name=="bcast": return np.broadcast_to(a[0:1, :], (a.shape[0], a.shape[1]))
    raise ValueError(name)

out = []
for tag,R,C in SIZES:
    iters,warm,rounds = pick(R*C)
    for dname,dt in DTYPES:
        base = ((np.arange(R*C) % 17) + 1).astype(dt).reshape(R,C)
        for lay in LAYOUTS:
            v = layout(base, lay)
            for op,fn in OPS.items():
                for axis in (0,1):
                    key = f"{tag}|{dname}|{lay}|{op}|ax{axis}"
                    try:
                        ms = best_ms(lambda fn=fn,v=v,axis=axis: fn(v,axis=axis), iters,warm,rounds)
                        out.append(f"{key}\t{ms:.6g}")
                    except Exception as e:
                        sys.stderr.write(f"{key}: {type(e).__name__}\n")
print("\n".join(out))
sys.stderr.write(f"[reduce_layout_bench.py] {len(out)} rows; numpy {np.__version__}\n")
