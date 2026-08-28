import numpy as np, time, sys

def best_ms(f, it, wm, rd):
    # Warmup; its timed tail (first call excluded — cache/alloc-cold) doubles as a per-call pilot.
    f()
    t = time.perf_counter()
    for _ in range(max(2, wm) - 1): f()
    per_call = (time.perf_counter() - t) * 1000.0 / max(1, max(2, wm) - 1)
    # Min-time policy: a call >20 ms/call runs EXACTLY 100 times (min over 100); everything else
    # batches ~1 ms windows and accumulates enough to span ~200 ms total (time-bound, not a round count).
    slow = per_call > 20.0
    it = 1 if slow else max(1, min(1_000_000, int(round(1.0 / max(per_call, 1e-6)))))
    rd = 100 if slow else max(2, int(200.0 / max(it * per_call, 1e-9)) + 1)
    best=float('inf')
    for _ in range(rd):
        t=time.perf_counter()
        for _ in range(it): f()
        best=min(best,(time.perf_counter()-t)/it)
    return best*1000.0
def pick(n): return (200,30,3) if n<=100_000 else (30,6,3)

SIZES=[("100K",316,316),("1M",1000,1000)]
DTYPES=[("f64",np.float64),("f32",np.float32),("c128",np.complex128),
        ("f16",np.float16),("i32",np.int32),("i64",np.int64)]
LAYOUTS=["C","F","T","strided","sliced","negrow","negcol","bcast"]
def layout(a,l):
    if l=="C": return a
    if l=="F": return np.asfortranarray(a)
    if l=="T": return a.T
    if l=="strided": return a[:, ::2]
    if l=="sliced": return a[1:a.shape[0]-1, 1:a.shape[1]-1]
    if l=="negrow": return a[::-1, :]
    if l=="negcol": return a[:, ::-1]
    if l=="bcast": return np.broadcast_to(a[0:1, :], (a.shape[0], a.shape[1]))
    raise ValueError(l)
def op(name,v):
    if name=="add": return v+v
    if name=="mul": return v*v
    if name=="neg": return -v
    if name=="abs": return np.abs(v)
    if name=="sqrt": return np.sqrt(v)
    if name=="less": return np.less(v,v)
    if name=="copy": return v.copy()
    raise ValueError(name)
OPS=["add","mul","neg","abs","sqrt","less","copy"]

out=[]
for tag,R,C in SIZES:
    it,wm,rd=pick(R*C)
    for dn,dt in DTYPES:
        base=((np.arange(R*C)%17)+1).astype(dt).reshape(R,C)
        for lay in LAYOUTS:
            v=layout(base,lay)
            for o in OPS:
                key=f"{tag}|{dn}|{lay}|{o}"
                try:
                    op(o,v)  # warm/validate
                    out.append(f"{key}\t{best_ms(lambda o=o,v=v: op(o,v), it,wm,rd):.6g}")
                except Exception as e:
                    sys.stderr.write(f"{key}: {type(e).__name__}\n")
print("\n".join(out))
sys.stderr.write(f"[elementwise_layout_bench.py] {len(out)} rows; numpy {np.__version__}\n")
