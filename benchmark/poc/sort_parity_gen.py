import numpy as np, os, json

OUT = "/tmp/sorttest"
os.makedirs(OUT, exist_ok=True)
cases = []
rng = np.random.default_rng(123)

dts = {'int32':np.int32,'int64':np.int64,'uint8':np.uint8,'int16':np.int16,'float32':np.float32,'float64':np.float64}
shapes = [(1000,), (37,53), (8,9,10)]

cid = 0
def emit(a, axis):
    global cid
    name = f"c{cid}"; cid += 1
    np.save(f"{OUT}/{name}_in.npy", a)
    s = np.sort(a, axis=axis)
    np.save(f"{OUT}/{name}_sort.npy", s)
    g = np.argsort(a, axis=axis, kind='stable').astype(np.int64)
    np.save(f"{OUT}/{name}_arg.npy", g)
    cases.append({'name':name,'dtype':str(a.dtype),'shape':list(a.shape),'axis':(-99 if axis is None else axis)})

for name,dt in dts.items():
    for shp in shapes:
        if dt in (np.float32,np.float64):
            a = (rng.random(shp)*200-100).astype(dt)
        else:
            info = np.iinfo(dt)
            a = rng.integers(info.min, info.max, size=shp, dtype=dt)
        for axis in range(len(shp)):
            emit(a, axis)
        emit(a, None)            # axis=None (flatten)
        emit(a, -1)              # negative axis

# NaN cases (float)
for dt in (np.float32, np.float64):
    a = (rng.random((20,30))*10-5).astype(dt)
    a[rng.random((20,30)) < 0.15] = np.nan
    a[0,0] = np.inf; a[1,1] = -np.inf
    emit(a, 0); emit(a, 1); emit(a, None)

# edge: empty, single, ties
emit(np.array([], dtype=np.int32), 0)
emit(np.array([5], dtype=np.int32), 0)
emit(np.array([3,1,3,1,3,1,2,2], dtype=np.int32), 0)
emit(np.full((5,5), 7, dtype=np.int32), 1)

json.dump(cases, open(f"{OUT}/manifest.json","w"))
print(f"wrote {len(cases)} cases to {OUT}")
