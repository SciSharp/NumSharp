import numpy as np, json, sys

# ---- value encoding (json-safe, round-trippable) ----
def enc(v, dt):
    if dt == 'complex':
        return [enc(v.real, 'double'), enc(v.imag, 'double')]
    if dt in ('single', 'double', 'half'):
        f = float(v)
        if np.isnan(f): return "nan"
        if f == np.inf: return "inf"
        if f == -np.inf: return "-inf"
        return f
    if dt == 'decimal':
        return repr(float(v))   # built via double in C#
    if dt == 'bool':
        return bool(v)
    if dt == 'char':
        return int(v)
    return int(v)

# NumSharp dtype name -> numpy dtype for building/compute. char modelled as uint16.
NP = {
 'bool': np.bool_, 'byte': np.uint8, 'sbyte': np.int8, 'int16': np.int16,
 'uint16': np.uint16, 'int32': np.int32, 'uint32': np.uint32, 'int64': np.int64,
 'uint64': np.uint64, 'char': np.uint16, 'half': np.float16, 'single': np.float32,
 'double': np.float64, 'decimal': np.float64, 'complex': np.complex128,
}

def base_values(dt, n):
    # deterministic, ensures distinct group extrema; includes negatives where signed
    if dt == 'bool':
        return [(i % 3 == 0) for i in range(n)]
    if dt in ('byte','uint16','uint32','uint64','char'):
        return [ (i*7 + 3) % 200 for i in range(n) ]
    if dt in ('sbyte','int16','int32','int64'):
        return [ ((i*13 + 5) % 120) - 60 for i in range(n) ]   # -60..59
    if dt in ('single','double','half'):
        return [ (((i*17 + 1) % 23) - 11) * 0.5 for i in range(n) ]  # negatives + .5
    if dt == 'decimal':
        return [ (((i*17 + 1) % 23) - 11) * 0.25 for i in range(n) ]
    if dt == 'complex':
        return [ complex(((i*5)%13)-6, ((i*3)%11)-5) for i in range(n) ]
    raise ValueError(dt)

def special_float(dt):
    # array with NaN, +inf, -inf, -0.0, +0.0, finite
    return [np.nan, -np.inf, 3.5, -2.0, np.inf, -0.0, 0.0, 1.0, np.nan, -7.0, 5.0, 2.0]

cases = []
def reduce_case(cid, dt, base, base_shape, transform, op, axis, keepdims):
    arr = np.array(base, dtype=NP[dt]).reshape(base_shape)
    if transform == 'c': v = arr
    elif transform == 'f': v = np.asfortranarray(arr)
    elif transform == 't': v = arr.T
    elif transform == 's2': v = arr[:, ::2]
    elif transform == 's2row': v = arr[::2, :]
    elif transform == 'rev': v = arr[::-1, :]
    elif transform == 'revcol': v = arr[:, ::-1]
    else: raise ValueError(transform)
    fn = np.amin if op == 'amin' else np.amax
    try:
        if axis is None:
            r = fn(v, keepdims=keepdims)
        else:
            r = fn(v, axis=axis, keepdims=keepdims)
        r = np.asarray(r)
        exp = [enc(x, dt) for x in r.ravel(order='C').tolist()] if dt!='complex' else [enc(x,dt) for x in r.ravel(order='C')]
        exp_shape = list(r.shape)
        raise_flag = False
    except ValueError:
        exp, exp_shape, raise_flag = [], [], True
    cases.append(dict(id=cid, dtype=dt, kind='reduce', base=[enc(x,dt) for x in base],
                      base_shape=list(base_shape), transform=transform, op=op,
                      axis=axis, keepdims=keepdims, expected=exp, expected_shape=exp_shape,
                      raises=raise_flag))

def binary_case(cid, dt, a, a_shape, b, b_shape, binop):
    A = np.array(a, dtype=NP[dt]).reshape(a_shape)
    B = np.array(b, dtype=NP[dt]).reshape(b_shape)
    fn = np.maximum if binop == 'maximum' else np.minimum
    r = np.asarray(fn(A, B))
    exp = [enc(x, dt) for x in r.ravel(order='C')] if dt=='complex' else [enc(x,dt) for x in r.ravel(order='C').tolist()]
    cases.append(dict(id=cid, dtype=dt, kind='binary', a=[enc(x,dt) for x in a], a_shape=list(a_shape),
                      b=[enc(x,dt) for x in b], b_shape=list(b_shape), binop=binop,
                      expected=exp, expected_shape=list(r.shape), raises=False))

DTYPES = ['bool','byte','sbyte','int16','uint16','int32','uint32','int64','uint64','char','half','single','double','decimal','complex']
TRANSFORMS = ['c','f','t','s2','s2row','rev','revcol']

# ---- reductions: every dtype x transform x axis x op (2D 4x6) ----
for dt in DTYPES:
    R, C = 4, 6
    base = base_values(dt, R*C)
    for tr in TRANSFORMS:
        for op in ('amin','amax'):
            for axis in (None, 0, 1, -1):
                reduce_case(f"red:{dt}:{tr}:{op}:ax{axis}", dt, base, (R,C), tr, op, axis, False)
    # keepdims sanity
    for op in ('amin','amax'):
        reduce_case(f"red:{dt}:c:{op}:ax1:kd", dt, base, (R,C), 'c', op, 1, True)
    # 3D axis mapping (C only, axes 0/1/2)
    base3 = base_values(dt, 2*3*4)
    for op in ('amin','amax'):
        for axis in (0,1,2,None):
            reduce_case(f"red:{dt}:3d:{op}:ax{axis}", dt, base3, (2,3,4), 'c', op, axis, False)

# ---- float/complex special-value reductions (NaN/inf/-0) ----
for dt in ('single','double','half','complex'):
    if dt == 'complex':
        sp = [complex(np.nan,1), complex(-np.inf,0), complex(3,4), complex(-2,-2),
              complex(np.inf,0), complex(0,0), complex(1,-1), complex(np.nan,np.nan),
              complex(-7,2), complex(5,5), complex(2,0), complex(-0.0,0.0)]
    else:
        sp = special_float(dt)
    for tr in ('c','t','s2'):
        for op in ('amin','amax'):
            for axis in (None,0,1):
                reduce_case(f"red:{dt}:SP:{tr}:{op}:ax{axis}", dt, sp, (3,4), tr, op, axis, False)

# ---- empty reduction (should raise) ----
for dt in ('int32','double'):
    reduce_case(f"red:{dt}:empty:amax:ax1", dt, [], (3,0), 'c', 'amax', 1, False)

# ---- elementwise maximum/minimum: dtype x broadcast/NaN ----
for dt in DTYPES:
    a = base_values(dt, 6); b = list(reversed(base_values(dt, 6)))
    for binop in ('maximum','minimum'):
        binary_case(f"bin:{dt}:{binop}:same", dt, a, (6,), b, (6,), binop)
        # broadcast (1,3)+(2,3)... use (2,3) vs (3,)
        a2 = base_values(dt, 6); b1 = base_values(dt, 3)
        binary_case(f"bin:{dt}:{binop}:bcast", dt, a2, (2,3), b1, (3,), binop)
for dt in ('single','double','half','complex'):
    if dt=='complex':
        a=[complex(np.nan,1),complex(2,2),complex(3,-1),complex(-1,np.nan)]
        b=[complex(1,1),complex(np.nan,0),complex(3,5),complex(2,2)]
    else:
        a=[np.nan,2.0,3.0,-1.0]; b=[1.0,np.nan,3.0,np.nan]
    for binop in ('maximum','minimum'):
        binary_case(f"bin:{dt}:{binop}:nan", dt, a, (4,), b, (4,), binop)

out = dict(cases=cases)
with open(sys.argv[1] if len(sys.argv)>1 else 'minmax_ref.json','w') as f:
    json.dump(out, f)
print(f"wrote {len(cases)} cases; numpy {np.__version__}")
