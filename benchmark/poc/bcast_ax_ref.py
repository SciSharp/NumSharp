import numpy as np, json, sys

def enc(v, dt):
    if dt == 'complex': return [enc(v.real,'double'), enc(v.imag,'double')]
    if dt in ('single','double','half','decimal'):
        f=float(v)
        if np.isnan(f): return "nan"
        if f==np.inf: return "inf"
        if f==-np.inf: return "-inf"
        return f
    if dt=='bool': return bool(v)
    return int(v)

NP={'bool':np.bool_,'byte':np.uint8,'sbyte':np.int8,'int16':np.int16,'uint16':np.uint16,
    'int32':np.int32,'uint32':np.uint32,'int64':np.int64,'uint64':np.uint64,'char':np.uint16,
    'half':np.float16,'single':np.float32,'double':np.float64,'decimal':np.float64,'complex':np.complex128}

def vals(dt,n):
    if dt=='bool': return [(i%3==0) for i in range(n)]
    if dt in ('byte','uint16','uint32','uint64','char'): return [ (i*7+3)%50+1 for i in range(n)]
    if dt in ('sbyte','int16','int32','int64'): return [ ((i*13+5)%40)-20 for i in range(n)]
    if dt in ('single','double','half'): return [ (((i*17+1)%23)-11)*0.5 for i in range(n)]
    if dt=='decimal': return [ (((i*17+1)%23)-11)*0.25 for i in range(n)]
    if dt=='complex': return [ complex(((i*5)%13)-6,((i*3)%11)-5) for i in range(n)]
    raise ValueError(dt)

def vals_z(dt,n):   # with zeros/falses sprinkled (for all/any/count_nonzero)
    if dt=='bool': return [(i%2==0) for i in range(n)]
    base=vals(dt,n)
    out=[]
    for i,x in enumerate(base):
        out.append(0 if (i%4==0) else x)
    return out

def vals_nan(dt,n): # NaN/inf laden (for nan-aware)
    pat=[np.nan,1.5,-2.0,np.inf,np.nan,3.0,-np.inf,0.5]
    if dt=='complex': return [complex(np.nan if (i%3==0) else i-3, 1.0) for i in range(n)]
    return [pat[i%len(pat)] for i in range(n)]

cases=[]
def enc_base(base, dt):
    flat = base.ravel() if dt=='complex' else base.ravel().tolist()
    return [enc(x,dt) for x in flat]
def enc_result(r):
    k=r.dtype.kind
    if k=='c': return ('c',[[enc(x.real,'double'),enc(x.imag,'double')] for x in r.ravel()])
    if k=='f': return ('f',[enc(x,'double') for x in r.ravel().tolist()])
    return (k,[int(x) for x in r.ravel().tolist()])

def make_base(dt, base_shape, kind='plain'):
    n=int(np.prod(base_shape))
    src = {'plain':vals,'z':vals_z,'nan':vals_nan}[kind]
    return np.array(src(dt,n),dtype=NP[dt]).reshape(base_shape)

def apply_pre(base, pre):
    if   pre=='none': return base
    elif pre=='T':    return base.T
    elif pre=='rev':  return base[tuple(slice(None,None,-1) for _ in base.shape)]
    elif pre=='slice':return base[tuple(slice(1,-1) for _ in base.shape)]
    raise ValueError(pre)

def reduce(fn_name, bc, axis, keepdims):
    if fn_name=='sum':  return np.sum(bc,axis=axis,keepdims=keepdims)
    if fn_name=='prod': return np.prod(bc,axis=axis,keepdims=keepdims)
    if fn_name=='min':  return np.amin(bc,axis=axis,keepdims=keepdims)
    if fn_name=='max':  return np.amax(bc,axis=axis,keepdims=keepdims)
    if fn_name=='mean': return np.mean(bc,axis=axis,keepdims=keepdims)
    if fn_name=='var':  return np.var(bc,axis=axis,keepdims=keepdims)
    if fn_name=='std':  return np.std(bc,axis=axis,keepdims=keepdims)
    if fn_name=='argmax': return np.argmax(bc,axis=axis,keepdims=keepdims) if axis is not None else np.argmax(bc)
    if fn_name=='argmin': return np.argmin(bc,axis=axis,keepdims=keepdims) if axis is not None else np.argmin(bc)
    if fn_name=='all':  return np.all(bc,axis=axis,keepdims=keepdims)
    if fn_name=='any':  return np.any(bc,axis=axis,keepdims=keepdims)
    if fn_name=='count_nonzero': return np.count_nonzero(bc,axis=axis,keepdims=keepdims)
    if fn_name=='nansum':  return np.nansum(bc,axis=axis,keepdims=keepdims)
    if fn_name=='nanmax':  return np.nanmax(bc,axis=axis,keepdims=keepdims)
    if fn_name=='nanmin':  return np.nanmin(bc,axis=axis,keepdims=keepdims)
    if fn_name=='nanmean': return np.nanmean(bc,axis=axis,keepdims=keepdims)
    if fn_name=='ptp':  return np.ptp(bc,axis=axis,keepdims=keepdims)
    if fn_name=='median': return np.median(bc,axis=axis,keepdims=keepdims)
    raise ValueError(fn_name)

def emit(cid, dt, base_shape, target, pre, op, axis, keepdims, kind='plain'):
    base=make_base(dt,base_shape,kind)
    b=apply_pre(base,pre)
    try: bc=np.broadcast_to(b,target)
    except Exception: return
    try:
        r=np.asarray(reduce(op,bc,axis,keepdims))
        rk,exp=enc_result(r)
        cases.append(dict(id=cid,dtype=dt,base=enc_base(base,dt),base_shape=list(base_shape),pre=pre,
                          target=list(target),op=op,axis=(-999 if axis is None else axis),keepdims=keepdims,
                          rkind=rk,expected=exp,expected_shape=list(r.shape),raises=False))
    except Exception as e:
        cases.append(dict(id=cid,dtype=dt,base=enc_base(base,dt),base_shape=list(base_shape),pre=pre,
                          target=list(target),op=op,axis=(-999 if axis is None else axis),keepdims=keepdims,
                          rkind='?',expected=[],expected_shape=[],raises=True,why=type(e).__name__))

DT15=['bool','byte','sbyte','int16','uint16','int32','uint32','int64','uint64','char','half','single','double','decimal','complex']

# layouts: (tag, base_shape, target, pre)  -- broadcast views, incl non-contig remainder
LAYOUTS=[
 ("a0",   (1,6),    (5,6),     'none'),    # axis0 broadcast
 ("a1",   (5,1),    (5,6),     'none'),    # axis1 broadcast
 ("ab",   (1,1),    (5,6),     'none'),    # both broadcast
 ("m3",   (2,1,3),  (2,4,3),   'none'),    # mid broadcast 3d
 ("o3",   (1,3,4),  (5,3,4),   'none'),    # outer broadcast 3d
 ("all3", (1,1,1),  (3,4,2),   'none'),    # all broadcast 3d
 ("ncT",  (4,6),    (3,6,4),   'T'),       # transpose then prepend bcast -> remainder non-contig
 ("ncS",  (5,6),    (3,3,4),   'slice'),   # slice then prepend bcast -> remainder non-contig + offset
]
def axes_of(target, pre, base_shape):
    nd=len(target)
    return [None]+list(range(nd))+[-1]

# 1) axis + flat reductions: sum/prod/min/max/mean over ALL 15 dtypes
for dt in DT15:
    for tag,bs,tg,pre in LAYOUTS:
        for op in ('sum','prod','min','max','mean'):
            for ax in axes_of(tg,pre,bs):
                for kd in (False,True):
                    emit(f"{tag}:{dt}:{op}:ax{ax}:kd{int(kd)}", dt, bs, tg, pre, op, ax, kd)

# 2) argmax/argmin over broadcast (fold EXCLUDES these -> untouched path)
for dt in ('bool','byte','sbyte','int32','int64','uint64','char','half','single','double','decimal','complex'):
    for tag,bs,tg,pre in LAYOUTS:
        for op in ('argmax','argmin'):
            for ax in axes_of(tg,pre,bs):
                emit(f"{tag}:{dt}:{op}:ax{ax}", dt, bs, tg, pre, op, ax, False)

# 3) var/std over broadcast
for dt in ('byte','int32','int64','half','single','double','decimal','complex'):
    for tag,bs,tg,pre in LAYOUTS[:6]:
        for op in ('var','std'):
            for ax in (None,0,1,-1):
                emit(f"{tag}:{dt}:{op}:ax{ax}", dt, bs, tg, pre, op, ax, False)

# 4) all/any/count_nonzero over broadcast (data with zeros)
for dt in ('bool','byte','int32','int64','double','complex'):
    for tag,bs,tg,pre in LAYOUTS[:6]:
        for op in ('all','any','count_nonzero'):
            for ax in (None,0,1):
                emit(f"{tag}:{dt}:{op}:ax{ax}", dt, bs, tg, pre, op, ax, False, kind='z')

# 5) nan-aware over broadcast
for dt in ('half','single','double','complex'):
    for tag,bs,tg,pre in LAYOUTS[:6]:
        for op in ('nansum','nanmax','nanmin','nanmean'):
            for ax in (None,0,1):
                emit(f"{tag}:{dt}:{op}:ax{ax}", dt, bs, tg, pre, op, ax, False, kind='nan')

# 6) ptp / median over broadcast
for dt in ('byte','int32','int64','single','double'):
    for tag,bs,tg,pre in LAYOUTS[:6]:
        for op in ('ptp','median'):
            for ax in (None,0,1):
                emit(f"{tag}:{dt}:{op}:ax{ax}", dt, bs, tg, pre, op, ax, False)

with open(sys.argv[1] if len(sys.argv)>1 else 'bcast_ax_ref.json','w') as f:
    json.dump(dict(cases=cases),f)
import collections
print(f"wrote {len(cases)} cases; numpy {np.__version__}")
print("ops:",dict(collections.Counter(c['op'] for c in cases)))
print("raises:",sum(c['raises'] for c in cases))
