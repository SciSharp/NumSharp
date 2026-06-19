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

cases=[]
def enc_base(base, dt):
    flat = base.ravel() if dt=='complex' else base.ravel().tolist()
    return [enc(x,dt) for x in flat]
def enc_result(r):
    # encode by RESULT dtype kind so integer (overflowing) results stay exact
    k=r.dtype.kind
    if k=='c': return ('c',[[enc(x.real,'double'),enc(x.imag,'double')] for x in r.ravel()])
    if k=='f': return ('f',[enc(x,'double') for x in r.ravel().tolist()])
    return (k,[int(x) for x in r.ravel().tolist()])   # i/u/b exact
def emit(cid, dt, base_shape, target, pre, op, keepdims=False):
    n=int(np.prod(base_shape))
    base=np.array(vals(dt,n),dtype=NP[dt]).reshape(base_shape)
    if   pre=='none': b=base
    elif pre=='T':    b=base.T
    elif pre=='rev':  b=base[::-1]
    elif pre=='revall': b=base[tuple(slice(None,None,-1) for _ in base_shape)]
    elif pre=='slice':b=base[tuple(slice(1,-1) for _ in base_shape)]
    else: raise ValueError(pre)
    try: bc=np.broadcast_to(b, target)
    except Exception: return
    fn={'sum':np.sum,'prod':np.prod,'min':np.amin,'max':np.amax,'mean':np.mean}[op]
    try:
        r=np.asarray(fn(bc, keepdims=keepdims))
        rk,exp=enc_result(r)
        cases.append(dict(id=cid,dtype=dt,base=enc_base(base,dt),base_shape=list(base_shape),pre=pre,
                          target=list(target),op=op,keepdims=keepdims,rkind=rk,expected=exp,
                          expected_shape=list(r.shape),raises=False))
    except Exception:
        cases.append(dict(id=cid,dtype=dt,base=enc_base(base,dt),base_shape=list(base_shape),pre=pre,
                          target=list(target),op=op,keepdims=keepdims,rkind='?',expected=[],
                          expected_shape=[],raises=True))

DT=['bool','byte','sbyte','int16','uint16','int32','uint32','int64','uint64','char','half','single','double','decimal','complex']
OPS=['sum','prod','min','max','mean']

# --- pure broadcasts: prepend / inner / both / 3D / high-rank ---
PURE=[
 ("ax0",   (1,6),   (5,6),   'none'),     # axis0 broadcast (canary shape)
 ("inner", (5,1),   (5,6),   'none'),     # inner axis broadcast
 ("both",  (1,1),   (5,6),   'none'),     # scalar broadcast -> collapses to scalar
 ("1d",    (1,),    (7,),    'none'),     # 1-D broadcast
 ("prepend",(6,),   (4,6),   'none'),     # broadcast_to PREPENDS a new axis
 ("3d_mid",(2,1,3), (2,4,3), 'none'),     # middle axis broadcast
 ("3d_outer",(1,3,4),(5,3,4),'none'),     # outer axis broadcast
 ("3d_all",(1,1,1), (3,4,2), 'none'),     # all broadcast -> scalar
 ("5d",    (1,2,1,2,1),(3,2,2,2,2),'none'),# high-rank mixed broadcast
]
for dt in DT:
    for tag,bs,tg,pre in PURE:
        for op in OPS:
            emit(f"{tag}:{dt}:{op}", dt, bs, tg, pre, op)

# --- broadcast PREPENDED onto a NON-CONTIGUOUS base (post-fold remainder is non-contig) ---
# base (4,6) -> transform -> broadcast_to (3,)+transformed.shape  (axis0 broadcast, inner non-contig)
NC=[("nc_T",(4,6),'T'), ("nc_rev",(4,6),'rev'), ("nc_revall",(4,6),'revall'), ("nc_slice",(5,6),'slice')]
for dt in DT:
    for tag,bs,pre in NC:
        n=int(np.prod(bs)); base=np.array(vals(dt,n),dtype=NP[dt]).reshape(bs)
        tb = {'T':base.T,'rev':base[::-1],'revall':base[::-1,::-1],'slice':base[1:-1,1:-1]}[pre]
        target=(3,)+tb.shape
        for op in OPS:
            emit(f"{tag}:{dt}:{op}", dt, bs, target, pre, op)

# --- special values: NaN / inf / -0 over broadcast (min/max propagation, sum) ---
def emit_special(cid, dt, flat, base_shape, target, op):
    base=np.array(flat,dtype=NP[dt]).reshape(base_shape)
    bc=np.broadcast_to(base,target)
    fn={'sum':np.sum,'prod':np.prod,'min':np.amin,'max':np.amax,'mean':np.mean}[op]
    r=np.asarray(fn(bc))
    rk,exp=enc_result(r)
    cases.append(dict(id=cid,dtype=dt,base=enc_base(base,dt),base_shape=list(base_shape),pre='none',
                      target=list(target),op=op,keepdims=False,rkind=rk,expected=exp,
                      expected_shape=list(r.shape),raises=False))
for dt in ('single','double','half'):
    for op in ('sum','min','max','mean'):
        emit_special(f"sp_nan:{dt}:{op}", dt, [np.nan,-np.inf,3.5,-2.0,np.inf,1.0], (1,6),(5,6), op)
        emit_special(f"sp_negzero:{dt}:{op}", dt, [-0.0,0.0,-0.0,0.0], (1,4),(4,4), op)
for op in ('sum','prod'):
    emit_special(f"sp_cx:{op}", 'complex', [complex(np.nan,1),complex(2,2),complex(-1,3)], (1,3),(4,3), op)

# --- integer prod OVERFLOW over broadcast (NumPy wraps) ---
for dt in ('int32','int64','uint32','byte'):
    emit_special(f"ovf:{dt}:prod", dt, [3,5,7,2,9,4,6,8], (1,8),(20,8), 'prod')

# --- keepdims over broadcast ---
for dt in ('double','int64'):
    for op in ('sum','min'):
        emit(f"kd:{dt}:{op}", dt, (1,6),(5,6),'none', op, keepdims=True)

with open(sys.argv[1] if len(sys.argv)>1 else 'bcast_ref.json','w') as f:
    json.dump(dict(cases=cases),f)
print(f"wrote {len(cases)} cases; numpy {np.__version__}")
