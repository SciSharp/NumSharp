import numpy as np, json, sys

def enc(v, dt):
    if dt == 'complex':
        return [enc(v.real,'double'), enc(v.imag,'double')]
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

def base_values(dt,n,small):
    if dt=='bool': return [(i%3==0) for i in range(n)]
    if dt in ('byte','uint16','uint32','uint64','char'):
        return [ (i%4)+1 for i in range(n)] if small else [ (i*7+3)%50 for i in range(n)]
    if dt in ('sbyte','int16','int32','int64'):
        return [ ((i%5)-2) for i in range(n)] if small else [ ((i*13+5)%40)-20 for i in range(n)]
    if dt in ('single','double','half'):
        return [ (((i%5)-2)*0.5) for i in range(n)] if small else [ (((i*17+1)%23)-11)*0.5 for i in range(n)]
    if dt=='decimal':
        return [ (((i%5)-2)*0.5) for i in range(n)] if small else [ (((i*17+1)%23)-11)*0.25 for i in range(n)]
    if dt=='complex':
        return [ complex(((i%5)-2),((i%3)-1)) for i in range(n)] if small else [ complex(((i*5)%13)-6,((i*3)%11)-5) for i in range(n)]
    raise ValueError(dt)

cases=[]
def case(cid,dt,base,bshape,tr,op,axis):
    arr=np.array(base,dtype=NP[dt]).reshape(bshape)
    if   tr=='c': v=arr
    elif tr=='f': v=np.asfortranarray(arr)
    elif tr=='t': v=arr.T
    elif tr=='s2': v=arr[:,::2]
    elif tr=='s2row': v=arr[::2,:]
    elif tr=='rev': v=arr[::-1,:]
    elif tr=='revcol': v=arr[:,::-1]
    elif tr=='slice': v=arr[1:3,1:3]
    elif tr=='slicestep': v=arr[1:4:2,1:5:2]
    else: raise ValueError(tr)
    fn={'sum':np.sum,'prod':np.prod,'mean':np.mean}[op]
    r=np.asarray(fn(v) if axis is None else fn(v,axis=axis))
    # Encode the RESULT generically (NOT by input dtype): sum/prod promote
    # (bool->int64, int->int64) and mean->float, so dt-encoding would bool-ify /
    # truncate the expected. Comparator parses doubles with tolerance.
    if dt=='complex':
        exp=[[enc(x.real,'double'),enc(x.imag,'double')] for x in r.ravel(order='C')]
    else:
        exp=[enc(x,'double') for x in r.ravel(order='C').tolist()]
    cases.append(dict(id=cid,dtype=dt,base=[enc(x,dt) for x in base],base_shape=list(bshape),
                      transform=tr,op=op,axis=axis,expected=exp,expected_shape=list(r.shape)))

DT=['bool','byte','sbyte','int16','uint16','int32','uint32','int64','uint64','char','half','single','double','decimal','complex']
TR=['c','f','t','s2','s2row','rev','revcol','slice','slicestep']
for dt in DT:
    R,C=4,6
    small = True   # keep prod from overflowing; sum/mean fine too
    base=base_values(dt,R*C,small)
    for tr in TR:
        for op in ('sum','prod','mean'):
            for axis in (None,0,1,-1):
                case(f"{op}:{dt}:{tr}:ax{axis}",dt,base,(R,C),tr,op,axis)

with open(sys.argv[1] if len(sys.argv)>1 else 'reduce_ref.json','w') as f:
    json.dump(dict(cases=cases),f)
print(f"wrote {len(cases)} cases; numpy {np.__version__}")
