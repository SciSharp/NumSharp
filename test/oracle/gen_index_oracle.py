#!/usr/bin/env python3
"""
gen_index_oracle.py — emit a committed, behaviour-exact NumPy 2.4.2 indexing oracle.

This is the indexing sibling of gen_oracle.py. Where gen_oracle.py proves elementwise/
reduction ops bit-identical to NumPy, this proves the GETTER and SETTER index surface
(basic + fancy + boolean + 0-d-bool + every mixed advanced combination) identical to NumPy.

NumPy is the oracle. Python evaluates each case once and records {ok, shape, vals} (or the
exception type). The C# replayer (Fuzz/IndexOracleTests.cs) rebuilds the SAME base array and
the SAME index from a portable TOKEN encoding, runs get/set, and bit-compares shape + values +
which-side-raised. No Python runs at test time or in CI.

Output (JSONL, one case per line), into test/NumSharp.UnitTest/Fuzz/corpus/:
  index_curated.jsonl  — deterministic basic/fancy/bool/0-d-bool/mixed matrix + setters
  index_dtype.jsonl    — a handful of index forms swept across all 13 NumPy dtypes
  index_random_<seed>.jsonl — seeded random fuzz over the whole index space

Token encoding (portable; both sides interpret identically):
  ["int",n] ["slice",start,stop,step] ["new"] ["ell"]
  ["arr",flat,shape] ["barr",flatbool,shape] ["b0",bool] ["a0",n]
value (setter): ["scalar",n] | ["arr",flat,shape]

Base recipes (mirrored in C#): S,V0,V1,V6,A,AT,ARS,ACS,ANR,ANC,ASO,ABC,B,BT,E03
(arange-filled; views built via the same slice/transpose/broadcast ops). All data int64 so
values compare exactly as int64; the dtype sweep re-encodes per dtype (complex -> re/im pairs,
bool -> 0/1, half -> double).

Regenerate (deterministic; needs numpy==2.4.2):
  python test/oracle/gen_index_oracle.py
then `dotnet build` (the csproj glob copies Fuzz/corpus/**/*.jsonl to test output).
"""
import numpy as np, json, itertools, os, random

RANDOM_SEED = 20240626   # pinned; surfaces in the random corpus filename for reproducibility

# ---------------- base array recipes (mirrored in C#) ----------------
def make_base(name):
    A  = np.arange(12).reshape(3,4)
    B  = np.arange(24).reshape(2,3,4)
    return {
        "S":   np.array(5),
        "V0":  np.arange(0),
        "V1":  np.arange(1),
        "V6":  np.arange(6),
        "A":   A.copy(),
        "AT":  A.T,                                    # transposed view (4,3)
        "ARS": A[::2],                                 # row-strided (2,4)
        "ACS": A[:, ::2],                              # col-strided (3,2)
        "ANR": A[::-1],                                # neg-row (3,4)
        "ANC": A[:, ::-1],                             # neg-col (3,4)
        "ASO": A[1:],                                  # sliced-offset (2,4)
        "ABC": np.broadcast_to(np.arange(4), (3,4)),  # broadcast (3,4)
        "B":   B.copy(),
        "BT":  B.T,                                    # (4,3,2)
        "E03": np.zeros((0,3), dtype=np.int64),        # empty 2-D
    }[name]

# dtype sweep bases: arange-filled (integer-valued so float/complex/half are exact)
DT = {"bool":np.bool_,"uint8":np.uint8,"int8":np.int8,"int16":np.int16,"uint16":np.uint16,
      "int32":np.int32,"uint32":np.uint32,"int64":np.int64,"uint64":np.uint64,
      "float16":np.float16,"float32":np.float32,"float64":np.float64,"complex128":np.complex128}
def make_dtype_base(dt):
    base = np.arange(12).reshape(3,4)
    if dt == "bool": return (base % 2).astype(np.bool_)
    return base.astype(DT[dt])

# ---------------- token -> numpy index ----------------
def tok_to_np(t):
    k = t[0]
    if k=="int":   return t[1]
    if k=="slice": return slice(t[1],t[2],t[3])
    if k=="new":   return None
    if k=="ell":   return Ellipsis
    if k=="arr":   return np.array(t[1], dtype=np.intp).reshape(t[2])
    if k=="barr":  return np.array(t[1], dtype=bool).reshape(t[2])
    if k=="b0":    return np.array(bool(t[1]))
    if k=="a0":    return np.array(t[1], dtype=np.intp)
    raise ValueError(k)

def np_index(tokens):
    return tuple(tok_to_np(t) for t in tokens)

def val_to_np(v):
    # v = ["scalar", n] | ["arr", flat, shape]
    if v[0]=="scalar": return np.int64(v[1])
    return np.array(v[1], dtype=np.int64).reshape(v[2])

def ravel_i64(a):
    return [int(x) for x in np.asarray(a).ravel(order="C").tolist()]

# ---------------- evaluate one case in numpy ----------------
def eval_get(base_name, tokens):
    try:
        b = make_base(base_name)
        r = b[np_index(tokens)]
        r = np.asarray(r)
        return {"ok":True, "shape":list(r.shape), "vals":ravel_i64(r)}
    except Exception as e:
        return {"ok":False, "err":type(e).__name__}

def eval_set(base_name, tokens, value):
    try:
        b = make_base(base_name).copy()
        b[np_index(tokens)] = val_to_np(value)
        return {"ok":True, "shape":list(b.shape), "vals":ravel_i64(b)}
    except Exception as e:
        return {"ok":False, "err":type(e).__name__}

def eval_get_dtype(dt, tokens):
    try:
        b = make_dtype_base(dt)
        r = np.asarray(b[np_index(tokens)])
        if dt=="complex128":
            vals=[]
            for x in r.ravel(order="C").tolist(): vals += [int(x.real), int(x.imag)]
        elif dt=="bool":
            vals=[1 if x else 0 for x in r.ravel(order="C").tolist()]
        else:
            vals=[int(x) for x in np.real(r).ravel(order="C").tolist()]
        return {"ok":True,"shape":list(r.shape),"vals":vals}
    except Exception as e:
        return {"ok":False,"err":type(e).__name__}

# ---------------- index-form generators ----------------
SLICES = [[None,None,None],[1,3,None],[None,None,2],[None,None,-1],[1,None,None],
          [None,2,None],[-2,None,None],[1,1,None],[3,0,-1],[None,None,3]]
def S(sd): return ["slice"]+sd
INTS   = [["int",0],["int",1],["int",-1],["int",2]]
BASIC2 = [["int",0],["int",1],["int",-1]] + [S(s) for s in SLICES[:6]] + [["new"],["ell"]]

curated=[]
def addg(base, tokens, tag):
    curated.append({"op":"get","base":base,"tokens":tokens,"tag":tag,"np":eval_get(base,tokens)})
def adds(base, tokens, value, tag):
    curated.append({"op":"set","base":base,"tokens":tokens,"value":value,"tag":tag,"np":eval_set(base,tokens,value)})

# 1) BASIC indexing — exhaustive small tuples on V/A/B and every layout
for base in ["V6","V1","V0","S"]:
    for t in [["int",0],["int",-1],["int",1],["int",5],["int",6],["int",-7],
              S([None,None,None]),S([1,3,None]),S([None,None,2]),S([None,None,-1]),
              S([1,1,None]),S([2,None,None]),S([-2,None,None]),["new"],["ell"]]:
        addg(base,[t],"basic1")
    addg(base,[],"emptytuple")
    addg(base,[["ell"]],"ell")
    addg(base,[["new"]],"new")

LAYOUTS2 = ["A","AT","ARS","ACS","ANR","ANC","ASO","ABC"]
for base in LAYOUTS2:
    addg(base,[],"emptytuple")
    for t in BASIC2:
        addg(base,[t],"basic1")
    # length-2 tuples (full cross of a compact alphabet)
    alpha = [["int",0],["int",1],["int",-1],S([None,None,None]),S([1,3,None]),
             S([None,None,2]),S([None,None,-1]),["new"],["ell"]]
    for a,b in itertools.product(alpha, alpha):
        addg(base,[a,b],"basic2")
    # a few length-3 (over-index + newaxis/ellipsis interplay)
    for combo in [[["int",0],S([None,None,None]),["new"]],
                  [S([None,None,None]),["new"],S([None,None,None])],
                  [["ell"],["int",-1]],
                  [["new"],["ell"],["new"]],
                  [S([None,None,None]),S([None,None,None]),S([None,None,None])],
                  [["int",0],["int",0],["int",0]]]:
        addg(base,combo,"basic3")

# 3-D
for combo_len in [1,2,3,4]:
    alpha3 = [["int",0],["int",1],["int",-1],S([None,None,None]),S([1,None,None]),
              S([None,None,-1]),["new"],["ell"]]
    # sample combos to bound size
    for ci, combo in enumerate(itertools.product(alpha3, repeat=combo_len)):
        if combo_len>=3 and (ci%4!=0):  # thin the explosion deterministically (by position)
            continue
        addg("B", list(combo), f"b{combo_len}")

# 2) FANCY indexing
FARR = {
 "f01":["arr",[0,1],[2]], "f02":["arr",[0,2],[2]], "f03":["arr",[2,0,1],[3]],
 "f04":["arr",[-1,-2],[2]], "f05":["arr",[0,-1,2],[3]], "f06":["arr",[1,1,1],[3]],
 "f07":["arr",[2],[1]], "f08":["arr",[],[0]], "f09":["arr",[5,4,3,2,1,0],[6]],
 "f2d":["arr",[0,1,1,0],[2,2]], "f3d":["arr",[0,1,1,0,1,0,0,1],[2,2,2]],
 "f21":["arr",[0,2],[2,1]], "f12":["arr",[0,2],[1,2]],
 "foob":["arr",[3],[1]], "fnegoob":["arr",[-4],[1]],
}
for base in ["V6","A","B"]:
    for nm,fa in FARR.items():
        addg(base,[fa],"fancy1:"+nm)
# multi-fancy (broadcast)
for base in ["A","B"]:
    for a in ["f01","f02","f21","f12","f2d"]:
        for b in ["f01","f02","f21","f12"]:
            addg(base,[FARR[a],FARR[b]],f"fancy2:{a},{b}")
# fancy + slice (mixed)
for base in LAYOUTS2:
    addg(base,[S([None,None,None]),FARR["f02"]],"mix:slice,fancy")
    addg(base,[FARR["f01"],S([None,None,None])],"mix:fancy,slice")
addg("B",[FARR["f01"],S([None,None,None]),FARR["f02"]],"mix:f,slice,f")
addg("B",[S([None,None,None]),FARR["f01"],FARR["f02"]],"mix:slice,f,f")
addg("B",[FARR["f01"],FARR["f02"],S([None,None,None])],"mix:f,f,slice")
# 0-d integer array
for base in ["A","B"]:
    addg(base,[["a0",1]],"a0")
    addg(base,[["a0",1],S([None,None,None])],"a0,slice")

# 3) BOOLEAN indexing
def mask_for(shape):
    n=int(np.prod(shape)); return ["barr",[(i%2==0) for i in range(n)], list(shape)]
addg("V6",[mask_for([6])],"bool:full")
addg("A",[mask_for([3,4])],"bool:full")
addg("A",[mask_for([3])],"bool:prefix")          # axis-0 mask
addg("B",[mask_for([2,3])],"bool:prefix2")
addg("A",[["barr",[True,False,True],[3]]],"bool:rows")
# 0-d bool combined with basic (HAS_0D_BOOL)
for base in ["A","B"]:
    for combo in [[["b0",True]],[["b0",False]],
                  [["b0",True],S([None,None,None])],
                  [S([None,None,None]),["b0",True]],
                  [["b0",False],S([None,None,None])],
                  [["int",1],["b0",True]],
                  [["b0",True],["int",1]],
                  [["b0",True],["b0",True]],
                  [["b0",True],["b0",False]]]:
        addg(base,combo,"bool0d")
# bool mask + basic
addg("A",[["barr",[True,False,True],[3]],["int",1]],"bool+int")
addg("A",[S([None,None,None]),["barr",[True,False,True,False],[4]]],"slice+boolcol")

# 4) SETTERS — mirror a representative slice of the above with values
SV = {"sc":["scalar",-1], "m4":["arr",[10,20,30,40],[4]], "m24":["arr",list(range(100,108)),[2,4]],
      "row1":["arr",[10],[1]], "bad2":["arr",[1,2],[2]], "bad3":["arr",[1,2,3],[3]],
      "m6":["arr",list(range(6)),[6]], "m34":["arr",list(range(100,112)),[3,4]]}
for base in ["V6","A","B"]+LAYOUTS2:
    adds(base,[["int",0]],SV["sc"],"set:int=scalar")
    adds(base,[S([None,None,None])],SV["sc"],"set:colon=scalar")
adds("A",[["int",0]],SV["m4"],"set:row=matched")
adds("A",[["int",0]],SV["bad2"],"set:row=bad2")
adds("A",[["int",0]],SV["bad3"],"set:row=bad3")
adds("A",[S([None,None,None])],SV["m34"],"set:all=matched")
adds("A",[S([None,None,None])],SV["m4"],"set:all=bcastrow")
adds("A",[S([None,None,None])],SV["m6"],"set:all=bad6")
adds("A",[FARR["f02"]],SV["sc"],"set:fancy=scalar")
adds("A",[FARR["f02"]],SV["m24"],"set:fancy=matched")
adds("A",[FARR["f02"]],SV["m4"],"set:fancy=bcastrow")
adds("A",[FARR["f01"],FARR["f02"]],["arr",[7,8],[2]],"set:multifancy=matched")
adds("A",[mask_for([3,4])],SV["sc"],"set:mask=scalar")
adds("A",[["barr",[True,False,True],[3]]],SV["sc"],"set:rowmask=scalar")
adds("A",[["b0",True],S([None,None,None])],SV["sc"],"set:0dbool=scalar")
adds("A",[["b0",False],S([None,None,None])],SV["sc"],"set:0dboolF=scalar")
for base in LAYOUTS2:
    adds(base,[S([None,None,None]),FARR["f02"]],SV["sc"],"set:slice,fancy=scalar")

# 4b) G6 (F6) — E03 (empty (0,3)): the recipe existed on both sides but had ZERO cases, so
# empty-array indexing (get AND set) was entirely ungated. Appended after the setter block so
# every pre-existing curated ordinal (and therefore case id) stays stable.
addg("E03",[],"emptytuple")                                   # -> (0,3) itself
addg("E03",[S([None,None,None])],"basic1")                    # [:]      -> (0,3)
addg("E03",[["int",0]],"basic1")                              # [0]      -> IndexError
addg("E03",[S([None,None,None]),["int",1]],"basic2")          # [:, 1]   -> (0,)
addg("E03",[mask_for([0,3])],"bool:full")                     # mask(0,3)-> (0,)
addg("E03",[FARR["f08"]],"fancy1:f08")                        # arr []   -> (0,3)
adds("E03",[S([None,None,None])],SV["sc"],"set:colon=scalar") # no-op OK
adds("E03",[["int",0]],SV["sc"],"set:int=scalar")             # err

# 5) DTYPE sweep — a handful of forms across all 13 dtypes
dtype_forms = {
  "int":[["int",1]], "slice":[S([1,3,None])], "negslice":[S([None,None,-1])],
  "fancy":[FARR["f02"]], "mask":[mask_for([3,4])], "rows":[["barr",[True,False,True],[3]]],
  "coord":[["int",1],["int",2]], "newaxis":[["new"],S([None,None,None])],
}
dtype_cases=[]
for dt in DT:
    for fnm,toks in dtype_forms.items():
        dtype_cases.append({"dtype":dt,"tokens":toks,"tag":fnm,"np":eval_get_dtype(dt,toks)})

# 5b) G15 — CROSS-DTYPE setters: the assigned value's dtype differs from the base's, exercising
# cast-on-set (float->int truncation toward zero, int->bool coercion, unsigned modular wrap for
# np-SCALAR values — all probed NumPy 2.4.2; python-int scalars would range-check instead, so the
# oracle assigns np.int64/np.float64). value spec: ["scalar",n] int64 | ["fscalar",x] float64 |
# ["farr",flat,shape] float64. Replayed by IndexOracleTests.Index_SetterDtype.
def setter_val_to_np(v):
    if v[0] == "scalar":  return np.int64(v[1])
    if v[0] == "fscalar": return np.float64(v[1])
    return np.array(v[1], dtype=np.float64).reshape(v[2])

def eval_set_dtype(dt, tokens, value):
    try:
        b = make_dtype_base(dt).copy()
        b[np_index(tokens)] = setter_val_to_np(value)
        if dt == "bool":
            vals = [1 if x else 0 for x in b.ravel(order="C").tolist()]
        else:
            vals = [int(x) for x in np.real(b).ravel(order="C").tolist()]
        return {"ok": True, "shape": list(b.shape), "vals": vals}
    except Exception as e:
        return {"ok": False, "err": type(e).__name__}

SETTER_DTYPE_CASES = [
    ("int32", [["int",0]], ["fscalar", 2.75]),                    # trunc -> 2
    ("int32", [["int",1]], ["fscalar", -3.9]),                    # trunc toward zero -> -3
    ("int32", [["int",0]], ["farr", [1.5, -2.5, 3.9, -0.1], [4]]),
    ("int32", [S([None,None,None])], ["fscalar", 7.5]),
    ("bool",  [["int",0]], ["scalar", 5]),                        # nonzero -> True
    ("bool",  [["int",1]], ["scalar", 0]),                        # zero -> False
    ("bool",  [S([None,None,None])], ["scalar", 3]),
    ("uint8", [["int",0]], ["scalar", -1]),                       # np.int64 wrap -> 255
    ("uint8", [S([None,None,None])], ["scalar", -2]),             # wrap -> 254
    ("uint8", [["int",2]], ["scalar", 300]),                      # wrap -> 44
]
setter_dtype_cases = []
for (dt, toks, val) in SETTER_DTYPE_CASES:
    setter_dtype_cases.append({"op": "set", "dtype": dt, "tokens": toks, "value": val,
                               "tag": "xdtype", "np": eval_set_dtype(dt, toks, val)})

# 6) RANDOM FUZZ — seeded, explores the space far beyond the curated forms
ND = {"V6":1,"V1":1,"V0":1,"S":0,"A":2,"AT":2,"ARS":2,"ACS":2,"ANR":2,"ANC":2,"ASO":2,"ABC":2,
      "B":3,"BT":3,"E03":2}
def rand_tokens(rng, ndim):
    L = rng.randint(0, ndim+2); toks=[]; used_ell=False
    for _ in range(L):
        r = rng.random()
        if r < 0.30:
            toks.append(["int", rng.randint(-6,6)])
        elif r < 0.55:
            rb = lambda: (None if rng.random()<0.35 else rng.randint(-7,7))
            toks.append(["slice", rb(), rb(), rng.choice([None,1,2,-1,-2,3])])
        elif r < 0.72:
            k=rng.randint(0,3); toks.append(["arr",[rng.randint(-5,5) for _ in range(k)],[k]])
        elif r < 0.80:
            shp=rng.choice([[2,2],[1,4],[4,1],[2,1],[1,2]]); need=1
            for d in shp: need*=d
            toks.append(["arr",[rng.randint(-3,3) for _ in range(need)],shp])
        elif r < 0.88:
            k=rng.randint(0,5); toks.append(["barr",[rng.random()<0.5 for _ in range(k)],[k]])
        elif r < 0.93:
            toks.append(["b0", rng.random()<0.5])
        elif r < 0.97:
            toks.append(["new"])
        else:
            if not used_ell: toks.append(["ell"]); used_ell=True
            else: toks.append(["int", rng.randint(-3,3)])
    return toks

random_get=[]; random_set=[]
def raddg(base, tokens, tag):
    random_get.append({"op":"get","base":base,"tokens":tokens,"tag":tag,"np":eval_get(base,tokens)})
def radds(base, tokens, value, tag):
    random_set.append({"op":"set","base":base,"tokens":tokens,"value":value,"tag":tag,"np":eval_set(base,tokens,value)})
def random_fuzz(seed, ng, ns):
    rng = random.Random(seed)
    # G6: E03 (empty 2-D) + V0 (empty 1-D) joined the getter pool, E03 the setter pool —
    # empty bases were previously absent from the whole random space.
    gpool=["V6","A","B","AT","ARS","ACS","ANR","ANC","ASO","ABC","BT","V1","S","E03","V0"]
    spool=["V6","A","B","ARS","ANR","ASO","ACS","E03"]
    for _ in range(ng):
        base=rng.choice(gpool); raddg(base, rand_tokens(rng, ND[base]), "rand")
    for _ in range(ns):
        base=rng.choice(spool); toks=rand_tokens(rng, ND[base])
        if rng.random()<0.5: val=["scalar", rng.randint(-9,9)]
        else:
            k=rng.randint(0,6); val=["arr",[rng.randint(0,99) for _ in range(k)],[k]]
        radds(base, toks, val, "rand")
random_fuzz(RANDOM_SEED, 7000, 3000)

# ---------------- write JSONL corpora ----------------
def write_jsonl(path, cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for i, c in enumerate(cases):
            row = dict(c)
            # stable id: op/base-or-dtype/tag/ordinal — appears in C# failure messages
            key = row.get("base", row.get("dtype", "?"))
            row["id"] = f"{row['op'] if 'op' in row else 'dget'}/{key}/{row['tag']}/{i}"
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

here = os.path.dirname(os.path.abspath(__file__))
corpus_dir = os.path.normpath(os.path.join(here, "..", "NumSharp.UnitTest", "Fuzz", "corpus"))
write_jsonl(os.path.join(corpus_dir, "index_curated.jsonl"), curated)
write_jsonl(os.path.join(corpus_dir, "index_dtype.jsonl"),   dtype_cases)
write_jsonl(os.path.join(corpus_dir, "index_setter_dtype.jsonl"), setter_dtype_cases)
write_jsonl(os.path.join(corpus_dir, f"index_random_{RANDOM_SEED}.jsonl"), random_get + random_set)

def nok(cs): return sum(1 for c in cs if c["np"]["ok"])
print(f"curated={len(curated)} (np_ok={nok(curated)} np_err={len(curated)-nok(curated)})")
print(f"dtype={len(dtype_cases)} (np_ok={nok(dtype_cases)})")
print(f"setter_dtype={len(setter_dtype_cases)} (np_ok={nok(setter_dtype_cases)})")
rnd = random_get + random_set
print(f"random[seed={RANDOM_SEED}]={len(rnd)} (get={len(random_get)} set={len(random_set)} np_ok={nok(rnd)} np_err={len(rnd)-nok(rnd)})")
print(f"-> {corpus_dir}")
