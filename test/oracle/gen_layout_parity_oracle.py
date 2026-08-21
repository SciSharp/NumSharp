"""Layout-parity oracle: NumPy 2.4.2 differential corpus for the modelled numpy-internal
representations (the formerly-[Misaligned] set, fixed on journey3):

  reshape   _reshape_with_copy_arg + _attempt_nocopy_reshape: nocopy strided VIEWS,
            views of the internal copy, order C/F/A resolution, -1, 0-d, empty, offset
  sortlike  np.sort / np.argsort / np.partition = copy(order='K') + in-place: KEEPORDER
            layouts (F stays F, 3-D transpose keeps its stride order, broadcast -> F-ish)
  join      np.concatenate / np.stack output layout = PyArray_CreateMultiSortedStridePerm
  nonzero   columns of ONE shared (count, ndim) multi-index buffer (+ where/argwhere/
            flatnonzero value twins)
  copyk     copy(order='K') / astype(order='K') true KEEPORDER allocation
  linspace  the in-place-mutated arange VIEW (owndata=False) vs owning astype copies
  reduce    full reductions return read-only numpy SCALARS (PyArray_Return, num=263)
  bcastw    views of a setflags(write=True)-re-enabled broadcast inherit the override

Unlike the flags oracle (flags records only), every case here also records the result's
SHAPE, byte STRIDES, OWNDATA/WRITEABLE, an exact shares-memory verdict against the case's
source, and the result VALUES as base64(tobytes(order='C')) — so the C# replay
(LayoutParityOracleTests) is bit-exact on values AND layout. Optional fields are null
where a surface is knowingly not byte-comparable (partition's between-anchor arrangement)
or where a PRE-EXISTING unrelated divergence would otherwise leak in (argwhere/flatnonzero
report values only — their owndata/layout is the documented base-of-transpose residue).

Regenerate:  python test/oracle/gen_layout_parity_oracle.py   (needs numpy==2.4.2)
Twin:        test/NumSharp.Tests/Backends/LayoutParityOracleTests.cs (builders keyed 1:1)
"""
import base64
import json
import os

import numpy as np

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "NumSharp.Tests", "Backends", "corpus", "layout_parity_oracle.jsonl")

DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
          "int64", "uint64", "float16", "float32", "float64", "complex128"]


# ---------------------------------------------------------------- sources (twin: Src())
def SRC(name, dt="int64"):
    dt = np.dtype(dt)
    if name == "c2d":    return np.arange(12).astype(dt).reshape(3, 4)
    if name == "f2d":    return np.asfortranarray(np.arange(12).astype(dt).reshape(3, 4))
    if name == "t2d":    return np.arange(12).astype(dt).reshape(3, 4).T
    if name == "st":     return np.arange(24).astype(dt).reshape(3, 8)[:, ::2]
    if name == "neg1d":  return np.arange(12).astype(dt)[::-1]
    if name == "negrow": return np.arange(12).astype(dt).reshape(3, 4)[::-1]
    if name == "bc":     return np.broadcast_to(np.arange(3).astype(dt), (4, 3))
    if name == "off":    return np.arange(20).astype(dt)[4:16].reshape(3, 4)
    if name == "t3d":    return np.transpose(np.arange(24).astype(dt).reshape(2, 3, 4), (2, 0, 1))
    if name == "hi5d":   return np.arange(32).astype(dt).reshape(2, 2, 2, 2, 2)
    if name == "e03":    return np.zeros((0, 3), dtype=dt)
    if name == "one":    return np.arange(1).astype(dt).reshape(1, 1)
    if name == "sc":     return np.arange(6).astype(dt)[:1].reshape(())
    # value-bearing zero-mix sources for the nonzero family
    if name == "c2z":    return (np.arange(12).astype(dt).reshape(3, 4) % 3).astype(dt)
    if name == "f2z":    return np.asfortranarray((np.arange(12) % 3).astype(dt).reshape(3, 4))
    if name == "z1d":    return np.array([0, 1, 2, 0, 3], dtype=dt)
    if name == "z3d":    return (np.arange(8) % 2).astype(dt).reshape(2, 2, 2)
    if name == "zall":   return np.zeros((3, 4), dtype=dt)
    if name == "zneg":   return (np.arange(12) % 3).astype(dt).reshape(3, 4)[::-1]
    if name == "zbc":    return np.broadcast_to(np.array([0, 1, 0]).astype(dt), (4, 3))
    raise KeyError(name)


TGT = {  # reshape target tokens (twin: Tgt())
    "12": (12,), "26": (2, 6), "62": (6, 2), "43": (4, 3), "34": (3, 4),
    "223": (2, 2, 3), "232": (2, 3, 2), "m1": (-1,), "46": (4, 6),
    "2223": (2, 2, 2, 3), "48": (4, 8), "32": (32,), "0": (0,), "30": (3, 0),
    "scalar": (), "1": (1,), "11": (1, 1), "111": (1, 1, 1), "24": (24,),
}


def rec(fam, key, dt, res, src=None, vals=True, layout=True, w=None, samebase=None):
    f = res.flags
    r = {"id": f"{fam}/{key}/{dt}", "fam": fam, "key": key, "dtype": dt,
         "shape": [int(x) for x in np.shape(res)]}
    if layout:
        r["strides"] = [int(x) for x in res.strides]
        r["num"] = int(f.num) & 0x7FFFFFFF
        r["own"] = 1 if f.owndata else 0
    else:
        r["strides"] = None; r["num"] = None; r["own"] = None
    r["w"] = (1 if f.writeable else 0) if w is None else w
    r["shares"] = None if src is None else (1 if np.shares_memory(res, src) else 0)
    r["samebase"] = samebase
    r["vals"] = base64.b64encode(np.ascontiguousarray(res).tobytes()).decode() if vals else None
    return r


def main():
    cases = []

    # ---------------------------------------------------------------- reshape
    # (src, tgt-token, order) — curated so every _reshape_with_copy_arg route appears:
    # same-shape view, contiguous relabel (C and F), nocopy combine/split (positive,
    # negative and zero strides, offset windows, 5-D), and the copy-in-order fallback.
    reshape_cells = [
        ("c2d", "12", "C"), ("c2d", "26", "C"), ("c2d", "26", "F"), ("c2d", "223", "C"),
        ("c2d", "m1", "C"), ("c2d", "34", "C"), ("c2d", "26", "A"), ("c2d", "12", "F"),
        ("f2d", "12", "C"), ("f2d", "12", "F"), ("f2d", "12", "A"), ("f2d", "26", "F"),
        ("f2d", "26", "C"), ("f2d", "62", "F"),
        ("t2d", "12", "A"), ("t2d", "12", "C"), ("t2d", "26", "F"), ("t2d", "m1", "C"),
        ("st", "26", "C"), ("st", "12", "C"), ("st", "62", "C"), ("st", "43", "C"),
        ("st", "223", "C"), ("st", "26", "F"), ("st", "34", "C"),
        ("neg1d", "34", "C"), ("neg1d", "26", "C"), ("neg1d", "12", "C"), ("neg1d", "232", "C"),
        ("negrow", "12", "C"), ("negrow", "34", "C"), ("negrow", "26", "C"),
        ("bc", "12", "C"), ("bc", "223", "C"), ("bc", "43", "C"),
        ("off", "12", "C"), ("off", "26", "C"), ("off", "26", "F"),
        ("t3d", "24", "C"), ("t3d", "46", "C"), ("t3d", "2223", "C"),
        ("hi5d", "48", "C"), ("hi5d", "32", "C"),
        ("e03", "0", "C"), ("e03", "30", "C"), ("e03", "0", "F"),
        ("one", "scalar", "C"), ("one", "111", "C"), ("one", "1", "C"),
        ("sc", "1", "C"), ("sc", "scalar", "C"), ("sc", "11", "C"),
    ]
    for s, t, o in reshape_cells:
        src = SRC(s)
        res = src.reshape(TGT[t], order=o)
        cases.append(rec("reshape", f"{s}_{t}_{o}", "int64", res, src=src))
    # dtype sweep on the flagship nocopy-view cell
    for dt in DTYPES:
        src = SRC("st", dt)
        cases.append(rec("reshape", "st_26_C", dt, src.reshape((2, 6), order="C"), src=src))

    # ---------------------------------------------------------------- sortlike
    sort_cells = [("c2d", -1), ("f2d", -1), ("f2d", 0), ("t2d", -1), ("t2d", 0),
                  ("st", -1), ("st", 0), ("negrow", -1), ("bc", -1), ("bc", 0),
                  ("t3d", -1), ("t3d", 0), ("off", -1)]
    for s, ax in sort_cells:
        src = SRC(s)
        cases.append(rec("sort", f"{s}_ax{ax}", "int64", np.sort(src, axis=ax), src=src))
    cases.append(rec("sort", "f2d_flat", "int64", np.sort(SRC("f2d"), axis=None), src=SRC("f2d")))
    cases.append(rec("sort", "st_flat", "int64", np.sort(SRC("st"), axis=None), src=SRC("st")))
    for dt in ["bool", "uint8", "int32", "float16", "float32", "float64"]:
        src = SRC("f2d", dt)
        cases.append(rec("sort", "f2d_ax-1", dt, np.sort(src, axis=-1), src=src))
    # nan_f: layout + NaN-goes-last pinned; VALUES excluded — NumPy's SIMD sort canonicalizes
    # every NaN to the positive quiet 0x7ff8… while NumSharp's radix emits .NET's negative
    # 0xfff8… (the long-documented divergence the set-ops NaN-rewrite pass exists for; the
    # C# replay asserts NaN-ness positionally instead).
    nanarr = np.asfortranarray(np.array([[3.0, np.nan, 1.0, 2.0]] * 3))
    cases.append(rec("sort", "nan_f", "float64", np.sort(nanarr, axis=-1), src=nanarr, vals=False))
    for s, ax in [("f2d", -1), ("t3d", 0), ("st", -1), ("bc", -1)]:
        src = SRC(s)
        cases.append(rec("argsort", f"{s}_ax{ax}", "int64", np.argsort(src, axis=ax), src=src))
    # partition: layout is contractual, the between-anchor arrangement is NOT (see
    # Fuzz/README) — record layout + the kth slice (np.take at kth along the axis).
    for s, kth, ax in [("f2d", 2, -1), ("st", 1, 0), ("t3d", 2, -1), ("c2d", 2, -1)]:
        src = SRC(s)
        res = np.partition(src, kth, axis=ax)
        r = rec("partition", f"{s}_k{kth}_ax{ax}", "int64", res, src=src, vals=False)
        r["kthvals"] = base64.b64encode(
            np.ascontiguousarray(np.take(res, kth, axis=ax)).tobytes()).decode()
        cases.append(r)

    # ---------------------------------------------------------------- join (concat/stack)
    def two(name): return SRC(name), SRC(name)
    join_cells = [
        ("stack", "f_ax0", lambda: np.stack(two("f2d"), axis=0)),
        ("stack", "f_ax1", lambda: np.stack(two("f2d"), axis=1)),
        ("stack", "f_ax2", lambda: np.stack(two("f2d"), axis=2)),
        ("stack", "c_ax0", lambda: np.stack(two("c2d"), axis=0)),
        ("stack", "t_ax1", lambda: np.stack(two("t2d"), axis=1)),
        ("stack", "st_ax0", lambda: np.stack(two("st"), axis=0)),
        ("stack", "n1_ax0", lambda: np.stack(two("neg1d"), axis=0)),
        ("stack", "n1_ax1", lambda: np.stack(two("neg1d"), axis=1)),
        ("stack", "t3_ax0", lambda: np.stack(two("t3d"), axis=0)),
        ("concat", "ff_ax0", lambda: np.concatenate(two("f2d"), axis=0)),
        ("concat", "ff_ax1", lambda: np.concatenate(two("f2d"), axis=1)),
        ("concat", "cc_ax0", lambda: np.concatenate(two("c2d"), axis=0)),
        ("concat", "cf_ax0", lambda: np.concatenate([SRC("c2d"), SRC("f2d")], axis=0)),
        ("concat", "tt_ax0", lambda: np.concatenate(two("t2d"), axis=0)),
        ("concat", "stst_ax1", lambda: np.concatenate(two("st"), axis=1)),
        ("concat", "negneg_ax0", lambda: np.concatenate(two("negrow"), axis=0)),
        ("concat", "col_ax1", lambda: np.concatenate(
            [np.arange(3).astype(np.int64).reshape(3, 1)] * 2, axis=1)),
        ("concat", "e_ax0", lambda: np.concatenate(two("e03"), axis=0)),
        ("concat", "three_ax0", lambda: np.concatenate([SRC("f2d")] * 3, axis=0)),
        ("concat", "bcbc_ax0", lambda: np.concatenate(two("bc"), axis=0)),
        ("concat", "mixdt_ax0", lambda: np.concatenate(
            [np.arange(12).astype(np.int32).reshape(3, 4), SRC("c2d")], axis=0)),
        ("concat", "flatnone", lambda: np.concatenate([SRC("f2d"), SRC("st")], axis=None)),
    ]
    for fam, key, build in join_cells:
        res = build()
        cases.append(rec(fam, key, str(res.dtype), res))

    # ---------------------------------------------------------------- nonzero family
    nz_cells = ["c2z", "f2z", "z1d", "z3d", "zall", "e03", "zneg", "zbc"]
    for s in nz_cells:
        src = SRC(s)
        nz = np.nonzero(src)
        for d in range(len(nz)):
            cases.append(rec("nonzero", f"{s}_{d}", "int64", nz[d],
                             samebase=1 if nz[d].base is nz[0].base else 0))
    for dt in ["bool", "uint8", "float64"]:
        nz = np.nonzero(SRC("c2z", dt))
        cases.append(rec("nonzero", "c2z_0", f"src_{dt}", nz[0], samebase=1))
    w1 = np.where(SRC("c2z") > 0)
    cases.append(rec("where1", "c2z_0", "int64", w1[0]))
    cases.append(rec("where1", "c2z_1", "int64", w1[1]))
    # argwhere/flatnonzero: VALUES only (their layout rides the documented
    # base-of-transpose / owning-engine residue, deliberately not pinned here)
    cases.append(rec("argwhere", "c2z", "int64", np.argwhere(SRC("c2z")), layout=False, w=None))
    cases.append(rec("flatnonzero", "c2z", "int64", np.flatnonzero(SRC("c2z")), layout=False, w=None))

    # ---------------------------------------------------------------- copy/astype K
    for s in ["t3d", "bc", "st", "negrow", "f2d"]:
        src = SRC(s)
        cases.append(rec("copyk", f"{s}_copy", "int64", src.copy(order="K"), src=src))
        cases.append(rec("copyk", f"{s}_astype", "float64",
                         src.astype(np.float64, order="K", copy=True), src=src))

    # ---------------------------------------------------------------- linspace
    lin_cells = [
        ("f64_5", lambda: np.linspace(0.0, 1.0, 5)),
        ("f64_2_3", lambda: np.linspace(2.0, 3.0, 5)),
        ("f64_11", lambda: np.linspace(0.0, 10.0, 11)),
        ("f64_num1", lambda: np.linspace(0.0, 1.0, 1)),
        ("f64_num1_noep", lambda: np.linspace(0.0, 1.0, 1, endpoint=False)),
        ("f64_num0", lambda: np.linspace(0.0, 1.0, 0)),
        ("f64_num2", lambda: np.linspace(0.0, 1.0, 2)),
        ("f64_noep", lambda: np.linspace(0.0, 1.0, 4, endpoint=False)),
        ("f32_5", lambda: np.linspace(0, 1, 5, dtype=np.float32)),
        ("i64_5", lambda: np.linspace(0, 10, 5, dtype=np.int64)),
    ]
    for key, build in lin_cells:
        res = build()
        # i64_5: layout/owndata pinned, VALUES excluded — NumPy floors the float lattice
        # (…7.5→7) where NumSharp's Converts.ToInt64 rounds half-to-even (7.5→8); a
        # pre-existing linspace int-dtype divergence outside this oracle's scope.
        cases.append(rec("linspace", key, str(res.dtype), res, vals=(key != "i64_5")))

    # ---------------------------------------------------------------- reductions -> scalars
    A = lambda dt="int64": np.arange(12).astype(dt).reshape(3, 4)
    A1 = lambda dt="int64": np.arange(3).astype(dt)
    red_cells = [
        ("sum_flat", lambda dt: np.sum(A(dt))), ("prod_flat", lambda dt: np.prod(A1(dt))),
        ("mean_flat", lambda dt: np.mean(A(dt))), ("std_flat", lambda dt: np.std(A(dt))),
        ("var_flat", lambda dt: np.var(A(dt))), ("amin_flat", lambda dt: np.amin(A(dt))),
        ("amax_flat", lambda dt: np.amax(A(dt))), ("median_flat", lambda dt: np.median(A(dt))),
        ("ptp_flat", lambda dt: np.ptp(A(dt))), ("trace_flat", lambda dt: np.trace(A(dt))),
        ("argmax_flat", lambda dt: np.argmax(A(dt))),
        ("sum_ax0_1d", lambda dt: np.sum(A1(dt), axis=0)),
        ("argmax_ax0_1d", lambda dt: np.argmax(A1(dt), axis=0)),
        ("percentile50", lambda dt: np.percentile(A(dt), 50)),
        ("quantile50", lambda dt: np.quantile(A(dt), 0.5)),
    ]
    for key, build in red_cells:
        for dt in ["int64", "float64"]:
            raw = build(dt)
            f = raw.flags
            r = {"id": f"reduce/{key}/{dt}", "fam": "reduce", "key": key, "dtype": dt,
                 "shape": [int(x) for x in np.shape(raw)],
                 "strides": [int(x) for x in raw.strides],
                 "num": int(f.num) & 0x7FFFFFFF, "own": 1 if f.owndata else 0,
                 "w": 1 if f.writeable else 0, "shares": None, "samebase": None,
                 "vals": base64.b64encode(np.asarray(raw).tobytes()).decode(),
                 "rdtype": str(np.asarray(raw).dtype)}
            cases.append(r)
    # nan-family + keepdims/out contrast rows (int64 only where numpy defines them)
    nanA = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    for key, res in [("nansum_flat", np.nansum(nanA)), ("nanmean_flat", np.nanmean(nanA)),
                     ("nanmax_flat", np.nanmax(nanA)), ("nanmedian_flat", np.nanmedian(nanA))]:
        f = res.flags
        cases.append({"id": f"reduce/{key}/float64", "fam": "reduce", "key": key,
                      "dtype": "float64", "shape": [], "strides": [],
                      "num": int(f.num) & 0x7FFFFFFF, "own": 1 if f.owndata else 0,
                      "w": 1 if f.writeable else 0, "shares": None, "samebase": None,
                      "vals": base64.b64encode(np.asarray(res).tobytes()).decode(),
                      "rdtype": str(np.asarray(res).dtype)})
    kd = np.sum(A(), keepdims=True)
    cases.append(rec("reduce", "sum_keepdims", "int64", kd))
    kd0 = np.sum(np.array(5, dtype=np.int64), keepdims=True)
    cases.append(rec("reduce", "sum_keepdims_0d", "int64", kd0))
    o = np.array(0, dtype=np.int64)
    ro = np.sum(A(), out=o)
    cases.append(rec("reduce", "sum_out0d", "int64", ro))
    # all/any: WRITEABLE pinned; num/own skipped (0-d generic-wrapper owndata residue)
    for key, res in [("all_flat", np.all(A() > -1)), ("any_flat", np.any(A() > 5)),
                     ("all_ax0_1d", np.all(A1() > -1, axis=0)), ("any_ax0_1d", np.any(A1() > 1, axis=0))]:
        cases.append({"id": f"reduce/{key}/bool", "fam": "reduce", "key": key, "dtype": "bool",
                      "shape": [], "strides": None, "num": None, "own": None,
                      "w": 1 if res.flags.writeable else 0, "shares": None, "samebase": None,
                      "vals": base64.b64encode(np.asarray(res).tobytes()).decode(),
                      "rdtype": "bool"})

    # ---------------------------------------------------------------- bcast writeable inherit
    def BCW():
        b = np.broadcast_to(np.arange(3).astype(np.int64), (4, 3))
        b.flags.writeable = True
        return b
    bcw_cells = [
        ("slice", lambda: BCW()[1:3]), ("T", lambda: BCW().T),
        ("step", lambda: BCW()[:, ::2]), ("squeeze", lambda: np.squeeze(BCW()[None])),
        ("row", lambda: BCW()[0]), ("chain", lambda: BCW()[1:3][0:1]),
        ("plain_slice", lambda: SRC("bc")[1:3]), ("plain_row", lambda: SRC("bc")[0]),
        ("rebroadcast", lambda: np.broadcast_to(BCW(), (2, 4, 3))),
    ]
    for key, build in bcw_cells:
        cases.append(rec("bcastw", key, "int64", build()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="\n") as fh:
        for c in cases:
            fh.write(json.dumps(c, separators=(",", ":")) + "\n")
    fams = {}
    for c in cases:
        fams[c["fam"]] = fams.get(c["fam"], 0) + 1
    print(f"wrote {len(cases)} cases -> {OUT}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(fams.items())))


if __name__ == "__main__":
    main()
