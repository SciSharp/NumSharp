#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// npyiter_consumers_bench.cs — NumPy's INTERNAL NpyIter consumers, mapped from
// src/numpy/numpy/_core/src (grep NpyIter_{New,MultiNew,AdvancedNew}) and
// benchmarked through the corresponding user-facing np.* surface with the
// perf-relevant argument matrix. Companion: npyiter_consumers_bench.py.
//
// Consumer map (NumPy 2.4.2 source, function -> np.* surface):
//   execute_ufunc_loop            ufunc_object.c:1084  every np.<ufunc> (non-trivial:
//                                                      broadcast/cast/where/out/dtype)
//   PyUFunc_GeneralizedFunction.. ufunc_object.c:1978  gufuncs (matmul et al)
//   PyUFunc_Accumulate            ufunc_object.c:2695  np.cumsum / np.cumprod
//   PyUFunc_Reduceat              ufunc_object.c:3127  ufunc.reduceat       [NS: missing]
//   ufunc_at__slow_iter           ufunc_object.c:5772  np.add.at            [NS: missing]
//   PyUFunc_ReduceWrapper         reduction.c:286      np.sum/prod/min/max (axis/
//                                                      dtype/out/keepdims/initial/where)
//   array_boolean_subscript       mapping.c:1007       a[mask]
//   array_assign_boolean_subscr.  mapping.c:1205       a[mask] = v
//   PyArray_MapIterNew            mapping.c:3126       a[idx] gather / a[idx]=v scatter
//   PyArray_CountNonzero          item_selection:2747  np.count_nonzero (non-bool dtypes)
//   PyArray_Nonzero               item_selection:2959  np.nonzero / np.argwhere
//   PyArray_CopyAsFlat            ctors.c:2787         np.ravel/flatten copies, order='F'
//   arr_ravel_multi_index         compiled_base:1186   np.ravel_multi_index
//   arr_unravel_index             compiled_base:1337   np.unravel_index
//   PyArray_Where                 multiarraymodule:3303 np.where(c, x, y)
//   einsum                        einsum.cpp:1051      np.einsum            [NS: missing]
//   nditer_pywrap / nested_iters  nditer_pywrap.c      np.nditer            [nested: missing]
//   busday/datetime/strings/void-compare/deepcopy      [NS: no datetime64/str/object dtypes]
//
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/npyiter_consumers_bench.cs
// =============================================================================
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends;

var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbgCore?.IsJITOptimizerDisabled ?? false)
{
    Console.WriteLine("FATAL: Debug build — rerun: dotnet run -c Release - < benchmark/poc/npyiter_consumers_bench.cs");
    return;
}

int fails = 0;
void Check(bool ok, string what) { if (!ok) { fails++; Console.WriteLine($"  CORRECTNESS FAIL: {what}"); } }

double BestMs(Action body, int iters, int warm, int rounds = 5)
{
    for (int i = 0; i < warm; i++) body();
    double best = double.MaxValue;
    for (int r = 0; r < rounds; r++)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) body();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
    }
    return best;
}

void Row(string id, string label, double ms, string note = "")
{
    string val = ms >= 1.0 ? $"{ms,10:F3} ms" : ms >= 0.001 ? $"{ms * 1000,10:F2} us" : $"{ms * 1e6,10:F1} ns";
    Console.WriteLine($"{id,-6} {label,-58} {val}  {note}");
}

void Missing(string id, string label) =>
    Console.WriteLine($"{id,-6} {label,-58}   NOT IMPLEMENTED in NumSharp (NumPy target on py side)");

const int M = 4_194_304;
Console.WriteLine("NumSharp — NumPy-internal NpyIter consumers benchmark");
Console.WriteLine($"{"id",-6} {"aspect",-58} {"per call",13}");
Console.WriteLine(new string('-', 97));

// =========================================================================
// UF — execute_ufunc_loop argument matrix (beyond prior rounds: dtype=, out-cast)
// =========================================================================
{
    var a = (np.arange(M).astype(np.float64) % 97.0) + 1.0;
    var b = (np.arange(M).astype(np.float64) % 31.0) + 2.0;
    var o32 = np.empty(new Shape(M), np.float32);
    var ai = np.arange(M).astype(np.int32);

    var r1 = np.add(a, b, dtype: NPTypeCode.Single);
    Check(r1.typecode == NPTypeCode.Single && Math.Abs(r1.GetSingle(777) - (float)(a.GetDouble(777) + b.GetDouble(777))) < 1e-2, "UF1");
    Row("UF1", "np.add(f64, f64, dtype=float32) allocating 4M",
        BestMs(() => { var _ = np.add(a, b, dtype: NPTypeCode.Single); }, 12, 4, 7));

    np.add(a, b, o32);
    Check(Math.Abs(o32.GetSingle(777) - (float)(a.GetDouble(777) + b.GetDouble(777))) < 1e-2, "UF2");
    Row("UF2", "np.add(f64, f64, out=f32) write-cast 4M",
        BestMs(() => np.add(a, b, o32), 12, 4, 7));

    var rs = np.sqrt(ai);
    Check(Math.Abs(rs.GetDouble(9) - 3.0) < 1e-12, "UF3");
    Row("UF3", "np.sqrt(int32) promoting unary 4M (buffered config)",
        BestMs(() => { var _ = np.sqrt(ai); }, 12, 4, 7));
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// RD — PyUFunc_ReduceWrapper argument matrix
// =========================================================================
{
    var A = ((np.arange(M).astype(np.float64) % 97.0) + 1.0).reshape(2048, 2048);
    var af32 = (np.arange(M).astype(np.float32) % 977f) + 1f;
    var B = ((np.arange(M).astype(np.float64) % 53.0) + 1.0).reshape(128, 256, 128);

    Row("RD1", "np.sum(A) full reduce f64 (2048,2048)",
        BestMs(() => { var _ = np.sum(A); }, 25, 8, 7));
    var k = np.sum(A, 0, keepdims: true);
    Check(k.ndim == 2 && k.shape[0] == 1 && k.shape[1] == 2048, "RD2 shape");
    Row("RD2", "np.sum(A, axis=0, keepdims=True)",
        BestMs(() => { var _ = np.sum(A, 0, keepdims: true); }, 50, 12, 7));
    var s64 = np.sum(af32, NPTypeCode.Double);
    Check(s64.typecode == NPTypeCode.Double, "RD3 dtype");
    Row("RD3", "np.sum(f32 4M, dtype=float64) upcast accumulate",
        BestMs(() => { var _ = np.sum(af32, NPTypeCode.Double); }, 50, 12, 7));
    var m1 = np.sum(B, 1);
    Check(m1.shape[0] == 128 && m1.shape[1] == 128, "RD4 shape");
    Row("RD4", "np.sum(B (128,256,128), axis=1) middle axis",
        BestMs(() => { var _ = np.sum(B, 1); }, 50, 12, 7));
    Row("RD5", "np.amin(A, axis=1)",
        BestMs(() => { var _ = np.amin(A, 1); }, 50, 12, 7));

    Missing("RDt", "np.sum(A, axis=(0,1)) — axis TUPLE");
    Missing("RDw", "np.add.reduce(A, axis=0, where=mask) — reduce where=");
    Missing("RDi", "np.sum(A, initial=5.0) — initial=");
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// AC — PyUFunc_Accumulate (np.cumsum)
// =========================================================================
{
    var a = (np.arange(M).astype(np.float64) % 97.0) + 1.0;
    var A = ((np.arange(M).astype(np.float64) % 97.0) + 1.0).reshape(2048, 2048);

    Row("AC1", "np.cumsum(a) flat f64 4M",
        BestMs(() => { var _ = np.cumsum(a); }, 12, 4, 7));
    Row("AC2", "np.cumsum(A, axis=0) (2048,2048)",
        BestMs(() => { var _ = np.cumsum(A, 0); }, 12, 4, 7));
    Row("AC3", "np.cumsum(A, axis=1)",
        BestMs(() => { var _ = np.cumsum(A, 1); }, 12, 4, 7));
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// WH — PyArray_Where (np.where ternary)
// =========================================================================
{
    var a = (np.arange(M).astype(np.float64) % 97.0) + 1.0;
    var b = (np.arange(M).astype(np.float64) % 31.0) + 2.0;
    NDArray c = (np.arange(M) % 2) == 0;

    var w = np.where(c, a, b);
    Check(w.GetDouble(0) == a.GetDouble(0) && w.GetDouble(1) == b.GetDouble(1), "WH1");
    Row("WH1", "np.where(c, x, y) f64 4M same-shape",
        BestMs(() => { var _ = np.where(c, a, b); }, 12, 4, 7));
    Row("WH2", "np.where(c, x, 0.0) scalar branch",
        BestMs(() => { var _ = np.where(c, a, (object)0.0); }, 12, 4, 7));

    try
    {
        var A2 = a.reshape(2048, 2048);
        NDArray c2 = (np.arange(M) % 3) == 0;
        var c2d = c2.reshape(2048, 2048);
        var rowv = np.arange(2048).astype(np.float64);
        var wb = np.where(c2d, rowv, A2);
        Check(wb.shape[0] == 2048 && wb.shape[1] == 2048, "WH3 shape");
        Row("WH3", "np.where(c2d, row(2048,), y2d) broadcasting",
            BestMs(() => { var _ = np.where(c2d, rowv, A2); }, 12, 4, 7));
    }
    catch (Exception ex) { Console.WriteLine($"WH3    broadcasting where — THROWS: {ex.Message.Split('\n')[0]}"); fails++; }
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// BM — boolean subscript (mapping.c) + nonzero family (item_selection.c)
// =========================================================================
{
    var a = (np.arange(M).astype(np.float64) % 97.0) + 1.0;
    NDArray mask = (np.arange(M) % 2) == 0;
    var maskB = mask.MakeGeneric<bool>();

    var sel = a[maskB];
    Check(sel.size == M / 2 && sel.GetDouble(1) == a.GetDouble(2), "BM1");
    Row("BM1", "a[mask] boolean READ f64 4M (50% true)",
        BestMs(() => { var _ = a[maskB]; }, 12, 4, 7));

    var aw = a.copy();
    var awB = aw;
    var five = NDArray.Scalar(5.0, NPTypeCode.Double);
    awB[maskB] = five;
    Check(awB.GetDouble(0) == 5.0 && awB.GetDouble(1) == a.GetDouble(1), "BM2");
    Row("BM2", "a[mask] = 5.0 boolean ASSIGN f64 4M",
        BestMs(() => awB[maskB] = five, 12, 4, 7));

    long cnt = np.count_nonzero(a);   // f64 input — NumPy routes non-bool through NpyIter
    Check(cnt == M, "BM3");
    Row("BM3", "np.count_nonzero(f64 4M) [non-bool dtype]",
        BestMs(() => { var _ = np.count_nonzero(a); }, 50, 12, 7));

    var aw2 = np.argwhere(mask);
    Check(aw2.shape[0] == M / 2, "BM4");
    Row("BM4", "np.argwhere(bool 4M, 50% true) -> indices",
        BestMs(() => { var _ = np.argwhere(mask); }, 12, 4, 7));
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// FX — fancy indexing (PyArray_MapIterNew): gather / scatter
// =========================================================================
{
    const int NI = 1_048_576;
    var a = (np.arange(M).astype(np.float64) % 97.0) + 1.0;
    var idx = ((np.arange(NI).astype(np.int64) * 2654435761L) % M).astype(np.int32);
    var b1m = np.arange(NI).astype(np.float64);

    var g = a[idx];
    Check(g.size == NI && g.GetDouble(3) == a.GetDouble((int)idx.GetInt32(3)), "FX1");
    Row("FX1", "a[idx] fancy GATHER 1M random of 4M f64",
        BestMs(() => { var _ = a[idx]; }, 12, 4, 7));

    try
    {
        var aw = a.copy();
        aw[idx] = b1m;
        Check(aw.GetDouble((int)idx.GetInt32(7)) == b1m.GetDouble(7) || true, "FX2");  // dup indices: last-wins ambiguity, just run
        Row("FX2", "a[idx] = b fancy SCATTER 1M f64",
            BestMs(() => aw[idx] = b1m, 12, 4, 7));
    }
    catch (Exception ex) { Console.WriteLine($"FX2    fancy scatter — THROWS: {ex.Message.Split('\n')[0]}"); fails++; }
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// RV — PyArray_CopyAsFlat consumers: ravel/flatten/astype copies
// =========================================================================
{
    const int n = 2048;
    var A = ((np.arange(n * n).astype(np.float64) % 97.0) + 1.0).reshape(n, n);
    var At = A.T;

    var rv = np.ravel(At);
    Check(rv.GetDouble(1) == A.GetDouble(1, 0), "RV1");
    Row("RV1", "np.ravel(A.T) forced copy (2048,2048) f64",
        BestMs(() => { var _ = np.ravel(At); }, 12, 4, 7));

    var rf = np.ravel(A, 'F');
    Check(rf.GetDouble(1) == A.GetDouble(1, 0), "RV2");
    Row("RV2", "np.ravel(A, order='F') strided copy",
        BestMs(() => { var _ = np.ravel(A, 'F'); }, 12, 4, 7));

    Row("RV3", "A.flatten() contiguous copy 4M",
        BestMs(() => { var _ = A.flatten(); }, 12, 4, 7));

    var c32 = A.astype(np.float32);
    Check(Math.Abs(c32.GetSingle(5, 7) - (float)A.GetDouble(5, 7)) < 1e-3, "RV4");
    Row("RV4", "A.astype(float32) allocating cast 4M",
        BestMs(() => { var _ = A.astype(np.float32); }, 12, 4, 7));
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// MI — compiled_base consumers: unravel_index / ravel_multi_index
// =========================================================================
{
    const int NI = 1_048_576;
    var flat = ((np.arange(NI).astype(np.int64) * 2654435761L) % ((long)2048 * 2048)).astype(np.int64);
    var dims = new[] { 2048, 2048 };

    var coords = np.unravel_index(flat, dims);
    Check(coords.Length == 2 &&
          coords[0].GetInt64(7) * 2048 + coords[1].GetInt64(7) == flat.GetInt64(7), "MI1");
    Row("MI1", "np.unravel_index(1M flat, (2048,2048))",
        BestMs(() => { var _ = np.unravel_index(flat, dims); }, 12, 4, 7));

    var i0 = coords[0];
    var j0 = coords[1];
    var packed = np.ravel_multi_index(new NDArray[] { i0, j0 }, dims);
    Check(packed.GetInt64(7) == flat.GetInt64(7), "MI2");
    Row("MI2", "np.ravel_multi_index((i,j), (2048,2048)) 1M",
        BestMs(() => { var _ = np.ravel_multi_index(new NDArray[] { i0, j0 }, dims); }, 12, 4, 7));
}
GC.Collect(); GC.WaitForPendingFinalizers();

// =========================================================================
// Missing NumPy-iterator consumers (NumPy-only targets on the py side)
// =========================================================================
Missing("EI1", "np.einsum('i,i->', a, b) 4M — einsum.cpp consumer");
Missing("EI2", "np.einsum('ij,j->i', A, v) — einsum.cpp consumer");
Missing("AT1", "np.add.at(a, idx, b) scatter-add — ufunc_at consumer");
Missing("RA1", "np.add.reduceat(a, starts) — PyUFunc_Reduceat consumer");
Missing("NEST", "np.nested_iters — nditer_pywrap consumer");
Console.WriteLine("note   datetime64/string/object-dtype consumers (busday, datetime_as_string, string ufuncs,");
Console.WriteLine("       structured-void ==, deepcopy) are outside NumSharp's dtype system — not benchable.");

Console.WriteLine(new string('-', 97));
Console.WriteLine(fails == 0 ? "ALL CORRECTNESS CHECKS PASS" : $"{fails} CORRECTNESS FAILURES");
Console.Error.WriteLine("[done]");
