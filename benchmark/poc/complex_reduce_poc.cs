#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

// ============================================================================
//  Complex128 axis-reduction POC — benchmarks every candidate so we can decide.
//  Metrics: ms/op, alloc B/op, correctness vs live np.sum.
//
//  0. Baseline    — live np.sum (boxing scalar helper)
//  C. NpyAxisIter — proven generic static-abstract per-output reduce (unwired)
//  B1. NpyIter 2-op REDUCE via ForEach (delegate)      [np.average template]
//  B2. NpyIter 2-op REDUCE via ExecuteGeneric (struct) [inlined dispatch]
//  A. Monolithic  — whole-array generic-math slab/innermost (Direct ceiling)
// ============================================================================

static double Time(Action a, int it)
{
    for (int i = 0; i < 3; i++) a();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int i = 0; i < it; i++) a();
    sw.Stop(); return sw.Elapsed.TotalMilliseconds / it;
}
static long Alloc(Action a, int it)
{
    for (int i = 0; i < 3; i++) a();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    long b0 = GC.GetAllocatedBytesForCurrentThread();
    for (int i = 0; i < it; i++) a();
    return (GC.GetAllocatedBytesForCurrentThread() - b0) / it;
}
static int Iters(long n) => n >= 1_000_000 ? 20 : 200;

static NDArray MakeComplex(int rows, int cols) =>
    (np.arange((long)rows * cols).astype(NPTypeCode.Double).reshape(rows, cols).astype(NPTypeCode.Complex))
    + new Complex(1, 1);

static long[] AxisRemoved(NDArray a, int axis)
{
    var l = new System.Collections.Generic.List<long>();
    for (int i = 0; i < a.ndim; i++) if (i != axis) l.Add(a.shape[i]);
    return l.ToArray();
}

// ---- C: NpyAxisIter proven generic per-output reduce ----
static NDArray SumAxis_NpyAxisIter(NDArray a, int axis)
{
    var outShape = AxisRemoved(a, axis);
    var ret = new NDArray(NPTypeCode.Complex, outShape.Length == 0 ? Shape.Scalar : new Shape(outShape));
    NpyAxisIter.ReduceNumeric<Complex, NpySumAxisKernel<Complex>>(a.Storage, ret.Storage, axis);
    return ret;
}

// ---- B1/B2: NpyIter 2-op REDUCE construction (np.average template) ----
static unsafe NpyIterRef BuildReduceIter(NDArray a, NDArray outArr, int axis)
{
    int ndim = a.ndim;
    int[] aAxes = new int[ndim];
    int[] outAxes = new int[ndim];
    int oc = 0;
    for (int i = 0; i < ndim; i++) { aAxes[i] = i; outAxes[i] = (i == axis) ? -1 : oc++; }
    return NpyIterRef.AdvancedNew(
        2, new[] { a, outArr },
        NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.EXTERNAL_LOOP,
        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
        new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
        null, ndim, new[] { aAxes, outAxes });
}
static unsafe void SumDelegate(void** p, long* strides, long count, void* aux)
{
    byte* inp = (byte*)p[0]; long inS = strides[0];
    byte* outp = (byte*)p[1]; long outS = strides[1];
    if (outS == 0) { Complex acc = *(Complex*)outp; for (long i = 0; i < count; i++) acc += *(Complex*)(inp + i * inS); *(Complex*)outp = acc; }
    else { for (long i = 0; i < count; i++) { Complex* o = (Complex*)(outp + i * outS); *o += *(Complex*)(inp + i * inS); } }
}
static unsafe NDArray SumAxis_NpyIter_ForEach(NDArray a, int axis)
{
    var outShape = AxisRemoved(a, axis);
    var ret = np.zeros(outShape.Length == 0 ? Shape.Scalar : new Shape(outShape), NPTypeCode.Complex);
    using var iter = BuildReduceIter(a, ret, axis);
    iter.ForEach(SumDelegate);
    return ret;
}
static unsafe NDArray SumAxis_NpyIter_Generic(NDArray a, int axis)
{
    var outShape = AxisRemoved(a, axis);
    var ret = np.zeros(outShape.Length == 0 ? Shape.Scalar : new Shape(outShape), NPTypeCode.Complex);
    using var iter = BuildReduceIter(a, ret, axis);
    iter.ExecuteGeneric(default(ReduceSumKernel<Complex>));
    return ret;
}

// ---- A: Monolithic whole-array generic-math (Direct whole-array ceiling) ----
static unsafe void MonoSlab<T>(T* p, T* o, int rows, int cols) where T : unmanaged, IAdditionOperators<T, T, T>
{ for (int r = 0; r < rows; r++) { T* row = p + (long)r * cols; for (int c = 0; c < cols; c++) o[c] += row[c]; } }
static unsafe void MonoInner<T>(T* p, T* o, int rows, int cols) where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
{ for (int r = 0; r < rows; r++) { T* row = p + (long)r * cols; T acc = T.AdditiveIdentity; for (int c = 0; c < cols; c++) acc += row[c]; o[r] = acc; } }
static unsafe NDArray SumAxis_Monolithic(NDArray a, int axis)
{
    int rows = (int)a.shape[0], cols = (int)a.shape[1];
    Complex* p = (Complex*)a.Address + a.Shape.offset;
    if (axis == 0)
    {
        var ret = np.zeros(new Shape(cols), NPTypeCode.Complex);
        MonoSlab<Complex>(p, (Complex*)ret.Address + ret.Shape.offset, rows, cols);
        return ret;
    }
    else
    {
        var ret = np.zeros(new Shape(rows), NPTypeCode.Complex);
        MonoInner<Complex>(p, (Complex*)ret.Address + ret.Shape.offset, rows, cols);
        return ret;
    }
}

// ---- Generic versions for the dtype sweep ----
static NDArray SumAxis_NpyAxisIter_T<T>(NDArray a, int axis, NPTypeCode tc)
    where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
{
    var outShape = AxisRemoved(a, axis);
    var ret = new NDArray(tc, outShape.Length == 0 ? Shape.Scalar : new Shape(outShape));
    NpyAxisIter.ReduceNumeric<T, NpySumAxisKernel<T>>(a.Storage, ret.Storage, axis);
    return ret;
}
static unsafe NDArray SumAxis_Mono_T<T>(NDArray a, int axis, NPTypeCode tc)
    where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
{
    int rows = (int)a.shape[0], cols = (int)a.shape[1];
    T* p = (T*)a.Address + a.Shape.offset;
    if (axis == 0) { var ret = new NDArray(tc, new Shape(cols)); MonoSlab<T>(p, (T*)ret.Address + ret.Shape.offset, rows, cols); return ret; }
    else { var ret = new NDArray(tc, new Shape(rows)); MonoInner<T>(p, (T*)ret.Address + ret.Shape.offset, rows, cols); return ret; }
}

static bool Same(NDArray x, NDArray reference)
{
    if (x.size != reference.size) return false;
    for (long i = 0; i < x.size; i++)
    {
        Complex a = x.GetComplex(i), b = reference.GetComplex(i);
        if (Math.Abs(a.Real - b.Real) > 1e-6 * (1 + Math.Abs(b.Real)) ||
            Math.Abs(a.Imaginary - b.Imaginary) > 1e-6 * (1 + Math.Abs(b.Imaginary))) return false;
    }
    return true;
}

// ============================== RUN ==============================
Console.WriteLine("=== Complex128 sum axis-reduction: candidate benchmark ===");
Console.WriteLine("NumPy refs (ms): sum0 100K=0.015 10M=7.25 | sum1 100K~0.033 10M~7.4");
Console.WriteLine();

int[] Ns = { 10_000, 100_000, 1_000_000, 10_000_000 };
foreach (int axis in new[] { 0, 1 })
{
    Console.WriteLine($"---- axis={axis} (ms/op) ----");
    Console.WriteLine($"{"N",-10}{"shape",-13}{"baseline",10}{"NpyAxisIt",10}{"NpyIt-FE",10}{"NpyIt-Gen",10}{"Monolith",10}   correctness");
    foreach (int N in Ns)
    {
        int rows = (int)Math.Sqrt(N), cols = N / rows;
        var a = MakeComplex(rows, cols);
        int it = Iters(N);
        var refr = np.sum(a, axis: axis);

        double t0 = Time(() => { using var _ = np.sum(a, axis: axis); }, it);
        double t1 = -1; bool ok1 = false;
        double t2 = -1; bool ok2 = false;
        double t3 = -1; bool ok3 = false;
        double t4 = -1; bool ok4 = false;
        try { ok1 = Same(SumAxis_NpyAxisIter(a, axis), refr); t1 = Time(() => { using var _ = SumAxis_NpyAxisIter(a, axis); }, it); } catch (Exception e) { Console.Write($"[AxIt:{e.GetType().Name}] "); }
        try { ok2 = Same(SumAxis_NpyIter_ForEach(a, axis), refr); t2 = Time(() => { using var _ = SumAxis_NpyIter_ForEach(a, axis); }, it); } catch (Exception e) { Console.Write($"[FE:{e.GetType().Name}] "); }
        try { ok3 = Same(SumAxis_NpyIter_Generic(a, axis), refr); t3 = Time(() => { using var _ = SumAxis_NpyIter_Generic(a, axis); }, it); } catch (Exception e) { Console.Write($"[Gen:{e.GetType().Name}] "); }
        try { ok4 = Same(SumAxis_Monolithic(a, axis), refr); t4 = Time(() => { using var _ = SumAxis_Monolithic(a, axis); }, it); } catch (Exception e) { Console.Write($"[Mono:{e.GetType().Name}] "); }

        string ok = $"AxIt={(ok1 ? "Y" : "N")} FE={(ok2 ? "Y" : "N")} Gen={(ok3 ? "Y" : "N")} Mono={(ok4 ? "Y" : "N")}";
        Console.WriteLine($"{N,-10}{$"{rows}x{cols}",-13}{t0,10:F4}{t1,10:F4}{t2,10:F4}{t3,10:F4}{t4,10:F4}   {ok}");
    }
    Console.WriteLine();
}

{
    var a = MakeComplex(3162, 3162);
    Console.WriteLine("Alloc B/op (10M, axis=0):");
    Console.WriteLine($"  baseline np.sum : {Alloc(() => { using var _ = np.sum(a, axis: 0); }, 20),14:N0}");
    Console.WriteLine($"  NpyAxisIter     : {Alloc(() => { using var _ = SumAxis_NpyAxisIter(a, 0); }, 20),14:N0}");
    Console.WriteLine($"  NpyIter ForEach : {Alloc(() => { using var _ = SumAxis_NpyIter_ForEach(a, 0); }, 20),14:N0}");
    Console.WriteLine($"  NpyIter Generic : {Alloc(() => { using var _ = SumAxis_NpyIter_Generic(a, 0); }, 20),14:N0}");
    Console.WriteLine($"  Monolithic      : {Alloc(() => { using var _ = SumAxis_Monolithic(a, 0); }, 20),14:N0}");
}

// ---- VARIATION: strided input a[::2,:] (1000x1000 view, row stride 2000) ----
Console.WriteLine();
Console.WriteLine("Strided input a[::2,:] (1000x1000 view) — correctness (Y=handles non-contig layout):");
{
    var full = MakeComplex(2000, 1000);
    var a = full["::2, :"]; // strided view
    foreach (int axis in new[] { 0, 1 })
    {
        var refr = np.sum(a, axis: axis);
        bool oAx = false, oFE = false, oGen = false, oMono = false;
        try { oAx = Same(SumAxis_NpyAxisIter(a, axis), refr); } catch { }
        try { oFE = Same(SumAxis_NpyIter_ForEach(a, axis), refr); } catch { }
        try { oGen = Same(SumAxis_NpyIter_Generic(a, axis), refr); } catch { }
        try { oMono = Same(SumAxis_Monolithic(a, axis), refr); } catch { }
        Console.WriteLine($"  axis={axis}: NpyAxisIt={(oAx ? "Y" : "N")} NpyIt-FE={(oFE ? "Y" : "N")} NpyIt-Gen={(oGen ? "Y" : "N")} Monolith={(oMono ? "Y" : "N (assumes C-contig)")}");
    }
}

// ---- VARIATION: dtype sweep (Half, Decimal) sum @ 1M, both axes ----
Console.WriteLine();
Console.WriteLine("Dtype sweep sum @ 1M (1000x1000), ms/op  [other excluded dtypes]:");
Console.WriteLine($"{"dtype/axis",-16}{"baseline",10}{"NpyAxisIt",11}{"Monolith",11}   ok");
{
    foreach (int axis in new[] { 0, 1 })
    {
        var h = (np.arange(1_000_000).astype(NPTypeCode.Double).reshape(1000, 1000).astype(NPTypeCode.Half));
        var hr = np.sum(h, axis: axis);
        bool hok = false; try { hok = Same2(SumAxis_NpyAxisIter_T<Half>(h, axis, NPTypeCode.Half), hr) & Same2(SumAxis_Mono_T<Half>(h, axis, NPTypeCode.Half), hr); } catch { }
        double hb = Time(() => { using var _ = np.sum(h, axis: axis); }, 20);
        double hax = Time(() => { using var _ = SumAxis_NpyAxisIter_T<Half>(h, axis, NPTypeCode.Half); }, 20);
        double hmo = Time(() => { using var _ = SumAxis_Mono_T<Half>(h, axis, NPTypeCode.Half); }, 20);
        Console.WriteLine($"{$"Half ax={axis}",-16}{hb,10:F4}{hax,11:F4}{hmo,11:F4}   {(hok ? "Y" : "N")}");

        var d = (np.arange(1_000_000).astype(NPTypeCode.Double).reshape(1000, 1000).astype(NPTypeCode.Decimal));
        var dr = np.sum(d, axis: axis);
        bool dok = false; try { dok = Same2(SumAxis_NpyAxisIter_T<decimal>(d, axis, NPTypeCode.Decimal), dr) & Same2(SumAxis_Mono_T<decimal>(d, axis, NPTypeCode.Decimal), dr); } catch { }
        double db = Time(() => { using var _ = np.sum(d, axis: axis); }, 20);
        double dax = Time(() => { using var _ = SumAxis_NpyAxisIter_T<decimal>(d, axis, NPTypeCode.Decimal); }, 20);
        double dmo = Time(() => { using var _ = SumAxis_Mono_T<decimal>(d, axis, NPTypeCode.Decimal); }, 20);
        Console.WriteLine($"{$"Decimal ax={axis}",-16}{db,10:F4}{dax,11:F4}{dmo,11:F4}   {(dok ? "Y" : "N")}");
    }
}

static bool Same2(NDArray x, NDArray reference)
{
    if (x.size != reference.size) return false;
    for (long i = 0; i < x.size; i++)
    {
        double a = Convert.ToDouble(x.GetAtIndex(i)), b = Convert.ToDouble(reference.GetAtIndex(i));
        if (Math.Abs(a - b) > 1e-2 * (1 + Math.Abs(b))) return false;
    }
    return true;
}

public readonly struct ReduceSumKernel<T> : INpyInnerLoop where T : unmanaged, IAdditionOperators<T, T, T>
{
    public unsafe void Execute(void** p, long* strides, long count)
    {
        byte* inp = (byte*)p[0]; long inS = strides[0];
        byte* outp = (byte*)p[1]; long outS = strides[1];
        if (outS == 0) { T acc = *(T*)outp; for (long i = 0; i < count; i++) acc += *(T*)(inp + i * inS); *(T*)outp = acc; }
        else { for (long i = 0; i < count; i++) { T* o = (T*)(outp + i * outS); *o += *(T*)(inp + i * inS); } }
    }
}
