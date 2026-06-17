#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp;

// ============================================================================
//  "Handle the fallback" — complex axis sum across ALL layouts.
//  Dispatch: inner-axis stride==1  -> SIMD double-pair (fast path, survives
//            row-slicing a[::2,:]); else -> strided scalar generic-math.
//  Verifies correctness vs live np.sum and times each layout.
// ============================================================================

static double Time(Action a, int it)
{
    for (int i = 0; i < 3; i++) a();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int i = 0; i < it; i++) a();
    sw.Stop(); return sw.Elapsed.TotalMilliseconds / it;
}

// contiguous complex row add: o[c] += row[c]  (== double add over 2*cols lanes)
static unsafe void SimdAddRow(Complex* o, Complex* row, int cols)
{
    double* od = (double*)o; double* rd = (double*)row; long len = 2L * cols; long i = 0;
    if (Avx.IsSupported) for (; i + 4 <= len; i += 4) Avx.Store(od + i, Avx.Add(Avx.LoadVector256(od + i), Avx.LoadVector256(rd + i)));
    for (; i < len; i++) od[i] += rd[i];
}
// contiguous complex run reduce -> one Complex (2-lane re/im accumulator)
static unsafe Complex SimdReduceRow(Complex* row, int cols)
{
    double* d = (double*)row; long len = 2L * cols; long i = 0;
    var acc = Vector256<double>.Zero;
    if (Avx.IsSupported) for (; i + 4 <= len; i += 4) acc = Avx.Add(acc, Avx.LoadVector256(d + i));
    double re = acc.GetElement(0) + acc.GetElement(2);
    double im = acc.GetElement(1) + acc.GetElement(3);
    for (; i < len; i += 2) { re += d[i]; im += d[i + 1]; }
    return new Complex(re, im);
}

// Layout-aware complex axis sum for 2-D (the fallback handler).
// Dispatch on which LOGICAL axis is contiguous (NumPy's strategy), not the
// physical last axis: slab when the output axis is contiguous, contiguous-reduce
// when the reduce axis is contiguous, else strided with smaller-stride innermost.
static unsafe NDArray SumAxisHandled(NDArray a, int axis, out string path)
{
    int rows = (int)a.shape[0], cols = (int)a.shape[1];
    long sRow = a.Shape.strides[0], sCol = a.Shape.strides[1]; // element strides
    Complex* basep = (Complex*)a.Address + a.Shape.offset;

    long sReduce = axis == 0 ? sRow : sCol;   // stride along reduced axis
    long sOther = axis == 0 ? sCol : sRow;    // stride along kept (output) axis
    int nReduce = axis == 0 ? rows : cols;
    int nOther = axis == 0 ? cols : rows;

    var ret = np.zeros(new Shape(nOther), NPTypeCode.Complex);
    Complex* o = (Complex*)ret.Address + ret.Shape.offset;

    if (sOther == 1)                  // output axis contiguous -> SIMD slab-fold
    {
        path = "SIMD-slab";
        for (int k = 0; k < nReduce; k++) SimdAddRow(o, basep + k * sReduce, nOther);
    }
    else if (sReduce == 1)            // reduce axis contiguous -> SIMD contiguous-reduce
    {
        path = "SIMD-red";
        for (int j = 0; j < nOther; j++) o[j] = SimdReduceRow(basep + j * sOther, nReduce);
    }
    else if (sOther <= sReduce)       // strided: iterate output (smaller stride) innermost
    {
        path = "strided-slab";
        for (int k = 0; k < nReduce; k++) { Complex* row = basep + k * sReduce; for (int j = 0; j < nOther; j++) o[j] += row[j * sOther]; }
    }
    else                              // strided: reduce axis (smaller stride) innermost
    {
        path = "strided-red";
        for (int j = 0; j < nOther; j++) { Complex* col = basep + j * sOther; Complex acc = Complex.Zero; for (int k = 0; k < nReduce; k++) acc += col[k * sReduce]; o[j] = acc; }
    }
    return ret;
}

static bool Same(NDArray x, NDArray refr)
{
    if (x.size != refr.size) return false;
    for (long i = 0; i < x.size; i++)
    {
        Complex a = x.GetComplex(i), b = refr.GetComplex(i);
        if (Math.Abs(a.Real - b.Real) > 1e-6 * (1 + Math.Abs(b.Real)) || Math.Abs(a.Imaginary - b.Imaginary) > 1e-6 * (1 + Math.Abs(b.Imaginary))) return false;
    }
    return true;
}

static void Run()
{
    int R = 4000, C = 4000; // base 16M; views carve ~4-16M
    var full = (np.arange((long)R * C).astype(NPTypeCode.Double).reshape(R, C).astype(NPTypeCode.Complex)) + new Complex(1, 1);

    var layouts = new (string name, NDArray a)[]
    {
        ("C-contig (4000x4000)", full),
        ("row-sliced a[::2,:]",  full["::2, :"]),
        ("col-sliced a[:,::2]",  full[":, ::2"]),
        ("transposed a.T",        full.T),
    };

    Console.WriteLine($"{"layout",-24}{"axis",5}{"contig?",9}{"elems",10}{"baseline",11}{"handled",11}{"path",14}  ok");
    foreach (var (name, a) in layouts)
    {
        for (int axis = 0; axis < 2; axis++)
        {
            long elems = a.size;
            int it = elems >= 4_000_000 ? 15 : 40;
            NDArray refr = np.sum(a, axis: axis);
            bool ok; string path;
            using (var probe = SumAxisHandled(a, axis, out path)) ok = Same(probe, refr);
            double tb = Time(() => { using var rb = np.sum(a, axis: axis); }, it);
            double th = Time(() => { using var rh = SumAxisHandled(a, axis, out _); }, it);
            Console.WriteLine($"{name,-24}{axis,5}{(a.Shape.IsContiguous ? "Y" : "N"),9}{elems,10}{tb,11:F4}{th,11:F4}{path,14}  {(ok ? "Y" : "N")}");
        }
    }
}

Console.WriteLine($"AVX={Avx.IsSupported}  — complex sum across layouts (ms/op); baseline=live np.sum\n");
Run();
