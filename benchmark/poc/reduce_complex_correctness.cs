#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;

// Phase 1 correctness gate: complex sum/prod/min/max axis on the NpyIter path,
// vs (a) NumPy's printed values for a known 3x4 array and (b) a brute-force C#
// reference across the layout matrix (C/F/sliced/transposed/3-D/keepdims/axis-1).

// Optimizer guard (CLAUDE.md): must run -c Release.
bool dbgScript = System.Diagnostics.Debugger.IsAttached;
bool optScript = !typeof(Program).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
bool optCore = !typeof(np).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
Console.WriteLine($"[opt] script={optScript} core={optCore}\n");

int fails = 0;

// ---- (a) known array vs NumPy printed values ----
var known = MakeKnown();
CheckVec("sum axis0", np.sum(known, axis: 0), new[]{ C(12,24),C(15,21),C(18,18),C(21,15) }, ref fails);
CheckVec("sum axis1", np.sum(known, axis: 1), new[]{ C(6,42),C(22,26),C(38,10) }, ref fails);
CheckVec("prod axis0", np.prod(known, axis: 0), new[]{ C(-960,0),C(-834,342),C(-624,624),C(-342,834) }, ref fails);
CheckVec("prod axis1", np.prod(known, axis: 1), new[]{ C(10512,-7344),C(-5328,-1776),C(4560,8400) }, ref fails);
CheckVec("min axis0", np.amin(known, axis: 0), new[]{ C(0,12),C(1,11),C(2,10),C(3,9) }, ref fails);
CheckVec("min axis1", np.amin(known, axis: 1), new[]{ C(0,12),C(4,8),C(8,4) }, ref fails);
CheckVec("max axis0", np.amax(known, axis: 0), new[]{ C(8,4),C(9,3),C(10,2),C(11,1) }, ref fails);
CheckVec("max axis1", np.amax(known, axis: 1), new[]{ C(3,9),C(7,5),C(11,1) }, ref fails);

// NaN propagation along an axis (matches NumPy: nan+3j sum, nan+0j min/max).
// Shape (3,1) reduce axis 0 -> exercises the pinned kernel with a NaN element.
var cc = np.array(new Complex[]{ C(1,1), new Complex(double.NaN,0), C(2,2) }).reshape(3,1);
CheckVec("sum nan axis0", np.sum(cc, axis: 0), new[]{ new Complex(double.NaN,3) }, ref fails);
CheckVec("min nan axis0", np.amin(cc, axis: 0), new[]{ new Complex(double.NaN,0) }, ref fails);
CheckVec("max nan axis0", np.amax(cc, axis: 0), new[]{ new Complex(double.NaN,0) }, ref fails);

// ---- (b) brute-force reference across layouts ----
var rnd = new Random(12345);
var shapes = new (int[] dims, string name)[]
{
    (new[]{7,5}, "2D 7x5"),
    (new[]{5,7}, "2D 5x7"),
    (new[]{4,6,3}, "3D 4x6x3"),
    (new[]{1,8}, "2D 1x8"),
    (new[]{8,1}, "2D 8x1"),
    (new[]{16}, "1D 16"),
};
foreach (var (dims, name) in shapes)
{
    var baseArr = MakeRandom(dims, rnd);
    var views = new (string tag, NDArray a)[]
    {
        ("C", baseArr),
        ("F", baseArr.copy(order: 'F')),
        ("T", baseArr.T),
        ("slice0", dims.Length>=1 && dims[0]>=4 ? baseArr["::2"] : baseArr),
    };
    foreach (var (tag, a) in views)
    {
        for (int ax = 0; ax < a.ndim; ax++)
        {
            if (a.shape[ax] == 1) continue;
            foreach (var op in new[]{"sum","prod","min","max","mean"})
            {
                foreach (var kd in new[]{false, true})
                {
                    NDArray got = Run(op, a, ax, kd);
                    var (refVals, refShape) = Reference(op, a, ax, kd);
                    if (!SameFlat(got, refVals, refShape, out string why))
                    {
                        Console.WriteLine($"FAIL {name}/{tag} {op} axis={ax} keepdims={kd}: {why}");
                        fails++;
                    }
                }
            }
        }
    }
}

// out= path (sum only)
{
    var a = MakeRandom(new[]{6,4}, rnd);
    var outArr = np.zeros(new Shape(4), NPTypeCode.Complex);
    var eng = (DefaultEngine)a.TensorEngine;
    var r = eng.ReduceAdd(a, 0, false, null, outArr);
    var (refv, refs) = Reference("sum", a, 0, false);
    if (!ReferenceEquals(r, outArr)) { Console.WriteLine("FAIL out= identity"); fails++; }
    if (!SameFlat(outArr, refv, refs, out string w2)) { Console.WriteLine($"FAIL out= values: {w2}"); fails++; }
}

Console.WriteLine(fails == 0 ? "\nALL CORRECT" : $"\n{fails} FAILURES");

// ===== helpers =====
static Complex C(double re, double im) => new Complex(re, im);

static NDArray MakeKnown()
{
    var data = new Complex[12];
    for (int i = 0; i < 12; i++) data[i] = new Complex(i, 12 - i);
    return np.array(data).reshape(3, 4);
}

static NDArray MakeRandom(int[] dims, Random rnd)
{
    long n = 1; foreach (var d in dims) n *= d;
    var data = new Complex[n];
    for (long i = 0; i < n; i++) data[i] = new Complex(Math.Round(rnd.NextDouble()*20-10,3), Math.Round(rnd.NextDouble()*20-10,3));
    var flat = np.array(data);
    var longDims = Array.ConvertAll(dims, x => (long)x);
    return flat.reshape(longDims);
}

static NDArray Run(string op, NDArray a, int ax, bool kd) => op switch
{
    "sum"  => np.sum(a, axis: ax, keepdims: kd),
    "mean" => np.mean(a, axis: ax, keepdims: kd),
    "prod" => np.prod(a, axis: ax, keepdims: kd),
    "min"  => np.amin(a, axis: ax, keepdims: kd),
    "max"  => np.amax(a, axis: ax, keepdims: kd),
    _ => throw new Exception()
};

// Brute-force reference: fold the reduce axis in logical order per output cell.
static (Complex[] vals, long[] shape) Reference(string op, NDArray a, int axis, bool kd)
{
    int ndim = a.ndim;
    var dims = new long[ndim];
    for (int i = 0; i < ndim; i++) dims[i] = a.shape[i];
    long axisN = dims[axis];

    var outDims = new System.Collections.Generic.List<long>();
    for (int i = 0; i < ndim; i++) if (i != axis) outDims.Add(dims[i]);
    long outSize = 1; foreach (var d in outDims) outSize *= d;

    var result = new Complex[outSize];
    // iterate output coords
    var outCoord = new long[outDims.Count];
    for (long oi = 0; oi < outSize; oi++)
    {
        // decode oi -> outCoord
        long rem = oi;
        for (int d = outDims.Count - 1; d >= 0; d--) { outCoord[d] = rem % outDims[d]; rem /= outDims[d]; }
        // build full coord
        var full = new long[ndim];
        for (int i = 0, od = 0; i < ndim; i++) if (i != axis) full[i] = outCoord[od++];

        Complex acc = op switch
        {
            "sum" => Complex.Zero,
            "mean" => Complex.Zero,
            "prod" => Complex.One,
            "min" => new Complex(double.PositiveInfinity, double.PositiveInfinity),
            "max" => new Complex(double.NegativeInfinity, double.NegativeInfinity),
            _ => Complex.Zero
        };
        for (long k = 0; k < axisN; k++)
        {
            full[axis] = k;
            Complex v = GetC(a, full);
            acc = op switch
            {
                "sum" => acc + v,
                "mean" => acc + v,
                "prod" => acc * v,
                "min" => LexPick(acc, v, false),
                "max" => LexPick(acc, v, true),
                _ => acc
            };
        }
        result[oi] = op == "mean" ? acc / (double)axisN : acc;
    }

    long[] shape = kd
        ? BuildKeep(dims, axis)
        : outDims.ToArray();
    return (result, shape);
}

static long[] BuildKeep(long[] dims, int axis)
{
    var r = (long[])dims.Clone(); r[axis] = 1; return r;
}

static Complex GetC(NDArray a, long[] coord)
{
    // flatten coord via a.shape (logical) using row-major over logical dims
    int ndim = a.ndim;
    long flat = 0, stride = 1;
    for (int i = ndim - 1; i >= 0; i--) { flat += coord[i] * stride; stride *= a.shape[i]; }
    return (System.Numerics.Complex)a.GetAtIndex(flat);
}

static Complex LexPick(Complex a, Complex b, bool greater)
{
    if (double.IsNaN(a.Real) || double.IsNaN(a.Imaginary)) return a;
    if (double.IsNaN(b.Real) || double.IsNaN(b.Imaginary)) return b;
    bool ag = a.Real > b.Real || (a.Real == b.Real && a.Imaginary > b.Imaginary);
    return greater ? (ag ? a : b) : (ag ? b : a);
}

static bool SameFlat(NDArray got, Complex[] refv, long[] refShape, out string why)
{
    why = "";
    if (got.size != refv.Length) { why = $"size {got.size} != {refv.Length}"; return false; }
    if (got.ndim != refShape.Length) { why = $"ndim {got.ndim} != {refShape.Length}"; return false; }
    for (int i = 0; i < refShape.Length; i++) if (got.shape[i] != refShape[i]) { why = $"shape[{i}] {got.shape[i]} != {refShape[i]}"; return false; }
    for (long i = 0; i < refv.Length; i++)
    {
        Complex g = (System.Numerics.Complex)got.GetAtIndex(i), r = refv[i];
        if (!Close(g, r)) { why = $"[{i}] got {g} ref {r}"; return false; }
    }
    return true;
}

static bool Close(Complex a, Complex b)
{
    return CloseD(a.Real, b.Real) && CloseD(a.Imaginary, b.Imaginary);
}
static bool CloseD(double a, double b)
{
    if (double.IsNaN(a) && double.IsNaN(b)) return true;
    if (double.IsInfinity(a) || double.IsInfinity(b)) return a == b;
    return Math.Abs(a - b) <= 1e-6 * (1 + Math.Abs(b));
}

static void CheckVec(string name, NDArray got, Complex[] expect, ref int fails)
{
    if (got.size != expect.Length) { Console.WriteLine($"FAIL {name}: size {got.size} != {expect.Length}"); fails++; return; }
    for (long i = 0; i < expect.Length; i++)
        if (!Close((System.Numerics.Complex)got.GetAtIndex(i), expect[i])) { Console.WriteLine($"FAIL {name}[{i}]: got {(System.Numerics.Complex)got.GetAtIndex(i)} expect {expect[i]}"); fails++; return; }
    Console.WriteLine($"ok   {name}");
}

static void Check1(string name, NDArray got, Complex expect, ref int fails)
{
    if (!Close((System.Numerics.Complex)got.GetAtIndex(0), expect)) { Console.WriteLine($"FAIL {name}: got {(System.Numerics.Complex)got.GetAtIndex(0)} expect {expect}"); fails++; }
    else Console.WriteLine($"ok   {name}");
}

partial class Program { }
