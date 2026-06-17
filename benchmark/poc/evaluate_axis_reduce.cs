#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using NumSharp;
using NumSharp.Backends.Iteration;

// Phase 5a correctness: evaluate(Reduce(a*b, axis)) == unfused np.reduce(a*b, axis)
// across dtypes / ops / axes / layouts / keepdims. The unfused side is already
// validated vs NumPy by the suite; a couple of absolute NumPy checks included.

bool optCore = !typeof(np).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
Console.WriteLine($"[opt] core={optCore}\n");
int fails = 0;

NDArray Fused(string op, NDArray a, NDArray b, int axis, bool kd)
{
    NpyExpr e = (NpyExpr)a * b;
    NpyExpr r = op switch
    {
        "sum"  => NpyExpr.Sum(e, axis, kd),
        "prod" => NpyExpr.Prod(e, axis, kd),
        "min"  => NpyExpr.Min(e, axis, kd),
        "max"  => NpyExpr.Max(e, axis, kd),
        "mean" => NpyExpr.Mean(e, axis, kd),
        _ => throw new Exception()
    };
    return np.evaluate(r);
}

NDArray Unfused(string op, NDArray a, NDArray b, int axis, bool kd)
{
    var p = a * b;
    return op switch
    {
        "sum"  => np.sum(p, axis: axis, keepdims: kd),
        "prod" => np.prod(p, axis: axis, keepdims: kd),
        "min"  => np.amin(p, axis: axis, keepdims: kd),
        "max"  => np.amax(p, axis: axis, keepdims: kd),
        "mean" => np.mean(p, axis: axis, keepdims: kd),
        _ => throw new Exception()
    };
}

bool Same(NDArray x, NDArray y, out string why)
{
    why = "";
    if (x.size != y.size) { why = $"size {x.size}!={y.size}"; return false; }
    if (x.ndim != y.ndim) { why = $"ndim {x.ndim}!={y.ndim}"; return false; }
    for (int d = 0; d < x.ndim; d++) if (x.shape[d] != y.shape[d]) { why = $"shape[{d}] {x.shape[d]}!={y.shape[d]}"; return false; }
    for (long i = 0; i < x.size; i++)
    {
        double a = Convert.ToDouble(x.GetAtIndex(i)), b = Convert.ToDouble(y.GetAtIndex(i));
        if (double.IsNaN(a) && double.IsNaN(b)) continue;
        if (Math.Abs(a - b) > 1e-6 * (1 + Math.Abs(b)) + 1e-9) { why = $"[{i}] {a} != {b}"; return false; }
    }
    return true;
}

var rnd = new Random(99);
NDArray Mk(int[] dims, NPTypeCode tc)
{
    long n = 1; foreach (var d in dims) n *= d;
    var data = new double[n];
    for (long i = 0; i < n; i++) data[i] = Math.Round(rnd.NextDouble() * 6 - 3, 2);
    return np.array(data).astype(tc).reshape(Array.ConvertAll(dims, x => (long)x));
}

foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Int32 })
foreach (var dims in new[] { new[] { 7, 5 }, new[] { 4, 6, 3 } })
{
    var aBase = Mk(dims, tc); var bBase = Mk(dims, tc);
    var views = new (string tag, NDArray a, NDArray b)[]
    {
        ("C", aBase, bBase),
        ("T", aBase.T, bBase.T),
        ("F", aBase.copy(order:'F'), bBase.copy(order:'F')),
    };
    foreach (var (tag, a, b) in views)
        for (int axis = 0; axis < a.ndim; axis++)
            foreach (var op in new[] { "sum", "prod", "min", "max", "mean" })
                foreach (var kd in new[] { false, true })
                {
                    // int prod overflow → skip (both sides wrap differently vs double check); keep sum/mean/min/max for int.
                    if (tc == NPTypeCode.Int32 && op == "prod") continue;
                    NDArray f, u;
                    try { f = Fused(op, a, b, axis, kd); u = Unfused(op, a, b, axis, kd); }
                    catch (Exception ex) { Console.WriteLine($"FAIL {tc}/{tag} {op} ax{axis} kd{kd}: EX {ex.GetType().Name} {ex.Message}"); fails++; continue; }
                    if (!Same(f, u, out string why)) { Console.WriteLine($"FAIL {tc}/{string.Join("x",dims)}/{tag} {op} ax{axis} kd{kd}: {why}"); fails++; }
                }
}

// Absolute NumPy spot-check: a=[[0,1,2,3],[4,5,6,7],[8,9,10,11]], b=a; sum(a*b,axis=0) etc.
var aa = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
var ff = np.evaluate(NpyExpr.Sum((NpyExpr)aa * aa, 0));
// NumPy: np.sum((np.arange(12).reshape(3,4))**2, axis=0) = [80, 107, 140, 179]
double[] expSum0 = { 80, 107, 140, 179 };
for (long i = 0; i < 4; i++) if (Math.Abs(Convert.ToDouble(ff.GetAtIndex(i)) - expSum0[i]) > 1e-9) { Console.WriteLine($"FAIL numpy sum0[{i}] got {ff.GetAtIndex(i)} exp {expSum0[i]}"); fails++; }
var mm = np.evaluate(NpyExpr.Mean((NpyExpr)aa * aa, 1));
// NumPy: np.mean((arange12.reshape(3,4))**2, axis=1) = [3.5, 42.5, 181.5]
double[] expMean1 = { 3.5, 31.5, 91.5 };
for (long i = 0; i < 3; i++) if (Math.Abs(Convert.ToDouble(mm.GetAtIndex(i)) - expMean1[i]) > 1e-9) { Console.WriteLine($"FAIL numpy mean1[{i}] got {mm.GetAtIndex(i)} exp {expMean1[i]}"); fails++; }

Console.WriteLine(fails == 0 ? "\nALL CORRECT" : $"\n{fails} FAILURES");
partial class Program { }
