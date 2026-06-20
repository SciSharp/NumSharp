#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// asiter_repl_bench.cs — find the best NpyIter-based replacement for each
// production AsIterator<T> call site (Half / Complex / bool flat reductions).
//
// Candidates per pattern:
//   A) AsIterator   : arr.AsIterator<T>() + while(HasNext()) MoveNext()  (CURRENT)
//   B) ExecReducing : NpyIterRef.New(arr, EXTERNAL_LOOP).ExecuteReducing<K,Acc>  (struct kernel)
//   C) DirectScan   : raw (T*)arr.Address pointer loop                   (contig only)
//   D) ReduceBool   : NpyIter.ReduceBool<bool,Kernel>                    (bool only; SIMD)
//
// Layouts: Contiguous (1-D) and Strided (transposed 2-D view, flat reduce).
// Report:  ms/op (best-of-rounds) + speedup vs AsIterator baseline (>1 = faster).
//
// Run:  dotnet run -c Release - < benchmark/poc/asiter_repl_bench.cs
//
using System;
using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbgScript = Attribute.GetCustomAttribute(System.Reflection.Assembly.GetExecutingAssembly(), typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
var dbgCore   = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if ((dbgScript?.IsJITOptimizerDisabled ?? false) || (dbgCore?.IsJITOptimizerDisabled ?? false))
{
    Console.WriteLine("FATAL: Debug-JITted assemblies — numbers would be INVALID. Use: dotnet run -c Release - < script.cs");
    return;
}

const int ROUNDS = 7;
int[] SIZES = { 4096, 4_000_000 };

var rng = new Random(12345);

// ---- builders ----------------------------------------------------------
NDArray HalfArr(int n)
{
    var a = new double[n];
    for (int i = 0; i < n; i++) a[i] = rng.NextDouble();        // [0,1) — finite Half sums
    return np.array(a).astype(NPTypeCode.Half);
}
NDArray CplxArr(int n)
{
    var a = new Complex[n];
    for (int i = 0; i < n; i++) a[i] = new Complex(rng.NextDouble(), rng.NextDouble());
    return np.array(a);
}
NDArray BoolArr(int n, bool allFalse)
{
    var a = new bool[n];
    if (!allFalse) for (int i = 0; i < n; i++) a[i] = (rng.Next(4) == 0);
    return np.array(a);
}
// transposed square view (non-contiguous flat reduction)
NDArray Strided(NDArray flat)
{
    int n = (int)flat.size;
    int r = (int)Math.Sqrt(n);
    while (r > 1 && n % r != 0) r--;
    return flat.reshape(r, n / r).T;   // shape (c,r) strides non-contiguous
}

// ---- timing ------------------------------------------------------------
double Bench(Func<double> op, long n, out double result)
{
    int reps = (int)Math.Max(1, 8_000_000L / n);
    // warmup
    double w = 0; for (int i = 0; i < 2; i++) w = op(); result = w;
    double best = double.MaxValue;
    for (int round = 0; round < ROUNDS; round++)
    {
        var sw = Stopwatch.StartNew();
        double last = 0;
        for (int i = 0; i < reps; i++) last = op();
        sw.Stop();
        double msPerOp = sw.Elapsed.TotalMilliseconds / reps;
        if (msPerOp < best) best = msPerOp;
        result = last;
    }
    return best;
}

void Row(string pat, string layout, long n, double baseMs, double bMs, double cMs, double dMs,
         double baseR, double bR, double cR, double dR, bool ok)
{
    string F(double ms) => ms == double.MaxValue ? "    -   " : (ms < 0.001 ? $"{ms*1000:F2}µs" : $"{ms,7:F4}");
    string S(double ms) => ms == double.MaxValue ? "  - " : $"{baseMs/ms,5:F2}x";
    Console.WriteLine($"{pat,-14} {layout,-9} {n,9} | A {F(baseMs)} | B {F(bMs)} {S(bMs)} | C {F(cMs)} {S(cMs)} | D {F(dMs)} {S(dMs)} | {(ok ? "ok" : "DIFF!")}");
}

bool Close(double a, double b) => double.IsNaN(a) && double.IsNaN(b) || Math.Abs(a - b) <= 1e-6 * (1 + Math.Abs(b));

Console.WriteLine("Legend: A=AsIterator(baseline)  B=ExecuteReducing(struct kernel)  C=DirectScan(contig)  D=ReduceBool");
Console.WriteLine("Speedup = A_ms / X_ms  (>1 = candidate FASTER than AsIterator). Best-of-" + ROUNDS + " rounds.\n");
Console.WriteLine($"{"pattern",-14} {"layout",-9} {"N",9} |   A (ms)   |   B (ms)   speedup |   C (ms)   speedup |   D (ms)   speedup | chk");
Console.WriteLine(new string('-', 120));

foreach (int n in SIZES)
{
    var hC = HalfArr(n);  var hS = Strided(HalfArr(n));
    var cC = CplxArr(n);  var cS = Strided(CplxArr(n));
    var bC = BoolArr(n, allFalse: true); var bS = Strided(BoolArr(n, allFalse: true));

    // ===== Half sum (covers sum/prod/mean/nansum) =====
    foreach (var (lay, arr) in new[] { ("contig", hC), ("strided", hS) })
    {
        double rb = Bench(() => HalfSum_AsIter(arr), n, out _);
        double rB = Bench(() => HalfSum_Exec(arr), n, out var vB);
        double rC = lay == "contig" ? Bench(() => HalfSum_Direct(arr), n, out _) : double.MaxValue;
        double refv = HalfSum_AsIter(arr);
        Row("Half.sum", lay, n, rb, rB, rC, double.MaxValue, 1, rb/rB, rb/rC, 0, Close(vB, refv));
    }
    // ===== Half max + NaN (covers min/max/argmin/argmax/nanmin/nanmax) =====
    foreach (var (lay, arr) in new[] { ("contig", hC), ("strided", hS) })
    {
        double rb = Bench(() => HalfMax_AsIter(arr), n, out _);
        double rB = Bench(() => HalfMax_Exec(arr), n, out var vB);
        double rC = lay == "contig" ? Bench(() => HalfMax_Direct(arr), n, out _) : double.MaxValue;
        double refv = HalfMax_AsIter(arr);
        Row("Half.max", lay, n, rb, rB, rC, double.MaxValue, 1, rb/rB, rb/rC, 0, Close(vB, refv));
    }
    // ===== Complex sum (covers Complex sum/prod/mean/nansum) =====
    foreach (var (lay, arr) in new[] { ("contig", cC), ("strided", cS) })
    {
        double rb = Bench(() => CplxSum_AsIter(arr), n, out _);
        double rB = Bench(() => CplxSum_Exec(arr), n, out var vB);
        double rC = lay == "contig" ? Bench(() => CplxSum_Direct(arr), n, out _) : double.MaxValue;
        double refv = CplxSum_AsIter(arr);
        Row("Complex.sum", lay, n, rb, rB, rC, double.MaxValue, 1, rb/rB, rb/rC, 0, Close(vB, refv));
    }
    // ===== Complex max lexicographic (covers Complex min/max/argmin/argmax) =====
    foreach (var (lay, arr) in new[] { ("contig", cC), ("strided", cS) })
    {
        double rb = Bench(() => CplxMax_AsIter(arr), n, out _);
        double rB = Bench(() => CplxMax_Exec(arr), n, out var vB);
        double rC = lay == "contig" ? Bench(() => CplxMax_Direct(arr), n, out _) : double.MaxValue;
        double refv = CplxMax_AsIter(arr);
        Row("Complex.max", lay, n, rb, rB, rC, double.MaxValue, 1, rb/rB, rb/rC, 0, Close(vB, refv));
    }
    // ===== bool any (= np.max(bool); min=All is symmetric) =====
    foreach (var (lay, arr) in new[] { ("contig", bC), ("strided", bS) })
    {
        double rb = Bench(() => BoolAny_AsIter(arr), n, out _);
        double rB = Bench(() => BoolAny_Exec(arr), n, out var vB);
        double rC = lay == "contig" ? Bench(() => BoolAny_Direct(arr), n, out _) : double.MaxValue;
        double rD = Bench(() => BoolAny_ReduceBool(arr), n, out var vD);
        double refv = BoolAny_AsIter(arr);
        Row("bool.any", lay, n, rb, rB, rC, rD, 1, rb/rB, rb/rC, rb/rD, Close(vB, refv) && Close(vD, refv));
    }
    Console.WriteLine();
}

// ======================================================================
// IMPLEMENTATIONS
// ======================================================================

// ---- Half sum ----
static double HalfSum_AsIter(NDArray arr)
{
    double s = 0; var it = arr.AsIterator<Half>();
    while (it.HasNext()) s += (double)it.MoveNext();
    return s;
}
static unsafe double HalfSum_Exec(NDArray arr)
{
    using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
    return it.ExecuteReducing<HalfSumK, double>(default, 0.0);
}
static unsafe double HalfSum_Direct(NDArray arr)
{
    long n = arr.size; Half* p = (Half*)((byte*)arr.Address + arr.Shape.offset * 2);
    double s = 0; for (long i = 0; i < n; i++) s += (double)p[i];
    return s;
}

// ---- Half max (NaN-propagate; data has no NaN so full scan) ----
static double HalfMax_AsIter(NDArray arr)
{
    var it = arr.AsIterator<Half>(); double best = double.NegativeInfinity; bool seen = false;
    while (it.HasNext()) { double v = (double)it.MoveNext(); if (double.IsNaN(v)) return double.NaN; if (!seen || v > best) { best = v; seen = true; } }
    return best;
}
static unsafe double HalfMax_Exec(NDArray arr)
{
    using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
    var a = it.ExecuteReducing<HalfMaxK, MinMaxAcc>(default, new MinMaxAcc { Best = double.NegativeInfinity });
    return a.SawNaN ? double.NaN : a.Best;
}
static unsafe double HalfMax_Direct(NDArray arr)
{
    long n = arr.size; Half* p = (Half*)((byte*)arr.Address + arr.Shape.offset * 2);
    double best = double.NegativeInfinity;
    for (long i = 0; i < n; i++) { double v = (double)p[i]; if (double.IsNaN(v)) return double.NaN; if (v > best) best = v; }
    return best;
}

// ---- Complex sum ----
static double CplxSum_AsIter(NDArray arr)
{
    var it = arr.AsIterator<Complex>(); Complex s = Complex.Zero;
    while (it.HasNext()) s += it.MoveNext();
    return s.Real + s.Imaginary;
}
static unsafe double CplxSum_Exec(NDArray arr)
{
    using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
    var s = it.ExecuteReducing<CplxSumK, Complex>(default, Complex.Zero);
    return s.Real + s.Imaginary;
}
static unsafe double CplxSum_Direct(NDArray arr)
{
    long n = arr.size; Complex* p = (Complex*)((byte*)arr.Address + arr.Shape.offset * 16);
    Complex s = Complex.Zero; for (long i = 0; i < n; i++) s += p[i];
    return s.Real + s.Imaginary;
}

// ---- Complex max (lexicographic real,imag) ----
static double CplxMax_AsIter(NDArray arr)
{
    var it = arr.AsIterator<Complex>(); Complex best = Complex.Zero; bool seen = false;
    while (it.HasNext()) { var v = it.MoveNext(); if (!seen || v.Real > best.Real || (v.Real == best.Real && v.Imaginary > best.Imaginary)) { best = v; seen = true; } }
    return best.Real + best.Imaginary;
}
static unsafe double CplxMax_Exec(NDArray arr)
{
    using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
    var a = it.ExecuteReducing<CplxMaxK, CplxAcc>(default, default);
    return a.Best.Real + a.Best.Imaginary;
}
static unsafe double CplxMax_Direct(NDArray arr)
{
    long n = arr.size; Complex* p = (Complex*)((byte*)arr.Address + arr.Shape.offset * 16);
    Complex best = Complex.Zero; bool seen = false;
    for (long i = 0; i < n; i++) { var v = p[i]; if (!seen || v.Real > best.Real || (v.Real == best.Real && v.Imaginary > best.Imaginary)) { best = v; seen = true; } }
    return best.Real + best.Imaginary;
}

// ---- bool any ----
static double BoolAny_AsIter(NDArray arr)
{
    var it = arr.AsIterator<bool>();
    while (it.HasNext()) if (it.MoveNext()) return 1;
    return 0;
}
static unsafe double BoolAny_Exec(NDArray arr)
{
    using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
    return it.ExecuteReducing<BoolAnyK, int>(default, 0);
}
static unsafe double BoolAny_Direct(NDArray arr)
{
    long n = arr.size; byte* p = (byte*)arr.Address + arr.Shape.offset;
    for (long i = 0; i < n; i++) if (p[i] != 0) return 1;
    return 0;
}
static double BoolAny_ReduceBool(NDArray arr)
    => NpyIter.ReduceBool<bool, NpyAnyKernel<bool>>(arr) ? 1 : 0;

// ======================================================================
// STRUCT KERNELS
// ======================================================================
public struct MinMaxAcc { public double Best; public bool Seen; public bool SawNaN; }
public struct CplxAcc { public Complex Best; public bool Seen; }

public readonly struct HalfSumK : INpyReducingInnerLoop<double>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref double sum)
    {
        byte* p = (byte*)dp[0]; long s = st[0]; double acc = sum;
        for (long i = 0; i < count; i++) acc += (double)*(Half*)(p + i * s);
        sum = acc; return true;
    }
}
public readonly struct HalfMaxK : INpyReducingInnerLoop<MinMaxAcc>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref MinMaxAcc a)
    {
        byte* p = (byte*)dp[0]; long s = st[0];
        double best = a.Best; bool seen = a.Seen;
        for (long i = 0; i < count; i++)
        {
            double v = (double)*(Half*)(p + i * s);
            if (double.IsNaN(v)) { a.SawNaN = true; a.Best = v; return false; }
            if (!seen || v > best) { best = v; seen = true; }
        }
        a.Best = best; a.Seen = seen; return true;
    }
}
public readonly struct CplxSumK : INpyReducingInnerLoop<Complex>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref Complex sum)
    {
        byte* p = (byte*)dp[0]; long s = st[0]; Complex acc = sum;
        for (long i = 0; i < count; i++) acc += *(Complex*)(p + i * s);
        sum = acc; return true;
    }
}
public readonly struct CplxMaxK : INpyReducingInnerLoop<CplxAcc>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref CplxAcc a)
    {
        byte* p = (byte*)dp[0]; long s = st[0];
        Complex best = a.Best; bool seen = a.Seen;
        for (long i = 0; i < count; i++)
        {
            var v = *(Complex*)(p + i * s);
            if (!seen || v.Real > best.Real || (v.Real == best.Real && v.Imaginary > best.Imaginary)) { best = v; seen = true; }
        }
        a.Best = best; a.Seen = seen; return true;
    }
}
public readonly struct BoolAnyK : INpyReducingInnerLoop<int>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref int found)
    {
        byte* p = (byte*)dp[0]; long s = st[0];
        for (long i = 0; i < count; i++) if (p[i * s] != 0) { found = 1; return false; }
        return true;
    }
}
