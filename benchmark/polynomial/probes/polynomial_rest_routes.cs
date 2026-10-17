#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Nullable=disable
// =============================================================================
// polynomial_rest_routes.cs — which IMPLEMENTATION should compute each numpy.polynomial name NumSharp still lacks?
//
// For every name the candidate routes run in ONE process, interleaved round by round (so their relative cost holds
// on a loaded host), and every route is byte-compared with the row's FIRST route, the eager composition, which is
// NumPy's own statement sequence. Sections (pass any subset as the first argument, default all):
//   W  weight functions: eager ufuncs vs one fused np.evaluate tree (and NDExpr.Power vs Square for x**2)
//   P  chebpts1 / chebpts2 / chebgauss's nodes: eager vs arange/linspace + one fused unary pass
//   R  polyvalfromroots: np.prod(x - r, 0) vs np.evaluate(Prod) vs the U3 kernel (polyval) as the cost proxy
//   F  the fits' _fit statements: vander, the column norm (eager vs fused streaming), scaling, lstsq, whole fit
//   G  leggauss / hermgauss compositions and their parts (companion, eigvalsh)
//   I  chebinterpolate's composition
//   C  the unit functions the classes' members call (class proxies)
// Every route disposes its intermediates, so no route pays another's finalizer backlog.
// Verdicts: docs/plans/numpy-polynomial-remaining-compute.md. Run from the repository root:
//   DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/polynomial/probes/polynomial_rest_routes.cs WPRFGIC
// =============================================================================

using System;
using System.Diagnostics;
using System.Linq;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Interop.OpenBLAS;

// Two P-cores (CPUs 2 and 4), never one: a single-CPU pin starves the background tier-1 compiler.
Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)0x14;
// One BLAS thread, the parity configuration of the oracle tiers (and NumPy's OPENBLAS_NUM_THREADS=1 twin).
OpenBlasEngine.Enable(threads: 1);
Console.WriteLine("blas: " + OpenBlasEngine.Info);

var P = np.polynomial.polynomial;
var C = np.polynomial.chebyshev;
string sel = args.Length > 0 ? args[0] : "WPRFGIC";

if (sel.Contains('W'))
{
    foreach (long n in new long[] { 1_000, 100_000, 1_000_000 })
    foreach (var dt in new[] { np.float64, np.float32, np.float16 })
    {
        using var x0 = Probe.Data(n, 0.95);
        using var x = x0.astype(dt);
        NDExpr X = x;
        var one = Probe.One; var zero = Probe.Zero; var two = Probe.Two; var mhalf = Probe.MinusHalf;
        Probe.Route("chebweight", dt, n, "",
            ("eager", () => { using var a = one + x; using var b = np.sqrt(a); using var c = one - x; using var d = np.sqrt(c); using var e = b * d; return one / e; }),
            ("fused", () => np.evaluate(1.0 / (NDExpr.Sqrt(1.0 + X) * NDExpr.Sqrt(1.0 - X)))));
        Probe.Route("legweight", dt, n, "",
            ("eager", () => { using var a = x * zero; return a + one; }),
            ("fused", () => np.evaluate(X * 0.0 + 1.0)));
        Probe.Route("lagweight", dt, n, "",
            ("eager", () => { using var a = np.negative(x); return np.exp(a); }),
            ("fused", () => np.evaluate(NDExpr.Exp(-X))));
        // NumPy's `x**2` on an array is np.square (number.c fast_scalar_power): the eager route spells it as NumPy's
        // np.power fast path, fusedSq as NumPy's operator, fusedPow as NDExpr's pow-per-element Power node.
        Probe.Route("hermweight", dt, n, "",
            ("eager", () => { using var a = np.power(x, two); using var b = np.negative(a); return np.exp(b); }),
            ("fusedPow", () => np.evaluate(NDExpr.Exp(-NDExpr.Power(X, 2)))),
            ("fusedSq", () => np.evaluate(NDExpr.Exp(-NDExpr.Square(X)))));
        Probe.Route("hermeweight", dt, n, "",
            ("eager", () => { using var a = np.power(x, two); using var b = np.multiply(mhalf, a); return np.exp(b); }),
            ("fusedPow", () => np.evaluate(NDExpr.Exp(-0.5 * NDExpr.Power(X, 2)))),
            ("fusedSq", () => np.evaluate(NDExpr.Exp(-0.5 * NDExpr.Square(X)))));
    }
}

if (sel.Contains('P'))
{
    foreach (long n in new long[] { 10, 100, 10_000, 1_000_000 })
    {
        // NumPy computes 0.5*np.pi/_npts as Python floats, left to right: (0.5*pi)/n.
        double c = 0.5 * Math.PI / n;
        NDArray cArr = c, pi = Math.PI, d2 = 2.0 * n;
        double d2d = 2.0 * n;
        Probe.Route("chebpts1", null, n, "",
            ("eager", () => { using var k = np.arange(-n + 1, n + 1, 2, np.int64); using var t = np.multiply(cArr, k); return np.sin(t); }),
            ("arange+fused", () => { using var k = np.arange(-n + 1, n + 1, 2, np.int64); return np.evaluate(NDExpr.Sin(c * (NDExpr)k)); }));
        Probe.Route("chebpts2", null, n, "",
            ("eager", () => { using var t = np.linspace(-Math.PI, 0.0, n); return np.cos(t); }),
            ("linspace+fused", () => { using var t = np.linspace(-Math.PI, 0.0, n); return np.evaluate(NDExpr.Cos(t)); }));
        Probe.Route("chebgauss.x", null, n, "",
            ("eager", () => { using var k = np.arange(1, 2 * n, 2, np.int64); using var a = np.multiply(pi, k); using var b = np.divide(a, d2); return np.cos(b); }),
            ("arange+fused", () => { using var k = np.arange(1, 2 * n, 2, np.int64); return np.evaluate(NDExpr.Cos((Math.PI * (NDExpr)k) / d2d)); }));
    }
}

if (sel.Contains('R'))
{
    foreach (long n in new long[] { 1_000, 100_000, 1_000_000 })
    foreach (int nr in new[] { 3, 10, 30 })
    {
        using var x = Probe.Data(n, 1.0);
        using var r0 = Probe.Data(nr, 0.9);
        using var r = r0.reshape(nr, 1);          // NumPy's tensor reshape of a 1-D r for a 1-D x
        using var cpoly = Probe.Data(nr + 1, 0.9);  // U3 proxy: nr Horner steps = nr sub+mul steps of the root product
        Probe.Route("polyvalfromroots", np.float64, n, $"r{nr}",
            ("compose", () => { using var t = np.subtract(x, r); return np.prod(t, axis: 0); }),
            ("evalProd", () => np.evaluate(NDExpr.Prod((NDExpr)x - (NDExpr)r, axis: 0))),
            ("U3polyval", () => P.polyval(x, cpoly)));
    }
}

if (sel.Contains('F'))
{
    foreach (long n in new long[] { 100, 1_000, 100_000 })
    foreach (int d in new[] { 3, 10, 30 })
    {
        using var x = Probe.Data(n, 1.0);
        using var x3 = np.multiply(3.0, x);
        using var y = np.sin(x3);
        using var van = C.chebvander(x, d);
        var lhs = van.T;
        using var sq = np.square(lhs);
        using var ss = np.sum(sq, axis: 1);
        using var scl = np.sqrt(ss);
        using var a = lhs.T / scl;
        Probe.Route("fit.vander", np.float64, n, $"d{d}", ("chebvander", () => C.chebvander(x, d)));
        Probe.Route("fit.norms", np.float64, n, $"d{d}",
            ("eager", () => { using var s1 = np.square(lhs); using var s2 = np.sum(s1, axis: 1); return np.sqrt(s2); }),
            ("fusedSum", () => { using var s = np.evaluate(NDExpr.Sum(NDExpr.Square(lhs), axis: 1)); return np.sqrt(s); }));
        Probe.Route("fit.scale", np.float64, n, $"d{d}", ("lhs.T/scl", () => lhs.T / scl));
        Probe.Route("fit.lstsq", np.float64, n, $"d{d}", ("lstsq", () => Probe.Solve(a, y, null)));
        Probe.Route("chebfit.compose", np.float64, n, $"d{d}", ("compose", () => Probe.ChebFit(x, y, d)));
    }
}

if (sel.Contains('G'))
{
    foreach (int deg in new[] { 5, 20, 50, 100 })
    {
        var cl = new long[deg + 1]; cl[deg] = 1;
        using var c = np.array(cl);
        using var m = np.polynomial.legendre.legcompanion(c);
        Probe.Route("gauss.legcompanion", null, deg, "", ("legcompanion", () => np.polynomial.legendre.legcompanion(c)));
        Probe.Route("gauss.eigvalsh", null, deg, "", ("eigvalsh", () => np.linalg.eigvalsh(m)));
        Probe.Route("leggauss.compose", null, deg, "", ("compose", () => Probe.LegGaussNodes(deg)));
        Probe.Route("hermgauss.compose", null, deg, "", ("compose", () => Probe.HermGaussNodes(deg)));
    }
}

if (sel.Contains('I'))
{
    foreach (int deg in new[] { 5, 20, 100 })
        Probe.Route("chebinterpolate.compose", null, deg, "", ("compose", () => Probe.ChebInterpolate(deg)));
}

if (sel.Contains('C'))
{
    using var c10 = Probe.Data(11, 1.0);
    using var c5 = Probe.Data(6, 1.0);
    using var x1k = Probe.Data(1_000, 1.0);
    using var x100k = Probe.Data(100_000, 1.0);
    var dom = new double[] { -1, 1 };
    Probe.Route("class.p+q proxy chebadd", null, 11, "", ("chebadd", () => C.chebadd(c10, c5)));
    Probe.Route("class.p*q proxy chebmul", null, 11, "", ("chebmul", () => C.chebmul(c10, c5)));
    Probe.Route("class.ch(x) proxy chebval", null, 1_000, "", ("chebval", () => C.chebval(x1k, c10)));
    Probe.Route("class.ch(x) proxy chebval", null, 100_000, "", ("chebval", () => C.chebval(x100k, c10)));
    Probe.Route("class.ch(x) proxy map+chebval", null, 100_000, "",
        ("mapdomain+chebval", () => { using var t = np.polynomial.polyutils.mapdomain(x100k, dom, dom); return C.chebval(t, c10); }));
}

/// <summary>Inputs, the NumPy statement sequences being compared, and the interleaved-timing harness.</summary>
static class Probe
{
    /// <summary>Held literal operands, converted once so no route allocates (and finalizes) a 0-d literal per call.</summary>
    public static readonly NDArray One = 1.0, Zero = 0.0, Two = 2, MinusHalf = -0.5;

    /// <summary>
    ///     The shared input <c>sin(k * 1.2345678901) * amp</c> for k = 0..n-1: deterministic, random-looking low bits, and
    ///     bit-identical to the NumPy twins' (an exact multiply and the same CRT <c>sin</c> on both sides).
    /// </summary>
    /// <param name="n">Element count.</param>
    /// <param name="amp">Amplitude; the values lie in [-amp, amp].</param>
    /// <returns>A fresh float64 array of <paramref name="n"/> elements.</returns>
    public static NDArray Data(long n, double amp)
    {
        using var k = np.arange(0, n, 1, np.int64);
        using var t = np.multiply(k, 1.2345678901);
        using var s = np.sin(t);
        return np.multiply(s, amp);
    }

    /// <summary><c>np.linalg.lstsq(a, b, rcond)</c>, keeping the solution and disposing the other three results.</summary>
    /// <param name="a">Design matrix.</param><param name="b">Right-hand side.</param>
    /// <param name="rcond">Cut-off; null is NumPy's default.</param>
    /// <returns>The solution.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed.</exception>
    public static NDArray Solve(NDArray a, NDArray b, double? rcond)
    {
        var t = np.linalg.lstsq(a, b, rcond);
        t.Residuals.Dispose(); t.Rank.Dispose(); t.SingularValues.Dispose();
        return t.Solution;
    }

    /// <summary>NumPy's <c>pu._fit</c> for the Chebyshev basis, statement for statement, on NumSharp's eager ufuncs.</summary>
    /// <param name="x">Sample points (1-D).</param><param name="y">Sample values (1-D).</param>
    /// <param name="deg">Degree.</param>
    /// <returns>The coefficients, low degree first.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed.</exception>
    public static NDArray ChebFit(NDArray x, NDArray y, int deg)
    {
        using var xx = x + Zero;
        using var yy = y + Zero;
        using var van = np.polynomial.chebyshev.chebvander(xx, deg);
        var lhs = van.T;
        var rhs = yy.T;
        using var sq = np.square(lhs);
        using var ss = np.sum(sq, axis: 1);
        using var scl = np.sqrt(ss);
        using var a = lhs.T / scl;
        // rcond = len(x) * finfo(float64).eps, as _fit computes it for a float64 x.
        using var c = Solve(a, rhs.T, (double)xx.size * 2.220446049250313e-16);
        using var cT = c.T / scl;
        return cT.T.copy();
    }

    /// <summary>NumPy's <c>leggauss(deg)</c> statement for statement; returns the nodes (the weights are computed and dropped).</summary>
    /// <param name="deg">Positive degree.</param>
    /// <returns>The symmetrised nodes.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed (eigvalsh).</exception>
    public static NDArray LegGaussNodes(int deg)
    {
        var L = np.polynomial.legendre;
        var cl = new long[deg + 1]; cl[deg] = 1;
        using var c = np.array(cl);
        using var m = L.legcompanion(c);
        var x = np.linalg.eigvalsh(m);
        using var dy = L.legval(x, c);
        using var dc = L.legder(c);
        using var df = L.legval(x, dc);
        using (var q = dy / df) np.subtract(x, q, @out: x);
        using var c1 = c["1:"];
        using var fm = L.legval(x, c1);
        using (var ab = np.abs(fm)) using (var mx = np.max(ab)) np.divide(fm, mx, @out: fm);
        using (var ab = np.abs(df)) using (var mx = np.max(ab)) np.divide(df, mx, @out: df);
        using var fd = fm * df;
        using var w = One / fd;
        using var wr = w["::-1"];
        using var ws = w + wr;
        using var w2 = ws / Two;
        using (var s = np.sum(w2)) using (var f = 2.0 / s) np.multiply(w2, f, @out: w2);
        using var xr = x["::-1"];
        using var xs = x - xr;
        x.Dispose();
        return xs / Two;
    }

    /// <summary>
    ///     NumPy's <c>_normed_hermite_n</c> (or <c>_normed_hermite_e_n</c>) composed from its array statements: the
    ///     route a U3 step-table program replaces (it costs ~6 array ops per step).
    /// </summary>
    /// <param name="x">Points (float64).</param><param name="n">Degree.</param>
    /// <param name="e">True for the probabilists' (HermiteE) recurrence.</param>
    /// <returns>The normalised polynomial's values at <paramref name="x"/>.</returns>
    public static NDArray NormedHermiteN(NDArray x, int n, bool e)
    {
        double c1v = 1.0 / Math.Sqrt(Math.Sqrt(e ? 2 * Math.PI : Math.PI));
        if (n == 0) return np.full(x.Shape, c1v);
        // c0 starts as the Python float 0.0 and c1 as an np.float64; both turn into arrays inside the loop.
        NDArray c0 = NDArray.Scalar(0.0), c1 = NDArray.Scalar(c1v);
        double nd = n;
        for (int i = 0; i < n - 1; i++)
        {
            var tmp = c0;
            using (var nc1 = -c1) using (NDArray s1 = Math.Sqrt((nd - 1.0) / nd)) c0 = nc1 * s1;
            using (var a = c1 * x) using (NDArray s2 = Math.Sqrt((e ? 1.0 : 2.0) / nd)) using (var b = a * s2) { var c1n = tmp + b; c1.Dispose(); c1 = c1n; }
            tmp.Dispose();
            nd -= 1.0;
        }
        using var cx = c1 * x;
        NDArray r;
        if (e) r = c0 + cx;
        else { using NDArray sq2 = Math.Sqrt(2.0); using var cxs = cx * sq2; r = c0 + cxs; }
        c0.Dispose(); c1.Dispose();
        return r;
    }

    /// <summary>NumPy's <c>hermgauss(deg)</c> statement for statement; returns the nodes.</summary>
    /// <param name="deg">Positive degree.</param>
    /// <returns>The symmetrised nodes.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed (eigvalsh).</exception>
    public static NDArray HermGaussNodes(int deg)
    {
        var cl = new double[deg + 1]; cl[deg] = 1;
        using var c = np.array(cl);
        using var m = np.polynomial.hermite.hermcompanion(c);
        var x = np.linalg.eigvalsh(m);
        using var dy = NormedHermiteN(x, deg, false);
        using var df0 = NormedHermiteN(x, deg - 1, false);
        using NDArray k = Math.Sqrt(2 * deg);
        using var df = df0 * k;
        using (var q = dy / df) np.subtract(x, q, @out: x);
        using var fm = NormedHermiteN(x, deg - 1, false);
        using (var ab = np.abs(fm)) using (var mx = np.max(ab)) np.divide(fm, mx, @out: fm);
        using var ff = fm * fm;
        using var w = One / ff;
        using var xr = x["::-1"];
        using var xs = x - xr;
        x.Dispose();
        return xs / Two;
    }

    /// <summary>NumPy's <c>chebinterpolate(np.sin, deg)</c> statement for statement.</summary>
    /// <param name="deg">Non-negative degree.</param>
    /// <returns>The Chebyshev coefficients.</returns>
    public static NDArray ChebInterpolate(int deg)
    {
        long order = deg + 1;
        using NDArray cc = 0.5 * Math.PI / order;
        using var k = np.arange(-order + 1, order + 1, 2, np.int64);
        using var t = np.multiply(cc, k);
        using var xc = np.sin(t);
        using var y = np.sin(xc);
        using var m = np.polynomial.chebyshev.chebvander(xc, deg);
        var c = np.dot(m.T, y);
        using (var c0 = c["0"]) using (NDArray o = (double)order) np.divide(c0, o, @out: c0);
        using (var c1 = c["1:"]) using (NDArray o = 0.5 * order) np.divide(c1, o, @out: c1);
        return c;
    }

    /// <summary>
    ///     Times every route of one row interleaved (7 rounds; the per-route minimum is kept) and prints each route's time
    ///     with its byte comparison against the row's first route.
    /// </summary>
    /// <param name="name">Row label.</param><param name="dt">Input dtype for the label, or null.</param>
    /// <param name="n">Size or degree for the label.</param><param name="extra">Extra label text.</param>
    /// <param name="routes">The routes; the first is the reference (NumPy's statement sequence).</param>
    public static void Route(string name, DType dt, long n, string extra, params (string label, Func<NDArray> f)[] routes)
    {
        foreach (var (_, f) in routes) { f().Dispose(); f().Dispose(); }   // tier-up before measuring
        var reference = Bytes(routes[0].f);
        var diffs = routes.Select((r, i) => i == 0 ? "" : DiffText(reference, Bytes(r.f))).ToArray();
        var iters = routes.Select(r => Calibrate(r.f)).ToArray();
        var best = Enumerable.Repeat(double.MaxValue, routes.Length).ToArray();
        for (int round = 0; round < 7; round++)
            for (int i = 0; i < routes.Length; i++)
                best[i] = Math.Min(best[i], Batch(routes[i].f, iters[i]));
        var cells = routes.Select((r, i) => $"{r.label}={best[i],11:F3}us{(diffs[i] == "" ? "" : " " + diffs[i])}");
        Console.WriteLine($"{name,-30} {(dt == null ? "" : dt.name),-8} {n,9} {extra,-4} " + string.Join("  ", cells));
    }

    /// <summary>One result's raw C-order bytes, dtype name and shape.</summary>
    /// <param name="f">The route.</param>
    /// <returns>The triple used for the byte comparison.</returns>
    private static (byte[] bytes, string dtype, string shape) Bytes(Func<NDArray> f)
    {
        using var r = f();
        return (r.tobytes(), r.dtype.name, string.Join(",", r.shape));
    }

    /// <summary><c>[same]</c>, <c>[DIFF k/n]</c> (k differing elements), or the dtype/shape mismatch.</summary>
    /// <param name="a">Reference.</param><param name="b">Candidate.</param>
    /// <returns>The comparison text.</returns>
    private static string DiffText((byte[] bytes, string dtype, string shape) a, (byte[] bytes, string dtype, string shape) b)
    {
        if (a.dtype != b.dtype || a.shape != b.shape) return $"[DIFF {b.dtype}({b.shape}) vs {a.dtype}({a.shape})]";
        if (a.bytes.AsSpan().SequenceEqual(b.bytes)) return "[same]";
        int item = a.dtype == "float64" ? 8 : a.dtype == "float32" ? 4 : a.dtype == "float16" ? 2 : 1;
        long diff = 0;
        for (int i = 0; i < a.bytes.Length; i += item)
            if (!a.bytes.AsSpan(i, item).SequenceEqual(b.bytes.AsSpan(i, item))) diff++;
        return $"[DIFF {diff}/{a.bytes.Length / item}]";
    }

    /// <summary>The smallest power-of-two call count whose batch takes at least 20 ms (capped at 2^20).</summary>
    /// <param name="f">The route.</param>
    /// <returns>Calls per timed batch.</returns>
    private static int Calibrate(Func<NDArray> f)
    {
        int n = 1;
        while (true)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < n; i++) f().Dispose();
            if (sw.Elapsed.TotalMilliseconds >= 20 || n >= 1 << 20) return n;
            n *= 2;
        }
    }

    /// <summary>Microseconds per call over one batch.</summary>
    /// <param name="f">The route.</param><param name="n">Calls.</param>
    /// <returns>Mean microseconds per call within the batch.</returns>
    private static double Batch(Func<NDArray> f, int n)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < n; i++) f().Dispose();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / n;
    }
}
