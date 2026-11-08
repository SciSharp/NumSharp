#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Nullable=disable
// =============================================================================
// polynomial_rest_h2h.cs — head-to-head C# side (twin: polynomial_rest_h2h_numpy.py)
//
// Each label runs the RECOMMENDED NumSharp route for that NumPy cell (docs/plans/numpy-polynomial-remaining-compute.md
// §4), or the existing kernel standing in for a kernel not written yet (its comment says "proxy"), plus the floors
// both sides share (float64 exp/sin/cos, lstsq, eigvalsh, dot) and the two spellings of the fit's column scaling.
// Each cell prints `label  microseconds`: the minimum over 25 batches of at least 5 ms, pinned to CPUs 2 and 4,
// OpenBLAS at one thread. Run the pair ALTERNATELY at least twice and keep each cell's minimum.
// =============================================================================

using System;
using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Interop.OpenBLAS;

Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)0x14;   // two P-cores, never one (tier-1 compiler)
OpenBlasEngine.Enable(threads: 1);
var P = np.polynomial.polynomial;
var C = np.polynomial.chebyshev;
var H = np.polynomial.hermite;
var PU = np.polynomial.polyutils;
NDArray one = H2H.One;

foreach (long n in new long[] { 1000, 1_000_000 })
{
    var x = H2H.Data(n, 0.95); NDExpr X = x;
    H2H.Bench($"chebweight f64 {n}", () => np.evaluate(1.0 / (NDExpr.Sqrt(1.0 + X) * NDExpr.Sqrt(1.0 - X))));
    // NumPy's x**2 on an array is np.square (number.c fast_scalar_power), so Square is the faithful node.
    H2H.Bench($"hermweight f64 {n}", () => np.evaluate(NDExpr.Exp(-NDExpr.Square(X))));
}
{
    var x = H2H.Data(1_000_000, 0.95); NDExpr X = x;
    H2H.Bench("legweight f64 1000000", () => np.evaluate(X * 0.0 + 1.0));
    H2H.Bench("lagweight f64 1000000", () => np.evaluate(NDExpr.Exp(-X)));
    var x32 = x.astype(np.float32); NDExpr X32 = x32;
    H2H.Bench("chebweight f32 1000000", () => np.evaluate(1.0 / (NDExpr.Sqrt(1.0 + X32) * NDExpr.Sqrt(1.0 - X32))));
    H2H.Bench("hermweight f32 1000000", () => np.evaluate(NDExpr.Exp(-NDExpr.Square(X32))));
    // float16 rides the eager house ufuncs: NDExpr's float16 path is 2-5x slower than their widen-compute-narrow SIMD.
    var x16 = H2H.Data(100_000, 0.95).astype(np.float16);
    H2H.Bench("chebweight f16 100000", () => { using var a = one + x16; using var b = np.sqrt(a); using var c = one - x16; using var d = np.sqrt(c); using var e = b * d; return one / e; });
}
foreach (long n in new long[] { 100, 1_000_000 })
{
    double cd = 0.5 * Math.PI / n;
    H2H.Bench($"chebpts1 {n}", () => { using var k = np.arange(-n + 1, n + 1, 2, np.int64); return np.evaluate(NDExpr.Sin(cd * (NDExpr)k)); });
    H2H.Bench($"chebpts2 {n}", () => { using var t = np.linspace(-Math.PI, 0.0, n); return np.cos(t); });
}
foreach (long n in new long[] { 50, 100_000 })
{
    double d2 = 2.0 * n, wv = Math.PI / n;
    H2H.Bench($"chebgauss {n}", () => { using var k = np.arange(1, 2 * n, 2, np.int64); using var w = np.full(new Shape(n), wv); return np.evaluate(NDExpr.Cos((Math.PI * (NDExpr)k) / d2)); });
}
foreach (int d in new[] { 5, 20, 100 })
{
    H2H.Bench($"leggauss {d}", () => H2H.LagOrLegGauss(d, legendre: true));
    H2H.Bench($"hermgauss {d}", () => H2H.HermGauss(d, normedE: false));
    // the kernel stand-in for one _normed_hermite_n pass: U3's hermval on the deg points (the same Clenshaw shape)
    using var xh = H2H.Data(d, 2.0);
    using var ch = np.ones(new Shape(d + 1));
    H2H.Bench($"proxy hermval(x[{d}], ones) x1", () => H.hermval(xh, ch));
    H2H.Bench($"part normed_hermite_n compose {d}", () => H2H.NormedHermiteN(xh, d, false));
}
H2H.Bench("laggauss 20", () => H2H.LagOrLegGauss(20, legendre: false));
H2H.Bench("hermegauss 20", () => H2H.HermGauss(20, normedE: true));
foreach (int d in new[] { 5, 20, 100 })
    H2H.Bench($"chebinterpolate {d}", () => H2H.ChebInterpolate(d));
foreach (var (n, d) in new[] { (100L, 3), (1000L, 10), (100_000L, 10) })
{
    var xf = H2H.Data(n, 1.0);
    var yf = np.sin(np.multiply(3.0, xf));
    H2H.Bench($"chebfit n{n} d{d}", () => H2H.Fit(C.chebvander, xf, yf, d));
}
{
    var xf = H2H.Data(1000, 1.0);
    var yf = np.sin(np.multiply(3.0, xf));
    H2H.Bench("polyfit n1000 d10", () => H2H.Fit(P.polyvander, xf, yf, 10));
}
foreach (long n in new long[] { 1000, 100_000, 1_000_000 })
{
    var xv = H2H.Data(n, 1.0);
    var c11 = H2H.Data(11, 0.9);
    H2H.Bench($"polyvalfromroots f64 n{n} r10", () => P.polyval(xv, c11));   // proxy: U3 kernel, same per-point op count
}
{
    var xv = H2H.Data(100_000, 1.0).ToArray<double>();
    var xc = np.array(Array.ConvertAll(xv, v => new Complex(v, -v)));
    var cc = np.array(Array.ConvertAll(H2H.Data(11, 0.9).ToArray<double>(), v => new Complex(v, 0.5)));
    H2H.Bench("polyvalfromroots c128 n100000 r10", () => P.polyval(xc, cc));   // proxy
    var c11 = H2H.Data(11, 0.9);
    H2H.Bench("polyvalfromroots scalar r10", () => P.polyval(0.3, c11));      // proxy (weak scalar x, 0-d result)
}
{
    var c10 = H2H.Data(11, 1.0);
    var c5 = H2H.Data(6, 1.0);
    var x1k = H2H.Data(1000, 1.0);
    var x100k = H2H.Data(100_000, 1.0);
    var dom = new double[] { -1, 1 };
    H2H.Bench("class p+q", () => P.polyadd(c10, c5));                                  // proxy: the module function
    H2H.Bench("class p*q", () => P.polymul(c10, c5));                                  // proxy
    H2H.Bench("class ch(x) 1000", () => { using var t = PU.mapdomain(x1k, dom, dom); return C.chebval(t, c10); });
    H2H.Bench("class ch(x) 100000", () => { using var t = PU.mapdomain(x100k, dom, dom); return C.chebval(t, c10); });
    H2H.Bench("class ch.convert(kind=Polynomial)", () => C.cheb2poly(c10));            // proxy (cost class only)
    var xf = H2H.Data(1000, 1.0);
    var yf = np.sin(np.multiply(3.0, xf));
    H2H.Bench("class Chebyshev.fit n1000 d10", () => { using var t = PU.mapdomain(xf, dom, dom); return H2H.Fit(C.chebvander, t, yf, 10); });
    H2H.Bench("format_float(0.1)", () => { _ = np.format_float_positional(0.1); return null; }); // proxy: the Dragon4 core
}
foreach (long n in new long[] { 1000, 1_000_000 })
{
    var xs = H2H.Data(n, 1.0);
    H2H.Bench($"floor np.exp f64 {n}", () => np.exp(xs));
    H2H.Bench($"floor np.sin f64 {n}", () => np.sin(xs));
    H2H.Bench($"floor np.cos f64 {n}", () => np.cos(xs));
}
foreach (var (n, d) in new[] { (100L, 3), (1000L, 10), (100_000L, 10) })
{
    var xs = H2H.Data(n, 1.0);
    var van = C.chebvander(xs, d);
    var ys = np.sin(np.multiply(3.0, xs));
    H2H.Bench($"floor lstsq n{n} d{d}", () => H2H.Solve(van, ys, null));
}
foreach (int deg in new[] { 5, 20, 100 })
{
    var cl = new long[deg + 1]; cl[deg] = 1;
    var m = np.polynomial.legendre.legcompanion(np.array(cl));
    H2H.Bench($"floor eigvalsh {deg}", () => np.linalg.eigvalsh(m));
}
foreach (int deg in new[] { 20, 100 })
{
    var xs = H2H.Data(deg + 1, 0.9);
    var mv = C.chebvander(xs, deg);
    var ys = np.sin(xs);
    H2H.Bench($"floor dot(m.T, y) {deg}", () => np.dot(mv.T, ys));
}
foreach (long n in new long[] { 1000, 100_000 })
{
    var lhs = H2H.Data(11 * n, 1.0).reshape(11, n);
    var scl = np.add(np.abs(np.sin(np.arange(0, 11, 1, np.int64))), 1.0);
    var sclc = scl.reshape(11, 1);
    H2H.Bench($"scale lhs.T / scl n{n} d10", () => lhs.T / scl);                     // NumPy's spelling (F result)
    H2H.Bench($"scale lhs / scl[:,None] n{n} d10", () => lhs / sclc);                // same bytes, C result
}

/// <summary>The recommended compositions and the timing harness of the head-to-head.</summary>
static class H2H
{
    /// <summary>Held literal operands, converted once so a route does not allocate (and finalize) a 0-d literal per call.</summary>
    public static readonly NDArray One = 1.0, Zero = 0.0, Two = 2;

    /// <summary>The shared input <c>sin(k * 1.2345678901) * amp</c>, bit-identical to the NumPy twin's.</summary>
    /// <param name="n">Element count.</param><param name="amp">Amplitude.</param>
    /// <returns>A fresh float64 array.</returns>
    public static NDArray Data(long n, double amp)
    {
        using var k = np.arange(0, n, 1, np.int64);
        using var t = np.multiply(k, 1.2345678901);
        using var s = np.sin(t);
        return np.multiply(s, amp);
    }

    /// <summary><c>np.linalg.lstsq(a, b, rcond)</c>, keeping the solution and disposing the other three results.</summary>
    /// <param name="a">Design matrix.</param><param name="b">Right-hand side.</param><param name="rcond">Cut-off.</param>
    /// <returns>The solution.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed.</exception>
    public static NDArray Solve(NDArray a, NDArray b, double? rcond)
    {
        var t = np.linalg.lstsq(a, b, rcond);
        t.Residuals.Dispose(); t.Rank.Dispose(); t.SingularValues.Dispose();
        return t.Solution;
    }

    /// <summary>
    ///     The recommended fit: NumPy's <c>pu._fit</c> with two bit-neutral savings — the column norm as one fused
    ///     streaming pass, and the scaling done in place on the fresh Vandermonde buffer's C-ordered rows
    ///     (<c>lhs / scl[:, None]</c> has the bytes of NumPy's <c>(lhs.T / scl).T</c>, element for element).
    /// </summary>
    /// <param name="vander">The basis' <c>{p}vander(x, deg)</c>.</param>
    /// <param name="x">Points.</param><param name="y">Values.</param><param name="deg">Degree.</param>
    /// <returns>The coefficients.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed.</exception>
    public static NDArray Fit(Func<object, int, NDArray> vander, NDArray x, NDArray y, int deg)
    {
        using var xx = x + Zero;
        using var yy = y + Zero;
        using var van = vander(xx, deg);
        var lhs = van.T;   // C-ordered (deg+1, n): {p}vander returns an F view of a C buffer
        using var s = np.evaluate(NDExpr.Sum(NDExpr.Square(lhs), axis: 1));
        using var scl = np.sqrt(s);
        using var sclc = scl.reshape(deg + 1, 1);
        np.divide(lhs, sclc, @out: lhs);   // one division of a fresh buffer: the van view now holds lhs.T / scl
        var c = Solve(van, yy, (double)xx.size * 2.220446049250313e-16);
        np.divide(c, scl, @out: c);
        return c;
    }

    /// <summary>NumPy's <c>leggauss</c> or <c>laggauss</c>, statement for statement; returns the nodes.</summary>
    /// <param name="deg">Positive degree.</param>
    /// <param name="legendre">True for Legendre (symmetrised, weights summing to 2), false for Laguerre (summing to 1).</param>
    /// <returns>The nodes.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed (eigvalsh).</exception>
    public static NDArray LagOrLegGauss(int deg, bool legendre)
    {
        var cl = new long[deg + 1]; cl[deg] = 1;
        using var c = np.array(cl);
        using var m = legendre ? np.polynomial.legendre.legcompanion(c) : np.polynomial.laguerre.lagcompanion(c);
        var x = np.linalg.eigvalsh(m);
        NDArray Val(NDArray at, NDArray coef) => legendre ? np.polynomial.legendre.legval(at, coef) : np.polynomial.laguerre.lagval(at, coef);
        using var dy = Val(x, c);
        using var dc = legendre ? np.polynomial.legendre.legder(c) : np.polynomial.laguerre.lagder(c);
        using var df = Val(x, dc);
        using (var q = dy / df) np.subtract(x, q, @out: x);
        using var c1 = c["1:"];
        using var fm = Val(x, c1);
        using (var ab = np.abs(fm)) using (var mx = np.max(ab)) np.divide(fm, mx, @out: fm);
        using (var ab = np.abs(df)) using (var mx = np.max(ab)) np.divide(df, mx, @out: df);
        using var fd = fm * df;
        using var w = One / fd;
        if (!legendre)
        {
            using (var s = np.sum(w)) np.divide(w, s, @out: w);
            return x;
        }
        using var wr = w["::-1"];
        using var ws = w + wr;
        using var w2 = ws / Two;
        using (var s = np.sum(w2)) using (var f = 2.0 / s) np.multiply(w2, f, @out: w2);
        using var xr = x["::-1"];
        using var xs = x - xr;
        x.Dispose();
        return xs / Two;
    }

    /// <summary>NumPy's <c>_normed_hermite_n</c> / <c>_normed_hermite_e_n</c> from array statements (the part a U3 program replaces).</summary>
    /// <param name="x">Points.</param><param name="n">Degree.</param><param name="e">True for HermiteE.</param>
    /// <returns>The normalised polynomial's values.</returns>
    public static NDArray NormedHermiteN(NDArray x, int n, bool e)
    {
        double c1v = 1.0 / Math.Sqrt(Math.Sqrt(e ? 2 * Math.PI : Math.PI));
        if (n == 0) return np.full(x.Shape, c1v);
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

    /// <summary>NumPy's <c>hermgauss</c> or <c>hermegauss</c>, statement for statement; returns the nodes.</summary>
    /// <param name="deg">Positive degree.</param><param name="normedE">True for HermiteE.</param>
    /// <returns>The nodes.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed (eigvalsh).</exception>
    public static NDArray HermGauss(int deg, bool normedE)
    {
        var cl = new double[deg + 1]; cl[deg] = 1;
        using var c = np.array(cl);
        using var m = normedE ? np.polynomial.hermite_e.hermecompanion(c) : np.polynomial.hermite.hermcompanion(c);
        var x = np.linalg.eigvalsh(m);
        using var dy = NormedHermiteN(x, deg, normedE);
        using var df0 = NormedHermiteN(x, deg - 1, normedE);
        using NDArray k = Math.Sqrt(normedE ? deg : 2 * deg);
        using var df = df0 * k;
        using (var q = dy / df) np.subtract(x, q, @out: x);
        using var fm = NormedHermiteN(x, deg - 1, normedE);
        using (var ab = np.abs(fm)) using (var mx = np.max(ab)) np.divide(fm, mx, @out: fm);
        using var ff = fm * fm;
        using var w = One / ff;
        using var wr = w["::-1"];
        using var ws = w + wr;
        using var w2 = ws / Two;
        using (var s = np.sum(w2)) using (var f = Math.Sqrt(normedE ? 2 * Math.PI : Math.PI) / s) np.multiply(w2, f, @out: w2);
        using var xr = x["::-1"];
        using var xs = x - xr;
        x.Dispose();
        return xs / Two;
    }

    /// <summary>The recommended <c>chebinterpolate(np.sin, deg)</c>: chebpts1 (fused), the callable, chebvander, np.dot, two in-place divides.</summary>
    /// <param name="deg">Non-negative degree.</param>
    /// <returns>The coefficients.</returns>
    public static NDArray ChebInterpolate(int deg)
    {
        long order = deg + 1;
        double cd = 0.5 * Math.PI / order;
        using var k = np.arange(-order + 1, order + 1, 2, np.int64);
        using var xc = np.evaluate(NDExpr.Sin(cd * (NDExpr)k));
        using var y = np.sin(xc);
        using var m = np.polynomial.chebyshev.chebvander(xc, deg);
        var c = np.dot(m.T, y);
        using (var c0 = c["0"]) using (NDArray o = (double)order) np.divide(c0, o, @out: c0);
        using (var c1 = c["1:"]) using (NDArray o = 0.5 * order) np.divide(c1, o, @out: c1);
        return c;
    }

    /// <summary>Prints the minimum microseconds per call over 25 batches of at least 5 ms (after three warm calls).</summary>
    /// <param name="label">The cell label (identical in the NumPy twin).</param>
    /// <param name="f">The route; a null result is allowed (string-returning cells).</param>
    public static void Bench(string label, Func<NDArray> f)
    {
        f()?.Dispose(); f()?.Dispose(); f()?.Dispose();
        int n = 1;
        while (true)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < n; i++) f()?.Dispose();
            if (sw.Elapsed.TotalMilliseconds >= 5 || n >= 1 << 20) break;
            n *= 2;
        }
        double best = double.MaxValue;
        for (int r = 0; r < 25; r++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < n; i++) f()?.Dispose();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / n);
        }
        Console.WriteLine($"{label,-40} {best,12:F3}");
    }
}
