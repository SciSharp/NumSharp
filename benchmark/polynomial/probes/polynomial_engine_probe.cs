#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
// =============================================================================
// polynomial_engine_probe.cs — which ENGINE should power the numpy.polynomial port?
//
// For each hot shape of the package the SAME NumPy 2.4.2 algorithm is implemented several ways,
// every variant is byte-compared with NumPy's own output (written by polynomial_engine_numpy.py) and
// timed best-of-7. The verdict and the full table live in docs/plans/numpy-polynomial.md §9.
//
//   P1     1-to-1 port: each NumPy array expression -> the matching np.* / NDArray op, element
//          reads/writes through typed accessors, temporaries reclaimed by an NDScope (what [NDScoped]
//          weaves into a real np.* method).
//   P1lit  P1 with NumPy's literal c[i] element spelling (0-d NDArray views).
//   P1out  the port rewritten to reuse buffers through out= (no per-step allocation).
//   Expr   np.evaluate over one NDExpr tree — only Horner is expressible (see §9).
//   Span   typed C# code: span loops for coefficient algebra, fused Vector256 kernels for
//          evaluation/Vandermonde, reproducing NumPy's per-element op sequence exactly.
//
// Usage (from this directory; generate data first with the NumPy twin):
//   set NS_PROBE_AFFINITY=0xFF        (same mask as the twin; hybrid-core hosts)
//   set DOTNET_TC_CallCountingDelayMs=0
//   dotnet run -c Release polynomial_engine_probe.cs -- data [SEVLM]
// Sections: S small-series algebra, E evaluation over x, V Vandermonde, L strided-x layout cost,
// M memory held by a scoped 1-to-1 port. Default: SEVL.
// PublishAot=false is load-bearing: without it DynamicMethod is unavailable and every NumSharp IL
// kernel silently falls back to a 10x slower path, which would corrupt every P1/P1out/Expr number.
// =============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends.Iteration;
#nullable disable

// Pin before anything is JIT-ed so every timed call runs on the chosen cores. Several logical CPUs
// (e.g. 0xFF), not one: a single pinned CPU starves the tiered-JIT background compiler.
if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } affinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

Probe.Data = args.Length > 0 ? args[0] : "data";
Probe.Run(args.Length > 1 ? args[1] : "SEVL");

/// <summary>
///     The probe body: a harness (timing + byte comparison) plus one implementation per strategy for
///     each measured shape. Every implementation mirrors NumPy 2.4.2's source op for op, so any byte
///     difference against NumPy is a real strategy defect, never an algorithm substitution.
/// </summary>
static unsafe class Probe
{
    /// <summary>Directory holding the .npy inputs and NumPy references written by the twin script.</summary>
    public static string Data = "data";

    // ======================================================================== harness

    /// <summary>Loads one .npy the twin wrote; fails loudly (FileNotFound) when the twin was not run.</summary>
    /// <param name="name">File stem inside <see cref="Data"/>.</param>
    /// <returns>The array exactly as NumPy saved it.</returns>
    /// <exception cref="FileNotFoundException">The twin script has not generated <paramref name="name"/>.</exception>
    static NDArray L(string name) => np.load_npy(Path.Combine(Data, name + ".npy"));

    /// <summary>
    ///     Disposes a variant's result so the timing loop measures steady-state allocation (a pooled
    ///     buffer is reused) instead of finalizer pressure — NumPy frees its result by refcount too.
    /// </summary>
    /// <param name="o">An NDArray, a (quotient, remainder) tuple, or anything else (ignored).</param>
    static void Release(object o)
    {
        if (o is NDArray a) a.Dispose();
        else if (o is ValueTuple<NDArray, NDArray> t) { t.Item1.Dispose(); t.Item2.Dispose(); }
    }

    /// <summary>
    ///     Best-of-7 microseconds per call. Warms up for 300 ms first (tiered JIT reaches Tier-1 on this
    ///     host only after repeated calls), then sizes each repeat to ~25 ms so timer resolution never
    ///     dominates a sub-microsecond cell. The minimum is reported because host noise only ever ADDS.
    /// </summary>
    /// <param name="f">The call under test; its result is released after every call.</param>
    /// <returns>Best per-call time in microseconds.</returns>
    static double Bench(Func<object> f)
    {
        var sw = Stopwatch.StartNew();
        int w = 0;
        while (sw.ElapsedMilliseconds < 300 || w < 3) { Release(f()); w++; }
        sw.Restart();
        Release(f());
        double one = Math.Max(sw.Elapsed.TotalMilliseconds, 1e-4);
        int iters = Math.Max(1, (int)(25.0 / one));
        double best = double.MaxValue;
        for (int r = 0; r < 7; r++)
        {
            sw.Restart();
            for (int i = 0; i < iters; i++) Release(f());
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
        }
        return best * 1000.0;
    }

    /// <summary>
    ///     Byte comparison in logical C order (so a moveaxis view compares like NumPy's saved copy).
    ///     Reports the count of differing float64 words and the largest difference in ULP.
    /// </summary>
    /// <param name="got">The variant's result.</param>
    /// <param name="want">NumPy's result.</param>
    /// <returns><c>exact</c>, or a mismatch summary.</returns>
    static string Check(NDArray got, NDArray want)
    {
        if (got.ndim != want.ndim) return $"NDIM {got.ndim} vs {want.ndim}";
        for (int d = 0; d < got.ndim; d++)
            if (got.shape[d] != want.shape[d]) return "SHAPE";
        double[] a = got.ToArray<double>(), b = want.ToArray<double>();
        long bad = 0, maxUlp = 0;
        for (long i = 0; i < a.Length; i++)
        {
            long ia = BitConverter.DoubleToInt64Bits(a[i]), ib = BitConverter.DoubleToInt64Bits(b[i]);
            if (ia != ib) { bad++; maxUlp = Math.Max(maxUlp, Math.Abs(ia - ib)); }
        }
        return bad == 0 ? "exact" : $"{bad}/{a.Length} differ (max {maxUlp} ulp)";
    }

    /// <summary>Checks one call's result, then times the call and prints a TSV row.</summary>
    /// <param name="cell">Cell label shared with the twin's output (the join key).</param>
    /// <param name="variant">Strategy name (P1, P1lit, P1out, Expr, Span).</param>
    /// <param name="f">The call under test.</param>
    /// <param name="check">Correctness check applied to the first call's result.</param>
    static void Row(string cell, string variant, Func<object> f, Func<object, string> check)
    {
        object first = f();
        string ok = check(first);
        Release(first);
        double us = Bench(f);
        Console.WriteLine($"{cell}\t{variant}\t{us:F3}\t{ok}");
    }

    /// <summary>A check against one NumPy reference array.</summary>
    /// <param name="want">NumPy's result.</param>
    /// <returns>A checker for <see cref="Row"/>.</returns>
    static Func<object, string> Vs(NDArray want) => o => Check((NDArray)o, want);

    /// <summary>A check against NumPy's (quotient, remainder) pair.</summary>
    /// <param name="wq">NumPy's quotient.</param>
    /// <param name="wr">NumPy's remainder.</param>
    /// <returns>A checker for <see cref="Row"/>.</returns>
    static Func<object, string> Vs2(NDArray wq, NDArray wr) => o =>
    {
        var (q, r) = ((NDArray, NDArray))o;
        return "q:" + Check(q, wq) + " r:" + Check(r, wr);
    };

    /// <summary>A fresh C-contiguous float64 buffer of <paramref name="like"/>'s dimensions.</summary>
    /// <remarks>
    ///     Built from the DIMENSIONS, never from <c>like.Shape</c>: the NDArray creation paths keep a
    ///     view Shape's strides while allocating only <c>size</c> elements, so a strided view's Shape
    ///     produces a buffer that writes past its end (probed 2026-09-24; see plan §9 side finding).
    /// </remarks>
    /// <param name="like">Array whose dimensions the buffer takes.</param>
    /// <returns>An uninitialised float64 array.</returns>
    static NDArray Fresh(NDArray like) => new NDArray(np.float64, new Shape(like.shape), false);

    /// <summary>Runs the requested sections.</summary>
    /// <param name="sections">Any of S, E, V, L, M.</param>
    public static void Run(string sections)
    {
        Console.WriteLine($"# Vector256 accelerated: {Vector256.IsHardwareAccelerated}");
        Console.WriteLine("cell\tvariant\tus\tvs_numpy");
        var c = new Dictionary<int, NDArray>();
        foreach (int d in new[] { 3, 5, 10, 30, 50 }) c[d] = L($"c{d}");
        if (sections.Contains('S')) SmallSeries(c);
        if (sections.Contains('E')) Evaluation(c);
        if (sections.Contains('V')) Vander();
        if (sections.Contains('L')) Layout(c);
        if (sections.Contains('M')) Memory(c);
    }

    // ======================================================================== S: small-series algebra

    /// <summary>
    ///     Coefficient algebra on 6-51 element series — the shape of U1/U2/U4, the U7 companion build
    ///     and every class operator. NumPy is interpreter-bound here, so the question is per-op cost.
    /// </summary>
    /// <param name="c">Seeded float64 series keyed by degree.</param>
    static void SmallSeries(Dictionary<int, NDArray> c)
    {
        var rAdd = L("ref_polyadd_5_10");
        Row("S polyadd 5+10", "P1", () => PolyaddP1(c[5], c[10]), Vs(rAdd));
        Row("S polyadd 5+10", "Span", () => ToND(AddS(Sp(c[5]), Sp(c[10]))), Vs(rAdd));

        // Expected ≠ for BOTH variants: NumPy convolves a 21-long z-series through cblas ddot.
        var rCm = L("ref_chebmul_10");
        Row("S chebmul 10x10", "P1", () => ChebmulP1(c[10], c[10]), Vs(rCm));
        Row("S chebmul 10x10", "Span", () => ToND(ChebmulS(Sp(c[10]), Sp(c[10]))), Vs(rCm));

        var rLm10 = L("ref_legmul_10");
        Row("S legmul 10x10", "P1lit", () => LegmulP1(c[10], c[10], literal: true), Vs(rLm10));
        Row("S legmul 10x10", "P1", () => LegmulP1(c[10], c[10], literal: false), Vs(rLm10));
        Row("S legmul 10x10", "Span", () => ToND(LegmulS(Sp(c[10]), Sp(c[10]))), Vs(rLm10));

        var rLm50 = L("ref_legmul_50");
        Row("S legmul 50x50", "P1", () => LegmulP1(c[50], c[50], literal: false), Vs(rLm50));
        Row("S legmul 50x50", "Span", () => ToND(LegmulS(Sp(c[50]), Sp(c[50]))), Vs(rLm50));

        var rq = L("ref_legdiv_q"); var rr = L("ref_legdiv_r");
        Row("S legdiv 50/10", "P1", () => LegdivP1(c[50], c[10]), Vs2(rq, rr));
        Row("S legdiv 50/10", "Span", () => { var (q, r) = LegdivS(Sp(c[50]), Sp(c[10])); return (ToND(q), ToND(r)); }, Vs2(rq, rr));

        var rCd = L("ref_chebder_50");
        Row("S chebder 50", "P1", () => ChebderP1(c[50]), Vs(rCd));
        Row("S chebder 50", "Span", () => ToND(ChebderS(Sp(c[50]))), Vs(rCd));

        var rLi = L("ref_legint_50");
        Row("S legint 50", "P1", () => LegintP1(c[50]), Vs(rLi));
        Row("S legint 50", "Span", () => ToND(LegintS(Sp(c[50]))), Vs(rLi));
    }

    /// <summary>Zero-copy read of a contiguous float64 array (all probe inputs are).</summary>
    /// <param name="a">A C-contiguous float64 array.</param>
    /// <returns>Its elements.</returns>
    static ReadOnlySpan<double> Sp(NDArray a) => a.Unsafe.ReadOnlySpan<double>();

    /// <summary>Wraps a span-port result into the NDArray a real np.* function would return (one copy).</summary>
    /// <param name="a">Result coefficients.</param>
    /// <returns>A new float64 array.</returns>
    static NDArray ToND(double[] a) => np.array(a);

    // ---- P1: polyutils ported onto NDArray ops (float64 probe) ----

    /// <summary>Port of <c>pu.trimseq</c>: drop trailing zeros but always keep one coefficient.</summary>
    /// <param name="seq">1-D series.</param>
    /// <returns><paramref name="seq"/> itself or a leading slice of it (a view).</returns>
    static NDArray Trimseq(NDArray seq)
    {
        long n = seq.size;
        if (n == 0 || seq.GetDouble(n - 1) != 0) return seq;
        long i;
        for (i = n - 1; i >= 0; i--) if (seq.GetDouble(i) != 0) break;
        // Python's loop leaves i == 0 when every coefficient is zero, keeping seq[:1].
        if (i < 0) i = 0;
        return seq[$":{i + 1}"];
    }

    /// <summary>Port of <c>pu.as_series(trim=True)</c>: 1-D check, trim, common dtype, COPY.</summary>
    /// <param name="arrs">Coefficient arrays.</param>
    /// <returns>Trimmed copies in <c>np.common_type</c> of the inputs.</returns>
    /// <exception cref="ArgumentException">An input is empty or not 1-D.</exception>
    static NDArray[] AsSeries(params NDArray[] arrs)
    {
        var list = new NDArray[arrs.Length];
        for (int k = 0; k < arrs.Length; k++)
        {
            var a = np.atleast_1d(arrs[k]);
            if (a.size == 0) throw new ArgumentException("Coefficient array is empty");
            if (a.ndim != 1) throw new ArgumentException("Coefficient array is not 1-d");
            list[k] = Trimseq(a);
        }
        var dt = np.common_type(list);
        for (int k = 0; k < list.Length; k++) list[k] = list[k].astype(dt, copy: true);
        return list;
    }

    /// <summary>Port of <c>pu._add</c>: add the shorter series into a copy of the longer one.</summary>
    /// <param name="a">First series.</param>
    /// <param name="b">Second series.</param>
    /// <returns>The trimmed sum.</returns>
    static NDArray AddP1(NDArray a, NDArray b)
    {
        var ab = AsSeries(a, b);
        NDArray c1 = ab[0], c2 = ab[1], ret;
        // c1[:c2.size] += c2 — an in-place slice update, so write through out= (C# += would rebind).
        if (c1.size > c2.size) { var v = c1[$":{c2.size}"]; np.add(v, c2, @out: v); ret = c1; }
        else { var v = c2[$":{c1.size}"]; np.add(v, c1, @out: v); ret = c2; }
        return Trimseq(ret);
    }

    /// <summary>Port of <c>pu._sub</c> (the negate-then-add spelling of the shorter-minuend branch included).</summary>
    /// <param name="a">Minuend series.</param>
    /// <param name="b">Subtrahend series.</param>
    /// <returns>The trimmed difference.</returns>
    static NDArray SubP1(NDArray a, NDArray b)
    {
        var ab = AsSeries(a, b);
        NDArray c1 = ab[0], c2 = ab[1], ret;
        if (c1.size > c2.size) { var v = c1[$":{c2.size}"]; np.subtract(v, c2, @out: v); ret = c1; }
        else { c2 = -c2; var v = c2[$":{c1.size}"]; np.add(v, c1, @out: v); ret = c2; }
        return Trimseq(ret);
    }

    /// <summary><c>polyadd</c> as a scoped 1-to-1 port.</summary>
    /// <param name="a">First series.</param>
    /// <param name="b">Second series.</param>
    /// <returns>The sum.</returns>
    static NDArray PolyaddP1(NDArray a, NDArray b)
    {
        using var s = NDScope.Open();
        return s.Returns(AddP1(a, b));
    }

    /// <summary><c>chebmul</c> as a scoped 1-to-1 port (z-series through <c>np.convolve</c>).</summary>
    /// <param name="a">First Chebyshev series.</param>
    /// <param name="b">Second Chebyshev series.</param>
    /// <returns>The product series.</returns>
    static NDArray ChebmulP1(NDArray a, NDArray b)
    {
        using var s = NDScope.Open();
        var ab = AsSeries(a, b);
        var z1 = CsToZs(ab[0]); var z2 = CsToZs(ab[1]);
        var prd = np.convolve(z1, z2);
        return s.Returns(Trimseq(ZsToCs(prd)));

        // _cseries_to_zseries: zs[n-1:] = c/2 ; return zs + zs[::-1]
        static NDArray CsToZs(NDArray c)
        {
            long n = c.size;
            var zs = np.zeros(new Shape(2 * n - 1), np.float64);
            zs[$"{n - 1}:"] = c / 2;
            return zs + zs["::-1"];
        }
        // _zseries_to_cseries: c = zs[n-1:].copy(); c[1:n] *= 2
        static NDArray ZsToCs(NDArray zs)
        {
            long n = (zs.size + 1) / 2;
            var cc = zs[$"{n - 1}:"].copy();
            var v = cc[$"1:{n}"];
            np.multiply(v, 2, @out: v);
            return cc;
        }
    }

    /// <summary><c>legmulx</c> ported onto NDArrays.</summary>
    /// <param name="cIn">Legendre series.</param>
    /// <param name="literal">
    ///     True: element reads/writes through NumPy's literal <c>c[i]</c> spelling (0-d NDArray views);
    ///     false: through typed <c>GetDouble</c>/<c>SetDouble</c> accessors.
    /// </param>
    /// <returns>The series multiplied by x.</returns>
    static NDArray LegmulxP1(NDArray cIn, bool literal)
    {
        var c = AsSeries(cIn)[0];
        long n = c.size;
        if (n == 1 && c.GetDouble(0) == 0) return c;
        var prd = np.empty(new Shape(n + 1), np.float64);
        if (literal)
        {
            prd[0] = c[0] * 0;
            prd[1] = c[0];
            for (int i = 1; i < n; i++)
            {
                int j = i + 1, k = i - 1, s = i + j;
                prd[j] = (c[i] * j) / s;
                prd[k] = prd[k] + (c[i] * i) / s;
            }
        }
        else
        {
            double c0 = c.GetDouble(0);
            prd.SetDouble(c0 * 0, 0);
            prd.SetDouble(c0, 1);
            for (long i = 1; i < n; i++)
            {
                long j = i + 1, k = i - 1, s = i + j;
                double ci = c.GetDouble(i);
                prd.SetDouble((ci * j) / s, j);
                prd.SetDouble(prd.GetDouble(k) + (ci * i) / s, k);
            }
        }
        return prd;
    }

    /// <summary>The unscoped body of <c>legmul</c> (backward recurrence over series), shared by legdiv.</summary>
    /// <param name="a">First Legendre series.</param>
    /// <param name="b">Second Legendre series.</param>
    /// <param name="literal">Element-access spelling for the inner <c>legmulx</c> (see <see cref="LegmulxP1"/>).</param>
    /// <returns>The product series.</returns>
    static NDArray LegmulCore(NDArray a, NDArray b, bool literal)
    {
        var ab = AsSeries(a, b);
        NDArray c, xs;
        if (ab[0].size > ab[1].size) { c = ab[1]; xs = ab[0]; } else { c = ab[0]; xs = ab[1]; }
        long len = c.size;
        NDArray c0, c1;
        if (len == 1) { c0 = c.GetDouble(0) * xs; c1 = np.array(new double[] { 0 }); }
        else if (len == 2) { c0 = c.GetDouble(0) * xs; c1 = c.GetDouble(1) * xs; }
        else
        {
            long nd = len;
            c0 = c.GetDouble(len - 2) * xs;
            c1 = c.GetDouble(len - 1) * xs;
            for (long i = 3; i <= len; i++)
            {
                var tmp = c0;
                nd = nd - 1;
                c0 = SubP1(c.GetDouble(len - i) * xs, (c1 * (nd - 1)) / nd);
                c1 = AddP1(tmp, (LegmulxP1(c1, literal) * (2 * nd - 1)) / nd);
            }
        }
        return AddP1(c0, LegmulxP1(c1, literal));
    }

    /// <summary><c>legmul</c> as a scoped 1-to-1 port.</summary>
    /// <param name="a">First Legendre series.</param>
    /// <param name="b">Second Legendre series.</param>
    /// <param name="literal">Element-access spelling (see <see cref="LegmulxP1"/>).</param>
    /// <returns>The product series.</returns>
    static NDArray LegmulP1(NDArray a, NDArray b, bool literal)
    {
        using var s = NDScope.Open();
        return s.Returns(LegmulCore(a, b, literal));
    }

    /// <summary><c>legdiv</c> = <c>pu._div(legmul, …)</c> as a scoped 1-to-1 port (the O(n⁴)-ish generic division).</summary>
    /// <param name="a">Dividend series.</param>
    /// <param name="b">Divisor series (degree ≥ 1 in this probe).</param>
    /// <returns>(quotient, remainder).</returns>
    static (NDArray, NDArray) LegdivP1(NDArray a, NDArray b)
    {
        using var s = NDScope.Open();
        var ab = AsSeries(a, b);
        NDArray c1 = ab[0], c2 = ab[1];
        long lc1 = c1.size, lc2 = c2.size;
        var quo = np.empty(new Shape(lc1 - lc2 + 1), np.float64);
        NDArray rem = c1;
        for (long i = lc1 - lc2; i >= 0; i--)
        {
            // mul_f([0]*i + [1], c2): NumPy builds the basis from a Python int list (int64 -> float64).
            var basis = np.zeros(new Shape(i + 1), np.int64);
            basis.SetInt64(1, i);
            var p = LegmulCore(basis, c2, literal: false);
            double q = rem.GetDouble(rem.size - 1) / p.GetDouble(p.size - 1);
            rem = rem[":-1"] - q * p[":-1"];
            quo.SetDouble(q, i);
        }
        return s.Returns((quo, Trimseq(rem)));
    }

    /// <summary><c>chebder</c> (m=1, scl=1, axis=0) as a scoped 1-to-1 port on a 1-D series.</summary>
    /// <param name="cIn">Chebyshev series.</param>
    /// <returns>The derivative series.</returns>
    static NDArray ChebderP1(NDArray cIn)
    {
        using var s = NDScope.Open();
        var c = cIn.copy();
        c = np.moveaxis(c, 0, 0);
        long n = c.size - 1;
        np.multiply(c, 1, @out: c);                       // c *= scl
        var der = np.empty(new Shape(n), np.float64);
        for (long j = n; j > 2; j--)
        {
            double cj = c.GetDouble(j);
            der.SetDouble((2 * j) * cj, j - 1);
            c.SetDouble(c.GetDouble(j - 2) + (j * cj) / (j - 2), j - 2);
        }
        if (n > 1) der.SetDouble(4 * c.GetDouble(2), 1);
        der.SetDouble(c.GetDouble(1), 0);
        return s.Returns(np.moveaxis(der, 0, 0));
    }

    /// <summary><c>legint</c> (m=1, k=[], lbnd=0, scl=1) as a scoped 1-to-1 port on a 1-D series.</summary>
    /// <param name="cIn">Legendre series.</param>
    /// <returns>The antiderivative series.</returns>
    static NDArray LegintP1(NDArray cIn)
    {
        using var s = NDScope.Open();
        var c = cIn.copy();
        c = np.moveaxis(c, 0, 0);
        long n = c.size;
        np.multiply(c, 1, @out: c);
        var tmp = np.empty(new Shape(n + 1), np.float64);
        tmp.SetDouble(c.GetDouble(0) * 0, 0);
        tmp.SetDouble(c.GetDouble(0), 1);
        if (n > 1) tmp.SetDouble(c.GetDouble(1) / 3, 2);
        for (long j = 2; j < n; j++)
        {
            double t = c.GetDouble(j) / (2 * j + 1);
            tmp.SetDouble(t, j + 1);
            tmp.SetDouble(tmp.GetDouble(j - 1) - t, j - 1);
        }
        // tmp[0] += k[i] - legval(lbnd, tmp) with k = 0, lbnd = 0 (a Python int x -> scalar arithmetic).
        tmp.SetDouble(tmp.GetDouble(0) + (0 - LegvalScalar(0, tmp.Unsafe.ReadOnlySpan<double>())), 0);
        return s.Returns(np.moveaxis(tmp, 0, 0));
    }

    // ---- Span ports: identical op order on double[] ----

    /// <summary><c>as_series</c> for one series: trim trailing zeros and COPY (NumPy's as_series always copies).</summary>
    /// <param name="s">Input coefficients.</param>
    /// <returns>A trimmed copy.</returns>
    static double[] TrimCopy(ReadOnlySpan<double> s)
    {
        int n = s.Length;
        if (n == 0 || s[n - 1] != 0) return s.ToArray();
        int i;
        for (i = n - 1; i >= 0; i--) if (s[i] != 0) break;
        if (i < 0) i = 0;
        return s.Slice(0, i + 1).ToArray();
    }

    /// <summary><c>pu.trimseq</c> on an owned array (returns it, or a shorter copy).</summary>
    /// <param name="a">Coefficients.</param>
    /// <returns>Trimmed coefficients.</returns>
    static double[] TrimSeq(double[] a)
    {
        int n = a.Length;
        if (n == 0 || a[n - 1] != 0) return a;
        int i;
        for (i = n - 1; i >= 0; i--) if (a[i] != 0) break;
        if (i < 0) i = 0;
        return a.AsSpan(0, i + 1).ToArray();
    }

    /// <summary><c>pu._add</c> on spans.</summary>
    /// <param name="a">First series.</param>
    /// <param name="b">Second series.</param>
    /// <returns>The trimmed sum.</returns>
    static double[] AddS(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var c1 = TrimCopy(a); var c2 = TrimCopy(b);
        double[] ret;
        if (c1.Length > c2.Length) { for (int i = 0; i < c2.Length; i++) c1[i] += c2[i]; ret = c1; }
        else { for (int i = 0; i < c1.Length; i++) c2[i] += c1[i]; ret = c2; }
        return TrimSeq(ret);
    }

    /// <summary><c>pu._sub</c> on spans, including NumPy's negate-then-add branch.</summary>
    /// <param name="a">Minuend series.</param>
    /// <param name="b">Subtrahend series.</param>
    /// <returns>The trimmed difference.</returns>
    static double[] SubS(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var c1 = TrimCopy(a); var c2 = TrimCopy(b);
        double[] ret;
        if (c1.Length > c2.Length) { for (int i = 0; i < c2.Length; i++) c1[i] -= c2[i]; ret = c1; }
        else { for (int i = 0; i < c2.Length; i++) c2[i] = -c2[i]; for (int i = 0; i < c1.Length; i++) c2[i] += c1[i]; ret = c2; }
        return TrimSeq(ret);
    }

    /// <summary><c>chebmul</c> on spans (z-series + a sequential convolution).</summary>
    /// <param name="a">First Chebyshev series.</param>
    /// <param name="b">Second Chebyshev series.</param>
    /// <returns>The product series.</returns>
    static double[] ChebmulS(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var c1 = TrimCopy(a); var c2 = TrimCopy(b);
        var z1 = CsToZs(c1); var z2 = CsToZs(c2);
        var prd = ConvolveSeq(z1, z2);
        int n = (prd.Length + 1) / 2;
        var cc = prd.AsSpan(n - 1).ToArray();
        for (int i = 1; i < n; i++) cc[i] *= 2;
        return TrimSeq(cc);

        static double[] CsToZs(double[] c)
        {
            int n = c.Length;
            var zs = new double[2 * n - 1];
            for (int i = 0; i < n; i++) zs[n - 1 + i] = c[i] / 2;
            var r = new double[zs.Length];
            for (int i = 0; i < zs.Length; i++) r[i] = zs[i] + zs[zs.Length - 1 - i];
            return r;
        }
    }

    /// <summary>
    ///     Full convolution with each output a sequential left-to-right sum — NumPy's order below its
    ///     BLAS switch only; long real operands (and every complex one) go through cblas ?dot in NumPy.
    /// </summary>
    /// <param name="a">First operand.</param>
    /// <param name="v">Second operand.</param>
    /// <returns>The full convolution.</returns>
    static double[] ConvolveSeq(double[] a, double[] v)
    {
        if (v.Length > a.Length) (a, v) = (v, a);
        int n = a.Length, m = v.Length;
        var o = new double[n + m - 1];
        for (int k = 0; k < o.Length; k++)
        {
            double sum = 0;
            int lo = Math.Max(0, k - m + 1), hi = Math.Min(k, n - 1);
            for (int i = lo; i <= hi; i++) sum += a[i] * v[k - i];
            o[k] = sum;
        }
        return o;
    }

    /// <summary><c>legmulx</c> on spans.</summary>
    /// <param name="cIn">Legendre series.</param>
    /// <returns>The series multiplied by x.</returns>
    static double[] LegmulxS(ReadOnlySpan<double> cIn)
    {
        var c = TrimCopy(cIn);
        int n = c.Length;
        if (n == 1 && c[0] == 0) return c;
        var prd = new double[n + 1];
        prd[0] = c[0] * 0;
        prd[1] = c[0];
        for (int i = 1; i < n; i++)
        {
            int j = i + 1, k = i - 1, s = i + j;
            prd[j] = (c[i] * j) / s;
            prd[k] += (c[i] * i) / s;
        }
        return prd;
    }

    /// <summary><c>legmul</c> on spans (same recurrence, same trims and copies as NumPy).</summary>
    /// <param name="a">First Legendre series.</param>
    /// <param name="b">Second Legendre series.</param>
    /// <returns>The product series.</returns>
    static double[] LegmulS(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var c1 = TrimCopy(a); var c2 = TrimCopy(b);
        double[] c, xs;
        if (c1.Length > c2.Length) { c = c2; xs = c1; } else { c = c1; xs = c2; }
        int len = c.Length;
        double[] a0, a1;
        if (len == 1) { a0 = Scale(c[0], xs); a1 = new double[] { 0 }; }
        else if (len == 2) { a0 = Scale(c[0], xs); a1 = Scale(c[1], xs); }
        else
        {
            int nd = len;
            a0 = Scale(c[len - 2], xs);
            a1 = Scale(c[len - 1], xs);
            for (int i = 3; i <= len; i++)
            {
                var tmp = a0;
                nd = nd - 1;
                var t = new double[a1.Length];
                // (c1*(nd-1))/nd — the multiply rounds before the divide, as in the NumPy source.
                for (int q = 0; q < t.Length; q++) t[q] = (a1[q] * (nd - 1)) / nd;
                a0 = SubS(Scale(c[len - i], xs), t);
                var mx = LegmulxS(a1);
                for (int q = 0; q < mx.Length; q++) mx[q] = (mx[q] * (2 * nd - 1)) / nd;
                a1 = AddS(tmp, mx);
            }
        }
        return AddS(a0, LegmulxS(a1));

        static double[] Scale(double s, double[] v)
        {
            var r = new double[v.Length];
            for (int q = 0; q < r.Length; q++) r[q] = s * v[q];
            return r;
        }
    }

    /// <summary><c>legdiv</c> = generic <c>_div</c> over <see cref="LegmulS"/> on spans.</summary>
    /// <param name="a">Dividend series.</param>
    /// <param name="b">Divisor series.</param>
    /// <returns>(quotient, remainder).</returns>
    static (double[], double[]) LegdivS(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var c1 = TrimCopy(a); var c2 = TrimCopy(b);
        int lc1 = c1.Length, lc2 = c2.Length;
        var quo = new double[lc1 - lc2 + 1];
        var rem = c1;
        for (int i = lc1 - lc2; i >= 0; i--)
        {
            var basis = new double[i + 1];
            basis[i] = 1;
            var p = LegmulS(basis, c2);
            double q = rem[rem.Length - 1] / p[p.Length - 1];
            var nr = new double[rem.Length - 1];
            for (int k = 0; k < nr.Length; k++) nr[k] = rem[k] - q * p[k];
            rem = nr;
            quo[i] = q;
        }
        return (quo, TrimSeq(rem));
    }

    /// <summary><c>chebder</c> on spans.</summary>
    /// <param name="cIn">Chebyshev series.</param>
    /// <returns>The derivative series.</returns>
    static double[] ChebderS(ReadOnlySpan<double> cIn)
    {
        var c = cIn.ToArray();
        int n = c.Length - 1;
        for (int q = 0; q < c.Length; q++) c[q] *= 1;
        var der = new double[n];
        for (int j = n; j > 2; j--)
        {
            der[j - 1] = (2 * j) * c[j];
            c[j - 2] += (j * c[j]) / (j - 2);
        }
        if (n > 1) der[1] = 4 * c[2];
        der[0] = c[1];
        return der;
    }

    /// <summary><c>legint</c> on spans.</summary>
    /// <param name="cIn">Legendre series.</param>
    /// <returns>The antiderivative series.</returns>
    static double[] LegintS(ReadOnlySpan<double> cIn)
    {
        var c = cIn.ToArray();
        int n = c.Length;
        for (int q = 0; q < n; q++) c[q] *= 1;
        var tmp = new double[n + 1];
        tmp[0] = c[0] * 0;
        tmp[1] = c[0];
        if (n > 1) tmp[2] = c[1] / 3;
        for (int j = 2; j < n; j++)
        {
            double t = c[j] / (2 * j + 1);
            tmp[j + 1] = t;
            tmp[j - 1] -= t;
        }
        tmp[0] += 0 - LegvalScalar(0, tmp);
        return tmp;
    }

    /// <summary><c>legval</c> at a scalar x (the <c>lbnd</c> constant): numpy-scalar arithmetic is double arithmetic.</summary>
    /// <param name="x">Evaluation point.</param>
    /// <param name="c">Legendre series.</param>
    /// <returns>The series value.</returns>
    static double LegvalScalar(double x, ReadOnlySpan<double> c)
    {
        int len = c.Length;
        double c0, c1;
        if (len == 1) { c0 = c[0]; c1 = 0; }
        else if (len == 2) { c0 = c[0]; c1 = c[1]; }
        else
        {
            int nd = len;
            c0 = c[len - 2]; c1 = c[len - 1];
            for (int i = 3; i <= len; i++)
            {
                double tmp = c0;
                nd = nd - 1;
                c0 = c[len - i] - c1 * ((double)(nd - 1) / nd);
                c1 = tmp + c1 * x * ((double)(2 * nd - 1) / nd);
            }
        }
        return c0 + c1 * x;
    }

    // ======================================================================== E: evaluation over x

    /// <summary>
    ///     Series evaluation over 1K/100K/10M points at degree 3/10/30 — U3, the class <c>__call__</c>,
    ///     and the Newton step of U8. 10M is timing-only (NumPy references are written up to 100K).
    /// </summary>
    /// <param name="c">Seeded float64 series keyed by degree.</param>
    static void Evaluation(Dictionary<int, NDArray> c)
    {
        foreach (int n in new[] { 1_000, 100_000, 10_000_000 })
        {
            var x = L($"x{n}");
            foreach (int d in new[] { 3, 10, 30 })
            {
                if (n == 10_000_000 && d != 10) continue;
                var cf = c[d];
                bool hasRef = n != 10_000_000;
                Func<object, string> vp = hasRef ? Vs(L($"ref_polyval_c{d}_x{n}")) : _ => "-";
                Func<object, string> vc = hasRef ? Vs(L($"ref_chebval_c{d}_x{n}")) : _ => "-";
                Func<object, string> vl = hasRef ? Vs(L($"ref_legval_c{d}_x{n}")) : _ => "-";
                // 0-d parameters, not literals: a literal is baked into the IL (one kernel per VALUE).
                var cParams = new NDArray[d + 1];
                for (int k = 0; k <= d; k++) cParams[k] = NDArray.Scalar(cf.GetDouble(k));
                string cellP = $"E polyval d{d} n{n}", cellC = $"E chebval d{d} n{n}", cellL = $"E legval d{d} n{n}";

                Row(cellP, "P1", () => PolyvalP1(x, cf), vp);
                Row(cellP, "P1out", () => PolyvalOut(x, cf), vp);
                Row(cellP, "Expr", () => PolyvalExpr(x, cParams), vp);
                Row(cellP, "Span", () => Eval(x, cf, 0), vp);
                Row(cellC, "P1", () => ChebvalP1(x, cf), vc);
                Row(cellC, "P1out", () => ChebvalOut(x, cf), vc);
                Row(cellC, "Span", () => Eval(x, cf, 1), vc);
                Row(cellL, "P1", () => LegvalP1(x, cf), vl);
                Row(cellL, "P1out", () => LegvalOut(x, cf), vl);
                Row(cellL, "Span", () => Eval(x, cf, 2), vl);
            }
        }

        var xx = L("x100000"); var yy = L("y100000"); var c55 = L("c55");
        var r2 = L("ref_chebval2d_x100000");
        Row("E chebval2d d5x5 n100000", "P1", () => Chebval2dP1(xx, yy, c55), Vs(r2));
        Row("E chebval2d d5x5 n100000", "Span", () => Chebval2dFused(xx, yy, c55), Vs(r2));
    }

    /// <summary>
    ///     <c>polyval</c> as a scoped 1-to-1 port. Every step allocates two full-size temporaries and
    ///     the scope keeps ALL of them alive until it closes (section M measures the cost).
    /// </summary>
    /// <param name="x">Evaluation points (float64).</param>
    /// <param name="c">Power series, low to high.</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray PolyvalP1(NDArray x, NDArray c)
    {
        using var s = NDScope.Open();
        int nc = (int)c.size;
        NDArray c0 = c.GetDouble(nc - 1) + x * 0;
        for (int i = 2; i <= nc; i++) c0 = c.GetDouble(nc - i) + c0 * x;
        return s.Returns(c0);
    }

    /// <summary><c>chebval</c> as a scoped 1-to-1 port (Clenshaw; c0/c1 start as numpy-scalar-like 0-d arrays).</summary>
    /// <param name="x">Evaluation points.</param>
    /// <param name="c">Chebyshev series (≥ 3 coefficients here).</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray ChebvalP1(NDArray x, NDArray c)
    {
        using var s = NDScope.Open();
        int nc = (int)c.size;
        var x2 = 2 * x;
        NDArray c0 = NDArray.Scalar(c.GetDouble(nc - 2)), c1 = NDArray.Scalar(c.GetDouble(nc - 1));
        for (int i = 3; i <= nc; i++)
        {
            var tmp = c0;
            c0 = c.GetDouble(nc - i) - c1;
            c1 = tmp + c1 * x2;
        }
        return s.Returns(c0 + c1 * x);
    }

    /// <summary><c>legval</c> as a scoped 1-to-1 port (Clenshaw with the (nd-1)/nd and (2nd-1)/nd factors).</summary>
    /// <param name="x">Evaluation points.</param>
    /// <param name="c">Legendre series (≥ 3 coefficients here).</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray LegvalP1(NDArray x, NDArray c)
    {
        using var s = NDScope.Open();
        int nc = (int)c.size, nd = nc;
        NDArray c0 = NDArray.Scalar(c.GetDouble(nc - 2)), c1 = NDArray.Scalar(c.GetDouble(nc - 1));
        for (int i = 3; i <= nc; i++)
        {
            var tmp = c0;
            nd = nd - 1;
            c0 = c.GetDouble(nc - i) - c1 * ((double)(nd - 1) / nd);
            c1 = tmp + c1 * x * ((double)(2 * nd - 1) / nd);
        }
        return s.Returns(c0 + c1 * x);
    }

    /// <summary>
    ///     <c>polyval</c> with buffer reuse through out= (legacy <c>np.polyval</c>'s technique): no
    ///     per-step allocation, but still two full memory passes per coefficient.
    /// </summary>
    /// <param name="x">Evaluation points (any layout — every op is a real ufunc).</param>
    /// <param name="c">Power series.</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray PolyvalOut(NDArray x, NDArray c)
    {
        int nc = (int)c.size;
        var y = np.multiply(x, 0);
        np.add(y, c.GetDouble(nc - 1), @out: y);
        for (int i = 2; i <= nc; i++)
        {
            np.multiply(y, x, @out: y);
            np.add(y, c.GetDouble(nc - i), @out: y);
        }
        return y;
    }

    /// <summary><c>chebval</c> with three rotating buffers: three full memory passes per coefficient.</summary>
    /// <param name="x">Evaluation points (any layout).</param>
    /// <param name="c">Chebyshev series (≥ 3 coefficients).</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray ChebvalOut(NDArray x, NDArray c)
    {
        int nc = (int)c.size;
        var x2 = np.multiply(x, 2);
        var shape = new Shape(x.shape);                      // dimensions only — see Fresh()
        var A = np.full(shape, c.GetDouble(nc - 2), np.float64);   // c0
        var B = np.full(shape, c.GetDouble(nc - 1), np.float64);   // c1
        var T = np.empty(shape, np.float64);
        for (int i = 3; i <= nc; i++)
        {
            // c0' = c[-i] - c1 -> T ; c1' = c0 + c1*x2 -> B ; the old c0 buffer becomes the spare.
            np.subtract(c.GetDouble(nc - i), B, @out: T);
            np.multiply(B, x2, @out: B);
            np.add(A, B, @out: B);
            (A, T) = (T, A);
        }
        np.multiply(B, x, @out: B);
        np.add(A, B, @out: A);
        B.Dispose(); T.Dispose(); x2.Dispose();
        return A;
    }

    /// <summary><c>legval</c> with rotating buffers: five full memory passes per coefficient.</summary>
    /// <param name="x">Evaluation points (any layout).</param>
    /// <param name="c">Legendre series (≥ 3 coefficients).</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray LegvalOut(NDArray x, NDArray c)
    {
        int nc = (int)c.size, nd = nc;
        var shape = new Shape(x.shape);
        var A = np.full(shape, c.GetDouble(nc - 2), np.float64);
        var B = np.full(shape, c.GetDouble(nc - 1), np.float64);
        var T = np.empty(shape, np.float64);
        for (int i = 3; i <= nc; i++)
        {
            nd = nd - 1;
            np.multiply(B, (double)(nd - 1) / nd, @out: T);
            np.subtract(c.GetDouble(nc - i), T, @out: T);
            np.multiply(B, x, @out: B);
            np.multiply(B, (double)(2 * nd - 1) / nd, @out: B);
            np.add(A, B, @out: B);
            (A, T) = (T, A);
        }
        np.multiply(B, x, @out: B);
        np.add(A, B, @out: A);
        B.Dispose(); T.Dispose();
        return A;
    }

    /// <summary>
    ///     Horner as ONE NDExpr tree evaluated by <c>np.evaluate</c> — the only basis NDExpr can express:
    ///     a Clenshaw step uses c1 twice and NDExpr re-emits a shared subtree per use, so those trees
    ///     grow ~1.6× per degree (1,033 nodes at degree 10, 15.8 M at degree 30).
    /// </summary>
    /// <param name="x">Evaluation points.</param>
    /// <param name="cParams">Coefficients as 0-d arrays (hoisted parameters: one kernel per degree).</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray PolyvalExpr(NDArray x, NDArray[] cParams)
    {
        int nc = cParams.Length;
        NDExpr X = x;
        NDExpr acc = (NDExpr)cParams[nc - 1] + X * 0;
        for (int k = nc - 2; k >= 0; k--) acc = (NDExpr)cParams[k] + acc * X;
        return np.evaluate(acc);
    }

    /// <summary>Runs one fused evaluation kernel into a fresh buffer.</summary>
    /// <param name="x">Contiguous float64 points.</param>
    /// <param name="c">Contiguous float64 series.</param>
    /// <param name="basis">0 power (Horner), 1 Chebyshev, 2 Legendre.</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray Eval(NDArray x, NDArray c, int basis)
    {
        var y = Fresh(x);
        double* xp = x.Unsafe.Pointer<double>(), yp = y.Unsafe.Pointer<double>(), cp = c.Unsafe.Pointer<double>();
        int nc = (int)c.size;
        if (basis == 0) PolyvalKernel(xp, cp, nc, yp, x.size);
        else if (basis == 1) ChebvalKernel(xp, cp, nc, yp, x.size);
        else LegvalKernel(xp, cp, nc, yp, x.size);
        return y;
    }

    /// <summary>
    ///     Horner in NumPy's exact order (<c>c0 = c[-1] + x*0; c0 = c[-i] + c0*x</c>). Four independent
    ///     vectors per iteration: the per-element chain is serial, so a single chain per loop body
    ///     would be latency-bound.
    /// </summary>
    /// <param name="x">Points.</param>
    /// <param name="c">Series.</param>
    /// <param name="nc">Coefficient count.</param>
    /// <param name="y">Output.</param>
    /// <param name="n">Point count.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void PolyvalKernel(double* x, double* c, int nc, double* y, long n)
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var vl = Vector256.Create(c[nc - 1]);
            var z = Vector256<double>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var x0 = Vector256.Load(x + i); var x1 = Vector256.Load(x + i + 4);
                var x2 = Vector256.Load(x + i + 8); var x3 = Vector256.Load(x + i + 12);
                // x*0 is kept literally: it is -0.0 for negative x and NaN for inf/NaN, as in NumPy.
                var a0 = vl + x0 * z; var a1 = vl + x1 * z; var a2 = vl + x2 * z; var a3 = vl + x3 * z;
                for (int k = nc - 2; k >= 0; k--)
                {
                    var ck = Vector256.Create(c[k]);
                    a0 = ck + a0 * x0; a1 = ck + a1 * x1; a2 = ck + a2 * x2; a3 = ck + a3 * x3;
                }
                a0.Store(y + i); a1.Store(y + i + 4); a2.Store(y + i + 8); a3.Store(y + i + 12);
            }
            for (; i + 4 <= n; i += 4)
            {
                var x0 = Vector256.Load(x + i);
                var a0 = vl + x0 * z;
                for (int k = nc - 2; k >= 0; k--) a0 = Vector256.Create(c[k]) + a0 * x0;
                a0.Store(y + i);
            }
        }
        for (; i < n; i++)
        {
            double xv = x[i], a = c[nc - 1] + xv * 0.0;
            for (int k = nc - 2; k >= 0; k--) a = c[k] + a * xv;
            y[i] = a;
        }
    }

    /// <summary>
    ///     <c>chebval</c> (≥ 3 coefficients) in NumPy's exact order: <c>x2 = 2*x; c0 = c[-2]; c1 = c[-1];
    ///     tmp = c0; c0 = c[-i] - c1; c1 = tmp + c1*x2; return c0 + c1*x</c>. Four chains per iteration.
    /// </summary>
    /// <param name="x">Points.</param>
    /// <param name="c">Series.</param>
    /// <param name="nc">Coefficient count (≥ 3).</param>
    /// <param name="y">Output.</param>
    /// <param name="n">Point count.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void ChebvalKernel(double* x, double* c, int nc, double* y, long n)
    {
        long i = 0;
        var two = Vector256.Create(2.0);
        if (Vector256.IsHardwareAccelerated)
        {
            var i0 = Vector256.Create(c[nc - 2]); var i1 = Vector256.Create(c[nc - 1]);
            for (; i + 16 <= n; i += 16)
            {
                var xa = Vector256.Load(x + i); var xb = Vector256.Load(x + i + 4);
                var xc = Vector256.Load(x + i + 8); var xd = Vector256.Load(x + i + 12);
                var ta = two * xa; var tb = two * xb; var tc = two * xc; var td = two * xd;
                Vector256<double> a0 = i0, b0 = i0, c0v = i0, d0 = i0, a1 = i1, b1 = i1, c1v = i1, d1 = i1;
                for (int k = nc - 3; k >= 0; k--)
                {
                    var ck = Vector256.Create(c[k]);
                    var s0 = a0; a0 = ck - a1; a1 = s0 + a1 * ta;
                    var s1 = b0; b0 = ck - b1; b1 = s1 + b1 * tb;
                    var s2 = c0v; c0v = ck - c1v; c1v = s2 + c1v * tc;
                    var s3 = d0; d0 = ck - d1; d1 = s3 + d1 * td;
                }
                (a0 + a1 * xa).Store(y + i); (b0 + b1 * xb).Store(y + i + 4);
                (c0v + c1v * xc).Store(y + i + 8); (d0 + d1 * xd).Store(y + i + 12);
            }
        }
        for (; i < n; i++)
        {
            double xv = x[i], x2 = 2 * xv, c0 = c[nc - 2], c1 = c[nc - 1];
            for (int k = nc - 3; k >= 0; k--) { double t = c0; c0 = c[k] - c1; c1 = t + c1 * x2; }
            y[i] = c0 + c1 * xv;
        }
    }

    /// <summary>
    ///     <c>legval</c> (≥ 3 coefficients) in NumPy's exact order: <c>c0 = c[-i] - c1*((nd-1)/nd);
    ///     c1 = tmp + c1*x*((2nd-1)/nd)</c>. The two factors are Python true divisions of ints, i.e.
    ///     correctly rounded doubles — precomputed once, bit-identical to NumPy's per-call values.
    /// </summary>
    /// <param name="x">Points.</param>
    /// <param name="c">Series.</param>
    /// <param name="nc">Coefficient count (≥ 3).</param>
    /// <param name="y">Output.</param>
    /// <param name="n">Point count.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void LegvalKernel(double* x, double* c, int nc, double* y, long n)
    {
        var k1 = stackalloc double[nc]; var k2 = stackalloc double[nc];
        for (int nd = 2; nd < nc; nd++) { k1[nd] = (double)(nd - 1) / nd; k2[nd] = (double)(2 * nd - 1) / nd; }
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var i0 = Vector256.Create(c[nc - 2]); var i1 = Vector256.Create(c[nc - 1]);
            for (; i + 16 <= n; i += 16)
            {
                var xa = Vector256.Load(x + i); var xb = Vector256.Load(x + i + 4);
                var xc = Vector256.Load(x + i + 8); var xd = Vector256.Load(x + i + 12);
                Vector256<double> a0 = i0, b0 = i0, c0v = i0, d0 = i0, a1 = i1, b1 = i1, c1v = i1, d1 = i1;
                int nd = nc;
                for (int k = nc - 3; k >= 0; k--)
                {
                    nd--;
                    var ck = Vector256.Create(c[k]); var q1 = Vector256.Create(k1[nd]); var q2 = Vector256.Create(k2[nd]);
                    // (c1*x)*q2 — left to right, as NumPy's `c1 * x * ((2*nd-1)/nd)` associates.
                    var s0 = a0; a0 = ck - a1 * q1; a1 = s0 + a1 * xa * q2;
                    var s1 = b0; b0 = ck - b1 * q1; b1 = s1 + b1 * xb * q2;
                    var s2 = c0v; c0v = ck - c1v * q1; c1v = s2 + c1v * xc * q2;
                    var s3 = d0; d0 = ck - d1 * q1; d1 = s3 + d1 * xd * q2;
                }
                (a0 + a1 * xa).Store(y + i); (b0 + b1 * xb).Store(y + i + 4);
                (c0v + c1v * xc).Store(y + i + 8); (d0 + d1 * xd).Store(y + i + 12);
            }
        }
        for (; i < n; i++)
        {
            double xv = x[i], c0 = c[nc - 2], c1 = c[nc - 1];
            int nd = nc;
            for (int k = nc - 3; k >= 0; k--) { nd--; double t = c0; c0 = c[k] - c1 * k1[nd]; c1 = t + c1 * xv * k2[nd]; }
            y[i] = c0 + c1 * xv;
        }
    }

    /// <summary>
    ///     <c>chebval2d</c> the way NumPy spells <c>_valnd(chebval, c, x, y)</c>: a tensor pass
    ///     (c (m0,m1) reshaped to (m0,m1,1) -> (m1, N)) then a per-point pass (tensor=False).
    /// </summary>
    /// <param name="x">First coordinates.</param>
    /// <param name="y">Second coordinates (same shape).</param>
    /// <param name="c">2-D coefficient array (≥ 3 × ≥ 3).</param>
    /// <returns>Values at the points.</returns>
    static NDArray Chebval2dP1(NDArray x, NDArray y, NDArray c)
    {
        using var s = NDScope.Open();
        var cr = c.reshape(c.shape[0], c.shape[1], 1);             // c.reshape(c.shape + (1,)*x.ndim)
        var r = ChebvalTensor(x, cr);                               // (m1, N)
        return s.Returns(ChebvalTensor(y, r));                      // tensor=False: plain broadcasting

        static NDArray ChebvalTensor(NDArray xv, NDArray cc)
        {
            long nc = cc.shape[0];
            var x2 = 2 * xv;
            NDArray c0 = cc[$"{nc - 2}"], c1 = cc[$"{nc - 1}"];
            for (long i = 3; i <= nc; i++)
            {
                var tmp = c0;
                c0 = cc[$"{nc - i}"] - c1;
                c1 = tmp + c1 * x2;
            }
            return c0 + c1 * xv;
        }
    }

    /// <summary>
    ///     <c>chebval2d</c> in ONE pass: per point, m1 Clenshaws down axis 0, then one Clenshaw over
    ///     those m1 values at y — the same per-element op sequence as NumPy's two passes.
    /// </summary>
    /// <param name="x">First coordinates (contiguous).</param>
    /// <param name="yv">Second coordinates (contiguous).</param>
    /// <param name="c">2-D coefficient array (contiguous, ≥ 3 × ≥ 3).</param>
    /// <returns>Values at the points.</returns>
    static NDArray Chebval2dFused(NDArray x, NDArray yv, NDArray c)
    {
        int m0 = (int)c.shape[0], m1 = (int)c.shape[1];
        var res = Fresh(x);
        Chebval2dKernel(x.Unsafe.Pointer<double>(), yv.Unsafe.Pointer<double>(), c.Unsafe.Pointer<double>(), m0, m1, res.Unsafe.Pointer<double>(), x.size);
        return res;
    }

    /// <summary>The fused 2-D Clenshaw (one vector of points per iteration; the per-point stage keeps m1 vectors in registers/stack).</summary>
    /// <param name="x">First coordinates.</param>
    /// <param name="yv">Second coordinates.</param>
    /// <param name="c">Row-major (m0, m1) coefficients.</param>
    /// <param name="m0">Degree+1 along x.</param>
    /// <param name="m1">Degree+1 along y.</param>
    /// <param name="o">Output.</param>
    /// <param name="n">Point count.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void Chebval2dKernel(double* x, double* yv, double* c, int m0, int m1, double* o, long n)
    {
        var r = stackalloc Vector256<double>[m1];
        var rs = stackalloc double[m1];
        var two = Vector256.Create(2.0);
        long i = 0;
        for (; i + 4 <= n; i += 4)
        {
            var xv = Vector256.Load(x + i); var x2 = two * xv;
            for (int k = 0; k < m1; k++)
            {
                var c0 = Vector256.Create(c[(m0 - 2) * m1 + k]); var c1 = Vector256.Create(c[(m0 - 1) * m1 + k]);
                for (int q = m0 - 3; q >= 0; q--) { var t = c0; c0 = Vector256.Create(c[q * m1 + k]) - c1; c1 = t + c1 * x2; }
                r[k] = c0 + c1 * xv;
            }
            var yy = Vector256.Load(yv + i); var y2 = two * yy;
            var d0 = r[m1 - 2]; var d1 = r[m1 - 1];
            for (int q = m1 - 3; q >= 0; q--) { var t = d0; d0 = r[q] - d1; d1 = t + d1 * y2; }
            (d0 + d1 * yy).Store(o + i);
        }
        for (; i < n; i++)
        {
            double xs = x[i], xs2 = 2 * xs;
            for (int k = 0; k < m1; k++)
            {
                double c0 = c[(m0 - 2) * m1 + k], c1 = c[(m0 - 1) * m1 + k];
                for (int q = m0 - 3; q >= 0; q--) { double t = c0; c0 = c[q * m1 + k] - c1; c1 = t + c1 * xs2; }
                rs[k] = c0 + c1 * xs;
            }
            double ys = yv[i], ys2 = 2 * ys, e0 = rs[m1 - 2], e1 = rs[m1 - 1];
            for (int q = m1 - 3; q >= 0; q--) { double t = e0; e0 = rs[q] - e1; e1 = t + e1 * ys2; }
            o[i] = e0 + e1 * ys;
        }
    }

    // ======================================================================== V: Vandermonde

    /// <summary><c>chebvander(x, 10)</c> at 100K points — write-bound (8.8 MB output).</summary>
    static void Vander()
    {
        var x = L("x100000");
        var rv = L("ref_chebvander_x100000_d10");
        Row("V chebvander d10 n100000", "P1", () => ChebvanderP1(x, 10), Vs(rv));
        Row("V chebvander d10 n100000", "P1out", () => ChebvanderOut(x, 10), Vs(rv));
        Row("V chebvander d10 n100000", "Span", () => ChebvanderFused(x, 10), Vs(rv));
    }

    /// <summary><c>chebvander</c> as a scoped 1-to-1 port (each row a temporary, copied into the row).</summary>
    /// <param name="xIn">Points.</param>
    /// <param name="deg">Degree.</param>
    /// <returns>NumPy's moveaxis view of the (deg+1, N) buffer.</returns>
    static NDArray ChebvanderP1(NDArray xIn, int deg)
    {
        using var s = NDScope.Open();
        var x = xIn + 0.0;
        var v = np.empty(new Shape(deg + 1, x.size), np.float64);
        v["0"] = x * 0 + 1;
        v["1"] = x;
        var x2 = 2 * x;
        for (int i = 2; i <= deg; i++) v[$"{i}"] = v[$"{i - 1}"] * x2 - v[$"{i - 2}"];
        return s.Returns(np.moveaxis(v, 0, -1));
    }

    /// <summary><c>chebvander</c> writing each row in place through out= (two passes per row, no temporaries).</summary>
    /// <param name="xIn">Points.</param>
    /// <param name="deg">Degree.</param>
    /// <returns>NumPy's moveaxis view of the (deg+1, N) buffer.</returns>
    static NDArray ChebvanderOut(NDArray xIn, int deg)
    {
        using var s = NDScope.Open();
        var x = xIn + 0.0;
        var v = np.empty(new Shape(deg + 1, x.size), np.float64);
        var r0 = v["0"]; np.multiply(x, 0, @out: r0); np.add(r0, 1, @out: r0);
        np.copyto(v["1"], x);
        var x2 = 2 * x;
        for (int i = 2; i <= deg; i++)
        {
            var ri = v[$"{i}"];
            np.multiply(v[$"{i - 1}"], x2, @out: ri);
            np.subtract(ri, v[$"{i - 2}"], @out: ri);
        }
        return s.Returns(np.moveaxis(v, 0, -1));
    }

    /// <summary><c>chebvander</c> as one fused fill of the (deg+1, N) buffer NumPy moveaxis-views.</summary>
    /// <param name="x">Contiguous float64 points.</param>
    /// <param name="deg">Degree.</param>
    /// <returns>NumPy's moveaxis view of the (deg+1, N) buffer.</returns>
    static NDArray ChebvanderFused(NDArray x, int deg)
    {
        long n = x.size;
        var v = new NDArray(np.float64, new Shape(deg + 1, n), false);
        ChebvanderKernel(x.Unsafe.Pointer<double>(), deg, v.Unsafe.Pointer<double>(), n);
        return np.moveaxis(v, 0, -1);
    }

    /// <summary>
    ///     Row-major fill: <c>v[0] = x*0+1; v[1] = x; x2 = 2*x; v[i] = v[i-1]*x2 - v[i-2]</c> — one read
    ///     of x and one write per output element (deg+1 contiguous write streams).
    /// </summary>
    /// <param name="x">Points.</param>
    /// <param name="deg">Degree.</param>
    /// <param name="v">(deg+1) × n output rows.</param>
    /// <param name="n">Point count.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void ChebvanderKernel(double* x, int deg, double* v, long n)
    {
        long i = 0;
        var one = Vector256.Create(1.0); var two = Vector256.Create(2.0); var z = Vector256<double>.Zero;
        for (; i + 4 <= n; i += 4)
        {
            var xv = Vector256.Load(x + i);
            var t0 = xv * z + one;
            t0.Store(v + i);
            if (deg == 0) continue;
            var t1 = xv;
            t1.Store(v + n + i);
            var x2 = two * xv;
            for (int k = 2; k <= deg; k++)
            {
                var t2 = t1 * x2 - t0;
                t2.Store(v + k * n + i);
                t0 = t1; t1 = t2;
            }
        }
        for (; i < n; i++)
        {
            double xv = x[i], t0 = xv * 0 + 1;
            v[i] = t0;
            if (deg == 0) continue;
            double t1 = xv;
            v[n + i] = t1;
            double x2 = 2 * xv;
            for (int k = 2; k <= deg; k++) { double t2 = t1 * x2 - t0; v[k * n + i] = t2; t0 = t1; t1 = t2; }
        }
    }

    // ======================================================================== L: strided x

    /// <summary>
    ///     Does the contiguous-only kernel need NDIter? Cost of normalising a strided x to a contiguous
    ///     buffer first vs the out= composition that walks any layout natively.
    /// </summary>
    /// <param name="c">Seeded float64 series keyed by degree.</param>
    static void Layout(Dictionary<int, NDArray> c)
    {
        var s50k = L("x100000")["::2"];
        var s5m = L("x10000000")["::2"];
        var rs = L("ref_chebval_c10_strided50k");
        Row("L chebval d10 strided 50K", "P1out", () => ChebvalOut(s50k, c[10]), Vs(rs));
        Row("L chebval d10 strided 50K", "Span", () => MaterializeThenEval(s50k, c[10]), Vs(rs));
        Row("L chebval d10 strided 5M", "P1out", () => ChebvalOut(s5m, c[10]), _ => "-");
        Row("L chebval d10 strided 5M", "Span", () => MaterializeThenEval(s5m, c[10]), _ => "-");
        Row("L materialize strided 5M", "copy", () => np.ascontiguousarray(s5m), _ => "-");
    }

    /// <summary>The "no NDIter" strategy for a non-contiguous x: one contiguous copy, then the fused kernel.</summary>
    /// <param name="x">Points in any layout.</param>
    /// <param name="c">Chebyshev series.</param>
    /// <returns>Values at <paramref name="x"/>.</returns>
    static NDArray MaterializeThenEval(NDArray x, NDArray c)
    {
        var xc = np.ascontiguousarray(x);
        var y = Eval(xc, c, 1);
        xc.Dispose();
        return y;
    }

    // ======================================================================== M: memory

    /// <summary>
    ///     Working set held by the scoped 1-to-1 <c>polyval</c> at 10M points / degree 10, sampled right
    ///     before its scope would close, vs the out= port (NumPy frees each step's temporary at once).
    /// </summary>
    /// <param name="c">Seeded float64 series keyed by degree.</param>
    static void Memory(Dictionary<int, NDArray> c)
    {
        var x = L("x10000000");
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long before = Environment.WorkingSet, peakScoped;
        using (var s = NDScope.Open())
        {
            int nc = (int)c[10].size;
            NDArray c0 = c[10].GetDouble(nc - 1) + x * 0;
            for (int i = 2; i <= nc; i++) c0 = c[10].GetDouble(nc - i) + c0 * x;
            peakScoped = Environment.WorkingSet - before;
        }
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        before = Environment.WorkingSet;
        var y = PolyvalOut(x, c[10]);
        long peakOut = Environment.WorkingSet - before;
        y.Dispose();
        Console.WriteLine($"M polyval d10 n10M\tscoped-1to1 {peakScoped / 1e6:F0} MB\tout= {peakOut / 1e6:F0} MB\t(x = 80 MB)");
    }
}
