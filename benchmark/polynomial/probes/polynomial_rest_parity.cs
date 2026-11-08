#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Nullable=disable
// =============================================================================
// polynomial_rest_parity.cs — is each candidate route NumPy-exact? (twin: polynomial_rest_parity_numpy.py)
//
// Prints `label  route  dtype[shape] sha256-prefix` for every candidate route of the numpy.polynomial names NumSharp
// still lacks, on inputs both sides build bit-identically. Compare line by line with the NumPy listing: equal hashes
// mean byte-identical results (NaN payloads and signed zeros included). OpenBLAS runs at one thread, the parity
// configuration of the host-pinned oracle tiers. Verdicts: docs/plans/numpy-polynomial-remaining-compute.md §3.
// =============================================================================

using System;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Interop.OpenBLAS;

OpenBlasEngine.Enable(threads: 1);
var C = np.polynomial.chebyshev;
var LG = np.polynomial.legendre;
var H = np.polynomial.hermite;
NDArray one = 1.0, zero = 0.0, two = 2, mhalf = -0.5;

foreach (var (dname, dt) in new[] { ("f64", np.float64), ("f32", np.float32), ("f16", np.float16) })
{
    using var x0 = Parity.Data(100_000, 0.95);
    using var x = x0.astype(dt);
    NDExpr X = x;
    Parity.Emit($"chebweight {dname}", "eager", 1.0 / (np.sqrt(one + x) * np.sqrt(one - x)));
    Parity.Emit($"chebweight {dname}", "fused", np.evaluate(1.0 / (NDExpr.Sqrt(1.0 + X) * NDExpr.Sqrt(1.0 - X))));
    Parity.Emit($"legweight {dname}", "eager", x * zero + one);
    Parity.Emit($"legweight {dname}", "fused", np.evaluate(X * 0.0 + 1.0));
    Parity.Emit($"lagweight {dname}", "eager", np.exp(np.negative(x)));
    Parity.Emit($"lagweight {dname}", "fused", np.evaluate(NDExpr.Exp(-X)));
    Parity.Emit($"hermweight {dname}", "eager", np.exp(np.negative(np.power(x, two))));
    Parity.Emit($"hermweight {dname}", "fusedPow", np.evaluate(NDExpr.Exp(-NDExpr.Power(X, 2))));
    Parity.Emit($"hermweight {dname}", "fusedSq", np.evaluate(NDExpr.Exp(-NDExpr.Square(X))));
    Parity.Emit($"hermeweight {dname}", "eager", np.exp(np.multiply(mhalf, np.power(x, two))));
    Parity.Emit($"hermeweight {dname}", "fusedPow", np.evaluate(NDExpr.Exp(-0.5 * NDExpr.Power(X, 2))));
    Parity.Emit($"hermeweight {dname}", "fusedSq", np.evaluate(NDExpr.Exp(-0.5 * NDExpr.Square(X))));
}
foreach (long n in new long[] { 10, 1000 })
{
    NDArray c = 0.5 * Math.PI / n, pi = Math.PI, d2 = 2.0 * n;
    double cd = 0.5 * Math.PI / n;
    using var k1 = np.arange(-n + 1, n + 1, 2, np.int64);
    using var k2 = np.arange(1, 2 * n, 2, np.int64);
    Parity.Emit($"chebpts1 {n}", "eager", np.sin(np.multiply(c, k1)));
    Parity.Emit($"chebpts1 {n}", "arange+fused", np.evaluate(NDExpr.Sin(cd * (NDExpr)k1)));
    Parity.Emit($"chebpts2 {n}", "eager", np.cos(np.linspace(-Math.PI, 0.0, n)));
    Parity.Emit($"chebgauss.x {n}", "eager", np.cos(np.divide(np.multiply(pi, k2), d2)));
    Parity.Emit($"chebgauss.x {n}", "arange+fused", np.evaluate(NDExpr.Cos((Math.PI * (NDExpr)k2) / (2.0 * n))));
    Parity.Emit($"chebgauss.w {n}", "ones*c", np.multiply(np.ones(new Shape(n)), (NDArray)(Math.PI / n)));
    Parity.Emit($"chebgauss.w {n}", "full", np.full(new Shape(n), Math.PI / n));
}
{
    using var x = Parity.Data(1000, 1.0);
    using var r0 = Parity.Data(10, 0.9);
    using var r = r0.reshape(10, 1);
    Parity.Emit("polyvalfromroots f64", "compose", np.prod(np.subtract(x, r), axis: 0));
    Parity.Emit("polyvalfromroots f64", "evalProd", np.evaluate(NDExpr.Prod((NDExpr)x - (NDExpr)r, axis: 0)));
    // The NumPy side builds x + v*1j; complex(x, v) is the same value bit for bit (see the twin's comment).
    var xv = x.ToArray<double>();
    var vv = Parity.Data(1000, 0.7).ToArray<double>().Reverse().ToArray();
    var rv = r0.ToArray<double>();
    var iv = Parity.Data(10, 0.5).ToArray<double>();
    using var xc = np.array(xv.Select((a, i) => new Complex(a, vv[i])).ToArray());
    using var rc0 = np.array(rv.Select((a, i) => new Complex(a, iv[i])).ToArray());
    using var rc = rc0.reshape(10, 1);
    Parity.Emit("polyvalfromroots c128", "compose", np.prod(np.subtract(xc, rc), axis: 0));
    Parity.Emit("polyvalfromroots c128", "evalProd", np.evaluate(NDExpr.Prod((NDExpr)xc - (NDExpr)rc, axis: 0)));
    NDArray xs = new Complex(xv[3], vv[3]);
    Parity.Emit("polyvalfromroots c128 scalar", "compose", np.prod(np.subtract(xs, rc0), axis: 0));
    using var y = np.sin(np.multiply(3.0, x));
    using var van = C.chebvander(x, 10);
    var lhs = van.T;
    Parity.Emit("chebfit norms", "eager", np.sqrt(np.sum(np.square(lhs), axis: 1)));
    Parity.Emit("chebfit norms", "fusedSum", np.sqrt(np.evaluate(NDExpr.Sum(NDExpr.Square(lhs), axis: 1))));
    Parity.Emit("chebfit", "compose", Parity.ChebFit(x, y, 10));
}
foreach (int d in new[] { 5, 20 })
{
    var (lx, lw) = Parity.LegGauss(d);
    Parity.Emit($"leggauss.x {d}", "compose", lx);
    Parity.Emit($"leggauss.w {d}", "compose", lw);
    var (hx, hw) = Parity.HermGauss(d);
    Parity.Emit($"hermgauss.x {d}", "compose", hx);
    Parity.Emit($"hermgauss.w {d}", "compose", hw);
}
foreach (int d in new[] { 5, 20 })
    Parity.Emit($"chebinterpolate {d}", "compose", Parity.ChebInterpolate(d));

/// <summary>The NumPy statement sequences under test, and the hash printer. Results are process-lifetime objects here
///     (the script prints and exits), so intermediates are not disposed: only values matter.</summary>
static class Parity
{
    private static readonly NDArray One = 1.0, Zero = 0.0, Two = 2;

    /// <summary>The shared input <c>sin(k * 1.2345678901) * amp</c>, bit-identical to the NumPy twin's.</summary>
    /// <param name="n">Element count.</param><param name="amp">Amplitude.</param>
    /// <returns>A fresh float64 array.</returns>
    public static NDArray Data(long n, double amp)
        => np.multiply(np.sin(np.multiply(np.arange(0, n, 1, np.int64), 1.2345678901)), amp);

    /// <summary>Prints <c>label route dtype[shape] hash</c> in the NumPy twin's format.</summary>
    /// <param name="label">The NumPy listing's label.</param><param name="route">The NumSharp route's name.</param>
    /// <param name="a">The route's result.</param>
    public static void Emit(string label, string route, NDArray a)
        => Console.WriteLine($"{label,-30} {route,-14} {a.dtype.name}[{string.Join(", ", a.shape)}] "
                             + Convert.ToHexString(SHA256.HashData(a.tobytes()))[..16].ToLowerInvariant());

    /// <summary>NumPy's <c>pu._fit</c> for the Chebyshev basis, statement for statement.</summary>
    /// <param name="x">Points.</param><param name="y">Values.</param><param name="deg">Degree.</param>
    /// <returns>The coefficients.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed.</exception>
    public static NDArray ChebFit(NDArray x, NDArray y, int deg)
    {
        var xx = x + Zero;
        var yy = y + Zero;
        var lhs = np.polynomial.chebyshev.chebvander(xx, deg).T;
        var scl = np.sqrt(np.sum(np.square(lhs), axis: 1));
        var t = np.linalg.lstsq(lhs.T / scl, yy.T.T, (double)xx.size * 2.220446049250313e-16);
        return (t.Solution.T / scl).T.copy();
    }

    /// <summary>NumPy's <c>leggauss(deg)</c>, statement for statement.</summary>
    /// <param name="deg">Positive degree.</param>
    /// <returns>Nodes and weights.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed.</exception>
    public static (NDArray x, NDArray w) LegGauss(int deg)
    {
        var L = np.polynomial.legendre;
        var cl = new long[deg + 1]; cl[deg] = 1;
        var c = np.array(cl);
        var x = np.linalg.eigvalsh(L.legcompanion(c));
        var dy = L.legval(x, c);
        var df = L.legval(x, L.legder(c));
        np.subtract(x, dy / df, @out: x);
        var fm = L.legval(x, c["1:"]);
        np.divide(fm, np.max(np.abs(fm)), @out: fm);
        np.divide(df, np.max(np.abs(df)), @out: df);
        var w = One / (fm * df);
        w = (w + w["::-1"]) / Two;
        x = (x - x["::-1"]) / Two;
        np.multiply(w, 2.0 / np.sum(w), @out: w);
        return (x, w);
    }

    /// <summary>NumPy's <c>_normed_hermite_n</c> as its array statements.</summary>
    /// <param name="x">Points.</param><param name="n">Degree.</param>
    /// <returns>The normalised Hermite polynomial at <paramref name="x"/>.</returns>
    public static NDArray NormedHermiteN(NDArray x, int n)
    {
        if (n == 0) return np.full(x.Shape, 1.0 / Math.Sqrt(Math.Sqrt(Math.PI)));
        NDArray c0 = NDArray.Scalar(0.0), c1 = NDArray.Scalar(1.0 / Math.Sqrt(Math.Sqrt(Math.PI)));
        double nd = n;
        for (int i = 0; i < n - 1; i++)
        {
            var tmp = c0;
            c0 = -c1 * (NDArray)Math.Sqrt((nd - 1.0) / nd);
            c1 = tmp + c1 * x * (NDArray)Math.Sqrt(2.0 / nd);
            nd -= 1.0;
        }
        return c0 + c1 * x * (NDArray)Math.Sqrt(2.0);
    }

    /// <summary>NumPy's <c>hermgauss(deg)</c>, statement for statement.</summary>
    /// <param name="deg">Positive degree.</param>
    /// <returns>Nodes and weights.</returns>
    /// <exception cref="OpenBlasMissingBackendException">No LAPACK backend is installed.</exception>
    public static (NDArray x, NDArray w) HermGauss(int deg)
    {
        var cl = new double[deg + 1]; cl[deg] = 1;
        var x = np.linalg.eigvalsh(np.polynomial.hermite.hermcompanion(np.array(cl)));
        var dy = NormedHermiteN(x, deg);
        var df = NormedHermiteN(x, deg - 1) * (NDArray)Math.Sqrt(2 * deg);
        np.subtract(x, dy / df, @out: x);
        var fm = NormedHermiteN(x, deg - 1);
        np.divide(fm, np.max(np.abs(fm)), @out: fm);
        var w = One / (fm * fm);
        w = (w + w["::-1"]) / Two;
        x = (x - x["::-1"]) / Two;
        np.multiply(w, Math.Sqrt(Math.PI) / np.sum(w), @out: w);
        return (x, w);
    }

    /// <summary>NumPy's <c>chebinterpolate(np.sin, deg)</c>, statement for statement.</summary>
    /// <param name="deg">Non-negative degree.</param>
    /// <returns>The coefficients.</returns>
    public static NDArray ChebInterpolate(int deg)
    {
        long order = deg + 1;
        NDArray cc = 0.5 * Math.PI / order;
        var xc = np.sin(np.multiply(cc, np.arange(-order + 1, order + 1, 2, np.int64)));
        var c = np.dot(np.polynomial.chebyshev.chebvander(xc, deg).T, np.sin(xc));
        np.divide(c["0"], (NDArray)(double)order, @out: c["0"]);
        np.divide(c["1:"], (NDArray)(0.5 * order), @out: c["1:"]);
        return c;
    }
}
