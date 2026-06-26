using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Multi-version coverage for the rounding-family SIMD ops (floor / ceil / round / trunc) across
/// BOTH the direct kernel (np.floor/ceil/round_/trunc) and the fused np.evaluate path.
///
/// <para>
/// These bind per-type <c>Vector{128,256,512}.{Floor|Ceiling|Round|Truncate}</c> BCL methods that
/// exist only for float/double on a capable runtime (Floor/Ceiling .NET 7+, Round/Truncate .NET 9+),
/// and only at widths the runtime provides (a runtime can ship Vector256.Round before Vector512.Round).
/// The engine probes that capability at runtime (<c>DirectILKernelGenerator.RoundingVectorSimdAvailable</c>)
/// and falls back to scalar otherwise, so the SAME result must come out on every (.NET version, CPU)
/// combination — SIMD where available, scalar where not, never a "Could not find &lt;op&gt; for
/// Vector{N}" kernel-compile crash.
/// </para>
///
/// <para>
/// Because the test project multi-targets net8.0 and net10.0, CI runs this class TWICE: on net8.0 it
/// exercises the scalar-fallback branch for Round/Truncate, on net10.0 the SIMD branch — both must
/// produce identical, NumPy-2.4.2-correct values. Within a single run, the "large" (SIMD body) and
/// "small" (scalar tail) variants exercise both code paths on the same runtime.
/// </para>
/// </summary>
[TestClass]
public class SimdRoundingMultiVersionTests
{
    // Deterministic mix: fractional, negatives, and exact .5 ties (banker's rounding probes).
    private static double[] SampleD(int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++)
        {
            int k = i % 8;
            a[i] = k switch
            {
                0 => 2.5,    // tie -> even (2)
                1 => 3.5,    // tie -> even (4)
                2 => -2.5,   // tie -> even (-2)
                3 => 0.5,    // tie -> even (0)
                4 => 1.7 + i,
                5 => -1.3 - i,
                6 => 10.49,
                _ => -10.51,
            };
        }
        return a;
    }

    private static double Oracle(string op, double x) => op switch
    {
        "floor" => Math.Floor(x),
        "ceil" => Math.Ceiling(x),
        "trunc" => Math.Truncate(x),
        "round" => Math.Round(x, MidpointRounding.ToEven), // NumPy round == banker's
        _ => throw new ArgumentException(op),
    };

    private static NDArray Direct(string op, NDArray a) => op switch
    {
        "floor" => np.floor(a),
        "ceil" => np.ceil(a),
        "trunc" => np.trunc(a),
        "round" => np.round_(a),
        _ => throw new ArgumentException(op),
    };

    private static NpyExpr Fused(string op, NpyExpr a) => op switch
    {
        "floor" => NpyExpr.Floor(a),
        "ceil" => NpyExpr.Ceil(a),
        "trunc" => NpyExpr.Truncate(a),
        "round" => NpyExpr.Round(a),
        _ => throw new ArgumentException(op),
    };

    private static readonly string[] Ops = { "floor", "ceil", "trunc", "round" };

    // 200 hits the unrolled SIMD body + remainder + tail; 3 is scalar-tail only.
    private static readonly int[] Sizes = { 3, 200 };

    // ----------------------------------------------------------------------------------
    // Direct kernel: np.floor / ceil / trunc / round_  (double + float, contiguous)
    // ----------------------------------------------------------------------------------

    [TestMethod]
    public void Direct_Double_Contiguous_MatchesNumPy()
    {
        foreach (var op in Ops)
            foreach (var n in Sizes)
            {
                var src = SampleD(n);
                var r = Direct(op, np.array(src));
                for (int i = 0; i < n; i++)
                    r.GetDouble(i).Should().Be(Oracle(op, src[i]),
                        $"np.{op}(double[{n}])[{i}] of {src[i]} must match NumPy on every .NET version/width");
            }
    }

    [TestMethod]
    public void Direct_Single_Contiguous_MatchesNumPy()
    {
        foreach (var op in Ops)
            foreach (var n in Sizes)
            {
                var src = SampleD(n);
                var f = new float[n];
                for (int i = 0; i < n; i++) f[i] = (float)src[i];
                var r = Direct(op, np.array(f));
                for (int i = 0; i < n; i++)
                    r.GetSingle(i).Should().Be((float)Oracle(op, f[i]),
                        $"np.{op}(float[{n}])[{i}] of {f[i]} must match NumPy on every .NET version/width");
            }
    }

    // ----------------------------------------------------------------------------------
    // Direct kernel: strided source (reversed view) — fused strided-SIMD gather path
    // ----------------------------------------------------------------------------------

    [TestMethod]
    public void Direct_Double_StridedReversed_MatchesNumPy()
    {
        foreach (var op in Ops)
        {
            const int n = 200;
            var src = SampleD(n);
            var rev = np.array(src)["::-1"];
            var r = Direct(op, rev);
            for (int i = 0; i < n; i++)
                r.GetDouble(i).Should().Be(Oracle(op, src[n - 1 - i]),
                    $"np.{op}(double[::-1])[{i}] must match NumPy (strided SIMD/scalar gather)");
        }
    }

    // ----------------------------------------------------------------------------------
    // Direct kernel: integer input must not crash and is an identity copy (NumSharp's
    // documented identity-loop behavior — there is no Vector{N}.Floor(int) at any width).
    // ----------------------------------------------------------------------------------

    [TestMethod]
    public void Direct_Integer_Identity_NoCrash()
    {
        var ints = np.array(new int[] { -3, -1, 0, 1, 2, 3, 7, 100, -100, 42 });
        foreach (var op in Ops)
        {
            NDArray r = null;
            new Action(() => r = Direct(op, ints)).Should().NotThrow(
                $"np.{op}(int[]) has no Vector{{N}} loop for integers — it must use the scalar " +
                "identity path, not hit the SIMD emitter's 'Could not find' throw");
            for (int i = 0; i < ints.size; i++)
                r.GetInt32(i).Should().Be(ints.GetInt32(i), $"integer {op} is an identity at index {i}");
        }
    }

    // ----------------------------------------------------------------------------------
    // Fused np.evaluate: float/double float-rounding (SIMD on .NET 9+, scalar on net8.0)
    // ----------------------------------------------------------------------------------

    [TestMethod]
    public void Fused_Double_MatchesNumPy()
    {
        foreach (var op in Ops)
            foreach (var n in Sizes)
            {
                var src = SampleD(n);
                var r = np.evaluate(Fused(op, NpyExpr.Arr(np.array(src))));
                for (int i = 0; i < n; i++)
                    r.GetDouble(i).Should().Be(Oracle(op, src[i]),
                        $"np.evaluate({op}(double[{n}]))[{i}] must match NumPy on every .NET version/width");
            }
    }

    [TestMethod]
    public void Fused_Single_MatchesNumPy()
    {
        foreach (var op in Ops)
            foreach (var n in Sizes)
            {
                var src = SampleD(n);
                var f = new float[n];
                for (int i = 0; i < n; i++) f[i] = (float)src[i];
                var r = np.evaluate(Fused(op, NpyExpr.Arr(np.array(f))));
                for (int i = 0; i < n; i++)
                    r.GetSingle(i).Should().Be((float)Oracle(op, f[i]),
                        $"np.evaluate({op}(float[{n}]))[{i}] must match NumPy on every .NET version/width");
            }
    }

    // ----------------------------------------------------------------------------------
    // Fused np.evaluate: integer input. Regression guard — this previously threw
    // "Could not find Floor for Vector256<Int32>" because the fused path green-lit the
    // vector route for an integer-preserving rounding op. It must now stay scalar.
    // ----------------------------------------------------------------------------------

    [TestMethod]
    public void Fused_Integer_Identity_NoCrash()
    {
        var ints = np.array(new int[] { -3, -1, 0, 1, 2, 3, 7, 100, -100, 42 });
        foreach (var op in Ops)
        {
            NDArray r = null;
            new Action(() => r = np.evaluate(Fused(op, NpyExpr.Arr(ints)))).Should().NotThrow(
                $"np.evaluate({op}(int[])) must not vectorize an integer rounding op (no " +
                "Vector{N} integer loop) — it must use the scalar identity emit");
            for (int i = 0; i < ints.size; i++)
                r.GetInt32(i).Should().Be(ints.GetInt32(i), $"fused integer {op} is an identity at index {i}");
        }
    }

    // ----------------------------------------------------------------------------------
    // Fused np.evaluate: chained Binary -> Unary so the type-aware SIMD gate must recurse
    // through the binary node into the rounding node (floor(a * a)).
    // ----------------------------------------------------------------------------------

    [TestMethod]
    public void Fused_Chained_FloorOfProduct_MatchesNumPy()
    {
        const int n = 200;
        var src = SampleD(n);
        var a = np.array(src);
        var r = np.evaluate(NpyExpr.Floor(NpyExpr.Arr(a) * NpyExpr.Arr(a)));
        for (int i = 0; i < n; i++)
            r.GetDouble(i).Should().Be(Math.Floor(src[i] * src[i]),
                $"np.evaluate(floor(a*a))[{i}] must match NumPy (Binary->Unary SIMD recursion)");
    }
}
