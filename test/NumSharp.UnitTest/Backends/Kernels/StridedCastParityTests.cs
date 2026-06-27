using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the cross-dtype strided cast path (<c>astype</c> on a non-contiguous view →
/// <c>NDIter.Copy</c> cross-dtype → <c>NDIterCasting.CopyStridedToStridedWithCast</c>).
///
/// That fallback (taken by every cast the SIMD <c>StridedCastKernel</c> rejects, i.e. any
/// pair involving Boolean/Char/Half/Decimal/Complex, AND every contiguous cast of those
/// types) used to recompute Σ coords·strides for BOTH operands per element and dispatch
/// the conversion type-switch per element — ~0.34-0.54× NumPy on the Vector-less dtypes.
/// It now walks the inner axis as a tight pointer run, advances outer axes incrementally,
/// and resolves a typed <c>Converts.FindConverter</c> delegate ONCE for primitive pairs
/// (Complex/Decimal keep the box-free scalar <c>ConvertValue</c> path for their exact
/// NumPy semantics). ~0.63-1.06× now, bit-exact.
///
/// Two guards: (1) addressing — a strided cast must equal casting the materialized copy,
/// for ALL 15×15 dtype pairs across awkward layouts; (2) semantics — hardcoded
/// NumPy-2.4.2-verified values for the divergence-prone directions (float→int sentinel/
/// wrap, signed→unsigned wrap, complex real-part drop / truthy-bool).
/// </summary>
[TestClass]
public class StridedCastParityTests
{
    private static readonly NPTypeCode[] AllDtypes =
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
        NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
        NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
        NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
    };

    public static System.Collections.Generic.IEnumerable<object[]> SrcDtypes()
    {
        foreach (var dt in AllDtypes)
            yield return new object[] { dt };
    }

    [DataTestMethod]
    [DynamicData(nameof(SrcDtypes), DynamicDataSourceType.Method)]
    public void StridedCast_ToEveryDtype_EqualsMaterializedCast(NPTypeCode src)
    {
        const int R = 6, C = 8;
        var baseArr = ((np.arange(R * C) % 13) + 1).astype(src).reshape(R, C);
        var layouts = new (string name, NDArray v)[]
        {
            ("strided", baseArr[":, ::2"]),     // strided inner axis (no tight memcpy run)
            ("sliced",  baseArr["1:5, 1:7"]),   // offset != 0, contiguous inner
            ("negrow",  baseArr["::-1, :"]),    // negative outer stride
            ("negcol",  baseArr[":, ::-1"]),    // negative inner stride
            ("F",       baseArr.copy(order: 'F')),
            ("T",       baseArr.T),
        };

        foreach (var (lname, v) in layouts)
        foreach (var dst in AllDtypes)
        {
            NDArray got = v.astype(dst);
            NDArray expect = v.copy().astype(dst); // materialized-then-cast reference
            Assert.AreEqual(expect.size, got.size, $"{src}->{dst}/{lname}: size");
            for (long i = 0; i < got.size; i++)
                Assert.AreEqual(expect.GetAtIndex(i), got.GetAtIndex(i),
                    $"{src}->{dst}/{lname}[{i}] strided cast != materialized cast");
        }
    }

    // NumPy 2.4.2-verified outputs for the conversion directions most likely to diverge.
    // Each row casts a 1-D source (made strided via [::1] won't str/de-contig, so we embed
    // into a 2-D array and take a strided column view) and checks one element.
    [TestMethod]
    public void StridedCast_EdgeSemantics_MatchNumPy()
    {
        // float64 -> int32 : NaN and ±inf map to the int MinValue sentinel; finite truncates.
        AssertStridedCast(new double[] { double.NaN, 1.9, -1.9, double.PositiveInfinity },
            NPTypeCode.Double, NPTypeCode.Int32,
            new double[] { -2147483648, 1, -1, -2147483648 });

        // int64 -> uint8 : modular wrap (NumPy unsigned narrowing).
        AssertStridedCast(new double[] { 256, -1, 255, 257 },
            NPTypeCode.Int64, NPTypeCode.Byte,
            new double[] { 0, 255, 255, 1 });

        // int64 -> int16 : wrap into signed 16-bit.
        AssertStridedCast(new double[] { 32768, -32769, 65536, -1 },
            NPTypeCode.Int64, NPTypeCode.Int16,
            new double[] { -32768, 32767, 0, -1 });

        // double -> uint16 : negative wraps, fractional truncates.
        AssertStridedCast(new double[] { -1, 65536, 3.9, 70000 },
            NPTypeCode.Double, NPTypeCode.UInt16,
            new double[] { 65535, 0, 3, 4464 });
    }

    [TestMethod]
    public void StridedCast_ComplexEdges_MatchNumPy()
    {
        // Complex -> Double : discard imaginary, keep real (NumPy ComplexWarning).
        var z = np.array(new Complex[] { new(3, 4), new(-2, 9), new(0, 5), new(7, -1) }).reshape(2, 2);
        var realCast = z.T.astype(NPTypeCode.Double); // z.T is non-contiguous
        var refReal = z.T.copy().astype(NPTypeCode.Double);
        for (long i = 0; i < realCast.size; i++)
            Assert.AreEqual(Convert.ToDouble(refReal.GetAtIndex(i)), Convert.ToDouble(realCast.GetAtIndex(i)), 0,
                $"complex->double real-part [{i}]");

        // Complex -> Boolean : true iff either part non-zero.
        var zb = np.array(new Complex[] { new(0, 0), new(0, 5), new(3, 0), new(0, 0) }).reshape(2, 2);
        var boolCast = zb.T.astype(NPTypeCode.Boolean);
        var refBool = zb.T.copy().astype(NPTypeCode.Boolean);
        for (long i = 0; i < boolCast.size; i++)
            Assert.AreEqual(refBool.GetAtIndex(i), boolCast.GetAtIndex(i), $"complex->bool truthy [{i}]");
    }

    // Build a (2, N) array from the source values (duplicated rows), take the strided column
    // view [:, ::1].T so the cast walks a non-contiguous operand, then check each element.
    private static void AssertStridedCast(double[] vals, NPTypeCode src, NPTypeCode dst, double[] expected)
    {
        int n = vals.Length;
        var flat = np.zeros(new Shape(2, n), NPTypeCode.Double);
        for (int i = 0; i < n; i++) { flat[0, i] = vals[i]; flat[1, i] = vals[i]; }
        var typed = flat.astype(src);
        var strided = typed.T;           // shape (n,2), non-contiguous
        Assert.IsFalse(strided.Shape.IsContiguous, $"{src}->{dst}: precondition strided");

        var got = strided.astype(dst);
        for (int i = 0; i < n; i++)
        {
            double g = ToDouble(got.GetAtIndex(i * 2)); // row i, col 0
            Assert.AreEqual(expected[i], g, 0, $"{src}->{dst} [{i}] (val {vals[i]})");
        }
    }

    private static double ToDouble(object o) => o switch
    {
        Half h => (double)h,
        char c => c,
        bool b => b ? 1 : 0,
        Complex z => z.Real,
        _ => Convert.ToDouble(o),
    };
}
