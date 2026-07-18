using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Tests for np.asarray_chkfinite — verified 1-to-1 against NumPy 2.4.2.
    /// The finiteness check runs only for the float family (Half/Single/Double/Complex),
    /// raising ValueError("array must not contain infs or NaNs") on any inf/NaN. Integer,
    /// boolean and char arrays are never checked. No copy is made when the input already matches.
    /// </summary>
    [TestClass]
    public class np_asarray_chkfinite_Tests
    {
        const double inf = double.PositiveInfinity;
        const double nan = double.NaN;

        static Action Chk(NDArray a, Type dtype = null) => () => np.asarray_chkfinite(a, dtype);

        // ─── finite arrays pass, all float dtypes ───────────────────────────

        [TestMethod]
        public void Finite_Double_Passes()
        {
            var r = np.asarray_chkfinite(np.array(new[] { 1.0, 2.0, 3.0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.shape.Should().Equal(new long[] { 3 });
        }

        [TestMethod]
        public void Finite_Single_Passes() => Chk(np.array(new[] { 1f, 2f, 3f })).Should().NotThrow();

        [TestMethod]
        public void Finite_Half_Passes() =>
            Chk(np.array(new[] { (Half)1, (Half)2, (Half)65504 })).Should().NotThrow(); // 65504 = max finite half

        [TestMethod]
        public void Finite_Complex_Passes() =>
            Chk(np.array(new[] { new Complex(1, 2), new Complex(3, 4) })).Should().NotThrow();

        // ─── inf / NaN raise ValueError with NumPy's exact message ──────────

        [TestMethod]
        public void PositiveInf_Double_Raises() =>
            Chk(np.array(new[] { 1.0, inf, 3.0 })).Should()
                .Throw<ValueError>().WithMessage("array must not contain infs or NaNs*");

        [TestMethod]
        public void NegativeInf_Single_Raises() =>
            Chk(np.array(new[] { 1f, float.NegativeInfinity })).Should().Throw<ValueError>();

        [TestMethod]
        public void NaN_Double_Raises() => Chk(np.array(new[] { 1.0, nan, 3.0 })).Should().Throw<ValueError>();

        [TestMethod]
        public void Inf_Half_Raises() =>
            Chk(np.array(new[] { (Half)1, Half.PositiveInfinity })).Should().Throw<ValueError>();

        [TestMethod]
        public void NaN_Half_Raises() => Chk(np.array(new[] { (Half)1, Half.NaN })).Should().Throw<ValueError>();

        [TestMethod]
        public void Complex_RealInf_Raises() =>
            Chk(np.array(new[] { new Complex(inf, 0), new Complex(3, 4) })).Should().Throw<ValueError>();

        [TestMethod]
        public void Complex_ImagNaN_Raises() =>
            Chk(np.array(new[] { new Complex(1, nan) })).Should().Throw<ValueError>();

        // ─── integer / bool never checked ───────────────────────────────────

        [TestMethod]
        public void Integer_NeverChecked() => Chk(np.array(new[] { 1, 2, 3 })).Should().NotThrow();

        [TestMethod]
        public void Bool_NeverChecked() => Chk(np.array(new[] { true, false })).Should().NotThrow();

        // ─── empty arrays pass (vacuous) ────────────────────────────────────

        [TestMethod]
        public void Empty_1D_Passes()
        {
            var r = np.asarray_chkfinite(np.zeros(new Shape(0), NPTypeCode.Double));
            r.shape.Should().Equal(new long[] { 0 });
        }

        [TestMethod]
        public void Empty_2D_Passes() =>
            Chk(np.zeros(new Shape(0, 3), NPTypeCode.Double)).Should().NotThrow();

        // ─── strided views: check the SELECTED physical elements ────────────

        [TestMethod]
        public void Strided_SkipsInf_Passes()
        {
            // buffer [1, inf, 2, 3, 4, 5]; [::2] selects indices 0,2,4 = 1,2,4 (all finite)
            var a = np.array(new[] { 1.0, inf, 2.0, 3.0, 4.0, 5.0 });
            Chk(a["::2"]).Should().NotThrow();
        }

        [TestMethod]
        public void Strided_HitsInf_Raises()
        {
            var a = np.array(new[] { 1.0, inf, 2.0, 3.0, 4.0, 5.0 });
            Chk(a["1::2"]).Should().Throw<ValueError>(); // indices 1,3,5 = inf,3,5
        }

        [TestMethod]
        public void Reversed_HitsInf_Raises()
        {
            var a = np.arange(40).astype(NPTypeCode.Double);
            a[25] = inf;
            Chk(a["::-1"]).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void Reversed_Finite_Passes()
        {
            var a = np.arange(40).astype(NPTypeCode.Double);
            Chk(a["::-1"]).Should().NotThrow();
        }

        [TestMethod]
        public void Transposed_HitsNaN_Raises()
        {
            var m = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, nan, 6.0 } });
            Chk(m.T).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void ColumnSlice_SkipsNaN_Passes()
        {
            var m = np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, nan, 6.0 } });
            Chk(m[":, :1"]).Should().NotThrow(); // column 0 = 1,4 finite
        }

        [TestMethod]
        public void Broadcast_HitsNaN_Raises()
        {
            var b = np.broadcast_to(np.array(new[] { 1.0, nan, 3.0 }), new Shape(4, 3));
            Chk(b).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void Broadcast_Finite_Passes()
        {
            var b = np.broadcast_to(np.array(new[] { 1.0, 2.0, 3.0 }), new Shape(4, 3));
            Chk(b).Should().NotThrow();
        }

        // ─── large contiguous scan (exercises the SIMD kernel) ──────────────

        [TestMethod]
        public void LargeContiguous_Finite_Passes() =>
            Chk(np.ones(new Shape(100000), NPTypeCode.Double)).Should().NotThrow();

        [TestMethod]
        public void LargeContiguous_TrailingNaN_Raises()
        {
            var a = np.ones(new Shape(100000), NPTypeCode.Double);
            a[99999] = nan; // last element — proves the whole array is scanned (no false early success)
            Chk(a).Should().Throw<ValueError>();
        }

        // ─── dtype conversion checks AFTER the cast ─────────────────────────

        [TestMethod]
        public void DtypeCast_IntToFloat_Passes()
        {
            var r = np.asarray_chkfinite(np.array(new[] { 1, 2, 3 }), typeof(float));
            r.dtype.Should().Be(typeof(float));
        }

        // ─── no-copy contract ───────────────────────────────────────────────

        [TestMethod]
        public void NoCopy_WhenAlreadyMatching()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            var r = np.asarray_chkfinite(a);
            ReferenceEquals(a.Storage, r.Storage).Should().BeTrue();
        }

        // ─── 0-D scalars ────────────────────────────────────────────────────

        [TestMethod]
        public void Scalar_Inf_Raises() => Chk(NDArray.Scalar(inf)).Should().Throw<ValueError>();

        [TestMethod]
        public void Scalar_Finite_Passes() => Chk(NDArray.Scalar(5.0)).Should().NotThrow();

        [TestMethod]
        public void NullInput_Throws() =>
            ((Action)(() => np.asarray_chkfinite((NDArray)null))).Should().Throw<ArgumentNullException>();

        // ─── second-pass edge cases (verified against NumPy) ────────────────

        [TestMethod]
        public void SignalingNaN_Double_Raises()
        {
            double sNaN = BitConverter.Int64BitsToDouble(unchecked((long)0x7ff0000000000001));
            Chk(np.array(new[] { sNaN, 1.0 })).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void SignalingNaN_Half_Raises()
        {
            Half sNaN = BitConverter.UInt16BitsToHalf(0x7c01);
            Chk(np.array(new[] { sNaN, (Half)1 })).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void SignedZero_Passes() => Chk(np.array(new[] { -0.0, 0.0, 1.0 })).Should().NotThrow();

        [TestMethod]
        public void Subnormal_And_MaxFinite_Pass() =>
            Chk(np.array(new[] { 5e-324, 1.7976931348623157e308, -1.7976931348623157e308 })).Should().NotThrow();

        [TestMethod]
        public void ColumnBroadcast_InnerStrideZero_HitsNaN_Raises()
        {
            // (3,1) broadcast to (3,4): inner axis has stride 0 — exercises the odometer's stride-0 run.
            var col = np.array(new[,] { { 1.0 }, { nan }, { 3.0 } });
            Chk(np.broadcast_to(col, new Shape(3, 4))).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void FiveD_StridedView_SkipsNaN_Passes()
        {
            var a5 = np.ones(new Shape(2, 3, 2, 3, 4), NPTypeCode.Double);
            a5[1, 2, 1, 2, 3] = inf;                 // lives at axis0=1, axis2=1
            Chk(a5["::2, :, ::2"]).Should().NotThrow(); // view selects axis0={0}, axis2={0} — inf excluded
        }

        [TestMethod]
        public void Stride3_Gather_HitsAndSkips()
        {
            var b = np.ones(new Shape(30), NPTypeCode.Double);
            b[9] = nan;
            Chk(b["::3"]).Should().Throw<ValueError>();  // 0,3,6,9,... hits 9
            var c = np.ones(new Shape(30), NPTypeCode.Double);
            c[9] = nan;
            Chk(c["1::3"]).Should().NotThrow();          // 1,4,7,10,... skips 9
        }

        [TestMethod]
        public void SimdBoundary_Sweep_NaNAtEveryPosition_AlwaysDetected()
        {
            // Sizes straddling the 4×-unroll SIMD body, the 1-vector remainder and the scalar tail.
            foreach (var n in new[] { 4, 5, 8, 9, 16, 17, 33, 64, 65 })
                for (int pos = 0; pos < n; pos++)
                {
                    var a = np.ones(new Shape(n), NPTypeCode.Double);
                    a[pos] = nan;
                    Chk(a).Should().Throw<ValueError>($"NaN at position {pos} of {n} must be found");
                }
        }
    }
}
