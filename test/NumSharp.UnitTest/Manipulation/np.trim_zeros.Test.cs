using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    ///     np.trim_zeros(filt, trim='fb', axis=None) — remove leading/trailing all-zero hyperplanes.
    ///     N-dimensional bounding-box semantics (NumPy 2.2.0+). Returns a view.
    /// </summary>
    [TestClass]
    public class np_trim_zeros_Test
    {
        // ---- 1-D ----
        [TestMethod]
        public void OneD_Default_TrimsBothSides()
        {
            var a = np.array(0, 0, 0, 1, 2, 3, 0, 2, 1, 0);
            np.trim_zeros(a).Should().BeOfValues(1, 2, 3, 0, 2, 1).And.BeShaped(6);
        }

        [TestMethod]
        public void OneD_TrimFront()
        {
            var a = np.array(0, 0, 0, 1, 2, 3, 0, 2, 1, 0);
            np.trim_zeros(a, "f").Should().BeOfValues(1, 2, 3, 0, 2, 1, 0).And.BeShaped(7);
        }

        [TestMethod]
        public void OneD_TrimBack()
        {
            var a = np.array(0, 0, 0, 1, 2, 3, 0, 2, 1, 0);
            np.trim_zeros(a, "b").Should().BeOfValues(0, 0, 0, 1, 2, 3, 0, 2, 1).And.BeShaped(9);
        }

        [TestMethod]
        public void OneD_TrimSpec_IsCaseInsensitive_And_bf_Alias()
        {
            var a = np.array(0, 0, 1, 2, 0);
            np.trim_zeros(a, "FB").Should().BeOfValues(1, 2).And.BeShaped(2);
            np.trim_zeros(a, "bf").Should().BeOfValues(1, 2).And.BeShaped(2);
            np.trim_zeros(a, "B").Should().BeOfValues(0, 0, 1, 2).And.BeShaped(4);
        }

        [TestMethod]
        public void OneD_AllZero_TrimsToEmpty_RegardlessOfTrim()
        {
            var z = np.zeros(new Shape(5)).astype(NPTypeCode.Int32);
            np.trim_zeros(z).Should().BeShaped(0);
            np.trim_zeros(z, "f").Should().BeShaped(0);
            np.trim_zeros(z, "b").Should().BeShaped(0);
        }

        [TestMethod]
        public void OneD_Empty_ReturnsEmpty()
        {
            np.trim_zeros(np.array(new int[0])).Should().BeShaped(0);
        }

        [TestMethod]
        public void OneD_SingleNonzero_Interior()
        {
            np.trim_zeros(np.array(0, 0, 5, 0, 0)).Should().BeOfValues(5).And.BeShaped(1);
        }

        // ---- 2-D bounding box ----
        [TestMethod]
        public void TwoD_BoundingBox()
        {
            var b = np.array(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).reshape(3, 6);
            np.trim_zeros(b).Should().BeOfValues(0, 2, 3, 1, 0, 3).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void TwoD_AxisNegative_TrimsThatAxisOnly()
        {
            var b = np.array(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).reshape(3, 6);
            np.trim_zeros(b, "fb", -1).Should().BeOfValues(0, 2, 3, 1, 0, 3, 0, 0, 0).And.BeShaped(3, 3);
        }

        [TestMethod]
        public void TwoD_Axis0_TrimsRowsOnly()
        {
            var b = np.array(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).reshape(3, 6);
            np.trim_zeros(b, "fb", 0).Should().BeOfValues(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0).And.BeShaped(2, 6);
        }

        [TestMethod]
        public void TwoD_TrimFront_And_Back()
        {
            var b = np.array(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).reshape(3, 6);
            np.trim_zeros(b, "f").Should().BeShaped(3, 5);
            np.trim_zeros(b, "b").Should().BeOfValues(0, 0, 2, 3, 0, 1, 0, 3).And.BeShaped(2, 4);
        }

        [TestMethod]
        public void TwoD_AxisSequence()
        {
            var b = np.array(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).reshape(3, 6);
            np.trim_zeros(b, "fb", new[] {0, 1}).Should().BeOfValues(0, 2, 3, 1, 0, 3).And.BeShaped(2, 3);
            np.trim_zeros(b, "fb", new[] {-1, -2}).Should().BeShaped(2, 3);
        }

        [TestMethod]
        public void EmptyAxisList_ReturnsUnmodified()
        {
            var b = np.array(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).reshape(3, 6);
            np.trim_zeros(b, "fb", new int[0]).Should().BeShaped(3, 6);
        }

        [TestMethod]
        public void AllZero_2D_TrimsRequestedAxes()
        {
            var z = np.zeros(new Shape(3, 4)).astype(NPTypeCode.Int32);
            np.trim_zeros(z).Should().BeShaped(0, 0);
            np.trim_zeros(z, "fb", 0).Should().BeShaped(0, 4);
            np.trim_zeros(z, "fb", 1).Should().BeShaped(3, 0);
        }

        // ---- 3-D ----
        [TestMethod]
        public void ThreeD_BoundingBox()
        {
            var c = np.zeros(new Shape(2, 3, 4)).astype(NPTypeCode.Int32);
            c[1, 1, 2] = (NDArray)9;
            c[0, 2, 1] = (NDArray)5;
            np.trim_zeros(c).Should().BeOfValues(0, 0, 5, 0, 0, 9, 0, 0).And.BeShaped(2, 2, 2);
        }

        // ---- errors ----
        [TestMethod]
        public void InvalidTrim_Throws()
        {
            var a = np.array(0, 1, 0);
            new Action(() => np.trim_zeros(a, "x")).Should()
                .Throw<ArgumentException>().WithMessage("*unexpected character(s) in `trim`*");
            new Action(() => np.trim_zeros(a, "")).Should().Throw<ArgumentException>();
            new Action(() => np.trim_zeros(a, "ff")).Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void RepeatedAxis_Throws()
        {
            var b = np.arange(6).reshape(2, 3);
            new Action(() => np.trim_zeros(b, "fb", new[] {0, 0})).Should()
                .Throw<ArgumentException>().WithMessage("*repeated axis*");
        }

        [TestMethod]
        public void OutOfRangeAxis_Throws()
        {
            var b = np.arange(6).reshape(2, 3);
            new Action(() => np.trim_zeros(b, "fb", 2)).Should()
                .Throw<AxisOutOfRangeException>().WithMessage("*out of bounds*");
            new Action(() => np.trim_zeros(b, "fb", -3)).Should()
                .Throw<AxisOutOfRangeException>().WithMessage("*out of bounds*");
        }

        // ---- 0-D ----
        [TestMethod]
        public void ZeroD_ReturnsUnmodified()
        {
            np.trim_zeros(NDArray.Scalar(5)).Should().BeShaped(size: 1, ndim: 0);
            np.trim_zeros(NDArray.Scalar(0)).Should().BeShaped(size: 1, ndim: 0);
        }

        [TestMethod]
        public void ZeroD_ExplicitAxis_Throws()
        {
            new Action(() => np.trim_zeros(NDArray.Scalar(5), "fb", 0)).Should()
                .Throw<AxisOutOfRangeException>();
        }

        // ---- dtypes / value semantics ----
        [TestMethod]
        public void Float_NanIsNonZero_NegZeroIsZero()
        {
            // -0.0 == 0 -> trimmed; NaN != 0 -> kept.
            var a = np.array(0.0, -0.0, double.NaN, 0.0, 1.5, 0.0);
            var r = np.trim_zeros(a);
            r.Should().BeShaped(3);
            double.IsNaN((double)r[0]).Should().BeTrue();
            ((double)r[1]).Should().Be(0.0);
            ((double)r[2]).Should().Be(1.5);
            // all -0.0 counts as all-zero
            np.trim_zeros(np.array(-0.0, -0.0, -0.0)).Should().BeShaped(0);
        }

        [TestMethod]
        public void Bool_TrimsFalse()
        {
            np.trim_zeros(np.array(false, false, true, false)).Should().BeShaped(1);
        }

        [TestMethod]
        public void AllDtypes_Trim1D()
        {
            var dtypes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
            };
            foreach (var t in dtypes)
            {
                var a = np.array(0, 0, 1, 2, 0).astype(t);
                var r = np.trim_zeros(a);
                r.Should().BeShaped(2);
                r.typecode.Should().Be(t);
            }
        }

        // ---- view semantics & layouts ----
        [TestMethod]
        public void ReturnsView_WriteThrough()
        {
            var a = np.array(0, 0, 1, 2, 3, 0);
            var t = np.trim_zeros(a);
            t[0] = (NDArray)99;
            ((int)a[2]).Should().Be(99); // t[0] aliases a[2]
        }

        [TestMethod]
        public void StridedAndReversedInputs()
        {
            var a = np.array(0, 0, 1, 2, 0, 3, 0, 0);
            np.trim_zeros(a["::2"]).Should().BeOfValues(1).And.BeShaped(1);     // [0,1,0,0] -> [1]
            np.trim_zeros(a["::-1"]).Should().BeOfValues(3, 0, 2, 1).And.BeShaped(4); // [0,0,3,0,2,1,0,0] -> [3,0,2,1]
        }

        // ---- second-pass edge coverage (all values probed against NumPy 2.4.2) ----
        [TestMethod]
        public void TwoD_NonContiguousLayouts()
        {
            var b = np.array(0, 0, 2, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).reshape(3, 6);
            np.trim_zeros(b["::-1"]).Should().BeOfValues(1, 0, 3, 0, 2, 3).And.BeShaped(2, 3);       // reversed rows
            np.trim_zeros(np.asfortranarray(b)).Should().BeOfValues(0, 2, 3, 1, 0, 3).And.BeShaped(2, 3); // F-order
            np.trim_zeros(b.T).Should().BeOfValues(0, 1, 2, 0, 3, 3).And.BeShaped(3, 2);              // transposed
        }

        [TestMethod]
        public void ZeroDimensionShapes()
        {
            np.trim_zeros(np.zeros(new Shape(0, 5)).astype(NPTypeCode.Int32)).Should().BeShaped(0, 0);
            np.trim_zeros(np.zeros(new Shape(3, 0)).astype(NPTypeCode.Int32)).Should().BeShaped(0, 0);
            np.trim_zeros(np.zeros(new Shape(2, 0, 3)).astype(NPTypeCode.Int32)).Should().BeShaped(0, 0, 0);
            // untrimmed axis is already empty, so it stays 0 too -> (0,0)
            np.trim_zeros(np.zeros(new Shape(0, 5)).astype(NPTypeCode.Int32), "fb", 1).Should().BeShaped(0, 0);
        }

        [TestMethod]
        public void AllNonzero_ReturnsFull()
        {
            np.trim_zeros(np.array(1, 2, 3)).Should().BeOfValues(1, 2, 3).And.BeShaped(3);
            np.trim_zeros(np.ones(new Shape(2, 3)).astype(NPTypeCode.Int32)).Should().BeShaped(2, 3);
        }

        [TestMethod]
        public void InteriorZeroHyperplane_Preserved()
        {
            // middle row is all-zero but lies inside the bounding box -> kept
            np.trim_zeros(np.array(1, 0, 0, 0, 0, 1).reshape(3, 2)).Should()
                .BeOfValues(1, 0, 0, 0, 0, 1).And.BeShaped(3, 2);
            // only the corner is non-zero
            np.trim_zeros(np.array(0, 0, 0, 5).reshape(2, 2)).Should().BeOfValues(5).And.BeShaped(1, 1);
        }

        [TestMethod]
        public void FourD_BoundingBox()
        {
            var c = np.zeros(new Shape(2, 2, 2, 2)).astype(NPTypeCode.Int32);
            c[1, 0, 1, 0] = (NDArray)7;
            np.trim_zeros(c).Should().BeOfValues(7).And.BeShaped(1, 1, 1, 1);
        }

        [TestMethod]
        public void NaN_IsNonZero_NotTrimmed()
        {
            var r = np.trim_zeros(np.array(double.NaN, double.NaN, double.NaN));
            r.Should().BeShaped(3);
            double.IsNaN((double)r[0]).Should().BeTrue();
        }

        [TestMethod]
        public void BroadcastInput_TrimsToView()
        {
            var b = np.broadcast_to(np.array(0, 1, 0), new Shape(2, 3));
            np.trim_zeros(b).Should().BeOfValues(1, 1).And.BeShaped(2, 1);
        }

        [TestMethod]
        public void TrimSpec_Whitespace_Throws()
        {
            var a = np.array(0, 1, 0);
            new Action(() => np.trim_zeros(a, " fb")).Should()
                .Throw<ArgumentException>().WithMessage("*unexpected character(s) in `trim`: ' fb'*");
        }

        [TestMethod]
        public void HalfDecimalComplex_Values()
        {
            np.trim_zeros(np.array(new Half[] {(Half)0, (Half)0, (Half)1.5, (Half)0})).Should().BeShaped(1);
            np.trim_zeros(np.array(new decimal[] {0m, 0m, 2m, 0m})).Should().BeShaped(1);
            // complex: 0+0j is zero, 0+2j is non-zero
            var cx = np.array(new System.Numerics.Complex[]
                {new(0, 0), new(0, 2), new(3, 0), new(0, 0)});
            np.trim_zeros(cx).Should().BeShaped(2);
        }
    }
}
