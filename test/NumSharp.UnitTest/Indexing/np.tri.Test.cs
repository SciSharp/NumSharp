using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing
{
    /// <summary>
    ///     np.tri / np.tril / np.triu — the triangular half of NumPy's two-dimensional base.
    ///     Every expectation below is real NumPy 2.4.2 output.
    /// </summary>
    [TestClass]
    public class np_tri_Test
    {
        // ---------------------------------------------------------------- tri

        [TestMethod]
        public void Tri_Square_DefaultsToFloat64AndMainDiagonal()
        {
            var t = np.tri(3);
            t.Should().BeShaped(3, 3);
            t.typecode.Should().Be(NPTypeCode.Double, "NumPy's tri defaults to dtype=float");
            t.Should().BeOfValues(1, 0, 0, 1, 1, 0, 1, 1, 1);
        }

        [TestMethod]
        public void Tri_Rectangular_And_KOffsets()
        {
            np.tri(3, 5).Should().BeShaped(3, 5)
                .And.BeOfValues(1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0);
            np.tri(3, 5, 1).Should()
                .BeOfValues(1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0);
            np.tri(3, 5, -1).Should()
                .BeOfValues(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0);
        }

        [TestMethod]
        public void Tri_SaturatingK_AllOnesOrAllZeros()
        {
            np.tri(3, 5, 10).Should().BeOfValues(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
            np.tri(3, 5, -10).Should().BeOfValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        [TestMethod]
        public void Tri_ZeroAndNegativeExtents_ClampToEmptyAxes()
        {
            np.tri(0).Should().BeShaped(0, 0);
            np.tri(3, 0).Should().BeShaped(3, 0);
            np.tri(0, 3).Should().BeShaped(0, 3);
            // NumPy reaches the extents through arange, so a negative count is an empty axis.
            np.tri(-2).Should().BeShaped(0, 0);
            np.tri(3, -2).Should().BeShaped(3, 0);
        }

        [TestMethod]
        public void Tri_HonoursDtype()
        {
            np.tri(3, 4, 0, NPTypeCode.Boolean).typecode.Should().Be(NPTypeCode.Boolean);
            np.tri(3, 4, 0, NPTypeCode.Boolean).Should()
                .BeOfValues(true, false, false, false, true, true, false, false, true, true, true, false);
            np.tri(3, 3, 0, typeof(int)).typecode.Should().Be(NPTypeCode.Int32);
            np.tri(3, 3, 0, NPTypeCode.Complex).typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Tri_AllDtypes_ProduceOnesBelowDiagonal()
        {
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
                NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
                NPTypeCode.UInt64, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
                NPTypeCode.Decimal, NPTypeCode.Complex
            };

            foreach (var tc in codes)
            {
                var t = np.tri(3, 4, 0, tc);
                t.typecode.Should().Be(tc);
                // 6 ones in a (3,4) k=0 lower triangle.
                np.count_nonzero(t).Should().Be(6, $"tri must fill the lower triangle for {tc}");
            }
        }

        // ---------------------------------------------------------------- tril / triu

        [TestMethod]
        public void Tril_And_Triu_TwoD()
        {
            var m = np.arange(12).reshape(3, 4);
            np.tril(m).Should().BeOfValues(0, 0, 0, 0, 4, 5, 0, 0, 8, 9, 10, 0);
            np.triu(m).Should().BeOfValues(0, 1, 2, 3, 0, 5, 6, 7, 0, 0, 10, 11);
        }

        [TestMethod]
        public void Tril_And_Triu_KOffsets()
        {
            var m = np.arange(12).reshape(3, 4);
            np.tril(m, 1).Should().BeOfValues(0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11);
            np.tril(m, -1).Should().BeOfValues(0, 0, 0, 0, 4, 0, 0, 0, 8, 9, 0, 0);
            np.triu(m, 1).Should().BeOfValues(0, 1, 2, 3, 0, 0, 6, 7, 0, 0, 0, 11);
            np.triu(m, -1).Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11);
        }

        [TestMethod]
        public void Tril_And_Triu_SaturatingK()
        {
            var m = np.arange(12).reshape(3, 4);
            np.tril(m, 10).Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            np.tril(m, -10).Should().BeOfValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            np.triu(m, 10).Should().BeOfValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            np.triu(m, -10).Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Tril_OneDInput_ProducesTwoDResult()
        {
            // NumPy quirk: m.shape[-2:] of a (n,) array is (n,), so the mask is tri(n) -> (n,n)
            // and the `where` broadcasts the row down it.
            np.tril(np.array(1, 2, 3)).Should()
                .BeShaped(3, 3).And.BeOfValues(1, 0, 0, 1, 2, 0, 1, 2, 3);
            np.triu(np.array(1, 2, 3)).Should()
                .BeShaped(3, 3).And.BeOfValues(1, 2, 3, 0, 2, 3, 0, 0, 3);
            np.tril(np.array(1, 2, 3), 1).Should()
                .BeOfValues(1, 2, 0, 1, 2, 3, 1, 2, 3);
            np.triu(np.array(1, 2, 3), -1).Should()
                .BeOfValues(1, 2, 3, 1, 2, 3, 0, 2, 3);
        }

        [TestMethod]
        public void Tril_ZeroD_Throws_NumPysLeakedText()
        {
            // NumPy leaks tri()'s own missing-argument TypeError, because `tri(*m.shape[-2:])`
            // receives no arguments. The text is reproduced verbatim.
            new Action(() => np.tril(NDArray.Scalar(5)))
                .Should().Throw<ArgumentException>()
                .WithMessage("tri() missing 1 required positional argument: 'N'");
            new Action(() => np.triu(NDArray.Scalar(5)))
                .Should().Throw<ArgumentException>()
                .WithMessage("tri() missing 1 required positional argument: 'N'");
        }

        [TestMethod]
        public void Tril_ThreeD_AppliesToLastTwoAxes()
        {
            np.tril(np.arange(24).reshape(2, 3, 4)).Should()
                .BeShaped(2, 3, 4)
                .And.BeOfValues(0, 0, 0, 0, 4, 5, 0, 0, 8, 9, 10, 0,
                                12, 0, 0, 0, 16, 17, 0, 0, 20, 21, 22, 0);
            np.triu(np.arange(24).reshape(2, 3, 4)).Should()
                .BeOfValues(0, 1, 2, 3, 0, 5, 6, 7, 0, 0, 10, 11,
                            12, 13, 14, 15, 0, 17, 18, 19, 0, 0, 22, 23);
        }

        [TestMethod]
        public void Tril_HandlesEveryMemoryLayout()
        {
            var m = np.arange(12).reshape(3, 4);

            // F-contiguous
            np.tril(np.asfortranarray(m)).Should().BeOfValues(0, 0, 0, 0, 4, 5, 0, 0, 8, 9, 10, 0);
            np.triu(np.asfortranarray(m)).Should().BeOfValues(0, 1, 2, 3, 0, 5, 6, 7, 0, 0, 10, 11);

            // Column-stepped view (last-axis stride != 1 -> composition path)
            np.tril(np.arange(24).reshape(4, 6)[":3, ::2"]).Should()
                .BeShaped(3, 3).And.BeOfValues(0, 0, 0, 6, 8, 0, 12, 14, 16);

            // Negative-stride view
            np.tril(np.arange(12).reshape(3, 4)[":, ::-1"]).Should()
                .BeShaped(3, 4).And.BeOfValues(3, 0, 0, 0, 7, 6, 0, 0, 11, 10, 9, 0);

            // Broadcast (read-only) source still yields a fresh writeable result
            var bc = np.tril(np.broadcast_to(np.arange(3), new Shape(3, 3)));
            bc.Should().BeOfValues(0, 0, 0, 0, 1, 0, 0, 1, 2);
            bc.Shape.IsWriteable.Should().BeTrue();
        }

        [TestMethod]
        public void Tril_ResultIsAFreshWriteableCopy()
        {
            var m = np.arange(12).reshape(3, 4);
            var t = np.tril(m);
            t.Shape.IsWriteable.Should().BeTrue();
            t[2, 0] = 99L;
            m.GetValue<long>(2, 0).Should().Be(8L, "tril returns a copy, never a view");
        }

        [TestMethod]
        public void Tril_EmptyAndDegenerateShapes()
        {
            np.tril(np.zeros(new Shape(0, 0), NPTypeCode.Int32)).Should().BeShaped(0, 0);
            np.tril(np.zeros(new Shape(3, 0), NPTypeCode.Int32)).Should().BeShaped(3, 0);
            np.tril(np.zeros(new Shape(0, 3), NPTypeCode.Int32)).Should().BeShaped(0, 3);
            np.tril(np.arange(4).reshape(1, 4)).Should().BeOfValues(0, 0, 0, 0);
            np.tril(np.arange(4).reshape(4, 1)).Should().BeOfValues(0, 1, 2, 3);
        }

        [TestMethod]
        public void Tril_PreservesDtype()
        {
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
                NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
                NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
                NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
            };

            foreach (var tc in codes)
            {
                var ones = np.ones(new Shape(3, 3), tc);
                np.tril(ones).typecode.Should().Be(tc, $"tril must preserve {tc}");
                np.triu(ones).typecode.Should().Be(tc, $"triu must preserve {tc}");
                np.count_nonzero(np.tril(ones)).Should().Be(6);
                np.count_nonzero(np.triu(ones)).Should().Be(6);
            }
        }

        [TestMethod]
        public void Tril_PreservesSpecialFloatValues()
        {
            // Kept cells keep NaN / +inf / -0.0 bit-for-bit; dropped cells become +0.0.
            var m = np.array(1.0d, double.NaN, double.PositiveInfinity, -0.0d).reshape(2, 2);
            var lower = np.tril(m);

            lower.GetValue<double>(0, 0).Should().Be(1.0d);
            double.IsNaN(lower.GetValue<double>(0, 1)).Should().BeFalse("the upper cell is zeroed");
            lower.GetValue<double>(0, 1).Should().Be(0.0d);
            double.IsPositiveInfinity(lower.GetValue<double>(1, 0)).Should().BeTrue();
            double.IsNegative(lower.GetValue<double>(1, 1)).Should().BeTrue("-0.0 survives on the diagonal");

            var upper = np.triu(m);
            double.IsNaN(upper.GetValue<double>(0, 1)).Should().BeTrue();
            upper.GetValue<double>(1, 0).Should().Be(0.0d);
        }
    }
}
