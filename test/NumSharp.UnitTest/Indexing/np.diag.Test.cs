using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing
{
    /// <summary>
    ///     np.diag(v, k=0) / np.diagflat(v, k=0) — extract or construct a diagonal.
    ///     Every expectation below is real NumPy 2.4.2 output.
    /// </summary>
    [TestClass]
    public class np_diag_Test
    {
        // ---------------------------------------------------------------- diag: 1-D → 2-D

        [TestMethod]
        public void Diag_OneD_MainDiagonal()
        {
            np.diag(np.array(1, 2, 3)).Should()
                .BeShaped(3, 3).And.BeOfValues(1, 0, 0, 0, 2, 0, 0, 0, 3);
        }

        [TestMethod]
        public void Diag_OneD_PositiveK_GrowsAndShiftsRight()
        {
            // np.diag([1,2,3], 1) -> 4x4 with the values on the first super-diagonal.
            np.diag(np.array(1, 2, 3), 1).Should()
                .BeShaped(4, 4).And.BeOfValues(0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0);
        }

        [TestMethod]
        public void Diag_OneD_NegativeK_GrowsAndShiftsDown()
        {
            np.diag(np.array(1, 2, 3), -1).Should()
                .BeShaped(4, 4).And.BeOfValues(0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0);
        }

        [TestMethod]
        public void Diag_OneD_LargeK_SizeIsLenPlusAbsK()
        {
            np.diag(np.array(1, 2, 3), 3).Should().BeShaped(6, 6);
            np.diag(np.array(1, 2, 3), -3).Should().BeShaped(6, 6);
        }

        [TestMethod]
        public void Diag_OneD_ResultIsFreshWriteableAndOwned()
        {
            var v = np.array(1, 2, 3);
            var d = np.diag(v);

            d.Shape.IsWriteable.Should().BeTrue();
            d[0, 0] = 99;
            v.GetAtIndex<int>(0).Should().Be(1, "the 1-D branch constructs a new array, it is not a view");
        }

        [TestMethod]
        public void Diag_OneD_Strided_ReadsLogicalOrder()
        {
            // np.diag(np.arange(10)[::2]) -> 5x5 with 0,2,4,6,8 on the diagonal.
            var strided = np.arange(10)["::2"];
            np.diagonal(np.diag(strided)).Should().BeOfValues(0, 2, 4, 6, 8);
        }

        [TestMethod]
        public void Diag_OneD_NegativeStride_ReadsLogicalOrder()
        {
            var reversed = np.arange(6)["::-1"];
            np.diagonal(np.diag(reversed)).Should().BeOfValues(5, 4, 3, 2, 1, 0);
        }

        [TestMethod]
        public void Diag_OneD_Empty_IsZeroByZero_ButKShiftGrowsIt()
        {
            var empty = np.array(new int[0]);
            np.diag(empty).Should().BeShaped(0, 0);
            // n = 0 + |k| => a k-shifted empty vector still yields a |k|x|k| zero matrix.
            np.diag(empty, 2).Should().BeShaped(2, 2).And.BeOfValues(0, 0, 0, 0);
            np.diag(empty, -2).Should().BeShaped(2, 2).And.BeOfValues(0, 0, 0, 0);
        }

        // ---------------------------------------------------------------- diag: 2-D → view

        [TestMethod]
        public void Diag_TwoD_ExtractsDiagonal()
        {
            var m = np.arange(12).reshape(3, 4);
            np.diag(m).Should().BeOfValues(0, 5, 10).And.BeShaped(3);
            np.diag(m, 1).Should().BeOfValues(1, 6, 11).And.BeShaped(3);
            np.diag(m, -1).Should().BeOfValues(4, 9).And.BeShaped(2);
        }

        [TestMethod]
        public void Diag_TwoD_KBeyondBounds_IsEmpty()
        {
            var m = np.arange(12).reshape(3, 4);
            np.diag(m, 4).Should().BeShaped(0);
            np.diag(m, 5).Should().BeShaped(0);
            np.diag(m, -3).Should().BeShaped(0);
            np.diag(m, -5).Should().BeShaped(0);
        }

        [TestMethod]
        public void Diag_TwoD_IsAReadOnlyView()
        {
            // NumPy: np.shares_memory(np.diag(m), m) is True and flags.writeable is False,
            // despite the docstring's mention of "a copy".
            var m = np.arange(12).reshape(3, 4);
            var d = np.diag(m);

            d.Shape.IsWriteable.Should().BeFalse();
            m[1, 1] = 77L;
            d.GetAtIndex<long>(1).Should().Be(77L, "the 2-D branch aliases the source storage");
        }

        [TestMethod]
        public void Diag_TwoD_FContiguousAndStrided()
        {
            var m = np.arange(12).reshape(3, 4);
            np.diag(np.asfortranarray(m)).Should().BeOfValues(0, 5, 10);
            np.diag(np.arange(24).reshape(4, 6)["::2, ::2"]).Should().BeOfValues(0, 14);
        }

        [TestMethod]
        public void Diag_TwoD_EmptyShapes()
        {
            np.diag(np.zeros(new Shape(0, 0), NPTypeCode.Int32)).Should().BeShaped(0);
            np.diag(np.zeros(new Shape(3, 0), NPTypeCode.Int32)).Should().BeShaped(0);
            np.diag(np.zeros(new Shape(0, 3), NPTypeCode.Int32)).Should().BeShaped(0);
        }

        // ---------------------------------------------------------------- diag: errors

        [TestMethod]
        public void Diag_ZeroD_And_ThreeD_Throw_VerbatimText()
        {
            // NumPy: ValueError("Input must be 1- or 2-d.")
            new Action(() => np.diag(NDArray.Scalar(5)))
                .Should().Throw<ArgumentException>().WithMessage("Input must be 1- or 2-d.");
            new Action(() => np.diag(np.zeros(new Shape(2, 2, 2))))
                .Should().Throw<ArgumentException>().WithMessage("Input must be 1- or 2-d.");
        }

        // ---------------------------------------------------------------- diagflat

        [TestMethod]
        public void Diagflat_OneD_MatchesDiag()
        {
            np.diagflat(np.array(1, 2, 3)).Should()
                .BeShaped(3, 3).And.BeOfValues(1, 0, 0, 0, 2, 0, 0, 0, 3);
        }

        [TestMethod]
        public void Diagflat_TwoD_RavelsFirst()
        {
            // Unlike diag, diagflat flattens: [[1,2],[3,4]] -> 4x4 diag(1,2,3,4).
            np.diagflat(np.array(1, 2, 3, 4).reshape(2, 2)).Should()
                .BeShaped(4, 4).And.BeOfValues(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4);
        }

        [TestMethod]
        public void Diagflat_ThreeD_IsAccepted()
        {
            np.diagflat(np.arange(8).reshape(2, 2, 2)).Should().BeShaped(8, 8);
        }

        [TestMethod]
        public void Diagflat_ZeroD_IsOneByOne()
        {
            np.diagflat(NDArray.Scalar(7)).Should().BeShaped(1, 1).And.BeOfValues(7);
            np.diagflat(NDArray.Scalar(7), 2).Should()
                .BeShaped(3, 3).And.BeOfValues(0, 0, 7, 0, 0, 0, 0, 0, 0);
        }

        [TestMethod]
        public void Diagflat_KShift()
        {
            np.diagflat(np.array(1, 2, 3), 1).Should()
                .BeShaped(4, 4).And.BeOfValues(0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0);
            np.diagflat(np.array(1, 2, 3), -1).Should()
                .BeShaped(4, 4).And.BeOfValues(0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0);
        }

        [TestMethod]
        public void Diagflat_Empty_IsZeroByZero()
        {
            np.diagflat(np.array(new int[0])).Should().BeShaped(0, 0);
        }

        [TestMethod]
        public void Diagflat_RavelUsesLogicalCOrder_NotMemoryOrder()
        {
            // F-contiguous (2,3) ravels to 0..5 in C order, NOT column order.
            np.diagonal(np.diagflat(np.asfortranarray(np.arange(6).reshape(2, 3))))
                .Should().BeOfValues(0, 1, 2, 3, 4, 5);

            // Strided column-step view: logical order is 0,2,4,6,8,10.
            np.diagonal(np.diagflat(np.arange(12).reshape(3, 4)[":, ::2"]))
                .Should().BeOfValues(0, 2, 4, 6, 8, 10);
        }

        // ---------------------------------------------------------------- dtypes

        [TestMethod]
        public void Diag_And_Diagflat_PreserveDtype()
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
                var v = np.ones(new Shape(3), tc);
                np.diag(v).typecode.Should().Be(tc, $"diag must preserve {tc}");
                np.diagflat(v).typecode.Should().Be(tc, $"diagflat must preserve {tc}");
                np.diag(np.ones(new Shape(3, 3), tc)).typecode.Should().Be(tc);
            }
        }

        [TestMethod]
        public void Diag_OneD_AllDtypes_PlaceValuesOnDiagonal()
        {
            var codes = new[]
            {
                NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal
            };

            foreach (var tc in codes)
            {
                var built = np.diag(np.ones(new Shape(3), tc), 1);
                built.Should().BeShaped(4, 4);
                built.typecode.Should().Be(tc);
                // The k=1 super-diagonal holds the ones; everything else is zero.
                // (Counted rather than compared — BeOfValues has no loop for Half/Decimal.)
                np.count_nonzero(np.diagonal(built, 1)).Should().Be(3, $"the k=1 diagonal carries the values for {tc}");
                np.count_nonzero(built).Should().Be(3, $"nothing outside the k=1 diagonal is set for {tc}");
            }
        }
    }
}
