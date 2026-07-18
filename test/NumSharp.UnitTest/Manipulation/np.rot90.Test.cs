using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    ///     Tests for <see cref="np.rot90"/>. All expected values are taken from running
    ///     NumPy 2.4.2 (see the probes in the implementing session).
    /// </summary>
    [TestClass]
    public class np_rot90_Test
    {
        // ----------------------------------------------------------------- 2-D, default axes (0, 1)

        [TestMethod]
        public void Rot90_2D_K1_Counterclockwise()
        {
            var m = np.arange(6).reshape(2, 3);          // [[0,1,2],[3,4,5]]
            np.rot90(m).Should()
                .BeOfValues(2, 5, 1, 4, 0, 3).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Rot90_2D_K2()
        {
            var m = np.arange(6).reshape(2, 3);
            np.rot90(m, 2).Should()
                .BeOfValues(5, 4, 3, 2, 1, 0).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void Rot90_2D_K3()
        {
            var m = np.arange(6).reshape(2, 3);
            np.rot90(m, 3).Should()
                .BeOfValues(3, 0, 4, 1, 5, 2).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Rot90_2D_K0_IsIdentity()
        {
            var m = np.arange(6).reshape(2, 3);
            np.rot90(m, 0).Should()
                .BeOfValues(0, 1, 2, 3, 4, 5).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void Rot90_2D_K4_FullTurn()
        {
            var m = np.arange(6).reshape(2, 3);
            np.rot90(m, 4).Should()
                .BeOfValues(0, 1, 2, 3, 4, 5).And.BeShaped(2, 3);
        }

        // ----------------------------------------------------------------- k modulo (Python-style)

        [TestMethod]
        public void Rot90_NegativeK_MatchesPositiveModulo()
        {
            var m = np.arange(6).reshape(2, 3);
            // k=-1 == k=3
            np.rot90(m, -1).Should().BeOfValues(3, 0, 4, 1, 5, 2).And.BeShaped(3, 2);
            // k=-2 == k=2
            np.rot90(m, -2).Should().BeOfValues(5, 4, 3, 2, 1, 0).And.BeShaped(2, 3);
            // k=-4 == k=0
            np.rot90(m, -4).Should().BeOfValues(0, 1, 2, 3, 4, 5).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void Rot90_LargeK_Modulo()
        {
            var m = np.arange(6).reshape(2, 3);
            np.rot90(m, 5).Should().BeOfValues(2, 5, 1, 4, 0, 3).And.BeShaped(3, 2);    // 5 % 4 == 1
            np.rot90(m, 100).Should().BeOfValues(0, 1, 2, 3, 4, 5).And.BeShaped(2, 3);  // 100 % 4 == 0
            np.rot90(m, -100).Should().BeOfValues(0, 1, 2, 3, 4, 5).And.BeShaped(2, 3);
        }

        // ----------------------------------------------------------------- reversed axes / negative axes

        [TestMethod]
        public void Rot90_AxesReversed_IsInverseRotation()
        {
            var m = np.arange(6).reshape(2, 3);
            // rot90(m, 1, (1,0)) is the reverse of rot90(m, 1, (0,1)) == rot90(m, 3, (0,1))
            np.rot90(m, 1, new[] {1, 0}).Should()
                .BeOfValues(3, 0, 4, 1, 5, 2).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Rot90_NegativeAxes_3D()
        {
            var m = np.arange(24).reshape(2, 3, 4);
            np.rot90(m, 1, new[] {-1, -2}).Should()
                .BeOfValues(8, 4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3,
                            20, 16, 12, 21, 17, 13, 22, 18, 14, 23, 19, 15)
                .And.BeShaped(2, 4, 3);
        }

        // ----------------------------------------------------------------- 3-D / 4-D axes

        [TestMethod]
        public void Rot90_3D_Axes_1_2()
        {
            var m = np.arange(24).reshape(2, 3, 4);
            np.rot90(m, 1, new[] {1, 2}).Should()
                .BeOfValues(3, 7, 11, 2, 6, 10, 1, 5, 9, 0, 4, 8,
                            15, 19, 23, 14, 18, 22, 13, 17, 21, 12, 16, 20)
                .And.BeShaped(2, 4, 3);
        }

        [TestMethod]
        public void Rot90_3D_Axes_0_2()
        {
            var m = np.arange(24).reshape(2, 3, 4);
            np.rot90(m, 1, new[] {0, 2}).Should()
                .BeOfValues(3, 15, 7, 19, 11, 23, 2, 14, 6, 18, 10, 22,
                            1, 13, 5, 17, 9, 21, 0, 12, 4, 16, 8, 20)
                .And.BeShaped(4, 3, 2);
        }

        [TestMethod]
        public void Rot90_4D_Axes_1_3_ShapeOnly()
        {
            var m = np.arange(120).reshape(2, 3, 4, 5);
            np.rot90(m, 1, new[] {1, 3}).Should().BeShaped(2, 5, 4, 3);
            np.rot90(m, 2, new[] {1, 3}).Should().BeShaped(2, 3, 4, 5);
            np.rot90(m, 3, new[] {1, 3}).Should().BeShaped(2, 5, 4, 3);
        }

        // ----------------------------------------------------------------- non-contiguous input

        [TestMethod]
        public void Rot90_NonContiguousInput()
        {
            // b = arange(24).reshape(4,6)[::2, 1::2]  ->  [[1,3,5],[13,15,17]]
            var b = np.arange(24).reshape(4, 6)["::2, 1::2"];
            np.rot90(b).Should()
                .BeOfValues(5, 17, 3, 15, 1, 13).And.BeShaped(3, 2);
        }

        // ----------------------------------------------------------------- view semantics

        [TestMethod]
        public void Rot90_ReturnsView_WriteThroughReachesParent()
        {
            var m = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var r = np.rot90(m);      // r[0,0] aliases m[0,2]
            r[0, 0] = 999;
            // m flattens contiguously: index 2 is m[0,2]
            m.flatten().GetData<int>()[2].Should().Be(999);
        }

        [TestMethod]
        public void Rot90_K0_ReturnsView()
        {
            var m = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var r = np.rot90(m, 0);
            r[1, 1] = 777;
            m.flatten().GetData<int>()[4].Should().Be(777);
        }

        [TestMethod]
        public void Rot90_K2_ReturnsView()
        {
            var m = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var r = np.rot90(m, 2);   // r[0,0] aliases m[1,2]
            r[0, 0] = 555;
            m.flatten().GetData<int>()[5].Should().Be(555);
        }

        [TestMethod]
        public void Rot90_BroadcastInput_ResultIsReadOnly()
        {
            var bc = np.broadcast_to(np.arange(3), (2, 3));   // read-only broadcast view
            var r = np.rot90(bc);
            r.Shape.IsWriteable.Should().BeFalse();
        }

        // ----------------------------------------------------------------- dtype coverage (all 15)

        [TestMethod]
        public void Rot90_PreservesDtype_AllTypes()
        {
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
            };

            foreach (var code in codes)
            {
                var a = np.arange(6).reshape(2, 3).astype(code);
                var r = np.rot90(a);
                r.typecode.Should().Be(code, $"rot90 must preserve dtype {code}");
                r.Should().BeShaped(3, 2);
            }
        }

        // ----------------------------------------------------------------- empty arrays

        [TestMethod]
        public void Rot90_Empty_2D()
        {
            var e = np.zeros(new int[] {0, 3});
            np.rot90(e).Should().BeShaped(3, 0);
        }

        [TestMethod]
        public void Rot90_Empty_3D_Axes()
        {
            var e = np.zeros(new int[] {2, 0, 3});
            np.rot90(e, 1, new[] {0, 2}).Should().BeShaped(3, 0, 2);
            np.rot90(e, 2, new[] {0, 2}).Should().BeShaped(2, 0, 3);
            np.rot90(e, 3, new[] {0, 2}).Should().BeShaped(3, 0, 2);
        }

        // ----------------------------------------------------------------- error parity

        [TestMethod]
        public void Rot90_AxesLengthNot2_Throws()
        {
            var m = np.arange(6).reshape(2, 3);
            ((Action)(() => np.rot90(m, 1, new[] {0}))).Should()
                .Throw<ArgumentException>().WithMessage("len(axes) must be 2.*");
            ((Action)(() => np.rot90(m, 1, new[] {0, 1, 2}))).Should()
                .Throw<ArgumentException>().WithMessage("len(axes) must be 2.*");
        }

        [TestMethod]
        public void Rot90_SameAxes_Throws()
        {
            var m = np.arange(6).reshape(2, 3);
            ((Action)(() => np.rot90(m, 1, new[] {0, 0}))).Should()
                .Throw<ArgumentException>().WithMessage("Axes must be different.*");
            // (0, -2) names the same physical axis on ndim=2 -> "Axes must be different."
            ((Action)(() => np.rot90(m, 1, new[] {0, -2}))).Should()
                .Throw<ArgumentException>().WithMessage("Axes must be different.*");
            // (0, 2): |0-2| == ndim == 2 -> reported as "Axes must be different.", NOT out-of-range.
            ((Action)(() => np.rot90(m, 1, new[] {0, 2}))).Should()
                .Throw<ArgumentException>().WithMessage("Axes must be different.*");
        }

        [TestMethod]
        public void Rot90_AxesOutOfRange_Throws()
        {
            var m = np.arange(6).reshape(2, 3);
            ((Action)(() => np.rot90(m, 1, new[] {0, 3}))).Should()
                .Throw<ArgumentException>().WithMessage("*out of range for array of ndim=2*");
            ((Action)(() => np.rot90(m, 1, new[] {-3, 0}))).Should()
                .Throw<ArgumentException>().WithMessage("*out of range for array of ndim=2*");
        }

        [TestMethod]
        public void Rot90_1D_Throws()
        {
            // default axes (0,1) on ndim=1: |0-1| == ndim == 1 -> "Axes must be different."
            var m = np.arange(3);
            ((Action)(() => np.rot90(m))).Should()
                .Throw<ArgumentException>().WithMessage("Axes must be different.*");
        }

        [TestMethod]
        public void Rot90_0D_Throws()
        {
            // default axes (0,1) on ndim=0: axis 0 is out of range.
            var m = NDArray.Scalar(5);
            ((Action)(() => np.rot90(m))).Should()
                .Throw<ArgumentException>().WithMessage("*out of range for array of ndim=0*");
        }
    }
}
