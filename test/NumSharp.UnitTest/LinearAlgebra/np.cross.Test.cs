using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     <c>np.cross</c> (main namespace) — every expectation probed against NumPy 2.4.2. Distinct
    ///     from <c>np.linalg.cross</c> (3-vectors only, single axis): this one takes the 2-vector
    ///     forms and separate <c>axisa</c>/<c>axisb</c>/<c>axisc</c>.
    /// </summary>
    [TestClass]
    public class np_cross_test
    {
        private static NDArray Ints(params int[] v) => np.array(v);

        [TestMethod]
        public void Cross_ThreeByThree()
        {
            np.cross(Ints(1, 2, 3), Ints(4, 5, 6)).Should().BeOfValues(-3, 6, -3).And.BeShaped(3);
        }

        [TestMethod]
        public void Cross_TwoByThree_MissingComponentIsZero()
        {
            np.cross(Ints(1, 2), Ints(4, 5, 6)).Should().BeOfValues(12, -6, -3).And.BeShaped(3);
        }

        [TestMethod]
        public void Cross_ThreeByTwo_MissingComponentIsZero()
        {
            np.cross(Ints(1, 2, 3), Ints(4, 5)).Should().BeOfValues(-15, 12, -3).And.BeShaped(3);
        }

        [TestMethod]
        public void Cross_TwoByTwo_ReturnsScalarZComponent()
        {
            // both 2-D vectors -> the scalar z-component, shape ().
            np.cross(Ints(1, 2), Ints(4, 5)).Should().BeOfValues(-3).And.BeShaped();
        }

        [TestMethod]
        public void Cross_DtypePromotion_IntTimesFloatIsFloat()
        {
            var r = np.cross(np.array(new int[] {1, 2, 3}), np.array(new double[] {4, 5, 6}));
            r.Should().BeOfValues(-3, 6, -3).And.BeShaped(3);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Cross_UnsignedResultWraps_LikeNumPy()
        {
            // cross([1,2,3],[4,5,6]) has z = 1*5-2*4 = -3, which wraps to 253 in uint8.
            var r = np.cross(np.array(new int[] {1, 2, 3}).astype(NPTypeCode.Byte),
                np.array(new int[] {4, 5, 6}).astype(NPTypeCode.Byte));
            r.Should().BeOfValues(253, 6, 253).And.BeShaped(3);
            r.typecode.Should().Be(NPTypeCode.Byte);
        }

        [TestMethod]
        public void Cross_Axisc_PlacesComponentAxis()
        {
            var a = np.array(new int[,] {{1, 2, 3}, {4, 5, 6}});
            var b = np.array(new int[,] {{4, 5, 6}, {1, 2, 3}});
            np.cross(a, b, axisc: 0).Should().BeOfValues(-3, 3, 6, -6, -3, 3).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Cross_AxisaAxisb_DefineVectorsAlongFirstAxis()
        {
            var a = np.array(new int[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
            var b = np.array(new int[,] {{7, 8, 9}, {4, 5, 6}, {1, 2, 3}});
            np.cross(a, b, axisa: 0, axisb: 0)
                .Should().BeOfValues(-24, 48, -24, -30, 60, -30, -36, 72, -36).And.BeShaped(3, 3);
        }

        [TestMethod]
        public void Cross_Axis_OverridesAllThree()
        {
            var a = np.array(new int[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
            var b = np.array(new int[,] {{7, 8, 9}, {4, 5, 6}, {1, 2, 3}});
            np.cross(a, b, axis: 0)
                .Should().BeOfValues(-24, -30, -36, 48, 60, 72, -24, -30, -36).And.BeShaped(3, 3);
        }

        [TestMethod]
        public void Cross_BroadcastsLeadingAxes()
        {
            var a = np.array(new int[,] {{1, 2, 3}});
            var b = np.array(new int[,] {{4, 5, 6}, {7, 8, 9}});
            np.cross(a, b).Should().BeOfValues(-3, 6, -3, -6, 12, -6).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void Cross_ReadsThroughNonContiguousLayouts()
        {
            var a = np.arange(1, 7).reshape(2, 3);
            var b = np.arange(7, 13).reshape(2, 3);
            // F-contiguous inputs read identically to C.
            np.cross(np.asfortranarray(a), np.asfortranarray(b))
                .Should().BeOfValues(-6, 12, -6, -6, 12, -6).And.BeShaped(2, 3);
            // reversed-row view.
            np.cross(a["::-1"], b).Should().BeOfValues(-3, 6, -3, -9, 18, -9).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void Cross_ZeroDimensionInput_Raises()
        {
            new Action(() => np.cross(np.array(5), Ints(1, 2, 3)))
                .Should().Throw<ValueError>().WithMessage("At least one array has zero dimension");
        }

        [TestMethod]
        public void Cross_BadVectorLength_Raises()
        {
            new Action(() => np.cross(Ints(1, 2, 3, 4), Ints(1, 2, 3)))
                .Should().Throw<ValueError>().WithMessage("incompatible dimensions for cross product*");
        }

        [TestMethod]
        public void Cross_AxisOutOfBounds_RaisesWithPrefix()
        {
            var a = np.array(new int[,] {{1, 2, 3}});
            new Action(() => np.cross(a, a, axisa: 5))
                .Should().Throw<AxisError>().WithMessage("axisa: axis 5 is out of bounds for array of dimension 2*");
        }
    }
}
