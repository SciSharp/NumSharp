using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Comprehensive tests for np.flip / np.fliplr / np.flipud, verified against NumPy 2.4.2 output.
    ///
    /// All three return VIEWS (negative-stride slices, constant time) — never copies.
    ///
    /// NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
    /// NumPy source: numpy/lib/_function_base_impl.py (flip), numpy/lib/_twodim_base_impl.py (fliplr/flipud)
    /// NumPy test source: numpy/_core/tests/test_function_base.py, numpy/lib/tests/test_twodim_base.py
    /// </summary>
    [TestClass]
    public class np_flip_Test
    {
        // ================================================================
        // np.flip — AXIS VARIANTS (baseline: np.arange(8).reshape(2,2,2))
        // ================================================================

        [TestMethod]
        public void Flip_1D()
        {
            // np.flip(np.arange(5)) => [4 3 2 1 0]
            var result = np.flip(np.arange(5));

            result.Should().BeShaped(5);
            result.Should().BeOfValues(4, 3, 2, 1, 0);
        }

        [TestMethod]
        public void Flip_3D_AllAxes()
        {
            // np.flip(np.arange(8).reshape(2,2,2)) => [[[7 6],[5 4]],[[3 2],[1 0]]]
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a);

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(7, 6, 5, 4, 3, 2, 1, 0);
        }

        [TestMethod]
        public void Flip_3D_Axis0()
        {
            // np.flip(a, 0) => [[[4 5],[6 7]],[[0 1],[2 3]]]
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, 0);

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(4, 5, 6, 7, 0, 1, 2, 3);
        }

        [TestMethod]
        public void Flip_3D_Axis1()
        {
            // np.flip(a, 1) => [[[2 3],[0 1]],[[6 7],[4 5]]]
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, 1);

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(2, 3, 0, 1, 6, 7, 4, 5);
        }

        [TestMethod]
        public void Flip_3D_Axis2()
        {
            // np.flip(a, 2) => [[[1 0],[3 2]],[[5 4],[7 6]]]
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, 2);

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(1, 0, 3, 2, 5, 4, 7, 6);
        }

        [TestMethod]
        public void Flip_3D_NegativeAxis()
        {
            // np.flip(a, -1) == np.flip(a, 2)
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, -1);

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(1, 0, 3, 2, 5, 4, 7, 6);
        }

        [TestMethod]
        public void Flip_3D_AxisTuple()
        {
            // np.flip(a, (0, 2)) => [[[5 4],[7 6]],[[1 0],[3 2]]]
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, new[] {0, 2});

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(5, 4, 7, 6, 1, 0, 3, 2);
        }

        [TestMethod]
        public void Flip_3D_AxisTuple01()
        {
            // np.flip(a, (0, 1)) => [[[6 7],[4 5]],[[2 3],[0 1]]]
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, new[] {0, 1});

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(6, 7, 4, 5, 2, 3, 0, 1);
        }

        [TestMethod]
        public void Flip_3D_AxisTupleNegative()
        {
            // np.flip(a, (-2,)) == np.flip(a, 1)
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, new[] {-2});

            result.Should().BeOfValues(2, 3, 0, 1, 6, 7, 4, 5);
        }

        [TestMethod]
        public void Flip_EmptyAxisTuple_FlipsNothing()
        {
            // np.flip(a, ()) => a (an unreversed full view)
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, new int[0]);

            result.Should().BeShaped(2, 2, 2);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7);
        }

        [TestMethod]
        public void Flip_NullAxisArray_FlipsAllAxes()
        {
            // C# null int[] maps to NumPy axis=None: flip over all axes.
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(a, (int[])null);

            result.Should().BeOfValues(7, 6, 5, 4, 3, 2, 1, 0);
        }

        [TestMethod]
        public void Flip_2x3x4_AllAxes()
        {
            // np.flip(np.arange(24).reshape(2,3,4)).ravel() => [23 22 ... 1 0]
            var B = np.arange(24).reshape(2, 3, 4);
            var result = np.flip(B);

            result.Should().BeShaped(2, 3, 4);
            result.Should().BeOfValues(23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                                       11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        }

        [TestMethod]
        public void Flip_2x3x4_AxisTuple02()
        {
            // np.flip(np.arange(24).reshape(2,3,4), (0,2)).ravel()
            //   => [15 14 13 12 19 18 17 16 23 22 21 20 3 2 1 0 7 6 5 4 11 10 9 8]
            var B = np.arange(24).reshape(2, 3, 4);
            var result = np.flip(B, new[] {0, 2});

            result.Should().BeOfValues(15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
                                       3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8);
        }

        // ================================================================
        // VIEW SEMANTICS
        // ================================================================

        [TestMethod]
        public void Flip_ReturnsView_WriteThrough()
        {
            // f = np.flip(x); f[0] = 99  =>  x[-1] == 99 (flip returns a view, not a copy)
            var x = np.arange(10);
            var f = np.flip(x);

            f["0"] = (NDArray)99;

            x.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 99);
        }

        [TestMethod]
        public void Flip_OfSlicedView_WritesThroughToRoot()
        {
            // x = np.arange(10); fv = np.flip(x[2:8]); fv[0] = 99  =>  x[7] == 99
            var x = np.arange(10);
            var fv = np.flip(x["2:8"]);

            fv.Should().BeOfValues(7, 6, 5, 4, 3, 2);

            fv["0"] = (NDArray)99;
            x.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 99, 8, 9);
        }

        [TestMethod]
        public void Flip_DoubleFlip_RestoresOrder()
        {
            // np.flip(np.flip(a, 0), 0) == a
            var a = np.arange(8).reshape(2, 2, 2);
            var result = np.flip(np.flip(a, 0), 0);

            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7);
        }

        [TestMethod]
        public void Flip_Broadcast_PreservesReadonly()
        {
            // b = np.broadcast_to(np.arange(3), (2,3)); np.flip(b, 1) => [[2 1 0],[2 1 0]], WRITEABLE=False
            var b = np.broadcast_to(np.arange(3), new Shape(2, 3));
            var result = np.flip(b, 1);

            result.Should().BeShaped(2, 3);
            result.Should().BeOfValues(2, 1, 0, 2, 1, 0);
            result.Shape.IsWriteable.Should().BeFalse("flip of a broadcast view stays read-only, matching NumPy");
        }

        // ================================================================
        // NON-CONTIGUOUS INPUTS
        // ================================================================

        [TestMethod]
        public void Flip_TransposedInput()
        {
            // t = np.arange(12).reshape(3,4).T; np.flip(t, 0) => [[3 7 11],[2 6 10],[1 5 9],[0 4 8]]
            var t = np.arange(12).reshape(3, 4).T;
            var result = np.flip(t, 0);

            result.Should().BeShaped(4, 3);
            result.Should().BeOfValues(3, 7, 11, 2, 6, 10, 1, 5, 9, 0, 4, 8);
        }

        [TestMethod]
        public void Flip_SteppedSliceInput()
        {
            // np.flip(np.arange(10)[1:9:2]) => [7 5 3 1]
            var sl = np.arange(10)["1:9:2"];
            var result = np.flip(sl);

            result.Should().BeOfValues(7, 5, 3, 1);
        }

        // ================================================================
        // 0-D / EMPTY / DEGENERATE
        // ================================================================

        [TestMethod]
        public void Flip_0d_ReturnsScalar()
        {
            // np.flip(np.array(5)) => np.int64(5) (0-d in, 0-d out)
            var s = NDArray.Scalar(5);
            var result = np.flip(s);

            result.ndim.Should().Be(0);
            ((int)result).Should().Be(5);
        }

        [TestMethod]
        public void Flip_0d_EmptyAxisTuple_ReturnsScalar()
        {
            // np.flip(np.array(5), ()) => np.int64(5)
            var s = NDArray.Scalar(5);
            var result = np.flip(s, new int[0]);

            result.ndim.Should().Be(0);
            ((int)result).Should().Be(5);
        }

        [TestMethod]
        public void Flip_Empty()
        {
            // np.flip(np.zeros((0,3))).shape => (0, 3), also per-axis
            var e = np.zeros(new Shape(0, 3));

            np.flip(e).Should().BeShaped(0, 3);
            np.flip(e, 0).Should().BeShaped(0, 3);
            np.flip(e, 1).Should().BeShaped(0, 3);
        }

        [TestMethod]
        public void Flip_OneElement()
        {
            // np.flip(np.array([7])) => [7]
            var result = np.flip(np.array(new[] {7}));

            result.Should().BeShaped(1);
            result.Should().BeOfValues(7);
        }

        // ================================================================
        // ERRORS (types and messages match NumPy 2.4.2 verbatim)
        // ================================================================

        [TestMethod]
        public void Flip_0d_WithAxis_Throws()
        {
            // np.flip(np.array(5), 0) => AxisError: axis 0 is out of bounds for array of dimension 0
            var s = NDArray.Scalar(5);

            new Action(() => np.flip(s, 0)).Should().Throw<AxisError>()
                .WithMessage("*axis 0 is out of bounds for array of dimension 0*");
        }

        [TestMethod]
        public void Flip_AxisOutOfBounds_Throws()
        {
            // np.flip(a, 3) => AxisError: axis 3 is out of bounds for array of dimension 3
            var a = np.arange(8).reshape(2, 2, 2);

            new Action(() => np.flip(a, 3)).Should().Throw<AxisError>()
                .WithMessage("*axis 3 is out of bounds for array of dimension 3*");
        }

        [TestMethod]
        public void Flip_NegativeAxisOutOfBounds_ThrowsWithOriginalAxis()
        {
            // np.flip(a, -4) => AxisError: axis -4 is out of bounds for array of dimension 3
            // (message reports the ORIGINAL negative axis, not the normalized one)
            var a = np.arange(8).reshape(2, 2, 2);

            new Action(() => np.flip(a, -4)).Should().Throw<AxisError>()
                .WithMessage("*axis -4 is out of bounds for array of dimension 3*");
        }

        [TestMethod]
        public void Flip_RepeatedAxis_Throws()
        {
            // np.flip(a, (0, 0)) => ValueError: repeated axis
            var a = np.arange(8).reshape(2, 2, 2);

            new Action(() => np.flip(a, new[] {0, 0})).Should().Throw<ValueError>()
                .WithMessage("*repeated axis*");
        }

        [TestMethod]
        public void Flip_RepeatedAxis_AfterNormalization_Throws()
        {
            // np.flip(a, (0, -3)) => ValueError: repeated axis (-3 normalizes to 0)
            var a = np.arange(8).reshape(2, 2, 2);

            new Action(() => np.flip(a, new[] {0, -3})).Should().Throw<ValueError>()
                .WithMessage("*repeated axis*");
        }

        [TestMethod]
        public void Flip_OutOfBoundsBeatsRepeated()
        {
            // np.flip(a, (0, 0, 5)) => AxisError (normalize_axis_tuple normalizes the WHOLE
            // tuple before the duplicate check, so out-of-bounds wins even after a duplicate)
            var a = np.arange(8).reshape(2, 2, 2);

            new Action(() => np.flip(a, new[] {0, 0, 5})).Should().Throw<AxisError>()
                .WithMessage("*axis 5 is out of bounds for array of dimension 3*");
        }

        [TestMethod]
        public void Flip_TupleAxisOutOfBounds_Throws()
        {
            // np.flip(a, (1, 5)) => AxisError: axis 5 is out of bounds for array of dimension 3
            var a = np.arange(8).reshape(2, 2, 2);

            new Action(() => np.flip(a, new[] {1, 5})).Should().Throw<AxisError>()
                .WithMessage("*axis 5 is out of bounds for array of dimension 3*");
        }

        // ================================================================
        // DTYPE COVERAGE (flip is a pure view op — dtype always preserved)
        // ================================================================

        [TestMethod]
        public void Flip_AllDtypes_PreservedAndReversed()
        {
            foreach (var tc in new[]
                     {
                         NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
                         NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
                         NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
                         NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
                     })
            {
                var src = np.arange(6).astype(tc).reshape(2, 3);
                var flipped = np.flip(src);

                flipped.typecode.Should().Be(tc, $"flip must preserve dtype {tc}");
                flipped.Should().BeShaped(2, 3);

                // first element of the flip == last element of the source
                flipped.flat.Cast<object>().First().Should().Be(src.flat.Cast<object>().Last(), $"dtype {tc}");

                // double flip restores the source bit-exactly
                np.array_equal(np.flip(flipped), src).Should().BeTrue($"double flip must round-trip for dtype {tc}");
            }
        }

        [TestMethod]
        public void Flip_Boolean_Values()
        {
            // z = np.array([[True,False,False],[False,True,True]])
            // np.flip(z)    => [[True True False],[False False True]]
            // np.flip(z, 0) => [[False True True],[True False False]]
            var z = np.array(new bool[,] {{true, false, false}, {false, true, true}});

            np.flip(z).Should().BeOfValues(true, true, false, false, false, true);
            np.flip(z, 0).Should().BeOfValues(false, true, true, true, false, false);
        }

        // ================================================================
        // np.fliplr
        // ================================================================

        [TestMethod]
        public void Fliplr_2D()
        {
            // np.fliplr(np.diag([1.,2.,3.])) => [[0 0 1],[0 2 0],[3 0 0]]
            var A = np.array(new double[,] {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
            var result = np.fliplr(A);

            result.Should().BeShaped(3, 3);
            result.Should().BeOfValues(0, 0, 1, 0, 2, 0, 3, 0, 0);
        }

        [TestMethod]
        public void Fliplr_3D()
        {
            // np.fliplr(np.arange(24).reshape(2,3,4)).ravel()
            //   => [8 9 10 11 4 5 6 7 0 1 2 3 20 21 22 23 16 17 18 19 12 13 14 15]
            var B = np.arange(24).reshape(2, 3, 4);
            var result = np.fliplr(B);

            result.Should().BeShaped(2, 3, 4);
            result.Should().BeOfValues(8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                                       20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15);
        }

        [TestMethod]
        public void Fliplr_EqualsFlipAxis1()
        {
            // np.fliplr(m) == np.flip(m, 1)
            var B = np.arange(24).reshape(2, 3, 4);

            np.array_equal(np.fliplr(B), np.flip(B, 1)).Should().BeTrue();
        }

        [TestMethod]
        public void Fliplr_IsView()
        {
            var A = np.arange(4).reshape(2, 2);
            var f = np.fliplr(A);

            f["0, 0"] = (NDArray)99; // f[0,0] aliases A[0,1]

            A.Should().BeOfValues(0, 99, 2, 3);
        }

        [TestMethod]
        public void Fliplr_Empty()
        {
            // np.fliplr(np.zeros((0,3))).shape => (0,3); np.fliplr(np.zeros((2,0))).shape => (2,0)
            np.fliplr(np.zeros(new Shape(0, 3))).Should().BeShaped(0, 3);
            np.fliplr(np.zeros(new Shape(2, 0))).Should().BeShaped(2, 0);
        }

        [TestMethod]
        public void Fliplr_1D_Throws()
        {
            // np.fliplr(np.array([1,2])) => ValueError: Input must be >= 2-d.
            new Action(() => np.fliplr(np.array(new[] {1, 2}))).Should().Throw<ValueError>()
                .WithMessage("*Input must be >= 2-d.*");
        }

        [TestMethod]
        public void Fliplr_0d_Throws()
        {
            // np.fliplr(np.array(5)) => ValueError: Input must be >= 2-d.
            new Action(() => np.fliplr(NDArray.Scalar(5))).Should().Throw<ValueError>()
                .WithMessage("*Input must be >= 2-d.*");
        }

        // ================================================================
        // np.flipud
        // ================================================================

        [TestMethod]
        public void Flipud_2D()
        {
            // np.flipud(np.diag([1.,2.,3.])) => [[0 0 3],[0 2 0],[1 0 0]]
            var A = np.array(new double[,] {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
            var result = np.flipud(A);

            result.Should().BeShaped(3, 3);
            result.Should().BeOfValues(0, 0, 3, 0, 2, 0, 1, 0, 0);
        }

        [TestMethod]
        public void Flipud_1D()
        {
            // np.flipud([1,2]) => [2 1]
            var result = np.flipud(np.array(new[] {1, 2}));

            result.Should().BeOfValues(2, 1);
        }

        [TestMethod]
        public void Flipud_3D()
        {
            // np.flipud(np.arange(24).reshape(2,3,4)).ravel() => [12..23, 0..11]
            var B = np.arange(24).reshape(2, 3, 4);
            var result = np.flipud(B);

            result.Should().BeShaped(2, 3, 4);
            result.Should().BeOfValues(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Flipud_EqualsFlipAxis0()
        {
            // np.flipud(m) == np.flip(m, 0)
            var B = np.arange(24).reshape(2, 3, 4);

            np.array_equal(np.flipud(B), np.flip(B, 0)).Should().BeTrue();
        }

        [TestMethod]
        public void Flipud_IsView()
        {
            var A = np.arange(4).reshape(2, 2);
            var f = np.flipud(A);

            f["0, 0"] = (NDArray)99; // f[0,0] aliases A[1,0]

            A.Should().BeOfValues(0, 1, 99, 3);
        }

        [TestMethod]
        public void Flipud_Empty()
        {
            // np.flipud(np.zeros((0,3))).shape => (0,3)
            np.flipud(np.zeros(new Shape(0, 3))).Should().BeShaped(0, 3);
        }

        [TestMethod]
        public void Flipud_0d_Throws()
        {
            // np.flipud(np.array(5)) => ValueError: Input must be >= 1-d.
            new Action(() => np.flipud(NDArray.Scalar(5))).Should().Throw<ValueError>()
                .WithMessage("*Input must be >= 1-d.*");
        }
    }
}
