using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     Tests for <see cref="np.apply_along_axis"/>, verified against NumPy 2.4.2 output.
    ///     Covers the axis variants, the three result-rank cases (scalar / 1-D / higher-D func),
    ///     dtype preservation, non-contiguous inputs, the *args overload, and the error parity
    ///     (empty iteration dimension, out-of-bounds axis, null func).
    ///
    ///     NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
    ///     NumPy source: numpy/lib/_shape_base_impl.py
    /// </summary>
    [TestClass]
    public class np_apply_along_axis_Test
    {
        // averaging func used in NumPy's own docstring examples: (a[0] + a[-1]) * 0.5
        private static NDArray Avg(NDArray a) => (a[0] + a["-1"]) * 0.5;

        // baseline 3x3 int matrix
        private static NDArray B() => np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        [TestMethod]
        public void Axis0_ColumnSlices()
        {
            // np.apply_along_axis(avg, 0, b) -> [4. 5. 6.]  (columns [1,4,7],[2,5,8],[3,6,9])
            var r = np.apply_along_axis(Avg, 0, B());
            r.Should().BeShaped(3);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.Should().BeOfValues(4.0, 5.0, 6.0);
        }

        [TestMethod]
        public void Axis1_RowSlices()
        {
            // np.apply_along_axis(avg, 1, b) -> [2. 5. 8.]
            np.apply_along_axis(Avg, 1, B()).Should().BeOfValues(2.0, 5.0, 8.0);
        }

        [TestMethod]
        public void NegativeAxis_SameAsLast()
        {
            np.apply_along_axis(Avg, -1, B()).Should().BeOfValues(2.0, 5.0, 8.0);
        }

        [TestMethod]
        public void ScalarFunc_DropsAxis_WidensToInt64()
        {
            // reduction to a scalar per slice; sum widens int32 -> int64 (NEP50)
            var r = np.apply_along_axis(a => a.sum(), 1, B());
            r.Should().BeShaped(3);
            r.typecode.Should().Be(NPTypeCode.Int64);
            r.Should().BeOfValues(6L, 15L, 24L);
        }

        [TestMethod]
        public void OneDimensionalInput_ProducesScalarResult()
        {
            // 1-D input, scalar func -> 0-d result (NumPy's trailing-ellipsis prevents earlier decay)
            var r = np.apply_along_axis(Avg, 0, np.array(new int[] { 1, 2, 3, 4 }));
            r.ndim.Should().Be(0);
            r.GetDouble(0).Should().Be(2.5);
        }

        [TestMethod]
        public void FuncReturns1D_KeepsRank()
        {
            // sort each row -> same shape, dtype preserved
            var c = np.array(new int[,] { { 8, 1, 7 }, { 4, 3, 9 }, { 5, 2, 6 } });
            var r = np.apply_along_axis(a => np.sort(a), 1, c);
            r.Should().BeShaped(3, 3);
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.Should().BeOfValues(1, 7, 8, 3, 4, 9, 2, 5, 6);
        }

        [TestMethod]
        public void FuncReturnsHigherDim_InsertsDimensions()
        {
            // np.diag over the last axis: (3,3) -> (3,3,3), the new dims replace `axis`
            var r = np.apply_along_axis(a => np.diag(a), -1, B());
            r.Should().BeShaped(3, 3, 3);
            r.Should().BeOfValues(1, 0, 0, 0, 2, 0, 0, 0, 3,
                                  4, 0, 0, 0, 5, 0, 0, 0, 6,
                                  7, 0, 0, 0, 8, 0, 0, 0, 9);
        }

        [TestMethod]
        public void ThreeDimensional_ScalarFunc()
        {
            // sum along axis 1 of arange(24).reshape(2,3,4) -> (2,4)
            var a3 = np.arange(24).reshape(2, 3, 4);
            var r = np.apply_along_axis(a => a.sum(), 1, a3);
            r.Should().BeShaped(2, 4);
            r.Should().BeOfValues(12L, 15L, 18L, 21L, 48L, 51L, 54L, 57L);
        }

        [TestMethod]
        public void ZeroLengthAxis_EmptySlicesReduce()
        {
            // M=0 ALONG the axis (iteration dims nonzero) is legal: sum of empty slice -> 0
            var r = np.apply_along_axis(a => a.sum(), 1, np.zeros(new Shape(3, 0), np.float64));
            r.Should().BeShaped(3);
            r.Should().BeOfValues(0.0, 0.0, 0.0);
        }

        [TestMethod]
        public void NonContiguousInput_TransposedReadCorrectly()
        {
            // sum along axis 0 of b.T (a transposed, non-contiguous view)
            var b = np.arange(6).reshape(2, 3);   // [[0,1,2],[3,4,5]]
            var r = np.apply_along_axis(a => a.sum(), 0, b.T);   // b.T is (3,2): rows [0,3],[1,4],[2,5]
            r.Should().BeShaped(2);
            r.Should().BeOfValues(3L, 12L);   // [0+1+2, 3+4+5]
        }

        [TestMethod]
        public void ArgsOverload_ForwardsExtraArguments()
        {
            // np.apply_along_axis(f, 1, arr, 10) with f(slice, args) -> sum*10
            var r = np.apply_along_axis((a, args) => a.sum() * (int)args[0], 1,
                                        np.array(new int[,] { { 1, 2 }, { 3, 4 } }), 10);
            r.Should().BeOfValues(30L, 70L);
        }

        [TestMethod]
        public void AllDtypes_PreserveViaFlip()
        {
            // apply_along_axis is dtype-agnostic (moves bytes): a dtype-preserving func keeps dtype
            // across all 15 supported types.
            DType[] dts =
            {
                np.bool8, np.uint8, np.int8, np.int16, np.uint16, np.int32, np.uint32,
                np.int64, np.uint64, np.@char, np.float16, np.float32, np.float64,
                np.@decimal, np.complex128
            };
            foreach (var dt in dts)
            {
                var a = np.arange(12).reshape(3, 4).astype(dt);
                var r = np.apply_along_axis(s => np.flip(s), 1, a);
                r.Should().BeShaped(3, 4);
                r.dtype.Should().Be(dt);
                // first row 0,1,2,3 reversed -> 3,2,1,0 (bool: [F,T,T,T] -> [T,T,T,F]).
                // Convert to float64 and flatten to read VALUES by flat index (GetDouble reinterprets
                // raw bytes for a non-double dtype, and indexes the first axis on a 2-D array).
                var f = r.astype(np.float64).flatten();
                f.GetDouble(0).Should().Be(dt == np.bool8 ? 1.0 : 3.0);
                f.GetDouble(3).Should().Be(0.0);
            }
        }

        // ================================================================
        //  Error parity
        // ================================================================

        [TestMethod]
        public void EmptyIterationDimension_Throws()
        {
            // A zero-length NON-axis dimension leaves no first slice to fix the result shape.
            ((Action)(() => np.apply_along_axis(a => a.sum(), 1, np.zeros(new Shape(0, 3), np.float64))))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("Cannot apply_along_axis when any iteration dimensions are 0");
        }

        [TestMethod]
        public void OutOfBoundsAxis_ThrowsAxisError()
        {
            // AxisError reports the ORIGINAL axis; .NET appends the parameter suffix (asserted with *).
            ((Action)(() => np.apply_along_axis(a => a, 5, np.zeros(new Shape(2, 3), np.float64))))
                .Should().ThrowExactly<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 2*");
        }

        [TestMethod]
        public void NullFunc_Throws()
        {
            ((Action)(() => np.apply_along_axis((Func<NDArray, NDArray>)null, 0, B())))
                .Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void NullArray_Throws()
        {
            ((Action)(() => np.apply_along_axis(a => a, 0, (NDArray)null)))
                .Should().Throw<ArgumentNullException>();
        }
    }
}
