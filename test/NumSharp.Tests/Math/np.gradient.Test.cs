using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.gradient — second-order central differences in the interior, first/second-order one-sided
    ///     at the edges. All expected values are from NumPy 2.4.2 (probed directly). gradient is a pure
    ///     composition of slicing + arithmetic (bit-exact with NumPy for finite data). Covers the
    ///     docstring examples, spacing (scalar / coordinate), edge_order, the dtype tier, the
    ///     tuple-or-single return (GradientResult), and the full error taxonomy.
    /// </summary>
    [TestClass]
    public class NpGradientTest
    {
        private static void AssertClose(NDArray got, double[] expected, double tol = 1e-12)
        {
            got.size.Should().Be(expected.Length);
            var g = got.astype(np.float64).Data<double>().ToArray();
            for (int i = 0; i < expected.Length; i++)
                g[i].Should().BeApproximately(expected[i], tol * (1.0 + System.Math.Abs(expected[i])), $"element {i}");
        }

        // ---------------------------- docstring examples ----------------------------

        [TestMethod]
        public void Gradient_1D_Unit()
        {
            // np.gradient([1,2,4,7,11,16]) -> [1., 1.5, 2.5, 3.5, 4.5, 5.]
            NDArray r = np.gradient(np.array(new long[] { 1, 2, 4, 7, 11, 16 }));
            r.dtype.Should().Be(np.float64);
            AssertClose(r, new[] { 1.0, 1.5, 2.5, 3.5, 4.5, 5.0 });
        }

        [TestMethod]
        public void Gradient_1D_ScalarSpacing()
        {
            // np.gradient([1,2,4,7,11,16], 2) -> [0.5, 0.75, 1.25, 1.75, 2.25, 2.5]
            NDArray r = np.gradient(np.array(new long[] { 1, 2, 4, 7, 11, 16 }), 2);
            AssertClose(r, new[] { 0.5, 0.75, 1.25, 1.75, 2.25, 2.5 });
        }

        [TestMethod]
        public void Gradient_1D_CoordinateArray_NonUniform()
        {
            // np.gradient([1,2,4,7,11,16], x=[0,1,1.5,3.5,4,6]) -> [1., 3., 3.5, 6.7, 6.9, 2.5]
            NDArray r = np.gradient(np.array(new long[] { 1, 2, 4, 7, 11, 16 }),
                                    np.array(new double[] { 0, 1, 1.5, 3.5, 4, 6 }));
            AssertClose(r, new[] { 1.0, 3.0, 3.5, 6.7, 6.9, 2.5 });
        }

        [TestMethod]
        public void Gradient_EdgeOrder2_Exact_On_XSquared()
        {
            // f = x**2 for x=0..4; np.gradient(f, edge_order=2) -> [0., 2., 4., 6., 8.] (exact)
            var f = np.power(np.arange(5).astype(np.float64), 2);
            NDArray r = np.gradient(f, edge_order: 2);
            AssertClose(r, new[] { 0.0, 2.0, 4.0, 6.0, 8.0 });
        }

        // ---------------------------- dtype tier ----------------------------

        [TestMethod]
        public void Gradient_Integer_ToFloat64()
        {
            NDArray r = np.gradient(np.array(new int[] { 1, 2, 4, 7, 11, 16 }));
            r.dtype.Should().Be(np.float64);
            AssertClose(r, new[] { 1.0, 1.5, 2.5, 3.5, 4.5, 5.0 });
        }

        [TestMethod]
        public void Gradient_Float32_StaysFloat32()
        {
            NDArray r = np.gradient(np.array(new float[] { 1, 2, 4, 7, 11, 16 }));
            r.dtype.Should().Be(np.float32);
            AssertClose(r, new[] { 1.0, 1.5, 2.5, 3.5, 4.5, 5.0 }, 1e-6);
        }

        [TestMethod]
        public void Gradient_Complex_StaysComplex128()
        {
            NDArray r = np.gradient(np.array(new System.Numerics.Complex[]
                { new(1, 0), new(2, 1), new(4, 3) }));
            r.dtype.Should().Be(np.complex128);
        }

        // ---------------------------- multi-axis (tuple return) ----------------------------

        [TestMethod]
        public void Gradient_2D_Default_ReturnsTuple()
        {
            // np.gradient([[1,2,6],[3,4,5]]) -> (d/axis0, d/axis1)
            var a = np.array(new double[,] { { 1, 2, 6 }, { 3, 4, 5 } });
            var g = np.gradient(a);
            g.IsSingle.Should().BeFalse();
            g.Length.Should().Be(2);
            AssertClose(g[0], new[] { 2.0, 2.0, -1.0, 2.0, 2.0, -1.0 });   // axis-0 component (2x3)
            AssertClose(g[1], new[] { 1.0, 2.5, 4.0, 1.0, 1.0, 1.0 });     // axis-1 component (2x3)
        }

        [TestMethod]
        public void Gradient_2D_SingleAxis_ReturnsBareArray()
        {
            var a = np.array(new double[,] { { 1, 2, 6 }, { 3, 4, 5 } });
            var g = np.gradient(a, axis: 0);
            g.IsSingle.Should().BeTrue();
            NDArray single = g;                 // implicit conversion to bare NDArray
            single.shape.Should().Equal(2, 3);
            AssertClose(single, new[] { 2.0, 2.0, -1.0, 2.0, 2.0, -1.0 });
        }

        [TestMethod]
        public void Gradient_Deconstruct()
        {
            var a = np.array(new double[,] { { 1, 2, 6 }, { 3, 4, 5 } });
            var (gy, gx) = np.gradient(a);
            AssertClose(gy, new[] { 2.0, 2.0, -1.0, 2.0, 2.0, -1.0 });
            AssertClose(gx, new[] { 1.0, 2.5, 4.0, 1.0, 1.0, 1.0 });
        }

        [TestMethod]
        public void Gradient_TupleAxis_ExplicitArrayForm()
        {
            // axis=(0,1) with no spacing needs the explicit-array form.
            var a = np.array(new double[,] { { 1, 2, 6 }, { 3, 4, 5 } });
            NDArray[] g = np.gradient(a, System.Array.Empty<object>(), axis: new[] { 0, 1 });
            g.Length.Should().Be(2);
            AssertClose(g[0], new[] { 2.0, 2.0, -1.0, 2.0, 2.0, -1.0 });
        }

        [TestMethod]
        public void Gradient_2D_PerAxisScalarSpacing()
        {
            // np.gradient([[1,2,6],[3,4,5]], 2., 3.) — dy=2 (axis0), dx=3 (axis1)
            var a = np.array(new double[,] { { 1, 2, 6 }, { 3, 4, 5 } });
            var g = np.gradient(a, 2.0, 3.0);
            g.Length.Should().Be(2);
            AssertClose(g[0], new[] { 1.0, 1.0, -0.5, 1.0, 1.0, -0.5 });
            AssertClose(g[1], new[] { 1.0 / 3, 2.5 / 3, 4.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3 });
        }

        // ---------------------------- 0-d -> empty tuple ----------------------------

        [TestMethod]
        public void Gradient_0D_IsEmptyTuple()
        {
            var g = np.gradient(np.array(5.0));   // NumPy returns an empty tuple
            g.Length.Should().Be(0);
        }

        // ---------------------------- errors ----------------------------

        [TestMethod]
        public void Gradient_Bool_Throws()
        {
            // NumPy: TypeError "numpy boolean subtract ..."; NumSharp: NotSupportedException, same text.
            Action act = () => np.gradient(np.array(new bool[] { true, false, true, false }));
            act.Should().Throw<NotSupportedException>().WithMessage("*boolean subtract*");
        }

        [TestMethod]
        public void Gradient_TooSmall_Throws()
        {
            Action act = () => np.gradient(np.array(new double[] { 5 }));
            act.Should().Throw<ValueError>().WithMessage("*too small*");
        }

        [TestMethod]
        public void Gradient_EdgeOrder3_Throws()
        {
            Action act = () => np.gradient(np.arange(5).astype(np.float64), edge_order: 3);
            act.Should().Throw<ValueError>().WithMessage("*edge_order*greater than 2*");
        }

        [TestMethod]
        public void Gradient_BadVarargsCount_Throws()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            Action act = () => np.gradient(a, 1.0, 2.0, 3.0);   // 3 spacings for a 2-D array
            act.Should().Throw<TypeError>().WithMessage("*invalid number of arguments*");
        }

        [TestMethod]
        public void Gradient_AxisOutOfBounds_Throws()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            Action act = () => np.gradient(a, axis: 5);
            act.Should().Throw<ArgumentOutOfRangeException>();
        }

        [TestMethod]
        public void Gradient_CoordLengthMismatch_Throws()
        {
            Action act = () => np.gradient(np.array(new double[] { 1, 2, 3 }),
                                           np.array(new double[] { 0, 1 }));
            act.Should().Throw<ValueError>().WithMessage("*distances must match*");
        }

        [TestMethod]
        public void Gradient_RepeatedAxis_Throws()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            Action act = () => np.gradient(a, System.Array.Empty<object>(), axis: new[] { 0, 0 });
            act.Should().Throw<ValueError>().WithMessage("*repeated axis*");
        }
    }
}
