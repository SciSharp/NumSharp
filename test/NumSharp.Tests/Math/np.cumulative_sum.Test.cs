using System;
using System.Linq;
using System.Numerics;
using System.Reflection;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.cumulative_sum</c> / <c>np.cumulative_prod</c>, NumPy 2.1's Array-API cumulative functions (NumPy's
    ///     <c>_cumulative_func</c>): the input is read as at least 1-D, the axis is required from two dimensions on (no
    ///     flattening, unlike cumsum), <c>include_initial</c> prepends the identity (0 / 1) along the axis, and the result
    ///     dtype follows cumsum / cumprod (int32 / bool → int64, uint8 → uint64). Every expected value is NumPy 2.4.2's
    ///     answer for the same call (probed 2026-11-11).
    /// </summary>
    [TestClass]
    public class CumulativeSumProdTests
    {
        /// <summary>The (2, 3) int32 input of the axis cases: <c>[[1, 2, 3], [4, 5, 6]]</c>.</summary>
        private static NDArray X => np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        /// <summary>1-D: <c>[1, 3, 6]</c> int64; with the initial zero <c>[0, 1, 3, 6]</c>; the product's initial one.</summary>
        [TestMethod]
        public void OneD_WithAndWithoutInitial_AsNumPy()
        {
            var x = np.array(new[] { 1, 2, 3 });
            var r = np.cumulative_sum(x);
            r.dtype.Should().Be(np.int64);
            r.ToArray<long>().Should().Equal(1L, 3L, 6L);
            np.cumulative_sum(x, include_initial: true).ToArray<long>().Should().Equal(0L, 1L, 3L, 6L);
            np.cumulative_prod(x, include_initial: true).ToArray<long>().Should().Equal(1L, 1L, 2L, 6L);
        }

        /// <summary>
        ///     2-D needs an axis: without one NumPy raises "For arrays which have more than one dimension ``axis`` argument is
        ///     required."; along axis 1 / -1 / 0 the identity row or column is prepended.
        /// </summary>
        [TestMethod]
        public void TwoD_AxisRequired_InitialAlongTheAxis()
        {
            Action noAxis = () => np.cumulative_sum(X);
            noAxis.Should().Throw<ValueError>()
                  .WithMessage("For arrays which have more than one dimension ``axis`` argument is required.");
            var last = np.cumulative_sum(X, axis: -1, include_initial: true);
            last.shape.Should().Equal(2L, 4L);
            last.flatten().ToArray<long>().Should().Equal(0L, 1L, 3L, 6L, 0L, 4L, 9L, 15L);
            var first = np.cumulative_sum(X, axis: 0, include_initial: true);
            first.shape.Should().Equal(3L, 3L);
            first.flatten().ToArray<long>().Should().Equal(0L, 0L, 0L, 1L, 2L, 3L, 5L, 7L, 9L);
            np.cumulative_prod(X, axis: 0, include_initial: true).flatten().ToArray<long>()
              .Should().Equal(1L, 1L, 1L, 1L, 2L, 3L, 4L, 10L, 18L);
        }

        /// <summary>
        ///     An out-of-range axis is the accumulate's AxisError, except with out= and include_initial together: NumPy
        ///     indexes its slice list with the axis first ("list assignment index out of range", IndexError).
        /// </summary>
        [TestMethod]
        public void AxisOutOfRange_ErrorDependsOnTheRoute()
        {
            Action plain = () => np.cumulative_sum(X, axis: 5);
            plain.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
            Action initial = () => np.cumulative_sum(X, axis: 5, include_initial: true);
            initial.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
            Action withOut = () => np.cumulative_sum(X, axis: 5, @out: np.zeros(new Shape(2, 4)), include_initial: true);
            withOut.Should().Throw<IndexError>().WithMessage("list assignment index out of range");
        }

        /// <summary>
        ///     out= with include_initial receives the sums after the first slot and the identity in it, cast to out's dtype,
        ///     and is returned; without include_initial out= is the accumulate's.
        /// </summary>
        [TestMethod]
        public void Out_WithAndWithoutInitial_IsWrittenAndReturned()
        {
            var o = np.zeros(new Shape(2, 4));
            var r = np.cumulative_sum(X, axis: 1, @out: o, include_initial: true);
            ReferenceEquals(r, o).Should().BeTrue();
            o.flatten().ToArray<double>().Should().Equal(0.0, 1.0, 3.0, 6.0, 0.0, 4.0, 9.0, 15.0);
            var o2 = np.zeros(new Shape(2, 3), np.int64);
            ReferenceEquals(np.cumulative_sum(X, axis: 1, @out: o2), o2).Should().BeTrue();
            o2.flatten().ToArray<long>().Should().Equal(1L, 3L, 6L, 4L, 9L, 15L);
            var o1 = np.zeros(new Shape(4));
            np.cumulative_sum(np.array(new[] { 1.0, 2.0, 3.0 }), @out: o1, include_initial: true);
            o1.ToArray<double>().Should().Equal(0.0, 1.0, 3.0, 6.0);
        }

        /// <summary>The accumulate's dtype rule and dtype=: float32 dtype=, bool dtype=bool (initial False), uint8 →
        ///     uint64, float16 and complex keep their dtype.</summary>
        [TestMethod]
        public void Dtypes_FollowCumsumsRule()
        {
            var f = np.cumulative_sum(np.array(new long[] { 1, 2 }), dtype: np.float32, include_initial: true);
            f.dtype.Should().Be(np.float32);
            f.ToArray<float>().Should().Equal(0f, 1f, 3f);
            var b = np.cumulative_sum(np.array(new[] { true, false }), dtype: np.@bool, include_initial: true);
            b.dtype.Should().Be(np.@bool);
            b.ToArray<bool>().Should().Equal(false, true, true);
            var u = np.cumulative_prod(np.array(new byte[] { 2, 3 }), include_initial: true);
            u.dtype.Should().Be(np.uint64);
            u.ToArray<ulong>().Should().Equal(1UL, 2UL, 6UL);
            var h = np.cumulative_sum(np.array(new[] { (Half)1.5, (Half)2.5 }), include_initial: true);
            h.dtype.Should().Be(np.float16);
            h.ToArray<Half>().Should().Equal((Half)0, (Half)1.5, (Half)4);
            var c = np.cumulative_sum(np.array(new[] { new Complex(1, 1), new Complex(2, 0) }), include_initial: true);
            c.ToArray<Complex>().Should().Equal(Complex.Zero, new Complex(1, 1), new Complex(3, 1));
            np.cumulative_sum(np.array(new sbyte[] { 100, 100 }), dtype: np.int8).ToArray<sbyte>()
              .Should().Equal((sbyte)100, (sbyte)-56);
        }

        /// <summary>
        ///     A 0-d input is read as 1-D (<c>[5]</c>, <c>[0, 5]</c>), axis 0 / -1 are its only axes; empty inputs prepend
        ///     only the identity block (<c>(1, 3)</c> along axis 0, <c>(0, 4)</c> along axis 1).
        /// </summary>
        [TestMethod]
        public void ZeroDAndEmpty_AsNumPy()
        {
            np.cumulative_sum(NDArray.Scalar(5L)).ToArray<long>().Should().Equal(5L);
            np.cumulative_sum(NDArray.Scalar(5L), include_initial: true).ToArray<long>().Should().Equal(0L, 5L);
            np.cumulative_sum(NDArray.Scalar(5L), axis: -1).ToArray<long>().Should().Equal(5L);
            Action axis1 = () => np.cumulative_sum(NDArray.Scalar(5L), axis: 1);
            axis1.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 1*");
            np.cumulative_sum(np.array(new double[0]), include_initial: true).ToArray<double>().Should().Equal(0.0);
            var e0 = np.cumulative_sum(np.zeros(new Shape(0, 3)), axis: 0, include_initial: true);
            e0.shape.Should().Equal(1L, 3L);
            np.cumulative_sum(np.zeros(new Shape(0, 3)), axis: 1, include_initial: true).shape.Should().Equal(0L, 4L);
        }

        /// <summary>Layouts: a 3-D middle axis, a strided view and a transposed view, NumPy's values.</summary>
        [TestMethod]
        public void Layouts_StridedTransposedThreeD_AsNumPy()
        {
            np.cumulative_sum(np.arange(8).reshape(2, 2, 2), axis: 1, include_initial: true).flatten().ToArray<long>()
              .Should().Equal(0L, 0L, 0L, 1L, 2L, 4L, 0L, 0L, 4L, 5L, 10L, 12L);
            np.cumulative_sum(np.arange(10)["::3"], include_initial: true).ToArray<long>().Should().Equal(0L, 0L, 3L, 9L, 18L);
            var t = np.cumulative_sum(np.arange(6).reshape(2, 3).T, axis: 0, include_initial: true);
            t.shape.Should().Equal(4L, 2L);
            t.flatten().ToArray<long>().Should().Equal(0L, 0L, 0L, 3L, 1L, 7L, 3L, 12L);
            np.cumulative_prod(np.array(new[] { 1.0, double.NaN, 2.0 }), include_initial: true).ToArray<double>()
              .Should().Equal(1.0, 1.0, double.NaN, double.NaN);
        }

        /// <summary>
        ///     A matrix operand stays a matrix on both routes (the accumulate's wrap; concat's GetSubType with the matrix's
        ///     priority), NumPy's <c>[[0, 0], [1, 2], [4, 6]]</c> and <c>[[1, 2], [4, 6]]</c>.
        /// </summary>
        [TestMethod]
        public void MatrixOperand_StaysAMatrix()
        {
            var m = np.asmatrix(np.array(new long[,] { { 1, 2 }, { 3, 4 } }));
            NDArray withInitial = np.cumulative_sum(m, axis: 0, include_initial: true);
            withInitial.Should().BeOfType<NDMatrix>();
            withInitial.shape.Should().Equal(3L, 2L);
            withInitial.flatten().ToArray<long>().Should().Equal(0L, 0L, 1L, 2L, 4L, 6L);
            NDArray plain = np.cumulative_sum(m, axis: 0);
            plain.Should().BeOfType<NDMatrix>();
            plain.flatten().ToArray<long>().Should().Equal(1L, 2L, 4L, 6L);
        }

        /// <summary>NumPy's signature: <c>(x, /, *, axis=None, dtype=None, out=None, include_initial=False)</c>, one
        ///     overload each.</summary>
        [TestMethod]
        public void Signatures_AreNumPys()
        {
            foreach (var name in new[] { "cumulative_sum", "cumulative_prod" })
            {
                var forms = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name).ToArray();
                forms.Should().HaveCount(1, name);
                forms[0].GetParameters().Select(p => p.Name).Should().Equal(new[] { "x", "axis", "dtype", "out", "include_initial" }, name);
            }
        }
    }
}
