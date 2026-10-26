using System;
using System.Linq;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     np.linalg held to NumPy 2.4.2's own statements (<c>numpy/linalg/_linalg.py</c> and the <c>_umath_linalg</c>
    ///     gufuncs): the objects it returns, its dtypes and its check order. Every expected value, flag and text was probed
    ///     against NumPy 2.4.2; the comments give the NumPy call. The tests run without a matrix backend, so they use the
    ///     functions the managed LU computes (det, slogdet, solve, inv) and the empty operands every factorisation answers
    ///     without arithmetic.
    /// </summary>
    [TestClass]
    public class NpLinalgNumPyPortTest
    {
        /// <summary>The (3, 3) matrix of the snapshot tool's <c>sq3</c> layout; its determinant is 48.</summary>
        private static NDArray Sq3() => np.array(new[] { 3.0, -1, 2, 0, 5, -4, 1, 7, -2 }).reshape(3, 3);

        /// <summary>
        ///     Asserts NumPy's scalar flags: a 0-d result that is C and F contiguous, owns its data and is read-only, as
        ///     NumPy's scalar reports them.
        /// </summary>
        /// <param name="label">The NumPy call, for the failure message.</param>
        /// <param name="r">The NumSharp result.</param>
        private static void AssertScalar(string label, NDArray r)
        {
            Assert.AreEqual(0, r.ndim, label + " ndim");
            Assert.IsTrue(r.flags.c_contiguous, label + " C_CONTIGUOUS");
            Assert.IsTrue(r.flags.f_contiguous, label + " F_CONTIGUOUS");
            Assert.IsTrue(r.flags.owndata, label + " OWNDATA");
            Assert.IsFalse(r.flags.writeable, label + " WRITEABLE");
        }

        /// <summary>
        ///     A gufunc's 0-d output is NumPy's scalar: <c>_umath_linalg.det</c> / <c>slogdet</c> of one matrix and
        ///     <c>lstsq</c>'s rank come back through the ufunc machinery's return step, which turns a 0-d output into a
        ///     scalar (C, F, OWNDATA, read-only), while a stack's determinants stay a writeable array. NumSharp returned
        ///     writeable 0-d arrays.
        /// </summary>
        [TestMethod]
        public void GufuncScalarOutputs_AreNumPysScalars()
        {
            // np.linalg.det(a): np.float64(48.00000000000001), flags C F OWNDATA, not WRITEABLE.
            var det = np.linalg.det(Sq3());
            AssertScalar("det(a)", det);
            Assert.AreEqual(48.0, det.GetDouble(), 1e-12);

            // np.linalg.det(np.zeros((0, 0))): np.float64(1.0), a scalar.
            var detEmpty = np.linalg.det(np.zeros(new Shape(0, 0)));
            AssertScalar("det(zeros((0, 0)))", detEmpty);
            Assert.AreEqual(1.0, detEmpty.GetDouble());

            // np.linalg.det(np.stack([a, a])): a (2,) array that owns its data and is writeable.
            var detStack = np.linalg.det(np.stack(new[] { Sq3(), Sq3() }));
            CollectionAssert.AreEqual(new long[] { 2 }, detStack.shape.Select(d => (long)d).ToArray());
            Assert.IsTrue(detStack.flags.owndata, "det(stack) OWNDATA");
            Assert.IsTrue(detStack.flags.writeable, "det(stack) WRITEABLE");

            // np.linalg.slogdet(a): (np.float64(1.0), np.float64(3.871201010907891)), both scalars.
            var (sign, logabsdet) = np.linalg.slogdet(Sq3());
            AssertScalar("slogdet(a).sign", sign);
            AssertScalar("slogdet(a).logabsdet", logabsdet);
            Assert.AreEqual(1.0, sign.GetDouble());
            Assert.AreEqual(System.Math.Log(48.0), logabsdet.GetDouble(), 1e-12);

            // np.linalg.slogdet(np.zeros((0, 0))): (1.0, 0.0), both scalars.
            var (signEmpty, logEmpty) = np.linalg.slogdet(np.zeros(new Shape(0, 0)));
            AssertScalar("slogdet(zeros((0, 0))).sign", signEmpty);
            AssertScalar("slogdet(zeros((0, 0))).logabsdet", logEmpty);

            // np.linalg.lstsq(np.zeros((0, 3)), np.zeros(0)): the rank is np.intc(0), a scalar; the solution, the
            // residuals and the singular values are writeable arrays.
            var (x, residuals, rank, s) = np.linalg.lstsq(np.zeros(new Shape(0, 3)), np.zeros(new Shape(0)));
            AssertScalar("lstsq(zeros((0, 3)), zeros(0)).rank", rank);
            Assert.AreEqual(np.int32, rank.dtype);
            Assert.AreEqual(0, rank.GetInt32());
            Assert.IsTrue(x.flags.writeable && residuals.flags.writeable && s.flags.writeable, "lstsq arrays WRITEABLE");
        }

        /// <summary>
        ///     norm's one-axis branch never normalizes the axis: a string order is refused ("Invalid norm order ... for
        ///     vectors") before any reduction would check the axis, while a numeric order's reduction raises NumPy's
        ///     AxisError. NumSharp normalized the axis first, so a string order with an out-of-range axis (a 0-d input's
        ///     axis 0 included) raised the AxisError.
        /// </summary>
        [TestMethod]
        public void VectorNorm_RefusesAStringOrderBeforeTheAxis()
        {
            var a = np.arange(9.0).reshape(3, 3);

            // np.linalg.norm(a, 'fro', axis=2): ValueError: Invalid norm order 'fro' for vectors.
            Action fro = () => np.linalg.norm(a, "fro", 2);
            fro.Should().Throw<ValueError>().WithMessage("Invalid norm order 'fro' for vectors");
            // np.linalg.norm(a, 'bogus', axis=-6): the same refusal of the name.
            Action bogus = () => np.linalg.norm(a, "bogus", -6);
            bogus.Should().Throw<ValueError>().WithMessage("Invalid norm order 'bogus' for vectors");
            // np.linalg.vector_norm(np.array(5.0), axis=0, ord='nuc'): a 0-d input has no axis 0, the name is refused first.
            Action nuc = () => np.linalg.vector_norm(np.array(5.0), 0, ord: "nuc");
            nuc.Should().Throw<ValueError>().WithMessage("Invalid norm order 'nuc' for vectors");

            // A numeric order reaches its reduction, which checks the axis: np.linalg.norm(a, 1, axis=2) and
            // np.linalg.vector_norm(np.array(5.0), axis=0) are AxisErrors.
            Action one = () => np.linalg.norm(a, 1, 2);
            one.Should().Throw<AxisError>().WithMessage("axis 2 is out of bounds for array of dimension 2*");
            Action zeroD = () => np.linalg.vector_norm(np.array(5.0), 0);
            zeroD.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        /// <summary>
        ///     Asserts a keepdims norm result is NumPy's reshape: the shape, a view that does not own its data, writeable,
        ///     C and F contiguous (every probed result has at most one axis longer than 1), and the values.
        /// </summary>
        /// <param name="label">The NumPy call, for the failure message.</param>
        /// <param name="r">The NumSharp result.</param>
        /// <param name="shape">NumPy's <c>r.shape</c>.</param>
        /// <param name="values">NumPy's values, in C order.</param>
        private static void AssertReshapedView(string label, NDArray r, long[] shape, params double[] values)
        {
            CollectionAssert.AreEqual(shape, r.shape.Select(d => (long)d).ToArray(), label + " shape");
            Assert.IsFalse(r.flags.owndata, label + " OWNDATA");
            Assert.IsTrue(r.flags.writeable, label + " WRITEABLE");
            Assert.IsTrue(r.flags.c_contiguous, label + " C_CONTIGUOUS");
            Assert.IsTrue(r.flags.f_contiguous, label + " F_CONTIGUOUS");
            var got = r.flatten().ToArray<double>();
            Assert.AreEqual(values.Length, got.Length, label + " size");
            for (int i = 0; i < values.Length; i++)
                Assert.AreEqual(values[i], got[i], 1e-12, label + " value " + i);
        }

        /// <summary>
        ///     A keepdims norm is NumPy's <c>reshape</c> of the result computed without keepdims: vector_norm always
        ///     computes <c>norm(x, axis=_axis, ord=ord)</c> and ends with <c>res.reshape(tuple(shape))</c>, norm's matrix
        ///     branch with <c>ret.reshape(ret_shape)</c> and its fast path with <c>ret.reshape(ndim * [1])</c>. Each is a
        ///     writeable view that does not own its data, a NumPy scalar's reshape included (a view of the fresh array NumPy
        ///     makes from the scalar). NumSharp passed keepdims into the reductions (owning arrays) or inserted the axes into
        ///     its read-only scalar stand-in (read-only views).
        /// </summary>
        [TestMethod]
        public void KeepdimsNorms_AreNumPysReshapes()
        {
            var a = np.arange(9.0).reshape(3, 3);
            var v = np.arange(3.0);

            // np.linalg.norm(a, 1, keepdims=True): [[15.]], a view of the reduced scalar's array, writeable.
            AssertReshapedView("norm(a, 1, keepdims=True)", np.linalg.norm(a, 1, (int?)null, true), new long[] { 1, 1 }, 15.0);
            // np.linalg.matrix_norm(a, keepdims=True, ord=np.inf): [[21.]].
            AssertReshapedView("matrix_norm(a, keepdims=True, ord=inf)",
                np.linalg.matrix_norm(a, true, double.PositiveInfinity), new long[] { 1, 1 }, 21.0);
            // np.linalg.norm(a, keepdims=True): the fast path, [[14.2828568570857]].
            AssertReshapedView("norm(a, keepdims=True)", np.linalg.norm(a, null, (int?)null, true), new long[] { 1, 1 },
                System.Math.Sqrt(204.0));
            // np.linalg.vector_norm(a, axis=1, keepdims=True): (3, 1), [[2.23606798], [7.07106781], [12.20655562]].
            AssertReshapedView("vector_norm(a, axis=1, keepdims=True)", np.linalg.vector_norm(a, 1, true),
                new long[] { 3, 1 }, System.Math.Sqrt(5.0), System.Math.Sqrt(50.0), System.Math.Sqrt(149.0));
            // np.linalg.vector_norm(v, axis=0, keepdims=True): (1,), [2.23606798].
            AssertReshapedView("vector_norm(v, axis=0, keepdims=True)", np.linalg.vector_norm(v, 0, true),
                new long[] { 1 }, System.Math.Sqrt(5.0));
            // np.linalg.vector_norm(a, keepdims=True): (1, 1), [[14.28285686]].
            AssertReshapedView("vector_norm(a, keepdims=True)", np.linalg.vector_norm(a, (int[])null, true),
                new long[] { 1, 1 }, System.Math.Sqrt(204.0));
            // np.linalg.vector_norm(a, axis=(0, 1), keepdims=True, ord=1): (1, 1), [[36.]].
            AssertReshapedView("vector_norm(a, axis=(0, 1), keepdims=True, ord=1)",
                np.linalg.vector_norm(a, new[] { 0, 1 }, true, 1), new long[] { 1, 1 }, 36.0);
        }
    }
}
