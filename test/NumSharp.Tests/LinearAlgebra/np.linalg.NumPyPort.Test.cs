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
    }
}
