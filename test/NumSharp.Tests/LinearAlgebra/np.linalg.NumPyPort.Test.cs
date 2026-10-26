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

        /// <summary>
        ///     norm's general vector order is NumPy's four statements, <c>absx = abs(x); absx **= ord; ret =
        ///     add.reduce(absx, axis=axis, keepdims=keepdims); ret **= reciprocal(ord, dtype=ret.dtype)</c>: a float16 /
        ///     float32 operand keeps its dtype, the reciprocal is computed in that dtype, and the last power is scalarmath's
        ///     <c>pow</c> when <c>ret</c> is 0-d (a NumPy scalar's <c>**=</c> rebinds the name) but the ufunc, with its fast
        ///     path, when <c>ret</c> is an array. NumSharp computed the reciprocal in float64 and raised a 0-d sum with
        ///     np.power on two 0-d arrays, which promote to float64.
        /// </summary>
        [TestMethod]
        public void VectorNorm_GeneralOrder_IsNumPysStatements()
        {
            var v16 = np.array(new[] { (Half)3, (Half)(-1), (Half)2 });
            var v32 = np.array(new[] { 3f, -1f, 2f });

            // np.linalg.norm(v32, 3): np.float32(3.3019273), bytes c7525340.
            var r32 = np.linalg.norm(v32, 3);
            AssertScalar("norm(v32, 3)", r32);
            Assert.AreEqual(np.float32, r32.dtype);
            Assert.AreEqual(0x405352C7u, BitConverter.SingleToUInt32Bits(r32.GetSingle()));
            // np.linalg.norm(v16, 3): np.float16(3.3), bytes 9a42.
            var r16 = np.linalg.norm(v16, 3);
            AssertScalar("norm(v16, 3)", r16);
            Assert.AreEqual(np.float16, r16.dtype);
            Assert.AreEqual((ushort)0x429A, BitConverter.HalfToUInt16Bits(r16.GetAtIndex<Half>(0)));
            // np.linalg.vector_norm(v16, ord=-2): np.float16(0.857), bytes db3a.
            Assert.AreEqual((ushort)0x3ADB,
                BitConverter.HalfToUInt16Bits(np.linalg.vector_norm(v16, (int[])null, false, -2).GetAtIndex<Half>(0)));

            // The reciprocal is float32's 1 / 1.1 (2eba683f; the float64 quotient rounded to float32 is 2fba683f), so
            // np.linalg.norm(v32, 1.1) is 5241af40 where the rounded quotient would give 5341af40.
            Assert.AreEqual(0x40AF4152u, BitConverter.SingleToUInt32Bits(np.linalg.norm(v32, 1.1).GetSingle()));
            // np.linalg.norm(v16, 0.3): bytes 8d54 (72.8125).
            Assert.AreEqual((ushort)0x548D, BitConverter.HalfToUInt16Bits(np.linalg.norm(v16, 0.3).GetAtIndex<Half>(0)));

            // A 0-d ret ** 2 (the order 0.5) is scalarmath's pow, an array's ret **= 2 the ufunc's fast path x * x: with
            // v = [757.7617475757002 ** 2], np.linalg.norm(v, 0.5) is pow's 0x1.185f5bb7003c2p+19 and
            // np.linalg.norm(v, 0.5, keepdims=True) the product's 0x1.185f5bb7003c1p+19.
            const double x = 757.7617475757002;
            var v = np.array(new[] { x * x });
            Assert.AreEqual(0x412185F5BB7003C2L, BitConverter.DoubleToInt64Bits(np.linalg.norm(v, 0.5).GetDouble()));
            Assert.AreEqual(0x412185F5BB7003C1L,
                BitConverter.DoubleToInt64Bits(np.linalg.norm(v, 0.5, (int?)null, true).GetDouble(0)));
            // vector_norm never hands keepdims to norm: np.linalg.vector_norm(v, ord=0.5, keepdims=True) is the scalar's pow.
            Assert.AreEqual(0x412185F5BB7003C2L,
                BitConverter.DoubleToInt64Bits(np.linalg.vector_norm(v, (int[])null, true, 0.5).GetDouble(0)));

            // An array result keeps the dtype: np.linalg.norm(a32, 1.1, axis=1) is float32 (3,), 5241af40 274a0741 fd131541.
            var a32 = np.array(new[] { 3f, -1, 2, 0, 5, -4, 1, 7, -2 }).reshape(3, 3);
            var rows = np.linalg.norm(a32, 1.1, 1);
            Assert.AreEqual(np.float32, rows.dtype);
            CollectionAssert.AreEqual(new uint[] { 0x40AF4152, 0x41074A27, 0x411513FD },
                rows.ToArray<float>().Select(BitConverter.SingleToUInt32Bits).ToArray());
        }

        /// <summary>
        ///     cond's inverse-based orders are NumPy's statements: <c>invx = _umath_linalg.inv(x, signature='d->d')</c> under
        ///     <c>errstate(all='ignore')</c> (float64 whatever the result dtype, NaN for a singular matrix where inv raises),
        ///     then <c>norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))</c> with both norms evaluated, the cast to the
        ///     real result dtype, and the NaN-to-inf tail. NumSharp inverted in the result dtype, raised on a singular matrix
        ///     inside a stack, and returned inf for a single singular matrix without evaluating the norms, so an order the
        ///     norm refuses was not refused.
        /// </summary>
        [TestMethod]
        public void Cond_InvertsInDoubleIgnoringErrors()
        {
            var singular = np.array(new[] { 1.0, 1, 1, 0, 1, 1, 1, 1, 1 }).reshape(3, 3);

            // np.linalg.cond(singular, 1): np.float64(inf), a scalar.
            var inf = np.linalg.cond(singular, 1);
            AssertScalar("cond(singular, 1)", inf);
            Assert.AreEqual(double.PositiveInfinity, inf.GetDouble());
            // np.linalg.cond(singular, 0): norm(x, 0) still runs and refuses the order.
            Action bad = () => np.linalg.cond(singular, 0);
            bad.Should().Throw<ValueError>().WithMessage("Invalid norm order for matrices.");

            // np.linalg.cond(np.stack([singular, a]), 1): [inf, 11.375000000000002] - the singular matrix is inf, the
            // other computes (the managed LU and LAPACK may differ in the last bit of a regular inverse).
            var stack = np.linalg.cond(np.stack(new[] { singular, Sq3() }), 1);
            CollectionAssert.AreEqual(new long[] { 2 }, stack.shape.Select(d => (long)d).ToArray());
            Assert.AreEqual(double.PositiveInfinity, stack.GetDouble(0));
            Assert.AreEqual(11.375000000000002, stack.GetDouble(1), 1e-12);

            // A float32 matrix is inverted in float64: np.linalg.cond(f32, 1) is np.float32(13.802469), bytes ead65c41,
            // where inverting in float32 gives e9d65c41; np.linalg.cond(f32, np.inf) is e4382e41 (float32: e3382e41).
            var f32 = np.array(new[] { 3f, -1, 2, 0, 5, -4, 1, 7, -2.5f }).reshape(3, 3);
            var c1 = np.linalg.cond(f32, 1);
            AssertScalar("cond(f32, 1)", c1);
            Assert.AreEqual(np.float32, c1.dtype);
            Assert.AreEqual(0x415CD6EAu, BitConverter.SingleToUInt32Bits(c1.GetSingle()));
            Assert.AreEqual(0x412E38E4u,
                BitConverter.SingleToUInt32Bits(np.linalg.cond(f32, double.PositiveInfinity).GetSingle()));
        }
    }
}
