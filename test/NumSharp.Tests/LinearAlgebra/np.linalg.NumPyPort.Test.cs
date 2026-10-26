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

        /// <summary>
        ///     tensorsolve is NumPy's statements: <c>oldshape = a.shape[-(an - b.ndim):]</c> (with as many axes in a as in b
        ///     the slice starts at -0, the whole shape), the size check <c>a.size != prod ** 2</c>, the square solve against
        ///     <c>b.ravel()</c> and <c>res.shape = oldshape</c>, which relabels solve's result in place. NumSharp checked
        ///     <c>prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])</c> and reshaped the solution into a view, so an empty or
        ///     1x1 pair of the same rank was refused or lost its shape, and a right-hand side with more axes than a overflowed
        ///     a negative array length.
        /// </summary>
        [TestMethod]
        public void Tensorsolve_IsNumPysStatements()
        {
            // np.linalg.tensorsolve(np.zeros((0, 0)), np.zeros((0, 0))): an empty (0, 0) float64 solution.
            var empty = np.linalg.tensorsolve(np.zeros(new Shape(0, 0)), np.zeros(new Shape(0, 0)));
            CollectionAssert.AreEqual(new long[] { 0, 0 }, empty.shape.Select(d => (long)d).ToArray());
            Assert.AreEqual(np.float64, empty.dtype);

            // np.linalg.tensorsolve([[2.]], [[4.]]): [[2.]], shape (1, 1), owning its data.
            var one = np.linalg.tensorsolve(np.array(new[,] { { 2.0 } }), np.array(new[,] { { 4.0 } }));
            CollectionAssert.AreEqual(new long[] { 1, 1 }, one.shape.Select(d => (long)d).ToArray());
            Assert.IsTrue(one.flags.owndata, "tensorsolve([[2.]], [[4.]]) OWNDATA");
            Assert.AreEqual(2.0, one.GetDouble(0, 0));

            // np.linalg.tensorsolve(np.array(2.), np.ones(7)): the (1, 1) system against 7 elements is solve's gufunc
            // error.
            Action wide = () => np.linalg.tensorsolve(np.array(2.0), np.ones(7));
            wide.Should().Throw<ValueError>().WithMessage(
                "solve1: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m)->(m) " +
                "(size 7 is different from 1)");

            // np.linalg.tensorsolve(np.eye(6).reshape(2, 3, 6), np.arange(6.).reshape(2, 3)): (6,) [0, 1, 2, 3, 4, 5],
            // owning its data (res.shape = oldshape, not a view).
            var x = np.linalg.tensorsolve(np.eye(6).reshape(2, 3, 6), np.arange(6.0).reshape(2, 3));
            CollectionAssert.AreEqual(new long[] { 6 }, x.shape.Select(d => (long)d).ToArray());
            Assert.IsTrue(x.flags.owndata, "tensorsolve(eye(6).reshape(2, 3, 6), ...) OWNDATA");
            CollectionAssert.AreEqual(new[] { 0.0, 1, 2, 3, 4, 5 }, x.ToArray<double>());

            // np.linalg.tensorsolve(np.zeros((0, 0), float16), np.zeros((0, 0))): solve's _commonType refusal, reached now
            // that the size check passes.
            Action half = () => np.linalg.tensorsolve(np.zeros(new Shape(0, 0), np.float16), np.zeros(new Shape(0, 0)));
            half.Should().Throw<TypeError>().WithMessage("array type float16 is unsupported in linalg");
        }

        /// <summary>
        ///     tensorinv's last statement is NumPy's <c>ia.reshape(*invshape)</c>: a 0-d operand's <c>invshape</c> is empty,
        ///     so Python calls <c>reshape()</c> with no argument and raises <c>TypeError: reshape() takes exactly 1 argument
        ///     (0 given)</c> after a successful 1x1 inverse, while a singular 0-d operand reports inv's error first. NumSharp
        ///     reshaped the inverse to <c>()</c> and returned a 0-d array.
        /// </summary>
        [TestMethod]
        public void Tensorinv_ZeroD_RaisesReshapesTypeError()
        {
            // np.linalg.tensorinv(np.array(2.)): TypeError: reshape() takes exactly 1 argument (0 given).
            Action zeroD = () => np.linalg.tensorinv(np.array(2.0));
            zeroD.Should().Throw<TypeError>().WithMessage("reshape() takes exactly 1 argument (0 given)");
            // np.linalg.tensorinv(np.array(0.)): the singular 1x1 matrix raises first - LinAlgError: Singular matrix.
            Action singular = () => np.linalg.tensorinv(np.array(0.0));
            singular.Should().Throw<LinAlgError>().WithMessage("Singular matrix");
            // np.linalg.tensorinv(np.array([2.])): (1,) [0.5], a view of the inverse (owndata False).
            var one = np.linalg.tensorinv(np.array(new[] { 2.0 }));
            CollectionAssert.AreEqual(new long[] { 1 }, one.shape.Select(d => (long)d).ToArray());
            Assert.IsFalse(one.flags.owndata, "tensorinv([2.]) OWNDATA");
            Assert.AreEqual(0.5, one.GetDouble(0));
        }

        /// <summary>
        ///     norm's 2-norm reduces NumPy's <c>(x.conj() * x).real</c>, and NumPy's complex product is <c>simd_cmul</c>: the
        ///     real part fuses <c>re * re</c> with the rounded <c>im * im</c>, one rounding where <c>re * re + im * im</c>
        ///     rounds twice. NumSharp summed the twice-rounded squares, one ULP off for these vectors.
        /// </summary>
        [TestMethod]
        public void VectorNorm_Complex_SquaresAsNumPysComplexProduct()
        {
            var z1 = np.array(new[]
            {
                new System.Numerics.Complex(0.348, -0.662), new System.Numerics.Complex(0.248, -0.838),
                new System.Numerics.Complex(1.099, -1.734), new System.Numerics.Complex(-1.285, 0.126)
            });
            var z2 = np.array(new[]
            {
                new System.Numerics.Complex(0.372, 1.61), new System.Numerics.Complex(-2.475, 1.84),
                new System.Numerics.Complex(-0.233, 1.426), new System.Numerics.Complex(0.532, -1.162)
            });

            // np.linalg.norm(z1, axis=0): 0x1.5792cb237ceb0p+1 (the twice-rounded squares give ...ceafp+1).
            Assert.AreEqual(0x4005792CB237CEB0L, BitConverter.DoubleToInt64Bits(np.linalg.norm(z1, null, 0).GetDouble()));
            // np.linalg.vector_norm(z2, axis=0): 0x1.ff674d62a1e57p+1 (twice-rounded: ...e56p+1).
            Assert.AreEqual(0x400FF674D62A1E57L, BitConverter.DoubleToInt64Bits(np.linalg.vector_norm(z2, 0).GetDouble()));
        }

        /// <summary>
        ///     np.linalg.cross is NumPy's length check and then <c>_core_cross</c>, np.cross: the dtypes promote strongly
        ///     (<c>promote_types</c>: an int8 matrix against an int32 row is int32, a bool one int32 too) and the result is
        ///     <c>moveaxis(cp, -1, axis)</c>, a view of a fresh C-ordered array (C order for the last axis, F order for a
        ///     matrix's first). NumSharp multiplied 0-d component views (which the ufuncs read as Python scalars) and
        ///     stacked the components along the first axis.
        /// </summary>
        [TestMethod]
        public void LinalgCross_IsNpCross()
        {
            var a = np.array(new sbyte[] { 3, -1, 2, 0, 5, -4, 1, 7, -2 }).reshape(3, 3);
            var row = np.array(new[] { 1, 2, 3 });

            // np.linalg.cross(int8 a, int32 row): int32 (3, 3) [[-7, -7, 7], [23, -4, -5], [25, -5, -5]], a C view.
            var r = np.linalg.cross(a, row);
            Assert.AreEqual(np.int32, r.dtype);
            CollectionAssert.AreEqual(new[] { -7, -7, 7, 23, -4, -5, 25, -5, -5 }, r.flatten().ToArray<int>());
            Assert.IsTrue(r.flags.c_contiguous && !r.flags.f_contiguous && !r.flags.owndata, "cross(a, row) C--W");
            // np.linalg.cross(a.astype(bool), row): int32 too.
            Assert.AreEqual(np.int32, np.linalg.cross(a.astype(np.bool_), row).dtype);

            // np.linalg.cross(a, a[::-1], axis=0): int8 (3, 3) [[0, -40, -16], [-8, 48, 0], [0, -40, -16]], an F view.
            var c = np.linalg.cross(a, a["::-1"], 0);
            Assert.AreEqual(np.int8, c.dtype);
            CollectionAssert.AreEqual(new sbyte[] { 0, -40, -16, -8, 48, 0, 0, -40, -16 }, c.flatten().ToArray<sbyte>());
            Assert.IsTrue(c.flags.f_contiguous && !c.flags.c_contiguous && !c.flags.owndata, "cross(a, a[::-1], axis=0) -F-W");
        }

        /// <summary>
        ///     norm's fast path (no axis, the default order) is NumPy's statements: <c>x = x.ravel(order='K')</c>, memory
        ///     order, then <c>x.dot(x)</c>, or for a complex operand <c>x_real.dot(x_real) + x_imag.dot(x_imag)</c>. NumSharp
        ///     raveled in C order and summed a complex operand's squares. Below 16 elements both dot products are a plain
        ///     sequential sum, so the bits do not depend on a matrix backend.
        /// </summary>
        [TestMethod]
        public void Norm_FastPath_RavelsInMemoryOrderAndDotsAComplexOperandsParts()
        {
            // a (3, 5) matrix held in F order: np.linalg.norm(np.asfortranarray(m)) is 0x1.1c1102d509a5bp+2, where the
            // C-order dot gives ...a5ap+2.
            var m = np.array(new[]
            {
                -0.803, 0.243, -1.656, 0.656, 1.143, -0.453, 0.43, 0.251, -0.394, -0.862, -2.033, 1.41, -0.048, 2.522, 0.826
            }).reshape(3, 5);
            Assert.AreEqual(0x4011C1102D509A5BL,
                BitConverter.DoubleToInt64Bits(np.linalg.norm(np.asfortranarray(m)).GetDouble()));

            // a complex vector: np.linalg.norm(z) is the two real dot products, 0x1.ff1a349a00ff0p+1, while
            // np.linalg.vector_norm(z) reduces the squares along the flattened axis, ...ff1p+1 (NumPy's own two answers).
            var z = np.array(new[]
            {
                new System.Numerics.Complex(-1.771, 1.752), new System.Numerics.Complex(1.394, 0.778),
                new System.Numerics.Complex(0.474, 1.528), new System.Numerics.Complex(-0.641, 0.489),
                new System.Numerics.Complex(0.068, 1.306), new System.Numerics.Complex(0.416, 1.448)
            });
            Assert.AreEqual(0x400FF1A349A00FF0L, BitConverter.DoubleToInt64Bits(np.linalg.norm(z).GetDouble()));
            Assert.AreEqual(0x400FF1A349A00FF1L, BitConverter.DoubleToInt64Bits(np.linalg.vector_norm(z).GetDouble()));
        }

        /// <summary>
        ///     NumPy's Frobenius norm over an axis pair is <c>sqrt(add.reduce((x.conj() * x).real, axis=axis))</c>: one
        ///     reduction over both axes, summed in the order its iterator visits the squares (laid out in the operand's K
        ///     order). NumSharp summed one axis and then the other, which rounds differently. Each case below is one
        ///     branch of that order, with NumPy 2.4.2's bits; the old two sums gave other bits in every one of them.
        /// </summary>
        [TestMethod]
        public void MatrixNorm_Frobenius_IsOneReductionInNumPysOrder()
        {
            var v = np.array(new[]
            {
                0.071, 2.703, -2.135, 2.692, -1.129, -0.46, 1.966, -0.545, 0.298, -2.835, 1.521, 0.229, -1.022, 1.731,
                -1.181, -0.279, -2.196, -0.581, -1.779, -1.426, 1.502, -1.318, -0.089, 2.884, 2.77, 1.349, 0.247, -1.339,
                -2.036, 2.82, 0.096, -2.305, 0.741, 1.66, 0.678, 2.504
            });

            // a 2-D C matrix, both axes: one flat reduction (0x1.7c03d0db37a9cp+2; two sums ...9dp+2), in either order.
            var a = v["0:12"].reshape(3, 4);
            Assert.AreEqual(0x4017C03D0DB37A9CL, BitConverter.DoubleToInt64Bits(np.linalg.norm(a, "fro", new[] {0, 1}).GetDouble()));
            Assert.AreEqual(0x4017C03D0DB37A9CL, BitConverter.DoubleToInt64Bits(np.linalg.norm(a, null, new[] {1, 0}).GetDouble()));

            // a C stack, the trailing pair: the two axes merge into one run per matrix, an owning result.
            var b = np.linalg.matrix_norm(v["0:24"].reshape(2, 3, 4));
            Assert.AreEqual(0x4017C03D0DB37A9CL, BitConverter.DoubleToInt64Bits(b.GetDouble(0)));
            Assert.AreEqual(0x40153CE021E56F24L, BitConverter.DoubleToInt64Bits(b.GetDouble(1)));
            Assert.IsTrue(b.flags.owndata);

            // an F-ordered matrix: the squares are reduced in memory order (0x1.8a5c325069fdfp+2; two sums ...dep+2).
            var c = np.array(new[] { -1.356, -2.957, 0.874, 1.319, 2.013, -1.309, -1.709, 0.836, 1.83, 2.782, -2.097, -0.107 })
                .reshape(4, 3).T;
            Assert.AreEqual(0x4018A5C325069FDFL, BitConverter.DoubleToInt64Bits(np.linalg.norm(c, "fro", new[] {0, 1}).GetDouble()));

            // float16, two reduced axes apart with the inner one innermost: HALF_add adds each run's float32 sum into the
            // float16 result and rounds once per run (0x46A4; two float16 sums round each run first, 0x46A3).
            var d = np.array(new[]
            {
                1.957, -2.629, -2.442, 2.779, 1.52, -0.973, -2.207, -0.68, -0.965, 2.247, -0.487, -2.508, 2.561, 0.734,
                -2.3, -2.321, -0.204, -2.447, 0.791, 0.698, -2.808, 1.845, 1.721, 2.492
            }).reshape(3, 2, 4).astype(np.float16);
            var rd = np.linalg.norm(d, "fro", new[] {0, 2});
            Assert.AreEqual((ushort)0x46A4, BitConverter.HalfToUInt16Bits(rd.GetAtIndex<Half>(0)));
            Assert.AreEqual((ushort)0x46D8, BitConverter.HalfToUInt16Bits(rd.GetAtIndex<Half>(1)));

            // two reduced axes apart with a kept axis innermost: every reduced element is added into its result in turn,
            // one sequential chain (two sums: ...03p+2 and ...8dp+2 for the first two).
            var e = np.linalg.norm(v.reshape(3, 2, 3, 2), "fro", new[] {0, 2});
            CollectionAssert.AreEqual(new long[] {2, 2}, e.shape);
            Assert.AreEqual(0x4013FDD36DC36502L, BitConverter.DoubleToInt64Bits(e.GetDouble(0, 0)));
            Assert.AreEqual(0x4015D077D52CF68CL, BitConverter.DoubleToInt64Bits(e.GetDouble(0, 1)));
            Assert.AreEqual(0x400C8688AFB33D7BL, BitConverter.DoubleToInt64Bits(e.GetDouble(1, 0)));
            Assert.AreEqual(0x401799E4DD56544CL, BitConverter.DoubleToInt64Bits(e.GetDouble(1, 1)));

            // a permuted operand: reduced in its memory order (the middle result, ...d7p+2; two sums ...d8p+2).
            var f = np.linalg.norm(v["0:24"].reshape(3, 2, 4).transpose(new[] {1, 0, 2}), "fro", new[] {0, 2});
            Assert.AreEqual(0x4013E7DAC696F813L, BitConverter.DoubleToInt64Bits(f.GetDouble(0)));
            Assert.AreEqual(0x401000AF8E6642D7L, BitConverter.DoubleToInt64Bits(f.GetDouble(1)));
            Assert.AreEqual(0x40130C4DFE8A825FL, BitConverter.DoubleToInt64Bits(f.GetDouble(2)));

            // a 4-D F operand: NumPy's (2, 5) result is F-ordered and owns its buffer (element strides (1, 2)).
            var g = np.linalg.norm(np.asfortranarray(np.arange(120).reshape(2, 3, 4, 5) / 7), "fro", new[] {1, 2});
            Assert.AreEqual(0x40307D1C15F4010AL, BitConverter.DoubleToInt64Bits(g.GetDouble(0, 1)));
            Assert.IsTrue(g.flags.f_contiguous);
            Assert.IsFalse(g.flags.c_contiguous);
            Assert.IsTrue(g.flags.owndata);
            CollectionAssert.AreEqual(new long[] {1, 2}, g.Shape.strides);
        }

        /// <summary>
        ///     np.linalg.multi_dot is NumPy's statements: <c>out</c> goes to the FINAL np.dot, whose check is strict (no cast,
        ///     no broadcast); the vector ends come back as <c>result[0, 0]</c> (a scalar) or <c>result.ravel()</c>; and the
        ///     three-matrix order reads its costs from <c>A.shape</c> and <c>C.shape</c>. NumSharp copied the product into
        ///     <c>out</c> with a same_kind cast, returned a writeable view for two vector ends, and read <c>B.shape</c>.
        ///     NumPy 2.4.2's answers.
        /// </summary>
        [TestMethod]
        public void MultiDot_IsNumPysStatements()
        {
            const string notAcceptable =
                "output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)";
            var m = np.arange(9).reshape(3, 3).astype(np.float64);
            var mi = np.arange(9).reshape(3, 3);
            // The texts are NumPy's ValueError messages, matched whole.
            void Refuses(string text, Action call) => call.Should().Throw<ValueError>().WithMessage(text);

            // out= is np.dot's: an integer or complex product refuses a float64 out, an F out or one of another rank is
            // refused, a same-rank shape mismatch is its own text; an acceptable out is returned itself.
            Refuses(notAcceptable,
                () => np.linalg.multi_dot(new[] {mi, mi.T}, np.zeros(new Shape(3, 3))));
            var mc = m.astype(np.complex128);
            Refuses(notAcceptable,
                () => np.linalg.multi_dot(new[] {mc, mc.T}, np.zeros(new Shape(3, 3))));
            Refuses(notAcceptable,
                () => np.linalg.multi_dot(new[] {mi, mi.T, mi}, np.zeros(new Shape(3, 3))));
            Refuses(notAcceptable,
                () => np.linalg.multi_dot(new[] {m, m.T, m}, np.asfortranarray(np.zeros(new Shape(3, 3)))));
            Refuses(notAcceptable,
                () => np.linalg.multi_dot(new[] {m, m.T, m}, np.zeros(new Shape(1, 3, 3))));
            Refuses("output array has wrong dimensions",
                () => np.linalg.multi_dot(new[] {m, m.T, m, m.T}, np.zeros(new Shape(3, 4))));
            var o = np.zeros(new Shape(3, 3));
            Assert.IsTrue(ReferenceEquals(o, np.linalg.multi_dot(new[] {m, m.T, m, m.T}, o)));

            // two vector ends: result[0, 0] is a scalar (a fresh read-only 0-d array here), never a view of the product
            // or of out, which keeps the (1, 1) product; a (3,) out is refused (the final dot wants (1, 1)).
            var s = np.linalg.multi_dot(new[] {m[":, 0"], m, m["0"]});
            Assert.AreEqual(0, s.ndim);
            Assert.AreEqual(180.0, s.GetDouble());
            Assert.IsTrue(s.flags.owndata);
            Assert.IsFalse(s.flags.writeable);
            var o11 = np.zeros(new Shape(1, 1));
            var s11 = np.linalg.multi_dot(new[] {m[":, 0"], m, m["0"]}, o11);
            Assert.AreEqual(180.0, s11.GetDouble());
            Assert.AreEqual(180.0, o11.GetDouble(0, 0));
            Assert.IsFalse(np.shares_memory(s11, o11));
            Refuses(notAcceptable,
                () => np.linalg.multi_dot(new[] {m[":, 0"], m, m["0"]}, np.zeros(new Shape(3))));

            // one vector end: result.ravel(), a view of the (1, 3) out.
            var o13 = np.zeros(new Shape(1, 3));
            var row = np.linalg.multi_dot(new[] {m[":, 0"], m, m}, o13);
            CollectionAssert.AreEqual(new long[] {3}, row.shape);
            Assert.IsTrue(np.shares_memory(row, o13));
            Assert.IsFalse(row.flags.owndata);

            // _multi_dot_three's costs use C.shape[0] (2 here), so the costs tie and NumPy takes A(BC): the misaligned
            // product it reports is B's, (2,1) against (2,1), where B.shape[1] (1) would have picked (AB)C and reported (1,1).
            // (np.dot's alignment error is the house IncorrectShapeException with NumPy's ValueError text.)
            Action misaligned = () =>
                np.linalg.multi_dot(new[] {np.ones(new Shape(1, 2)), np.ones(new Shape(2, 1)), np.ones(new Shape(2, 1))});
            misaligned.Should().Throw<IncorrectShapeException>()
                .WithMessage("shapes (2,1) and (2,1) not aligned: 1 (dim 1) != 2 (dim 0)");
        }
    }
}
