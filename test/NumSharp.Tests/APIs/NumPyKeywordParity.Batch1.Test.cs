using System;
using System.Linq;
using System.Numerics;
using System.Reflection;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 1 (2026-11-11, docs/plans/API_PARITY_DIFF.md): the np.* functions whose C# spelling refused a
    ///     ported NumPy call - a keyword NumSharp spelled differently (<c>premute</c>, <c>a</c> / <c>b</c>, <c>arr</c>,
    ///     <c>typeCode</c>, <c>indices</c> / <c>values</c>, <c>obj</c>), a None default a value type could not take
    ///     (<c>axis</c>, <c>order</c>, <c>density</c>), or a second overload. Each case is the NumPy call written the NumPy way,
    ///     with NumPy 2.4.2's answer for it (probed 2026-11-11), so a renamed or retyped keyword fails here even when the
    ///     values would be right; the reflection cases pin the one public signature.
    /// </summary>
    /// <remarks>
    ///     Two subclass rules bind the public function's parameters BY NAME (NumSharp.Build loads <c>null</c> for a name
    ///     the function lacks): <c>NDSubclassRules.Round</c> reads np.round's <c>a</c> and <c>NDSubclassRules.Iscomplex</c>
    ///     np.iscomplex's <c>x</c>. The matrix cases below fail if a rename leaves a rule behind - the result comes back as
    ///     a plain array instead of NumPy's matrix.
    /// </remarks>
    [TestClass]
    public class NumPyKeywordParityBatch1Tests
    {
        /// <summary>The public static overloads of <c>np.<paramref name="name"/></c>.</summary>
        /// <param name="name">The function name.</param>
        /// <returns>Every public static method of that name on <see cref="np"/>.</returns>
        private static MethodInfo[] NpForms(string name)
            => typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name).ToArray();

        /// <summary>The parameter names of <paramref name="method"/>, in order.</summary>
        /// <param name="method">The method.</param>
        /// <returns>Its parameter names.</returns>
        private static string[] Names(MethodInfo method) => method.GetParameters().Select(p => p.Name).ToArray();

        #region transpose (fix 1)

        /// <summary><c>np.transpose(a, axes=(1, 0, 2))</c> binds by NumPy's keyword and permutes the axes.</summary>
        [TestMethod]
        public void Transpose_AxesKeyword_PermutesAsNumPy()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var r = np.transpose(a, axes: new[] { 1, 0, 2 });
            r.shape.Should().Equal(3L, 2L, 4L);
            r.GetInt64(1, 0, 2).Should().Be(6L);
            Names(NpForms("transpose").Single()).Should().Equal("a", "axes");
        }

        /// <summary>
        ///     <c>a.transpose(*axes)</c>: separate integers, one array, no argument and None all bind the one method, as in
        ///     NumPy; the method's parameter is a C# <c>params</c> array.
        /// </summary>
        [TestMethod]
        public void NDArrayTranspose_StarAxes_AcceptsNumPysCallShapes()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            a.transpose(2, 0, 1).shape.Should().Equal(4L, 2L, 3L);
            a.transpose(new[] { 1, 0, 2 }).shape.Should().Equal(3L, 2L, 4L);
            a.transpose().shape.Should().Equal(4L, 3L, 2L);
            a.transpose(null).shape.Should().Equal(4L, 3L, 2L);
            var method = typeof(NDArray).GetMethods().Single(m => m.Name == "transpose" && m.DeclaringType == typeof(NDArray));
            method.GetParameters().Single().IsDefined(typeof(ParamArrayAttribute)).Should().BeTrue();
        }

        /// <summary>
        ///     np.transpose keeps NumPy's reading of an EMPTY permutation (it fits only a 0-d array):
        ///     <c>np.transpose(zeros((2, 3)), ())</c> raises "axes don't match array".
        /// </summary>
        [TestMethod]
        public void Transpose_EmptyAxes_RaisesForNdArrays_AsNumPy()
        {
            Action twoD = () => np.transpose(np.zeros(new Shape(2, 3)), axes: new int[0]);
            twoD.Should().Throw<ValueError>().WithMessage("axes don't match array");
            np.transpose(NDArray.Scalar(5L), axes: new int[0]).ndim.Should().Be(0);
        }

        /// <summary><c>matrix.transpose(1, 0)</c> is a (3, 2) matrix, NumPy's <c>[[0, 3], [1, 4], [2, 5]]</c>.</summary>
        [TestMethod]
        public void MatrixTranspose_StarAxes_IsTheTransposedMatrix()
        {
            var m = np.asmatrix(np.arange(6).reshape(2, 3));
            NDArray r = m.transpose(1, 0);
            r.Should().BeOfType<NDMatrix>();
            r.shape.Should().Equal(3L, 2L);
            r.flatten().ToArray<long>().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
        }

        /// <summary><c>MaskedArray.transpose(1, 0)</c> transposes the mask with the data (NumPy's mask
        ///     <c>[[F, F], [T, F], [F, T]]</c>).</summary>
        [TestMethod]
        public void MaskedArrayTranspose_StarAxes_TransposesTheMask()
        {
            var ma = np.ma.masked_array(np.arange(6).reshape(2, 3),
                mask: np.array(new[,] { { false, true, false }, { false, false, true } }));
            var r = ma.transpose(1, 0);
            r.shape.Should().Equal(3L, 2L);
            r.mask.flatten().ToArray<bool>().Should().Equal(false, false, true, false, false, true);
            r.filled(-1).flatten().ToArray<long>().Should().Equal(0L, 3L, -1L, 4L, 2L, -1L);
        }

        #endregion

        #region array_equal, type predicates, array_repr (fixes 2-4)

        /// <summary><c>np.array_equal(a1=..., a2=..., equal_nan=True)</c>: NaNs at one position compare equal only with
        ///     equal_nan.</summary>
        [TestMethod]
        public void ArrayEqual_A1A2Keywords_AsNumPy()
        {
            var x = np.array(new[] { 1.0, double.NaN });
            var y = np.array(new[] { 1.0, double.NaN });
            np.array_equal(a1: x, a2: y, equal_nan: true).Should().BeTrue();
            np.array_equal(a1: x, a2: y).Should().BeFalse();
            Names(NpForms("array_equal").Single()).Should().Equal("a1", "a2", "equal_nan");
        }

        /// <summary>The type predicates take NumPy's <c>x</c> and np.isscalar its <c>element</c>.</summary>
        [TestMethod]
        public void TypePredicates_XAndElementKeywords_AsNumPy()
        {
            var c = np.array(new[] { new Complex(1, 0), new Complex(1, 2) });
            np.iscomplex(x: c).ToArray<bool>().Should().Equal(false, true);
            np.isreal(x: c).ToArray<bool>().Should().Equal(true, false);
            np.iscomplexobj(x: c).Should().BeTrue();
            np.isrealobj(x: c).Should().BeFalse();
            np.isscalar(element: 3.0).Should().BeTrue();
            np.isscalar(element: new[] { 1 }).Should().BeFalse();
            foreach (var name in new[] { "iscomplex", "isreal", "iscomplexobj", "isrealobj" })
                Names(NpForms(name).Single()).Should().Equal(new[] { "x" }, name);
            Names(NpForms("isscalar").Single()).Should().Equal("element");
        }

        /// <summary>
        ///     <c>np.iscomplex</c> of a complex matrix is a matrix (<c>imag(x) != 0</c> is a ufunc the matrix wraps); of a
        ///     float matrix a plain array (<c>zeros(shape, bool)</c>). Pins the by-name binding of
        ///     <c>NDSubclassRules.Iscomplex</c> to the renamed <c>x</c>.
        /// </summary>
        [TestMethod]
        public void Iscomplex_MatrixOperand_FollowsNumPysRoute()
        {
            NDArray complexMask = np.iscomplex(np.asmatrix(np.array(new[,] { { new Complex(1, 1), new Complex(2, 0) } })));
            complexMask.Should().BeOfType<NDMatrix>();
            complexMask.flatten().ToArray<bool>().Should().Equal(true, false);
            NDArray floatMask = np.iscomplex(np.asmatrix(np.array(new[,] { { 1.0, 2.0 } })));
            floatMask.Should().BeOfType<NDArray>();
            floatMask.shape.Should().Equal(1L, 2L);
        }

        /// <summary><c>np.array_repr(arr=..., precision=2)</c> is NumPy's <c>'array([1.12, 2.5 ])'</c>.</summary>
        [TestMethod]
        public void ArrayRepr_ArrKeyword_AsNumPy()
        {
            np.array_repr(arr: np.array(new[] { 1.123456, 2.5 }), precision: 2).Should().Be("array([1.12, 2.5 ])");
            Names(NpForms("array_repr").Single()).Should().Equal("arr", "max_line_width", "precision", "suppress_small");
        }

        #endregion

        #region cumsum / cumprod, put (fixes 5-6)

        /// <summary><c>np.cumsum(a=x, axis=0, dtype=float32)</c> and <c>np.cumprod(a=x, dtype=int64)</c> bind NumPy's
        ///     keywords.</summary>
        [TestMethod]
        public void CumsumCumprod_AAndDtypeKeywords_AsNumPy()
        {
            var x = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var s = np.cumsum(a: x, axis: 0, dtype: np.float32);
            s.dtype.Should().Be(np.float32);
            s.flatten().ToArray<float>().Should().Equal(1f, 2f, 3f, 5f, 7f, 9f);
            var p = np.cumprod(a: x, dtype: np.int64);
            p.ToArray<long>().Should().Equal(1L, 2L, 6L, 24L, 120L, 720L);
            Names(NpForms("cumsum").Single()).Should().Equal("a", "axis", "dtype", "out");
            Names(NpForms("cumprod").Single()).Should().Equal("a", "axis", "dtype", "out");
        }

        /// <summary>
        ///     <c>np.put(a, ind, v, mode)</c> is one overload: the scalar call <c>np.put(p, ind=1, v=5)</c> binds it (the
        ///     literals convert to the Python values' 0-d arrays) and <c>mode='clip'</c> clips index 9 to the last slot.
        /// </summary>
        [TestMethod]
        public void Put_OneSignature_TakesNumPysKeywords()
        {
            var p = np.zeros(new Shape(4));
            np.put(a: p, ind: np.array(new long[] { 0, 9 }), v: np.array(new[] { 9.0, 8.0 }), mode: "clip");
            p.ToArray<double>().Should().Equal(9.0, 0.0, 0.0, 8.0);
            var q = np.zeros(new Shape(4));
            np.put(q, ind: 1, v: 5);
            q.ToArray<double>().Should().Equal(0.0, 5.0, 0.0, 0.0);
            Names(NpForms("put").Single()).Should().Equal("a", "ind", "v", "mode");
        }

        /// <summary>
        ///     A literal int value keeps NumPy's Python-int range check through the one overload: <c>np.put(int8, 1, 1000)</c>
        ///     raises "Python integer 1000 out of bounds for int8" (the call was CS0121-ambiguous before the merge).
        /// </summary>
        [TestMethod]
        public void Put_LiteralOutOfRange_RaisesNumPysOverflowError()
        {
            Action put = () => np.put(np.zeros(new Shape(3), np.int8), 1, 1000);
            put.Should().Throw<OverflowError>().WithMessage("Python integer 1000 out of bounds for int8");
        }

        #endregion

        #region round / around (fix 7)

        /// <summary>
        ///     <c>np.round</c> exists under NumPy 2.x's name and equals np.around; a matrix operand keeps its class for
        ///     <c>decimals=0</c> (the rint ufunc) and comes back plain otherwise - NumPy's <c>[[2., 2.]]</c> matrix and
        ///     <c>[[1.6, 2.4]]</c> ndarray. Pins the by-name binding of <c>NDSubclassRules.Round</c> to the renamed
        ///     <c>a</c>.
        /// </summary>
        [TestMethod]
        public void Round_NumPyName_AndMatrixClassFollowDecimals()
        {
            var m = np.asmatrix(np.array(new[,] { { 1.5, 2.5 } }));
            NDArray r0 = np.round(a: m);
            r0.Should().BeOfType<NDMatrix>();
            r0.flatten().ToArray<double>().Should().Equal(2.0, 2.0);
            NDArray around0 = np.around(a: m);
            around0.Should().BeOfType<NDMatrix>();
            NDArray r1 = np.round(np.asmatrix(np.array(new[,] { { 1.55, 2.45 } })), decimals: 1);
            r1.Should().BeOfType<NDArray>();
            r1.flatten().ToArray<double>().Should().Equal(1.6, 2.4);
            typeof(np).GetMethod("round_").Should().BeNull("NumPy 2.0 removed np.round_");
        }

        /// <summary><c>np.round(a, out=o)</c> writes into and returns <c>o</c>, as NumPy's.</summary>
        [TestMethod]
        public void Round_Out_ReturnsTheGivenArray()
        {
            var o = np.zeros(new Shape(5));
            var r = np.round(np.array(new[] { 0.5, 1.5, 2.5, -0.5, 1.25 }), @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(0.0, 2.0, 2.0, -0.0, 1.0);
        }

        #endregion

        #region squeeze, order=None, density=None (fixes 11-13)

        /// <summary>
        ///     <c>np.squeeze(a, axis=None)</c> is one overload: None removes every length-1 axis, an integer only that axis
        ///     (negative counts from the end), a longer axis raises NumPy's ValueError and an out-of-range one its AxisError.
        /// </summary>
        [TestMethod]
        public void Squeeze_OneSignature_AxisNoneAndInteger_AsNumPy()
        {
            var z = np.zeros(new Shape(1, 3, 1));
            np.squeeze(z, axis: null).shape.Should().Equal(3L);
            np.squeeze(z, axis: 0).shape.Should().Equal(3L, 1L);
            np.squeeze(z, axis: -1).shape.Should().Equal(1L, 3L);
            Action longAxis = () => np.squeeze(z, axis: 1);
            longAxis.Should().Throw<ValueError>().WithMessage("cannot select an axis to squeeze out which has size not equal to one");
            Action outOfRange = () => np.squeeze(z, axis: 3);
            outOfRange.Should().Throw<AxisError>().WithMessage("axis 3 is out of bounds for array of dimension 3*");
            Names(NpForms("squeeze").Single()).Should().Equal("a", "axis");
        }

        /// <summary>
        ///     <c>order=None</c>: np.asarray and np.asarray_chkfinite read it as 'K' (an F array comes back itself, still F),
        ///     np.ndarray as 'C'.
        /// </summary>
        [TestMethod]
        public void Order_None_IsEachFunctionsNumPyDefault()
        {
            var f = np.asfortranarray(np.arange(6.0).reshape(2, 3));
            var viaAsarray = np.asarray(f, order: null);
            ReferenceEquals(viaAsarray, f).Should().BeTrue();
            viaAsarray.Shape.IsFContiguous.Should().BeTrue();
            np.asarray_chkfinite(f, order: null).Shape.IsFContiguous.Should().BeTrue();
            var fresh = np.ndarray(new Shape(2, 3), order: null);
            fresh.Shape.IsContiguous.Should().BeTrue();
            fresh.Shape.IsFContiguous.Should().BeFalse();
        }

        /// <summary><c>density=None</c> is False for the three histograms: counts <c>[1, 2, 1]</c>, NumPy's 2-d and n-d
        ///     counts.</summary>
        [TestMethod]
        public void Density_None_GivesCounts_AsNumPy()
        {
            var h = np.array(new[] { 1.0, 2, 2, 3 });
            np.histogram(h, bins: 3, density: null).hist.ToArray<long>().Should().Equal(1L, 2L, 1L);
            np.histogram2d(h, h, bins: 2, density: null).H.flatten().ToArray<double>().Should().Equal(1.0, 0.0, 0.0, 3.0);
            np.histogramdd(np.array(new[,] { { 1.0, 2 }, { 3, 4 }, { 1, 4 } }), bins: 2, density: null).H.flatten()
                .ToArray<double>().Should().Equal(1.0, 1.0, 0.0, 1.0);
        }

        #endregion
    }
}
