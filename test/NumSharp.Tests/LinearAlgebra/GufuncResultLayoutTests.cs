using System.Linq;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     The product gufuncs (<c>np.matmul</c>, <c>np.vecdot</c>, <c>np.matvec</c>, <c>np.vecmat</c>) return NumPy's
    ///     result object: a fresh array that owns its data, laid out as NumPy's iterator allocates it. A placement of the
    ///     output core (a dropped promoted axis, a kept reduced axis, an <c>axes=</c> output entry) relabels the computed
    ///     buffer in place; it used to return a view of an internal buffer (OWNDATA false). Every expected layout, byte
    ///     strides and value list was probed against NumPy 2.4.2; the comments give the NumPy call.
    /// </summary>
    [TestClass]
    public class GufuncResultLayoutTests
    {
        /// <summary>
        ///     Asserts a float64 result's shape, byte strides, contiguity and values, and that it is a writeable owner, as
        ///     every NumPy gufunc output is.
        /// </summary>
        /// <param name="label">The NumPy call, for the failure message.</param>
        /// <param name="r">The NumSharp result.</param>
        /// <param name="shape">NumPy's <c>r.shape</c>.</param>
        /// <param name="strides">NumPy's <c>r.strides</c>, in bytes.</param>
        /// <param name="c">NumPy's <c>r.flags.c_contiguous</c>.</param>
        /// <param name="f">NumPy's <c>r.flags.f_contiguous</c>.</param>
        /// <param name="values">NumPy's <c>r.ravel(order='C')</c>.</param>
        private static void AssertLayout(string label, NDArray r, long[] shape, long[] strides, bool c, bool f,
            params double[] values)
        {
            Assert.AreEqual(np.float64, r.dtype, label);
            CollectionAssert.AreEqual(shape, r.shape.Select(d => (long)d).ToArray(), label + " shape");
            CollectionAssert.AreEqual(strides, r.strides, label + " strides");
            Assert.AreEqual(c, r.flags.c_contiguous, label + " C_CONTIGUOUS");
            Assert.AreEqual(f, r.flags.f_contiguous, label + " F_CONTIGUOUS");
            Assert.IsTrue(r.flags.owndata, label + " OWNDATA");
            Assert.IsTrue(r.flags.writeable, label + " WRITEABLE");
            CollectionAssert.AreEqual(values, r.flatten().ToArray<double>(), label + " values");
        }

        [TestMethod]
        public void MatvecVecmat_DropThePromotedAxisFromTheOwnedProduct()
        {
            var m = np.arange(12.0).reshape(3, 4);
            // np.matvec(m, np.arange(4.)) -> (3,), strides (8,), C F, OWNDATA True
            AssertLayout("matvec", np.matvec(m, np.arange(4.0)), new long[] { 3 }, new long[] { 8 }, true, true,
                14, 38, 62);
            // np.vecmat(np.arange(3.), m) -> (4,), strides (8,), C F, OWNDATA True
            AssertLayout("vecmat", np.vecmat(np.arange(3.0), m), new long[] { 4 }, new long[] { 8 }, true, true,
                20, 23, 26, 29);
        }

        [TestMethod]
        public void Vecdot_Keepdims_InsertsTheAxisIntoTheOwnedResult()
        {
            var m = np.arange(12.0).reshape(3, 4);
            // np.vecdot(m, m, keepdims=True) -> (3,1), strides (8,8), C F, OWNDATA True
            AssertLayout("vecdot keepdims", np.vecdot(m, m, keepdims: true), new long[] { 3, 1 }, new long[] { 8, 8 },
                true, true, 14, 126, 366);
            // np.vecdot(m, m, axis=0, keepdims=True) -> (1,4), strides (8,8), C F, OWNDATA True
            AssertLayout("vecdot axis=0 keepdims", np.vecdot(m, m, axis: 0, keepdims: true), new long[] { 1, 4 },
                new long[] { 8, 8 }, true, true, 80, 107, 140, 179);
        }

        [TestMethod]
        public void AxesOutputEntry_RelabelsTheComputedBuffer()
        {
            // np.matmul(a, b, axes=[(1,2),(1,2),(0,1)]) on (2,3,4) @ (2,4,5) -> (3,5,2), strides (40,8,120), neither C nor
            // F: the loop axis stays outermost in memory, as NumPy's iterator allocated it.
            var a = np.arange(24.0).reshape(2, 3, 4);
            var b = np.arange(40.0).reshape(2, 4, 5);
            AssertLayout("matmul axes 3-D", np.matmul(a, b, axes: new[] { new[] { 1, 2 }, new[] { 1, 2 }, new[] { 0, 1 } }),
                new long[] { 3, 5, 2 }, new long[] { 40, 8, 120 }, false, false,
                70, 1510, 76, 1564, 82, 1618, 88, 1672, 94, 1726, 190, 1950, 212, 2020, 234, 2090, 256, 2160, 278, 2230,
                310, 2390, 348, 2476, 386, 2562, 424, 2648, 462, 2734);

            // np.matmul(m, n, axes=[(0,1),(0,1),(1,0)]) -> (5,3), strides (8,40), F
            var m = np.arange(12.0).reshape(3, 4);
            var n = np.arange(20.0).reshape(4, 5);
            AssertLayout("matmul axes 2-D", np.matmul(m, n, axes: new[] { new[] { 0, 1 }, new[] { 0, 1 }, new[] { 1, 0 } }),
                new long[] { 5, 3 }, new long[] { 8, 40 }, false, true,
                70, 190, 310, 76, 212, 348, 82, 234, 386, 88, 256, 424, 94, 278, 462);

            // np.matvec(c, np.arange(4.), axes=[(0,1),(0,),(0,)]) on (3,4,5) -> (3,5), strides (8,24), F
            var c3 = np.arange(60.0).reshape(3, 4, 5);
            AssertLayout("matvec axes", np.matvec(c3, np.arange(4.0), axes: new[] { new[] { 0, 1 }, new[] { 0 }, new[] { 0 } }),
                new long[] { 3, 5 }, new long[] { 8, 24 }, false, true,
                70, 76, 82, 88, 94, 190, 196, 202, 208, 214, 310, 316, 322, 328, 334);

            // np.vecmat(np.arange(4.), d, axes=[(0,),(0,1),(0,)]) on d (4,3,5) -> (3,5), strides (8,24), F
            var d = np.arange(60.0).reshape(4, 3, 5);
            AssertLayout("vecmat axes", np.vecmat(np.arange(4.0), d, axes: new[] { new[] { 0 }, new[] { 0, 1 }, new[] { 0 } }),
                new long[] { 3, 5 }, new long[] { 8, 24 }, false, true,
                210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294);
        }

        /// <summary>
        ///     Asserts NumSharp's stand-in for a NumPy scalar result: a 0-d array that owns its data and is not
        ///     writeable, as the scalar's flags are (C, F, OWNDATA, not WRITEABLE).
        /// </summary>
        /// <param name="label">The NumPy call, for the failure message.</param>
        /// <param name="r">The NumSharp result.</param>
        /// <param name="dtype">NumPy's scalar dtype.</param>
        /// <param name="value">NumPy's <c>r.item()</c>.</param>
        private static void AssertScalarStandIn(string label, NDArray r, DType dtype, double value)
        {
            Assert.AreEqual(dtype, r.dtype, label);
            Assert.AreEqual(0, r.ndim, label + " ndim");
            Assert.IsTrue(r.flags.c_contiguous && r.flags.f_contiguous, label + " contiguity");
            Assert.IsTrue(r.flags.owndata, label + " OWNDATA");
            Assert.IsFalse(r.flags.writeable, label + " WRITEABLE");
            Assert.AreEqual(value, r.astype(np.float64).GetDouble(), label + " value");
        }

        [TestMethod]
        public void ZeroDimensionalResult_IsNumPysScalarStandIn()
        {
            // NumPy hands a gufunc's 0-d output back as a scalar (replace_with_wrapped_result_and_return ->
            // npy_apply_wrap with return_scalar): its flags are C, F, OWNDATA and not WRITEABLE.
            var a = np.arange(3.0);
            // np.matmul(a, a) -> np.float64(5.0); np.matmul(a, a[::-1]) -> 1.0 (NumSharp's was a writeable 0-d view)
            AssertScalarStandIn("matmul", np.matmul(a, a), np.float64, 5);
            AssertScalarStandIn("matmul reversed", np.matmul(a, a["::-1"]), np.float64, 1);
            // np.matmul(np.zeros(0), np.zeros(0)) -> 0.0
            AssertScalarStandIn("matmul empty", np.matmul(np.zeros(new Shape(0L)), np.zeros(new Shape(0L))), np.float64, 0);
            // np.vecdot(np.zeros(0), np.zeros(0)) -> 0.0, and of float16 vectors np.float16(5.0) (NumSharp's were writeable)
            AssertScalarStandIn("vecdot empty", np.vecdot(np.zeros(new Shape(0L)), np.zeros(new Shape(0L))), np.float64, 0);
            var h = np.arange(3.0).astype(np.float16);
            AssertScalarStandIn("vecdot float16", np.vecdot(h, h), np.float16, 5);
            // np.vecdot of int64 vectors -> np.int64(5); np.linalg's forms return the same
            AssertScalarStandIn("vecdot int64", np.vecdot(np.arange(3), np.arange(3)), np.int64, 5);
            AssertScalarStandIn("linalg.matmul", np.linalg.matmul(a, a), np.float64, 5);
            AssertScalarStandIn("linalg.vecdot", np.linalg.vecdot(a, a), np.float64, 5);

            // With out= NumPy returns the out itself, writeable: return_scalar is off.
            var o = np.zeros(new Shape());
            var r = np.matmul(a, a, o);
            Assert.IsTrue(ReferenceEquals(o, r), "matmul out= returns the out");
            Assert.IsTrue(r.flags.writeable, "matmul out= stays writeable");
            Assert.AreEqual(5.0, r.GetDouble());
            var o2 = np.zeros(new Shape());
            var r2 = np.vecdot(a, a, o2);
            Assert.IsTrue(ReferenceEquals(o2, r2), "vecdot out= returns the out");
            Assert.IsTrue(r2.flags.writeable, "vecdot out= stays writeable");
            Assert.AreEqual(5.0, r2.GetDouble());
        }

        [TestMethod]
        public void MatmulOrder_GivesNumPysOwnedLayout()
        {
            var m = np.arange(12.0).reshape(3, 4);
            var n = np.arange(20.0).reshape(4, 5);
            var fm = np.asfortranarray(m);
            var fn = np.asfortranarray(n);
            double[] values = { 70, 76, 82, 88, 94, 190, 212, 234, 256, 278, 310, 348, 386, 424, 462 };
            // np.matmul(m, n, order='F') -> (3,5), strides (8,24), F
            AssertLayout("order F", np.matmul(m, n, order: 'F'), new long[] { 3, 5 }, new long[] { 8, 24 }, false, true,
                values);
            // np.matmul(fm, fn, order='A') -> F, as both inputs are
            AssertLayout("order A of F inputs", np.matmul(fm, fn, order: 'A'), new long[] { 3, 5 }, new long[] { 8, 24 },
                false, true, values);
            // np.matmul(fm, fn) (order='K') -> C: the core axes are not part of K's ordering
            AssertLayout("order K of F inputs", np.matmul(fm, fn), new long[] { 3, 5 }, new long[] { 40, 8 }, true, false,
                values);
            // np.matmul(m, n, order='C') -> C
            AssertLayout("order C", np.matmul(m, n, order: 'C'), new long[] { 3, 5 }, new long[] { 40, 8 }, true, false,
                values);
        }

        [TestMethod]
        public void MatmulOrder_KeepsAVectorPairsScalar()
        {
            // np.matmul(a, a, order=o) of two vectors is a scalar for every o (NumSharp's order= copies returned shape (1,))
            var a = np.arange(3.0);
            foreach (char o in "CFAK")
                AssertScalarStandIn($"order {o}", np.matmul(a, a, order: o), np.float64, 5);
            // np.matmul(a, a, order='F', dtype=np.float32) -> np.float32(5.0); of two empty vectors -> 0.0
            AssertScalarStandIn("order F, dtype float32", np.matmul(a, a, dtype: np.float32, order: 'F'), np.float32, 5);
            AssertScalarStandIn("order F, empty", np.matmul(np.zeros(new Shape(0L)), np.zeros(new Shape(0L)), order: 'F'),
                np.float64, 0);
            // np.matmul(a, a, order='F', out=np.zeros(())) -> the out itself, shape ()
            var z = np.zeros(new Shape());
            var r = np.matmul(a, a, z, order: 'F');
            Assert.IsTrue(ReferenceEquals(z, r), "out= returns the out");
            Assert.AreEqual(0, r.ndim);
            Assert.AreEqual(5.0, r.GetDouble());
            // A 1-D product keeps its one axis under every order: np.matmul(m, v, order='F') -> (3,), C and F
            var m = np.arange(12.0).reshape(3, 4);
            AssertLayout("matrix-vector order F", np.matmul(m, np.arange(4.0), order: 'F'), new long[] { 3 },
                new long[] { 8 }, true, true, 14, 38, 62);
            AssertLayout("vector-matrix order C", np.matmul(np.arange(3.0), m, order: 'C'), new long[] { 4 },
                new long[] { 8 }, true, true, 20, 23, 26, 29);
        }
    }
}
