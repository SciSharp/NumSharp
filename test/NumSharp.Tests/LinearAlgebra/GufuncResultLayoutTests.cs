using System;
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

        /// <summary>(arange(prod(shape)) % 7 - 3) as float64 in C order: the operand values of the loop-order probes.</summary>
        private static NDArray Base(params long[] shape)
        {
            long n = 1;
            foreach (long d in shape)
                n *= d;
            return (np.arange(n).astype(np.float64).reshape(new Shape(shape)) % 7.0) - 3.0;
        }

        /// <summary>The same values, F-ordered: NumPy's <c>np.asfortranarray</c>.</summary>
        private static NDArray FOrder(params long[] shape) => np.asfortranarray(Base(shape));

        /// <summary>The same values with the first two axes swapped in memory: <c>base(s1, s0, ...).swapaxes(0, 1)</c>.</summary>
        private static NDArray Swapped(params long[] shape)
        {
            var t = (long[])shape.Clone();
            (t[0], t[1]) = (t[1], t[0]);
            return np.swapaxes(Base(t), 0, 1);
        }

        /// <summary>
        ///     Asserts a result's NumPy layout and that its values equal the same product over C-ordered copies of the
        ///     operands (the layout must not change a value).
        /// </summary>
        /// <param name="label">The NumPy call, for the failure message.</param>
        /// <param name="r">The NumSharp result.</param>
        /// <param name="reference">The product over C-ordered copies.</param>
        /// <param name="strides">NumPy's byte strides.</param>
        /// <param name="c">NumPy's <c>flags.c_contiguous</c>.</param>
        /// <param name="f">NumPy's <c>flags.f_contiguous</c>.</param>
        private static void AssertLoopLayout(string label, NDArray r, NDArray reference, long[] strides, bool c, bool f)
        {
            CollectionAssert.AreEqual(reference.shape, r.shape, label + " shape");
            CollectionAssert.AreEqual(strides, r.strides, label + " strides");
            Assert.AreEqual(c, r.flags.c_contiguous, label + " C_CONTIGUOUS");
            Assert.AreEqual(f, r.flags.f_contiguous, label + " F_CONTIGUOUS");
            Assert.IsTrue(r.flags.owndata, label + " OWNDATA");
            Assert.IsTrue(np.array_equal(r, reference), label + " values");
        }

        [TestMethod]
        public void LoopAxes_FollowNumPysIteratorOrder()
        {
            // NumPy's gufunc iterator sorts the loop axes by the inputs' strides (K order); the output's core axes stay
            // innermost in C order. Probed 2.4.2 on (arange % 7 - 3) values; the reference is the product of C copies.
            var fa = FOrder(2, 3, 4, 5);
            var fb = FOrder(2, 3, 5, 6);
            var ca = Base(2, 3, 4, 5);
            var cb = Base(2, 3, 5, 6);
            var mm = np.matmul(ca, cb);
            // np.matmul(F, F) -> (2,3,4,6), strides (192,384,48,8): the loop axes F-ordered, the core block C-ordered
            AssertLoopLayout("matmul F@F", np.matmul(fa, fb), mm, new long[] { 192, 384, 48, 8 }, false, false);
            // np.matmul(F, C) -> C (576,192,48,8): where the inputs disagree, C order wins
            AssertLoopLayout("matmul F@C", np.matmul(fa, cb), mm, new long[] { 576, 192, 48, 8 }, true, false);
            // np.matmul(P, P) with the loop axes swapped in memory -> (192,384,48,8)
            var pa = Swapped(2, 3, 4, 5);
            var pb = Swapped(2, 3, 5, 6);
            AssertLoopLayout("matmul P@P", np.matmul(pa, pb),
                np.matmul(np.ascontiguousarray(pa), np.ascontiguousarray(pb)), new long[] { 192, 384, 48, 8 }, false, false);
            // np.matmul(F, v) -> (2,3,4), strides (32,64,8): the vector abstains
            var v = Base(5);
            AssertLoopLayout("matmul F@v", np.matmul(fa, v), np.matmul(ca, v), new long[] { 32, 64, 8 }, false, false);
            // np.matmul(F, F, axes=[(2,3),(2,3),(0,1)]) -> (4,6,2,3), strides (48,8,192,384)
            AssertLoopLayout("matmul F@F axes", np.matmul(fa, fb, axes: new[] { new[] { 2, 3 }, new[] { 2, 3 }, new[] { 0, 1 } }),
                np.matmul(ca, cb, axes: new[] { new[] { 2, 3 }, new[] { 2, 3 }, new[] { 0, 1 } }),
                new long[] { 48, 8, 192, 384 }, false, false);
            // order='C' and 'F' lay out the whole output instead
            AssertLoopLayout("matmul F@F order C", np.matmul(fa, fb, order: 'C'), mm, new long[] { 576, 192, 48, 8 }, true,
                false);
            AssertLoopLayout("matmul F@F order F", np.matmul(fa, fb, order: 'F'), mm, new long[] { 8, 16, 48, 192 }, false,
                true);

            // np.matvec(F, v) -> (2,3,4), strides (32,64,8); of reversed (C) rows -> C (96,32,8)
            AssertLoopLayout("matvec F@v", np.matvec(fa, v), np.matvec(ca, v), new long[] { 32, 64, 8 }, false, false);
            var ra = Base(2, 3, 4, 5)["::-1, ::-1"];
            AssertLoopLayout("matvec R@v", np.matvec(ra, v), np.matvec(np.ascontiguousarray(ra), v),
                new long[] { 96, 32, 8 }, true, false);
            // np.vecmat(v, F) -> (2,3,4), strides (32,64,8)
            var fm = FOrder(2, 3, 5, 4);
            AssertLoopLayout("vecmat v@F", np.vecmat(v, fm), np.vecmat(v, Base(2, 3, 5, 4)), new long[] { 32, 64, 8 },
                false, false);

            // np.vecdot(F, F) -> F (8,16,48); np.vecdot(P, C) -> C (96,32,8): C wins the disagreement
            AssertLoopLayout("vecdot F.F", np.vecdot(fa, fa), np.vecdot(ca, ca), new long[] { 8, 16, 48 }, false, true);
            AssertLoopLayout("vecdot P.C", np.vecdot(pa, ca), np.vecdot(np.ascontiguousarray(pa), ca),
                new long[] { 96, 32, 8 }, true, false);
            // np.vecdot(F, F, axis=0, keepdims=True) -> (1,3,4,5), strides (8,8,24,96): the kept axis innermost
            AssertLoopLayout("vecdot F.F axis=0 keepdims", np.vecdot(fa, fa, axis: 0, keepdims: true),
                np.vecdot(ca, ca, axis: 0, keepdims: true), new long[] { 8, 8, 24, 96 }, false, true);
            // np.vecdot(F, C(4,5)) -> F (8,16,48): the broadcast operand votes on the loop axes only, never on the core
            // axis (NumSharp's elementwise product counted the core axis and came out C)
            var c45 = Base(4, 5);
            AssertLoopLayout("vecdot F.Cb", np.vecdot(fa, c45), np.vecdot(ca, c45), new long[] { 8, 16, 48 }, false, true);
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

        /// <summary>A float16 array of the given raw bits.</summary>
        /// <param name="bits">The raw float16 bits.</param>
        /// <returns>A 1-D array.</returns>
        private static NDArray HalfBits(params ushort[] bits) => np.array(bits.Select(BitConverter.UInt16BitsToHalf).ToArray());

        /// <summary>Asserts a float16 result's raw bits in C order.</summary>
        /// <param name="label">The NumPy call, for the failure message.</param>
        /// <param name="r">The NumSharp result.</param>
        /// <param name="bits">NumPy's raw bits.</param>
        private static void AssertHalfBits(string label, NDArray r, params ushort[] bits)
        {
            Assert.AreEqual(np.float16, r.dtype, label);
            var got = np.ascontiguousarray(r).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).ToArray();
            CollectionAssert.AreEqual(bits, got,
                $"{label}: numpy [{string.Join(",", bits.Select(b => b.ToString("x4")))}], numsharp [{string.Join(",", got.Select(b => b.ToString("x4")))}]");
        }

        [TestMethod]
        public void Float16Vecdot_IsNumPysHalfDot()
        {
            // NumPy's float16 vecdot loop is HALF_dot: each pair widened to float32 and multiplied there, the products
            // summed in float32, the sum narrowed once. NumSharp rounded every product to float16 first; these rows differ
            // by that rounding (probed 2.4.2, raw bits).
            var a = HalfBits(0x3988, 0x3e93, 0x3949, 0xbc4c, 0x3ca6, 0x39d5, 0xbe41, 0xb81d, 0x242b).reshape(3, 3);
            var b = HalfBits(0xc136, 0x3f3e, 0x3b24, 0x38b5, 0x2b47, 0x3c60, 0xb869, 0x412d, 0x4007).reshape(3, 3);
            AssertHalfBits("vecdot rows", np.vecdot(a, b), 0x3f0e, 0x3367, 0xb6f9);
            AssertHalfBits("vecdot axis=0 of the transposes", np.vecdot(a.T, b.T, axis: 0), 0x3f0e, 0x3367, 0xb6f9);
            var kept = np.vecdot(a, b, keepdims: true);
            CollectionAssert.AreEqual(new long[] { 3, 1 }, kept.shape, "keepdims shape");
            AssertHalfBits("vecdot keepdims", kept, 0x3f0e, 0x3367, 0xb6f9);
            AssertHalfBits("linalg.vecdot", np.linalg.vecdot(a, b), 0x3f0e, 0x3367, 0xb6f9);

            // The float16 loop by request: np.vecdot(u, u, dtype=np.float16) of uint8 rows [[200,100,50],[255,3,7]]
            var u = np.array(new byte[,] { { 200, 100, 50 }, { 255, 3, 7 } });
            AssertHalfBits("vecdot uint8 dtype=float16", np.vecdot(u, u, dtype: np.float16), 0x7a69, 0x7bf2);
            // A mixed pair (float16 with int8) runs the float16 loop too, both cast to it: -> 0xbe14
            AssertHalfBits("vecdot float16 . int8",
                np.vecdot(HalfBits(0x3988, 0x3e93, 0x3949), np.array(new sbyte[] { 3, -5, 7 })), 0xbe14);

            // The float32 sum: 1 + nine 2^-11 is 1.00390625 (a float16 sum stays 1.0), ones(3000) . ones(3000) is 3000
            var small = np.array(new[] { 1.0 }.Concat(Enumerable.Repeat(System.Math.Pow(2, -11), 9)).ToArray()).astype(np.float16);
            AssertHalfBits("vecdot 1 + 9 * 2^-11", np.vecdot(small, np.ones(new Shape(10L), np.float16)), 0x3c04);
            var ones = np.ones(new Shape(3000L), np.float16);
            AssertHalfBits("vecdot ones(3000)", np.vecdot(ones, ones), 0x69dc);
        }

        [TestMethod]
        public void MatmulOrder_LaysOutTheIteratorAxes()
        {
            // NumPy applies order= to the iterator's axes (loop, then core) and only then remaps them for axes=, with exact
            // C / F strides (an extent-1 axis included). Probed 2.4.2.
            var a = Base(2, 3, 4);
            var b = Base(2, 4, 5);
            var axes = new[] { new[] { 1, 2 }, new[] { 1, 2 }, new[] { 0, 1 } };
            var reference = np.matmul(a, b, axes: axes);
            // order='C' -> (3,5,2), strides (40,8,120); order='F' -> (16,48,8): neither is C- or F-contiguous
            AssertLoopLayout("axes order C", np.matmul(a, b, axes: axes, order: 'C'), reference, new long[] { 40, 8, 120 },
                false, false);
            AssertLoopLayout("axes order F", np.matmul(a, b, axes: axes, order: 'F'), reference, new long[] { 16, 48, 8 },
                false, false);
            // order='A' over F-ordered inputs is F order; over C-ordered ones C order
            AssertLoopLayout("axes order A of F inputs",
                np.matmul(np.asfortranarray(a), np.asfortranarray(b), axes: axes, order: 'A'), reference,
                new long[] { 16, 48, 8 }, false, false);
            AssertLoopLayout("axes order A of C inputs", np.matmul(a, b, axes: axes, order: 'A'), reference,
                new long[] { 40, 8, 120 }, false, false);

            // order='F' gives an extent-1 axis its F stride: np.matmul(np.arange(2.), C(1,2,3), order='F') -> (1,3),
            // strides (8,8) (a C-ordered (1,3) array, F-contiguous by its flags too, has (24,8))
            var v = Base(2);
            var c123 = Base(1, 2, 3);
            AssertLoopLayout("vector @ (1,2,3) order F", np.matmul(v, c123, order: 'F'), np.matmul(v, c123),
                new long[] { 8, 8 }, true, true);
            // np.matmul(C(2,1,3,4), C(4,5), order='F') -> (2,1,3,5), strides (8,16,16,48)
            var c2134 = Base(2, 1, 3, 4);
            var c45 = Base(4, 5);
            AssertLoopLayout("(2,1,3,4) @ (4,5) order F", np.matmul(c2134, c45, order: 'F'), np.matmul(c2134, c45),
                new long[] { 8, 16, 16, 48 }, false, true);
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
