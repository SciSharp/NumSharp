using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Generic;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 9 (the owner approved all fifteen picks): every expected value, dtype, mask and text is NumPy
    ///     2.4.2's (scratchpad batch9_numpy_evidence*.txt and batch9_wrap_probe.txt), for the call exactly as the C# fix makes
    ///     it. The finfo / iinfo / can_cast migrations (fixes 14 and 15) and the isdtype / issubdtype ones (fix 2) live in
    ///     those functions' own suites.
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch9Tests
    {
        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="T"/> with NumPy's exact text.</summary>
        /// <typeparam name="T">The exception type NumPy's error maps to.</typeparam>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw <typeparamref name="T"/> with that text.</exception>
        private static void Throws<T>(Action act, string message) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(message);

        /// <summary>A masked array's data in C order, as float64.</summary>
        /// <param name="m">The masked array.</param>
        /// <returns>Its data values (masked slots included).</returns>
        private static double[] Data(NDMaskedArray m) => np.ma.getdata(m).astype(np.float64).ToArray<double>();

        /// <summary>A masked array's full mask in C order (all False for nomask).</summary>
        /// <param name="m">The masked array.</param>
        /// <returns>The mask values.</returns>
        private static bool[] Mask(NDMaskedArray m) => np.ma.getmaskarray(m).ToArray<bool>();

        /// <summary>Whether a masked array holds NumPy's <c>nomask</c> (no mask array at all).</summary>
        /// <param name="m">The masked array.</param>
        /// <returns>True for nomask.</returns>
        private static bool IsNoMask(NDMaskedArray m) => ReferenceEquals(np.ma.getmask(m), np.ma.nomask);

        /// <summary>NumPy's <c>ma.array(np.arange(6.0).reshape(2, 3), mask=[[0, 1, 0], [0, 0, 1]])</c>.</summary>
        /// <returns>A new masked (2, 3) float64 array.</returns>
        private static NDMaskedArray M23()
            => np.ma.array(np.arange(6.0).reshape(2, 3), mask: new[,] { { false, true, false }, { false, false, true } });

        /// <summary>NumPy's <c>ma.array(np.arange(6).reshape(2, 3), mask=[[0, 1, 0], [0, 0, 1]])</c> (int64).</summary>
        /// <returns>A new masked (2, 3) int64 array.</returns>
        private static NDMaskedArray I23()
            => np.ma.array(np.arange(6L).reshape(2, 3), mask: new[,] { { false, true, false }, { false, false, true } });

        /// <summary>NumPy's <c>ma.array([[-1., 4., 0.], [9., -4., 1.]], mask=[[0, 0, 0], [0, 0, 1]])</c>.</summary>
        /// <returns>A new masked (2, 3) float64 array with negatives (sqrt / log domain failures).</returns>
        private static NDMaskedArray Neg()
            => np.ma.array(np.array(new[,] { { -1.0, 4.0, 0.0 }, { 9.0, -4.0, 1.0 } }), mask: new[,] { { false, false, false }, { false, false, true } });

        /// <summary>NumPy's masked out of -7 with row 0 masked: <c>ma.array(np.full((2, 3), -7), mask=[[1, 1, 1], [0, 0, 0]])</c>.</summary>
        /// <param name="dtype">The out's dtype (float64 when null).</param>
        /// <param name="hard">Whether its mask is hard.</param>
        /// <returns>The pre-masked out.</returns>
        private static NDMaskedArray Premasked(DType dtype = null, bool hard = false)
            => np.ma.array(np.full(new Shape(2, 3), -7, dtype ?? np.float64),
                           mask: new[,] { { true, true, true }, { false, false, false } }, hard_mask: hard);

        /// <summary>The where mask the ufunc tests share: NumPy's <c>[[T, F, T], [F, T, F]]</c>.</summary>
        /// <returns>A new bool (2, 3) array.</returns>
        private static NDArray W() => np.array(new[,] { { true, false, true }, { false, true, false } });

        // ---- fix 1: Shape takes C# collection expressions; a shape enumerates its dimensions ----------------------------

        /// <summary>NumPy's list shapes: <c>np.zeros([2, 3])</c> is float64 (2, 3), <c>np.ones([5], np.int32)</c> int32,
        ///     <c>np.full([2], 5.0)</c>, an F-ordered <c>np.empty([2, 2], order='F')</c>, the 0-d <c>np.zeros([])</c>,
        ///     <c>np.indices([2, 3])</c> (2, 2, 3), <c>np.tile([1, 2], [2, 2])</c>, <c>np.reshape(a6, [3, 2])</c>,
        ///     <c>np.broadcast_to(a6, [2, 6])</c> and a random <c>size=[2, 3]</c>.</summary>
        [TestMethod]
        public void Shape_CollectionExpressions_AreNumPysListShapes()
        {
            var z = np.zeros([2, 3]);
            z.shape.Should().Equal(2L, 3L);
            z.dtype.Should().Be(np.float64);
            np.ones([5], np.int32).ToArray<int>().Should().Equal(1, 1, 1, 1, 1);
            np.full([2], 5.0).ToArray<double>().Should().Equal(5.0, 5.0);
            np.empty([2, 2], order: 'F').Shape.IsFContiguous.Should().BeTrue();
            var scalar = np.zeros([]);
            scalar.ndim.Should().Be(0);
            scalar.GetDouble().Should().Be(0.0);
            np.indices([2, 3]).shape.Should().Equal(2L, 2L, 3L);
            np.tile(np.array(new long[] { 1, 2 }), [2, 2]).ToArray<long>().Should().Equal(1L, 2L, 1L, 2L, 1L, 2L, 1L, 2L);
            var a6 = np.arange(6L);
            np.reshape(a6, [3, 2]).shape.Should().Equal(3L, 2L);
            np.broadcast_to(a6, [2, 6]).shape.Should().Equal(2L, 6L);
            np.random.normal(0, 1, [2, 3]).shape.Should().Equal(2L, 3L);
            // A collection expression of variables and of longs builds the same shape.
            int rows = 2;
            long cols = 3;
            np.zeros([rows, cols]).shape.Should().Equal(2L, 3L);
        }

        /// <summary>Python's <c>list(np.zeros((2, 3, 4)).shape)</c> is [2, 3, 4]: <c>foreach</c> walks a shape's dimensions,
        ///     outermost first, and a 0-d shape yields nothing.</summary>
        [TestMethod]
        public void Shape_Enumerates_ItsDimensions()
        {
            var dims = new List<long>();
            foreach (long d in np.zeros((2, 3, 4)).Shape)
                dims.Add(d);
            dims.Should().Equal(2L, 3L, 4L);
            var none = new List<long>();
            foreach (long d in Shape.Scalar)
                none.Add(d);
            none.Should().BeEmpty();
        }

        // ---- fix 3: the reshape methods: NumPy's (shape, order, copy) plus loose ints ------------------------------------

        /// <summary><c>a6.reshape([3, 2])</c> (CS0121 until batch 9) is (3, 2); a typed array keeps its type through NumPy's
        ///     order and copy keywords (<c>a6.reshape((3, 2), order='F')</c> is [[0, 3], [1, 4], [2, 5]] F-contiguous, and
        ///     <c>copy=True</c> shares no memory); the matrix and chararray methods take the same spellings.</summary>
        [TestMethod]
        public void Reshape_MethodSpellings_OrderAndCopyOnEveryClass()
        {
            var a6 = np.arange(6L);
            a6.reshape([3, 2]).shape.Should().Equal(3L, 2L);
            a6.reshape(3, 2).shape.Should().Equal(3L, 2L);
            a6.reshape(new int[] { 3, 2 }).shape.Should().Equal(3L, 2L);

            NDArray<long> t = np.arange(6L).MakeGeneric<long>();
            NDArray<long> f = t.reshape((3, 2), order: 'F');
            f.ToArray<long>().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
            f.Shape.IsFContiguous.Should().BeTrue();
            ((long)f[0, 1]).Should().Be(3L);
            ((long)f[2, 0]).Should().Be(2L);
            NDArray<long> copied = t.reshape((3, 2), copy: true);
            np.shares_memory(copied, t).Should().BeFalse();
            t.reshape([2, 3]).shape.Should().Equal(2L, 3L);

            var m = np.asmatrix(np.arange(6L));
            var mf = m.reshape((3, 2), order: 'F');
            mf.Should().BeOfType<NDMatrix>();
            ((long)mf[0, 1]).Should().Be(3L);
            m.reshape([3, 2]).shape.Should().Equal(3L, 2L);

            var c = np.@char.array(new[] { "a", "b", "c", "d" });
            var cc = c.reshape((2, 2), copy: true);
            cc.Should().BeOfType<NDCharArray>();
            np.shares_memory(cc, c).Should().BeFalse();
            c.reshape([2, 2]).shape.Should().Equal(2L, 2L);
        }

        // ---- fix 4: numpy.ma's ufuncs: out=, where=, dtype= ------------------------------------------------------------

        /// <summary><c>ma.add(m23, 1, dtype=np.float32)</c> is float32 with m23's mask, <c>ma.equal(m23, 2, dtype=bool)</c>
        ///     keeps the input's data under the mask, and <c>dtype=np.int64</c> is NumPy's input cast refusal.</summary>
        [TestMethod]
        public void MaUfuncs_Dtype()
        {
            var r = np.ma.add(M23(), 1, dtype: np.float32);
            r.dtype.Should().Be(np.float32);
            Data(r).Should().Equal(1.0, 1.0, 3.0, 4.0, 5.0, 5.0);
            Mask(r).Should().Equal(false, true, false, false, false, true);

            var e = np.ma.equal(M23(), 2, dtype: np.@bool);
            np.ma.getdata(e).ToArray<bool>().Should().Equal(false, true, true, false, false, true);
            Mask(e).Should().Equal(false, true, false, false, false, true);

            Throws<UFuncInputCastingError>(() => np.ma.add(M23(), 1, dtype: np.int64),
                "Cannot cast ufunc 'add' input 0 from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
        }

        /// <summary>A plain out: the ufunc writes it, the masked slots get the input back, and the returned masked array views
        ///     it (<c>np.shares_memory</c> is True). <c>ma.add(m23, 1, out=po)</c> gives po = [[1, 1, 3], [4, 5, 5]];
        ///     <c>ma.sqrt(neg, out=po)</c> masks the negatives and keeps their data; <c>ma.divide(m23, b, out=po)</c> masks the
        ///     zero divisors and adds the dividend back; with <c>where</c> an unselected slot keeps the out's -7.</summary>
        [TestMethod]
        public void MaUfuncs_PlainOut()
        {
            var po = np.full(new Shape(2, 3), -7.0);
            var r = np.ma.add(M23(), 1, @out: po);
            po.ToArray<double>().Should().Equal(1.0, 1.0, 3.0, 4.0, 5.0, 5.0);
            Mask(r).Should().Equal(false, true, false, false, false, true);
            np.shares_memory(np.ma.getdata(r), po).Should().BeTrue();

            var ps = np.full(new Shape(2, 3), -7.0);
            var s = np.ma.sqrt(Neg(), @out: ps);
            ps.ToArray<double>().Should().Equal(-1.0, 2.0, 0.0, 3.0, -4.0, 1.0);
            Mask(s).Should().Equal(true, false, false, false, true, true);

            var pd = np.full(new Shape(2, 3), -7.0);
            var dv = np.ma.divide(M23(), np.array(new[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 0.0 } }), @out: pd);
            pd.ToArray<double>().Should().Equal(0.0, 1.0, 1.0, 3.0, 4.0, 5.0);
            Mask(dv).Should().Equal(true, true, false, true, false, true);

            var pw = np.full(new Shape(2, 3), -7.0);
            var rw = np.ma.add(M23(), 1, @out: pw, where: W());
            pw.ToArray<double>().Should().Equal(1.0, 1.0, 3.0, -7.0, 5.0, 5.0);
            Mask(rw).Should().Equal(false, true, false, false, false, true);
        }

        /// <summary>A masked out runs the ufunc's <c>__array_wrap__</c>: a binary ufunc leaves it nomask, a unary one an
        ///     all-False ARRAY; the call returns a new masked array (not the out) with the operand's mask, fill value and
        ///     hardness (<c>ma.add(hard m fill -3, 1, out=ma.zeros)</c> is hard with fill -3, the out keeps its own).</summary>
        [TestMethod]
        public void MaUfuncs_MaskedOut_TheWrapReplacesItsMask()
        {
            var o = Premasked();
            var r = np.ma.add(M23(), 1, @out: o);
            ReferenceEquals(r, o).Should().BeFalse();
            Data(o).Should().Equal(1.0, 1.0, 3.0, 4.0, 5.0, 5.0);
            IsNoMask(o).Should().BeTrue("NumPy: the binary ufunc's wrap stores mask_or of two all-False masks, shrunk to nomask");
            Mask(r).Should().Equal(false, true, false, false, false, true);
            np.shares_memory(np.ma.getdata(r), np.ma.getdata(o)).Should().BeTrue();

            var on = Premasked();
            var n = np.ma.negative(M23(), @out: on);
            Data(on).Should().Equal(-0.0, 1.0, -2.0, -3.0, -4.0, 5.0);
            IsNoMask(on).Should().BeFalse("NumPy: a unary wrap's reduce over ONE input keeps that all-False array");
            Mask(on).Should().Equal(false, false, false, false, false, false);
            Mask(n).Should().Equal(false, true, false, false, false, true);

            var ow = Premasked();
            np.ma.add(M23(), 1, @out: ow, where: W());
            Data(ow).Should().Equal(1.0, 1.0, 3.0, -7.0, 5.0, 5.0);
            IsNoMask(ow).Should().BeTrue();

            var ob = Premasked(np.@bool);
            var eq = np.ma.equal(M23(), 2, @out: ob);
            np.ma.getdata(ob).ToArray<bool>().Should().Equal(false, true, true, false, false, true);
            IsNoMask(ob).Should().BeTrue();
            Mask(eq).Should().Equal(false, true, false, false, false, true);

            var hard = np.ma.array(np.arange(6.0).reshape(2, 3), mask: new[,] { { false, true, false }, { false, false, true } },
                                   hard_mask: true, fill_value: -3.0);
            var oz = np.ma.zeros(new Shape(2, 3));
            var rh = np.ma.add(hard, 1, @out: oz);
            rh.hardmask.Should().BeTrue();
            Convert.ToDouble(rh.fill_value).Should().Be(-3.0);
            oz.hardmask.Should().BeFalse();
            Convert.ToDouble(oz.fill_value).Should().Be(1e20);
        }

        /// <summary>The domained ufuncs into a masked out: the wrap writes the registered fill (0.0 for sqrt, 1.0 for log, 1
        ///     for the division family) where the domain fails and makes that the out's mask (a hard one too), the wrapper then
        ///     puts the input back under its own mask, and the division family adds nothing back where the out's wrap mask is
        ///     set (<c>MaskedArray.__iadd__</c>), so those slots keep 0 where a plain out gets the dividend. NumPy's returned
        ///     mask is itself a masked array there (its <c>tolist()</c> shows None where the out's wrap masked); NumSharp
        ///     returns the plain bool mask with the same values.</summary>
        [TestMethod]
        [Misaligned]
        public void MaUfuncs_DomainedIntoMaskedOut()
        {
            var o = Premasked();
            var s = np.ma.sqrt(Neg(), @out: o);
            Data(o).Should().Equal(-1.0, 2.0, 0.0, 3.0, -4.0, 1.0);
            Mask(o).Should().Equal(true, false, false, false, true, false);
            Mask(s).Should().Equal(true, false, false, false, true, true);

            var oh = Premasked(hard: true);
            np.ma.sqrt(Neg(), @out: oh);
            Mask(oh).Should().Equal(true, false, false, false, true, false);
            oh.hardmask.Should().BeTrue();

            var ol = Premasked();
            var l = np.ma.log(Neg(), @out: ol);
            Data(ol).Should().Equal(-1.0, System.Math.Log(4.0), 0.0, System.Math.Log(9.0), -4.0, 1.0);
            Mask(ol).Should().Equal(true, false, true, false, true, false);
            Mask(l).Should().Equal(true, false, true, false, true, true);

            var od = Premasked();
            var dv = np.ma.divide(M23(), np.array(new[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 0.0 } }), @out: od);
            Data(od).Should().Equal(0.0, 1.0, 1.0, 0.0, 4.0, 0.0);
            Mask(od).Should().Equal(true, false, false, true, false, true);
            Mask(dv).Should().Equal(true, true, false, true, false, true);

            var of = Premasked();
            np.ma.floor_divide(M23(), 0, @out: of);
            Data(of).Should().Equal(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            Mask(of).Should().Equal(true, true, true, true, true, true);

            var ow = Premasked();
            var sw = np.ma.sqrt(Neg(), @out: ow, where: W());
            Data(ow).Should().Equal(-1.0, -7.0, 0.0, -7.0, -4.0, 1.0);
            Mask(ow).Should().Equal(true, false, false, false, true, false);
            Mask(sw).Should().Equal(true, false, false, false, true, true);
        }

        /// <summary>The ufunc refuses an out it cannot cast into before anything is written (<c>ma.add(m23, 1,
        ///     out=masked int64)</c>: the out keeps its -7 and its mask), and a 0-d result returns the out itself
        ///     (<c>ma.add(2.0, 3.0, out=ma.zeros(()))</c> is the out, 5.0, nomask).</summary>
        [TestMethod]
        public void MaUfuncs_OutRefusalAndZeroDim()
        {
            var oi = Premasked(np.int64);
            Throws<UFuncOutputCastingError>(() => np.ma.add(M23(), 1, @out: oi),
                "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
            Data(oi).Should().Equal(-7.0, -7.0, -7.0, -7.0, -7.0, -7.0);
            Mask(oi).Should().Equal(true, true, true, false, false, false);

            var o0 = np.ma.zeros(Shape.Scalar);
            var r0 = np.ma.add(2.0, 3.0, @out: o0);
            ReferenceEquals(r0, o0).Should().BeTrue();
            Data(o0).Should().Equal(5.0);
            IsNoMask(o0).Should().BeTrue();

            // NumPy: ValueError: operands could not be broadcast together with shapes (2,3) () (2,2) — raised by np.add itself,
            // whose out path reports it as an ArgumentException with NumPy's text (np.add's own mapping, not numpy.ma's).
            new Action(() => np.ma.add(M23(), 1, @out: np.zeros(new Shape(2, 2)))).Should().Throw<ArgumentException>()
                .WithMessage("operands could not be broadcast together with shapes (2,3) () (2,2)*");
        }

        // ---- fix 5: ma.sort / ma.argsort and the methods: kind / order / stable ------------------------------------------

        /// <summary>v = [3, 1, --, 5]: <c>ma.sort(v, 0, 'stable')</c> is [1, 3, 5, --], <c>ma.sort(v, -1, None, None,
        ///     False)</c> [--, 1, 3, 5], <c>ma.argsort(v, 0, 'mergesort')</c> [1, 0, 3, 2] and <c>v.argsort(kind='stable',
        ///     endwith=False)</c> [2, 1, 0, 3]; stable=True is refused by ma.sort and both methods but ignored by
        ///     ma.argsort (NumPy passes stable=None on); a bad kind and an order are NumPy's ValueErrors.</summary>
        [TestMethod]
        public void MaSortArgsort_KindOrderStable()
        {
            NDMaskedArray V() => np.ma.array(np.array(new[] { 3.0, 1.0, 2.0, 5.0 }), mask: new[] { false, false, true, false });
            var s = np.ma.sort(V(), 0, "stable");
            Data(s).Should().Equal(1.0, 3.0, 5.0, 2.0);
            Mask(s).Should().Equal(false, false, false, true);
            var front = np.ma.sort(V(), -1, null, null, false);
            Data(front).Should().Equal(2.0, 1.0, 3.0, 5.0);
            Mask(front).Should().Equal(true, false, false, false);
            np.ma.argsort(V(), 0, "mergesort").ToArray<long>().Should().Equal(1L, 0L, 3L, 2L);
            np.ma.argsort(V(), -1, null, null, false).ToArray<long>().Should().Equal(2L, 1L, 0L, 3L);
            V().argsort(kind: "stable", endwith: false).ToArray<long>().Should().Equal(2L, 1L, 0L, 3L);

            const string stableRefusal = "`stable` parameter is not supported for masked arrays.";
            Throws<ValueError>(() => np.ma.sort(V(), stable: true), stableRefusal);
            Throws<ValueError>(() => np.ma.sort(V(), kind: "quicksort", stable: true), stableRefusal);
            Throws<ValueError>(() => V().sort(stable: true), stableRefusal);
            Throws<ValueError>(() => V().argsort(stable: true), stableRefusal);
            Throws<ValueError>(() => V().argsort(0, stable: true), stableRefusal);
            Throws<ValueError>(() => np.ma.sort(V(), order: "f"), "Cannot specify order when the array has no fields.");

            var v2 = np.ma.array(np.array(new[,] { { 3.0, 1.0 }, { 2.0, 5.0 } }), mask: new[,] { { false, true }, { false, false } });
            np.ma.argsort(v2, 0, stable: true).ToArray<long>().Should().Equal(1L, 1L, 0L, 0L);
            np.ma.argsort(v2).ToArray<long>().Should().Equal(2L, 0L, 3L, 1L);
            np.ma.argsort(v2, kind: "stable").ToArray<long>().Should().Equal(2L, 0L, 3L, 1L);
            np.ma.argsort(v2, null, "stable").ToArray<long>().Should().Equal(2L, 0L, 3L, 1L);
            Throws<ValueError>(() => np.ma.sort(v2, 0, "bogus"), "sort kind must be one of 'quick', 'heap', or 'stable' (got 'bogus')");
            Throws<ValueError>(() => np.ma.argsort(v2, 0, "bogus"), "sort kind must be one of 'quick', 'heap', or 'stable' (got 'bogus')");
            var heap = np.ma.sort(v2, null, "heapsort", null, false, -1.0);
            Data(heap).Should().Equal(1.0, 2.0, 3.0, 5.0);
            Mask(heap).Should().Equal(true, false, false, false);
        }

        // ---- fix 6: ma.clip: np.clip over the masked operand -----------------------------------------------------------

        /// <summary><c>ma.clip(m23, 1, 3)</c> keeps m23's mask; a masked bound masks too (<c>ma.clip(m23, [1(m), 1, 1], 3)</c>
        ///     is masked [[T, T, F], [T, F, T]], NumSharp lost it until batch 9); the min / max keywords and their two errors;
        ///     fill_value, hardmask and dtype; no bounds is a copy; a hard operand gives a hard result unless hardmask=False;
        ///     a plain operand with a masked bound is masked.</summary>
        [TestMethod]
        public void MaClip_EveryOperandsMask_AndTheKeywords()
        {
            Data(np.ma.clip(M23(), 1, 3)).Should().Equal(1.0, 1.0, 2.0, 3.0, 3.0, 3.0);
            Mask(np.ma.clip(M23(), 1, 3)).Should().Equal(false, true, false, false, false, true);

            var lo = np.ma.array(np.array(new[] { 1.0, 1.0, 1.0 }), mask: new[] { true, false, false });
            Mask(np.ma.clip(M23(), lo, 3)).Should().Equal(true, true, false, true, false, true);
            var hiM = np.ma.array(np.array(new[] { 3.0, 3.0, 3.0 }), mask: new[] { false, false, true });
            Mask(np.ma.clip(M23(), min: 1, max: hiM)).Should().Equal(false, true, true, false, false, true);

            Data(np.ma.clip(M23(), min: 1, max: 3)).Should().Equal(1.0, 1.0, 2.0, 3.0, 3.0, 3.0);
            Throws<ValueError>(() => np.ma.clip(M23(), 1, 3, min: 1),
                "Passing `min` or `max` keyword argument when `a_min` and `a_max` are provided is forbidden.");
            Throws<TypeError>(() => np.ma.clip(M23(), 1, min: 1), "clip() missing 1 required positional argument: 'a_max'");

            Convert.ToDouble(np.ma.clip(M23(), 1, 3, fill_value: 9).fill_value).Should().Be(9.0);
            np.ma.clip(M23(), 1, 3, hardmask: true).hardmask.Should().BeTrue();
            var f32 = np.ma.clip(I23(), 1, 3, dtype: np.float32);
            f32.dtype.Should().Be(np.float32);
            Mask(f32).Should().Equal(false, true, false, false, false, true);

            var copy = np.ma.clip(M23());
            Data(copy).Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            Mask(copy).Should().Equal(false, true, false, false, false, true);
            Data(np.ma.clip(M23(), null, 3)).Should().Equal(0.0, 1.0, 2.0, 3.0, 3.0, 3.0);
            Data(np.ma.clip(M23(), 1, null)).Should().Equal(1.0, 1.0, 2.0, 3.0, 4.0, 5.0);

            var hard = np.ma.array(np.arange(6.0).reshape(2, 3), mask: new[,] { { false, true, false }, { false, false, true } },
                                   hard_mask: true, fill_value: -9.0);
            var hc = np.ma.clip(hard, 1, 3);
            hc.hardmask.Should().BeTrue();
            Convert.ToDouble(hc.fill_value).Should().Be(-9.0);
            np.ma.clip(hard, 1, 3, hardmask: false).hardmask.Should().BeFalse();

            var plainBound = np.ma.clip(np.arange(3.0), lo, 3);
            Data(plainBound).Should().Equal(1.0, 1.0, 2.0);
            Mask(plainBound).Should().Equal(true, false, false);
            IsNoMask(np.ma.clip(np.arange(6.0).reshape(2, 3), 1, 3)).Should().BeTrue();
        }

        /// <summary>The outs: <c>ma.clip(m23, 1, 3, out=mo)</c> returns a new view of mo (not mo) and REPLACES mo's mask with
        ///     m23's, a hard one too (the out keeps its hardness, which the view takes); a plain out gives a mask-less view;
        ///     the METHOD returns the out itself.</summary>
        [TestMethod]
        public void MaClip_Out()
        {
            var mo = np.ma.array(np.zeros(new Shape(2, 3)), mask: new[,] { { true, true, true }, { false, false, false } }, hard_mask: true);
            var r = np.ma.clip(M23(), 1, 3, @out: mo);
            ReferenceEquals(r, mo).Should().BeFalse();
            Data(mo).Should().Equal(1.0, 1.0, 2.0, 3.0, 3.0, 3.0);
            Mask(mo).Should().Equal(false, true, false, false, false, true);
            mo.hardmask.Should().BeTrue();
            r.hardmask.Should().BeTrue();
            Mask(r).Should().Equal(false, true, false, false, false, true);

            var po = np.zeros(new Shape(2, 3));
            var rp = np.ma.clip(M23(), 1, 3, @out: po);
            IsNoMask(rp).Should().BeTrue();
            po.ToArray<double>().Should().Equal(1.0, 1.0, 2.0, 3.0, 3.0, 3.0);

            var mm = np.ma.zeros(new Shape(2, 3));
            var rm = M23().clip(1, 3, @out: mm);
            ReferenceEquals(rm, mm).Should().BeTrue();
            Mask(mm).Should().Equal(false, true, false, false, false, true);
            M23().clip(1, 3, dtype: np.float32).dtype.Should().Be(np.float32);
            Data(M23().clip()).Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        }

        // ---- fix 7: ma.median(a, axis, out, overwrite_input, keepdims) ---------------------------------------------------

        /// <summary>NumPy's order (keepdims fifth), tuple and int[] axes (<c>median(m234, (0, 2))</c> is [7.5, 16, 16]; a
        ///     repeated axis and an out-of-range one are NumPy's errors), and the plain operand path.</summary>
        [TestMethod]
        public void MaMedian_OrderAndTupleAxes()
        {
            Data(np.ma.median(M23(), 1)).Should().Equal(1.0, 3.5);
            var kd = np.ma.median(M23(), 1, null, false, true);
            kd.shape.Should().Equal(2L, 1L);
            Data(kd).Should().Equal(1.0, 3.5);
            Data(np.ma.median(M23(), (0, 1))).Should().Equal(2.5);
            Data(np.ma.median(M23(), new[] { 0, 1 })).Should().Equal(2.5);
            var kdAll = np.ma.median(M23(), (0, 1), keepdims: true);
            kdAll.shape.Should().Equal(1L, 1L);
            Data(kdAll).Should().Equal(2.5);

            var data234 = np.arange(24.0).reshape(2, 3, 4);
            var m234 = np.ma.array(data234, mask: np.equal(np.remainder(np.arange(24L).reshape(2, 3, 4), 5L), 0L));
            Data(np.ma.median(m234, (0, 2))).Should().Equal(7.5, 16.0, 16.0);
            Data(np.ma.median(m234, (2, 0))).Should().Equal(7.5, 16.0, 16.0);
            var kd3 = np.ma.median(m234, (0, 2), keepdims: true);
            kd3.shape.Should().Equal(1L, 3L, 1L);
            Data(kd3).Should().Equal(7.5, 16.0, 16.0);
            Throws<ValueError>(() => np.ma.median(m234, (0, 0)), "repeated axis");
            new Action(() => np.ma.median(m234, (0, 5))).Should().Throw<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 3*");

            var plain = np.ma.median(np.arange(6.0).reshape(2, 3), 1);
            Data(plain).Should().Equal(1.0, 4.0);
            IsNoMask(plain).Should().BeTrue();
        }

        /// <summary>A fully masked median is the <c>masked</c> singleton, which keepdims turns into an ordinary float64
        ///     array of ones-shape, data 0.0, masked (NumPy's <c>r[(Ellipsis,) + index_r]</c>): a 1-D bool array along axis
        ///     0 or flattened gives (1,), a 2-D int array flattened (1, 1) and along axis 1 (2, 1); without keepdims it stays
        ///     the singleton. The result is a copy, never a view of the process-wide constant (the oracle's ma_reduce tier
        ///     caught the first cut returning the singleton).</summary>
        [TestMethod]
        public void MaMedian_FullyMasked_Keepdims_IsAnArrayOfOnesShape()
        {
            var b = np.ma.array(np.array(new[] { true, false, false, true, false, false, true, false }),
                                mask: new[] { true, true, true, true, true, true, true, true });
            foreach (var r in new[] { np.ma.median(b, 0, keepdims: true), np.ma.median(b, keepdims: true) })
            {
                r.shape.Should().Equal(1L);
                r.dtype.Should().Be(np.float64);
                Data(r).Should().Equal(0.0);
                Mask(r).Should().Equal(true);
                ReferenceEquals(r, np.ma.masked).Should().BeFalse();
            }
            ReferenceEquals(np.ma.median(b, 0), np.ma.masked).Should().BeTrue();

            var i = np.ma.array(np.arange(6L).reshape(2, 3), mask: new[,] { { true, true, true }, { true, true, true } });
            var flat = np.ma.median(i, keepdims: true);
            flat.shape.Should().Equal(1L, 1L);
            Mask(flat).Should().Equal(true);
            var rows = np.ma.median(i, 1, keepdims: true);
            rows.shape.Should().Equal(2L, 1L);
            Data(rows).Should().Equal(0.0, 0.0);
            Mask(rows).Should().Equal(true, true);
        }

        /// <summary>The outs: a masked out is filled and returned with the result's mask as an array (an all-masked slice
        ///     masked, its data 0), an integer operand too; a plain out takes an integer operand's median but is NumPy's
        ///     TypeError for a float operand reduced along an axis; the flattened median into a 0-d masked out leaves it
        ///     nomask (an even count halves through a ufunc into the out); keepdims writes a mask-less out without masking
        ///     it.</summary>
        [TestMethod]
        public void MaMedian_Out()
        {
            var o = np.ma.zeros(new Shape(2));
            ReferenceEquals(np.ma.median(M23(), 1, @out: o), o).Should().BeTrue();
            Data(o).Should().Equal(1.0, 3.5);
            IsNoMask(o).Should().BeFalse();
            Mask(o).Should().Equal(false, false);

            var oi = np.ma.zeros(new Shape(2));
            np.ma.median(I23(), 1, @out: oi);
            Data(oi).Should().Equal(1.0, 3.5);

            var allMasked = np.ma.array(np.arange(6.0).reshape(2, 3), mask: new[,] { { true, true, true }, { false, false, true } });
            var oa = np.ma.zeros(new Shape(2));
            np.ma.median(allMasked, 1, @out: oa);
            Data(oa).Should().Equal(0.0, 3.5);
            Mask(oa).Should().Equal(true, false);

            var pi = np.zeros(new Shape(2));
            np.ma.median(I23(), 1, @out: pi);
            pi.ToArray<double>().Should().Equal(1.0, 3.5);
            Throws<TypeError>(() => np.ma.median(M23(), 1, @out: np.zeros(new Shape(2))), "return arrays must be of ArrayType");

            var o0 = np.ma.zeros(Shape.Scalar);
            ReferenceEquals(np.ma.median(M23(), null, @out: o0), o0).Should().BeTrue();
            Data(o0).Should().Equal(2.5);
            IsNoMask(o0).Should().BeTrue();
            var p0 = np.zeros(Shape.Scalar);
            np.ma.median(M23(), null, @out: p0);
            p0.GetDouble().Should().Be(2.5);

            var ok = np.ma.zeros(new Shape(2, 1));
            np.ma.median(M23(), 1, @out: ok, keepdims: true);
            Data(ok).Should().Equal(1.0, 3.5);
            IsNoMask(ok).Should().BeTrue();

            var on = np.ma.zeros(new Shape(2));
            np.ma.median(np.ma.array(np.arange(6.0).reshape(2, 3)), 1, @out: on);
            Data(on).Should().Equal(1.0, 4.0);
            Mask(on).Should().Equal(false, false);

            new Action(() => np.ma.median(M23(), 1, @out: np.ma.zeros(new Shape(3)))).Should().Throw<ValueError>();
        }

        /// <summary><c>overwrite_input=True</c> sorts the operand in place (masked slots last): along axis 0 m23 becomes
        ///     [[0, 4, 2], [3, --, --]] (data [[0, 4, 2], [3, 1, 5]]); flattened, [[5, --, 3], [4, 0, --]] becomes
        ///     [[0, 3, 4], [5, --, --]]; without it the operand is untouched.</summary>
        [TestMethod]
        public void MaMedian_OverwriteInput_SortsInPlace()
        {
            var mc = M23();
            Data(np.ma.median(mc, 0, overwrite_input: true)).Should().Equal(1.5, 4.0, 2.0);
            Data(mc).Should().Equal(0.0, 4.0, 2.0, 3.0, 1.0, 5.0);
            Mask(mc).Should().Equal(false, false, false, false, true, true);

            var mf = np.ma.array(np.array(new[,] { { 5.0, 1.0, 3.0 }, { 4.0, 0.0, 2.0 } }), mask: new[,] { { false, true, false }, { false, false, true } });
            Data(np.ma.median(mf, overwrite_input: true)).Should().Equal(3.5);
            Data(mf).Should().Equal(0.0, 3.0, 4.0, 5.0, 1.0, 2.0);
            Mask(mf).Should().Equal(false, false, false, false, true, true);

            var untouched = M23();
            np.ma.median(untouched);
            Data(untouched).Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            Mask(untouched).Should().Equal(false, true, false, false, false, true);
        }

        // ---- fix 8: ma.amax / ma.amin are np.amax / np.amin --------------------------------------------------------------

        /// <summary><c>ma.amax(m23, 0)</c> is [3, 4, 2]; NumPy's out (a masked out's mask overwritten, a plain one NaN at an
        ///     all-masked slot, a plain integer out MaskError); tuple axes; keepdims fourth; initial / where are MaskedArray.max's
        ///     TypeError on a masked operand and np.amax's keywords on a plain one.</summary>
        [TestMethod]
        public void MaAmaxAmin_AreNpAmaxAmin()
        {
            Data(np.ma.amax(M23(), 0)).Should().Equal(3.0, 4.0, 2.0);
            var om = np.ma.array(np.zeros(new Shape(3)), mask: new[] { true, false, false });
            np.ma.amax(M23(), 0, @out: om);
            Data(om).Should().Equal(3.0, 4.0, 2.0);
            Mask(om).Should().Equal(false, false, false);

            var colMasked = np.ma.array(np.arange(6.0).reshape(2, 3), mask: new[,] { { false, true, false }, { false, true, true } });
            var po = np.zeros(new Shape(3));
            np.ma.amax(colMasked, 0, @out: po);
            po.ToArray<double>()[0].Should().Be(3.0);
            double.IsNaN(po.ToArray<double>()[1]).Should().BeTrue();
            var intMasked = np.ma.array(np.arange(6L).reshape(2, 3), mask: new[,] { { false, true, false }, { false, true, true } });
            Throws<MaskError>(() => np.ma.amax(intMasked, 0, @out: np.zeros(new Shape(3), np.int64)),
                "Masked data information would be lost in one or more location.");

            Data(np.ma.amax(M23(), (0, 1))).Should().Equal(4.0);
            Data(np.ma.amin(M23(), new[] { 0, 1 })).Should().Equal(0.0);
            var kd = np.ma.amin(M23(), 0, null, true);
            kd.shape.Should().Equal(1L, 3L);
            Data(kd).Should().Equal(0.0, 4.0, 2.0);
            np.ma.amax(M23(), keepdims: true).shape.Should().Equal(1L, 1L);

            Throws<TypeError>(() => np.ma.amax(M23(), initial: 10), "MaskedArray.max() got an unexpected keyword argument 'initial'");
            Throws<TypeError>(() => np.ma.amax(M23(), initial: 1, where: np.array(true)),
                "MaskedArray.max() got an unexpected keyword argument 'initial'");
            Throws<TypeError>(() => np.ma.amin(M23(), where: np.array(true)), "MaskedArray.min() got an unexpected keyword argument 'where'");

            var plain = np.arange(6.0).reshape(2, 3);
            Data(np.ma.amax(plain, 0, initial: 10)).Should().Equal(10.0, 10.0, 10.0);
            Data(np.ma.amin(plain, 0, initial: -5)).Should().Equal(-5.0, -5.0, -5.0);
            Data(np.ma.amax(plain, 0, where: np.array(new[,] { { true, false, true }, { true, true, true } }), initial: -1))
                .Should().Equal(3.0, 4.0, 5.0);
            var kdp = np.ma.amax(plain, 0, initial: 10, keepdims: true);
            kdp.shape.Should().Equal(1L, 3L);
        }

        // ---- fix 9 / 10 / 11 / 12 / 13: the numpy.ma keyword gaps and np.insert's values ---------------------------------

        /// <summary><c>ma.row_stack([m23[0], b3], dtype=np.float32)</c> is float32 [[0, --, 2], [7, 8, 9]]; casting='safe'
        ///     into int8 is NumPy's cast refusal (InvalidCastException, NumSharp's house mapping, NumPy's text).</summary>
        [TestMethod]
        public void MaRowStack_DtypeAndCasting()
        {
            var b3 = np.array(new[] { 7.0, 8.0, 9.0 });
            var r = np.ma.row_stack(new object[] { M23()[0], b3 }, dtype: np.float32);
            r.dtype.Should().Be(np.float32);
            Data(r).Should().Equal(0.0, 1.0, 2.0, 7.0, 8.0, 9.0);
            Mask(r).Should().Equal(false, true, false, false, false, false);
            Throws<InvalidCastException>(() => np.ma.row_stack(new object[] { M23()[0], b3 }, dtype: np.int8, casting: "safe"),
                "Cannot cast array data from dtype('float64') to dtype('int8') according to the rule 'safe'");
            Data(np.ma.row_stack(M23()[0], b3)).Should().Equal(0.0, 1.0, 2.0, 7.0, 8.0, 9.0);
        }

        /// <summary><c>list(ma.ndenumerate(e, compressed=False))</c> for e = [1, --, 3] is [((0,), 1), ((1,), masked),
        ///     ((2,), 3)]; the default skips the masked slot; a plain array yields every element either way.</summary>
        [TestMethod]
        public void MaNdenumerate_Compressed()
        {
            var e = np.ma.array(np.array(new long[] { 1, 2, 3 }), mask: new[] { false, true, false });
            var skipped = np.ma.ndenumerate(e).ToList();
            skipped.Select(p => p.index[0]).Should().Equal(0L, 2L);
            var all = np.ma.ndenumerate(e, compressed: false).ToList();
            all.Select(p => p.index[0]).Should().Equal(0L, 1L, 2L);
            Convert.ToInt64(all[0].value).Should().Be(1L);
            ReferenceEquals(all[1].value, np.ma.masked).Should().BeTrue();
            Convert.ToInt64(all[2].value).Should().Be(3L);
            np.ma.ndenumerate(np.array(new long[] { 4, 5 }), compressed: false).Select(p => Convert.ToInt64(p.value)).Should().Equal(4L, 5L);
        }

        /// <summary><c>ma.arange(5, dtype=np.float32)</c>, <c>fill_value=7</c>, <c>hardmask=True</c>, <c>device='cpu'</c>, and
        ///     NumPy's errors for <c>device='gpu'</c> and <c>fill_value='x'</c>.</summary>
        [TestMethod]
        public void MaArange_Keywords()
        {
            var f = np.ma.arange(5, dtype: np.float32);
            f.dtype.Should().Be(np.float32);
            Data(f).Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0);
            IsNoMask(f).Should().BeTrue();
            np.ma.arange(5).dtype.Should().Be(np.int64);
            Convert.ToInt64(np.ma.arange(5, fill_value: 7).fill_value).Should().Be(7L);
            np.ma.arange(5, hardmask: true).hardmask.Should().BeTrue();
            Data(np.ma.arange(0, 5, 1, device: "cpu")).Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0);
            Data(np.ma.arange(1, 5, 2)).Should().Equal(1.0, 3.0);
            Data(np.ma.arange(0.5, 3)).Should().Equal(0.5, 1.5, 2.5);
            Throws<ValueError>(() => np.ma.arange(5, device: "gpu"), "Device not understood. Only \"cpu\" is allowed, but received: gpu");
            Throws<TypeError>(() => np.ma.arange(5, fill_value: "x"), "Cannot set fill value of string with array of dtype int64");
        }

        /// <summary><c>ma.left_shift(s, n=1)</c> is [2, --, 8] and <c>ma.right_shift(s, n=1)</c> [0, --, 2] for s = [1, --,
        ///     4]: NumPy's parameter name.</summary>
        [TestMethod]
        public void MaShifts_NKeyword()
        {
            var sh = np.ma.array(np.array(new long[] { 1, 2, 4 }), mask: new[] { false, true, false });
            var l = np.ma.left_shift(sh, n: 1);
            Data(l).Should().Equal(2.0, 0.0, 8.0);
            Mask(l).Should().Equal(false, true, false);
            var r = np.ma.right_shift(sh, n: 1);
            Data(r).Should().Equal(0.0, 0.0, 2.0);
            Mask(r).Should().Equal(false, true, false);
        }

        /// <summary><c>np.insert(a6, 1, values=99)</c> is [0, 99, 1, 2, 3, 4, 5]: the object forms take NumPy's
        ///     <c>values</c> keyword.</summary>
        [TestMethod]
        public void Insert_ValuesKeyword_OnTheObjectForms()
        {
            var a6 = np.arange(6L);
            np.insert(a6, 1, values: 99).ToArray<long>().Should().Equal(0L, 99L, 1L, 2L, 3L, 4L, 5L);
            np.insert(a6, 1L, values: 99).ToArray<long>().Should().Equal(0L, 99L, 1L, 2L, 3L, 4L, 5L);
            np.insert(a6, new[] { 1, 3 }, values: new[] { 7, 8 }).ToArray<long>().Should().Equal(0L, 7L, 1L, 2L, 8L, 3L, 4L, 5L);
        }
    }
}
