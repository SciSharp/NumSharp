using System;
using System.Linq;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    /// Tests for the <c>np.ma</c> masked-array ufunc family. Every expected value/mask was probed against
    /// NumPy 2.4.2 (<c>numpy.ma.*</c>) and bit-compared. The contract these pin: the underlying <c>np.*</c>
    /// op runs on the DATA, and the mask propagates — pass-through for a domain-free unary, logical-OR of the
    /// operand masks for a binary, and OR'd with the invalid-input domain for sqrt/log/divide; masked
    /// positions have the input data restored into <c>.data</c> (NumPy's <c>copyto(result, d, where=m)</c>).
    /// </summary>
    [TestClass]
    public class NDMaskedArrayTests
    {
        /// <summary>Data of a masked result as float64, in logical C-order (masked positions included).</summary>
        private static double[] D(NDMaskedArray r) => np.ma.getdata(r).astype(np.float64).ToArray<double>();

        /// <summary>Full boolean mask of a masked result (never the 0-d nomask sentinel).</summary>
        private static bool[] M(NDMaskedArray r) => np.ma.getmaskarray(r).ToArray<bool>();

        private static NDMaskedArray A() =>
            np.ma.array(np.array(new double[] { 1, -2, 3, -4 }), np.array(new bool[] { false, true, false, false }));
        private static NDMaskedArray B() =>
            np.ma.array(np.array(new double[] { 10, 20, 30, 40 }), np.array(new bool[] { false, false, true, false }));

        /// <summary>abs/absolute apply |·| to the data and carry the mask through UNCHANGED; the masked slot
        /// keeps its (restored) input value, not |input|.</summary>
        [TestMethod]
        public void Abs_And_Absolute_PassMaskThrough()
        {
            foreach (var r in new[] { np.ma.abs(A()), np.ma.absolute(A()) })
            {
                // NumPy: data=[1,-2,3,4] (masked slot restored to input -2), mask=[F,T,F,F].
                Assert.IsTrue(D(r).SequenceEqual(new double[] { 1, -2, 3, 4 }));
                Assert.IsTrue(M(r).SequenceEqual(new[] { false, true, false, false }));
            }
        }

        /// <summary>add masks a position wherever EITHER operand was masked (logical-OR), and restores the
        /// LEFT operand's data at masked slots.</summary>
        [TestMethod]
        public void Add_OrsMasks_AndRestoresLeftData()
        {
            var r = np.ma.add(A(), B());
            // NumPy: data=[11,-2,3,36] (slots 1,2 restored to a's data), mask=[F,T,T,F].
            Assert.IsTrue(D(r).SequenceEqual(new double[] { 11, -2, 3, 36 }));
            Assert.IsTrue(M(r).SequenceEqual(new[] { false, true, true, false }));
        }

        /// <summary>Two unmasked operands keep the nomask fast path: no mask is materialized and
        /// <c>getmask</c> returns the shared nomask sentinel.</summary>
        [TestMethod]
        public void Add_TwoUnmasked_StaysNomask()
        {
            var r = np.ma.add(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 3, 4, 5 }));
            Assert.IsTrue(D(r).SequenceEqual(new double[] { 4, 6, 8 }));
            Assert.IsTrue(ReferenceEquals(np.ma.getmask(r), np.ma.nomask)); // nomask, not a False array
            Assert.IsFalse(M(r).Any(x => x));                               // getmaskarray fills all-False
        }

        /// <summary>sqrt additionally MASKS inputs &lt; 0 (its domain) on top of the incoming mask, and the
        /// masked slot's data is the restored input, not NaN.</summary>
        [TestMethod]
        public void Sqrt_MasksNegativeDomain()
        {
            // c = [-1, 0, 4, 9] with index 2 already masked.
            var c = np.ma.array(np.array(new double[] { -1, 0, 4, 9 }), np.array(new bool[] { false, false, true, false }));
            var r = np.ma.sqrt(c);
            // NumPy: data=[-1,0,4,3] (idx0 negative→restored -1; idx2 restored 4), mask=[T,F,T,F].
            Assert.IsTrue(D(r).SequenceEqual(new double[] { -1, 0, 4, 3 }));
            Assert.IsTrue(M(r).SequenceEqual(new[] { true, false, true, false }));
        }

        /// <summary>divide masks positions where the denominator is (near) zero, so no inf/NaN leaks into an
        /// unmasked slot.</summary>
        [TestMethod]
        public void Divide_MasksDivisionByZero()
        {
            var d = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }));
            var e = np.ma.array(np.array(new double[] { 0, 2, 0, 4 }));
            var r = np.ma.divide(d, e);
            // NumPy: quotient [_,1,_,1] with div-by-zero slots masked; masked data restored to numerator.
            Assert.IsTrue(M(r).SequenceEqual(new[] { true, false, true, false }));
            Assert.AreEqual(1.0, D(r)[1]);
            Assert.AreEqual(1.0, D(r)[3]);
        }

        /// <summary>Comparisons return a masked boolean array whose mask is the OR of the operand masks.</summary>
        [TestMethod]
        public void Greater_ReturnsMaskedBool()
        {
            var r = np.ma.greater(A(), B());
            // a>b is all False on the unmasked slots (0,3); mask=[F,T,T,F].
            Assert.IsTrue(M(r).SequenceEqual(new[] { false, true, true, false }));
            Assert.AreEqual(0.0, D(r)[0]);
            Assert.AreEqual(0.0, D(r)[3]);
        }

        /// <summary><c>filled</c> substitutes the fill value at masked positions and returns a plain array of
        /// the data dtype; unmasked positions are untouched.</summary>
        [TestMethod]
        public void Filled_SubstitutesAtMaskedPositions()
        {
            var filled = np.ma.add(A(), B()).filled(99.0).ToArray<double>();
            // mask was [F,T,T,F] over data [11,-2,3,36] ⇒ [11, 99, 99, 36].
            Assert.IsTrue(filled.SequenceEqual(new double[] { 11, 99, 99, 36 }));
        }

        /// <summary>An operation that reduces to a single fully-masked 0-D element returns the <c>masked</c>
        /// singleton, exactly as NumPy's <c>np.ma.masked</c>.</summary>
        [TestMethod]
        public void ScalarFullyMasked_ReturnsMaskedSingleton()
        {
            var r = np.ma.add(np.ma.NDMasked, NDArray.Scalar(5.0));
            Assert.IsTrue(ReferenceEquals(r, np.ma.NDMasked));
        }

        /// <summary>
        ///     NumPy's names <c>np.ma.masked</c> / <c>np.ma.masked_singleton</c> and NumSharp's <c>ND*</c> names
        ///     <c>np.ma.NDMasked</c> / <c>np.ma.NDMaskedSingleton</c> are ONE object, so an identity check written
        ///     against any spelling agrees with what the ufuncs and reductions return. Probed against NumPy 2.4.2:
        ///     <c>masked is masked_singleton</c> → True, <c>type(masked).__name__</c> → <c>MaskedConstant</c>, a 0-D
        ///     float64 <c>0.0</c> under a <c>True</c> mask, <c>repr(masked)</c> → <c>masked</c>, and
        ///     <c>ma.array([1., 2.], mask=[1, 1]).sum() is masked</c> → True.
        /// </summary>
        [TestMethod]
        public void MaskedConstant_NumPyNames_AliasTheNDMaskedSingleton()
        {
            Assert.IsTrue(ReferenceEquals(np.ma.masked, np.ma.NDMasked));
            Assert.IsTrue(ReferenceEquals(np.ma.masked_singleton, np.ma.NDMasked));
            Assert.IsTrue(ReferenceEquals(np.ma.NDMaskedSingleton, np.ma.NDMasked));
            Assert.IsInstanceOfType(np.ma.masked, typeof(NDMaskedConstant));
            Assert.AreEqual("masked", np.ma.masked.ToString());                // NumPy repr(np.ma.masked)

            // The singleton's payload: a 0-D float64 0.0 datum under a 0-D True mask.
            var data = np.ma.getdata(np.ma.masked);
            Assert.AreEqual(0, data.ndim);
            Assert.AreEqual(np.float64, data.dtype);
            Assert.AreEqual(0.0, data.GetAtIndex<double>(0));
            Assert.IsTrue(np.ma.getmaskarray(np.ma.masked).GetAtIndex<bool>(0));

            // A fully-masked reduction returns the NumPy-named constant too — identity, not mere equality.
            var allMasked = np.ma.array(np.array(new double[] { 1, 2 }), np.array(new[] { true, true }));
            Assert.IsTrue(ReferenceEquals(np.ma.sum(allMasked), np.ma.masked));

            // Assigning the NumPy-named constant masks the slot (creating the mask) and leaves the data alone.
            var y = np.ma.array(np.array(new double[] { 1, 2, 3 }));
            y[1] = np.ma.masked;
            Assert.IsTrue(M(y).SequenceEqual(new[] { false, true, false }));
            Assert.IsTrue(D(y).SequenceEqual(new double[] { 1, 2, 3 }));
        }

        /// <summary>abs and absolute are the same op; both are domain-free (mask unchanged).</summary>
        [TestMethod]
        public void Mod_And_TrueDivide_AreAliases()
        {
            var x = np.array(new double[] { 7, 8 });
            var y = np.array(new double[] { 3, 5 });
            Assert.IsTrue(D(np.ma.mod(x, y)).SequenceEqual(D(np.ma.remainder(x, y))));
            Assert.IsTrue(D(np.ma.true_divide(x, y)).SequenceEqual(D(np.ma.divide(x, y))));
        }

        /// <summary>getdata/getmask/getmaskarray behave as NumPy: a plain array has no mask; getmaskarray
        /// still yields an all-False array of the right shape.</summary>
        [TestMethod]
        public void Substrate_GetdataGetmaskGetmaskarray()
        {
            var plain = np.array(new double[] { 1, 2, 3 });
            Assert.IsTrue(ReferenceEquals(np.ma.getmask(plain), np.ma.nomask));
            Assert.IsFalse(np.ma.getmaskarray(plain).ToArray<bool>().Any(x => x));
            Assert.IsTrue(np.ma.getdata(A()).astype(np.float64).ToArray<double>()
                .SequenceEqual(new double[] { 1, -2, 3, -4 }));
        }

        // m = [1, 2(masked), 3, 4(masked)] — reductions see only {1, 3}.
        private static NDMaskedArray M13() =>
            np.ma.array(np.array(new double[] { 1, 2, 3, 4 }), np.array(new bool[] { false, true, false, true }));

        /// <summary>Reductions exclude masked elements: sum/mean/min/max/prod/count/ptp over {1,3}.</summary>
        [TestMethod]
        public void Reductions_ExcludeMaskedElements()
        {
            Assert.AreEqual(4.0, np.ma.sum(M13()).data.GetDouble(0));   // 1+3
            Assert.AreEqual(2.0, np.ma.mean(M13()).data.GetDouble(0));  // (1+3)/2
            Assert.AreEqual(1.0, np.ma.min(M13()).data.GetDouble(0));
            Assert.AreEqual(3.0, np.ma.max(M13()).data.GetDouble(0));
            Assert.AreEqual(3.0, np.ma.prod(M13()).data.GetDouble(0));  // 1*3
            Assert.AreEqual(2L, np.ma.count(M13()).GetInt64(0));        // two unmasked
            Assert.AreEqual(2.0, np.ma.ptp(M13()).data.GetDouble(0));   // 3-1
            Assert.AreEqual(1.0, np.ma.var(M13()).data.GetDouble(0));   // mean 2 → ((1-2)²+(3-2)²)/2
            Assert.AreEqual(1.0, np.ma.std(M13()).data.GetDouble(0));
        }

        /// <summary>A reduction whose every element is masked returns the <c>masked</c> singleton.</summary>
        [TestMethod]
        public void Reduction_AllMasked_ReturnsMaskedSingleton()
        {
            var allMasked = np.ma.array(np.array(new double[] { 1, 2, 3 }), np.array(new bool[] { true, true, true }));
            Assert.IsTrue(ReferenceEquals(np.ma.sum(allMasked), np.ma.NDMasked));
            Assert.IsTrue(ReferenceEquals(np.ma.mean(allMasked), np.ma.NDMasked));
        }

        /// <summary>cumsum treats masked slots as 0 for the running total but keeps their POSITIONS masked.</summary>
        [TestMethod]
        public void Cumsum_PreservesMaskedPositions()
        {
            var m = np.ma.array(np.array(new double[] { 1, 2, 3 }), np.array(new bool[] { false, true, false }));
            var r = np.ma.cumsum(m);
            Assert.IsTrue(np.ma.getdata(r).astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 1, 1, 4 }));
            Assert.IsTrue(np.ma.getmaskarray(r).ToArray<bool>().SequenceEqual(new[] { false, true, false }));
        }

        /// <summary>argmin/argmax skip masked elements (masked filled with the dtype's extreme).</summary>
        [TestMethod]
        public void ArgMinMax_IgnoreMasked()
        {
            var m = np.ma.array(np.array(new double[] { 5, 1, 3 }), np.array(new bool[] { false, true, false }));
            Assert.AreEqual(2L, np.ma.argmin(m).GetInt64(0)); // min of {5,3} is 3 at index 2 (1 is masked)
            Assert.AreEqual(0L, np.ma.argmax(m).GetInt64(0)); // max of {5,3} is 5 at index 0
        }

        /// <summary>masked_greater/masked_invalid build masks from a predicate, OR'ing onto any existing mask.</summary>
        [TestMethod]
        public void MaskedConstructors_Build_Masks()
        {
            var g = np.ma.masked_greater(np.array(new double[] { 1, 5, 2, 9 }), 3.0);
            Assert.IsTrue(np.ma.getmaskarray(g).ToArray<bool>().SequenceEqual(new[] { false, true, false, true }));

            var inv = np.ma.masked_invalid(np.array(new double[] { 1, double.NaN, double.PositiveInfinity, 4 }));
            Assert.IsTrue(np.ma.getmaskarray(inv).ToArray<bool>().SequenceEqual(new[] { false, true, true, false }));
        }

        /// <summary>Element-wise maximum picks the larger per position; a masked slot yields the other operand.</summary>
        [TestMethod]
        public void Maximum_ElementWise()
        {
            var a = np.ma.array(np.array(new double[] { 1, 8 }), np.array(new bool[] { false, true }));
            var b = np.array(new double[] { 5, 2 });
            var r = np.ma.maximum(a, b);
            Assert.AreEqual(5.0, np.ma.getdata(r).GetDouble(0)); // max(1,5)
            Assert.AreEqual(2.0, np.ma.getdata(r).GetDouble(1)); // a masked → b's 2
        }

        /// <summary>Instance methods and arithmetic operators mirror the module functions.</summary>
        [TestMethod]
        public void InstanceMethods_And_Operators()
        {
            Assert.AreEqual(4.0, M13().sum().data.GetDouble(0));
            Assert.AreEqual(2.0, M13().mean().data.GetDouble(0));
            var s = M13() + M13();                                 // [2, --, 6, --]
            Assert.IsTrue(np.ma.getmaskarray(s).ToArray<bool>().SequenceEqual(new[] { false, true, false, true }));
            Assert.AreEqual(2.0, np.ma.getdata(s).GetDouble(0));
        }

        /// <summary>Creation makes fresh UNMASKED arrays; shape manip keeps mask aligned; compressed drops masked.</summary>
        [TestMethod]
        public void Creation_Manip_Compressed()
        {
            Assert.IsFalse(np.ma.getmaskarray(np.ma.zeros(new Shape(2, 2))).ToArray<bool>().Any(x => x));
            Assert.IsTrue(np.ma.arange(4).data.astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 0, 1, 2, 3 }));
            // compressed drops the masked elements (M13 → {1,3}).
            Assert.IsTrue(np.ma.compressed(M13()).astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 1, 3 }));
            // transpose keeps the mask aligned.
            var x = np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), np.array(new bool[,] { { false, true }, { false, false } }));
            Assert.IsTrue(np.ma.getmaskarray(np.ma.transpose(x)).ToArray<bool>().SequenceEqual(new[] { false, false, true, false }));
        }

        /// <summary>average (uniform + weighted), median, and ediff1d over unmasked elements.</summary>
        [TestMethod]
        public void Extras_Average_Median_Ediff1d()
        {
            Assert.AreEqual(2.0, np.ma.average(M13()).data.GetDouble(0));   // mean of {1,3}
            // weighted: [1, 2, 3(masked)] with weights [1,2,3] → (1·1 + 2·2)/(1+2) = 5/3
            var wa = np.ma.average(np.ma.array(np.array(new double[] { 1, 2, 3 }), np.array(new bool[] { false, false, true })),
                                   weights: np.array(new double[] { 1, 2, 3 }));
            Assert.AreEqual(5.0 / 3.0, wa.data.GetDouble(0), 1e-12);
            // median of {1,2,3,4} (100 masked) = 2.5
            var med = np.ma.median(np.ma.array(np.array(new double[] { 1, 2, 3, 4, 100 }), np.array(new bool[] { false, false, false, false, true })));
            Assert.AreEqual(2.5, med.data.GetDouble(0));
            Assert.IsTrue(np.ma.ediff1d(np.array(new double[] { 1, 2, 4, 7, 0 })).data.astype(np.float64).ToArray<double>()
                .SequenceEqual(new double[] { 1, 2, 3, -7 }));
        }

        /// <summary>ma.sort pushes masked entries to the end (keeping their data) and re-masks them; argsort
        /// treats masked as the largest.</summary>
        [TestMethod]
        public void Sort_And_Argsort_MaskedToEnd()
        {
            var m = np.ma.array(np.array(new double[] { 3, 1, 2 }), np.array(new bool[] { false, true, false }));
            var s = np.ma.sort(m);
            Assert.IsTrue(np.ma.getdata(s).astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 2, 3, 1 }));
            Assert.IsTrue(np.ma.getmaskarray(s).ToArray<bool>().SequenceEqual(new[] { false, false, true }));
            Assert.IsTrue(np.ma.argsort(m).astype(np.int64).ToArray<long>().SequenceEqual(new long[] { 2, 0, 1 }));
        }

        /// <summary>ma.unique returns the sorted unique unmasked values, plus one trailing masked entry when
        /// any element was masked.</summary>
        [TestMethod]
        public void Unique_UnmaskedValues_PlusOneMasked()
        {
            var u = np.ma.unique(np.ma.array(np.array(new double[] { 1, 2, 2, 3 }), np.array(new bool[] { false, false, true, false })));
            // unmasked values {1,2,3} (idx2's 2 is masked) → [1,2,3] + 1 masked slot.
            Assert.IsTrue(np.ma.getmaskarray(u).ToArray<bool>().SequenceEqual(new[] { false, false, false, true }));
            var vals = np.ma.getdata(u).astype(np.float64).ToArray<double>();
            Assert.IsTrue(vals.Take(3).SequenceEqual(new double[] { 1, 2, 3 })); // trailing masked entry's data is a hidden don't-care
        }

        /// <summary>count_masked/masked_all/dot/allclose behave with masked semantics.</summary>
        [TestMethod]
        public void CountMasked_MaskedAll_Dot_Allclose()
        {
            Assert.AreEqual(2L, np.ma.count_masked(M13()).GetInt64(0)); // 2 masked
            Assert.IsTrue(np.ma.getmaskarray(np.ma.masked_all(new Shape(3))).ToArray<bool>().All(x => x));
            // dot treats masked as 0: [1(m),2]·[3,4] = 0·3 + 2·4 = 8
            var dot = np.ma.dot(np.ma.array(np.array(new double[] { 1, 2 }), np.array(new bool[] { true, false })), np.array(new double[] { 3, 4 }));
            Assert.AreEqual(8.0, np.ma.getdata(dot).GetDouble(0));
            Assert.IsTrue(np.ma.allclose(np.ma.array(np.array(new double[] { 1, 2, 3 }), np.array(new bool[] { false, false, true })),
                                         np.array(new double[] { 1, 2, 99 }))); // masked position ignored
        }

        /// <summary>mean's result dtype follows NumPy exactly: int/bool/float32→float64 (NumPy's <c>dsum*1.</c>
        /// promotes float32), float16→float16 (computed in float32, cast back), complex128 preserved.</summary>
        [TestMethod]
        public void Mean_ResultDtype_MatchesNumPy()
        {
            NDMaskedArray Mk<T>(T[] v) => np.ma.array(np.array(v), np.array(new bool[] { false, true, false }));
            Assert.AreEqual(np.float64, np.ma.getdata(np.ma.mean(Mk(new int[] { 1, 2, 3 }))).dtype);
            Assert.AreEqual(np.float64, np.ma.getdata(np.ma.mean(Mk(new float[] { 1, 2, 3 }))).dtype);   // f4 → f8 (dsum*1.)
            Assert.AreEqual(np.float64, np.ma.getdata(np.ma.mean(Mk(new double[] { 1, 2, 3 }))).dtype);
            Assert.AreEqual(np.complex128, np.ma.getdata(np.ma.mean(Mk(new System.Numerics.Complex[] { 1, 2, 3 }))).dtype);
            Assert.AreEqual(np.float16, np.ma.getdata(np.ma.mean(np.ma.array(np.array(new double[] { 1, 2, 3 }).astype(np.float16),
                                                                             np.array(new bool[] { false, true, false })))).dtype);
        }

        /// <summary>Mixed NDMaskedArray/NDArray operators (both orders) compile and return a mask-aware
        /// NDMaskedArray — matching NumPy where <c>ma+nd</c>/<c>nd+ma</c> are MaskedArrays. (These were
        /// CS0034-ambiguous before the explicit (NDMaskedArray,NDArray)/(NDArray,NDMaskedArray) overloads.)</summary>
        [TestMethod]
        public void Operators_MixedWithNDArray_AreMaskAware()
        {
            var m = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }), np.array(new bool[] { false, true, false, true }));
            var nd = np.array(new double[] { 10, 20, 30, 40 });
            foreach (var r in new[] { m + nd, nd + m })
            {
                Assert.IsTrue(np.ma.getmaskarray(r).ToArray<bool>().SequenceEqual(new[] { false, true, false, true }));
                Assert.AreEqual(11.0, np.ma.getdata(r).GetDouble(0)); // 1+10
                Assert.AreEqual(33.0, np.ma.getdata(r).GetDouble(2)); // 3+30
            }
            Assert.AreEqual(90.0, np.ma.getdata(m * nd).GetDouble(2));                 // 3*30
            Assert.IsTrue(np.ma.getmaskarray(m < nd).ToArray<bool>().SequenceEqual(new[] { false, true, false, true }));
            Assert.AreEqual(1.0, np.ma.getdata(m < nd).astype(np.float64).GetDouble(0)); // 1<10 True
        }

        /// <summary>astype casts the DATA and PRESERVES the mask (NumPy's MaskedArray.astype); asarray/asanyarray
        /// wrap an ndarray as unmasked and keep an incoming mask.</summary>
        [TestMethod]
        public void Astype_And_Asarray_MatchNumPy()
        {
            var m = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }), np.array(new bool[] { false, true, false, true }));
            var asI = m.astype(np.int32);
            Assert.AreEqual(np.int32, np.ma.getdata(asI).dtype);
            Assert.IsTrue(np.ma.getmaskarray(asI).ToArray<bool>().SequenceEqual(new[] { false, true, false, true }));

            var nd = np.array(new double[] { 5, 6, 7 });
            Assert.IsTrue(ReferenceEquals(np.ma.getmask(np.ma.asarray(nd)), np.ma.nomask)); // plain → unmasked
            Assert.IsTrue(np.ma.getmaskarray(np.ma.asarray(m)).ToArray<bool>()               // masked → mask kept
                .SequenceEqual(new[] { false, true, false, true }));
            NDMaskedArray implicitlyWrapped = nd;                                            // implicit NDArray→NDMaskedArray
            Assert.IsTrue(ReferenceEquals(np.ma.getmask(implicitlyWrapped), np.ma.nomask));
        }

        // ── Set operations. Every masked element in the inputs collapses to at most ONE trailing masked
        //    entry (masked values are equal only to one another). The observable value at that slot is its
        //    fill, so these compare via FILLED values + the mask + the dtype (the raw masked datum is
        //    arbitrary and hidden, matching NumPy). Expected outputs probed against NumPy 2.4.2. ──

        /// <summary>Filled values (masked slot → dtype default) as float64, for value comparison that
        /// normalizes the arbitrary hidden datum exactly as NumPy's <c>.filled()</c> does.</summary>
        private static double[] FD(NDMaskedArray r) => r.filled().astype(np.float64).ToArray<double>();

        private static NDMaskedArray MaL(long[] d, bool[] m) => np.ma.array(np.array(d), np.array(m));

        /// <summary>intersect1d keeps values UNMASKED in BOTH inputs; a masked entry survives only when BOTH
        /// inputs carry a masked element (masked == masked). Result is always a masked array.</summary>
        [TestMethod]
        public void Intersect1d_MatchesNumPy()
        {
            var x = MaL(new long[] { 1, 3, 3, 3 }, new[] { false, false, false, true });
            var y = MaL(new long[] { 3, 1, 1, 1 }, new[] { false, false, false, true });
            var r = np.ma.intersect1d(x, y);                                    // both masked ⇒ trailing masked
            Assert.IsTrue(FD(r).SequenceEqual(new double[] { 1, 3, 999999 }));
            Assert.IsTrue(M(r).SequenceEqual(new[] { false, false, true }));
            Assert.AreEqual(np.int64, r.dtype);

            // masked in only ONE input ⇒ NO trailing masked entry.
            var a = MaL(new long[] { 1, 2, 3, 9 }, new[] { false, false, false, true });
            var b = MaL(new long[] { 2, 3, 4 }, new[] { false, false, false });
            var r2 = np.ma.intersect1d(a, b);
            Assert.IsTrue(FD(r2).SequenceEqual(new double[] { 2, 3 }));
            Assert.IsFalse(M(r2).Any(v => v));

            // Disjoint unmasked sets ⇒ empty, dtype preserved (int64, not float64).
            var r3 = np.ma.intersect1d(MaL(new long[] { 1, 2 }, new[] { false, false }),
                                       MaL(new long[] { 3, 4 }, new[] { false, false }));
            Assert.AreEqual(0, r3.size);
            Assert.AreEqual(np.int64, r3.dtype);
        }

        /// <summary>union1d keeps every UNMASKED value from either input; a masked entry survives when EITHER
        /// input carries a masked element. int+float promotes to float64; two all-masked inputs collapse to a
        /// single masked int64 entry (dtype preserved through the empty compression).</summary>
        [TestMethod]
        public void Union1d_MatchesNumPy()
        {
            var x = MaL(new long[] { 1, 3, 3, 3 }, new[] { false, false, false, true });
            var y = MaL(new long[] { 3, 1, 1, 1 }, new[] { false, false, false, true });
            var r = np.ma.union1d(x, y);
            Assert.IsTrue(FD(r).SequenceEqual(new double[] { 1, 3, 999999 }));
            Assert.IsTrue(M(r).SequenceEqual(new[] { false, false, true }));

            // Promotion: int ∪ float ⇒ float64, no masked entry.
            var rp = np.ma.union1d(np.array(new long[] { 1, 2 }), np.array(new double[] { 2.5, 3.0 }));
            Assert.IsTrue(FD(rp).SequenceEqual(new double[] { 1, 2, 2.5, 3 }));
            Assert.AreEqual(np.float64, rp.dtype);
            Assert.IsFalse(M(rp).Any(v => v));

            // Both fully masked ⇒ ONE masked entry, int64 preserved (not float64 from an empty concatenate).
            var am = MaL(new long[] { 1, 2 }, new[] { true, true });
            var bm = MaL(new long[] { 3 }, new[] { true });
            var ra = np.ma.union1d(am, bm);
            Assert.AreEqual(1, ra.size);
            Assert.IsTrue(M(ra).SequenceEqual(new[] { true }));
            Assert.AreEqual(np.int64, ra.dtype);
        }

        /// <summary>setxor1d keeps values in EXACTLY ONE input's unmasked set; the masked entry survives iff
        /// exactly one input carries a masked element (XOR of masked-presence).</summary>
        [TestMethod]
        public void Setxor1d_MatchesNumPy()
        {
            // masked in exactly one ⇒ trailing masked entry.
            var a = MaL(new long[] { 1, 2, 3, 9 }, new[] { false, false, false, true });
            var b = MaL(new long[] { 2, 3, 4 }, new[] { false, false, false });
            var r = np.ma.setxor1d(a, b);
            Assert.IsTrue(FD(r).SequenceEqual(new double[] { 1, 4, 999999 }));
            Assert.IsTrue(M(r).SequenceEqual(new[] { false, false, true }));

            // masked in BOTH ⇒ NO masked entry (masked cancels itself out of the xor).
            var c = MaL(new long[] { 1, 2, 9 }, new[] { false, false, true });
            var d = MaL(new long[] { 2, 3, 9 }, new[] { false, false, true });
            var r2 = np.ma.setxor1d(c, d);
            Assert.IsTrue(FD(r2).SequenceEqual(new double[] { 1, 3 }));
            Assert.IsFalse(M(r2).Any(v => v));
        }

        /// <summary>setdiff1d keeps ar1's unmasked values absent from ar2; ar1's masked entry survives iff ar2
        /// does NOT also carry a masked element (masked in ar2 removes it). Directional.</summary>
        [TestMethod]
        public void Setdiff1d_MatchesNumPy()
        {
            var a = MaL(new long[] { 1, 2, 3, 9 }, new[] { false, false, false, true });  // has masked
            var b = MaL(new long[] { 2, 3, 4 }, new[] { false, false, false });           // no masked
            var r = np.ma.setdiff1d(a, b);                                                // masked survives
            Assert.IsTrue(FD(r).SequenceEqual(new double[] { 1, 999999 }));
            Assert.IsTrue(M(r).SequenceEqual(new[] { false, true }));

            // Reverse direction: b has no masked ⇒ result has none, and only b's extra value 4 survives.
            var r2 = np.ma.setdiff1d(b, a);
            Assert.IsTrue(FD(r2).SequenceEqual(new double[] { 4 }));
            Assert.IsFalse(M(r2).Any(v => v));
        }

        // ── New coverage (audit MA_MODULE_AUDIT.md §6): predicates, shape queries, aliases, mask helpers,
        //    fill-value surface, functional gaps, parameter parity, operators, and the instance surface.
        //    Every expected value was probed against NumPy 2.4.2. ──

        private static NDMaskedArray Ma(double[] d, bool[] m) => np.ma.array(np.array(d), np.array(m));

        /// <summary><c>is_masked</c> is the VALUE predicate (any element masked?), distinct from the
        /// <c>isMaskedArray</c> TYPE check: an all-False mask and a plain array both read False.</summary>
        [TestMethod]
        public void IsMasked_ValuePredicate()
        {
            Assert.IsFalse(np.ma.is_masked(np.ma.array(np.array(new double[] { 1, 2, 3 }))));
            Assert.IsTrue(np.ma.is_masked(Ma(new double[] { 1, 2, 3 }, new[] { false, true, false })));
            Assert.IsFalse(np.ma.is_masked(Ma(new double[] { 1, 2, 3 }, new[] { false, false, false })));
            Assert.IsFalse(np.ma.is_masked(np.array(new double[] { 1, 2, 3 })));
            Assert.IsTrue(np.ma.is_masked(np.ma.NDMasked)); // the singleton is masked
        }

        /// <summary>Module-level <c>ndim</c>/<c>shape</c>/<c>size</c> mirror the instance props and accept a
        /// plain array too; <c>copy</c> deep-copies data AND mask.</summary>
        [TestMethod]
        public void ShapeQueries_And_Copy()
        {
            var x = np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), np.array(new bool[,] { { false, true }, { false, false } }));
            Assert.AreEqual(2, np.ma.ndim(x));
            Assert.IsTrue(np.ma.shape(x).SequenceEqual(new long[] { 2, 2 }));
            Assert.AreEqual(4L, np.ma.size(x));
            Assert.AreEqual(1, np.ma.ndim(np.array(new double[] { 1, 2, 3 })));

            var c = np.ma.copy(x);
            Assert.IsTrue(np.ma.is_masked(c));
            Assert.IsTrue(np.ma.getmaskarray(c).ToArray<bool>().SequenceEqual(new[] { false, true, false, false }));
        }

        /// <summary>The deprecated/renamed aliases forward to their canonical op (amax/amin/alltrue/sometrue/
        /// round_/innerproduct/outerproduct/isMA/isarray).</summary>
        [TestMethod]
        public void Aliases_ForwardToCanonical()
        {
            Assert.AreEqual(5.0, np.ma.amax(Ma(new double[] { 1, 5, 3 }, new[] { false, false, true })).data.GetDouble(0));
            Assert.AreEqual(3.0, np.ma.amin(Ma(new double[] { 1, 5, 3 }, new[] { true, false, false })).data.GetDouble(0));
            Assert.IsTrue(np.ma.alltrue(Ma(new double[] { 1, 1, 0 }, new[] { false, false, true })).data.GetBoolean(0));
            Assert.IsFalse(np.ma.sometrue(Ma(new double[] { 0, 0, 1 }, new[] { false, false, true })).data.GetBoolean(0));
            var rnd = np.ma.round_(Ma(new double[] { 1.4, 2.6 }, new[] { false, true }));
            Assert.IsTrue(np.ma.getmaskarray(rnd).ToArray<bool>().SequenceEqual(new[] { false, true }));
            Assert.AreEqual(32.0, np.ma.innerproduct(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 4, 5, 6 })).data.GetDouble(0));
            Assert.IsTrue(np.ma.getdata(np.ma.outerproduct(np.array(new long[] { 1, 2 }), np.array(new long[] { 3, 4 })))
                .astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 3, 4, 6, 8 }));
            Assert.IsTrue(np.ma.isMA(Ma(new double[] { 1 }, new[] { false })));
            Assert.IsTrue(np.ma.isarray(Ma(new double[] { 1 }, new[] { false })));
        }

        /// <summary>Complex fill values are COMPLEX, not a bare real (NumPy: 1e20+0j / inf+infj / -inf-infj).</summary>
        [TestMethod]
        public void ComplexFillValues_AreComplex()
        {
            Assert.AreEqual(new System.Numerics.Complex(1e20, 0), (System.Numerics.Complex)np.ma.default_fill_value(np.complex128));
            var cplx = np.array(new System.Numerics.Complex[] { System.Numerics.Complex.ImaginaryOne });
            Assert.AreEqual(new System.Numerics.Complex(double.PositiveInfinity, double.PositiveInfinity), (System.Numerics.Complex)np.ma.minimum_fill_value(cplx));
            Assert.AreEqual(new System.Numerics.Complex(double.NegativeInfinity, double.NegativeInfinity), (System.Numerics.Complex)np.ma.maximum_fill_value(cplx));
            Assert.AreEqual(1e20, (double)np.ma.default_fill_value(np.float64)); // float unchanged
        }

        /// <summary>make_mask coerces by non-zero-ness and shrinks an all-False result to nomask; make_mask_none
        /// is an all-False array; mask_or is the nomask-aware OR.</summary>
        [TestMethod]
        public void MaskConstruction_MakeMask_MaskOr()
        {
            Assert.IsTrue(np.ma.make_mask(np.array(new[] { 1, 0, 1 })).ToArray<bool>().SequenceEqual(new[] { true, false, true }));
            Assert.IsTrue(ReferenceEquals(np.ma.make_mask(np.array(new[] { 0, 0, 0 })), np.ma.nomask)); // shrinks
            Assert.IsTrue(np.ma.make_mask(np.array(new[] { 0, 0, 0 }), shrink: false).ToArray<bool>().SequenceEqual(new[] { false, false, false }));
            Assert.IsTrue(np.ma.make_mask_none(new Shape(3)).ToArray<bool>().SequenceEqual(new[] { false, false, false }));
            Assert.IsTrue(np.ma.mask_or(np.ma.nomask, np.array(new[] { true, false })).ToArray<bool>().SequenceEqual(new[] { true, false }));
            Assert.IsTrue(np.ma.mask_or(np.array(new[] { true, false }), np.array(new[] { false, true })).ToArray<bool>().SequenceEqual(new[] { true, true }));
            Assert.IsTrue(ReferenceEquals(np.ma.mask_or(np.ma.nomask, np.ma.nomask), np.ma.nomask));
        }

        /// <summary>fill_value get/set is per-instance; common_fill_value returns the shared fill or null;
        /// set_fill_value mutates in place.</summary>
        [TestMethod]
        public void FillValue_Surface()
        {
            var z = np.ma.array(np.array(new long[] { 1, 2, 3 }), np.array(new[] { false, true, false }));
            Assert.AreEqual(999999L, Convert.ToInt64(z.fill_value));
            z.fill_value = 42L;
            Assert.AreEqual(42L, Convert.ToInt64(z.fill_value));
            Assert.IsTrue(z.filled().astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 1, 42, 3 }));

            var a = np.ma.array(np.array(new long[] { 1, 2 }), fill_value: 999L);
            var b = np.ma.array(np.array(new long[] { 3, 4 }), fill_value: 999L);
            Assert.AreEqual(999L, Convert.ToInt64(np.ma.common_fill_value(a, b)));
            Assert.IsNull(np.ma.common_fill_value(a, np.ma.array(np.array(new long[] { 3, 4 }), fill_value: 7L)));

            var w = np.ma.array(np.array(new double[] { 1, 2, 3 }));
            np.ma.set_fill_value(w, 7.0);
            Assert.AreEqual(7.0, Convert.ToDouble(w.fill_value));
        }

        /// <summary>fix_invalid masks NaN/±inf AND writes the fill into their data slots (the array's own
        /// fill_value attribute is left at the dtype default).</summary>
        [TestMethod]
        public void FixInvalid_MasksAndFillsData()
        {
            var fi = np.ma.fix_invalid(np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, 4 }));
            Assert.IsTrue(np.ma.getmaskarray(fi).ToArray<bool>().SequenceEqual(new[] { false, true, true, false }));
            var data = fi.data.ToArray<double>();
            Assert.AreEqual(1.0, data[0]);
            Assert.AreEqual(1e20, data[1]); // default fill written into the masked (invalid) slot
            var fi2 = np.ma.fix_invalid(np.array(new[] { 1.0, double.NaN, double.PositiveInfinity }), fill_value: -1.0);
            Assert.IsTrue(fi2.data.ToArray<double>().SequenceEqual(new[] { 1.0, -1.0, -1.0 }));
        }

        /// <summary>left_shift/right_shift are mask-OR binary ufuncs.</summary>
        [TestMethod]
        public void BitShift_Ufuncs()
        {
            var ls = np.ma.left_shift(np.ma.array(np.array(new long[] { 1, 2, 3 }), np.array(new[] { false, true, false })), 2);
            Assert.AreEqual(4.0, D(ls)[0]);
            Assert.AreEqual(12.0, D(ls)[2]);
            Assert.IsTrue(M(ls).SequenceEqual(new[] { false, true, false }));
            var rs = np.ma.right_shift(np.ma.array(np.array(new long[] { 8, 16, 32 }), np.array(new[] { false, true, false })), 1);
            Assert.AreEqual(4.0, D(rs)[0]);
            Assert.AreEqual(16.0, D(rs)[2]);
            Assert.IsTrue(M(rs).SequenceEqual(new[] { false, true, false }));
        }

        /// <summary>clip preserves the mask; choose picks the chosen element's mask; compress drops by
        /// condition keeping mask; diagonal/trace/nonzero treat masked appropriately.</summary>
        [TestMethod]
        public void FunctionalGaps_Clip_Choose_Compress_Diagonal_Trace_Nonzero()
        {
            var cl = np.ma.clip(np.ma.array(np.array(new long[] { 1, 5, 10, 15 }), np.array(new[] { false, true, false, false })), 3, 12);
            Assert.IsTrue(D(cl).SequenceEqual(new double[] { 3, 5, 10, 12 }));
            Assert.IsTrue(M(cl).SequenceEqual(new[] { false, true, false, false }));

            // choose: [0,1,0] over {maskedArr, plainArr} → [1, 20, 3], unmasked (chosen elements all unmasked).
            var ch = np.ma.choose(np.array(new[] { 0, 1, 0 }),
                new object[] { np.ma.array(np.array(new long[] { 1, 2, 3 }), np.array(new[] { false, true, false })), np.array(new long[] { 10, 20, 30 }) });
            Assert.IsTrue(D(ch).SequenceEqual(new double[] { 1, 20, 3 }));
            Assert.IsFalse(M(ch).Any(v => v));
            // choose picking a masked element propagates its mask.
            var ch2 = np.ma.choose(np.array(new[] { 0, 0 }),
                new object[] { np.ma.array(np.array(new long[] { 1, 2 }), np.array(new[] { true, false })), np.array(new long[] { 10, 20 }) });
            Assert.IsTrue(M(ch2).SequenceEqual(new[] { true, false }));

            var cmp = np.ma.compress(np.array(new[] { true, false, true }), np.ma.array(np.array(new long[] { 1, 2, 3 }), np.array(new[] { false, false, true })));
            Assert.AreEqual(1.0, D(cmp)[0]);
            Assert.IsTrue(M(cmp).SequenceEqual(new[] { false, true }));

            var mat = np.ma.array(np.array(new long[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }).reshape(3, 3),
                                  np.array(new[] { false, true, false, false, false, false, false, false, true }).reshape(3, 3));
            var dg = np.ma.diagonal(mat);
            Assert.AreEqual(1.0, D(dg)[0]);
            Assert.AreEqual(5.0, D(dg)[1]);
            Assert.IsTrue(M(dg).SequenceEqual(new[] { false, false, true }));

            var tr = np.ma.trace(mat);          // 1 + 5 + (masked→0) = 6, float64 (astype(None) quirk)
            Assert.AreEqual(6.0, tr.GetDouble(0));
            Assert.AreEqual(np.float64, tr.dtype);

            var nz = np.ma.nonzero(np.ma.array(np.array(new long[] { 0, 1, 0, 2, 3 }), np.array(new[] { false, true, false, false, false })));
            Assert.IsTrue(nz[0].ToArray<long>().SequenceEqual(new long[] { 3, 4 })); // index 1 (value 1) masked → excluded
        }

        /// <summary>diff propagates the mask (a window with any masked element is masked); append flattens and
        /// concatenates, masks aligned.</summary>
        [TestMethod]
        public void Diff_And_Append()
        {
            var df = np.ma.diff(np.ma.array(np.array(new long[] { 1, 2, 4, 7, 11 }), np.array(new[] { false, true, false, false, false })));
            Assert.AreEqual(3.0, D(df)[2]);
            Assert.AreEqual(4.0, D(df)[3]);
            Assert.IsTrue(M(df).SequenceEqual(new[] { true, true, false, false }));

            var df2 = np.ma.diff(np.ma.array(np.array(new long[] { 1, 2, 4, 7, 11 })), n: 2);
            Assert.IsTrue(D(df2).SequenceEqual(new double[] { 1, 1, 1 }));
            Assert.IsFalse(M(df2).Any(v => v));

            var ap = np.ma.append(np.ma.array(np.array(new long[] { 1, 2 }), np.array(new[] { false, true })),
                                  np.ma.array(np.array(new long[] { 3, 4 }), np.array(new[] { true, false })));
            Assert.IsTrue(M(ap).SequenceEqual(new[] { false, true, true, false }));
        }

        /// <summary>sort/argsort endwith=False sends masked entries to the FRONT (filled with the dtype's
        /// smallest key), the mirror of the endwith=True default.</summary>
        [TestMethod]
        public void Sort_Argsort_Endwith()
        {
            var s = Ma(new double[] { 3, 1, 2 }, new[] { false, true, false });
            var sEnd = np.ma.sort(s, endwith: true);
            Assert.AreEqual(2.0, D(sEnd)[0]);
            Assert.AreEqual(3.0, D(sEnd)[1]);
            Assert.IsTrue(M(sEnd).SequenceEqual(new[] { false, false, true }));

            var sFront = np.ma.sort(s, endwith: false);
            Assert.IsTrue(M(sFront).SequenceEqual(new[] { true, false, false }));
            Assert.AreEqual(2.0, D(sFront)[1]);
            Assert.AreEqual(3.0, D(sFront)[2]);

            Assert.IsTrue(np.ma.argsort(s, endwith: false).astype(np.int64).ToArray<long>().SequenceEqual(new long[] { 1, 2, 0 }));
            Assert.IsTrue(np.ma.argsort(s, endwith: true).astype(np.int64).ToArray<long>().SequenceEqual(new long[] { 2, 0, 1 }));
        }

        /// <summary>average keepdims keeps the reduced axis; average_returned yields (avg, sum-of-weights) —
        /// the sum-of-weights being the unmasked count (uniform) or Σweights (weighted).</summary>
        [TestMethod]
        public void Average_Keepdims_And_Returned()
        {
            var av = np.ma.average(np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), np.array(new bool[,] { { false, true }, { false, false } })),
                                   axis: 1, keepdims: true);
            Assert.IsTrue(av.shape.SequenceEqual(new long[] { 2, 1 }));
            Assert.AreEqual(1.0, av.data.GetDouble(0)); // row0: only col0 unmasked
            Assert.AreEqual(3.5, av.data.GetDouble(1));

            var (avg, sws) = np.ma.average_returned(Ma(new double[] { 1, 2, 3 }, new[] { false, false, true }));
            Assert.AreEqual(1.5, avg.data.GetDouble(0));
            Assert.AreEqual(2.0, sws.GetDouble(0)); // 2 unmasked

            var (avgW, swsW) = np.ma.average_returned(Ma(new double[] { 1, 2, 3 }, new[] { false, false, true }),
                                                      weights: np.array(new double[] { 1, 2, 3 }));
            Assert.AreEqual(5.0 / 3.0, avgW.data.GetDouble(0), 1e-12);
            Assert.AreEqual(3.0, swsW.GetDouble(0)); // 1 + 2 (masked weight dropped)
        }

        /// <summary>median keepdims on the flat path yields an all-ones shape of the input rank.</summary>
        [TestMethod]
        public void Median_Keepdims()
        {
            var m = np.ma.median(np.array(new double[] { 1, 2, 3 }), keepdims: true);
            Assert.IsTrue(m.shape.SequenceEqual(new long[] { 1 }));
            Assert.AreEqual(2.0, m.data.GetDouble(0));
        }

        /// <summary>array(dtype=…) casts the data; isin(assume_unique=…) is accepted (result unchanged).</summary>
        [TestMethod]
        public void Array_Dtype_And_Isin_AssumeUnique()
        {
            Assert.AreEqual(np.float32, np.ma.array(np.array(new double[] { 1, 2, 3 }), np.array(new[] { false, true, false }), dtype: np.float32).dtype);
            var r = np.ma.isin(Ma(new double[] { 1, 2, 3 }, new[] { false, true, false }), np.array(new double[] { 1, 3 }), assume_unique: true);
            Assert.IsTrue(np.ma.getdata(r).ToArray<bool>().SequenceEqual(new[] { true, false, true }));
        }

        /// <summary>The %/&amp;/|/^/~ operators wire to the np.ma.* funcs (mask = OR of operands', division-domain
        /// for %; invert carries the mask through).</summary>
        [TestMethod]
        public void Operators_Modulo_Bitwise_Invert()
        {
            NDMaskedArray L(long[] d, bool[] m) => np.ma.array(np.array(d), np.array(m));
            var modr = L(new long[] { 7, 8, 9 }, new[] { false, true, false }) % L(new long[] { 3, 5, 2 }, new[] { false, false, true });
            Assert.AreEqual(1.0, D(modr)[0]);
            Assert.IsTrue(M(modr).SequenceEqual(new[] { false, true, true }));

            var andr = L(new long[] { 6, 7 }, new[] { false, true }) & L(new long[] { 3, 5 }, new[] { false, false });
            Assert.AreEqual(2.0, D(andr)[0]);
            Assert.IsTrue(M(andr).SequenceEqual(new[] { false, true }));

            var orr = L(new long[] { 6, 7 }, new[] { false, true }) | L(new long[] { 1, 8 }, new[] { false, false });
            Assert.AreEqual(7.0, D(orr)[0]);

            var xorr = L(new long[] { 6, 7 }, new[] { false, true }) ^ L(new long[] { 3, 5 }, new[] { false, false });
            Assert.AreEqual(5.0, D(xorr)[0]);

            var inv = ~L(new long[] { 6, 7 }, new[] { false, true });
            Assert.AreEqual(-7.0, D(inv)[0]);
            Assert.IsTrue(M(inv).SequenceEqual(new[] { false, true }));
        }

        /// <summary>Instance surface: compressed/ravel/reshape/T/clip/round/item/real/imag mirror the module
        /// funcs; item on a masked singleton returns the masked constant.</summary>
        [TestMethod]
        public void InstanceSurface_WiresThrough()
        {
            var inst = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, true });
            Assert.IsTrue(inst.compressed().astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 1, 3 }));
            Assert.AreEqual(4L, inst.ravel().size);
            Assert.IsTrue(inst.reshape(2, 2).shape.SequenceEqual(new long[] { 2, 2 }));

            var mat = np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), np.array(new bool[,] { { false, true }, { false, false } }));
            Assert.IsTrue(np.ma.getmaskarray(mat.T).ToArray<bool>().SequenceEqual(new[] { false, false, true, false }));

            Assert.AreEqual(4, D(inst.clip(2, 3)).Length);
            Assert.AreEqual(4L, inst.round().size);

            Assert.IsTrue(ReferenceEquals(np.ma.array(np.array(new double[] { 5 }), np.array(new[] { true })).item(), np.ma.NDMasked));
            Assert.AreEqual(5.0, Convert.ToDouble(np.ma.array(np.array(new double[] { 5 }), np.array(new[] { false })).item()));

            var cplx = np.ma.array(np.array(new System.Numerics.Complex[] { new System.Numerics.Complex(1, 2), new System.Numerics.Complex(3, 4) }), np.array(new[] { false, true }));
            Assert.IsTrue(cplx.real.data.astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 1, 3 }));
            Assert.IsTrue(cplx.imag.data.astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 2, 4 }));
        }

        // ── The indexer (§6 item 6, the keystone) + put/putmask/resize/mask_rowcols/compress_rowcols/
        //    notmasked_*/ids that it unblocks. Every expected value probed against NumPy 2.4.2. ──

        /// <summary>Indexer GET: a scalar index yields the bare value (unmasked) or the masked singleton; a slice
        /// yields a sub-NDMaskedArray VIEW that writes through; boolean/fancy indices yield COPIES; 2-D reduces per axis.</summary>
        [TestMethod]
        public void Indexer_Get()
        {
            var x = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, true });
            Assert.AreEqual(1.0, Convert.ToDouble(x[0]));                 // unmasked scalar
            Assert.IsTrue(ReferenceEquals(x[1], np.ma.NDMasked));        // masked scalar → singleton

            var xs = (NDMaskedArray)x["1:3"];                            // slice → sub-NDMaskedArray
            Assert.IsTrue(D(xs).SequenceEqual(new double[] { 2, 3 }));
            Assert.IsTrue(M(xs).SequenceEqual(new[] { true, false }));
            xs[0] = 99.0;                                                // VIEW: writes through to x[1] + unmasks
            Assert.AreEqual(99.0, np.ma.getdata(x).astype(np.float64).ToArray<double>()[1]);
            Assert.IsTrue(M(x).SequenceEqual(new[] { false, false, false, true }));

            var x2 = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, true });
            var bsel = (NDMaskedArray)x2[np.array(new[] { true, false, true, false })];
            Assert.IsTrue(D(bsel).SequenceEqual(new double[] { 1, 3 }) && !M(bsel).Any(v => v));
            var fsel = (NDMaskedArray)x2[np.array(new[] { 1, 3 })];
            Assert.IsTrue(M(fsel).SequenceEqual(new[] { true, true }));

            var m2 = np.ma.array(np.array(new double[,] { { 1, 2 }, { 3, 4 } }), np.array(new bool[,] { { false, true }, { false, false } }));
            var row = (NDMaskedArray)m2[0];
            Assert.IsTrue(D(row).SequenceEqual(new double[] { 1, 2 }) && M(row).SequenceEqual(new[] { false, true }));
            Assert.IsTrue(ReferenceEquals(m2[0, 1], np.ma.NDMasked));    // masked element
            Assert.AreEqual(3.0, Convert.ToDouble(m2[1, 0]));            // unmasked element
        }

        /// <summary>Indexer SET: a plain value writes+UNMASKS; <c>= masked</c> masks in place (data untouched);
        /// a masked-array value propagates its mask; masking a nomask array creates the mask.</summary>
        [TestMethod]
        public void Indexer_Set()
        {
            var y = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, true });
            y[0] = 100.0;
            Assert.IsTrue(D(y)[0] == 100.0 && M(y)[0] == false);
            y[1] = 200.0;                                                // masked slot → value UNMASKS
            Assert.IsTrue(D(y)[1] == 200.0 && M(y)[1] == false);
            y[2] = np.ma.NDMasked;                                      // mask (data unchanged)
            Assert.IsTrue(D(y)[2] == 3.0 && M(y)[2] == true);

            var z = np.ma.array(np.array(new double[] { 1, 2, 3 }));    // nomask
            Assert.IsTrue(ReferenceEquals(np.ma.getmask(z), np.ma.nomask));
            z[1] = np.ma.NDMasked;                                      // creates a mask
            Assert.IsTrue(M(z).SequenceEqual(new[] { false, true, false }));

            var w = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, true });
            w["1:3"] = 0.0;                                             // slice assign unmasks
            Assert.IsTrue(D(w).SequenceEqual(new double[] { 1, 0, 0, 4 }) && M(w).SequenceEqual(new[] { false, false, false, true }));

            var v = np.ma.array(np.array(new double[] { 1, 2, 3, 4 }));
            v["0:2"] = np.ma.array(np.array(new double[] { 10, 20 }), np.array(new[] { true, false }));
            Assert.IsTrue(M(v).SequenceEqual(new[] { true, false, false, false }) && D(v)[1] == 20.0);
        }

        /// <summary>put scatters into the flat data (cycling short values); masked values mask the slots, a
        /// masked-array propagates, a plain value unmasks.</summary>
        [TestMethod]
        public void Put_MaskAware()
        {
            var pa = Ma(new double[] { 1, 2, 3, 4, 5 }, new[] { false, true, false, true, false });
            np.ma.put(pa, np.array(new[] { 0, 2 }), np.array(new double[] { 100, 300 }));
            Assert.IsTrue(D(pa).SequenceEqual(new double[] { 100, 2, 300, 4, 5 }));
            Assert.IsTrue(M(pa).SequenceEqual(new[] { false, true, false, true, false }));

            var pb = np.ma.array(np.array(new double[] { 1, 2, 3, 4, 5 }));
            np.ma.put(pb, np.array(new[] { 1, 3 }), np.ma.NDMasked);
            Assert.IsTrue(M(pb).SequenceEqual(new[] { false, true, false, true, false }));

            var pc = np.ma.array(np.array(new double[] { 1, 2, 3, 4, 5 }));
            np.ma.put(pc, np.array(new[] { 1, 3 }), np.ma.array(np.array(new double[] { 20, 40 }), np.array(new[] { true, false })));
            Assert.IsTrue(M(pc).SequenceEqual(new[] { false, true, false, false, false }) && D(pc)[3] == 40.0);
        }

        /// <summary>putmask writes values where the mask is True and unmasks those slots (masked-off slots
        /// untouched, so a pre-masked one stays masked).</summary>
        [TestMethod]
        public void PutMask_MaskAware()
        {
            var d = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, false });
            np.ma.putmask(d, np.array(new[] { true, false, true, false }), np.array(new double[] { 10, 20, 30, 40 }));
            Assert.IsTrue(D(d).SequenceEqual(new double[] { 10, 2, 30, 4 }));
            Assert.IsTrue(M(d).SequenceEqual(new[] { false, true, false, false }));
        }

        /// <summary>resize tiles BOTH data and mask into the new shape (never in place); shrinking truncates.</summary>
        [TestMethod]
        public void Resize_TilesDataAndMask()
        {
            var e = Ma(new double[] { 1, 2, 3 }, new[] { false, true, false });
            var rl = np.ma.resize(e, new Shape(2, 3));
            Assert.IsTrue(rl.shape.SequenceEqual(new long[] { 2, 3 }));
            Assert.IsTrue(M(rl).SequenceEqual(new[] { false, true, false, false, true, false }));
            var rs = np.ma.resize(e, new Shape(2));
            Assert.IsTrue(D(rs).SequenceEqual(new double[] { 1, 2 }) && M(rs).SequenceEqual(new[] { false, true }));
        }

        private static NDMaskedArray G() => np.ma.array(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }),
                                                      np.array(new bool[,] { { false, false, false }, { false, true, false }, { false, false, false } }));

        /// <summary>mask_rowcols masks whole rows and/or cols (from the ORIGINAL mask) containing any masked
        /// element: axis=None both, 0 rows, 1 cols.</summary>
        [TestMethod]
        public void MaskRowcols()
        {
            Assert.IsTrue(M(np.ma.mask_rowcols(G())).SequenceEqual(new[] { false, true, false, true, true, true, false, true, false }));
            Assert.IsTrue(M(np.ma.mask_rowcols(G(), 0)).SequenceEqual(new[] { false, false, false, true, true, true, false, false, false }));
            Assert.IsTrue(M(np.ma.mask_rowcols(G(), 1)).SequenceEqual(new[] { false, true, false, false, true, false, false, true, false }));
        }

        /// <summary>compress_rowcols drops rows and/or cols with any masked value, returning a plain array.</summary>
        [TestMethod]
        public void CompressRowcols()
        {
            Assert.IsTrue(np.ma.compress_rowcols(G()).astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 1, 3, 7, 9 }));
            Assert.IsTrue(np.ma.compress_rowcols(G(), 0).astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 1, 2, 3, 7, 8, 9 }));
        }

        /// <summary>notmasked_edges gives the first/last unmasked flat indices; notmasked_contiguous the unmasked
        /// runs; ids returns the (data, mask) buffer addresses (mask 0 for nomask).</summary>
        [TestMethod]
        public void NotmaskedEdges_Contiguous_Ids()
        {
            var h = Ma(new double[] { 1, 2, 3, 4, 5 }, new[] { true, false, false, true, false });
            // The flat/1-D forms return long[] / Slice[] boxed as object (NumPy's return is polymorphic).
            Assert.IsTrue(((long[])np.ma.notmasked_edges(h)).SequenceEqual(new long[] { 1, 4 }));
            Assert.AreEqual(2, ((Slice[])np.ma.notmasked_contiguous(h)).Length);

            var (dp, mp) = np.ma.ids(h);
            Assert.AreNotEqual(0L, dp);
            Assert.AreNotEqual(0L, mp);
            Assert.AreEqual(0L, np.ma.ids(np.ma.array(np.array(new double[] { 1, 2 }))).mask); // nomask → 0
        }

        /// <summary>frombuffer/fromfunction build UNMASKED arrays; hsplit splits data AND mask alike;
        /// ndenumerate yields (index, value) for the UNMASKED elements only.</summary>
        [TestMethod]
        public void QuickWins_Frombuffer_Fromfunction_Hsplit_Ndenumerate()
        {
            var buf = np.array(new double[] { 1, 2, 3, 4 }).tobytes();
            var fb = np.ma.frombuffer(buf, np.float64);
            Assert.IsTrue(D(fb).SequenceEqual(new double[] { 1, 2, 3, 4 }) && !M(fb).Any(v => v));

            var ff = np.ma.fromfunction((i, j) => i + j, new Shape(2, 3), np.int64);
            Assert.IsTrue(D(ff).SequenceEqual(new double[] { 0, 1, 2, 1, 2, 3 }) && !M(ff).Any(v => v));

            var g = np.ma.array(np.array(new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }),
                                np.array(new bool[,] { { false, true, false, false }, { false, false, false, true } }));
            var parts = np.ma.hsplit(g, 2);
            Assert.AreEqual(2, parts.Length);
            Assert.IsTrue(M(parts[0]).SequenceEqual(new[] { false, true, false, false }));
            Assert.IsTrue(M(parts[1]).SequenceEqual(new[] { false, false, false, true }));

            var x = Ma(new double[] { 1, 2, 3 }, new[] { false, true, false });
            var pairs = np.ma.ndenumerate(x).ToList();
            Assert.AreEqual(2, pairs.Count); // the masked element is skipped
            Assert.IsTrue(pairs[0].index.SequenceEqual(new long[] { 0 }) && Convert.ToDouble(pairs[0].value) == 1.0);
            Assert.IsTrue(pairs[1].index.SequenceEqual(new long[] { 2 }) && Convert.ToDouble(pairs[1].value) == 3.0);
        }

        /// <summary>fromfunction's DEFAULT grid dtype is float64 — NumPy's <c>fromfunction(..., dtype=float)</c>
        /// (probed 2.4.2: <c>ma.fromfunction(lambda i, j: i + j, (2, 3))</c> is float64 <c>[[0,1,2],[1,2,3]]</c>).
        /// The default used to fall through to np.indices' int64, so the plain spelling returned int64.</summary>
        [TestMethod]
        public void Fromfunction_DefaultDtype_IsFloat64LikeNumPy()
        {
            var ff = np.ma.fromfunction((i, j) => i + j, new Shape(2, 3));
            Assert.AreEqual(np.float64, ff.dtype);
            Assert.IsTrue(ff.data.ToArray<double>().SequenceEqual(new double[] { 0, 1, 2, 1, 2, 3 }));
        }

        /// <summary>MAError/MaskError exception types match NumPy's hierarchy (MaskError : MAError : Exception).</summary>
        [TestMethod]
        public void ExceptionTypes_Hierarchy()
        {
            Assert.IsTrue(new MaskError("x") is MAError);
            Assert.IsTrue(new MAError("x") is Exception);
        }

        /// <summary>cov is PAIRWISE-COMPLETE over the mask (each entry divides by the count of both-unmasked
        /// observations); corrcoef normalizes it by the outer product of the per-variable std deviations.</summary>
        [TestMethod]
        public void Cov_And_Corrcoef_PairwiseComplete()
        {
            var x = np.ma.array(np.array(new double[,] { { 1, 2, 3, 4 }, { 2, 4, 6, 8 } }));
            Assert.IsTrue(D(np.ma.cov(x)).Zip(new double[] { 5.0 / 3, 10.0 / 3, 10.0 / 3, 20.0 / 3 }, (a, e) => System.Math.Abs(a - e) < 1e-9).All(v => v));

            // pairwise mask: dropping x[0,2] changes only the pairs that use it.
            var xm = np.ma.array(np.array(new double[,] { { 1, 2, 3, 4 }, { 2, 4, 6, 8 } }),
                                 np.array(new bool[,] { { false, false, true, false }, { false, false, false, false } }));
            Assert.IsTrue(D(np.ma.cov(xm)).Zip(new double[] { 7.0 / 3, 14.0 / 3, 14.0 / 3, 20.0 / 3 }, (a, e) => System.Math.Abs(a - e) < 1e-9).All(v => v));
            Assert.IsFalse(M(np.ma.cov(xm)).Any(v => v));

            Assert.IsTrue(D(np.ma.cov(x, bias: true)).Zip(new double[] { 1.25, 2.5, 2.5, 5.0 }, (a, e) => System.Math.Abs(a - e) < 1e-9).All(v => v));

            // rowvar=False: variables are columns.
            var xt = np.ma.array(np.array(new double[,] { { 1, 2 }, { 2, 4 }, { 3, 6 }, { 4, 8 } }));
            Assert.IsTrue(D(np.ma.cov(xt, rowvar: false)).Zip(new double[] { 5.0 / 3, 10.0 / 3, 10.0 / 3, 20.0 / 3 }, (a, e) => System.Math.Abs(a - e) < 1e-9).All(v => v));

            // corrcoef: perfectly correlated variables → all 1.
            Assert.IsTrue(D(np.ma.corrcoef(x)).All(v => System.Math.Abs(v - 1.0) < 1e-9));
        }

        /// <summary>convolve/correlate propagate the mask by sliding the boolean masks against ones: with
        /// propagate_mask a result is masked if ANY masked cell contributed; without it, only if NO unmasked
        /// cell did (and the data comes from the 0-filled inputs).</summary>
        [TestMethod]
        public void Convolve_And_Correlate_MaskPropagation()
        {
            var a = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, false });
            var v = np.array(new double[] { 0.5, 0.5 });

            var cp = np.ma.convolve(a, v);
            Assert.IsTrue(D(cp).Zip(new double[] { 0.5, 1.5, 2.5, 3.5, 2.0 }, (x, e) => System.Math.Abs(x - e) < 1e-9).All(b => b));
            Assert.IsTrue(M(cp).SequenceEqual(new[] { false, true, true, false, false }));

            var cn = np.ma.convolve(a, v, propagate_mask: false);
            Assert.IsTrue(D(cn).Zip(new double[] { 0.5, 0.5, 1.5, 3.5, 2.0 }, (x, e) => System.Math.Abs(x - e) < 1e-9).All(b => b));
            Assert.IsFalse(M(cn).Any(x => x));

            var cr = np.ma.correlate(a, v, mode: "full");
            Assert.IsTrue(M(cr).SequenceEqual(new[] { false, true, true, false, false }));
        }

        /// <summary>apply_over_axes applies a masked reduction over each axis in turn, re-expanding a
        /// keepdims-less result so the passes compose (NumPy's ma.apply_over_axes).</summary>
        [TestMethod]
        public void ApplyOverAxes()
        {
            var data = np.arange(24).reshape(2, 3, 4).astype(np.float64);
            var mask = np.array(Enumerable.Range(0, 24).Select(i => i % 5 == 0).ToArray()).reshape(2, 3, 4);
            var a = np.ma.array(data, mask);
            var r = np.ma.apply_over_axes((m, ax) => np.ma.sum(m, ax), a, new[] { 0, 2 });
            Assert.IsTrue(r.shape.SequenceEqual(new long[] { 1, 3, 1 }));
            Assert.IsTrue(np.ma.getdata(r).astype(np.float64).ToArray<double>().SequenceEqual(new double[] { 45, 87, 94 }));
            Assert.IsFalse(np.ma.getmaskarray(r).ToArray<bool>().Any(x => x));
        }

        /// <summary>apply_along_axis hands each 1-D masked slice to the function and assembles the results — a
        /// scalar result drops the axis (an all-masked slice → masked), a 1-D result replaces it (mask preserved).</summary>
        [TestMethod]
        public void ApplyAlongAxis()
        {
            var data = np.arange(12).reshape(3, 4).astype(np.float64);
            var mask = np.array(Enumerable.Range(0, 12).Select(i => i % 4 == 1).ToArray()).reshape(3, 4);
            var a = np.ma.array(data, mask);

            // scalar result: sum along axis 1 / axis 0 (all-masked column → masked).
            var s1 = np.ma.apply_along_axis(m => np.ma.sum(m), 1, a);
            Assert.IsTrue(D(s1).SequenceEqual(new double[] { 5, 17, 29 }) && s1.shape.SequenceEqual(new long[] { 3 }));
            var s0 = np.ma.apply_along_axis(m => np.ma.sum(m), 0, a);
            Assert.IsTrue(D(s0).SequenceEqual(new double[] { 12, 0, 18, 21 }));
            Assert.IsTrue(M(s0)[1]); // the all-masked column reduces to masked

            // 1-D result: cumsum along axis 1 keeps the masked positions masked.
            var c1 = np.ma.apply_along_axis(m => np.ma.cumsum(m), 1, a);
            Assert.IsTrue(c1.shape.SequenceEqual(new long[] { 3, 4 }));
            Assert.IsTrue(D(c1).SequenceEqual(new double[] { 0, 0, 2, 5, 4, 4, 10, 17, 8, 8, 18, 29 }));
            Assert.IsTrue(M(c1).SequenceEqual(new[] { false, true, false, false, false, true, false, false, false, true, false, false }));
        }

        /// <summary>Per-slice masked median along an explicit axis (NumPy's <c>ma.median</c> axis branch): masked
        /// entries are sorted behind the valid ones, so the middle is taken over each slice's UNMASKED count; an
        /// all-masked slice → masked, an integer input promotes to float64, an unmasked NaN forces the slice's
        /// median to NaN, and keepdims re-inserts the reduced axis.</summary>
        [TestMethod]
        public void MedianMaskedAxis()
        {
            var a = np.ma.array(np.array(new double[,] { { 3, 1, 2, 8 }, { 9, 5, 4, 7 }, { 1, 2, 3, 4 } }),
                                np.array(new bool[,] { { false, true, false, false }, { false, false, true, false }, { true, true, true, true } }));
            // axis 1: median of {3,2,8}=3, {9,5,7}=7, all-masked → masked.
            var r1 = np.ma.median(a, 1);
            Assert.IsTrue(D(r1).SequenceEqual(new double[] { 3, 7, 0 }));
            Assert.IsTrue(M(r1).SequenceEqual(new[] { false, false, true }));
            // axis 0: columns, the fully-masked slice contributes nothing.
            var r0 = np.ma.median(a, 0);
            Assert.IsTrue(D(r0).SequenceEqual(new double[] { 6, 5, 2, 7.5 }));
            Assert.IsFalse(M(r0).Any(v => v));
            // negative axis and keepdims.
            Assert.IsTrue(D(np.ma.median(a, -1)).SequenceEqual(new double[] { 3, 7, 0 }));
            Assert.IsTrue(np.ma.median(a, 1, keepdims: true).shape.SequenceEqual(new long[] { 3, 1 }));

            // integer input → float64.
            var ai = np.ma.array(np.array(new int[,] { { 3, 1, 2, 8 }, { 9, 5, 4, 7 } }),
                                 np.array(new bool[,] { { false, true, false, false }, { false, false, false, false } }));
            var ri = np.ma.median(ai, 1);
            Assert.IsTrue(D(ri).SequenceEqual(new double[] { 3, 6 }));
            Assert.AreEqual(np.float64, ri.data.dtype);

            // an unmasked NaN in a slice makes that slice's median NaN.
            var an = np.ma.array(np.array(new double[,] { { 1, double.NaN, 3 }, { 4, 5, 6 } }),
                                 np.array(new bool[,] { { false, false, false }, { false, true, false } }));
            var rn = D(np.ma.median(an, 1));
            Assert.IsTrue(double.IsNaN(rn[0]) && rn[1] == 5.0);

            // a 1-D masked array with axis=0 (the general path also handles rank 1).
            var d1 = Ma(new double[] { 3, 1, 2, 8, 5 }, new[] { false, true, false, false, false });
            Assert.AreEqual(4.0, np.ma.median(d1, 0).data.GetDouble(0));
        }

        /// <summary>ma.dot's strict flag PROPAGATES the mask along the contracted axes first — any masked value in a
        /// row/column masks that whole vector — so a result entry is masked whenever a masked value touched it
        /// (NumPy's <c>ma.dot(..., strict=True)</c>); the default treats masked as 0.</summary>
        [TestMethod]
        public void Dot_Strict()
        {
            var a = np.ma.array(np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }),
                                np.array(new bool[,] { { true, false, false }, { false, false, false } }));
            var b = np.ma.array(np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }),
                                np.array(new bool[,] { { true, false }, { false, false }, { false, false } }));
            var loose = np.ma.dot(a, b);
            Assert.IsTrue(D(loose).SequenceEqual(new double[] { 21, 26, 45, 64 }));
            Assert.IsFalse(M(loose).Any(v => v));
            // strict: a's row 0 and b's column 0 are tainted, so only result[1,1] survives.
            Assert.IsTrue(M(np.ma.dot(a, b, strict: true)).SequenceEqual(new[] { true, true, true, false }));
        }

        /// <summary>ma.take honors the out-of-bounds mode (wrap/clip) on BOTH the data and the mask gather, so a
        /// wrapped index carries the wrapped element's masked-ness (NumPy's <c>ma.take(..., mode=…)</c>).</summary>
        [TestMethod]
        public void Take_Mode()
        {
            var t = Ma(new double[] { 10, 20, 30, 40 }, new[] { false, true, false, false });
            var idx = np.array(new int[] { 0, 3, 5 });
            var w = np.ma.take(t, idx, mode: "wrap");   // index 5 wraps to 1 (masked)
            Assert.IsTrue(D(w).SequenceEqual(new double[] { 10, 40, 20 }));
            Assert.IsTrue(M(w).SequenceEqual(new[] { false, false, true }));
            var c = np.ma.take(t, idx, mode: "clip");   // index 5 clips to 3 (unmasked)
            Assert.IsTrue(D(c).SequenceEqual(new double[] { 10, 40, 40 }));
            Assert.IsFalse(M(c).Any(v => v));
        }

        /// <summary>ma.squeeze with an explicit axis drops ONLY that size-1 axis (mask squeezed alike), matching
        /// NumPy's <c>squeeze(axis=…)</c>; the default drops every size-1 axis.</summary>
        [TestMethod]
        public void Squeeze_Axis()
        {
            var sq = np.ma.array(np.array(new int[, ,] { { { 1 }, { 2 } } }),   // shape (1, 2, 1)
                                 np.array(new bool[, ,] { { { false }, { true } } }));
            Assert.IsTrue(np.ma.squeeze(sq, 0).shape.SequenceEqual(new long[] { 2, 1 }));
            Assert.IsTrue(np.ma.squeeze(sq, 2).shape.SequenceEqual(new long[] { 1, 2 }));
            Assert.IsTrue(np.ma.squeeze(sq).shape.SequenceEqual(new long[] { 2 })); // drop all size-1 axes
        }

        /// <summary>ma.polyfit DROPS the masked observations, then fits the survivors through <c>np.polyfit</c>
        /// (which needs the LAPACK backend). A clean line with one masked outlier recovers slope 2, intercept 1.
        /// Skips loudly if no LAPACK-capable BLAS is present.</summary>
        [TestMethod]
        public void Polyfit_DropsMaskedObservations()
        {
            try { OpenBlasEngine.Enable(); }
            catch (Exception e) { Assert.Inconclusive("no CBLAS on this host: " + e.Message.Split('\n')[0]); }
            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("the loaded BLAS exports no LAPACK lstsq (a bare reference CBLAS).");
            try
            {
                // The masked entry at index 2 (data 999) is a wild outlier; dropping it recovers y = 2x + 1.
                var x = np.ma.array(np.array(new double[] { 0, 1, 2, 3, 4, 5 }), np.array(new bool[] { false, false, true, false, false, false }));
                var y = np.ma.array(np.array(new double[] { 1, 3, 999, 7, 9, 11 }), np.array(new bool[] { false, false, true, false, false, false }));
                NDArray c = np.ma.polyfit(x, y, 1);
                var cv = c.ravel().ToArray<double>();
                Assert.AreEqual(2.0, cv[0], 1e-9);
                Assert.AreEqual(1.0, cv[1], 1e-9);
            }
            finally { OpenBlasEngine.Disable(); }
        }

        /// <summary>notmasked_edges with an explicit axis on a >1-D array returns NumPy's <c>[tuple(mins),
        /// tuple(maxs)]</c> — one compressed first/last-unmasked coordinate array per dimension (returned as
        /// NDArray[][] boxed in object, mirroring NumPy's polymorphic list return).</summary>
        [TestMethod]
        public void NotmaskedEdges_NDAxis()
        {
            // 3x3 with [1:,1:] masked → unmasked coords {(0,0),(0,1),(0,2),(1,0),(2,0)}.
            var a = np.arange(9).reshape(3, 3);
            var m = np.zeros(new Shape(3, 3), np.@bool);
            m[1, 1] = true; m[1, 2] = true; m[2, 1] = true; m[2, 2] = true;
            var am = np.ma.array(a, m);

            var e0 = (NDArray[][])np.ma.notmasked_edges(am, 0);
            Assert.IsTrue(e0[0][0].ToArray<long>().SequenceEqual(new long[] { 0, 0, 0 }));   // mins, dim0
            Assert.IsTrue(e0[0][1].ToArray<long>().SequenceEqual(new long[] { 0, 1, 2 }));   // mins, dim1
            Assert.IsTrue(e0[1][0].ToArray<long>().SequenceEqual(new long[] { 2, 0, 0 }));   // maxs, dim0
            Assert.IsTrue(e0[1][1].ToArray<long>().SequenceEqual(new long[] { 0, 1, 2 }));   // maxs, dim1

            var e1 = (NDArray[][])np.ma.notmasked_edges(am, 1);
            Assert.IsTrue(e1[0][0].ToArray<long>().SequenceEqual(new long[] { 0, 1, 2 }));
            Assert.IsTrue(e1[1][0].ToArray<long>().SequenceEqual(new long[] { 0, 1, 2 }));
            Assert.IsTrue(e1[1][1].ToArray<long>().SequenceEqual(new long[] { 2, 0, 0 }));

            // The flat form still yields the plain [first, last] (boxed).
            Assert.IsTrue(((long[])np.ma.notmasked_edges(am)).SequenceEqual(new long[] { 0, 6 }));
        }

        /// <summary>notmasked_contiguous with an explicit axis on a 2-D array returns NumPy's list-of-lists — one
        /// inner Slice[] of contiguous unmasked runs per line along the OTHER axis (empty for a fully-masked
        /// line); >2-D raises NumPy's "Currently limited to at most 2D array." (NotSupportedException here).</summary>
        [TestMethod]
        public void NotmaskedContiguous_2DAxis()
        {
            var a = np.arange(12).reshape(3, 4);
            var mask = np.zeros(new Shape(3, 4), np.@bool);
            mask[1, 0] = true; mask[1, 1] = true; mask[1, 2] = true; mask[2, 1] = true; mask[2, 2] = true; mask[0, 1] = true;
            var ma = np.ma.array(a, mask);

            var c0 = (Slice[][])np.ma.notmasked_contiguous(ma, 0);  // per column
            Assert.AreEqual(4, c0.Length);
            Assert.AreEqual("0:1,2:3", string.Join(",", c0[0].Select(s => $"{s.Start}:{s.Stop}")));
            Assert.AreEqual(0, c0[1].Length);                       // fully-masked column → []
            Assert.AreEqual("0:1", string.Join(",", c0[2].Select(s => $"{s.Start}:{s.Stop}")));
            Assert.AreEqual("0:3", string.Join(",", c0[3].Select(s => $"{s.Start}:{s.Stop}")));

            var c1 = (Slice[][])np.ma.notmasked_contiguous(ma, 1);  // per row
            Assert.AreEqual(3, c1.Length);
            Assert.AreEqual("0:1,2:4", string.Join(",", c1[0].Select(s => $"{s.Start}:{s.Stop}")));
            Assert.AreEqual("3:4", string.Join(",", c1[1].Select(s => $"{s.Start}:{s.Stop}")));

            Assert.ThrowsException<NotSupportedException>(
                () => np.ma.notmasked_contiguous(np.ma.array(np.arange(8).reshape(2, 2, 2)), 0));
        }

        /// <summary>ma.power's third (modulus) argument is rejected with NumPy's MaskError "3-argument power not
        /// supported." — the slot exists only for signature parity.</summary>
        [TestMethod]
        public void Power_ThirdArg_Throws()
        {
            var ex = Assert.ThrowsException<MaskError>(() => np.ma.power(np.ma.array(np.array(new int[] { 2, 3 })), 2, 3));
            Assert.AreEqual("3-argument power not supported.", ex.Message);
        }

        /// <summary>var's NumPy-2.0 `mean` param centers by the SUPPLIED mean instead of the slice mean; std
        /// ACCEPTS `mean` but IGNORES it (a NumPy quirk — ma.std forwards only keepdims to var, so ma.std(mean=X)
        /// is the PLAIN std). Both are reproduced exactly.</summary>
        [TestMethod]
        public void VarStd_MeanParam()
        {
            var a = Ma(new double[] { 1, 2, 3, 4 }, new[] { false, true, false, false }); // [1,--,3,4]
            // var centered by 2.0: devs [-1,_,1,2] → (1+1+4)/3 = 2.0.
            Assert.AreEqual(2.0, np.ma.var(a, mean: 2.0).data.GetDouble(0), 1e-12);
            // std IGNORES mean → plain std = sqrt(var()) (NOT sqrt(var(mean=2.0))).
            double plain = np.ma.std(a).data.GetDouble(0);
            Assert.AreEqual(plain, np.ma.std(a, mean: 2.0).data.GetDouble(0), 1e-12);
            Assert.AreNotEqual(System.Math.Sqrt(2.0), np.ma.std(a, mean: 2.0).data.GetDouble(0), 1e-6);

            // axis form: per-row keepdims mean centers each row.
            var d = np.ma.array(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }),
                                np.array(new bool[,] { { false, true, false }, { false, false, false } }));
            var vr = np.ma.var(d, 1, mean: np.array(new double[,] { { 2.0 }, { 5.0 } })).data.astype(np.float64).ToArray<double>();
            Assert.AreEqual(1.0, vr[0], 1e-12);
            Assert.AreEqual(2.0 / 3.0, vr[1], 1e-12);

            // unregressed: plain var (no mean) unchanged.
            Assert.AreEqual(14.0 / 9.0, np.ma.var(a).data.GetDouble(0), 1e-12);
        }

        // ── Conversion / export + flag-surface instance members (pass 9). Every value probed vs NumPy 2.4.2. ──

        /// <summary>tolist returns the nested data with <c>null</c> at masked positions by default (NumPy's None),
        /// the fill substituted when given, the plain nested data for nomask, and a bare scalar/null for 0-D.</summary>
        [TestMethod]
        public void Tolist_NestedNullAtMasked_AndFill()
        {
            static string Str(object o) => o switch
            {
                null => "None",
                object[] arr => "[" + string.Join(", ", arr.Select(Str)) + "]",
                _ => System.Convert.ToString(o, System.Globalization.CultureInfo.InvariantCulture)
            };
            var d = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            var m = np.array(new bool[] { false, true, false, true, false, true, false, true, false }).reshape(3, 3);
            var x = np.ma.array(d, m);
            Assert.AreEqual("[[1, None, 3], [None, 5, None], [7, None, 9]]", Str(x.tolist()));
            Assert.AreEqual("[[1, -999, 3], [-999, 5, -999], [7, -999, 9]]", Str(x.tolist(-999)));
            Assert.AreEqual("[1.5, 2.5, 3.5]", Str(np.ma.array(np.array(new double[] { 1.5, 2.5, 3.5 })).tolist()));
            Assert.AreEqual("None", Str(np.ma.array(np.array(7), np.array(true)).tolist()));
            Assert.AreEqual("7", Str(np.ma.array(np.array(7), np.array(false)).tolist()));
        }

        /// <summary>tobytes fills masked slots FIRST (default fill or the given one) then encodes — byte-identical
        /// to NumPy; F-order reorders the (asymmetric) elements.</summary>
        [TestMethod]
        public void Tobytes_FillsThenEncodes()
        {
            var b = np.ma.array(np.array(new int[,] { { 1, 2 }, { 3, 4 } }),
                                np.array(new bool[] { false, true, true, false }).reshape(2, 2));
            Assert.AreEqual("010000003f420f003f420f0004000000", System.Convert.ToHexString(b.tobytes()).ToLowerInvariant());
            Assert.AreEqual("01000000ffffffffffffffff04000000", System.Convert.ToHexString(b.tobytes(-1)).ToLowerInvariant());
            // F-order of [[1,2],[3,4]] int32 (no mask): columns 1,3 then 2,4.
            var fo = np.ma.array(np.array(new int[,] { { 1, 2 }, { 3, 4 } }));
            Assert.AreEqual("01000000030000000200000004000000", System.Convert.ToHexString(fo.tobytes(order: 'F')).ToLowerInvariant());
        }

        /// <summary>view reinterprets the data dtype (same-itemsize keeps the shape, so the mask rides through and
        /// the bytes round-trip); an itemsize-changing view of a MASKED array is refused.</summary>
        [TestMethod]
        public void View_ReinterpretsDtype_MaskCarried()
        {
            var v = np.ma.array(np.array(new int[] { 1, 2, 3, 4 }), np.array(new bool[] { false, true, false, true }));
            var vv = v.view(np.float32);
            Assert.AreEqual(np.float32, vv.dtype);
            Assert.IsTrue(M(vv).SequenceEqual(new[] { false, true, false, true }));
            Assert.IsTrue(vv.data.view(np.int32).ToArray<int>().SequenceEqual(new[] { 1, 2, 3, 4 }));
            // Itemsize change (int32 -> int8) with a live mask: refused (the bool mask can't be reinterpreted).
            Assert.ThrowsException<NotSupportedException>(() => v.view(np.int8));
        }

        /// <summary>iscontiguous reads the CONTIGUOUS flag (True for a fresh 1-D, False for a transpose); ids
        /// returns nonzero data/mask pointers when masked and a zero mask pointer for nomask; baseclass/hardmask/
        /// recordmask report the fixed non-structured, soft-mask surface.</summary>
        [TestMethod]
        public void Flags_Ids_Iscontiguous_Baseclass()
        {
            Assert.IsTrue(np.ma.array(np.array(new int[] { 1, 2, 3 })).iscontiguous());
            Assert.IsFalse(np.ma.array(np.array(new int[,] { { 1, 2 }, { 3, 4 } })).T.iscontiguous());

            var h = np.ma.array(np.array(new int[] { 1, 2, 3 }), np.array(new bool[] { false, true, false }));
            var (dp, mp) = h.ids();
            Assert.IsTrue(dp != 0 && mp != 0);
            Assert.AreEqual(0L, np.ma.array(np.array(new int[] { 1, 2, 3 })).ids().mask); // nomask -> 0

            Assert.IsFalse(h.hardmask);
            Assert.IsTrue(h.recordmask.ToArray<bool>().SequenceEqual(new[] { false, true, false }));
            Assert.AreEqual(typeof(NDArray), h.baseclass);
        }

        /// <summary>Instance putmask writes values where the mask is True and UNMASKS those slots (plain values),
        /// leaving the others' masks intact — the mutating instance form.</summary>
        [TestMethod]
        public void Putmask_Instance_WritesAndUnmasks()
        {
            var p = np.ma.array(np.array(new int[] { 1, 2, 3, 4 }), np.array(new bool[] { false, true, false, true }));
            p.putmask(np.array(new bool[] { true, false, true, false }), np.array(new int[] { 10, 20, 30, 40 }));
            Assert.IsTrue(p.data.ToArray<int>().SequenceEqual(new[] { 10, 2, 30, 4 }));
            Assert.IsTrue(M(p).SequenceEqual(new[] { false, true, false, true }));
        }

        /// <summary>flat returns a raveled masked VIEW (NumSharp's NDArray.flat house convention, not NumPy's
        /// iterator) — data and mask flattened in C-order.</summary>
        [TestMethod]
        public void Flat_RaveledMaskedView()
        {
            var f = np.ma.array(np.array(new int[,] { { 1, 2 }, { 3, 4 } }),
                                np.array(new bool[] { false, true, false, false }).reshape(2, 2));
            var flat = f.flat;
            Assert.IsTrue(flat.data.ToArray<int>().SequenceEqual(new[] { 1, 2, 3, 4 }));
            Assert.IsTrue(M(flat).SequenceEqual(new[] { false, true, false, false }));
        }

        /// <summary>Instance resize refuses in place exactly as NumPy does (a masked array does not own its data);
        /// unshare_mask copies the mask so a later in-place mask write on a view cannot reach the source.</summary>
        [TestMethod]
        public void Resize_Raises_And_UnshareMask_Isolates()
        {
            var ex = Assert.ThrowsException<NotSupportedException>(
                () => np.ma.array(np.array(new int[] { 1, 2, 3 })).resize(new Shape(2, 3)));
            Assert.AreEqual(
                "A masked array does not own its data and therefore cannot be resized.\n" +
                "Use the numpy.ma.resize function instead.", ex.Message);

            // A view shares the source's mask reference — a masking write propagates WITHOUT unshare.
            var src2 = np.ma.array(np.array(new int[] { 1, 2, 3 }), np.array(new bool[] { false, true, false }));
            var view2 = src2.view();
            view2[0] = np.ma.NDMasked;
            Assert.IsTrue(src2.mask.GetBoolean(0));

            // After unshare_mask the view owns its mask, so the same write leaves the source alone.
            var src = np.ma.array(np.array(new int[] { 1, 2, 3 }), np.array(new bool[] { false, true, false }));
            var view = src.view();
            view.unshare_mask();
            view[0] = np.ma.NDMasked;
            Assert.IsFalse(src.mask.GetBoolean(0));
        }

        /// <summary>harden_mask makes the mask HARD: a plain-value assignment (indexer/put) no longer unmasks a
        /// slot (masked slots keep their data AND mask), while `= masked` still masks and soften_mask restores
        /// unmasking. putmask under hard WRITES data everywhere but FREEZES the mask (no unmask). Every value
        /// probed against NumPy 2.4.2.</summary>
        [TestMethod]
        public void HardMask_BlocksUnmasking()
        {
            static NDMaskedArray Mk() => np.ma.array(np.array(new int[] { 1, 2, 3, 4 }), np.array(new bool[] { false, true, false, true }));
            static int[] DI(NDMaskedArray r) => r.data.ToArray<int>();

            // Indexer: a masked slot is ignored (data & mask unchanged); an unmasked slot writes.
            var x = Mk().harden_mask();
            Assert.IsTrue(x.hardmask);
            x[1] = 99; x[0] = 88;
            Assert.IsTrue(DI(x).SequenceEqual(new[] { 88, 2, 3, 4 }));
            Assert.IsTrue(M(x).SequenceEqual(new[] { false, true, false, true }));

            // Slice/broadcast assign honors hard: only unmasked positions change.
            var y = Mk().harden_mask(); y[":"] = 7;
            Assert.IsTrue(DI(y).SequenceEqual(new[] { 7, 2, 7, 4 }));
            Assert.IsTrue(M(y).SequenceEqual(new[] { false, true, false, true }));

            // Masking a slot is still allowed under a hard mask.
            var z = Mk().harden_mask(); z[0] = np.ma.NDMasked;
            Assert.IsTrue(DI(z).SequenceEqual(new[] { 1, 2, 3, 4 }));
            Assert.IsTrue(M(z).SequenceEqual(new[] { true, true, false, true }));

            // put under hard: masked targets are dropped (both slots masked → whole call is a no-op).
            var p = Mk().harden_mask(); p.put(np.array(new int[] { 1, 3 }), np.array(new int[] { 50, 60 }));
            Assert.IsTrue(DI(p).SequenceEqual(new[] { 1, 2, 3, 4 }));
            Assert.IsTrue(M(p).SequenceEqual(new[] { false, true, false, true }));

            // putmask under hard: the DATA is written even at a masked slot, but the mask is frozen (no unmask).
            var q = Mk().harden_mask();
            np.ma.putmask(q, np.array(new bool[] { true, true, false, false }), np.array(new int[] { 50, 60, 70, 80 }));
            Assert.IsTrue(DI(q).SequenceEqual(new[] { 50, 60, 3, 4 }));
            Assert.IsTrue(M(q).SequenceEqual(new[] { false, true, false, true }));

            // soften_mask restores unmasking; the module harden_mask mutates the SAME instance in place.
            var s = Mk().harden_mask().soften_mask();
            Assert.IsFalse(s.hardmask);
            s[1] = 99;
            Assert.IsTrue(M(s).SequenceEqual(new[] { false, false, false, true }));
            var m2 = Mk();
            Assert.IsTrue(ReferenceEquals(m2, np.ma.harden_mask(m2)));
            Assert.IsTrue(m2.hardmask);
        }

        /// <summary>flatten_mask ravels any mask to a fresh 1-D BOOL array in C-order (non-bool coerced by
        /// non-zero-ness); a 0-D input flattens to length 1, an empty to length 0. Values probed against
        /// NumPy 2.4.2 (whose structured recursion never branches without structured dtypes, so this is
        /// exactly its non-structured path).</summary>
        [TestMethod]
        public void FlattenMask_RavelsToBool()
        {
            // 2-D integer mask → raveled, coerced != 0.
            var r = np.ma.flatten_mask(np.array(new[,] { { 0, 2 }, { 0, 0 } }));
            Assert.AreEqual(1, r.ndim);
            Assert.IsTrue(r.dtype == np.@bool);
            Assert.IsTrue(r.ToArray<bool>().SequenceEqual(new[] { false, true, false, false }));

            Assert.IsTrue(np.ma.flatten_mask(np.array(new[] { true, false, true })).ToArray<bool>()
                .SequenceEqual(new[] { true, false, true }));
            Assert.IsTrue(np.ma.flatten_mask(false).ToArray<bool>().SequenceEqual(new[] { false }));   // 0-D → len 1
            Assert.AreEqual(0, np.ma.flatten_mask(np.array(new bool[0])).ToArray<bool>().Length);       // empty → len 0
            Assert.IsTrue(np.ma.flatten_mask(np.array(new[] { 0.0, 1.5, 0.0, -2.0 })).ToArray<bool>()
                .SequenceEqual(new[] { false, true, false, true }));                                     // float !=0
        }

        /// <summary>make_mask_descr returns the BOOLEAN dtype for every data dtype — NumSharp has no structured
        /// dtypes, so the recursion NumPy uses over a structured dtype never branches and every dtype maps to
        /// the single scalar bool descriptor NumPy returns for a non-structured dtype.</summary>
        [TestMethod]
        public void MakeMaskDescr_AlwaysBool()
        {
            Assert.IsTrue(np.ma.make_mask_descr(np.float64) == np.@bool);
            Assert.IsTrue(np.ma.make_mask_descr(np.int32) == np.@bool);
            Assert.IsTrue(np.ma.make_mask_descr(np.complex128) == np.@bool);
            Assert.IsTrue(np.ma.make_mask_descr("i4") == np.@bool);          // dtype-string converts in
            Assert.IsTrue(np.ma.make_mask_descr(typeof(int)) == np.@bool);   // CLR Type converts in
        }

        /// <summary>mr_ is the masked counterpart of np.r_: concatenates masked arrays / slices / scalars along
        /// the first axis (or a directive axis), propagating the mask; two unmasked operands stay nomask; a lone
        /// string raises MAError (NumPy's rejection). Data uses NumSharp's colon-string slice spelling and every
        /// value/mask was probed against NumPy 2.4.2 (its real-slice form).</summary>
        [TestMethod]
        public void Mr_ConcatenatesMasked()
        {
            var a = np.ma.array(np.array(new[] { 1, 2, 3 }), np.array(new[] { false, true, false }));
            var b = np.ma.array(np.array(new[] { 4, 5, 6 }), np.array(new[] { true, false, false }));

            var r = np.ma.mr_[a, b];
            Assert.IsTrue(D(r).SequenceEqual(new double[] { 1, 2, 3, 4, 5, 6 }));
            Assert.IsTrue(M(r).SequenceEqual(new[] { false, true, false, true, false, false }));

            // scalar entry → one unmasked slot.
            var r2 = np.ma.mr_[a, 7, b];
            Assert.IsTrue(D(r2).SequenceEqual(new double[] { 1, 2, 3, 7, 4, 5, 6 }));
            Assert.IsTrue(M(r2).SequenceEqual(new[] { false, true, false, false, true, false, false }));

            // colon-slice (NumSharp spelling) + masked  ≙ numpy mr_[0:3, a].
            var r3 = np.ma.mr_["0:3", a];
            Assert.IsTrue(D(r3).SequenceEqual(new double[] { 0, 1, 2, 1, 2, 3 }));
            Assert.IsTrue(M(r3).SequenceEqual(new[] { false, false, false, false, true, false }));

            // axis directive over 2-D.
            var A2 = np.ma.array(np.array(new[,] { { 1, 2 }, { 3, 4 } }), np.array(new[,] { { false, true }, { false, false } }));
            var B2 = np.ma.array(np.array(new[,] { { 5, 6 }, { 7, 8 } }), np.array(new[,] { { true, false }, { false, true } }));
            var r4 = np.ma.mr_["1", A2, B2];
            Assert.AreEqual(4L, r4.shape[1]);
            Assert.IsTrue(D(r4).SequenceEqual(new double[] { 1, 2, 5, 6, 3, 4, 7, 8 }));
            Assert.IsTrue(M(r4).SequenceEqual(new[] { false, true, true, false, false, false, false, true }));

            // no mask anywhere → nomask fast path.
            var r5 = np.ma.mr_[np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 })];
            Assert.IsFalse(np.ma.is_masked(r5));
            Assert.IsTrue(D(r5).SequenceEqual(new double[] { 1, 2, 3, 4 }));

            // lone string → MAError verbatim.
            foreach (var k in new[] { "r", "0:3", "0" })
            {
                var ex = Assert.ThrowsException<MAError>(() => { var _ = np.ma.mr_[k]; });
                Assert.AreEqual("Unavailable for masked array.", ex.Message);
            }
        }

        // ── Masked printing (str/repr) + masked_print_option — byte-exact port of NumPy 2.4.2's
        //    MaskedArray.__str__/__repr__ (object-array substitution of the `--` display token). All expected
        //    strings were probed against NumPy 2.4.2. C# int[] → int32, so int64-default cases cast explicitly. ──

        /// <summary>The <c>str()</c> form shows masked slots inline as <c>--</c> via object-array formatting
        /// (unaligned), covering int/float/complex/bool, 2-D, 0-D, nomask, all-False mask and summarization —
        /// each byte-exact vs NumPy 2.4.2.</summary>
        [TestMethod]
        public void MaskedPrint_Str_BitExact()
        {
            NDMaskedArray mi(int[] d, bool[] m) => np.ma.array(np.array(d).astype(np.int64), np.array(m));
            Assert.AreEqual("[1 -- 3 --]", mi(new[] { 1, 2, 3, 4 }, new[] { false, true, false, true }).ToString());
            // masked slots are NOT padded to the wide value's width (object layout, not numeric alignment).
            Assert.AreEqual("[1 -- 3 --]", mi(new[] { 1, 200, 3, 40000 }, new[] { false, true, false, true }).ToString());
            Assert.AreEqual("[1.5 -- 3.5]", np.ma.array(np.array(new[] { 1.5, 2.5, 3.5 }), np.array(new[] { false, true, false })).ToString());
            Assert.AreEqual("[[1 --]\n [3 4]]", np.ma.array(np.array(new[,] { { 1, 2 }, { 3, 4 } }), np.array(new[,] { { false, true }, { false, false } })).ToString());
            // nomask → normal (aligned) numeric print; all-False real mask → unaligned object print.
            Assert.AreEqual("[1 2 3]", np.ma.array(np.array(new[] { 1, 2, 3 })).ToString());
            Assert.AreEqual("[1 200 3 40000]", mi(new[] { 1, 200, 3, 40000 }, new[] { false, false, false, false }).ToString());
            Assert.AreEqual("--", np.ma.array(np.array(5).astype(np.int64), np.array(true)).ToString());
            Assert.AreEqual("5", np.ma.array(np.array(5).astype(np.int64), np.array(false)).ToString());
            Assert.AreEqual("[(1+2j) --]", np.ma.array(np.array(new[] { new System.Numerics.Complex(1, 2), new System.Numerics.Complex(3, 4) }), np.array(new[] { false, true })).ToString());
            Assert.AreEqual("[True -- True]", np.ma.array(np.array(new[] { true, false, true }), np.array(new[] { false, true, false })).ToString());
            // summarization threads through the mask (edgeitems=3): [first3 ... last3].
            var big = np.ma.array(np.arange(2000), (np.arange(2000) % 2).astype(np.@bool));
            Assert.AreEqual("[0 -- 2 ... -- 1998 --]", big.ToString());
        }

        /// <summary>The <c>repr()</c> form (<c>ToString(true)</c>) reproduces NumPy's
        /// <c>masked_array(data=…, mask=…, fill_value=…[, dtype=…])</c> template exactly — one-row vs 2-D layout,
        /// the <c>dtype=</c> line for non-implied/all-masked dtypes, and the <c>np.&lt;type&gt;(value)</c> fill
        /// wrapper.</summary>
        [TestMethod]
        public void MaskedPrint_Repr_BitExact()
        {
            NDMaskedArray mi(int[] d, bool[] m) => np.ma.array(np.array(d).astype(np.int64), np.array(m));
            Assert.AreEqual(
                "masked_array(data=[1, --, 3, --],\n             mask=[False,  True, False,  True],\n       fill_value=999999)",
                mi(new[] { 1, 2, 3, 4 }, new[] { false, true, false, true }).ToString(true));
            Assert.AreEqual(
                "masked_array(\n  data=[[1, --],\n        [3, 4]],\n  mask=[[False,  True],\n        [False, False]],\n  fill_value=999999)",
                np.ma.array(np.array(new[,] { { 1, 2 }, { 3, 4 } }).astype(np.int64), np.array(new[,] { { false, true }, { false, false } })).ToString(true));
            Assert.AreEqual(
                "masked_array(data=[1, 2, 3],\n             mask=False,\n       fill_value=999999)",
                np.ma.array(np.array(new[] { 1, 2, 3 }).astype(np.int64)).ToString(true));
            // all-masked → dtype shown even though int64 is implied (values never appear).
            Assert.AreEqual(
                "masked_array(data=[--, --, --],\n             mask=[ True,  True,  True],\n       fill_value=999999,\n            dtype=int64)",
                mi(new[] { 1, 2, 3 }, new[] { true, true, true }).ToString(true));
            // non-implied dtype → dtype line + promoted-fill wrapper np.<type>(value).
            Assert.AreEqual(
                "masked_array(data=[1.5, --, 3.5],\n             mask=[False,  True, False],\n       fill_value=np.float64(1e+20),\n            dtype=float32)",
                np.ma.array(np.array(new[] { 1.5f, 2.5f, 3.5f }), np.array(new[] { false, true, false })).ToString(true));
            Assert.AreEqual(
                "masked_array(data=[1, --, 3],\n             mask=[False,  True, False],\n       fill_value=np.int64(999999),\n            dtype=int32)",
                np.ma.array(np.array(new[] { 1, 2, 3 }), np.array(new[] { false, true, false })).ToString(true));
            Assert.AreEqual(
                "masked_array(data=[],\n             mask=[],\n       fill_value=1e+20,\n            dtype=float64)",
                np.ma.array(np.array(new double[] { }), np.array(new bool[] { })).ToString(true));
            Assert.AreEqual(
                "masked_array(data=--,\n             mask=True,\n       fill_value=999999,\n            dtype=int64)",
                np.ma.array(np.array(5).astype(np.int64), np.array(true)).ToString(true));
        }

        /// <summary><c>masked_print_option</c> is a live global: <c>set_display</c> changes the token and
        /// <c>enable(false)</c> switches to the fill-value (aligned) rendering — both observable through
        /// <c>ToString()</c>, restored afterward so the shared option does not leak into other tests.</summary>
        [TestMethod]
        public void MaskedPrintOption_DisplayAndEnable()
        {
            var mpo = np.ma.masked_print_option;
            Assert.AreEqual("--", mpo.display());
            Assert.IsTrue(mpo.enabled());
            Assert.AreEqual("--", mpo.ToString());

            var a = np.ma.array(np.array(new[] { 1, 2, 3, 4 }).astype(np.int64), np.array(new[] { false, true, false, true }));
            try
            {
                mpo.enable(false); // disabled → masked slots print the fill value, numeric-aligned.
                Assert.AreEqual("[     1 999999      3 999999]", a.ToString());
                mpo.enable(true);
                mpo.set_display("N/A");
                Assert.AreEqual("[1 N/A 3 N/A]", a.ToString());
            }
            finally
            {
                mpo.set_display("--"); // restore the shared global regardless of assertion outcome.
                mpo.enable(true);
            }
        }

        /// <summary>The DATA-only instance members inherited from ndarray: <c>fill</c> overwrites the data and
        /// keeps the mask, <c>searchsorted</c> reads the data and returns a plain array, <c>byteswap</c> swaps the
        /// data and carries the mask into a fresh array.</summary>
        [TestMethod]
        public void Instance_Fill_Searchsorted_Byteswap()
        {
            var af = np.ma.array(np.array(new[] { 1, 2, 3, 4 }).astype(np.int64), np.array(new[] { false, true, false, true }));
            af.fill(99);
            Assert.IsTrue(D(af).SequenceEqual(new double[] { 99, 99, 99, 99 })); // data overwritten…
            Assert.IsTrue(M(af).SequenceEqual(new[] { false, true, false, true })); // …mask untouched.

            var ss = np.ma.array(np.array(new[] { 1, 2, 3, 4, 5 }), np.array(new[] { false, false, true, false, false }));
            Assert.AreEqual(2, ss.searchsorted(np.array(3)).GetInt32(0));
            Assert.IsTrue(ss.searchsorted(np.array(new[] { 2.5, 4.5 })).ToArray<long>().SequenceEqual(new long[] { 2, 4 }));

            var bs = np.ma.array(np.array(new short[] { 1, 2, 3 }), np.array(new[] { false, true, false }));
            var bsr = bs.byteswap();
            Assert.IsTrue(D(bsr).SequenceEqual(new double[] { 256, 512, 768 }));
            Assert.IsTrue(M(bsr).SequenceEqual(new[] { false, true, false }));
            Assert.IsTrue(D(bs).SequenceEqual(new double[] { 1, 2, 3 })); // non-inplace: source unchanged.
        }

        /// <summary><c>partition</c>/<c>argpartition</c> reorder only the DATA and leave the mask in its ORIGINAL
        /// positions (NumPy's documented mask-oblivious footgun — it warns, NumSharp has no warning channel).</summary>
        [TestMethod]
        public void Instance_Partition_Argpartition_IgnoreMask()
        {
            var pt = np.ma.array(np.array(new[] { 5, 3, 8, 1, 9, 2 }), np.array(new[] { false, false, true, false, false, false }));
            pt.partition(2);
            Assert.IsTrue(D(pt).SequenceEqual(new double[] { 1, 2, 3, 5, 8, 9 })); // data partitioned…
            Assert.IsTrue(M(pt).SequenceEqual(new[] { false, false, true, false, false, false })); // …mask NOT moved.

            var ap = np.ma.array(np.array(new[] { 5, 3, 8, 1, 9, 2 }), np.array(new[] { false, false, true, false, false, false }));
            var apr = ap.argpartition(2);
            Assert.IsTrue(np.ma.getdata(apr).astype(np.int64).ToArray<long>().SequenceEqual(new long[] { 3, 5, 1, 0, 2, 4 }));
            Assert.IsTrue(M(apr).SequenceEqual(new[] { false, false, true, false, false, false }));
        }

        /// <summary>The data-describing instance properties (<c>itemsize</c>/<c>nbytes</c>/<c>strides</c>) mirror
        /// the DATA buffer and EXCLUDE the mask, matching NumPy (a masked array's <c>nbytes</c> equals its unmasked
        /// data's; strides are reported in bytes).</summary>
        [TestMethod]
        public void Instance_DataDescribingProperties()
        {
            // int32 (2,3): itemsize 4, nbytes 24 (mask's 6 bytes excluded), strides (12, 4) in bytes.
            var a = np.ma.array(np.arange(6).astype(np.int32).reshape(2, 3), np.array(new[,] { { false, true, false }, { false, false, true } }));
            Assert.AreEqual(4, a.itemsize);
            Assert.AreEqual(24L, a.nbytes);
            Assert.IsTrue(a.strides.SequenceEqual(new long[] { 12, 4 }));
        }

        /// <summary>numpy.ma treats a bare scalar operand of a binary op as a STRONG default-width array (python
        /// int→int64, float→float64), so a narrow-dtype array op a scalar WIDENS — unlike the plain-np weak-scalar
        /// path. The inputs are widened (not the result), so a value that would overflow the narrow dtype is
        /// computed wide, exactly as NumPy does. Probed against NumPy 2.4.2.</summary>
        [TestMethod]
        public void MaBinary_ScalarOperand_PromotesLikeNumPy()
        {
            var i32 = np.ma.array(np.array(new[] { 1, 2, 3 }).astype(np.int32), np.array(new[] { false, true, false }));
            var i8 = np.ma.array(np.array(new[] { 1, 2, 3 }).astype(np.int8), np.array(new[] { false, true, false }));
            var f32 = np.ma.array(np.array(new[] { 1f, 2f, 3f }), np.array(new[] { false, true, false }));
            var i64 = np.ma.array(np.array(new[] { 1L, 2L, 3L }), np.array(new[] { false, true, false }));

            // Integer array × C# int scalar → int64 (NumPy's ma wraps the scalar as a strong int64 array).
            np.ma.multiply(i32, 2).dtype.Should().Be(np.int64);
            np.ma.add(i8, 2).dtype.Should().Be(np.int64);
            np.ma.floor_divide(i32, 2).dtype.Should().Be(np.int64);   // DomainedBinary path
            np.ma.bitwise_and(i32, 2).dtype.Should().Be(np.int64);
            np.ma.power(i32, 2).dtype.Should().Be(np.int64);          // own path
            // Float array (or float scalar) → float64; int64 array unchanged.
            np.ma.multiply(f32, 2).dtype.Should().Be(np.float64);
            np.ma.multiply(i32, 2.5).dtype.Should().Be(np.float64);
            np.ma.multiply(i64, 2).dtype.Should().Be(np.int64);
            // maximum/minimum/where route through np.where and widen the same way.
            np.ma.maximum(i32, 2).dtype.Should().Be(np.int64);
            np.ma.minimum(i32, 2).dtype.Should().Be(np.int64);
            np.ma.where(np.array(new[] { true, false, true }), i32, 2).dtype.Should().Be(np.int64);
            // array × array is UNTOUCHED — int32 × int32 stays int32 (no scalar operand).
            np.ma.multiply(i32, i32).dtype.Should().Be(np.int32);
            // Inputs widened BEFORE the op: 2e9 (int32) * 2 = 4e9 computed at int64, not the int32-wrapped value.
            var big = np.ma.array(np.array(new[] { 2_000_000_000 }).astype(np.int32), np.array(new[] { false }));
            var r = np.ma.multiply(big, 2);
            r.dtype.Should().Be(np.int64);
            Assert.AreEqual(4_000_000_000L, Convert.ToInt64(np.ma.getdata(r).GetAtIndex(0)));
        }

        /// <summary>The instance wrappers over module functions the surface diff found missing: <c>copy()</c> deep-copies
        /// data+mask (independent), <c>product()</c> aliases <c>prod</c>, and <c>choose()</c> selects from choices by
        /// this array's (masked) indices. Probed against NumPy 2.4.2.</summary>
        [TestMethod]
        public void Instance_Copy_Choose_Product()
        {
            var a = np.ma.array(np.array(new[] { 1, 2, 3, 4 }).astype(np.int64), np.array(new[] { false, true, false, true }));
            var c = a.copy();
            c[0] = 99;                                        // mutate the copy
            Assert.IsTrue(D(a).SequenceEqual(new double[] { 1, 2, 3, 4 })); // original untouched → deep copy
            Assert.IsTrue(D(c).SequenceEqual(new double[] { 99, 2, 3, 4 }));
            Assert.IsTrue(M(c).SequenceEqual(new[] { false, true, false, true })); // slot 0 was already unmasked → mask unchanged

            // product == prod (of the unmasked elements: 1*3 = 3; slots 1,3 masked).
            Assert.AreEqual(3L, Convert.ToInt64(np.ma.getdata(a.product()).GetAtIndex(0)));

            // choose: idx [0,1,2,1,0] (slots 2,4 masked) selects choices[idx][pos]; masked idx → masked result.
            var idx = np.ma.array(np.array(new[] { 0, 1, 2, 1, 0 }).astype(np.int64), np.array(new[] { false, false, true, false, true }));
            var choices = new object[]
            {
                np.array(new[]{10,11,12,13,14}).astype(np.int64),
                np.array(new[]{20,21,22,23,24}).astype(np.int64),
                np.array(new[]{30,31,32,33,34}).astype(np.int64),
            };
            var ch = idx.choose(choices);
            // unmasked picks: pos0→choices[0][0]=10, pos1→choices[1][1]=21, pos3→choices[1][3]=23; pos2,4 masked.
            Assert.IsTrue(M(ch).SequenceEqual(new[] { false, false, true, false, true }));
            Assert.AreEqual(10L, Convert.ToInt64(np.ma.getdata(ch).GetAtIndex(0)));
            Assert.AreEqual(21L, Convert.ToInt64(np.ma.getdata(ch).GetAtIndex(1)));
            Assert.AreEqual(23L, Convert.ToInt64(np.ma.getdata(ch).GetAtIndex(3)));
        }

        // ── Pass-13 audit fixes (found by a 6055-case behavioral differential fuzz). ──

        /// <summary>
        ///     REGRESSION FIX: <c>ma.left_shift</c>/<c>right_shift</c> keep the value operand's dtype for a bare
        ///     scalar shift-count (NumPy's shifts are plain functions, NOT <c>_MaskedBinaryOperation</c>, so —
        ///     unlike add/multiply/bitwise — the scalar is NOT widened to int64). Pass-12's blanket scalar-strong
        ///     promotion had wrongly widened them; verified bit-exact vs NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void MaShift_ScalarCount_KeepsArrayDtype()
        {
            foreach (var (dt, name) in new[] { (np.int8, "int8"), (np.int16, "int16"), (np.int32, "int32"), (np.uint64, "uint64") })
            {
                var a = np.ma.array(np.array(new[] { 3, 5, 6 }).astype(dt), np.array(new[] { false, true, false }));
                np.ma.left_shift(a, 1).dtype.Should().Be(dt);  // stays the array dtype, NOT int64
                np.ma.right_shift(a, 1).dtype.Should().Be(dt);
                // unmasked values (positions 0,2): 3<<1=6, 6<<1=12 (slot 1 is masked).
                Assert.IsTrue(np.ma.compressed(np.ma.left_shift(a, 1)).astype(np.int64).ToArray<long>().SequenceEqual(new long[] { 6, 12 }));
            }
            // contrast: a non-shift binary op DOES widen a scalar to int64 (the pass-12 behavior, unchanged).
            var i16 = np.ma.array(np.array(new[] { 3, 5, 6 }).astype(np.int16), np.array(new[] { false, true, false }));
            np.ma.add(i16, 1).dtype.Should().Be(np.int64);
        }

        /// <summary>
        ///     CRASH FIX: <c>arctan2</c> on complex inputs must throw cleanly, not corrupt memory. arctan2 is a
        ///     float-only ufunc (NumPy raises <c>TypeError</c> for complex); NumSharp's kernel previously lacked
        ///     the complex-input guard and fatally crashed the CLR (<c>ExecutionEngineException</c> in
        ///     <c>ExecuteATan2Kernel</c> — found by the differential fuzz). Both the plain <c>np.arctan2</c> and the
        ///     masked <c>np.ma.arctan2</c> paths now raise the NumPy-verbatim message.
        /// </summary>
        [TestMethod]
        public void Arctan2_Complex_ThrowsCleanly_NoCrash()
        {
            var ca = np.array(new[] { new System.Numerics.Complex(1, -1), new System.Numerics.Complex(-2, 3) });
            var cb = np.array(new[] { new System.Numerics.Complex(2, 0), new System.Numerics.Complex(3, 1) });
            // Plain np path (the core fix).
            Assert.ThrowsException<IncorrectTypeException>(() => np.arctan2(ca, cb));
            // Masked path (delegates to np.arctan2).
            var ma = np.ma.array(ca);
            var mb = np.ma.array(cb);
            Assert.ThrowsException<IncorrectTypeException>(() => np.ma.arctan2(ma, mb));
        }

        /// <summary>
        ///     DTYPE FIX: masked <c>median</c> along an axis of a 1-D narrow-float array preserves the input float
        ///     dtype (float16/float32), matching NumPy. The 1-D result is 0-D, and NumSharp's 0-D-is-weak rule had
        ///     widened the <c>/2</c> to float64 (the 2-D path kept the dtype — an internal asymmetry). Dividing a
        ///     float by 2 is exact, so the restore is bit-exact.
        /// </summary>
        [TestMethod]
        public void MaMedian_1D_NarrowFloat_PreservesDtype()
        {
            foreach (var dt in new[] { np.float16, np.float32, np.float64 })
            {
                var a = np.ma.array(np.array(new[] { 5f, 6f, 7f, 8f }).astype(dt), np.array(new[] { false, false, true, false }));
                var med = np.ma.median(a, axis: -1);            // unmasked {5,6,8} → median 6
                med.dtype.Should().Be(dt);                       // preserves float16/float32/float64
                Assert.AreEqual(6.0, Convert.ToDouble(np.ma.getdata(med).astype(np.float64).GetAtIndex(0)));
            }
        }
    }
}
