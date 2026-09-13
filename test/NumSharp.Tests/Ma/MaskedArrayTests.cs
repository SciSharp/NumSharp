using System;
using System.Linq;

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
    public class MaskedArrayTests
    {
        /// <summary>Data of a masked result as float64, in logical C-order (masked positions included).</summary>
        private static double[] D(MaskedArray r) => np.ma.getdata(r).astype(np.float64).ToArray<double>();

        /// <summary>Full boolean mask of a masked result (never the 0-d nomask sentinel).</summary>
        private static bool[] M(MaskedArray r) => np.ma.getmaskarray(r).ToArray<bool>();

        private static MaskedArray A() =>
            np.ma.array(np.array(new double[] { 1, -2, 3, -4 }), np.array(new bool[] { false, true, false, false }));
        private static MaskedArray B() =>
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
            var r = np.ma.add(np.ma.masked, NDArray.Scalar(5.0));
            Assert.IsTrue(ReferenceEquals(r, np.ma.masked));
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
        private static MaskedArray M13() =>
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
            Assert.IsTrue(ReferenceEquals(np.ma.sum(allMasked), np.ma.masked));
            Assert.IsTrue(ReferenceEquals(np.ma.mean(allMasked), np.ma.masked));
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
    }
}
