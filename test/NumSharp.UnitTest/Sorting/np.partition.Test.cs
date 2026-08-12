using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Sorting
{
    /// <summary>
    /// np.partition / np.argpartition / ndarray.partition parity with NumPy 2.4.2.
    /// Implementation: AxisSort's IterAllButAxis drive + QuickSelect introselect (the primitive
    /// already backing median/percentile). The partition CONTRACT fixes only the kth elements and
    /// the two-sided ≤/≥ invariant — the arrangement inside each side is algorithm-specific in
    /// NumPy too, so tests assert kth values + invariants (probed values where NumPy's own
    /// arrangement is matched are noted).
    /// </summary>
    [TestClass]
    public class NpPartitionTests
    {
        /// <summary>Asserts the NumPy partition contract for a 1-D result: result is a permutation
        /// of the input, result[k] == sorted(input)[k] for every k, left ≤ result[k] ≤ right.</summary>
        private static void AssertPartitionInvariant(NDArray input1d, NDArray result, params int[] ks)
        {
            var srt = np.sort(input1d.astype(np.float64), null);
            var res = result.astype(np.float64);
            foreach (var k in ks)
            {
                double pv = res.GetValue<double>(k), sv = srt.GetValue<double>(k);
                if (double.IsNaN(sv)) double.IsNaN(pv).Should().BeTrue($"result[{k}] must be NaN");
                else pv.Should().Be(sv, $"result[{k}] must equal sorted[{k}]");
                if (double.IsNaN(pv)) continue;
                for (int i = 0; i < k; i++)
                {
                    double v = res.GetValue<double>(i);
                    (double.IsNaN(v) || v <= pv).Should().BeTrue($"left of kth={k}: [{i}]={v} must be <= {pv}");
                }
                for (int i = k + 1; i < res.size; i++)
                {
                    double v = res.GetValue<double>(i);
                    (double.IsNaN(v) || v >= pv).Should().BeTrue($"right of kth={k}: [{i}]={v} must be >= {pv}");
                }
            }
            // permutation check via sorted equality
            var sortedRes = np.sort(res, null);
            for (int i = 0; i < srt.size; i++)
            {
                double e = srt.GetValue<double>(i), g = sortedRes.GetValue<double>(i);
                if (double.IsNaN(e)) double.IsNaN(g).Should().BeTrue("multiset must be preserved");
                else g.Should().Be(e, "multiset must be preserved");
            }
        }

        // -------------------- basics --------------------
        [TestMethod]
        public void Partition_1D_Basic()
        {
            // np.partition([3,4,2,1], 3) == [1,2,3,4] (probed — NumPy's own arrangement)
            var r = np.partition(np.array(new[] { 3, 4, 2, 1 }), 3);
            r.ToArray<int>().Should().BeEquivalentTo(new[] { 1, 2, 3, 4 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Partition_MultiKth()
        {
            // np.partition(a, (4, 8)): p[4]==2 and p[8]==5 with the doc's three-range invariant
            var a = np.array(new[] { 0, 1, 2, 1, 2, 3, 3, 2, 5, 6, 7, 7, 7, 7 });
            var r = np.partition(a, new[] { 4, 8 });
            r.GetValue<int>(4).Should().Be(2);
            r.GetValue<int>(8).Should().Be(5);
            AssertPartitionInvariant(a, r, 4, 8);
        }

        [TestMethod]
        public void Partition_NegativeKth_Wraps()
        {
            // np.partition([3,4,2,1], -1)[3] == 4
            var r = np.partition(np.array(new[] { 3, 4, 2, 1 }), -1);
            r.GetValue<int>(3).Should().Be(4);
        }

        [TestMethod]
        public void Partition_KthUnsorted_SortedInternally()
        {
            // np.partition(descending, [7,2]) — kths are sorted before use (NumPy sorts kthrvl)
            var a = np.array(new double[] { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });
            var r = np.partition(a, new[] { 7, 2 });
            AssertPartitionInvariant(a, r, 2, 7);
        }

        [TestMethod]
        public void Partition_ReturnsCopy_InputUntouched()
        {
            var a = np.array(new[] { 3, 4, 2, 1 });
            np.partition(a, 2);
            a.ToArray<int>().Should().BeEquivalentTo(new[] { 3, 4, 2, 1 }, o => o.WithStrictOrdering());
        }

        // -------------------- axis --------------------
        [TestMethod]
        public void Partition_2D_Axis0()
        {
            // np.partition([[3,1],[2,0]], 0, axis=0) == [[2,0],[3,1]] (probed)
            var r = np.partition(np.array(new[,] { { 3, 1 }, { 2, 0 } }), 0, axis: 0);
            r.GetValue<int>(0, 0).Should().Be(2); r.GetValue<int>(0, 1).Should().Be(0);
            r.GetValue<int>(1, 0).Should().Be(3); r.GetValue<int>(1, 1).Should().Be(1);
        }

        [TestMethod]
        public void Partition_AxisNone_Flattens()
        {
            // np.partition([[3,1],[2,0]], 2, axis=None) == [0,1,2,3] (probed)
            var r = np.partition(np.array(new[,] { { 3, 1 }, { 2, 0 } }), 2, axis: null);
            r.Shape.Should().Be(new Shape(4));
            r.GetValue<int>(2).Should().Be(2);
        }

        [TestMethod]
        public void Partition_DefaultAxis_IsLast()
        {
            // kth validated against the LAST axis: np.partition([[3,1],[2,0]], 2) → kth(=2) oob (2)
            var act = () => np.partition(np.array(new[,] { { 3, 1 }, { 2, 0 } }), 2);
            act.Should().Throw<ValueError>().WithMessage("kth(=2) out of bounds (2)*");
        }

        [TestMethod]
        public void Partition_3D_MiddleAxis()
        {
            // arange(24)[::-1].reshape(2,3,4), kth=1, axis=1:每 column partitioned
            var a = np.arange(24)["::-1"].reshape(2, 3, 4);
            var r = np.partition(a, 1, axis: 1);
            // probed NumPy: r[0,1,:] == [19,18,17,16] and r[1,1,:] == [7,6,5,4]
            r.GetValue<long>(0, 1, 0).Should().Be(19);
            r.GetValue<long>(1, 1, 3).Should().Be(4);
        }

        // -------------------- validation (NumPy's probed order + verbatim texts) --------------------
        [TestMethod]
        public void Partition_KthOutOfBounds()
        {
            var act = () => np.partition(np.array(new[] { 3, 4, 2, 1 }), 4);
            act.Should().Throw<ValueError>().WithMessage("kth(=4) out of bounds (4)*");
        }

        [TestMethod]
        public void Partition_NegativeKthOOB_ReportsPostWrapValue()
        {
            // np.partition(a4, -5) → "kth(=-1) out of bounds (4)" — the POST-wrap value, like NumPy
            var act = () => np.partition(np.array(new[] { 3, 4, 2, 1 }), -5);
            act.Should().Throw<ValueError>().WithMessage("kth(=-1) out of bounds (4)*");
        }

        [TestMethod]
        public void Partition_BadKind()
        {
            var act = () => np.partition(np.array(new[] { 3, 1 }), 0, kind: "quickselect");
            act.Should().Throw<ValueError>().WithMessage("select kind must be 'introselect' (got 'quickselect')*");
        }

        [TestMethod]
        public void Partition_OrderNotSupported()
        {
            var act = () => np.partition(np.array(new[] { 3, 1 }), 0, order: "f");
            act.Should().Throw<ValueError>().WithMessage("Cannot specify order when the array has no fields.*");
        }

        [TestMethod]
        public void Partition_ValidationOrder_KindBeforeOrderBeforeAxisBeforeKth()
        {
            // NumPy probed precedence with everything invalid at once
            var a = np.array(new[] { 3.0, 1.0, 2.0 });
            ((Action)(() => np.partition(a, 99, axis: 9, kind: "bad", order: "f")))
                .Should().Throw<ValueError>().WithMessage("select kind must be*");
            ((Action)(() => np.partition(a, 99, axis: 9, order: "f")))
                .Should().Throw<ValueError>().WithMessage("Cannot specify order*");
            ((Action)(() => np.partition(a, 99, axis: 9)))
                .Should().Throw<ArgumentException>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
            ((Action)(() => np.partition(a, 99)))
                .Should().Throw<ValueError>().WithMessage("kth(=99) out of bounds (3)*");
        }

        [TestMethod]
        public void Partition_Empty_SkipsKthValidation()
        {
            // np.partition([], 5) == [] — the C source skips kth bounds for size-0 arrays
            var r = np.partition(np.array(new double[0]), 5);
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void Partition_0d_Throws_ButAxisNoneFlattens()
        {
            // np.partition(np.array(5.0), 0) → AxisError: axis -1 ... dimension 0
            ((Action)(() => np.partition(NDArray.Scalar(5.0), 0)))
                .Should().Throw<ArgumentException>().WithMessage("axis -1 is out of bounds for array of dimension 0*");
            // np.partition(np.array(5.0), 0, axis=None) == array([5.]) shape (1,)
            var r = np.partition(NDArray.Scalar(5.0), 0, axis: null);
            r.Shape.Should().Be(new Shape(1));
            r.GetValue<double>(0).Should().Be(5.0);
        }

        [TestMethod]
        public void Partition_EmptyKthArray_IsValidNoOp()
        {
            // C# int[0] ≙ NumPy np.array([], dtype=intp): a valid no-op copy (documented divergence
            // from Python's bare [] which is float64 → TypeError)
            var a = np.array(new[] { 3, 4, 2, 1 });
            var r = np.partition(a, new int[0]);
            r.ToArray<int>().Should().BeEquivalentTo(new[] { 3, 4, 2, 1 }, o => o.WithStrictOrdering());
        }

        // -------------------- NaN / special values --------------------
        [TestMethod]
        public void Partition_NaN_SortsLast()
        {
            // np.partition([nan,1,7,2], 1)[1] == 2; kth=3 lands on nan
            var a = np.array(new[] { double.NaN, 1.0, 7.0, 2.0 });
            np.partition(a, 1).GetValue<double>(1).Should().Be(2.0);
            double.IsNaN(np.partition(a, 3).GetValue<double>(3)).Should().BeTrue();
            AssertPartitionInvariant(a, np.partition(a, new[] { 1, 3 }), 1, 3);
        }

        [TestMethod]
        public void Partition_NaN_PreservesBitPatterns()
        {
            // NumPy moves NaNs by swaps — payload/sign bits survive. NumSharp stashes and rewrites
            // the ORIGINAL bits (unlike np.sort's canonicalizing float radix path).
            double negNan = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL));
            var a = np.array(new[] { negNan, 1.0, 2.0 });
            var r = np.partition(a, 0);
            long bits = BitConverter.DoubleToInt64Bits(r.GetValue<double>(2));
            bits.Should().Be(unchecked((long)0xFFF8000000000000UL), "the NaN's sign/payload bits must survive");
        }

        [TestMethod]
        public void Partition_Complex_NumPyOrdering()
        {
            // np.partition([3+1j,1+2j,2-1j,1+0j], 1)[1] == 1+2j (sorted: [1+0j,1+2j,2-1j,3+1j])
            var a = np.array(new[] { new Complex(3, 1), new Complex(1, 2), new Complex(2, -1), new Complex(1, 0) });
            var r = np.partition(a, 1);
            r.GetValue<Complex>(1).Should().Be(new Complex(1, 2));
            r.GetValue<Complex>(0).Should().Be(new Complex(1, 0));
        }

        [TestMethod]
        public void Partition_Complex_NaNOrdering_NotCompactable()
        {
            // sorted order (NumPy CDOUBLE_LT): [0+1j, 2+3j, 1+nanj, nan+1j] — (1,nan) BEFORE (nan,1)
            var a = np.array(new[] { new Complex(double.NaN, 1), new Complex(2, 3), new Complex(1, double.NaN), new Complex(0, 1) });
            var r = np.partition(a, 2);
            var v2 = r.GetValue<Complex>(2);
            (v2.Real == 1.0 && double.IsNaN(v2.Imaginary)).Should().BeTrue("sorted[2] is 1+nanj");
            var v3 = r.GetValue<Complex>(3);
            double.IsNaN(v3.Real).Should().BeTrue("sorted[3] is nan+1j");
        }

        // -------------------- dtypes --------------------
        [TestMethod]
        public void Partition_AllDtypes_KthValueMatchesSorted()
        {
            void Check<T>(T[] values, int k) where T : unmanaged
            {
                var a = np.array(values);
                var r = np.partition(a, k);
                var s = np.sort(a, null);
                r.GetValue<T>(k).Should().Be(s.GetValue<T>(k), $"dtype {a.typecode} kth={k}");
                r.typecode.Should().Be(a.typecode);
            }

            Check(new[] { true, false, true, false }, 1);
            Check(new byte[] { 9, 1, 5, 3 }, 2);
            Check(new sbyte[] { -9, 1, -5, 3 }, 2);
            Check(new short[] { -900, 100, -500, 300 }, 1);
            Check(new ushort[] { 900, 100, 500, 300 }, 1);
            Check(new[] { 'd', 'a', 'c', 'b' }, 2);
            Check(new[] { -90000, 10000, -50000, 30000 }, 2);
            Check(new uint[] { 90000, 10000, 50000, 30000 }, 2);
            Check(new[] { -9000000000L, 1000000000L, -5000000000L, 3000000000L }, 1);
            Check(new ulong[] { 9223372036854775813UL, 3, 18446744073709551615UL }, 1);
            Check(new[] { (Half)9, (Half)1, (Half)5, (Half)3 }, 2);
            Check(new[] { 9f, 1f, 5f, 3f }, 2);
            Check(new[] { 9.0, 1.0, 5.0, 3.0 }, 2);
            Check(new decimal[] { 9, 1, 5, 3 }, 2);
            Check(new[] { new Complex(3, 1), new Complex(1, 2), new Complex(2, -1) }, 1);
        }

        [TestMethod]
        public void Partition_UInt64_ExactAboveInt64Range()
        {
            // values above 2^63 order by exact unsigned value (no double roundtrip)
            var r = np.partition(np.array(new ulong[] { 9223372036854775813UL, 3, 18446744073709551615UL }), 1);
            r.GetValue<ulong>(0).Should().Be(3UL);
            r.GetValue<ulong>(1).Should().Be(9223372036854775813UL);
            r.GetValue<ulong>(2).Should().Be(18446744073709551615UL);
        }

        // -------------------- layouts / in-place --------------------
        [TestMethod]
        public void Partition_StridedView_Input()
        {
            // np.partition(a[::2], 1) == [7,8,9] for a=[9,1,8,2,7,3] (probed)
            var a = np.array(new[] { 9.0, 1.0, 8.0, 2.0, 7.0, 3.0 });
            var r = np.partition(a["::2"], 1);
            r.GetValue<double>(1).Should().Be(8.0);
            AssertPartitionInvariant(a["::2"], r, 1);
        }

        [TestMethod]
        public void PartitionInPlace_StridedView_WritesThrough()
        {
            // v = a[::2]; v.partition(1) mutates the base through the view (probed: base [7,1,8,2,9,3])
            var a = np.array(new[] { 9.0, 1.0, 8.0, 2.0, 7.0, 3.0 });
            var v = a["::2"];
            v.partition(1);
            v.GetValue<double>(1).Should().Be(8.0);
            a.GetValue<double>(1).Should().Be(1.0, "the untouched interleaved elements stay");
            a.GetValue<double>(3).Should().Be(2.0);
        }

        [TestMethod]
        public void PartitionInPlace_ReadOnly_Throws()
        {
            // NumPy: ValueError: partition array is read-only
            var bc = np.broadcast_to(np.array(new[] { 3.0 }), new Shape(4));
            var act = () => bc.partition(1);
            act.Should().Throw<ValueError>().WithMessage("partition array is read-only*");
        }

        [TestMethod]
        public void Partition_TransposedInput()
        {
            var a = np.arange(6).reshape(2, 3).T;   // shape (3,2), non-contiguous
            var r = np.partition(a, 0, axis: 0);
            var s = np.sort(a.copy(), 0);
            r.GetValue<long>(0, 0).Should().Be(s.GetValue<long>(0, 0));
            r.GetValue<long>(0, 1).Should().Be(s.GetValue<long>(0, 1));
        }

        // -------------------- NDArray kth (NumPy's array-kth form; probed verbatim) --------------------
        [TestMethod]
        public void Partition_NDArrayKth_Basic()
        {
            // np.partition([3,4,2,1], np.array([1,3])) — kth values, 0-d kth, empty kth all legal
            var b = np.array(new double[] { 3, 4, 2, 1 });
            var r = np.partition(b, np.array(new[] { 1, 3 }));
            r.GetValue<double>(1).Should().Be(2.0);
            r.GetValue<double>(3).Should().Be(4.0);
            np.partition(b, NDArray.Scalar(2)).GetValue<double>(2).Should().Be(3.0);
            np.partition(b, np.array(new long[0])).ToArray<double>()
                .Should().BeEquivalentTo(new[] { 3.0, 4.0, 2.0, 1.0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Partition_NDArrayKth_DtypeRejections_Verbatim()
        {
            // the array form is what makes NumPy's kth-dtype errors REACHABLE:
            var b = np.array(new double[] { 3, 4, 2, 1 });
            ((Action)(() => np.partition(b, np.array(new[] { true }))))
                .Should().Throw<ValueError>().WithMessage("Booleans unacceptable as partition index*");
            ((Action)(() => np.partition(b, np.array(new[] { 1.0 }))))
                .Should().Throw<TypeError>().WithMessage("Partition index must be integer*");
            ((Action)(() => np.partition(b, np.array(new[,] { { 1 } }))))
                .Should().Throw<ValueError>().WithMessage("object too deep for desired array*");
            // decimal has no NumPy analog — not integer-family, takes the non-integer exit (house)
            ((Action)(() => np.partition(b, np.array(new decimal[] { 1m }))))
                .Should().Throw<TypeError>().WithMessage("Partition index must be integer*");
            // argpartition shares the route
            ((Action)(() => np.argpartition(b, np.array(new[] { true, false }))))
                .Should().Throw<ValueError>().WithMessage("Booleans unacceptable as partition index*");
            ((Action)(() => np.argpartition(b, np.array(new[] { 2.5 }))))
                .Should().Throw<TypeError>().WithMessage("Partition index must be integer*");
        }

        [TestMethod]
        public void Partition_NDArrayKth_ValidationStaging()
        {
            // probed NumPy staging: kind → order → too-deep (BEFORE axis) → axis → bool/float dtype
            var b = np.array(new double[] { 3, 4, 2, 1 });
            ((Action)(() => np.partition(b, np.array(new[,] { { 1 } }), kind: "bad")))
                .Should().Throw<ValueError>().WithMessage("select kind must be*");
            ((Action)(() => np.partition(b, np.array(new[,] { { 1 } }), order: "f")))
                .Should().Throw<ValueError>().WithMessage("Cannot specify order*");
            ((Action)(() => np.partition(b, np.array(new[,] { { 1 } }), axis: 9)))
                .Should().Throw<ValueError>().WithMessage("object too deep for desired array*");
            ((Action)(() => np.partition(b, np.array(new[] { true }), axis: 9)))
                .Should().Throw<ArgumentException>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void Partition_NDArrayKth_UInt64_ModularWrap()
        {
            // NumPy casts kth to intp MODULARLY: 2^63 wraps negative (its bounds error reports the
            // wrapped value), 2^64-1 wraps to -1 — a LEGAL last-element kth (probed verbatim)
            var b = np.array(new double[] { 3, 4, 2, 1 });
            ((Action)(() => np.partition(b, np.array(new ulong[] { 9223372036854775808UL }))))
                .Should().Throw<ValueError>().WithMessage("kth(=-9223372036854775804) out of bounds (4)*");
            var r = np.partition(b, np.array(new ulong[] { 18446744073709551615UL }));
            r.GetValue<double>(3).Should().Be(4.0);
        }

        [TestMethod]
        public void Partition_StridedDecimalAndComplex_GatherPath()
        {
            // decimal/Complex never ride the NumPy corpus (no analog / lexicographic axis-0 only),
            // so the strided gather→introselect→scatter branch gets pinned here explicitly.
            var d = np.array(new decimal[] { 9, 1, 8, 2, 7, 3 })["::2"];   // [9, 8, 7]
            var rd = np.partition(d, 1);
            rd.GetValue<decimal>(1).Should().Be(8m);
            var gd = np.argpartition(d, 1);
            d.GetValue<decimal>((int)gd.GetValue<long>(1)).Should().Be(8m);

            var c = np.array(new[] { new Complex(9, 0), new Complex(1, 0), new Complex(8, 0), new Complex(2, 0), new Complex(7, 0), new Complex(3, 0) })["::2"];
            var rc = np.partition(c, 1);
            rc.GetValue<Complex>(1).Should().Be(new Complex(8, 0));
        }
    }

    [TestClass]
    public class NpArgPartitionTests
    {
        /// <summary>Asserts NumPy's argpartition contract: result is a permutation of 0..n-1 and
        /// gathering through it yields a valid partition (kth values == sorted values).</summary>
        private static void AssertArgPartitionInvariant(NDArray input1d, NDArray indices, params int[] ks)
        {
            indices.typecode.Should().Be(NPTypeCode.Int64);
            int n = (int)input1d.size;
            var seen = new bool[n];
            for (int i = 0; i < n; i++)
            {
                long ix = indices.GetValue<long>(i);
                ix.Should().BeInRange(0, n - 1);
                seen[ix].Should().BeFalse("indices must be a permutation");
                seen[ix] = true;
            }
            var gathered = input1d.astype(np.float64);
            var srt = np.sort(gathered, null);
            foreach (var k in ks)
            {
                double v = gathered.GetValue<double>((int)indices.GetValue<long>(k));
                double sv = srt.GetValue<double>(k);
                if (double.IsNaN(sv)) double.IsNaN(v).Should().BeTrue($"a[g[{k}]] must be NaN");
                else v.Should().Be(sv, $"a[g[{k}]] must equal sorted[{k}]");
            }
        }

        [TestMethod]
        public void ArgPartition_1D_Basic()
        {
            // np.argpartition([3,4,2,1], 3) == [3,2,0,1] (probed — NumPy's own arrangement matched)
            var a = np.array(new[] { 3, 4, 2, 1 });
            var g = np.argpartition(a, 3);
            a.GetValue<int>((int)g.GetValue<long>(3)).Should().Be(4);
            AssertArgPartitionInvariant(a, g, 3);
        }

        [TestMethod]
        public void ArgPartition_ReturnsInt64_InputUntouched()
        {
            var a = np.array(new[] { 3.0, 1.0, 2.0 });
            var g = np.argpartition(a, 0);
            g.typecode.Should().Be(NPTypeCode.Int64);
            a.ToArray<double>().Should().BeEquivalentTo(new[] { 3.0, 1.0, 2.0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void ArgPartition_MultiKth()
        {
            // np.argpartition(desc10, (7,2)) — kths sorted internally, both positions exact
            var a = np.array(new double[] { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });
            var g = np.argpartition(a, new[] { 7, 2 });
            AssertArgPartitionInvariant(a, g, 2, 7);
        }

        [TestMethod]
        public void ArgPartition_NaN_TailKeepsEncounterOrder()
        {
            // np.argpartition([nan,1,7,2], 1) == [1,3,2,0] (probed): value at kth=1 is 2.0, NaN index last
            var a = np.array(new[] { double.NaN, 1.0, 7.0, 2.0 });
            var g = np.argpartition(a, 1);
            a.GetValue<double>((int)g.GetValue<long>(1)).Should().Be(2.0);
            g.GetValue<long>(3).Should().Be(0L, "the single NaN's index sits in the tail");
            AssertArgPartitionInvariant(a, g, 1);
        }

        [TestMethod]
        public void ArgPartition_2D_Axis0()
        {
            // np.argpartition([[3,1],[2,0]], 0, axis=0) == [[1,1],[0,0]] (probed exact)
            var g = np.argpartition(np.array(new[,] { { 3, 1 }, { 2, 0 } }), 0, axis: 0);
            g.GetValue<long>(0, 0).Should().Be(1);
            g.GetValue<long>(0, 1).Should().Be(1);
            g.GetValue<long>(1, 0).Should().Be(0);
            g.GetValue<long>(1, 1).Should().Be(0);
        }

        [TestMethod]
        public void ArgPartition_AxisNone_FlattensIndices()
        {
            // np.argpartition([[3,1],[2,0]], 2, axis=None) == [3,1,2,0] (probed exact)
            var g = np.argpartition(np.array(new[,] { { 3, 1 }, { 2, 0 } }), 2, axis: null);
            g.Shape.Should().Be(new Shape(4));
            g.GetValue<long>(0).Should().Be(3);
        }

        [TestMethod]
        public void ArgPartition_0d_RavelsToOne()
        {
            // NumPy's arg-side 0-d quirk: np.argpartition(np.array(5.0), 0) == array([0]) shape (1,)
            var g = np.argpartition(NDArray.Scalar(5.0), 0);
            g.Shape.Should().Be(new Shape(1));
            g.GetValue<long>(0).Should().Be(0L);
            // axis errors then report dimension 1, kth errors size 1 (probed verbatim)
            ((Action)(() => np.argpartition(NDArray.Scalar(5.0), 0, axis: 5)))
                .Should().Throw<ArgumentException>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
            ((Action)(() => np.argpartition(NDArray.Scalar(5.0), 3)))
                .Should().Throw<ValueError>().WithMessage("kth(=3) out of bounds (1)*");
        }

        [TestMethod]
        public void ArgPartition_Empty_SkipsKthValidation()
        {
            var g = np.argpartition(np.array(new double[0]), 5);
            g.size.Should().Be(0);
            g.typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void ArgPartition_ReadOnlyInput_Works()
        {
            // argpartition only reads — a broadcast (read-only) input is fine, like NumPy
            var bc = np.broadcast_to(np.array(new[] { 3.0 }), new Shape(4));
            var g = np.argpartition(bc, 1);
            g.size.Should().Be(4);
        }

        [TestMethod]
        public void ArgPartition_Complex()
        {
            // np.argpartition([3+1j,1+2j,2-1j,1+0j], 1) == [3,1,2,0] (probed): x[g[1]] == 1+2j
            var x = np.array(new[] { new Complex(3, 1), new Complex(1, 2), new Complex(2, -1), new Complex(1, 0) });
            var g = np.argpartition(x, 1);
            x.GetValue<Complex>((int)g.GetValue<long>(1)).Should().Be(new Complex(1, 2));
        }

        [TestMethod]
        public void ArgPartition_StridedView()
        {
            // np.argpartition(a[::2], 1) — indices address the VIEW: a[::2][g[1]] == 8
            var a = np.array(new[] { 9.0, 1.0, 8.0, 2.0, 7.0, 3.0 });
            var v = a["::2"];
            var g = np.argpartition(v, 1);
            v.GetValue<double>((int)g.GetValue<long>(1)).Should().Be(8.0);
        }

        [TestMethod]
        public void ArgPartition_ValidationSharedWithPartition()
        {
            var a = np.array(new[] { 3.0, 1.0, 2.0 });
            ((Action)(() => np.argpartition(a, 1, kind: "bad")))
                .Should().Throw<ValueError>().WithMessage("select kind must be 'introselect' (got 'bad')*");
            ((Action)(() => np.argpartition(a, 1, order: "x")))
                .Should().Throw<ValueError>().WithMessage("Cannot specify order when the array has no fields.*");
            ((Action)(() => np.argpartition(a, 99)))
                .Should().Throw<ValueError>().WithMessage("kth(=99) out of bounds (3)*");
        }

        [TestMethod]
        public void ArgPartition_AllDtypes_GatherMatchesSorted()
        {
            void Check<T>(T[] values, int k) where T : unmanaged
            {
                var a = np.array(values);
                var g = np.argpartition(a, k);
                var s = np.sort(a, null);
                a.GetValue<T>((int)g.GetValue<long>(k)).Should().Be(s.GetValue<T>(k), $"dtype {a.typecode}");
            }

            Check(new[] { true, false, true }, 1);
            Check(new byte[] { 9, 1, 5, 3 }, 2);
            Check(new sbyte[] { -9, 1, -5, 3 }, 2);
            Check(new short[] { -900, 100, -500, 300 }, 1);
            Check(new ushort[] { 900, 100, 500, 300 }, 1);
            Check(new[] { 'd', 'a', 'c', 'b' }, 2);
            Check(new[] { -90000, 10000, -50000, 30000 }, 2);
            Check(new uint[] { 90000, 10000, 50000, 30000 }, 2);
            Check(new[] { -9000000000L, 1000000000L, -5000000000L, 3000000000L }, 1);
            Check(new ulong[] { 9223372036854775813UL, 3, 18446744073709551615UL }, 1);
            Check(new[] { (Half)9, (Half)1, (Half)5, (Half)3 }, 2);
            Check(new[] { 9f, 1f, 5f, 3f }, 2);
            Check(new[] { 9.0, 1.0, 5.0, 3.0 }, 2);
            Check(new decimal[] { 9, 1, 5, 3 }, 2);
        }
    }
}
