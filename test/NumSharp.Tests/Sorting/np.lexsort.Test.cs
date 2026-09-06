using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    /// np.lexsort parity with NumPy 2.4.2 (every expected value produced by running NumPy).
    /// Implementation: port of PyArray_LexSort — successive STABLE argsorts from the first key to
    /// the last (the LAST key is the primary sort key), each pass re-sorting the running
    /// permutation via take_along_axis. NumSharp's radix argsort is stable, so results are
    /// bit-identical to NumPy's mergesort passes.
    /// </summary>
    [TestClass]
    public class NpLexsortTests
    {
        private static void AssertIdx(NDArray r, params long[] expected)
        {
            r.dtype.Should().Be(typeof(long));
            r.ToArray<long>().Should().BeEquivalentTo(expected, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Lexsort_LastKeyIsPrimary()
        {
            // a=[1,5,1,4,3,4,4], b=[9,4,0,4,0,2,1]
            // np.lexsort((a, b)) == [2,4,6,5,3,1,0]  (sorted by b, ties by a)
            // np.lexsort((b, a)) == [2,0,4,6,5,3,1]  (sorted by a, ties by b — the doc example)
            var a = np.array(new[] { 1, 5, 1, 4, 3, 4, 4 });
            var b = np.array(new[] { 9, 4, 0, 4, 0, 2, 1 });
            AssertIdx(np.lexsort(new[] { a, b }), 2, 4, 6, 5, 3, 1, 0);
            AssertIdx(np.lexsort(new[] { b, a }), 2, 0, 4, 6, 5, 3, 1);
        }

        [TestMethod]
        public void Lexsort_SingleKey_IsStableArgsort()
        {
            // np.lexsort((k,)) == np.argsort(k, kind='stable')
            AssertIdx(np.lexsort(new[] { np.array(new[] { 3, 1, 2 }) }), 1, 2, 0);
        }

        [TestMethod]
        public void Lexsort_Stability_AllTiesKeepInputOrder()
        {
            // np.lexsort(([1,1,1,1],)) == [0,1,2,3]
            AssertIdx(np.lexsort(new[] { np.array(new[] { 1, 1, 1, 1 }) }), 0, 1, 2, 3);
            // all-tie PRIMARY key falls through to the earlier key, stably:
            // np.lexsort(([1,2,1,2], [0,0,0,0])) == [0,2,1,3]
            AssertIdx(np.lexsort(new[] { np.array(new[] { 1, 2, 1, 2 }), np.array(new[] { 0, 0, 0, 0 }) }), 0, 2, 1, 3);
        }

        [TestMethod]
        public void Lexsort_2DArray_RowsAreKeys()
        {
            // np.lexsort(np.array([[3,1,2],[1,1,0]])) == [2,1,0] — last ROW is the primary key
            AssertIdx(np.lexsort(np.array(new[,] { { 3, 1, 2 }, { 1, 1, 0 } })), 2, 1, 0);
        }

        [TestMethod]
        public void Lexsort_1DArray_DegeneratesToScalarKeys()
        {
            // np.lexsort(np.array([3,1,2])) == np.int64(0), shape () — each element is a 0-d key
            // (NumPy's probed sequence-of-scalars quirk, not an error)
            var r = np.lexsort(np.array(new[] { 3, 1, 2 }));
            r.Shape.IsScalar.Should().BeTrue();
            ((long)r).Should().Be(0L);
        }

        [TestMethod]
        public void Lexsort_2DKeys_LastAxisDefault()
        {
            // k0=[[3,1,2],[1,1,0]], k1=[[0,0,0],[1,0,1]]:
            // np.lexsort((k0, k1)) == [[1,2,0],[1,2,0]]
            var k0 = np.array(new[,] { { 3, 1, 2 }, { 1, 1, 0 } });
            var k1 = np.array(new[,] { { 0, 0, 0 }, { 1, 0, 1 } });
            var r = np.lexsort(new[] { k0, k1 });
            r.Shape.Should().Be(new Shape(2, 3));
            r.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 2, 0, 1, 2, 0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Lexsort_2DKeys_Axis0()
        {
            // np.lexsort((k0, k1), axis=0) == [[0,1],[1,0]] (probed)
            var k0 = np.array(new[,] { { 3, 1 }, { 1, 2 } });
            var k1 = np.array(new[,] { { 0, 1 }, { 1, 0 } });
            var r = np.lexsort(new[] { k0, k1 }, 0);
            r.ToArray<long>().Should().BeEquivalentTo(new long[] { 0, 1, 1, 0 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Lexsort_3DKeys_MiddleAxis()
        {
            // k = arange(8).reshape(2,2,2)[::-1]; np.lexsort((k,), axis=1) == [[[0,0],[1,1]],[[0,0],[1,1]]]
            var k = np.arange(8).reshape(2, 2, 2)["::-1"];
            var r = np.lexsort(new[] { k }, 1);
            r.Shape.Should().Be(new Shape(2, 2, 2));
            r.ToArray<long>().Should().BeEquivalentTo(new long[] { 0, 0, 1, 1, 0, 0, 1, 1 }, o => o.WithStrictOrdering());
        }

        // -------------------- validation (verbatim NumPy) --------------------
        [TestMethod]
        public void Lexsort_EmptyKeys_Throws()
        {
            // np.lexsort(()) → TypeError: need sequence of keys with len > 0 in lexsort
            ((Action)(() => np.lexsort(new NDArray[0])))
                .Should().Throw<TypeError>().WithMessage("need sequence of keys with len > 0 in lexsort*");
            // 0-d / zero-length array forms of the same rejection
            ((Action)(() => np.lexsort(NDArray.Scalar(5.0))))
                .Should().Throw<TypeError>().WithMessage("need sequence of keys with len > 0 in lexsort*");
        }

        [TestMethod]
        public void Lexsort_MismatchedShapes_Throws()
        {
            // np.lexsort(([1,2], [1,2,3])) → ValueError: all keys need to be the same shape
            ((Action)(() => np.lexsort(new[] { np.array(new[] { 1, 2 }), np.array(new[] { 1, 2, 3 }) })))
                .Should().Throw<ValueError>().WithMessage("all keys need to be the same shape*");
            // ndim mismatch takes the same text
            ((Action)(() => np.lexsort(new[] { np.array(new[] { 1, 2 }), np.array(new[,] { { 1, 2 } }) })))
                .Should().Throw<ValueError>().WithMessage("all keys need to be the same shape*");
        }

        [TestMethod]
        public void Lexsort_AxisOutOfBounds()
        {
            // np.lexsort((k,), axis=2) → AxisError: axis 2 is out of bounds for array of dimension 1
            ((Action)(() => np.lexsort(new[] { np.array(new[] { 1, 2 }) }, 2)))
                .Should().Throw<ArgumentException>().WithMessage("axis 2 is out of bounds for array of dimension 1*");
        }

        // -------------------- degenerate shapes --------------------
        [TestMethod]
        public void Lexsort_0dKeys_AxisCompatQuirk()
        {
            // 0-d keys: axis 0 and -1 slip through (backwards compat) → np.int64(0), shape ()
            var r = np.lexsort(new[] { NDArray.Scalar(5), NDArray.Scalar(3) });
            r.Shape.IsScalar.Should().BeTrue();
            ((long)r).Should().Be(0L);
            // any other axis raises: axis 1 is out of bounds for array of dimension 0
            ((Action)(() => np.lexsort(new[] { NDArray.Scalar(5), NDArray.Scalar(3) }, 1)))
                .Should().Throw<ArgumentException>().WithMessage("axis 1 is out of bounds for array of dimension 0*");
        }

        [TestMethod]
        public void Lexsort_EmptyAndSingleElementKeys()
        {
            // np.lexsort(([], [])) == array([], dtype=int64)
            var e = np.lexsort(new[] { np.array(new double[0]), np.array(new double[0]) });
            e.size.Should().Be(0);
            e.dtype.Should().Be(typeof(long));
            // np.lexsort(([7], [2])) == [0]
            AssertIdx(np.lexsort(new[] { np.array(new[] { 7 }), np.array(new[] { 2 }) }), 0);
        }

        // -------------------- dtypes / values --------------------
        [TestMethod]
        public void Lexsort_MixedKeyDtypes()
        {
            // np.lexsort(([1.5,0.5,1.5], [2,1,0])) == [2,1,0] — each key compares in its own dtype
            AssertIdx(np.lexsort(new[] { np.array(new[] { 1.5, 0.5, 1.5 }), np.array(new[] { 2, 1, 0 }) }), 2, 1, 0);
        }

        [TestMethod]
        public void Lexsort_NaNKey_SortsLast()
        {
            // np.lexsort(([nan,1,nan,2],)) == [1,3,0,2] — NaN last, stable among NaNs
            AssertIdx(np.lexsort(new[] { np.array(new[] { double.NaN, 1.0, double.NaN, 2.0 }) }), 1, 3, 0, 2);
        }

        [TestMethod]
        public void Lexsort_ComplexKey()
        {
            // np.lexsort(([3+1j, 1+2j, 1+0j],)) == [2,1,0] — real then imag
            AssertIdx(np.lexsort(new[] { np.array(new[] { new Complex(3, 1), new Complex(1, 2), new Complex(1, 0) }) }), 2, 1, 0);
        }

        [TestMethod]
        public void Lexsort_BoolKeys()
        {
            // np.lexsort(([True,False,True], [False,False,True])) == [1,0,2]
            AssertIdx(np.lexsort(new[] { np.array(new[] { true, false, true }), np.array(new[] { false, false, true }) }), 1, 0, 2);
        }

        [TestMethod]
        public void Lexsort_StridedAndReadOnlyKeys()
        {
            // np.lexsort((a[::2], a[1::2])) == [0,1,2] for a=[9,1,8,2,7,3] (keys only read)
            var a = np.array(new[] { 9.0, 1.0, 8.0, 2.0, 7.0, 3.0 });
            AssertIdx(np.lexsort(new[] { a["::2"], a["1::2"] }), 0, 1, 2);
            // broadcast (read-only) key works — all ties → identity
            var bc = np.broadcast_to(np.array(new[] { 3.0 }), new Shape(4));
            AssertIdx(np.lexsort(new[] { bc }), 0, 1, 2, 3);
        }

        [TestMethod]
        public void Lexsort_DecimalKey()
        {
            // decimal (no NumPy analog) rides the same stable argsort route as every other dtype
            AssertIdx(np.lexsort(new[] { np.array(new decimal[] { 1.5m, 0.5m, 1.5m }), np.array(new[] { 2, 1, 0 }) }), 2, 1, 0);
        }

        [TestMethod]
        public void Lexsort_KeysNotMutated()
        {
            var x = np.array(new[] { 3, 1, 2 });
            np.lexsort(new[] { x });
            x.ToArray<int>().Should().BeEquivalentTo(new[] { 3, 1, 2 }, o => o.WithStrictOrdering());
        }

        [TestMethod]
        public void Lexsort_NullKey_Throws()
        {
            // deliberate C# divergence (house fill_diagonal precedent): a null entry is a caller bug
            ((Action)(() => np.lexsort(new NDArray[] { np.array(new[] { 1, 2 }), null })))
                .Should().Throw<ArgumentNullException>();
        }
    }
}
