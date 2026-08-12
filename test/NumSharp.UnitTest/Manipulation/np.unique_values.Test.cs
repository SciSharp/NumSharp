using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// np.unique_values / unique_counts / unique_inverse / unique_all parity with NumPy 2.4.2
    /// (Array API set functions — numpy/lib/_arraysetops_impl.py). All expected values were
    /// produced by running NumPy 2.4.2. These are unique(x, ..., equal_nan=False), so every NaN
    /// is a distinct value. indices/inverse_indices/counts are int64; values dtype is preserved.
    /// </summary>
    [TestClass]
    public class UniqueValuesTests
    {
        // ---- helpers ----------------------------------------------------------------------
        private static void AssertInt64(NDArray a, params long[] expected)
        {
            a.typecode.Should().Be(NPTypeCode.Int64);
            a.ToArray<long>().Should().Equal(expected);
        }

        private static void AssertInts(NDArray a, params int[] expected)
        {
            a.typecode.Should().Be(NPTypeCode.Int32);
            a.ToArray<int>().Should().Equal(expected);
        }

        // NaN != NaN, so a NaN-carrying values array needs NaN-aware, position-wise compare.
        private static void AssertDoubles(NDArray a, params double[] expected)
        {
            var got = a.astype(NPTypeCode.Double).ToArray<double>();
            got.Length.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                if (double.IsNaN(expected[i])) double.IsNaN(got[i]).Should().BeTrue($"index {i}");
                else got[i].Should().Be(expected[i], $"index {i}");
            }
        }

        private static void AssertShape(NDArray a, params int[] shape)
        {
            a.ndim.Should().Be(shape.Length);
            var got = a.shape;
            for (int i = 0; i < shape.Length; i++) ((long)got[i]).Should().Be(shape[i], $"dim {i}");
        }

        // ---- unique_all -------------------------------------------------------------------
        [TestMethod]
        public void UniqueAll_Basic_Int()
        {
            // >>> np.unique_all(np.array([1,2,6,4,2,3,2]))
            // values=[1 2 3 4 6] indices=[0 1 5 3 2] inverse=[0 1 4 3 1 2 1] counts=[1 3 1 1 1]
            var (values, indices, inverse, counts) = np.unique_all(np.array(new[] { 1, 2, 6, 4, 2, 3, 2 }));
            AssertInts(values, 1, 2, 3, 4, 6);
            AssertInt64(indices, 0, 1, 5, 3, 2);
            AssertInt64(inverse, 0, 1, 4, 3, 1, 2, 1);
            AssertInt64(counts, 1, 3, 1, 1, 1);
        }

        [TestMethod]
        public void UniqueAll_FirstOccurrenceIndices()
        {
            // >>> np.unique_all(np.array([3,1,2,1,3,2,5,4]))
            var r = np.unique_all(np.array(new[] { 3, 1, 2, 1, 3, 2, 5, 4 }));
            AssertInts(r.values, 1, 2, 3, 4, 5);
            AssertInt64(r.indices, 1, 2, 0, 7, 6);
            AssertInt64(r.inverse_indices, 2, 0, 1, 0, 2, 1, 4, 3);
            AssertInt64(r.counts, 2, 2, 2, 1, 1);
        }

        [TestMethod]
        public void UniqueAll_Bool()
        {
            // >>> np.unique_all(np.array([True, False, True, True]))
            var r = np.unique_all(np.array(new[] { true, false, true, true }));
            r.values.typecode.Should().Be(NPTypeCode.Boolean);
            r.values.ToArray<bool>().Should().Equal(false, true);
            AssertInt64(r.indices, 1, 0);
            AssertInt64(r.inverse_indices, 1, 0, 1, 1);
            AssertInt64(r.counts, 1, 3);
        }

        [TestMethod]
        public void UniqueAll_UInt64_Large()
        {
            // >>> np.unique_all(np.array([2**63+5, 3, 2**63+5, 1], dtype=np.uint64))
            var r = np.unique_all(np.array(new ulong[] { (ulong)long.MaxValue + 6, 3, (ulong)long.MaxValue + 6, 1 }));
            r.values.typecode.Should().Be(NPTypeCode.UInt64);
            r.values.ToArray<ulong>().Should().Equal(1, 3, (ulong)long.MaxValue + 6);
            AssertInt64(r.indices, 3, 1, 0);
            AssertInt64(r.inverse_indices, 2, 1, 2, 0);
            AssertInt64(r.counts, 1, 1, 2);
        }

        [TestMethod]
        public void UniqueAll_Complex_WithNaN()
        {
            // >>> np.unique_all(np.array([1+2j,3+0j,1+2j,complex(nan,1),2+2j]))
            var r = np.unique_all(np.array(new[]
            {
                new Complex(1, 2), new Complex(3, 0), new Complex(1, 2),
                new Complex(double.NaN, 1), new Complex(2, 2)
            }));
            r.values.typecode.Should().Be(NPTypeCode.Complex);
            var v = r.values.ToArray<Complex>();
            v.Length.Should().Be(4);
            v[0].Should().Be(new Complex(1, 2));
            v[1].Should().Be(new Complex(2, 2));
            v[2].Should().Be(new Complex(3, 0));
            (double.IsNaN(v[3].Real) && v[3].Imaginary == 1).Should().BeTrue();
            AssertInt64(r.indices, 0, 4, 1, 3);
            AssertInt64(r.inverse_indices, 0, 2, 0, 3, 1);
            AssertInt64(r.counts, 2, 1, 1, 1);
        }

        // ---- NaN semantics (equal_nan=False — the whole point of the Array API variants) ---
        [TestMethod]
        public void UniqueAll_NaN_EachDistinct()
        {
            // >>> np.unique_all(np.array([1.0, nan, 2.0, nan, 1.0, nan]))
            // values=[1 2 nan nan nan] indices=[0 2 1 3 5] inverse=[0 2 1 3 0 4] counts=[2 1 1 1 1]
            var r = np.unique_all(np.array(new[] { 1.0, double.NaN, 2.0, double.NaN, 1.0, double.NaN }));
            AssertDoubles(r.values, 1.0, 2.0, double.NaN, double.NaN, double.NaN);
            AssertInt64(r.indices, 0, 2, 1, 3, 5);
            AssertInt64(r.inverse_indices, 0, 2, 1, 3, 0, 4);
            AssertInt64(r.counts, 2, 1, 1, 1, 1);
        }

        // ---- unique_counts ----------------------------------------------------------------
        [TestMethod]
        public void UniqueCounts_Basic()
        {
            // >>> np.unique_counts(np.array([1,2,6,4,2,3,2]))
            var (values, counts) = np.unique_counts(np.array(new[] { 1, 2, 6, 4, 2, 3, 2 }));
            AssertInts(values, 1, 2, 3, 4, 6);
            AssertInt64(counts, 1, 3, 1, 1, 1);
        }

        [TestMethod]
        public void UniqueCounts_Float32_DtypePreserved()
        {
            var (values, counts) = np.unique_counts(np.array(new[] { 3f, 1f, 2f, 1f, 3f }));
            values.typecode.Should().Be(NPTypeCode.Single);
            values.ToArray<float>().Should().Equal(1f, 2f, 3f);
            AssertInt64(counts, 2, 1, 2);
        }

        // ---- unique_inverse ---------------------------------------------------------------
        [TestMethod]
        public void UniqueInverse_ReconstructInvariant()
        {
            var x = np.array(new[] { 1, 2, 6, 4, 2, 3, 2 });
            var (values, inverse) = np.unique_inverse(x);
            AssertInts(values, 1, 2, 3, 4, 6);
            AssertInt64(inverse, 0, 1, 4, 3, 1, 2, 1);
            // take(values, inverse) reconstructs x
            np.take(values, inverse).array_equal(x).Should().BeTrue();
        }

        [TestMethod]
        public void UniqueInverse_2D_ReshapesToInputShape()
        {
            // >>> np.unique_inverse(np.array([[1,2,6],[4,2,3]]))
            // inverse_indices shape (2,3) = [[0,1,4],[3,1,2]]
            var x = np.array(new[,] { { 1, 2, 6 }, { 4, 2, 3 } });
            var r = np.unique_inverse(x);
            AssertInts(r.values, 1, 2, 3, 4, 6);
            AssertShape(r.inverse_indices, 2, 3);
            r.inverse_indices.ToArray<long>().Should().Equal(0, 1, 4, 3, 1, 2);
            np.take(r.values, r.inverse_indices).array_equal(x).Should().BeTrue();
        }

        [TestMethod]
        public void UniqueInverse_3D_ReshapesToInputShape()
        {
            var x = (np.arange(24).reshape(2, 3, 4) % 5);
            var r = np.unique_inverse(x);
            AssertInt64(r.values.astype(NPTypeCode.Int64), 0, 1, 2, 3, 4);
            AssertShape(r.inverse_indices, 2, 3, 4);
            np.take(r.values, r.inverse_indices).array_equal(x).Should().BeTrue();
        }

        [TestMethod]
        public void UniqueInverse_Transposed_FlattensInCOrder()
        {
            // a.T of [[1,2,6],[4,2,3]] is shape (3,2), C-flat [1,4,2,2,6,3].
            // >>> np.unique_all(a.T): indices=[0 2 5 1 4] inverse (3,2)=[[0,3],[1,1],[4,2]]
            var at = np.array(new[,] { { 1, 2, 6 }, { 4, 2, 3 } }).T;
            var r = np.unique_all(at);
            AssertInts(r.values, 1, 2, 3, 4, 6);
            AssertInt64(r.indices, 0, 2, 5, 1, 4);
            AssertShape(r.inverse_indices, 3, 2);
            r.inverse_indices.ToArray<long>().Should().Equal(0, 3, 1, 1, 4, 2);
            AssertInt64(r.counts, 1, 2, 1, 1, 1);
        }

        // ---- 0-d input: inverse is a 0-d () scalar (NumPy 2.0) -----------------------------
        [TestMethod]
        public void Unique_ZeroDim_InverseIsScalar()
        {
            // >>> np.unique_all(np.array(5)): values=[5] indices=[0] inverse=0 (shape ()) counts=[1]
            var r = np.unique_all(np.array(5));
            AssertInt64(r.values.astype(NPTypeCode.Int64), 5);
            AssertShape(r.values, 1);
            AssertInt64(r.indices, 0);
            r.inverse_indices.ndim.Should().Be(0);   // () scalar
            r.inverse_indices.astype(NPTypeCode.Int64).ToArray<long>()[0].Should().Be(0);
            AssertInt64(r.counts, 1);
        }

        // ---- empty ------------------------------------------------------------------------
        [TestMethod]
        public void Unique_Empty_AllOutputsEmpty()
        {
            var r = np.unique_all(np.array(Array.Empty<double>()));
            r.values.size.Should().Be(0);
            r.values.typecode.Should().Be(NPTypeCode.Double);   // dtype preserved
            r.indices.size.Should().Be(0);
            r.indices.typecode.Should().Be(NPTypeCode.Int64);
            r.inverse_indices.size.Should().Be(0);
            r.inverse_indices.typecode.Should().Be(NPTypeCode.Int64);
            r.counts.size.Should().Be(0);
            r.counts.typecode.Should().Be(NPTypeCode.Int64);
        }

        // ---- unique_values ----------------------------------------------------------------
        [TestMethod]
        public void UniqueValues_ContainsCorrectSet()
        {
            // NumSharp returns SORTED (see the [Misaligned] note below). As a SET it equals NumPy.
            np.unique_values(np.array(new[] { 3, 1, 2, 1, 3, 2, 5, 4 }))
                .array_equal(np.array(new[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void UniqueValues_NaN_EachDistinct()
        {
            // equal_nan=False → 3 NaNs survive; NumSharp sorts them to the end.
            AssertDoubles(np.unique_values(np.array(new[] { 1.0, double.NaN, 2.0, double.NaN, 1.0, double.NaN })),
                1.0, 2.0, double.NaN, double.NaN, double.NaN);
        }

        [TestMethod]
        public void UniqueValues_DtypePreserved()
        {
            np.unique_values(np.array(new byte[] { 3, 1, 2, 1 })).typecode.Should().Be(NPTypeCode.Byte);
            np.unique_values(np.array(new Half[] { (Half)2, (Half)1, (Half)2 })).typecode.Should().Be(NPTypeCode.Half);
        }

        /// <summary>
        /// DELIBERATE DIVERGENCE: NumPy 2.4.2's unique_values uses sorted=False, returning values in
        /// std::unordered_set iteration order — a platform/compiler-specific hash order that is neither
        /// sorted nor first-occurrence (e.g. [0,8,16,24,1,9,17] -> [24,16,8,0,17,9,1]) and is not
        /// reproducible in C#. NumSharp returns the SORTED values instead: deterministic, portable, equal
        /// AS A SET, and consistent with the values field of unique_all/counts/inverse. The Array API
        /// leaves the order unspecified, so this is spec-compliant.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void UniqueValues_Sorted_NotNumPyHashOrder()
        {
            var v = np.unique_values(np.array(new[] { 0, 8, 16, 24, 1, 9, 17 }));
            // NumSharp: sorted. (NumPy on win-amd64: [24,16,8,0,17,9,1] — hash order.)
            v.array_equal(np.array(new[] { 0, 1, 8, 9, 16, 17, 24 })).Should().BeTrue();
        }

        // ---- result-container ergonomics --------------------------------------------------
        [TestMethod]
        public void ResultTypes_DeconstructIndexerImplicitConversion()
        {
            var x = np.array(new[] { 1, 2, 2, 3 });

            var all = np.unique_all(x);
            all.Length.Should().Be(4);
            all[0].array_equal(all.values).Should().BeTrue();
            all[3].array_equal(all.counts).Should().BeTrue();
            NDArray[] asArray = all;   // implicit conversion, NumPy field order
            asArray.Length.Should().Be(4);
            asArray[2].array_equal(all.inverse_indices).Should().BeTrue();

            var counts = np.unique_counts(x);
            counts.Length.Should().Be(2);
            counts[1].array_equal(counts.counts).Should().BeTrue();

            var inv = np.unique_inverse(x);
            inv.Length.Should().Be(2);
            inv[1].array_equal(inv.inverse_indices).Should().BeTrue();
        }
    }
}
