using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// np.union1d / intersect1d / setxor1d / setdiff1d parity with NumPy 2.4.2
    /// (arraysetops — sorted/unique set operations, flattened inputs, promoted dtype).
    /// </summary>
    [TestClass]
    public class SetOpsTests
    {
        // np.array_equal treats NaN != NaN (NumPy's equal_nan=False default), so results carrying NaN
        // need a NaN-aware element compare (NaN matches NaN by position; finite values by value).
        private static void AssertDoubleEqual(NDArray result, double[] expected)
        {
            result.typecode.Should().Be(NPTypeCode.Double);
            var got = result.ToArray<double>();
            got.Length.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                if (double.IsNaN(expected[i])) double.IsNaN(got[i]).Should().BeTrue($"index {i}");
                else got[i].Should().Be(expected[i], $"index {i}");
            }
        }

        // ---- union1d --------------------------------------------------------------------
        [TestMethod]
        public void Union_Basic()
        {
            np.union1d(np.array(new long[] { -1, 0, 1 }), np.array(new long[] { -2, 0, 2 }))
                .array_equal(np.array(new long[] { -2, -1, 0, 1, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Union_MixedDtype_Promotes()
        {
            var r = np.union1d(np.array(new sbyte[] { 1, 2 }), np.array(new[] { 3, 400 }));
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.array_equal(np.array(new[] { 1, 2, 3, 400 })).Should().BeTrue();
        }

        [TestMethod]
        public void Union_2D_Flattened()
        {
            np.union1d(np.array(new long[,] { { 1, 2 } }), np.array(new long[,] { { 3, 4 }, { 1, 5 } }))
                .array_equal(np.array(new long[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Union_Empty_IsFloat64()
        {
            var r = np.union1d(np.zeros(new Shape(0), NPTypeCode.Double), np.zeros(new Shape(0), NPTypeCode.Double));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Union_Nan_KeptAtEnd()
        {
            AssertDoubleEqual(np.union1d(np.array(new[] { 1.0, double.NaN }), np.array(new[] { double.NaN, 2.0 })),
                new[] { 1.0, 2.0, double.NaN });
        }

        // ---- intersect1d ----------------------------------------------------------------
        [TestMethod]
        public void Intersect_Basic()
        {
            np.intersect1d(np.array(new long[] { 1, 3, 4, 3 }), np.array(new long[] { 3, 1, 2, 1 }))
                .array_equal(np.array(new long[] { 1, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Intersect_MixedDtype_Promotes()
        {
            var r = np.intersect1d(np.array(new[] { 1, 2, 3 }), np.array(new[] { 2.0, 3.0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.array_equal(np.array(new[] { 2.0, 3.0 })).Should().BeTrue();
        }

        [TestMethod]
        public void Intersect_ReturnIndices()
        {
            // xy, x_ind, y_ind = intersect1d([1,1,2,3,4], [2,1,4,6], return_indices=True)
            var r = np.intersect1d(np.array(new long[] { 1, 1, 2, 3, 4 }), np.array(new long[] { 2, 1, 4, 6 }),
                assume_unique: false, return_indices: true);
            r.Length.Should().Be(3);
            r[0].array_equal(np.array(new long[] { 1, 2, 4 })).Should().BeTrue();   // values
            r[1].array_equal(np.array(new long[] { 0, 2, 4 })).Should().BeTrue();   // first idx in ar1
            r[2].array_equal(np.array(new long[] { 1, 0, 2 })).Should().BeTrue();   // first idx in ar2
        }

        [TestMethod]
        public void Intersect_ReturnIndices_AssumeUnique()
        {
            var r = np.intersect1d(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 2, 3, 4 }),
                assume_unique: true, return_indices: true);
            r[0].array_equal(np.array(new long[] { 2, 3 })).Should().BeTrue();
            r[1].array_equal(np.array(new long[] { 1, 2 })).Should().BeTrue();
            r[2].array_equal(np.array(new long[] { 0, 1 })).Should().BeTrue();
        }

        [TestMethod]
        public void Intersect_Empty()
        {
            var r = np.intersect1d(np.array(new long[] { 1, 2 }), np.array(new long[] { 3, 4 }));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void Intersect_Nan_Excluded()
        {
            // NaN != NaN, so a shared NaN is NOT in the intersection.
            np.intersect1d(np.array(new[] { 1.0, double.NaN }), np.array(new[] { double.NaN, 1.0 }))
                .array_equal(np.array(new[] { 1.0 })).Should().BeTrue();
        }

        // ---- setxor1d -------------------------------------------------------------------
        [TestMethod]
        public void Setxor_Basic()
        {
            np.setxor1d(np.array(new long[] { 1, 2, 3, 2, 4 }), np.array(new long[] { 2, 3, 5, 7, 5 }))
                .array_equal(np.array(new long[] { 1, 4, 5, 7 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setxor_AssumeUnique()
        {
            np.setxor1d(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 2, 3, 4 }), assume_unique: true)
                .array_equal(np.array(new long[] { 1, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setxor_Empty_IsFloat64()
        {
            var r = np.setxor1d(np.zeros(new Shape(0), NPTypeCode.Double), np.zeros(new Shape(0), NPTypeCode.Double));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Setxor_Nan_BothKept()
        {
            // Each NaN is distinct (NaN != NaN), so both survive the xor.
            AssertDoubleEqual(np.setxor1d(np.array(new[] { 1.0, double.NaN }), np.array(new[] { double.NaN, 2.0 })),
                new[] { 1.0, 2.0, double.NaN, double.NaN });
        }

        // ---- setdiff1d ------------------------------------------------------------------
        [TestMethod]
        public void Setdiff_Basic()
        {
            np.setdiff1d(np.array(new long[] { 1, 2, 3, 2, 4, 1 }), np.array(new long[] { 3, 4, 5, 6 }))
                .array_equal(np.array(new long[] { 1, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setdiff_AssumeUnique_KeepsDuplicates()
        {
            // assume_unique skips the unique() on ar1, so its duplicates are preserved.
            np.setdiff1d(np.array(new long[] { 1, 1, 2, 3 }), np.array(new long[] { 3 }), assume_unique: true)
                .array_equal(np.array(new long[] { 1, 1, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setdiff_ResultDtype_IsAr1Dtype()
        {
            var r = np.setdiff1d(np.array(new sbyte[] { 1, 2, 3 }), np.array(new long[] { 2 }));
            r.typecode.Should().Be(NPTypeCode.SByte);
            r.array_equal(np.array(new sbyte[] { 1, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setdiff_EmptyAr1_IsFloat64()
        {
            var r = np.setdiff1d(np.zeros(new Shape(0), NPTypeCode.Double), np.array(new long[] { 1, 2 }));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Setdiff_Nan_Kept()
        {
            AssertDoubleEqual(np.setdiff1d(np.array(new[] { 1.0, double.NaN, 2.0 }), np.array(new[] { 2.0 })),
                new[] { 1.0, double.NaN });
        }
    }
}
