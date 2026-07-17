using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using AwesomeAssertions;

namespace NumSharp.UnitTest
{
    /// <summary>
    /// Regression guards for typed data extraction on a dtype MISMATCH.
    ///
    /// Context (interop finding "C3"): Numpy.NET's <c>NDarray.GetData&lt;int&gt;()</c> on an int64 array
    /// silently REINTERPRETS the raw buffer as int32 and truncates to the element count, e.g.
    /// <c>[1, 5_000_000_000, -1]</c> -&gt; <c>[1, 0, 705032704]</c> — which is numpy's
    /// <c>a.view(np.int32)[:3]</c>, i.e. raw byte garbage, NOT a value conversion. That is data corruption.
    ///
    /// NumSharp must never behave that way. The contract these tests pin:
    ///   * <see cref="NDArray.GetData{T}"/> performs a VALUE CAST that matches numpy's <c>a.astype(...)</c> exactly.
    ///   * <see cref="NDArray.ToArray{T}"/> is STRICT and throws <see cref="ArrayTypeMismatchException"/> on mismatch.
    /// Oracle values were produced with NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class GetDataTypeMismatchTests
    {
        private static int[] GetInts(NDArray nd)
        {
            var g = nd.GetData<int>();
            var r = new int[nd.size];
            for (int i = 0; i < r.Length; i++) r[i] = g[i];
            return r;
        }

        private static long[] GetLongs(NDArray nd)
        {
            var g = nd.GetData<long>();
            var r = new long[nd.size];
            for (int i = 0; i < r.Length; i++) r[i] = g[i];
            return r;
        }

        [TestMethod]
        public void GetData_Int64ToInt32_ValueCasts_NotReinterpret()
        {
            var i64 = np.array(new long[] { 1, 5_000_000_000L, -1 });
            i64.typecode.Should().Be(NPTypeCode.Int64);

            var got = GetInts(i64);

            // numpy: np.array([1, 5_000_000_000, -1], np.int64).astype(np.int32).tolist() == [1, 705032704, -1]
            got.Should().Equal(1, 705032704, -1);

            // Must NOT equal numpy's a.view(np.int32)[:3] == [1, 0, 705032704] — the Numpy.NET "C3" reinterpret bug.
            got.Should().NotEqual(new[] { 1, 0, 705032704 });
        }

        [TestMethod]
        public void GetData_Float64ToInt32_TruncatesTowardZero_MatchesNumpyAstype()
        {
            var f64 = np.array(new double[] { 1.9, -2.9, 3_500_000_000.0 });

            // numpy: np.array([1.9, -2.9, 3.5e9]).astype(np.int32).tolist() == [1, -2, -2147483648]
            GetInts(f64).Should().Equal(1, -2, -2147483648);
        }

        [TestMethod]
        public void GetData_Int32ToInt64_Widens_MatchesNumpyAstype()
        {
            var i32 = np.array(new int[] { 1, 2, -3 });

            // numpy: np.array([1, 2, -3], np.int32).astype(np.int64).tolist() == [1, 2, -3]
            GetLongs(i32).Should().Equal(1L, 2L, -3L);
        }

        [TestMethod]
        public void GetData_MatchingType_ReturnsValuesUnchanged()
        {
            var i64 = np.array(new long[] { 1, 5_000_000_000L, -1 });

            GetLongs(i64).Should().Equal(1L, 5_000_000_000L, -1L);
        }

        [TestMethod]
        public void ToArray_TypeMismatch_Throws()
        {
            var i64 = np.array(new long[] { 1, 5_000_000_000L, -1 });

            Action act = () => i64.ToArray<int>();
            act.Should().Throw<ArrayTypeMismatchException>();
        }
    }
}
