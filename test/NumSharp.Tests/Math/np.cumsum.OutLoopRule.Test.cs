using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.cumsum / np.cumprod / np.nancumsum / np.nancumprod with an <c>out=</c> and no <c>dtype=</c> are NumPy's
    ///     <c>add.accumulate</c> / <c>multiply.accumulate</c> with an out operand: the loop is
    ///     <c>promote_types(out.dtype, a.dtype)</c> without the small-integer widening an accumulation applies without
    ///     <c>out</c> (<c>reducelike_promote_and_resolve</c>, DefaultEngine.AccumulateLoop), the result stored with an unsafe
    ///     cast. Every literal was probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Before the fix the accumulations ran in NEP 50's accumulator whatever the out: an int8 run into a bool out stored
    ///     the truth of the int64 running sum (True where NumPy's int8 sum wraps to 0 and stores False), a float32 run into a
    ///     float64 out added in float32, a float16 product into a float32 out overflowed to inf.
    /// </remarks>
    [TestClass]
    public class np_cumsum_out_loop_rule_tests
    {
        /// <summary>The add accumulation's loop is promote_types(out, a): it narrows, widens and changes kind with the out.</summary>
        [TestMethod]
        public void CumSum_LoopIsThePromotionOfOutAndInput()
        {
            // int8 [100, 100, 56] into bool: the int8 loop's running sum wraps to 0 at the third element.
            using (var a = np.array(new sbyte[] { 100, 100, 56 }))
            using (var o = np.zeros(new Shape(3), np.@bool))
            {
                var r = np.cumsum(a, null, null, o);
                ReferenceEquals(r, o).Should().BeTrue();
                o.ToArray<bool>().Should().Equal(true, true, false);
            }
            // float32 [16777216, 1, 1] into float64: added in float64, not in float32 (where 16777216 + 1 rounds back).
            using (var a = np.array(new float[] { 16777216f, 1f, 1f }))
            using (var o = np.zeros(new Shape(3), np.float64))
            {
                np.cumsum(a, null, null, o);
                o.ToArray<double>().Should().Equal(16777216.0, 16777217.0, 16777218.0);
            }
            // uint64 [2**63 + 1, 0] into int64: promote_types(int64, uint64) is float64, whose 9.223372036854776e18 casts to
            // int64 as INT64_MIN (NumPy's unsafe float -> int cast), not the uint64 sum's wrapped bits.
            using (var a = np.array(new ulong[] { (1UL << 63) + 1, 0 }))
            using (var o = np.zeros(new Shape(2), np.int64))
            {
                np.cumsum(a, null, null, o);
                o.ToArray<long>().Should().Equal(long.MinValue, long.MinValue);
            }
            // A dtype= still decides the loop: int8 into bool with dtype=int64 sums 256 and stores True.
            using (var a = np.array(new sbyte[] { 100, 100, 56 }))
            using (var o = np.zeros(new Shape(3), np.@bool))
            {
                np.cumsum(a, null, np.int64, o);
                o.ToArray<bool>().Should().Equal(true, true, true);
            }
        }

        /// <summary>The multiply accumulation follows the same rule.</summary>
        [TestMethod]
        public void CumProd_LoopIsThePromotionOfOutAndInput()
        {
            // int8 [16, 16] into bool: 16 * 16 wraps to 0 in the int8 loop.
            using (var a = np.array(new sbyte[] { 16, 16 }))
            using (var o = np.zeros(new Shape(2), np.@bool))
            {
                var r = np.cumprod(a, null, null, o);
                ReferenceEquals(r, o).Should().BeTrue();
                o.ToArray<bool>().Should().Equal(true, false);
            }
            // float16 [300, 300] into float32: multiplied in float32 (90000), not in float16 (inf).
            using (var a = np.array(new Half[] { (Half)300, (Half)300 }))
            using (var o = np.zeros(new Shape(2), np.float32))
            {
                np.cumprod(a, null, null, o);
                o.ToArray<float>().Should().Equal(300f, 90000f);
            }
        }

        /// <summary>
        ///     np.nancumsum / np.nancumprod are np.cumsum / np.cumprod of the NaN-replaced input with the same keywords, so
        ///     they follow the rule: the NaN counts as 0 (1) and the run is accumulated in the out's promotion.
        /// </summary>
        [TestMethod]
        public void NanAccumulations_FollowTheRule()
        {
            using (var a = np.array(new float[] { 16777216f, float.NaN, 1f, 1f }))
            using (var o = np.zeros(new Shape(4), np.float64))
            {
                np.nancumsum(a, null, null, o);
                o.ToArray<double>().Should().Equal(16777216.0, 16777216.0, 16777217.0, 16777218.0);
            }
            using (var a = np.array(new Half[] { (Half)300, Half.NaN, (Half)300 }))
            using (var o = np.zeros(new Shape(3), np.float32))
            {
                np.nancumprod(a, null, null, o);
                o.ToArray<float>().Should().Equal(300f, 300f, 90000f);
            }
            using (var a = np.array(new sbyte[] { 100, 100, 56 }))
            using (var o = np.zeros(new Shape(3), np.@bool))
            {
                np.nancumsum(a, null, null, o);
                o.ToArray<bool>().Should().Equal(true, true, false);
            }
        }

        /// <summary>
        ///     The running value stays in the loop dtype across NumPy's buffer fills: an int8 run of 24,576 elements (three
        ///     np.getbufsize() fills) into a bool out is False at every third element and only there.
        /// </summary>
        [TestMethod]
        public void CumSum_RunLongerThanTheBuffer_KeepsTheLoopValue()
        {
            const int n = 3 * 8192;
            using var a = np.tile(np.array(new sbyte[] { 100, 100, 56 }), n / 3);
            using var o = np.zeros(new Shape(n), np.@bool);
            np.cumsum(a, null, null, o);
            var values = o.ToArray<bool>();
            var falses = Enumerable.Range(0, n).Where(i => !values[i]).ToArray();
            falses.Length.Should().Be(8192);
            falses.Take(5).Should().Equal(2, 5, 8, 11, 14);
            falses.All(i => i % 3 == 2).Should().BeTrue();
        }
    }
}
