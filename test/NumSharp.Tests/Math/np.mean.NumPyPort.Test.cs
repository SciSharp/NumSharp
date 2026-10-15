using System;
using System.Collections.Generic;
using NumSharp.Backends;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.mean</c> as NumPy's <c>_methods._mean</c>, pinned to the bit: the sum in the loop dtype (<c>dtype=</c>,
    ///     else float64 for bool / integers and float32 for float16), then <c>true_divide(ret, rcount, out=ret,
    ///     casting='unsafe')</c> with a strong <c>np.intp</c> count. Every expected string is NumPy 2.4.2's output for the
    ///     same call, in <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format (<c>dtype (shape) bits…</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Before the port (plan <c>docs/plans/api-parameter-parity.md</c> item X9, 2026-10-05), np.mean along an axis ran
    ///     its own kernel: a cast loop's elements were read as doubles and summed in one sequential pass, a complex sum was
    ///     divided part by part, float16 summed in double, an integer or bool loop on an empty input threw
    ///     <c>InvalidCastException</c> or gave a float64 NaN, a flat integer loop did not wrap, and the flat fallback read a
    ///     broadcast uint64 as int64. A 9,660-case probe found 2,936 cases off; the oracle tier <c>mean_loops.jsonl</c>
    ///     (zero excuses) covers the same ground.
    ///     </para>
    ///     <para>
    ///     There is no <c>np.mean</c> overload with <c>dtype=</c> and no axis yet (plan item B3): the flat <c>dtype=</c>
    ///     cases call the engine's np.mean, <c>TensorEngine.Mean</c>, which every np.mean overload reaches.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_mean_numpy_port_tests
    {
        /// <summary>
        ///     A complex loop divides by <c>rcount + 0j</c> with NumPy's complex division (Smith's algorithm): each part
        ///     times <c>1/n</c>, so <c>14 * (1/3)</c> is <c>…aaaa</c> where <c>14 / 3</c> is <c>…aaab</c>; an infinite part
        ///     turns the other part into NaN. The old axis route divided each part by <c>n</c>.
        /// </summary>
        [TestMethod]
        public void ComplexLoop_DividesLikeNumPysComplexLoop()
        {
            var fails = new List<string>();
            var i32 = np.array(new int[,] { { 5, 3 }, { 5, 4 }, { 4, 3 } });
            Check(fails, "mean i32 axis0 dtype=c128", () => np.mean(i32, 0, np.complex128),
                  "complex128 (2,) 4012aaaaaaaaaaaa 0000000000000000 400aaaaaaaaaaaaa 0000000000000000");
            var fi = np.array(new double[,] { { double.PositiveInfinity, 1.0 }, { 1.0, 2.0 } });
            Check(fails, "mean f64 inf axis0 dtype=c128", () => np.mean(fi, 0, np.complex128),
                  "complex128 (2,) 7ff0000000000000 fff8000000000000 3ff8000000000000 0000000000000000");
            AssertNoMismatch(fails, "NumPy's complex mean divides in its complex loop");
        }

        /// <summary>
        ///     An integer or bool loop divides in float64 and casts the quotient back unsafely: an empty slice's
        ///     <c>0 / 0</c> is NaN, which becomes INT_MIN in an integer loop (a scalar too) and True in a bool loop. The
        ///     old routes threw <c>InvalidCastException</c> along an axis and returned a float64 NaN flat.
        /// </summary>
        [TestMethod]
        public void IntegerAndBoolLoops_EmptySlice_IsTheCastOfNaN()
        {
            var fails = new List<string>();
            var e = np.zeros(new Shape(0, 3), np.int8);
            Check(fails, "mean i8 (0,3) axis0 dtype=i32", () => np.mean(e, 0, np.int32),
                  "int32 (3,) -2147483648 -2147483648 -2147483648");
            var b = np.zeros(new Shape(0), np.@bool);
            Check(fails, "mean bool (0,) dtype=i32", () => b.TensorEngine.Mean(b, null, np.int32, false), "int32 () -2147483648");
            var f = np.zeros(new Shape(3, 0));
            Check(fails, "mean f64 (3,0) axis1 dtype=bool", () => np.mean(f, 1, np.@bool), "bool (3,) True True True");
            AssertNoMismatch(fails, "NumPy casts 0/0 into the loop: INT_MIN for an integer, True for bool");
        }

        /// <summary>
        ///     A flat integer loop sums with the loop's wraparound (NumPy's int32 add), then divides in float64 and
        ///     truncates. The old fallback summed without wrapping.
        /// </summary>
        [TestMethod]
        public void FlatIntegerLoop_SumWraps()
        {
            var fails = new List<string>();
            var wrap = np.array(new[] { 2_000_000_000, 2_000_000_000, 2_000_000_000, -7 });
            Check(fails, "mean i32 dtype=i32", () => wrap.TensorEngine.Mean(wrap, null, np.int32, false), "int32 () 426258174");
            var wide = wrap.astype(np.int64) * 3L;
            Check(fails, "mean i64 dtype=i32", () => wide.TensorEngine.Mean(wide, null, np.int32, false), "int32 () 205032698");
            AssertNoMismatch(fails, "NumPy's integer loop wraps the sum before the division");
        }

        /// <summary>
        ///     A float16 input sums in float32 and returns float16: along an axis of 2,049 values the float32 sum rounds
        ///     differently from a double one. A float16 loop (<c>dtype=float16</c>) sums in float16 and overflows to inf.
        /// </summary>
        [TestMethod]
        public void Float16_SumsInItsLoop()
        {
            var fails = new List<string>();
            var h = ((np.arange(2 * 2049) % 37) * 0.173 + 8.0).astype(np.float16).reshape(2, 2049);
            Check(fails, "mean f16 (2,2049) axis1", () => np.mean(h, 1), "float16 (2,) 498d 498f");
            Check(fails, "mean f16 (2,2049) axis1 keepdims", () => np.mean(h, 1, true), "float16 (2,1) 498d 498f");
            Check(fails, "mean f16 (2049,) axis0", () => np.mean(h["0"], 0), "float16 () 498d");
            var i = (np.arange(24).astype(np.int32) * 3001).reshape(4, 6);
            Check(fails, "mean i32 (4,6) axis1 dtype=f16", () => np.mean(i, 1, np.float16), "float16 (4,) 6f53 7c00 7c00 7c00");
            AssertNoMismatch(fails, "NumPy's float16 mean sums in float32, and a float16 loop sums in float16");
        }

        /// <summary>
        ///     A cast into float32 along an axis rounds every element into float32 and sums there (the old route summed
        ///     doubles and rounded once).
        /// </summary>
        [TestMethod]
        public void Float32Loop_AlongAnAxis_SumsInFloat32()
        {
            var fails = new List<string>();
            var x = (np.arange(24).astype(np.float32) * 0.37f + 1000.0f).astype(np.float64).reshape(4, 6) * 1.0001;
            Check(fails, "mean f64 (4,6) axis0 dtype=f32", () => np.mean(x, 0, np.float32),
                  "float32 (6,) 447adb8a 447af33a 447b0ae8 447b2297 447b3a46 447b51f4");
            AssertNoMismatch(fails, "NumPy's float32 loop casts each element first");
        }

        /// <summary>
        ///     int64 values near 2**61 lose bits in float64, so the order counts: NumPy sums the innermost axis pairwise and
        ///     a slab (a non-innermost axis) sequentially; both match whether the operand is C- or F-ordered.
        /// </summary>
        [TestMethod]
        public void Int64_LargeValues_FollowNumPysOrder()
        {
            var fails = new List<string>();
            var vals = new long[33];
            for (int k = 1; k <= 33; k++)
                vals[k - 1] = k % 2 == 1 ? -k : (1L << 61) - k;
            var ib = np.array(vals).reshape(3, 11);
            Check(fails, "mean i64 (3,11) axis1", () => np.mean(ib, 1), "float64 (3,) 43ad1745d1745d17 43b1745d1745d174 43ad1745d1745d17");
            Check(fails, "mean i64 (11,3).T axis0", () => np.mean(ib.T, 0), "float64 (3,) 43ad1745d1745d17 43b1745d1745d174 43ad1745d1745d17");
            Check(fails, "mean i64 (3,11) axis0", () => np.mean(ib, 0),
                  "float64 (11,) 43a5555555555555 43b5555555555555 43a5555555555555 43b5555555555555 43a5555555555555 "
                  + "43b5555555555555 43a5555555555555 43b5555555555555 43a5555555555555 43b5555555555555 43a5555555555555");
            AssertNoMismatch(fails, "NumPy's float64 loop over int64 rounds each element and sums in its schedule");
        }

        /// <summary>
        ///     A flat mean of a broadcast uint64 operand reads each element as uint64 (the old fallback read
        ///     <c>2**64 - 1</c> as -1: 101.8 instead of 3.69e18).
        /// </summary>
        [TestMethod]
        public void BroadcastUInt64_Flat_ReadsUnsigned()
        {
            var fails = new List<string>();
            var big = np.array(new ulong[] { 0, ulong.MaxValue, 127, 128, 255 });
            var b = np.broadcast_to(big, new Shape(4, 5));
            Check(fails, "mean u64 broadcast (4,5)", () => np.mean(b), "float64 () 43c999999999999a");
            AssertNoMismatch(fails, "NumPy's mean of uint64 elements is unsigned");
        }

        /// <summary>
        ///     np.mean normalizes the axis itself, so a 0-d input refuses every axis (np.sum would take 0 and -1).
        /// </summary>
        [TestMethod]
        public void ZeroD_RefusesEveryAxis()
        {
            var fails = new List<string>();
            var s = NDArray.Scalar(3.0);
            CheckThrows<AxisError>(fails, "mean f64 0-d axis0", () => np.mean(s, 0), "axis 0 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "mean f64 0-d axis0 dtype=i32", () => np.mean(s, 0, np.int32), "axis 0 is out of bounds for array of dimension 0");
            AssertNoMismatch(fails, "NumPy's _count_reduce_items normalizes the axis against the operand's rank");
        }

        /// <summary>
        ///     A flat mean of a decimal array stays decimal (NumSharp's own dtype, no NumPy analog): the axis route and the
        ///     decimal oracle always kept decimal, the flat fallback returned float64.
        /// </summary>
        [TestMethod]
        public void Decimal_Flat_StaysDecimal()
        {
            var d = np.array(new[] { 1.5m, 2.25m, 3m, 4.125m });
            var r = np.mean(d);
            r.typecode.Should().Be(NPTypeCode.Decimal);
            r.GetValue<decimal>().Should().Be(10.875m / 4m);
        }
    }
}
