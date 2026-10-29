using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Pins <see cref="np.gcd(NDArray,NDArray,NDArray,NDArray,DType)"/> and
    ///     <see cref="np.lcm(NDArray,NDArray,NDArray,NDArray,DType)"/> against NumPy 2.4.2. The broad
    ///     value × dtype × layout matrix is gated bit-exactly by the differential-fuzz <c>gcd</c> tier;
    ///     these tests target what that tier does NOT reach — the precise wrapping edges (signed MIN,
    ///     lcm overflow), the promotion rules, and the full error taxonomy (integer-only loops, the
    ///     uint64+signed → float64 no-loop, and the <c>dtype=</c> loop-selection errors) — so a regression
    ///     in any of those surfaces here rather than silently.
    /// </summary>
    [TestClass]
    public class GcdLcmTest
    {
        /// <summary>Read a scalar element as a signed 64-bit value, converting from the array's native
        /// dtype (so a narrow int8/int16 result is widened, never byte-reinterpreted).</summary>
        /// <param name="a">A size-1 / scalar result array.</param>
        /// <returns>The first element as <see cref="long"/>.</returns>
        private static long S(NDArray a) => Convert.ToInt64(a.GetAtIndex(0));

        /// <summary>Read a scalar element as an unsigned 64-bit value (for the uint dtype results).</summary>
        /// <param name="a">A size-1 / scalar result array.</param>
        /// <returns>The first element as <see cref="ulong"/>.</returns>
        private static ulong U(NDArray a) => Convert.ToUInt64(a.GetAtIndex(0));

        // ---------------------------------------------------------------- values

        /// <summary>Basic gcd/lcm identities including the two zero edges NumPy pins: gcd(0,0)==0,
        /// gcd(0,x)==|x|, lcm(x,0)==0 (never a divide-by-zero).</summary>
        [TestMethod]
        public void BasicValues_And_ZeroEdges()
        {
            Assert.AreEqual(4L, S(np.gcd(np.array(new[] { 12 }), np.array(new[] { 8 }))));
            Assert.AreEqual(12L, S(np.lcm(np.array(new[] { 4 }), np.array(new[] { 6 }))));
            Assert.AreEqual(0L, S(np.gcd(np.array(new[] { 0 }), np.array(new[] { 0 }))));
            Assert.AreEqual(5L, S(np.gcd(np.array(new[] { 0 }), np.array(new[] { 5 }))));
            Assert.AreEqual(0L, S(np.lcm(np.array(new[] { 0 }), np.array(new[] { 5 }))));
            Assert.AreEqual(0L, S(np.lcm(np.array(new[] { 0 }), np.array(new[] { 0 }))));
        }

        /// <summary>gcd/lcm operate on MAGNITUDES: the sign of either operand never reaches the result
        /// (gcd/lcm of ±12,±8 are all 4/24), matching NumPy's <c>npy_gcd(a,b)=npy_gcdu(|a|,|b|)</c>.</summary>
        [TestMethod]
        public void NegativeOperands_UseMagnitude()
        {
            foreach (var (a, b) in new[] { (-12, 8), (12, -8), (-12, -8) })
                Assert.AreEqual(4L, S(np.gcd(np.array(new[] { a }), np.array(new[] { b }))), $"gcd({a},{b})");
            foreach (var (a, b) in new[] { (-4, 6), (4, -6), (-4, -6) })
                Assert.AreEqual(12L, S(np.lcm(np.array(new[] { a }), np.array(new[] { b }))), $"lcm({a},{b})");
        }

        /// <summary>The signed-MIN wrap edge, which is the one case a result can be NEGATIVE:
        /// |int8 -128| == 128 has no positive int8, so gcd(-128,-128) and lcm(-128,±) wrap to -128 —
        /// bit-for-bit NumPy. A naive "abs then gcd" that saturated instead of wrapping would fail here.</summary>
        [TestMethod]
        public void SignedMinValue_WrapsNegative_LikeNumPy()
        {
            Assert.AreEqual(-128L, S(np.gcd(np.array(new sbyte[] { -128 }), np.array(new sbyte[] { -128 }))));
            Assert.AreEqual(-128L, S(np.lcm(np.array(new sbyte[] { -128 }), np.array(new sbyte[] { 6 }))));
            Assert.AreEqual(2L, S(np.gcd(np.array(new[] { int.MinValue }), np.array(new[] { 6 }))));
            Assert.AreEqual((long)int.MinValue, S(np.gcd(np.array(new[] { int.MinValue }), np.array(new[] { int.MinValue }))));
            Assert.AreEqual(long.MinValue, S(np.lcm(np.array(new[] { long.MinValue }), np.array(new[] { 1L }))));
        }

        /// <summary>lcm's product |a|/gcd*|b| WRAPS the dtype on overflow (NumPy does not widen): the
        /// int16 and int32 cases here exceed their range and wrap to the exact NumPy value.</summary>
        [TestMethod]
        public void Lcm_Overflow_Wraps_LikeNumPy()
        {
            Assert.AreEqual(-23536L, S(np.lcm(np.array(new short[] { 21000 }), np.array(new short[] { 14000 }))));
            Assert.AreEqual(-728379968L, S(np.lcm(np.array(new[] { 1000000 }), np.array(new[] { 999999 }))));
            Assert.AreEqual(44L, S(np.lcm(np.array(new sbyte[] { 100 }), np.array(new sbyte[] { 3 }))));   // 300 mod 256
            // uint64 coprime pair: lcm wraps to 2 (max*(max-1) mod 2^64), gcd is 1.
            Assert.AreEqual(1UL, U(np.gcd(np.array(new[] { ulong.MaxValue }), np.array(new[] { ulong.MaxValue - 1 }))));
            Assert.AreEqual(2UL, U(np.lcm(np.array(new[] { ulong.MaxValue }), np.array(new[] { ulong.MaxValue - 1 }))));
        }

        // ---------------------------------------------------------------- promotion

        /// <summary>Uniform NEP50 promotion: the two operands and the output share one integer dtype
        /// (int8+int16→int16, uint8+int8→int16, uint32+int32→int64, uint64+uint64→uint64).</summary>
        [TestMethod]
        public void Promotion_SharesOnePromotedIntegerDtype()
        {
            np.gcd(np.array(new sbyte[] { 12 }), np.array(new short[] { 8 })).dtype.Should().Be(typeof(short));
            np.gcd(np.array(new byte[] { 12 }), np.array(new sbyte[] { 8 })).dtype.Should().Be(typeof(short));
            np.gcd(np.array(new uint[] { 12 }), np.array(new[] { 8 })).dtype.Should().Be(typeof(long));
            np.gcd(np.array(new[] { 12u }), np.array(new[] { 8u })).dtype.Should().Be(typeof(uint));
        }

        /// <summary>bool promotes to the integer it is paired with (there is no bool gcd loop, but
        /// bool→int is a valid promotion): gcd(bool, int8) runs the int8 loop, result int8.</summary>
        [TestMethod]
        public void BoolPairedWithInteger_PromotesToThatInteger()
        {
            var r = np.gcd(np.array(new[] { true, false }), np.array(new sbyte[] { 6, 9 }));
            r.dtype.Should().Be(typeof(sbyte));
            Assert.AreEqual(1L, Convert.ToInt64(r.GetAtIndex(0)));   // gcd(1,6)
            Assert.AreEqual(9L, Convert.ToInt64(r.GetAtIndex(1)));   // gcd(0,9)
        }

        /// <summary>A C# integer literal is a NEP50 weak scalar: gcd(int8[], 8) stays int8 (adopts the
        /// array's dtype) rather than promoting to int32.</summary>
        [TestMethod]
        public void WeakScalarLiteral_AdoptsArrayDtype()
        {
            var r = np.gcd(np.array(new sbyte[] { 12, 16 }), 8);
            r.dtype.Should().Be(typeof(sbyte));
            Assert.AreEqual(4L, Convert.ToInt64(r.GetAtIndex(0)));
            Assert.AreEqual(8L, Convert.ToInt64(r.GetAtIndex(1)));
        }

        /// <summary>Char (NumSharp's unsigned-16-bit extension, no NumPy dtype) is supported like uint16.</summary>
        [TestMethod]
        public void Char_IsSupported_LikeUInt16()
        {
            var g = np.gcd(np.array(new[] { (char)12 }), np.array(new[] { (char)8 }));
            g.dtype.Should().Be(typeof(char));
            Assert.AreEqual(4, (int)Convert.ToUInt16(g.GetAtIndex(0)));
            Assert.AreEqual(24, (int)Convert.ToUInt16(np.lcm(np.array(new[] { (char)12 }), np.array(new[] { (char)8 })).GetAtIndex(0)));
        }

        // ---------------------------------------------------------------- out / where / broadcast / dtype

        /// <summary>out= returns the SAME instance and stores the result (NumPy ufunc out= identity).</summary>
        [TestMethod]
        public void Out_ReturnsSameInstance_AndStores()
        {
            var outp = np.zeros(new Shape(3), np.int32);
            var r = np.gcd(np.array(new[] { 12, 15, 9 }), np.array(new[] { 8, 10, 6 }), @out: outp);
            Assert.IsTrue(ReferenceEquals(r, outp));
            Assert.AreEqual(4L, Convert.ToInt64(r.GetAtIndex(0)));
            Assert.AreEqual(5L, Convert.ToInt64(r.GetAtIndex(1)));
            Assert.AreEqual(3L, Convert.ToInt64(r.GetAtIndex(2)));
        }

        /// <summary>where= computes only mask-true elements; masked-off slots keep their prior contents.</summary>
        [TestMethod]
        public void Where_LeavesMaskedOffUntouched()
        {
            var outp = np.array(new[] { 99, 99, 99 });
            np.gcd(np.array(new[] { 12, 15, 9 }), np.array(new[] { 8, 10, 6 }),
                @out: outp, where: np.array(new[] { true, false, true }));
            Assert.AreEqual(4L, Convert.ToInt64(outp.GetAtIndex(0)));
            Assert.AreEqual(99L, Convert.ToInt64(outp.GetAtIndex(1)));   // masked off
            Assert.AreEqual(3L, Convert.ToInt64(outp.GetAtIndex(2)));
        }

        /// <summary>dtype= selects the loop: gcd(int16, int16, dtype=int64) computes in and returns int64.</summary>
        [TestMethod]
        public void Dtype_SelectsWiderLoop()
        {
            var r = np.gcd(np.array(new short[] { 12 }), np.array(new short[] { 8 }), dtype: np.int64);
            r.dtype.Should().Be(typeof(long));
            Assert.AreEqual(4L, S(r));
        }

        /// <summary>An explicit dtype= lets a uint64+signed pair (which would otherwise promote to
        /// float64 and raise) compute on the named integer loop — NumPy's loop-selection semantics.</summary>
        [TestMethod]
        public void Dtype_UnblocksUInt64SignedPair()
        {
            var r = np.gcd(np.array(new ulong[] { 12 }), np.array(new[] { 8L }), dtype: np.int64);
            r.dtype.Should().Be(typeof(long));
            Assert.AreEqual(4L, S(r));
        }

        // ---------------------------------------------------------------- error taxonomy (verbatim NumPy)

        /// <summary>Float/complex/bool-pair inputs name no integer loop → NumPy's verbatim
        /// "did not contain a loop with signature matching types (…, …) -> None" (as <see cref="TypeError"/>).</summary>
        [TestMethod]
        public void NonIntegerInputs_RaiseNoLoop_Verbatim()
        {
            var e1 = Assert.ThrowsException<TypeError>(() => np.gcd(np.array(new[] { 1.0 }), np.array(new[] { 2.0 })));
            StringAssert.Contains(e1.Message,
                "ufunc 'gcd' did not contain a loop with signature matching types " +
                "(<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.Float64DType'>) -> None");

            var e2 = Assert.ThrowsException<TypeError>(() => np.gcd(np.array(new[] { true }), np.array(new[] { false })));
            StringAssert.Contains(e2.Message, "(<class 'numpy.dtypes.BoolDType'>, <class 'numpy.dtypes.BoolDType'>) -> None");

            var e3 = Assert.ThrowsException<TypeError>(() => np.lcm(
                np.array(new Complex[] { 1 }), np.array(new Complex[] { 2 })));
            StringAssert.Contains(e3.Message, "ufunc 'lcm' did not contain a loop");
        }

        /// <summary>uint64 paired with a signed integer promotes to float64 under NEP50 → no integer
        /// loop → the no-loop error naming BOTH input dtype classes in order.</summary>
        [TestMethod]
        public void UInt64PlusSigned_PromotesToFloat64_NoLoop()
        {
            var e = Assert.ThrowsException<TypeError>(() => np.gcd(np.array(new[] { 1L }), np.array(new ulong[] { 2 })));
            StringAssert.Contains(e.Message,
                "(<class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.UInt64DType'>) -> None");
        }

        /// <summary>A non-integer dtype= names the missing OUTPUT loop: the tail becomes
        /// "-> {dtype}DType" (not "-> None"), matching NumPy's dtype-path no-loop text.</summary>
        [TestMethod]
        public void NonIntegerDtype_RaisesNoLoop_WithOutputTail()
        {
            var e1 = Assert.ThrowsException<TypeError>(() =>
                np.gcd(np.array(new[] { 1 }), np.array(new[] { 2 }), dtype: np.float64));
            StringAssert.Contains(e1.Message,
                "(<class 'numpy.dtypes.Int32DType'>, <class 'numpy.dtypes.Int32DType'>) -> <class 'numpy.dtypes.Float64DType'>");

            var e2 = Assert.ThrowsException<TypeError>(() =>
                np.gcd(np.array(new[] { 1 }), np.array(new[] { 2 }), dtype: np.@bool));
            StringAssert.Contains(e2.Message, "-> <class 'numpy.dtypes.BoolDType'>");
        }

        /// <summary>Float INPUTS with an integer dtype= fail on the input→loop cast (not the output loop):
        /// NumPy's verbatim "Cannot cast ufunc 'gcd' input 0 … with casting rule 'same_kind'".</summary>
        [TestMethod]
        public void FloatInputs_IntegerDtype_RaiseInputCast()
        {
            var e = Assert.ThrowsException<ArgumentException>(() =>
                np.gcd(np.array(new[] { 1.0 }), np.array(new[] { 2.0 }), dtype: np.int64));
            StringAssert.Contains(e.Message,
                "Cannot cast ufunc 'gcd' input 0 from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
        }
    }
}
