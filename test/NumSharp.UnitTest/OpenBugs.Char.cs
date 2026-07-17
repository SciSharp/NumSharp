using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Open bugs in NumSharp's <b>Char</b> dtype, surfaced by weaving Char into the
    ///     differential-fuzz grids (gen_oracle.char_tier). NumSharp documents Char as a 2-byte
    ///     UNSIGNED value, bit-identical to uint16 (see CLAUDE.md "Supported Types"), so the
    ///     CORRECT behavior of every Char op is exactly NumPy's uint16 behavior — verified against
    ///     NumPy 2.4.2 on 2026-06-30 and quoted per test.
    ///
    ///     These combinations are CARVED OUT of the green corpus (they would otherwise red the
    ///     woven tiers) and reproduced here under [OpenBugs]. Each test asserts the CORRECT
    ///     (uint16-equivalent) result, so it FAILS while the bug exists and PASSES when fixed —
    ///     at which point re-add the carved pair/op in gen_oracle.char_tier and delete the test.
    ///
    ///     Root causes (six distinct bugs):
    ///       1. promote(Char, Byte) -> Byte  — Char ranks BELOW uint8 in the promotion table, so
    ///          Char×uint8 truncates the Char's high byte. Corrupts arithmetic AND comparisons.
    ///       2. reciprocal(Char) -> Double   — Char not recognised as an integer dtype.
    ///       3. power(Char, Single) -> Double — Char mis-ranked ABOVE float32 (inconsistent with #1).
    ///       4. power(Char, …) scalar path crashes — kernel routes through Convert.To*(char), which
    ///          throws InvalidCastException for System.Char.
    ///       5. bitwise_*(bool, Char) -> KeyNotFoundException — (Boolean, Char) mixed-type kernel
    ///          pair is unregistered.
    ///       6. invert(Char) with N>=16 -> NotSupportedException — the SIMD Vector256&lt;ushort&gt;
    ///          bitwise-not path omits Char (N&lt;=15 scalar path works; size-dependent).
    /// </summary>
    [TestClass]
    public class OpenBugsCharTests : TestClass
    {
        // -- helpers: build Char / Byte NDArrays from int codes, read a scalar code back ----------
        private static NDArray Chars(params int[] v)
        {
            var a = new NDArray(NPTypeCode.Char, new Shape(v.Length), false);
            for (int i = 0; i < v.Length; i++) a.SetValue((char)v[i], i);
            return a;
        }

        private static NDArray CharScalar(int v)
        {
            var a = new NDArray(NPTypeCode.Char, new Shape(), false);
            a.SetValue((char)v, new int[0]);
            return a;
        }

        private static NDArray Bytes(params int[] v)
        {
            var a = new NDArray(NPTypeCode.Byte, new Shape(v.Length), false);
            for (int i = 0; i < v.Length; i++) a.SetValue((byte)v[i], i);
            return a;
        }

        private static int Code(NDArray a, int i) => (char)a.GetValue(i);

        // ============================================================================
        //  BUG 1: promote(Char, Byte) -> Byte (arithmetic) — silent data corruption.
        //
        //  np.array([321],u16) + np.array([65],u8) == [386], dtype uint16  (NumPy 2.4.2)
        //  NumSharp: char[321] + uint8[65] -> dtype Byte, value 130 (321 truncated to 0x41=65,
        //  then 65+65). Char ranks below uint8 so the wider operand "wins" downward.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Char_Add_Byte_TruncatesToByte()
        {
            var r = Chars(321) + Bytes(65);
            r.typecode.Should().Be(NPTypeCode.Char,
                "NumPy uint16(321)+uint8(65) -> uint16; Char is bit-identical to uint16 so the " +
                "result dtype must be Char, not Byte (promote(Char,Byte) wrongly returns Byte).");
            Code(r, 0).Should().Be(386,
                "NumPy: 321+65 == 386. NumSharp truncates the Char to its low byte (65) before adding.");
        }

        // ============================================================================
        //  BUG 1b: the SAME promotion bug corrupts COMPARISONS (wrong booleans).
        //
        //  np.greater(u16 321, u8 65) == True ; np.equal(u16 321, u8 65) == False  (NumPy 2.4.2)
        //  NumSharp compares at Byte precision after truncating 321 -> 65, so it reports
        //  greater == False and equal == True. This is a wrong ANSWER, not just a wrong dtype.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Char_Compare_Byte_TruncatesHighByte()
        {
            ((bool)np.greater(Chars(321), Bytes(65)).GetValue(0)).Should().BeTrue(
                "NumPy: 321 > 65 is True. NumSharp truncates Char 321 -> 65 and compares 65>65 -> False.");
            ((bool)np.equal(Chars(321), Bytes(65)).GetValue(0)).Should().BeFalse(
                "NumPy: 321 == 65 is False. NumSharp truncates Char 321 -> 65 and compares 65==65 -> True.");
        }

        // ============================================================================
        //  BUG 2: reciprocal(Char) -> Double instead of the integer reciprocal.
        //
        //  np.reciprocal(np.array([2,3,4],u16)) == [0,0,0], dtype uint16  (integer 1//x).
        //  NumSharp returns dtype Double — Char is not treated as an integer dtype here.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Char_Reciprocal_ReturnsDouble()
        {
            var r = np.reciprocal(Chars(2, 3, 4));
            r.typecode.Should().Be(NPTypeCode.Char,
                "NumPy reciprocal(uint16) stays uint16 (integer 1//x); Char must preserve its dtype, " +
                "but NumSharp promotes Char -> Double.");
        }

        // ============================================================================
        //  BUG 3: power(Char, Single) -> Double instead of Single.
        //
        //  np.power(u16[2,3], f32[2,2]) == [4.,9.], dtype float32.
        //  NumSharp returns dtype Double — Char is mis-ranked ABOVE float32 (the opposite
        //  direction from BUG 1, where it sits below Byte: the promotion table is simply wrong).
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Char_Power_Single_ReturnsDouble()
        {
            var r = np.power(Chars(2, 3), new NDArray(new float[] { 2f, 2f }));
            r.typecode.Should().Be(NPTypeCode.Single,
                "NumPy result_type(uint16, float32) == float32; NumSharp returns Double for Char^float32.");
        }

        // ============================================================================
        //  BUG 4: power crashes on a Char operand in the scalar/broadcast path.
        //
        //  np.power(u16[2,3,4], u16(2)) == [4,9,16], dtype uint16.
        //  NumSharp throws InvalidCastException: the scalar power kernel routes through
        //  Convert.ToDouble(char)/Convert.ToInt32(char), neither of which supports System.Char.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Char_Power_ScalarChar_Crashes()
        {
            NDArray r = null;
            Action act = () => r = np.power(Chars(2, 3, 4), CharScalar(2));
            act.Should().NotThrow("NumPy power(uint16, uint16) computes [4,9,16]; NumSharp throws " +
                                  "InvalidCastException because the kernel calls Convert.To*(char).");
            r.typecode.Should().Be(NPTypeCode.Char);
            Code(r, 0).Should().Be(4);
            Code(r, 1).Should().Be(9);
            Code(r, 2).Should().Be(16);
        }

        // ============================================================================
        //  BUG 5: bitwise_and/or/xor(bool, Char) -> KeyNotFoundException '(Boolean, Char)'.
        //
        //  np.bitwise_and([T,F,T], u16(5)) == [1,0,1], dtype uint16.
        //  NumSharp's mixed-type kernel cache has no (Boolean, Char) entry and throws.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Char_BitwiseAnd_Bool_KeyNotFound()
        {
            var b = new NDArray(new bool[] { true, false, true });
            NDArray r = null;
            Action act = () => r = np.bitwise_and(b, CharScalar(5));
            act.Should().NotThrow("NumPy bitwise_and(bool, uint16) -> uint16 [1,0,1]; NumSharp throws " +
                                  "KeyNotFoundException because the (Boolean, Char) kernel pair is unregistered.");
            r.typecode.Should().Be(NPTypeCode.Char);
            Code(r, 0).Should().Be(1);
            Code(r, 1).Should().Be(0);
            Code(r, 2).Should().Be(1);
        }

        // ============================================================================
        //  BUG 6: invert(Char) throws on the N>=16 SIMD path (size-dependent).
        //
        //  np.invert(u16[1..20]) == [65534,65533,…,65515], dtype uint16.
        //  NumSharp's Vector256<ushort> bitwise-not path omits Char and throws
        //  NotSupportedException once N reaches the 16-lane SIMD width (N<=15 scalar path works).
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Char_Invert_LargeN_NotSupported()
        {
            var c = Chars(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20); // N=20 -> SIMD
            NDArray r = null;
            Action act = () => r = np.invert(c);
            act.Should().NotThrow("NumPy invert(uint16) computes the bitwise complement; NumSharp throws " +
                                  "NotSupportedException on the Vector256<ushort> path for Char at N>=16.");
            r.typecode.Should().Be(NPTypeCode.Char);
            Code(r, 0).Should().Be(65534); // ~1
            Code(r, 19).Should().Be(65515); // ~20
        }
    }
}
