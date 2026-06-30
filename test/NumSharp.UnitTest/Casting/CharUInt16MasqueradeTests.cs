using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// Char is NumSharp's representation of a 16-bit unsigned integer (System.Char /
    /// UTF-16 code unit). It has NO direct NumPy dtype; its closest analogue is uint16,
    /// so Char MUST promote and compute bit-identically to uint16 everywhere.
    ///
    /// These tests pin the six Char≡uint16 bugs that the differential-fuzz Char gate
    /// surfaced (all expected values probed against NumPy 2.4.2 as uint16), plus the
    /// broader bugs found alongside them. The 14×op×order sweep at the bottom proves the
    /// masquerade holds against the already-NumPy-validated uint16 path.
    /// </summary>
    [TestClass]
    public class CharUInt16MasqueradeTests : TestClass
    {
        // ---- helpers ---------------------------------------------------------
        private static NDArray Chars(params int[] xs) => np.array(xs.Select(v => (char)v).ToArray());
        private static NDArray U16(params int[] xs) => np.array(xs.Select(v => (ushort)v).ToArray());

        /// <summary>Flatten + read any integer/char result as long[] (Convert.ToInt64(char) throws, so unbox char by hand).</summary>
        private static long[] Ints(NDArray r)
        {
            r = r.flatten();
            var o = new long[r.size];
            for (int i = 0; i < r.size; i++)
            {
                var v = r.GetValue(i);
                o[i] = v is char c ? c : Convert.ToInt64(v);
            }
            return o;
        }

        private static double[] Doubles(NDArray r)
        {
            r = r.flatten();
            var o = new double[r.size];
            for (int i = 0; i < r.size; i++)
            {
                var v = r.GetValue(i);
                o[i] = v is char c ? c : Convert.ToDouble(v);
            }
            return o;
        }

        /// <summary>Char and UInt16 are the same dtype for the masquerade.</summary>
        private static NPTypeCode Norm(NPTypeCode t) => t == NPTypeCode.Char ? NPTypeCode.UInt16 : t;

        // =====================================================================
        // Bug 1 — promote(Char, *) ranked Char below uint8 and truncated the high byte.
        //   char[321] + uint8[65] gave dtype Byte / value 130 (386 mod 256).
        //   NumPy: np.array([321],uint16) + np.array([65],uint8) -> uint16, 386.
        // =====================================================================
        [TestMethod]
        public void Bug1_CharPlusByte_PromotesToUInt16Width_NoTruncation()
        {
            var r = Chars(321) + np.array(new byte[] { 65 });
            Norm(r.typecode).Should().Be(NPTypeCode.UInt16, "char + uint8 promotes like uint16 + uint8 -> uint16");
            Ints(r).Should().BeEquivalentTo(new long[] { 386 }, "386 must NOT truncate to a byte (130)");
        }

        [TestMethod]
        public void Bug1_CharPlusByte_BitIdenticalToUInt16PlusByte()
        {
            var rc = Chars(321) + np.array(new byte[] { 65 });
            var ru = U16(321) + np.array(new byte[] { 65 });
            Norm(rc.typecode).Should().Be(Norm(ru.typecode));
            Ints(rc).Should().BeEquivalentTo(Ints(ru));
        }

        // =====================================================================
        // Bug 1b — the same mis-rank corrupted comparisons (compared at the truncated width).
        //   greater(char 321, uint8 65) gave False; equal gave True (321 & 65 both -> 65).
        //   NumPy: greater -> True, equal -> False, less -> False.
        // =====================================================================
        [TestMethod]
        public void Bug1b_Comparisons_UseFullUInt16Width()
        {
            var a = Chars(321);
            var b = np.array(new byte[] { 65 });
            np.greater(a, b).GetBoolean(0).Should().BeTrue("321 > 65");
            np.equal(a, b).GetBoolean(0).Should().BeFalse("321 != 65 (no high-byte truncation)");
            np.less(a, b).GetBoolean(0).Should().BeFalse("321 is not < 65");
        }

        // =====================================================================
        // Bug 2 — reciprocal(char) returned Double (Char not seen as integer).
        //   NumPy: reciprocal(uint16) -> uint16 with C-truncating 1//x (0 for |x|>=2, 1 for 1).
        // =====================================================================
        [TestMethod]
        public void Bug2_Reciprocal_Char_TakesIntegerLoop()
        {
            var r = np.reciprocal(Chars(2, 3, 5));
            Norm(r.typecode).Should().Be(NPTypeCode.UInt16, "reciprocal(uint16) stays integer, not float");
            Ints(r).Should().BeEquivalentTo(new long[] { 0, 0, 0 });

            var r1 = np.reciprocal(Chars(1));
            Norm(r1.typecode).Should().Be(NPTypeCode.UInt16);
            Ints(r1).Should().BeEquivalentTo(new long[] { 1 });
        }

        // =====================================================================
        // Bug 3 — power(char, float32) returned Double (Char mis-ranked above float32).
        //   NumPy: power(uint16, float32) -> float32. The override that caused this also
        //   wrongly upcast EVERY {bool,int8,int16,uint8,uint16} ** float32 to float64.
        // =====================================================================
        [TestMethod]
        public void Bug3_Power_CharBase_Float32Exp_IsFloat32()
        {
            var r = np.power(Chars(3), np.array(new float[] { 2f }));
            r.typecode.Should().Be(NPTypeCode.Single, "power(uint16, float32) -> float32, not float64");
            Doubles(r).Should().BeEquivalentTo(new double[] { 9.0 });
        }

        [DataTestMethod]
        // probed NumPy 2.4.2: small ints fit in float32, int32+ need float64
        [DataRow(NPTypeCode.SByte, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Int16, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Byte, NPTypeCode.Single)]
        [DataRow(NPTypeCode.UInt16, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Double)]
        [DataRow(NPTypeCode.UInt32, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Int64, NPTypeCode.Double)]
        public void Bug3b_Power_IntBase_Float32Exp_FollowsResultType(NPTypeCode baseType, NPTypeCode expected)
        {
            // Collected bug: the removed power override forced float64 for ALL int-base ** float-exp.
            var baseArr = np.array(new int[] { 2 }).astype(baseType);
            var r = np.power(baseArr, np.array(new float[] { 3f }));
            r.typecode.Should().Be(expected);
            Doubles(r).Should().BeEquivalentTo(new double[] { 8.0 });
        }

        // =====================================================================
        // Bug 4 — power crashed on a char scalar exponent (Convert.ToDouble(char) throws
        //   InvalidCastException in the scalar-exponent fast path).
        //   NumPy: power(uint16, 2) -> uint16 [9, 16].
        // =====================================================================
        [TestMethod]
        public void Bug4_Power_CharScalarExponent_DoesNotThrow()
        {
            var baseArr = Chars(3, 4);

            var rChar = np.power(baseArr, NDArray.Scalar((char)2));   // char scalar (the crash)
            var rInt = np.power(baseArr, 2);                          // int scalar
            var rArr = np.power(baseArr, Chars(2, 2));                // char array exponent

            foreach (var r in new[] { rChar, rInt, rArr })
            {
                Norm(r.typecode).Should().Be(NPTypeCode.UInt16);
                Ints(r).Should().BeEquivalentTo(new long[] { 9, 16 });
            }
        }

        // =====================================================================
        // Bug 5 — (Boolean, Char) was missing from the arr_scalar promotion table, so a
        //   bool array op a char SCALAR threw KeyNotFoundException '(Boolean, Char)'.
        //   Not bitwise-specific — any binary op hit it. NumPy(bool, uint16): and->[1,0],
        //   or->[5,7], xor->[4,7], add->[6,5].
        // =====================================================================
        [TestMethod]
        public void Bug5_BoolArray_CharScalar_DoesNotThrow()
        {
            var b = np.array(new bool[] { true, false });
            var s = NDArray.Scalar((char)5);

            Ints(np.bitwise_and(b, s)).Should().BeEquivalentTo(new long[] { 1, 0 });
            Ints(np.bitwise_or(b, s)).Should().BeEquivalentTo(new long[] { 5, 5 });
            Ints(np.bitwise_xor(b, s)).Should().BeEquivalentTo(new long[] { 4, 5 });
            Ints(np.add(b, s)).Should().BeEquivalentTo(new long[] { 6, 5 });

            Norm(np.bitwise_and(b, s).typecode).Should().Be(NPTypeCode.UInt16);
        }

        [TestMethod]
        public void Bug5_BoolArray_CharArray_BitwiseMatchesUInt16()
        {
            var b = np.array(new bool[] { true, false });
            Ints(np.bitwise_and(b, Chars(5, 7))).Should().BeEquivalentTo(new long[] { 1, 0 });
            Ints(np.bitwise_or(b, Chars(5, 7))).Should().BeEquivalentTo(new long[] { 5, 7 });
            Ints(np.bitwise_xor(b, Chars(5, 7))).Should().BeEquivalentTo(new long[] { 4, 7 });
        }

        // =====================================================================
        // Bug 6 — invert(char) with N >= SIMD width threw NotSupportedException: the
        //   BitwiseNot SIMD path emitted Vector<char>, which does not exist. N<16 (scalar
        //   path) worked. NumPy: invert(uint16) -> uint16 ones-complement at every size.
        // =====================================================================
        [DataTestMethod]
        [DataRow(1)]
        [DataRow(3)]
        [DataRow(15)]   // last scalar-path size on V256
        [DataRow(16)]   // first SIMD-width size (the crash)
        [DataRow(17)]
        [DataRow(64)]   // multiple vectors + tail
        public void Bug6_Invert_Char_AllSizes(int n)
        {
            var vals = Enumerable.Range(0, n).Select(i => i * 7 % 65536).ToArray();
            var r = np.invert(Chars(vals));
            Norm(r.typecode).Should().Be(NPTypeCode.UInt16);
            var expected = vals.Select(v => (long)(ushort)~(ushort)v).ToArray();
            Ints(r).Should().BeEquivalentTo(expected, "invert is ~x as 16-bit unsigned");
        }

        [TestMethod]
        public void Bug6_Invert_Char_MatchesUInt16_AcrossSimdBoundary()
        {
            var vals = Enumerable.Range(0, 40).Select(i => i * 1000 % 65536).ToArray();
            Ints(np.invert(Chars(vals))).Should().BeEquivalentTo(Ints(np.invert(U16(vals))));
        }

        // =====================================================================
        // Collected: AsNumpyDtypeName(Char) reported "uint8" (Char is 2 bytes -> uint16).
        // =====================================================================
        [TestMethod]
        public void Collected_AsNumpyDtypeName_Char_IsUInt16()
        {
            NPTypeCode.Char.AsNumpyDtypeName().Should().Be("uint16");
            NPTypeCode.Char.AsNumpyDtypeName().Should().NotBe(NPTypeCode.Byte.AsNumpyDtypeName());
        }

        // =====================================================================
        // Differential: Char promotes identically to uint16 in BOTH promotion tables.
        // uint16 is already validated against NumPy, so this proves the masquerade.
        // =====================================================================
        private static readonly NPTypeCode[] AllTypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
        };

        [TestMethod]
        public void Differential_ArrArrPromotion_CharEqualsUInt16()
        {
            foreach (var x in AllTypes)
            {
                Norm(np._FindCommonArrayType(NPTypeCode.Char, x))
                    .Should().Be(Norm(np._FindCommonArrayType(NPTypeCode.UInt16, x)), $"char⊗{x} == uint16⊗{x}");
                Norm(np._FindCommonArrayType(x, NPTypeCode.Char))
                    .Should().Be(Norm(np._FindCommonArrayType(x, NPTypeCode.UInt16)), $"{x}⊗char == {x}⊗uint16");
            }
        }

        [TestMethod]
        public void Differential_ArrScalarPromotion_CharEqualsUInt16()
        {
            foreach (var x in AllTypes)
            {
                // char ARRAY vs scalar X  ==  uint16 ARRAY vs scalar X
                Norm(np._FindCommonArrayScalarType(NPTypeCode.Char, x))
                    .Should().Be(Norm(np._FindCommonArrayScalarType(NPTypeCode.UInt16, x)), $"char-array ⊗ {x}-scalar");
                // X ARRAY vs char SCALAR  ==  X ARRAY vs uint16 SCALAR (this is where (bool,char) was missing)
                Norm(np._FindCommonArrayScalarType(x, NPTypeCode.Char))
                    .Should().Be(Norm(np._FindCommonArrayScalarType(x, NPTypeCode.UInt16)), $"{x}-array ⊗ char-scalar");
            }
        }

        // =====================================================================
        // Memory-layout (DOD): strided / reversed / broadcast / axis char ops match uint16.
        // =====================================================================
        [TestMethod]
        public void Layout_StridedReversedBroadcast_CharMatchesUInt16()
        {
            int[] d = { 10, 20, 30, 40, 50, 60 };

            Ints(np.invert(Chars(d)["::2"])).Should().BeEquivalentTo(Ints(np.invert(U16(d)["::2"])));
            Ints(np.reciprocal(Chars(d)["::-1"])).Should().BeEquivalentTo(Ints(np.reciprocal(U16(d)["::-1"])));
            Ints((Chars(d)["1::2"] * np.array(new byte[] { 2, 2, 2 })))
                .Should().BeEquivalentTo(Ints((U16(d)["1::2"] * np.array(new byte[] { 2, 2, 2 }))));

            // broadcast (2,1) + (1,3)
            var cb = Chars(1, 2).reshape(2, 1) + Chars(10, 20, 30).reshape(1, 3);
            var ub = U16(1, 2).reshape(2, 1) + U16(10, 20, 30).reshape(1, 3);
            Ints(cb).Should().BeEquivalentTo(Ints(ub));

            // axis reductions on a 2-D char view
            Ints(np.sum(Chars(d).reshape(2, 3), 0)).Should().BeEquivalentTo(Ints(np.sum(U16(d).reshape(2, 3), 0)));
            Ints(np.max(Chars(d).reshape(2, 3), 1)).Should().BeEquivalentTo(Ints(np.max(U16(d).reshape(2, 3), 1)));
        }
    }
}
