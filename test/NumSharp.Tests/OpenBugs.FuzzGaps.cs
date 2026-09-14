using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs surfaced by the WS-GAPS fuzz-completeness pass (closing false-premise
    ///     exclusions and dead wiring in the differential-fuzz corpus — see
    ///     Fuzz/COMPLETENESS_PLAN.md). Each is a NumPy-parity gap found while widening a
    ///     corpus tier; the offending (op, dtype/param) cell is CARVED OUT of the green corpus
    ///     (carve comment at the generator site points here) and reproduced under [OpenBugs].
    ///     NumPy 2.4.2 is the source of truth (probed 2026-07-07). Remove the generator carve
    ///     and the test together when fixed.
    /// </summary>
    [TestClass]
    public class OpenBugsFuzzGapsTests : TestClass
    {
        // NOTE: two former round_ bugs (bool -> float16 dtype at decimals==0, and the Complex
        // decimals!=0 no-op) were FIXED by the Default.Round PyArray_Round port; their positive
        // parity tests now live in Math/np.round.Test.cs, and gen_oracle's rounding tier gates them.

        // ============================================================================
        //  BUG: np.dot on two 1-D Char vectors throws instead of the uint16-modular
        //  inner product.
        //
        //  Char is bit-identical to uint16; the NumPy proxy says dot(uint16, uint16)
        //  -> 0-D uint16 (modular). NumSharp's vector-dot path reduces through
        //  DefaultEngine.sum_elementwise_il with an EXPLICIT Char result typecode,
        //  and that switch has no Char arm:
        //      NotSupportedException: Sum not supported for type Char
        //  np.matmul on the same 1-D Char vectors works (different path), as do all
        //  2-D+ Char matmul/dot/outer cases and flat np.sum(char) (which accumulates
        //  to UInt64 via GetAccumulatingType, hitting an existing arm).
        //  Carved from the char matmul weave (gen_oracle.char_tier "matmul").
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Dot_Char_1D_Throws()
        {
            var a = np.array(new char[] { (char)1, (char)2, (char)3, (char)4 });
            var b = np.array(new char[] { (char)5, (char)6, (char)7, (char)8 });
            NDArray r = null;
            Action act = () => r = np.dot(a, b);
            act.Should().NotThrow("NumPy dot(uint16, uint16) is the modular uint16 inner product; " +
                                  "Char is bit-identical to uint16");
            r.typecode.Should().Be(NPTypeCode.Char);
            ((int)(char)r.GetValue(0)).Should().Be(70, "1*5 + 2*6 + 3*7 + 4*8 = 70");
        }
    }
}
