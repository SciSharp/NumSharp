using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
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
        // ============================================================================
        //  BUG: np.round_ on a Boolean array resolves the wrong result dtype.
        //
        //  NumPy 2.4.2: np.round(np.array([False, True]), 0) -> float16 [0., 1.]
        //  (bool takes the rint float-tier: bool/i8/u8 -> float16). NumSharp's round_
        //  resolves Boolean -> Double instead ([0., 1.] as float64). Values match,
        //  dtype does not. Carved from ROUND_DTYPES in gen_oracle.py.
        //  Root cause: DefaultEngine.Round's decimals==0 path — Boolean fails
        //  IsInteger() (so it skips the int identity-copy) and ResolveUnaryReturnType
        //  maps bool -> Double, while the rint tier table maps bool -> Half.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Round_Bool_Dtype_Diverges()
        {
            var r = np.round_(np.array(new bool[] { false, true }), 0);
            r.typecode.Should().Be(NPTypeCode.Half,
                "NumPy round(bool, 0) returns float16 (the rint float-tier); NumSharp resolves Double");
            ((double)(Half)r.GetValue(0)).Should().Be(0.0);
            ((double)(Half)r.GetValue(1)).Should().Be(1.0);
        }

        // ============================================================================
        //  BUG: np.round_ with decimals != 0 is a NO-OP for Complex input.
        //
        //  NumPy 2.4.2 rounds real and imaginary parts via the multiply -> rint ->
        //  divide composition: np.round([1.55+2.45j], 1) -> [1.6+2.4j].
        //  NumSharp returns the input unchanged ([1.55+2.45j]) — the decimals!=0
        //  path (RoundDecimalsCore) never touches Complex components.
        //  decimals==0 IS correct (probed: [1.5+2.5j] -> [2+2j], banker's).
        //  Carved from the rounding tier: complex128 kept at decimals=0 only.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Round_Complex_NonzeroDecimals_NoOp()
        {
            var a = np.array(new System.Numerics.Complex[] { new(1.55, 2.45) });
            var r = np.round_(a, 1);
            var v = (System.Numerics.Complex)r.GetValue(0);
            v.Real.Should().Be(1.6, "NumPy rounds the real part: 1.55*10=15.500000000000002 -> rint 16 -> 1.6");
            v.Imaginary.Should().Be(2.4, "NumPy rounds the imaginary part: 2.45*10=24.499999999999996 -> rint 24 -> 2.4");
        }
    }
}
