using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// Parity coverage for NumPy's <c>ndarray.astype(dtype, casting=...)</c> gate. NumSharp's astype
    /// historically had no casting parameter (always cast unsafely). The parameter now mirrors NumPy
    /// 2.4.2: default <c>'unsafe'</c> permits any conversion; a stricter rule raises
    /// (InvalidCastException — NumSharp's TypeError analogue, same message shape as np.copyto) when the
    /// source dtype cannot reach the target under that rule. Rule outcomes verified against NumPy.
    /// </summary>
    [TestClass]
    public class AstypeCastingParamTests
    {
        [TestMethod]
        public void Astype_DefaultCasting_IsUnsafe_NeverThrows()
        {
            // No casting arg == NumPy default 'unsafe': narrowing/lossy casts must still work.
            var i32 = np.array(new int[] { 1, 2, 300 });
            i32.astype(NPTypeCode.Int16).typecode.Should().Be(NPTypeCode.Int16);
            i32.astype(typeof(byte)).typecode.Should().Be(NPTypeCode.Byte);
            np.array(new double[] { 1.9 }).astype(NPTypeCode.Int32).GetInt32(0).Should().Be(1);
        }

        [TestMethod]
        public void Astype_Safe_AllowsWidening_RejectsNarrowingAndCrossKind()
        {
            var i32 = np.array(new int[] { 1, 2, 3 });
            // widening within signed ints is safe
            i32.astype(NPTypeCode.Int64, true, 'K', "safe").typecode.Should().Be(NPTypeCode.Int64);
            // narrowing is NOT safe
            new Action(() => i32.astype(NPTypeCode.Int16, true, 'K', "safe")).Should().Throw<InvalidCastException>();
            // int32 -> float32 is NOT safe (precision), int32 -> float64 IS safe
            new Action(() => i32.astype(NPTypeCode.Single, true, 'K', "safe")).Should().Throw<InvalidCastException>();
            i32.astype(NPTypeCode.Double, true, 'K', "safe").typecode.Should().Be(NPTypeCode.Double);
            // float -> int is never safe
            new Action(() => np.array(new double[] { 1.5 }).astype(NPTypeCode.Int32, true, 'K', "safe"))
                .Should().Throw<InvalidCastException>();
        }

        [TestMethod]
        public void Astype_SameKind_MatchesNumPyDirectionalRule()
        {
            var i32 = np.array(new int[] { 1, 2, 3 });
            // int -> narrower int and int -> float are same_kind
            i32.astype(NPTypeCode.Int16, true, 'K', "same_kind").typecode.Should().Be(NPTypeCode.Int16);
            i32.astype(NPTypeCode.Single, true, 'K', "same_kind").typecode.Should().Be(NPTypeCode.Single);
            // signed -> unsigned is NOT same_kind
            new Action(() => i32.astype(NPTypeCode.UInt32, true, 'K', "same_kind")).Should().Throw<InvalidCastException>();
            // float -> int is NOT same_kind
            new Action(() => np.array(new double[] { 1.5 }).astype(NPTypeCode.Int32, true, 'K', "same_kind"))
                .Should().Throw<InvalidCastException>();
        }

        [TestMethod]
        public void Astype_Unsafe_AllowsEverything()
        {
            var f64 = np.array(new double[] { 1.9, -2.9, 300.5 });
            f64.astype(NPTypeCode.Int32, true, 'K', "unsafe").typecode.Should().Be(NPTypeCode.Int32);
            f64.astype(NPTypeCode.Byte, true, 'K', "unsafe").typecode.Should().Be(NPTypeCode.Byte);
        }

        [TestMethod]
        public void Astype_RejectedCast_MessageMatchesNumPyShape()
        {
            var i32 = np.array(new int[] { 1, 2, 3 });
            new Action(() => i32.astype(NPTypeCode.Int16, true, 'K', "safe"))
                .Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast array data from dtype('int32') to dtype('int16') according to the rule 'safe'");
        }

        [TestMethod]
        public void Astype_TypeOverload_AlsoValidatesCasting()
        {
            var i32 = np.array(new int[] { 1, 2, 3 });
            new Action(() => i32.astype(typeof(short), true, 'K', "safe")).Should().Throw<InvalidCastException>();
            i32.astype(typeof(long), true, 'K', "safe").typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void Astype_SameDtype_AnyCasting_IsNoOpAllowed()
        {
            // from == to is allowed under every rule (including 'no').
            var i32 = np.array(new int[] { 1, 2, 3 });
            i32.astype(NPTypeCode.Int32, true, 'K', "no").typecode.Should().Be(NPTypeCode.Int32);
            i32.astype(NPTypeCode.Int32, true, 'K', "safe").typecode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Astype_AllowedCast_StillProducesCorrectValues()
        {
            // The gate must not change the produced bytes for an allowed cast.
            var i64 = np.array(new long[] { 1, 5_000_000_000L, -1 });
            var strict = i64.astype(NPTypeCode.Int32, true, 'K', "same_kind");
            var unsafe_ = i64.astype(NPTypeCode.Int32);
            Convert.ToHexString(strict.tobytes()).Should().Be(Convert.ToHexString(unsafe_.tobytes()));
        }
    }
}
