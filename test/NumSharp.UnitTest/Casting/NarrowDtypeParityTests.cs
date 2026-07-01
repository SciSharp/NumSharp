using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    /// Regression coverage for the "narrow dtype avoidance" family — code that historically
    /// ignored int8 (SByte), float16 (Half), or the no-bool-loop int8 promotion, diverging from
    /// NumPy 2.4.2. Each assertion below was verified against actual NumPy output.
    ///
    /// Bugs pinned here:
    ///  - NPTypeHierarchy._concreteParent was MISSING SByte and Half, so issubdtype / can_cast /
    ///    maximum_sctype treated int8 and float16 as uncategorized (Generic).
    ///  - can_cast(..., "same_kind") used a symmetric "same category" model instead of NumPy's
    ///    DIRECTIONAL kind ordering (bool &lt; unsigned &lt; signed &lt; float &lt; complex).
    ///  - power / floor_divide / mod / square on bool returned bool instead of int8 (these ufuncs
    ///    have no bool loop in NumPy).
    /// </summary>
    [TestClass]
    public class NarrowDtypeParityTests
    {
        // ---- issubdtype: int8 (SByte) and float16 (Half) must be categorized ----------------

        [TestMethod]
        public void IsSubDtype_Int8_IsSignedInteger()
        {
            np.issubdtype(NPTypeCode.SByte, "signedinteger").Should().BeTrue();
            np.issubdtype(NPTypeCode.SByte, "integer").Should().BeTrue();
            np.issubdtype(NPTypeCode.SByte, "number").Should().BeTrue();
            np.issubdtype(NPTypeCode.SByte, "unsignedinteger").Should().BeFalse();
            np.issubdtype(NPTypeCode.SByte, "floating").Should().BeFalse();
        }

        [TestMethod]
        public void IsSubDtype_Float16_IsFloating()
        {
            np.issubdtype(NPTypeCode.Half, "floating").Should().BeTrue();
            np.issubdtype(NPTypeCode.Half, "inexact").Should().BeTrue();
            np.issubdtype(NPTypeCode.Half, "number").Should().BeTrue();
            np.issubdtype(NPTypeCode.Half, "integer").Should().BeFalse();
        }

        // ---- maximum_sctype: int8 -> int64, float16 -> float64 -------------------------------

        [TestMethod]
        public void MaximumSctype_Int8_And_Float16()
        {
            np.maximum_sctype(NPTypeCode.SByte).Should().Be(NPTypeCode.Int64);
            np.maximum_sctype(NPTypeCode.Half).Should().Be(NPTypeCode.Double);
        }

        // ---- can_cast same_kind: DIRECTIONAL kind ordering, incl. int8 / float16 -------------

        [TestMethod]
        public void CanCast_SameKind_KindOrdering_Directional()
        {
            // unsigned -> signed allowed; signed -> unsigned NOT (asymmetric).
            np.can_cast(NPTypeCode.Byte, NPTypeCode.SByte, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.SByte, NPTypeCode.Byte, "same_kind").Should().BeFalse();
            // int -> float allowed; float -> int NOT.
            np.can_cast(NPTypeCode.Int64, NPTypeCode.Half, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Half, NPTypeCode.Int64, "same_kind").Should().BeFalse();
            // signed downcast to int8 allowed (same kind); float downcast to float16 allowed.
            np.can_cast(NPTypeCode.Int64, NPTypeCode.SByte, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Double, NPTypeCode.Half, "same_kind").Should().BeTrue();
            // complex -> float moves down a kind: rejected.
            np.can_cast(NPTypeCode.Complex, NPTypeCode.Half, "same_kind").Should().BeFalse();
        }

        // ---- no-bool-loop ufuncs promote bool -> int8 (dtype AND values) ---------------------

        [TestMethod]
        public void Power_BoolBool_IsInt8()
        {
            var b = np.array(new bool[] { true, false, true, true });
            var r = np.power(b, b);
            r.typecode.Should().Be(NPTypeCode.SByte, "power has no bool loop; bool**bool -> int8");
            r.GetSByte(0).Should().Be(1); // 1**1
            r.GetSByte(1).Should().Be(1); // 0**0 == 1
        }

        [TestMethod]
        public void FloorDivide_BoolBool_IsInt8()
        {
            var b = np.array(new bool[] { true, false, true, true });
            var r = np.floor_divide(b, b);
            r.typecode.Should().Be(NPTypeCode.SByte, "floor_divide has no bool loop; bool//bool -> int8");
            r.GetSByte(0).Should().Be(1); // 1//1
            r.GetSByte(1).Should().Be(0); // 0//0 -> 0
        }

        [TestMethod]
        public void Mod_BoolBool_IsInt8()
        {
            var b = np.array(new bool[] { true, true, true, true });
            var r = np.mod(b, b);
            r.typecode.Should().Be(NPTypeCode.SByte, "remainder has no bool loop; bool%bool -> int8");
            r.GetSByte(0).Should().Be(0); // 1%1
        }

        [TestMethod]
        public void Square_Bool_IsInt8()
        {
            var b = np.array(new bool[] { true, false, true });
            var r = np.square(b);
            r.typecode.Should().Be(NPTypeCode.SByte, "square has no bool loop; square(bool) -> int8");
            r.GetSByte(0).Should().Be(1);
            r.GetSByte(1).Should().Be(0);
            r.GetSByte(2).Should().Be(1);
        }

        [TestMethod]
        public void NonBool_Ops_PreserveDtype_Unaffected()
        {
            // Sanity: the bool->int8 remap is scoped to bool×bool only.
            np.power(np.array(new int[] { 2, 3 }), np.array(new int[] { 2, 3 }))
                .typecode.Should().Be(NPTypeCode.Int32);
            np.square(np.array(new short[] { 2, 3 }))
                .typecode.Should().Be(NPTypeCode.Int16);
            np.mod(np.array(new byte[] { 3, 4 }), np.array(new byte[] { 2, 2 }))
                .typecode.Should().Be(NPTypeCode.Byte);
        }
    }
}
