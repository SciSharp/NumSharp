using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest
{
    // =====================================================================
    // OPEN BUG — NEP50 out-of-range python-int scalar is NOT range-checked
    // =====================================================================
    //
    // Collected while making Char promote as the uint16 masquerade (the Char≡uint16
    // differential surfaced it; it is NOT Char-specific — it affects every integer
    // dtype). NumPy 2.x (NEP50) treats a Python int operand as "weak" and folds it
    // into the array's dtype, but FIRST checks it is representable in that dtype:
    // an out-of-range value raises OverflowError BEFORE any element-wise work.
    //
    //   >>> np.array([1,2], np.uint16) + 70000   # 70000 > 65535
    //   OverflowError: Python integer 70000 out of bounds for uint16
    //   >>> np.array([1,2], np.uint16) * -1       # -1 < 0 (unsigned)
    //   OverflowError: Python integer -1 out of bounds for uint16
    //   >>> np.array([1],  np.int8)  + 200        # 200 > 127
    //   OverflowError: Python integer 200 out of bounds for int8
    //   >>> np.power(np.array([2,3], np.uint16), -1)
    //   OverflowError: Python integer -1 out of bounds for uint16
    //
    // In-range scalars whose RESULT overflows are fine and wrap (probed 2.4.2):
    //   >>> np.array([1,2], np.uint16) - 5        # 5 fits; result wraps
    //   array([65532, 65533], dtype=uint16)
    //   >>> np.array([1], np.int8) + 100          # 100 fits
    //   array([101], dtype=int8)
    //
    // NumSharp's promotion of a C# scalar is purely TYPE-based (the arr_scalar
    // table), so it silently coerces/wraps the value instead of range-checking it:
    //   uint16[1,2] + 70000 -> [4465,4466]   (70000 & 0xFFFF == 4464, +1/+2)
    //   uint16[1,2] * -1     -> [65535,65534]
    //   int8[1]    + 200     -> wraps, no throw
    //   power(uint16[2,3],-1)-> [0, 43691]   (43691 is the modular inverse of 3!)
    //
    // Inconsistency proving this is a gap, not a deliberate design choice: NumSharp's
    // OWN fused path ALREADY enforces the rule — np.evaluate throws the exact
    // OverflowException (NDExpr.Typing.cs: "Python integer {value} out of bounds for
    // {dtype}"). Only the operator / ufunc path skips the check.
    //
    // EXPECTED (when fixed): each op below throws OverflowException (NumPy's
    // OverflowError), matching np.evaluate. Remove [OpenBugs] when fixed.
    // =====================================================================
    public partial class OpenBugs
    {
        private static NDArray CharArr(params int[] xs) => np.array(xs.Select(v => (char)v).ToArray());

        [TestMethod, OpenBugs(IssueUrl = "nep50-scalar-out-of-range")]
        public void ScalarOverflow_UInt16_PlusTooLargeInt_ShouldRaise()
        {
            // NumPy: uint16 array + 70000 -> OverflowError (70000 > 65535).
            Action act = () => { var _ = np.array(new ushort[] { 1, 2 }) + 70000; };
            act.Should().Throw<OverflowException>(
                "a Python int outside the uint16 range must raise, not wrap to [4465,4466]");
        }

        [TestMethod, OpenBugs(IssueUrl = "nep50-scalar-out-of-range")]
        public void ScalarOverflow_Char_PlusTooLargeInt_ShouldRaise()
        {
            // Char is the uint16 masquerade -> same rule as uint16.
            Action act = () => { var _ = CharArr(1, 2) + 70000; };
            act.Should().Throw<OverflowException>();
        }

        [TestMethod, OpenBugs(IssueUrl = "nep50-scalar-out-of-range")]
        public void ScalarOverflow_Unsigned_TimesNegativeOne_ShouldRaise()
        {
            // NumPy: uint16 array * -1 -> OverflowError (-1 < 0 for unsigned).
            Action actU16 = () => { var _ = np.array(new ushort[] { 1, 2 }) * -1; };
            actU16.Should().Throw<OverflowException>("-1 is out of range for an unsigned dtype");

            Action actByte = () => { var _ = np.array(new byte[] { 1, 2 }) + (-1); };
            actByte.Should().Throw<OverflowException>("-1 is out of range for uint8");
        }

        [TestMethod, OpenBugs(IssueUrl = "nep50-scalar-out-of-range")]
        public void ScalarOverflow_Signed_OutOfRangeInt_ShouldRaise()
        {
            // NumPy: int8 array + 200 -> OverflowError (200 > 127); int16 + 40000 -> OverflowError.
            Action actI8 = () => { var _ = np.array(new sbyte[] { 1 }) + 200; };
            actI8.Should().Throw<OverflowException>("200 is out of range for int8");

            Action actI16 = () => { var _ = np.array(new short[] { 1 }) + 40000; };
            actI16.Should().Throw<OverflowException>("40000 is out of range for int16");
        }

        [TestMethod, OpenBugs(IssueUrl = "nep50-scalar-out-of-range")]
        public void ScalarOverflow_Power_UnsignedBase_NegativeIntExponent_ShouldRaise()
        {
            // NumPy: power(uint16, -1) -> OverflowError (-1 out of uint16 range), checked
            // BEFORE the negative-exponent rule. NumSharp instead computes a modular
            // inverse ([0, 43691]) — doubly wrong (no throw AND a nonsense value).
            Action act = () => { var _ = np.power(CharArr(2, 3), -1); };
            act.Should().Throw<OverflowException>();
        }
    }
}
