using System;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #529, "[NEP50 Core] Type promotion diverges from NumPy 2.x NEP 50 for unsigned
    ///     int array + signed int scalar" (https://github.com/SciSharp/NumSharp/issues/529): since 0.80.0 a C# number literal
    ///     is NumPy's weak Python scalar on every route, so an unsigned array keeps its dtype against any integer literal and
    ///     an out-of-range literal raises instead of wrapping. Expected values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue529
    {
        /// <summary>
        ///     The issue's 12 table entries, written as C# literals: uint8 / uint16 / uint32 / uint64 arrays with an int16
        ///     ((short)1), int32 (1) or int64 (1L) literal keep the array's dtype, for <c>+</c> and <c>*</c> alike. A C# literal of
        ///     any integer type is a Python int, which is why the int16 and int64 rows hold too.
        /// </summary>
        [TestMethod]
        public void UnsignedArray_WithIntegerLiteral_KeepsTheArrayDtype()
        {
            var arrays = new (NPTypeCode code, NDArray array)[]
            {
                (NPTypeCode.Byte, np.array(new byte[] { 1 })), (NPTypeCode.UInt16, np.array(new ushort[] { 1 })),
                (NPTypeCode.UInt32, np.array(new uint[] { 1 })), (NPTypeCode.UInt64, np.array(new ulong[] { 1 })),
            };
            foreach (var (code, a) in arrays)
            {
                (a + 1).typecode.Should().Be(code, "a + 1 keeps the unsigned dtype");
                (a * 3).typecode.Should().Be(code, "a * 3 keeps the unsigned dtype");
                (a + (short)1).typecode.Should().Be(code, "an int16 C# literal is a weak Python int too");
                (a + 1L).typecode.Should().Be(code, "an int64 C# literal is a weak Python int too");
            }

            (np.array(new byte[] { 1 }) + 1).GetByte(0).Should().Be(2);
        }

        /// <summary>
        ///     Only literals are weak: array + array still widens (uint8 + int32 array is int32), and an explicit 0-d int16 array
        ///     is a strong NumPy scalar (uint8 + int16 0-d is int16), both as in NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void ArrayOperands_StayStrong()
        {
            (np.array(new byte[] { 1 }) + np.array(new int[] { 1 })).typecode.Should().Be(NPTypeCode.Int32);
            (np.array(new byte[] { 1 }) + np.array((short)1)).typecode.Should().Be(NPTypeCode.Int16);
        }

        /// <summary>
        ///     An integer literal outside the array's dtype raises NumPy's text ("Python integer 300 out of bounds for uint8";
        ///     NumPy raises OverflowError, NumSharp OverflowException) instead of wrapping, for both ends of the range.
        /// </summary>
        [TestMethod]
        public void OutOfRangeLiteral_Raises_WithNumPysText()
        {
            var u8 = np.array(new byte[] { 1 });

            new Action(() => { var _ = u8 + 300; }).Should().Throw<OverflowException>()
                .WithMessage("Python integer 300 out of bounds for uint8");
            new Action(() => { var _ = u8 + (-1); }).Should().Throw<OverflowException>()
                .WithMessage("Python integer -1 out of bounds for uint8");
        }
    }
}
