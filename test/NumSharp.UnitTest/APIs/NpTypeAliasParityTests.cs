using System;
using System.Numerics;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.APIs
{
    /// <summary>
    ///     NumPy 2.4.2 parity for class-level type aliases on <c>np</c>.
    ///     Every assertion was cross-checked against
    ///     <c>python -c "import numpy as np; print(np.dtype(np.&lt;name&gt;))"</c> on Windows 64-bit
    ///     and matches the LLP64/LP64 C-data-model convention for platform-dependent types.
    /// </summary>
    [TestClass]
    public class NpTypeAliasParityTests
    {
        private static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static bool Is64Bit  => IntPtr.Size == 8;

        // ---------------------------------------------------------------------
        // Fixed-size NumPy aliases — same on every platform
        // ---------------------------------------------------------------------

        [TestMethod] public void NpBool_Is_Bool()       => np.@bool.Should().Be(typeof(bool));
        [TestMethod] public void NpBoolUnderscore_Is_Bool() => np.bool_.Should().Be(typeof(bool));
        [TestMethod] public void NpBool8_Is_Bool()      => np.bool8.Should().Be(typeof(bool));

        [TestMethod] public void NpInt8_Is_SByte()      => np.int8.Should().Be(typeof(sbyte));
        [TestMethod] public void NpUInt8_Is_Byte()      => np.uint8.Should().Be(typeof(byte));
        [TestMethod] public void NpInt16_Is_Int16()     => np.int16.Should().Be(typeof(short));
        [TestMethod] public void NpUInt16_Is_UInt16()   => np.uint16.Should().Be(typeof(ushort));
        [TestMethod] public void NpInt32_Is_Int32()     => np.int32.Should().Be(typeof(int));
        [TestMethod] public void NpUInt32_Is_UInt32()   => np.uint32.Should().Be(typeof(uint));
        [TestMethod] public void NpInt64_Is_Int64()     => np.int64.Should().Be(typeof(long));
        [TestMethod] public void NpUInt64_Is_UInt64()   => np.uint64.Should().Be(typeof(ulong));
        [TestMethod] public void NpFloat16_Is_Half()    => np.float16.Should().Be(typeof(Half));
        [TestMethod] public void NpFloat32_Is_Single()  => np.float32.Should().Be(typeof(float));
        [TestMethod] public void NpFloat64_Is_Double()  => np.float64.Should().Be(typeof(double));
        [TestMethod] public void NpComplex128_Is_Complex() => np.complex128.Should().Be(typeof(Complex));
        [TestMethod] public void NpComplex_Is_Complex() => np.complex_.Should().Be(typeof(Complex));

        // ---------------------------------------------------------------------
        // NumPy C-type aliases — fixed
        // ---------------------------------------------------------------------

        [TestMethod] public void NpByte_Is_Int8_NumPyConvention()
        {
            // NumPy: np.byte = int8 (signed, C char convention). NumSharp follows NumPy.
            np.@byte.Should().Be(typeof(sbyte));
        }

        [TestMethod] public void NpUByte_Is_UInt8() => np.ubyte.Should().Be(typeof(byte));

        [TestMethod] public void NpShort_Is_Int16() => np.@short.Should().Be(typeof(short));
        [TestMethod] public void NpUShort_Is_UInt16() => np.@ushort.Should().Be(typeof(ushort));

        [TestMethod] public void NpIntc_Is_Int32() => np.intc.Should().Be(typeof(int));
        [TestMethod] public void NpUIntc_Is_UInt32() => np.uintc.Should().Be(typeof(uint));

        [TestMethod] public void NpLongLong_Is_Int64() => np.longlong.Should().Be(typeof(long));
        [TestMethod] public void NpULongLong_Is_UInt64() => np.ulonglong.Should().Be(typeof(ulong));

        [TestMethod] public void NpHalf_Is_Half() => np.half.Should().Be(typeof(Half));
        [TestMethod] public void NpSingle_Is_Single() => np.single.Should().Be(typeof(float));
        [TestMethod] public void NpDouble_Is_Double() => np.@double.Should().Be(typeof(double));
        [TestMethod] public void NpFloat_Is_Double()  => np.float_.Should().Be(typeof(double));

        // ---------------------------------------------------------------------
        // NumPy pointer-sized aliases (intp) — int64 on 64-bit, int32 on 32-bit
        // ---------------------------------------------------------------------

        [TestMethod]
        public void NpIntp_Is_PointerSizedSigned()
        {
            var expected = Is64Bit ? typeof(long) : typeof(int);
            np.intp.Should().Be(expected);
            // Critical: must NOT be typeof(nint) — that Type has NPTypeCode.Empty
            // and breaks np.zeros/np.empty dispatch.
            np.intp.Should().NotBe(typeof(nint));
        }

        [TestMethod]
        public void NpUIntp_Is_PointerSizedUnsigned()
        {
            var expected = Is64Bit ? typeof(ulong) : typeof(uint);
            np.uintp.Should().Be(expected);
            np.uintp.Should().NotBe(typeof(nuint));
        }

        [TestMethod]
        public void NpIntUnderscore_Is_Intp_NumPy2x()
        {
            // NumPy 2.x: np.int_ ≡ np.intp.
            np.int_.Should().Be(np.intp);
        }

        [TestMethod]
        public void NpUInt_Is_UIntp_NumPy2x()
        {
            // NumPy 2.x: np.uint ≡ np.uintp.
            np.@uint.Should().Be(np.uintp);
        }

        // ---------------------------------------------------------------------
        // NumPy C-long aliases — platform-dependent (LLP64 vs LP64)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void NpLong_MatchesPlatformCLong()
        {
            // Windows (MSVC LLP64): C long = 32 bits → typeof(int)
            // Linux/Mac 64-bit (gcc LP64): C long = 64 bits → typeof(long)
            var expected = (IsWindows || !Is64Bit) ? typeof(int) : typeof(long);
            np.@long.Should().Be(expected);
        }

        [TestMethod]
        public void NpULong_MatchesPlatformCULong()
        {
            var expected = (IsWindows || !Is64Bit) ? typeof(uint) : typeof(ulong);
            np.@ulong.Should().Be(expected);
        }

        // ---------------------------------------------------------------------
        // Consistency: np.X (class) matches np.dtype("X") — NumPy 2.x guarantees this
        // ---------------------------------------------------------------------

        [TestMethod] public void Consistent_int_() => np.int_.Should().Be(np.dtype("int_").type);
        [TestMethod] public void Consistent_intp() => np.intp.Should().Be(np.dtype("intp").type);
        [TestMethod] public void Consistent_uint() => np.@uint.Should().Be(np.dtype("uint").type);
        [TestMethod] public void Consistent_uintp() => np.uintp.Should().Be(np.dtype("uintp").type);
        [TestMethod] public void Consistent_long() => np.@long.Should().Be(np.dtype("long").type);
        [TestMethod] public void Consistent_ulong() => np.@ulong.Should().Be(np.dtype("ulong").type);
        [TestMethod] public void Consistent_longlong() => np.longlong.Should().Be(np.dtype("longlong").type);
        [TestMethod] public void Consistent_ulonglong() => np.ulonglong.Should().Be(np.dtype("ulonglong").type);
        [TestMethod] public void Consistent_short() => np.@short.Should().Be(np.dtype("short").type);
        [TestMethod] public void Consistent_ushort() => np.@ushort.Should().Be(np.dtype("ushort").type);
        [TestMethod] public void Consistent_byte() => np.@byte.Should().Be(np.dtype("byte").type);
        [TestMethod] public void Consistent_ubyte() => np.ubyte.Should().Be(np.dtype("ubyte").type);
        [TestMethod] public void Consistent_single() => np.single.Should().Be(np.dtype("single").type);
        [TestMethod] public void Consistent_double() => np.@double.Should().Be(np.dtype("double").type);
        [TestMethod] public void Consistent_float_() => np.float_.Should().Be(np.dtype("float").type);
        [TestMethod] public void Consistent_half() => np.half.Should().Be(np.dtype("half").type);
        [TestMethod] public void Consistent_int8() => np.int8.Should().Be(np.dtype("int8").type);
        [TestMethod] public void Consistent_uint8() => np.uint8.Should().Be(np.dtype("uint8").type);
        [TestMethod] public void Consistent_intc() => np.intc.Should().Be(np.dtype("intc").type);
        [TestMethod] public void Consistent_uintc() => np.uintc.Should().Be(np.dtype("uintc").type);
        [TestMethod] public void Consistent_complex128() => np.complex128.Should().Be(np.dtype("complex128").type);

        // ---------------------------------------------------------------------
        // Regression: np.intp must be usable to create arrays (was broken when np.intp = typeof(nint))
        // ---------------------------------------------------------------------

        [TestMethod]
        public void NpIntp_Works_For_ArrayCreation()
        {
            // Prior bug: typeof(nint).GetTypeCode() returned NPTypeCode.Empty, causing
            // np.zeros/np.empty to throw on dispatch. Now np.intp = typeof(long) on 64-bit.
            var shape = new Shape(3);
            var arr = np.zeros(shape, np.intp.GetTypeCode());
            arr.typecode.Should().Be(Is64Bit ? NPTypeCode.Int64 : NPTypeCode.Int32);
        }

        [TestMethod]
        public void NpUIntp_Works_For_ArrayCreation()
        {
            var shape = new Shape(3);
            var arr = np.zeros(shape, np.uintp.GetTypeCode());
            arr.typecode.Should().Be(Is64Bit ? NPTypeCode.UInt64 : NPTypeCode.UInt32);
        }
    }
}
