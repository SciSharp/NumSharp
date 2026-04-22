using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     Exhaustive NumPy 2.x parity tests for <see cref="np.dtype(string)"/> string parsing.
    ///
    ///     Every expectation here is verified against NumPy's actual output via
    ///     <c>python -c "import numpy as np; np.dtype(...)"</c>. NumPy types NumSharp doesn't
    ///     implement (bytestring, unicode, datetime, timedelta, object, void) throw
    ///     <see cref="NotSupportedException"/>; NumPy's complex64 widens to
    ///     NumSharp's complex128 (System.Numerics.Complex) since the 64-bit form isn't supported.
    ///
    ///     Platform note: 'l'/'L' and 'int'/'uint' follow the <b>Windows</b> NumPy convention
    ///     (C long = 32-bit). On 64-bit Linux NumPy these would be 64-bit; NumSharp is fixed
    ///     at the Windows convention.
    /// </summary>
    [TestClass]
    public class DTypeStringParityTests
    {
        private static void Expect(string input, NPTypeCode expected) =>
            np.dtype(input).typecode.Should().Be(expected, $"input='{input}'");

        private static void ExpectThrow(string input)
        {
            Action act = () => np.dtype(input);
            act.Should().Throw<Exception>($"input='{input}' should throw")
                .Which.Should().Match(ex => ex is NotSupportedException || ex is ArgumentNullException);
        }

        // ---------------------------------------------------------------------
        // Single-char NumPy type codes
        // ---------------------------------------------------------------------

        [TestMethod] public void SingleChar_QuestionMark_Bool()    => Expect("?", NPTypeCode.Boolean);
        [TestMethod] public void SingleChar_b_Int8()               => Expect("b", NPTypeCode.SByte);
        [TestMethod] public void SingleChar_B_UInt8()              => Expect("B", NPTypeCode.Byte);
        [TestMethod] public void SingleChar_h_Int16()              => Expect("h", NPTypeCode.Int16);
        [TestMethod] public void SingleChar_H_UInt16()             => Expect("H", NPTypeCode.UInt16);
        [TestMethod] public void SingleChar_i_Int32()              => Expect("i", NPTypeCode.Int32);
        [TestMethod] public void SingleChar_I_UInt32()             => Expect("I", NPTypeCode.UInt32);
        [TestMethod] public void SingleChar_l_PlatformDependent()
        {
            // 'l' = C long: 32-bit on Windows (MSVC), 64-bit on Linux/Mac LP64 (gcc).
            var expected = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) || IntPtr.Size != 8
                ? NPTypeCode.Int32 : NPTypeCode.Int64;
            Expect("l", expected);
        }
        [TestMethod] public void SingleChar_L_PlatformDependent()
        {
            var expected = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) || IntPtr.Size != 8
                ? NPTypeCode.UInt32 : NPTypeCode.UInt64;
            Expect("L", expected);
        }
        [TestMethod] public void SingleChar_q_Int64()              => Expect("q", NPTypeCode.Int64);
        [TestMethod] public void SingleChar_Q_UInt64()             => Expect("Q", NPTypeCode.UInt64);
        [TestMethod] public void SingleChar_e_Float16()            => Expect("e", NPTypeCode.Half);
        [TestMethod] public void SingleChar_f_Float32()            => Expect("f", NPTypeCode.Single);
        [TestMethod] public void SingleChar_d_Float64()            => Expect("d", NPTypeCode.Double);
        [TestMethod] public void SingleChar_g_LongDouble_AsFloat64() => Expect("g", NPTypeCode.Double);
        [TestMethod] public void SingleChar_F_Complex64_Throws()   => ExpectThrow("F");
        [TestMethod] public void SingleChar_D_Complex128()         => Expect("D", NPTypeCode.Complex);
        [TestMethod] public void SingleChar_G_LongDoubleComplex()  => Expect("G", NPTypeCode.Complex);
        [TestMethod] public void SingleChar_p_IntPtr_Int64()       => Expect("p", NPTypeCode.Int64);
        [TestMethod] public void SingleChar_P_UIntPtr_UInt64()     => Expect("P", NPTypeCode.UInt64);

        // ---------------------------------------------------------------------
        // Sized variants (letter + size digits)
        // ---------------------------------------------------------------------

        [TestMethod] public void Sized_b1_Bool()        => Expect("b1", NPTypeCode.Boolean);
        [TestMethod] public void Sized_i1_Int8()        => Expect("i1", NPTypeCode.SByte);
        [TestMethod] public void Sized_u1_UInt8()       => Expect("u1", NPTypeCode.Byte);
        [TestMethod] public void Sized_i2_Int16()       => Expect("i2", NPTypeCode.Int16);
        [TestMethod] public void Sized_u2_UInt16()      => Expect("u2", NPTypeCode.UInt16);
        [TestMethod] public void Sized_f2_Half()        => Expect("f2", NPTypeCode.Half);
        [TestMethod] public void Sized_i4_Int32()       => Expect("i4", NPTypeCode.Int32);
        [TestMethod] public void Sized_u4_UInt32()      => Expect("u4", NPTypeCode.UInt32);
        [TestMethod] public void Sized_f4_Single()      => Expect("f4", NPTypeCode.Single);
        [TestMethod] public void Sized_c8_Complex64_Throws() => ExpectThrow("c8");
        [TestMethod] public void Sized_i8_Int64()       => Expect("i8", NPTypeCode.Int64);
        [TestMethod] public void Sized_u8_UInt64()      => Expect("u8", NPTypeCode.UInt64);
        [TestMethod] public void Sized_f8_Double()      => Expect("f8", NPTypeCode.Double);
        [TestMethod] public void Sized_c16_Complex128() => Expect("c16", NPTypeCode.Complex);

        // ---------------------------------------------------------------------
        // Named forms — NumPy lowercase (everything `np.dtype('<name>')` returns)
        // ---------------------------------------------------------------------

        [TestMethod] public void Named_int8_SByte()      => Expect("int8", NPTypeCode.SByte);
        [TestMethod] public void Named_uint8_Byte()      => Expect("uint8", NPTypeCode.Byte);
        [TestMethod] public void Named_int16()           => Expect("int16", NPTypeCode.Int16);
        [TestMethod] public void Named_uint16()          => Expect("uint16", NPTypeCode.UInt16);
        [TestMethod] public void Named_int32()           => Expect("int32", NPTypeCode.Int32);
        [TestMethod] public void Named_uint32()          => Expect("uint32", NPTypeCode.UInt32);
        [TestMethod] public void Named_int64()           => Expect("int64", NPTypeCode.Int64);
        [TestMethod] public void Named_uint64()          => Expect("uint64", NPTypeCode.UInt64);
        [TestMethod] public void Named_float16()         => Expect("float16", NPTypeCode.Half);
        [TestMethod] public void Named_half()            => Expect("half", NPTypeCode.Half);
        [TestMethod] public void Named_float32()         => Expect("float32", NPTypeCode.Single);
        [TestMethod] public void Named_float64()         => Expect("float64", NPTypeCode.Double);
        [TestMethod] public void Named_float_AsDouble()  => Expect("float", NPTypeCode.Double);
        [TestMethod] public void Named_double()          => Expect("double", NPTypeCode.Double);
        [TestMethod] public void Named_single()          => Expect("single", NPTypeCode.Single);
        [TestMethod] public void Named_complex64_Throws() => ExpectThrow("complex64");
        [TestMethod] public void Named_complex128()      => Expect("complex128", NPTypeCode.Complex);
        [TestMethod] public void Named_complex()         => Expect("complex", NPTypeCode.Complex);
        [TestMethod] public void Named_bool()            => Expect("bool", NPTypeCode.Boolean);
        [TestMethod] public void Named_byte_IsSigned()   => Expect("byte", NPTypeCode.SByte); // NumPy quirk
        [TestMethod] public void Named_ubyte()           => Expect("ubyte", NPTypeCode.Byte);
        [TestMethod] public void Named_short()           => Expect("short", NPTypeCode.Int16);
        [TestMethod] public void Named_ushort()          => Expect("ushort", NPTypeCode.UInt16);
        [TestMethod] public void Named_intc()            => Expect("intc", NPTypeCode.Int32);
        [TestMethod] public void Named_uintc()           => Expect("uintc", NPTypeCode.UInt32);
        [TestMethod] public void Named_intp()            => Expect("intp", NPTypeCode.Int64);
        [TestMethod] public void Named_uintp()           => Expect("uintp", NPTypeCode.UInt64);
        [TestMethod] public void Named_longlong()        => Expect("longlong", NPTypeCode.Int64);
        [TestMethod] public void Named_ulonglong()       => Expect("ulonglong", NPTypeCode.UInt64);
        [TestMethod] public void Named_int_IsIntp()
        {
            // NumPy 2.x: 'int' is an alias for 'intp' (pointer-sized).
            var expected = IntPtr.Size == 8 ? NPTypeCode.Int64 : NPTypeCode.Int32;
            Expect("int", expected);
        }
        [TestMethod] public void Named_intUnderscore_Intp()
        {
            var expected = IntPtr.Size == 8 ? NPTypeCode.Int64 : NPTypeCode.Int32;
            Expect("int_", expected);
        }
        [TestMethod] public void Named_long_PlatformDependent()
        {
            var expected = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) || IntPtr.Size != 8
                ? NPTypeCode.Int32 : NPTypeCode.Int64;
            Expect("long", expected);
        }
        [TestMethod] public void Named_ulong_PlatformDependent()
        {
            var expected = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) || IntPtr.Size != 8
                ? NPTypeCode.UInt32 : NPTypeCode.UInt64;
            Expect("ulong", expected);
        }
        [TestMethod] public void Named_boolUnderscore()  => Expect("bool_", NPTypeCode.Boolean);
        [TestMethod] public void Named_longdouble_AsDouble() => Expect("longdouble", NPTypeCode.Double);
        [TestMethod] public void Named_clongdouble()     => Expect("clongdouble", NPTypeCode.Complex);

        // ---------------------------------------------------------------------
        // NumSharp-specific friendly C# aliases (PascalCase / .NET names)
        // ---------------------------------------------------------------------

        [TestMethod] public void Alias_SByte()    => Expect("SByte", NPTypeCode.SByte);
        [TestMethod] public void Alias_sbyte()    => Expect("sbyte", NPTypeCode.SByte);
        [TestMethod] public void Alias_Byte()     => Expect("Byte", NPTypeCode.Byte);
        [TestMethod] public void Alias_Int16()    => Expect("Int16", NPTypeCode.Int16);
        [TestMethod] public void Alias_UInt16()   => Expect("UInt16", NPTypeCode.UInt16);
        [TestMethod] public void Alias_Int32()    => Expect("Int32", NPTypeCode.Int32);
        [TestMethod] public void Alias_UInt32()   => Expect("UInt32", NPTypeCode.UInt32);
        [TestMethod] public void Alias_Int64()    => Expect("Int64", NPTypeCode.Int64);
        [TestMethod] public void Alias_UInt64()   => Expect("UInt64", NPTypeCode.UInt64);
        [TestMethod] public void Alias_Half()     => Expect("Half", NPTypeCode.Half);
        [TestMethod] public void Alias_Single()   => Expect("Single", NPTypeCode.Single);
        [TestMethod] public void Alias_Float()    => Expect("Float", NPTypeCode.Single);
        [TestMethod] public void Alias_Double()   => Expect("Double", NPTypeCode.Double);
        [TestMethod] public void Alias_Complex()  => Expect("Complex", NPTypeCode.Complex);
        [TestMethod] public void Alias_Boolean()  => Expect("Boolean", NPTypeCode.Boolean);
        [TestMethod] public void Alias_Bool()     => Expect("Bool", NPTypeCode.Boolean);
        [TestMethod] public void Alias_boolean()  => Expect("boolean", NPTypeCode.Boolean);
        [TestMethod] public void Alias_Char()     => Expect("Char", NPTypeCode.Char);
        [TestMethod] public void Alias_char()     => Expect("char", NPTypeCode.Char);
        [TestMethod] public void Alias_Decimal()  => Expect("Decimal", NPTypeCode.Decimal);
        [TestMethod] public void Alias_decimal()  => Expect("decimal", NPTypeCode.Decimal);
        [TestMethod] public void Alias_String()   => Expect("String", NPTypeCode.String);

        // ---------------------------------------------------------------------
        // Byte-order prefix
        // ---------------------------------------------------------------------

        [TestMethod] public void ByteOrder_LittleEndian() => Expect("<i4", NPTypeCode.Int32);
        [TestMethod] public void ByteOrder_BigEndian()    => Expect(">i4", NPTypeCode.Int32);
        [TestMethod] public void ByteOrder_Native()       => Expect("=i4", NPTypeCode.Int32);
        [TestMethod] public void ByteOrder_NotApplicable() => Expect("|i4", NPTypeCode.Int32);
        [TestMethod] public void ByteOrder_Little_f8()    => Expect("<f8", NPTypeCode.Double);
        [TestMethod] public void ByteOrder_Big_c16()      => Expect(">c16", NPTypeCode.Complex);
        [TestMethod] public void ByteOrder_Little_questionmark() => Expect("<?", NPTypeCode.Boolean);

        // ---------------------------------------------------------------------
        // Invalid size suffixes (NumPy TypeError)
        // ---------------------------------------------------------------------

        [TestMethod] public void Invalid_b4()  => ExpectThrow("b4");
        [TestMethod] public void Invalid_qm1() => ExpectThrow("?1");  // ? is not sized
        [TestMethod] public void Invalid_i3()  => ExpectThrow("i3");
        [TestMethod] public void Invalid_i5()  => ExpectThrow("i5");
        [TestMethod] public void Invalid_i16() => ExpectThrow("i16");
        [TestMethod] public void Invalid_i32() => ExpectThrow("i32");
        [TestMethod] public void Invalid_u3()  => ExpectThrow("u3");
        [TestMethod] public void Invalid_u16() => ExpectThrow("u16");
        [TestMethod] public void Invalid_f1()  => ExpectThrow("f1");
        [TestMethod] public void Invalid_f3()  => ExpectThrow("f3");
        [TestMethod] public void Invalid_f5()  => ExpectThrow("f5");
        [TestMethod] public void Invalid_f16() => ExpectThrow("f16");
        [TestMethod] public void Invalid_c1()  => ExpectThrow("c1");
        [TestMethod] public void Invalid_c2()  => ExpectThrow("c2");
        [TestMethod] public void Invalid_c4()  => ExpectThrow("c4");
        [TestMethod] public void Invalid_c32() => ExpectThrow("c32");

        // ---------------------------------------------------------------------
        // NumPy types NumSharp doesn't implement → NotSupportedException
        // ---------------------------------------------------------------------

        [TestMethod] public void Unsupported_S()    => ExpectThrow("S");
        [TestMethod] public void Unsupported_S10()  => ExpectThrow("S10");
        [TestMethod] public void Unsupported_S1000() => ExpectThrow("S1000");
        [TestMethod] public void Unsupported_U()    => ExpectThrow("U");
        [TestMethod] public void Unsupported_U32()  => ExpectThrow("U32");
        [TestMethod] public void Unsupported_V()    => ExpectThrow("V");
        [TestMethod] public void Unsupported_V16()  => ExpectThrow("V16");
        [TestMethod] public void Unsupported_O()    => ExpectThrow("O");
        [TestMethod] public void Unsupported_M()    => ExpectThrow("M");
        [TestMethod] public void Unsupported_M8()   => ExpectThrow("M8");
        [TestMethod] public void Unsupported_m()    => ExpectThrow("m");
        [TestMethod] public void Unsupported_m8()   => ExpectThrow("m8");
        [TestMethod] public void Unsupported_a()    => ExpectThrow("a");
        [TestMethod] public void Unsupported_a5()   => ExpectThrow("a5");
        [TestMethod] public void Unsupported_c_IsS1_NotComplex() => ExpectThrow("c");
        [TestMethod] public void Unsupported_str()  => ExpectThrow("str");
        [TestMethod] public void Unsupported_str_() => ExpectThrow("str_");
        [TestMethod] public void Unsupported_bytes_() => ExpectThrow("bytes_");
        [TestMethod] public void Unsupported_object() => ExpectThrow("object");
        [TestMethod] public void Unsupported_object_() => ExpectThrow("object_");
        [TestMethod] public void Unsupported_datetime64() => ExpectThrow("datetime64");
        [TestMethod] public void Unsupported_timedelta64() => ExpectThrow("timedelta64");

        // ---------------------------------------------------------------------
        // Case-sensitive: NumPy is case-sensitive for single chars — 'I4' throws
        // ---------------------------------------------------------------------

        [TestMethod] public void CaseSensitive_I4_Throws() => ExpectThrow("I4");
        [TestMethod] public void CaseSensitive_F4_Throws() => ExpectThrow("F4");
        [TestMethod] public void CaseSensitive_D8_Throws() => ExpectThrow("D8");

        // ---------------------------------------------------------------------
        // Nonsense / whitespace
        // ---------------------------------------------------------------------

        [TestMethod] public void Whitespace_Leading() => ExpectThrow("   i4");
        [TestMethod] public void Whitespace_Trailing() => ExpectThrow("i4   ");
        [TestMethod] public void Whitespace_Empty()   => ExpectThrow("");
        [TestMethod] public void Whitespace_SpaceOnly() => ExpectThrow(" ");
        [TestMethod] public void Invalid_xyz()        => ExpectThrow("xyz");
        [TestMethod] public void Invalid_True()       => ExpectThrow("True");
        [TestMethod] public void Invalid_None()       => ExpectThrow("None");
        [TestMethod] public void Invalid_Random()     => ExpectThrow("not_a_dtype");

        [TestMethod]
        public void NullInput_Throws()
        {
            Action act = () => np.dtype(null);
            act.Should().Throw<ArgumentNullException>();
        }

        // ---------------------------------------------------------------------
        // Resolved DType round-trip sanity: ensure type and itemsize match
        // ---------------------------------------------------------------------

        [TestMethod] public void Round_Int8_TypeAndSize()
        {
            var d = np.dtype("int8");
            d.type.Should().Be(typeof(sbyte));
            d.itemsize.Should().Be(1);
            d.kind.Should().Be('i');
        }

        [TestMethod] public void Round_UInt8_TypeAndSize()
        {
            var d = np.dtype("uint8");
            d.type.Should().Be(typeof(byte));
            d.itemsize.Should().Be(1);
            d.kind.Should().Be('u');
        }

        [TestMethod] public void Round_Float16_TypeAndSize()
        {
            var d = np.dtype("float16");
            d.type.Should().Be(typeof(Half));
            d.itemsize.Should().Be(2);
            d.kind.Should().Be('f');
        }

        [TestMethod] public void Round_Complex128_TypeAndSize()
        {
            var d = np.dtype("complex128");
            d.type.Should().Be(typeof(Complex));
            d.itemsize.Should().Be(16);
            d.kind.Should().Be('c');
        }

        [TestMethod] public void Round_SingleChar_b_Is_Int8()
        {
            var d = np.dtype("b");
            d.type.Should().Be(typeof(sbyte));
            d.itemsize.Should().Be(1);
        }

        [TestMethod] public void Round_SingleChar_B_Is_UInt8()
        {
            var d = np.dtype("B");
            d.type.Should().Be(typeof(byte));
            d.itemsize.Should().Be(1);
        }
    }
}
