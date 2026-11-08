using System;
using System.Numerics;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     Exhaustive NumPy 2.x parity tests for <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> string parsing.
    ///
    ///     Every expectation here is verified against NumPy's actual output via
    ///     <c>python -c "import numpy as np; np.dtype(...)"</c>. The text codes parse to NumPy's descriptors
    ///     (<c>S</c> / <c>S10</c> / <c>a</c> / <c>a5</c> to bytestrings, <c>U</c> / <c>U32</c> to unicode), and the
    ///     datetime / timedelta codes to descriptors that have no storage yet; the NumPy types NumSharp doesn't
    ///     implement (complex64, object, void) throw <see cref="NotSupportedException"/>, and a string NumPy rejects
    ///     raises NumPy's <see cref="TypeError"/> with NumPy's text.
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

        /// <summary>
        ///     A string NumPy rejects: NumPy's <c>TypeError: data type '…' not understood</c>. None of the strings here
        ///     needs Python's repr escaping, so the quoted text is the input between single quotes.
        /// </summary>
        /// <param name="input">The dtype string.</param>
        private static void ExpectNotUnderstood(string input)
        {
            Action act = () => np.dtype(input);
            act.Should().Throw<TypeError>($"input='{input}'").Which.Message.Should().Be($"data type '{input}' not understood");
        }

        /// <summary>A valid NumPy dtype NumSharp has no class for (complex64, object, void): NumSharp's limit, not NumPy's error.</summary>
        /// <param name="input">The dtype string.</param>
        private static void ExpectUnsupported(string input)
        {
            Action act = () => np.dtype(input);
            act.Should().Throw<NotSupportedException>($"input='{input}'").Which.Message.Should().Be($"NumPy dtype '{input}' is not supported by NumSharp");
        }

        /// <summary>
        ///     <c>'f16'</c> / <c>'c32'</c> name the host's C long double in NumPy where it is 16 bytes (float128 /
        ///     complex256 on 64-bit Linux): unsupported there, NumPy's "not understood" where NumPy has no such width.
        /// </summary>
        /// <param name="input">The dtype string.</param>
        private static void ExpectLongDoubleWidth(string input)
        {
            if (DTypeRegistry.CLongDoubleSize == 16)
                ExpectUnsupported(input);
            else
                ExpectNotUnderstood(input);
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
        [TestMethod] public void SingleChar_F_Complex64_Throws()   => ExpectUnsupported("F");
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
        [TestMethod] public void Sized_c8_Complex64_Throws() => ExpectUnsupported("c8");
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
        [TestMethod] public void Named_complex64_Throws() => ExpectUnsupported("complex64");
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

        [TestMethod] public void Invalid_b4()  => ExpectNotUnderstood("b4");
        [TestMethod] public void Invalid_qm1() => ExpectNotUnderstood("?1");  // ? is not sized
        [TestMethod] public void Invalid_i3()  => ExpectNotUnderstood("i3");
        [TestMethod] public void Invalid_i5()  => ExpectNotUnderstood("i5");
        [TestMethod] public void Invalid_i16() => ExpectNotUnderstood("i16");
        [TestMethod] public void Invalid_i32() => ExpectNotUnderstood("i32");
        [TestMethod] public void Invalid_u3()  => ExpectNotUnderstood("u3");
        [TestMethod] public void Invalid_u16() => ExpectNotUnderstood("u16");
        [TestMethod] public void Invalid_f1()  => ExpectNotUnderstood("f1");
        [TestMethod] public void Invalid_f3()  => ExpectNotUnderstood("f3");
        [TestMethod] public void Invalid_f5()  => ExpectNotUnderstood("f5");
        [TestMethod] public void Invalid_f16() => ExpectLongDoubleWidth("f16");
        [TestMethod] public void Invalid_c1()  => ExpectNotUnderstood("c1");
        [TestMethod] public void Invalid_c2()  => ExpectNotUnderstood("c2");
        [TestMethod] public void Invalid_c4()  => ExpectNotUnderstood("c4");
        [TestMethod] public void Invalid_c32() => ExpectLongDoubleWidth("c32");

        // ---------------------------------------------------------------------
        // NumPy types NumSharp doesn't implement → NotSupportedException
        // ---------------------------------------------------------------------

        // The fixed-width string dtypes parse (probed 2.4.2: np.dtype(spelling).str); the unsized forms are the class
        // singletons S0 / U0, which an allocation turns into one character.
        [TestMethod] public void String_S()     => np.dtype("S").str.Should().Be("|S0");
        [TestMethod] public void String_S10()   => np.dtype("S10").str.Should().Be("|S10");
        [TestMethod] public void String_S1000() => np.dtype("S1000").str.Should().Be("|S1000");
        [TestMethod] public void String_U()     => np.dtype("U").str.Should().Be("<U0");
        [TestMethod] public void String_U32()   => np.dtype("U32").str.Should().Be("<U32");
        [TestMethod] public void Unsupported_V()    => ExpectUnsupported("V");
        [TestMethod] public void Unsupported_V16()  => ExpectUnsupported("V16");
        [TestMethod] public void Unsupported_O()    => ExpectUnsupported("O");
        [TestMethod] public void String_a_IsDeprecatedBytesAlias() => np.dtype("a").str.Should().Be("|S0");
        // NumPy's PyArray_DescrFromType maps the bare datetime LETTERS to the generic descriptors: np.dtype('M') is
        // dtype('<M8') and np.dtype('m') is dtype('<m8') (probed 2.4.2).
        [TestMethod] public void SingleChar_M_GenericDatetime()  => np.dtype("M").Should().Be(np.dtype("M8"));
        [TestMethod] public void SingleChar_m_GenericTimedelta() => np.dtype("m").Should().Be(np.dtype("m8"));

        // ---------------------------------------------------------------------
        // datetime64 / timedelta64: DESCRIPTOR-level support (dtype-system Stage A). The typestrs parse into
        // parametric descriptors — kind 'M'/'m', itemsize 8, generic unit — but carry no storage yet, so the
        // NPTypeCode / Type conversions raise NotSupportedException naming the missing piece.
        // ---------------------------------------------------------------------

        [TestMethod] public void Datetime_M8_IsGenericDatetimeDescriptor()
        {
            var d = np.dtype("M8");
            d.kind.Should().Be('M');
            d.@char.Should().Be('M');
            d.itemsize.Should().Be(8);
            d.num.Should().Be(21);
            d.name.Should().Be("datetime64");
            d.str.Should().Be("<M8");
            d.Meta.Should().BeSameAs(np.dtypes.DateTime64DType);
            np.datetime_data(d).Should().Be(("generic", 1));
            Action act = () => { NPTypeCode _ = d; };
            act.Should().Throw<NotSupportedException>();
        }

        [TestMethod] public void Datetime_m8_IsGenericTimedeltaDescriptor()
        {
            var d = np.dtype("m8");
            d.kind.Should().Be('m');
            d.@char.Should().Be('m');
            d.itemsize.Should().Be(8);
            d.num.Should().Be(22);
            d.name.Should().Be("timedelta64");
            d.str.Should().Be("<m8");
            d.Meta.Should().BeSameAs(np.dtypes.TimeDelta64DType);
        }

        [TestMethod] public void Datetime_Names_Parse()
        {
            np.dtype("datetime64").Should().Be(np.dtype("M8"));
            np.dtype("timedelta64").Should().Be(np.dtype("m8"));
            np.dtype("datetime64[ns]").Should().Be(np.dtype("M8[ns]"));
            np.dtype("timedelta64[ms]").str.Should().Be("<m8[ms]");
        }
        [TestMethod] public void String_a5()       => np.dtype("a5").str.Should().Be("|S5");
        [TestMethod] public void String_c_IsS1_NotComplex() => np.dtype("c").str.Should().Be("|S1");
        [TestMethod] public void String_str()      => np.dtype("str").str.Should().Be("<U0");
        [TestMethod] public void String_str_()     => np.dtype("str_").str.Should().Be("<U0");
        [TestMethod] public void String_bytes_()   => np.dtype("bytes_").str.Should().Be("|S0");
        [TestMethod] public void Unsupported_object() => ExpectUnsupported("object");
        [TestMethod] public void Unsupported_object_() => ExpectUnsupported("object_");
        // datetime64 / timedelta64 parse into generic-unit descriptors since dtype-system Stage A (see Datetime_* below).
        [TestMethod] public void Named_datetime64_IsGenericDatetime()  => np.dtype("datetime64").str.Should().Be("<M8");
        [TestMethod] public void Named_timedelta64_IsGenericTimedelta() => np.dtype("timedelta64").str.Should().Be("<m8");

        // ---------------------------------------------------------------------
        // Case-sensitive: NumPy is case-sensitive for single chars — 'I4' throws
        // ---------------------------------------------------------------------

        [TestMethod] public void CaseSensitive_I4_Throws() => ExpectNotUnderstood("I4");
        [TestMethod] public void CaseSensitive_F4_Throws() => ExpectNotUnderstood("F4");
        [TestMethod] public void CaseSensitive_D8_Throws() => ExpectNotUnderstood("D8");

        // ---------------------------------------------------------------------
        // Nonsense / whitespace
        // ---------------------------------------------------------------------

        [TestMethod] public void Whitespace_Leading() => ExpectNotUnderstood("   i4");
        [TestMethod] public void Whitespace_Trailing() => ExpectNotUnderstood("i4   ");
        [TestMethod] public void Whitespace_Empty()   => ExpectNotUnderstood("");
        [TestMethod] public void Whitespace_SpaceOnly() => ExpectNotUnderstood(" ");
        [TestMethod] public void Invalid_xyz()        => ExpectNotUnderstood("xyz");
        [TestMethod] public void Invalid_True()       => ExpectNotUnderstood("True");
        [TestMethod] public void Invalid_None()       => ExpectNotUnderstood("None");
        [TestMethod] public void Invalid_Random()     => ExpectNotUnderstood("not_a_dtype");

        /// <summary>
        ///     A null string is NumPy's <c>None</c>: <c>np.dtype(None) is np.dtype('f8')</c> (NumPy 2.4.2,
        ///     <c>_convert_from_any</c>'s first test). np.dtype has string / Type / NPTypeCode / DType / NDScalarType
        ///     overloads, so a bare <c>null</c> literal is ambiguous at compile time; the string spelling is the one this
        ///     suite exercises.
        /// </summary>
        [TestMethod]
        public void NullInput_IsNumPysNone_Float64()
        {
            np.dtype((string)null).Should().BeSameAs(DType.Double);
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
