using System;
using System.Linq;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The dtype-string grammar of <see cref="np.dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> — the port of <c>descriptor.c::_convert_from_str</c>
    ///     plus the datetime metadata grammar of <c>datetime.c</c>. Every expectation (values AND error texts) was probed
    ///     against <c>numpy==2.4.2</c>; <c>DTypeStringParityTests</c> keeps the pre-existing name/code table, this file
    ///     covers what the new grammar adds: datetime typestrs, divisors, byte-order retention, NumPy's error wording,
    ///     and the <c>strtol</c> quirks of the sized codes.
    /// </summary>
    [TestClass]
    public class DTypeStringGrammarTests
    {
        private static (string unit, int count) Data(string spelling) => np.datetime_data(np.dtype(spelling));

        [DataTestMethod]
        [DataRow("M8[ns]", "<M8[ns]", "ns", 1)]
        [DataRow("M8[10ns]", "<M8[10ns]", "ns", 10)]
        [DataRow("m8[2W]", "<m8[2W]", "W", 2)]
        [DataRow("M8[Y]", "<M8[Y]", "Y", 1)]
        [DataRow("m8[Y]", "<m8[Y]", "Y", 1)]
        [DataRow("datetime64[D]", "<M8[D]", "D", 1)]
        [DataRow("timedelta64[us]", "<m8[us]", "us", 1)]
        [DataRow("datetime64[ns]", "<M8[ns]", "ns", 1)]
        [DataRow("timedelta64[ms]", "<m8[ms]", "ms", 1)]
        [DataRow("datetime64[2D]", "<M8[2D]", "D", 2)]
        [DataRow("M8[1s]", "<M8[s]", "s", 1)]
        [DataRow("M8[01s]", "<M8[s]", "s", 1)]
        [DataRow("M8[0s]", "<M8[0s]", "s", 0)]
        [DataRow("M8[00s]", "<M8[0s]", "s", 0)]
        [DataRow("M8[μs]", "<M8[us]", "us", 1)]
        [DataRow("M8[1generic]", "<M8", "generic", 1)]
        [DataRow("M8[2generic]", "<M8", "generic", 2)]
        [DataRow("m8[generic]", "<m8", "generic", 1)]
        [DataRow("M8", "<M8", "generic", 1)]
        [DataRow("m8", "<m8", "generic", 1)]
        [DataRow("datetime64", "<M8", "generic", 1)]
        [DataRow("timedelta64", "<m8", "generic", 1)]
        [DataRow("M", "<M8", "generic", 1)]
        [DataRow("m", "<m8", "generic", 1)]
        public void Datetime_Typestr_Parses(string spelling, string str, string unit, int count)
        {
            np.dtype(spelling).str.Should().Be(str);
            Data(spelling).Should().Be((unit, count));
        }

        [DataTestMethod]
        [DataRow("m8[s/2]", "<m8[500ms]", "ms", 500)]
        [DataRow("M8[Y/2]", "<M8[6M]", "M", 6)]
        [DataRow("m8[D/3]", "<m8[8h]", "h", 8)]
        [DataRow("M8[W/2]", "<M8[84h]", "h", 84)]
        [DataRow("m8[h/4]", "<m8[15m]", "m", 15)]
        [DataRow("m8[ms/8]", "<m8[125us]", "us", 125)]
        [DataRow("m8[us/1000]", "<m8[ns]", "ns", 1)]
        [DataRow("m8[fs/2]", "<m8[500as]", "as", 500)]
        [DataRow("m8[M/2]", "<m8[2W]", "W", 2)]
        [DataRow("m8[M/5]", "<m8[6D]", "D", 6)]
        [DataRow("m8[2s/4]", "<m8[500ms]", "ms", 500)]
        [DataRow("m8[3s/2]", "<m8[1500ms]", "ms", 1500)]
        [DataRow("m8[W/7]", "<m8[D]", "D", 1)]
        [DataRow("m8[Y/12]", "<m8[M]", "M", 1)]
        [DataRow("m8[D/24]", "<m8[h]", "h", 1)]
        [DataRow("m8[s/1000]", "<m8[ms]", "ms", 1)]
        [DataRow("m8[ns/1000]", "<m8[ps]", "ps", 1)]
        [DataRow("M8[10Y/2]", "<M8[60M]", "M", 60)]
        [DataRow("M8[s/-2]", "<M8[-500ms]", "ms", -500)]
        [DataRow("M8[W/11]", "<M8[0Y]", "Y", 0)]
        public void Divisor_BecomesMultipleOfFinerUnit(string spelling, string str, string unit, int count)
        {
            // convert_datetime_divisor_to_multiple, table quirks included: a negative divisor is accepted and W/11 walks
            // off the end of NumPy's multiples table into a zero entry ([0Y]) — both probed on 2.4.2.
            np.dtype(spelling).str.Should().Be(str);
            Data(spelling).Should().Be((unit, count));
        }

        [DataTestMethod]
        [DataRow("m8[s/7]", "divisor (7) is not a multiple of a lower-unit in datetime metadata \"[s/7]\"")]
        [DataRow("m8[as/2]", "divisor (2) is not a multiple of a lower-unit in datetime metadata \"[as/2]\"")]
        [DataRow("m8[Y/7]", "divisor (7) is not a multiple of a lower-unit in datetime metadata \"[Y/7]\"")]
        [DataRow("m8[generic/2]", "Can't use 'den' divisor with generic units")]
        public void Divisor_Errors_AreNumPysValueErrors(string spelling, string message)
        {
            Action act = () => np.dtype(spelling);
            act.Should().Throw<ValueError>().WithMessage(message);
        }

        [DataTestMethod]
        [DataRow("M8[]", "Invalid datetime metadata string \"[]\"")]
        [DataRow("M8[", "Invalid datetime metadata string \"[\"")]
        [DataRow("M8]", "Invalid datetime metadata string \"]\"")]
        [DataRow("M8[5]", "Invalid datetime metadata string \"[5]\" at position 2")]
        [DataRow("M8[5x]", "Invalid datetime unit in metadata string \"[5x]\"")]
        [DataRow("M8[x]", "Invalid datetime unit in metadata string \"[x]\"")]
        [DataRow("M8[ x]", "Invalid datetime unit in metadata string \"[ x]\"")]
        [DataRow("M8[s", "Invalid datetime metadata string \"[s\"")]
        [DataRow("M8[s]x", "Invalid datetime metadata string \"[s]x\" at position 3")]
        [DataRow("M8[Y]x", "Invalid datetime metadata string \"[Y]x\" at position 3")]
        [DataRow("M8[-1s]", "Invalid datetime metadata string \"[-1s]\" at position 1")]
        [DataRow("M8[9999999999s]", "Invalid datetime metadata string \"[9999999999s]\" at position 1")]
        [DataRow("M8[s/]", "Invalid datetime metadata string \"[s/]\" at position 3")]
        [DataRow("M8[/2]", "Invalid datetime metadata string \"[/2]\" at position 1")]
        [DataRow("M8[5/2]", "Invalid datetime metadata string \"[5/2]\" at position 2")]
        [DataRow("datetime64[]", "Invalid datetime metadata string \"[]\"")]
        [DataRow("datetime64x", "Invalid datetime metadata string \"x\"")]
        [DataRow("M8x", "Invalid datetime metadata string \"x\"")]
        public void Metadata_Errors_AreNumPysTypeErrors_Verbatim(string spelling, string message)
        {
            Action act = () => np.dtype(spelling);
            act.Should().Throw<TypeError>().WithMessage(message);
        }

        [TestMethod]
        [Misaligned]
        public void ZeroDivisor_RaisesInsteadOfCrashing()
        {
            // np.dtype('M8[s/0]') crashes the CPython interpreter (integer division by zero in
            // convert_datetime_divisor_to_multiple); NumSharp raises the metadata TypeError instead.
            Action act = () => np.dtype("M8[s/0]");
            act.Should().Throw<TypeError>().WithMessage("Invalid datetime metadata string \"[s/0]\" at position 3");
        }

        [DataTestMethod]
        [DataRow("garbage", "data type 'garbage' not understood")]
        [DataRow("", "data type '' not understood")]
        [DataRow(">", "data type '>' not understood")]
        [DataRow("i3", "data type 'i3' not understood")]
        [DataRow("f32", "data type 'f32' not understood")]
        [DataRow("I4", "data type 'I4' not understood")]
        [DataRow(" i4", "data type ' i4' not understood")]
        [DataRow("i4 ", "data type 'i4 ' not understood")]
        [DataRow("M9", "data type 'M9' not understood")]
        [DataRow("datetime6", "data type 'datetime6' not understood")]
        [DataRow("f-8", "data type 'f-8' not understood")]
        [DataRow("?1", "data type '?1' not understood")]
        [DataRow("bool8", "Alias 'bool8' was removed in NumPy 2.0. Use a name without a digit at the end.")]
        [DataRow("int0", "Alias 'int0' was removed in NumPy 2.0. Use a name without a digit at the end.")]
        public void InvalidStrings_RaiseNumPysTypeError(string spelling, string message)
        {
            Action act = () => np.dtype(spelling);
            act.Should().Throw<TypeError>().Which.Message.Should().Be(message);
        }

        /// <summary>
        ///     NumPy's own TypeError for a string it rejects, and NumSharp's NotSupportedException for a valid NumPy dtype
        ///     it has no class for, are told apart by the host: the width of the C long double is a NumPy dtype only
        ///     where the long double has it (<c>'f16'</c> / <c>'float128'</c> on a 16-byte host, 64-bit Linux), and is
        ///     NumPy's "not understood" elsewhere (Windows, Apple arm64). Probed: Windows NumPy 2.4.2 raises
        ///     <c>TypeError: data type 'f16' not understood</c>, Linux NumPy 2.4.2 answers <c>dtype('float128')</c>.
        /// </summary>
        [TestMethod]
        public void LongDoubleWidth_IsNotUnderstood_OnlyWhereNumPyLacksIt()
        {
            int ld = DTypeRegistry.CLongDoubleSize;
            foreach (var (spec, has) in new[] { ("f16", ld == 16), ("c32", ld == 16), ("float128", ld == 16), ("complex256", ld == 16),
                                                ("f12", ld == 12), ("c24", ld == 12), ("float96", ld == 12), ("complex192", ld == 12) })
            {
                Action act = () => np.dtype(spec);
                if (has)
                    act.Should().Throw<NotSupportedException>(spec).Which.Message.Should().Be($"NumPy dtype '{spec}' is not supported by NumSharp");
                else
                    act.Should().Throw<TypeError>(spec).Which.Message.Should().Be($"data type '{spec}' not understood");
            }
            // A byte-order prefix before a kind + size is a byte order, before a name it is part of the name (NumPy
            // looks names up by the whole string), so '<float128' is "not understood" on every host.
            new Action(() => np.dtype("<float128")).Should().Throw<TypeError>().Which.Message.Should().Be("data type '<float128' not understood");
        }

        [DataTestMethod]
        [DataRow("i04", "<i4")]
        [DataRow("i 4", "<i4")]
        [DataRow("i+4", "<i4")]
        [DataRow("u08", "<u8")]
        [DataRow("f002", "<f2")]
        public void SizedCodes_FollowStrtol(string spelling, string str)
        {
            // NumPy parses the tail with strtol: leading whitespace, an explicit '+' and leading zeros are accepted.
            np.dtype(spelling).str.Should().Be(str);
        }

        [TestMethod]
        public void UnsupportedNumPyTypes_StillRaise()
        {
            // The fixed-width string spellings (S, S10, U, U32, c, str, bytes_) parse since the string dtypes
            // landed (see StringDTypeGrammar_ParsesNumPysSpellings); void, object, complex64 and structured stay refused.
            foreach (var s in new[] { "V", "V16", "O", "c8", "complex64", "F", "object", "i4,f8", "3f8", "(2,3)f8" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<NotSupportedException>(s);
            }
        }

        /// <summary>
        ///     NumPy looks a NAME up in its type dictionary by the WHOLE string (<c>PyDict_GetItemWithError(typeDict, obj)</c>),
        ///     and no name starts with a byte-order character, so a prefix before a name is "not understood" — while the same
        ///     prefix before a type code or a kind + size is read as a byte order. NumSharp's own C# aliases follow the
        ///     same rule. Probed against NumPy 2.4.2.
        /// </summary>
        [DataTestMethod]
        [DataRow("<float64")]
        [DataRow(">float64")]
        [DataRow("=float64")]
        [DataRow("|float64")]
        [DataRow("|bool")]
        [DataRow(">double")]
        [DataRow("<int")]
        [DataRow("=complex128")]
        [DataRow("<cdouble")]
        [DataRow("<datetime")]
        [DataRow("<Int32")]
        public void ByteOrderPrefix_BeforeAName_IsNotUnderstood(string spelling)
        {
            Action act = () => np.dtype(spelling);
            act.Should().Throw<TypeError>().WithMessage($"data type '{spelling}' not understood");
        }

        /// <summary>The same prefixes before type codes and sized codes are byte orders (NumPy strips them first).</summary>
        [TestMethod]
        public void ByteOrderPrefix_BeforeACode_IsAByteOrder()
        {
            np.dtype("<f8").Should().BeSameAs(DType.Double, "an explicit host order folds into the native singleton");
            np.dtype(">d").ToString(true).Should().Be("dtype('>f8')");
            np.dtype("|?").Should().BeSameAs(DType.Boolean);
            np.dtype("=b1").Should().BeSameAs(DType.Boolean);
            np.dtype(">n").ToString(true).Should().Be(IntPtr.Size == 8 ? "dtype('>i8')" : "dtype('>i4')");
        }

        /// <summary>
        ///     NumPy 2.x names and codes the old table lacked: <c>'cdouble'</c> (complex128's C name, in
        ///     <c>np.sctypeDict</c>) and the intp letters <c>'n'</c> / <c>'N'</c> (NumPy 2.0 moved <c>np.intp</c> to
        ///     <c>'n'</c>; <c>'p'</c> stays pointer-sized, the same integer on every platform NumSharp runs on).
        /// </summary>
        [TestMethod]
        public void NumPy2Spellings_cdouble_n_N()
        {
            np.dtype("cdouble").Should().BeSameAs(DType.Complex);
            np.dtype("n").Should().BeSameAs(IntPtr.Size == 8 ? DType.Int64 : DType.Int32);
            np.dtype("N").Should().BeSameAs(IntPtr.Size == 8 ? DType.UInt64 : DType.UInt32);
            np.dtype("p").Should().BeSameAs(np.dtype("n"));
            foreach (var key in np.sctypeDict.Keys)
                np.dtype(key).Should().Be(np.sctypeDict[key], $"np.dtype('{key}') reads the same table as np.sctypeDict");
        }

        /// <summary>
        ///     NumPy's error texts quote the dtype string with Python's <c>repr</c> (<c>%R</c>): a single quote inside
        ///     switches to double quotes and a control character is escaped; the removed-alias text quotes the whole
        ///     string, prefix included.
        /// </summary>
        [TestMethod]
        public void ErrorTexts_QuoteTheStringWithPythonRepr()
        {
            Action quote = () => np.dtype("a'b");
            quote.Should().Throw<TypeError>().WithMessage("data type \"a'b\" not understood");
            Action newline = () => np.dtype("i4\n");
            newline.Should().Throw<TypeError>().WithMessage("data type 'i4\\n' not understood");
            Action prefixedAlias = () => np.dtype("<int0");
            prefixedAlias.Should().Throw<TypeError>()
                .WithMessage("Alias '<int0' was removed in NumPy 2.0. Use a name without a digit at the end.");
        }

        /// <summary>
        ///     The size tail is parsed with C's <c>strtol</c>, which skips only C's six ASCII whitespace characters — a
        ///     tab or a line feed before the size is fine, a no-break space or any other Unicode space is not.
        /// </summary>
        [TestMethod]
        public void SizeTail_SkipsOnlyCWhitespace()
        {
            np.dtype("f\t8").Should().BeSameAs(DType.Double);
            np.dtype("f\n8").Should().BeSameAs(DType.Double);
            np.dtype("i\v4").Should().BeSameAs(DType.Int32);
            Action nbsp = () => np.dtype("f\u00a08");
            nbsp.Should().Throw<TypeError>().WithMessage("data type 'f\\xa08' not understood");
            Action enQuad = () => np.dtype("f\u20008");
            enQuad.Should().Throw<TypeError>().WithMessage("data type 'f\\u20008' not understood");
        }

        /// <summary>
        ///     A lone character below <c>'\x18'</c> is a TYPE NUMBER to NumPy's <c>PyArray_DescrFromType</c>, not a letter:
        ///     <c>'\x00'</c> is bool, <c>'\x0c'</c> float64, <c>'\x17'</c> float16, <c>'\x0d'</c> the long double (NumSharp's
        ///     own <c>LongDoubleDType</c> over the 8-byte float64 lane, <c>dtype('float64')</c> with character <c>'g'</c> as on
        ///     NumPy's Windows build), <c>'\x12'</c> / <c>'\x13'</c> the unsized bytes / str builtins (<c>dtype('S')</c>,
        ///     <c>dtype('&lt;U')</c>); a byte-order prefix still applies, and a number NumSharp has no class for (complex64,
        ///     object, void) is refused as unsupported. <c>'\x18'</c> (the expired <c>NPY_CHAR</c>) is not a dtype.
        /// </summary>
        [TestMethod]
        public void ControlCharacters_AreTypeNumbers()
        {
            np.dtype("\x00").Should().BeSameAs(DType.Boolean);
            np.dtype("\x01").Should().BeSameAs(DType.SByte);
            np.dtype("\x0c").Should().BeSameAs(DType.Double);
            np.dtype("\x0d").Should().BeSameAs(DTypeRegistry.LongDouble.singleton);
            np.dtype("\x0d").ToString(true).Should().Be("dtype('float64')");
            np.dtype("\x0d").@char.Should().Be('g');
            np.dtype("\x0f").Should().BeSameAs(DType.Complex);
            np.dtype("\x12").Should().BeSameAs(DTypeRegistry.Bytes.singleton);
            np.dtype("\x13").Should().BeSameAs(DTypeRegistry.Str.singleton);
            np.dtype("\x17").Should().BeSameAs(DType.Half);
            np.dtype("\x15").Should().BeSameAs(DTypeRegistry.DateTime64.singleton);
            np.dtype(">\x0c").ToString(true).Should().Be("dtype('>f8')");
            Action complex64 = () => np.dtype("\x0e");
            complex64.Should().Throw<NotSupportedException>();
            Action obj = () => np.dtype("\x11");
            obj.Should().Throw<NotSupportedException>();
            Action voidType = () => np.dtype("\x14");
            voidType.Should().Throw<NotSupportedException>();
            Action expiredChar = () => np.dtype("\x18");
            expiredChar.Should().Throw<TypeError>().WithMessage("data type '\\x18' not understood");
        }

        /// <summary>
        ///     Parentheses outside a comma string are no sub-array to NumPy: <c>'(2)f8'</c>, <c>'( )f8'</c> and
        ///     <c>'f8()'</c> are "not understood". A leading empty tuple, also after a byte-order character, IS a comma
        ///     string (<c>_check_for_commastring</c>): <c>(type, ())</c> is the type itself, so <c>'()f8'</c> is the float64
        ///     singleton and <c>'&gt;()i4'</c> a big-endian int32, while <c>'(2,)f8'</c> is a sub-array NumSharp refuses.
        /// </summary>
        [TestMethod]
        public void Parentheses_FollowNumPysCommaStringTest()
        {
            foreach (var s in new[] { "(2)f8", "( )f8", "f8()", "(f8)" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<TypeError>().WithMessage($"data type '{s}' not understood");
            }
            np.dtype("()f8").Should().BeSameAs(DType.Double);
            np.dtype("<()f8").Should().BeSameAs(DType.Double);
            np.dtype(">()i4").ToString(true).Should().Be("dtype('>i4')");
            Action subArray = () => np.dtype("(2,)f8");
            subArray.Should().Throw<NotSupportedException>().WithMessage("NumPy dtype '(2,)f8' (a sub-array dtype) is not supported by NumSharp");
        }

        /// <summary>
        ///     NumPy's <c>_convert_from_tuple</c> over a comma string's <c>(dtype, repeats)</c>: a count SIZES an unsized
        ///     string class (a fresh descriptor, byte order kept), and the texts of every refusal are NumPy's. Values from
        ///     NumPy 2.5.4 (which removed the <c>'a'</c> alias of <c>'S'</c>: <c>'2a'</c> is no longer a dtype).
        /// </summary>
        [TestMethod]
        public void CommaString_RepeatsSizeAnUnsizedStringClass()
        {
            var u2 = np.dtype("2U");
            u2.ToString(true).Should().Be("dtype('<U2')");
            u2.itemsize.Should().Be(8);
            u2.isbuiltin.Should().Be(0);
            np.dtype("2>U").ToString(true).Should().Be("dtype('>U2')");
            np.dtype(">2U").ToString(true).Should().Be("dtype('>U2')");
            np.dtype("2S").ToString(true).Should().Be("dtype('S2')");
            np.dtype("2>S").ToString(true).Should().Be("dtype('S2')");
            Action alias = () => np.dtype("2a");
            alias.Should().Throw<TypeError>().WithMessage("data type 'a' not understood");
            // '0U' is a FRESH unsized U (np.dtype('U') is the singleton): NumPy's PyArray_DESCR_REPLACE copies it.
            var u0 = np.dtype("0U");
            u0.ToString(true).Should().Be("dtype('<U')");
            u0.isbuiltin.Should().Be(0);
            u0.Should().NotBeSameAs(np.dtype("U"));
            np.dtype("536870911U").itemsize.Should().Be(2147483644);
            np.dtype("2147483647S").itemsize.Should().Be(int.MaxValue);

            foreach (var s in new[] { "536870912U", "2147483648S", "(1,)U", "()U", "()S", "(2,3)S", "99999999999999999999U", "()V", "(1,)>0f8" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<ValueError>().WithMessage("invalid itemsize in generic type tuple", s);
            }
        }

        /// <summary>
        ///     A comma string's sub-array shape is validated as NumPy validates it before NumSharp refuses the sub-array:
        ///     C int dimensions, the byte size in a C int (left to right in <c>npy_intp</c>, so a 0 after an overflow comes
        ///     too late), at most 64 dimensions, and nothing past <c>npy_intp</c>. A <c>StringDType</c> base is NumPy 2.5's
        ///     TypeError (<c>arraydescr_new_from_subarray</c>), raised once the shape converts and before its dimensions are
        ///     validated, so only the converter's own refusals still beat it. Texts from NumPy 2.5.4.
        /// </summary>
        [TestMethod]
        public void CommaString_SubArrayShapes_AreCheckedBeforeTheRefusal()
        {
            ExpectValueError("3000000000f8", "invalid shape in fixed-type tuple: dimension does not fit into a C int.");
            ExpectValueError("(0,3000000000)f8", "invalid shape in fixed-type tuple: dimension does not fit into a C int.");
            ExpectValueError("99999999999999999999f8", "invalid shape in fixed-type tuple.");
            ExpectValueError("(" + string.Join(",", Enumerable.Repeat("1", 65)) + ")f8", "invalid shape in fixed-type tuple.");
            ExpectValueError("(65536,65536)f8", "invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int.");
            ExpectValueError("300000000f8", "invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int.");
            ExpectValueError("(2147483647,2147483647,2147483647,0)f8", "invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int.");
            // NumPy 2.4.2 built StringDType sub-arrays, so '134217728T' failed the byte-size check; NumPy 2.5 refuses the
            // base first, whatever the dimensions, and NumSharp has no sub-array to refuse after it.
            const string noStringSubarray = "numpy.dtypes.StringDType is not currently supported within subarray dtypes.";
            foreach (var s in new[] { "134217728T", "2T", "(2,)T", "1T", "0T", "3000000000T", "(65536,65536)T", "2,T", "T,2T" })
                ExpectTypeError(s, noStringSubarray);
            ExpectValueError("99999999999999999999T", "invalid shape in fixed-type tuple.");
            ExpectValueError("(" + string.Join(",", Enumerable.Repeat("1", 65)) + ")T", "invalid shape in fixed-type tuple.");
            // A dtype NumSharp lacks still has NumPy's item size: complex64 is 8 bytes.
            ExpectValueError("300000000F", "invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int.");

            foreach (var s in new[] { "268435455f8", "(2147483647,2,0)f8", "(" + string.Join(",", Enumerable.Repeat("1", 64)) + ")f8", "2c", "2>3f8", "2,3f8", "2 ,f8" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<NotSupportedException>().WithMessage("*(a sub-array dtype) is not supported by NumSharp", s);
            }
        }

        /// <summary>
        ///     The fields of a comma string: NumPy's separator and padding (Python's Unicode whitespace), byte-order merge and
        ///     field-list texts, and NumPy's ORDER - every field is split and literal-evaluated before any converts, so a
        ///     SyntaxError in field 3 beats a TypeError in field 1, and a dtype NumSharp lacks in field 1 (<c>'F'</c>) does
        ///     not hide NumPy's TypeError in field 2.
        /// </summary>
        [TestMethod]
        public void CommaString_Fields_FollowNumPysOrderAndTexts()
        {
            foreach (var s in new[] { "i4,f8", "f8, i4", "f8 ,i4", "i4,", "i4,<", "i4, ", "f8, i4", "f8,\x1ci4", "f8,i4　", "U,S", "2f8,i4" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<NotSupportedException>().WithMessage("*(a structured dtype) is not supported by NumSharp", s);
            }

            ExpectValueError("f8,x y", "format number 2 of \"f8,x y\" is not recognized");
            ExpectValueError("f8,​i4", "format number 2 of \"f8,​i4\" is not recognized");
            ExpectValueError("()()f8", "format number 1 of \"()()f8\" is not recognized");
            ExpectValueError("<()>f8", "inconsistent byte-order specification < and >");
            ExpectValueError(">2=i4", "inconsistent byte-order specification > and <");
            ExpectValueError("|2=i4", "inconsistent byte-order specification | and <");
            ExpectValueError("\t,", "Expected at least one field name");

            ExpectTypeError("\t,i4", "data type '' not understood");
            ExpectTypeError("i4,>", "data type '>' not understood");
            ExpectTypeError("()", "data type '' not understood");
            ExpectTypeError("i4,2>3x", "data type '>x' not understood");
            ExpectTypeError("F,xyz", "data type 'xyz' not understood");
            ExpectTypeError("2V,xyz", "data type 'xyz' not understood");
            ExpectTypeError("2M8[s,x]", "Invalid datetime unit in metadata string \"[s,x]\"");

            Action syntaxFirst = () => np.dtype("xyz,f8,,");
            syntaxFirst.Should().Throw<SyntaxError>().WithMessage("invalid syntax (<unknown>, line 1)");
            Action unsupportedLast = () => np.dtype("()F");
            unsupportedLast.Should().Throw<NotSupportedException>().WithMessage("NumPy dtype '()F' is not supported by NumSharp");
        }

        /// <summary>
        ///     <c>ast.literal_eval</c> of a comma string's repeat group raises CPython 3.12's SyntaxError texts, with the
        ///     tokenizer's errors first (leading zeros, an unmatched <c>')'</c>) and the parser's two-pass messages after.
        ///     The port is checked against every string of the group's language up to 7 characters; these are its cases.
        /// </summary>
        [TestMethod]
        public void CommaString_RepeatGroup_RaisesCPythonsSyntaxErrors()
        {
            ExpectSyntaxError("01f8", "leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers (<unknown>, line 1)");
            ExpectSyntaxError("i4,(1 02)f8", "leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers (<unknown>, line 1)");
            ExpectSyntaxError("1 2f8", "invalid syntax (<unknown>, line 1)");
            ExpectSyntaxError("(,)f8", "invalid syntax (<unknown>, line 1)");
            ExpectSyntaxError("i4,(1 2)f8", "invalid syntax. Perhaps you forgot a comma? (<unknown>, line 1)");
            ExpectSyntaxError("i4,(1 2,", "invalid syntax. Perhaps you forgot a comma? (<unknown>, line 1)");
            ExpectSyntaxError("i4,(1 2", "'(' was never closed (<unknown>, line 1)");
            ExpectSyntaxError("f8,(1", "'(' was never closed (<unknown>, line 1)");
            ExpectSyntaxError("i4,)", "unmatched ')' (<unknown>, line 1)");
            ExpectSyntaxError("i4,< f8", "invalid syntax (<unknown>, line 0)");

            // Valid groups: '00' is the int 0 (only a zero may lead a decimal literal of zeros), spaces separate nothing
            // ('2 U' is 2), and a trailing comma makes a TUPLE, which no item size reads ('2,U').
            np.dtype("00U").ToString(true).Should().Be("dtype('<U')");
            np.dtype("2 U").ToString(true).Should().Be("dtype('<U2')");
            ExpectValueError("2,U", "invalid itemsize in generic type tuple");
            // '(2)U' is no comma string at all (neither a leading digit nor '()' nor a comma): NumPy's "not understood".
            ExpectTypeError("(2)U", "data type '(2)U' not understood");
        }

        /// <summary>Asserts <c>np.dtype(spec)</c> raises <see cref="ValueError"/> with <paramref name="text"/>.</summary>
        /// <param name="spec">The dtype string.</param>
        /// <param name="text">NumPy's message.</param>
        private static void ExpectValueError(string spec, string text)
        {
            Action act = () => np.dtype(spec);
            act.Should().Throw<ValueError>().Which.Message.Should().Be(text, spec);
        }

        /// <summary>Asserts <c>np.dtype(spec)</c> raises <see cref="TypeError"/> with <paramref name="text"/>.</summary>
        /// <param name="spec">The dtype string.</param>
        /// <param name="text">NumPy's message.</param>
        private static void ExpectTypeError(string spec, string text)
        {
            Action act = () => np.dtype(spec);
            act.Should().Throw<TypeError>().Which.Message.Should().Be(text, spec);
        }

        /// <summary>Asserts <c>np.dtype(spec)</c> raises <see cref="SyntaxError"/> with <paramref name="text"/>.</summary>
        /// <param name="spec">The dtype string.</param>
        /// <param name="text">CPython's <c>str(e)</c>.</param>
        private static void ExpectSyntaxError(string spec, string text)
        {
            Action act = () => np.dtype(spec);
            act.Should().Throw<SyntaxError>().Which.Message.Should().Be(text, spec);
        }

        /// <summary>
        ///     Valid NumPy dtypes NumSharp has no class for (complex64, object, void) are "not supported"; the same spellings
        ///     NumPy itself refuses keep NumPy's "not understood" (a prefixed name - NumPy looks names up by the whole string,
        ///     so <c>'&lt;str'</c> and <c>'&lt;bytes'</c> too -, <c>'O16'</c>, and every spelling of the <c>'a'</c> bytes
        ///     alias NumPy 2.5 removed: <c>'a'</c>, <c>'|a'</c>, <c>'a5'</c>, <c>'&gt;a4'</c>, <c>'a0'</c>). The string classes
        ///     are NumSharp's own: <c>'bytes'</c>, <c>'&gt;S'</c>, <c>'T'</c> and <c>'&lt;T'</c> parse.
        /// </summary>
        [TestMethod]
        public void UnsupportedClasses_AreTold_FromInvalidStrings()
        {
            foreach (var s in new[] { "csingle", "complex64", "object", "O4", "O8", "<O8", "void", "V" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<NotSupportedException>().WithMessage("*is not supported by NumSharp", s);
            }
            foreach (var s in new[] { "<object", "|a", "O16", "O2", "<csingle", "<str", "<bytes", "a", "a5", "|a5", ">a4", "a0" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<TypeError>().WithMessage($"data type '{s}' not understood", s);
            }
            np.dtype("bytes").Should().BeSameAs(DTypeRegistry.Bytes.singleton);
            np.dtype(">S").Should().BeSameAs(DTypeRegistry.Bytes.singleton);
            np.dtype("T").Meta.Should().BeSameAs(DTypeRegistry.VString);
            np.dtype("<T").Meta.Should().BeSameAs(DTypeRegistry.VString);
        }

        [TestMethod]
        public void StringDTypeGrammar_ParsesNumPysSpellings()
        {
            // probed 2.5.4: np.dtype(spelling).str (the 'a' / 'a5' alias spellings are gone since 2.5)
            foreach (var (spelling, str) in new[]
                     {
                         ("S", "|S0"), ("S10", "|S10"), ("U", "<U0"), ("U32", "<U32"),
                         ("c", "|S1"), ("str", "<U0"), ("bytes_", "|S0"), ("<U3", "<U3"), ("|S3", "|S3"), ("=U2", "<U2"),
                     })
                np.dtype(spelling).str.Should().Be(str, spelling);
        }

        [TestMethod]
        public void ByteOrderPrefix_IsRetainedOnlyWhenNonNative()
        {
            np.dtype(">i4").byteorder.Should().Be('>');
            np.dtype(">M8[ns]").byteorder.Should().Be('>');
            np.dtype(">m8").byteorder.Should().Be('>');
            np.dtype("<M8[ns]").byteorder.Should().Be('=');
            np.dtype("=M8[ns]").byteorder.Should().Be('=');
            np.dtype("|M8[ns]").byteorder.Should().Be('=');
            np.dtype(">b1").byteorder.Should().Be('|');
            np.dtype(">i4").Should().Be(np.dtype("i4").newbyteorder(">"));
        }

        [TestMethod]
        public void DatetimeMetaData_Struct_Api()
        {
            var m = DatetimeMetaData.Parse("[10ns]");
            m.@base.Should().Be(NPY_DATETIMEUNIT.NPY_FR_ns);
            m.num.Should().Be(10);
            m.UnitString.Should().Be("ns");
            m.ToString().Should().Be("[10ns]");
            m.ToString(skipBrackets: true).Should().Be("10ns");
            DatetimeMetaData.Parse("").Should().Be(DatetimeMetaData.Generic);
            DatetimeMetaData.Generic.ToString().Should().Be("");
            DatetimeMetaData.Generic.ToString(skipBrackets: true).Should().Be("generic");
            new DatetimeMetaData(NPY_DATETIMEUNIT.NPY_FR_s).ToString().Should().Be("[s]");
            DatetimeMetaData.ParseUnit("as").Should().Be(NPY_DATETIMEUNIT.NPY_FR_as);
            DatetimeMetaData.ParseUnit("μs").Should().Be(NPY_DATETIMEUNIT.NPY_FR_us);
            DatetimeMetaData.ParseUnit("generic").Should().Be(NPY_DATETIMEUNIT.NPY_FR_GENERIC);
            Action bad = () => DatetimeMetaData.ParseUnit("x");
            bad.Should().Throw<TypeError>().WithMessage("Invalid datetime unit \"x\" in metadata");
            (m == new DatetimeMetaData(NPY_DATETIMEUNIT.NPY_FR_ns, 10)).Should().BeTrue();
            (m != DatetimeMetaData.Generic).Should().BeTrue();
            ((int)NPY_DATETIMEUNIT.NPY_FR_D).Should().Be(4, "the enum keeps NumPy's gap where the 1.6 business-day unit was");
        }
    }
}
