using System;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The dtype-string grammar of <see cref="np.dtype(string)"/> — the port of <c>descriptor.c::_convert_from_str</c>
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
        [DataRow("f16", "data type 'f16' not understood")]
        [DataRow("I4", "data type 'I4' not understood")]
        [DataRow(" i4", "data type ' i4' not understood")]
        [DataRow("i4 ", "data type 'i4 ' not understood")]
        [DataRow("M9", "data type 'M9' not understood")]
        [DataRow("datetime6", "data type 'datetime6' not understood")]
        [DataRow("f-8", "data type 'f-8' not understood")]
        [DataRow("?1", "data type '?1' not understood")]
        [DataRow("bool8", "Alias 'bool8' was removed in NumPy 2.0. Use a name without a digit at the end.")]
        [DataRow("int0", "Alias 'int0' was removed in NumPy 2.0. Use a name without a digit at the end.")]
        public void InvalidStrings_KeepNotSupportedException_WithNumPysWording(string spelling, string message)
        {
            Action act = () => np.dtype(spelling);
            act.Should().Throw<NotSupportedException>().WithMessage(message);
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
            foreach (var s in new[] { "S", "S10", "U", "U32", "V", "V16", "O", "a", "a5", "c", "c8", "complex64", "F", "object", "str", "bytes_", "i4,f8", "3f8", "(2,3)f8" })
            {
                Action act = () => np.dtype(s);
                act.Should().Throw<NotSupportedException>(s);
            }
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
            m.Base.Should().Be(NPY_DATETIMEUNIT.NPY_FR_ns);
            m.Num.Should().Be(10);
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
