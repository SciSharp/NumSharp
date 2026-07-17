using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Pure map tests — no Python engine required, so these run on every machine and CI image.
    /// </summary>
    [TestClass]
    public class DtypeMapTests
    {
        private static readonly NPTypeCode[] NumpyMappable =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex, NPTypeCode.Char,
        };

        [TestMethod]
        public void ToNumpyDtypeStr_CoversEveryMappableDtype()
        {
            foreach (var tc in NumpyMappable)
            {
                string s = PythonConvert.ToNumpyDtypeStr(tc);
                s.Should().NotBeNullOrEmpty(tc.ToString());
                "<|".Should().Contain(s[0].ToString(), "only little-endian or order-irrelevant markers may be produced");
            }

            PythonConvert.ToNumpyDtypeStr(NPTypeCode.Char).Should().Be("<u2", "a C# char is a UTF-16 code unit");
            PythonConvert.ToNumpyDtypeStr(NPTypeCode.Boolean).Should().Be("|b1");
            PythonConvert.ToNumpyDtypeStr(NPTypeCode.Complex).Should().Be("<c16");
        }

        [TestMethod]
        public void ToNumpyDtypeStr_Decimal_Throws()
        {
            ((Action)(() => PythonConvert.ToNumpyDtypeStr(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>().WithMessage("*decimal*");
        }

        [TestMethod]
        public void FromNumpyDtypeStr_RoundTripsEveryExport()
        {
            foreach (var tc in NumpyMappable)
            {
                var expected = tc == NPTypeCode.Char ? NPTypeCode.UInt16 : tc;   // char is exported as its u2 proxy
                PythonConvert.FromNumpyDtypeStr(PythonConvert.ToNumpyDtypeStr(tc)).Should().Be(expected, tc.ToString());
            }
        }

        [TestMethod]
        public void FromNumpyDtypeStr_AcceptsMarkerVariants_RejectsForeign()
        {
            PythonConvert.FromNumpyDtypeStr("f8").Should().Be(NPTypeCode.Double, "bare code");
            PythonConvert.FromNumpyDtypeStr("=f4").Should().Be(NPTypeCode.Single, "native marker");
            PythonConvert.FromNumpyDtypeStr("|b1").Should().Be(NPTypeCode.Boolean);

            ((Action)(() => PythonConvert.FromNumpyDtypeStr(">f8")))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian*", "silent byte-swapping is a data corruption");
            ((Action)(() => PythonConvert.FromNumpyDtypeStr("<M8[ns]")))
                .Should().Throw<NotSupportedException>("datetime64 has no NumSharp dtype");
            ((Action)(() => PythonConvert.FromNumpyDtypeStr("V16")))
                .Should().Throw<NotSupportedException>("void/record dtypes have no NumSharp dtype");
        }

        [TestMethod]
        public void ToBufferFormat_RoundTripsThroughFromBufferFormat()
        {
            foreach (var tc in NumpyMappable)
            {
                string fmt = PythonConvert.ToBufferFormat(tc);
                var expected = tc == NPTypeCode.Char ? NPTypeCode.UInt16 : tc;
                PythonConvert.FromBufferFormat(fmt, tc.SizeOf()).Should().Be(expected, $"{tc} -> '{fmt}'");
            }

            ((Action)(() => PythonConvert.ToBufferFormat(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void FromBufferFormat_PlatformAmbiguousCodes_UseItemSize()
        {
            PythonConvert.FromBufferFormat("l", 4).Should().Be(NPTypeCode.Int32, "32-bit C long (Windows)");
            PythonConvert.FromBufferFormat("l", 8).Should().Be(NPTypeCode.Int64, "64-bit C long (unix)");
            PythonConvert.FromBufferFormat("L", 4).Should().Be(NPTypeCode.UInt32);
            PythonConvert.FromBufferFormat("L", 8).Should().Be(NPTypeCode.UInt64);
            PythonConvert.FromBufferFormat("@d", 8).Should().Be(NPTypeCode.Double, "native-order marker strips");
            PythonConvert.FromBufferFormat("", 1).Should().Be(NPTypeCode.Byte, "empty format means raw bytes");
        }

        [TestMethod]
        public void FromBufferFormat_BigEndian_RejectedForMultiByte_AllowedForSingleByte()
        {
            ((Action)(() => PythonConvert.FromBufferFormat(">i", 4)))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian*");
            ((Action)(() => PythonConvert.FromBufferFormat("!H", 2)))
                .Should().Throw<NotSupportedException>("'!' is network byte order, i.e. big-endian");

            PythonConvert.FromBufferFormat(">B", 1).Should().Be(NPTypeCode.Byte, "byte order is irrelevant for 1-byte types");
            PythonConvert.FromBufferFormat(">?", 1).Should().Be(NPTypeCode.Boolean);
        }

        [TestMethod]
        public void FromBufferFormat_Complex64_ThrowsWithWideningGuidance()
        {
            ((Action)(() => PythonConvert.FromBufferFormat("Zf", 8)))
                .Should().Throw<NotSupportedException>().WithMessage("*complex64*ToNDArray*");
        }

        [TestMethod]
        public void FromBufferFormat_UnknownCode_Throws()
        {
            ((Action)(() => PythonConvert.FromBufferFormat("x", 1)))
                .Should().Throw<NotSupportedException>().WithMessage("*'x'*");
        }
    }
}
