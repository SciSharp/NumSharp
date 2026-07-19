using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;

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
                string s = NDArrayPythonInterop.ToNumpyDtypeStr(tc);
                s.Should().NotBeNullOrEmpty(tc.ToString());
                "<|".Should().Contain(s[0].ToString(), "only little-endian or order-irrelevant markers may be produced");
            }

            NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Char).Should().Be("<u2", "a C# char is a UTF-16 code unit");
            NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Boolean).Should().Be("|b1");
            NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Complex).Should().Be("<c16");
        }

        [TestMethod]
        public void ToNumpyDtypeStr_Decimal_Throws()
        {
            ((Action)(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>().WithMessage("*decimal*");
        }

        [TestMethod]
        public void FromNumpyDtypeStr_RoundTripsEveryExport()
        {
            foreach (var tc in NumpyMappable)
            {
                var expected = tc == NPTypeCode.Char ? NPTypeCode.UInt16 : tc;   // char is exported as its u2 proxy
                NDArrayPythonInterop.FromNumpyDtypeStr(NDArrayPythonInterop.ToNumpyDtypeStr(tc)).Should().Be(expected, tc.ToString());
            }
        }

        [TestMethod]
        public void FromNumpyDtypeStr_AcceptsMarkerVariants_RejectsForeign()
        {
            NDArrayPythonInterop.FromNumpyDtypeStr("f8").Should().Be(NPTypeCode.Double, "bare code");
            NDArrayPythonInterop.FromNumpyDtypeStr("=f4").Should().Be(NPTypeCode.Single, "native marker");
            NDArrayPythonInterop.FromNumpyDtypeStr("|b1").Should().Be(NPTypeCode.Boolean);

            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr(">f8")))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian*", "silent byte-swapping is a data corruption");
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr("<M8[ns]")))
                .Should().Throw<NotSupportedException>("datetime64 has no NumSharp dtype");
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr("V16")))
                .Should().Throw<NotSupportedException>("void/record dtypes have no NumSharp dtype");
        }

        [TestMethod]
        public void ToBufferFormat_RoundTripsThroughFromBufferFormat()
        {
            foreach (var tc in NumpyMappable)
            {
                string fmt = NDArrayPythonInterop.ToBufferFormat(tc);
                var expected = tc == NPTypeCode.Char ? NPTypeCode.UInt16 : tc;
                NDArrayPythonInterop.FromBufferFormat(fmt, tc.SizeOf()).Should().Be(expected, $"{tc} -> '{fmt}'");
            }

            ((Action)(() => NDArrayPythonInterop.ToBufferFormat(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void FromBufferFormat_PlatformAmbiguousCodes_UseItemSize()
        {
            NDArrayPythonInterop.FromBufferFormat("l", 4).Should().Be(NPTypeCode.Int32, "32-bit C long (Windows)");
            NDArrayPythonInterop.FromBufferFormat("l", 8).Should().Be(NPTypeCode.Int64, "64-bit C long (unix)");
            NDArrayPythonInterop.FromBufferFormat("L", 4).Should().Be(NPTypeCode.UInt32);
            NDArrayPythonInterop.FromBufferFormat("L", 8).Should().Be(NPTypeCode.UInt64);
            NDArrayPythonInterop.FromBufferFormat("@d", 8).Should().Be(NPTypeCode.Double, "native-order marker strips");
            NDArrayPythonInterop.FromBufferFormat("", 1).Should().Be(NPTypeCode.Byte, "empty format means raw bytes");
        }

        [TestMethod]
        public void FromBufferFormat_BigEndian_RejectedForMultiByte_AllowedForSingleByte()
        {
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat(">i", 4)))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian*");
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("!H", 2)))
                .Should().Throw<NotSupportedException>("'!' is network byte order, i.e. big-endian");

            NDArrayPythonInterop.FromBufferFormat(">B", 1).Should().Be(NPTypeCode.Byte, "byte order is irrelevant for 1-byte types");
            NDArrayPythonInterop.FromBufferFormat(">?", 1).Should().Be(NPTypeCode.Boolean);
        }

        [TestMethod]
        public void FromBufferFormat_Complex64_ThrowsWithWideningGuidance()
        {
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("Zf", 8)))
                .Should().Throw<NotSupportedException>().WithMessage("*complex64*ToNDArray*");
        }

        [TestMethod]
        public void FromBufferFormat_UnknownCode_Throws()
        {
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("x", 1)))
                .Should().Throw<NotSupportedException>().WithMessage("*'x'*");
        }
    }
}
