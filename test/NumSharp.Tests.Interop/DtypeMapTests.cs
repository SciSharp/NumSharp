using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;

namespace NumSharp.Tests.Interop
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

        [TestMethod]
        public void FromBufferFormat_WcharTextUnits_MapByWidth()
        {
            // 'u' is wchar_t: 2 bytes on windows — a UTF-16 code unit, which IS System.Char. numpy
            // itself cannot import 'u' (its _pep3118_unsupported_map lists it as "UCS-2 strings");
            // NumSharp natively has the dtype numpy lacks, so the mapping legitimately exceeds numpy.
            NDArrayPythonInterop.FromBufferFormat("u", 2).Should().Be(NPTypeCode.Char, "a 2-byte wchar_t is exactly a UTF-16 code unit");
            NDArrayPythonInterop.FromBufferFormat("<u", 2).Should().Be(NPTypeCode.Char, "ctypes.c_wchar arrays export '<u'");
            NDArrayPythonInterop.FromBufferFormat("u", 1).Should().Be(NPTypeCode.Byte, "a degenerate single-byte text unit is raw bytes");

            // A 4-byte unit is one UCS-4 code point: NumSharp's U1 holds exactly that, so it is a dtype of its own (not a
            // narrowed Char) — the same bytes numpy's '<U1' holds.
            NDArrayPythonInterop.FromBufferFormat("u", 4).Should().Be(np.dtype("U1"), "4-byte wchar_t (linux/macOS) is one UCS-4 code point");
            NDArrayPythonInterop.FromBufferFormat("w", 4).Should().Be(np.dtype("U1"), "'w' is UCS-4 by definition (array.array('w') on 3.13+)");
            NDArrayPythonInterop.FromBufferFormat("1w", 4).Should().Be(np.dtype("U1"), "numpy '<U1' exports the count-prefixed '1w'");
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat(">u", 2)))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian*", "UTF-16 is byte-order sensitive — 2-byte 'u' is not exempt like 1-byte codes");
        }

        /// <summary>
        ///     The PEP 3118 text formats numpy's buffers report map to NumSharp's sized string dtypes (and back): a count is
        ///     characters, UCS-4 has a byte order (a big-endian view is refused with the copy route), a byte string does not,
        ///     a count that disagrees with the item size matches nothing, an object buffer and a <c>StringDType</c> refuse.
        /// </summary>
        [TestMethod]
        public void FromBufferFormat_TextCodes_MapToSizedStringDtypes()
        {
            // numpy's own buffers count characters: '5s' is S5 (5 bytes), '5w' is U5 (5 code points, 20 bytes); a lone
            // code means one character. The byte order of UCS-4 matters; a byte string's does not.
            NDArrayPythonInterop.FromBufferFormat("5s", 5).Should().Be(np.dtype("S5"));
            NDArrayPythonInterop.FromBufferFormat("s", 1).Should().Be(np.dtype("S1"));
            NDArrayPythonInterop.FromBufferFormat(">5s", 5).Should().Be(np.dtype("S5"), "a byte string has no byte order");
            NDArrayPythonInterop.FromBufferFormat("5w", 20).Should().Be(np.dtype("U5"));
            NDArrayPythonInterop.FromBufferFormat("=2w", 8).Should().Be(np.dtype("U2"), "numpy marks an unaligned U field '=2w'");
            NDArrayPythonInterop.FromBufferFormat("<12w", 48).Should().Be(np.dtype("U12"));
            NDArrayPythonInterop.FromBufferFormat("5w", 20).itemsize.Should().Be(20);

            ((Action)(() => NDArrayPythonInterop.FromBufferFormat(">5w", 20)))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian UCS-4*byte-swaps*U5*");
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("!3w", 12)))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian UCS-4*");
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("5w", 16)))
                .Should().Throw<NotSupportedException>().WithMessage("*5 character(s) (20 bytes)*itemsize 16*", "a count that disagrees with the itemsize matches no dtype");
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("O", 8)))
                .Should().Throw<NotSupportedException>().WithMessage("*object dtype 'O'*str*StringDType copy*");

            // The reverse map writes exactly the formats numpy's buffers report.
            NDArrayPythonInterop.ToBufferFormat(np.dtype("S5")).Should().Be("5s");
            NDArrayPythonInterop.ToBufferFormat(np.dtype("U3")).Should().Be("3w");
            ((Action)(() => NDArrayPythonInterop.ToBufferFormat(np.dtypes.StringDType())))
                .Should().Throw<NotSupportedException>().WithMessage("*cannot include dtype 'numpy.dtypes.StringDType' in a buffer*");
        }

        /// <summary>
        ///     numpy's text typestrs map to NumSharp's sized string dtypes and back (<c>'&lt;U1'</c> is <c>U1</c>, not
        ///     <c>Char</c>); a big-endian <c>U</c> typestr, a <c>StringDType</c> typestr and the object typestr refuse with
        ///     the copy route in the message.
        /// </summary>
        [TestMethod]
        public void TextTypestrs_MapBothWays()
        {
            // The typestrs __array_interface__ and dtype.str give: '|S5', '<U5'; a byte string ignores any byte order.
            NDArrayPythonInterop.FromNumpyDtypeStr("|S5").Should().Be(np.dtype("S5"));
            NDArrayPythonInterop.FromNumpyDtypeStr("<S5").Should().Be(np.dtype("S5"));
            NDArrayPythonInterop.FromNumpyDtypeStr(">S5").Should().Be(np.dtype("S5"));
            NDArrayPythonInterop.FromNumpyDtypeStr("<U1").Should().Be(np.dtype("U1"), "numpy '<U1' is NumSharp's U1 (not Char)");
            NDArrayPythonInterop.FromNumpyDtypeStr("=U7").Should().Be(np.dtype("U7"));
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr(">U3")))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian UCS-4 text '>U3'*byte-swaps*U3*");
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr("StringDType()")))
                .Should().Throw<NotSupportedException>().WithMessage("*StringDType() has no buffer form*copy*");
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr("StringDType(na_object=None)")))
                .Should().Throw<NotSupportedException>().WithMessage("*StringDType(na_object=None) has no buffer form*");
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr("|O")))
                .Should().Throw<NotSupportedException>().WithMessage("*object dtype '|O'*");

            NDArrayPythonInterop.ToNumpyDtypeStr(np.dtype("S5")).Should().Be("|S5");
            NDArrayPythonInterop.ToNumpyDtypeStr(np.dtype("U5")).Should().Be("<U5");
            NDArrayPythonInterop.ToNumpyDtypeStr(np.dtype("U1")).Should().Be("<U1");
            ((Action)(() => NDArrayPythonInterop.ToNumpyDtypeStr(np.dtypes.StringDType(null, coerce: false))))
                .Should().Throw<NotSupportedException>().WithMessage("*StringDType(na_object=None, coerce=False) has no buffer form*");
        }

        [TestMethod]
        public void FromBufferFormat_LongDouble_MapsByWidth()
        {
            // np.longdouble exports buffer format 'g' at EVERY width. On MSVC long double IS IEEE
            // double (itemsize 8) — bit-exact, so it views; extended-precision widths have no
            // NumSharp dtype and are refused with astype guidance.
            NDArrayPythonInterop.FromBufferFormat("g", 8).Should().Be(NPTypeCode.Double, "MSVC long double is IEEE double");
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("g", 16)))
                .Should().Throw<NotSupportedException>().WithMessage("*float64*", "x87/quad long double has no NumSharp dtype");
        }

        /// <summary>
        ///     The complex64 typestr refuses with the widening-copy guidance the buffer-format path gives, not the generic
        ///     "has no NumSharp dtype".
        /// </summary>
        [TestMethod]
        public void FromNumpyDtypeStr_Complex64_ThrowsWithConversionGuidance()
        {
            // The typestr twin of buffer 'Zf' — reachable through __array_interface__ when a NON-contiguous complex64 numpy
            // array is viewed — must carry the same guidance as the buffer-format path, not the generic "has no NumSharp
            // dtype". (Its text twins, '<U1' / '<U3', are real NumSharp dtypes now: TextTypestrs_MapBothWays.)
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr("<c8")))
                .Should().Throw<NotSupportedException>().WithMessage("*complex64*ToNDArray*");
        }
    }
}
