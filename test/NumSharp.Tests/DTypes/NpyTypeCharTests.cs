using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     NumPy's type characters: the <see cref="NPY_TYPECHAR"/> mirror of NumPy 2.4.2's header, the two mappings
    ///     <see cref="NPTypeCodeExtensions.ToTypeCode(NPY_TYPECHAR)"/> / <see cref="NPTypeCodeExtensions.ToTYPECHAR"/> and the
    ///     single-character dtype strings of <c>np.dtype</c>. Every expectation is NumPy 2.4.2's, probed on Windows (C long is
    ///     32-bit) and Linux (64-bit): <c>np.dtype(c)</c> for every printable ASCII character and <c>np.dtype(t).char</c> for
    ///     every builtin dtype. The full header comparison of the enum is <see cref="DTypeOracleTests.CEnums_MirrorNumPysHeader"/>.
    /// </summary>
    [TestClass]
    public class NpyTypeCharTests
    {
        /// <summary>Whether C <c>long</c> is 32-bit on this host (Windows, or any 32-bit process), the switch NumPy's 'l'/'L' follow.</summary>
        private static readonly bool CLongIs32 = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || IntPtr.Size != 8;

        /// <summary>
        ///     The characters NumPy 2.4.2 resolves to a dtype NumSharp stores, with that storage code on this host. Every other
        ///     printable ASCII character is refused: NumPy does not know it ("data type 'x' not understood"), or NumSharp has
        ///     no such dtype ('F' complex64, 'S'/'a'/'c' bytes, 'U' str, 'V' void, 'O' object, 'T' StringDType), or the dtype
        ///     has no storage yet ('M'/'m' datetimes).
        /// </summary>
        private static Dictionary<char, NPTypeCode> NumPyStorageCodes() => new()
        {
            ['?'] = NPTypeCode.Boolean,
            ['b'] = NPTypeCode.SByte,
            ['B'] = NPTypeCode.Byte,
            ['h'] = NPTypeCode.Int16,
            ['H'] = NPTypeCode.UInt16,
            ['i'] = NPTypeCode.Int32,
            ['I'] = NPTypeCode.UInt32,
            ['l'] = CLongIs32 ? NPTypeCode.Int32 : NPTypeCode.Int64,
            ['L'] = CLongIs32 ? NPTypeCode.UInt32 : NPTypeCode.UInt64,
            ['q'] = NPTypeCode.Int64,
            ['Q'] = NPTypeCode.UInt64,
            ['n'] = IntPtr.Size == 8 ? NPTypeCode.Int64 : NPTypeCode.Int32,
            ['N'] = IntPtr.Size == 8 ? NPTypeCode.UInt64 : NPTypeCode.UInt32,
            ['p'] = IntPtr.Size == 8 ? NPTypeCode.Int64 : NPTypeCode.Int32,
            ['P'] = IntPtr.Size == 8 ? NPTypeCode.UInt64 : NPTypeCode.UInt32,
            ['e'] = NPTypeCode.Half,
            ['f'] = NPTypeCode.Single,
            ['d'] = NPTypeCode.Double,
            ['g'] = NPTypeCode.Double,
            ['D'] = NPTypeCode.Complex,
            ['G'] = NPTypeCode.Complex,
        };

        /// <summary>
        ///     Every printable ASCII character maps to NumPy's dtype or is refused, with no third outcome: the old switch read
        ///     'D' as float64, 'F' as float32, Windows' 'l' as int64, 'S'/'U' as char, 'c' as complex, 'V' as Empty and the
        ///     kind letter 'u' as uint64, and threw for 'G', 'n' and 'N'.
        /// </summary>
        [TestMethod]
        public void ToTypeCode_EveryPrintableAsciiCharacter_IsNumPysDtype()
        {
            var expected = NumPyStorageCodes();
            var problems = new List<string>();
            for (char c = ' '; c <= '~'; c++)
            {
                NPTypeCode? got;
                try { got = ((NPY_TYPECHAR)c).ToTypeCode(); }
                catch (NotSupportedException) { got = null; }

                if (expected.TryGetValue(c, out var want))
                {
                    if (got != want)
                        problems.Add($"'{c}': expected {want}, got {(got?.ToString() ?? "a refusal")}");
                }
                else if (got != null)
                {
                    problems.Add($"'{c}': NumPy 2.4.2 gives no NumSharp dtype, got {got}");
                }
            }

            problems.Should().BeEmpty();
        }

        /// <summary>
        ///     <see cref="NPTypeCodeExtensions.ToTypeCode(NPY_TYPECHAR)"/> is the parser's answer, character for character:
        ///     the two can no longer disagree. Checked over the whole Latin-1 range, where most characters are refusals.
        /// </summary>
        [TestMethod]
        public void ToTypeCode_IsTheParsersAnswer_ForEveryLatin1Character()
        {
            for (int code = 0; code <= 0xFF; code++)
            {
                string parsed, mapped;
                try { parsed = np.dtype(((char)code).ToString()).GetTypeCode().ToString(); }
                catch (Exception e) { parsed = e.GetType().Name; }
                try { mapped = ((NPY_TYPECHAR)code).ToTypeCode().ToString(); }
                catch (Exception e) { mapped = e.GetType().Name; }
                mapped.Should().Be(parsed, $"character U+{code:X4}");
            }
        }

        /// <summary>
        ///     Being the parser's answer, a control character below '\x18' is a TYPE NUMBER, as in NumPy 2.4.2 (probed:
        ///     np.dtype('\x00') is bool, '\x0c' float64, '\x0f' complex128, '\x17' float16, '\x18' "not understood"), so a
        ///     value that is no type character can still resolve. The platform-free numbers only: '\x07' is C long.
        /// </summary>
        [TestMethod]
        public void ToTypeCode_ControlCharacterBelow0x18_IsATypeNumber()
        {
            ((NPY_TYPECHAR)0x00).ToTypeCode().Should().Be(NPTypeCode.Boolean);
            ((NPY_TYPECHAR)0x0C).ToTypeCode().Should().Be(NPTypeCode.Double);
            ((NPY_TYPECHAR)0x0F).ToTypeCode().Should().Be(NPTypeCode.Complex);
            ((NPY_TYPECHAR)0x17).ToTypeCode().Should().Be(NPTypeCode.Half);
            Action past = () => ((NPY_TYPECHAR)0x18).ToTypeCode();
            past.Should().Throw<NotSupportedException>().WithMessage("*not understood*");
        }

        /// <summary>
        ///     A kind letter is not a type character: 'u' is refused as NumPy refuses it, and the complex kind letter shares
        ///     the value of NumPy's one-byte string type 'c', so it is refused as bytes and never read as complex.
        /// </summary>
        [TestMethod]
        public void ToTypeCode_KindLetters_AreNotTypeCharacters()
        {
            Action unsigned = () => NPY_TYPECHAR.NPY_UNSIGNEDLTR.ToTypeCode();
            unsigned.Should().Throw<NotSupportedException>().WithMessage("*'u'*");

            Action complexKind = () => NPY_TYPECHAR.NPY_COMPLEXLTR.ToTypeCode();
            complexKind.Should().Throw<NotSupportedException>("'c' is S1 in NumPy, a dtype NumSharp has no storage for");

            NPY_TYPECHAR.NPY_CDOUBLELTR.ToTypeCode().Should().Be(NPTypeCode.Complex);
            NPY_TYPECHAR.NPY_GENBOOLLTR.ToTypeCode().Should().Be(NPTypeCode.SByte, "the bool kind letter 'b' is the int8 type character");
        }

        /// <summary>A value that is no UTF-16 character is refused instead of wrapping onto an unrelated character in the (char) cast.</summary>
        [TestMethod]
        public void ToTypeCode_ValueOutsideTheCharRange_Throws()
        {
            Action above = () => ((NPY_TYPECHAR)('d' + 0x10000)).ToTypeCode();
            above.Should().Throw<ArgumentOutOfRangeException>();
            Action negative = () => ((NPY_TYPECHAR)(-1)).ToTypeCode();
            negative.Should().Throw<ArgumentOutOfRangeException>();
        }

        /// <summary>
        ///     Every storage code with a NumPy dtype maps to NumPy's character for it on this host, which is its descriptor's
        ///     character: complex128 is 'D' (the old mapping gave the complex KIND letter 'c', which is NumPy's S1), the str
        ///     slot is 'U' (was 'S', bytes), and the 32- and 64-bit integers follow C long ('l'/'q' on Windows, 'i'/'l' on
        ///     Linux; the old mapping was Linux's on every host).
        /// </summary>
        [TestMethod]
        public void ToTYPECHAR_EveryNumPyDtype_IsNumPysCharacter()
        {
            var expected = new Dictionary<NPTypeCode, char>
            {
                [NPTypeCode.Boolean] = '?',
                [NPTypeCode.SByte] = 'b',
                [NPTypeCode.Byte] = 'B',
                [NPTypeCode.Int16] = 'h',
                [NPTypeCode.UInt16] = 'H',
                [NPTypeCode.Int32] = CLongIs32 ? 'l' : 'i',
                [NPTypeCode.UInt32] = CLongIs32 ? 'L' : 'I',
                [NPTypeCode.Int64] = CLongIs32 ? 'q' : 'l',
                [NPTypeCode.UInt64] = CLongIs32 ? 'Q' : 'L',
                [NPTypeCode.Half] = 'e',
                [NPTypeCode.Single] = 'f',
                [NPTypeCode.Double] = 'd',
                [NPTypeCode.Complex] = 'D',
                [NPTypeCode.String] = 'U',
            };
            foreach (var (code, ch) in expected)
            {
                ((char)code.ToTYPECHAR()).Should().Be(ch, code.ToString());
                ((char)code.ToTYPECHAR()).Should().Be(DType.From(code).@char, $"{code}: the descriptor's own character");
            }
        }

        /// <summary>
        ///     NumSharp's own dtypes have no NumPy character, so the mapping refuses them: it used to hand out decimal as 'q'
        ///     (C long long, so <c>find_common_type</c> searched from int64 for it) and char as 'c' (NumPy's S1). Empty is no
        ///     dtype at all.
        /// </summary>
        [TestMethod]
        public void ToTYPECHAR_NumSharpOnlyDtypes_AndEmpty_Refuse()
        {
            Action dec = () => NPTypeCode.Decimal.ToTYPECHAR();
            dec.Should().Throw<NotSupportedException>().WithMessage("*decimal dtype has no NumPy type character*");
            Action ch = () => NPTypeCode.Char.ToTYPECHAR();
            ch.Should().Throw<NotSupportedException>().WithMessage("*char dtype has no NumPy type character*");
            Action empty = () => NPTypeCode.Empty.ToTYPECHAR();
            empty.Should().Throw<ArgumentException>();
        }

        /// <summary>
        ///     Character and storage code round-trip for every NumPy dtype <c>np.dtype</c> can parse (the str slot is excluded:
        ///     NumSharp refuses <c>np.dtype('U')</c>). The old pair lost identity silently for NumSharp's own dtypes: decimal
        ///     came back as int64 (through 'q') and char as complex (through 'c'); they now refuse instead (see
        ///     <see cref="ToTYPECHAR_NumSharpOnlyDtypes_AndEmpty_Refuse"/>).
        /// </summary>
        [TestMethod]
        public void ToTypeCode_OfToTYPECHAR_RoundTrips()
        {
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
                NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
                NPTypeCode.Complex,
            };
            foreach (var code in codes)
                code.ToTYPECHAR().ToTypeCode().Should().Be(code);
        }

        /// <summary>
        ///     NumPy 2's 'n'/'N' (np.intp / np.uintp) parse to the int64/uint64 descriptor on a 64-bit host, whose character is
        ///     NumPy's for int64 there ('q' on Windows, 'l' on Linux): NumPy resolves them to the same descriptor.
        /// </summary>
        [TestMethod]
        public void Parser_n_N_AreIntpAndUintp()
        {
            np.dtype("n").Should().Be(np.intp);
            np.dtype("N").Should().Be(np.uintp);
            if (IntPtr.Size == 8)
            {
                np.dtype("n").@char.Should().Be(CLongIs32 ? 'q' : 'l');
                np.dtype("N").@char.Should().Be(CLongIs32 ? 'Q' : 'L');
            }
        }

        /// <summary>'T' is a dtype NumPy 2.x knows (StringDType), so NumSharp says it does not support it, never that it does not understand it.</summary>
        [TestMethod]
        public void Parser_T_IsNotSupported_RatherThanNotUnderstood()
        {
            Action t = () => np.dtype("T");
            t.Should().Throw<NotSupportedException>().WithMessage("*'T' is not supported by NumSharp*");
            Action unknown = () => np.dtype("x");
            unknown.Should().Throw<NotSupportedException>().WithMessage("*'x' not understood*");
        }

        /// <summary>
        ///     The members this fix changed, beside the full header comparison of the oracle: NumPy 2.4.2 has
        ///     <see cref="NPY_TYPECHAR.NPY_VSTRINGLTR"/> and spells the 'a' alias <see cref="NPY_TYPECHAR.NPY_DEPRECATED_STRINGLTR2"/>,
        ///     and removed <c>NPY_INTPLTR</c>/<c>NPY_UINTPLTR</c>.
        /// </summary>
        [TestMethod]
        public void Enum_HasNumPy2sMembers()
        {
            ((char)NPY_TYPECHAR.NPY_VSTRINGLTR).Should().Be('T');
            ((char)NPY_TYPECHAR.NPY_DEPRECATED_STRINGLTR2).Should().Be('a');
            var names = Enum.GetNames(typeof(NPY_TYPECHAR));
            names.Should().NotContain(new[] { "NPY_INTPLTR", "NPY_UINTPLTR", "NPY_STRINGLTR2" });
            Enum.IsDefined(typeof(NPY_TYPECHAR), (int)'p').Should().BeFalse("'p' is a valid character with no member since NumPy 2.0");
        }

        /// <summary>NumPy's name of the str_ dtype is "str"; NumPy has no dtype named "string".</summary>
        [TestMethod]
        public void AsNumpyDtypeName_String_IsStr()
        {
            NPTypeCode.String.AsNumpyDtypeName().Should().Be("str");
            DType.From(NPTypeCode.String).name.Should().Be("str");
        }
    }
}
