using System;
using System.Text;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Python's five attributes on NumSharp's Unicode errors (the owner's decision of 2026-11-10, "Both errors"):
    ///     <c>encoding</c>, <c>object</c>, <c>start</c>, <c>end</c> and <c>reason</c>, filled by every raise site and by
    ///     Python's five-argument constructors. Every expected value is NumPy 2.4.2's (CPython 3.12) answer to the same call,
    ///     probed with <c>except UnicodeDecodeError as e: e.encoding, e.object, e.start, e.end, e.reason</c>.
    /// </summary>
    /// <remarks>
    ///     The raise paths differ by codec and by route (the ASCII class conversions of <c>astype</c>, the codecs of
    ///     <c>np.strings.encode</c> / <c>decode</c>, the <c>StringDType</c> cast), so each route has a case: a fixed list of
    ///     factories builds the errors, and a factory that dropped one value would show here.
    /// </remarks>
    [TestClass]
    public class UnicodeErrorAttributesTests : TestClass
    {
        /// <summary>Asserts the five attributes and the message of a decode error.</summary>
        /// <param name="e">The caught error.</param>
        /// <param name="encoding">NumPy's <c>e.encoding</c>.</param>
        /// <param name="bytes">NumPy's <c>e.object</c>.</param>
        /// <param name="start">NumPy's <c>e.start</c>.</param>
        /// <param name="end">NumPy's <c>e.end</c>.</param>
        /// <param name="reason">NumPy's <c>e.reason</c>.</param>
        /// <param name="message">NumPy's <c>str(e)</c>.</param>
        private static void Check(UnicodeDecodeError e, string encoding, byte[] bytes, long start, long end, string reason, string message)
        {
            e.encoding.Should().Be(encoding);
            e.@object.Should().Equal(bytes);
            e.start.Should().Be(start);
            e.end.Should().Be(end);
            e.reason.Should().Be(reason);
            e.Message.Should().Be(message);
        }

        /// <summary>Asserts the five attributes and the message of an encode error.</summary>
        /// <param name="e">The caught error.</param>
        /// <param name="encoding">NumPy's <c>e.encoding</c>.</param>
        /// <param name="text">NumPy's <c>e.object</c>.</param>
        /// <param name="start">NumPy's <c>e.start</c> (code points).</param>
        /// <param name="end">NumPy's <c>e.end</c>.</param>
        /// <param name="reason">NumPy's <c>e.reason</c>.</param>
        /// <param name="message">NumPy's <c>str(e)</c>.</param>
        private static void Check(UnicodeEncodeError e, string encoding, string text, long start, long end, string reason, string message)
        {
            e.encoding.Should().Be(encoding);
            e.@object.Should().Be(text);
            e.start.Should().Be(start);
            e.end.Should().Be(end);
            e.reason.Should().Be(reason);
            e.Message.Should().Be(message);
        }

        // ---- the class conversions (ASCII) ------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.array([b"a\xffz"], dtype="S5").astype("U5")</c>: <c>UnicodeDecodeError('ascii', b'a\xffz', 1, 2,
        ///     'ordinal not in range(128)')</c> - the object is the element WITHOUT its NUL padding, as NumPy reads it.
        /// </summary>
        [TestMethod]
        public void BytesToUnicode_NamesTheElementWithoutItsPadding()
        {
            using var s = np.array(new object[] { new byte[] { 0x61, 0xFF, 0x7A } }, dtype: "S5");
            var e = new Action(() => s.astype("U5")).Should().Throw<UnicodeDecodeError>().Which;
            Check(e, "ascii", new byte[] { 0x61, 0xFF, 0x7A }, 1, 2, "ordinal not in range(128)",
                "'ascii' codec can't decode byte 0xff in position 1: ordinal not in range(128)");
        }

        /// <summary><c>np.array(["a\xe9"], dtype="U3").astype("S3")</c>: <c>('ascii', 'a\xe9', 1, 2, ...)</c>.</summary>
        [TestMethod]
        public void UnicodeToBytes_NamesTheText()
        {
            using var u = np.array(new[] { "aé" }, dtype: "U3");
            var e = new Action(() => u.astype("S3")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "ascii", "aé", 1, 2, "ordinal not in range(128)",
                "'ascii' codec can't encode character '\\xe9' in position 1: ordinal not in range(128)");
        }

        /// <summary>
        ///     <c>np.array(["a\U0001F600b"], dtype="U4").astype("S4")</c>: <c>('ascii', 'a\U0001f600b', 1, 2, ...)</c> - the
        ///     positions count code points, so the astral character is ONE position though the C# text holds it as two chars.
        /// </summary>
        [TestMethod]
        public void UnicodeToBytes_AstralCharacter_CountsOnePosition()
        {
            using var u = np.array(new[] { "a\U0001F600b" }, dtype: "U4");
            var e = new Action(() => u.astype("S4")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "ascii", "a\U0001F600b", 1, 2, "ordinal not in range(128)",
                "'ascii' codec can't encode character '\\U0001f600' in position 1: ordinal not in range(128)");
        }

        /// <summary><c>np.array(["ab\xe9\xe8c"], dtype="U6").astype("S6")</c>: a run, <c>('ascii', ..., 2, 4, ...)</c>.</summary>
        [TestMethod]
        public void UnicodeToBytes_Run_ReportsItsRange()
        {
            using var u = np.array(new[] { "abéèc" }, dtype: "U6");
            var e = new Action(() => u.astype("S6")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "ascii", "abéèc", 2, 4, "ordinal not in range(128)",
                "'ascii' codec can't encode characters in position 2-3: ordinal not in range(128)");
        }

        /// <summary>
        ///     <c>np.array(["a\xe9"], dtype=StringDType()).astype("S3")</c>: <c>('ascii', 'a\xe9', 1, 2, ...)</c>, the element's
        ///     whole text.
        /// </summary>
        [TestMethod]
        public void StringDTypeToBytes_NamesTheText()
        {
            using var t = np.array(new[] { "aé" }, np.dtypes.StringDType());
            var e = new Action(() => t.astype("S3")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "ascii", "aé", 1, 2, "ordinal not in range(128)",
                "'ascii' codec can't encode character '\\xe9' in position 1: ordinal not in range(128)");
        }

        // ---- np.strings.decode ------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.strings.decode(np.array([b"ok", b"\xff\xfe"]), "utf-8")</c>: <c>('utf-8', b'\xff\xfe', 0, 1, 'invalid start
        ///     byte')</c> - the failing element, not the array.
        /// </summary>
        [TestMethod]
        public void DecodeUtf8_NamesTheFailingElement()
        {
            using var s = np.array(new object[] { Encoding.ASCII.GetBytes("ok"), new byte[] { 0xFF, 0xFE } }, dtype: "S2");
            var e = new Action(() => np.strings.decode(s, "utf-8")).Should().Throw<UnicodeDecodeError>().Which;
            Check(e, "utf-8", new byte[] { 0xFF, 0xFE }, 0, 1, "invalid start byte",
                "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte");
        }

        /// <summary><c>np.strings.decode(np.array([b"ab\x80"]), "ascii")</c>: <c>('ascii', b'ab\x80', 2, 3, ...)</c>.</summary>
        [TestMethod]
        public void DecodeAscii_NamesTheElement()
        {
            using var s = np.array(new object[] { new byte[] { 0x61, 0x62, 0x80 } }, dtype: "S3");
            var e = new Action(() => np.strings.decode(s, "ascii")).Should().Throw<UnicodeDecodeError>().Which;
            Check(e, "ascii", new byte[] { 0x61, 0x62, 0x80 }, 2, 3, "ordinal not in range(128)",
                "'ascii' codec can't decode byte 0x80 in position 2: ordinal not in range(128)");
        }

        /// <summary><c>np.strings.decode(np.array([b"\x81"]), "cp1252")</c>: <c>('charmap', b'\x81', 0, 1, ...)</c>.</summary>
        [TestMethod]
        public void DecodeCharmap_NamesTheElement()
        {
            using var s = np.array(new object[] { new byte[] { 0x81 } }, dtype: "S1");
            var e = new Action(() => np.strings.decode(s, "cp1252")).Should().Throw<UnicodeDecodeError>().Which;
            Check(e, "charmap", new byte[] { 0x81 }, 0, 1, "character maps to <undefined>",
                "'charmap' codec can't decode byte 0x81 in position 0: character maps to <undefined>");
        }

        /// <summary><c>np.strings.decode(np.array([b"a"]), "utf-16-le")</c>: <c>('utf-16-le', b'a', 0, 1, 'truncated data')</c>.</summary>
        [TestMethod]
        public void DecodeUtf16_TruncatedData()
        {
            using var s = np.array(new object[] { new byte[] { 0x61 } }, dtype: "S1");
            var e = new Action(() => np.strings.decode(s, "utf-16-le")).Should().Throw<UnicodeDecodeError>().Which;
            Check(e, "utf-16-le", new byte[] { 0x61 }, 0, 1, "truncated data",
                "'utf-16-le' codec can't decode byte 0x61 in position 0: truncated data");
        }

        // ---- np.strings.encode ------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.strings.encode(np.array(["ok", "a\ud800b"]), "utf-8")</c>: <c>('utf-8', 'a\ud800b', 1, 2, 'surrogates not
        ///     allowed')</c>.
        /// </summary>
        [TestMethod]
        public void EncodeUtf8_LoneSurrogate()
        {
            using var u = np.array(new[] { "ok", "a\ud800b" });
            var e = new Action(() => np.strings.encode(u, "utf-8")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "utf-8", "a\ud800b", 1, 2, "surrogates not allowed",
                "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed");
        }

        /// <summary><c>np.strings.encode(np.array(["中文"]), "latin-1")</c>: <c>('latin-1', '中文', 0, 2, 'ordinal not in range(256)')</c>.</summary>
        [TestMethod]
        public void EncodeLatin1_Run()
        {
            using var u = np.array(new[] { "中文" });
            var e = new Action(() => np.strings.encode(u, "latin-1")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "latin-1", "中文", 0, 2, "ordinal not in range(256)",
                "'latin-1' codec can't encode characters in position 0-1: ordinal not in range(256)");
        }

        /// <summary><c>np.strings.encode(np.array(["a€"]), "cp437")</c>: <c>('charmap', 'a€', 1, 2, 'character maps to &lt;undefined&gt;')</c>.</summary>
        [TestMethod]
        public void EncodeCharmap_Undefined()
        {
            using var u = np.array(new[] { "a€" });
            var e = new Action(() => np.strings.encode(u, "cp437")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "charmap", "a€", 1, 2, "character maps to <undefined>",
                "'charmap' codec can't encode character '\\u20ac' in position 1: character maps to <undefined>");
        }

        /// <summary><c>np.strings.encode(np.array(["x\udc00"]), "utf-16")</c>: <c>('utf-16', 'x\udc00', 1, 2, 'surrogates not allowed')</c>.</summary>
        [TestMethod]
        public void EncodeUtf16_LoneSurrogate()
        {
            using var u = np.array(new[] { "x\udc00" });
            var e = new Action(() => np.strings.encode(u, "utf-16")).Should().Throw<UnicodeEncodeError>().Which;
            Check(e, "utf-16", "x\udc00", 1, 2, "surrogates not allowed",
                "'utf-16' codec can't encode character '\\udc00' in position 1: surrogates not allowed");
        }

        // ---- Python's constructors --------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>str(UnicodeDecodeError(...))</c>: one byte inside the data quotes it, any other range (a longer one, or a
        ///     start past the data's end) prints the range; <c>repr()</c> is CPython's.
        /// </summary>
        [TestMethod]
        public void DecodeConstructor_FormatsCPythonsText()
        {
            var one = new UnicodeDecodeError("utf-8", new byte[] { 0xFF }, 0, 1, "invalid start byte");
            one.Message.Should().Be("'utf-8' codec can't decode byte 0xff in position 0: invalid start byte");
            one.PythonRepr().Should().Be("UnicodeDecodeError('utf-8', b'\\xff', 0, 1, 'invalid start byte')");
            new UnicodeDecodeError("utf-8", Encoding.ASCII.GetBytes("ab"), 0, 2, "x").Message
                .Should().Be("'utf-8' codec can't decode bytes in position 0-1: x");
            new UnicodeDecodeError("utf-8", Encoding.ASCII.GetBytes("a"), 5, 6, "x").Message
                .Should().Be("'utf-8' codec can't decode bytes in position 5-5: x");
        }

        /// <summary>
        ///     <c>str(UnicodeEncodeError(...))</c>: one character quotes it as <c>\xNN</c> / <c>\uNNNN</c> / <c>\UNNNNNNNN</c> by
        ///     its size, a run or a start past the text's end prints the range.
        /// </summary>
        [TestMethod]
        public void EncodeConstructor_FormatsCPythonsText()
        {
            new UnicodeEncodeError("ascii", "aé", 1, 2, "r").Message.Should().Be("'ascii' codec can't encode character '\\xe9' in position 1: r");
            new UnicodeEncodeError("ascii", "a中", 1, 2, "r").Message.Should().Be("'ascii' codec can't encode character '\\u4e2d' in position 1: r");
            new UnicodeEncodeError("ascii", "a\U0001F600", 1, 2, "r").Message.Should().Be("'ascii' codec can't encode character '\\U0001f600' in position 1: r");
            new UnicodeEncodeError("ascii", "abc", 0, 3, "r").Message.Should().Be("'ascii' codec can't encode characters in position 0-2: r");
            new UnicodeEncodeError("ascii", "a", 4, 5, "r").Message.Should().Be("'ascii' codec can't encode characters in position 4-4: r");
        }

        /// <summary>The constructors refuse a null text with CPython's TypeError texts, argument by argument.</summary>
        [TestMethod]
        public void Constructors_RefuseNullsLikeCPython()
        {
            new Action(() => _ = new UnicodeDecodeError("utf-8", null, 0, 1, "x")).Should().Throw<TypeError>()
                .WithMessage("a bytes-like object is required, not 'NoneType'");
            new Action(() => _ = new UnicodeDecodeError(null, new byte[1], 0, 1, "x")).Should().Throw<TypeError>()
                .WithMessage("argument 1 must be str, not None");
            new Action(() => _ = new UnicodeDecodeError("utf-8", new byte[1], 0, 1, null)).Should().Throw<TypeError>()
                .WithMessage("argument 5 must be str, not None");
            new Action(() => _ = new UnicodeEncodeError("utf-8", null, 0, 1, "x")).Should().Throw<TypeError>()
                .WithMessage("argument 2 must be str, not None");
            new Action(() => _ = new UnicodeEncodeError(null, "a", 0, 1, "x")).Should().Throw<TypeError>()
                .WithMessage("argument 1 must be str, not None");
            new Action(() => _ = new UnicodeEncodeError("utf-8", "a", 0, 1, null)).Should().Throw<TypeError>()
                .WithMessage("argument 5 must be str, not None");
        }

        /// <summary>The message-only constructors leave the attributes empty (null texts, zero positions).</summary>
        [TestMethod]
        public void MessageOnlyConstructors_LeaveTheAttributesEmpty()
        {
            var d = new UnicodeDecodeError("text");
            d.encoding.Should().BeNull();
            d.@object.Should().BeNull();
            d.reason.Should().BeNull();
            (d.start, d.end).Should().Be((0L, 0L));
            d.PythonRepr().Should().Be("UnicodeDecodeError('text')");
            var e = new UnicodeEncodeError("text");
            e.encoding.Should().BeNull();
            e.@object.Should().BeNull();
            e.reason.Should().BeNull();
            (e.start, e.end).Should().Be((0L, 0L));
        }
    }
}
