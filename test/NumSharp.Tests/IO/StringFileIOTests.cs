using System;
using System.IO;
using System.Text;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     File and text I/O of the fixed-width string dtypes (<c>S</c> / <c>U</c>) against NumPy 2.4.2 (every expected value
    ///     probed on 2026-10-07): <c>ndarray.tofile</c> (binary and text), <c>np.fromfile</c>, <c>np.fromstring</c>,
    ///     <c>np.loadtxt</c> and <c>np.savetxt</c>. These go through files and text writers, which the string oracle's
    ///     operand corpus cannot express, so they are pinned here.
    /// </summary>
    [TestClass]
    public class StringFileIOTests
    {
        /// <summary>A fresh temp file path, deleted by the caller's finally.</summary>
        /// <returns>The path.</returns>
        private static string TempFile() => Path.Combine(Path.GetTempPath(), "ns_strio_" + Guid.NewGuid().ToString("N"));

        /// <summary>Writes <paramref name="a"/> with <c>tofile</c> and returns the file's bytes as Latin-1 text.</summary>
        /// <param name="a">The array.</param>
        /// <param name="sep">The separator ("" for binary).</param>
        /// <param name="format">The text format.</param>
        /// <returns>The file content, one char per byte.</returns>
        private static string ToFileText(NDArray a, string sep, string format = "%s")
        {
            var path = TempFile();
            try
            {
                a.tofile(path, sep, format);
                return Encoding.Latin1.GetString(File.ReadAllBytes(path));
            }
            finally
            {
                File.Delete(path);
            }
        }

        // ---- ndarray.tofile -----------------------------------------------------------------------------------------

        /// <summary>
        ///     Guards that a text <c>tofile</c> writes <c>str()</c> of the Python object <c>getitem</c> returns: a bytes
        ///     value prints as <c>b'ab'</c> (not the <c>np.bytes_(b'ab')</c> a 0-d array prints), a str value as its text.
        /// </summary>
        [TestMethod]
        public void ToFile_Text_WritesPythonStr()
        {
            ToFileText(np.array(new[] { "ab", "c" }, dtype: "S2"), ",").Should().Be("b'ab',b'c'");
            ToFileText(np.array(new[] { "ab", "c" }), ",").Should().Be("ab,c");
            ToFileText(np.array(new object[] { new byte[] { (byte)'a', (byte)'b' }, new byte[] { 0xFF } }, dtype: "S2"), ",")
                .Should().Be("b'ab',b'\\xff'");
            ToFileText(np.array(new[,] { { "ab", "c" }, { "d", "e" } }, dtype: "S2").T, "|")
                .Should().Be("b'ab'|b'd'|b'c'|b'e'", "the logical C order of a transposed view");
            ToFileText(np.array("xy"), ",").Should().Be("xy");
            ToFileText(np.zeros(new Shape(0), "U2"), ",").Should().Be("");
        }

        /// <summary>Guards <c>format % item</c> for text: <c>%s</c>, <c>%r</c> of the Python object, widths and padding.</summary>
        [TestMethod]
        public void ToFile_Text_FormatConversions()
        {
            ToFileText(np.array(new[] { "ab", "c" }), ",", "%s!").Should().Be("ab!,c!");
            ToFileText(np.array(new[] { "ab", "c" }, dtype: "S2"), ",", "%r").Should().Be("b'ab',b'c'");
            ToFileText(np.array(new[] { "ab", "c" }, dtype: "S2"), ",", "%5s").Should().Be("b'ab', b'c'");
            ToFileText(np.array(new[] { "ab", "c" }), ",", "%r").Should().Be("'ab','c'");
            ToFileText(np.array(new[] { "ab", "c" }), ",", "%-3s|").Should().Be("ab |,c  |");
            ToFileText(np.array(new[] { "ab", "c" }), ",", "%.1s").Should().Be("a,c");

            Action numeric = () => ToFileText(np.array(new[] { "ab", "c" }), ",", "%d");
            numeric.Should().Throw<TypeError>().WithMessage("%d format: a real number is required, not str");
            Action hex = () => ToFileText(np.array(new[] { "ab" }, dtype: "S2"), ",", "%x");
            hex.Should().Throw<TypeError>().WithMessage("%x format: an integer is required, not bytes");
            Action fixedPoint = () => ToFileText(np.array(new[] { "ab" }), ",", "%f");
            fixedPoint.Should().Throw<TypeError>().WithMessage("must be real number, not str");
        }

        /// <summary>
        ///     Pins a text <c>tofile</c> of non-ASCII text: NumPy 2.4.2 calls <c>PyUnicode_AsASCIIString</c> without a NULL
        ///     check and CRASHES the process; NumSharp raises the <c>UnicodeEncodeError</c> CPython's call reports.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void ToFile_Text_NonAscii_RaisesInsteadOfCrashing()
        {
            Action act = () => ToFileText(np.array(new[] { "ab", "dé" }), ",");
            act.Should().Throw<UnicodeEncodeError>()
                .WithMessage("'ascii' codec can't encode character '\\xe9' in position 1: ordinal not in range(128)");
        }

        /// <summary>Guards a binary <c>tofile</c>: the raw element bytes in logical C order (UCS-4 units for U).</summary>
        [TestMethod]
        public void ToFile_Binary_WritesRawElements()
        {
            // 'ab', 'c' as <U2: two UCS-4 units per element, NUL-padded.
            Convert.ToHexString(Encoding.Latin1.GetBytes(ToFileText(np.array(new[] { "ab", "c" }), "")))
                .Should().Be("61000000620000006300000000000000");
            // The transpose's logical C order: 'ab', 'd', 'c', 'e' (NumPy's bytes, probed).
            Convert.ToHexString(Encoding.Latin1.GetBytes(ToFileText(np.array(new[,] { { "ab", "c" }, { "d", "e" } }).T, "")))
                .Should().Be("6100000062000000" + "6400000000000000" + "6300000000000000" + "6500000000000000");
        }

        // ---- np.fromfile ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Guards a binary <c>np.fromfile</c> with a string dtype: whole elements read from the bytes, a trailing partial
        ///     element dropped, count and offset honored, the result owning its data.
        /// </summary>
        [TestMethod]
        public void FromFile_Binary_ReadsWholeElements()
        {
            var path = TempFile();
            try
            {
                File.WriteAllBytes(path, Encoding.ASCII.GetBytes("abcdefg"));
                using var all = np.fromfile(path, np.dtype("S3"));
                all.ToString(true).Should().Be("array([b'abc', b'def'], dtype='|S3')");
                all.flags.owndata.Should().BeTrue();
                np.fromfile(path, np.dtype("S3"), count: 1).ToString(true).Should().Be("array([b'abc'], dtype='|S3')");
                np.fromfile(path, np.dtype("S3"), count: 5).ToString(true).Should().Be("array([b'abc', b'def'], dtype='|S3')");
                np.fromfile(path, np.dtype("S3"), offset: 2).ToString(true).Should().Be("array([b'cde'], dtype='|S3')");

                np.array(new[] { "ab", "c" }).tofile(path);
                np.fromfile(path, np.dtype("<U2")).ToString(true).Should().Be("array(['ab', 'c'], dtype='<U2')");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>Guards NumPy's refusals: an unsized class (count -1), and a text file of a string dtype.</summary>
        [TestMethod]
        public void FromFile_Refusals_AreNumPys()
        {
            var path = TempFile();
            try
            {
                File.WriteAllBytes(path, Encoding.ASCII.GetBytes("a,b"));
                Action unsized = () => np.fromfile(path, np.dtype("S"));
                unsized.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
                Action text = () => np.fromfile(path, np.dtype("S1"), sep: ",");
                text.Should().Throw<ValueError>().WithMessage("Unable to read character files of that array type");
                Action offsetText = () => np.fromfile(path, np.dtype("S1"), sep: ",", offset: 1);
                offsetText.Should().Throw<TypeError>().WithMessage("'offset' argument only permitted for binary files");
            }
            finally
            {
                File.Delete(path);
            }
        }

        // ---- np.fromstring -------------------------------------------------------------------------------------------

        /// <summary>Guards np.fromstring's refusals for string dtypes, in NumPy's order.</summary>
        [TestMethod]
        public void FromString_StringDtype_IsRefusedWithNumPysTexts()
        {
            Action binary = () => np.fromstring("ab", np.dtype("S1"));
            binary.Should().Throw<ValueError>().WithMessage("The binary mode of fromstring is removed, use frombuffer instead");
            Action unsized = () => np.fromstring("a b", np.dtype("S"), sep: " ");
            unsized.Should().Throw<ValueError>().WithMessage("zero-valued itemsize");
            Action text = () => np.fromstring("a b", np.dtype("U1"), sep: " ");
            text.Should().Throw<ValueError>().WithMessage("don't know how to read character strings with that array type");
        }

        // ---- np.loadtxt ----------------------------------------------------------------------------------------------

        /// <summary>
        ///     Guards np.loadtxt with string dtypes: fields stored as text (truncated to a sized width, an unsized class taking
        ///     the longest field), <c>S</c> fields encoded as Latin-1.
        /// </summary>
        [TestMethod]
        public void LoadTxt_StringDtypes_StoreTheFields()
        {
            np.loadtxt(new StringReader("ab cd\nef gh\n"), np.dtype("U2")).ToString(true)
                .Should().Be("array([['ab', 'cd'],\n       ['ef', 'gh']], dtype='<U2')");
            np.loadtxt(new StringReader("ab cd\nef gh\n"), np.dtype("U1")).ToString(true)
                .Should().Be("array([['a', 'c'],\n       ['e', 'g']], dtype='<U1')");
            np.loadtxt(new StringReader("ab cd\nef gh\n"), np.dtype("S2")).ToString(true)
                .Should().Be("array([[b'ab', b'cd'],\n       [b'ef', b'gh']], dtype='|S2')");
            np.loadtxt(new StringReader("é x\n"), np.dtype("S2")).ToString(true)
                .Should().Be("array([b'\\xe9', b'x'], dtype='|S2')");
            np.loadtxt(new StringReader("ab cde\n"), np.dtype("U")).ToString(true)
                .Should().Be("array(['ab', 'cde'], dtype='<U3')");
            np.loadtxt(new StringReader("ab cde\n"), np.dtype("S")).ToString(true)
                .Should().Be("array([b'ab', b'cde'], dtype='|S3')");
            np.loadtxt(new StringReader("\"a b\" c\n"), np.dtype("U3"), quotechar: "\"").ToString(true)
                .Should().Be("array(['a b', 'c'], dtype='<U3')");
            np.loadtxt(new StringReader("a,,c\n"), np.dtype("U3"), delimiter: ",").ToString(true)
                .Should().Be("array(['a', '', 'c'], dtype='<U3')");
            np.loadtxt(new StringReader("a b c\n"), np.dtype("U3"), usecols: new[] { 2, 0 }).ToString(true)
                .Should().Be("array(['c', 'a'], dtype='<U3')");
            np.loadtxt(new StringReader("a b\n"), np.dtype("U3"), converters: (Func<string, object>)(s => s.ToUpperInvariant())).ToString(true)
                .Should().Be("array(['A', 'B'], dtype='<U3')");
            np.loadtxt(new StringReader("a\n"), np.dtype("U3")).ToString(true).Should().Be("array('a', dtype='<U3')");
            np.loadtxt(new StringReader(""), np.dtype("U3")).ToString(true).Should().Be("array([], dtype='<U3')");
        }

        /// <summary>Guards the <c>S</c> refusal of a field Latin-1 cannot encode, with NumPy's text.</summary>
        [TestMethod]
        public void LoadTxt_BytesField_PastLatin1_IsRefused()
        {
            Action act = () => np.loadtxt(new StringReader("中 x\n"), np.dtype("S2"));
            act.Should().Throw<ValueError>().WithMessage("could not convert string '中' to |S2 at row 0, column 1.");
        }

        // ---- np.savetxt ----------------------------------------------------------------------------------------------

        /// <summary>
        ///     Guards np.savetxt's <c>format % tuple(row)</c> over NumPy SCALARS: <c>%s</c> of an np.bytes_ is <c>b'ab'</c>,
        ///     <c>%r</c> of an np.str_ is <c>np.str_('ab')</c>, and a numeric conversion is NumPy's mismatch TypeError
        ///     naming <c>str(X.dtype)</c>.
        /// </summary>
        [TestMethod]
        public void SaveTxt_StringRows_FormatNumPyScalars()
        {
            var w = new StringWriter();
            np.savetxt(w, np.array(new[] { "ab", "c" }, dtype: "S2"), fmt: "%s");
            w.ToString().Should().Be("b'ab'\nb'c'\n");

            w = new StringWriter();
            np.savetxt(w, np.array(new[,] { { "ab", "c" } }), fmt: "%r");
            w.ToString().Should().Be("np.str_('ab') np.str_('c')\n");

            Action mismatch = () => np.savetxt(new StringWriter(), np.array(new[,] { { "ab", "c" } }), fmt: "%d");
            mismatch.Should().Throw<TypeError>().WithMessage("Mismatch between array dtype ('<U2') and format specifier ('%d %d')");
        }
    }
}
