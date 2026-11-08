using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Printing;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     CPython 3.12's <c>%</c> operator as <c>np.savetxt</c> and <c>ndarray.tofile</c> apply it
    ///     (<see cref="PercentFormat"/> + <see cref="PrintfFormatter"/>), against CPython 3.12.12 / NumPy 2.4.2 (every
    ///     expected value probed on 2026-10-15): the parse and the order of its errors, <c>*</c> arguments, NumPy scalars
    ///     versus Python objects, CPython's overflow quirks, and the two writers' own rules.
    /// </summary>
    /// <remarks>
    ///     The string oracle's section tio replays the NumPy cases of both writers over every NumPy dtype (the format matrix
    ///     of 2026-10-15); these tests pin what a corpus cannot hold: the exact exception TYPES (the wrapped and the raw
    ///     <see cref="TypeError"/>, <see cref="OutOfMemoryException"/> for CPython's <c>MemoryError</c>), the parse
    ///     driven directly, NumSharp's <c>Char</c> / <c>Decimal</c> elements (no NumPy dtype), the case where NumPy
    ///     crashes, and the file <c>tofile</c> must not create.
    /// </remarks>
    [TestClass]
    public class PercentFormatTests
    {
        /// <summary><c>format % args</c> over Python objects (<see cref="PercentFormat.PyObject"/>).</summary>
        /// <param name="format">The format.</param>
        /// <param name="args">The tuple's items.</param>
        /// <returns>The text.</returns>
        private static string Py(string format, params object[] args)
            => Run(format, args, Enumerable.Repeat(PercentFormat.PyObject, args.Length).ToArray());

        /// <summary><c>format % args</c> with explicit kinds.</summary>
        /// <param name="format">The format.</param>
        /// <param name="args">The tuple's items.</param>
        /// <param name="kinds">Their kinds.</param>
        /// <returns>The text.</returns>
        private static string Run(string format, object[] args, NPTypeCode[] kinds)
        {
            var sb = new StringBuilder();
            PrintfFormatter.Compile(format).FormatInto(sb, args, kinds);
            return sb.ToString();
        }

        /// <summary>Asserts the exact exception type and its message.</summary>
        /// <typeparam name="T">The exception type (exactly).</typeparam>
        /// <param name="f">The call.</param>
        /// <param name="msg">CPython's message.</param>
        /// <returns>The exception.</returns>
        private static T Throws<T>(Action f, string msg) where T : Exception
        {
            var e = Assert.ThrowsException<T>(f);
            Assert.AreEqual(msg, e.Message);
            return e;
        }

        /// <summary>ndarray.tofile's text output to a stream, decoded as Latin-1 (the elements are ASCII).</summary>
        /// <param name="a">The array.</param>
        /// <param name="format">The format.</param>
        /// <param name="sep">The separator.</param>
        /// <returns>The text written.</returns>
        private static string ToFile(NDArray a, string format, string sep = ",")
        {
            using var ms = new MemoryStream();
            a.tofile(ms, sep, format);
            return Encoding.Latin1.GetString(ms.ToArray());
        }

        /// <summary>np.savetxt's text output to a writer.</summary>
        /// <param name="a">The array.</param>
        /// <param name="fmt">The format.</param>
        /// <returns>The text written.</returns>
        private static string SaveTxt(NDArray a, string fmt)
        {
            var sw = new StringWriter();
            np.savetxt(sw, a, fmt);
            return sw.ToString();
        }

        // ---- the parse ----------------------------------------------------------------------------------------------

        /// <summary>
        ///     CPython parses as it formats: the argument is taken before the conversion character is checked, and each
        ///     parse error raises where the parser meets it, with CPython's text and code-point index.
        /// </summary>
        [TestMethod]
        public void Parse_ErrorsRaiseWhereCPythonMeetsThem()
        {
            // The second spec has no argument: that error comes before 'y' is found unsupported.
            Throws<PrintfArgumentException>(() => Py("%d %y", 1), "not enough arguments for format string");
            Throws<PrintfArgumentException>(() => Py("%y"), "not enough arguments for format string");
            Throws<ValueError>(() => Py("%y", 1), "unsupported format character 'y' (0x79) at index 1");
            // ONE length modifier is skipped; a second is the conversion character.
            Assert.AreEqual("1", Py("%ld", 1));
            Throws<ValueError>(() => Py("%lld", 1), "unsupported format character 'l' (0x6c) at index 2");
            Throws<ValueError>(() => Py("%5.2l", 1), "incomplete format");
            Throws<ValueError>(() => Py("%", 1), "incomplete format");
            Throws<ValueError>(() => Py("% ", 1), "incomplete format"); // ' ' is a flag
            // A literal percent takes nothing; a leftover argument is CPython's last check.
            Assert.AreEqual("%", Py("%%"));
            Throws<PrintfArgumentException>(() => Py("%%", 1), "not all arguments converted during string formatting");
            // The arguments are a tuple, never a mapping.
            Throws<PrintfArgumentException>(() => Py("%(k)s", 1), "format requires a mapping");
            Throws<ValueError>(() => Py("%-(k)s", 1), "unsupported format character '(' (0x28) at index 2");
            // The character is shown only when printable ASCII; the index counts code points (a surrogate pair is one).
            Throws<ValueError>(() => Py("%\U0001F600", 1), "unsupported format character '?' (0x1f600) at index 1");
            Throws<ValueError>(() => Py("\U0001F600%y", 1), "unsupported format character 'y' (0x79) at index 2");
            Throws<ValueError>(() => Py("%\0", 1), "unsupported format character '?' (0x0) at index 1");
            // CPython's "printable" test is 31 <= ch <= 126: U+001F is shown as itself, U+001E and U+007F are not.
            Throws<ValueError>(() => Py("%\u001f", 1), "unsupported format character '\u001f' (0x1f) at index 1");
            Throws<ValueError>(() => Py("%\u001e", 1), "unsupported format character '?' (0x1e) at index 1");
            Throws<ValueError>(() => Py("%~", 1), "unsupported format character '~' (0x7e) at index 1");
            Throws<ValueError>(() => Py("%\u007f", 1), "unsupported format character '?' (0x7f) at index 1");
            Throws<ValueError>(() => Py("%é", 1), "unsupported format character '?' (0xe9) at index 1");
            // Digits are checked as they accumulate: past PY_SSIZE_T_MAX / INT_MAX.
            Throws<ValueError>(() => Py("%9223372036854775808d", 1), "width too big");
            Throws<ValueError>(() => Py("%.2147483648d", 1), "precision too big");
            Throws<PrintfArgumentException>(() => Py("%9223372036854775807d"), "not enough arguments for format string");
            // A '*' width's argument is taken before the precision's digits are refused.
            Throws<PrintfArgumentException>(() => Py("%*.99999999999d", "x", 1), "* wants int");
            Throws<ValueError>(() => Py("%*.99999999999d", 5, 1), "precision too big");
        }

        /// <summary>
        ///     A <c>*</c> width or precision takes the next argument, which must be a Python <c>int</c> (a <c>bool</c> is
        ///     one, no NumPy scalar is); a negative width left-justifies (the most negative one pads nothing, as CPython's
        ///     negation overflows), a negative precision is 0.
        /// </summary>
        [TestMethod]
        public void StarArguments_TakePythonIntsOnly()
        {
            Assert.AreEqual("    3", Py("%*d", 5, 3));
            Assert.AreEqual("3    |", Py("%*d|", -5, 3));
            Assert.AreEqual("3", Py("%*d", true, 3));
            Assert.AreEqual("1", Py("%*d", long.MinValue, 1));
            Assert.AreEqual("2", Py("%.*f", -3, 1.5));
            Assert.AreEqual("0004", Py("%.*d", 4, 4));
            Throws<OverflowException>(() => Py("%*d", BigInteger.Pow(2, 63), 3), "Python int too large to convert to C ssize_t");
            Throws<OverflowException>(() => Py("%.*d", 1L << 31, 3), "Python int too large to convert to C int");
            Throws<PrintfArgumentException>(() => Py("%*d", 5.0, 3), "* wants int");
            Throws<PrintfArgumentException>(() => Py("%*d", "5", 3), "* wants int");
            Throws<PrintfArgumentException>(() => Py("%*d", null, 3), "* wants int");
            // np.int64 is not an int subclass.
            Throws<PrintfArgumentException>(() => Run("%*d", new object[] { 5L, 3L }, new[] { NPTypeCode.Int64, NPTypeCode.Int64 }),
                "* wants int");
            Throws<ValueError>(() => Py("%*", 5), "incomplete format");
            Throws<PrintfArgumentException>(() => Py("%*", 5.0), "* wants int");
        }

        /// <summary>
        ///     NumPy scalars (savetxt's <c>tuple(row)</c>): <c>repr</c> names the type, a <c>TypeError</c> quotes it, an
        ///     <c>np.bool</c> is not an integer, an <c>np.uint64</c> is rounded once to a double.
        /// </summary>
        [TestMethod]
        public void NumPyScalars_ReprAndTypeNames()
        {
            string S(string f, object v, NPTypeCode tc) => Run(f, new[] { v }, new[] { tc });
            Assert.AreEqual("np.float32(0.1)", S("%r", 0.1f, NPTypeCode.Single));
            Assert.AreEqual("np.True_", S("%r", true, NPTypeCode.Boolean));
            Assert.AreEqual("np.uint8(200)", S("%r", (byte)200, NPTypeCode.Byte));
            Assert.AreEqual("np.float64(nan)", S("%r", double.NaN, NPTypeCode.Double));
            Assert.AreEqual("np.float16(1.5)", S("%a", (Half)1.5, NPTypeCode.Half));
            Assert.AreEqual("np.complex128(1+2j)", S("%r", new Complex(1, 2), NPTypeCode.Complex));
            Assert.AreEqual("np.str_('ab')", S("%r", "ab", NPTypeCode.String));
            Assert.AreEqual("0.1", S("%s", 0.1f, NPTypeCode.Single)); // the float32's own str, not the widened double
            Assert.AreEqual("1.000000e+00", S("%e", true, NPTypeCode.Boolean));
            Assert.AreEqual("1", S("%d", true, NPTypeCode.Boolean));
            Assert.AreEqual("9.223372036854777856e+18", S("%.18e", (1UL << 63) + 1025, NPTypeCode.UInt64));
            Throws<PrintfArgumentException>(() => S("%x", true, NPTypeCode.Boolean),
                "%x format: an integer is required, not numpy.bool");
            Throws<PrintfArgumentException>(() => S("%o", (Half)1.5, NPTypeCode.Half),
                "%o format: an integer is required, not numpy.float16");
            Throws<PrintfArgumentException>(() => S("%c", true, NPTypeCode.Boolean), "%c requires int or char");
            Throws<PrintfArgumentException>(() => S("%c", 65.0, NPTypeCode.Double), "%c requires int or char");
            Throws<PrintfArgumentException>(() => S("%d", "a", NPTypeCode.String),
                "%d format: a real number is required, not numpy.str_");
            Throws<ValueError>(() => S("%d", (Half)double.NaN, NPTypeCode.Half), "cannot convert float NaN to integer");
            Throws<OverflowException>(() => S("%i", float.PositiveInfinity, NPTypeCode.Single),
                "cannot convert float infinity to integer");
        }

        /// <summary>
        ///     Python objects (getitem's results): a <c>bool</c> is an <c>int</c>, a <c>float</c> is not one, a
        ///     <c>str</c> of one code point is a <c>%c</c>, an <c>int</c> of more than 4300 digits has no decimal text.
        /// </summary>
        [TestMethod]
        public void PythonObjects_FollowTheirPythonTypes()
        {
            Assert.AreEqual("1", Py("%x", true));
            Assert.AreEqual("\x01", Py("%c", true));
            Assert.AreEqual("\U0001F600", Py("%c", "\U0001F600"));
            Assert.AreEqual("\ud800", Py("%c", 0xD800));
            Throws<PrintfArgumentException>(() => Py("%x", 1.0), "%x format: an integer is required, not float");
            Throws<PrintfArgumentException>(() => Py("%c", "ab"), "%c requires int or char");
            Throws<PrintfArgumentException>(() => Py("%e", new Complex(1, 2)), "must be real number, not complex");
            Throws<OverflowException>(() => Py("%c", -2), "%c arg not in range(0x110000)");
            Throws<OverflowException>(() => Py("%c", BigInteger.Pow(2, 70)), "%c arg not in range(0x110000)");
            Throws<ValueError>(() => Py("%d", BigInteger.Pow(10, 4300)),
                "Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit");
            // Power-of-two bases have no limit.
            Assert.AreEqual(3572, Py("%x", BigInteger.Pow(10, 4300)).Length);
            Assert.AreEqual(4300, Py("%d", BigInteger.Pow(10, 4299)).Length);
        }

        /// <summary>
        ///     CPython's arithmetic at the limits: <c>%.2147483647e</c> is <c>%.0e</c> (its <c>precision + 1</c> wraps),
        ///     <c>%g</c> strips zeros past any precision, an integer precision past <c>INT_MAX - 3</c> is <c>precision too
        ///     large</c>, and a width no string can meet is <c>MemoryError</c> (<see cref="OutOfMemoryException"/>).
        /// </summary>
        [TestMethod]
        public void LimitQuirks_AreCPythons()
        {
            Assert.AreEqual("2e+00", Py("%.2147483647e", 1.5));
            Assert.AreEqual("2E+00", Py("%.2147483647E", 2.5));
            Assert.AreEqual("2.e+00", Py("%#.2147483647e", 1.5));
            Assert.AreEqual("0.1000000000000000055511151231257827021181583404541015625", Py("%.2147483647g", 0.1));
            Assert.AreEqual("ab", Py("%.2147483647s", "ab"));
            Throws<OverflowException>(() => Py("%.2147483647d", 1), "precision too large");
            Throws<OverflowException>(() => Py("%.2147483645x", 1), "precision too large");
            Assert.ThrowsException<OutOfMemoryException>(() => Py("%9223372036854775807d", 1));
            // The conversion's own error comes first, as CPython converts before it writes.
            Throws<PrintfArgumentException>(() => Py("%9223372036854775807d", "a"),
                "%d format: a real number is required, not str");
        }

        /// <summary>
        ///     The one-shot entry points (<c>format % (value,)</c> compiled and run once), and the boundary map's other
        ///     Python objects: a C# <c>char</c> is a one-character <c>str</c>, every C# integer an <c>int</c>, a
        ///     <c>float</c> / <c>decimal</c> a <c>float</c>, a list / tuple / dict named by their Python type, an ndarray
        ///     refused by NumPy's own conversion slot (<c>array_float</c>) where it has one.
        /// </summary>
        [TestMethod]
        public void OneShotEntryPoints_AndTheBoundaryMap()
        {
            Assert.AreEqual("  2.2", PrintfFormatter.Format("%5.1f", 2.25, NPTypeCode.Double)); // the tie rounds to even
            Assert.AreEqual("'ab'", PrintfFormatter.FormatPyObject("%r", "ab"));
            Assert.ThrowsException<ArgumentNullException>(() => PrintfFormatter.Format(null, 1, PercentFormat.PyObject));
            Assert.ThrowsException<ArgumentException>(() =>
                PrintfFormatter.Compile("%d").FormatInto(new StringBuilder(), new object[] { 1 }, Array.Empty<NPTypeCode>()));
            Assert.AreEqual("x|x", Py("%s|%c", 'x', 'x'));
            Throws<PrintfArgumentException>(() => Py("%d", 'x'), "%d format: a real number is required, not str");
            Assert.AreEqual("-3 7 9", Py("%d %i %u", (short)-3, (ushort)7, 9u));
            Assert.AreEqual("1.500000e+00 2.500000E+00", Py("%e %E", 1.5f, 2.5m));
            Throws<PrintfArgumentException>(() => Py("%d", new System.Collections.Generic.List<int> { 1 }),
                "%d format: a real number is required, not list");
            Throws<PrintfArgumentException>(() => Py("%d", (1, 2)), "%d format: a real number is required, not tuple");
            Throws<PrintfArgumentException>(() => Py("%x", new System.Collections.Generic.Dictionary<int, int>()),
                "%x format: an integer is required, not dict");
            // An ndarray HAS nb_float (array_float), so PyFloat_AsDouble raises NumPy 2.4's own refusal for an array of one
            // or more dimensions, not CPython's "must be real number" (CPython 3.12 + NumPy 2.4.2: '%e' % np.array([1, 2])).
            Throws<PrintfArgumentException>(() => Py("%e", np.array(new[] { 1, 2 })),
                "only 0-dimensional arrays can be converted to Python scalars");
            Throws<PrintfArgumentException>(() => Py("%d", new object()), "%d format: a real number is required, not Object");
            // NumPy scalars of the dtypes savetxt never hands over alone: a Char is a uint16, a complex128 converts its real
            // part (NumPy warns ComplexWarning).
            Assert.AreEqual("6.5e+01", Run("%.1e", new object[] { 'A' }, new[] { NPTypeCode.Char }));
            Assert.AreEqual("1.000000e+00 1", Run("%e %d", new object[] { new Complex(1, 2), new Complex(1.9, 2) },
                new[] { NPTypeCode.Complex, NPTypeCode.Complex }));
            Throws<PrintfArgumentException>(() => Run("%x", new object[] { new Complex(1, 2) }, new[] { NPTypeCode.Complex }),
                "%x format: an integer is required, not numpy.complex128");
            // A precision .NET's F / E formats cannot take: the result is past any .NET string (CPython builds it).
            Assert.ThrowsException<OutOfMemoryException>(() => Py("%.1000000000f", 1.0));
        }

        // ---- ndarray.tofile ----------------------------------------------------------------------------------------

        /// <summary>A text larger than the writer's batch is flushed in pieces, byte-identical to one write.</summary>
        [TestMethod]
        public void ToFile_LargeOutput_FlushesInBatches()
        {
            var a = np.arange(20_000);
            Assert.AreEqual(string.Join(",", Enumerable.Range(0, 20_000)), ToFile(a, "%d"));
            Assert.AreEqual(string.Join(";", Enumerable.Range(0, 20_000)), ToFile(a, "%s", ";"));
        }

        /// <summary>
        ///     tofile formats <c>(item,)</c> of the Python object getitem returns: a <c>bool</c> (an <c>int</c> to
        ///     <c>%</c>), a <c>float</c> for every float dtype, NumSharp's <c>Char</c> as an <c>int</c> and its
        ///     <c>Decimal</c> as a <c>float</c>; a <c>TypeError</c> surfaces unwrapped.
        /// </summary>
        [TestMethod]
        public void ToFile_FormatsGetItemObjects()
        {
            Assert.AreEqual("1,0", ToFile(np.array(new[] { true, false }), "%x"));
            Assert.AreEqual("0.10000000149011612", ToFile(np.array(new[] { 0.1f }), "%r"));
            Assert.AreEqual("1.5", ToFile(np.array(new[] { (Half)1.5 }), "%s"));
            var e = Throws<TypeError>(() => ToFile(np.array(new long[] { 7 }), "%*d"), "not enough arguments for format string");
            Assert.AreEqual(typeof(TypeError), e.GetType(), "tofile does not wrap the TypeError; its exact type is TypeError");
            Throws<OverflowException>(() => ToFile(np.array(new[] { ulong.MaxValue }), "%*d"),
                "Python int too large to convert to C ssize_t");
            Throws<TypeError>(() => ToFile(np.array(new[] { 1.5 }), "%x"), "%x format: an integer is required, not float");
            // Char: an int (uint16) — %s / %d its number, %c its character, a '*' width.
            var chars = np.array(new[] { 'A', 'b' });
            Assert.AreEqual("65,98", ToFile(chars, "%d"));
            Assert.AreEqual("A,b", ToFile(chars, "%c"));
            Assert.AreEqual("41,62", ToFile(chars, "%x"));
            Throws<TypeError>(() => ToFile(chars, "%*d"), "not enough arguments for format string");
            // Decimal: the float64 of its value.
            var decs = np.array(new[] { 1.5m, 2m });
            Assert.AreEqual("1.5,2.0", ToFile(decs, "%r"));
            Assert.AreEqual("1,2", ToFile(decs, "%d"));
            Throws<TypeError>(() => ToFile(decs, "%x"), "%x format: an integer is required, not float");
        }

        /// <summary>
        ///     An element whose text is not ASCII raises CPython's <c>UnicodeEncodeError</c> after the elements before it are
        ///     written. NumPy 2.4.2 crashes the process there (it does not check <c>PyUnicode_AsASCIIString</c>'s NULL).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void ToFile_NonAsciiText_IsUnicodeEncodeError()
        {
            using var ms = new MemoryStream();
            var a = np.array(new byte[] { 65, 200, 66 });
            Throws<UnicodeEncodeError>(() => a.tofile(ms, ",", "%c"),
                "'ascii' codec can't encode character '\\xc8' in position 0: ordinal not in range(128)");
            Assert.AreEqual("A,", Encoding.Latin1.GetString(ms.ToArray()), "the element before it and its separator are written");
            Throws<UnicodeEncodeError>(() => ToFile(np.array(new long[] { 1 }), "é%d"),
                "'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");
            // The separator is written as its UTF-8 bytes, as NumPy writes the C string.
            using var ms2 = new MemoryStream();
            np.array(new long[] { 1, 2 }).tofile(ms2, "é", "%d");
            CollectionAssert.AreEqual(new byte[] { 0x31, 0xC3, 0xA9, 0x32 }, ms2.ToArray());
        }

        /// <summary>
        ///     tofile's <c>sep</c> and <c>format</c> pass CPython's <c>"s"</c> converter before the file is opened: a NUL or
        ///     a lone surrogate raises and no file is created (an existing one is not truncated).
        /// </summary>
        [TestMethod]
        public void ToFile_ChecksSepAndFormatBeforeOpeningTheFile()
        {
            string path = Path.Combine(Path.GetTempPath(), "ns_pct_" + Guid.NewGuid().ToString("N") + ".bin");
            try
            {
                var a = np.array(new long[] { 1, 2 });
                Throws<ValueError>(() => a.tofile(path, ",", "%s\0"), "embedded null character");
                Throws<ValueError>(() => a.tofile(path, "\0", "%s"), "embedded null character");
                Throws<ValueError>(() => a.tofile(path, "", "%s\0"), "embedded null character"); // binary mode too
                Throws<UnicodeEncodeError>(() => a.tofile(path, ",", "\udc80%s"),
                    "'utf-8' codec can't encode character '\\udc80' in position 0: surrogates not allowed");
                // sep is checked first, and a surrogate before a NUL (CPython encodes, then looks for a NUL).
                Throws<UnicodeEncodeError>(() => a.tofile(path, "\ud800\0", "%s\0"),
                    "'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed");
                Assert.IsFalse(File.Exists(path), "no file is created");
                File.WriteAllText(path, "keep");
                Throws<ValueError>(() => a.tofile(path, "\0", "%s"), "embedded null character");
                Assert.AreEqual("keep", File.ReadAllText(path), "an existing file is not truncated");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     Both writers read only their array's elements: arrays over <see cref="FencedMemory"/> (an inaccessible page
        ///     against either end of the buffer) in contiguous, reversed, stepped and 2-D layouts, written by savetxt and
        ///     by tofile (text and binary), give the bytes a copy gives. A read outside the buffer ends the test host.
        /// </summary>
        [TestMethod]
        public unsafe void Writers_OfAFencedBuffer_ReadOnlyItsElements()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");
            using var fence = new FencedMemory(pages: 2);
            var dtypes = new[] { NPTypeCode.Double, NPTypeCode.Int32, NPTypeCode.Byte, NPTypeCode.Half, NPTypeCode.Boolean, NPTypeCode.Complex };
            foreach (var t in dtypes)
            foreach (long n in new long[] { 1, 6, 17 })
            foreach (bool atEnd in new[] { false, true })
            {
                long bytes = n * t.SizeOf();
                byte* p = fence.Place(bytes, atEnd);
                var fenced = np.frombuffer((IntPtr)p, bytes, t);
                np.copyto(fenced, (np.arange(n) % 5).astype(t));
                foreach (string slice in new[] { ":", "::-1", "1::2", "::3" })
                {
                    var view = fenced[slice];
                    using var copy = view.copy();
                    Assert.AreEqual(ToFile(copy, "%s"), ToFile(view, "%s"), $"{t} n={n} {slice} tofile %s");
                    Assert.AreEqual(ToFile(copy, t == NPTypeCode.Complex ? "%r" : "%e"), ToFile(view, t == NPTypeCode.Complex ? "%r" : "%e"));
                    Assert.AreEqual(SaveTxt(copy, "%s"), SaveTxt(view, "%s"), $"{t} n={n} {slice} savetxt");
                    using var b1 = new MemoryStream();
                    using var b2 = new MemoryStream();
                    copy.tofile(b1);
                    view.tofile(b2);
                    CollectionAssert.AreEqual(b1.ToArray(), b2.ToArray(), $"{t} n={n} {slice} binary");
                }
                if (n == 6)
                {
                    var m = fenced.reshape(2, 3);
                    Assert.AreEqual(SaveTxt(m.copy(), "%s"), SaveTxt(m, "%s"));
                    Assert.AreEqual(SaveTxt(m.T.copy(), "%s"), SaveTxt(m.T, "%s"));
                }
            }
        }

        // ---- np.savetxt --------------------------------------------------------------------------------------------

        /// <summary>
        ///     savetxt formats NumPy scalars: <c>%r</c> names their type (a complex row's parts are <c>np.float64</c>); the
        ///     real branch wraps a <c>TypeError</c> into its Mismatch text, the complex branch raises it as it is, and a
        ///     <c>ValueError</c> of the format propagates from both.
        /// </summary>
        [TestMethod]
        public void SaveTxt_NumPyScalars_AndTheBranchesErrors()
        {
            Assert.AreEqual("np.float32(0.1)\n", SaveTxt(np.array(new[] { 0.1f }), "%r"));
            Assert.AreEqual(" (np.float64(1.0)+np.float64(2.0)j)\n", SaveTxt(np.array(new[] { new Complex(1, 2) }), "%r"));
            var m = Throws<TypeError>(() => SaveTxt(np.array(new[] { true }), "%x"),
                "Mismatch between array dtype ('bool') and format specifier ('%x')");
            Assert.AreEqual("%x format: an integer is required, not numpy.bool", m.InnerException?.Message);
            var raw = Throws<TypeError>(() => SaveTxt(np.array(new[] { new Complex(1, 2) }), "%x"),
                "%x format: an integer is required, not numpy.float64");
            Assert.AreEqual(typeof(TypeError), raw.GetType());
            Throws<TypeError>(() => SaveTxt(np.array(new long[] { 7 }), "%*d"),
                "Mismatch between array dtype ('int64') and format specifier ('%*d')");
            Throws<ValueError>(() => SaveTxt(np.array(new long[] { 7 }), "%y"), "unsupported format character 'y' (0x79) at index 1");
            Throws<ValueError>(() => SaveTxt(np.array(new[] { new Complex(1, 2) }), "%y"),
                "unsupported format character 'y' (0x79) at index 3");
        }

        /// <summary>A text larger than savetxt's batch is delivered in pieces, identical to one write.</summary>
        [TestMethod]
        public void SaveTxt_LargeOutput_FlushesInBatches()
        {
            var a = np.arange(20_000);
            Assert.AreEqual(string.Concat(Enumerable.Range(0, 20_000).Select(i => i + "\n")), SaveTxt(a, "%d"));
            using var ms = new MemoryStream();
            np.savetxt(ms, a, "%d");
            Assert.AreEqual(string.Concat(Enumerable.Range(0, 20_000).Select(i => i + "\n")), Encoding.Latin1.GetString(ms.ToArray()));
        }

        /// <summary>
        ///     NumSharp's dtypes without a NumPy scalar under savetxt: <c>Char</c> is an <c>np.uint16</c> (its str a number,
        ///     <c>%c</c> its character), <c>Decimal</c> an <c>np.float64</c> of its value.
        /// </summary>
        [TestMethod]
        public void SaveTxt_CharAndDecimal_FollowTheirAnalogs()
        {
            var chars = np.array(new[] { 'A' });
            Assert.AreEqual("65\n", SaveTxt(chars, "%s"));
            Assert.AreEqual("np.uint16(65)\n", SaveTxt(chars, "%r"));
            Assert.AreEqual("A\n", SaveTxt(chars, "%c"));
            Assert.AreEqual("0x41\n", SaveTxt(chars, "%#x"));
            var decs = np.array(new[] { 1.5m });
            Assert.AreEqual("np.float64(1.5)\n", SaveTxt(decs, "%r"));
            Assert.AreEqual("1.500e+00\n", SaveTxt(decs, "%.3e"));
            var x = Throws<TypeError>(() => SaveTxt(decs, "%x"), "Mismatch between array dtype ('float64') and format specifier ('%x')");
            Assert.AreEqual("%x format: an integer is required, not numpy.float64", x.InnerException?.Message);
        }

        /// <summary>
        ///     A StringDType row is Python objects, so an NA object that is an int can be a <c>*</c> width: the row
        ///     <c>[NA, 'ab']</c> with <c>fmt=['%*s', '']</c> is <c>'%*s ' % (4, 'ab')</c>; the most negative width pads
        ///     nothing.
        /// </summary>
        [TestMethod]
        public void SaveTxt_StringDTypeIntNA_IsAStarWidth()
        {
            var sw = new StringWriter();
            np.savetxt(sw, np.array(new object[] { 4, "ab" }, np.dtypes.StringDType(na_object: 4)).reshape(1, 2), new[] { "%*s", "" });
            Assert.AreEqual("  ab \n", sw.ToString());
            sw = new StringWriter();
            np.savetxt(sw, np.array(new object[] { long.MinValue, long.MinValue }, np.dtypes.StringDType(na_object: long.MinValue)).reshape(1, 2),
                new[] { "%*d", "" });
            Assert.AreEqual("-9223372036854775808 \n", sw.ToString());
            sw = new StringWriter();
            np.savetxt(sw, np.array(new object[] { -5, -5 }, np.dtypes.StringDType(na_object: -5)).reshape(1, 2), new[] { "%*d", "" });
            Assert.AreEqual("-5    \n", sw.ToString());
        }
    }
}
