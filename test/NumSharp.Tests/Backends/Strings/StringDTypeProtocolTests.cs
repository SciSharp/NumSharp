using System;
using System.Collections;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The CPython protocols NEP 55 <c>StringDType</c> runs on text and on its missing-data object (strings phase 1,
    ///     P1.2): the strict UTF-8 codec (<see cref="NDUtf8"/>) and <c>str()</c> / <c>repr()</c> / <c>==</c> / hash of a
    ///     C# value standing in for a Python object (<see cref="NDPyObject"/>). Expected values are CPython 3.12's
    ///     (win-amd64, NumPy 2.4.2's interpreter). The full comparison these cases are drawn from ran 373,249 codec inputs
    ///     (every 1- and 2-byte sequence, 3- and 4-byte sweeps, 60,000 random byte strings, ~36,000 strings with lone
    ///     surrogates) and 83 objects × <c>str</c> / <c>repr</c> plus a 64 × 64 equality matrix, all identical.
    /// </summary>
    [TestClass]
    public class StringDTypeProtocolTests
    {
        // ------------------------------------------------------------------------------------------------------------
        // NDUtf8: decoding
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     CPython's strict decoder diagnosis, one case per branch of <c>utf8_decode</c>: a bad start byte, a cut-off
        ///     sequence (one byte, two bytes, three bytes), a missing continuation at each position, the overlong / surrogate
        ///     / past-U+10FFFF second-byte limits, and a cut-off sequence an early byte already proves invalid.
        /// </summary>
        /// <param name="hex">The input bytes, as hexadecimal.</param>
        /// <param name="message">CPython 3.12's <c>bytes.decode('utf-8')</c> error text for them.</param>
        [TestMethod]
        [DataRow("80", "'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte")]
        [DataRow("c180", "'utf-8' codec can't decode byte 0xc1 in position 0: invalid start byte")]
        [DataRow("c0af", "'utf-8' codec can't decode byte 0xc0 in position 0: invalid start byte")]
        [DataRow("f5", "'utf-8' codec can't decode byte 0xf5 in position 0: invalid start byte")]
        [DataRow("ff", "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte")]
        [DataRow("c2", "'utf-8' codec can't decode byte 0xc2 in position 0: unexpected end of data")]
        [DataRow("e1", "'utf-8' codec can't decode byte 0xe1 in position 0: unexpected end of data")]
        [DataRow("e0a0", "'utf-8' codec can't decode bytes in position 0-1: unexpected end of data")]
        [DataRow("f09080", "'utf-8' codec can't decode bytes in position 0-2: unexpected end of data")]
        [DataRow("c241", "'utf-8' codec can't decode byte 0xc2 in position 0: invalid continuation byte")]
        [DataRow("e18041", "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte")]
        [DataRow("f09041", "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte")]
        [DataRow("f0908041", "'utf-8' codec can't decode bytes in position 0-2: invalid continuation byte")]
        [DataRow("e08080", "'utf-8' codec can't decode byte 0xe0 in position 0: invalid continuation byte")]
        [DataRow("e080", "'utf-8' codec can't decode byte 0xe0 in position 0: invalid continuation byte")]
        [DataRow("eda080", "'utf-8' codec can't decode byte 0xed in position 0: invalid continuation byte")]
        [DataRow("edbfbf", "'utf-8' codec can't decode byte 0xed in position 0: invalid continuation byte")]
        [DataRow("f0808080", "'utf-8' codec can't decode byte 0xf0 in position 0: invalid continuation byte")]
        [DataRow("f4908080", "'utf-8' codec can't decode byte 0xf4 in position 0: invalid continuation byte")]
        [DataRow("414280", "'utf-8' codec can't decode byte 0x80 in position 2: invalid start byte")]
        [DataRow("c3a9e282ac80", "'utf-8' codec can't decode byte 0x80 in position 5: invalid start byte")]
        public void Decode_InvalidBytes_RaisesCPythonsError(string hex, string message)
        {
            var bytes = Convert.FromHexString(hex);
            Action check = () => NDUtf8.ThrowIfInvalid(bytes);
            check.Should().Throw<UnicodeDecodeError>().WithMessage(message);
            Action decode = () => NDUtf8.DecodeStrict(bytes);
            decode.Should().Throw<UnicodeDecodeError>().WithMessage(message);
        }

        /// <summary>
        ///     Valid UTF-8 at the edges of each sequence length (U+007F, U+0080, U+07FF, U+0800, U+D7FF, U+E000, U+FFFF,
        ///     U+10000, U+10FFFF) decodes exactly as .NET's decoder; the trusted decoder agrees and maps empty input to the
        ///     empty string.
        /// </summary>
        [TestMethod]
        public void Decode_ValidBytes_RoundTrip()
        {
            foreach (var hex in new[] { "", "41", "7f", "c280", "dfbf", "e0a080", "ed9fbf", "ee8080", "efbfbf", "f0908080", "f48fbfbf", "c3a9e282acf09f9880" })
            {
                var bytes = Convert.FromHexString(hex);
                NDUtf8.ThrowIfInvalid(bytes);
                NDUtf8.DecodeStrict(bytes).Should().Be(Encoding.UTF8.GetString(bytes), hex);
                NDUtf8.DecodeTrusted(bytes).Should().Be(Encoding.UTF8.GetString(bytes), hex);
            }
            NDUtf8.DecodeTrusted(ReadOnlySpan<byte>.Empty).Should().BeSameAs(string.Empty);
        }

        /// <summary>The error factory: one byte names the byte, a range names the positions.</summary>
        [TestMethod]
        public void DecodeError_FormatsOneByteAndRanges()
        {
            var b = new byte[] { 0x41, 0xab, 0xcd };
            NDUtf8.DecodeError(b, 1, 2, "invalid start byte").Message.Should().Be("'utf-8' codec can't decode byte 0xab in position 1: invalid start byte");
            NDUtf8.DecodeError(b, 1, 3, "unexpected end of data").Message.Should().Be("'utf-8' codec can't decode bytes in position 1-2: unexpected end of data");
        }

        // ------------------------------------------------------------------------------------------------------------
        // NDUtf8: encoding
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The lone-surrogate cases of <see cref="Encode_LoneSurrogate_RaisesCPythonsError"/>, with CPython 3.12's
        ///     <c>str.encode('utf-8')</c> texts.
        /// </summary>
        /// <remarks>
        ///     Kept in the method body, not in <c>[DataRow]</c> / <c>[DynamicData]</c>: an attribute argument is stored as
        ///     UTF-8 in the metadata, which cannot hold a lone surrogate (it reads back as replacement characters), and the
        ///     test platform serializes data-source rows as XML, which refuses one (a high surrogate followed by a high
        ///     surrogate fails to deserialize). A string literal in a method body is UTF-16 and keeps it.
        /// </remarks>
        private static readonly (string s, string message)[] LoneSurrogateCases =
        {
            ("\udfff", "'utf-8' codec can't encode character '\\udfff' in position 0: surrogates not allowed"),
            ("a\udfff中", "'utf-8' codec can't encode character '\\udfff' in position 1: surrogates not allowed"),
            ("中\udfff\udfff中a", "'utf-8' codec can't encode characters in position 1-2: surrogates not allowed"),
            ("😀\ud800a\udfff", "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed"),
            ("😀\udfff\udbff", "'utf-8' codec can't encode characters in position 1-2: surrogates not allowed"),
            ("é\U00010000\udc00😀", "'utf-8' codec can't encode character '\\udc00' in position 2: surrogates not allowed"),
            ("😀éé中\ud800", "'utf-8' codec can't encode character '\\ud800' in position 4: surrogates not allowed"),
            ("\ud800\udbff", "'utf-8' codec can't encode characters in position 0-1: surrogates not allowed"),
            ("\udc00é😀", "'utf-8' codec can't encode character '\\udc00' in position 0: surrogates not allowed"),
        };

        /// <summary>
        ///     A lone surrogate is CPython's <c>UnicodeEncodeError</c>; positions count CODE POINTS (a valid pair is one),
        ///     and a run of consecutive lone surrogates is one range. <see cref="NDUtf8.GetByteCount"/> and both encoders
        ///     raise the same text, for every case of <see cref="LoneSurrogateCases"/>.
        /// </summary>
        [TestMethod]
        public void Encode_LoneSurrogate_RaisesCPythonsError()
        {
            foreach (var (s, message) in LoneSurrogateCases)
            {
                Action count = () => NDUtf8.GetByteCount(s);
                count.Should().Throw<UnicodeEncodeError>().WithMessage(message);
                Action span = () => NDUtf8.Encode(s, new byte[s.Length * 4]);
                span.Should().Throw<UnicodeEncodeError>().WithMessage(message);
                Action array = () => NDUtf8.Encode(s);
                array.Should().Throw<UnicodeEncodeError>().WithMessage(message);
            }
        }

        /// <summary>
        ///     Valid text: the byte count and the bytes are .NET's UTF-8 (a surrogate PAIR is one 4-byte character — a C#
        ///     string maps to the Python <c>str</c> of its UTF-16 decoding); ASCII takes the fast path; a short destination
        ///     is an <see cref="ArgumentException"/>.
        /// </summary>
        [TestMethod]
        public void Encode_ValidText_MatchesUtf8()
        {
            foreach (var s in new[] { "", "abc", "é", "中", "😀", "a😀b", "\U00010000", "\u007f\u0080߿ࠀ￿" })
            {
                var want = Encoding.UTF8.GetBytes(s);
                NDUtf8.GetByteCount(s).Should().Be(want.Length, s);
                NDUtf8.Encode(s).Should().Equal(want, s);
                var dst = new byte[want.Length + 3];
                NDUtf8.Encode(s, dst).Should().Be(want.Length);
                dst.AsSpan(0, want.Length).ToArray().Should().Equal(want);
            }
            NDUtf8.GetByteCount(new string('x', 1000)).Should().Be(1000);
            Action shortDst = () => NDUtf8.Encode("中", new byte[2]);
            shortDst.Should().Throw<ArgumentException>();
        }

        // ------------------------------------------------------------------------------------------------------------
        // NDPyObject: str / repr
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>str()</c> and <c>repr()</c> of numbers under the boundary map: Python ints (bool prints True / False),
        ///     Python floats (a C# float is the Python float of its widened value), complex numbers (CPython's pure
        ///     imaginary <c>1j</c> / <c>0j</c> and signed zeros), <c>np.float16</c> (NumPy's scalar digits, repr naming the
        ///     type) and decimal (printed as the float64 it converts to).
        /// </summary>
        [TestMethod]
        public void StrRepr_Numbers()
        {
            void Same(object o, string text)
            {
                NDPyObject.Str(o).Should().Be(text);
                NDPyObject.Repr(o).Should().Be(text);
            }
            Same(null, "None");
            Same(true, "True");
            Same(false, "False");
            Same(-5, "-5");
            Same((byte)255, "255");
            Same(long.MaxValue, "9223372036854775807");
            Same(ulong.MaxValue, "18446744073709551615");
            Same(-BigInteger.Pow(2, 70), "-1180591620717411303424");
            Same(1.5, "1.5");
            Same(-0.0, "-0.0");
            Same(1e16, "1e+16");
            Same(1e15, "1000000000000000.0");
            Same(1e-5, "1e-05");
            Same(1e-4, "0.0001");
            Same(123456789012345678.0, "1.2345678901234568e+17");
            Same(-double.NaN, "nan");
            Same(double.NegativeInfinity, "-inf");
            Same(5e-324, "5e-324");
            Same(0.1f, "0.10000000149011612");
            Same(3.4e38f, "3.3999999521443642e+38");
            Same(1.5m, "1.5");
            Same(new Complex(1, 2), "(1+2j)");
            Same(new Complex(0, 1), "1j");
            Same(new Complex(-0.0, 1), "(-0+1j)");
            Same(new Complex(1, -0.0), "(1-0j)");
            Same(new Complex(double.NaN, 0), "(nan+0j)");
            Same(new Complex(0, double.NaN), "nanj");
            Same(new Complex(double.PositiveInfinity, double.NegativeInfinity), "(inf-infj)");
            Same(new Complex(1.5, -2.5e-10), "(1.5-2.5e-10j)");
            Same(new Complex(0, 0), "0j");
            Same(new Complex(-0.0, -0.0), "(-0-0j)");

            NDPyObject.Str((Half)65504).Should().Be("6.55e+04");
            NDPyObject.Repr((Half)65504).Should().Be("np.float16(6.55e+04)");
            NDPyObject.Str((Half)0.1).Should().Be("0.1");
            NDPyObject.Repr(Half.NaN).Should().Be("np.float16(nan)");
            NDPyObject.Repr((Half)(-0.0)).Should().Be("np.float16(-0.0)");
            NDPyObject.Repr((Half)1e-7).Should().Be("np.float16(1e-07)");
            NDPyObject.Repr((Half)6e-8).Should().Be("np.float16(6e-08)");
        }

        /// <summary>
        ///     Text: <c>str()</c> is the text, <c>repr()</c> is CPython's (quote choice, escapes for control characters,
        ///     non-printable separators and lone surrogates; printable non-ASCII kept); a char is a one-character str;
        ///     bytes print their repr under both protocols.
        /// </summary>
        [TestMethod]
        public void StrRepr_TextAndBytes()
        {
            void Text(string s, string repr)
            {
                NDPyObject.Str(s).Should().Be(s);
                NDPyObject.Repr(s).Should().Be(repr);
            }
            Text("NA", "'NA'");
            Text("", "''");
            Text("it's", "\"it's\"");
            Text("q\"z", "'q\"z'");
            Text("both'\"", "'both\\'\"'");
            Text("a\nb", "'a\\nb'");
            Text("tab\t", "'tab\\t'");
            Text("\0", "'\\x00'");
            Text("\x7f", "'\\x7f'");
            Text("\u0080", "'\\x80'");
            Text("\u00a0", "'\\xa0'");
            Text("é", "'é'");
            Text("\u200b", "'\\u200b'");
            Text("\u2028", "'\\u2028'");
            Text("\ud800", "'\\ud800'");
            Text("a😀b", "'a😀b'");
            Text("\\", "'\\\\'");
            NDPyObject.Str('x').Should().Be("x");
            NDPyObject.Repr('\'').Should().Be("\"'\"");

            void Bytes(byte[] b, string repr)
            {
                NDPyObject.Str(b).Should().Be(repr);
                NDPyObject.Repr(b).Should().Be(repr);
            }
            Bytes(new byte[0], "b''");
            Bytes(new byte[] { 0x78 }, "b'x'");
            Bytes(new byte[] { 0x80 }, "b'\\x80'");
            Bytes(Encoding.ASCII.GetBytes("it's"), "b\"it's\"");
            Bytes(Encoding.ASCII.GetBytes("both'\""), "b'both\\'\"'");
            Bytes(new byte[] { 0x0a, 0x09, 0x5c, 0x0d }, "b'\\n\\t\\\\\\r'");
            Bytes(new byte[] { 0x00, 0x7f }, "b'\\x00\\x7f'");
        }

        /// <summary>
        ///     Containers print Python's container reprs under both protocols (a one-element tuple keeps its comma); an
        ///     <see cref="NDArray"/> prints <c>str(array)</c> / <c>repr(array)</c>; any other object prints its
        ///     <see cref="object.ToString"/>.
        /// </summary>
        [TestMethod]
        public void StrRepr_ContainersArraysAndOtherObjects()
        {
            void Same(object o, string text)
            {
                NDPyObject.Str(o).Should().Be(text);
                NDPyObject.Repr(o).Should().Be(text);
            }
            Same(ValueTuple.Create(1), "(1,)");
            Same(ValueTuple.Create(), "()");
            Same((1, 2), "(1, 2)");
            Same(new object[0], "[]");
            Same(new object[] { 1, "a" }, "[1, 'a']");
            Same(new Hashtable(), "{}");
            Same(new Hashtable { { "k", 2 } }, "{'k': 2}");
            Same(new object[] { ValueTuple.Create(1), new Hashtable { { "a", new object[] { null } } } }, "[(1,), {'a': [None]}]");
            Same(((object)null, true, 1.5, "x", new byte[] { 0x79 }), "(None, True, 1.5, 'x', b'y')");
            Same(Tuple.Create(1, "a"), "(1, 'a')");

            var a = np.array(new[] { 1, 2 });
            NDPyObject.Str(a).Should().Be("[1 2]");
            NDPyObject.Repr(a).Should().Be("array([1, 2], dtype=int32)");
            Same(new Opaque(), "opaque!");
        }

        /// <summary>An object outside the boundary map: Python's default str / repr have no analog, so its ToString stands in.</summary>
        private sealed class Opaque
        {
            /// <inheritdoc/>
            public override string ToString() => "opaque!";
        }

        // ------------------------------------------------------------------------------------------------------------
        // NDPyObject: equality, hash, NaN-likeness
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>na_eq_cmp</c>: identity; Python-float NaNs; Python's cross-type numeric equality (exact for integral
        ///     values: <c>2**53 + 1 != 2.0**53</c>); text ordinal (a char equals its one-character string); bytes by
        ///     content and never equal to text; arrays only by identity; a NumPy <c>np.float16</c> NaN is not a Python float.
        /// </summary>
        [TestMethod]
        public void Equal_FollowsNaEqCmp()
        {
            var cases = new (object a, object b, bool eq)[]
            {
                (null, null, true), (null, 0, false), (0, null, false),
                (double.NaN, double.NaN, true), (double.NaN, -double.NaN, true), (float.NaN, double.NaN, true),
                (double.NaN, 0.0, false), (Half.NaN, Half.NaN, false), (Half.NaN, double.NaN, false),
                (new Complex(double.NaN, 0), new Complex(double.NaN, 0), false),
                (5, 5.0, true), (5, 5L, true), ((byte)5, (ulong)5, true), (true, 1, true), (false, 0.0, true), (true, 2, false),
                (new Complex(1, 0), 1, true), (new Complex(1, 1), 1, false), (0.0, -0.0, true), (1.5m, 1.5, true), ((Half)1.5, 1.5f, true),
                (BigInteger.Pow(2, 70), System.Math.Pow(2, 70), true), ((1L << 53) + 1, System.Math.Pow(2, 53), false), (ulong.MaxValue, System.Math.Pow(2, 64), false),
                (double.PositiveInfinity, double.PositiveInfinity, true), (double.PositiveInfinity, float.PositiveInfinity, true),
                (0.1, 0.1f, false), (1, "1", false), ("a", "a", true), ("a", "A", false), ("x", 'x', true), ('x', "x", true), ('x', 'x', true),
                ("xy", 'x', false), (new byte[] { 1 }, new byte[] { 1 }, true), (new byte[] { 1 }, new byte[] { 2 }, false),
                (new byte[] { 0x78 }, "x", false), ("x", new byte[] { 0x78 }, false),
                // CPython compares an int and a float exactly, even where the int's double would overflow to inf.
                (BigInteger.Pow(2, 1100), double.PositiveInfinity, false), (BigInteger.Pow(2, 70), System.Math.Pow(2, 70) + 1e6, false),
                // A NumPy scalar (np.float16; decimal as np.float64) converts a Python number to its own dtype first.
                ((Half)2048, 2049, true), ((Half)2048, 2050, false), ((Half)0.1, 0.1, true), ((Half)65504, 65519, true),
                ((Half)65504, 65520, false), (Half.PositiveInfinity, BigInteger.Pow(2, 70), true),
                (Half.PositiveInfinity, BigInteger.Pow(2, 1100), false), ((Half)0.1, new Complex(0.1, 0), false),
                ((Half)1, new Complex(1, 0), true), ((Half)1, true, true), ((Half)(-0.0), 0, true), ((Half)0.1, 0.1m, false),
                (9007199254740992m, (1L << 53) + 1, true), (0.1m, 0.1, true), (0.1m, 0.1f, false), (0.1m, new Complex(0.1, 0), true),
                // A NumPy scalar against a sequence: the array's elementwise ==, true only for a single leaf, strong promotion.
                ((Half)2048, new object[] { 2049 }, false), ((Half)2048, new object[] { new object[] { 2048 } }, true),
                ((Half)0.1, new object[] { 0.1 }, false), ((Half)0.1, new object[] { (Half)0.1 }, true), ((Half)1, ValueTuple.Create(1), true),
                ((Half)1, new object[] { true }, true), ((Half)1, new object[] { 1, 1 }, false), ((Half)1, new object[0], false),
                ((Half)1, new object[] { "1" }, false), ((Half)1, new object[] { BigInteger.Pow(2, 70) }, false),
                (9007199254740992m, new object[] { (1L << 53) + 1 }, true), (1.0, new object[] { 1 }, false),
                // Containers compare item by item with Python's rule; a tuple never equals a list.
                ((1, 2), (1L, 2.0), true), ((1, 2), (1, 3), false), (new object[] { 1 }, ValueTuple.Create(1), false),
                (new object[] { 1, "a" }, new System.Collections.Generic.List<object> { 1.0, "a" }, true),
                (new Hashtable { { "k", 1 } }, new Hashtable { { "k", 1.0 } }, true), (new Hashtable { { "k", 1 } }, new Hashtable { { "k", 2 } }, false),
                (new Hashtable { { "k", 1 } }, new Hashtable { { "j", 1 } }, false), (new object[] { double.NaN }, new object[] { double.NaN }, false),
            };
            foreach (var (a, b, eq) in cases)
            {
                NDPyObject.Equal(a, b).Should().Be(eq, $"{NDPyObject.Repr(a)} == {NDPyObject.Repr(b)}");
                if (eq)
                    NDPyObject.Hash(a).Should().Be(NDPyObject.Hash(b), $"hash({NDPyObject.Repr(a)}) == hash({NDPyObject.Repr(b)})");
            }
            var arr = np.array(new[] { 1.0 });
            NDPyObject.Equal(arr, arr).Should().BeTrue();
            NDPyObject.Equal(arr, np.array(new[] { 1.0 })).Should().BeFalse();
            var h = (object)Half.NaN;
            NDPyObject.Equal(h, h).Should().BeTrue("the same object is equal to itself");
            var opaque = new Opaque();
            NDPyObject.Equal(opaque, opaque).Should().BeTrue();
            NDPyObject.Equal(opaque, new Opaque()).Should().BeFalse();
            NDPyObject.Equal((1, "a"), (1, "a")).Should().BeTrue("a value tuple compares its items");
        }

        /// <summary>
        ///     NumPy's NaN-likeness (<c>na != na</c> truthy or raising): float / float16 / complex NaNs and arrays whose
        ///     comparison has no single truth value are NaN-like; None, numbers, text and a one-element non-NaN array are not.
        /// </summary>
        [TestMethod]
        public void IsNaNLike_FollowsNaNotEqualNa()
        {
            NDPyObject.IsNaNLike(null).Should().BeFalse();
            NDPyObject.IsNaNLike(double.NaN).Should().BeTrue();
            NDPyObject.IsNaNLike(float.NaN).Should().BeTrue();
            NDPyObject.IsNaNLike(Half.NaN).Should().BeTrue();
            NDPyObject.IsNaNLike(new Complex(0, double.NaN)).Should().BeTrue();
            NDPyObject.IsNaNLike(new Complex(double.NaN, 1)).Should().BeTrue();
            NDPyObject.IsNaNLike(1.5).Should().BeFalse();
            NDPyObject.IsNaNLike(double.PositiveInfinity).Should().BeFalse();
            NDPyObject.IsNaNLike(5).Should().BeFalse();
            NDPyObject.IsNaNLike("nan").Should().BeFalse();
            NDPyObject.IsNaNLike(np.array(new[] { 1.0, 2.0 })).Should().BeTrue("a two-element comparison has no truth value");
            NDPyObject.IsNaNLike(np.array(new double[0])).Should().BeTrue("an empty comparison has no truth value in NumPy 2.4");
            NDPyObject.IsNaNLike(np.array(new[] { double.NaN })).Should().BeTrue();
            NDPyObject.IsNaNLike(np.array(new[] { 1.0 })).Should().BeFalse();
        }

        /// <summary>
        ///     The protocol corners a sentinel can reach: a dict of several entries prints CPython's
        ///     <c>{'a': 1, 'b': 2}</c>; <c>None</c> equals nothing but itself; dicts of different sizes differ; an
        ///     <c>np.float16</c> scalar against a one-element list holding a complex compares elementwise in complex128
        ///     (<c>np.float16(1) == [1+0j]</c> is True), against a C# unsigned integer (a Python int) as a float; an array hashes
        ///     by identity (it equals only itself) and an object outside the boundary map by its own hash.
        /// </summary>
        [TestMethod]
        public void Protocol_Corners()
        {
            NDPyObject.Repr(new Hashtable { { "a", 1 } }).Should().Be("{'a': 1}");
            var two = new System.Collections.Generic.SortedDictionary<string, int> { { "a", 1 }, { "b", 2 } };
            NDPyObject.Repr(two).Should().Be("{'a': 1, 'b': 2}");
            NDPyObject.PyEq(null, "x").Should().BeFalse();
            NDPyObject.PyEq("x", null).Should().BeFalse();
            NDPyObject.PyEq(new Hashtable { { "a", 1 } }, new Hashtable { { "a", 1 }, { "b", 2 } }).Should().BeFalse();
            NDPyObject.PyEq((Half)1, new object[] { new Complex(1, 0) }).Should().BeTrue();
            NDPyObject.PyEq((Half)1, new object[] { new Complex(1, 1) }).Should().BeFalse();
            NDPyObject.PyEq((Half)1, (byte)1).Should().BeTrue();
            NDPyObject.PyEq((Half)2, ulong.MaxValue).Should().BeFalse();
            var a = np.array(new[] { 1.0 });
            NDPyObject.Hash(a).Should().Be(System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(a));
            var opaque = new Opaque();
            NDPyObject.Hash(opaque).Should().Be(opaque.GetHashCode());
        }
    }
}
