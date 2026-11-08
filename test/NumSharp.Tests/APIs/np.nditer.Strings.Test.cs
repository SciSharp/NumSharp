using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     <c>np.nditer</c> over fixed-width strings (<c>U</c> and <c>S</c>): the value stream over every layout and order,
    ///     the tracked indices, external-loop chunks, buffered width and class conversions with every casting rule, the copy
    ///     routes, writes through the views and the setter, allocated outputs, operand pairs, ranges, nested iterators,
    ///     seeks and copies, and NumPy's refusals. Every expected value and every error text was probed against NumPy 2.4.2
    ///     (2026-10-09); the three tests of <see cref="np_nditer_NumPyParity_Test"/> that cover strings stay where they are.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Reading a value: a <c>U</c> element is a <see cref="string"/>, an <c>S</c> element a <see cref="byte"/> array,
    ///     and both drop their trailing NULs, as NumPy's <c>str_</c> / <c>bytes_</c> scalars do. <see cref="Text"/> reads
    ///     either as text (an <c>S</c> element through Latin-1, so every byte survives).
    ///     </para>
    ///     <para>
    ///     A conversion between text and numbers runs NumPy's legacy casts: a number is written as its scalar's
    ///     <c>str()</c>, a text is parsed by <c>int()</c> / <c>float()</c> / <c>complex()</c>
    ///     (<see cref="TextNumberConversions_FormatAndParse"/>).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_nditer_Strings_Test
    {
        /// <summary>A <c>U</c> array of the given texts, as NumPy's <c>np.array(list_of_str)</c>: its width is the longest.</summary>
        /// <param name="s">The elements.</param>
        /// <returns>A new 1-D <c>&lt;Un</c> array.</returns>
        private static NDArray U(params string[] s) => np.array(s);

        /// <summary>An <c>S</c> array of the ASCII bytes of the given texts, as NumPy's <c>np.array(list_of_bytes)</c>.</summary>
        /// <param name="s">The elements (ASCII text).</param>
        /// <returns>A new 1-D <c>|Sn</c> array.</returns>
        private static NDArray S(params string[] s)
            => np.array(s.Select(x => (object)Encoding.ASCII.GetBytes(x)).ToArray(), dtype: "S");

        /// <summary>One element as text: a <c>U</c> element is its string, an <c>S</c> element its bytes through Latin-1.</summary>
        /// <param name="element">A boxed element read with <see cref="NDArray.GetAtIndex"/>.</param>
        /// <returns>The text (every byte of an <c>S</c> element maps to one character, so nothing is lost).</returns>
        private static string Text(object element) => element is byte[] b ? Encoding.Latin1.GetString(b) : (string)element;

        /// <summary>Walks an iterator to its end and reads operand 0 at every step, as text.</summary>
        /// <param name="it">The iterator to walk; it is finished afterwards.</param>
        /// <returns>The texts in iteration order.</returns>
        private static List<string> Walk(np.NDIterator it) => it.Select(v => Text(v[0].GetAtIndex(0))).ToList();

        /// <summary>Every element of a chunk (a 1-D external-loop view), as text.</summary>
        /// <param name="chunk">The view.</param>
        /// <returns>The texts in the view's order.</returns>
        private static string[] Chunk(NDArray chunk)
            => Enumerable.Range(0, (int)chunk.size).Select(i => Text(chunk.GetAtIndex(i))).ToArray();

        /// <summary>Every element of an array in C order, as text.</summary>
        /// <param name="a">The array (any rank and layout).</param>
        /// <returns>The texts in C order.</returns>
        private static string[] Elements(NDArray a)
        {
            // A flat C-order copy, so a view of any layout reads in its logical order.
            using var flat = a.flatten();
            return Chunk(flat);
        }

        /// <summary>Writes one text into the current element of operand 0 through <c>np.copyto</c> (NumPy's <c>x[...] = w</c>).</summary>
        /// <param name="view">The element view the walk handed out.</param>
        /// <param name="text">The text; a <c>U</c> source, cast into the view's dtype with <c>'same_kind'</c>.</param>
        private static void Write(NDArray view, string text)
        {
            using var source = np.array(text);
            np.copyto(view, source);
        }

        // ------------------------------------------------------------------ the value stream

        /// <summary>
        ///     A <c>U</c> array yields 0-d views of its OWN <c>&lt;U3</c> dtype (not a <c>U1</c> or a wider one), each holding
        ///     its text without padding; <c>dtypes</c> reports the width.
        /// </summary>
        [TestMethod]
        public void UnicodeArray_YieldsZeroDViewsOfItsOwnDtype()
        {
            using var u = U("ab", "c", "def");
            using var it = np.nditer(u);
            it.dtypes.Select(d => d.str).Should().Equal("<U3");
            it.Select(v => (Text(v[0].GetAtIndex(0)), v[0].dtype.str, v[0].ndim)).Should()
              .Equal(("ab", "<U3", 0), ("c", "<U3", 0), ("def", "<U3", 0));
        }

        /// <summary>An <c>S</c> array yields byte strings of its own <c>|S3</c> dtype.</summary>
        [TestMethod]
        public void BytesArray_YieldsByteStringsOfItsOwnDtype()
        {
            using var s = S("ab", "c", "def");
            using var it = np.nditer(s);
            it.dtypes.Select(d => d.str).Should().Equal("|S3");
            var raw = it.Select(v => v[0].GetAtIndex(0)).ToList();
            raw.Should().AllBeOfType<byte[]>();
            raw.Select(Text).Should().Equal("ab", "c", "def");
        }

        /// <summary>A tracked multi-index walks a 2-D string array in C order with the coordinates of each element.</summary>
        [TestMethod]
        public void MultiIndex_WalksTheCoordinates()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using var it = np.nditer(u, new[] { "multi_index" });
            it.Select(v => $"{string.Join(",", it.multi_index)}:{Text(v[0].GetAtIndex(0))}").Should()
              .Equal("0,0:a", "0,1:b", "0,2:c", "1,0:d", "1,1:e", "1,2:f");
        }

        /// <summary>
        ///     The order argument works on strings as on numbers: <c>'F'</c> over a C array and <c>'A'</c> over an F array both
        ///     walk <c>a d b e c f</c>.
        /// </summary>
        [TestMethod]
        public void OrderF_AndOrderAOverAnFArray()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using (var it = np.nditer(u, order: 'F'))
                Walk(it).Should().Equal("a", "d", "b", "e", "c", "f");

            using var f = np.asfortranarray(u);
            using (var it = np.nditer(f, order: 'A'))
                Walk(it).Should().Equal("a", "d", "b", "e", "c", "f");
        }

        /// <summary>
        ///     Under <c>'K'</c> a transposed view and a reversed view walk MEMORY order (<c>a … f</c>, <c>a b c d</c>); under
        ///     <c>'C'</c> the reversed view walks its logical order, <c>d c b a</c>.
        /// </summary>
        [TestMethod]
        public void KOrder_FollowsMemory_COrderFollowsTheView()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using var t = u.T;
            using (var it = np.nditer(t))
                Walk(it).Should().Equal("a", "b", "c", "d", "e", "f");

            using var four = U("a", "b", "c", "d");
            using var reversed = four["::-1"];
            using (var it = np.nditer(reversed))
                Walk(it).Should().Equal("a", "b", "c", "d");
            using (var it = np.nditer(reversed, order: 'C'))
                Walk(it).Should().Equal("d", "c", "b", "a");
        }

        /// <summary>A stepped view skips elements, and a broadcast view repeats its row: <c>a c e</c>, <c>x yz x yz</c>.</summary>
        [TestMethod]
        public void SteppedAndBroadcastViews()
        {
            using var six = U("a", "b", "c", "d", "e", "f");
            using var stepped = six["::2"];
            using (var it = np.nditer(stepped))
                Walk(it).Should().Equal("a", "c", "e");

            using var row = U("x", "yz");
            using var broadcast = np.broadcast_to(row, new Shape(2, 2));
            using (var it = np.nditer(broadcast))
                Walk(it).Should().Equal("x", "yz", "x", "yz");
        }

        /// <summary>
        ///     A <c>c_index</c> under an F walk and an <c>f_index</c> under the default walk name each element's position in
        ///     their own order: <c>0 3 1 4 2 5</c> and <c>0 2 4 1 3 5</c>.
        /// </summary>
        [TestMethod]
        public void CIndexAndFIndex_NameTheFlatPositions()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using (var it = np.nditer(u, new[] { "c_index" }, order: 'F'))
            {
                it.Select(v => $"{it.index}:{Text(v[0].GetAtIndex(0))}").Should()
                  .Equal("0:a", "3:d", "1:b", "4:e", "2:c", "5:f");
            }

            using (var it = np.nditer(u, new[] { "f_index" }))
            {
                it.Select(v => $"{it.index}:{Text(v[0].GetAtIndex(0))}").Should()
                  .Equal("0:a", "2:b", "4:c", "1:d", "3:e", "5:f");
            }
        }

        /// <summary>
        ///     Supplementary-plane characters are ONE character each (NumPy's UCS-4 units): <c>'😀x'</c> makes a
        ///     <c>&lt;U2</c> array and reads back whole.
        /// </summary>
        [TestMethod]
        public void SupplementaryCharacters_CountOnce()
        {
            using var u = U("\U0001F600x", "y");
            u.dtype.str.Should().Be("<U2");
            using var it = np.nditer(u);
            Walk(it).Should().Equal("\U0001F600x", "y");
        }

        /// <summary>Empty elements iterate as empty text.</summary>
        [TestMethod]
        public void EmptyElements_IterateAsEmptyText()
        {
            using var u = U("", "a", "");
            using var it = np.nditer(u);
            Walk(it).Should().Equal("", "a", "");
        }

        /// <summary>
        ///     An <c>S</c> element drops its TRAILING NULs on read but keeps a leading one: <c>b'a\x00'</c> reads <c>b'a'</c>,
        ///     <c>b'\x00b'</c> stays two bytes.
        /// </summary>
        [TestMethod]
        public void BytesTrailingNuls_AreDroppedOnRead()
        {
            using var s = np.array(new object[] { new byte[] { (byte)'a', 0 }, new byte[] { 0, (byte)'b' } }, dtype: "S2");
            using var it = np.nditer(s);
            Walk(it).Should().Equal("a", "\0b");
        }

        /// <summary>A 3-D <c>S</c> array walked in F order with its multi-index: the first axis varies fastest.</summary>
        [TestMethod]
        public void Bytes3D_FOrderMultiIndex()
        {
            using var s = S("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l").reshape(2, 3, 2);
            using var it = np.nditer(s, new[] { "multi_index" }, order: 'F');
            it.Select(v => $"{string.Join(",", it.multi_index)}:{Text(v[0].GetAtIndex(0))}").Should()
              .Equal("0,0,0:a", "1,0,0:g", "0,1,0:c", "1,1,0:i", "0,2,0:e", "1,2,0:k",
                     "0,0,1:b", "1,0,1:h", "0,1,1:d", "1,1,1:j", "0,2,1:f", "1,2,1:l");
        }

        // ------------------------------------------------------------------ external loop

        /// <summary>
        ///     <c>external_loop</c> over a C string array is ONE chunk of every element, in the array's own <c>&lt;U2</c>
        ///     dtype.
        /// </summary>
        [TestMethod]
        public void ExternalLoop_IsOneChunkOfTheArraysDtype()
        {
            using var u = U("a", "bb", "c", "dd", "e", "ff").reshape(2, 3);
            using var it = np.nditer(u, new[] { "external_loop" });
            var chunks = it.Select(v => (v[0].shape, v[0].dtype.str, Chunk(v[0]))).ToList();
            chunks.Should().HaveCount(1);
            chunks[0].Item1.Should().Equal(6L);
            chunks[0].Item2.Should().Be("<U2");
            chunks[0].Item3.Should().Equal("a", "bb", "c", "dd", "e", "ff");
        }

        /// <summary>
        ///     Views whose strides coalesce stay ONE chunk: a column-stepped <c>(2, 4)[:, ::2]</c> view is one strided chunk
        ///     <c>a c e g</c>, a reversed view one chunk in memory order <c>a b c d</c>.
        /// </summary>
        [TestMethod]
        public void ExternalLoop_CoalescedViews_AreOneChunk()
        {
            using var eight = U("a", "b", "c", "d", "e", "f", "g", "h").reshape(2, 4);
            using var stepped = eight[":, ::2"];
            using (var it = np.nditer(stepped, new[] { "external_loop" }))
            {
                var chunks = it.Select(v => Chunk(v[0])).ToList();
                chunks.Should().HaveCount(1);
                chunks[0].Should().Equal("a", "c", "e", "g");
            }

            using var four = U("a", "b", "c", "d");
            using var reversed = four["::-1"];
            using (var it = np.nditer(reversed, new[] { "external_loop" }))
            {
                var chunks = it.Select(v => Chunk(v[0])).ToList();
                chunks.Should().HaveCount(1);
                chunks[0].Should().Equal("a", "b", "c", "d");
            }
        }

        /// <summary>
        ///     A buffered cast with <c>external_loop</c> chunks by the buffer size, each chunk in the REQUESTED dtype
        ///     (<c>&lt;U4</c>): <c>[a, b] [c, d] [e]</c> — over a reversed view too, in memory order.
        /// </summary>
        [TestMethod]
        public void BufferedExternalLoop_ChunksByTheBufferSize()
        {
            using var u = U("a", "b", "c", "d", "e");
            using (var it = np.nditer(u, new[] { "buffered", "external_loop" }, op_dtypes: new[] { np.dtype("<U4") }, buffersize: 2))
            {
                var chunks = it.Select(v => (v[0].dtype.str, Chunk(v[0]))).ToList();
                chunks.Select(c => c.Item1).Should().Equal("<U4", "<U4", "<U4");
                chunks.Select(c => string.Join(",", c.Item2)).Should().Equal("a,b", "c,d", "e");
            }

            using var reversed = u["::-1"];
            using (var it = np.nditer(reversed, new[] { "buffered", "external_loop" }, op_dtypes: new[] { np.dtype("<U3") },
                       buffersize: 2))
                it.Select(v => string.Join(",", Chunk(v[0]))).Should().Equal("a,b", "c,d", "e");
        }

        /// <summary>A buffered cast under <c>order='C'</c> walks a reversed view in its logical order: <c>e d c b a</c>.</summary>
        [TestMethod]
        public void BufferedCast_COrderOverAReversedView()
        {
            using var u = U("a", "b", "c", "d", "e");
            using var reversed = u["::-1"];
            using var it = np.nditer(reversed, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U3") }, order: 'C');
            Walk(it).Should().Equal("e", "d", "c", "b", "a");
        }

        // ------------------------------------------------------------------ buffered requests

        /// <summary>A wider request widens through the buffer: <c>dtypes</c> and every view are <c>&lt;U5</c>.</summary>
        [TestMethod]
        public void WiderRequest_WidensThroughTheBuffer()
        {
            using var u = U("ab", "c");
            using var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U5") });
            it.dtypes.Select(d => d.str).Should().Equal("<U5");
            it.Select(v => (Text(v[0].GetAtIndex(0)), v[0].dtype.str)).Should().Equal(("ab", "<U5"), ("c", "<U5"));
        }

        /// <summary>
        ///     A narrower request truncates under <c>'same_kind'</c> (<c>abc de f</c> → <c>a d f</c>) and is refused under
        ///     <c>'safe'</c>, which may not lose characters.
        /// </summary>
        [TestMethod]
        public void NarrowerRequest_TruncatesUnderSameKind_RefusedUnderSafe()
        {
            using var u = U("abc", "de", "f");
            using (var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U1") }, casting: "same_kind"))
                Walk(it).Should().Equal("a", "d", "f");

            var safe = () => np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U1") }, casting: "safe");
            safe.Should().Throw<TypeError>()
                .WithMessage("Iterator operand 0 dtype could not be cast from dtype('<U3') to dtype('<U1') according to the rule 'safe'");
        }

        /// <summary>
        ///     <c>U</c> → <c>S</c> changes the class, so it needs <c>'unsafe'</c> (ASCII-encoded bytes); <c>'same_kind'</c> is
        ///     refused with NumPy's text, which prints the bytes dtype without its <c>|</c>.
        /// </summary>
        [TestMethod]
        public void UnicodeToBytes_NeedsUnsafe()
        {
            using var u = U("ab", "c");
            using (var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("|S2") }, casting: "unsafe"))
            {
                var raw = it.Select(v => v[0].GetAtIndex(0)).ToList();
                raw.Should().AllBeOfType<byte[]>();
                raw.Select(Text).Should().Equal("ab", "c");
            }

            var sameKind = () => np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("|S2") }, casting: "same_kind");
            sameKind.Should().Throw<TypeError>()
                    .WithMessage("Iterator operand 0 dtype could not be cast from dtype('<U2') to dtype('S2') according to the rule 'same_kind'");
        }

        /// <summary>
        ///     <c>S</c> → <c>U</c> decodes and is <c>'safe'</c> when the width holds every byte; a narrower <c>U</c> is refused
        ///     under <c>'safe'</c>.
        /// </summary>
        [TestMethod]
        public void BytesToUnicode_IsSafeWhenItFits()
        {
            using var s = S("ab", "c");
            using (var it = np.nditer(s, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U2") }, casting: "safe"))
            {
                var raw = it.Select(v => v[0].GetAtIndex(0)).ToList();
                raw.Should().AllBeOfType<string>();
                raw.Should().Equal("ab", "c");
            }

            using var s3 = S("abc", "d");
            var narrow = () => np.nditer(s3, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U2") }, casting: "safe");
            narrow.Should().Throw<TypeError>()
                  .WithMessage("Iterator operand 0 dtype could not be cast from dtype('S3') to dtype('<U2') according to the rule 'safe'");
        }

        /// <summary>
        ///     An UNSIZED request takes the width a cast from the operand picks: <c>'U'</c> over <c>|S3</c> is <c>&lt;U3</c>,
        ///     <c>'S'</c> over <c>&lt;U3</c> is <c>|S3</c>.
        /// </summary>
        [TestMethod]
        public void UnsizedRequests_TakeTheOperandsWidth()
        {
            using var s = S("abc", "d");
            using (var it = np.nditer(s, new[] { "buffered" }, op_dtypes: new[] { np.dtype("U") }))
            {
                it.dtypes.Select(d => d.str).Should().Equal("<U3");
                Walk(it).Should().Equal("abc", "d");
            }

            using var u = U("abc", "d");
            using (var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("S") }, casting: "unsafe"))
            {
                it.dtypes.Select(d => d.str).Should().Equal("|S3");
                Walk(it).Should().Equal("abc", "d");
            }
        }

        /// <summary>
        ///     The <c>S</c> class widens (<c>|S4</c> views) and truncates under <c>'same_kind'</c> (<c>abc de</c> →
        ///     <c>a d</c>) like the <c>U</c> class.
        /// </summary>
        [TestMethod]
        public void BytesRequests_WidenAndTruncate()
        {
            using var s = S("ab", "c");
            using (var it = np.nditer(s, new[] { "buffered" }, op_dtypes: new[] { np.dtype("|S4") }))
                it.Select(v => (Text(v[0].GetAtIndex(0)), v[0].dtype.str)).Should().Equal(("ab", "|S4"), ("c", "|S4"));

            using var s3 = S("abc", "de");
            using (var it = np.nditer(s3, new[] { "buffered" }, op_dtypes: new[] { np.dtype("|S1") }, casting: "same_kind"))
                Walk(it).Should().Equal("a", "d");
        }

        /// <summary>
        ///     A non-native request (<c>'&gt;U2'</c>) is reported as requested and reads the same texts (the buffer holds
        ///     native data, a documented NumSharp difference that the values do not show).
        /// </summary>
        [TestMethod]
        public void NonNativeRequest_IsReportedAsRequested()
        {
            using var u = U("ab", "c");
            using var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype(">U2") });
            it.dtypes.Select(d => d.str).Should().Equal(">U2");
            Walk(it).Should().Equal("ab", "c");
        }

        /// <summary>
        ///     The class conversions are ASCII only, with NumPy's codec errors: a non-ASCII character into <c>S</c> is a
        ///     UnicodeEncodeError, a byte ≥ 0x80 into <c>U</c> a UnicodeDecodeError.
        /// </summary>
        [TestMethod]
        public void NonAsciiConversions_RaiseNumPysCodecErrors()
        {
            using var u = U("ok", "é");
            var encode = () =>
            {
                using var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.dtype("|S2") }, casting: "unsafe");
                foreach (var _ in it) { }
            };
            encode.Should().Throw<UnicodeEncodeError>()
                  .WithMessage("'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");

            using var s = np.array(new object[] { Encoding.ASCII.GetBytes("ok"), new byte[] { 0xff } }, dtype: "S2");
            var decode = () =>
            {
                using var it = np.nditer(s, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U2") }, casting: "unsafe");
                foreach (var _ in it) { }
            };
            decode.Should().Throw<UnicodeDecodeError>()
                  .WithMessage("'ascii' codec can't decode byte 0xff in position 0: ordinal not in range(128)");
        }

        /// <summary>
        ///     Text ↔ number conversions through the iterator's buffers run NumPy's legacy casts (probed against NumPy 2.4.2):
        ///     an int64 operand requested as <c>'U'</c> reads <c>&lt;U21</c> text <c>'0' '1' '2'</c>; <c>'1' '22' '-3'</c>
        ///     requested as int64 reads 1, 22, -3; <c>common_dtype</c> over <c>&lt;U2</c> and int64 is <c>&lt;U21</c> for
        ///     both, the numbers read as text; and <c>it[0] = 5</c> into a <c>&lt;U2</c> operand stores <c>'5'</c>.
        /// </summary>
        [TestMethod]
        public void TextNumberConversions_FormatAndParse()
        {
            using (var a = np.arange(3))
            using (var it = np.nditer(a, new[] { "buffered" }, op_dtypes: new[] { np.dtype("U") }, casting: "unsafe"))
            {
                it.dtypes[0].str.Should().Be("<U21");
                Walk(it).Should().Equal("0", "1", "2");
            }

            using (var u = U("1", "22", "-3"))
            using (var it = np.nditer(u, new[] { "buffered" }, op_dtypes: new[] { np.int64 }, casting: "unsafe"))
                it.Select(v => (long)v[0].GetAtIndex(0)).ToList().Should().Equal(1L, 22L, -3L);

            using (var u = U("ab", "c"))
            using (var n = np.arange(2))
            using (var it = np.nditer(new[] { u, n }, new[] { "common_dtype", "buffered" }, casting: "unsafe"))
            {
                it.dtypes.Select(d => d.str).Should().Equal("<U21", "<U21");
                it.Select(v => Text(v[0].GetAtIndex(0)) + "|" + Text(v[1].GetAtIndex(0))).ToList().Should().Equal("ab|0", "c|1");
            }

            using var text = U("ab", "c");
            using (var writer = np.nditer(text, op_flags: new[] { "readwrite" }))
            using (var five = NDArray.Scalar(5))
                writer[0] = five;
            NDArray.AsStringArray(text).Should().Equal("5", "c");
        }

        // ------------------------------------------------------------------ refusals

        /// <summary>
        ///     A width change is a cast, and a cast needs a buffer or a copy: an unbuffered <c>&lt;U5</c> request over
        ///     <c>&lt;U2</c> is NumPy's "required copying or buffering" TypeError.
        /// </summary>
        [TestMethod]
        public void UnbufferedWidthChange_IsRefused()
        {
            using var u = U("ab", "c");
            var act = () => np.nditer(u, op_dtypes: new[] { np.dtype("<U5") });
            act.Should().Throw<TypeError>()
               .WithMessage("Iterator operand required copying or buffering, but neither copying nor buffering was enabled");
        }

        /// <summary>A broadcast string view is read-only: a <c>readwrite</c> walk over it is refused at construction.</summary>
        [TestMethod]
        public void ReadWriteOverABroadcastView_IsRefused()
        {
            using var row = U("x", "yz");
            using var broadcast = np.broadcast_to(row, new Shape(2, 2));
            var act = () => np.nditer(broadcast, op_flags: new[] { "readwrite" });
            act.Should().Throw<ValueError>().WithMessage("operand array with iterator write flag set is read-only");
        }

        /// <summary>
        ///     An <c>updateifcopy</c> temporary is written back, so its write-back cast must pass the rule: <c>&lt;U4</c> →
        ///     <c>&lt;U2</c> is not <c>'safe'</c>.
        /// </summary>
        [TestMethod]
        public void UpdateIfCopy_UnsafeWriteBack_IsRefused()
        {
            using var u = U("ab", "c");
            var act = () => np.nditer(u, op_flags: new[] { "readwrite", "updateifcopy" }, op_dtypes: new[] { np.dtype("<U4") },
                casting: "safe");
            act.Should().Throw<TypeError>()
               .WithMessage("Iterator requested dtype could not be cast from dtype('<U4') to dtype('<U2'), the operand 0 dtype, " +
                            "according to the rule 'safe'");
        }

        // ------------------------------------------------------------------ writes

        /// <summary>
        ///     A readwrite walk writes through its views, and a text longer than the element is TRUNCATED to the width
        ///     (<c>'TUVW'</c> into <c>&lt;U3</c> stores <c>'TUV'</c>) — unbuffered, and buffered across several buffers.
        /// </summary>
        [TestMethod]
        public void ReadWrite_WritesThroughAndTruncates()
        {
            var writes = new[] { "Q", "RS", "TUVW" };
            using (var u = U("ab", "c", "def"))
            {
                using (var it = np.nditer(u, op_flags: new[] { "readwrite" }))
                {
                    int i = 0;
                    foreach (var v in it)
                        Write(v[0], writes[i++]);
                }

                Elements(u).Should().Equal("Q", "RS", "TUV");
            }

            using (var u = U("ab", "c", "def"))
            {
                using (var it = np.nditer(u, new[] { "buffered" }, new[] { "readwrite" }, buffersize: 2))
                {
                    int i = 0;
                    foreach (var v in it)
                        Write(v[0], writes[i++]);
                }

                Elements(u).Should().Equal("Q", "RS", "TUV");
            }
        }

        /// <summary>
        ///     A WIDER buffered request holds the whole write in the buffer, and the write-back truncates it to the
        ///     operand's width: <c>'TUVWX'</c> through <c>&lt;U5</c> into <c>&lt;U3</c> stores <c>'TUV'</c>.
        /// </summary>
        [TestMethod]
        public void ReadWrite_WiderBuffer_TruncatesOnWriteBack()
        {
            using var u = U("ab", "c", "def");
            using (var it = np.nditer(u, new[] { "buffered" }, new[] { "readwrite" }, new[] { np.dtype("<U5") }, casting: "same_kind"))
            {
                var writes = new[] { "Q", "RS", "TUVWX" };
                int i = 0;
                foreach (var v in it)
                    Write(v[0], writes[i++]);
            }

            Elements(u).Should().Equal("Q", "RS", "TUV");
        }

        /// <summary>
        ///     A <c>U</c> buffer over an <c>S</c> operand encodes on the way back: writes <c>'XYZ'</c>, <c>'Q'</c> store
        ///     <c>b'XY'</c>, <c>b'Q'</c> (truncated to <c>|S2</c>); a non-ASCII write fails the write-back with NumPy's
        ///     UnicodeEncodeError.
        /// </summary>
        [TestMethod]
        public void ReadWrite_UnicodeBufferOverBytes_EncodesOnWriteBack()
        {
            using (var s = S("ab", "c"))
            {
                using (var it = np.nditer(s, new[] { "buffered" }, new[] { "readwrite" }, new[] { np.dtype("<U3") }, casting: "unsafe"))
                {
                    var writes = new[] { "XYZ", "Q" };
                    int i = 0;
                    foreach (var v in it)
                        Write(v[0], writes[i++]);
                }

                Elements(s).Should().Equal("XY", "Q");
            }

            using (var s = S("ab", "c"))
            {
                var act = () =>
                {
                    using var it = np.nditer(s, new[] { "buffered" }, new[] { "readwrite" }, new[] { np.dtype("<U3") }, casting: "unsafe");
                    var writes = new[] { "é", "Q" };
                    int i = 0;
                    foreach (var v in it)
                        Write(v[0], writes[i++]);
                };
                act.Should().Throw<UnicodeEncodeError>()
                   .WithMessage("'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");
            }
        }

        /// <summary>A <c>writeonly</c> walk stores every element without reading any: <c>X YY ZZZ</c>.</summary>
        [TestMethod]
        public void WriteOnly_StoresEveryElement()
        {
            using var u = U("ab", "c", "def");
            using (var it = np.nditer(u, op_flags: new[] { "writeonly" }))
            {
                var writes = new[] { "X", "YY", "ZZZ" };
                int i = 0;
                foreach (var v in it)
                    Write(v[0], writes[i++]);
            }

            Elements(u).Should().Equal("X", "YY", "ZZZ");
        }

        /// <summary>
        ///     The setter assigns text with the element's width: <c>'xy'</c>, <c>'toolong'</c> → <c>'too'</c>, and
        ///     <c>''</c>; a <c>U</c> element also takes a byte string (decoded).
        /// </summary>
        [TestMethod]
        public void Setter_AssignsTextTruncatedToTheWidth()
        {
            using (var u = U("ab", "c", "def"))
            {
                using (var it = np.nditer(u, op_flags: new[] { "readwrite" }))
                {
                    var writes = new[] { "xy", "toolong", "" };
                    int i = 0;
                    foreach (var _ in it)
                    {
                        using var value = np.array(writes[i++]);
                        it[0] = value;
                    }
                }

                Elements(u).Should().Equal("xy", "too", "");
            }

            using (var u = U("ab", "c"))
            {
                using (var it = np.nditer(u, op_flags: new[] { "readwrite" }))
                {
                    using var bytes = np.array("zz", np.dtype("S"));
                    it[0] = bytes;
                }

                Elements(u).Should().Equal("zz", "c");
            }
        }

        /// <summary>
        ///     The setter encodes text into an <c>S</c> element (<c>b'xy'</c>, <c>b'z'</c>), and refuses a non-ASCII text
        ///     with NumPy's UnicodeEncodeError.
        /// </summary>
        [TestMethod]
        public void Setter_EncodesTextIntoBytes()
        {
            using (var s = S("ab", "c"))
            {
                using (var it = np.nditer(s, op_flags: new[] { "readwrite" }))
                {
                    var writes = new[] { "xy", "z" };
                    int i = 0;
                    foreach (var _ in it)
                    {
                        using var value = np.array(writes[i++]);
                        it[0] = value;
                    }
                }

                var raw = Enumerable.Range(0, 2).Select(i => s.GetAtIndex(i)).ToList();
                raw.Should().AllBeOfType<byte[]>();
                raw.Select(Text).Should().Equal("xy", "z");
            }

            using (var s = S("ab", "c"))
            {
                using var it = np.nditer(s, op_flags: new[] { "readwrite" });
                using var e = np.array("é");
                var act = () => it[0] = e;
                act.Should().Throw<UnicodeEncodeError>()
                   .WithMessage("'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");
            }
        }

        /// <summary>
        ///     Written NULs follow NumPy's rule: a TRAILING NUL is padding (<c>'a\x00'</c> reads <c>'a'</c>), an inner or
        ///     leading one is text (<c>'\x00b'</c>, <c>'x\x00y'</c> keep their lengths 2 and 3).
        /// </summary>
        [TestMethod]
        public void WrittenNuls_TrailingIsPadding_InnerIsText()
        {
            using var u = U("ab", "c", "def");
            using (var it = np.nditer(u, op_flags: new[] { "readwrite" }))
            {
                var writes = new[] { "a\0", "\0b", "x\0y" };
                int i = 0;
                foreach (var v in it)
                    Write(v[0], writes[i++]);
            }

            Elements(u).Should().Equal("a", "\0b", "x\0y");
        }

        /// <summary>A read-only operand's string view is not writeable (NumPy's write flags).</summary>
        [TestMethod]
        public void ReadOnlyOperand_ViewIsNotWriteable()
        {
            using var u = U("ab", "c");
            using var it = np.nditer(u);
            it[0].flags.writeable.Should().BeFalse();
        }

        /// <summary>
        ///     <c>copy_if_overlap</c> over two views of one string buffer reads the ORIGINAL texts: shifting <c>u[:-1]</c>
        ///     into <c>u[1:]</c> gives <c>a a b c</c>.
        /// </summary>
        [TestMethod]
        public void CopyIfOverlap_ShiftsTheOriginalTexts()
        {
            using var u = U("a", "b", "c", "d");
            using (var it = np.nditer(new[] { u[":-1"], u["1:"] }, new[] { "copy_if_overlap" },
                       new[] { new[] { "readonly" }, new[] { "writeonly" } }))
            {
                foreach (var v in it)
                    np.copyto(v[1], v[0]);
            }

            Elements(u).Should().Equal("a", "a", "b", "c");
        }

        // ------------------------------------------------------------------ copy routes

        /// <summary>
        ///     A read-only <c>copy</c> operand is iterated as a temporary in the requested width: <c>operands[0]</c> is a new
        ///     <c>&lt;U5</c> array holding the same texts.
        /// </summary>
        [TestMethod]
        public void CopyRoute_IteratesAWiderTemporary()
        {
            using var u = U("ab", "c");
            using var it = np.nditer(u, op_flags: new[] { "readonly", "copy" }, op_dtypes: new[] { np.dtype("<U5") });
            var temp = it.operands[0];
            temp.dtype.str.Should().Be("<U5");
            Elements(temp).Should().Equal("ab", "c");
            ReferenceEquals(temp, u).Should().BeFalse();
            Walk(it).Should().Equal("ab", "c");
        }

        /// <summary>
        ///     An <c>updateifcopy</c> temporary (<c>&lt;U5</c> over <c>&lt;U2</c>) is written back at <c>close()</c>, not
        ///     before, truncated to the operand's width: <c>['ab', 'c']</c> until close, <c>['XY', 'Q']</c> after.
        /// </summary>
        [TestMethod]
        public void UpdateIfCopy_WritesBackAtCloseTruncated()
        {
            using var u = U("ab", "c");
            var it = np.nditer(u, op_flags: new[] { "readwrite", "updateifcopy" }, op_dtypes: new[] { np.dtype("<U5") },
                casting: "same_kind");
            try
            {
                var temp = it.operands[0];
                temp.dtype.str.Should().Be("<U5");
                ReferenceEquals(temp, u).Should().BeFalse();

                var writes = new[] { "XYZW", "Q" };
                int i = 0;
                foreach (var v in it)
                    Write(v[0], writes[i++]);
                Elements(u).Should().Equal("ab", "c");
            }
            finally
            {
                it.close();
            }

            Elements(u).Should().Equal("XY", "Q");
        }

        /// <summary>
        ///     A read-only 0-d <c>S</c> operand is copied into the requested <c>&lt;U2</c> without being asked (no buffer, no
        ///     <c>copy</c> flag): the walk reads <c>'ab'</c>.
        /// </summary>
        [TestMethod]
        public void ZeroDBytesOperand_IsCopiedForADecode()
        {
            using var s = np.array("ab", np.dtype("S"));
            using var it = np.nditer(s, op_dtypes: new[] { np.dtype("<U2") });
            it.dtypes.Select(d => d.str).Should().Equal("<U2");
            Walk(it).Should().Equal("ab");
        }

        // ------------------------------------------------------------------ allocated outputs

        /// <summary>
        ///     An output allocated without a request takes the input's <c>&lt;U2</c>, and copying each element fills it with
        ///     the same texts.
        /// </summary>
        [TestMethod]
        public void AllocatedOutput_TakesTheInputDtype()
        {
            using var u = U("ab", "c");
            using var it = np.nditer(new NDArray[] { u, null });
            foreach (var v in it)
                np.copyto(v[1], v[0], casting: "unsafe");
            it.dtypes.Select(d => d.str).Should().Equal("<U2", "<U2");
            var output = it.operands[1];
            output.dtype.str.Should().Be("<U2");
            Elements(output).Should().Equal("ab", "c");
        }

        /// <summary>
        ///     A sized request allocates that width: <c>&lt;U5</c> keeps every text, <c>&lt;U2</c> truncates <c>'cde'</c> to
        ///     <c>'cd'</c>, and <c>|S4</c> encodes them.
        /// </summary>
        [TestMethod]
        public void AllocatedOutput_TakesASizedRequest()
        {
            using var u = U("ab", "cde");
            using (var it = np.nditer(new NDArray[] { u, null }, op_dtypes: new[] { null, np.dtype("<U5") }))
            {
                foreach (var v in it)
                    np.copyto(v[1], v[0], casting: "unsafe");
                it.dtypes.Select(d => d.str).Should().Equal("<U3", "<U5");
                it.operands[1].dtype.str.Should().Be("<U5");
                Elements(it.operands[1]).Should().Equal("ab", "cde");
            }

            using (var it = np.nditer(new NDArray[] { u, null }, op_dtypes: new[] { null, np.dtype("<U2") }, casting: "unsafe"))
            {
                foreach (var v in it)
                    np.copyto(v[1], v[0], casting: "unsafe");
                it.operands[1].dtype.str.Should().Be("<U2");
                Elements(it.operands[1]).Should().Equal("ab", "cd");
            }

            using (var it = np.nditer(new NDArray[] { u, null }, op_dtypes: new[] { null, np.dtype("|S4") }, casting: "unsafe"))
            {
                foreach (var v in it)
                    np.copyto(v[1], v[0], casting: "unsafe");
                it.dtypes.Select(d => d.str).Should().Equal("<U3", "|S4");
                it.operands[1].dtype.str.Should().Be("|S4");
                Elements(it.operands[1]).Should().Equal("ab", "cde");
            }
        }

        /// <summary>
        ///     An UNSIZED request for an allocated output is reported as requested (<c>&lt;U0</c>) while the array is
        ///     allocated with one character (<c>&lt;U1</c>), as in NumPy.
        /// </summary>
        [TestMethod]
        public void AllocatedOutput_UnsizedRequest_IsOneCharacterWide()
        {
            using var u = U("ab", "cde");
            using var it = np.nditer(new NDArray[] { u, null }, op_dtypes: new[] { null, np.dtype("U") });
            it.dtypes.Select(d => d.str).Should().Equal("<U3", "<U0");
            it.operands[1].dtype.str.Should().Be("<U1");
        }

        // ------------------------------------------------------------------ operand pairs

        /// <summary>
        ///     Two string operands without <c>common_dtype</c> keep their own dtypes (<c>&lt;U2</c>, <c>|S2</c>), and
        ///     <c>value</c> holds one view of each.
        /// </summary>
        [TestMethod]
        public void Pair_KeepsEachDtype()
        {
            using var u = U("ab", "c");
            using var s = S("x", "yz");
            using var it = np.nditer(new[] { u, s });
            it.dtypes.Select(d => d.str).Should().Equal("<U2", "|S2");
            it.value.Select(x => (x.dtype.str, Text(x.GetAtIndex(0)))).Should().Equal(("<U2", "ab"), ("|S2", "x"));
            it.Select(v => (Text(v[0].GetAtIndex(0)), Text(v[1].GetAtIndex(0)))).Should().Equal(("ab", "x"), ("c", "yz"));
        }

        /// <summary>
        ///     <c>common_dtype</c> with a buffer converts both to their result type: <c>&lt;U2</c> for a <c>U</c> and an
        ///     <c>S</c> operand (the bytes decoded), <c>&lt;U5</c> for two widths. Without a buffer the conversion is refused.
        /// </summary>
        [TestMethod]
        public void Pair_CommonDtype_NeedsABuffer()
        {
            using var u = U("ab", "c");
            using var s = S("x", "yz");
            using (var it = np.nditer(new[] { u, s }, new[] { "common_dtype", "buffered" }))
            {
                it.dtypes.Select(d => d.str).Should().Equal("<U2", "<U2");
                it.Select(v => (v[1].GetAtIndex(0) is string, Text(v[0].GetAtIndex(0)), Text(v[1].GetAtIndex(0)))).Should()
                  .Equal((true, "ab", "x"), (true, "c", "yz"));
            }

            using var wide = U("vwxyz", "q");
            using (var it = np.nditer(new[] { u, wide }, new[] { "common_dtype", "buffered" }))
            {
                it.dtypes.Select(d => d.str).Should().Equal("<U5", "<U5");
                it.Select(v => (Text(v[0].GetAtIndex(0)), Text(v[1].GetAtIndex(0)))).Should().Equal(("ab", "vwxyz"), ("c", "q"));
            }

            var unbuffered = () => np.nditer(new[] { u, s }, new[] { "common_dtype" });
            unbuffered.Should().Throw<TypeError>()
                      .WithMessage("Iterator operand required copying or buffering, but neither copying nor buffering was enabled");
        }

        /// <summary>A string operand next to a bool operand keeps both dtypes: <c>(ab, True)</c>, <c>(c, False)</c>.</summary>
        [TestMethod]
        public void Pair_TextAndBool_KeepTheirDtypes()
        {
            using var u = U("ab", "c");
            using var flags = np.array(new[] { true, false });
            using var it = np.nditer(new[] { u, flags });
            it.dtypes.Select(d => d.str).Should().Equal("<U2", "|b1");
            it.Select(v => (Text(v[0].GetAtIndex(0)), (bool)v[1].GetAtIndex(0))).Should().Equal(("ab", true), ("c", false));
        }

        /// <summary>Two string operands broadcast like numbers: a row of 2 against a column of 3 walks six pairs.</summary>
        [TestMethod]
        public void Pair_Broadcasts()
        {
            using var row = U("x", "yy");
            using var column = U("p", "q", "r").reshape(3, 1);
            using var it = np.nditer(new[] { row, column });
            it.Select(v => $"{Text(v[0].GetAtIndex(0))}/{Text(v[1].GetAtIndex(0))}").Should()
              .Equal("x/p", "yy/p", "x/q", "yy/q", "x/r", "yy/r");
        }

        // ------------------------------------------------------------------ op_axes, itershape and the construction checks

        /// <summary>
        ///     <c>op_axes</c> remaps a string operand like a numeric one: <c>[1, 0]</c> over a <c>(2, 3)</c> <c>U</c> array
        ///     iterates its transpose, and the K-order walk still follows memory (the remapped axis 0 is the innermost), so
        ///     the multi-index runs <c>(0,0) (1,0) (2,0) (0,1) …</c> over <c>a b c d e f</c>.
        /// </summary>
        [TestMethod]
        public void OpAxes_RemapAStringOperand()
        {
            using var grid = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using var it = np.nditer(grid, new[] { "multi_index" }, op_axes: new[] { new[] { 1, 0 } });
            var seen = new List<string>();
            foreach (var v in it)
                seen.Add($"{string.Join(",", it.multi_index)}:{Text(v[0].GetAtIndex(0))}");
            seen.Should().Equal("0,0:a", "1,0:b", "2,0:c", "0,1:d", "1,1:e", "2,1:f");
        }

        /// <summary>
        ///     A negative <c>itershape</c> entry is deduced from a string operand (<c>(2, -1)</c> over <c>['ab', 'cd', 'ef']</c>
        ///     walks the row twice as <c>&lt;U2</c>), and an allocated output takes the input's string dtype at the deduced
        ///     shape.
        /// </summary>
        [TestMethod]
        public void Itershape_DeducedOverAStringOperand()
        {
            using var row = U("ab", "cd", "ef");
            using (var it = np.nditer(row, itershape: new long[] { 2, -1 }))
            {
                it.dtypes.Select(d => d.str).Should().Equal("<U2");
                Walk(it).Should().Equal("ab", "cd", "ef", "ab", "cd", "ef");
            }

            using var letters = U("a", "b", "c");
            using var withOutput = np.nditer(new NDArray[] { letters, null },
                op_flags: new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, itershape: new long[] { 2, -1 });
            using var output = withOutput.operands[1];
            output.shape.Should().Equal(2L, 3L);
            output.dtype.str.Should().Be("<U1");
        }

        /// <summary>
        ///     The construction refusals are dtype-free and keep NumPy's texts for string operands: a <c>U</c> and an <c>S</c>
        ///     operand that do not broadcast, a <c>no_broadcast</c> <c>U</c> row against a <c>(2, 2)</c> <c>S</c> grid, an
        ///     <c>itershape</c> the operands do not fit (the remapped text, then the requested shape) and a duplicate
        ///     <c>op_axes</c> entry.
        /// </summary>
        [TestMethod]
        public void ConstructionRefusals_AreNumPysForStringOperands()
        {
            var mismatch = () => np.nditer(new[] { U("ab", "c"), S("x", "yz", "w") });
            mismatch.Should().Throw<ValueError>().WithMessage("operands could not be broadcast together with shapes (2,) (3,) ");

            var stretched = () => np.nditer(new[] { U("ab", "c"), S("x", "yz", "p", "q").reshape(2, 2) },
                op_flags: new[] { new[] { "readonly", "no_broadcast" }, new[] { "readonly" } });
            stretched.Should().Throw<ValueError>()
                     .WithMessage("non-broadcastable operand with shape (2,) doesn't match the broadcast shape (2,2)");

            var requested = () => np.nditer(new[] { U("ab", "c"), S("x", "yz") }, op_axes: new[] { new[] { 0 }, new[] { 0 } },
                itershape: new long[] { 3 });
            requested.Should().Throw<ValueError>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (2,)->(2,) (2,)->(2,)  " +
                "and requested shape (3,)");

            var duplicate = () => np.nditer(U("a", "b", "c", "d").reshape(2, 2), op_axes: new[] { new[] { 1, 1 } });
            duplicate.Should().Throw<ValueError>()
                     .WithMessage("The 'op_axes' provided to the iterator constructor for operand 0 contained duplicate value 1");
        }

        // ------------------------------------------------------------------ 0-d and empty

        /// <summary>
        ///     A 0-d string array is one element with no axes, and under <c>external_loop</c> one chunk of shape <c>(1,)</c>.
        /// </summary>
        [TestMethod]
        public void ZeroD_IsOneElement()
        {
            using var u = np.array("abc");
            using (var it = np.nditer(u))
            {
                it.ndim.Should().Be(0);
                it.itersize.Should().Be(1);
                it.dtypes.Select(d => d.str).Should().Equal("<U3");
                Walk(it).Should().Equal("abc");
            }

            using (var it = np.nditer(u, new[] { "external_loop" }))
            {
                var chunks = it.Select(v => (v[0].shape, Chunk(v[0]))).ToList();
                chunks.Should().HaveCount(1);
                chunks[0].Item1.Should().Equal(1L);
                chunks[0].Item2.Should().Equal("abc");
            }
        }

        /// <summary>
        ///     A zero-size string array under <c>zerosize_ok</c> keeps its dtype, is finished from the start and walks
        ///     nothing, element by element or by chunks.
        /// </summary>
        [TestMethod]
        public void ZeroSize_IsFinishedFromTheStart()
        {
            using var u = np.zeros(new Shape(0, 3), np.dtype("<U3"));
            using (var it = np.nditer(u, new[] { "zerosize_ok" }))
            {
                it.itersize.Should().Be(0);
                it.finished.Should().BeTrue();
                it.dtypes.Select(d => d.str).Should().Equal("<U3");
                Walk(it).Should().BeEmpty();
            }

            using var flat = np.zeros(new Shape(0), np.dtype("<U2"));
            using (var it = np.nditer(flat, new[] { "zerosize_ok", "external_loop" }))
                it.Should().BeEmpty();
        }

        // ------------------------------------------------------------------ the protocol over strings

        /// <summary>
        ///     The single-operand protocol reads strings like numbers: <c>value</c> is the 0-d <c>&lt;U2</c> view,
        ///     <c>operands[0]</c> the caller's array itself, <c>shape</c> without a multi-index the coalesced <c>(6,)</c>, and a
        ///     <c>while (!finished)</c> loop over <c>iternext()</c> visits every element.
        /// </summary>
        [TestMethod]
        public void Protocol_ValueOperandsShapeAndIternext()
        {
            using var u = U("ab", "c");
            using (var it = np.nditer(u))
            {
                var value = it.value[0];
                value.dtype.str.Should().Be("<U2");
                Text(value.GetAtIndex(0)).Should().Be("ab");
                ReferenceEquals(it.operands[0], u).Should().BeTrue();
            }

            using var grid = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using (var it = np.nditer(grid))
            {
                it.shape.Should().Equal(6L);
                it.ndim.Should().Be(1);
                it.itersize.Should().Be(6);
            }

            using var abc = U("a", "b", "c");
            using (var it = np.nditer(abc))
            {
                var seen = new List<string>();
                while (!it.finished)
                {
                    seen.Add(Text(it[0].GetAtIndex(0)));
                    it.iternext();
                }

                seen.Should().Equal("a", "b", "c");
            }
        }

        /// <summary>
        ///     Seeks land on string elements: <c>multi_index = (1, 0)</c> walks <c>d e f</c>, an <c>f_index</c> of 1 walks
        ///     <c>d e f</c> with indices <c>1 3 5</c>, and a buffered <c>iterindex = 3</c> refills the window from <c>d</c>.
        /// </summary>
        [TestMethod]
        public void Seeks_LandOnTheElement()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using (var it = np.nditer(u, new[] { "multi_index" }))
            {
                it.multi_index = new long[] { 1, 0 };
                it.Select(v => $"{string.Join(",", it.multi_index)}:{Text(v[0].GetAtIndex(0))}").Should()
                  .Equal("1,0:d", "1,1:e", "1,2:f");
            }

            using (var it = np.nditer(u, new[] { "f_index" }))
            {
                it.index = 1;
                it.Select(v => $"{it.index}:{Text(v[0].GetAtIndex(0))}").Should().Equal("1:d", "3:e", "5:f");
            }

            using var five = U("a", "b", "c", "d", "e");
            using (var it = np.nditer(five, new[] { "buffered" }, op_dtypes: new[] { np.dtype("<U3") }, buffersize: 2))
            {
                it.iterindex = 3;
                Walk(it).Should().Equal("d", "e");
            }
        }

        /// <summary>
        ///     A buffered range walks only its positions (<c>[1, 4)</c> is <c>b c d</c>); an empty range is finished at once.
        /// </summary>
        [TestMethod]
        public void Ranged_WalksTheSubRange()
        {
            using var u = U("a", "b", "c", "d", "e", "f");
            using var it = np.nditer(u, new[] { "buffered", "ranged" }, op_dtypes: new[] { np.dtype("<U5") }, casting: "same_kind");
            it.iterrange = (1, 4);
            Walk(it).Should().Equal("b", "c", "d");
            it.iterrange = (5, 5);
            it.finished.Should().BeTrue();
            Walk(it).Should().BeEmpty();
        }

        /// <summary>A copy taken after one pull resumes with the second element, independently of the original.</summary>
        [TestMethod]
        public void Copy_MidWalk_ResumesWithTheNextElement()
        {
            using var u = U("a", "b", "c", "d");
            using var it = np.nditer(u);
            it.MoveNext();
            using var copy = it.copy();
            Walk(copy).Should().Equal("b", "c", "d");
            Walk(it).Should().Equal("b", "c", "d");
        }

        /// <summary><c>itviews</c> exposes a string operand in the iterator's axis order, keeping its <c>&lt;U2</c> dtype.</summary>
        [TestMethod]
        public void ItViews_KeepTheDtype()
        {
            using var u = U("ab", "c", "d", "e").reshape(2, 2);
            using var it = np.nditer(u, new[] { "multi_index" });
            var view = it.itviews[0];
            view.shape.Should().Equal(2L, 2L);
            view.dtype.str.Should().Be("<U2");
            Elements(view).Should().Equal("ab", "c", "d", "e");
        }

        /// <summary>
        ///     <c>remove_axis(0)</c> walks the remaining axis at row 0 (<c>a b c</c>, shape <c>(3,)</c>), and
        ///     <c>remove_multi_index</c> coalesces the array into one axis of 6.
        /// </summary>
        [TestMethod]
        public void RemoveAxis_AndRemoveMultiIndex()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            using (var it = np.nditer(u, new[] { "multi_index" }))
            {
                it.remove_axis(0);
                it.shape.Should().Equal(3L);
                it.Select(v => $"{string.Join(",", it.multi_index)}:{Text(v[0].GetAtIndex(0))}").Should().Equal("0:a", "1:b", "2:c");
            }

            using (var it = np.nditer(u, new[] { "multi_index" }))
            {
                it.remove_multi_index();
                it.shape.Should().Equal(6L);
                Walk(it).Should().Equal("a", "b", "c", "d", "e", "f");
            }
        }

        // ------------------------------------------------------------------ np.nested_iters

        /// <summary>
        ///     <c>np.nested_iters</c> over a string array: the outer level walks the rows, the inner level each row's texts.
        /// </summary>
        [TestMethod]
        public void NestedIters_WalkRowsThenColumns()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            var its = np.nested_iters(u, new[] { new[] { 0 }, new[] { 1 } });
            try
            {
                var rows = new List<string>();
                foreach (var _ in its[0])
                    rows.Add(string.Join(",", its[1].Select(v => Text(v[0].GetAtIndex(0)))));
                rows.Should().Equal("a,b,c", "d,e,f");
            }
            finally
            {
                foreach (var it in its)
                    it.Dispose();
            }
        }

        /// <summary>
        ///     A buffered width request in <c>np.nested_iters</c> lands in the innermost level: with the outer level on axis 1,
        ///     each column is read as <c>&lt;U5</c> views (<c>a d</c>, <c>b e</c>, <c>c f</c>).
        /// </summary>
        [TestMethod]
        public void NestedIters_BufferedRequest_ReachesTheInnerLevel()
        {
            using var u = U("a", "b", "c", "d", "e", "f").reshape(2, 3);
            var its = np.nested_iters(u, new[] { new[] { 1 }, new[] { 0 } }, new[] { "buffered" },
                op_dtypes: new[] { np.dtype("<U5") });
            try
            {
                var rows = new List<string>();
                foreach (var _ in its[0])
                    rows.Add(string.Join(",", its[1].Select(v => $"{Text(v[0].GetAtIndex(0))}{v[0].dtype.str}")));
                rows.Should().Equal("a<U5,d<U5", "b<U5,e<U5", "c<U5,f<U5");
            }
            finally
            {
                foreach (var it in its)
                    it.Dispose();
            }
        }
    }
}
