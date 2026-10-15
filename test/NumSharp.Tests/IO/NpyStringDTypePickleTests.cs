using System;
using System.IO;
using System.Linq;
using System.Text;
using NumSharp.Backends.Strings;
using NumSharp.IO;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     What the committed <c>.npy</c> corpus cannot express about the <c>StringDType</c> pickle (decision D3): several
    ///     pickled arrays in one stream, a header whose shape the pickle overrides, the refusal of every other pickle (an
    ///     object array, a call to an arbitrary global — never executed —, protocol 0, a dict), truncation, the
    ///     missing-data objects the writer refuses, a NaN missing-data object's bits, the allocators' lifetime, a text that
    ///     is not UTF-8 (NumPy's <c>Unpickling a python object failed</c> wrapper), a frame length that claims a terabyte
    ///     (from a seekable and a non-seekable stream), and CPython's memo length across a sparse <c>LONG_BINPUT</c>.
    ///     Expected values were probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class NpyStringDTypePickleTests
    {
        /// <summary>The refusal every non-StringDType pickle starts with.</summary>
        private const string OnlyStringDType = "NumSharp reads only the pickle NumPy writes for a StringDType array*";

        /// <summary>
        ///     Arrays saved one after another into one stream load one after another: the reader stops exactly at each
        ///     pickle's STOP (frames are read whole, as their lengths say), at NumPy's positions 319 / 471 / 769, and a fourth
        ///     load reports the end of the stream.
        /// </summary>
        [TestMethod]
        public void Stream_SeveralArrays_EachLoadStopsAtItsPickle()
        {
            using var a1 = np.array(new[] { "a", "bcdefghijklmnopqrs" }, np.dtypes.StringDType());
            using var a2 = np.arange(3.0);
            using var a3 = np.array(new object[] { "z" }, np.dtypes.StringDType(na_object: null));
            using var ms = new MemoryStream();
            foreach (var a in new[] { a1, a2, a3 })
                np.save(ms, a);
            ms.Position = 0;
            var positions = new long[3];
            for (int i = 0; i < 3; i++)
            {
                using var loaded = (NDArray)np.load(ms, allow_pickle: true);
                positions[i] = ms.Position;
                if (i == 0)
                    NDArray.AsStringArray(loaded).Should().Equal("a", "bcdefghijklmnopqrs");
                if (i == 1)
                    loaded.ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
                if (i == 2)
                    loaded.dtype.Should().Be(np.dtypes.StringDType(na_object: null));
            }
            positions.Should().Equal(319, 471, 769);
            Action fourth = () => np.load(ms, allow_pickle: true);
            fourth.Should().Throw<EndOfStreamException>().WithMessage("No data left in file");
        }

        /// <summary>
        ///     NumPy's <c>read_array</c> ignores the header's shape and order for an object array and returns what the pickle
        ///     holds: a header claiming <c>(9,)</c> over a pickled <c>(2,)</c> array still loads <c>(2,)</c>.
        /// </summary>
        [TestMethod]
        public void HeaderShape_IsIgnoredForAPickle()
        {
            using var a = np.array(new[] { "a", "bcdefghijklmnopqrs" }, np.dtypes.StringDType());
            byte[] raw = np.save(a);
            string text = Encoding.Latin1.GetString(raw);
            text.Should().Contain("'shape': (2,)");
            byte[] patched = Encoding.Latin1.GetBytes(text.Replace("'shape': (2,)", "'shape': (9,)"));
            using var loaded = (NDArray)np.load(patched, allow_pickle: true);
            loaded.shape.Should().Equal(2L);
            NDArray.AsStringArray(loaded).Should().Equal("a", "bcdefghijklmnopqrs");
        }

        /// <summary>
        ///     A NumPy object array (<c>np.array([{'a': 1}], dtype=object)</c>, the corpus case <c>dtype_object</c>) is refused
        ///     with <c>allow_pickle</c> too: NumSharp has no object dtype, and its reader stops at the dict opcode.
        /// </summary>
        [TestMethod]
        public void ObjectArray_IsRefusedEvenWithAllowPickle()
        {
            NpyCase c = NpyOracleCorpus.Cases.Single(x => x.Name == "dtype_object");
            Action load = () => np.load(c.Bytes, allow_pickle: true);
            load.Should().Throw<NotSupportedException>().WithMessage(OnlyStringDType);
        }

        /// <summary>
        ///     The reader never executes what a pickle names: a pickle that calls <c>os.system('echo hi')</c> (a valid
        ///     protocol-4 stream) is parsed into an inert record and refused by name.
        /// </summary>
        [TestMethod]
        public void ArbitraryGlobal_IsNeverCalled()
        {
            var b = new MemoryStream();
            b.WriteByte(0x80); b.WriteByte(4);                                    // PROTO 4
            WriteShortText(b, "os"); WriteShortText(b, "system");
            b.WriteByte(0x93);                                                    // STACK_GLOBAL
            WriteShortText(b, "echo hi");
            b.WriteByte(0x85);                                                    // TUPLE1
            b.WriteByte(0x52);                                                    // REDUCE
            b.WriteByte(0x2E);                                                    // STOP
            Action load = () => np.load(b.ToArray(), allow_pickle: true);
            load.Should().Throw<NotSupportedException>().WithMessage("*a call to the global os.system*");
        }

        /// <summary>Other pickles NumPy can read and NumSharp refuses by name: a dict, and a text-mode protocol 0 stream.</summary>
        [TestMethod]
        public void OtherPickles_AreRefused()
        {
            Action dict = () => np.load(new byte[] { 0x80, 4, 0x7D, 0x94, 0x2E }, allow_pickle: true);   // {}
            dict.Should().Throw<NotSupportedException>().WithMessage(OnlyStringDType);
            byte[] protocol0 = Encoding.ASCII.GetBytes("cnumpy._core.multiarray\n_reconstruct\np0\n");
            Action p0 = () => np.load(protocol0, allow_pickle: true);
            p0.Should().Throw<NotSupportedException>().WithMessage(OnlyStringDType);
            byte[] protocol1Header = { 0x80, 1, 0x4E, 0x2E };
            Action p1 = () => np.load(protocol1Header, allow_pickle: true);
            p1.Should().Throw<NotSupportedException>().WithMessage("NumSharp reads pickle protocols 2 to 5, not protocol 1.");
        }

        /// <summary>A truncated pickle is a <see cref="FormatException"/>, at any cut (inside a frame or before it).</summary>
        [TestMethod]
        public void Truncated_IsFormatException()
        {
            using var a = np.array(new[] { "a", "bcdefghijklmnopqrs", "c" }, np.dtypes.StringDType());
            byte[] raw = np.save(a);
            foreach (int cut in new[] { raw.Length - 1, raw.Length - 20, 140, 135 })
            {
                Action load = () => np.load(raw.Take(cut).ToArray(), allow_pickle: true);
                load.Should().Throw<FormatException>($"cut at {cut} of {raw.Length}");
            }
        }

        /// <summary>
        ///     The writer refuses a missing-data object it cannot pickle — a <see cref="Half"/> (NumPy's <c>np.float16</c>,
        ///     whose pickle names <c>numpy._core.multiarray.scalar</c>) or a <see cref="decimal"/> — after the header, as a
        ///     failing pickle leaves NumPy's file.
        /// </summary>
        [TestMethod]
        public void UnpicklableNAObject_IsRefusedOnSave()
        {
            foreach (object na in new object[] { Half.NaN, 1.5m })
            {
                using var a = np.array(new object[] { "x" }, np.dtypes.StringDType(na));
                using var ms = new MemoryStream();
                Action save = () => np.save(ms, a);
                save.Should().Throw<NotSupportedException>().WithMessage($"*missing-data object of type {na.GetType().Name}*");
                Encoding.Latin1.GetString(ms.ToArray()).Should().Contain("'descr': '|O'");
            }
        }

        /// <summary>
        ///     A NaN missing-data object travels by its bits (BINFLOAT is the double's 8 bytes): a payload NaN round-trips with
        ///     its payload, and NumPy's <c>np.nan</c> (<c>0x7FF8000000000000</c>) loads as itself.
        /// </summary>
        [TestMethod]
        public void NaNMissingDataObject_KeepsItsBits()
        {
            double payload = BitConverter.Int64BitsToDouble(0x7FF8000000000ABCL);
            using var a = np.array(new object[] { "x", payload }, np.dtypes.StringDType(payload));
            using var loaded = (NDArray)np.load(np.save(a), allow_pickle: true);
            BitConverter.DoubleToInt64Bits((double)loaded.dtype.VString.na_object).Should().Be(0x7FF8000000000ABCL);
            loaded.size.Should().Be(2);
            loaded.Storage.GetAtIndex(0).Should().Be("x");
            loaded.Storage.GetAtIndex(1).Should().BeOfType<double>();
        }

        /// <summary>Every loaded array's allocator is freed with the array: the live allocator count returns to its baseline.</summary>
        /// <remarks>
        ///     The count is process-wide, so another test's undisposed <c>StringDType</c> arrays lower it whenever a
        ///     collection finalizes them; a window that a collection disturbed is measured again, never compared
        ///     (<see cref="GcQuiescence"/>). Measured unguarded, the test failed whenever the string tests ran before it in
        ///     one process (3 live allocators at the baseline, 1 after the loop).
        /// </remarks>
        [TestMethod]
        public void LoadedArrays_ReleaseTheirAllocators()
        {
            using var a = np.array(new[] { "a", "bcdefghijklmnopqrs", "c" }, np.dtypes.StringDType());
            byte[] raw = np.save(a);
            for (int attempt = 0; ; attempt++)
            {
                GcQuiescence.CollectYoung();
                var window = GcQuiescence.OpenWindow();
                long baseline = VStringAllocator.LiveAllocators;
                for (int i = 0; i < 50; i++)
                {
                    using var loaded = (NDArray)np.load(raw, allow_pickle: true);
                    loaded.size.Should().Be(3);
                }
                long after = VStringAllocator.LiveAllocators;
                if (GcQuiescence.Undisturbed(window))
                {
                    after.Should().Be(baseline, "every loaded array's allocator is freed with the array");
                    return;
                }
                if (attempt == 4)
                    Assert.Inconclusive("Every measurement window was disturbed by a collection.");
            }
        }

        /// <summary>
        ///     A pickled text that is not UTF-8 fails as NumPy's <c>np.load</c> fails: <c>read_array</c> wraps the unpickler's
        ///     <c>UnicodeDecodeError</c> in a <c>UnicodeError</c> that prints its repr (probed against NumPy 2.4.2). A
        ///     surrogate, which CPython's <c>surrogatepass</c> decode lets through and NumPy then fails on with a
        ///     <c>SystemError</c> (a NumPy bug), is refused the same way, with the strict decoder's error.
        /// </summary>
        [TestMethod]
        public void NonUtf8Text_IsNumPysUnpicklingError()
        {
            using var a = np.array(new[] { "abc", "a long text over fifteen bytes" }, np.dtypes.StringDType());
            byte[] raw = np.save(a);
            int at = IndexOf(raw, new byte[] { 0x8C, 3, (byte)'a', (byte)'b', (byte)'c' }) + 2;

            byte[] bad = (byte[])raw.Clone();
            bad[at] = 0xFF; bad[at + 1] = 0xFE; bad[at + 2] = 0xFD;
            Action load = () => np.load(bad, allow_pickle: true);
            var e = load.Should().Throw<UnicodeError>().Which;
            e.GetType().Should().Be(typeof(UnicodeError), "NumPy raises UnicodeError itself, from the decode error");
            e.Message.Should().Be("Unpickling a python object failed: UnicodeDecodeError('utf-8', b'\\xff\\xfe\\xfd', 0, 1, " +
                                  "'invalid start byte')\nYou may need to pass the encoding= option to numpy.load");
            e.InnerException.Should().BeOfType<UnicodeDecodeError>()
             .Which.Message.Should().Be("'utf-8' codec can't decode byte 0xff in position 0: invalid start byte");

            byte[] surrogate = (byte[])raw.Clone();
            surrogate[at] = 0xED; surrogate[at + 1] = 0xA0; surrogate[at + 2] = 0x80;
            Action loadSurrogate = () => np.load(surrogate, allow_pickle: true);
            loadSurrogate.Should().Throw<UnicodeError>().WithMessage(
                "Unpickling a python object failed: UnicodeDecodeError('utf-8', b'\\xed\\xa0\\x80', 0, 1, 'invalid continuation byte')*");
        }

        /// <summary>
        ///     A frame whose length claims a terabyte in a file of a few hundred bytes fails as a truncated pickle, from a
        ///     seekable and from a non-seekable stream: the reader grows its buffer only by what the stream delivers, so it
        ///     never asks for the claim (which would be an <see cref="OutOfMemoryException"/>).
        /// </summary>
        [TestMethod]
        public void HostileFrameLength_FailsWithoutAllocatingTheClaim()
        {
            using var a = np.array(new[] { "a", "bcdefghijklmnopqrs" }, np.dtypes.StringDType());
            byte[] raw = np.save(a);
            // The pickle starts after the 128-byte header with PROTO 4 and the first FRAME: its 8-byte length follows.
            raw[128].Should().Be(0x80);
            raw[130].Should().Be(0x95);
            BitConverter.TryWriteBytes(raw.AsSpan(131, 8), 1L << 40).Should().BeTrue();

            Action seekable = () => np.load(raw, allow_pickle: true);
            seekable.Should().Throw<FormatException>().WithMessage("EOF: reading pickle frame, expected 1099511627776 bytes got *");
            // np.load needs a seekable stream to detect the file type; np.load_npy reads a .npy straight from any stream.
            Action streamed = () => np.load_npy(new NonSeekableStream(raw), allow_pickle: true);
            streamed.Should().Throw<FormatException>().WithMessage("EOF: reading pickle frame, expected 1099511627776 bytes got *");
        }

        /// <summary>
        ///     The memo follows CPython's: a <c>LONG_BINPUT</c> far past the filled slots (index 100000, which the reader keeps
        ///     outside its memo array) counts towards <c>memo_len</c>, so the next <c>MEMOIZE</c> takes index 1, and both come
        ///     back through <c>LONG_BINGET</c> / <c>BINGET</c>; a <c>None</c> element stores <c>'None'</c>. A hand-made pickle
        ///     NumPy 2.4.2 loads as <c>array(['a', 'b', 'a', 'b', 'None'], dtype=StringDType())</c>.
        /// </summary>
        [TestMethod]
        public void SparseMemo_FollowsCPythonsMemoLength()
        {
            const string hex =
                "934e554d5059010076007b276465736372273a20277c4f272c2027666f727472616e5f6f72646572273a2046616c73652c202773" +
                "68617065273a2028352c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020" +
                "20202020202020202020202020202020202020202020200a80048c166e756d70792e5f636f72652e6d756c746961727261798c0c" +
                "5f7265636f6e737472756374938c056e756d70798c076e646172726179934b00854301628752284b014b05858c156e756d70792e" +
                "5f636f72652e5f696e7465726e616c8c1e5f636f6e766572745f746f5f737472696e6764747970655f6b7761726773934b018552" +
                "895d288c016172a08601008c0162946aa086010068014e6574622e";
            using var loaded = (NDArray)np.load(Convert.FromHexString(hex), allow_pickle: true);
            loaded.dtype.Should().Be(np.dtypes.StringDType());
            NDArray.AsStringArray(loaded).Should().Equal("a", "b", "a", "b", "None");
        }

        /// <summary>The index of the first occurrence of <paramref name="pattern"/> in <paramref name="data"/>.</summary>
        /// <param name="data">The bytes searched.</param>
        /// <param name="pattern">The bytes looked for.</param>
        /// <returns>The index (the test fails when the pattern is missing).</returns>
        private static int IndexOf(byte[] data, byte[] pattern)
        {
            int i = data.AsSpan().IndexOf(pattern);
            i.Should().BeGreaterThanOrEqualTo(0, "the pattern must be in the saved bytes");
            return i;
        }

        /// <summary>A SHORT_BINUNICODE opcode with its text.</summary>
        /// <param name="s">The output.</param>
        /// <param name="text">An ASCII text.</param>
        private static void WriteShortText(Stream s, string text)
        {
            s.WriteByte(0x8C);
            s.WriteByte((byte)text.Length);
            s.Write(Encoding.ASCII.GetBytes(text));
        }

        /// <summary>A read-only stream over bytes that cannot seek (no <c>Length</c>), as a network or pipe stream.</summary>
        private sealed class NonSeekableStream : Stream
        {
            private readonly MemoryStream _inner;

            /// <summary>Wraps <paramref name="data"/>.</summary>
            /// <param name="data">The bytes the stream delivers.</param>
            internal NonSeekableStream(byte[] data) => _inner = new MemoryStream(data, writable: false);

            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
            public override void Flush() { }
            public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inner.Dispose();
                base.Dispose(disposing);
            }
        }
    }
}
