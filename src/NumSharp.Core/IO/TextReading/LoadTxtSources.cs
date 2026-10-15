using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NumSharp.IO
{
    /// <summary>
    ///     What a <see cref="LoadTxtStream"/> buffer is (NumPy's <c>textreading/stream.h</c>): a chunk of a file that may
    ///     hold several lines, exactly one line, or the end of the input.
    /// </summary>
    internal enum LoadTxtBufferState
    {
        /// <summary><c>BUFFER_MAY_CONTAIN_NEWLINE</c>: a chunk of a file; lines end wherever a newline is.</summary>
        MayContainNewline = 0,

        /// <summary><c>BUFFER_IS_LINEND</c>: one line of a line-iterated source; its end is the line's end, and a newline
        ///     before it (outside quotes) is NumPy's "unquoted embedded newline" error.</summary>
        IsLineEnd = 1,

        /// <summary><c>BUFFER_IS_FILEEND</c>: the input is exhausted.</summary>
        IsFileEnd = 2,
    }

    /// <summary>
    ///     The input of NumPy's text reader (<c>textreading/stream_pyobject.c</c>): a source of character buffers the
    ///     tokenizer walks. A buffer stays valid until the next <see cref="NextBuffer"/> call — the tokenizer copies every
    ///     field it reads out of it.
    /// </summary>
    internal abstract class LoadTxtStream : IDisposable
    {
        /// <summary>The next buffer.</summary>
        /// <param name="buffer">The characters (empty at the end of the input, and for an empty line).</param>
        /// <returns>What the buffer is.</returns>
        internal abstract LoadTxtBufferState NextBuffer(out ReadOnlyMemory<char> buffer);

        /// <summary>Releases what the source owns (the file it opened); a caller's reader or stream is left open.</summary>
        public virtual void Dispose() { }
    }

    /// <summary>
    ///     A source of LINES as Python's iteration yields them — <c>iter(fname)</c> for anything that is not a file name
    ///     (<c>_read</c>'s <c>filelike=False</c> branch), and the input of the multi-character comment stripping.
    /// </summary>
    internal abstract class LoadTxtLineReader : IDisposable
    {
        /// <summary>The next line, with its line terminator when it has one.</summary>
        /// <param name="line">The line (valid until the next call).</param>
        /// <returns>False at the end of the input.</returns>
        internal abstract bool TryReadLine(out ReadOnlyMemory<char> line);

        /// <summary>Releases what the reader owns.</summary>
        public virtual void Dispose() { }
    }

    /// <summary>
    ///     The lines of a C# string sequence — NumPy's list / generator of <c>str</c>: each element is one line exactly as
    ///     given (an element may end in a newline; one inside it raises NumPy's "embedded newline" error in the tokenizer).
    /// </summary>
    internal sealed class LoadTxtSequenceLineReader : LoadTxtLineReader
    {
        private readonly IEnumerator<string> _lines;

        /// <summary>Wraps the sequence.</summary>
        /// <param name="lines">The lines.</param>
        internal LoadTxtSequenceLineReader(IEnumerable<string> lines) => _lines = lines.GetEnumerator();

        /// <summary>
        ///     True when the lines pass through NumPy's <c>_preprocess_comments</c> first (multiple or multi-character
        ///     comments): a <c>None</c> element then meets <c>line.split(c, 1)</c> before the C reader sees it, which is
        ///     <c>AttributeError: 'NoneType' object has no attribute 'split'</c>. Set by <see cref="LoadTxtLineStream"/>.
        /// </summary>
        internal bool NullLineIsAttributeError { get; set; }

        /// <inheritdoc/>
        /// <exception cref="TypeError">A null element (NumPy's <c>non-string returned while reading data</c>, which any
        ///     element that is not a <c>str</c> raises).</exception>
        /// <exception cref="AttributeError">A null element when <see cref="NullLineIsAttributeError"/> is set.</exception>
        internal override bool TryReadLine(out ReadOnlyMemory<char> line)
        {
            if (!_lines.MoveNext())
            {
                line = default;
                return false;
            }
            string s = _lines.Current;
            if (s == null)
            {
                if (NullLineIsAttributeError)
                    throw new AttributeError("'NoneType' object has no attribute 'split'");
                throw new TypeError("non-string returned while reading data");
            }
            line = s.AsMemory();
            return true;
        }

        /// <inheritdoc/>
        public override void Dispose() => _lines.Dispose();
    }

    /// <summary>
    ///     The lines of a <see cref="TextReader"/>, split as Python splits a text stream when it iterates it: after every
    ///     <c>'\n'</c>, the <c>'\n'</c> kept, a <c>'\r'</c> left as an ordinary character (<c>io.StringIO</c>, whose
    ///     <c>newline='\n'</c> translates nothing). The reader is the caller's and stays open.
    /// </summary>
    internal sealed class LoadTxtTextLineReader : LoadTxtLineReader
    {
        private readonly TextReader _reader;
        private char[] _block = new char[1 << 14];
        private int _start, _end;
        private bool _eof;

        /// <summary>Wraps the reader.</summary>
        /// <param name="reader">The text, read from its current position.</param>
        internal LoadTxtTextLineReader(TextReader reader) => _reader = reader;

        /// <inheritdoc/>
        internal override bool TryReadLine(out ReadOnlyMemory<char> line)
        {
            int scan = _start;
            while (true)
            {
                // A vectorized search for the next '\n' in the unread characters.
                int nl = _block.AsSpan(scan, _end - scan).IndexOf('\n');
                if (nl >= 0)
                {
                    nl += scan;
                    line = new ReadOnlyMemory<char>(_block, _start, nl + 1 - _start);
                    _start = nl + 1;
                    return true;
                }
                if (_eof)
                {
                    if (_start == _end)
                    {
                        line = default;
                        return false;
                    }
                    line = new ReadOnlyMemory<char>(_block, _start, _end - _start);
                    _start = _end;
                    return true;
                }
                // Move the unread characters to the block's start (growing it when a line fills it) and read more; the
                // search resumes after what was already searched.
                int keep = _end - _start;
                if (keep == _block.Length)
                    Array.Resize(ref _block, _block.Length * 2);
                if (_start > 0)
                {
                    Array.Copy(_block, _start, _block, 0, keep);
                    _start = 0;
                    _end = keep;
                }
                scan = _end;
                int read = _reader.Read(_block, _end, _block.Length - _end);
                if (read <= 0)
                    _eof = true;
                else
                    _end += read;
            }
        }
    }

    /// <summary>
    ///     The lines of a byte <see cref="Stream"/> — Python's iteration of a binary file object (<c>io.BytesIO</c>,
    ///     <c>open(path, 'rb')</c>): split after every <c>0x0A</c> byte, the byte kept, then each line decoded on its own
    ///     with the codec (NumPy's <c>PyUnicode_FromEncodedObject(line, encoding)</c>: <c>bytes.decode</c>, so a BOM codec
    ///     reads a BOM per line), strictly. No newline translation: a <c>"\r\n"</c> line keeps its <c>'\r'</c>.
    /// </summary>
    internal sealed class LoadTxtByteLineReader : LoadTxtLineReader
    {
        private readonly Stream _stream;
        private readonly TextIOCodec _codec;
        private readonly Exception _lookupFailure;
        private byte[] _block = new byte[1 << 16];
        private int _start, _end;
        private bool _eof;

        /// <summary>Wraps the stream (left open).</summary>
        /// <param name="stream">The bytes, read from the current position.</param>
        /// <param name="codec">The codec each line is decoded with, or null when the encoding name's lookup failed.</param>
        /// <param name="lookupFailure">With a null <paramref name="codec"/>: the lookup's error (a <see cref="LookupError"/>,
        ///     or the <see cref="NotSupportedException"/> of a codec NumSharp does not implement), raised at the first line.
        ///     NumPy decodes each line as it reads it (<c>stream_python_iterable</c>'s
        ///     <c>PyUnicode_FromEncodedObject(line, encoding)</c>, <c>_preprocess_comments</c>' <c>line.decode(encoding)</c>),
        ///     so the codec is looked up at the first line only — and never by a stream without lines.</param>
        internal LoadTxtByteLineReader(Stream stream, TextIOCodec codec, Exception lookupFailure = null)
        {
            _stream = stream;
            _codec = codec;
            _lookupFailure = lookupFailure;
        }

        /// <inheritdoc/>
        /// <exception cref="UnicodeDecodeError">A line the codec cannot decode (CPython's text; positions count bytes of
        ///     that line).</exception>
        /// <exception cref="LookupError">The first line, when the encoding name is unknown or not a text encoding.</exception>
        /// <exception cref="NotSupportedException">The first line, when the codec is one NumSharp does not implement.</exception>
        internal override bool TryReadLine(out ReadOnlyMemory<char> line)
        {
            int scan = _start;
            while (true)
            {
                // A vectorized search for the next 0x0A byte in the unread bytes.
                int nl = _block.AsSpan(scan, _end - scan).IndexOf((byte)'\n');
                if (nl >= 0)
                {
                    nl += scan;
                    if (_codec is null)
                        throw _lookupFailure;
                    line = _codec.Decode(new ReadOnlySpan<byte>(_block, _start, nl + 1 - _start)).AsMemory();
                    _start = nl + 1;
                    return true;
                }
                if (_eof)
                {
                    if (_start == _end)
                    {
                        line = default;
                        return false;
                    }
                    if (_codec is null)
                        throw _lookupFailure;
                    line = _codec.Decode(new ReadOnlySpan<byte>(_block, _start, _end - _start)).AsMemory();
                    _start = _end;
                    return true;
                }
                int keep = _end - _start;
                if (keep == _block.Length)
                    Array.Resize(ref _block, _block.Length * 2);
                if (_start > 0)
                {
                    Buffer.BlockCopy(_block, _start, _block, 0, keep);
                    _start = 0;
                    _end = keep;
                }
                scan = _end;
                int read = _stream.Read(_block, _end, _block.Length - _end);
                if (read <= 0)
                    _eof = true;
                else
                    _end += read;
            }
        }
    }

    /// <summary>
    ///     A file NumPy opened ITSELF from a name (<c>np.lib._datasource.open(fname, 'rt', encoding=...)</c>): a port of the
    ///     read side of CPython's <c>TextIOWrapper</c> over the file's bytes — the codec's incremental decoder inside an
    ///     <c>IncrementalNewlineDecoder</c> (universal newlines: <c>"\r\n"</c> and <c>'\r'</c> become <c>'\n'</c>) —, with
    ///     TextIOWrapper's chunking, which decides where a decode error's position counts from.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Chunks.</b> <c>read(n)</c> (NumPy's C reader calls <c>fh.read(16384)</c>) takes the decoded characters left
    ///     from the previous chunk first, then reads chunks of <c>max(8192, b2cratio * remaining)</c> bytes, where
    ///     <c>b2cratio</c> is the previous chunk's bytes per decoded (translated) character, until it has <c>n</c>
    ///     characters; a line iteration (<see cref="TryReadLine"/>, the path of multiple or multi-character comments)
    ///     reads chunks of 8192 bytes. Each chunk is ONE raw read of that many bytes (<c>BufferedReader.read1</c> on an
    ///     empty buffer), fewer only at the end of the file — so a stateless codec's decode error (ASCII, Latin-1, a
    ///     charmap) counts its position within its chunk, a buffered one's (UTF-8, UTF-16, UTF-32) within the chunk plus the
    ///     bytes left over from the previous one, exactly as NumPy reports it.
    ///     </para>
    ///     <para>
    ///     A <c>.gz</c> file's chunks are filled from the decompressed stream the same way; CPython's gzip reader hands
    ///     TextIOWrapper whatever one decompression step produced, so a decode error's position inside a compressed file can
    ///     differ from NumPy's (the values read never do).
    ///     </para>
    /// </remarks>
    internal sealed class LoadTxtTextFile : IDisposable
    {
        /// <summary>TextIOWrapper's <c>_CHUNK_SIZE</c>.</summary>
        private const int ChunkSize = 8192;

        private readonly Stream _raw;
        private readonly TextIONewlineDecoder _decoder;
        // Pooled at the size of read(16384)'s first chunk: a small file costs one rent of each instead of an allocation
        // (and its zeroing) that the first chunk would immediately outgrow. Returned by Dispose.
        private byte[] _bytes = ArrayPool<byte>.Shared.Rent(2 * ChunkSize);
        // decoded_chars: the last chunk's decoded (translated) characters, [_decodedStart, _decodedEnd) not yet returned.
        private char[] _decoded = ArrayPool<char>.Shared.Rent(2 * ChunkSize + 16);
        private int _decodedStart, _decodedEnd;
        private double _b2cratio;
        // A line that spans chunks is assembled here (TryReadLine).
        private char[] _line = new char[256];

        /// <summary>Wraps the raw file stream, which this reader owns.</summary>
        /// <param name="raw">The file's bytes (a gzip name's decompressed bytes).</param>
        /// <param name="codec">The text encoding.</param>
        internal LoadTxtTextFile(Stream raw, TextIOCodec codec)
        {
            _raw = raw;
            _decoder = new TextIONewlineDecoder(codec.NewIncrementalDecoder());
        }

        /// <summary>
        ///     <c>TextIOWrapper.read(n)</c> with <c>n = dest.Length</c>: the characters left from the last chunk, then chunk
        ///     after chunk until <c>n</c> characters are read or the file ends.
        /// </summary>
        /// <param name="dest">The destination; its length is <c>n</c>.</param>
        /// <returns>The number of characters read (fewer than <c>n</c> only at the end of the file; 0 there).</returns>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode.</exception>
        /// <exception cref="UnicodeError">A BOM codec whose file does not start with a BOM; the <c>undefined</c> codec.</exception>
        internal int Read(Span<char> dest)
        {
            int got = TakeDecoded(dest);
            while (got < dest.Length)
            {
                if (!ReadChunk(dest.Length - got))
                    break;
                got += TakeDecoded(dest.Slice(got));
            }
            return got;
        }

        /// <summary>
        ///     <c>TextIOWrapper.__next__</c> (its readline): the next line, through its <c>'\n'</c> (the last line may lack
        ///     one), reading 8192-byte chunks as they are needed.
        /// </summary>
        /// <param name="line">The line (valid until the next call).</param>
        /// <returns>False at the end of the file.</returns>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode.</exception>
        /// <exception cref="UnicodeError">A BOM codec whose file does not start with a BOM; the <c>undefined</c> codec.</exception>
        internal bool TryReadLine(out ReadOnlyMemory<char> line)
        {
            int length = 0; // characters assembled in _line (a line that spans chunks)
            while (true)
            {
                if (_decodedStart == _decodedEnd)
                {
                    if (!ReadChunk(0))
                    {
                        line = length == 0 ? default : new ReadOnlyMemory<char>(_line, 0, length);
                        return length > 0;
                    }
                    continue;
                }
                var available = _decoded.AsSpan(_decodedStart, _decodedEnd - _decodedStart);
                int nl = available.IndexOf('\n');
                int take = nl >= 0 ? nl + 1 : available.Length;
                if (nl >= 0 && length == 0)
                {
                    // The whole line is in this chunk: hand out a slice of it, no copy.
                    line = new ReadOnlyMemory<char>(_decoded, _decodedStart, take);
                    _decodedStart += take;
                    return true;
                }
                if (_line.Length < length + take)
                    Array.Resize(ref _line, Math.Max(length + take, _line.Length * 2));
                available.Slice(0, take).CopyTo(_line.AsSpan(length));
                length += take;
                _decodedStart += take;
                if (nl >= 0)
                {
                    line = new ReadOnlyMemory<char>(_line, 0, length);
                    return true;
                }
            }
        }

        /// <summary>Copies the unreturned decoded characters (at most <c>dest.Length</c>) and advances past them.</summary>
        /// <param name="dest">The destination.</param>
        /// <returns>The number copied.</returns>
        private int TakeDecoded(Span<char> dest)
        {
            int take = Math.Min(dest.Length, _decodedEnd - _decodedStart);
            _decoded.AsSpan(_decodedStart, take).CopyTo(dest);
            _decodedStart += take;
            return take;
        }

        /// <summary>
        ///     <c>textiowrapper_read_chunk(size_hint)</c>: one raw read of <c>max(8192, (Py_ssize_t)(max(b2cratio, 1.0) *
        ///     size_hint))</c> bytes (8192 for a line iteration's <c>size_hint</c> of 0), decoded — final when the read came
        ///     back empty — into the decoded characters, which it replaces.
        /// </summary>
        /// <param name="sizeHint">The characters still wanted (0 for a line iteration).</param>
        /// <returns>False at the end of the file (an empty read that decoded nothing).</returns>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode.</exception>
        /// <exception cref="UnicodeError">The codec's BOM / undefined errors.</exception>
        private bool ReadChunk(int sizeHint)
        {
            long hint = sizeHint > 0 ? (long)(Math.Max(_b2cratio, 1.0) * sizeHint) : 0;
            int chunkSize = (int)Math.Min(Math.Max(ChunkSize, hint), int.MaxValue / 4);
            if (_bytes.Length < chunkSize)
            {
                ArrayPool<byte>.Shared.Return(_bytes);
                _bytes = ArrayPool<byte>.Shared.Rent(chunkSize);
            }
            // read1 on an empty buffer is one raw read of chunk_size bytes, which a regular file fills completely unless
            // the file ends; a .NET stream may return fewer per call, so the call is repeated until full or at the end.
            int nbytes = 0;
            while (nbytes < chunkSize)
            {
                int n = _raw.Read(_bytes, nbytes, chunkSize - nbytes);
                if (n <= 0)
                    break;
                nbytes += n;
            }
            bool eof = nbytes == 0;
            int capacity = _decoder.MaxChars(nbytes);
            if (_decoded.Length < capacity)
            {
                // Nothing unreturned is kept: read_chunk replaces the decoded characters (they were all taken).
                ArrayPool<char>.Shared.Return(_decoded);
                _decoded = ArrayPool<char>.Shared.Rent(Math.Max(capacity, _decoded.Length * 2));
            }
            int nchars = _decoder.Decode(_bytes.AsSpan(0, nbytes), eof, _decoded, out int start);
            _decodedStart = start;
            _decodedEnd = start + nchars;
            // b2cratio = nbytes / nchars over the TRANSLATED characters, as TextIOWrapper measures it.
            _b2cratio = nchars > 0 ? (double)nbytes / nchars : 0.0;
            if (nchars > 0)
                eof = false; // the final decode flushed something (a held-back '\r')
            return !eof;
        }

        /// <summary>Closes the file and returns the pooled buffers (the reader must not be used afterwards).</summary>
        public void Dispose()
        {
            _raw.Dispose();
            if (_bytes != null)
            {
                ArrayPool<byte>.Shared.Return(_bytes);
                ArrayPool<char>.Shared.Return(_decoded);
                _bytes = null;
                _decoded = null;
            }
        }
    }

    /// <summary>A file NumPy opened itself, iterated line by line (<see cref="LoadTxtTextFile.TryReadLine"/>): the input of
    ///     the comment stripping when multiple or multi-character comments are given.</summary>
    internal sealed class LoadTxtFileLineReader : LoadTxtLineReader
    {
        private readonly LoadTxtTextFile _file;

        /// <summary>Wraps the file, which this reader owns.</summary>
        /// <param name="file">The opened file.</param>
        internal LoadTxtFileLineReader(LoadTxtTextFile file) => _file = file;

        /// <inheritdoc/>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode.</exception>
        /// <exception cref="UnicodeError">The codec's BOM / undefined errors.</exception>
        internal override bool TryReadLine(out ReadOnlyMemory<char> line) => _file.TryReadLine(out line);

        /// <inheritdoc/>
        public override void Dispose() => _file.Dispose();
    }

    /// <summary>
    ///     The chunks of a file NumPy opened itself (<c>stream_python_file</c>: <c>fh.read(16384)</c> per buffer, its newlines
    ///     already translated by the TextIOWrapper): each chunk may hold several lines, and a quoted field may continue into
    ///     the next.
    /// </summary>
    internal sealed class LoadTxtChunkStream : LoadTxtStream
    {
        private readonly LoadTxtTextFile _file;
        // fh.read(16384)'s destination, pooled (returned by Dispose); a rented array may be longer, so reads take exactly
        // the first 16384 characters of it.
        private char[] _chunk = ArrayPool<char>.Shared.Rent(ChunkChars);

        /// <summary>The characters <c>stream_python_file</c> asks for per read (<c>READ_CHUNKSIZE</c>, <c>1 &lt;&lt; 14</c>).</summary>
        private const int ChunkChars = 1 << 14;

        /// <summary>Wraps the opened file (which this stream owns).</summary>
        /// <param name="file">The file.</param>
        internal LoadTxtChunkStream(LoadTxtTextFile file) => _file = file;

        /// <inheritdoc/>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode.</exception>
        /// <exception cref="UnicodeError">The codec's BOM / undefined errors.</exception>
        internal override LoadTxtBufferState NextBuffer(out ReadOnlyMemory<char> buffer)
        {
            int n = _file.Read(_chunk.AsSpan(0, ChunkChars));
            if (n <= 0)
            {
                buffer = default;
                return LoadTxtBufferState.IsFileEnd;
            }
            buffer = new ReadOnlyMemory<char>(_chunk, 0, n);
            return LoadTxtBufferState.MayContainNewline;
        }

        /// <summary>Closes the file and returns the pooled chunk.</summary>
        public override void Dispose()
        {
            _file.Dispose();
            if (_chunk != null)
            {
                ArrayPool<char>.Shared.Return(_chunk);
                _chunk = null;
            }
        }
    }

    /// <summary>
    ///     A line-iterated source as the tokenizer reads it (<c>stream_python_iterable</c>): every line one
    ///     <see cref="LoadTxtBufferState.IsLineEnd"/> buffer, after NumPy's <c>_preprocess_comments</c> when multiple or
    ///     multi-character comments are given (each comment, in order, is <c>line = line.split(c, 1)[0]</c>: a string cuts
    ///     the line at its first occurrence — which also drops the line's newline —, <c>None</c> keeps the first
    ///     whitespace-separated word, any other object is <c>str.split</c>'s TypeError).
    /// </summary>
    internal sealed class LoadTxtLineStream : LoadTxtStream
    {
        private readonly LoadTxtLineReader _lines;
        private readonly object[] _comments;

        /// <summary>Wraps the line reader (which this stream owns).</summary>
        /// <param name="lines">The lines.</param>
        /// <param name="comments">The comments to strip as Python objects (see <see cref="NextBuffer"/>), or null.</param>
        /// <remarks>With comments, a C# sequence's null line is <c>_preprocess_comments</c>' <c>line.split</c> on
        ///     <c>None</c> (an AttributeError) rather than the C reader's TypeError, so the reader is told to hand it over.</remarks>
        internal LoadTxtLineStream(LoadTxtLineReader lines, object[] comments)
        {
            _lines = lines;
            _comments = comments;
            if (comments != null && lines is LoadTxtSequenceLineReader sequence)
                sequence.NullLineIsAttributeError = true;
        }

        /// <inheritdoc/>
        /// <exception cref="IndexError">A <c>None</c> comment over a line that is empty or only whitespace
        ///     (<c>str.split(None, 1)</c> returns <c>[]</c>, and <c>[0]</c> is <c>list index out of range</c>).</exception>
        /// <exception cref="TypeError">A comment that is neither a str nor <c>None</c> (<c>must be str or None, not int</c>),
        ///     raised at the first line read, skipped lines included, as NumPy's generator raises it.</exception>
        internal override LoadTxtBufferState NextBuffer(out ReadOnlyMemory<char> buffer)
        {
            if (!_lines.TryReadLine(out var line))
            {
                buffer = default;
                return LoadTxtBufferState.IsFileEnd;
            }
            if (_comments != null)
            {
                var span = line.Span;
                foreach (object c in _comments)
                {
                    if (c is string s)
                    {
                        int at = span.IndexOf(s.AsSpan(), StringComparison.Ordinal);
                        if (at >= 0)
                            span = span.Slice(0, at);
                    }
                    else if (c is null)
                    {
                        // str.split(None, 1)[0]: leading whitespace dropped, then everything from the first whitespace
                        // character on (Python's whitespace — the tokenizer's set, U+001C..U+001F included).
                        int start = 0;
                        while (start < span.Length && LoadTxtTokenizer.IsSpace(span[start]))
                            start++;
                        if (start == span.Length)
                            throw new IndexError("list index out of range");
                        int end = start;
                        while (end < span.Length && !LoadTxtTokenizer.IsSpace(span[end]))
                            end++;
                        span = span.Slice(start, end - start);
                        // A slice that does not start at the line's first character: the buffer is the slice's own memory.
                        line = line.Slice(start);
                    }
                    else
                    {
                        throw new TypeError($"must be str or None, not {Backends.Strings.NDPyObject.TypeName(c)}");
                    }
                }
                if (span.Length != line.Length)
                    line = line.Slice(0, span.Length);
            }
            buffer = line;
            return LoadTxtBufferState.IsLineEnd;
        }

        /// <inheritdoc/>
        public override void Dispose() => _lines.Dispose();
    }
}
