using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp.IO
{
    /// <summary>
    ///     NumPy 2.4.2's text-reader tokenizer (<c>numpy/_core/src/multiarray/textreading/tokenize.cpp</c>), statement for
    ///     statement: a state machine that splits the input into rows of fields, buffer by buffer, the way
    ///     <c>np.loadtxt</c> reads every dtype.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Quoting.</b> A field is quoted when its first character (after leading whitespace in whitespace mode) is the
    ///     quote character. Inside, delimiters, comments and NEWLINES are ordinary characters, two quotes are one literal
    ///     quote, and the second quote ends the quoting — anything after it continues the field unquoted
    ///     (<c>"ABC"DEF</c> is <c>ABCDEF</c>). A quote anywhere else is data. A quoted field may span lines, and even buffers
    ///     of a line-iterated source: <c>['"a', 'b"']</c> reads one field <c>ab</c> (no newline is added).
    ///     </para>
    ///     <para>
    ///     <b>Rows.</b> A row ends at <c>'\n'</c>, <c>'\r'</c> or <c>"\r\n"</c> outside quotes, at a comment character (the
    ///     rest of the line is skipped), or at the end of a line buffer. A line buffer that holds more after its row ended is
    ///     NumPy's <c>Found an unquoted embedded newline within a single line of input.</c> error.
    ///     </para>
    ///     <para>
    ///     <b>Empty fields.</b> Every row starts a field and every delimiter starts the next, so two rules drop an empty
    ///     last field: a row of exactly one empty field is an empty line, and in whitespace mode an empty last field is
    ///     dropped (Python's <c>" 1 ".split()</c>) — both only when the row's FIRST field is unquoted (NumPy tests
    ///     <c>ts-&gt;fields-&gt;quoted</c>, field 0, so <c>'"a" b '</c> in whitespace mode keeps a third, empty field).
    ///     </para>
    ///     <para>
    ///     <b>Field storage.</b> As NumPy's <c>field_buffer</c>: the fields of a row are copied one after the other into one
    ///     character buffer, each followed by a NUL separator; field <c>i</c> spans
    ///     <c>[offset[i], offset[i + 1] - 1)</c>, and an extra offset closes the last field.
    ///     </para>
    /// </remarks>
    internal sealed class LoadTxtTokenizer
    {
        /// <summary>The tokenizer states (<c>tokenizer_parsing_state</c>).</summary>
        private enum State
        {
            Init,
            CheckQuoted,
            Unquoted,
            UnquotedWhitespace,
            Quoted,
            QuotedCheckDoubleQuote,
            LineEnd,
            EatCrlf,
            GotoLineEnd,
        }

        /// <summary>A control character that is not set: no character of the input can equal it.</summary>
        internal const int None = -1;

        /// <summary>
        ///     Every character <see cref="IsSpace"/> accepts (NumPy's <c>Py_UNICODE_ISSPACE</c>): a fixed set of BMP
        ///     characters, built once by testing the whole BMP so it cannot drift from <see cref="IsSpace"/>.
        /// </summary>
        private static readonly string SpaceChars = BuildSpaceChars();

        /// <summary>The characters that end an unquoted field's run, per (delimiter, comment) configuration: a loadtxt call
        ///     reuses the search set and class table of an earlier call with the same control characters (building them costs
        ///     microseconds, a measurable share of a small call).</summary>
        private static readonly ConcurrentDictionary<(int Delimiter, int Comment), StopSet> StopsCache = new();

        /// <summary><see cref="StopSet.AsciiClass"/>: the character ends an unquoted run.</summary>
        private const byte StopBit = 1;

        /// <summary><see cref="StopSet.AsciiClass"/>: the character is whitespace CheckQuoted strips in whitespace mode
        ///     (<see cref="IsSpace"/>, but not <c>'\r'</c> or <c>'\n'</c>).</summary>
        private const byte StripBit = 2;

        /// <summary>
        ///     The stop characters of one (delimiter, comment) configuration in the two forms the tokenizer searches them in.
        /// </summary>
        private sealed class StopSet
        {
            /// <summary>The vectorized set (<c>'\r'</c>, <c>'\n'</c>, the delimiter or every whitespace character, the
            ///     comment), for long runs.</summary>
            internal readonly SearchValues<char> Search;

            /// <summary>One entry per ASCII character: <see cref="StopBit"/> when <see cref="Search"/> holds it, plus
            ///     <see cref="StripBit"/> for the whitespace CheckQuoted strips — one table load per character for the scalar
            ///     loops that walk short runs.</summary>
            internal readonly byte[] AsciiClass = new byte[128];

            /// <summary>Builds the set of one configuration.</summary>
            /// <param name="delimiter">The delimiter, or <see cref="None"/> for whitespace mode.</param>
            /// <param name="comment">The single comment character, or <see cref="None"/>.</param>
            internal StopSet(int delimiter, int comment)
            {
                Search = BuildStops(delimiter, comment);
                for (int c = 0; c < 128; c++)
                {
                    if (Search.Contains((char)c))
                        AsciiClass[c] |= StopBit;
                    if (IsSpace((char)c) && c != '\r' && c != '\n')
                        AsciiClass[c] |= StripBit;
                }
            }
        }

        /// <summary>The two line-end characters <c>GotoLineEnd</c> searches for.</summary>
        private static readonly SearchValues<char> LineEnds = SearchValues.Create("\r\n");

        private readonly int _delimiter, _quote, _comment;
        private readonly bool _ignoreLeadingWhitespace;
        private readonly State _unquotedState;

        /// <summary>
        ///     The characters that can end an unquoted run in this configuration: <c>'\r'</c>, <c>'\n'</c>, the delimiter
        ///     (every whitespace character in whitespace mode) and the comment. The unquoted states find the first of them with
        ///     one vectorized search instead of NumPy's per-character loop; which character it is then decides the state
        ///     exactly as the loop's ordered tests do.
        /// </summary>
        private readonly SearchValues<char> _stops;

        /// <summary>The ASCII class table of the configuration (<see cref="StopSet.AsciiClass"/>).</summary>
        private readonly byte[] _asciiClass;

        /// <summary>How many characters <see cref="FindStop"/> tests one by one before it switches to the vectorized search,
        ///     and the longest chunk <see cref="CopyToFieldBuffer"/> copies element by element.</summary>
        private const int ScalarScanLength = 16;

        private State _state = State.Init;
        private LoadTxtBufferState _bufState = LoadTxtBufferState.MayContainNewline;
        private ReadOnlyMemory<char> _buffer;
        private int _pos, _end;

        private char[] _fieldBuffer = new char[32];
        private int _fieldBufferPos;
        private int[] _offsets = new int[4];
        private bool[] _quoted = new bool[4];
        private int _numFields;

        // Where the row's fields are once a row is tokenized: field i is [_offsets[i], _ends[i]) of RowChars, which is the
        // field buffer (the state machine copies every field there, NumPy's layout: _ends[i] is the NUL after the field) or,
        // for a row TokenizeUnquotedLine read, the line buffer itself — a line without quotes needs no copy, every field is a
        // plain slice of it.
        private int[] _ends = new int[4];
        private bool _rowInLine;

        /// <summary>Configures the tokenizer (<c>npy_tokenizer_init</c> with the parser config).</summary>
        /// <param name="delimiter">The delimiter character, or <see cref="None"/> for whitespace (runs of whitespace separate
        ///     fields, and leading whitespace is skipped before the quote check, as Python's <c>str.split(None)</c>).</param>
        /// <param name="quote">The quote character, or <see cref="None"/>.</param>
        /// <param name="comment">The single comment character, or <see cref="None"/> (multi-character comments are stripped
        ///     from the lines before the tokenizer sees them).</param>
        internal LoadTxtTokenizer(int delimiter, int quote, int comment)
        {
            bool whitespace = delimiter == None;
            _delimiter = delimiter;
            _quote = quote;
            _comment = comment;
            _ignoreLeadingWhitespace = whitespace;
            _unquotedState = whitespace ? State.UnquotedWhitespace : State.Unquoted;
            var set = StopsCache.GetOrAdd((delimiter, comment), static k => new StopSet(k.Delimiter, k.Comment));
            _stops = set.Search;
            _asciiClass = set.AsciiClass;
        }

        /// <summary>Collects every BMP character <see cref="IsSpace"/> accepts.</summary>
        /// <returns>The characters, in code order.</returns>
        private static string BuildSpaceChars()
        {
            var sb = new StringBuilder();
            for (int c = 0; c <= char.MaxValue; c++)
                if (IsSpace((char)c))
                    sb.Append((char)c);
            return sb.ToString();
        }

        /// <summary>Builds the stop set of one configuration (see <see cref="_stops"/>).</summary>
        /// <param name="delimiter">The delimiter, or <see cref="None"/> for whitespace mode.</param>
        /// <param name="comment">The single comment character, or <see cref="None"/>.</param>
        /// <returns>The search set.</returns>
        /// <remarks>The control characters are BMP characters (a character past U+FFFF is refused before a tokenizer is
        ///     made), and none of them is a newline, so each stop character means exactly one of the loop's outcomes.</remarks>
        private static SearchValues<char> BuildStops(int delimiter, int comment)
        {
            var sb = new StringBuilder("\r\n");
            if (delimiter == None)
                sb.Append(SpaceChars);
            else
                sb.Append((char)delimiter);
            if (comment != None)
                sb.Append((char)comment);
            return SearchValues.Create(sb.ToString());
        }

        /// <summary>The number of fields of the row just tokenized.</summary>
        internal int NumFields => _numFields;

        /// <summary>The characters of field <paramref name="i"/> of the row just tokenized (valid until the next row).</summary>
        /// <param name="i">The field index (0 ≤ i &lt; <see cref="NumFields"/>).</param>
        /// <returns>The field's characters.</returns>
        internal ReadOnlySpan<char> Field(int i) => RowChars.Slice(_offsets[i], _ends[i] - _offsets[i]);

        /// <summary>The characters the row's fields are slices of (see <see cref="FieldStarts"/>; valid until the next
        ///     row). A caller reading every field takes this once per row rather than <see cref="Field"/> per field.</summary>
        internal ReadOnlySpan<char> RowChars => _rowInLine ? _buffer.Span : _fieldBuffer;

        /// <summary>Field <c>i</c> starts at <c>FieldStarts[i]</c> of <see cref="RowChars"/> (0 ≤ i &lt;
        ///     <see cref="NumFields"/>; the array may be longer).</summary>
        internal int[] FieldStarts => _offsets;

        /// <summary>Field <c>i</c> ends (exclusive) at <c>FieldEnds[i]</c> of <see cref="RowChars"/>.</summary>
        internal int[] FieldEnds => _ends;

        /// <summary>
        ///     Skips one line (<c>read_rows</c>' <c>skiplines</c>: the state is set to <c>TOKENIZE_GOTO_LINE_END</c> before a
        ///     tokenize call): quotes are not honored, and a line buffer is skipped whole.
        /// </summary>
        /// <param name="stream">The input.</param>
        /// <returns>1 when the input ended, else 0.</returns>
        internal int SkipLine(LoadTxtStream stream)
        {
            _state = State.GotoLineEnd;
            return Tokenize(stream);
        }

        /// <summary>
        ///     Tokenizes one row (<c>npy_tokenize</c>): afterwards <see cref="NumFields"/> and <see cref="Field"/> describe it
        ///     (zero fields for an empty or comment-only line).
        /// </summary>
        /// <param name="stream">The input.</param>
        /// <returns>1 when the input ended with this row (the row may still hold fields), else 0.</returns>
        /// <exception cref="ValueError">A line buffer that goes on after its row ended outside quotes (NumPy's
        ///     <c>Found an unquoted embedded newline within a single line of input.  This is currently not
        ///     supported.</c>).</exception>
        internal int Tokenize(LoadTxtStream stream)
        {
            int finishedReadingFile = 0;
            _fieldBufferPos = 0;
            _numFields = 0;

            // The state, the buffer's characters and the position live in locals for the whole row (written back once at the
            // end): NumPy's tokenizer_core is a separate step function, but a step per call reloaded all three through `this`
            // and re-derived the span of the ReadOnlyMemory each time, about a third of the cost of a short field.
            State state = _state;
            int pos = _pos;
            int end = _end;
            ReadOnlySpan<char> buf;
            if (state == State.Init && pos >= end && _bufState != LoadTxtBufferState.IsLineEnd)
            {
                // A row that starts with its buffer consumed (every row of a line source): the loop's first steps, taken here
                // — add_field of the first field, then the fetch and its two checks — so a whole line without the quote
                // character goes straight to the unquoted-line tokenizer, which reads it in one pass with the same fields,
                // end position and errors as the steps below (no field of such a row can be quoted). Any other buffer
                // continues in the loop exactly where the loop's own fetch would.
                AddField();
                state = State.CheckQuoted;
                _bufState = stream.NextBuffer(out _buffer);
                buf = _buffer.Span;
                pos = 0;
                end = buf.Length;
                if (_bufState == LoadTxtBufferState.IsFileEnd)
                {
                    finishedReadingFile = 1;
                    pos = end; // the stream ensures this
                    goto RowDone;
                }
                if (pos == end)
                    goto RowDone; // this must be an empty line (and it must be indicated!)
                if (_bufState == LoadTxtBufferState.IsLineEnd && (_quote == None || buf.IndexOf((char)_quote) < 0))
                    return TokenizeUnquotedLine(buf);
            }
            else
            {
                // A consumed buffer is never read: the span is taken when one is fetched.
                buf = pos < end ? _buffer.Span : default;
            }

            while (true)
            {
                // This loop adds new fields to the result (to make up a full row) until the row ends (typically a line end or
                // the file end).
                if (state == State.Init)
                {
                    // Start a new field.
                    AddField();
                    state = State.CheckQuoted;
                }

                if (pos >= end)
                {
                    if (_bufState == LoadTxtBufferState.IsLineEnd && state != State.Quoted)
                    {
                        // Finished line, do not read anymore (also do not eat \n). If we are in a quoted field and the "line"
                        // does not end with a newline, the quoted field will not have it either.
                        break;
                    }
                    // Fetch new data.
                    _bufState = stream.NextBuffer(out _buffer);
                    buf = _buffer.Span;
                    pos = 0;
                    end = buf.Length;
                    if (_bufState == LoadTxtBufferState.IsFileEnd)
                    {
                        finishedReadingFile = 1;
                        pos = end; // the stream ensures this
                        break;
                    }
                    if (pos == end)
                    {
                        // This must be an empty line (and it must be indicated!).
                        break;
                    }
                }

                // One step of tokenizer_core, inline so the state, buffer and position stay in registers: it consumes
                // characters of the current buffer in the current state. pos is below end here (a buffer was fetched first),
                // and a step can leave it at end + 1: NumPy steps past the character that ended a run even when the run ended
                // at the buffer's end, and the test above reads any position at or past the end as "fetch the next buffer".
                if (state == State.CheckQuoted)
                {
                    // Before we can check for quotes, strip leading whitespace.
                    if (_ignoreLeadingWhitespace)
                    {
                        while (pos < end && IsSpace(buf[pos]) && buf[pos] != '\r' && buf[pos] != '\n')
                            pos++;
                        if (pos == end)
                            continue; // tokenizer_core returns here; the row loop fetches the next buffer
                    }

                    // Setting chunk effectively starts the field.
                    if (buf[pos] == _quote)
                    {
                        _quoted[_numFields - 1] = true;
                        state = State.Quoted;
                        pos++; // Quoted is OK with pos == end
                    }
                    else
                    {
                        // Set to Unquoted or UnquotedWhitespace.
                        state = _unquotedState;
                    }
                }

                int chunkStart;
                switch (state)
                {
                    case State.Unquoted:
                    case State.UnquotedWhitespace:
                    {
                        // NumPy's per-character loop stops at the first '\r' (EatCrlf), '\n' (LineEnd), delimiter — any
                        // whitespace character in whitespace mode — (Init) or comment (GotoLineEnd). FindStop finds that
                        // character; the tests below then classify it in the loop's order, so the outcome is the loop's.
                        // Running off the buffer leaves pos == end and the state unchanged, as the loop does.
                        chunkStart = pos;
                        pos = FindStop(buf, pos, end);
                        if (pos < end)
                        {
                            char c = buf[pos];
                            if (c == '\r')
                                state = State.EatCrlf;
                            else if (c == '\n')
                                state = State.LineEnd;
                            else if (state == State.UnquotedWhitespace ? IsSpace(c) : c == _delimiter)
                                state = State.Init;
                            else
                                state = State.GotoLineEnd; // the only other stop character is the comment
                        }
                        CopyToFieldBuffer(buf.Slice(chunkStart, pos - chunkStart));
                        pos++;
                        break;
                    }

                    case State.Quoted:
                    {
                        // Inside quotes only the quote character matters: one vectorized search for it.
                        chunkStart = pos;
                        int k = buf.Slice(pos, end - pos).IndexOf((char)_quote);
                        if (k < 0)
                        {
                            pos = end;
                        }
                        else
                        {
                            pos += k;
                            state = State.QuotedCheckDoubleQuote;
                        }
                        CopyToFieldBuffer(buf.Slice(chunkStart, Math.Min(pos, end) - chunkStart));
                        pos++;
                        break;
                    }

                    case State.QuotedCheckDoubleQuote:
                        if (buf[pos] == _quote)
                        {
                            // Copy the quote character directly from the config.
                            AppendToFieldBuffer((char)_quote);
                            state = State.Quoted;
                            pos++;
                        }
                        else
                        {
                            // Continue parsing as if unquoted.
                            state = _unquotedState;
                        }
                        break;

                    case State.GotoLineEnd:
                    {
                        if (_bufState != LoadTxtBufferState.MayContainNewline)
                        {
                            pos = end; // advance to next buffer
                            state = State.LineEnd;
                            break;
                        }
                        // The rest of a commented (or skipped) line: one vectorized search for its end.
                        int k = buf.Slice(pos, end - pos).IndexOfAny(LineEnds);
                        if (k < 0)
                        {
                            pos = end;
                        }
                        else
                        {
                            pos += k;
                            state = buf[pos] == '\r' ? State.EatCrlf : State.LineEnd;
                        }
                        pos++;
                        break;
                    }

                    case State.EatCrlf:
                        // "Universal newline" support: remove \n in \r\n.
                        if (buf[pos] == '\n')
                            pos++;
                        state = State.LineEnd;
                        break;
                }

                if (state == State.LineEnd)
                    break;
            }

        RowDone:
            _pos = pos;
            _end = end;

            // We have finished tokenizing a full row into fields, finalize result.
            if (_bufState == LoadTxtBufferState.IsLineEnd)
            {
                // This line is "finished", make sure we don't touch it again.
                _bufState = LoadTxtBufferState.MayContainNewline;
                if (pos < end)
                    throw new ValueError("Found an unquoted embedded newline within a single line of input.  This is currently not supported.");
            }

            // Finish the last field (we "append" one to store the last one's length).
            AddField();
            _numFields -= 1;

            // We always start a new field (at the very beginning and whenever a delimiter was found). This gives two
            // scenarios where the last field must be ignored if it is empty: (1) exactly one empty (unquoted) field — the
            // whole line is empty; (2) splitting on whitespace, a last empty field is ignored to match Python's
            // `" 1 ".split()`. NumPy tests the FIRST field's quoted flag in both (ts->fields->quoted).
            if (_numFields == 1 || (_numFields > 0 && _unquotedState == State.UnquotedWhitespace))
            {
                int offsetLast = _offsets[_numFields - 1];
                int endLast = _offsets[_numFields];
                if (!_quoted[0] && endLast - offsetLast == 1)
                    _numFields--;
            }

            // The fields are in the field buffer, each ending at the NUL before the next one's start.
            _rowInLine = false;
            for (int i = 0; i < _numFields; i++)
                _ends[i] = _offsets[i + 1] - 1;
            _state = State.Init;
            return finishedReadingFile;
        }

        /// <summary>
        ///     Tokenizes a row that is one whole line buffer holding no quote character, start to finish: the steps
        ///     <see cref="Tokenize"/> takes for it (CheckQuoted's whitespace strip, the unquoted run, then add_field at a
        ///     delimiter, EatCrlf at <c>'\r'</c>, LineEnd at <c>'\n'</c>, GotoLineEnd at the comment) and its end of row (the
        ///     embedded-newline check, the empty-last-field rules), with the fields left as slices of the line itself.
        /// </summary>
        /// <param name="buf">The line, read from its start. On entry the row's first field is open and empty.</param>
        /// <returns>0: a line was read, so the input did not end with this row.</returns>
        /// <exception cref="ValueError">The line goes on after a newline (NumPy's <c>Found an unquoted embedded newline within
        ///     a single line of input.  This is currently not supported.</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     Without a quote character no field can be quoted, so the quote test of CheckQuoted is always false and the
        ///     quoted states are never entered; every other step is the state machine's, in its order. A line buffer ends the
        ///     row at its end (BUFFER_IS_LINEND), which is why a comment skips the rest of the line outright: GotoLineEnd only
        ///     searches for the newline in a buffer that may contain more lines.
        ///     </para>
        ///     <para>
        ///     The state machine copies each field into the field buffer because a field can span buffers or lose its quotes;
        ///     here every field is one unbroken run of the line, so it is recorded as [start, end) of the line
        ///     (<see cref="RowChars"/> is then the line buffer, valid until the next row) and no character is copied.
        ///     </para>
        /// </remarks>
        private int TokenizeUnquotedLine(ReadOnlySpan<char> buf)
        {
            int end = buf.Length;
            // A row cannot have more fields than characters plus one: growing once here keeps the loop free of checks.
            if (_offsets.Length < end + 2)
            {
                int size = Math.Max(end + 2, _offsets.Length * 2);
                Array.Resize(ref _offsets, size);
                Array.Resize(ref _quoted, size);
                Array.Resize(ref _ends, size);
            }

            // The class table is read through a reference (an index below 128, no bounds check).
            ref byte cls = ref MemoryMarshal.GetArrayDataReference(_asciiClass);
            int[] starts = _offsets;
            int[] ends = _ends;
            bool whitespace = _ignoreLeadingWhitespace;
            int fields = 1; // the row's first field is open; field fields - 1 is the open one
            int pos = 0;
            while (true)
            {
                // CheckQuoted: strip leading whitespace (whitespace mode; IsSpace, but not '\r' or '\n'); the field is
                // unquoted.
                if (whitespace)
                {
                    while (pos < end)
                    {
                        char w = buf[pos];
                        if (w < 128 ? (Unsafe.Add(ref cls, w) & StripBit) == 0 : !IsSpace(w))
                            break;
                        pos++;
                    }
                    if (pos == end)
                    {
                        // The line is consumed with the field open and empty: the row ends there.
                        starts[fields - 1] = end;
                        ends[fields - 1] = end;
                        break;
                    }
                }

                // The unquoted run: the first ScalarScanLength characters tested one by one, a longer run continued by the
                // vectorized search.
                starts[fields - 1] = pos;
                int limit = Math.Min(end, pos + ScalarScanLength);
                while (pos < limit)
                {
                    char r = buf[pos];
                    if (r < 128 ? (Unsafe.Add(ref cls, r) & StopBit) != 0 : IsNonAsciiStop(r))
                        break;
                    pos++;
                }
                if (pos == limit && pos < end)
                {
                    int k = buf.Slice(pos).IndexOfAny(_stops);
                    pos = k < 0 ? end : pos + k;
                }
                ends[fields - 1] = pos;
                if (pos == end)
                {
                    pos++; // the run reached the line's end: NumPy steps past it, and the row ends
                    break;
                }

                char c = buf[pos++];
                if (c == '\r')
                {
                    // EatCrlf: a '\n' right after is part of the newline (unless the line ends at the '\r').
                    if (pos < end && buf[pos] == '\n')
                        pos++;
                    break;
                }
                if (c == '\n')
                    break; // LineEnd
                if (whitespace ? IsSpace(c) : c == _delimiter)
                {
                    // Init: add_field opens the next field.
                    fields++;
                    if (pos >= end)
                    {
                        // The new field is open at the line's end: the row ends with it empty.
                        starts[fields - 1] = end;
                        ends[fields - 1] = end;
                        break;
                    }
                    continue;
                }
                // The comment: GotoLineEnd skips the rest of a line buffer.
                pos = end;
                break;
            }

            // The end of the row, as Tokenize finishes it: the line must be consumed, then an empty last field is dropped
            // when it is the only field or the split is on whitespace (field 0 is never quoted here).
            _pos = pos;
            _end = end;
            _bufState = LoadTxtBufferState.MayContainNewline;
            _state = State.Init;
            if (pos < end)
                throw new ValueError("Found an unquoted embedded newline within a single line of input.  This is currently not supported.");
            if ((fields == 1 || whitespace) && ends[fields - 1] == starts[fields - 1])
                fields--;
            _numFields = fields;
            _rowInLine = true;
            return 0;
        }

        /// <summary>
        ///     The first character at or after <paramref name="pos"/> that ends an unquoted run (see <see cref="_stops"/>), or
        ///     <paramref name="stop"/> when the buffer holds none.
        /// </summary>
        /// <param name="buf">The buffer's characters.</param>
        /// <param name="pos">Where the run starts.</param>
        /// <param name="stop">The buffer's end.</param>
        /// <returns>The stop character's position, or <paramref name="stop"/>.</returns>
        /// <remarks>
        ///     Most fields are a few characters long, and a vectorized <see cref="SearchValues{T}"/> search costs about as much
        ///     to start as a short field takes to walk (the whitespace stop set holds non-ASCII characters, which makes it the
        ///     probabilistic searcher): the first <see cref="ScalarScanLength"/> characters are tested one by one against the
        ///     ASCII class table, as NumPy's loop tests them, and only a longer run continues with the vectorized search.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int FindStop(ReadOnlySpan<char> buf, int pos, int stop)
        {
            ref byte cls = ref MemoryMarshal.GetArrayDataReference(_asciiClass);
            int limit = Math.Min(stop, pos + ScalarScanLength);
            for (; pos < limit; pos++)
            {
                char c = buf[pos];
                if (c < 128 ? (Unsafe.Add(ref cls, c) & StopBit) != 0 : IsNonAsciiStop(c))
                    return pos;
            }
            if (pos == stop)
                return stop;
            int k = buf.Slice(pos, stop - pos).IndexOfAny(_stops);
            return k < 0 ? stop : pos + k;
        }

        /// <summary>Whether a character past U+007F ends an unquoted run: any whitespace character in whitespace mode, or the
        ///     delimiter or comment when they are not ASCII.</summary>
        /// <param name="c">The character (at least U+0080).</param>
        /// <returns>True for a stop character.</returns>
        private bool IsNonAsciiStop(char c) => (_ignoreLeadingWhitespace && IsSpace(c)) || c == _delimiter || c == _comment;

        /// <summary>
        ///     Appends characters to the current field (<c>copy_to_field_buffer</c>), keeping room for the NUL separator and
        ///     the extra closing offset <see cref="AddField"/> writes.
        /// </summary>
        /// <param name="chunk">The characters.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CopyToFieldBuffer(ReadOnlySpan<char> chunk)
        {
            // Space for length +1 termination, +2 additional padding for AddField.
            int size = chunk.Length + _fieldBufferPos + 3;
            if (_fieldBuffer.Length < size)
                Array.Resize(ref _fieldBuffer, Math.Max(size, _fieldBuffer.Length * 2));
            Span<char> dst = _fieldBuffer.AsSpan(_fieldBufferPos);
            // A short field is copied element by element: Memmove's call costs more than the copy itself.
            if (chunk.Length <= ScalarScanLength)
            {
                for (int i = 0; i < chunk.Length; i++)
                    dst[i] = chunk[i];
            }
            else
            {
                chunk.CopyTo(dst);
            }
            _fieldBufferPos += chunk.Length;
            _fieldBuffer[_fieldBufferPos] = '\0'; // always ensure we end with NUL
        }

        /// <summary>Appends one character to the current field (the doubled quote of a quoted field).</summary>
        /// <param name="c">The character.</param>
        private void AppendToFieldBuffer(char c)
        {
            // Room for the character, the NUL separator and AddField's extra closing offset, as CopyToFieldBuffer keeps.
            if (_fieldBuffer.Length < _fieldBufferPos + 4)
                Array.Resize(ref _fieldBuffer, Math.Max(_fieldBufferPos + 4, _fieldBuffer.Length * 2));
            _fieldBuffer[_fieldBufferPos++] = c;
            _fieldBuffer[_fieldBufferPos] = '\0';
        }

        /// <summary>Closes the current field and opens the next one (<c>add_field</c>).</summary>
        private void AddField()
        {
            // The previous field is done, advance to keep a NUL at the end.
            _fieldBufferPos += 1;
            if (_numFields + 1 >= _offsets.Length)
            {
                Array.Resize(ref _offsets, _offsets.Length * 2);
                Array.Resize(ref _quoted, _quoted.Length * 2);
                // The end positions the row's finish computes are as many as the offsets.
                Array.Resize(ref _ends, _offsets.Length);
            }
            if (_fieldBuffer.Length <= _fieldBufferPos)
                Array.Resize(ref _fieldBuffer, _fieldBuffer.Length * 2);
            _offsets[_numFields] = _fieldBufferPos;
            _quoted[_numFields] = false;
            _numFields += 1;
            // Ensure this (currently empty) word is NUL terminated.
            _fieldBuffer[_fieldBufferPos] = '\0';
        }

        /// <summary>
        ///     CPython's <c>Py_UNICODE_ISSPACE</c>: .NET's <see cref="char.IsWhiteSpace(char)"/> over the whole BMP plus the
        ///     four C0 information separators <c>U+001C</c>..<c>U+001F</c> (the only difference, verified exhaustively).
        /// </summary>
        /// <param name="c">The character.</param>
        /// <returns>True for whitespace.</returns>
        internal static bool IsSpace(char c) => char.IsWhiteSpace(c) || (c >= '\u001c' && c <= '\u001f');
    }
}
