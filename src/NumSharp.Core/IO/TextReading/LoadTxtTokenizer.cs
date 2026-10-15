using System;
using System.Buffers;
using System.Collections.Concurrent;
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
        ///     reuses the vectorized search set of an earlier call with the same control characters (building one costs
        ///     about a microsecond, a measurable share of a small call).</summary>
        private static readonly ConcurrentDictionary<(int Delimiter, int Comment), SearchValues<char>> StopsCache = new();

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

        private State _state = State.Init;
        private LoadTxtBufferState _bufState = LoadTxtBufferState.MayContainNewline;
        private ReadOnlyMemory<char> _buffer;
        private int _pos, _end;

        private char[] _fieldBuffer = new char[32];
        private int _fieldBufferPos;
        private int[] _offsets = new int[4];
        private bool[] _quoted = new bool[4];
        private int _numFields;

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
            _stops = StopsCache.GetOrAdd((delimiter, comment), static k => BuildStops(k.Delimiter, k.Comment));
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
        internal ReadOnlySpan<char> Field(int i) => new ReadOnlySpan<char>(_fieldBuffer, _offsets[i], _offsets[i + 1] - _offsets[i] - 1);

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

            while (true)
            {
                // This loop adds new fields to the result (to make up a full row) until the row ends (typically a line end or
                // the file end).
                if (_state == State.Init)
                {
                    // Start a new field.
                    AddField();
                    _state = State.CheckQuoted;
                }

                if (_pos >= _end)
                {
                    if (_bufState == LoadTxtBufferState.IsLineEnd && _state != State.Quoted)
                    {
                        // Finished line, do not read anymore (also do not eat \n). If we are in a quoted field and the "line"
                        // does not end with a newline, the quoted field will not have it either.
                        break;
                    }
                    // Fetch new data.
                    _bufState = stream.NextBuffer(out _buffer);
                    _pos = 0;
                    _end = _buffer.Length;
                    if (_bufState == LoadTxtBufferState.IsFileEnd)
                    {
                        finishedReadingFile = 1;
                        _pos = _end; // the stream ensures this
                        break;
                    }
                    if (_pos == _end)
                    {
                        // This must be an empty line (and it must be indicated!).
                        break;
                    }
                }

                TokenizerCore();
                if (_state == State.LineEnd)
                    break;
            }

            // We have finished tokenizing a full row into fields, finalize result.
            if (_bufState == LoadTxtBufferState.IsLineEnd)
            {
                // This line is "finished", make sure we don't touch it again.
                _bufState = LoadTxtBufferState.MayContainNewline;
                if (_pos < _end)
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
            _state = State.Init;
            return finishedReadingFile;
        }

        /// <summary>One step of <c>tokenizer_core</c>: consumes characters of the current buffer in the current state.</summary>
        private void TokenizerCore()
        {
            ReadOnlySpan<char> buf = _buffer.Span;
            int pos = _pos;
            int stop = _end;

            if (_state == State.CheckQuoted)
            {
                // Before we can check for quotes, strip leading whitespace.
                if (_ignoreLeadingWhitespace)
                {
                    while (pos < stop && IsSpace(buf[pos]) && buf[pos] != '\r' && buf[pos] != '\n')
                        pos++;
                    if (pos == stop)
                    {
                        _pos = pos;
                        return;
                    }
                }

                // Setting chunk effectively starts the field.
                if (buf[pos] == _quote)
                {
                    _quoted[_numFields - 1] = true;
                    _state = State.Quoted;
                    pos++; // Quoted is OK with pos == stop
                }
                else
                {
                    // Set to Unquoted or UnquotedWhitespace.
                    _state = _unquotedState;
                }
            }

            int chunkStart;
            switch (_state)
            {
                case State.Unquoted:
                case State.UnquotedWhitespace:
                {
                    // NumPy's per-character loop stops at the first '\r' (EatCrlf), '\n' (LineEnd), delimiter — any whitespace
                    // character in whitespace mode — (Init) or comment (GotoLineEnd). One vectorized search over the stop set
                    // finds that character; the tests below then classify it in the loop's order, so the outcome is the
                    // loop's. Running off the buffer leaves pos == stop and the state unchanged, as the loop does.
                    chunkStart = pos;
                    int k = buf.Slice(pos, stop - pos).IndexOfAny(_stops);
                    if (k < 0)
                    {
                        pos = stop;
                    }
                    else
                    {
                        pos += k;
                        char c = buf[pos];
                        if (c == '\r')
                            _state = State.EatCrlf;
                        else if (c == '\n')
                            _state = State.LineEnd;
                        else if (_state == State.UnquotedWhitespace ? IsSpace(c) : c == _delimiter)
                            _state = State.Init;
                        else
                            _state = State.GotoLineEnd; // the only other stop character is the comment
                    }
                    CopyToFieldBuffer(buf.Slice(chunkStart, pos - chunkStart));
                    pos++;
                    break;
                }

                case State.Quoted:
                {
                    // Inside quotes only the quote character matters: one vectorized search for it.
                    chunkStart = pos;
                    int k = buf.Slice(pos, stop - pos).IndexOf((char)_quote);
                    if (k < 0)
                    {
                        pos = stop;
                    }
                    else
                    {
                        pos += k;
                        _state = State.QuotedCheckDoubleQuote;
                    }
                    CopyToFieldBuffer(buf.Slice(chunkStart, Math.Min(pos, stop) - chunkStart));
                    pos++;
                    break;
                }

                case State.QuotedCheckDoubleQuote:
                    if (buf[pos] == _quote)
                    {
                        // Copy the quote character directly from the config.
                        Span<char> q = stackalloc char[1];
                        q[0] = (char)_quote;
                        CopyToFieldBuffer(q);
                        _state = State.Quoted;
                        pos++;
                    }
                    else
                    {
                        // Continue parsing as if unquoted.
                        _state = _unquotedState;
                    }
                    break;

                case State.GotoLineEnd:
                {
                    if (_bufState != LoadTxtBufferState.MayContainNewline)
                    {
                        pos = stop; // advance to next buffer
                        _state = State.LineEnd;
                        break;
                    }
                    // The rest of a commented (or skipped) line: one vectorized search for its end.
                    int k = buf.Slice(pos, stop - pos).IndexOfAny(LineEnds);
                    if (k < 0)
                    {
                        pos = stop;
                    }
                    else
                    {
                        pos += k;
                        _state = buf[pos] == '\r' ? State.EatCrlf : State.LineEnd;
                    }
                    pos++;
                    break;
                }

                case State.EatCrlf:
                    // "Universal newline" support: remove \n in \r\n.
                    if (buf[pos] == '\n')
                        pos++;
                    _state = State.LineEnd;
                    break;
            }

            _pos = pos;
        }

        /// <summary>
        ///     Appends characters to the current field (<c>copy_to_field_buffer</c>), keeping room for the NUL separator and
        ///     the extra closing offset <see cref="AddField"/> writes.
        /// </summary>
        /// <param name="chunk">The characters.</param>
        private void CopyToFieldBuffer(ReadOnlySpan<char> chunk)
        {
            // Space for length +1 termination, +2 additional padding for AddField.
            int size = chunk.Length + _fieldBufferPos + 3;
            if (_fieldBuffer.Length < size)
                Array.Resize(ref _fieldBuffer, Math.Max(size, _fieldBuffer.Length * 2));
            chunk.CopyTo(_fieldBuffer.AsSpan(_fieldBufferPos));
            _fieldBufferPos += chunk.Length;
            _fieldBuffer[_fieldBufferPos] = '\0'; // always ensure we end with NUL
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
