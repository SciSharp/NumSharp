using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     One committed JSONL corpus file held as raw UTF-8 for a single pass: it counts its cases without
    ///     parsing them and hands them out ONE AT A TIME, so a replay loop keeps only the case it is working
    ///     on alive instead of the whole file's parsed list.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why streaming matters here.</b> A parsed <see cref="FuzzCorpus.Case"/> carries its operand and
    ///     expected buffers as hex strings — UTF-16, so four bytes per payload byte — and the corpus is ~170 MB of
    ///     them. <see cref="FuzzCorpus.Load"/> materializes a whole file, so every case of a 17 MB tier survives
    ///     until the tier ends and the GC copies it gen0 → gen1 → gen2 on the way; in the Oracle test host that
    ///     promotion traffic, not the JSON parsing, was the dominant cost (a 2026-09-24 CPU trace attributed ~65 % of
    ///     the coverage gates' time and ~2 s of the leak gate's to GC pauses). Enumerating a <see cref="CorpusFile"/>
    ///     instead leaves each case garbage the moment the loop moves on, so it dies in gen0 for nothing.</para>
    ///     <para><b>Same bytes in, same cases out.</b> Lines follow exactly the rules the previous
    ///     <see cref="File.ReadLines(string)"/> + <see cref="string.IsNullOrWhiteSpace"/> loader applied (see
    ///     <see cref="TryReadLine"/>), and each line goes to the SAME serializer options through its UTF-8
    ///     overload, which skips the UTF-16 transcode of the whole line that the string overload paid.</para>
    ///     <para><b>Ownership.</b> The file's bytes live in a buffer rented from <see cref="ArrayPool{T}.Shared"/>
    ///     (tiers up to 17 MB, so reusing one large array beats a fresh large-object-heap allocation per tier);
    ///     <see cref="Dispose"/> returns it. Always use the instance in a <c>using</c>: an undisposed instance only
    ///     costs the pool its array (the GC still reclaims it), but an enumeration after
    ///     <see cref="Dispose"/> would read a buffer someone else may already be filling, so it is refused.</para>
    /// </remarks>
    public sealed class CorpusFile : IEnumerable<FuzzCorpus.Case>, IDisposable
    {
        private static ReadOnlySpan<byte> Utf8Bom => [0xEF, 0xBB, 0xBF];

        private byte[] _bytes;
        private readonly int _length;

        /// <summary>
        ///     Reads <paramref name="path"/> whole into a pooled buffer and counts its case lines.
        /// </summary>
        /// <param name="name">The corpus file name (for messages; <see cref="Name"/>).</param>
        /// <param name="path">The file's full path.</param>
        /// <exception cref="IOException">The file cannot be opened or read.</exception>
        /// <exception cref="InvalidDataException">The file is larger than one managed array can hold (2 GB).</exception>
        internal CorpusFile(string name, string path)
        {
            Name = name;
            _bytes = RentFileBytes(path, out _length);
            try
            {
                Count = CountLines(Content);
            }
            catch
            {
                // Counting cannot fail today, but a constructor that throws after renting must not strand the
                // pooled array — nobody could Dispose an instance that was never returned.
                ArrayPool<byte>.Shared.Return(_bytes);
                _bytes = null;
                throw;
            }
        }

        /// <summary>The corpus file name this instance was opened from (e.g. <c>reduce.jsonl</c>).</summary>
        public string Name { get; }

        /// <summary>
        ///     The number of cases the file holds — its non-blank lines, counted without parsing. Equal to the
        ///     <c>Count</c> of the list <see cref="FuzzCorpus.Load"/> returns for the same file, so a replay can
        ///     check its committed case floor BEFORE it spends time replaying.
        /// </summary>
        public int Count { get; }

        /// <summary>
        ///     The file's UTF-8 content after an optional byte-order mark — what the line walkers read.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The instance was disposed (its buffer is back in the pool).</exception>
        internal ReadOnlySpan<byte> Content
        {
            get
            {
                ObjectDisposedException.ThrowIf(_bytes == null, this);
                var all = new ReadOnlySpan<byte>(_bytes, 0, _length);
                // StreamReader (behind File.ReadLines) consumes a UTF-8 byte-order mark instead of yielding it as
                // text; skipping it keeps the first line identical to what that loader produced.
                return all.StartsWith(Utf8Bom) ? all[Utf8Bom.Length..] : all;
            }
        }

        /// <summary>
        ///     Parses and yields the cases in file order, one per non-blank line; each enumeration parses anew.
        /// </summary>
        /// <returns>An enumerator over freshly parsed cases.</returns>
        /// <exception cref="ObjectDisposedException">The instance was disposed.</exception>
        /// <exception cref="System.Text.Json.JsonException">A line is not valid corpus JSON (raised while enumerating).</exception>
        public IEnumerator<FuzzCorpus.Case> GetEnumerator()
        {
            // Checked eagerly (not on the first MoveNext) so a use-after-dispose fails at the call site.
            ObjectDisposedException.ThrowIf(_bytes == null, this);
            return Parse();
        }

        /// <inheritdoc cref="GetEnumerator"/>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        ///     The line-by-line parse behind <see cref="GetEnumerator"/>. An iterator cannot keep a span alive across
        ///     <c>yield</c>, so the cursor is an integer offset and each line is re-sliced from the buffer per step.
        /// </summary>
        /// <returns>The cases, lazily parsed.</returns>
        /// <remarks>
        ///     Every enumeration is watched (<see cref="CaseWatchdog"/>): the case handed out last is the one the replay loop
        ///     is running, so it arms the enumeration's slot with that case's id, and a case that never finishes terminates
        ///     the test host by name instead of hanging the run. The slot opens on the first <c>MoveNext</c> (the iterator
        ///     body starts there) and closes in the iterator's <c>finally</c> — on completion, on an early exit from the
        ///     consumer's loop, and when a parse throws.
        /// </remarks>
        private IEnumerator<FuzzCorpus.Case> Parse()
        {
            using CaseWatchdog.Slot watch = CaseWatchdog.Watch(Name);
            int position = 0;
            while (true)
            {
                FuzzCorpus.Case next;
                {
                    // Re-reading Content each step also re-checks disposal: a Dispose between two MoveNext calls
                    // must not let the next step parse a buffer the pool may have handed to someone else.
                    ReadOnlySpan<byte> content = Content;
                    if (!TryReadLine(content, ref position, out int start, out int length))
                        yield break;
                    next = FuzzCorpus.ParseLine(content.Slice(start, length));
                }
                // The consumer runs this case between this yield and its next MoveNext. (A literal `null` line, which no
                // generator writes, parses to null and disarms the slot rather than throwing here.)
                watch.Enter(next?.Id);
                yield return next;
            }
        }

        /// <summary>
        ///     Walks the non-blank lines of the content as UTF-8 spans, for callers that read a line without
        ///     materializing a <see cref="FuzzCorpus.Case"/> (the header-only <see cref="CorpusSurvey"/>).
        /// </summary>
        /// <returns>A <c>foreach</c>-able enumerator of line spans; each span is valid until the next step.</returns>
        /// <exception cref="ObjectDisposedException">The instance was disposed.</exception>
        internal LineEnumerator EnumerateLines() => new(Content);

        /// <summary>
        ///     Returns the file's buffer to the pool. Idempotent; the instance cannot be enumerated afterwards.
        /// </summary>
        public void Dispose()
        {
            byte[] bytes = _bytes;
            _bytes = null;
            if (bytes != null)
                ArrayPool<byte>.Shared.Return(bytes);
        }

        /// <summary>
        ///     Reads a whole file into an array rented from <see cref="ArrayPool{T}.Shared"/>.
        /// </summary>
        /// <param name="path">The file to read.</param>
        /// <param name="length">Receives the number of bytes read (the rented array may be longer).</param>
        /// <returns>The rented array; the caller returns it to <see cref="ArrayPool{T}.Shared"/>.</returns>
        /// <exception cref="IOException">The file cannot be opened or read.</exception>
        /// <exception cref="InvalidDataException">The file does not fit in one managed array.</exception>
        internal static byte[] RentFileBytes(string path, out int length)
        {
            // FileShare.Read is what File.ReadLines opened with, so a reader racing a regeneration behaves the same.
            using SafeFileHandle handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                                                          FileOptions.SequentialScan);
            long fileLength = RandomAccess.GetLength(handle);
            if (fileLength > Array.MaxLength)
                throw new InvalidDataException($"corpus file '{path}' is {fileLength} bytes — larger than one managed array");

            byte[] buffer = ArrayPool<byte>.Shared.Rent((int)fileLength);
            try
            {
                int total = 0;
                while (total < fileLength)
                {
                    int read = RandomAccess.Read(handle, buffer.AsSpan(total, (int)fileLength - total), total);
                    if (read == 0)
                        break;   // the file shrank under us: keep what exists, exactly as a streaming reader would
                    total += read;
                }
                length = total;
                return buffer;
            }
            catch
            {
                // A read failure must not strand the rented array.
                ArrayPool<byte>.Shared.Return(buffer);
                throw;
            }
        }

        /// <summary>
        ///     Counts the non-blank lines of <paramref name="content"/> — the number of cases it holds.
        /// </summary>
        /// <param name="content">UTF-8 content (byte-order mark already skipped).</param>
        /// <returns>The case-line count.</returns>
        internal static int CountLines(ReadOnlySpan<byte> content)
        {
            int position = 0, count = 0;
            while (TryReadLine(content, ref position, out _, out _))
                count++;
            return count;
        }

        /// <summary>
        ///     Finds the next non-blank line at or after <paramref name="position"/>.
        /// </summary>
        /// <remarks>
        ///     The rules are <see cref="StreamReader.ReadLine"/>'s — <c>\n</c>, <c>\r\n</c> and a lone <c>\r</c> each end
        ///     a line — followed by <see cref="string.IsNullOrWhiteSpace"/>'s blank test, so the lines are exactly the
        ///     ones the previous string loader deserialized. Splitting at every <c>\r</c> and <c>\n</c> separately turns a
        ///     <c>\r\n</c> pair into a line end plus an empty line, which the blank test then drops — the same outcome.
        /// </remarks>
        /// <param name="content">UTF-8 content (byte-order mark already skipped).</param>
        /// <param name="position">The scan cursor; advanced past the returned line (and past skipped blank lines).</param>
        /// <param name="start">Receives the line's start offset within <paramref name="content"/>.</param>
        /// <param name="length">Receives the line's length in bytes (terminator excluded).</param>
        /// <returns>False when no non-blank line remains.</returns>
        internal static bool TryReadLine(ReadOnlySpan<byte> content, ref int position, out int start, out int length)
        {
            while (position < content.Length)
            {
                ReadOnlySpan<byte> rest = content[position..];
                int end = rest.IndexOfAny((byte)'\n', (byte)'\r');
                int lineLength = end < 0 ? rest.Length : end;
                int lineStart = position;
                position += end < 0 ? rest.Length : end + 1;
                if (!IsBlank(content.Slice(lineStart, lineLength)))
                {
                    start = lineStart;
                    length = lineLength;
                    return true;
                }
            }
            start = length = 0;
            return false;
        }

        /// <summary>
        ///     <see cref="string.IsNullOrWhiteSpace"/> for one UTF-8 line, without decoding the common case.
        /// </summary>
        /// <param name="line">The line, terminator excluded.</param>
        /// <returns>True when the line is empty or whitespace only.</returns>
        internal static bool IsBlank(ReadOnlySpan<byte> line)
        {
            foreach (byte b in line)
            {
                // A non-ASCII byte may begin a Unicode space (U+00A0, U+2028, …) that IsNullOrWhiteSpace counts as
                // blank: decide on the decoded text. A corpus line starts with '{' and exits on its first byte anyway.
                if (b >= 0x80)
                    return string.IsNullOrWhiteSpace(Encoding.UTF8.GetString(line));
                if (!char.IsWhiteSpace((char)b))
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     A <c>foreach</c> enumerator over the non-blank lines of UTF-8 content (see <see cref="TryReadLine"/>).
        /// </summary>
        internal ref struct LineEnumerator
        {
            private readonly ReadOnlySpan<byte> _content;
            private int _position;

            /// <summary>Starts an enumeration at the beginning of <paramref name="content"/>.</summary>
            /// <param name="content">UTF-8 content (byte-order mark already skipped).</param>
            internal LineEnumerator(ReadOnlySpan<byte> content)
            {
                _content = content;
                _position = 0;
                Current = default;
            }

            /// <summary>The current line (terminator excluded); valid until the next <see cref="MoveNext"/>.</summary>
            public ReadOnlySpan<byte> Current { get; private set; }

            /// <summary>Returns this enumerator, so the struct itself is what <c>foreach</c> drives.</summary>
            /// <returns>This enumerator.</returns>
            public readonly LineEnumerator GetEnumerator() => this;

            /// <summary>Advances to the next non-blank line.</summary>
            /// <returns>False when the content is exhausted.</returns>
            public bool MoveNext()
            {
                if (!TryReadLine(_content, ref _position, out int start, out int length))
                    return false;
                Current = _content.Slice(start, length);
                return true;
            }
        }
    }
}
