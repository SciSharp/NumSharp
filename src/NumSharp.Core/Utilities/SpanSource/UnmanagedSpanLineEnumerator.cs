// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Collections.Generic;

namespace System.Text
{
    /// <summary>
    /// Enumerates the lines of a <see cref="ReadOnlyUnmanagedSpan{Char}"/>.
    /// </summary>
    /// <remarks>
    /// To get an instance of this type, use <see cref="MemoryExtensions.EnumerateLines(ReadOnlyUnmanagedSpan{char})"/>.
    /// </remarks>
    public ref struct UnmanagedSpanLineEnumerator : IEnumerator<ReadOnlyUnmanagedSpan<char>>
    {
        private ReadOnlyUnmanagedSpan<char> _remaining;
        private ReadOnlyUnmanagedSpan<char> _current;
        private bool _isEnumeratorActive;

        internal UnmanagedSpanLineEnumerator(ReadOnlyUnmanagedSpan<char> buffer)
        {
            _remaining = buffer;
            _current = default;
            _isEnumeratorActive = true;
        }

        /// <summary>
        /// Gets the line at the current position of the enumerator.
        /// </summary>
        public ReadOnlyUnmanagedSpan<char> Current => _current;

        /// <summary>
        /// Returns this instance as an enumerator.
        /// </summary>
        public UnmanagedSpanLineEnumerator GetEnumerator() => this;

        /// <summary>
        /// Advances the enumerator to the next line of the span.
        /// </summary>
        /// <returns>
        /// True if the enumerator successfully advanced to the next line; false if
        /// the enumerator has advanced past the end of the span.
        /// </returns>
        public bool MoveNext()
        {
            if (!_isEnumeratorActive)
            {
                _current = default;
                return false; // EOF previously reached or enumerator was never initialized
            }

            ReadOnlyUnmanagedSpan<char> remaining = _remaining;

            int idx = remaining.IndexOfAny(string.SearchValuesStorage.NewLineChars);

            if ((uint)idx < (uint)remaining.Length)
            {
                int stride = 1;

                if (remaining[idx] == '\r' && (uint)(idx + 1) < (uint)remaining.Length && remaining[idx + 1] == '\n')
                {
                    stride = 2;
                }

                _current = remaining.Slice(0, idx);
                _remaining = remaining.Slice(idx + stride);
            }
            else
            {
                // We've reached EOF, but we still need to return 'true' for this final
                // iteration so that the caller can query the Current property once more.

                _current = remaining;
                _remaining = default;
                _isEnumeratorActive = false;
            }

            return true;
        }

        /// <inheritdoc />
        object IEnumerator.Current => throw new NotSupportedException();

        /// <inheritdoc />
        void IEnumerator.Reset() => throw new NotSupportedException();

        /// <inheritdoc />
        void IDisposable.Dispose() { }
    }
}
