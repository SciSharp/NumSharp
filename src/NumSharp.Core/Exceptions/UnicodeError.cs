using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible <c>UnicodeError</c> — Python's base class for text encoding and decoding failures, a
    ///     subclass of <see cref="ValueError"/>. NumSharp raises its two concrete subclasses,
    ///     <see cref="UnicodeEncodeError"/> and <see cref="UnicodeDecodeError"/>, when a value crosses between a
    ///     fixed-width Unicode (<c>U</c>) and a byte-string (<c>S</c>) dtype the way NumPy's ASCII codec refuses it.
    /// </summary>
    /// <remarks>
    ///     Catching <see cref="ValueError"/> (or <see cref="ArgumentException"/>) catches these too, exactly as
    ///     <c>except ValueError</c> catches Python's Unicode errors. The message is CPython's verbatim
    ///     <c>UnicodeError.__str__</c> text, so error parity with NumPy is a string comparison.
    /// </remarks>
    public class UnicodeError : ValueError
    {
        /// <summary>Creates the error with CPython's message text.</summary>
        /// <param name="message">The verbatim message (<c>'ascii' codec can't encode …</c>).</param>
        public UnicodeError(string message) : base(message) { }

        /// <summary>Creates the error with CPython's message text and the failure that caused it.</summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The underlying failure.</param>
        public UnicodeError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     NumPy-compatible <c>UnicodeEncodeError</c>: a Unicode string held a character the target encoding cannot
    ///     represent. NumSharp raises it where NumPy's ASCII codec does — a <c>U</c> value with a code point at or above
    ///     128 stored into an <c>S</c> (byte-string) array, cast to <c>S</c>, or given to <c>np.array(…, dtype='S')</c>.
    /// </summary>
    /// <remarks>
    ///     The message follows CPython's <c>UnicodeEncodeError.__str__</c> exactly: one offending character reads
    ///     <c>'ascii' codec can't encode character '\xe9' in position 1: ordinal not in range(128)</c> (the character as
    ///     <c>\xNN</c>, <c>\uNNNN</c> or <c>\UNNNNNNNN</c>), a run of consecutive offending characters reads
    ///     <c>'ascii' codec can't encode characters in position 1-2: ordinal not in range(128)</c>. Positions count code
    ///     points of the element, not UTF-16 units, as Python does.
    /// </remarks>
    public class UnicodeEncodeError : UnicodeError
    {
        /// <summary>Creates the error with CPython's message text.</summary>
        /// <param name="message">The verbatim message.</param>
        public UnicodeEncodeError(string message) : base(message) { }
    }

    /// <summary>
    ///     NumPy-compatible <c>UnicodeDecodeError</c>: bytes that the source encoding cannot read. NumSharp raises it where
    ///     NumPy's ASCII codec does — an <c>S</c> (byte-string) value holding a byte at or above <c>0x80</c> stored into a
    ///     <c>U</c> array, cast to <c>U</c>, or given to <c>np.array(…, dtype='U')</c>.
    /// </summary>
    /// <remarks>
    ///     The message follows CPython's <c>UnicodeDecodeError.__str__</c> for the ASCII codec, which reports the FIRST
    ///     bad byte only: <c>'ascii' codec can't decode byte 0xe9 in position 1: ordinal not in range(128)</c>.
    /// </remarks>
    public class UnicodeDecodeError : UnicodeError
    {
        /// <summary>Creates the error with CPython's message text.</summary>
        /// <param name="message">The verbatim message.</param>
        public UnicodeDecodeError(string message) : base(message) { }
    }
}
