using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>StopIteration</c>: an iterator has no more items. A generator's <c>return value</c> travels in it as
    ///     <see cref="value"/>, which is how Python's <c>yield from</c> receives a sub-generator's result.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from <see cref="NumSharpException"/>: .NET has no exception with the same meaning (an
    ///     <see cref="System.Collections.IEnumerator"/> ends by returning false from <c>MoveNext</c>, and NumSharp's
    ///     iterators — <c>np.nditer</c>, <c>np.ndindex</c>, <c>flatiter</c> — are .NET enumerables that end the same way).
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it; the class exists so a port of code that drives an iterator by hand
    ///     (<c>next(it)</c> until <c>StopIteration</c>) compiles and behaves as in Python.
    ///     </para>
    /// </remarks>
    public class StopIteration : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise StopIteration</c>: an empty
        ///     <see cref="Exception.Message"/> and a null <see cref="value"/>, as <c>str(StopIteration())</c> is <c>''</c>
        ///     and its <c>value</c> is <c>None</c>.
        /// </summary>
        public StopIteration() : base(string.Empty) { }

        /// <summary>
        ///     Python's <c>StopIteration(text)</c>: the text is both the message and <see cref="value"/>.
        /// </summary>
        /// <param name="message">The text, or null for Python's <c>StopIteration(None)</c> (whose message is <c>None</c>
        ///     and value null).</param>
        /// <remarks>Python keeps a lone argument as <c>value</c> whatever it is; a string argument binds here, any other
        ///     value to <see cref="StopIteration(object)"/>, with the same result.</remarks>
        public StopIteration(string message) : base(message ?? "None")
        {
            value = message;
        }

        /// <summary>
        ///     Python's <c>StopIteration(value)</c>: a generator's return value. The message is Python's <c>str(value)</c>
        ///     (<c>5</c>, <c>1.5</c>, <c>True</c>, <c>None</c>).
        /// </summary>
        /// <param name="value">The value, or null for <c>None</c>.</param>
        /// <exception cref="ValueError"><paramref name="value"/> is a Python int (<see cref="System.Numerics.BigInteger"/>)
        ///     of more than 4300 decimal digits, whose <c>str()</c> CPython refuses.</exception>
        public StopIteration(object value) : base(NDPyObject.Str(value))
        {
            this.value = value;
        }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise StopIteration(message)
        ///     from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>; the message is also
        ///     <see cref="value"/>.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public StopIteration(string message, Exception innerException) : base(message ?? "None", innerException)
        {
            value = message;
        }

        /// <summary>
        ///     The iterator's final value (Python's <c>value</c>): the constructor's argument, or null after the bare
        ///     constructor (Python's <c>None</c>).
        /// </summary>
        public object value { get; }
    }

    /// <summary>
    ///     Python's <c>StopAsyncIteration</c>: an asynchronous iterator (<c>__anext__</c>) has no more items.
    /// </summary>
    /// <remarks>
    ///     Unlike <see cref="StopIteration"/> it has no <c>value</c> attribute, as in Python. It derives from
    ///     <see cref="NumSharpException"/> (an <see cref="System.Collections.Generic.IAsyncEnumerator{T}"/> ends by
    ///     returning false); NumSharp never raises it, and the class exists for ports.
    /// </remarks>
    public class StopAsyncIteration : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise StopAsyncIteration</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(StopAsyncIteration())</c> is <c>''</c>.
        /// </summary>
        public StopAsyncIteration() : base(string.Empty) { }

        /// <summary>Creates the error with the port's message.</summary>
        /// <param name="message">The text, verbatim.</param>
        public StopAsyncIteration(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     StopAsyncIteration(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public StopAsyncIteration(string message, Exception innerException) : base(message, innerException) { }
    }
}
