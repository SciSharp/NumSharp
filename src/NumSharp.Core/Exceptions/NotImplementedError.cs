using System;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>NotImplementedError</c>: an operation declared for an input it is not implemented for, or an abstract
    ///     method a subclass did not override. NumSharp raises it where NumPy does, with NumPy's text: a masked array's
    ///     <c>tofile</c>, the numpy.ma functions limited to two dimensions (<c>compress_rows</c>, <c>compress_cols</c>,
    ///     <c>compress_rowcols</c>, <c>mask_rowcols</c>, <c>notmasked_contiguous</c>) and a seedless
    ///     <c>SeedSequence</c>'s <c>generate_state</c>; a port throws it where its Python source raises it (a base class's
    ///     <c>raise NotImplementedError</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from <see cref="RuntimeError"/>, as Python's does (<c>NotImplementedError → RuntimeError →
    ///     Exception</c>), so <c>catch (RuntimeError)</c> catches it as <c>except RuntimeError</c> does.
    ///     </para>
    ///     <para>
    ///     Footgun: it is NOT a .NET <see cref="NotImplementedException"/>, and a <c>catch (NotImplementedException)</c> does
    ///     not see it. NumSharp keeps .NET's <see cref="NotImplementedException"/> / <see cref="NotSupportedException"/> for
    ///     what NumSharp itself lacks, so the two never mean the same thing: this one is NumPy's (or the ported code's)
    ///     refusal.
    ///     </para>
    /// </remarks>
    public class NotImplementedError : RuntimeError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise NotImplementedError</c> (the usual body of an abstract
        ///     method): an empty <see cref="Exception.Message"/>, as <c>str(NotImplementedError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>NotImplementedError</c>.</remarks>
        public NotImplementedError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error
        ///     (<c>MaskedArray.tofile() not implemented yet.</c>).</param>
        public NotImplementedError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     NotImplementedError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>
        ///     as Python keeps it in <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public NotImplementedError(string message, Exception innerException) : base(message, innerException) { }
    }
}
