using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.linalg.lapack_lite.LapackError</c>: the error NumPy's bundled <c>lapack_lite</c> module raises
    ///     when a LAPACK call gets an argument it cannot pass (<c>Parameter a is not contiguous in lapack_lite.dgelsd</c>,
    ///     <c>Parameter a is not of type float64 in lapack_lite.dgelsd</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from <see cref="NumSharpException"/>, as NumPy's derives from Python's <c>Exception</c> directly (it
    ///     is NOT a <see cref="LinAlgError"/>: that one reports a mathematical failure, this one a wrong call).
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it: it has no <c>lapack_lite</c> module, and its LAPACK seam
    ///     (<c>TensorEngine.Blas</c>) prepares every argument itself. The class exists so a port of code that raises or
    ///     catches it compiles and behaves as in Python.
    ///     </para>
    /// </remarks>
    public class LapackError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise LapackError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(LapackError())</c> is <c>''</c>.
        /// </summary>
        public LapackError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim (<c>Parameter a is not contiguous in lapack_lite.dgelsd</c>).</param>
        public LapackError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise LapackError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public LapackError(string message, Exception innerException) : base(message, innerException) { }
    }
}
