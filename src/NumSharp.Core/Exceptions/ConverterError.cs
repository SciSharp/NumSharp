using System;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.lib._iotools.ConverterError</c> (also <c>numpy.lib.npyio.ConverterError</c>): a string value
    ///     that <c>np.genfromtxt</c>'s column converter cannot convert (<c>Converter #2 is locked and cannot be upgraded:
    ///     (occurred line #4 for value 'abc')</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from <see cref="NumSharpException"/>, as NumPy's derives from Python's <c>Exception</c> directly (it
    ///     is NOT a <see cref="ValueError"/>, so <c>catch (ValueError)</c> misses it, as <c>except ValueError</c> does).
    ///     Its subclass <see cref="ConverterLockError"/> reports a locked converter.
    ///     </para>
    ///     <para>
    ///     NumSharp does not raise it yet: it has no <c>np.genfromtxt</c> (and <c>np.loadtxt</c>, which it has, reports a
    ///     bad value as NumPy's <see cref="ValueError"/>). The class exists so a port of code that raises or catches it
    ///     compiles and behaves as in Python.
    ///     </para>
    /// </remarks>
    public class ConverterError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ConverterError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ConverterError())</c> is <c>''</c>.
        /// </summary>
        public ConverterError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim.</param>
        public ConverterError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise ConverterError(message)
        ///     from innerException</c> (NumPy's <c>genfromtxt</c> chains the converter's own failure this way).
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ConverterError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     NumPy's <c>numpy.lib._iotools.ConverterLockError</c>: an attempt to upgrade a locked column converter of
    ///     <c>np.genfromtxt</c> (one whose dtype the caller fixed) to a wider type.
    /// </summary>
    /// <remarks>
    ///     A subclass of <see cref="ConverterError"/>, as in NumPy, so <c>catch (ConverterError)</c> catches it. NumSharp
    ///     does not raise it yet (no <c>np.genfromtxt</c>); the class exists for ports.
    /// </remarks>
    public class ConverterLockError : ConverterError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ConverterLockError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ConverterLockError())</c> is <c>''</c>.
        /// </summary>
        public ConverterLockError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim (<c>Converter is locked and cannot be upgraded</c>, NumPy's
        ///     <c>StringConverter.upgrade</c>).</param>
        public ConverterLockError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     ConverterLockError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ConverterLockError(string message, Exception innerException) : base(message, innerException) { }
    }
}
