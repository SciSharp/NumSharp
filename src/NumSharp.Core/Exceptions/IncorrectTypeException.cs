namespace NumSharp
{
    /// <summary>
    ///     Raised when a dtype cannot satisfy an operation — the ufunc "no loop matching the
    ///     specified signature and casting was found", a dtype= request with no matching loop, or a
    ///     dtype that a function forbids ("i0 not supported for complex values", "x may not be
    ///     complex"). Every one of these is a NumPy <c>TypeError</c>, which is why this derives from
    ///     <see cref="TypeError"/>: a caller writing <c>catch (TypeError)</c> (the way NumPy code
    ///     writes <c>except TypeError</c>) now catches it, while <c>catch (NumSharpException)</c>
    ///     still works because <see cref="TypeError"/> is itself a <see cref="NumSharpException"/>.
    ///     Do NOT re-parent this to <see cref="NumSharpException"/> directly — that silently drops it
    ///     out of the <c>TypeError</c> hierarchy and reopens the parity gap (a ufunc type error that
    ///     a <c>catch (TypeError)</c> misses).
    /// </summary>
    public class IncorrectTypeException : TypeError
    {
        /// <summary>Creates the exception with NumSharp's generic dtype-unsupported message.</summary>
        public IncorrectTypeException() : base("This method does not work with this dtype or was not already implemented.") { }

        /// <summary>Creates the exception with NumPy's verbatim ufunc/type message.</summary>
        /// <param name="message">The message, spelled exactly as NumPy 2.4.2 words it, so the differential-fuzz error tier matches byte-for-byte.</param>
        public IncorrectTypeException(string message) : base(message) { }

        /// <summary>
        ///     Creates the exception with the message and the exception that caused it, .NET's standard constructor every
        ///     NumSharp exception has (Python's <c>raise TypeError(message) from innerException</c>).
        /// </summary>
        /// <param name="message">The message, kept verbatim (NumPy's text when it mirrors a NumPy failure).</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public IncorrectTypeException(string message, System.Exception innerException) : base(message, innerException) { }
    }
}
