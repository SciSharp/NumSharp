namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     A <see cref="TypeError"/> that CPython's <c>%</c> operator raises while formatting (see
    ///     <see cref="PercentFormat.FormatInto"/>): an argument-count mismatch, a <c>%(key)</c> spec without a mapping, a
    ///     <c>*</c> argument that is not a Python <c>int</c>, or a conversion the value's type does not take
    ///     (<c>%x format: an integer is required, not numpy.float64</c>, <c>%c requires int or char</c>).
    /// </summary>
    /// <remarks>
    ///     It IS a <see cref="TypeError"/>, and its own type marks exactly the errors NumPy's <c>np.savetxt</c> catches:
    ///     the real branch catches it and re-raises the dtype / format mismatch message (the original as the cause),
    ///     while the complex branch and <c>ndarray.tofile</c>, which do not wrap the failure, surface a plain
    ///     <see cref="TypeError"/> with CPython's raw text. A <c>ValueError</c> or <c>OverflowError</c> of the same
    ///     formatting is not one of these and propagates as it is in every writer.
    /// </remarks>
    internal sealed class PrintfArgumentException : TypeError
    {
        /// <summary>Creates the error with CPython's message.</summary>
        /// <param name="message">CPython's text, verbatim (<c>not enough arguments for format string</c>).</param>
        public PrintfArgumentException(string message) : base(message) { }
    }
}
