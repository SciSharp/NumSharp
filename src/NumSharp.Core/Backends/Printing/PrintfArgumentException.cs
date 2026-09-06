namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     Signals the two conditions under which CPython's <c>%</c> operator raises a
    ///     <see cref="TypeError"/> while formatting a row tuple in <see cref="PrintfFormatter.FormatRow"/>:
    ///     an argument-count mismatch, or a <c>%c</c> conversion on a non-integral operand.
    /// </summary>
    /// <remarks>
    ///     It IS a <see cref="TypeError"/> so that <c>np.savetxt</c>'s complex branch — which, matching
    ///     NumPy, does not wrap the failure — surfaces a plain <see cref="TypeError"/> with CPython's raw
    ///     text, while its real branch catches this type and re-raises the dtype/format mismatch message.
    /// </remarks>
    internal sealed class PrintfArgumentException : TypeError
    {
        public PrintfArgumentException(string message) : base(message) { }
    }
}
