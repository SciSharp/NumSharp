using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Return the number of dimensions of an array (NumPy's <c>np.ndim</c>, the module function beside the
        ///     <see cref="NDArray.ndim"/> property).
        /// </summary>
        /// <param name="a">Input array (NumPy's keyword <c>a</c>). A C# scalar converts to a 0-d array and a C# array to
        ///     an array of its rank, so <c>np.ndim(5)</c> is 0 as in NumPy.</param>
        /// <returns>The number of dimensions in <paramref name="a"/>; a scalar is zero-dimensional.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (NumSharp has no object dtype for
        ///     NumPy's <c>np.ndim(None)</c>, which is 0; <see cref="size"/> refuses null the same way).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndim.html
        ///     <para>NumPy's statements: <c>a.ndim</c>, or <c>asarray(a).ndim</c> for an object without the attribute (the
        ///     implicit conversions to <see cref="NDArray"/> are that <c>asarray</c> here). No collision with the
        ///     <c>ndim</c> members of <see cref="NDArray"/> or of the iterator / memory-view types nested in <c>np</c>: those
        ///     are instance members of other classes, and inside a nested type its own member wins the name lookup.</para>
        /// </remarks>
        public static int ndim(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            return a.ndim;
        }
    }
}
