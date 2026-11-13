using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Return the shape of an array (NumPy's <c>np.shape</c>, the module function beside the
        ///     <see cref="NDArray.shape"/> property).
        /// </summary>
        /// <param name="a">Input array (NumPy's keyword <c>a</c>). A C# scalar converts to a 0-d array and a C# array to
        ///     an array of its shape, so <c>np.shape(5)</c> is empty as NumPy's <c>()</c>.</param>
        /// <returns>The lengths of <paramref name="a"/>'s dimensions, NumPy's tuple as a fresh <c>long[]</c> (the same
        ///     copy <see cref="NDArray.shape"/> returns, so writing into it never reshapes the array); empty for a
        ///     0-d input.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (NumSharp has no object dtype for
        ///     NumPy's <c>np.shape(None)</c>, which is <c>()</c>; <see cref="size"/> refuses null the same way).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.shape.html
        ///     <para>NumPy's statements: <c>a.shape</c>, or <c>asarray(a).shape</c> for an object without the attribute (the
        ///     implicit conversions to <see cref="NDArray"/> are that <c>asarray</c> here). The return type follows
        ///     <see cref="NDArray.shape"/> and numpy.ma's <c>np.ma.shape</c>. No collision with the <c>shape</c> members of
        ///     <see cref="NDArray"/> or of the types nested in <c>np</c> (<c>np.broadcast.shape</c>, ...): those are members
        ///     of other classes, and inside a nested type its own member wins the name lookup.</para>
        /// </remarks>
        public static long[] shape(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NDArray.shape is already a copy of the shared dimension array (bug 2b of view-shape-open-bugs.md).
            return a.shape;
        }
    }
}
