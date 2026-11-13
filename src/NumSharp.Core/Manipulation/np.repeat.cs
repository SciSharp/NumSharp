using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Repeat elements of an array with per-element repeat counts. Mirrors NumPy
        ///     <c>np.repeat(a, repeats, axis)</c>: scalar / size-1 <paramref name="repeats"/> broadcasts to
        ///     every element along the (flattened or selected) axis; otherwise the length must match.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">Repeat counts. Either a 0-d/size-1 array (broadcast) or a 1-D array of length equal to <c>a.size</c> (axis=None) or <c>a.shape[axis]</c>.</param>
        /// <param name="axis">Axis along which to repeat. <c>null</c> flattens the input.</param>
        /// <returns>A new array with elements repeated according to <paramref name="repeats"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="repeats"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="repeats"/> has more than one dimension (<c>object too deep for desired
        ///     array</c>, checked first).</exception>
        /// <exception cref="TypeError"><paramref name="repeats"/> does not cast 'safely' to int64 — uint64, a float, complex,
        ///     decimal or text (<c>Cannot cast array data from dtype('float64') to dtype('int64') according to the rule
        ///     'safe'</c>; <c>Cannot cast scalar …</c> for a 0-d repeats).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ValueError">A negative entry of a per-element count array (NumPy's <c>repeats may not contain
        ///     negative values.</c>).</exception>
        /// <exception cref="ArgumentException">The repeats' length does not match, or one broadcast count is negative
        ///     (NumPy's text for that one is <c>negative dimensions are not allowed</c>: pinned).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        ///     <para>
        ///     NumPy converts the repeats with <c>PyArray_ContiguousFromAny(repeats, NPY_INTP, 0, 1)</c> before it reads the axis:
        ///     the rank (at most 1) first, then the 'safe' cast — so a 2-D repeats is the rank's ValueError whatever its dtype,
        ///     and both precede an out-of-range axis. (A 2-D repeats used to be flattened.)
        ///     </para>
        ///     <para>One signature, NumPy's <c>repeat(a, repeats, axis=None)</c>: the <c>int</c> / <c>long</c> count
        ///     overloads went with API-parity batch 3 (2026-11-11). A C# integer converts to a 0-d array, which broadcasts
        ///     over every element through the same tiling kernel, so <c>np.repeat(a, 3)</c> keeps its answer. The scalar form
        ///     <see cref="repeat{T}(T, NDArray)"/> stays beside it (the owner's call).</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // calculating overload: reclaims the ravel temps
        public static NDArray repeat(NDArray a, NDArray repeats, int? axis = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (repeats is null)
                throw new ArgumentNullException(nameof(repeats));
            return UnmanagedStorage.Repeat(a, repeats, axis);
        }

        /// <summary>
        ///     Repeat a C# scalar value: NumPy's <c>np.repeat(scalar, repeats)</c>, whose 0-d operand ravels to one
        ///     element, as a 1-D array of <typeparamref name="T"/>'s dtype.
        /// </summary>
        /// <typeparam name="T">The scalar's type; its dtype is the result's (a C# <c>int</c> gives int32).</typeparam>
        /// <param name="a">The value to repeat.</param>
        /// <param name="repeats">The count: a 0-d integer array (a C# integer converts to one) fills the result directly;
        ///     anything else - a size-1 array, a float count - takes <see cref="repeat(NDArray, NDArray, int?)"/>'s
        ///     route with <paramref name="a"/> as a 0-d array, NumPy's checks and texts included.</param>
        /// <returns>A new 1-D array of <paramref name="a"/> repeated.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="repeats"/> is null.</exception>
        /// <exception cref="ArgumentException">A negative count (NumPy raises ValueError; pinned as an [OpenBugs] case with
        ///     the array route's same text).</exception>
        /// <exception cref="ValueError">Counts that do not broadcast to one element, or a repeats of more than one
        ///     dimension.</exception>
        /// <exception cref="TypeError">A repeats that does not cast 'safely' to int64.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        ///     <para>NumSharp's scalar form, kept by the owner beside NumPy's one signature (API-parity batch 3): it
        ///     replaces the former <c>repeat&lt;T&gt;(T, int)</c> / <c>repeat&lt;T&gt;(T, long)</c> pair, so a C# value
        ///     needs no 0-d wrapper for the common count.</para></remarks>
        public static unsafe NDArray repeat<T>(T a, NDArray repeats) where T : unmanaged
        {
            if (repeats is null)
                throw new ArgumentNullException(nameof(repeats));

            // The common call, np.repeat(5, 3): a 0-d integer count fills the result directly (the former typed forms'
            // loop). Its checks are the array route's, so both forms answer alike.
            if (repeats.ndim == 0 && UnmanagedStorage.IsSafeToInt64(repeats.GetTypeCode))
            {
                long count = Convert.ToInt64(repeats.GetAtIndex(0), System.Globalization.CultureInfo.InvariantCulture);
                if (count < 0)
                    throw new ArgumentException("repeats may not contain negative values");

                if (count == 0)
                    return new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(0));

                var ret = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(count));
                var dst = (T*)ret.Address;
                for (long j = 0; j < count; j++)
                    dst[j] = a;
                return ret;
            }

            // Every other count is NumPy's general route over the 0-d operand: broadcast, rank and cast checks, texts.
            using var scalar = NDArray.Scalar(a);
            return repeat(scalar, repeats);
        }
    }
}
