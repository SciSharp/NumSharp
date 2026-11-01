using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Construct an array by repeating <paramref name="A"/> the number of times given by <paramref name="reps"/>.
        ///     <para>
        ///     If <paramref name="reps"/> has length d, the result has dimension <c>max(d, A.ndim)</c>.
        ///     If <c>A.ndim &lt; d</c>, A is promoted to be d-dimensional by prepending size-1 axes.
        ///     If <c>A.ndim &gt; d</c>, <paramref name="reps"/> is promoted to A.ndim by prepending 1s.
        ///     </para>
        /// </summary>
        /// <param name="A">The input array.</param>
        /// <param name="reps">The number of repetitions of A along each axis. Each rep must be non-negative.</param>
        /// <returns>The tiled output array. Expanded outputs are C-contiguous; all-one reps produce a keep-order copy. Dtype matches <paramref name="A"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tile.html</remarks>
        /// <exception cref="ArgumentNullException">If <paramref name="A"/> or <paramref name="reps"/> is null.</exception>
        /// <exception cref="ArgumentException">If any element of <paramref name="reps"/> is negative.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray tile(NDArray A, params int[] reps)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(A))
                return NDMatrixBoundary.Wrapped((A, reps), static s => tile(s.A, s.reps));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(A))
                return NDCharArrayBoundary.Kept((A, reps), static s => tile(s.A, s.reps));
            if (A is null) throw new ArgumentNullException(nameof(A));
            if (reps is null) throw new ArgumentNullException(nameof(reps));

            // NumPy's statements are the array object's, over the repetitions widened to long.
            return NDArray.tile(A, ToLongArray(reps));
        }

        /// <summary>
        ///     Construct an array by repeating <paramref name="A"/> the number of times given by <paramref name="reps"/>.
        ///     <para>Long overload — see <see cref="tile(NDArray, int[])"/>.</para>
        /// </summary>
        /// <param name="A">The input array.</param>
        /// <param name="reps">The number of repetitions of A along each axis. Each rep must be non-negative.</param>
        /// <returns>The tiled output array (see <see cref="tile(NDArray, int[])"/>).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException">If <paramref name="A"/> or <paramref name="reps"/> is null.</exception>
        /// <exception cref="ArgumentException">If any element of <paramref name="reps"/> is negative (the message's index
        ///     counts the repetitions padded with leading 1s to <paramref name="A"/>'s rank).</exception>
        /// <remarks>
        ///     The statements are the array object's (<c>NDArray.tile</c>), which releases its intermediates: the interleaved
        ///     reshape of <paramref name="A"/>, its broadcast view and the materialized copy, the reshape of a lower-rank
        ///     <paramref name="A"/> on the all-ones path, and the zero-filled array behind an empty result's view.
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray tile(NDArray A, long[] reps)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(A))
                return NDMatrixBoundary.Wrapped((A, reps), static s => tile(s.A, s.reps));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(A))
                return NDCharArrayBoundary.Kept((A, reps), static s => tile(s.A, s.reps));
            if (A is null) throw new ArgumentNullException(nameof(A));
            if (reps is null) throw new ArgumentNullException(nameof(reps));

            // NumPy's statements (the padding, the all-ones copy, the empty result, the broadcast repeat) are the array
            // object's.
            return NDArray.tile(A, reps);
        }

        /// <summary>
        ///     Widens <see cref="int"/> repetitions to the <see cref="long"/> list <see cref="NDArray"/>'s tile reads (a copy:
        ///     the caller's array is never kept).
        /// </summary>
        /// <param name="arr">The repetitions (not null).</param>
        /// <returns>A new array of the same values.</returns>
        private static long[] ToLongArray(int[] arr)
        {
            var result = new long[arr.Length];
            for (int i = 0; i < arr.Length; i++) result[i] = arr[i];
            return result;
        }
    }
}
