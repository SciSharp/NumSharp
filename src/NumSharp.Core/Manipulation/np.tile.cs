using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Construct an array by repeating <paramref name="A"/> the number of times given by <paramref name="reps"/> —
        ///     NumPy's <c>np.tile(A, reps)</c>.
        ///     <para>
        ///     If <paramref name="reps"/> has length d, the result has dimension <c>max(d, A.ndim)</c>.
        ///     If <c>A.ndim &lt; d</c>, A is promoted to be d-dimensional by prepending size-1 axes.
        ///     If <c>A.ndim &gt; d</c>, <paramref name="reps"/> is promoted to A.ndim by prepending 1s.
        ///     </para>
        /// </summary>
        /// <param name="A">The input array.</param>
        /// <param name="reps">The number of repetitions of A along each axis, as ONE argument: an int (<c>2</c>), a tuple
        ///     (<c>(2, 3)</c>), an <c>int[]</c> / <c>long[]</c> or a <see cref="Shape"/> (all convert implicitly; only the
        ///     values are read). An empty <paramref name="reps"/> is a copy of <paramref name="A"/>.</param>
        /// <returns>The tiled output array, of <paramref name="A"/>'s dtype: C-contiguous for a real repetition, a keep-order
        ///     copy for all-one repetitions. A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy
        ///     creates it as the subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="A"/> is null.</exception>
        /// <exception cref="ValueError">A negative repetition that NumPy's <c>repeat</c> reaches while the tile still holds
        ///     elements (<c>negative dimensions are not allowed</c>); a <c>numpy.matrix</c> operand whose result cannot be a
        ///     matrix (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException">A negative repetition of an empty tile: NumPy's final reshape refuses
        ///     the negative extent (<c>cannot reshape array of size 0 into shape (0,newaxis)</c>, NumPy's ValueError text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.tile.html — NumPy's statements are the array object's
        ///     (<c>NDArray.tile</c>), which releases its intermediates: the interleaved reshape of <paramref name="A"/>, its
        ///     broadcast view and the materialized copy, the reshape of a lower-rank <paramref name="A"/> on the all-ones
        ///     path, and the zero-filled array behind an empty result's view.
        ///     <para>One signature since API-parity batch 8 (2026-11-12): the <c>params int[] reps</c> and <c>long[] reps</c>
        ///     overloads went. A tuple now compiles (<c>np.tile(a, (2, 2))</c>), and the loose <c>np.tile(a, 2, 2)</c> — NumPy's
        ///     <c>TypeError: tile() takes 2 positional arguments but 3 were given</c> — no longer does. A negative repetition
        ///     raised an ArgumentException naming the padded index before.</para>
        /// </remarks>
        public static NDArray tile(NDArray A, Shape reps)
        {
            if (A is null) throw new ArgumentNullException(nameof(A));

            // Only the repetition counts are NumPy's `tuple(reps)`: a default(Shape) has none, which is the empty tuple.
            return NDArray.tile(A, reps.dimensions ?? Array.Empty<long>());
        }
    }
}
