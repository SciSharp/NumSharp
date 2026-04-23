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
        /// <returns>The tiled output array. Always C-contiguous, dtype matches <paramref name="A"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tile.html</remarks>
        /// <exception cref="ArgumentNullException">If <paramref name="A"/> or <paramref name="reps"/> is null.</exception>
        /// <exception cref="ArgumentException">If any element of <paramref name="reps"/> is negative.</exception>
        public static NDArray tile(NDArray A, params int[] reps)
        {
            if (A is null) throw new ArgumentNullException(nameof(A));
            if (reps is null) throw new ArgumentNullException(nameof(reps));

            return tile(A, ToLongArray(reps));
        }

        /// <summary>
        ///     Construct an array by repeating <paramref name="A"/> the number of times given by <paramref name="reps"/>.
        ///     <para>Long overload — see <see cref="tile(NDArray, int[])"/>.</para>
        /// </summary>
        public static NDArray tile(NDArray A, long[] reps)
        {
            if (A is null) throw new ArgumentNullException(nameof(A));
            if (reps is null) throw new ArgumentNullException(nameof(reps));

            int d = reps.Length;
            int aDim = A.ndim;
            int outDim = Math.Max(d, aDim);

            // Pad A's shape with leading 1s when reps has more entries than A.ndim.
            // Pad reps with leading 1s when A.ndim is larger than reps' length.
            // Both yield a common ndim = max(d, aDim) where in[i] aligns with rep[i].
            var aShape = new long[outDim];
            var tup = new long[outDim];
            for (int i = 0; i < outDim - aDim; i++) aShape[i] = 1;
            for (int i = 0; i < aDim; i++) aShape[outDim - aDim + i] = A.shape[i];
            for (int i = 0; i < outDim - d; i++) tup[i] = 1;
            for (int i = 0; i < d; i++) tup[outDim - d + i] = reps[i];

            for (int i = 0; i < outDim; i++)
                if (tup[i] < 0)
                    throw new ArgumentException($"reps[{i}] must be non-negative, got {tup[i]}.", nameof(reps));

            // Compute output shape.
            var outShape = new long[outDim];
            long outSize = 1;
            for (int i = 0; i < outDim; i++)
            {
                outShape[i] = aShape[i] * tup[i];
                outSize *= outShape[i];
            }

            // Empty result: any rep==0 or any aShape[i]==0 → return zero-element array of the
            // correct shape and dtype. NumPy: tile([], 3) → array([], shape=(0,), dtype=float64).
            if (outSize == 0)
                return zeros(new Shape(outShape), A.dtype);

            // Trivial case: all reps are 1 → return a copy preserving the (possibly promoted) shape.
            // Matches NumPy's array(A, copy=True, ndmin=d) shortcut.
            bool allOnes = true;
            for (int i = 0; i < outDim; i++) if (tup[i] != 1) { allOnes = false; break; }
            if (allOnes)
            {
                var c = aDim == outDim ? A.copy() : A.reshape(new Shape(aShape)).copy();
                return c;
            }

            // General case: insert size-1 axes between A's axes to create a tile axis next to each
            // input axis, then broadcast and copy to materialize, then collapse.
            //
            //   A.shape (a0, a1, ..., a_{n-1})
            //     ↓ reshape to interleaved (1, a0, 1, a1, ..., 1, a_{n-1})
            //     ↓ broadcast_to (r0, a0, r1, a1, ..., r_{n-1}, a_{n-1})  — each leading 1 expands
            //     ↓ copy() → contiguous (size = product of all)
            //     ↓ reshape to (r0*a0, r1*a1, ..., r_{n-1}*a_{n-1})
            //
            // This composes broadcast + copy + reshape (all O(N)) and produces NumPy-aligned output.
            var interleaved = new long[2 * outDim];
            var broadcastTarget = new long[2 * outDim];
            for (int i = 0; i < outDim; i++)
            {
                interleaved[2 * i] = 1;
                interleaved[2 * i + 1] = aShape[i];
                broadcastTarget[2 * i] = tup[i];
                broadcastTarget[2 * i + 1] = aShape[i];
            }

            var promoted = A.reshape(new Shape(interleaved));
            var broadcasted = broadcast_to(promoted, new Shape(broadcastTarget));
            var contiguous = broadcasted.copy();
            return contiguous.reshape(new Shape(outShape));
        }

        private static long[] ToLongArray(int[] arr)
        {
            var result = new long[arr.Length];
            for (int i = 0; i < arr.Length; i++) result[i] = arr[i];
            return result;
        }
    }
}
