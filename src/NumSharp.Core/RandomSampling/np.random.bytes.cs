using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return random bytes.
        /// </summary>
        /// <param name="length">Number of random bytes.</param>
        /// <returns>
        ///     A 1-D <see cref="NDArray{T}"/> of <see cref="byte"/> (dtype <c>uint8</c>), length
        ///     <paramref name="length"/> — the NumSharp analogue of NumPy's <c>bytes</c> object.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.bytes.html
        ///     <br/>
        ///     Byte-identical to NumPy's <c>RandomState.bytes</c>: it draws
        ///     <c>ceil(length/4)</c> uint32 values from the MT19937 stream, packs them
        ///     little-endian, and truncates to <paramref name="length"/> bytes. The word
        ///     count follows NumPy's exact formula <c>(length - 1) / 4 + 1</c> (C integer
        ///     division — <c>mtrand.pyx</c> sets <c>cdivision=True</c>, so <c>//</c> truncates
        ///     toward zero and C# integer division matches), so the same seed consumes the same
        ///     amount of state — including the quirks that <c>bytes(0)</c> still draws one word
        ///     and a negative length Python-slices the tail (NumPy's <c>tobytes()[:length]</c>).
        ///     <br/>
        ///     NumPy's <c>length</c> is a 64-bit <c>npy_intp</c> and its <c>bytes</c> result can
        ///     exceed 2 GiB. NumSharp returns an <see cref="NDArray{T}"/> backed by unmanaged
        ///     memory (addressed by <c>long</c>), so it matches that capability: a request larger
        ///     than a managed <c>byte[]</c> can hold (<see cref="Array.MaxLength"/> ≈ 2 GiB) still
        ///     succeeds. To recover a managed array from a small result use
        ///     <c>bytes(n).ToArray&lt;byte&gt;()</c> (itself capped at <see cref="Array.MaxLength"/>).
        /// </remarks>
        public NDArray<byte> bytes(long length)
            => BytesCore(length, static self => self.randomizer.NextUInt32(), this);

        // Shared byte-string builder for RandomState.bytes and Generator.bytes. Draws exactly
        // ceil(length/4) uint32 words (matching NumPy's stream consumption regardless of the final
        // slice) and stores only the first 'end' little-endian bytes into a fresh unmanaged
        // NDArray<byte> — which, unlike a byte[], can exceed 2 GiB (NumPy's npy_intp length).
        internal static unsafe NDArray<byte> BytesCore<TState>(long length, Func<TState, uint> nextUInt32, TState state)
        {
            // NumPy: n_uint32 = ((length - 1) // 4 + 1) where '//' is C truncation-toward-zero
            //        because 'length' is typed npy_intp and the random pyx set cdivision=True.
            //        C# integer division matches.
            long nUint32 = (length - 1) / 4 + 1;

            if (nUint32 < 0)
                // Reaches np.random.randint(size=n_uint32) with a negative size in NumPy.
                throw new ValueError("negative dimensions are not allowed");

            long totalBytes = nUint32 * 4;

            // Python slice tobytes()[:length]: for length >= 0 keep the first 'length' bytes;
            // for length < 0 drop |length| bytes from the end (clamped at 0). Either way the result
            // is the first 'end' bytes of the little-endian word stream.
            long end = length >= 0
                ? Math.Min(length, totalBytes)
                : Math.Max(0, totalBytes + length);

            // A 1-D uint8 array of exactly 'end' elements. fillZeros:false — the loop below writes
            // every one of the 'end' bytes exactly once (fullWordsStored*4 + tailBytes == end).
            var result = new NDArray<byte>(end, false);
            byte* p = end > 0 ? (byte*)result.Address : null;

            long fullWordsStored = end / 4;
            int tailBytes = (int)(end % 4);
            long pos = 0;
            for (long w = 0; w < nUint32; w++)
            {
                uint r = nextUInt32(state); // every word is drawn (state consumption), some discarded
                if (w < fullWordsStored)
                {
                    p[pos] = (byte)r;
                    p[pos + 1] = (byte)(r >> 8);
                    p[pos + 2] = (byte)(r >> 16);
                    p[pos + 3] = (byte)(r >> 24);
                    pos += 4;
                }
                else if (w == fullWordsStored && tailBytes > 0)
                {
                    for (int b = 0; b < tailBytes; b++)
                        p[pos++] = (byte)(r >> (8 * b));
                }
            }
            return result;
        }
    }
}
