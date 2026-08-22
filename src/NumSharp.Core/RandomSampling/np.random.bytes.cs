using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return random bytes.
        /// </summary>
        /// <param name="length">Number of random bytes.</param>
        /// <returns>A byte array of length <paramref name="length"/>.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.bytes.html
        ///     <br/>
        ///     Byte-identical to NumPy's <c>RandomState.bytes</c>: it draws
        ///     <c>ceil(length/4)</c> uint32 values from the MT19937 stream, packs them
        ///     little-endian, and truncates to <paramref name="length"/> bytes. The word
        ///     count follows NumPy's exact formula <c>(length - 1) / 4 + 1</c> (C integer
        ///     division), so the same seed consumes the same amount of state — including
        ///     the quirks that <c>bytes(0)</c> still draws one word and a negative length
        ///     Python-slices the tail (NumPy's <c>tobytes()[:length]</c>).
        ///     <br/>
        ///     NumPy's <c>length</c> is a 64-bit <c>npy_intp</c> and it returns a Python <c>bytes</c>
        ///     object, so it can exceed 2 GiB. A .NET <c>byte[]</c> cannot (<see cref="Array.MaxLength"/>
        ///     ≈ 2.0 GiB), so a request whose result would exceed that raises — the same <c>byte[]</c>
        ///     ceiling <c>np.save</c>/<c>np.savez</c> carry. For a &gt; 2 GiB array of random bytes use
        ///     an NDArray path such as <c>default_rng().integers(0, 256, size, np.uint8)</c>.
        /// </remarks>
        public byte[] bytes(long length)
            => BytesCore(length, static self => self.randomizer.NextUInt32(), this);

        // Shared byte-string builder for RandomState.bytes and Generator.bytes. Draws exactly
        // ceil(length/4) uint32 words (matching NumPy's stream consumption regardless of the final
        // slice) and stores only the first 'end' little-endian bytes, so a length just under the
        // byte[] ceiling never over-allocates the extra 0..3 padding bytes.
        internal static byte[] BytesCore<TState>(long length, Func<TState, uint> nextUInt32, TState state)
        {
            // NumPy: n_uint32 = ((length - 1) // 4 + 1) where '//' is C truncation-toward-zero
            //        because 'length' is typed npy_intp. C# integer division matches.
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

            if (end > Array.MaxLength)
                throw new OverflowException(
                    $"np.random.bytes({length}) would produce {end} bytes, exceeding the maximum .NET " +
                    $"byte[] length ({Array.MaxLength} ≈ 2 GiB). NumPy returns a 64-bit-length bytes object; " +
                    "for a larger array of random bytes use an NDArray path, e.g. " +
                    "default_rng().integers(0, 256, size, np.uint8).");

            var result = new byte[end];
            long fullWordsStored = end / 4;
            int tailBytes = (int)(end % 4);
            long pos = 0;
            for (long w = 0; w < nUint32; w++)
            {
                uint r = nextUInt32(state); // every word is drawn (state consumption), some discarded
                if (w < fullWordsStored)
                {
                    result[pos] = (byte)r;
                    result[pos + 1] = (byte)(r >> 8);
                    result[pos + 2] = (byte)(r >> 16);
                    result[pos + 3] = (byte)(r >> 24);
                    pos += 4;
                }
                else if (w == fullWordsStored && tailBytes > 0)
                {
                    for (int b = 0; b < tailBytes; b++)
                        result[pos++] = (byte)(r >> (8 * b));
                }
            }
            return result;
        }
    }
}
