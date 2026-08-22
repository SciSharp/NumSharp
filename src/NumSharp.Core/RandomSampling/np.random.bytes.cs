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
        /// </remarks>
        public byte[] bytes(long length)
        {
            // NumPy: n_uint32 = ((length - 1) // 4 + 1) where '//' is C truncation-toward-zero
            //        because 'length' is typed npy_intp. C# integer division matches.
            long nUint32 = (length - 1) / 4 + 1;

            if (nUint32 < 0)
                // Reaches np.random.randint(size=n_uint32) with a negative size in NumPy.
                throw new ValueError("negative dimensions are not allowed");

            long totalBytes = nUint32 * 4;
            var full = new byte[totalBytes];
            long pos = 0;
            for (long w = 0; w < nUint32; w++)
            {
                uint r = randomizer.NextUInt32();
                full[pos++] = (byte)r;
                full[pos++] = (byte)(r >> 8);
                full[pos++] = (byte)(r >> 16);
                full[pos++] = (byte)(r >> 24);
            }

            // Python slice tobytes()[:length]: for length >= 0 keep the first 'length'
            // bytes; for length < 0 drop |length| bytes from the end (clamped at 0).
            long end = length >= 0
                ? Math.Min(length, totalBytes)
                : Math.Max(0, totalBytes + length);

            if (end == totalBytes)
                return full;

            var result = new byte[end];
            Array.Copy(full, result, end);
            return result;
        }
    }
}
