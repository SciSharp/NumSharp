using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // numpy random_interval : uniform integer in [0, max] via mask-rejection (NOT Lemire).
        // This is the sampler shuffle/permutation use (choice uses Lemire via _shuffle_int).
        private ulong RandomInterval(ulong max)
        {
            if (max == 0)
                return 0;
            ulong mask = max;
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;
            ulong value;
            if (max <= 0xffffffffUL)
                while ((value = _bitGenerator.NextUInt32() & mask) > max) { }
            else
                while ((value = _bitGenerator.NextUInt64() & mask) > max) { }
            return value;
        }

        /// <summary>
        ///     Modify an array in-place by shuffling its contents along the given axis.
        /// </summary>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.shuffle.html
        ///     <br/>Fisher–Yates using <c>random_interval</c> (mask-rejection), byte-identical to NumPy.
        /// </remarks>
        public void shuffle(NDArray x, int axis = 0)
        {
            if (!x.Shape.IsWriteable)
                throw new ValueError("array is read-only");

            int nd = x.ndim;
            int ax = axis < 0 ? axis + nd : axis;
            if (nd == 0 || ax < 0 || ax >= nd)
                throw new ArgumentException($"axis {axis} is out of bounds for array of dimension {nd}");

            if (x.size == 0)
                return;

            if (nd == 1)
            {
                Shuffle1D(x);
                return;
            }

            long m = x.shape[ax];
            if (m <= 1)
                return;

            // NumPy's N-D path swaps whole sub-arrays with random_interval (skipping i==j but always
            // drawing). Running the same swap sequence over an index array yields the identical
            // permutation, which we then apply via take — byte-exact and view-agnostic.
            long[] idx = FisherYatesIndices(m);
            var reordered = np.take(x, np.array(idx), axis: ax);
            CopyInPlace(x, reordered);
        }

        /// <summary>
        ///     Randomly permute a sequence, or return a permuted range.
        /// </summary>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html
        /// </remarks>
        public NDArray permutation(long x)
        {
            var arr = np.arange(x);
            Shuffle1D(arr);
            return arr;
        }

        /// <inheritdoc cref="permutation(long)"/>
        public NDArray permutation(NDArray x, int axis = 0)
        {
            int nd = x.ndim;
            int ax = axis < 0 ? axis + nd : axis;
            if (nd == 0 || ax < 0 || ax >= nd)
                throw new ArgumentException($"axis {axis} is out of bounds for array of dimension {nd}");

            if (nd == 1)
            {
                var c = x.copy();
                Shuffle1D(c);
                return c;
            }

            long[] idx = FisherYatesIndices(x.shape[ax]);
            return np.take(x, np.array(idx), axis: ax);
        }

        // Build the permutation produced by in-place Fisher–Yates over [0, m).
        private long[] FisherYatesIndices(long m)
        {
            var idx = new long[m];
            for (long k = 0; k < m; k++)
                idx[k] = k;
            for (long i = m - 1; i >= 1; i--)
            {
                ulong j = RandomInterval((ulong)i);
                if ((long)j != i)
                    (idx[i], idx[j]) = (idx[j], idx[i]);
            }
            return idx;
        }

        // In-place byte-level Fisher–Yates for a 1-D array (numpy _shuffle_raw), honouring the
        // element stride so strided 1-D views shuffle correctly.
        private unsafe void Shuffle1D(NDArray x)
        {
            long n = x.shape[0];
            if (n <= 1)
                return;
            int itemsize = x.dtypesize;
            long strideBytes = x.Shape.strides[0] * itemsize;
            byte* basePtr = x.Storage.Address + x.Shape.offset * itemsize;
            byte* buf = stackalloc byte[16]; // widest dtype (Complex/Decimal) = 16 bytes

            for (long i = n - 1; i >= 1; i--)
            {
                ulong j = RandomInterval((ulong)i);
                if ((long)j == i)
                    continue;
                byte* pi = basePtr + i * strideBytes;
                byte* pj = basePtr + (long)j * strideBytes;
                Buffer.MemoryCopy(pj, buf, 16, itemsize);
                Buffer.MemoryCopy(pi, pj, itemsize, itemsize);
                Buffer.MemoryCopy(buf, pi, itemsize, itemsize);
            }
        }

        // Overwrite dst's contents (in logical C order) from a freshly-materialised source.
        private void CopyInPlace(NDArray dst, NDArray src)
        {
            long n = dst.size;
            for (long i = 0; i < n; i++)
                dst.SetAtIndex(src.GetAtIndex(i), i);
        }
    }
}
