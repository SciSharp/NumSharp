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
        [NDScoped] // void boundary: the N-D path's index array + reordered gather are reclaimed after CopyInPlace
        public void shuffle(NDArray x, int axis = 0)
        {
            // NumPy evaluates `n = len(x)` before any other check, so a 0-d array raises TypeError here
            // ("len() of unsized object") rather than an axis error.
            if (x.ndim == 0)
                throw new TypeError("len() of unsized object");

            if (!x.Shape.IsWriteable)
                throw new ValueError("array is read-only");

            int nd = x.ndim;
            int ax = axis < 0 ? axis + nd : axis;
            if (ax < 0 || ax >= nd)
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
        [NDScoped] // reclaims the N-D path's index-array temp; the gathered result is yielded
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

        /// <summary>
        ///     Randomly permute <paramref name="x"/> along <paramref name="axis"/>. Unlike
        ///     <see cref="shuffle"/>, each slice along the axis is shuffled INDEPENDENTLY of the others.
        /// </summary>
        /// <param name="x">Array to shuffle (at least 1-D when an axis is given).</param>
        /// <param name="axis">Axis whose slices are each shuffled; <c>null</c> shuffles the flattened array.</param>
        /// <param name="out">Optional destination (must match <paramref name="x"/>'s shape); returned when given.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permuted.html
        ///     <br/>Byte-identical to NumPy: <c>axis=None</c> shuffles the C-order flattened copy;
        ///     an explicit axis runs an independent <c>random_interval</c> Fisher–Yates over each 1-D
        ///     slice, iterating the remaining axes in C-order (NumPy's <c>PyArray_IterAllButAxis</c>).
        /// </remarks>
        public NDArray permuted(NDArray x, int? axis = null, NDArray @out = null)
        {
            NDArray target;
            if (@out is null)
            {
                target = x.copy();
            }
            else
            {
                if (!@out.Shape.IsWriteable)
                    throw new ValueError("out is read-only");
                if (!@out.Shape.Equals(x.Shape))
                    throw new ValueError("out must have the same shape as x");
                np.copyto(@out, x);
                target = @out;
            }

            if (axis is null)
            {
                if (target.ndim == 0)
                    shuffle(target); // NumPy shuffles `out` here → 0-d raises TypeError("len() of unsized object")
                else
                {
                    // Shuffle the flattened array (C-order for a contiguous target; the common path).
                    NDArray flat = target.Shape.IsContiguous ? target.reshape(target.size) : target.copy().reshape(target.size);
                    Shuffle1D(flat);
                    if (!target.Shape.IsContiguous)
                        CopyInPlace(target, flat);
                }
                return target;
            }

            int nd = target.ndim;
            int ax = axis.Value < 0 ? axis.Value + nd : axis.Value;
            if (nd == 0 || ax < 0 || ax >= nd)
                throw new ArgumentException($"axis {axis} is out of bounds for array of dimension {nd}");

            PermutedAlongAxis(target, ax);
            return target;
        }

        // Independent Fisher–Yates (random_interval) over every 1-D slice along `ax`, visiting the
        // remaining axes in C-order — the byte-exact analog of NumPy's IterAllButAxis loop in permuted.
        private unsafe void PermutedAlongAxis(NDArray target, int ax)
        {
            long axlen = target.shape[ax];
            if (axlen <= 1 || target.size == 0)
                return;

            int itemsize = target.dtypesize;
            long axStrideBytes = target.Shape.strides[ax] * itemsize;
            byte* basePtr = target.Storage.Address + target.Shape.offset * itemsize;
            byte* buf = stackalloc byte[16]; // widest dtype (Complex/Decimal)

            int nd = target.ndim;
            // The non-axis dimensions and their byte strides, in original (C) order.
            var dims = new long[nd - 1];
            var strides = new long[nd - 1];
            int k = 0;
            for (int d = 0; d < nd; d++)
                if (d != ax) { dims[k] = target.shape[d]; strides[k] = target.Shape.strides[d] * itemsize; k++; }

            long outerCount = target.size / axlen;
            var coord = new long[nd - 1];
            for (long o = 0; o < outerCount; o++)
            {
                long baseOff = 0;
                for (int d = 0; d < nd - 1; d++) baseOff += coord[d] * strides[d];
                byte* slice = basePtr + baseOff;

                for (long i = axlen - 1; i >= 1; i--)
                {
                    ulong j = RandomInterval((ulong)i);
                    if ((long)j == i) continue;
                    byte* pi = slice + i * axStrideBytes;
                    byte* pj = slice + (long)j * axStrideBytes;
                    Buffer.MemoryCopy(pj, buf, 16, itemsize);
                    Buffer.MemoryCopy(pi, pj, itemsize, itemsize);
                    Buffer.MemoryCopy(buf, pi, itemsize, itemsize);
                }

                // C-order odometer over the non-axis dimensions (last dim fastest).
                for (int d = nd - 2; d >= 0; d--) { if (++coord[d] < dims[d]) break; coord[d] = 0; }
            }
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
