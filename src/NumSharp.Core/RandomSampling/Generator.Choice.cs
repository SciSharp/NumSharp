using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Modify an array in-place by shuffling its contents along the given axis.
        /// </summary>
        /// <param name="x">The writeable array to shuffle (at least 1-D).</param>
        /// <param name="axis">The axis whose slices are permuted (the other axes move with them).</param>
        /// <exception cref="TypeError"><paramref name="x"/> is null (<c>object of type 'NoneType' has no len()</c>) or 0-d
        ///     (<c>len() of unsized object</c>) — NumPy evaluates <c>len(x)</c> first.</exception>
        /// <exception cref="ValueError"><paramref name="x"/> is read-only.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-x.ndim, x.ndim)</c> (NumPy's <c>normalize_axis_index</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.shuffle.html
        ///     <br/>Fisher–Yates using <c>random_interval</c> (mask-rejection), byte-identical to NumPy. The draws
        ///     hold the bit generator's lock.
        /// </remarks>
        [NDScoped] // void boundary: the N-D path's index array + reordered gather are reclaimed after the copy-back
        public void shuffle(NDArray x, int axis = 0)
        {
            // NumPy evaluates `n = len(x)` before any other check, so None and a 0-d array raise TypeError here
            // rather than an axis error.
            if (x is null)
                throw new TypeError("object of type 'NoneType' has no len()");
            if (x.ndim == 0)
                throw new TypeError("len() of unsized object");

            if (!x.Shape.IsWriteable)
                throw new ValueError("array is read-only");

            int nd = x.ndim;
            if (axis < -nd || axis >= nd)
                throw new AxisError(axis, nd);
            int ax = axis < 0 ? axis + nd : axis;

            if (x.size == 0)
                return;

            if (nd == 1)
            {
                lock (_bitGenerator.@lock)
                    Shuffle1D(x);
                return;
            }

            long m = x.shape[ax];
            if (m <= 1)
                return;

            // NumPy's N-D path swaps whole sub-arrays with random_interval (skipping i==j but always
            // drawing). Running the same swap sequence over an index array yields the identical
            // permutation, which is then gathered with take and written back through the NDIter copy —
            // byte-exact and view-agnostic.
            long[] idx;
            lock (_bitGenerator.@lock)
                idx = BoundedIntegers.FisherYatesIndices(_bitGenerator, m);
            var reordered = np.take(x, np.array(idx), axis: ax);
            np.copyto(x, reordered);
        }

        /// <summary>
        ///     Randomly permute <c>arange(x)</c>.
        /// </summary>
        /// <param name="x">The length of the range.</param>
        /// <param name="axis">Accepted for NumPy's signature <c>permutation(x, axis=0)</c> and, as in NumPy, ignored for an
        ///     integer <paramref name="x"/> (its <c>isinstance(x, int)</c> branch shuffles <c>np.arange(x)</c> before the axis
        ///     is ever normalized), so any value is accepted.</param>
        /// <returns>A shuffled int64 range.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html
        /// </remarks>
        public NDArray permutation(long x, int axis = 0)
        {
            var arr = np.arange(x);
            lock (_bitGenerator.@lock)
                Shuffle1D(arr);
            return arr;
        }

        /// <summary>
        ///     Randomly permute a copy of <paramref name="x"/> along <paramref name="axis"/>.
        /// </summary>
        /// <param name="x">The array to permute (at least 1-D).</param>
        /// <param name="axis">The axis whose slices are permuted.</param>
        /// <returns>A permuted copy (C-contiguous); <paramref name="x"/> is not modified.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-x.ndim, x.ndim)</c> — which every axis is for a 0-d
        /// <paramref name="x"/>, and for a null one (NumPy's <c>np.asarray(None)</c> is 0-d): <c>axis 0 is out of bounds for
        /// array of dimension 0</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html
        ///     <br/>1-D input: a copy shuffled in place; N-D input: a shuffled index vector gathered with
        ///     <c>take</c> (NumPy's <c>arr[tuple(slices)]</c>), consuming the stream identically.
        /// </remarks>
        [NDScoped] // reclaims the N-D path's index-array temp; the gathered result is yielded
        public NDArray permutation(NDArray x, int axis = 0)
        {
            // NumPy: arr = np.asarray(x) — None is a 0-d object array, so it takes the 0-d axis error below.
            int nd = x is null ? 0 : x.ndim;
            if (axis < -nd || axis >= nd)
                throw new AxisError(axis, nd);
            int ax = axis < 0 ? axis + nd : axis;

            if (nd == 1)
            {
                var c = x.copy();
                lock (_bitGenerator.@lock)
                    Shuffle1D(c);
                return c;
            }

            long[] idx;
            lock (_bitGenerator.@lock)
                idx = BoundedIntegers.FisherYatesIndices(_bitGenerator, x.shape[ax]);
            return np.take(x, np.array(idx), axis: ax);
        }

        /// <summary>
        ///     Randomly permute <paramref name="x"/> along <paramref name="axis"/>. Unlike
        ///     <see cref="shuffle"/>, each slice along the axis is shuffled INDEPENDENTLY of the others.
        /// </summary>
        /// <param name="x">Array to shuffle (at least 1-D when an axis is given).</param>
        /// <param name="axis">Axis whose slices are each shuffled; <c>null</c> shuffles the flattened array.</param>
        /// <param name="out">Optional writeable destination of <paramref name="x"/>'s shape, filled with <c>casting='safe'</c>; returned when given.</param>
        /// <returns>The permuted array: <paramref name="out"/> when given, else a copy of <paramref name="x"/> with its memory layout kept (<c>order='K'</c>).</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only, or its shape differs from <paramref name="x"/>'s.</exception>
        /// <exception cref="TypeError"><paramref name="x"/>'s dtype cannot be cast to <paramref name="out"/>'s under <c>'safe'</c>; or <paramref name="axis"/> is null and <paramref name="x"/> is 0-d.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-x.ndim, x.ndim)</c>.</exception>
        /// <remarks>
        ///     A null <paramref name="x"/> is NumPy's <c>np.asarray(None)</c>: a 0-d OBJECT array, failing exactly where any
        ///     0-d input fails (the <c>len()</c> TypeError without an axis, the AxisError with one), and an <paramref name="out"/>
        ///     can only mismatch its shape or refuse the object-to-dtype cast.
        /// </remarks>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permuted.html
        ///     <br/>Byte-identical to NumPy: <c>axis=None</c> shuffles <c>out.ravel(order='A')</c> — the MEMORY
        ///     order of a C- or F-contiguous output (so an F-ordered input permutes a different sequence than its
        ///     C-order flattening) — or, for a non-contiguous output, a C-contiguous write-back copy in C order.
        ///     An explicit axis runs an independent <c>random_interval</c> Fisher–Yates over each 1-D slice,
        ///     iterating the remaining axes in C order (NumPy's <c>PyArray_IterAllButAxis</c>).
        /// </remarks>
        [NDScoped] // reclaims the write-back copy of a non-contiguous out; the target is yielded
        public NDArray permuted(NDArray x, int? axis = null, NDArray @out = null)
        {
            if (x is null)
            {
                // NumPy's x = np.asarray(None) is 0-d with dtype object; its checks run in NumPy's order.
                if (@out is not null)
                {
                    if (!@out.Shape.IsWriteable)
                        throw new ValueError("out is read-only");
                    if (@out.ndim != 0)
                        throw new ValueError("out must have the same shape as x");
                    throw new TypeError($"Cannot cast scalar from dtype('O') to {@out.dtype.ToString(true)} according to the rule 'safe'");
                }
                if (axis is null)
                    throw new TypeError("len() of unsized object");
                throw new AxisError(axis.Value, 0);
            }

            NDArray target;
            if (@out is null)
            {
                // NumPy: out = x.copy(order='K') — the copy keeps x's memory order, which decides the
                // axis=None walk below.
                target = x.copy('K');
            }
            else
            {
                if (!@out.Shape.IsWriteable)
                    throw new ValueError("out is read-only");
                if (!@out.Shape.Equals(x.Shape))
                    throw new ValueError("out must have the same shape as x");
                // NumPy: np.copyto(out, x, casting='safe') — reported with copyto's TypeError text, which calls a 0-d
                // source a "scalar".
                if (!np.can_cast(x.dtype, @out.dtype, "safe"))
                    throw new TypeError($"Cannot cast {(x.ndim == 0 ? "scalar" : "array data")} from {x.dtype.ToString(true)} to {@out.dtype.ToString(true)} according to the rule 'safe'");
                np.copyto(@out, x, "safe");
                target = @out;
            }

            if (axis is null)
            {
                if (target.ndim > 1)
                {
                    if (target.Shape.IsContiguous || target.Shape.IsFContiguous)
                    {
                        // out.ravel(order='A') is a view of the whole buffer in memory order for any C- or
                        // F-contiguous array, so NumPy's shuffle of it is a Fisher-Yates over the raw buffer.
                        lock (_bitGenerator.@lock)
                            ShuffleMemoryOrder(target);
                    }
                    else
                    {
                        // NumPy: a C-contiguous WRITEBACKIFCOPY copy is shuffled (C order) and written back.
                        var staging = target.copy('C');
                        lock (_bitGenerator.@lock)
                            ShuffleMemoryOrder(staging);
                        np.copyto(target, staging);
                    }
                }
                else
                {
                    // 1-D (any stride) shuffles in place; 0-d raises NumPy's len() TypeError.
                    shuffle(target);
                }
                return target;
            }

            int nd = target.ndim;
            if (axis.Value < -nd || axis.Value >= nd)
                throw new AxisError(axis.Value, nd);
            int ax = axis.Value < 0 ? axis.Value + nd : axis.Value;

            lock (_bitGenerator.@lock)
                PermutedAlongAxis(target, ax);
            return target;
        }

        /// <summary>
        ///     Independent Fisher–Yates (random_interval) over every 1-D slice along <paramref name="ax"/>,
        ///     visiting the remaining axes in C order — the byte-exact analog of NumPy's
        ///     <c>PyArray_IterAllButAxis</c> loop in <c>permuted</c>.
        /// </summary>
        /// <param name="target">The writeable array permuted in place (any layout; walked through its strides).</param>
        /// <param name="ax">The normalized axis.</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        private unsafe void PermutedAlongAxis(NDArray target, int ax)
        {
            long axlen = target.shape[ax];
            if (axlen <= 1 || target.size == 0)
                return;

            int itemsize = target.dtypesize;
            long axStrideBytes = target.Shape.strides[ax] * itemsize;
            byte* basePtr = target.Storage.Address + target.Shape.offset * itemsize;

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

                BoundedIntegers.ShuffleRaw(_bitGenerator, basePtr + baseOff, axlen, axStrideBytes, itemsize);

                // C-order odometer over the non-axis dimensions (last dim fastest).
                for (int d = nd - 2; d >= 0; d--) { if (++coord[d] < dims[d]) break; coord[d] = 0; }
            }
        }

        /// <summary>
        ///     In-place byte-level Fisher–Yates for a 1-D array (numpy <c>_shuffle_raw</c>), honouring the element
        ///     stride so strided 1-D views shuffle correctly.
        /// </summary>
        /// <param name="x">The 1-D writeable array.</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        private unsafe void Shuffle1D(NDArray x)
        {
            long n = x.shape[0];
            if (n <= 1)
                return;
            int itemsize = x.dtypesize;
            BoundedIntegers.ShuffleRaw(_bitGenerator, x.Storage.Address + x.Shape.offset * itemsize, n, x.Shape.strides[0] * itemsize, itemsize);
        }

        /// <summary>
        ///     Fisher–Yates over the whole buffer of a C- or F-contiguous array in ADDRESS order — NumPy's
        ///     <c>shuffle(out.ravel(order='A'))</c> for <c>permuted(axis=None)</c>.
        /// </summary>
        /// <param name="contiguous">A C- or F-contiguous writeable array (its elements fill <c>[base, base + size·itemsize)</c>).</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        private unsafe void ShuffleMemoryOrder(NDArray contiguous)
        {
            long n = contiguous.size;
            if (n <= 1)
                return;
            int itemsize = contiguous.dtypesize;
            BoundedIntegers.ShuffleRaw(_bitGenerator, contiguous.Storage.Address + contiguous.Shape.offset * itemsize, n, itemsize, itemsize);
        }
    }
}
