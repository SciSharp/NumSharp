using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Replace elements of <paramref name="a"/> with given values, at the
        ///     specified flat indices. In-place — modifies <paramref name="a"/>.
        ///     Equivalent to <c>a.flat[indices] = values</c> with cyclic
        ///     broadcasting of <paramref name="values"/>.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="indices">
        ///     Integer array of flat indices (cast to int64 internally). Indexing
        ///     is into the C-order flattening of <paramref name="a"/>.
        /// </param>
        /// <param name="values">
        ///     Values to write. Cast to <paramref name="a"/>'s dtype. Cycles
        ///     modulo its size — shorter than <paramref name="indices"/> is fine.
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default), <c>"wrap"</c>, or <c>"clip"</c>.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.put.html</remarks>
        public static void put(NDArray a, NDArray indices, NDArray values, string mode = "raise")
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (indices is null) throw new ArgumentNullException(nameof(indices));
            if (values is null) throw new ArgumentNullException(nameof(values));

            int modeInt = ParseMode(mode, nameof(mode));

            long indicesCount = indices.size;
            if (indicesCount == 0)
                return;   // nothing to do

            if (a.size == 0)
                throw new IndexOutOfRangeException("cannot replace elements of an empty array");

            if (values.size == 0)
                throw new ArgumentException("put: values cannot be empty when indices is non-empty.", nameof(values));

            // NumPy WRITEBACKIFCOPY semantics: when the target is non-contig (e.g. a
            // strided view), allocate a contig scratch with the view's content, run
            // the kernel into the scratch, then np.copyto back to the view which
            // propagates the writes through the original strides to the underlying
            // storage. Slower than the contig fast path (extra O(view.size) traffic)
            // but matches NumPy's "writes flow through to parent" semantics.
            if (!a.Shape.IsContiguous)
            {
                var scratch = np.ascontiguousarray(a);
                try
                {
                    PutImpl(scratch, indices, values, modeInt);
                    np.copyto(a, scratch, casting: "unsafe");
                }
                finally { scratch.Dispose(); }
                return;
            }

            PutImpl(a, indices, values, modeInt);
        }

        /// <summary>
        ///     Scalar-index, scalar-value convenience overload.
        /// </summary>
        public static void put(NDArray a, long index, object value, string mode = "raise")
            => put(a, NDArray.Scalar(index), np.asanyarray(value), mode);

        private static unsafe void PutImpl(NDArray a, NDArray indices, NDArray values, int mode)
        {
            // Cast indices to contig int64; cast values to contig a.dtype.
            var idx64 = CastIndicesToInt64(indices, out bool ownIdx);

            NDArray valsCast;
            bool ownVals;
            if (values.GetTypeCode == a.GetTypeCode && values.Shape.IsContiguous)
            {
                valsCast = values;
                ownVals = false;
            }
            else if (values.GetTypeCode == a.GetTypeCode)
            {
                var c = np.ascontiguousarray(values);
                valsCast = c;
                ownVals = !ReferenceEquals(c, values);
            }
            else
            {
                valsCast = values.astype(a.GetTypeCode);
                ownVals = true;
            }

            try
            {
                var kernel = ILKernelGenerator.GetPutKernel();
                if (kernel == null)
                    throw new NotSupportedException("np.put: IL kernel unavailable");

                byte* dstPtr = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
                long* idxPtr = (long*)idx64.Storage.Address + idx64.Shape.offset;
                byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

                long status = kernel(dstPtr, idxPtr, indices.size,
                                      valsPtr, valsCast.size,
                                      a.size, a.dtypesize, mode);
                if (status < indices.size)
                {
                    long badI = status;
                    long badVal = idxPtr[badI];
                    throw new IndexOutOfRangeException(
                        $"index {badVal} is out of bounds for axis 0 with size {a.size}");
                }
            }
            finally
            {
                if (ownIdx) idx64.Dispose();
                if (ownVals) valsCast.Dispose();
            }
        }
    }
}
