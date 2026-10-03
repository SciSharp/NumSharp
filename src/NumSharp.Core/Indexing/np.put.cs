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
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is read-only (checked first, as NumPy does).</exception>
        /// <exception cref="ArgumentException"><paramref name="mode"/> is not a known mode, or
        ///     <paramref name="values"/> is empty while <paramref name="indices"/> is not.</exception>
        /// <exception cref="IndexOutOfRangeException"><paramref name="a"/> is empty while <paramref name="indices"/>
        ///     is not, or an index is out of bounds under <c>mode="raise"</c>.</exception>
        /// <exception cref="TypeError">The index dtype cannot be cast to int64 under the <c>safe</c> rule
        ///     (float, complex, uint64).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.put.html
        ///     <para>Scoped by <c>[NDScoped]</c>: the contiguous scratch of a non-contiguous target and
        ///     <see cref="PutImpl"/>'s int64 and dtype-converted copies are released on return, exceptions included.
        ///     The caller's arrays were built before the scope opened, so the scope never disposes them.</para>
        /// </remarks>
        [NDScoped]
        public static void put(NDArray a, NDArray indices, NDArray values, string mode = "raise")
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (indices is null) throw new ArgumentNullException(nameof(indices));
            if (values is null) throw new ArgumentNullException(nameof(values));

            // Writeability is checked FIRST — before the empty-indices no-op and the bounds/mode
            // checks (probed 2.4.2: put(ro, [], []) and put(ro, [oob], v) both report read-only).
            // The contiguous fast path (PutImpl) writes a.Address directly; the non-contiguous path
            // is already guarded (it routes through np.copyto).
            NumSharpException.ThrowIfNotWriteable(a.Shape, "put: output array");

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
                // The scratch is released by this method's scope (exceptions included).
                var scratch = np.ascontiguousarray(a);
                PutImpl(scratch, indices, values, modeInt);
                np.copyto(a, scratch, casting: "unsafe");
                return;
            }

            PutImpl(a, indices, values, modeInt);
        }

        /// <summary>
        ///     Scalar-index, scalar-value convenience overload.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="index">The flat index into the C-order flattening of <paramref name="a"/>.</param>
        /// <param name="value">The value to write, converted with <see cref="np.asanyarray(in object, DType, string)"/> and cast to
        ///     <paramref name="a"/>'s dtype.</param>
        /// <param name="mode">Boundary mode: <c>"raise"</c> (default), <c>"wrap"</c>, or <c>"clip"</c>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is read-only.</exception>
        /// <exception cref="ArgumentException"><paramref name="mode"/> is not a known mode.</exception>
        /// <exception cref="IndexOutOfRangeException"><paramref name="a"/> is empty, or <paramref name="index"/> is
        ///     out of bounds under <c>mode="raise"</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the 0-d index array and the conversion of <paramref name="value"/> are built
        ///     in THIS frame, before the array overload's scope opens, so this method needs a scope of its own to
        ///     release them (they were left to the finalizer before). An <see cref="NDArray"/> passed as
        ///     <paramref name="value"/> comes back from <see cref="np.asanyarray(in object, DType, string)"/> unchanged and is never disposed.
        /// </remarks>
        [NDScoped]
        public static void put(NDArray a, long index, object value, string mode = "raise")
            => put(a, NDArray.Scalar(index), np.asanyarray(value), mode);

        /// <summary>
        ///     The contiguous core of <see cref="put(NDArray, NDArray, NDArray, string)"/>: casts the indices to int64
        ///     (a C-contiguous int32 array is read in place) and the values to a C-contiguous copy of
        ///     <paramref name="a"/>'s dtype, then runs the IL scatter kernel straight into <paramref name="a"/>'s buffer.
        /// </summary>
        /// <param name="a">The C-contiguous target (the caller's array, or its contiguous scratch).</param>
        /// <param name="indices">The flat indices, any integer dtype castable to int64 under <c>safe</c>.</param>
        /// <param name="values">The values, cycled modulo their size; cast to <paramref name="a"/>'s dtype.</param>
        /// <param name="mode">The parsed boundary mode: 0 raise, 1 wrap, 2 clip.</param>
        /// <exception cref="TypeError">The index dtype cannot be cast to int64 under <c>safe</c>.</exception>
        /// <exception cref="NotSupportedException">No IL put kernel exists for the element width.</exception>
        /// <exception cref="IndexOutOfRangeException">An index is out of bounds under mode raise.</exception>
        /// <remarks>
        ///     Covered by <c>[NDScopedCovered]</c>: the only caller is the <c>[NDScoped]</c> array overload of
        ///     <c>put</c>, so the int64 and dtype-converted copies are released by that caller's scope, exceptions
        ///     included. The scope also keeps them reachable while the kernel reads them through raw pointers.
        /// </remarks>
        [NDScopedCovered]
        private static unsafe void PutImpl(NDArray a, NDArray indices, NDArray values, int mode)
        {
            // Cast indices to contig int64 (a contig int32 array is read in place); cast values to contig a.dtype.
            // A converted copy is built under the caller's scope and released with it; an operand used as-is is the
            // caller's and is never disposed.
            var idx64 = CastIndicesToInt64(indices, "safe", out _, out bool idx32);

            NDArray valsCast;
            if (values.GetTypeCode == a.GetTypeCode && values.Shape.IsContiguous)
            {
                valsCast = values;
            }
            else if (values.GetTypeCode == a.GetTypeCode)
            {
                valsCast = np.ascontiguousarray(values);
            }
            else
            {
                valsCast = values.astype(a.GetTypeCode);
            }

            int copyKind = DirectILKernelGenerator.CopyKindFor(a.dtypesize);

            byte* dstPtr = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            // idx32: the index buffer holds int32 values; the kernels' `long*` parameter is a reinterpret.
            long* idxPtr = idx32
                ? (long*)((int*)idx64.Storage.Address + idx64.Shape.offset)
                : (long*)idx64.Storage.Address + idx64.Shape.offset;
            byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

            long status;
            // The lean flat scatter (DirectILKernelGenerator.GatherFlat.cs) serves RAISE mode with a
            // compile-time element width; wrap/clip keep the general kernel.
            var flat = mode == 0 ? DirectILKernelGenerator.GetPutFlatKernel(copyKind, idx32) : null;
            if (flat != null)
            {
                status = flat(dstPtr, idxPtr, indices.size, valsPtr, valsCast.size, a.size);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetPutKernel(copyKind, idx32);
                if (kernel == null)
                    throw new NotSupportedException("np.put: IL kernel unavailable");
                status = kernel(dstPtr, idxPtr, indices.size,
                                valsPtr, valsCast.size,
                                a.size, a.dtypesize, mode);
            }
            if (status < indices.size)
            {
                long badI = status;
                long badVal = idx32 ? ((int*)idxPtr)[badI] : idxPtr[badI];
                throw new IndexOutOfRangeException(
                    $"index {badVal} is out of bounds for axis 0 with size {a.size}");
            }
        }
    }
}
