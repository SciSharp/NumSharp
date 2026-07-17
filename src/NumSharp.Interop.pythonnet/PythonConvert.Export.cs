using System;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop
{
    public static partial class PythonConvert
    {
        // ===========================  NumSharp  ->  Python (numpy)  ===========================

        /// <summary>
        ///     Wrap a NumSharp array as a numpy array that SHARES its unmanaged buffer (mutations visible
        ///     both ways). Requires numpy to be importable.
        ///
        ///     <para><b>Layout fidelity:</b> every NumSharp layout is exported zero-copy — C-contiguous,
        ///     sliced, transposed, Fortran-order and negative-stride views become numpy views with the
        ///     same strides; broadcast (stride-0) views become read-only numpy views, matching NumSharp's
        ///     own write protection. Scalars become 0-d arrays. <see cref="NPTypeCode.Char"/> is exported
        ///     as <c>uint16</c> (UTF-16 code units); <see cref="NPTypeCode.Decimal"/> has no numpy dtype
        ///     and throws — convert first.</para>
        ///
        ///     <para><b>Lifetime:</b> this export takes its own atomic reference on the NumSharp buffer, so
        ///     the memory stays valid even if the source <see cref="NDArray"/> — and the returned
        ///     <see cref="PyObject"/> wrapper — are disposed or garbage-collected. The reference is
        ///     released by a <c>weakref.finalize</c> on the exported array's base buffer object when the
        ///     LAST Python-side view over it (including derived views like <c>arr[1:]</c>, <c>arr.T</c>)
        ///     is collected, or at interpreter exit. While exported, <c>ndarray.resize(refcheck=True)</c>
        ///     on the source refuses to reallocate — the same protection NumPy applies to exported
        ///     buffers.</para>
        /// </summary>
        public static unsafe PyObject ToNumpy(NDArray source)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
            InteropRuntime.EnsureEngine();
            InteropRuntime.DrainPending();

            string dtypeStr = ToNumpyDtypeStr(source.typecode);   // throws for Decimal

            using (Py.GIL())
            {
                dynamic np = InteropRuntime.Numpy;

                var shape = source.Shape;
                if (shape.Size == 0)
                {
                    // Nothing to share — an empty array of the right dtype/shape IS the view.
                    using var emptyDims = MakeTuple(shape.Dimensions);
                    return (PyObject)np.empty(emptyDims, dtypeStr);
                }

                // Pin the buffer for the whole build; on success ownership of this reference
                // moves to the ExportKeeper (released by Python's weakref.finalize).
                IArraySlice slice = source.Storage.InternalArray;
                if (!slice.TryAddRef())
                    throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");

                try
                {
                    dynamic arr = BuildSharedView(np, source, slice, out PyObject baseBuffer);
                    using (baseBuffer)
                    {
                        var keeper = new ExportKeeper(source, slice);
                        RootOnPythonObject(baseBuffer, keeper);
                        InteropRuntime.TrackExport(keeper);
                    }

                    return (PyObject)arr;
                }
                catch
                {
                    slice.Release();
                    throw;
                }
            }
        }

        /// <summary>
        ///     <see cref="ToNumpy(NDArray)"/> when <paramref name="copy"/> is <c>false</c> (zero-copy shared
        ///     view), <see cref="ToNumpyCopy"/> when <c>true</c> (independent numpy array).
        /// </summary>
        public static PyObject ToNumpy(NDArray source, bool copy) => copy ? ToNumpyCopy(source) : ToNumpy(source);

        /// <summary>
        ///     Copy a NumSharp array into an independent, C-contiguous numpy array (no shared memory, no
        ///     lifetime coupling). Values follow the source's logical layout, so sliced / transposed /
        ///     broadcast views copy element-exact. Same dtype mapping as <see cref="ToNumpy(NDArray)"/>.
        /// </summary>
        public static unsafe PyObject ToNumpyCopy(NDArray source)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
            InteropRuntime.EnsureEngine();
            InteropRuntime.DrainPending();

            string dtypeStr = ToNumpyDtypeStr(source.typecode);   // throws for Decimal

            using (Py.GIL())
            {
                dynamic np = InteropRuntime.Numpy;

                var shape = source.Shape;
                if (shape.Size == 0)
                {
                    using var emptyDims = MakeTuple(shape.Dimensions);
                    return (PyObject)np.empty(emptyDims, dtypeStr);
                }

                // Guard the buffer against a concurrent Dispose for the duration of the copy only.
                IArraySlice slice = source.Storage.InternalArray;
                if (!slice.TryAddRef())
                    throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");

                try
                {
                    dynamic view = BuildSharedView(np, source, slice, out PyObject baseBuffer);
                    using (baseBuffer)
                    {
                        return (PyObject)np.array(view);   // np.array copies by default
                    }
                }
                finally
                {
                    slice.Release();
                    GC.KeepAlive(source);
                }
            }
        }

        /// <summary>
        ///     Expose a C-contiguous NumSharp array as a Python <c>memoryview</c> of raw bytes
        ///     (format 'B') — for non-numpy consumers of the buffer protocol (<c>PIL.Image.frombuffer</c>,
        ///     <c>struct.unpack_from</c>, sockets, ...). Zero-copy and writable: mutations through the
        ///     memoryview are visible in NumSharp. The NumSharp buffer is rooted exactly like
        ///     <see cref="ToNumpy(NDArray)"/> — it survives until the last Python-side reference dies.
        /// </summary>
        /// <exception cref="InvalidOperationException">The source is not C-contiguous — materialize first
        /// (<c>np.ascontiguousarray(nd)</c> or <c>nd.copy()</c>).</exception>
        public static unsafe PyObject ToMemoryView(NDArray source)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
            InteropRuntime.EnsureEngine();
            InteropRuntime.DrainPending();

            _ = ToBufferFormat(source.typecode);   // dtype gate (Decimal throws)

            var shape = source.Shape;
            if (!shape.IsContiguous || shape.Offset != 0)
                throw new InvalidOperationException(
                    "the array is not C-contiguous; a flat memoryview would misrepresent it. Materialize first: np.ascontiguousarray(nd) or nd.copy().");

            using (Py.GIL())
            {
                IArraySlice slice = source.Storage.InternalArray;
                if (!slice.TryAddRef())
                    throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");

                try
                {
                    dynamic ct = InteropRuntime.Ctypes;
                    dynamic builtins = InteropRuntime.Builtins;

                    long nbytes = shape.Size * source.dtypesize;
                    dynamic ctBuf = (ct.c_char * nbytes).from_address((long)slice.Address);
                    dynamic mv = builtins.memoryview(ctBuf).cast("B");

                    using (PyObject baseBuffer = (PyObject)ctBuf)
                    {
                        var keeper = new ExportKeeper(source, slice);
                        RootOnPythonObject(baseBuffer, keeper);
                        InteropRuntime.TrackExport(keeper);
                    }

                    return (PyObject)mv;
                }
                catch
                {
                    slice.Release();
                    throw;
                }
            }
        }

        // ---- shared construction ----------------------------------------------------------------

        /// <summary>
        ///     Build the numpy view over <paramref name="slice"/>'s memory WITHOUT lifetime rooting
        ///     (the caller holds an ARC reference for the duration). <paramref name="baseBuffer"/> is the
        ///     deepest Python base object (the ctypes buffer) every derived numpy view chains to —
        ///     the correct attachment point for a keep-alive.
        /// </summary>
        private static unsafe dynamic BuildSharedView(dynamic np, NDArray source, IArraySlice slice, out PyObject baseBuffer)
        {
            dynamic ct = InteropRuntime.Ctypes;

            var shape = source.Shape;
            int itemsize = source.dtypesize;
            long[] dims = shape.Dimensions;
            long[] elemStrides = shape.Strides;
            long offset = shape.Offset;

            // The buffer window starts at the view's element-0 pointer. Negative strides address
            // memory BELOW that pointer — still inside the real allocation (NumSharp's own offset
            // places element 0 so every coordinate lands within the block); numpy's as_strided does
            // no bounds checking, so the window only needs to be a valid region, which it is.
            long dataPtr = (long)slice.Address + offset * itemsize;
            long tailBytes = (slice.Count - offset) * itemsize;

            dynamic ctBuf = (ct.c_char * tailBytes).from_address(dataPtr);
            string dtypeStr = ToNumpyDtypeStr(source.typecode);
            dynamic flat = np.frombuffer(ctBuf, dtypeStr);

            dynamic arr;
            if (shape.IsContiguous && offset == 0 && dims.Length == 1)
            {
                arr = flat;
            }
            else if (shape.IsContiguous && offset == 0 && dims.Length > 1)
            {
                using var dimsTuple = MakeTuple(dims);
                arr = flat.reshape(dimsTuple);
            }
            else
            {
                // Strided / offset / broadcast / scalar: express the exact NumSharp layout.
                var byteStrides = new long[elemStrides.Length];
                for (int i = 0; i < elemStrides.Length; i++)
                    byteStrides[i] = elemStrides[i] * itemsize;
                using var dimsTuple = MakeTuple(dims);
                using var stridesTuple = MakeTuple(byteStrides);
                arr = np.lib.stride_tricks.as_strided(flat, dimsTuple, stridesTuple);
            }

            if (!shape.IsWriteable)
                arr.setflags(write: false);   // broadcast views are read-only, as in NumSharp/NumPy

            baseBuffer = (PyObject)ctBuf;
            return arr;
        }

        /// <summary>
        ///     Register <c>weakref.finalize(pythonObject, keeper.Release)</c>: CPython keeps the finalize
        ///     object alive in a global registry until either the target is collected or the interpreter
        ///     exits (an atexit pass is guaranteed) — in both cases the marshaled delegate runs under the
        ///     GIL and releases the NumSharp buffer reference. CLR-only work, safe at any engine phase.
        /// </summary>
        private static void RootOnPythonObject(PyObject target, ExportKeeper keeper)
        {
            dynamic weakref = InteropRuntime.Weakref;
            using PyObject callback = ((Action)keeper.Release).ToPython();
            using PyObject finalizer = (PyObject)weakref.finalize(target, callback);
        }

        private static PyTuple MakeTuple(long[] values)
        {
            var items = new PyObject[values.Length];
            try
            {
                for (int i = 0; i < values.Length; i++)
                    items[i] = new PyInt(values[i]);
                return new PyTuple(items);   // increfs the items
            }
            finally
            {
                for (int i = 0; i < items.Length; i++)
                    items[i]?.Dispose();
            }
        }
    }
}
