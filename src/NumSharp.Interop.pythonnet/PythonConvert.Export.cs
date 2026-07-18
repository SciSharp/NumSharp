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

            _ = ToNumpyDtypeStr(source.typecode);   // dtype gate — throws for Decimal before any Python work

            using (Py.GIL())
            {
                var shape = source.Shape;
                if (shape.Size == 0)
                {
                    // Nothing to share — an empty array of the right dtype/shape IS the view.
                    using var emptyDims = MakeTuple(shape.Dimensions);
                    return InteropRuntime.NpEmpty.Invoke(emptyDims, InteropRuntime.DtypeString(source.typecode));
                }

                // Pin the buffer for the whole build; on success ownership of this reference
                // moves to the ExportKeeper (released by Python's weakref.finalize).
                IArraySlice slice = source.Storage.InternalArray;
                if (!slice.TryAddRef())
                    throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");

                PyObject arr = null;
                ExportKeeper keeper = null;
                try
                {
                    arr = BuildSharedView(source, slice, out PyObject baseBuffer);
                    using (baseBuffer)
                    {
                        keeper = new ExportKeeper(source, slice);
                        RootOnPythonObject(baseBuffer, keeper);
                        InteropRuntime.TrackExport(keeper);
                    }

                    return arr;
                }
                catch
                {
                    arr?.Dispose();
                    // Once the weakref.finalize is registered the keeper owns the reference and
                    // WILL be invoked when the base buffer dies — release through the keeper
                    // (idempotent) so a late finalize callback cannot double-release.
                    if (keeper is null)
                        slice.Release();
                    else
                        keeper.Release();
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

            _ = ToNumpyDtypeStr(source.typecode);   // dtype gate — throws for Decimal before any Python work

            using (Py.GIL())
            {
                var shape = source.Shape;
                if (shape.Size == 0)
                {
                    using var emptyDims = MakeTuple(shape.Dimensions);
                    return InteropRuntime.NpEmpty.Invoke(emptyDims, InteropRuntime.DtypeString(source.typecode));
                }

                // Guard the buffer against a concurrent Dispose for the duration of the copy only.
                IArraySlice slice = source.Storage.InternalArray;
                if (!slice.TryAddRef())
                    throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");

                try
                {
                    using PyObject view = BuildSharedView(source, slice, out PyObject baseBuffer);
                    using (baseBuffer)
                    {
                        return InteropRuntime.NpArray.Invoke(view);   // np.array copies by default
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

                PyObject mv = null;
                ExportKeeper keeper = null;
                try
                {
                    long nbytes = shape.Size * source.dtypesize;
                    using PyObject ctBuf = MakeCtypesWindow((long)slice.Address, nbytes);

                    using (PyObject raw = InteropRuntime.BuiltinsMemoryview.Invoke(ctBuf))
                    using (PyObject cast = raw.GetAttr(InteropRuntime.NameCast))
                        mv = cast.Invoke(InteropRuntime.StrB);   // the memoryview holds its own exporter ref on ctBuf

                    keeper = new ExportKeeper(source, slice);
                    RootOnPythonObject(ctBuf, keeper);
                    InteropRuntime.TrackExport(keeper);

                    return mv;
                }
                catch
                {
                    mv?.Dispose();
                    // Same keeper-owned release rule as ToNumpy: after weakref.finalize registration
                    // the late callback and this path must share one idempotent Release.
                    if (keeper is null)
                        slice.Release();
                    else
                        keeper.Release();
                    throw;
                }
            }
        }

        // ---- shared construction ----------------------------------------------------------------

        /// <summary>
        ///     Build the numpy view over <paramref name="slice"/>'s memory WITHOUT lifetime rooting
        ///     (the caller holds an ARC reference for the duration). <paramref name="baseBuffer"/> is the
        ///     deepest Python base object (the ctypes buffer) every derived numpy view chains to —
        ///     the correct attachment point for a keep-alive. All Python work runs through the
        ///     session-cached callables on <see cref="InteropRuntime"/> (no dynamic dispatch), and every
        ///     intermediate wrapper is disposed deterministically instead of drifting to pythonnet's
        ///     finalizer queue — numpy's own reference chain (arr → flat → ctBuf) carries the lifetime.
        /// </summary>
        private static unsafe PyObject BuildSharedView(NDArray source, IArraySlice slice, out PyObject baseBuffer)
        {
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

            PyObject ctBuf = null, flat = null, arr = null;
            try
            {
                ctBuf = MakeCtypesWindow(dataPtr, tailBytes);
                flat = InteropRuntime.NpFrombuffer.Invoke(ctBuf, InteropRuntime.DtypeString(source.typecode));

                // The trivial paths require the view to cover its backing window EXACTLY: a contiguous
                // offset-0 slice like ring["0:16"] keeps the full base block as its InternalArray
                // (NumSharp only re-slices the array when offset > 0), so flat/reshape would leak the
                // whole buffer into the export — the strided path below trims to the true extent.
                bool wholeWindow = shape.IsContiguous && offset == 0 && shape.Size == slice.Count;
                if (wholeWindow && dims.Length == 1)
                {
                    arr = flat;
                    flat = null;   // the flat view IS the result — ownership moves to arr
                }
                else if (wholeWindow && dims.Length > 1)
                {
                    using var dimsTuple = MakeTuple(dims);
                    using PyObject reshape = flat.GetAttr(InteropRuntime.NameReshape);
                    arr = reshape.Invoke(dimsTuple);
                }
                else
                {
                    // Strided / offset / prefix-window / broadcast / scalar: express the exact layout.
                    var byteStrides = new long[elemStrides.Length];
                    for (int i = 0; i < elemStrides.Length; i++)
                        byteStrides[i] = elemStrides[i] * itemsize;
                    using var dimsTuple = MakeTuple(dims);
                    using var stridesTuple = MakeTuple(byteStrides);
                    arr = InteropRuntime.NpAsStrided.Invoke(flat, dimsTuple, stridesTuple);
                }

                if (!shape.IsWriteable)
                {
                    // broadcast views are read-only, as in NumSharp/NumPy.
                    // setflags(write=None, align=None, uic=None) — the first positional IS write.
                    using PyObject setflags = arr.GetAttr(InteropRuntime.NameSetflags);
                    using PyObject none = setflags.Invoke(InteropRuntime.FalseLiteral);
                }

                baseBuffer = ctBuf;
                PyObject result = arr;
                ctBuf = null;   // ownership of both transfers to the caller
                arr = null;
                return result;
            }
            finally
            {
                flat?.Dispose();    // intermediate on every path that did not return it
                arr?.Dispose();     // non-null here only when a later step threw
                ctBuf?.Dispose();   // non-null here only when a later step threw
            }
        }

        /// <summary>
        ///     <c>(ctypes.c_char * nbytes).from_address(address)</c> through the cached bound
        ///     <c>c_char.__mul__</c> — the raw-pointer window every zero-copy export is built over.
        /// </summary>
        private static PyObject MakeCtypesWindow(long address, long nbytes)
        {
            using var size = new PyInt(nbytes);
            using PyObject arrayType = InteropRuntime.CCharMul.Invoke(size);
            using PyObject fromAddress = arrayType.GetAttr(InteropRuntime.NameFromAddress);
            using var addr = new PyInt(address);
            return fromAddress.Invoke(addr);
        }

        /// <summary>
        ///     Register <c>weakref.finalize(pythonObject, keeper.Release)</c>: CPython keeps the finalize
        ///     object alive in a global registry until either the target is collected or the interpreter
        ///     exits (an atexit pass is guaranteed) — in both cases the marshaled delegate runs under the
        ///     GIL and releases the NumSharp buffer reference. CLR-only work, safe at any engine phase.
        /// </summary>
        private static void RootOnPythonObject(PyObject target, ExportKeeper keeper)
        {
            using PyObject callback = ((Action)keeper.Release).ToPython();
            using PyObject finalizer = InteropRuntime.WeakrefFinalize.Invoke(target, callback);
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
