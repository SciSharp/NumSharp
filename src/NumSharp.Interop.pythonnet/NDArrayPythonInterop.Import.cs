using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    public static partial class NDArrayPythonInterop
    {
        // ===========================  Python  ->  NumSharp  ==================================

        /// <summary>
        ///     Copy any PEP 3118 buffer object (numpy array, memoryview, bytes, bytearray, array.array,
        ///     PIL image, torch tensor, ...) into a fresh C-contiguous NumSharp array. Honors strides /
        ///     Fortran order (non-contiguous sources are linearized by CPython's
        ///     <c>memoryview.tobytes('C')</c>). numpy-agnostic; the result owns its memory — no lifetime
        ///     coupling to the source.
        ///
        ///     <para>0-d exporters produce scalar NDArrays. complex64 buffers (format 'Zf') are widened
        ///     to <see cref="NPTypeCode.Complex"/> (complex128) during the copy.</para>
        /// </summary>
        /// <param name="obj">The buffer-protocol exporter to copy.</param>
        /// <param name="requireGIL">
        ///     <c>true</c>: acquire the GIL for this call (re-entrant under an outer <see cref="Py.GIL"/>);
        ///     <c>false</c>: no GIL management — the calling thread must ALREADY hold the GIL;
        ///     <c>null</c> (default): follow <see cref="RequireGIL"/>.
        /// </param>
        public static unsafe NDArray ToNDArray(this PyObject obj, bool? requireGIL = null)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();

            using (AcquireGil(requireGIL))
            {
                using PyObject mv = OpenMemoryView(obj);

                string format = mv.format;
                long itemsize = mv.itemsize;
                bool widenComplex64 = IsComplex64Format(format);
                NPTypeCode tc = widenComplex64 ? NPTypeCode.Complex : FromBufferFormat(format, itemsize);

                long[] dims = mv.shape;
                Shape shape = dims.Length == 0 ? new Shape() : new Shape(dims);
                if (shape.Size == 0)
                    return new NDArray(tc, shape, fillZeros: false);

                var dest = new NDArray(tc, shape, fillZeros: false);
                long expectedSourceBytes = shape.Size * itemsize;

                if (mv.c_contiguous)
                {
                    // Read through the MEMORYVIEW, not the raw object — same reason as the view path:
                    // pythonnet 3.0.x's obj.GetBuffer is per-exporter buggy (a raw ctypes array
                    // hard-crashes it on every flag), while the memoryview over the same memory is
                    // uniformly safe. Only a read-only SIMPLE lock is needed here; it is released as
                    // soon as the bytes are blitted.
                    using PyBuffer buf = mv.GetBuffer(PyBUF.SIMPLE);
                    CopyBuffer((void*)buf.Buffer, buf.Length, dest, expectedSourceBytes, widenComplex64);
                }
                else
                {
                    // Linearize through CPython (correct for every stride pattern incl. suboffsets),
                    // then blit the C-ordered bytes. The bytes object is a plain contiguous exporter.
                    using PyObject bytesObj = mv.tobytes("C");
                    using PyBuffer buf = bytesObj.GetBuffer(PyBUF.SIMPLE);
                    CopyBuffer((void*)buf.Buffer, buf.Length, dest, expectedSourceBytes, widenComplex64);
                }

                return dest;
            }
        }

        /// <summary>
        ///     Zero-copy NumSharp view over Python memory: NumSharp SHARES the exporter's buffer
        ///     (mutations visible both ways).
        ///
        ///     <para><b>Three zero-copy routes:</b></para>
        ///     <list type="bullet">
        ///       <item><b>C-contiguous PEP 3118 exporters</b> (any object): the buffer is acquired with
        ///         <c>PyBUF.WRITABLE</c>, which pins the exporter and — for resizable objects like
        ///         <c>bytearray</c> — blocks reallocation for the lease's lifetime.</item>
        ///       <item><b>Non-contiguous numpy arrays</b> (slices, transposes, Fortran order, broadcasts):
        ///         imported through <c>__array_interface__</c> as a strided NumSharp view with identical
        ///         layout; broadcast (stride-0) sources become read-only NumSharp views. The numpy array
        ///         is kept alive by a strong reference (numpy's <c>resize(refcheck=True)</c> refuses to
        ///         reallocate while it exists).</item>
        ///       <item><b>Non-contiguous NON-numpy exporters</b> (a sliced / offset / reversed
        ///         <c>memoryview</c>, a strided <c>memoryview</c> of an <c>array.array</c>, ...): the base
        ///         pointer comes from a <c>PyBUF.STRIDED</c> buffer and the exact shape/strides from the
        ///         <c>memoryview</c> itself, reconstructing the strided view (incl. negative strides) —
        ///         so a view is produced whenever the layout is representable, not only for numpy. Only
        ///         genuinely irreducible layouts (complex64, big-endian, non-element strides) decline.</item>
        ///     </list>
        ///
        ///     <para><b>Lifetime:</b> the lease is released when the LAST NumSharp view over the memory —
        ///     including derived views like <c>nd["1:"]</c> — is disposed or garbage-collected (NumSharp's
        ///     memory-block reference counting drives it; the Python-side release is marshaled to the GIL
        ///     safely, never on a raw finalizer thread). Import views are tied to the interpreter: after
        ///     <see cref="PythonEngine.Shutdown"/> their memory is gone and they must not be touched (the
        ///     shutdown handler releases all outstanding leases crash-free).</para>
        /// </summary>
        /// <param name="obj">The exporter to view.</param>
        /// <param name="allowReadonly">
        ///     Accept read-only sources (<c>bytes</c>, read-only numpy arrays, ...) and return a
        ///     NON-WRITEABLE view (<see cref="Shape.IsWriteable"/> is <c>false</c>; guarded write paths
        ///     raise NumPy's "assignment destination is read-only") — exactly how numpy marks arrays
        ///     over read-only buffers <c>writeable=False</c>. Default <c>false</c>: read-only sources
        ///     throw with guidance instead.
        /// </param>
        /// <param name="requireGIL">
        ///     <c>true</c>: acquire the GIL for this call (re-entrant under an outer <see cref="Py.GIL"/>);
        ///     <c>false</c>: no GIL management — the calling thread must ALREADY hold the GIL;
        ///     <c>null</c> (default): follow <see cref="RequireGIL"/>.
        /// </param>
        /// <remarks>
        ///     The returned array does NOT own its data (its storage reports view semantics, like
        ///     numpy's <c>flags.owndata == False</c> for foreign buffers): a size-changing
        ///     <see cref="NDArray.resize(Shape, bool)"/> refuses with numpy's "cannot resize this
        ///     array: it does not own its data" instead of silently reallocating away from the
        ///     shared Python memory, and <c>np.require(..., "O")</c> produces an owning copy.
        /// </remarks>
        public static unsafe NDArray ToNDArrayView(PyObject obj, bool allowReadonly = false, bool? requireGIL = null)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();

            using (AcquireGil(requireGIL))
            {
                PyObject mv = null;
                try
                {
                    try
                    {
                        mv = OpenMemoryView(obj);
                    }
                    catch (NotSupportedException) when (obj.HasAttr("__array_interface__"))
                    {
                        // No buffer protocol but numpy-interface metadata exists (e.g. exotic dtypes
                        // fail memoryview with a numpy-side error) — let the interface path decide.
                        return ViewViaArrayInterface(obj, allowReadonly);
                    }

                    if (!mv.c_contiguous)
                    {
                        // numpy arrays carry the richest layout metadata (F-order, >1-D strides,
                        // broadcasts) in __array_interface__ — prefer it.
                        if (obj.HasAttr("__array_interface__"))
                            return ViewViaArrayInterface(obj, allowReadonly);

                        // ANY other buffer-protocol exporter (a sliced / offset / reversed memoryview,
                        // a strided memoryview of an array.array, ...) is STILL viewable: the buffer
                        // protocol hands us the base pointer via a PyBUF.STRIDED request and the
                        // memoryview reports the exact shape/strides. Reconstruct the strided view
                        // rather than declining — only genuinely irreducible layouts (complex64,
                        // big-endian, non-element strides) throw here, and in Auto mode those become
                        // the copy fallback. Extract the metadata, then release the metadata view
                        // before taking the lease buffer (same discipline as the contiguous path).
                        string sFormat = mv.format;
                        long sItemsize = mv.itemsize;
                        long[] sDims = mv.shape;
                        long[] sByteStrides = GetLongTuple(mv, PythonRuntimeInterop.NameStrides);
                        bool sReadonly = GetBool(mv, PythonRuntimeInterop.NameReadonly);
                        mv.Dispose();
                        mv = null;
                        return ViewViaBufferStrides(obj, sFormat, sItemsize, sDims, sByteStrides, sReadonly, allowReadonly);
                    }

                    string format = mv.format;
                    long itemsize = mv.itemsize;
                    NPTypeCode tc = FromBufferFormat(format, itemsize);   // 'Zf' (complex64) throws with copy guidance

                    long[] dims = mv.shape;
                    Shape shape = dims.Length == 0 ? new Shape() : new Shape(dims);
                    if (shape.Size == 0)
                        return new NDArray(tc, shape, fillZeros: false);

                    // Read writeability from the metadata view: it is the authoritative signal and lets
                    // AcquireBuffer AVOID probing PyBUF.WRITABLE on a read-only source. That probe is not
                    // merely wasteful — GetBuffer(PyBUF.WRITABLE) on a read-only *memoryview* HARD-CRASHES
                    // pythonnet 3.0.5 (bytes throws cleanly, a memoryview segfaults), so it must never be
                    // attempted when we already know the source is read-only.
                    bool sourceReadonly = GetBool(mv, PythonRuntimeInterop.NameReadonly);

                    // Take the lease buffer FROM THE MEMORYVIEW WRAPPER, not the raw object. The
                    // memoryview is CPython's canonical, uniformly-behaved buffer exporter, so acquiring
                    // through it sidesteps pythonnet 3.0.x's per-exporter GetBuffer bugs: a raw ctypes
                    // array, for example, hard-crashes obj.GetBuffer for EVERY flag, while the memoryview
                    // over the very same memory leases cleanly. The PyBuffer keeps the memoryview alive
                    // (Py_buffer.obj holds it), which in turn keeps the source pinned — so the wrapper is
                    // disposed right after, and any resize-lock on the source (bytearray) still holds for
                    // the lease's lifetime through the retained memoryview.
                    PyBuffer buf = AcquireBuffer(mv, allowReadonly, sourceReadonly, PyBUF.WRITABLE, PyBUF.SIMPLE, out bool readOnly);
                    mv.Dispose();
                    mv = null;

                    var lease = new ImportLease(buf, holder: null, bytes: buf.Length);
                    try
                    {
                        if (buf.Length != shape.Size * itemsize)
                            throw new InvalidOperationException(
                                $"exporter reported {buf.Length} bytes but shape {shape} x itemsize {itemsize} needs {shape.Size * itemsize}.");

                        // numpy marks arrays over read-only buffers writeable=False; carry the same
                        // flag so NumSharp's guarded write paths raise "assignment destination is
                        // read-only" instead of corrupting an immutable Python object. Derived views
                        // inherit it (Shape.Slice / GetView carry non-writeability through).
                        if (readOnly)
                            shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

                        IArraySlice slice = WrapExternal(tc, (void*)buf.Buffer, shape.Size, lease.Release);
                        // Alias() so the storage reports VIEW semantics (numpy: flags.owndata == False
                        // for foreign buffers): ndarray.resize then refuses to reallocate ("cannot
                        // resize this array: it does not own its data") instead of silently detaching
                        // the view from Python's memory, and np.require(..., "O") copies.
                        var nd = new NDArray(new UnmanagedStorage(slice, shape).Alias());
                        PythonRuntimeInterop.TrackImport(lease);
                        return nd;
                    }
                    catch
                    {
                        PythonRuntimeInterop.TrackImport(lease);   // let the release path account for it
                        lease.Release();
                        throw;
                    }
                }
                finally
                {
                    mv?.Dispose();
                }
            }
        }

        /// <summary>
        ///     Fluent alias of <see cref="ToNDArrayView(PyObject, bool, bool?)"/> following numpy's
        ///     <c>array</c>/<c>asarray</c> naming: <c>To…</c> copies, <c>As…</c> shares. Returns a
        ///     zero-copy NumSharp view over the exporter's memory (shared mutation, shared lifetime).
        /// </summary>
        /// <inheritdoc cref="ToNDArrayView(PyObject, bool, bool?)"/>
        public static NDArray AsNDArray(this PyObject obj, bool allowReadonly = false, bool? requireGIL = null)
            => ToNDArrayView(obj, allowReadonly, requireGIL);

        // ---- zero-copy import internals ----------------------------------------------------------

        private static PyObject OpenMemoryView(PyObject obj)
        {
            try
            {
                return builtins.memoryview(obj);
            }
            catch (PythonException e)
            {
                throw new NotSupportedException(
                    $"the object does not export a PEP 3118 buffer ({e.Message}). " +
                    "Only buffer-protocol objects (numpy arrays, memoryview, bytes, bytearray, array.array, ...) can be converted.", e);
            }
        }

        /// <summary>
        ///     Lease the exporter's buffer, requesting a WRITABLE lock when the source reports itself
        ///     writable and a read-only lock otherwise. <paramref name="writableFlag"/> /
        ///     <paramref name="readonlyFlag"/> select the buffer shape: <c>WRITABLE</c>/<c>SIMPLE</c> for
        ///     a C-contiguous view, <c>STRIDED</c>/<c>STRIDED_RO</c> for a strided one.
        /// </summary>
        private static PyBuffer AcquireBuffer(PyObject obj, bool allowReadonly, bool sourceReadonly, PyBUF writableFlag, PyBUF readonlyFlag, out bool readOnly)
        {
            // A writable lease is what makes the view's shared MUTATION legal — but only REQUEST it
            // when the source reports itself writable (<paramref name="sourceReadonly"/> comes from the
            // exporter's own memoryview.readonly). A writable buffer request must never be attempted on
            // a read-only source: on a read-only *memoryview* it hard-crashes pythonnet 3.0.5 (bytes
            // merely throws BufferError). We therefore gate on the known flag instead of probing by
            // exception — which is also one fewer failed C-API call + throw on every read-only import.
            if (!sourceReadonly)
            {
                try
                {
                    PyBuffer buf = obj.GetBuffer(writableFlag);
                    readOnly = false;
                    return buf;
                }
                catch (PythonException)
                {
                    // Defensive: the source claimed writable yet the lock request still failed. Fall
                    // through to the read-only handling rather than surface a raw BufferError.
                }
            }

            if (!allowReadonly)
                throw new InvalidOperationException(
                    "the exporter's buffer is read-only; writing through a NumSharp view would corrupt an immutable Python object. " +
                    "Use ToNDArray (copy), or pass allowReadonly:true to take a NON-WRITEABLE view (guarded writes through it throw).");
            readOnly = true;
            return obj.GetBuffer(readonlyFlag);
        }

        /// <summary>
        ///     Strided zero-copy import for ANY non-contiguous buffer-protocol exporter that is NOT a
        ///     numpy array (a sliced / offset / reversed <c>memoryview</c>, a strided memoryview of an
        ///     <c>array.array</c>, ...). The base pointer comes from a <c>PyBUF.STRIDED</c>(<c>_RO</c>)
        ///     buffer; the exact <paramref name="dims"/> / <paramref name="byteStrides"/> come from the
        ///     exporter's own memoryview. The window is normalized so element offsets stay non-negative
        ///     (PEP 3118's <c>buf</c> addresses element 0; negative strides address memory below it),
        ///     mirroring <see cref="ViewViaArrayInterface"/> and NumSharp's own reversed views.
        /// </summary>
        private static unsafe NDArray ViewViaBufferStrides(PyObject obj, string format, long itemsize, long[] dims, long[] byteStrides, bool sourceReadonly, bool allowReadonly)
        {
            NPTypeCode tc = FromBufferFormat(format, itemsize);   // 'Zf' (complex64) / big-endian throw → copy fallback in Auto
            if (tc.SizeOf() != itemsize)
                throw new NotSupportedException(
                    $"buffer itemsize {itemsize} does not match NumSharp dtype {tc} ({tc.SizeOf()} bytes); a zero-copy view is not possible. Use ToNDArray (copy).");

            long sizeFromDims = 1;
            for (int i = 0; i < dims.Length; i++)
                sizeFromDims *= dims[i];
            if (sizeFromDims == 0)
                return new NDArray(tc, dims.Length == 0 ? new Shape() : new Shape(dims), fillZeros: false);

            if (byteStrides is null || byteStrides.Length != dims.Length)
                throw new NotSupportedException(
                    "the exporter did not report per-dimension strides for a non-contiguous buffer; a zero-copy view is not possible. Use ToNDArray (copy).");

            var elemStrides = new long[byteStrides.Length];
            for (int i = 0; i < byteStrides.Length; i++)
            {
                if (byteStrides[i] % itemsize != 0)
                    throw new NotSupportedException(
                        $"stride {byteStrides[i]} bytes is not a multiple of itemsize {itemsize}; NumSharp strides are element-based. Use ToNDArray (copy).");
                elemStrides[i] = byteStrides[i] / itemsize;
            }

            // Normalize the window: PEP 3118's buf pointer addresses element 0; negative strides put
            // other elements BELOW it. NumSharp offsets are relative to the block start, so shift the
            // base down to the lowest touched element.
            long minOffset = 0, maxOffset = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                long extent = (dims[i] - 1) * elemStrides[i];
                if (extent < 0) minOffset += extent;
                else maxOffset += extent;
            }
            long spanElements = maxOffset - minOffset + 1;

            PyBuffer buf = AcquireBuffer(obj, allowReadonly, sourceReadonly, PyBUF.STRIDED, PyBUF.STRIDED_RO, out bool readOnly);
            var lease = new ImportLease(buf, holder: null, bytes: spanElements * itemsize);
            try
            {
                long basePtr = (long)buf.Buffer + minOffset * itemsize;
                Shape shape = new Shape(dims, elemStrides, offset: -minOffset, bufferSize: spanElements);
                if (readOnly)
                    shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

                IArraySlice slice = WrapExternal(tc, (void*)basePtr, spanElements, lease.Release);
                // The strided shape's logical size differs from the physical span, so Alias a flat
                // storage with the strided shape (as ViewViaArrayInterface / NumSharp slicing do).
                var storage = new UnmanagedStorage(slice, Shape.Vector(spanElements)).Alias(shape);
                var nd = new NDArray(storage);
                PythonRuntimeInterop.TrackImport(lease);
                return nd;
            }
            catch
            {
                PythonRuntimeInterop.TrackImport(lease);
                lease.Release();
                throw;
            }
        }

        /// <summary>
        ///     Strided zero-copy import for numpy arrays whose layout the buffer protocol cannot hand us
        ///     on pythonnet 3.0.1 (non-contiguous views). Reconstructs the exact numpy layout as a
        ///     NumSharp <see cref="Shape"/>: the buffer window is normalized so element offsets stay
        ///     non-negative (numpy's data pointer addresses element 0; negative strides address memory
        ///     below it), mirroring how NumSharp lays out its own reversed views.
        /// </summary>
        private static unsafe NDArray ViewViaArrayInterface(PyObject obj, bool allowReadonly)
        {
            using PyObject aiObj = obj.__array_interface__;
            using var ai = new PyDict(aiObj);

            string typestr;
            using (PyObject t = ai[PythonRuntimeInterop.NameTypestr]) typestr = t.As<string>();
            NPTypeCode tc = FromNumpyDtypeStr(typestr);   // rejects big-endian / datetime / object dtypes
            int itemsize = tc.SizeOf();

            long dataPtr;
            bool readOnly;
            using (PyObject data = ai[PythonRuntimeInterop.NameData])
            {
                using var dataTuple = PyTuple.AsTuple(data);
                using (PyObject p = dataTuple[0]) dataPtr = p.As<long>();
                using (PyObject r = dataTuple[1]) readOnly = r.As<bool>();
            }

            if (readOnly && !allowReadonly)
                throw new InvalidOperationException(
                    "the numpy array is read-only; writing through a NumSharp view would break its immutability contract. " +
                    "Use ToNDArray (copy), or pass allowReadonly:true to take a NON-WRITEABLE view (guarded writes through it throw).");

            long[] dims;
            using (PyObject s = ai[PythonRuntimeInterop.NameShape]) dims = TupleToLongs(s);

            long sizeFromDims = 1;
            for (int i = 0; i < dims.Length; i++)
                sizeFromDims *= dims[i];
            if (sizeFromDims == 0)
                return new NDArray(tc, new Shape(dims), fillZeros: false);

            long[] byteStrides = null;
            if (ai.HasKey(PythonRuntimeInterop.NameStrides))
                using (PyObject s = ai[PythonRuntimeInterop.NameStrides])
                    if (!s.IsNone())
                        byteStrides = TupleToLongs(s);

            Shape shape;
            long spanElements, basePtr;
            if (byteStrides is null)
            {
                // strides=None means C-contiguous.
                shape = dims.Length == 0 ? new Shape() : new Shape(dims);
                spanElements = shape.Size;
                basePtr = dataPtr;
            }
            else
            {
                var elemStrides = new long[byteStrides.Length];
                for (int i = 0; i < byteStrides.Length; i++)
                {
                    if (byteStrides[i] % itemsize != 0)
                        throw new NotSupportedException(
                            $"stride {byteStrides[i]} bytes is not a multiple of itemsize {itemsize}; NumSharp strides are element-based. Use ToNDArray (copy).");
                    elemStrides[i] = byteStrides[i] / itemsize;
                }

                // Normalize the window: numpy's data pointer addresses view element 0; negative
                // strides put other elements BELOW it. NumSharp offsets are relative to the block
                // start, so shift the base down to the lowest touched element.
                long minOffset = 0, maxOffset = 0;
                for (int i = 0; i < dims.Length; i++)
                {
                    long extent = (dims[i] - 1) * elemStrides[i];
                    if (extent < 0) minOffset += extent;
                    else maxOffset += extent;
                }

                spanElements = maxOffset - minOffset + 1;
                basePtr = dataPtr + minOffset * itemsize;
                shape = new Shape(dims, elemStrides, offset: -minOffset, bufferSize: spanElements);
            }

            // The interface's data tuple is (pointer, readonly): numpy reports readonly=True for
            // writeable=False arrays. Mirror it as a non-writeable NumSharp shape so guarded write
            // paths raise "assignment destination is read-only" (broadcast sources are additionally
            // non-writeable via their stride-0 BROADCASTED flag either way).
            if (readOnly)
                shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            // Keep the numpy array alive with our OWN strong reference (independent of the caller's
            // PyObject wrapper): a single-element Python list is an unambiguous, public-API container.
            var holder = new PyList();
            holder.Append(obj);

            var lease = new ImportLease(buffer: null, holder: holder, bytes: spanElements * itemsize);
            try
            {
                IArraySlice slice = WrapExternal(tc, (void*)basePtr, spanElements, lease.Release);
                // The strided shape's logical size differs from the physical span, so the
                // validating (slice, shape) ctor cannot be used — build a flat storage over the
                // span and Alias it with the strided shape, exactly how NumSharp's own slicing
                // constructs non-contiguous views. The contiguous branch aliases too, purely for
                // the ownership contract: numpy arrays over foreign buffers have owndata == False,
                // and it is view semantics that make ndarray.resize refuse to reallocate away from
                // the shared Python memory.
                UnmanagedStorage storage = byteStrides is null
                    ? new UnmanagedStorage(slice, shape).Alias()
                    : new UnmanagedStorage(slice, Shape.Vector(spanElements)).Alias(shape);
                var nd = new NDArray(storage);
                PythonRuntimeInterop.TrackImport(lease);
                return nd;
            }
            catch
            {
                PythonRuntimeInterop.TrackImport(lease);
                lease.Release();
                throw;
            }
        }

        // ---- copy internals ----------------------------------------------------------------------

        private static unsafe void CopyBuffer(void* src, long srcBytes, NDArray dest, long expectedSourceBytes, bool widenComplex64)
        {
            if (srcBytes != expectedSourceBytes)
                throw new InvalidOperationException($"exporter produced {srcBytes} bytes, expected {expectedSourceBytes}.");

            if (widenComplex64)
            {
                var s = (float*)src;
                var d = (Complex*)dest.Storage.Address;
                long n = dest.size;
                for (long i = 0; i < n; i++)
                    d[i] = new Complex(s[2 * i], s[2 * i + 1]);
            }
            else
            {
                long destBytes = (long)dest.size * dest.dtypesize;
                Buffer.MemoryCopy(src, dest.Storage.Address, destBytes, srcBytes);
            }
        }

        private static bool IsComplex64Format(string format)
        {
            if (string.IsNullOrEmpty(format))
                return false;
            int i = "<>=@!".IndexOf(format[0]) >= 0 ? 1 : 0;
            return format.Substring(i) == "Zf";
        }

        private static long[] TupleToLongs(PyObject tupleLike)
        {
            using var tup = PyTuple.AsTuple(tupleLike);
            int n = (int)tup.Length();
            var values = new long[n];
            for (int i = 0; i < n; i++) { using var e = tup[i]; values[i] = e.As<long>(); }
            return values;
        }
    }
}
