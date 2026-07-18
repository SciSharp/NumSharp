using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop
{
    public static partial class PythonConvert
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
        public static unsafe NDArray ToNDArray(PyObject obj)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            InteropRuntime.EnsureEngine();
            InteropRuntime.DrainPending();

            using (Py.GIL())
            {
                using PyObject mv = OpenMemoryView(obj);

                string format = GetStr(mv, InteropRuntime.NameFormat);
                long itemsize = GetLong(mv, InteropRuntime.NameItemsize);
                bool widenComplex64 = IsComplex64Format(format);
                NPTypeCode tc = widenComplex64 ? NPTypeCode.Complex : FromBufferFormat(format, itemsize);

                long[] dims = GetLongTuple(mv, InteropRuntime.NameShape);
                Shape shape = dims.Length == 0 ? new Shape() : new Shape(dims);
                if (shape.Size == 0)
                    return new NDArray(tc, shape, fillZeros: false);

                var dest = new NDArray(tc, shape, fillZeros: false);
                long expectedSourceBytes = shape.Size * itemsize;

                if (GetBool(mv, InteropRuntime.NameCContiguous))
                {
                    using PyBuffer buf = obj.GetBuffer(PyBUF.SIMPLE);
                    CopyBuffer((void*)buf.Buffer, buf.Length, dest, expectedSourceBytes, widenComplex64);
                }
                else
                {
                    // Linearize through CPython (correct for every stride pattern incl. suboffsets),
                    // then blit the C-ordered bytes. The bytes object is a plain contiguous exporter.
                    using PyObject tobytes = mv.GetAttr(InteropRuntime.NameTobytes);
                    using PyObject bytesObj = tobytes.Invoke(InteropRuntime.StrC);
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
        ///     <para><b>Two zero-copy routes:</b></para>
        ///     <list type="bullet">
        ///       <item><b>C-contiguous PEP 3118 exporters</b> (any object): the buffer is acquired with
        ///         <c>PyBUF.WRITABLE</c>, which pins the exporter and — for resizable objects like
        ///         <c>bytearray</c> — blocks reallocation for the lease's lifetime.</item>
        ///       <item><b>Non-contiguous numpy arrays</b> (slices, transposes, Fortran order, broadcasts):
        ///         imported through <c>__array_interface__</c> as a strided NumSharp view with identical
        ///         layout; broadcast (stride-0) sources become read-only NumSharp views. The numpy array
        ///         is kept alive by a strong reference (numpy's <c>resize(refcheck=True)</c> refuses to
        ///         reallocate while it exists).</item>
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
        ///     Accept read-only sources (<c>bytes</c>, read-only numpy arrays, ...) and return a view that
        ///     NumSharp cannot mark immutable — the caller promises not to write through it. Default
        ///     <c>false</c>: read-only sources throw with guidance instead of risking corruption of
        ///     immutable Python objects.
        /// </param>
        public static unsafe NDArray ToNDArrayView(PyObject obj, bool allowReadonly = false)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            InteropRuntime.EnsureEngine();
            InteropRuntime.DrainPending();

            using (Py.GIL())
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

                    if (!GetBool(mv, InteropRuntime.NameCContiguous))
                    {
                        if (obj.HasAttr("__array_interface__"))
                            return ViewViaArrayInterface(obj, allowReadonly);
                        throw new InvalidOperationException(
                            "buffer is not C-contiguous and the exporter is not a numpy array; a zero-copy view is not possible. Use ToNDArray (copy).");
                    }

                    string format = GetStr(mv, InteropRuntime.NameFormat);
                    long itemsize = GetLong(mv, InteropRuntime.NameItemsize);
                    NPTypeCode tc = FromBufferFormat(format, itemsize);   // 'Zf' (complex64) throws with copy guidance

                    long[] dims = GetLongTuple(mv, InteropRuntime.NameShape);
                    Shape shape = dims.Length == 0 ? new Shape() : new Shape(dims);
                    if (shape.Size == 0)
                        return new NDArray(tc, shape, fillZeros: false);

                    // Dispose the metadata view BEFORE taking the lease buffer: some exporters
                    // count every open view (an extra one is harmless but untidy).
                    mv.Dispose();
                    mv = null;

                    PyBuffer buf = AcquireBuffer(obj, allowReadonly);
                    var lease = new ImportLease(buf, holder: null, bytes: buf.Length);
                    try
                    {
                        if (buf.Length != shape.Size * itemsize)
                            throw new InvalidOperationException(
                                $"exporter reported {buf.Length} bytes but shape {shape} x itemsize {itemsize} needs {shape.Size * itemsize}.");

                        IArraySlice slice = WrapExternal(tc, (void*)buf.Buffer, shape.Size, lease.Release);
                        var nd = new NDArray(new UnmanagedStorage(slice, shape));
                        InteropRuntime.TrackImport(lease);
                        return nd;
                    }
                    catch
                    {
                        InteropRuntime.TrackImport(lease);   // let the release path account for it
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

        // ---- zero-copy import internals ----------------------------------------------------------

        private static PyObject OpenMemoryView(PyObject obj)
        {
            try
            {
                return InteropRuntime.BuiltinsMemoryview.Invoke(obj);
            }
            catch (PythonException e)
            {
                throw new NotSupportedException(
                    $"the object does not export a PEP 3118 buffer ({e.Message}). " +
                    "Only buffer-protocol objects (numpy arrays, memoryview, bytes, bytearray, array.array, ...) can be converted.", e);
            }
        }

        private static PyBuffer AcquireBuffer(PyObject obj, bool allowReadonly)
        {
            // WRITABLE first: a writable lease is what makes the view's shared MUTATION legal.
            // Read-only exporters (bytes, arr.setflags(write=False), ...) refuse it with BufferError.
            try
            {
                return obj.GetBuffer(PyBUF.WRITABLE);
            }
            catch (PythonException e)
            {
                if (!allowReadonly)
                    throw new InvalidOperationException(
                        "the exporter's buffer is read-only; writing through a NumSharp view would corrupt an immutable Python object. " +
                        "Use ToNDArray (copy), or pass allowReadonly:true if you promise not to write through the view.", e);
                return obj.GetBuffer(PyBUF.SIMPLE);
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
            using PyObject aiObj = obj.GetAttr(InteropRuntime.NameArrayInterface);
            using var ai = new PyDict(aiObj);

            string typestr;
            using (PyObject t = ai[InteropRuntime.NameTypestr]) typestr = t.As<string>();
            NPTypeCode tc = FromNumpyDtypeStr(typestr);   // rejects big-endian / datetime / object dtypes
            int itemsize = tc.SizeOf();

            long dataPtr;
            bool readOnly;
            using (PyObject data = ai[InteropRuntime.NameData])
            {
                using var dataTuple = PyTuple.AsTuple(data);
                using (PyObject p = dataTuple[0]) dataPtr = p.As<long>();
                using (PyObject r = dataTuple[1]) readOnly = r.As<bool>();
            }

            if (readOnly && !allowReadonly)
                throw new InvalidOperationException(
                    "the numpy array is read-only; writing through a NumSharp view would break its immutability contract. " +
                    "Use ToNDArray (copy), or pass allowReadonly:true if you promise not to write through the view.");

            long[] dims;
            using (PyObject s = ai[InteropRuntime.NameShape]) dims = TupleToLongs(s);

            long sizeFromDims = 1;
            for (int i = 0; i < dims.Length; i++)
                sizeFromDims *= dims[i];
            if (sizeFromDims == 0)
                return new NDArray(tc, new Shape(dims), fillZeros: false);

            long[] byteStrides = null;
            if (ai.HasKey(InteropRuntime.NameStrides))
                using (PyObject s = ai[InteropRuntime.NameStrides])
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
                // constructs non-contiguous views.
                UnmanagedStorage storage = byteStrides is null
                    ? new UnmanagedStorage(slice, shape)
                    : new UnmanagedStorage(slice, Shape.Vector(spanElements)).Alias(shape);
                var nd = new NDArray(storage);
                InteropRuntime.TrackImport(lease);
                return nd;
            }
            catch
            {
                InteropRuntime.TrackImport(lease);
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
