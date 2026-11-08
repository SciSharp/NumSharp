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
        ///     Copy any supported Python array object into a fresh C-contiguous NumSharp array. Native
        ///     inputs are PEP 3118 buffers (numpy, memoryview, bytes, bytearray, array.array, PyArrow,
        ///     ...); registered <see cref="IPythonArrayAdapter"/> instances first expose library arrays
        ///     such as <c>torch.Tensor</c> and Pandas containers through their official buffer/NumPy
        ///     adapter. Honors strides /
        ///     Fortran order (non-contiguous sources are linearized by CPython's
        ///     <c>memoryview.tobytes('C')</c>). The native buffer route is numpy-agnostic; an adapter may
        ///     use its library's official NumPy interchange object. The result owns its memory — no
        ///     lifetime coupling to the source.
        ///
        ///     <para>0-d exporters produce scalar NDArrays. complex64 buffers (format 'Zf') are widened
        ///     to <see cref="NPTypeCode.Complex"/> (complex128) during the copy, and big-endian multi-byte
        ///     elements are byte-reversed to native order.</para>
        ///
        ///     <para><b>Text.</b> Fixed-width text copies as NumSharp's own sized dtype with the same bytes:
        ///     numpy <c>S&lt;n&gt;</c> / <c>U&lt;n&gt;</c> (format <c>'&lt;n&gt;s'</c> / <c>'&lt;n&gt;w'</c>), a
        ///     4-byte <c>wchar_t</c> buffer as <c>U1</c> (a 2-byte one as <see cref="NPTypeCode.Char"/>), a
        ///     big-endian <c>&gt;U&lt;n&gt;</c> byte-reversed per code point, and <c>np.str_</c> / <c>np.bytes_</c>
        ///     scalars as the 0-d <c>U&lt;n&gt;</c> / <c>S&lt;n&gt;</c> numpy reads (an empty one as a one-character
        ///     <c>''</c>). A NEP 55 <c>StringDType</c> array, which has no buffer, is copied element by element
        ///     through numpy's <c>NpyString</c> C API into a NumSharp <c>StringDType</c> array with the same
        ///     <c>na_object</c> and <c>coerce</c> (every NA value stays NA). An object array whose every
        ///     element is a <c>str</c> copies to <c>StringDType()</c>; any other object array is refused.</para>
        /// </summary>
        /// <param name="obj">The buffer exporter or registered-adapter source to copy.</param>
        /// <param name="requireGIL">
        ///     <c>true</c>: acquire the GIL for this call (re-entrant under an outer <see cref="Py.GIL"/>);
        ///     <c>false</c>: no GIL management — the calling thread must ALREADY hold the GIL;
        ///     <c>null</c> (default): follow <see cref="RequireGIL"/>.
        /// </param>
        public static NDArray ToNDArray(this PyObject obj, bool? requireGIL = null)
            => ToNDArrayCore(obj, requireGIL, useAdapters: true);

        /// <summary>Codec-only route used when adapter decoding is disabled but native buffers remain enabled.</summary>
        internal static NDArray ToNDArrayWithoutAdapters(PyObject obj, bool? requireGIL = null)
            => ToNDArrayCore(obj, requireGIL, useAdapters: false);

        private static unsafe NDArray ToNDArrayCore(PyObject obj, bool? requireGIL, bool useAdapters)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();

            using (AcquireGil(requireGIL))
            {
                // Copy semantics permit an adapter to detach/transfer/materialize. The adapted object is
                // only a front door: every byte/dtype/layout decision below remains the existing bridge.
                using PyObject adapted = useAdapters
                    ? PythonArrayAdapterRegistry.TryAdapt(obj, allowCopy: true)
                    : null;
                PyObject source = adapted ?? obj;
                PyObject opened = TryOpenMemoryView(source);
                if (opened is null)
                {
                    // NEP 55 StringDType exports no buffer (its text lives in its allocator): numpy's own NpyString C API
                    // copies it exactly, NA values and every layout included. Asked only after the buffer failed, so the
                    // numeric paths pay nothing for it; anything else re-raises the exporter's own error with guidance.
                    if (IsStringDTypeArray(source))
                        return ImportStringDType(source);
                    opened = OpenMemoryView(source);
                }

                string format;
                long itemsize;
                long[] dims;
                bool contiguous;
                try
                {
                    format = opened.format;
                    itemsize = opened.itemsize;
                    dims = opened.shape;
                    contiguous = opened.c_contiguous;
                }
                catch
                {
                    opened.Dispose();
                    throw;
                }

                if (format == "O")
                {
                    // An object array: refused, except an array of str — text that cannot be viewed, so it is copied as a
                    // StringDType array. The buffer lock is released first (the conversion reads the objects itself).
                    opened.Dispose();
                    return ImportObjectArrayOfStr(source);
                }

                using PyObject mv = opened;
                DtypeCompatibilityKind kind;
                DType descr;
                int swapUnit;
                long sourceItemsize = itemsize;
                if (IsNumpyBytesScalar(source, format, dims))
                {
                    // numpy reads an np.bytes_ (a bytes subclass exporting its n bytes) as the 0-d |S<n> it is:
                    // np.asarray(np.bytes_(b'ab')) is dtype 'S2', not two uint8 values. A plain bytes object stays raw bytes.
                    descr = TextDescr('S', dims[0]);
                    sourceItemsize = dims[0];
                    dims = Array.Empty<long>();
                    kind = DtypeCompatibilityKind.Blit;
                    swapUnit = 0;
                }
                else
                {
                    kind = ResolveDtypeCompatibility(format, itemsize, out descr, out swapUnit);
                }

                Shape shape = dims.Length == 0 ? new Shape() : new Shape(dims);
                // An empty np.str_ / np.bytes_ is a zero-width element; numpy grows it to one character (np.asarray(np.str_(''))
                // is '<U1'), and so does the copy — a NumSharp string element always has a positive width.
                DType destinationDescr = GrowZeroWidthText(descr);
                if (shape.Size == 0)
                    return new NDArray(destinationDescr, shape, fillZeros: false);
                if (!ReferenceEquals(destinationDescr, descr))
                    return new NDArray(destinationDescr, shape);   // every element is '' (zero-filled); nothing to read

                // Every byte is written below, so the destination skips the zero fill a U allocation otherwise gets.
                var dest = NDArray.ForOverwrite(descr, shape);
                long expectedSourceBytes = shape.Size * sourceItemsize;
                try
                {
                    if (contiguous)
                    {
                        // Read through the MEMORYVIEW, not the raw object — same reason as the view path:
                        // pythonnet 3.0.x's obj.GetBuffer is per-exporter buggy (a raw ctypes array
                        // hard-crashes it on every flag), while the memoryview over the same memory is
                        // uniformly safe. Only a read-only SIMPLE lock is needed here; it is released as
                        // soon as the bytes are blitted.
                        using PyBuffer buf = mv.GetBuffer(PyBUF.SIMPLE);
                        CopyBuffer((void*)buf.Buffer, buf.Length, dest, expectedSourceBytes, kind, swapUnit);
                    }
                    else
                    {
                        // Linearize through CPython (correct for every stride pattern incl. suboffsets),
                        // then blit the C-ordered bytes. The bytes object is a plain contiguous exporter.
                        using PyObject bytesObj = mv.tobytes("C");
                        using PyBuffer buf = bytesObj.GetBuffer(PyBUF.SIMPLE);
                        CopyBuffer((void*)buf.Buffer, buf.Length, dest, expectedSourceBytes, kind, swapUnit);
                    }
                }
                catch
                {
                    // The half-written copy is this call's own; release its buffer now rather than at a finalizer.
                    dest.Dispose();
                    throw;
                }

                return dest;
            }
        }

        /// <summary>
        ///     Materialize ANY array-like Python object — a <c>list</c>, <c>tuple</c>, nested sequence, a
        ///     Python scalar (<c>int</c> / <c>float</c> / <c>bool</c> / <c>complex</c>), or a buffer
        ///     exporter — into a fresh, independent NumSharp array by routing it through
        ///     <c>numpy.asarray</c> first.
        ///
        ///     <para>This is the general array-like companion of <see cref="ToNDArray(PyObject, bool?)"/>:
        ///     <see cref="ToNDArray"/> accepts PEP 3118 exporters plus registered adapter sources,
        ///     whereas this also accepts everyday Python containers a numpy call or plain Python code
        ///     hands back — at the cost of requiring numpy and one extra materialization (<c>numpy.asarray</c>
        ///     builds the ndarray, then <see cref="ToNDArray"/> copies it into NumSharp). The result owns
        ///     its memory — no lifetime coupling to the source. dtype follows numpy's own inference
        ///     (<c>[1, 2, 3]</c> → int64, <c>[1.0, 2.0]</c> → float64); anything numpy can only express as
        ///     an object array (a ragged list, a bignum outside int64 range, a <c>dict</c>) has no NumSharp
        ///     dtype and throws.</para>
        /// </summary>
        /// <param name="obj">The array-like object to materialize.</param>
        /// <param name="requireGIL">GIL policy, exactly as on <see cref="ToNDArray(PyObject, bool?)"/>.</param>
        public static NDArray FromArrayLike(this PyObject obj, bool? requireGIL = null)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();

            using (AcquireGil(requireGIL))
            {
                using PyObject arr = np.asarray(obj);       // list/tuple/nested/scalar -> ndarray (an exporter passes through)
                return ToNDArray(arr, requireGIL: false);   // GIL already held; copy into a fresh, owning NumSharp array
            }
        }

        /// <summary>
        ///     Zero-copy NumSharp view over Python memory: NumSharp SHARES the exporter's buffer
        ///     (mutations visible both ways). A registered <see cref="IPythonArrayAdapter"/> may first
        ///     expose a library object such as a CPU <c>torch.Tensor</c> or homogeneous Pandas object
        ///     through its official shareable array adapter; all memory decisions below remain unchanged.
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
        ///     <para><b>Text.</b> Fixed-width numpy <c>S&lt;n&gt;</c> / <c>U&lt;n&gt;</c> arrays (<c>&lt;U1</c> is
        ///     NumSharp's <c>U1</c>, the same 4-byte code points) and <c>np.str_</c> / <c>np.bytes_</c> scalars
        ///     view through these routes like numbers. Text without a buffer — a NEP 55 <c>StringDType</c>
        ///     array, an object array — and an empty text scalar (no byte to share) are refused with the copy
        ///     route (<see cref="ToNDArray"/>) in the message.</para>
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
        ///     throw with guidance instead. The size does not matter: an EMPTY read-only source
        ///     (<c>np.frombuffer(b'')</c>, an empty broadcast) is refused or non-writeable the same way.
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
        public static NDArray ToNDArrayView(PyObject obj, bool allowReadonly = false, bool? requireGIL = null)
            => ToNDArrayViewCore(obj, allowReadonly, requireGIL, useAdapters: true);

        /// <summary>Codec-only route used when adapter decoding is disabled but native buffers remain enabled.</summary>
        internal static NDArray ToNDArrayViewWithoutAdapters(
            PyObject obj,
            bool allowReadonly = false,
            bool? requireGIL = null)
            => ToNDArrayViewCore(obj, allowReadonly, requireGIL, useAdapters: false);

        private static unsafe NDArray ToNDArrayViewCore(
            PyObject obj,
            bool allowReadonly,
            bool? requireGIL,
            bool useAdapters)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();

            using (AcquireGil(requireGIL))
            {
                // View semantics forbid adapter-side materialization. For Torch this is Tensor.numpy()
                // without force; CPU/grad/layout failures propagate here and Auto codec mode can then
                // choose the ordinary copy fallback through ToNDArray.
                using PyObject adapted = useAdapters
                    ? PythonArrayAdapterRegistry.TryAdapt(obj, allowCopy: false)
                    : null;
                PyObject source = adapted ?? obj;
                PyObject mv = null;
                try
                {
                    mv = TryOpenMemoryView(source);
                    if (mv is null)
                    {
                        // No buffer protocol but numpy-interface metadata exists (e.g. exotic dtypes, a StringDType
                        // array, fail memoryview with a numpy-side error) — let the interface path decide; anything else
                        // re-raises the exporter's own error with guidance.
                        if (source.HasAttr(PythonRuntimeInterop.NameArrayInterface))
                            return ViewViaArrayInterface(source, allowReadonly);
                        mv = OpenMemoryView(source);
                    }

                    if (!mv.c_contiguous)
                    {
                        // numpy arrays carry the richest layout metadata (F-order, >1-D strides,
                        // broadcasts) in __array_interface__ — prefer it.
                        if (source.HasAttr("__array_interface__"))
                            return ViewViaArrayInterface(source, allowReadonly);

                        // ANY other buffer-protocol exporter (a sliced / offset / reversed memoryview,
                        // a strided memoryview of an array.array, ...) is STILL viewable: the buffer
                        // protocol hands us the base pointer via a PyBUF.STRIDED request and the
                        // memoryview reports the exact shape/strides. Reconstruct the strided view
                        // rather than declining — only genuinely irreducible layouts (complex64,
                        // UCS-4 text, big-endian, non-element strides) throw here, and in Auto mode
                        // those become the copy fallback. Extract the metadata, then release the
                        // metadata view before taking the lease buffer (same discipline as the
                        // contiguous path).
                        string sFormat = mv.format;
                        long sItemsize = mv.itemsize;
                        long[] sDims = mv.shape;
                        long[] sByteStrides = GetLongTuple(mv, PythonRuntimeInterop.NameStrides);
                        bool sReadonly = GetBool(mv, PythonRuntimeInterop.NameReadonly);
                        mv.Dispose();
                        mv = null;
                        return ViewViaBufferStrides(source, sFormat, sItemsize, sDims, sByteStrides, sReadonly, allowReadonly);
                    }

                    string format = mv.format;
                    long itemsize = mv.itemsize;
                    long[] dims = mv.shape;
                    DType descr;
                    if (IsNumpyBytesScalar(source, format, dims))
                    {
                        // numpy reads an np.bytes_ as the 0-d |S<n> it is (its n bytes are ONE element), so the view
                        // is that element over the same bytes. A plain bytes object stays raw uint8 bytes.
                        descr = TextDescr('S', dims[0]);
                        dims = Array.Empty<long>();
                    }
                    else
                    {
                        descr = FromBufferFormat(format, itemsize);   // 'Zf' (complex64) / big-endian / 'O' throw with copy guidance
                    }

                    // A zero-width text element (an empty np.str_ / np.bytes_) has no byte to share; the copy grows it to
                    // one character, as numpy does.
                    if (descr.Meta is StringDTypeMeta && descr.itemsize == 0)
                        throw new NotSupportedException(
                            "a zero-width text element (an empty np.str_ / np.bytes_) has no byte a NumSharp view could share. Use ToNDArray (copy): numpy reads it as a one-character '' element.");

                    Shape shape = dims.Length == 0 ? new Shape() : new Shape(dims);

                    // Read writeability from the metadata view: it is the authoritative signal and lets
                    // AcquireBuffer AVOID probing PyBUF.WRITABLE on a read-only source. That probe is not
                    // merely wasteful — GetBuffer(PyBUF.WRITABLE) on a read-only *memoryview* HARD-CRASHES
                    // pythonnet 3.0.5 (bytes throws cleanly, a memoryview segfaults), so it must never be
                    // attempted when we already know the source is read-only. Read before the empty case too, which
                    // carries it (an empty read-only source is refused or non-writeable, like any other).
                    bool sourceReadonly = GetBool(mv, PythonRuntimeInterop.NameReadonly);
                    if (shape.Size == 0)
                        return EmptyView(descr, shape, sourceReadonly, allowReadonly);
                    // The element width comes from the descriptor: an np.bytes_ buffer reports 1-byte items for its one
                    // n-byte element.
                    long elementBytes = descr.itemsize;

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
                        if (buf.Length != shape.Size * elementBytes)
                            throw new InvalidOperationException(
                                $"exporter reported {buf.Length} bytes but shape {shape} x itemsize {elementBytes} needs {shape.Size * elementBytes}.");

                        // numpy marks arrays over read-only buffers writeable=False; carry the same
                        // flag so NumSharp's guarded write paths raise "assignment destination is
                        // read-only" instead of corrupting an immutable Python object. Derived views
                        // inherit it (Shape.Slice / GetView carry non-writeability through).
                        if (readOnly)
                            shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

                        IArraySlice slice = WrapExternal(descr, (void*)buf.Buffer, shape.Size, lease.Release);
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

        /// <summary>
        ///     <c>memoryview(obj)</c>, or <see langword="null"/> when <paramref name="obj"/> exports no PEP 3118 buffer.
        ///     Call under the GIL.
        /// </summary>
        /// <param name="obj">The candidate exporter.</param>
        /// <returns>A new memoryview the caller owns, or <see langword="null"/>.</returns>
        /// <remarks>
        ///     The failure is caught inside Python (<see cref="PythonRuntimeInterop.TryMemoryview"/>), so the routes that
        ///     go on after it — a <c>StringDType</c> copy, the <c>__array_interface__</c> view — pay no .NET exception;
        ///     a caller that has nothing else to try calls <see cref="OpenMemoryView"/>, which raises the exporter's own
        ///     error with the bridge's guidance.
        /// </remarks>
        private static PyObject TryOpenMemoryView(PyObject obj)
        {
            PyObject mv = PythonRuntimeInterop.TryMemoryview.Invoke(obj);
            if (!mv.IsNone())
                return mv;
            mv.Dispose();
            return null;
        }

        /// <summary><c>memoryview(obj)</c>, raising when <paramref name="obj"/> exports no buffer. Call under the GIL.</summary>
        /// <param name="obj">The candidate exporter.</param>
        /// <returns>A new memoryview the caller owns.</returns>
        /// <exception cref="NotSupportedException">The object exports no PEP 3118 buffer; the message carries Python's own
        ///     error and names the objects that do convert.</exception>
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
                    "Only buffer-protocol objects (numpy arrays, memoryview, bytes, bytearray, array.array, ...) " +
                    "or objects handled by a registered IPythonArrayAdapter can be converted.", e);
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
                throw ReadonlyBufferRefusal();
            readOnly = true;
            return obj.GetBuffer(readonlyFlag);
        }

        /// <summary>
        ///     The refusal of a writable view over a read-only exporter (<c>bytes</c>, a <c>writeable=False</c> numpy array,
        ///     a read-only memoryview), naming the two routes that work. Shared by the buffer request and the empty view, so
        ///     an empty read-only source is refused with the same text as a non-empty one.
        /// </summary>
        /// <returns>The exception to throw.</returns>
        private static InvalidOperationException ReadonlyBufferRefusal()
            => new InvalidOperationException(
                "the exporter's buffer is read-only; writing through a NumSharp view would corrupt an immutable Python object. " +
                "Use ToNDArray (copy), or pass allowReadonly:true to take a NON-WRITEABLE view (guarded writes through it throw).");

        /// <summary>
        ///     The view of an EMPTY exporter: a fresh empty array of the descriptor and shape (no byte exists to share), with the
        ///     source's writeability — numpy's own view of empty read-only memory (<c>np.frombuffer(b'')</c>, an empty
        ///     broadcast) is <c>writeable=False</c>, and the read-only refusal does not depend on the size.
        /// </summary>
        /// <param name="descr">The element descriptor.</param>
        /// <param name="shape">The shape (at least one dimension is 0).</param>
        /// <param name="sourceReadonly">Whether the exporter reported its memory read-only.</param>
        /// <param name="allowReadonly">The caller's opt-in to a non-writeable view.</param>
        /// <returns>The empty array, non-writeable when the source is read-only.</returns>
        /// <exception cref="InvalidOperationException">The source is read-only and <paramref name="allowReadonly"/> is false.</exception>
        private static NDArray EmptyView(DType descr, Shape shape, bool sourceReadonly, bool allowReadonly)
        {
            if (sourceReadonly && !allowReadonly)
                throw ReadonlyBufferRefusal();
            var nd = new NDArray(descr, shape, fillZeros: false);
            if (sourceReadonly)
                nd.setflags(write: false);
            return nd;
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
            DType descr = FromBufferFormat(format, itemsize);   // 'Zf' (complex64) / big-endian / 'O' throw → copy fallback in Auto
            // Also refuses a zero-width text element (itemsize 0): there is no byte to share, and no stride to divide by.
            if (descr.itemsize != itemsize || itemsize == 0)
                throw new NotSupportedException(
                    $"buffer itemsize {itemsize} does not match NumSharp dtype {descr} ({descr.itemsize} bytes); a zero-copy view is not possible. Use ToNDArray (copy).");

            long sizeFromDims = 1;
            for (int i = 0; i < dims.Length; i++)
                sizeFromDims *= dims[i];
            if (sizeFromDims == 0)
                return EmptyView(descr, dims.Length == 0 ? new Shape() : new Shape(dims), sourceReadonly, allowReadonly);

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

                IArraySlice slice = WrapExternal(descr, (void*)basePtr, spanElements, lease.Release);
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
            // Rejects big-endian / datetime / object / StringDType (copy-only) dtypes; a fixed-width string maps to its sized
            // descriptor, whose item size is the element stride unit below.
            DType descr = FromNumpyDtypeStr(typestr);
            int itemsize = descr.itemsize;
            if (itemsize == 0)
                throw new NotSupportedException(
                    $"numpy dtype '{typestr}' has zero-width elements; no NumSharp view can share them. Use ToNDArray (copy).");

            long dataPtr;
            bool readOnly;
            if (!ai.HasKey(PythonRuntimeInterop.NameData))
                throw new NotSupportedException(
                    "__array_interface__ has no 'data' entry — the spec then defers to the object's own buffer protocol, which this object does not export. Use ToNDArray (copy), or np.asarray(obj) first.");
            using (PyObject data = ai[PythonRuntimeInterop.NameData])
            {
                // The spec allows 'data' to be the (pointer, readonly) TUPLE or a buffer-like object
                // (PIL.Image emits bytes). Only the tuple form names an address a view can share — and
                // the gate must be a real type check: PySequence_Tuple would happily turn bytes into a
                // tuple of BYTE VALUES, silently promoting the first pixel byte to a pointer.
                if (!PyTuple.IsTupleType(data))
                    throw new NotSupportedException(
                        "__array_interface__['data'] is not a (pointer, readonly) tuple — buffer-object data (PIL images, ...) names no address a zero-copy view could share. Use ToNDArray (copy), or np.asarray(obj) first.");
                using var dataTuple = PyTuple.AsTuple(data);
                if (dataTuple.Length() != 2)
                    throw new NotSupportedException(
                        $"__array_interface__['data'] tuple has {dataTuple.Length()} items, expected (pointer, readonly).");
                using (PyObject p = dataTuple[0]) dataPtr = p.As<long>();
                // The readonly flag is read by TRUTHINESS, not As<bool>: the spec shows a bool, but
                // real-world producers emit 0/1 ints too, and pythonnet's bool conversion rejects ints.
                using (PyObject r = dataTuple[1]) readOnly = r.IsTrue();
            }

            if (readOnly && !allowReadonly)
                throw new InvalidOperationException(
                    "the numpy array is read-only; writing through a NumSharp view would break its immutability contract. " +
                    "Use ToNDArray (copy), or pass allowReadonly:true to take a NON-WRITEABLE view (guarded writes through it throw).");

            long[] dims;
            using (PyObject s = ai[PythonRuntimeInterop.NameShape]) dims = TupleToLongs(s);

            // A hostile/buggy __array_interface__ producer can name a negative dimension; numpy
            // rejects it ("negative dimensions are not allowed") rather than build a view with a
            // negative extent (which downstream flows to a negative buffer count). Match numpy.
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] < 0)
                    throw new NotSupportedException(
                        $"__array_interface__ 'shape' entry {i} is negative ({dims[i]}); negative dimensions are not allowed.");

            long sizeFromDims = 1;
            for (int i = 0; i < dims.Length; i++)
                sizeFromDims *= dims[i];
            if (sizeFromDims == 0)
                return EmptyView(descr, dims.Length == 0 ? new Shape() : new Shape(dims), readOnly, allowReadonly);

            // A non-empty array must name a real address to share. numpy rejects a NULL data pointer
            // here ("data is NULL but array contains data"); without this guard a null (or otherwise
            // absent) pointer yields a NumSharp view over address 0 whose first read faults deep in
            // UnmanagedStorage — the memory-unsafe opposite of this package's zero-copy-safety contract.
            if (dataPtr == 0)
                throw new NotSupportedException(
                    "__array_interface__ 'data' pointer is NULL but the array is non-empty; there is no memory to share for a zero-copy view. Use ToNDArray (copy), or np.asarray(obj) first.");

            long[] byteStrides = null;
            if (ai.HasKey(PythonRuntimeInterop.NameStrides))
                using (PyObject s = ai[PythonRuntimeInterop.NameStrides])
                    if (!s.IsNone())
                        byteStrides = TupleToLongs(s);

            // numpy rejects a strides tuple whose length differs from the shape ("mismatch in length
            // of strides and shape"). Without this a too-long tuple builds a Shape whose Strides.Length
            // != ndim (corrupt view over a mis-normalized window), and a too-short one throws a raw
            // IndexOutOfRangeException from the stride-normalization loop below.
            if (byteStrides != null && byteStrides.Length != dims.Length)
                throw new NotSupportedException(
                    $"__array_interface__ 'strides' has {byteStrides.Length} entr{(byteStrides.Length == 1 ? "y" : "ies")} but 'shape' has {dims.Length}; mismatch in length of strides and shape.");

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
                IArraySlice slice = WrapExternal(descr, (void*)basePtr, spanElements, lease.Release);
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

        /// <summary>
        ///     How <see cref="ToNDArray"/> materializes one PEP 3118 element type into a NumSharp dtype.
        ///     A source whose element type is bit-identical to a NumSharp dtype is a straight blit (possibly
        ///     byte-reversed per unit for a big-endian source); complex64, whose only NumSharp counterpart
        ///     is a DIFFERENT width, needs an element-wise conversion during the copy (and so can never be a
        ///     zero-copy view — the view path lets <see cref="FromBufferFormat"/> throw the copy guidance
        ///     instead). Resolved once per import by <see cref="ResolveDtypeCompatibility"/> and consumed by
        ///     <see cref="CopyBuffer"/>.
        /// </summary>
        /// <remarks>
        ///     UCS-4 text used to be a third kind (narrowed to <see cref="NPTypeCode.Char"/>, BMP only). Since NumSharp
        ///     has fixed-width string dtypes, numpy's <c>'&lt;U1'</c> is NumSharp's <c>U1</c> — the same bytes — so text is
        ///     a blit like any other buffer.
        /// </remarks>
        private enum DtypeCompatibilityKind
        {
            /// <summary>Element type is bit-identical to its NumSharp dtype — a straight byte blit.</summary>
            Blit,

            /// <summary>complex64 ('Zf'): widen each (float32, float32) pair to a 16-byte <see cref="Complex"/>.</summary>
            WidenComplex64,
        }

        /// <summary>
        ///     Classify a PEP 3118 element type by the <see cref="DtypeCompatibilityKind"/> path
        ///     <see cref="ToNDArray"/> must take, hand back the destination <paramref name="descr"/> the
        ///     path implies, and report the sub-element byte width <paramref name="swapUnit"/> that
        ///     <see cref="CopyBuffer"/> must byte-reverse (0 = no swap — native-endian source).
        ///
        ///     <para><b>Big-endian is the copy path's alone.</b> A zero-copy VIEW over big-endian memory
        ///     is impossible on a native-endian NumSharp buffer, so <see cref="FromBufferFormat"/> — the
        ///     view path's dtype gate — refuses every big-endian multi-byte format. The COPY path,
        ///     however, can byte-reverse each element as it blits, exactly as it widens complex64;
        ///     so a big-endian source decodes to a byteswapped copy here rather than failing outright.
        ///     Single-byte big-endian formats ('&gt;b', '&gt;B', '&gt;?') are byte-order-irrelevant and
        ///     fall through to <see cref="FromBufferFormat"/> unchanged.</para>
        /// </summary>
        /// <param name="format">The source's <c>memoryview.format</c>.</param>
        /// <param name="itemsize">The source's <c>memoryview.itemsize</c>.</param>
        /// <param name="descr">Receives the destination descriptor.</param>
        /// <param name="swapUnit">Receives the byte width each reversed group spans (0: no swap).</param>
        /// <returns>The materialization kind.</returns>
        /// <exception cref="NotSupportedException">The format has no NumSharp dtype at any byte order (see
        ///     <see cref="FromBufferFormat"/> / <see cref="ResolveBigEndianBlitDtype"/>).</exception>
        private static DtypeCompatibilityKind ResolveDtypeCompatibility(string format, long itemsize, out DType descr, out int swapUnit)
        {
            swapUnit = 0;
            bool bigEndian = IsBigEndianFormat(format);

            if (IsComplex64Format(format))
            {
                // The endianness check must live HERE, not in FromBufferFormat: complex64 short-circuits
                // to WidenComplex64 before the format ever reaches the endian-aware FromBufferFormat, so
                // a big-endian '>Zf' was previously widened by reading its float32 halves as native —
                // silent data corruption. Byte-reverse each 4-byte half first when the source is BE.
                descr = DType.Complex;
                if (bigEndian) swapUnit = 4;
                return DtypeCompatibilityKind.WidenComplex64;
            }

            if (bigEndian && itemsize > 1)
            {
                descr = ResolveBigEndianBlitDtype(format, itemsize, out swapUnit);
                return DtypeCompatibilityKind.Blit;
            }

            descr = FromBufferFormat(format, itemsize);
            return DtypeCompatibilityKind.Blit;
        }

        /// <summary>
        ///     True for the PEP 3118 big-endian / network byte-order markers ('&gt;' and '!'). Native
        ///     order ('&lt;', '=', '@', or no marker) is false. Byte order only matters for multi-byte
        ///     elements, so callers that decide on a byteswap additionally gate on <c>itemsize &gt; 1</c>.
        /// </summary>
        private static bool IsBigEndianFormat(string format)
        {
            if (string.IsNullOrEmpty(format))
                return false;
            char c0 = format[0];
            return c0 == '>' || c0 == '!';
        }

        /// <summary>
        ///     Map a BIG-ENDIAN, multi-byte PEP 3118 format to its native NumSharp descriptor and the
        ///     sub-element byte width <see cref="ByteSwapCopy"/> must reverse. Reached ONLY by
        ///     <see cref="ToNDArray"/> (the copy path): the view path's <see cref="FromBufferFormat"/>
        ///     refuses big-endian because a native-endian zero-copy view is impossible. complex128 ('Zd')
        ///     reverses each 8-byte half independently (real, then imag) — never the 16-byte element as
        ///     one unit; UCS-4 text ('&gt;5w', numpy's '&gt;U5') reverses each 4-byte code point; a byte
        ///     string ('&gt;5s') has no byte order and is copied as it is.
        /// </summary>
        /// <param name="format">The big-endian format (it starts with '&gt;' or '!').</param>
        /// <param name="itemSize">The element size in bytes.</param>
        /// <param name="swapUnit">Receives the byte width of each reversed group (0 for a byte string).</param>
        /// <returns>The native NumSharp descriptor.</returns>
        /// <exception cref="NotSupportedException">The format has no NumSharp dtype at ANY byte order
        ///     (extended-precision long double, structured, a text count that disagrees with the itemsize) —
        ///     matching <see cref="FromBufferFormat"/>'s refusals.</exception>
        private static DType ResolveBigEndianBlitDtype(string format, long itemSize, out int swapUnit)
        {
            string code = format.Substring(1);   // starts with '>' or '!' — guaranteed by IsBigEndianFormat
            if (TryParseCountedText(code, out char textCode, out long count))
            {
                long width = textCode == 'w' ? count * 4 : count;
                if (width == itemSize)
                {
                    // A UCS-4 code point is a 4-byte unit; a byte string's characters are single bytes (no order).
                    swapUnit = textCode == 'w' ? 4 : 0;
                    return TextDescr(textCode == 'w' ? 'U' : 'S', count);
                }
            }

            switch (code)
            {
                case "h": swapUnit = 2; return DType.Int16;
                case "H": swapUnit = 2; return DType.UInt16;
                case "i": case "l": swapUnit = (int)itemSize; return itemSize == 8 ? DType.Int64 : DType.Int32;
                case "I": case "L": swapUnit = (int)itemSize; return itemSize == 8 ? DType.UInt64 : DType.UInt32;
                case "n": case "q": swapUnit = 8; return DType.Int64;
                case "N": case "Q": swapUnit = 8; return DType.UInt64;
                case "e": swapUnit = 2; return DType.Half;
                case "f": swapUnit = 4; return DType.Single;
                case "d": swapUnit = 8; return DType.Double;
                case "g":                                      // C long double: IEEE double only at width 8 (MSVC)
                    if (itemSize == 8) { swapUnit = 8; return DType.Double; }
                    break;
                case "u":                                      // wchar_t text unit: a 2-byte BE unit is a UTF-16 code unit,
                    if (itemSize == 2) { swapUnit = 2; return DType.Char; }
                    if (itemSize == 4) { swapUnit = 4; return TextDescr('U', 1); }   // a 4-byte one a UCS-4 code point
                    break;                                     // 1-byte never reaches here (itemsize > 1)
                case "Zd": swapUnit = 8; return DType.Complex;   // complex128 — each 8-byte half reversed
            }

            swapUnit = 0;
            throw new NotSupportedException(
                $"big-endian buffer format '{format}' (itemsize {itemSize}) has no NumSharp dtype.");
        }

        /// <summary>
        ///     Materializes <paramref name="srcBytes"/> source bytes (C-ordered elements) into the fresh
        ///     destination by the resolved <paramref name="kind"/>: a straight or byte-reversing blit, or the
        ///     complex64 widening.
        /// </summary>
        /// <param name="src">The first source byte.</param>
        /// <param name="srcBytes">The source byte count.</param>
        /// <param name="dest">The fresh C-contiguous destination (every byte is overwritten).</param>
        /// <param name="expectedSourceBytes">The byte count the shape and the source element size imply.</param>
        /// <param name="kind">The materialization kind.</param>
        /// <param name="swapUnit">The byte width of each reversed group (0 or 1: no swap).</param>
        /// <exception cref="InvalidOperationException">The exporter produced another byte count than its shape implies.</exception>
        private static unsafe void CopyBuffer(void* src, long srcBytes, NDArray dest, long expectedSourceBytes, DtypeCompatibilityKind kind, int swapUnit)
        {
            if (srcBytes != expectedSourceBytes)
                throw new InvalidOperationException($"exporter produced {srcBytes} bytes, expected {expectedSourceBytes}.");

            switch (kind)
            {
                case DtypeCompatibilityKind.WidenComplex64:
                {
                    var d = (Complex*)dest.Storage.Address;
                    long n = dest.size;
                    if (swapUnit == 4)
                    {
                        // Big-endian complex64: byte-reverse each 4-byte float32 half before widening.
                        var sb = (byte*)src;
                        for (long i = 0; i < n; i++)
                        {
                            float re = ReadBigEndianSingle(sb + (2 * i) * 4);
                            float im = ReadBigEndianSingle(sb + (2 * i + 1) * 4);
                            d[i] = new Complex(re, im);
                        }
                    }
                    else
                    {
                        var s = (float*)src;
                        for (long i = 0; i < n; i++)
                            d[i] = new Complex(s[2 * i], s[2 * i + 1]);
                    }
                    break;
                }

                default:   // Blit — bit-identical (native) or byte-reversed-per-unit (big-endian)
                {
                    long destBytes = (long)dest.size * dest.dtypesize;
                    if (swapUnit > 1)
                        ByteSwapCopy((byte*)src, (byte*)dest.Storage.Address, srcBytes, swapUnit);
                    else
                        Buffer.MemoryCopy(src, dest.Storage.Address, destBytes, srcBytes);
                    break;
                }
            }
        }

        /// <summary>
        ///     Copy <paramref name="srcBytes"/> bytes from <paramref name="src"/> to <paramref name="dest"/>,
        ///     reversing every consecutive <paramref name="unit"/>-byte group — the big-endian → native
        ///     blit. <paramref name="srcBytes"/> is always a whole multiple of <paramref name="unit"/> (a
        ///     buffer holds an integral number of elements, and <paramref name="unit"/> divides the element
        ///     size: it IS the element size for real types, half of it for complex128, whose two IEEE
        ///     halves are byte-reversed independently, and one code point for UCS-4 text).
        /// </summary>
        private static unsafe void ByteSwapCopy(byte* src, byte* dest, long srcBytes, int unit)
        {
            for (long b = 0; b < srcBytes; b += unit)
                for (int k = 0; k < unit; k++)
                    dest[b + k] = src[b + (unit - 1 - k)];
        }

        /// <summary>Read a big-endian IEEE-754 float32 from <paramref name="p"/> (4 bytes) into a native float.</summary>
        private static unsafe float ReadBigEndianSingle(byte* p)
        {
            uint bits = ((uint)p[0] << 24) | ((uint)p[1] << 16) | ((uint)p[2] << 8) | p[3];
            return BitConverter.UInt32BitsToSingle(bits);
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
