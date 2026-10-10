using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.OnnxRuntime
{
    public static partial class NDArrayOnnxInterop
    {
        // ===========================  ONNX Runtime  ->  NumSharp  ===========================

        /// <summary>
        ///     Copy an ORT tensor (typically a <c>session.Run</c> output) into a fresh, owning, C-contiguous
        ///     <see cref="NDArray"/> — the safe default: the <see cref="OrtValue"/> can be disposed the moment
        ///     this returns, and the result has no lifetime coupling to ORT.
        ///
        ///     <para>The dtype follows <see cref="FromTensorElementType"/> (Float16 → Half; UInt16 → UInt16, never
        ///     Char), the shape is the tensor's row-major shape (0-d and empty tensors included). A STRING tensor
        ///     comes back as a text array: fixed-width Unicode (<c>U</c>), as wide as its longest string, so a string
        ///     that ends in NUL characters loses them (NumSharp has no variable-width string dtype yet). The tensor
        ///     must be a dense tensor in CPU-addressable memory: sequences / maps / sparse tensors and
        ///     device-resident (CUDA / DirectML) outputs are refused with a message naming the alternative.</para>
        /// </summary>
        /// <remarks>
        ///     <para>Every refusal is decided before the result is allocated. A tensor of more than 2 GB is refused: ORT's
        ///     managed API exposes a tensor's memory only as a <see cref="Span{T}"/> of bytes, which cannot cover it (see
        ///     <see cref="ImportTooLargeMessage"/> for the route that works at any size — ORT writing into a pre-allocated
        ///     NumSharp output).</para>
        ///     <para>A value this package made (<see cref="OrtTensor.Value"/> of an <see cref="AsOrtValue"/> handle) is
        ///     the exception: its memory is a NumSharp buffer, so it is copied from that buffer directly, at any size.</para>
        /// </remarks>
        /// <param name="value">A dense tensor OrtValue in CPU-accessible memory; it stays the caller's to dispose.</param>
        /// <returns>A new owning C-contiguous array holding a copy of the tensor's elements.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a dense tensor; a BFloat16 element type; or a tensor of more than 2 GB.</exception>
        /// <exception cref="ArgumentException">The tensor's shape has a negative (symbolic) dimension.</exception>
        /// <exception cref="InvalidOperationException">The tensor lives in device memory the CPU cannot address, or ORT reports a byte count its shape contradicts.</exception>
        /// <exception cref="ObjectDisposedException">The value is one this package made and its handle was disposed while it was read.</exception>
        public static unsafe NDArray ToNDArray(this OrtValue value)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            // A value this package made over a NumSharp buffer: copy from the buffer itself. No span is involved, so the
            // 2 GB limit of ORT's managed data API does not apply, and no native call is made. (A TEXT value the package
            // made is a string tensor ORT owns: it takes the string path below like any other.)
            if (OrtTensor.TryGetOwner(value, out OrtTensor owner) && owner.IsNumSharpMemory)
                return CopyExportedValue(owner);

            EnsureTensor(value, nameof(ToNDArray));
            OrtTensorTypeAndShapeInfo info = value.GetTensorTypeAndShape();
            if (info.IsString)
                return TextToNDArray(value, info.Shape);   // a new U array; string tensors are always CPU memory
            NPTypeCode tc = FromTensorElementType(info.ElementDataType);
            Shape shape = ShapeFrom(info.Shape);
            EnsureCpuAccessible(value, nameof(ToNDArray));

            // Decide the size and read the source span BEFORE allocating the result. The result used to be allocated
            // first, so a tensor past 2 GB cost a full-size allocation (2 GB for a 2 GB tensor, 4 GB for a 4 GB one) that
            // was then abandoned to the finalizer when ORT's span failed — with ArgumentOutOfRangeException when the byte
            // count's int cast came out negative, or the "ORT reports 16 tensor bytes" InvalidOperationException below
            // when it wrapped to a small positive number. Neither said why.
            int itemsize = tc.SizeOf();
            long count = info.ElementCount;
            if (count > int.MaxValue / itemsize)
                throw new NotSupportedException(ImportTooLargeMessage(nameof(ToNDArray), count, itemsize));
            Span<byte> src = count > 0 ? value.GetTensorMutableRawData() : Span<byte>.Empty;
            long nbytes = shape.Size * itemsize;
            if (src.Length != nbytes)
                throw new InvalidOperationException($"ORT reports {src.Length} tensor bytes but shape {shape} × itemsize {itemsize} needs {nbytes}.");

            // Nothing below can fail once the result exists: the lengths match, so the copy cannot throw.
            var nd = new NDArray(tc, shape, fillZeros: false);
            if (nbytes > 0)
                src.CopyTo(new Span<byte>(nd.Storage.InternalArray.Address, src.Length));
            // The span reads ORT's native memory, which the OrtValue frees when it is finalized, and the span does not keep
            // the OrtValue alive: once its last use above is behind us, a caller that holds no other reference to it (a
            // value made just for this call) lets the GC collect it — during the allocation of the result or while the copy
            // runs — and the copy read freed memory. Probed: copying 2^14-element outputs nobody else referenced, under GC
            // pressure, ended the process within seconds (a fatal access violation inside the copy). ORT's own rule: a
            // tensor's span is valid only while its OrtValue is alive.
            GC.KeepAlive(value);
            return nd;
        }

        /// <summary>
        ///     View an ORT tensor as a NumSharp array over the tensor's OWN buffer — zero-copy, mutations visible both
        ///     ways (<c>OrtValue.GetTensorMutableDataAsSpan</c> and the NDArray read the same bytes). The view is a
        ///     genuine NumSharp array: slice it, reduce it, feed it to kernels, or hand it straight back to ORT
        ///     with <see cref="AsOrtValue"/>.
        ///
        ///     <para><b>Lifetime of a value ORT owns</b> (a <c>session.Run</c> output, a value from an ORT factory). By
        ///     default (<paramref name="ownsValue"/> = <c>false</c>) the OrtValue stays the caller's: the view holds a
        ///     strong reference so the value cannot be finalized underneath it, but the caller must not <c>Dispose</c>
        ///     the OrtValue (or the collection <c>session.Run</c> returned) while any view — or any slice derived from
        ///     it — is still in use, exactly the rule ORT documents for <c>GetTensorDataAsSpan</c>. With
        ///     <paramref name="ownsValue"/> = <c>true</c> the view TAKES OWNERSHIP: the OrtValue is disposed when the
        ///     last NumSharp view over the memory dies (disposal or collection), and the caller must not dispose it
        ///     again — the safe way to keep an output alive past <c>session.Run</c> without a copy (what
        ///     <c>session.Run(..., zeroCopyOutputs: true)</c> does).</para>
        ///
        ///     <para><b>Lifetime of a value this package made</b> (<see cref="OrtTensor.Value"/> of an
        ///     <see cref="AsOrtValue"/> handle). Its memory is a NumSharp buffer, so the view is an ordinary NumSharp
        ///     alias of that buffer: it holds its own counted reference and stays valid after the handle, the OrtValue
        ///     and the source array are all disposed. It is read-only when the exported array was. With
        ///     <paramref name="ownsValue"/> = <c>true</c> the handle is disposed as soon as the view exists.</para>
        ///
        ///     <para>The view does not own its data (<c>flags.owndata == False</c>, like <c>np.frombuffer</c>): a
        ///     size-changing <c>ndarray.resize</c> refuses instead of detaching from the memory. CPU-addressable
        ///     tensors only; an empty tensor yields an empty array (nothing to share).</para>
        /// </summary>
        /// <remarks>
        ///     <para>A tensor of more than 2 GB that ORT owns is refused, whatever its exact size: ORT's managed API hands
        ///     out a tensor's memory only through a <see cref="Span{T}"/> of bytes, whose length it computes with an
        ///     unchecked <c>int</c> cast, so past 2 GB the span either fails to construct or comes back truncated (see
        ///     <see cref="ImportTooLargeMessage"/>). A value this package made has no such limit.</para>
        ///     <para>Every refusal and every failure leaves <paramref name="value"/> untouched and still the caller's, even
        ///     with <paramref name="ownsValue"/> = <c>true</c>: ownership transfers only with a returned array.</para>
        ///     <para><b>Modes.</b> A dense numeric tensor in CPU memory can always be viewed, so
        ///     <see cref="OnnxCodecMode.Auto"/> and <see cref="OnnxCodecMode.View"/> do the same for it.
        ///     <see cref="OnnxCodecMode.Copy"/> returns an independent copy (what <see cref="ToNDArray(OrtValue)"/>
        ///     returns, with its refusals) and, with <paramref name="ownsValue"/> = <c>true</c>, disposes the value once
        ///     the copy exists. A STRING tensor is the one case where the modes differ: it cannot be viewed, so
        ///     <see cref="OnnxCodecMode.Auto"/> returns the copy (a text array) and <see cref="OnnxCodecMode.View"/>
        ///     throws. No mode makes a tensor in device memory or a non-tensor value readable.</para>
        ///     <para><b>Memory pressure.</b> A view that owns an ORT buffer reports the buffer's size to the GC for its
        ///     lifetime. A non-owning view reports nothing: the caller's OrtValue owns that memory.</para>
        ///     <para><b>A value ORT returned over memory that is not ORT's.</b> For a model output that is a FED INPUT
        ///     passed through (no node produces it), ONNX Runtime returns a value over the fed tensor's memory. When you
        ///     ran the session yourself and fed an <see cref="AsOrtValue"/> handle, that memory is the handle's NumSharp
        ///     buffer, and this verb cannot know it (the returned value is another object than the handle's): the view is
        ///     then valid only while the handle, or the array you fed, is alive, even with <paramref name="ownsValue"/> =
        ///     <c>true</c>. The <c>Run</c> tier of this package knows what it fed and returns a counted view for such an
        ///     output.</para>
        /// </remarks>
        /// <param name="value">A dense tensor OrtValue in CPU-accessible memory.</param>
        /// <param name="ownsValue"><c>true</c> to dispose the OrtValue when the last view dies; <c>false</c> (default) to leave it the caller's.</param>
        /// <param name="mode">View (also the meaning of the default here) or an independent copy; see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A NumSharp view over the tensor's buffer (an empty array for an empty tensor, which has nothing to share), or a copy under <see cref="OnnxCodecMode.Copy"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="NotSupportedException">Not a dense tensor; a BFloat16 element type; or a tensor of more than 2 GB that ORT owns.</exception>
        /// <exception cref="ArgumentException">The tensor's shape has a negative (symbolic) dimension.</exception>
        /// <exception cref="InvalidOperationException">The tensor lives in device memory the CPU cannot address; or it is a string tensor and <paramref name="mode"/> is <see cref="OnnxCodecMode.View"/>.</exception>
        /// <exception cref="ObjectDisposedException">The value is one this package made and its handle was disposed while the view was built.</exception>
        public static unsafe NDArray AsNDArray(this OrtValue value, bool ownsValue = false, OnnxCodecMode mode = OnnxCodecMode.Auto)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            EnsureDefined(mode);

            if (mode == OnnxCodecMode.Copy)
            {
                // An independent copy needs nothing from the value afterwards. Owning means "the result is responsible
                // for the value", and a copy has no use for it, so it is disposed now — after the copy exists, so a
                // failed copy leaves the value the caller's.
                NDArray copy = value.ToNDArray();
                if (ownsValue)
                    DisposeValue(value);
                return copy;
            }

            // A value this package made over a NumSharp buffer is NumSharp memory, not foreign memory: the view is a
            // counted alias of the buffer. It used to be leased like an ORT buffer, so the view rooted only the OrtValue,
            // which owns nothing: once the handle and the source were disposed the buffer went back to the pool under
            // the view (a probe read another array's 777s through it).
            if (OrtTensor.TryGetOwner(value, out OrtTensor owner) && owner.IsNumSharpMemory)
                return AliasExportedValue(owner, ownsValue);

            EnsureTensor(value, nameof(AsNDArray));
            OrtTensorTypeAndShapeInfo info = value.GetTensorTypeAndShape();
            if (info.IsString)
            {
                // Text cannot be viewed: ORT's strings are separate UTF-8 objects, NumSharp's are fixed-width code
                // points. Auto falls back to the copy; the strict mode says so.
                if (mode == OnnxCodecMode.View)
                    throw new InvalidOperationException(TextViewMessage(nameof(AsNDArray)));
                NDArray text = TextToNDArray(value, info.Shape);
                if (ownsValue)
                    DisposeValue(value);   // the copy needs nothing from the value; owning it means freeing it
                return text;
            }
            NPTypeCode tc = FromTensorElementType(info.ElementDataType);
            Shape shape = ShapeFrom(info.Shape);
            EnsureCpuAccessible(value, nameof(AsNDArray));

            long count = info.ElementCount;
            // The pointer comes from GetTensorMutableRawData, whose (int) byte count is negative for some sizes past 2 GB
            // (2 GB + 16 threw ArgumentOutOfRangeException) and wraps to a small positive number for others (4 GB + 16
            // returned a 16-byte span at the right address, so the view happened to work). Which one a tensor got depended
            // on bit 31 of its byte count; refusing every size past the span limit makes the outcome the documented one.
            int itemsize = tc.SizeOf();
            if (count > int.MaxValue / itemsize)
                throw new NotSupportedException(ImportTooLargeMessage(nameof(AsNDArray), count, itemsize));
            if (count == 0)
            {
                // Nothing to share — an empty array of the right dtype/shape IS the view.
                var empty = new NDArray(tc, shape, fillZeros: false);
                if (ownsValue)
                    value.Dispose();
                return empty;
            }

            Span<byte> raw = value.GetTensorMutableRawData();
            // The span fronts NATIVE memory (ORT hands out its tensor pointer), so taking the address is stable —
            // nothing the GC can move. It is the only public route to the pointer in the managed API.
            byte* data = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(raw));

            // Owning: the last view disposes the OrtValue. Non-owning: the closure ROOTS the OrtValue for the
            // life of the lease (the lease is referenced from the memory block's dispose hook), so the value's
            // finalizer cannot free the buffer while a view still reads it.
            // Only the owning lease reports the buffer to the GC: its bytes live exactly as long as the view. A
            // non-owning view's bytes belong to the caller's OrtValue, and reporting them once per view made the GC
            // collect more for memory no collection can free.
            Action release = ownsValue ? value.Dispose : () => GC.KeepAlive(value);
            var lease = new ImportLease(release, ownedBytes: ownsValue ? raw.Length : 0);
            try
            {
                IArraySlice slice = WrapExternal(tc, data, count, lease.Release);
                // Alias() so the storage reports VIEW semantics (numpy: flags.owndata == False for foreign
                // buffers): ndarray.resize then refuses to reallocate instead of silently detaching from ORT's memory.
                return new NDArray(new UnmanagedStorage(slice, shape).Alias());
            }
            catch
            {
                // The view was not returned, so ownership did not transfer: end the lease WITHOUT its release action.
                // Releasing it disposed the caller's OrtValue on a failure the documentation says leaves it untouched,
                // and the memory block that already carries the lease would have disposed it later from its finalizer.
                lease.Disarm();
                throw;
            }
        }

        /// <summary>
        ///     Disposes an <see cref="OrtValue"/> the way its owner must: through its export handle when this package made
        ///     the value (the handle then also gives back its buffer reference and its temporary copy, and is not left to
        ///     its finalizer), directly otherwise.
        /// </summary>
        /// <param name="value">The value to dispose.</param>
        internal static void DisposeValue(OrtValue value)
        {
            if (OrtTensor.TryGetOwner(value, out OrtTensor owner))
                owner.Dispose();
            else
                value.Dispose();
        }

        /// <summary>
        ///     The view <see cref="AsNDArray(OrtValue, bool, OnnxCodecMode)"/> returns for a value this package made over a
        ///     NumSharp buffer: an alias of that buffer whose reference counting is the buffer's own
        ///     (<c>ArraySlice.WrapShared</c>), so the view keeps the buffer alive exactly as any NumSharp view does, whatever
        ///     happens to the handle, the OrtValue and the exported array.
        /// </summary>
        /// <remarks>
        ///     <para>The pointer, the shape and the element type come from the handle, not from ORT: no span is needed, so a
        ///     tensor of any size can be viewed. The dtype is the CROSSED one (a Char array comes back as UInt16), as for
        ///     any tensor.</para>
        ///     <para>The view is read-only when the exported array was: a round trip through ORT must not hand out a
        ///     writeable view of read-only memory (a broadcast, a read-only memory map).</para>
        ///     <para>The Run tier calls this too, for a model output that is a FED INPUT passed through: ONNX Runtime
        ///     returns such an output as a value over the fed tensor's memory, which is the handle's NumSharp buffer
        ///     (see <c>InferenceSessionExtensions.FedTensor</c>).</para>
        /// </remarks>
        /// <param name="owner">The value's export handle; it must be over a NumSharp buffer (<see cref="OrtTensor.IsNumSharpMemory"/>).</param>
        /// <param name="ownsValue"><c>true</c> to dispose the handle (and with it the OrtValue) once the view exists.</param>
        /// <returns>A view sharing the buffer, holding its own reference on it (an empty array for an empty tensor).</returns>
        /// <exception cref="ObjectDisposedException">The handle has been disposed, or the buffer is already released.</exception>
        internal static unsafe NDArray AliasExportedValue(OrtTensor owner, bool ownsValue)
        {
            // A temporary reference: the liveness check, and what keeps the buffer alive until the view holds its own
            // (the NDArray constructor takes the view's reference without checking that it succeeded).
            IArraySlice buffer = owner.Slice;
            if (owner.IsDisposed || !buffer.TryAddRef())
                throw new ObjectDisposedException(nameof(OrtTensor), "the OrtTensor handle that owns this OrtValue has been disposed; its NumSharp buffer is no longer pinned.");
            try
            {
                NPTypeCode tc = FromTensorElementType(owner.ElementType);
                Shape shape = ShapeFrom(owner.ShapeNoCopy);
                long count = shape.Size;
                NDArray view;
                if (count == 0)
                {
                    // Nothing to share — an empty array of the right dtype/shape IS the view.
                    view = new NDArray(tc, shape, fillZeros: false);
                }
                else
                {
                    if (!owner.SourceWriteable)
                        shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                    IArraySlice alias = WrapShared(tc, owner.Data, count, buffer);
                    view = new NDArray(new UnmanagedStorage(alias, shape).Alias());   // takes the view's reference, forwarded to the buffer
                }

                // Ownership transfers only now that the view exists. The view's own reference keeps the buffer (for a
                // copy handle: the temporary's buffer) alive past the handle.
                if (ownsValue)
                    owner.Dispose();
                return view;
            }
            finally
            {
                buffer.Release();   // the temporary reference; the view's own keeps the count above zero
                GC.KeepAlive(owner);
            }
        }

        /// <summary>
        ///     The copy <see cref="ToNDArray(OrtValue)"/> returns for a value this package made over a NumSharp buffer: read
        ///     from the buffer itself, under a reference on it, with the byte count in a <see cref="long"/>.
        /// </summary>
        /// <remarks>
        ///     The Run tier calls this too, for a model output that is a fed input passed through (see
        ///     <see cref="AliasExportedValue"/>): the copy then needs no span, so it works past 2 GB as well.
        /// </remarks>
        /// <param name="owner">The value's export handle; it must be over a NumSharp buffer (<see cref="OrtTensor.IsNumSharpMemory"/>).</param>
        /// <returns>A new owning C-contiguous array of the crossed dtype holding the tensor's elements.</returns>
        /// <exception cref="ObjectDisposedException">The handle has been disposed, or the buffer is already released.</exception>
        internal static unsafe NDArray CopyExportedValue(OrtTensor owner)
        {
            IArraySlice buffer = owner.Slice;
            if (owner.IsDisposed || !buffer.TryAddRef())
                throw new ObjectDisposedException(nameof(OrtTensor), "the OrtTensor handle that owns this OrtValue has been disposed; its NumSharp buffer is no longer pinned.");
            try
            {
                NPTypeCode tc = FromTensorElementType(owner.ElementType);
                Shape shape = ShapeFrom(owner.ShapeNoCopy);
                var nd = new NDArray(tc, shape, fillZeros: false);
                long nbytes = shape.Size * tc.SizeOf();
                if (nbytes > 0)
                    Buffer.MemoryCopy(owner.Data, nd.Storage.InternalArray.Address, nbytes, nbytes);
                return nd;
            }
            finally
            {
                buffer.Release();
                GC.KeepAlive(owner);
            }
        }

        /// <summary>
        ///     Copy an ORT <see cref="DenseTensor{T}"/> (the legacy surface — <c>DisposableNamedOnnxValue.AsTensor&lt;T&gt;()</c>,
        ///     or a tensor you filled yourself) into a fresh owning C-contiguous <see cref="NDArray"/>. A
        ///     column-major (<c>IsReversedStride</c>) tensor is transposed back to its logical order on the way.
        ///     <see cref="Float16"/> reads back as <see cref="NPTypeCode.Half"/>.
        /// </summary>
        /// <remarks>
        ///     Any size a <see cref="DenseTensor{T}"/> can hold crosses (up to <see cref="int.MaxValue"/> elements, so up to
        ///     16 GB of bytes): the copy counts bytes in a <see cref="long"/>. It counted them in an <c>int</c> through
        ///     <see cref="MemoryMarshal.AsBytes{T}(Span{T})"/>, which threw <see cref="OverflowException"/> for every tensor
        ///     past 2 GB — including the ones <see cref="AsDenseTensor{T}"/> and <see cref="ToDenseTensor{T}"/> produce.
        ///     The buffer is pinned for the copy, so a NumSharp buffer behind an <see cref="AsDenseTensor{T}"/> handle stays
        ///     alive while it is read (and is refused once released) and a managed array cannot move.
        /// </remarks>
        /// <typeparam name="T">The tensor's element type: bool, the eight integers, <see cref="Float16"/>, float or double.</typeparam>
        /// <param name="tensor">The tensor to copy; row- or column-major.</param>
        /// <returns>A new owning C-contiguous array holding the tensor's elements in logical (row-major) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no NumSharp dtype (BFloat16, string, ...).</exception>
        /// <exception cref="ObjectDisposedException">The tensor's memory is a NumSharp buffer whose <see cref="OrtTensor{T}"/> handle has been disposed.</exception>
        public static unsafe NDArray ToNDArray<T>(this DenseTensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            NPTypeCode tc = TypeCodeOf<T>();
            long[] dims = ToLongDims(tensor.Dimensions);
            long count = tensor.Length;
            bool reversedStride = tensor.IsReversedStride && dims.Length >= 2;

            // Column-major memory is the C-order layout of the REVERSED shape: copy it as such, transpose back (a view)
            // and take a C-order copy of the logical array.
            long[] copied = dims;
            if (reversedStride)
            {
                copied = new long[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    copied[i] = dims[dims.Length - 1 - i];
            }

            if (count == 0)
            {
                // Nothing to read, and nothing to pin: ORT's own memory manager (a legacy output's OrtValueTensor) refuses
                // to pin element 0 of an empty tensor. The span is still asked for, so a NumSharp buffer behind a disposed
                // AsDenseTensor handle is refused exactly as a non-empty one is.
                _ = tensor.Buffer.Span;
                var empty = new NDArray(tc, ShapeFrom(copied), fillZeros: false);
                if (!reversedStride)
                    return empty;
                using (empty)
                using (NDArray transposedEmpty = empty.transpose())
                    return transposedEmpty.copy();
            }

            // Pin before allocating, so a released NumSharp buffer is refused with nothing allocated.
            MemoryHandle pin = tensor.Buffer.Pin();
            try
            {
                var flat = new NDArray(tc, ShapeFrom(copied), fillZeros: false);
                CopyPinnedTensor(pin, flat, count * sizeof(T));
                if (!reversedStride)
                    return flat;
                using (flat)
                using (NDArray transposed = flat.transpose())
                    return transposed.copy();
            }
            finally
            {
                pin.Dispose();
                // The pin does not always keep the memory alive: ORT's memory manager for a legacy output
                // (OrtValueTensor<T>) hands out a bare pointer, and its native buffer lives only as long as the tensor
                // (tensor → Memory → manager → OrtValue, whose finalizer frees it). Without this the tensor's last use was
                // the Pin() above, so a tensor the caller does not otherwise hold (`results.First().AsTensor<T>()` of a
                // dropped results list) could be collected while the copy read it.
                GC.KeepAlive(tensor);
            }
        }

        /// <summary>
        ///     Copies <paramref name="nbytes"/> bytes from a pinned tensor buffer into a freshly allocated array — the one
        ///     copy both layouts of <see cref="ToNDArray{T}(DenseTensor{T})"/> share, with the byte count in a
        ///     <see cref="long"/> so a tensor past 2 GB crosses.
        /// </summary>
        /// <param name="pin">The pinned source buffer; the caller keeps the pin for the duration of the call.</param>
        /// <param name="destination">A fresh C-contiguous array of at least <paramref name="nbytes"/> bytes.</param>
        /// <param name="nbytes">The number of bytes to copy (the tensor's element count × element size).</param>
        private static unsafe void CopyPinnedTensor(MemoryHandle pin, NDArray destination, long nbytes) =>
            Buffer.MemoryCopy(pin.Pointer, destination.Storage.InternalArray.Address, nbytes, nbytes);

        /// <summary>
        ///     View an ORT <see cref="DenseTensor{T}"/> as a NumSharp array over the tensor's own
        ///     <see cref="DenseTensor{T}.Buffer"/> — zero-copy. The buffer is pinned for the view's lifetime (the pin is
        ///     released when the last NumSharp view over it dies), and the view keeps the tensor itself reachable, so neither
        ///     a managed array nor a native buffer can be collected under it. Column-major tensors come back as
        ///     Fortran-ordered views — no transpose copy.
        /// </summary>
        /// <remarks>
        ///     <para>What the pin guarantees depends on who owns the memory. A managed array (a tensor you built) is pinned, so it
        ///     neither moves nor dies. A NumSharp buffer behind <see cref="AsDenseTensor{T}"/> is not pinned at all: the view
        ///     is an ordinary NumSharp alias of that buffer, holding a counted reference on it like any view, so it outlives the
        ///     handle and the source and is released through NumSharp's own rules (no import lease, so <see cref="LiveImports"/>
        ///     does not count it); it is read-only when the exported array was. An ORT output's memory (the legacy
        ///     <c>session.Run</c> results, <c>DisposableNamedOnnxValue.AsTensor&lt;T&gt;()</c>) is a bare native pointer: the
        ///     view keeps the tensor — and through it the <see cref="OrtValue"/> — reachable, so a results collection that is
        ///     merely DROPPED is not finalized under the view, but disposing it explicitly while a view lives frees the memory
        ///     under the view; keep the results alive, or copy with <see cref="ToNDArray{T}(DenseTensor{T})"/>. That is the
        ///     rule <see cref="AsNDArray(OrtValue, bool, OnnxCodecMode)"/> with <c>ownsValue: false</c> follows too.</para>
        ///     <para><b>Modes.</b> A dense tensor can always be viewed, so <see cref="OnnxCodecMode.Auto"/> and
        ///     <see cref="OnnxCodecMode.View"/> do the same here; <see cref="OnnxCodecMode.Copy"/> returns what
        ///     <see cref="ToNDArray{T}(DenseTensor{T})"/> returns.</para>
        ///     <para>The view reports no memory pressure to the GC: it never owns the tensor's memory.</para>
        /// </remarks>
        /// <typeparam name="T">The tensor's element type: bool, the eight integers, <see cref="Float16"/>, float or double.</typeparam>
        /// <param name="tensor">The tensor to view; row- or column-major.</param>
        /// <param name="mode">View (also the meaning of the default here) or an independent copy; see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A NumSharp view over the tensor's buffer (an empty array for an empty tensor, which has nothing to share), or a copy under <see cref="OnnxCodecMode.Copy"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no NumSharp dtype (BFloat16, string, ...).</exception>
        /// <exception cref="ObjectDisposedException">The tensor's memory is a NumSharp buffer whose <see cref="OrtTensor{T}"/> handle has been disposed.</exception>
        public static unsafe NDArray AsNDArray<T>(this DenseTensor<T> tensor, OnnxCodecMode mode = OnnxCodecMode.Auto) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            EnsureDefined(mode);
            if (mode == OnnxCodecMode.Copy)
                return tensor.ToNDArray();
            NPTypeCode tc = TypeCodeOf<T>();
            long[] dims = ToLongDims(tensor.Dimensions);
            Shape shape = ShapeFrom(dims);
            long count = tensor.Length;
            if (count == 0)
            {
                // Nothing to share and nothing to pin (ORT's legacy memory manager refuses to pin element 0 of an empty
                // tensor), but the span is still asked for: a NumSharp buffer behind a disposed AsDenseTensor handle is
                // refused for an empty tensor exactly as for a non-empty one, as ToNDArray refuses it.
                _ = tensor.Buffer.Span;
                return new NDArray(tc, shape, fillZeros: false);
            }

            // A tensor over NumSharp's own memory (an AsDenseTensor handle's) is viewed as an ordinary NumSharp alias of
            // that buffer, not leased like foreign memory.
            if (MemoryMarshal.TryGetMemoryManager<T, UnmanagedMemoryManager<T>>(tensor.Buffer, out UnmanagedMemoryManager<T> numsharp, out int start, out _))
                return AliasNumSharpTensor(numsharp, start, tc, dims, shape, count, tensor.IsReversedStride);

            MemoryHandle handle = tensor.Buffer.Pin();
            // The lease roots the TENSOR as well as releasing the pin. ORT's memory manager for a legacy output
            // (OrtValueTensor<T>) hands out a bare pointer whose pin keeps nothing alive, so once the caller dropped the
            // results, the OrtValue's finalizer freed the native buffer under the view and the next run's output showed
            // through it (20 of 20 rounds in a probe). The tensor holds its Memory, the Memory its manager, the manager
            // its OrtValue: keeping the tensor reachable keeps the buffer valid until the last view dies.
            // No memory pressure: the view owns none of these bytes. A managed array is already counted by the GC, and a
            // legacy output's native buffer belongs to its results collection.
            var lease = new ImportLease(() =>
            {
                handle.Dispose();
                GC.KeepAlive(tensor);
            }, ownedBytes: 0);
            try
            {
                IArraySlice slice = WrapExternal(tc, handle.Pointer, count, lease.Release);
                UnmanagedStorage storage = tensor.IsReversedStride && dims.Length > 1
                    // the strided (Fortran) shape's logical layout differs from the flat physical span, so alias a flat
                    // storage with the strided shape — the pattern NumSharp's own slicing and the pythonnet bridge use
                    ? new UnmanagedStorage(slice, Shape.Vector(count)).Alias(new Shape(dims, FortranStrides(dims), offset: 0, bufferSize: count))
                    : new UnmanagedStorage(slice, shape).Alias();
                return new NDArray(storage);
            }
            catch
            {
                lease.Release();
                throw;
            }
        }

        /// <summary>
        ///     The view <see cref="AsNDArray{T}(DenseTensor{T}, OnnxCodecMode)"/> returns for a tensor over NumSharp's own
        ///     memory (an <see cref="AsDenseTensor{T}"/> handle's): an alias of the buffer whose reference counting is the
        ///     buffer's own (<c>ArraySlice.WrapShared</c>, the alias NumSharp builds for <c>np.real</c> or <c>view(dtype)</c>),
        ///     so the view keeps the buffer alive exactly as any NumSharp view does, whatever happens to the handle and the source.
        /// </summary>
        /// <remarks>
        ///     <para>It used to be leased like foreign memory: the view pinned the tensor's Memory and the pin's release ran
        ///     when the view's memory block was released — from the block's FINALIZER when the view was collected. The pin's
        ///     release then claimed the eager free at zero from a finalizer, which frees a buffer that an uncounted alias
        ///     (a bare <c>IArraySlice</c> of the source) may still be reading; and the import counter treated NumSharp's own
        ///     memory as foreign. An alias forwards its finalizer path to <see cref="IArraySlice.Abandon"/> like every view.</para>
        ///     <para>A tensor over a SLICE of the handle's Memory (<c>new DenseTensor&lt;T&gt;(h.Memory.Slice(k), …)</c>) is
        ///     an alias of that window: <paramref name="start"/> is the slice's offset into the manager's elements.</para>
        ///     <para>The view is read-only when the exported array was (<see cref="UnmanagedMemoryManager{T}.Writeable"/>):
        ///     the round trip array → tensor → array used to return a writeable view of a read-only array.</para>
        /// </remarks>
        /// <typeparam name="T">The tensor's element type (ORT's type for the dtype: Float16 for Half, ushort for Char).</typeparam>
        /// <param name="manager">The NumSharp memory manager behind the tensor's Memory.</param>
        /// <param name="start">The element offset of the tensor's Memory within the manager's memory.</param>
        /// <param name="tc">The dtype the view carries (<see cref="TypeCodeOf{T}"/>).</param>
        /// <param name="dims">The tensor's dimensions.</param>
        /// <param name="shape">The tensor's row-major shape.</param>
        /// <param name="count">The tensor's element count (non-zero).</param>
        /// <param name="reversedStride">Whether the tensor is column-major (<c>IsReversedStride</c>).</param>
        /// <returns>A view sharing the buffer, holding its own reference on it.</returns>
        /// <exception cref="ObjectDisposedException">The tensor's handle has been disposed, or the buffer is already released.</exception>
        private static unsafe NDArray AliasNumSharpTensor<T>(UnmanagedMemoryManager<T> manager, int start, NPTypeCode tc, long[] dims, Shape shape, long count, bool reversedStride)
            where T : unmanaged
        {
            // The same refusal the tensor's Memory gives once its handle is disposed (Pin and Span throw then): a disposed
            // handle refuses new access, and a view is new access.
            if (manager.IsDisposed)
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>), "the NumSharp buffer behind this Memory<T> has been released (its OrtTensor handle was disposed).");

            // A temporary reference: the liveness check (it fails once every reference is gone) and what keeps the buffer
            // alive until the view holds its own. The NDArray constructor takes the view's reference without checking that
            // it succeeded, so the check must come first.
            IArraySlice owner = manager.Owner;
            if (!owner.TryAddRef())
                throw new ObjectDisposedException(nameof(UnmanagedMemoryManager<T>), "the NumSharp buffer behind this Memory<T> has already been released.");
            try
            {
                IArraySlice slice = WrapShared(tc, manager.Pointer + start, count, owner);
                // The base storage carries the read-only flag; both Alias forms inherit it (a view of a read-only array
                // stays read-only).
                Shape flat = Shape.Vector(count);
                if (!manager.Writeable)
                {
                    shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                    flat = flat.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                }
                UnmanagedStorage storage = reversedStride && dims.Length > 1
                    // the Fortran view of a flat storage, as for foreign memory below
                    ? new UnmanagedStorage(slice, flat).Alias(new Shape(dims, FortranStrides(dims), offset: 0, bufferSize: count))
                    : new UnmanagedStorage(slice, shape).Alias();
                return new NDArray(storage);   // takes the view's reference, forwarded to the buffer
            }
            finally
            {
                // The temporary reference. Once the view exists its own reference keeps the count above zero, so this never
                // frees; if building the view failed, it gives the buffer back like any deterministic release.
                owner.Release();
            }
        }

        /// <summary>
        ///     Copy the tensor inside a legacy <see cref="NamedOnnxValue"/> / <c>DisposableNamedOnnxValue</c>
        ///     (the <c>session.Run(IReadOnlyCollection&lt;NamedOnnxValue&gt;)</c> result shape) into a fresh owning
        ///     <see cref="NDArray"/>, dispatching on the tensor's element type at runtime so no
        ///     <c>AsTensor&lt;T&gt;()</c> guess is needed.
        /// </summary>
        /// <remarks>
        ///     Every element type the generic <see cref="ToNDArray{T}(DenseTensor{T})"/> accepts is dispatched, including
        ///     the two a caller's own tensor can carry but an ORT output never does: <see cref="char"/> (read back as
        ///     <see cref="NPTypeCode.Char"/>) and <see cref="Half"/> (as <see cref="NPTypeCode.Half"/>, like ORT's
        ///     <see cref="Float16"/>). They used to be refused here although the generic verb took them.
        /// </remarks>
        /// <param name="value">A named value that holds a tensor.</param>
        /// <returns>A new owning C-contiguous array holding the tensor's elements in logical (row-major) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="NotSupportedException">The value is a sequence / map, or a BFloat16 tensor, or a tensor of a type with no NumSharp dtype.</exception>
        /// <exception cref="InvalidOperationException">The named value holds no value.</exception>
        public static NDArray ToNDArray(this NamedOnnxValue value)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            if (value.ValueType != OnnxValueType.ONNX_TYPE_TENSOR)
                throw new NotSupportedException(
                    $"NamedOnnxValue '{value.Name}' is {value.ValueType}, not a tensor; read sequences with AsEnumerable<T>() and maps with AsDictionary<K, V>().");

            switch (value.Value)
            {
                case DenseTensor<float> t: return t.ToNDArray();
                case DenseTensor<double> t: return t.ToNDArray();
                case DenseTensor<sbyte> t: return t.ToNDArray();
                case DenseTensor<byte> t: return t.ToNDArray();
                case DenseTensor<short> t: return t.ToNDArray();
                case DenseTensor<ushort> t: return t.ToNDArray();
                case DenseTensor<int> t: return t.ToNDArray();
                case DenseTensor<uint> t: return t.ToNDArray();
                case DenseTensor<long> t: return t.ToNDArray();
                case DenseTensor<ulong> t: return t.ToNDArray();
                case DenseTensor<bool> t: return t.ToNDArray();
                case DenseTensor<Float16> t: return t.ToNDArray();
                case DenseTensor<Half> t: return t.ToNDArray();
                case DenseTensor<char> t: return t.ToNDArray();
                case DenseTensor<BFloat16>: throw new NotSupportedException(UnsupportedImportMessage(TensorElementType.BFloat16));
                case Tensor<string> t: return TextTensorToNDArray(t);   // a new U array, in the tensor's logical order
                // a non-dense Tensor<T> (a caller's own subclass): densify through ORT's own copy
                case Tensor<float> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<double> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<sbyte> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<byte> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<short> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<ushort> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<int> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<uint> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<long> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<ulong> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<bool> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<Float16> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<Half> t: return t.ToDenseTensor().ToNDArray();
                case Tensor<char> t: return t.ToDenseTensor().ToNDArray();
                case null: throw new InvalidOperationException($"NamedOnnxValue '{value.Name}' holds no value.");
                default:
                    throw new NotSupportedException($"NamedOnnxValue '{value.Name}' holds a {value.Value.GetType()}, which has no NumSharp dtype.");
            }
        }

        private static long[] ToLongDims(ReadOnlySpan<int> dims)
        {
            var result = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++)
                result[i] = dims[i];
            return result;
        }
    }
}
