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
        ///     Char), the shape is the tensor's row-major shape (0-d and empty tensors included). The tensor
        ///     must be a dense tensor in CPU-addressable memory: sequences / maps / sparse tensors and
        ///     device-resident (CUDA / DirectML) outputs are refused with a message naming the alternative.</para>
        /// </summary>
        /// <remarks>
        ///     Every refusal is decided before the result is allocated. A tensor of more than 2 GB is refused: ORT's
        ///     managed API exposes a tensor's memory only as a <see cref="Span{T}"/> of bytes, which cannot cover it (see
        ///     <see cref="ImportTooLargeMessage"/> for the route that works at any size — ORT writing into a pre-allocated
        ///     NumSharp output).
        /// </remarks>
        /// <param name="value">A dense tensor OrtValue in CPU-accessible memory; it stays the caller's to dispose.</param>
        /// <returns>A new owning C-contiguous array holding a copy of the tensor's elements.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a dense tensor; a String / BFloat16 element type; or a tensor of more than 2 GB.</exception>
        /// <exception cref="ArgumentException">The tensor's shape has a negative (symbolic) dimension.</exception>
        /// <exception cref="InvalidOperationException">The tensor lives in device memory the CPU cannot address, or ORT reports a byte count its shape contradicts.</exception>
        public static unsafe NDArray ToNDArray(this OrtValue value)
        {
            EnsureTensor(value, nameof(ToNDArray));
            OrtTensorTypeAndShapeInfo info = value.GetTensorTypeAndShape();
            if (info.IsString)
                throw new NotSupportedException(UnsupportedImportMessage(TensorElementType.String));
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
            return nd;
        }

        /// <summary>
        ///     View an ORT tensor as a NumSharp array over ORT's OWN buffer — zero-copy, mutations visible both
        ///     ways (<c>OrtValue.GetTensorMutableDataAsSpan</c> and the NDArray read the same bytes). The view is a
        ///     genuine NumSharp array: slice it, reduce it, feed it to kernels, or hand it straight back to ORT
        ///     with <see cref="AsOrtValue"/>.
        ///
        ///     <para><b>Lifetime.</b> By default (<paramref name="ownsValue"/> = <c>false</c>) the OrtValue stays
        ///     the caller's: the view holds a strong reference so the value cannot be finalized underneath it,
        ///     but the caller must not <c>Dispose</c> the OrtValue (or the collection <c>session.Run</c> returned)
        ///     while any view — or any slice derived from it — is still in use, exactly the rule ORT documents for
        ///     <c>GetTensorDataAsSpan</c>. With <paramref name="ownsValue"/> = <c>true</c> the view TAKES OWNERSHIP:
        ///     the OrtValue is disposed when the last NumSharp view over the memory dies (disposal or collection),
        ///     and the caller must not dispose it again — the safe way to keep an output alive past
        ///     <c>session.Run</c> without a copy (what <c>InferenceSessionExtensions.Run(..., zeroCopyOutputs: true)</c>
        ///     does).</para>
        ///
        ///     <para>The view does not own its data (<c>flags.owndata == False</c>, like <c>np.frombuffer</c>): a
        ///     size-changing <c>ndarray.resize</c> refuses instead of detaching from ORT's memory. CPU-addressable
        ///     tensors only; an empty tensor yields an empty array (nothing to share).</para>
        /// </summary>
        /// <remarks>
        ///     A tensor of more than 2 GB is refused, whatever its exact size: ORT's managed API hands out a tensor's memory
        ///     only through a <see cref="Span{T}"/> of bytes, whose length it computes with an unchecked <c>int</c> cast, so
        ///     past 2 GB the span either fails to construct or comes back truncated (see <see cref="ImportTooLargeMessage"/>).
        ///     Every refusal leaves <paramref name="value"/> untouched and still the caller's, even with
        ///     <paramref name="ownsValue"/> = <c>true</c>: ownership transfers only with a returned view.
        /// </remarks>
        /// <param name="value">A dense tensor OrtValue in CPU-accessible memory.</param>
        /// <param name="ownsValue"><c>true</c> to dispose the OrtValue when the last view dies; <c>false</c> (default) to leave it the caller's.</param>
        /// <returns>A NumSharp view over ORT's buffer (an empty array for an empty tensor, which has nothing to share).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a dense tensor; a String / BFloat16 element type; or a tensor of more than 2 GB.</exception>
        /// <exception cref="ArgumentException">The tensor's shape has a negative (symbolic) dimension.</exception>
        /// <exception cref="InvalidOperationException">The tensor lives in device memory the CPU cannot address.</exception>
        public static unsafe NDArray AsNDArray(this OrtValue value, bool ownsValue = false)
        {
            EnsureTensor(value, nameof(AsNDArray));
            OrtTensorTypeAndShapeInfo info = value.GetTensorTypeAndShape();
            if (info.IsString)
                throw new NotSupportedException(UnsupportedImportMessage(TensorElementType.String));
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
            Action release = ownsValue ? value.Dispose : () => GC.KeepAlive(value);
            var lease = new ImportLease(release, raw.Length);
            try
            {
                IArraySlice slice = WrapExternal(tc, data, count, lease.Release);
                // Alias() so the storage reports VIEW semantics (numpy: flags.owndata == False for foreign
                // buffers): ndarray.resize then refuses to reallocate instead of silently detaching from ORT's memory.
                return new NDArray(new UnmanagedStorage(slice, shape).Alias());
            }
            catch
            {
                lease.Release();
                throw;
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
        ///     What the pin guarantees depends on who owns the memory. A managed array (a tensor you built) is pinned, so it
        ///     neither moves nor dies. A NumSharp buffer behind <see cref="AsDenseTensor{T}"/> takes a reference of its own,
        ///     so the view outlives the handle and the source. An ORT output's memory (the legacy <c>session.Run</c> results,
        ///     <c>DisposableNamedOnnxValue.AsTensor&lt;T&gt;()</c>) is a bare native pointer: the view keeps the tensor — and
        ///     through it the <see cref="OrtValue"/> — reachable, so a results collection that is merely DROPPED is not
        ///     finalized under the view, but disposing it explicitly while a view lives frees the memory under the view; keep
        ///     the results alive, or copy with <see cref="ToNDArray{T}(DenseTensor{T})"/>. That is the rule
        ///     <see cref="AsNDArray(OrtValue, bool)"/> with <c>ownsValue: false</c> follows too.
        /// </remarks>
        /// <typeparam name="T">The tensor's element type: bool, the eight integers, <see cref="Float16"/>, float or double.</typeparam>
        /// <param name="tensor">The tensor to view; row- or column-major.</param>
        /// <returns>A NumSharp view over the tensor's buffer (an empty array for an empty tensor, which has nothing to share).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no NumSharp dtype (BFloat16, string, ...).</exception>
        /// <exception cref="ObjectDisposedException">The tensor's memory is a NumSharp buffer whose <see cref="OrtTensor{T}"/> handle has been disposed.</exception>
        public static unsafe NDArray AsNDArray<T>(this DenseTensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
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

            MemoryHandle handle = tensor.Buffer.Pin();
            // The lease roots the TENSOR as well as releasing the pin. ORT's memory manager for a legacy output
            // (OrtValueTensor<T>) hands out a bare pointer whose pin keeps nothing alive, so once the caller dropped the
            // results, the OrtValue's finalizer freed the native buffer under the view and the next run's output showed
            // through it (20 of 20 rounds in a probe). The tensor holds its Memory, the Memory its manager, the manager
            // its OrtValue: keeping the tensor reachable keeps the buffer valid until the last view dies.
            var lease = new ImportLease(() =>
            {
                handle.Dispose();
                GC.KeepAlive(tensor);
            }, count * sizeof(T));
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
        ///     Copy the tensor inside a legacy <see cref="NamedOnnxValue"/> / <c>DisposableNamedOnnxValue</c>
        ///     (the <c>session.Run(IReadOnlyCollection&lt;NamedOnnxValue&gt;)</c> result shape) into a fresh owning
        ///     <see cref="NDArray"/>, dispatching on the tensor's element type at runtime so no
        ///     <c>AsTensor&lt;T&gt;()</c> guess is needed.
        /// </summary>
        /// <exception cref="NotSupportedException">The value is a sequence / map, or a string / BFloat16 tensor.</exception>
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
                case DenseTensor<BFloat16>: throw new NotSupportedException(UnsupportedImportMessage(TensorElementType.BFloat16));
                case Tensor<string>: throw new NotSupportedException(UnsupportedImportMessage(TensorElementType.String));
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
