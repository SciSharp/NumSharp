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
        /// <exception cref="NotSupportedException">Not a dense tensor, or a String / BFloat16 element type.</exception>
        /// <exception cref="InvalidOperationException">The tensor lives in device memory the CPU cannot address.</exception>
        public static unsafe NDArray ToNDArray(this OrtValue value)
        {
            EnsureTensor(value, nameof(ToNDArray));
            OrtTensorTypeAndShapeInfo info = value.GetTensorTypeAndShape();
            if (info.IsString)
                throw new NotSupportedException(UnsupportedImportMessage(TensorElementType.String));
            NPTypeCode tc = FromTensorElementType(info.ElementDataType);
            Shape shape = ShapeFrom(info.Shape);
            EnsureCpuAccessible(value, nameof(ToNDArray));

            var nd = new NDArray(tc, shape, fillZeros: false);
            if (info.ElementCount > 0)
            {
                Span<byte> src = value.GetTensorMutableRawData();
                long nbytes = shape.Size * nd.dtypesize;
                if (src.Length != nbytes)
                    throw new InvalidOperationException($"ORT reports {src.Length} tensor bytes but shape {shape} × itemsize {nd.dtypesize} needs {nbytes}.");
                src.CopyTo(new Span<byte>(nd.Storage.InternalArray.Address, src.Length));
            }

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
        /// <param name="value">A dense tensor OrtValue in CPU-accessible memory.</param>
        /// <param name="ownsValue"><c>true</c> to dispose the OrtValue when the last view dies; <c>false</c> (default) to leave it the caller's.</param>
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
        public static unsafe NDArray ToNDArray<T>(this DenseTensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            NPTypeCode tc = TypeCodeOf<T>();
            long[] dims = ToLongDims(tensor.Dimensions);
            long count = tensor.Length;
            ReadOnlySpan<byte> src = MemoryMarshal.AsBytes(tensor.Buffer.Span.Slice(0, (int)count));

            if (!tensor.IsReversedStride || dims.Length < 2)
            {
                var nd = new NDArray(tc, ShapeFrom(dims), fillZeros: false);
                if (count > 0)
                    src.CopyTo(new Span<byte>(nd.Storage.InternalArray.Address, src.Length));
                return nd;
            }

            // Column-major memory is the C-order layout of the REVERSED shape: copy it as such, transpose back
            // (a view) and take a C-order copy of the logical array.
            var reversed = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++)
                reversed[i] = dims[dims.Length - 1 - i];
            using var tmp = new NDArray(tc, ShapeFrom(reversed), fillZeros: false);
            if (count > 0)
                src.CopyTo(new Span<byte>(tmp.Storage.InternalArray.Address, src.Length));
            using NDArray transposed = tmp.transpose();
            return transposed.copy();
        }

        /// <summary>
        ///     View an ORT <see cref="DenseTensor{T}"/> as a NumSharp array over the tensor's own
        ///     <see cref="DenseTensor{T}.Buffer"/> — zero-copy. A managed buffer is pinned for the view's lifetime
        ///     (the pin is released when the last NumSharp view over it dies); a buffer that already fronts native
        ///     memory (an ORT output's tensor, or a NumSharp buffer from <see cref="AsDenseTensor{T}"/>) is simply
        ///     addressed. Column-major tensors come back as Fortran-ordered views — no transpose copy.
        /// </summary>
        public static unsafe NDArray AsNDArray<T>(this DenseTensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            NPTypeCode tc = TypeCodeOf<T>();
            long[] dims = ToLongDims(tensor.Dimensions);
            Shape shape = ShapeFrom(dims);
            long count = tensor.Length;
            if (count == 0)
                return new NDArray(tc, shape, fillZeros: false);

            MemoryHandle handle = tensor.Buffer.Pin();
            var lease = new ImportLease(handle.Dispose, count * sizeof(T));
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
