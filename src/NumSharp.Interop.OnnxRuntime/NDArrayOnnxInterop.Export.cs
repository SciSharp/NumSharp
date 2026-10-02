using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.OnnxRuntime
{
    public static partial class NDArrayOnnxInterop
    {
        // ===========================  NumSharp  ->  ONNX Runtime  ===========================

        /// <summary>
        ///     Wrap a NumSharp array as an ONNX Runtime <see cref="OrtValue"/> tensor that SHARES its unmanaged
        ///     buffer — no copy, no per-element fill loop. The tensor is built over the array's own pointer
        ///     (<c>OrtValue.CreateTensorValueWithData</c>), so ORT reads (and, for a pre-allocated output,
        ///     writes) NumSharp's memory directly.
        ///
        ///     <para><b>Layout:</b> ORT tensors are dense row-major and carry no strides, so only a
        ///     <b>C-contiguous</b> array can be shared. A sliced / transposed / negative-stride / broadcast /
        ///     Fortran-order view throws — materialize it first (<c>np.ascontiguousarray(nd)</c>,
        ///     <c>nd.copy()</c>) or use <see cref="ToOrtValue"/>, which copies any layout. Contiguous views at a
        ///     non-zero offset (<c>np.split</c> / <c>np.unstack</c> children) share exactly their window. Scalars
        ///     become 0-d tensors; empty arrays become empty tensors.</para>
        ///
        ///     <para><b>Dtypes:</b> bool, the eight integers, float and double cross as themselves;
        ///     <see cref="NPTypeCode.Half"/> crosses as <see cref="TensorElementType.Float16"/> (bit-identical,
        ///     NaN payloads included); <see cref="NPTypeCode.Char"/> as <see cref="TensorElementType.UInt16"/>.
        ///     <see cref="NPTypeCode.Decimal"/> has no ONNX type: it is converted to a float64 temporary the
        ///     returned handle owns (mutations by ORT land in the temporary). <see cref="NPTypeCode.Complex"/>
        ///     throws.</para>
        ///
        ///     <para><b>Lifetime:</b> the handle takes its own atomic reference on the NumSharp buffer, so the
        ///     memory survives even if <paramref name="source"/> is disposed or collected while ORT holds the
        ///     tensor; disposing the handle disposes the <see cref="OrtValue"/> and releases the reference.
        ///     <b>Keep the handle alive across <c>session.Run</c></b> (a <c>using</c> around the run is the
        ///     idiom). While exported, <c>ndarray.resize(refcheck: true)</c> on the source refuses to reallocate,
        ///     exactly as NumPy refuses for a referenced buffer.</para>
        /// </summary>
        /// <param name="source">The NumSharp array to share.</param>
        /// <returns>A disposable handle owning the <see cref="OrtValue"/> (<see cref="OrtTensor.Value"/>) and the buffer pin.</returns>
        /// <exception cref="InvalidOperationException">The array is not C-contiguous.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX tensor element type (Complex).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static OrtTensor AsOrtValue(this NDArray source) => AsOrtValueCore(source, ownsSource: false);

        /// <summary>
        ///     <see cref="AsOrtValue"/> whose returned handle also owns <paramref name="source"/> when
        ///     <paramref name="ownsSource"/> is <c>true</c> (a temporary the caller wants disposed with the
        ///     handle). Ownership transfers only on success; on failure the caller still owns its array.
        /// </summary>
        /// <param name="source">The array to share (a Decimal array is converted to a float64 temporary the handle owns).</param>
        /// <param name="ownsSource"><c>true</c> when <paramref name="source"/> is a temporary the handle should dispose.</param>
        /// <returns>A handle owning the <see cref="OrtValue"/>, the buffer pin and any temporary.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="InvalidOperationException">The array (after any Decimal conversion) is not C-contiguous.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX tensor element type (Complex).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static unsafe OrtTensor AsOrtValueCore(NDArray source, bool ownsSource)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            // Decimal has no ONNX element type (16-byte, non-IEEE), so no zero-copy view is possible; convert it
            // to float64 — the exact conversion this package's own guidance recommends (astype(Double), lossy
            // beyond ~16 significant digits). The converted array is a TEMPORARY nobody but this export can ever
            // dispose, so the handle takes ownership of it and disposes it when the handle dies. The low-level
            // dtype maps stay honest and still refuse Decimal; only the array-producing verbs convert.
            // The conversion is the first read of the source, so it runs under a reference on the source's buffer
            // (ConvertPinned): a bare astype read a RELEASED buffer silently, and the Pin below then succeeded on
            // the fresh conversion, so the tensor carried whatever the pool had put in the freed memory.
            NDArray converted = null;
            NDArray fed = source;
            if (source.typecode == NPTypeCode.Decimal)
                fed = converted = ConvertPinned(source, NPTypeCode.Double);

            try
            {
                TensorElementType elementType = ToTensorElementType(fed.typecode);
                Shape shape = fed.Shape;
                if (!shape.IsContiguous)
                    throw new InvalidOperationException(NotCContiguousMessage);

                long[] dims = LongDims(shape);
                IArraySlice slice = Pin(fed, out byte* data, out long nbytes);
                OrtValue value = null;
                try
                {
                    // Does NOT own or free the memory; the handle's ARC reference guarantees validity for as long
                    // as the OrtValue exists. Row-major, no strides — which is why C-contiguity was required above.
                    value = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, elementType, dims, (IntPtr)data, nbytes);
                    var handle = new OrtTensor(value, fed, slice, ownsSource: converted is not null || ownsSource, elementType, dims);
                    if (converted is not null && ownsSource)
                        source.Dispose();   // the caller's temporary is superseded by the conversion the handle now owns
                    converted = null;       // ownership moved to the handle
                    return handle;
                }
                catch
                {
                    value?.Dispose();
                    slice.Release();
                    throw;
                }
            }
            finally
            {
                converted?.Dispose();   // only reached on failure — the success path nulled it
            }
        }

        /// <summary>
        ///     Copy a NumSharp array into an independent ORT-allocated tensor (no shared memory, no lifetime
        ///     coupling — dispose the <see cref="OrtValue"/> when done, as with any ORT value). Any layout: a
        ///     sliced / transposed / broadcast view is copied in its logical (C) order. Same dtype rules as
        ///     <see cref="AsOrtValue"/> (Decimal → float64; Complex throws).
        /// </summary>
        /// <remarks>
        ///     Refusals come in the order of what they cost to decide: the dtype and the 2 GB limit first (they need
        ///     only the dtype and the shape, so nothing is converted, copied or allocated for an array that cannot
        ///     cross), then the buffer's liveness, then the copy. A tensor past 2 GB cannot go through this verb at
        ///     all — share it with <see cref="AsOrtValue"/>, which hands ORT a native pointer.
        /// </remarks>
        /// <param name="source">The array to copy; any layout.</param>
        /// <returns>A new ORT-allocated tensor holding the array's values in logical (C) order; the caller disposes it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX tensor element type, or the array exceeds the 2 GB span limit of ORT's managed API.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static unsafe OrtValue ToOrtValue(this NDArray source)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            // Decide everything the dtype and the shape alone decide BEFORE the source is read. The 2 GB check used
            // to run after the source had been converted and densified and ORT had allocated the tensor, so a large
            // strided array cost two full-size allocations just to be refused (or died with OutOfMemoryException in
            // the densifying copy, which is what a 2^40-element broadcast view did). The byte count is compared
            // without multiplying (size > int.MaxValue / itemsize ⇔ size × itemsize > int.MaxValue for positive
            // integers), because a broadcast view's element count can be large enough to overflow the product.
            NPTypeCode feedCode = FeedTypeCode(source.typecode);
            TensorElementType elementType = ToTensorElementType(feedCode);
            long size = source.Shape.Size;
            int feedItemSize = feedCode.SizeOf();
            if (size > int.MaxValue / feedItemSize)
                throw new NotSupportedException(TooLargeForSpanMessage(size, feedItemSize));

            // Take the reference on the SOURCE's buffer before reading it in any way: Pin's TryAddRef is the only
            // check that the buffer has not been released. The Decimal conversion and the densifying copy below
            // used to run first, and the Pin further down then succeeded on their fresh result, so a released buffer
            // was read silently and the tensor carried another array's values. Holding the reference across every
            // read also keeps the buffer alive while it is read. Even an empty array is checked.
            IArraySlice sourcePin = Pin(source);
            try
            {
                using NDArray converted = source.typecode == NPTypeCode.Decimal ? source.astype(NPTypeCode.Double) : null;
                NDArray logical = converted ?? source;

                // ORT tensors are dense row-major: a non-C-contiguous view is read in LOGICAL order through a C-order
                // copy (0-d stays 0-d; np.ascontiguousarray would lift a scalar to shape (1,), so copy() it is).
                using NDArray dense = logical.Shape.IsContiguous ? null : logical.copy();
                NDArray src = dense ?? logical;
                Shape shape = src.Shape;
                long[] dims = LongDims(shape);

                IArraySlice slice = Pin(src, out byte* data, out long nbytes);
                try
                {
                    OrtValue value = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, dims);
                    try
                    {
                        if (nbytes > 0)
                        {
                            // Refused up front already (same element count, same feed itemsize); kept beside the
                            // (int) cast below so the cast can never truncate whatever the up-front check becomes.
                            if (nbytes > int.MaxValue)
                                throw new NotSupportedException(TooLargeForSpanMessage(shape.Size, src.dtypesize));
                            Span<byte> dst = value.GetTensorMutableRawData();
                            new ReadOnlySpan<byte>(data, (int)nbytes).CopyTo(dst);
                        }

                        return value;
                    }
                    catch
                    {
                        value.Dispose();
                        throw;
                    }
                }
                finally
                {
                    slice.Release();
                    GC.KeepAlive(src);
                }
            }
            finally
            {
                sourcePin.Release();
                GC.KeepAlive(source);
            }
        }

        /// <summary>
        ///     Wrap a NumSharp array as an ORT <see cref="DenseTensor{T}"/> that SHARES its buffer — the legacy
        ///     tensor surface behind <c>NamedOnnxValue.CreateFromTensor</c> and the
        ///     <c>session.Run(IReadOnlyCollection&lt;NamedOnnxValue&gt;)</c> overloads. The tensor's
        ///     <see cref="DenseTensor{T}.Buffer"/> is a <see cref="Memory{T}"/> over NumSharp's unmanaged memory
        ///     (an internal <see cref="System.Buffers.MemoryManager{T}"/>); the returned handle holds the buffer
        ///     pin and must outlive every use of the tensor.
        ///
        ///     <para><typeparamref name="T"/> must be the C# type ORT uses for the array's dtype
        ///     (<see cref="ToTensorElementClrType"/>): <see cref="Float16"/> for <see cref="NPTypeCode.Half"/>,
        ///     <see cref="ushort"/> for <see cref="NPTypeCode.Char"/>, otherwise the dtype's own type — a
        ///     mismatch throws rather than silently reinterpreting the bytes. A C-contiguous array shares
        ///     row-major; a Fortran-contiguous array shares column-major (<c>reverseStride: true</c> — the one
        ///     non-C layout ORT's tensor can express); any other view throws. <see cref="NPTypeCode.Decimal"/>
        ///     converts to an owned float64 temporary (request <c>AsDenseTensor&lt;double&gt;</c>).</para>
        /// </summary>
        /// <typeparam name="T">The C# element type ORT uses for the array's dtype (<see cref="ToTensorElementClrType"/>); anything else is refused.</typeparam>
        /// <param name="source">The array to share; C- or Fortran-contiguous (any layout once Decimal is converted).</param>
        /// <returns>A disposable handle owning the tensor and the buffer pin; keep it alive while the tensor is used.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the element type ORT uses for the array's dtype.</exception>
        /// <exception cref="InvalidOperationException">The array is neither C- nor Fortran-contiguous.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX element type (Complex), or the array has more elements than a <see cref="Memory{T}"/> can index.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static OrtTensor<T> AsDenseTensor<T>(this NDArray source) where T : unmanaged => AsDenseTensorCore<T>(source, ownsSource: false);

        /// <summary>
        ///     <see cref="AsDenseTensor{T}"/> whose returned handle also owns <paramref name="source"/> when
        ///     <paramref name="ownsSource"/> is <c>true</c>. Ownership transfers only on success; on failure the caller
        ///     still owns its array.
        /// </summary>
        /// <typeparam name="T">The C# element type ORT uses for the (converted) array's dtype.</typeparam>
        /// <param name="source">The array to share (a Decimal array is converted to a float64 temporary the handle owns).</param>
        /// <param name="ownsSource"><c>true</c> when <paramref name="source"/> is a temporary the handle should dispose.</param>
        /// <returns>A handle owning the tensor, the buffer pin and any temporary.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the element type ORT uses for the array's dtype.</exception>
        /// <exception cref="InvalidOperationException">The array is neither C- nor Fortran-contiguous.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX element type (Complex), or the array has more elements than a <see cref="Memory{T}"/> can index.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static unsafe OrtTensor<T> AsDenseTensorCore<T>(NDArray source, bool ownsSource) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            // The Decimal → float64 conversion is the first read of the source, so it runs under a reference on the
            // source's buffer (ConvertPinned, see AsOrtValueCore): a bare astype read a released buffer silently.
            NDArray converted = null;
            NDArray fed = source;
            if (source.typecode == NPTypeCode.Decimal)
                fed = converted = ConvertPinned(source, NPTypeCode.Double);

            try
            {
                EnsureElementType<T>(fed.typecode, nameof(AsDenseTensor));
                Shape shape = fed.Shape;
                bool reverseStride;
                if (shape.IsContiguous)
                    reverseStride = false;
                else if (shape.IsFContiguous)
                    reverseStride = true;   // column-major memory: DenseTensor reads it in place with reversed strides
                else
                    throw new InvalidOperationException(
                        "the array is neither C- nor Fortran-contiguous; DenseTensor is dense (row- or column-major) and cannot " +
                        "express a sliced / transposed / negative-stride / broadcast view. Materialize first (np.ascontiguousarray(nd) " +
                        "or nd.copy()) or call ToDenseTensor(nd) for a managed copy.");

                if (shape.Size > int.MaxValue)
                    throw new NotSupportedException($"AsDenseTensor: Memory<T> holds at most {int.MaxValue} elements, the array has {shape.Size}. Use AsOrtValue (a native pointer, no such limit).");

                int[] dims = IntDims(shape, nameof(AsDenseTensor));
                IArraySlice slice = Pin(fed, out byte* data, out _);
                try
                {
                    var manager = new UnmanagedMemoryManager<T>((T*)data, (int)shape.Size);
                    var tensor = new DenseTensor<T>(manager.Memory, dims, reverseStride);
                    var handle = new OrtTensor<T>(tensor, manager, fed, slice, ownsSource: converted is not null || ownsSource);
                    if (converted is not null && ownsSource)
                        source.Dispose();
                    converted = null;
                    return handle;
                }
                catch
                {
                    slice.Release();
                    throw;
                }
            }
            finally
            {
                converted?.Dispose();
            }
        }

        /// <summary>
        ///     Copy a NumSharp array into an independent, row-major <see cref="DenseTensor{T}"/> over a fresh
        ///     managed <c>T[]</c> (no shared memory, no lifetime coupling). Any layout is read in logical (C)
        ///     order. <typeparamref name="T"/> follows the same rule as <see cref="AsDenseTensor{T}"/>.
        /// </summary>
        /// <remarks>
        ///     The element type, the size and the dimensions are checked before anything is converted, copied or
        ///     allocated; the buffer's liveness is checked next, even for an empty array.
        /// </remarks>
        /// <typeparam name="T">The C# element type ORT uses for the array's dtype (float64 for Decimal); anything else is refused.</typeparam>
        /// <param name="source">The array to copy; any layout.</param>
        /// <returns>A new row-major tensor over a managed array holding the values in logical (C) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the element type ORT uses for the array's dtype.</exception>
        /// <exception cref="NotSupportedException">
        ///     The dtype has no ONNX element type (Complex); the array has more elements than a managed <c>T[]</c> can
        ///     hold (<see cref="Array.MaxLength"/>, 2,147,483,591 — a little below <see cref="int.MaxValue"/>); or an
        ///     axis is longer than <see cref="DenseTensor{T}"/>'s <see cref="int"/> dimensions allow.
        /// </exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static unsafe DenseTensor<T> ToDenseTensor<T>(this NDArray source) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            // Decide everything the dtype and the shape alone decide BEFORE the source is read: the element type is
            // that of the FED dtype (Decimal is fed as float64), and the conversion and the copy never change the
            // shape, so the source's shape is the tensor's.
            EnsureElementType<T>(FeedTypeCode(source.typecode), nameof(ToDenseTensor));

            // The limit is Array.MaxLength, NOT int.MaxValue: a managed array holds at most 2,147,483,591 elements, so
            // comparing against int.MaxValue let every size in between reach `new T[n]`, which fails with
            // OutOfMemoryException ("Array dimensions exceeded supported range") instead of this documented refusal.
            Shape sourceShape = source.Shape;
            if (sourceShape.Size > Array.MaxLength)
                throw new NotSupportedException(
                    $"ToDenseTensor: a managed T[] holds at most {Array.MaxLength} elements (Array.MaxLength), the array has {sourceShape.Size}. " +
                    "Share the buffer zero-copy instead: AsDenseTensor (up to int.MaxValue elements) or AsOrtValue (a native pointer, no such limit).");
            int[] dims = IntDims(sourceShape, nameof(ToDenseTensor));

            // Take the reference on the SOURCE's buffer before reading it in any way: Pin's TryAddRef is the only
            // check that the buffer has not been released. The Decimal conversion and the densifying copy below used
            // to run first (and an empty array was never pinned at all), so a released buffer was read silently and
            // the tensor carried another array's values. Holding the reference across every read also keeps the
            // buffer alive while it is read; the managed array is allocated only once the source is known to be live.
            IArraySlice sourcePin = Pin(source);
            try
            {
                using NDArray converted = source.typecode == NPTypeCode.Decimal ? source.astype(NPTypeCode.Double) : null;
                NDArray logical = converted ?? source;
                Shape shape = logical.Shape;

                var data = new T[shape.Size];
                if (shape.Size > 0)
                {
                    using NDArray dense = shape.IsContiguous ? null : logical.copy();
                    NDArray src = dense ?? logical;
                    IArraySlice slice = Pin(src, out byte* p, out long nbytes);
                    try
                    {
                        fixed (T* dst = data)
                            Buffer.MemoryCopy(p, dst, nbytes, nbytes);
                    }
                    finally
                    {
                        slice.Release();
                        GC.KeepAlive(src);
                    }
                }

                return new DenseTensor<T>(data, dims);
            }
            finally
            {
                sourcePin.Release();
                GC.KeepAlive(source);
            }
        }

        /// <summary>
        ///     The refusal for a copy through ONNX Runtime's managed data API, whose <see cref="Span{T}"/> of bytes
        ///     cannot exceed 2 GB. The byte count is formatted from the element count and the item size in
        ///     <see cref="decimal"/>, because a broadcast view can hold enough elements to overflow a <see cref="long"/>
        ///     product.
        /// </summary>
        /// <param name="count">The number of elements to copy.</param>
        /// <param name="itemsize">The size in bytes of one element of the dtype that crosses.</param>
        /// <returns>The message, naming the zero-copy verbs that have no such limit.</returns>
        internal static string TooLargeForSpanMessage(long count, int itemsize) =>
            $"the tensor is {(decimal)count * itemsize} bytes, beyond the 2 GB System.Span limit of ONNX Runtime's managed data API " +
            "(GetTensorMutableRawData / GetTensorDataAsSpan). Share the buffer zero-copy with AsOrtValue / AsNDArray instead of copying it.";
    }
}
