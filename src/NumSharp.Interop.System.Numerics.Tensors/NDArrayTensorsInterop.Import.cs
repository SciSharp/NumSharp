using System;
using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.Tensors
{
    public static partial class NDArrayTensorsInterop
    {
        // ===========================  System.Numerics.Tensors  ->  NumSharp  ===========================

        /// <summary>
        ///     Copy a <see cref="Tensor{T}"/> into a fresh, owning, C-contiguous <see cref="NDArray"/> — the safe
        ///     default: the tensor can be discarded the moment this returns, and the result has no lifetime
        ///     coupling. A sliced / strided (non-dense) tensor is read in its logical (C) order.
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static unsafe NDArray ToNDArray<T>(this Tensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));

            NPTypeCode tc = TypeCodeOf<T>();
            Shape shape = ShapeFrom(tensor.Lengths);
            long count = shape.Size;

            var nd = new NDArray(tc, shape, fillZeros: false);
            if (count > 0)
            {
                void* dst = nd.Storage.InternalArray.Address;
                if (tensor.IsDense)
                {
                    // Dense (incl. a 0-d scalar): the backing store IS the logical C-order buffer — raw copy it.
                    using MemoryHandle mh = tensor.GetPinnedHandle();
                    long nbytes = count * sizeof(T);
                    Buffer.MemoryCopy(mh.Pointer, dst, nbytes, nbytes);
                }
                else
                {
                    // Non-dense (a sliced/strided view): FlattenTo materializes the logical C order for us.
                    if (count > int.MaxValue)
                        throw new NotSupportedException($"ToNDArray: FlattenTo fills a Span<T> (max {int.MaxValue} elements); the tensor has {count}. Densify with tensor.ToDenseTensor() first.");
                    tensor.FlattenTo(new Span<T>(dst, (int)count));
                }
            }

            return nd;
        }

        /// <summary>
        ///     View a <see cref="Tensor{T}"/> as a NumSharp array over the tensor's OWN managed backing store —
        ///     zero-copy, mutations visible both ways. The backing array is PINNED for the view's lifetime (the
        ///     pin is released when the last NumSharp view over it — derived slices included — is disposed or
        ///     collected). A dense tensor comes back as a C-contiguous view; a strided (non-dense) tensor comes
        ///     back as a strided view sharing the same memory — no densifying copy.
        ///
        ///     <para>The view does not own its data (<c>flags.owndata == False</c>, like <c>np.frombuffer</c>):
        ///     a size-changing <c>ndarray.resize</c> refuses instead of detaching from the tensor's memory. Keep
        ///     the <see cref="Tensor{T}"/> reachable in your own code as usual; the view roots it for its lifetime.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static unsafe NDArray AsNDArray<T>(this Tensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));

            NPTypeCode tc = TypeCodeOf<T>();
            long[] dims = ToLongDims(tensor.Lengths);
            Shape shape = ShapeFrom(dims);
            long count = shape.Size;
            if (count == 0)
                return new NDArray(tc, shape, fillZeros: false);

            MemoryHandle mh = tensor.GetPinnedHandle();
            // The lease roots the Tensor<T> for its lifetime and unpins the backing array when the last view dies.
            var lease = new ImportLease(() => { mh.Dispose(); GC.KeepAlive(tensor); }, count * sizeof(T));
            try
            {
                void* ptr = mh.Pointer;   // the tensor's LOGICAL element 0 (GetPinnableReference), pinned
                if (tensor.IsDense)
                {
                    IArraySlice slice = WrapExternal(tc, ptr, count, lease.Release);
                    return new NDArray(new UnmanagedStorage(slice, shape).Alias());
                }

                // Strided: alias a flat storage of the reachable window with the tensor's own element strides.
                long[] strides = ToLongDims(tensor.Strides);
                long reach = ReachFrom(dims, strides);
                IArraySlice slab = WrapExternal(tc, ptr, reach, lease.Release);
                UnmanagedStorage storage = new UnmanagedStorage(slab, Shape.Vector(reach))
                    .Alias(new Shape(dims, strides, offset: 0, bufferSize: reach));
                return new NDArray(storage);
            }
            catch
            {
                lease.Release();
                throw;
            }
        }

        /// <summary>
        ///     Copy a <see cref="ReadOnlyTensorSpan{T}"/> into a fresh owning C-contiguous <see cref="NDArray"/>.
        ///     A <c>ref struct</c> span cannot be leased (it is stack-only), so this always copies — read in the
        ///     span's logical (C) order.
        /// </summary>
        [Experimental("NUMSHARP_TENSORS")]
        public static unsafe NDArray ToNDArray<T>(this ReadOnlyTensorSpan<T> span) where T : unmanaged
        {
            NPTypeCode tc = TypeCodeOf<T>();
            long[] dims = ToLongDims(span.Lengths);
            Shape shape = ShapeFrom(dims);
            long count = shape.Size;

            var nd = new NDArray(tc, shape, fillZeros: false);
            if (count > 0)
            {
                if (count > int.MaxValue)
                    throw new NotSupportedException($"ToNDArray: FlattenTo fills a Span<T> (max {int.MaxValue} elements); the span has {count}.");
                span.FlattenTo(new Span<T>(nd.Storage.InternalArray.Address, (int)count));
            }
            return nd;
        }

        /// <summary><see cref="ToNDArray{T}(ReadOnlyTensorSpan{T})"/> for the mutable <see cref="TensorSpan{T}"/>.</summary>
        [Experimental("NUMSHARP_TENSORS")]
        public static NDArray ToNDArray<T>(this TensorSpan<T> span) where T : unmanaged
            => span.AsReadOnlyTensorSpan().ToNDArray();

        // ---- shared import helpers ----------------------------------------------------------------------

        /// <summary>A tensor's <c>ReadOnlySpan&lt;nint&gt;</c> lengths → a NumSharp <see cref="Shape"/> (rank 0 → the scalar).</summary>
        internal static Shape ShapeFrom(ReadOnlySpan<nint> lengths) => ShapeFrom(ToLongDims(lengths));

        internal static Shape ShapeFrom(long[] dims)
        {
            if (dims is null || dims.Length == 0)
                return Shape.Scalar;
            return new Shape(dims);
        }

        internal static long[] ToLongDims(ReadOnlySpan<nint> src)
        {
            if (src.Length == 0)
                return Array.Empty<long>();
            var result = new long[src.Length];
            for (int i = 0; i < src.Length; i++)
                result[i] = src[i];
            return result;
        }

        /// <summary>Elements reachable from logical element 0 given non-negative strides: <c>1 + Σ (len_i-1)·stride_i</c>.</summary>
        internal static long ReachFrom(long[] dims, long[] strides)
        {
            long reach = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] <= 0)
                    return 0;
                reach += (dims[i] - 1) * strides[i];
            }
            return reach;
        }
    }
}
