using System;
using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
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
        /// <remarks>
        ///     A tensor that starts past element 0 of its backing array (a <c>Slice</c> or range indexer, or
        ///     <c>Tensor.Create(array, start, …)</c>) is copied from its own start, not from the array's head (see
        ///     <see cref="LogicalStart{T}"/>). A rank-0 tensor (<c>Tensor&lt;T&gt;.Empty</c>) holds no element and
        ///     comes back as the empty vector <c>(0,)</c> (see <see cref="ImportShape"/>).
        /// </remarks>
        /// <typeparam name="T">The tensor's element type; must be one of the 15 NumSharp element types.</typeparam>
        /// <param name="tensor">The tensor to copy; any layout, dense or strided.</param>
        /// <returns>A new array that owns its buffer, with the tensor's lengths and its values in logical (C) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        ///     <typeparamref name="T"/> is not a NumSharp element type, or a non-dense tensor holds more than
        ///     <see cref="int.MaxValue"/> elements (the copy goes through an int-indexed <see cref="Span{T}"/>; it is
        ///     refused before anything is allocated — view it with <see cref="AsNDArray{T}"/> instead, which has no
        ///     element-count limit).
        /// </exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static unsafe NDArray ToNDArray<T>(this Tensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));

            NPTypeCode tc = TypeCodeOf<T>();
            Shape shape = ImportShape(tensor.Lengths, tensor.FlattenedLength);
            long count = shape.Size;
            bool dense = tensor.IsDense;

            // Refuse BEFORE allocating: the check used to follow the allocation, so a too-large tensor first asked
            // for its whole result (4 TB for a 2^40-element stride-0 tensor: OutOfMemoryException instead of this
            // error), and an allocatable size was allocated only to be thrown away. A dense tensor is backed by one
            // managed T[] and can never exceed the limit.
            if (!dense && count > int.MaxValue)
                throw new NotSupportedException(
                    $"ToNDArray: FlattenTo fills a Span<T> (max {int.MaxValue} elements); the tensor has {count}. " +
                    "View it zero-copy with AsNDArray instead (no element-count limit).");

            var nd = new NDArray(tc, shape, fillZeros: false);
            try
            {
                if (count > 0)
                {
                    void* dst = nd.Storage.InternalArray.Address;
                    if (dense)
                    {
                        // Dense: the tensor's elements are contiguous in logical C order FROM ITS OWN START, which
                        // is not the pinned handle's pointer for an offset tensor (LogicalStart).
                        using MemoryHandle mh = tensor.GetPinnedHandle();   // pins the backing array for the copy
                        long nbytes = count * sizeof(T);
                        Buffer.MemoryCopy(LogicalStart(tensor), dst, nbytes, nbytes);
                    }
                    else
                    {
                        // Non-dense (a sliced/strided view): FlattenTo materializes the logical C order for us.
                        tensor.FlattenTo(new Span<T>(dst, (int)count));
                    }
                }
                return nd;
            }
            catch
            {
                // The result was never handed out: free its buffer now rather than leave it to the finalizer.
                nd.Dispose();
                throw;
            }
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
        /// <remarks>
        ///     The view starts at the tensor's own first element even when the tensor starts past element 0 of its
        ///     backing array (a <c>Slice</c>, a range indexer, <c>Tensor.Create(array, start, …)</c>), so a write
        ///     through it lands on the tensor element it names (see <see cref="LogicalStart{T}"/>). A stride-0
        ///     (broadcast) tensor comes back as a read-only broadcast view of any size, since only one element per
        ///     broadcast run exists. A tensor with no element — including the rank-0 <c>Tensor&lt;T&gt;.Empty</c> —
        ///     comes back as a fresh empty array (there is nothing to share; see <see cref="ImportShape"/>).
        /// </remarks>
        /// <typeparam name="T">The tensor's element type; must be one of the 15 NumSharp element types.</typeparam>
        /// <param name="tensor">The tensor to view; any layout, dense or strided.</param>
        /// <returns>A NumSharp view over the tensor's memory (or a fresh empty array when the tensor has no element).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not a NumSharp element type.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static unsafe NDArray AsNDArray<T>(this Tensor<T> tensor) where T : unmanaged
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));

            NPTypeCode tc = TypeCodeOf<T>();
            long[] dims = ToLongDims(tensor.Lengths);
            Shape shape = ImportShape(tensor.Lengths, tensor.FlattenedLength);
            long count = shape.Size;
            if (count == 0)
                return new NDArray(tc, shape, fillZeros: false);

            MemoryHandle mh = tensor.GetPinnedHandle();
            // The lease roots the Tensor<T> for its lifetime and unpins the backing array when the last view dies.
            var lease = new ImportLease(() => { mh.Dispose(); GC.KeepAlive(tensor); }, count * sizeof(T));
            try
            {
                // The tensor's own first element, NOT mh.Pointer (the backing array's head — see LogicalStart). The
                // pin above keeps it from moving for as long as the lease lives.
                void* ptr = LogicalStart(tensor);
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
        /// <remarks>
        ///     A rank-0 span (<c>ReadOnlyTensorSpan&lt;T&gt;.Empty</c>, a <c>default</c> span) holds no element and
        ///     comes back as the empty vector <c>(0,)</c> (see <see cref="ImportShape"/>).
        /// </remarks>
        /// <typeparam name="T">The span's element type; must be one of the 15 NumSharp element types.</typeparam>
        /// <param name="span">The span to copy; any layout, offset and strides included.</param>
        /// <returns>A new array that owns its buffer, with the span's lengths and its values in logical (C) order.</returns>
        /// <exception cref="NotSupportedException">
        ///     <typeparamref name="T"/> is not a NumSharp element type, or the span holds more than
        ///     <see cref="int.MaxValue"/> elements (the copy goes through an int-indexed <see cref="Span{T}"/>; it is
        ///     refused before anything is allocated).
        /// </exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static unsafe NDArray ToNDArray<T>(this ReadOnlyTensorSpan<T> span) where T : unmanaged
        {
            NPTypeCode tc = TypeCodeOf<T>();
            Shape shape = ImportShape(span.Lengths, span.FlattenedLength);
            long count = shape.Size;

            // Refuse BEFORE allocating (the check used to follow the allocation: see ToNDArray(Tensor<T>)).
            if (count > int.MaxValue)
                throw new NotSupportedException($"ToNDArray: FlattenTo fills a Span<T> (max {int.MaxValue} elements); the span has {count}.");

            var nd = new NDArray(tc, shape, fillZeros: false);
            try
            {
                if (count > 0)
                    span.FlattenTo(new Span<T>(nd.Storage.InternalArray.Address, (int)count));
                return nd;
            }
            catch
            {
                // The result was never handed out: free its buffer now rather than leave it to the finalizer.
                nd.Dispose();
                throw;
            }
        }

        /// <summary><see cref="ToNDArray{T}(ReadOnlyTensorSpan{T})"/> for the mutable <see cref="TensorSpan{T}"/>.</summary>
        /// <typeparam name="T">The span's element type; must be one of the 15 NumSharp element types.</typeparam>
        /// <param name="span">The span to copy; any layout.</param>
        /// <returns>A new array that owns its buffer, with the span's lengths and its values in logical (C) order.</returns>
        /// <exception cref="NotSupportedException">
        ///     <typeparamref name="T"/> is not a NumSharp element type, or the span holds more than
        ///     <see cref="int.MaxValue"/> elements.
        /// </exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static NDArray ToNDArray<T>(this TensorSpan<T> span) where T : unmanaged
            => span.AsReadOnlyTensorSpan().ToNDArray();

        // ---- shared import helpers ----------------------------------------------------------------------

        /// <summary>
        ///     The NumSharp shape an imported tensor or span maps to. A rank-0 value in <c>System.Numerics.Tensors</c>
        ///     is the degenerate EMPTY one (<c>Tensor&lt;T&gt;.Empty</c>, <c>ReadOnlyTensorSpan&lt;T&gt;.Empty</c>, a
        ///     <c>default</c> span): it holds no element, so it maps to the empty vector <c>(0,)</c>. It must NOT map
        ///     to NumSharp's 0-d shape, which claims one element: the copy then asked <c>FlattenTo</c> for an element
        ///     that does not exist (IndexOutOfRangeException), and the zero-copy view wrapped one element past the end
        ///     of the tensor's empty backing array. Every other rank keeps the lengths as they are.
        /// </summary>
        /// <param name="lengths">The tensor's or span's lengths.</param>
        /// <param name="flattenedLength">Its element count, which decides what a rank-0 value is.</param>
        /// <returns>
        ///     <c>(0,)</c> for a rank-0 value with no element; NumSharp's 0-d scalar for a rank-0 value that holds one
        ///     (a shape the BCL does not build today); otherwise a shape with the same lengths.
        /// </returns>
        internal static Shape ImportShape(ReadOnlySpan<nint> lengths, nint flattenedLength)
        {
            if (lengths.Length == 0)
                return flattenedLength == 0 ? Shape.Vector(0) : Shape.Scalar;
            return ShapeFrom(lengths);
        }

        /// <summary>
        ///     The address of a tensor's own first element — where its data starts. The caller must already have
        ///     pinned the backing array (<see cref="Tensor{T}.GetPinnedHandle"/>) and keep it pinned while the
        ///     address is used.
        /// </summary>
        /// <remarks>
        ///     <see cref="Tensor{T}.GetPinnedHandle"/>'s <c>Pointer</c> is NOT this address: the handle pins the whole
        ///     backing <c>T[]</c> and points at the ARRAY's element 0, ignoring the tensor's start offset (verified on
        ///     System.Numerics.Tensors 10.0.0, net8.0 and net10.0: for <c>t.Slice(1..3, ..)</c> of a 3×4 tensor the
        ///     handle points at backing[0] while the slice starts at backing[4]). Every tensor produced by a
        ///     <c>Slice</c> or range indexer, or by <c>Tensor.Create(array, start, …)</c>, starts past element 0, so
        ///     reading from the handle's pointer imported the wrong elements, and a write through the zero-copy view
        ///     landed outside the tensor. The span's <c>GetPinnableReference()</c> is the tensor's own start.
        /// </remarks>
        /// <typeparam name="T">The tensor's element type.</typeparam>
        /// <param name="tensor">A tensor with at least one element (an empty tensor's reference is null).</param>
        /// <returns>A pointer to the tensor's first element, valid while the backing array stays pinned.</returns>
        internal static unsafe void* LogicalStart<T>(Tensor<T> tensor) where T : unmanaged
            => Unsafe.AsPointer(ref tensor.AsTensorSpan().GetPinnableReference());

        /// <summary>
        ///     A tensor's <c>ReadOnlySpan&lt;nint&gt;</c> lengths → a NumSharp <see cref="Shape"/> (rank 0 → the
        ///     scalar). Importing a BCL tensor or span goes through <see cref="ImportShape"/> instead, because a
        ///     BCL rank-0 value has NO element and must not become a one-element 0-d array.
        /// </summary>
        /// <param name="lengths">The lengths to convert.</param>
        /// <returns>A shape with the same lengths, or NumSharp's 0-d scalar for empty lengths.</returns>
        internal static Shape ShapeFrom(ReadOnlySpan<nint> lengths) => ShapeFrom(ToLongDims(lengths));

        /// <summary>
        ///     Element dimensions → a NumSharp <see cref="Shape"/>; null or empty dimensions give the 0-d scalar (see
        ///     <see cref="ImportShape"/> for why an imported BCL value must not reach that case).
        /// </summary>
        /// <param name="dims">The dimensions, or null.</param>
        /// <returns>A C-contiguous shape with those dimensions, or the 0-d scalar.</returns>
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
