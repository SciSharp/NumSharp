using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.Tensors
{
    public static partial class NDArrayTensorsInterop
    {
        // ===========================  NumSharp  ->  System.Numerics.Tensors  ===========================

        /// <summary>
        ///     View a NumSharp array as a <see cref="TensorSpan{T}"/> / <see cref="ReadOnlyTensorSpan{T}"/> that
        ///     SHARES its unmanaged buffer — no copy. The returned handle owns the buffer pin; take the span from
        ///     it (<see cref="TensorSpanHandle{T}.Span"/> / <see cref="TensorSpanHandle{T}.ReadOnlySpan"/>) inside
        ///     a <c>using</c>.
        ///
        ///     <para><b>Layout.</b> A <see cref="TensorSpan{T}"/> carries explicit lengths+strides, so ANY
        ///     non-negative-stride NumSharp layout shares zero-copy: C-contiguous, an offset slice, a transposed
        ///     or otherwise strided view, and even a <b>broadcast</b> view (stride-0 dimensions — exposed through
        ///     <see cref="TensorSpanHandle{T}.ReadOnlySpan"/> only, since writing overlapping lanes would
        ///     corrupt data). A <b>negative-stride</b> view (a reversed slice <c>a[::-1]</c>) is the one refusal:
        ///     <c>System.Numerics.Tensors</c> forbids negative strides. Materialize it first
        ///     (<c>np.ascontiguousarray(nd)</c>, <c>nd.copy()</c>) or use <see cref="ToTensor{T}"/>.</para>
        ///
        ///     <para><b>Dtype.</b> <typeparamref name="T"/> must be the array's own element type
        ///     (<see cref="ToTensorElementClrType"/>) — <see cref="System.Half"/> for Half, <see cref="char"/>
        ///     for Char, <see cref="decimal"/> for Decimal, <see cref="System.Numerics.Complex"/> for Complex,
        ///     otherwise the dtype's C# type. A mismatch throws rather than reinterpreting the bytes. There are no
        ///     conversions and no unsupported dtypes: all 15 cross as themselves.</para>
        ///
        ///     <para><b>Lifetime.</b> The handle takes its own atomic reference on the NumSharp buffer, so the
        ///     memory survives even if <paramref name="source"/> is disposed or collected while a span is in use;
        ///     disposing the handle releases the reference. There is NO 2 GB limit (the span fronts a native
        ///     pointer with an <see cref="System.IntPtr"/> length, unlike a <c>Memory&lt;T&gt;</c>-backed tensor).</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">The array is a negative-stride view.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the array's element type.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static TensorSpanHandle<T> AsTensorSpan<T>(this NDArray source) where T : unmanaged
            => AsTensorSpanCore<T>(source, ownsSource: false);

        internal static unsafe TensorSpanHandle<T> AsTensorSpanCore<T>(NDArray source, bool ownsSource) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            EnsureElementType<T>(source.typecode, nameof(AsTensorSpan));

            Shape shape = source.Shape;
            bool writeable = shape.IsWriteable;

            // Edge shapes System.Numerics.Tensors handles awkwardly, resolved explicitly:
            //   empty (size 0)     -> the span is TensorSpan<T>.Empty (its ctor rejects a 0-length backing).
            //   0-d scalar (ndim 0)-> a rank-1 [1] span (the pointer ctor infers rank-1 from empty lengths and
            //                         cannot express rank 0, so a scalar crosses as a single-element vector).
            bool empty = shape.Size == 0;
            nint[] lengths, strides;
            nint dataLength;
            if (empty)
            {
                lengths = Array.Empty<nint>();
                strides = Array.Empty<nint>();
                dataLength = 0;
            }
            else if (shape.NDim == 0)
            {
                lengths = new nint[] { 1 };
                strides = new nint[] { 0 };   // a single-element axis is never stepped (BCL requires stride 0)
                dataLength = 1;
            }
            else
            {
                lengths = NIntLengths(shape);
                strides = NIntStrides(shape, nameof(AsTensorSpan));   // throws on a negative-stride view
                dataLength = ReachElements(shape);
            }

            IArraySlice slice = Pin(source, out byte* data);
            try
            {
                return new TensorSpanHandle<T>((T*)data, dataLength, lengths, strides, writeable, empty, slice, source, ownsSource);
            }
            catch
            {
                slice.Release();
                throw;
            }
        }

        /// <summary>
        ///     Copy a NumSharp array into an independent, dense C-contiguous <see cref="Tensor{T}"/> over a fresh
        ///     managed <c>T[]</c> — no shared memory, no lifetime coupling. Any layout is read in logical (C)
        ///     order, so a negative-stride / broadcast / transposed view is fine here (unlike
        ///     <see cref="AsTensorSpan{T}"/>). <typeparamref name="T"/> follows the same rule as
        ///     <see cref="AsTensorSpan{T}"/>.
        /// </summary>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the array's element type.</exception>
        /// <exception cref="NotSupportedException">The array exceeds the <see cref="int.MaxValue"/>-element limit of a managed <c>T[]</c>.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static unsafe Tensor<T> ToTensor<T>(this NDArray source) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            EnsureElementType<T>(source.typecode, nameof(ToTensor));

            Shape shape = source.Shape;
            if (shape.Size > int.MaxValue)
                throw new NotSupportedException(
                    $"ToTensor: a managed T[] holds at most {int.MaxValue} elements, the array has {shape.Size}. " +
                    "Share the buffer zero-copy with AsTensorSpan (a native pointer, no such limit) instead.");

            var data = new T[shape.Size];
            if (shape.Size > 0)
            {
                using NDArray dense = shape.IsContiguous ? null : source.copy();
                NDArray src = dense ?? source;
                IArraySlice slice = Pin(src, out byte* p);
                try
                {
                    long nbytes = shape.Size * src.dtypesize;
                    fixed (T* dst = data)
                        Buffer.MemoryCopy(p, dst, nbytes, nbytes);
                }
                finally
                {
                    slice.Release();
                    GC.KeepAlive(src);
                }
            }

            return Tensor.Create(data, NIntLengths(shape));
        }
    }
}
