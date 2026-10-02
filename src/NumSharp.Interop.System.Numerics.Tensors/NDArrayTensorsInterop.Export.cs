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
        ///
        ///     <para><b>Edge shapes.</b> <c>System.Numerics.Tensors</c> has no rank 0, so a 0-d scalar crosses as a
        ///     single-element vector <c>[1]</c> (the same shape <see cref="ToTensor{T}"/> gives it). An empty array keeps
        ///     its shape, with every stride 0: a <c>(3,0,4)</c> array is a rank-3 span with <c>FlattenedLength</c> 0,
        ///     which the BCL can flatten, fill and reduce like any other span.</para>
        /// </summary>
        /// <typeparam name="T">The array's own element type (<see cref="ToTensorElementClrType"/>); anything else is refused.</typeparam>
        /// <param name="source">The array to share. It stays usable, and may even be disposed while the handle lives.</param>
        /// <returns>A handle that owns one reference on the buffer; dispose it when the span is no longer used.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="InvalidOperationException">The array is a negative-stride view.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the array's element type.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static TensorSpanHandle<T> AsTensorSpan<T>(this NDArray source) where T : unmanaged
            => AsTensorSpanCore<T>(source, ownsSource: false);

        /// <summary>
        ///     The body of <see cref="AsTensorSpan{T}"/>: resolve the lengths, element strides and reachable length the
        ///     BCL's pointer constructor takes, then take the ARC reference the returned handle owns. The ARC reference
        ///     is taken LAST, after every check that can throw, so a refused array never leaves a reference behind.
        /// </summary>
        /// <typeparam name="T">The array's own element type; anything else is refused.</typeparam>
        /// <param name="source">The array to share.</param>
        /// <param name="ownsSource">
        ///     <c>true</c> when the handle should dispose <paramref name="source"/> together with its own reference
        ///     (for a temporary array created only to be exported); <c>false</c> leaves the caller's array alone.
        /// </param>
        /// <returns>The export handle, holding one reference on the buffer.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the array's element type.</exception>
        /// <exception cref="InvalidOperationException">The array is a negative-stride view.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static unsafe TensorSpanHandle<T> AsTensorSpanCore<T>(NDArray source, bool ownsSource) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            EnsureElementType<T>(source.typecode, nameof(AsTensorSpan));

            Shape shape = source.Shape;
            bool writeable = shape.IsWriteable;

            // Edge shapes System.Numerics.Tensors handles awkwardly, resolved explicitly:
            //   empty (size 0)     -> the array's own lengths with EVERY stride 0. The BCL refuses a zero-length
            //                         backing only when a stride is nonzero ("would allow you to access elements
            //                         outside the provided memory"), so all-zero strides are what make a shaped
            //                         empty span legal. TensorSpan<T>.Empty is no substitute: it is RANK 0, so the
            //                         shape would be lost, and the BCL's own FlattenTo / Tensor.Sum throw
            //                         IndexOutOfRangeException on it. (Every stride is irrelevant here: nothing
            //                         is ever addressed, so an empty negative-stride view is fine too.)
            //   0-d scalar (ndim 0)-> a rank-1 [1] span. The BCL has no rank 0: EMPTY lengths do not mean "scalar"
            //                         to it, they build a rank-1 span of length 0, so a scalar crosses as a
            //                         single-element vector with explicit lengths.
            bool empty = shape.Size == 0;
            nint[] lengths, strides;
            nint dataLength;
            if (empty)
            {
                lengths = NIntLengths(shape);
                strides = new nint[lengths.Length];   // all zero
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
        /// <remarks>
        ///     A 0-d scalar becomes a single-element vector tensor <c>[1]</c>, because the BCL has no rank 0 (passing
        ///     it empty lengths builds a rank-1 tensor of LENGTH 0, which would drop the value); importing the tensor
        ///     back therefore gives shape <c>(1,)</c>, not <c>()</c>. An empty array keeps its shape.
        /// </remarks>
        /// <typeparam name="T">The array's own element type (<see cref="ToTensorElementClrType"/>); anything else is refused.</typeparam>
        /// <param name="source">The array to copy; any layout.</param>
        /// <returns>A new dense tensor holding the array's values in logical (C) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the array's element type.</exception>
        /// <exception cref="NotSupportedException">The array exceeds the <see cref="int.MaxValue"/>-element limit of a managed <c>T[]</c>.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
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

            // Take the reference on the SOURCE's buffer before reading it in any way: Pin's TryAddRef is the only
            // check that the buffer has not been released. A strided source is densified with copy() below, and
            // that copy used to run before any check, so a released buffer was read silently — once the pool had
            // handed it out again, the tensor held another array's data. Holding the reference across the copy
            // also keeps the buffer alive while it is read. Even an empty array is checked, as AsTensorSpan does.
            IArraySlice pin = Pin(source, out byte* p);
            try
            {
                var data = new T[shape.Size];
                if (shape.Size > 0)
                {
                    long nbytes = shape.Size * source.dtypesize;
                    if (shape.IsContiguous)
                    {
                        fixed (T* dst = data)
                            Buffer.MemoryCopy(p, dst, nbytes, nbytes);
                    }
                    else
                    {
                        // copy() lays the logical C order out contiguously; the raw copy below then reads it whole.
                        using NDArray dense = source.copy();
                        IArraySlice densePin = Pin(dense, out byte* q);
                        try
                        {
                            fixed (T* dst = data)
                                Buffer.MemoryCopy(q, dst, nbytes, nbytes);
                        }
                        finally
                        {
                            densePin.Release();
                        }
                    }
                }

                // A 0-d scalar needs explicit lengths [1] (see remarks): empty lengths would make a length-0 tensor.
                nint[] lengths = shape.NDim == 0 ? new nint[] { 1 } : NIntLengths(shape);
                return Tensor.Create(data, lengths);
            }
            finally
            {
                pin.Release();
                GC.KeepAlive(source);
            }
        }
    }
}
