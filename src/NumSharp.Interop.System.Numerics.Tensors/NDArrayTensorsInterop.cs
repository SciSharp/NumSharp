using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Threading;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.Tensors
{
    /// <summary>
    ///     Zero-copy interop between NumSharp <see cref="NDArray"/> and the BCL's
    ///     <c>System.Numerics.Tensors</c> types — <see cref="System.Numerics.Tensors.Tensor{T}"/>,
    ///     <see cref="System.Numerics.Tensors.TensorSpan{T}"/> and
    ///     <see cref="System.Numerics.Tensors.ReadOnlyTensorSpan{T}"/>.
    ///
    ///     <para>This is a CONVERSION library in the mould of the ONNX Runtime / ML.NET bridges: it moves data
    ///     (or rather, descriptions of the same memory) across the boundary. It is NOT an engine backend —
    ///     <c>System.Numerics.Tensors</c> does not compute NumSharp operations (and routing NumSharp's kernels
    ///     through <c>TensorPrimitives</c> would abandon NumSharp's bit-exact-with-NumPy results), so there is
    ///     no <c>TensorEngine</c> seam, no <c>[ModuleInitializer]</c>, and referencing the package changes
    ///     nothing until a verb is called.</para>
    ///
    ///     <para><b>The verbs</b> follow the house convention: <c>As…</c> shares memory (zero-copy view),
    ///     <c>To…</c> copies.</para>
    ///     <list type="table">
    ///         <item><term><see cref="Export.AsTensorSpan{T}"/></term><description>NumSharp → a <see cref="System.Numerics.Tensors.TensorSpan{T}"/> / <see cref="System.Numerics.Tensors.ReadOnlyTensorSpan{T}"/> over the NDArray's own buffer (ANY non-negative-stride layout — contiguous, sliced, transposed, strided, broadcast; the buffer is ARC-rooted for the handle's lifetime)</description></item>
    ///         <item><term><see cref="Export.ToTensor{T}"/></term><description>NumSharp → an independent, dense <see cref="System.Numerics.Tensors.Tensor{T}"/> copy (any layout, no lifetime coupling)</description></item>
    ///         <item><term><see cref="Import.ToNDArray{T}(System.Numerics.Tensors.Tensor{T})"/></term><description>Tensors → NumSharp, a fresh owning C-contiguous copy (the safe default)</description></item>
    ///         <item><term><see cref="Import.AsNDArray{T}(System.Numerics.Tensors.Tensor{T}, bool)"/></term><description>Tensors → NumSharp, a zero-copy view over the pinned managed backing store</description></item>
    ///     </list>
    ///
    ///     <para><b>Dtype map.</b> Unlike ONNX Runtime (a fixed <c>TensorElementType</c> enum),
    ///     <c>System.Numerics.Tensors</c> containers are pure generics over an unmanaged <c>T</c> with NO dtype
    ///     tag, so <b>all 15 NumSharp dtypes cross zero-copy as their own CLR type</b> — bool, the eight
    ///     integers, <see cref="System.Char"/>, <see cref="System.Half"/> (directly, no <c>Float16</c>
    ///     wrapper), float, double, <see cref="System.Decimal"/> and <see cref="System.Numerics.Complex"/>.
    ///     Nothing is refused and nothing is converted. See <see cref="ToTensorElementClrType"/> /
    ///     <see cref="TypeCodeOf{T}"/>.</para>
    ///
    ///     <para><b>Layout.</b> A <see cref="System.Numerics.Tensors.TensorSpan{T}"/> carries explicit
    ///     lengths+strides, so a sliced / transposed / strided / broadcast NDArray view shares zero-copy —
    ///     a real advantage over the row-major-only ONNX bridge. The one exception is a <b>negative-stride</b>
    ///     view (e.g. <c>a[::-1]</c>): <c>System.Numerics.Tensors</c> forbids negative strides, so those are
    ///     refused with a message pointing at <see cref="Export.ToTensor{T}"/> / <c>np.ascontiguousarray</c>.</para>
    ///
    ///     <para><b>Experimental.</b> The BCL marks <c>Tensor{T}</c>/<c>TensorSpan{T}</c>
    ///     <c>[Experimental("SYSLIB5001")]</c>; this surface re-marks itself
    ///     <c>[Experimental("NUMSHARP_TENSORS")]</c> so a consumer acknowledges that churn ONCE, here, rather
    ///     than at every tensor touch. Suppress with <c>&lt;NoWarn&gt;NUMSHARP_TENSORS&lt;/NoWarn&gt;</c> or a
    ///     <c>#pragma warning disable NUMSHARP_TENSORS</c>.</para>
    /// </summary>
    [Experimental("NUMSHARP_TENSORS")]
    public static partial class NDArrayTensorsInterop
    {
        private static int _liveExports;
        private static int _liveImports;

        /// <summary>
        ///     Number of live export handles (<see cref="TensorSpanHandle{T}"/>) — NumSharp buffers currently
        ///     pinned for a <see cref="System.Numerics.Tensors.TensorSpan{T}"/>. Every handle increments it on
        ///     creation and decrements it when disposed (or finalized); a steady non-zero count is a leak.
        /// </summary>
        public static int LiveExports => Volatile.Read(ref _liveExports);

        /// <summary>
        ///     Number of live import leases — NumSharp views (<see cref="Import.AsNDArray{T}(System.Numerics.Tensors.Tensor{T}, bool)"/>)
        ///     currently holding a pinned <see cref="System.Numerics.Tensors.Tensor{T}"/> backing store. Released
        ///     when the LAST NumSharp view over the memory — derived slices included — is disposed or collected.
        /// </summary>
        public static int LiveImports => Volatile.Read(ref _liveImports);

        internal static void ExportOpened() => Interlocked.Increment(ref _liveExports);
        internal static void ExportClosed() => Interlocked.Decrement(ref _liveExports);
        internal static void ImportOpened() => Interlocked.Increment(ref _liveImports);
        internal static void ImportClosed() => Interlocked.Decrement(ref _liveImports);

        // ===================================  the dtype map  ===================================

        /// <summary>
        ///     The C# element type a <c>System.Numerics.Tensors</c> container uses for a NumSharp dtype — which
        ///     is simply the dtype's OWN CLR type: <see cref="System.Half"/> for <see cref="NPTypeCode.Half"/>
        ///     (directly, no wrapper), <see cref="char"/> for <see cref="NPTypeCode.Char"/>,
        ///     <see cref="System.Decimal"/> for <see cref="NPTypeCode.Decimal"/>,
        ///     <see cref="System.Numerics.Complex"/> for <see cref="NPTypeCode.Complex"/>. Every dtype maps —
        ///     the containers are unconstrained generics, so none is refused.
        /// </summary>
        public static Type ToTensorElementClrType(NPTypeCode code)
        {
            switch (code)
            {
                case NPTypeCode.Boolean: return typeof(bool);
                case NPTypeCode.Byte: return typeof(byte);
                case NPTypeCode.SByte: return typeof(sbyte);
                case NPTypeCode.Int16: return typeof(short);
                case NPTypeCode.UInt16: return typeof(ushort);
                case NPTypeCode.Int32: return typeof(int);
                case NPTypeCode.UInt32: return typeof(uint);
                case NPTypeCode.Int64: return typeof(long);
                case NPTypeCode.UInt64: return typeof(ulong);
                case NPTypeCode.Char: return typeof(char);
                case NPTypeCode.Half: return typeof(Half);
                case NPTypeCode.Single: return typeof(float);
                case NPTypeCode.Double: return typeof(double);
                case NPTypeCode.Decimal: return typeof(decimal);
                case NPTypeCode.Complex: return typeof(Complex);
                default: throw new NotSupportedException($"NumSharp dtype {code} has no CLR element type.");
            }
        }

        /// <summary>
        ///     The NumSharp dtype a <c>System.Numerics.Tensors</c> element type <typeparamref name="T"/> reads
        ///     back as — the inverse of <see cref="ToTensorElementClrType"/>. Directional like the whole family:
        ///     <see cref="ushort"/> comes back as <see cref="NPTypeCode.UInt16"/>, never <see cref="NPTypeCode.Char"/>
        ///     (a tensor carries no "these are characters" bit — request <c>&lt;char&gt;</c> explicitly for that).
        /// </summary>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not one of the 15 NumSharp element types.</exception>
        public static NPTypeCode TypeCodeOf<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(bool)) return NPTypeCode.Boolean;
            if (typeof(T) == typeof(byte)) return NPTypeCode.Byte;
            if (typeof(T) == typeof(sbyte)) return NPTypeCode.SByte;
            if (typeof(T) == typeof(short)) return NPTypeCode.Int16;
            if (typeof(T) == typeof(ushort)) return NPTypeCode.UInt16;
            if (typeof(T) == typeof(int)) return NPTypeCode.Int32;
            if (typeof(T) == typeof(uint)) return NPTypeCode.UInt32;
            if (typeof(T) == typeof(long)) return NPTypeCode.Int64;
            if (typeof(T) == typeof(ulong)) return NPTypeCode.UInt64;
            if (typeof(T) == typeof(char)) return NPTypeCode.Char;
            if (typeof(T) == typeof(Half)) return NPTypeCode.Half;
            if (typeof(T) == typeof(float)) return NPTypeCode.Single;
            if (typeof(T) == typeof(double)) return NPTypeCode.Double;
            if (typeof(T) == typeof(decimal)) return NPTypeCode.Decimal;
            if (typeof(T) == typeof(Complex)) return NPTypeCode.Complex;
            throw new NotSupportedException(
                $"{typeof(T).Name} is not a NumSharp element type. The 15 dtypes are bool, byte, sbyte, short, ushort, int, " +
                "uint, long, ulong, char, System.Half, float, double, System.Decimal and System.Numerics.Complex.");
        }

        /// <summary>
        ///     Validate that a caller-chosen <typeparamref name="T"/> is the element type
        ///     (<see cref="ToTensorElementClrType"/>) of the array's dtype. A mismatch would silently
        ///     reinterpret the bytes as a different dtype, so it is refused up front.
        /// </summary>
        internal static void EnsureElementType<T>(NPTypeCode code, string verb) where T : unmanaged
        {
            Type want = ToTensorElementClrType(code);
            if (typeof(T) != want)
                throw new ArgumentException(
                    $"{verb}<{typeof(T).Name}>: a NumSharp {code} array crosses as {want.Name} " +
                    $"({(code == NPTypeCode.Char ? "chars are their own 16-bit type — request <char>" : "the dtype's own CLR type")}); " +
                    $"reinterpreting it as {typeof(T).Name} would hand back the wrong dtype. Call {verb}<{want.Name}>() or convert with astype() first.");
        }

        // ===================================  shared plumbing  ===================================

        internal const string NegativeStrideMessage =
            "the array is a negative-stride view (e.g. a reversed slice a[::-1]); System.Numerics.Tensors forbids " +
            "negative strides (its TensorSpan ctor throws \"Strides cannot be less than 0\"), so it cannot be shared " +
            "zero-copy. Materialize first — np.ascontiguousarray(nd) or nd.copy() — or call ToTensor(nd) for an " +
            "independent dense copy (which reads any layout in logical order).";

        /// <summary>
        ///     Take an ARC reference on the array's buffer and resolve the pointer to its first logical element
        ///     (<c>slice.Address + Shape.Offset × itemsize</c> — right for both a contiguous slice that re-seats
        ///     the storage address and a strided view that keeps the base address with a non-zero offset). The
        ///     caller owns the reference and must <see cref="IArraySlice.Release"/> it. Identical to the ONNX
        ///     bridge's <c>Pin</c>.
        /// </summary>
        internal static unsafe IArraySlice Pin(NDArray source, out byte* data)
        {
            IArraySlice slice = source.Storage.InternalArray;
            if (!slice.TryAddRef())
                throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");
            Shape shape = source.Shape;
            int itemsize = source.dtypesize;
            data = (byte*)slice.Address + shape.offset * itemsize;
            return slice;
        }

        /// <summary>A NumSharp shape's dimensions as the <c>nint</c> lengths <c>System.Numerics.Tensors</c> uses (0-d → empty).</summary>
        internal static nint[] NIntLengths(Shape shape)
        {
            int n = shape.NDim;
            if (n == 0)
                return Array.Empty<nint>();
            long[] dims = shape.dimensions;
            var lengths = new nint[n];
            for (int i = 0; i < n; i++)
                lengths[i] = (nint)dims[i];
            return lengths;
        }

        /// <summary>
        ///     A NumSharp shape's ELEMENT strides as <c>nint</c> (System.Numerics.Tensors strides are in
        ///     elements, exactly like NumSharp's internal <c>Shape.strides</c> — the PUBLIC <c>nd.strides</c> is
        ///     in bytes). Throws for a negative-stride layout, which the tensor types forbid.
        /// </summary>
        internal static nint[] NIntStrides(Shape shape, string verb)
        {
            int n = shape.NDim;
            if (n == 0)
                return Array.Empty<nint>();
            long[] dims = shape.dimensions;
            long[] strides = shape.strides;
            var result = new nint[n];
            for (int i = 0; i < n; i++)
            {
                // A length-≤1 axis is never stepped, so its stride is irrelevant to addressing — but
                // System.Numerics.Tensors REQUIRES it be 0 (a nonzero stride on such an axis reads as
                // over-claiming the buffer: "would allow you to access elements outside the provided memory").
                // NumSharp assigns unit axes a nonzero element stride, so normalize them to 0 here.
                if (dims[i] <= 1)
                {
                    result[i] = 0;
                    continue;
                }
                if (strides[i] < 0)
                    throw new InvalidOperationException($"{verb}: {NegativeStrideMessage}");
                result[i] = (nint)strides[i];
            }
            return result;
        }

        /// <summary>
        ///     The number of elements reachable from the logical-start pointer — the <c>dataLength</c> the
        ///     <c>TensorSpan&lt;T&gt;(T*, nint dataLength, …)</c> ctor bounds against. <c>bufferSize - offset</c>
        ///     always covers a valid view's reach (for non-negative strides the max reachable index is
        ///     <c>Σ (len_i-1)·stride_i ≤ bufferSize - offset - 1</c>).
        /// </summary>
        internal static nint ReachElements(Shape shape)
        {
            long reach = shape.bufferSize - shape.offset;
            return reach < 0 ? 0 : (nint)reach;
        }

        /// <summary>
        ///     Wrap foreign (pinned managed) memory as a NumSharp <see cref="IArraySlice"/> whose memory-block
        ///     Disposer invokes <paramref name="dispose"/> exactly once when the LAST NumSharp reference (any
        ///     view sharing the block) is released — deterministically via <see cref="NDArray.Dispose"/> or by
        ///     the finalizer safety net. The same primitive the ONNX / pythonnet bridges lease foreign buffers with.
        /// </summary>
        internal static unsafe IArraySlice WrapExternal(NPTypeCode tc, void* p, long count, Action dispose)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)p, count, dispose));
                case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)p, count, dispose));
                case NPTypeCode.SByte: return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)p, count, dispose));
                case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)p, count, dispose));
                case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)p, count, dispose));
                case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)p, count, dispose));
                case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)p, count, dispose));
                case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)p, count, dispose));
                case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)p, count, dispose));
                case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>((char*)p, count, dispose));
                case NPTypeCode.Half: return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)p, count, dispose));
                case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)p, count, dispose));
                case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)p, count, dispose));
                case NPTypeCode.Decimal: return new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>((decimal*)p, count, dispose));
                case NPTypeCode.Complex: return new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>((Complex*)p, count, dispose));
                default: throw new NotSupportedException(tc.ToString());
            }
        }
    }
}
