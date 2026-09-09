using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        // Built-in .NET (BCL) views over the NDArray's unmanaged buffer, reached through nd.Unsafe.
        //
        // Every accessor here is UNSAFE in one specific sense: the returned Span<T>/ReadOnlySpan<T>/
        // Memory<T>/pointer ALIASES the NDArray's unmanaged buffer and does NOT keep it alive — none
        // of them takes an ARC reference on the memory holder. If the NDArray is disposed or
        // garbage-collected while one of these is still in use, it dangles and corrupts memory. Keep
        // the NDArray alive (side by side) for the whole lifetime of the returned view.
        //
        // They span the array's LOGICAL C-order elements ([offset, offset+size)), which requires the
        // array to be C-contiguous (an offset slice IS C-contiguous — its Storage.Address is
        // re-seated). A transposed / F-contiguous / strided / broadcast array has no single
        // contiguous memory region for its logical elements, so the Span/Memory/Bytes accessors throw
        // (and the TryGet* forms return false) — copy first with np.ascontiguousarray(a) / a.copy(),
        // or use a strided walk (np.nditer<T>), or the whole-buffer Unsafe.AsSpan<T>() (UnmanagedSpan,
        // which is long-indexable but also NumSharp-typed). The raw Pointer<T>() works for any layout
        // (it points at logical element 0; you walk it yourself with nd.strides).
        public readonly unsafe partial struct _Unsafe
        {
            // ---- guards -------------------------------------------------------------------------

            private void RequireDtype<T>() where T : unmanaged
            {
                if (_this.typecode != InfoOf<T>.NPTypeCode)
                    throw new ArgumentException(
                        $"nd.Unsafe view requested as {typeof(T).Name} but the array's dtype is " +
                        $"{_this.dtype.type.Name}. A Span/Memory over the raw buffer cannot convert; reinterpreting " +
                        $"the bytes would be wrong. Cast first (a.astype(typeof({typeof(T).Name}))), use a.view(dtype) " +
                        $"for a same-size reinterpret, or take a raw Pointer<T>().");
            }

            private void RequireContiguousSpannable<T>() where T : unmanaged
            {
                RequireDtype<T>();
                if (!_this.Shape.IsContiguous)
                    throw new InvalidOperationException(
                        "nd.Unsafe.Span/Memory requires a C-contiguous array (a transposed / F-contiguous / strided / " +
                        "broadcast view has no single contiguous region for its logical elements). Call " +
                        "np.ascontiguousarray(a) or a.copy() first, or walk it with np.nditer<T>. The raw Pointer<T>() " +
                        "works for any layout.");
                if (_this.size > int.MaxValue)
                    throw new InvalidOperationException(
                        $"nd.Unsafe.Span/Memory<{typeof(T).Name}> cannot represent {_this.size} elements — a Span<T>/" +
                        $"Memory<T> length is a 32-bit int. Use Unsafe.AsSpan<{typeof(T).Name}>() (a long-indexable " +
                        $"UnmanagedSpan) or Pointer<{typeof(T).Name}>() for arrays larger than int.MaxValue.");
            }

            // Pointer to the array's logical element 0 for ANY layout:
            //   Storage.Address is the buffer base for offset 0 of this Shape (a contiguous slice
            //   re-seats it and keeps Shape.offset == 0; a strided view keeps the base + a non-zero
            //   offset), so base + offset*itemsize is right for both — the documented NDArray rule.
            private T* LogicalStart<T>() where T : unmanaged
                => (T*)_this.Storage.Address + _this.Shape.offset;

            // ---- Span<T> / ReadOnlySpan<T> ------------------------------------------------------

            /// <summary>
            ///     A <see cref="System.Span{T}"/> aliasing the array's logical C-order elements.
            ///     <b>Unsafe:</b> does not root the buffer — keep the NDArray alive. Requires a
            ///     C-contiguous array of the exact dtype and ≤ <see cref="int.MaxValue"/> elements
            ///     (throws otherwise; see <see cref="TryGetSpan{T}"/> for the non-throwing form).
            /// </summary>
            public Span<T> Span<T>() where T : unmanaged
            {
                RequireContiguousSpannable<T>();
                return new Span<T>(LogicalStart<T>(), (int)_this.size);
            }

            /// <summary>
            ///     A read-only <see cref="System.ReadOnlySpan{T}"/> aliasing the array's logical
            ///     C-order elements. <b>Unsafe</b> and C-contiguous, as <see cref="Span{T}()"/>.
            /// </summary>
            public ReadOnlySpan<T> ReadOnlySpan<T>() where T : unmanaged
            {
                RequireContiguousSpannable<T>();
                return new ReadOnlySpan<T>(LogicalStart<T>(), (int)_this.size);
            }

            // ---- Memory<T> / ReadOnlyMemory<T> --------------------------------------------------

            /// <summary>
            ///     A <see cref="System.Memory{T}"/> aliasing the array's logical C-order elements,
            ///     backed by an <see cref="UnmanagedMemoryManager{T}"/>. <b>Unsafe:</b> the manager
            ///     holds only a raw pointer, so the returned <c>Memory</c> keeps the manager alive but
            ///     NOT the NDArray — keep the NDArray alive for the <c>Memory</c>'s whole lifetime.
            ///     Requires a C-contiguous array of the exact dtype, ≤ <see cref="int.MaxValue"/>
            ///     elements. Allocates the manager (one small object); the buffer is not copied.
            /// </summary>
            public Memory<T> Memory<T>() where T : unmanaged
            {
                RequireContiguousSpannable<T>();
                return new UnmanagedMemoryManager<T>(LogicalStart<T>(), (int)_this.size).Memory;
            }

            /// <summary>
            ///     A <see cref="System.ReadOnlyMemory{T}"/> aliasing the array's logical C-order
            ///     elements. <b>Unsafe</b>, C-contiguous, manager-backed — as <see cref="Memory{T}()"/>.
            /// </summary>
            public ReadOnlyMemory<T> ReadOnlyMemory<T>() where T : unmanaged
            {
                RequireContiguousSpannable<T>();
                return new UnmanagedMemoryManager<T>(LogicalStart<T>(), (int)_this.size).Memory;
            }

            // ---- raw bytes ----------------------------------------------------------------------

            /// <summary>
            ///     A <see cref="System.Span{T}">Span&lt;byte&gt;</see> over the raw bytes of the
            ///     array's logical C-order window (<c>size * itemsize</c> bytes), dtype-agnostic —
            ///     useful for serialization or bit-level reinterpretation. <b>Unsafe</b> and
            ///     C-contiguous, ≤ <see cref="int.MaxValue"/> bytes.
            /// </summary>
            public Span<byte> Bytes()
            {
                RequireContiguousBytes();
                return new Span<byte>(BytesStart(), checked((int)(_this.size * _this.dtypesize)));
            }

            /// <summary>Read-only counterpart of <see cref="Bytes"/>.</summary>
            public ReadOnlySpan<byte> ReadOnlyBytes()
            {
                RequireContiguousBytes();
                return new ReadOnlySpan<byte>(BytesStart(), checked((int)(_this.size * _this.dtypesize)));
            }

            private void RequireContiguousBytes()
            {
                if (!_this.Shape.IsContiguous)
                    throw new InvalidOperationException(
                        "nd.Unsafe.Bytes requires a C-contiguous array. Call np.ascontiguousarray(a) / a.copy() first.");
                if (_this.size * _this.dtypesize > int.MaxValue)
                    throw new InvalidOperationException(
                        "nd.Unsafe.Bytes cannot represent more than int.MaxValue bytes — use Pointer<T>() with BytesLength.");
            }

            private byte* BytesStart() => _this.Storage.Address + _this.Shape.offset * _this.dtypesize;

            // ---- raw pointer (any layout) -------------------------------------------------------

            /// <summary>
            ///     A raw <c>T*</c> pointer to the array's logical element 0, valid for ANY layout
            ///     (contiguous or not). For a non-contiguous array this is just the first element's
            ///     address — walk the rest yourself using <see cref="NDArray.strides"/> (byte strides).
            ///     <b>Unsafe:</b> does not root the buffer. Requires the exact dtype.
            /// </summary>
            public T* Pointer<T>() where T : unmanaged
            {
                RequireDtype<T>();
                return LogicalStart<T>();
            }

            // ---- non-throwing forms -------------------------------------------------------------

            /// <summary>
            ///     Tries to alias the array's logical C-order elements as a <see cref="System.Span{T}"/>.
            ///     Returns <see langword="false"/> (and <c>default</c>) instead of throwing when the
            ///     dtype does not match, the array is not C-contiguous, or it has more than
            ///     <see cref="int.MaxValue"/> elements. <b>Unsafe</b> aliasing, as <see cref="Span{T}()"/>.
            /// </summary>
            public bool TryGetSpan<T>(out Span<T> span) where T : unmanaged
            {
                if (_this.typecode == InfoOf<T>.NPTypeCode && _this.Shape.IsContiguous && _this.size <= int.MaxValue)
                {
                    span = new Span<T>(LogicalStart<T>(), (int)_this.size);
                    return true;
                }

                span = default;
                return false;
            }

            /// <summary>
            ///     Tries to alias the array's logical C-order elements as a <see cref="System.Memory{T}"/>
            ///     (manager-backed). Returns <see langword="false"/> under the same conditions as
            ///     <see cref="TryGetSpan{T}"/>. <b>Unsafe</b> aliasing, as <see cref="Memory{T}()"/>.
            /// </summary>
            public bool TryGetMemory<T>(out Memory<T> memory) where T : unmanaged
            {
                if (_this.typecode == InfoOf<T>.NPTypeCode && _this.Shape.IsContiguous && _this.size <= int.MaxValue)
                {
                    memory = new UnmanagedMemoryManager<T>(LogicalStart<T>(), (int)_this.size).Memory;
                    return true;
                }

                memory = default;
                return false;
            }
        }
    }
}
