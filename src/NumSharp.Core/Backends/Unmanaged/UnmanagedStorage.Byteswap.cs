using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // NumPy's PyArray_Byteswap (methods.c), the body of ndarray.byteswap: the array object's own member (the np-layer dtype
    // plan's D1); NDArray.byteswap is the facade over it.
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class UnmanagedStorage
    {
        // =====================================================================
        // NDArray.byteswap — port of NumPy 2.4.2's ndarray.byteswap / PyArray_Byteswap
        // (numpy/_core/src/multiarray/methods.c).
        //
        //   Reverse the bytes WITHIN each element, toggling the low/big-endian
        //   representation. The dtype is unchanged — only the raw bytes move, so the
        //   *reinterpreted* values change (int32 1 -> 0x01000000). It is a pure
        //   byte permutation: it depends ONLY on the element's byte width, never on
        //   the dtype's numeric meaning, so ONE width-dispatched SIMD kernel covers
        //   every dtype instead of a per-NPTypeCode path (Utilities/NDByteOrder.cs,
        //   shared with the dtype classes' copyswapn slot).
        //
        //   The "swap unit" is the element size, EXCEPT complex, whose real and
        //   imaginary halves are swapped independently (NumPy's copyswapn for a
        //   complex dtype swaps per component: '<c16' is two 8-byte doubles, not one
        //   16-byte blob). 1-byte dtypes (bool/i1/u1) are a value no-op.
        //
        //   NumSharp-only dtypes have no NumPy analog, so they take the consistent
        //   raw-itemsize rule: Char (2 bytes) swaps in 2s, Decimal (16 bytes) in 16s.
        //
        //   Fixed-width strings swap per CHARACTER, as NumPy's UNICODE/STRING copyswapn
        //   do: a 'U' element is N four-byte code points, each reversed on its own
        //   (the result holds code points NumPy itself cannot decode); an 'S' element
        //   is N single bytes, so it is a value no-op (still a copy unless inplace).
        //
        //   A NEP 55 StringDType ('T') element has NO byte order (its descriptor's
        //   byteorder is '|': a cell holds UTF-8 text, an arena offset or a heap
        //   pointer that belong to its buffer's allocator). NumPy 2.4.2 CRASHES the
        //   process here (its copyswapn for StringDType is not usable), so NumSharp
        //   takes the owner's decision D14 (docs/plans/strings-phase1-completion.md):
        //   the swap is the identity, so inplace=True leaves the cells alone and
        //   returns self (after NumPy's writeable check), and inplace=False returns
        //   PyArray_NewCopy(self, NPY_ANYORDER), the step NumPy runs before it crashes.
        //   Swapping a cell's bytes would corrupt its offset or pointer.
        //
        //   inplace=False (default): return a fresh COPY (never shares memory) in
        //     NumPy's PyArray_NewCopy(ANYORDER) layout — F iff the source is
        //     F-contiguous-and-not-C, else C — then swap it. Even a 1-byte dtype
        //     returns a copy.
        //   inplace=True: swap the viewed elements in place and return self. Requires
        //     a writeable array, else ValueError (a broadcast/read-only view raises).
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>PyArray_Byteswap</c> (<c>methods.c</c>), the body of <c>ndarray.byteswap</c>: each element's bytes
        ///     reversed, in place (the array itself, after NumPy's writeability check) or in a copy in NumPy's
        ///     <c>ANYORDER</c> layout.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="inplace">True to swap <paramref name="a"/>'s own elements and return it; false for a swapped
        ///     copy.</param>
        /// <returns><paramref name="a"/> when <paramref name="inplace"/>, else the copy.</returns>
        /// <exception cref="ValueError"><paramref name="inplace"/> on a read-only array (<c>array to be byte-swapped is
        ///     read-only</c>).</exception>
        internal static NDArray Byteswap(NDArray a, bool inplace)
        {
            // A StringDType cell is text in its buffer's allocator, never a byte-ordered number: the swap is the identity.
            // The writeable check runs first, as NumPy's PyArray_FailUnlessWriteable does before the swap it cannot make.
            if (a.Storage.IsVString)
            {
                if (!inplace)
                    return a.copy(a.Shape.Order);
                if (!a.Shape.IsWriteable)
                    throw new ValueError("array to be byte-swapped is read-only");
                return a;
            }

            int itemsize = a.itemsize;
            // Complex swaps each component (real/imag) independently — half the itemsize per unit; a string swaps
            // each character (4 bytes for 'U', 1 for 'S').
            int unit = a.typecode == NPTypeCode.Complex ? itemsize / 2
                : a.typecode == NPTypeCode.String ? ((StringDTypeMeta)a.dtype.Meta).UnitSize
                : itemsize;

            if (inplace)
            {
                // NumPy: PyArray_FailUnlessWriteable(self, "array to be byte-swapped").
                if (!a.Shape.IsWriteable)
                    throw new ValueError("array to be byte-swapped is read-only");

                if (unit > 1 && a.size > 0)
                    ByteswapInPlace(a, unit, itemsize);
                return a;
            }

            // NumPy: PyArray_NewCopy(self, NPY_ANYORDER) then byteswap the copy in place — i.e. a
            // copy pass THEN a swap pass. When the source is C-contiguous (the common case) both
            // fuse into ONE pass: allocate an uninitialized C-contiguous result and read→shuffle→
            // write straight across, halving the memory traffic and skipping the copy machinery.
            // A 1-byte dtype has no bytes to swap, so it is a straight contiguous copy.
            if (a.Shape.IsContiguous && a.size > 0)
            {
                // The descriptor allocates: a string's width lives on it, not on the storage lane.
                var result = new NDArray(a.dtype, Shape.Adopt(a.Shape.dimensions), fillZeros: false);
                unsafe
                {
                    byte* src = (byte*)a.Storage.Address + a.Shape.offset * (long)itemsize;
                    byte* dst = (byte*)result.Storage.Address;   // fresh -> offset 0
                    long bytes = (long)a.size * itemsize;
                    if (unit > 1) NDByteOrder.CopyReverseRun(src, dst, bytes, unit);
                    else Buffer.MemoryCopy(src, dst, bytes, bytes);
                }
                return result;
            }

            // F-contiguous-and-not-C / strided / broadcast / empty: copy in NumPy's ANYORDER layout
            // (F only for an F-contiguous-and-not-C source, via Shape.Order), then swap it.
            NDArray fallback = a.copy(a.Shape.Order);
            if (unit > 1 && fallback.size > 0)
            {
                unsafe
                {
                    byte* p = (byte*)fallback.Storage.Address + fallback.Shape.offset * (long)itemsize;
                    NDByteOrder.ReverseRun(p, (long)fallback.size * itemsize, unit);
                }
            }
            return fallback;
        }

        /// <summary>In-place byteswap over any layout. One-segment arrays swap their whole contiguous
        /// byte block; strided/transposed/reversed views drive a single-operand EXTERNAL_LOOP NDIter
        /// (memory order) whose kernel swaps each chunk — NumPy's IterAllButAxis inplace path.</summary>
        /// <param name="a">The array (writeable, not empty).</param>
        /// <param name="unit">The byte-order unit: the item size, half of it for complex, a U element's code unit.</param>
        /// <param name="itemsize">The item size in bytes.</param>
        private static unsafe void ByteswapInPlace(NDArray a, int unit, int itemsize)
        {
            // One segment (C- OR F-contiguous): the logical elements occupy a single contiguous byte
            // run of size*itemsize (column-major layout is still one block), so swap it wholesale.
            if (a.Shape.IsContiguous || a.Shape.IsFContiguous)
            {
                byte* p = (byte*)a.Storage.Address + a.Shape.offset * (long)itemsize;
                NDByteOrder.ReverseRun(p, (long)a.size * itemsize, unit);
                return;
            }

            // Non-contiguous: visit every element via a single READWRITE EXTERNAL_LOOP iterator in
            // memory order (K), so a strided array's contiguous inner runs coalesce into large chunks.
            var ctx = new SwapCtx { itemsize = itemsize, unit = unit };
            var iter = NDIterRef.AdvancedNew(1, new[] { a }, NDIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                new[] { NDIterPerOpFlags.READWRITE });
            try { iter.ForEach(ByteswapStridedKernel, &ctx); }
            finally { iter.Dispose(); }
        }

        private struct SwapCtx { public int itemsize; public int unit; }

        // NDInnerLoopFunc: swap one inner-loop chunk of `count` elements at byte stride strides[0].
        // A unit-stride chunk (the coalesced inner axis) is a contiguous block -> SIMD; any other
        // stride (non-unit / negative) reverses each element's bytes individually.
        private static unsafe void ByteswapStridedKernel(void** dataptrs, long* strides, long count, void* auxdata)
        {
            byte* data = (byte*)dataptrs[0];
            long s = strides[0];
            SwapCtx* c = (SwapCtx*)auxdata;
            int itemsize = c->itemsize, unit = c->unit;

            if (s == itemsize)
            {
                NDByteOrder.ReverseRun(data, count * itemsize, unit);
                return;
            }
            for (long i = 0; i < count; i++)
                ReverseElement(data + i * s, itemsize, unit);
        }

        /// <summary>Reverse each of the <paramref name="unit"/>-byte components of one strided element
        /// (one group for every dtype but complex, whose two halves reverse independently).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ReverseElement(byte* e, int itemsize, int unit)
        {
            for (int u = 0; u < itemsize; u += unit)
                NDByteOrder.ReverseUnit(e + u, unit);
        }
    }
}
