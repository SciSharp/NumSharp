using NumSharp.Backends.Iteration;

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
        //   *reinterpreted* values change (int32 1 -> 0x01000000). As in NumPy, the
        //   swap is the dtype class's copyswapn slot, which knows its own byte-order
        //   unit; its contiguous runs use the width-dispatched SIMD kernels of
        //   Utilities/NDByteOrder.cs.
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
        ///     reversed by the class's <see cref="DTypeMeta.copyswapn"/> (its own byte-order unit), in place (the array
        ///     itself, after NumPy's writeability check) or in a copy in NumPy's <c>ANYORDER</c> layout.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="inplace">True to swap <paramref name="a"/>'s own elements and return it; false for a swapped
        ///     copy.</param>
        /// <returns><paramref name="a"/> when <paramref name="inplace"/>, else the copy.</returns>
        /// <exception cref="ValueError"><paramref name="inplace"/> on a read-only array (<c>array to be byte-swapped is
        ///     read-only</c>).</exception>
        internal static NDArray Byteswap(NDArray a, bool inplace)
        {
            DType descr = a.dtype;
            if (inplace)
            {
                // NumPy: PyArray_FailUnlessWriteable(self, "array to be byte-swapped"), then copyswapn over the array (a
                // StringDType's is the identity: the owner's decision D14).
                if (!a.Shape.IsWriteable)
                    throw new ValueError("array to be byte-swapped is read-only");
                if (a.size > 0)
                    SwapInPlace(a, descr);
                return a;
            }

            // NumPy: PyArray_NewCopy(self, NPY_ANYORDER), then the copy swapped in place - a copy pass THEN a swap pass. A
            // C-contiguous source whose items hold no references fuses both into ONE pass: copyswapn copies and swaps
            // straight into an uninitialized C-contiguous result, halving the memory traffic. (A StringDType copy
            // duplicates its strings through the allocator: the general route below.)
            if (a.Shape.IsContiguous && a.size > 0 && !descr.Meta.HasReferences)
            {
                // The descriptor allocates: a string's width lives on it, not on the storage lane.
                var result = new NDArray(descr, Shape.Adopt(a.Shape.dimensions), fillZeros: false);
                unsafe
                {
                    int itemsize = descr.itemsize;
                    byte* src = (byte*)a.Storage.Address + a.Shape.offset * (long)itemsize;
                    // A fresh result: offset 0, its run disjoint from the source's.
                    descr.Meta.copyswapn(descr, (byte*)result.Storage.Address, itemsize, src, itemsize, a.size, swap: true);
                }
                return result;
            }

            // F-contiguous-and-not-C / strided / broadcast / a StringDType: copy in NumPy's ANYORDER layout (F only for an
            // F-contiguous-and-not-C source, via Shape.Order), then swap the copy in place.
            NDArray copy = a.copy(a.Shape.Order);
            if (copy.size > 0)
                SwapInPlace(copy, descr);
            return copy;
        }

        /// <summary>
        ///     The in-place half of <c>PyArray_Byteswap</c> over any layout: one <see cref="DTypeMeta.copyswapn"/> call with
        ///     no source over a one-segment array's whole run (NumPy's <c>PyArray_ISONESEGMENT</c> branch), else one call per
        ///     inner run of a single-operand EXTERNAL_LOOP iterator in memory order (NumPy walks <c>IterAllButAxis</c>).
        /// </summary>
        /// <param name="a">The array (writeable, not empty).</param>
        /// <param name="descr">Its descriptor, whose class swaps the elements.</param>
        private static unsafe void SwapInPlace(NDArray a, DType descr)
        {
            int itemsize = descr.itemsize;
            // One segment (C- OR F-contiguous): the logical elements occupy a single contiguous byte run of size * itemsize
            // (a column-major layout is still one block), so one call swaps it wholesale.
            if (a.Shape.IsContiguous || a.Shape.IsFContiguous)
            {
                byte* p = (byte*)a.Storage.Address + a.Shape.offset * (long)itemsize;
                descr.Meta.copyswapn(descr, p, itemsize, null, itemsize, a.size, swap: true);
                return;
            }

            // Non-contiguous: a single READWRITE EXTERNAL_LOOP iterator in memory order (K), so a strided array's contiguous
            // inner runs coalesce into large chunks; each chunk is one copyswapn call at the chunk's byte stride (a
            // unit-stride chunk is swapped with SIMD, any other element by element). The kernel is a closure over the
            // descriptor: the iterator's auxdata is a raw pointer, which cannot carry a managed object.
            DTypeMeta meta = descr.Meta;
            NDInnerLoopFunc swapChunk = (dataptrs, strides, count, auxdata) =>
                meta.copyswapn(descr, (byte*)dataptrs[0], strides[0], null, strides[0], count, swap: true);
            var iter = NDIterRef.AdvancedNew(1, new[] { a }, NDIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                new[] { NDIterPerOpFlags.READWRITE });
            try { iter.ForEach(swapChunk); }
            finally { iter.Dispose(); }
        }
    }
}
