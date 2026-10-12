using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The reference-aware copy core of NEP 55 <c>StringDType</c> arrays — NumPy's <c>PyArray_AssignArray</c> with the
    ///     <c>string_to_string</c> cast loop of <c>stringdtype/casts.cpp</c>: every element is DUPLICATED into the
    ///     destination's allocator (a short string moves as its 16-byte cell, an arena or heap string gets a new allocation),
    ///     never copied as raw cell bytes, because a cell is meaningful only with its own buffer's allocator.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Semantics</b> (<c>string_to_string</c>). A source cell whose text the destination cell already shares (the same
    ///     allocator and buffer: <c>a[...] = a</c>) is skipped; an NA cell copied from a descriptor WITH an NA object into one
    ///     WITHOUT becomes the source NA object's <c>str()</c> (<c>'None'</c>, <c>'nan'</c>: the lossy unsafe cast); anything
    ///     else is <c>free_and_copy</c> — the destination's old text is freed, the source string duplicated (an NA stays NA).
    ///     </para>
    ///     <para>
    ///     <b>Iteration and overlap</b> (<c>PyArray_AssignArray</c>). The source broadcasts to the destination
    ///     (<c>np.broadcast_to</c>'s errors, as every NumSharp copy). An overlapping source is copied first into a fresh buffer
    ///     with its own allocator when NumPy does (a multi-dimensional destination, or a 1-D one whose strides run against the
    ///     source's); a 1-D destination whose runs go the same way is walked in NumPy's raw order instead (backward when the
    ///     source starts below the destination inside it), reading cells as that walk leaves them. Everything else is walked
    ///     by <see cref="NDIterRef"/> over the storages' cell views (<see cref="UnmanagedStorage.VStringCellView"/>), with the
    ///     allocators locked once for the whole copy (NumPy's <c>NpyString_acquire_allocators</c>).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDVStringCopy
    {
        /// <summary>The operation NumPy names in a copy's <c>MemoryError</c> texts.</summary>
        private const string Location = "string to string cast";

        /// <summary>
        ///     <c>dst[...] = src</c> for two <c>StringDType</c> storages (same descriptor or not): the source broadcasts to the
        ///     destination's shape and every element is copied with NumPy's <c>string_to_string</c> rules.
        /// </summary>
        /// <param name="dst">The destination (a writeable <c>StringDType</c> storage).</param>
        /// <param name="src">The source (a <c>StringDType</c> storage of any layout, broadcastable to <paramref name="dst"/>).</param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError"><paramref name="dst"/> is read-only.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="src"/> does not broadcast to <paramref name="dst"/>
        ///     (<c>np.broadcast_to</c>'s error).</exception>
        /// <exception cref="InvalidOperationException">Either storage is not a <c>StringDType</c> storage (the caller's routing).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>Failed to allocate string in string to string cast</c>.</exception>
        internal static void Copy(UnmanagedStorage dst, UnmanagedStorage src)
        {
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (!dst.IsVString || !src.IsVString)
                throw new InvalidOperationException("The StringDType copy core needs two StringDType storages.");
            // NumPy's text for a read-only destination (NumSharpException.ThrowIfNotWriteable's).
            if (!dst.Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");

            // np.broadcast_to's validation and stride-0 view, as NDIter.Copy's numeric path (CreateCopyState) uses.
            var bshape = ShapesMatch(src.Shape, dst.Shape) ? src.Shape : np.broadcast_to(src.Shape, dst.Shape);
            if (dst.Shape.size == 0)
                return;

            UnmanagedStorage temp = null;
            try
            {
                bool mayOverlap = ReferenceEquals(dst.VStringSlice.Allocator, src.VStringSlice.Allocator)
                                  && NDMemOverlap.StoragesMayShareMemory(src, dst);
                if (mayOverlap)
                {
                    if (NeedsTemp(dst.Shape, bshape))
                    {
                        // PyArray_AssignArray: "Allocate a temporary copy array" — a fresh buffer and allocator holding the
                        // source's strings, then the copy reads from it.
                        temp = src.Clone();
                        src = temp;
                        bshape = ShapesMatch(src.Shape, dst.Shape) ? src.Shape : np.broadcast_to(src.Shape, dst.Shape);
                    }
                    else if (dst.Shape.NDim == 1)
                    {
                        CopyRaw1D(dst, src, bshape);
                        return;
                    }
                }
                CopyIterated(dst, src, bshape);
            }
            finally
            {
                if (temp != null)
                    temp.InternalArray.DangerousFree(); // the temporary's only reference: free it (its clear loop runs) now
            }
        }

        /// <summary>
        ///     NumPy's <c>raw_array_assign_array</c> walk (<see cref="NDVStringCasting.Walk"/>: the axes sorted by the
        ///     DESTINATION's strides, the source broadcast); every innermost run runs the <c>string_to_string</c> body. The
        ///     destination order decides where each duplicated arena string lands in the destination's arena, so a copy's cells
        ///     (<c>tobytes()</c>) are NumPy's: <c>np.copy(c_array, order='F')</c> duplicates in F order.
        /// </summary>
        /// <param name="dst">The destination storage.</param>
        /// <param name="src">The source storage.</param>
        /// <param name="bshape">The source's shape broadcast to <paramref name="dst"/>'s dimensions.</param>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        private static void CopyIterated(UnmanagedStorage dst, UnmanagedStorage src, Shape bshape)
        {
            var inAllocator = src.VStringSlice.Allocator;
            var outAllocator = dst.VStringSlice.Allocator;
            byte[] naName = NAReplacement(src.VStringSlice, dst.VStringSlice);
            // The walk reads raw pointers of both storages, no NDArray wrapper is made (so no reference is counted or
            // released here: the caller keeps both buffers alive for the call).
            var srcView = src.Alias(bshape);
            VStringAllocator.AcquirePair(inAllocator, outAllocator);
            try
            {
                NDVStringCasting.Walk(dst, srcView, (s, ss, d, ds, count) =>
                    CopyRun(s, ss, d, ds, count, inAllocator, outAllocator, naName));
            }
            finally
            {
                VStringAllocator.ReleasePair(inAllocator, outAllocator);
            }
        }

        /// <summary>
        ///     NumPy's 1-D <c>raw_array_assign_array</c> order for an overlapping copy NumPy does not copy first: a negative
        ///     destination stride is flipped (with the source's), then the walk runs backward when the source starts below the
        ///     destination and runs into it — so cells overwritten earlier in the walk are read as they now are.
        /// </summary>
        /// <param name="dst">The 1-D destination.</param>
        /// <param name="src">The source (same allocator).</param>
        /// <param name="bshape">The source's broadcast shape (1-D).</param>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        private static void CopyRaw1D(UnmanagedStorage dst, UnmanagedStorage src, Shape bshape)
        {
            long n = dst.Shape.dimensions[0];
            long ds = dst.Shape.strides[0] * VStringAllocator.CellSize;
            long ss = (bshape.NDim == 0 ? 0 : bshape.strides[bshape.NDim - 1]) * VStringAllocator.CellSize;
            byte* d = dst.Address + dst.Shape.offset * VStringAllocator.CellSize;
            byte* s = src.Address + bshape.offset * VStringAllocator.CellSize;
            // PyArray_PrepareTwoRawArrayIter, ndim 1: "Always make a positive stride for the first operand".
            if (ds < 0)
            {
                d += ds * (n - 1);
                s += ss * (n - 1);
                ds = -ds;
                ss = -ss;
            }
            // raw_array_assign_array: "Overlap check for the 1D case".
            if (s < d && s + n * ss > d)
            {
                s += (n - 1) * ss;
                d += (n - 1) * ds;
                ss = -ss;
                ds = -ds;
            }
            var allocator = dst.VStringSlice.Allocator;
            byte[] naName = NAReplacement(src.VStringSlice, dst.VStringSlice);
            allocator.Acquire();
            try
            {
                CopyRun(s, ss, d, ds, n, allocator, allocator, naName);
            }
            finally
            {
                allocator.Release();
            }
        }

        /// <summary>
        ///     The <c>string_to_string</c> loop over one run of cells (the allocators are locked by the caller).
        /// </summary>
        /// <param name="s">The first source cell.</param>
        /// <param name="sStride">The source byte stride (0 broadcasts one cell).</param>
        /// <param name="d">The first destination cell.</param>
        /// <param name="dStride">The destination byte stride.</param>
        /// <param name="count">The number of cells.</param>
        /// <param name="inAllocator">The source's allocator.</param>
        /// <param name="outAllocator">The destination's allocator (may be <paramref name="inAllocator"/>).</param>
        /// <param name="naName">The UTF-8 <c>str()</c> of the source's NA object when its NA cells must become text (a
        ///     descriptor with an NA object into one without), else null.</param>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static void CopyRun(byte* s, long sStride, byte* d, long dStride, long count,
            VStringAllocator inAllocator, VStringAllocator outAllocator, byte[] naName)
        {
            for (long i = 0; i < count; i++, s += sStride, d += dStride)
            {
                if (VStringAllocator.ShareMemory(s, inAllocator, d, outAllocator))
                    continue;
                if (naName != null && VStringAllocator.IsNull(s))
                {
                    // lossy but this is an unsafe cast so this is OK
                    fixed (byte* name = naName)
                        if (outAllocator.Pack(d, name, naName.Length) < 0)
                            throw new OutOfMemoryException("Failed to pack string in string to string cast.");
                }
                else
                {
                    VStringAllocator.FreeAndCopy(inAllocator, outAllocator, s, d, Location);
                }
            }
        }

        /// <summary>
        ///     The text an NA cell becomes in this copy: the source NA object's <c>str()</c> as UTF-8 when the source descriptor
        ///     has an NA object and the destination's does not (NumPy's <c>in_has_null &amp;&amp; !out_has_null</c>), else null.
        /// </summary>
        /// <param name="src">The source slice.</param>
        /// <param name="dst">The destination slice.</param>
        /// <returns>The UTF-8 bytes, or null when NA cells are copied as NA.</returns>
        internal static byte[] NAReplacement(VStringArraySlice src, VStringArraySlice dst)
            => src.HasNAObject && !dst.HasNAObject ? NDUtf8.Encode(src.Descr.VString.NAName) : null;

        /// <summary>Whether a copy of an overlapping source must go through a temporary (NumPy's <c>PyArray_AssignArray</c>):
        ///     a multi-dimensional destination, or a 1-D one whose stride runs against the source's last stride.</summary>
        /// <param name="dst">The destination shape.</param>
        /// <param name="src">The source shape (broadcast).</param>
        /// <returns>True when NumPy copies the source first.</returns>
        private static bool NeedsTemp(Shape dst, Shape src)
        {
            int dn = (int)dst.NDim;
            if (dn > 1)
                return true;
            int sn = (int)src.NDim;
            return dn == 1 && sn >= 1 && dst.strides[0] * src.strides[sn - 1] < 0;
        }

        /// <summary>Whether two shapes have identical dimensions (no broadcast needed).</summary>
        /// <param name="a">The first shape.</param>
        /// <param name="b">The second shape.</param>
        /// <returns>True for the same rank and extents.</returns>
        private static bool ShapesMatch(Shape a, Shape b)
        {
            if (a.NDim != b.NDim)
                return false;
            for (int i = 0; i < a.NDim; i++)
                if (a.dimensions[i] != b.dimensions[i])
                    return false;
            return true;
        }
    }
}
