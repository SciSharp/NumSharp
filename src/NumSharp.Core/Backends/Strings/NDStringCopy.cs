using System;
using System.Linq;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The copy route of the fixed-width string lane — what <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>
    ///     runs when either operand holds strings. It is NumPy's <c>string_to_string</c> zero-pad copy and its legacy
    ///     <c>S ↔ U</c> casts, executed on the existing numeric machinery through UNIT VIEWS
    ///     (<see cref="UnmanagedStorage.StringUnitView"/>): a string array of shape <c>D</c> is copied as a <c>uint8</c> /
    ///     <c>uint32</c> array of shape <c>D + (chars,)</c>, so broadcasting, every view layout, the overlap guard and the
    ///     SIMD copy / cast kernels are the numeric ones, byte for byte.
    /// </summary>
    /// <remarks>
    ///     <list type="bullet">
    ///       <item><description><b>Same class</b> (<c>S → S</c>, <c>U → U</c>): the first <c>min(src, dst)</c> characters
    ///       are copied and a longer destination is NUL-padded (<c>PyArray_GetStridedZeroPadCopyFn</c>). No decoding —
    ///       any unit, even one that is not a valid code point, moves unchanged.</description></item>
    ///       <item><description><b><c>S → U</c></b>: every SOURCE element is ASCII-checked whole (NumPy decodes the full
    ///       <c>bytes</c> before truncating), then the bytes widen to code points.</description></item>
    ///       <item><description><b><c>U → S</c></b>: every source element is checked whole — a unit above <c>U+10FFFF</c>
    ///       is NumPy's <c>SystemError</c>, a code point at or above 128 its <c>UnicodeEncodeError</c> — then the code
    ///       points narrow to bytes.</description></item>
    ///       <item><description><b>Numbers ↔ text</b> are the computational phase: refused with
    ///       <see cref="NotSupportedException"/>.</description></item>
    ///     </list>
    ///     <para>
    ///     <b>Errors.</b> NumPy converts element by element in the order its raw assignment visits the DESTINATION
    ///     (<c>PyArray_PrepareTwoRawArrayIter</c>: axes by the destination's absolute strides, outermost first; an axis with
    ///     a negative destination stride walked from its end) and raises at the first element that cannot convert, so the
    ///     REPORTED element is that one here too (<see cref="ThrowIfCannotConvert"/>). The difference: NumSharp checks
    ///     before it writes, so a failing cast leaves the destination untouched where NumPy leaves the elements before the
    ///     failing one written.
    ///     </para>
    ///     <para>
    ///     <b>Overlap.</b> A 1-D same-width copy between overlapping views runs NumPy's raw element order
    ///     (<c>raw_array_assign_array</c>: the destination stride made positive, then a backward walk when the source starts
    ///     below the destination inside it), so <c>a[3:19:2] = a[6:14]</c> reads its own writes exactly as NumPy does.
    ///     Every other overlapping copy is copied first, by the unit-view copy's own overlap guard.
    ///     </para>
    /// </remarks>
    internal static unsafe class NDStringCopy
    {
        /// <summary>
        ///     Copies <paramref name="src"/> into <paramref name="dst"/> (broadcasting <paramref name="src"/>) when either is a
        ///     string storage.
        /// </summary>
        /// <param name="dst">The destination storage (writeable).</param>
        /// <param name="src">The source storage (broadcastable to <paramref name="dst"/>; at most its rank).</param>
        /// <exception cref="ValueError">The destination is read-only.</exception>
        /// <exception cref="IncorrectShapeException">The source does not broadcast to the destination.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U → S</c> copy meets a code point at or above 128.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S → U</c> copy meets a byte at or above <c>0x80</c>.</exception>
        /// <exception cref="SystemError">A <c>U → S</c> copy meets a unit above <c>U+10FFFF</c>.</exception>
        /// <exception cref="StringComputationNotSupportedException">One operand is numeric and the destination has elements
        ///     (number ↔ text conversion is not part of this phase). An EMPTY destination converts nothing, so it returns
        ///     after the writeability and broadcast checks, as NumPy's raw assignment does.</exception>
        public static void Copy(UnmanagedStorage dst, UnmanagedStorage src)
        {
            NumSharpException.ThrowIfNotWriteable(dst.Shape);
            // Validate the broadcast at the STRING level so a mismatch reports the arrays' own shapes, not the unit
            // views' (which carry one extra axis).
            if (!ShapesMatch(src.Shape, dst.Shape))
                np.broadcast_to(src.Shape, dst.Shape);
            if (dst.Shape.size == 0)
                return;
            // Numbers <-> text: NumPy reaches the per-element conversion only now (after the shape checks, with an element
            // to convert) — formatting / parsing values is the computational phase.
            if (!dst.IsString || !src.IsString)
                throw MixedKinds(src, dst);

            var dSlice = dst.StringSlice;
            var sSlice = src.StringSlice;

            if (dSlice.IsUnicode == sSlice.IsUnicode)
            {
                // The same class and width over the same dense layout is one block move.
                if (dSlice.ItemSize == sSlice.ItemSize && TryCopyDenseBlock(dst, src))
                    return;
                // NumPy's raw 1-D order for an overlapping same-width copy (reads its own writes, like NumPy).
                if (dSlice.ItemSize == sSlice.ItemSize && TryAssignRaw1D(dst, src))
                    return;
                // A non-overlapping copy over at most two effective axes moves whole elements (any widths of the class).
                if (TryCopyElementGrid(dst, src))
                    return;
                CopyUnits(dst.StringUnitView(), dSlice.Chars, src.StringUnitView(), sSlice.Chars);
                return;
            }

            // S <-> U: NumPy's error for the first element (in NumPy's visiting order) that is not ASCII, before any byte
            // is written; then a plain widen / narrow of the units, which every remaining value survives exactly.
            ThrowIfCannotConvert(dst, src, null);
            CopyUnits(dst.StringUnitView(), dSlice.Chars, src.StringUnitView(), sSlice.Chars);
        }

        /// <summary>
        ///     The conversion half of <see cref="Copy"/> WITHOUT the ASCII check: copies <paramref name="src"/> (broadcast)
        ///     into <paramref name="dst"/>, truncating / NUL-padding and widening / narrowing the units. A unit that is not
        ///     ASCII is carried by the unit cast (a <c>uint32</c> narrowed to <c>uint8</c> keeps its low byte), so the caller
        ///     must have proven that every element it will READ converts (<see cref="ThrowIfCannotConvert"/>) — the masked
        ///     copy stages every source element but writes only the selected ones.
        /// </summary>
        /// <param name="dst">A fresh, writeable string storage (no overlap with <paramref name="src"/>).</param>
        /// <param name="src">A string storage broadcastable to <paramref name="dst"/>.</param>
        internal static void CopyUnchecked(UnmanagedStorage dst, UnmanagedStorage src)
            => CopyUnits(dst.StringUnitView(), dst.StringSlice.Chars, src.StringUnitView(), src.StringSlice.Chars);

        /// <summary>
        ///     NumPy's error for an <c>S ↔ U</c> conversion that cannot run, raised before any byte is written: the FIRST
        ///     element whose text is not ASCII, in the order NumPy's raw assignment visits the destination — the axes sorted by
        ///     the destination's absolute strides (outermost first, ties by ascending axis: <c>PyArray_CreateSortedStridePerm</c>)
        ///     and an axis with a negative destination stride walked from its end — counting only the elements
        ///     <paramref name="where"/> selects. NumPy converts element by element in exactly that order and raises at the
        ///     first failure, so the error names the same element and character position.
        /// </summary>
        /// <param name="dst">The destination string storage (its shape and strides give the order).</param>
        /// <param name="src">The source string storage of the other class, broadcastable to <paramref name="dst"/>.</param>
        /// <param name="where">A bool mask broadcastable to <paramref name="dst"/>, or null for every element.</param>
        /// <exception cref="UnicodeEncodeError">A selected <c>U</c> element holds a code point at or above 128.</exception>
        /// <exception cref="UnicodeDecodeError">A selected <c>S</c> element holds a byte at or above <c>0x80</c>.</exception>
        /// <exception cref="SystemError">A selected <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        /// <remarks>
        ///     The common case costs one SIMD scan (or one max reduction for a non-contiguous source): when no unit of any
        ///     DISTINCT source element exceeds <c>0x7F</c> nothing can fail. Only then are the per-element verdicts computed —
        ///     a max over each element's units, broadcast to the destination, masked, and searched in NumPy's visiting order
        ///     by viewing the verdicts in that order (transpose + flips) and taking the flat <c>argmax</c>.
        /// </remarks>
        internal static void ThrowIfCannotConvert(UnmanagedStorage dst, UnmanagedStorage src, NDArray where)
        {
            var sSlice = src.StringSlice;
            int chars = sSlice.Chars;
            bool unicode = sSlice.IsUnicode;
            var distinct = DistinctElements(src);
            if (!AnyNonAscii(distinct, chars, unicode))
                return;

            // Per-element verdict over the distinct elements: the largest unit of each element above 0x7F.
            using var units = new NDArray(distinct.StringUnitView());
            using var elementMax = np.amax(units, -1);
            using var bad = unicode ? elementMax > 0x7Fu : elementMax > (byte)0x7F;

            var dstShape = dst.Shape;
            int nd = dstShape.NDim;
            using var spread = np.broadcast_to(bad, new Shape((long[])dstShape.dimensions.Clone()));
            using var hit = where is null ? spread.view() : np.logical_and(spread, where);

            // NumPy's visiting order: axes by |dst stride| descending (stable: ties keep ascending axis), negative strides
            // walked backward. Viewing the verdicts in that order makes the flat C-order argmax the first visited hit.
            long[] dstStrides = dstShape.strides;
            int[] perm = Enumerable.Range(0, nd).OrderByDescending(d => Math.Abs(dstStrides[d])).ToArray();
            NDArray ordered = nd > 1 ? np.transpose(hit, perm) : hit.view();
            try
            {
                for (int j = 0; j < nd; j++)
                {
                    if (dstStrides[perm[j]] < 0)
                    {
                        var flipped = np.flip(ordered, j);
                        ordered.Dispose();
                        ordered = flipped;
                    }
                }

                long first = nd == 0 ? 0 : np.argmax(ordered);
                // Unravel over the ordered view, then undo the flips and the permutation: the destination coordinate.
                var dims = ordered.Shape.dimensions;
                var dstCoord = new long[nd];
                long rest = first;
                for (int j = nd - 1; j >= 0; j--)
                {
                    long c = rest % dims[j];
                    rest /= dims[j];
                    int axis = perm[j];
                    dstCoord[axis] = dstStrides[axis] < 0 ? dims[j] - 1 - c : c;
                }
                if (!hit.GetBoolean(dstCoord))
                    return; // every element that cannot convert is masked off

                // The source element read at that destination coordinate (right-aligned; a broadcast axis reads index 0).
                var sShape = src.Shape;
                int sn = sShape.NDim;
                long offset = sShape.offset;
                for (int d = 0; d < sn; d++)
                {
                    long c = dstCoord[nd - sn + d];
                    if (sShape.dimensions[d] != 1)
                        offset += c * sShape.strides[d];
                }

                byte* element = src.Address + offset * (long)sSlice.ItemSize;
                if (unicode)
                    NDStringCodec.ThrowIfNotAscii((uint*)element, chars);
                else
                    NDStringCodec.ThrowIfNotAscii(new ReadOnlySpan<byte>(element, chars));
            }
            finally
            {
                ordered.Dispose();
            }
        }

        /// <summary>
        ///     The source with every broadcast axis (stride 0, extent above 1) collapsed to extent 1: each DISTINCT element once,
        ///     so a scan over it reads no element twice and a broadcast of a huge destination costs nothing.
        /// </summary>
        /// <param name="src">A string storage.</param>
        /// <returns><paramref name="src"/> itself when it repeats nothing, else an alias with the collapsed extents.</returns>
        private static UnmanagedStorage DistinctElements(UnmanagedStorage src)
        {
            var shape = src.Shape;
            long[] dims = shape.dimensions, strides = shape.strides;
            bool repeats = false;
            for (int d = 0; d < dims.Length; d++)
            {
                if (strides[d] == 0 && dims[d] > 1)
                {
                    repeats = true;
                    break;
                }
            }
            if (!repeats)
                return src;

            var compact = (long[])dims.Clone();
            for (int d = 0; d < compact.Length; d++)
                if (strides[d] == 0)
                    compact[d] = 1;
            var view = new Shape(compact, (long[])strides.Clone(), shape.offset, shape.bufferSize);
            if (!shape.IsWriteable && view.IsWriteable)
                view = view.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            return src.Alias(view);
        }

        /// <summary>
        ///     Whether any unit of <paramref name="src"/> exceeds <c>0x7F</c> — one SIMD scan over a contiguous source, one
        ///     max reduction over the unit view of any other layout.
        /// </summary>
        /// <param name="src">A string storage (no broadcast axes: <see cref="DistinctElements"/>).</param>
        /// <param name="chars">Units per element.</param>
        /// <param name="unicode">True for <c>U</c> (<c>uint32</c> units), false for <c>S</c> (bytes).</param>
        /// <returns>True when at least one unit is above <c>0x7F</c>.</returns>
        private static bool AnyNonAscii(UnmanagedStorage src, int chars, bool unicode)
        {
            var shape = src.Shape;
            if (shape.size == 0)
                return false;
            if (shape.IsContiguous)
            {
                byte* first = src.Address + shape.offset * (long)src.StringSlice.ItemSize;
                long total = shape.size * chars;
                return unicode ? FirstAbove((uint*)first, total, 0x7Fu) >= 0 : FirstAbove(first, total, (byte)0x7F) >= 0;
            }

            using var units = new NDArray(src.StringUnitView());
            using var max = np.amax(units);
            return unicode ? max.GetAtIndex<uint>(0) > 0x7F : max.GetAtIndex<byte>(0) > 0x7F;
        }

        /// <summary>
        ///     The unit-view copy with NumPy's zero-pad rule: the first <c>min(dstChars, srcChars)</c> units of every element
        ///     are copied (cast between <c>uint8</c> and <c>uint32</c> when the classes differ — every unit the caller reads is
        ///     ASCII by then, so the cast is exact), and when the destination is wider its remaining units are zeroed.
        /// </summary>
        /// <param name="dstUnits">The destination's unit view.</param>
        /// <param name="dstChars">Characters per destination element.</param>
        /// <param name="srcUnits">The source's unit view.</param>
        /// <param name="srcChars">Characters per source element.</param>
        private static void CopyUnits(UnmanagedStorage dstUnits, int dstChars, UnmanagedStorage srcUnits, int srcChars)
        {
            if (dstChars == srcChars)
            {
                NDIter.Copy(dstUnits, srcUnits);
                return;
            }

            int common = Math.Min(dstChars, srcChars);
            NDIter.Copy(LastAxisRange(dstUnits, 0, common), LastAxisRange(srcUnits, 0, common));
            if (dstChars > srcChars)
            {
                // NUL-pad the rest of every destination element: a scalar-broadcast fill of the trailing units.
                var tail = LastAxisRange(dstUnits, srcChars, dstChars - srcChars);
                var zero = dstUnits.TypeCode == NPTypeCode.UInt32 ? UnmanagedStorage.Scalar(0u) : UnmanagedStorage.Scalar((byte)0);
                NDIter.Copy(tail, zero);
            }
        }

        /// <summary>
        ///     A view of <paramref name="units"/> restricted to <paramref name="count"/> units starting at unit
        ///     <paramref name="start"/> of every element (the trailing axis): the same strides, a shorter last axis, the
        ///     offset advanced by <paramref name="start"/> (the last axis has stride 1).
        /// </summary>
        /// <param name="units">A unit view (trailing axis = characters, stride 1).</param>
        /// <param name="start">The first unit kept.</param>
        /// <param name="count">The units kept.</param>
        /// <returns>The narrowed view (shares memory).</returns>
        private static UnmanagedStorage LastAxisRange(UnmanagedStorage units, long start, long count)
        {
            var shape = units.Shape;
            var dims = (long[])shape.dimensions.Clone();
            dims[dims.Length - 1] = count;
            var narrowed = new Shape(dims, (long[])shape.strides.Clone(), shape.offset + start, shape.bufferSize);
            if (!shape.IsWriteable && narrowed.IsWriteable)
                narrowed = narrowed.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            return units.Alias(narrowed);
        }

        /// <summary>
        ///     The copy between two string storages of the SAME class and width that lay their elements out identically:
        ///     equal dimensions (no broadcast) and both C-contiguous, or both F-contiguous. Their element grids then coincide
        ///     byte for byte, so the whole copy is one block move — no unit views, no iterator.
        /// </summary>
        /// <param name="dst">The destination string storage (writeable; same descriptor as <paramref name="src"/>).</param>
        /// <param name="src">The source string storage.</param>
        /// <returns>True when the copy was performed here; false (nothing done) for any other layout pair.</returns>
        /// <remarks>
        ///     <see cref="Buffer.MemoryCopy(void*, void*, long, long)"/> has memmove semantics, which is exactly NumPy's answer for an OVERLAPPING pair
        ///     of the same layout: a multi-dimensional copy is copied first (<c>raw_array_assign_array</c> sees the overlap),
        ///     and a 1-D one walks away from the overlap — either way every element reads its value from before the copy.
        /// </remarks>
        private static bool TryCopyDenseBlock(UnmanagedStorage dst, UnmanagedStorage src)
        {
            var dShape = dst.Shape;
            var sShape = src.Shape;
            if (!ShapesMatch(sShape, dShape))
                return false; // a broadcast source
            bool sameLayout = (dShape.IsContiguous && sShape.IsContiguous) || (dShape.IsFContiguous && sShape.IsFContiguous);
            if (!sameLayout)
                return false;

            long itemsize = dst.StringSlice.ItemSize;
            long bytes = dShape.size * itemsize;
            byte* d = dst.Address + dShape.offset * itemsize;
            byte* s = src.Address + sShape.offset * itemsize;
            if (d != s)
                Buffer.MemoryCopy(s, d, bytes, bytes);
            return true;
        }

        /// <summary>
        ///     The same-class copy (<c>S → S</c>, <c>U → U</c>, any widths) of two storages that do not share memory, when the
        ///     destination's axes of extent above 1 reduce to at most TWO after merging the axes both operands continue
        ///     contiguously: one <see cref="DirectILKernelGenerator.GetElementCopyKernel"/> call moves whole elements
        ///     (truncating or NUL-padding to the destination width) over that 2-D byte-strided grid — a 1-D strided or reversed
        ///     view, a transposed matrix, a contiguous array of another width, a broadcast source (stride 0).
        /// </summary>
        /// <param name="dst">The destination string storage (writeable; the broadcast already validated).</param>
        /// <param name="src">The source string storage of the same class, broadcastable to <paramref name="dst"/>.</param>
        /// <returns>True when the copy was performed here; false (nothing done) when the operands may overlap, the grid has
        ///     more than two effective axes, or IL generation is unavailable — the unit route then runs.</returns>
        /// <remarks>
        ///     The unit route (a <c>uint8</c> / <c>uint32</c> view with one extra character axis) runs each element as a short
        ///     inner row of the numeric iterator; for narrow strided rows that per-row cost dominated (a stepped <c>&lt;U8</c>
        ///     copy, a <c>U8 → U4</c> cast at 0.6x NumPy). Overlapping operands keep NumPy's order through
        ///     <see cref="TryAssignRaw1D"/> or the unit route's overlap guard.
        /// </remarks>
        private static bool TryCopyElementGrid(UnmanagedStorage dst, UnmanagedStorage src)
        {
            var dShape = dst.Shape;
            var sShape = src.Shape;
            int nd = dShape.NDim;
            int lead = nd - sShape.NDim; // the source's axes align to the destination's trailing axes
            if (lead < 0 || NDMemOverlap.StoragesMayShareMemory(src, dst))
                return false;

            // The destination's axes of extent > 1, outermost first, with the source's stride on each (0 where the source
            // broadcasts); an axis both operands continue contiguously from the previous one merges into it.
            Span<long> dims = stackalloc long[2];
            Span<long> dStrides = stackalloc long[2];
            Span<long> sStrides = stackalloc long[2];
            int k = 0;
            for (int axis = 0; axis < nd; axis++)
            {
                long extent = dShape.dimensions[axis];
                if (extent == 1)
                    continue;
                int sAxis = axis - lead;
                long sStride = sAxis < 0 || sShape.dimensions[sAxis] == 1 ? 0 : sShape.strides[sAxis];
                long dStride = dShape.strides[axis];
                if (k > 0 && dStrides[k - 1] == dStride * extent && sStrides[k - 1] == sStride * extent)
                {
                    dims[k - 1] *= extent;
                    dStrides[k - 1] = dStride;
                    sStrides[k - 1] = sStride;
                    continue;
                }
                if (k == 2)
                    return false; // three independent axes: the unit route
                dims[k] = extent;
                dStrides[k] = dStride;
                sStrides[k] = sStride;
                k++;
            }

            long dW = dst.StringSlice.ItemSize;
            long sW = src.StringSlice.ItemSize;
            int copy = (int)Math.Min(dW, sW);
            int zero = (int)Math.Max(0, dW - sW);
            var kernel = DirectILKernelGenerator.GetElementCopyKernel(copy, zero);
            if (kernel is null)
                return false;

            long rows = k == 2 ? dims[0] : 1;
            long cols = k == 0 ? 1 : dims[k - 1];
            long dRow = k == 2 ? dStrides[0] * dW : 0;
            long sRow = k == 2 ? sStrides[0] * sW : 0;
            long dCol = k == 0 ? 0 : dStrides[k - 1] * dW;
            long sCol = k == 0 ? 0 : sStrides[k - 1] * sW;
            kernel(dst.Address + dShape.offset * dW, src.Address + sShape.offset * sW, rows, cols, dRow, dCol, sRow, sCol);
            return true;
        }

        /// <summary>
        ///     NumPy's raw 1-D assignment order for an overlapping same-width string copy — the case
        ///     <c>raw_array_assign_array</c> does NOT copy first. Returns false (nothing done) when it does not apply: a
        ///     multi-dimensional destination, an opposite-direction pair, a broadcast or 1-element source, or no overlap.
        /// </summary>
        /// <param name="dst">The destination string storage.</param>
        /// <param name="src">The source string storage (same class and width).</param>
        /// <returns>True when the copy was performed here.</returns>
        /// <remarks>
        ///     The walk itself is <see cref="DirectILKernelGenerator.AssignRaw1D"/> (the masked-assign walker under a constant
        ///     true mask, lane by lane for a string element): it flips a negative destination stride and walks backward when
        ///     the source starts below the destination inside it, exactly as NumPy does.
        /// </remarks>
        private static bool TryAssignRaw1D(UnmanagedStorage dst, UnmanagedStorage src)
        {
            var dShape = dst.Shape;
            if (dShape.NDim != 1 || dShape.dimensions[0] <= 1)
                return false;
            long n = dShape.dimensions[0];

            // The source's effective 1-D stride: leading axes of extent 1, a last axis of extent n (or a broadcast).
            var sShape = src.Shape;
            int sn = sShape.NDim;
            if (sn == 0)
                return false;
            for (int axis = 0; axis < sn - 1; axis++)
                if (sShape.dimensions[axis] != 1)
                    return false;
            long last = sShape.dimensions[sn - 1];
            if (last != n)
                return false; // a broadcast (or invalid) source: NumPy's general path
            long ss = sShape.strides[sn - 1];
            if (ss == 0)
                return false;

            long ds = dShape.strides[0];
            // Opposite directions are copied first (NeedsAssignmentTemp); a same-direction unit-stride pair is a memmove,
            // which the unit copy already reproduces.
            if (ds * ss < 0 || (ds == 1 && ss == 1))
                return false;
            if (!NDMemOverlap.StoragesMayShareMemory(src, dst))
                return false;

            int itemsize = dst.StringSlice.ItemSize;
            byte* d = dst.Address + dShape.offset * (long)itemsize;
            byte* s = src.Address + sShape.offset * (long)itemsize;
            DirectILKernelGenerator.AssignRaw1D(d, ds * itemsize, s, ss * itemsize, n, itemsize);
            return true;
        }

        /// <summary>The index of the first unit above <paramref name="max"/> in a contiguous run (SIMD, in int-sized chunks).</summary>
        /// <typeparam name="T">The unit type (<see cref="byte"/> or <see cref="uint"/>).</typeparam>
        /// <param name="p">The first unit.</param>
        /// <param name="length">The run length in units.</param>
        /// <param name="max">The largest accepted unit value.</param>
        /// <returns>The index, or -1 when every unit is accepted.</returns>
        private static long FirstAbove<T>(T* p, long length, T max) where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            long done = 0;
            while (done < length)
            {
                int chunk = (int)Math.Min(length - done, int.MaxValue);
                int hit = new ReadOnlySpan<T>(p + done, chunk).IndexOfAnyExceptInRange(default, max);
                if (hit >= 0)
                    return done + hit;
                done += chunk;
            }
            return -1;
        }

        /// <summary>True when two shapes have identical dimensions (no broadcast needed).</summary>
        /// <param name="a">One shape.</param>
        /// <param name="b">The other shape.</param>
        /// <returns>True for equal ranks and extents.</returns>
        private static bool ShapesMatch(Shape a, Shape b)
        {
            var da = a.dimensions;
            var db = b.dimensions;
            if (da.Length != db.Length)
                return false;
            for (int i = 0; i < da.Length; i++)
                if (da[i] != db[i])
                    return false;
            return true;
        }

        /// <summary>The refusal for a copy between a string and a numeric storage.</summary>
        /// <param name="src">The source storage.</param>
        /// <param name="dst">The destination storage.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static StringComputationNotSupportedException MixedKinds(UnmanagedStorage src, UnmanagedStorage dst)
            => new StringComputationNotSupportedException(
                $"Cannot copy {src.Descr.ToString(true)} data into a {dst.Descr.ToString(true)} array: converting between numbers and " +
                "text is not part of NumSharp's fixed-width string storage phase yet (string arrays copy only between string dtypes).");
    }
}
