using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     Copies between a fixed-width string array (<c>S</c> / <c>U</c>) and a numeric array — the conversions NumPy's
    ///     legacy casts compute (<c>@FROMTYPE@_to_UNICODE</c>, <c>UNICODE_to_@TOTYPE@</c>, ... in <c>arraytypes.c.src</c>):
    ///     a number is written as its scalar's <c>str()</c> (<see cref="NDTextFormat"/>), a text is parsed by CPython's
    ///     <c>int()</c> / <c>float()</c> / <c>complex()</c> (<see cref="NDTextParse"/>), and a text cast to bool is its truth
    ///     value. Every route that moves values between text and numbers ends here: <c>astype</c>, <c>np.copyto</c>,
    ///     assignment, the string copy of <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>NumPy's order and commit blocks.</b> NumPy assigns through <c>raw_array_assign_array</c>: the axes sorted by
    ///     the DESTINATION's absolute strides (a negative destination axis walked from its end, adjacent axes coalesced —
    ///     <c>PyArray_PrepareTwoRawArrayIter</c>, here <see cref="RawArrayIter.PrepareThree"/>), one cast call per
    ///     innermost run. The legacy cast functions are wrapped in buffers of <see cref="BlockSize"/> elements
    ///     (<c>NPY_LOWLEVEL_BUFFER_BLOCKSIZE</c>): each block is converted into a temporary and copied out only when every
    ///     element of it converted. So a parse error at element 200 of a contiguous copy leaves elements 0..127 written and
    ///     128.. untouched (probed against 2.4.2), and the error names the FIRST failing element in that order. This class
    ///     walks the same order with the same blocks, so both the error and what an existing destination holds afterwards
    ///     are NumPy's.
    ///     </para>
    ///     <para>
    ///     <b>Canonical blocks.</b> A block of numbers is first converted to its family's canonical type
    ///     (<see cref="TextNumberKind"/> / <see cref="NDTextParse.Family"/>) by the house cast kernel
    ///     (<see cref="DirectILKernelGenerator.TryGetInnerCastKernel"/>), so the formatter and the parser have one routine
    ///     per family and the per-dtype step is the existing, NumPy-exact cast (exact for a parsed integer, which is range
    ///     checked first; float64 → float32 / float16 rounds once, as NumPy's <c>FLOAT_setitem</c> /
    ///     <c>npy_double_to_half</c> do).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDTextConversion
    {
        /// <summary>
        ///     NumPy's <c>NPY_LOWLEVEL_BUFFER_BLOCKSIZE</c>: the block a wrapped legacy cast converts at a time and commits
        ///     whole. A parse that fails inside a block leaves the whole block unwritten.
        /// </summary>
        internal const int BlockSize = 128;

        /// <summary>The largest canonical value (<see cref="System.Numerics.Complex"/>, 16 bytes).</summary>
        private const int CanonicalMaxBytes = 16;

        /// <summary>
        ///     Copies <paramref name="src"/> (broadcast) into <paramref name="dst"/> where exactly one of them holds text:
        ///     numbers are formatted into a text destination, texts are parsed (or truth-tested for bool) into a numeric
        ///     destination.
        /// </summary>
        /// <param name="dst">The destination: writeable, its shape already checked against <paramref name="src"/>'s, at least
        ///     one element (the caller — <see cref="NDStringCopy.Copy"/> — does those checks in NumPy's order).</param>
        /// <param name="src">The source, broadcastable to <paramref name="dst"/>.</param>
        /// <exception cref="ArgumentException">Both or neither of the storages hold text.</exception>
        /// <exception cref="ValueError">A text is not a number of the destination's kind (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts, quoting <c>np.str_('…')</c> / <c>np.bytes_(b'…')</c>), or an integer
        ///     text has more than 4300 digits.</exception>
        /// <exception cref="OverflowException">An integer text does not fit the destination (NumPy's LP64 OverflowError
        ///     texts).</exception>
        /// <exception cref="SystemError">A <c>U</c> element is one unit above U+10FFFF.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> element converted to complex is not UTF-8.</exception>
        /// <param name="style">How a parse error quotes a text: an array element's NumPy scalar (<c>np.str_('x')</c>, the
        ///     default, NumPy's cast), or the Python object NumPy's setitem received (<c>'x'</c>: a list of strings packed
        ///     into a numeric dtype).</param>
        /// <remarks>On an error the destination keeps the blocks NumPy would have committed before it (see the class
        ///     remarks); the rest is untouched.</remarks>
        internal static void Convert(UnmanagedStorage dst, UnmanagedStorage src, TextReprStyle style = TextReprStyle.ArrayElement)
        {
            if (dst.IsString == src.IsString)
                throw new ArgumentException("A text <-> number conversion needs exactly one string operand.");

            // NumPy's raw assignment copies an overlapping source first (arrays_overlap). Text and numbers share a buffer
            // only through a reinterpreting view (np.frombuffer, view(dtype)), so the guard almost never fires.
            if (MayOverlap(dst, src))
            {
                var copy = src.Clone();
                try
                {
                    Walk(dst, copy, null, style);
                }
                finally
                {
                    // A bare cloned storage was never ARC-initialized (refcount 0): take the one owning reference it
                    // implicitly holds and release it, returning the buffer to the pool now (NDIter.Copy's overlap clone
                    // does the same) instead of leaving it to the finalizer.
                    var block = copy.InternalArray;
                    if (block is not null)
                    {
                        block.TryAddRef();
                        block.Release();
                    }
                }
                return;
            }
            Walk(dst, src, null, style);
        }

        /// <summary>
        ///     The masked form of <see cref="Convert(UnmanagedStorage, UnmanagedStorage)"/>: converts only the elements
        ///     <paramref name="where"/> selects — NumPy's <c>raw_array_wheremasked_assign_array</c>, whose masked transfer
        ///     (<c>_strided_masked_wrapper_transfer_function</c>) calls the wrapped cast once per run of selected elements,
        ///     so an element the mask leaves out is never parsed (it cannot raise) and the 128-element commit blocks start at
        ///     each selected run.
        /// </summary>
        /// <param name="dst">The destination (writeable, shapes already checked, at least one element).</param>
        /// <param name="src">The source, broadcastable to <paramref name="dst"/>.</param>
        /// <param name="where">A bool mask broadcastable to <paramref name="dst"/>, or null for every element.</param>
        /// <exception cref="ArgumentException">Both or neither of <paramref name="dst"/> and <paramref name="src"/> hold
        ///     text, or the mask is not bool.</exception>
        /// <exception cref="ValueError">A selected text is not a number of the destination's kind.</exception>
        /// <exception cref="OverflowException">A selected integer text does not fit the destination.</exception>
        /// <exception cref="SystemError">A selected <c>U</c> element is one unit above U+10FFFF.</exception>
        /// <exception cref="UnicodeDecodeError">A selected <c>S</c> element converted to complex is not UTF-8.</exception>
        internal static void Convert(UnmanagedStorage dst, UnmanagedStorage src, UnmanagedStorage where)
        {
            if (where is null)
            {
                Convert(dst, src);
                return;
            }
            if (dst.IsString == src.IsString)
                throw new ArgumentException("A text <-> number conversion needs exactly one string operand.");
            if (where.TypeCode != NPTypeCode.Boolean)
                throw new ArgumentException($"A where mask is bool, not {where.Descr}.", nameof(where));
            if (MayOverlap(dst, src))
            {
                var copy = src.Clone();
                try
                {
                    Walk(dst, copy, where);
                }
                finally
                {
                    // See Convert(dst, src): the bare clone's one owning reference, taken and released.
                    var block = copy.InternalArray;
                    if (block is not null)
                    {
                        block.TryAddRef();
                        block.Release();
                    }
                }
                return;
            }
            Walk(dst, src, where);
        }

        /// <summary>Whether the byte ranges <paramref name="a"/> and <paramref name="b"/> address can intersect.</summary>
        /// <param name="a">One storage.</param>
        /// <param name="b">The other.</param>
        /// <returns>True when their extents overlap (a conservative bounds test, NumPy's <c>NPY_MAY_SHARE_BOUNDS</c>).</returns>
        private static bool MayOverlap(UnmanagedStorage a, UnmanagedStorage b)
        {
            Extent(a, out byte* aLo, out byte* aHi);
            Extent(b, out byte* bLo, out byte* bHi);
            return aLo < bHi && bLo < aHi;
        }

        /// <summary>The half-open byte range [lo, hi) a storage's elements occupy.</summary>
        /// <param name="s">The storage.</param>
        /// <param name="lo">Receives the lowest byte.</param>
        /// <param name="hi">Receives one past the highest byte.</param>
        private static void Extent(UnmanagedStorage s, out byte* lo, out byte* hi)
        {
            var shape = s.Shape;
            long item = ItemSize(s);
            byte* first = s.Address + shape.offset * item;
            lo = first;
            hi = first + item;
            for (int d = 0; d < shape.NDim; d++)
            {
                long span = (shape.dimensions[d] - 1) * shape.strides[d] * item;
                if (shape.dimensions[d] == 0)
                {
                    hi = lo;
                    return;
                }
                if (span < 0)
                    lo += span;
                else
                    hi += span;
            }
        }

        /// <summary>The element size of a storage in bytes (a string's descriptor width, a number's type size).</summary>
        /// <param name="s">The storage.</param>
        /// <returns>The size.</returns>
        private static int ItemSize(UnmanagedStorage s) => s.IsString ? s.StringSlice.ItemSize : s.TypeCode.SizeOf();

        /// <summary>
        ///     The raw assignment walk: NumPy's axis order and coalescing over the destination's strides, one
        ///     <see cref="FormatRun"/> / <see cref="ParseRun"/> call per innermost run (per run of selected elements under a
        ///     mask).
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="src">The source (not overlapping <paramref name="dst"/>). A source of higher rank than the
        ///     destination has extent-1 leading axes (the caller's broadcast check proved it); they are dropped, as NumPy's
        ///     assignment strips them.</param>
        /// <param name="mask">A bool mask broadcastable to <paramref name="dst"/>, or null.</param>
        /// <param name="style">How a parse error quotes a text.</param>
        private static void Walk(UnmanagedStorage dst, UnmanagedStorage src, UnmanagedStorage mask = null,
            TextReprStyle style = TextReprStyle.ArrayElement)
        {
            var ds = dst.Shape;
            int ndim = ds.NDim;
            int rank = Math.Max(ndim, 1);
            long* shape = stackalloc long[rank];
            long* sa = stackalloc long[rank];
            long* sb = stackalloc long[rank];
            long* sm = stackalloc long[rank];
            long* oShape = stackalloc long[rank];
            long* oa = stackalloc long[rank];
            long* ob = stackalloc long[rank];
            long* om = stackalloc long[rank];

            int dItem = ItemSize(dst), sItem = ItemSize(src);
            for (int d = 0; d < ndim; d++)
            {
                shape[d] = ds.dimensions[d];
                sa[d] = ds.strides[d] * dItem;
            }
            BroadcastStrides(src.Shape, ndim, sItem, sb);
            byte* a = dst.Address + ds.offset * dItem;
            byte* b = src.Address + src.Shape.offset * sItem;
            byte* m;
            if (mask is null)
            {
                // No mask: the third operand repeats the source (it coalesces exactly when the source does).
                for (int d = 0; d < ndim; d++)
                    sm[d] = sb[d];
                m = b;
            }
            else
            {
                BroadcastStrides(mask.Shape, ndim, 1, sm);
                m = mask.Address + mask.Shape.offset;
            }

            int n = RawArrayIter.PrepareThree(ndim, shape, ref a, sa, ref b, sb, ref m, sm, oShape, oa, ob, om);
            if (oShape[0] == 0)
                return;

            RunAction run;
            if (dst.IsString)
            {
                var slice = dst.StringSlice;
                NPTypeCode from = src.TypeCode;
                run = (bp, bs, ap, asd, count) => FormatRun(bp, bs, from, ap, asd, slice.IsUnicode, slice.Chars, count);
            }
            else
            {
                var slice = src.StringSlice;
                // One scratch for the whole walk: the ASCII rewrite of the widest element and its underscore-free copy.
                byte[] scratch = new byte[NDTextParse.ScratchBytes(slice.Chars)];
                NPTypeCode target = dst.TypeCode;
                run = (bp, bs, ap, asd, count) =>
                    ParseRun(bp, bs, slice.IsUnicode, slice.Chars, ap, asd, target, count, scratch, style);
            }
            RunOuter(n, oShape, oa, ob, om, a, b, mask is null ? null : m, run);
            GC.KeepAlive(dst);
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
        }

        /// <summary>
        ///     The byte strides of an operand broadcast to <paramref name="ndim"/> axes: right-aligned, a missing or
        ///     length-1 axis broadcasting with stride 0; a higher-rank operand's extra leading axes (extent 1) dropped.
        /// </summary>
        /// <param name="s">The operand's shape.</param>
        /// <param name="ndim">The destination's rank.</param>
        /// <param name="item">The operand's element size in bytes.</param>
        /// <param name="strides">Receives <paramref name="ndim"/> byte strides.</param>
        private static void BroadcastStrides(Shape s, int ndim, int item, long* strides)
        {
            int lead = ndim - s.NDim;
            for (int d = 0; d < ndim; d++)
            {
                int sd = d - lead;
                strides[d] = sd < 0 || s.dimensions[sd] == 1 ? 0 : s.strides[sd] * item;
            }
        }

        /// <summary>One innermost run: source pointer and byte stride, destination pointer and byte stride, length.</summary>
        /// <param name="src">The run's first source element.</param>
        /// <param name="srcStride">The source byte stride.</param>
        /// <param name="dst">The run's first destination element.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="count">The run length.</param>
        private delegate void RunAction(byte* src, long srcStride, byte* dst, long dstStride, long count);

        /// <summary>
        ///     Walks the outer axes (1..<paramref name="n"/>-1, axis 1 fastest) of a prepared raw iteration and calls
        ///     <paramref name="run"/> for each innermost run — for each run of SELECTED elements when a mask is given — in
        ///     NumPy's order.
        /// </summary>
        /// <param name="n">The prepared number of axes.</param>
        /// <param name="shape">The prepared extents, innermost first.</param>
        /// <param name="da">The destination's prepared byte strides.</param>
        /// <param name="db">The source's prepared byte strides.</param>
        /// <param name="dm">The mask's prepared byte strides (ignored without a mask).</param>
        /// <param name="a">The destination's first element (after the reversal of negative axes).</param>
        /// <param name="b">The source's first element (moved alongside).</param>
        /// <param name="m">The mask's first element, or null for no mask.</param>
        /// <param name="run">The per-run converter.</param>
        private static void RunOuter(int n, long* shape, long* da, long* db, long* dm, byte* a, byte* b, byte* m, RunAction run)
        {
            long* coord = stackalloc long[n];
            for (int k = 0; k < n; k++)
                coord[k] = 0;
            long inner = shape[0];
            while (true)
            {
                if (m == null)
                    run(b, db[0], a, da[0], inner);
                else
                {
                    // npy_memchr pairs: skip a run of masked-out elements, convert the run of selected ones after it.
                    long i = 0, ms = dm[0];
                    while (i < inner)
                    {
                        while (i < inner && m[i * ms] == 0)
                            i++;
                        long start = i;
                        while (i < inner && m[i * ms] != 0)
                            i++;
                        if (i > start)
                            run(b + start * db[0], db[0], a + start * da[0], da[0], i - start);
                    }
                }

                int axis = 1;
                for (; axis < n; axis++)
                {
                    a += da[axis];
                    b += db[axis];
                    if (m != null)
                        m += dm[axis];
                    if (++coord[axis] < shape[axis])
                        break;
                    a -= da[axis] * shape[axis];
                    b -= db[axis] * shape[axis];
                    if (m != null)
                        m -= dm[axis] * shape[axis];
                    coord[axis] = 0;
                }
                if (axis >= n)
                    return;
            }
        }

        /// <summary>
        ///     The cast kernel between a numeric dtype and a canonical type: the house IL inner cast loop (direct
        ///     <c>Converts.To{Dst}</c> calls), or the scalar <see cref="NDIterCasting.ConvertValue"/> loop where dynamic code is
        ///     unavailable. Both are the NumPy-exact cast table.
        /// </summary>
        /// <param name="from">The source dtype.</param>
        /// <param name="to">The destination dtype.</param>
        /// <returns>The kernel.</returns>
        private static DirectILKernelGenerator.InnerCastLoop CastLoop(NPTypeCode from, NPTypeCode to)
            => DirectILKernelGenerator.TryGetInnerCastKernel(from, to) ?? ((s, ss, d, ds, n) =>
            {
                byte* sp = (byte*)s, dp = (byte*)d;
                for (long i = 0; i < n; i++, sp += ss, dp += ds)
                    NDIterCasting.ConvertValue(sp, dp, from, to);
            });

        // ---------------------------------------------------------------------------------------------
        // Runs
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Formats a run of numbers into text elements: each value's NumPy scalar <c>str()</c>, truncated to the element
        ///     width and NUL-padded. Formatting cannot fail.
        /// </summary>
        /// <param name="src">The first number.</param>
        /// <param name="srcStride">The source byte stride (any sign, 0 for a broadcast source).</param>
        /// <param name="srcType">The numeric dtype.</param>
        /// <param name="dst">The first text element.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="unicode">True for a <c>U</c> destination, false for <c>S</c>.</param>
        /// <param name="chars">The destination width in characters.</param>
        /// <param name="count">The number of elements.</param>
        internal static void FormatRun(byte* src, long srcStride, NPTypeCode srcType, byte* dst, long dstStride,
            bool unicode, int chars, long count)
        {
            var kind = NDTextFormat.KindOf(srcType);
            var canonType = NDTextFormat.CanonicalType(kind);
            Span<char> buf = stackalloc char[NDTextFormat.MaxTextLength];

            // A bool, int64, uint64, float or complex source is already its canonical value: format it in place.
            if (srcType == canonType)
            {
                for (long i = 0; i < count; i++, src += srcStride, dst += dstStride)
                {
                    int n = NDTextFormat.Format(kind, src, buf);
                    NDTextFormat.Store(buf.Slice(0, n), dst, unicode, chars);
                }
                return;
            }

            var cast = CastLoop(srcType, canonType);
            int canonSize = canonType.SizeOf();
            byte* block = stackalloc byte[BlockSize * CanonicalMaxBytes];
            for (long done = 0; done < count; done += BlockSize)
            {
                int k = (int)Math.Min(BlockSize, count - done);
                cast(src + done * srcStride, srcStride, block, canonSize, k);
                byte* e = dst + done * dstStride;
                for (int j = 0; j < k; j++, e += dstStride)
                {
                    int n = NDTextFormat.Format(kind, block + j * canonSize, buf);
                    NDTextFormat.Store(buf.Slice(0, n), e, unicode, chars);
                }
            }
        }

        /// <summary>
        ///     Parses a run of text elements into numbers, in blocks of <see cref="BlockSize"/>: a block is parsed into its
        ///     canonical values and written to the destination only when every element parsed (NumPy's wrapped legacy cast).
        /// </summary>
        /// <param name="src">The first text element.</param>
        /// <param name="srcStride">The source byte stride.</param>
        /// <param name="unicode">True for a <c>U</c> source, false for <c>S</c>.</param>
        /// <param name="chars">The source width in characters.</param>
        /// <param name="dst">The first destination number.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="dstType">The numeric destination dtype (bool takes the truth value).</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="scratch">Work space of <see cref="NDTextParse.ScratchBytes"/>(<paramref name="chars"/>) bytes.</param>
        /// <param name="style">How an error quotes the text.</param>
        /// <exception cref="ValueError">A text is not a number of the destination's kind, or has too many digits.</exception>
        /// <exception cref="OverflowException">An integer text does not fit the destination.</exception>
        /// <exception cref="SystemError">A <c>U</c> element is one unit above U+10FFFF.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> element converted to complex is not UTF-8.</exception>
        internal static void ParseRun(byte* src, long srcStride, bool unicode, int chars, byte* dst, long dstStride,
            NPTypeCode dstType, long count, Span<byte> scratch, TextReprStyle style)
        {
            byte* block = stackalloc byte[BlockSize * CanonicalMaxBytes];
            if (dstType == NPTypeCode.Boolean)
            {
                TruthRun(src, srcStride, unicode, chars, dst, dstStride, count, block);
                return;
            }

            var family = NDTextParse.FamilyOf(dstType);
            var canonType = NDTextParse.CanonicalType(family);
            int canonSize = canonType.SizeOf();
            var cast = canonType == dstType ? null : CastLoop(canonType, dstType);
            for (long done = 0; done < count; done += BlockSize)
            {
                int k = (int)Math.Min(BlockSize, count - done);
                byte* e = src + done * srcStride;
                for (int j = 0; j < k; j++, e += srcStride)
                {
                    if (NDTextParse.ParseElement(e, unicode, chars, dstType, block + j * canonSize, scratch) != TextParseStatus.Ok)
                        // The block is not committed: NumPy's wrapper copies a block out only after its cast call succeeded.
                        throw NDTextParse.ElementError(e, unicode, chars, dstType, style);
                }
                Commit(block, canonSize, dst + done * dstStride, dstStride, k, cast);
            }
        }

        /// <summary>
        ///     Writes a converted block into the destination: a same-type move, or the cast from the canonical type.
        /// </summary>
        /// <param name="block">The canonical values, packed.</param>
        /// <param name="canonSize">The canonical value size.</param>
        /// <param name="dst">The block's first destination element.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="k">The block length.</param>
        /// <param name="cast">The canonical → destination cast, or null when they are the same type.</param>
        private static void Commit(byte* block, int canonSize, byte* dst, long dstStride, int k,
            DirectILKernelGenerator.InnerCastLoop cast)
        {
            if (cast != null)
            {
                cast(block, canonSize, dst, dstStride, k);
                return;
            }
            if (dstStride == canonSize)
            {
                Buffer.MemoryCopy(block, dst, (long)k * canonSize, (long)k * canonSize);
                return;
            }
            for (int j = 0; j < k; j++, dst += dstStride)
                Buffer.MemoryCopy(block + j * canonSize, dst, canonSize, canonSize);
        }

        /// <summary>
        ///     Text → bool for a run: an element is True when it holds a non-NUL unit (NumPy's legacy string-to-bool cast,
        ///     the truth of the <c>np.str_</c> with its trailing NULs dropped), committed in blocks like a parse.
        /// </summary>
        /// <param name="src">The first text element.</param>
        /// <param name="srcStride">The source byte stride.</param>
        /// <param name="unicode">True for <c>U</c>.</param>
        /// <param name="chars">The source width in characters.</param>
        /// <param name="dst">The first bool.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="block">A block buffer of at least <see cref="BlockSize"/> bytes.</param>
        /// <exception cref="SystemError">A <c>U</c> element is one unit above U+10FFFF (NumPy cannot build its
        ///     <c>np.str_</c>).</exception>
        private static void TruthRun(byte* src, long srcStride, bool unicode, int chars, byte* dst, long dstStride, long count, byte* block)
        {
            for (long done = 0; done < count; done += BlockSize)
            {
                int k = (int)Math.Min(BlockSize, count - done);
                byte* e = src + done * srcStride;
                for (int j = 0; j < k; j++, e += srcStride)
                {
                    bool any = false;
                    if (unicode)
                    {
                        uint* u = (uint*)e;
                        int last = -1;
                        for (int i = 0; i < chars; i++)
                            if (u[i] != 0)
                                last = i;
                        // A lone unit above U+10FFFF: the one-character np.str_ CPython refuses.
                        if (last == 0 && u[0] > NDStringCodec.MaxCodePoint)
                            throw new SystemError(NDStringCodec.InvalidMaxCharMessage);
                        any = last >= 0;
                    }
                    else
                    {
                        for (int i = 0; i < chars && !any; i++)
                            any = e[i] != 0;
                    }
                    block[j] = any ? (byte)1 : (byte)0;
                }
                byte* o = dst + done * dstStride;
                for (int j = 0; j < k; j++, o += dstStride)
                    *o = block[j];
            }
        }
    }
}
