using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Sorting;
using NumSharp.Utilities;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     NumPy's <c>multiply.reduce</c> of INTEGER repeat counts into a TEXT <c>out=</c> (<c>S</c>, <c>U</c> or
    ///     <c>StringDType</c>): the string repeat loop run as a reduction. NumPy resolves the loop on (out, counts) — the
    ///     out's text class, the counts read as <c>int64</c> (<c>uint64</c> keeps its own <c>StringDType</c> loop) — copies
    ///     each lane's FIRST count into its out cell through the integer-to-text cast (its <c>str()</c>, cut to a
    ///     fixed-width out's width), and then repeats the out's text by every later count in place:
    ///     <c>np.prod([2, 3], out=np.zeros((), 'U20'))</c> is <c>'222'</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>One write per lane.</b> NumPy re-copies the whole running text at every step; here a lane's later counts only
    ///     update the text's LENGTH, with NumPy's own per-step checks, and the text is written once at the end. That is
    ///     exact because a repetition of a repetition is a prefix of the first text repeated: a fixed-width lane ends as the
    ///     prefix of <c>first^∞</c> of length <c>min(len(first) · Π counts, width)</c> (or empty once a count is below 1),
    ///     and a <c>StringDType</c> lane as <c>first</c> repeated <c>Π counts</c> times.
    ///     </para>
    ///     <para>
    ///     <b>The per-step checks</b> (<c>string_multiply</c> in <c>string_ufuncs.cpp</c>, <c>multiply_loop_core</c> in
    ///     <c>stringdtype_ufuncs.cpp</c>): a fixed-width step with a count below 1 or an empty text leaves the text empty for
    ///     good; a one-character text repeats without an overflow check; any other text raises <c>Overflow detected in
    ///     string multiply</c> when <c>count · length</c> passes <c>2^63 - 1</c>, even when the out's width would have cut
    ///     the result. A <c>StringDType</c> step multiplies the size by the count as <c>size_t</c> (a negative count is
    ///     <c>2^64 + count</c>) and raises <c>Overflow encountered in string multiply</c> past <c>2^63 - 1</c>.
    ///     </para>
    ///     <para>
    ///     <b>Two documented differences</b> (<c>[Misaligned]</c>): an empty <c>StringDType</c> text with a negative count
    ///     makes NumPy's copy loop run <c>2^64 - k</c> times copying nothing (it never returns; NumSharp keeps the empty
    ///     text, as its binary repeat does); and NumPy allocates every intermediate <c>StringDType</c> text, so an
    ///     intermediate size the host cannot allocate is a <c>MemoryError</c> there even when a later count of 0 empties
    ///     the text — NumSharp raises that error for any intermediate past the 2^47-byte address space of 64-bit hosts
    ///     (every host's <c>malloc</c> fails there) and otherwise only when the final text cannot be allocated. When a
    ///     check raises, NumPy may have repeated other lanes further than NumSharp has (its iterator walks the counts in
    ///     another order); the exception is the same.
    ///     </para>
    /// </remarks>
    internal static unsafe class NDStringRepeatReduction
    {
        /// <summary>
        ///     The largest intermediate <c>StringDType</c> text NumSharp lets a step reach: the 2^47-byte user address space of
        ///     64-bit hosts, past which NumPy's per-step <c>PyMem_RawMalloc</c> fails on every host.
        /// </summary>
        private const ulong MaxAllocatableTextSize = 1UL << 47;

        /// <summary>
        ///     <c>multiply.reduce(counts, axis=axis, out=out, keepdims=keepdims)</c> after the caller's argument checks (the
        ///     out's read-only check and the axis, in NumPy's order), with <c>PyUFunc_ReduceWrapper</c>'s own checks in its
        ///     order: the string repeat loop is not reorderable, the out's rank and shape, the missing identity of an empty
        ///     reduction — then the computation.
        /// </summary>
        /// <param name="counts">The integer counts (one of the eight integer widths).</param>
        /// <param name="axis">The axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Whether the out keeps the reduced axis with length 1.</param>
        /// <param name="out">The text out (<c>S</c>, <c>U</c> or <c>StringDType</c>), writeable.</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="ValueError">Two or more reduced axes (<c>reduction operation 'multiply' is not reorderable, so
        ///     at most one axis may be specified</c>), an out of the wrong rank or length, or an empty reduced axis
        ///     (<c>zero-size array to reduction operation multiply which has no identity</c>).</exception>
        /// <exception cref="IncorrectShapeException">The out does not broadcast against the result.</exception>
        /// <exception cref="OverflowException">A repeat whose size passes <c>2^63 - 1</c> (NumPy's two texts, see the class
        ///     remarks).</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> text that cannot be allocated (<c>Failed to allocate
        ///     string in multiply</c>) or packed (<c>Failed to pack string in multiply</c>).</exception>
        internal static NDArray ReduceNumPy(NDArray counts, int? axis, bool keepdims, NDArray @out)
        {
            int nd = counts.ndim;
            int[] axes = axis is { } a0 && nd > 0 ? new[] { a0 < 0 ? a0 + nd : a0 } : null;
            if (NDStringUfuncs.VStringNotReorderable("multiply", counts, axes) is { } many)
                throw many;
            ReductionArguments.ValidateOperands(counts, ReductionArguments.AxisFlags(nd, axis), keepdims, @out, null, "multiply");
            if (NDStringUfuncs.VStringNoIdentity("multiply", counts, axes) is { } empty)
                throw empty;
            Reduce(counts, axes is null ? 0 : axes[0], keepdims, @out);
            return @out;
        }

        /// <summary>
        ///     The computation: the first counts cast into the out cells, then each lane's later counts folded into its text.
        /// </summary>
        /// <param name="counts">The counts (checks done).</param>
        /// <param name="axis">The normalized reduced axis (0 for a 0-d or a flat 1-D input).</param>
        /// <param name="keepdims">Whether the out keeps the reduced axis.</param>
        /// <param name="out">The text out.</param>
        [NDScoped] // the lane views (the first slice, the int64 counts, the out with the input's rank) are this call's own
        private static void Reduce(NDArray counts, int axis, bool keepdims, NDArray @out)
        {
            if (@out.size == 0)
                return;
            int nd = counts.ndim;
            if (nd == 0)
            {
                // A 0-d input reduces over no axis: its one element is the result (the first-element copy alone).
                np.copyto(@out, counts, casting: "unsafe");
                return;
            }
            // The out with the input's rank (the reduced axis at length 1), as the lane walker visits it.
            NDArray outLanes = keepdims ? @out : np.expand_dims(@out, axis);
            // NumPy's PyArray_CopyInitialReduceValues: the first count of every lane through the integer-to-text cast (a
            // uint64 count prints as itself, not as its int64 reading).
            var first = new Slice[nd];
            for (int d = 0; d < nd; d++)
                first[d] = d == axis ? new Slice(0, 1) : Slice.All;
            np.copyto(outLanes, counts[first], casting: "unsafe");
            long n = counts.Shape.dimensions[axis];
            if (n < 2)
                return;
            // The later counts as the loop reads them: int64 (a uint64 count's bits read as int64 — the fixed-width loop's
            // cast; the StringDType uint64 loop reads the same 64 bits as size_t, which is what the fold does with them).
            NDArray counts64 = counts.typecode == NPTypeCode.Int64 ? counts : counts.astype(np.int64, true, 'K', "unsafe");
            bool vstring = NDStringUfuncs.IsVString(@out);
            var ctx = new RepeatContext
            {
                N = n,
                CountStride = counts64.Shape.strides[axis] * sizeof(long),
            };
            if (vstring)
                ctx.Out = outLanes.Storage.VStringSlice.Allocator;
            else
            {
                var meta = (StringDTypeMeta)@out.dtype.Meta;
                ctx.Unit = meta.UnitSize;
                ctx.Width = @out.dtype.itemsize / meta.UnitSize;
            }
            var handle = GCHandle.Alloc(ctx);
            if (vstring)
                ctx.Out.acquire_allocator();
            try
            {
                IntPtr h = GCHandle.ToIntPtr(handle);
                AxisSort.DriveAllButAxis(new[] { counts64, outLanes }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE },
                    axis, vstring ? s_vstringLane : s_fixedLane, &h);
            }
            finally
            {
                if (vstring)
                    ctx.Out.release_allocator();
                handle.Free();
            }
        }

        /// <summary>The per-call state the lane kernels read through their aux pointer.</summary>
        private sealed class RepeatContext
        {
            /// <summary>The reduced length (the lane's count of counts).</summary>
            public long N;

            /// <summary>The byte stride between a lane's int64 counts.</summary>
            public long CountStride;

            /// <summary>A fixed-width out's unit size in bytes (1 for <c>S</c>, 4 for <c>U</c>).</summary>
            public int Unit;

            /// <summary>A fixed-width out's width in units.</summary>
            public long Width;

            /// <summary>A <c>StringDType</c> out's allocator (locked for the call).</summary>
            public VStringAllocator Out;

            /// <summary>The context behind a lane kernel's aux pointer (a <see cref="GCHandle"/>).</summary>
            /// <param name="aux">The aux pointer.</param>
            /// <returns>The context.</returns>
            public static RepeatContext FromAux(void* aux) => (RepeatContext)GCHandle.FromIntPtr(*(IntPtr*)aux).Target;
        }

        /// <summary>The fixed-width lane kernel (<see cref="FixedLane"/>), one call per lane.</summary>
        private static readonly NDInnerLoopFunc s_fixedLane = static (p, s, c, a) => FixedLane((byte*)p[0], (byte*)p[1], RepeatContext.FromAux(a));

        /// <summary>The <c>StringDType</c> lane kernel (<see cref="VStringLane"/>), one call per lane.</summary>
        private static readonly NDInnerLoopFunc s_vstringLane = static (p, s, c, a) => VStringLane((byte*)p[0], (byte*)p[1], RepeatContext.FromAux(a));

        /// <summary>
        ///     One fixed-width lane: the out cell holds the first count's text; the later counts update its length with
        ///     <c>string_multiply</c>'s checks, and the cell is rewritten once.
        /// </summary>
        /// <param name="lane">The lane's first int64 count.</param>
        /// <param name="cell">The lane's out cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <exception cref="OverflowException"><c>Overflow detected in string multiply</c>.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void FixedLane(byte* lane, byte* cell, RepeatContext ctx)
        {
            int unit = ctx.Unit;
            long width = ctx.Width;
            // num_codepoints: the text before its trailing NUL units (the first text is digits and a sign: no NUL inside).
            long period = width;
            if (unit == 1)
                while (period > 0 && cell[period - 1] == 0) period--;
            else
                while (period > 0 && ((uint*)cell)[period - 1] == 0) period--;
            long len = period;
            for (long k = 1; k < ctx.N; k++)
            {
                long count = *(long*)(lane + k * ctx.CountStride);
                // string_multiply: an empty text or a count below 1 empties the out, and every later step returns early.
                if (len == 0 || count < 1)
                {
                    len = 0;
                    break;
                }
                // A one-character text is memset without an overflow check.
                if (len == 1)
                {
                    len = Math.Min(count, width);
                    continue;
                }
                // npy_mul_with_overflow_size_t(count, len) or a product past PY_SSIZE_T_MAX: exact as one division.
                if (count > long.MaxValue / len)
                    throw new OverflowException("Overflow detected in string multiply");
                len = Math.Min(count * len, width);
            }
            if (len == period)
                return;
            long bytes = len * unit, done = period * unit;
            if (len == 0)
            {
                // The whole cell NUL: string_multiply's buffer_fill_with_zeros_after_index(0).
                NativeMemory.Clear(cell, (nuint)(width * unit));
                return;
            }
            // len > period: the repetitions of the period after it, by doubling copies (the units past len stay NUL).
            while (done < bytes)
            {
                long chunk = Math.Min(done, bytes - done);
                Buffer.MemoryCopy(cell, cell + done, bytes - done, chunk);
                done += chunk;
            }
        }

        /// <summary>
        ///     One <c>StringDType</c> lane: the out cell holds the first count's text; the later counts update its size with
        ///     <c>multiply_loop_core</c>'s checks, and the cell is packed once with the repetition.
        /// </summary>
        /// <param name="lane">The lane's first int64 count.</param>
        /// <param name="cell">The lane's out cell (its allocator locked).</param>
        /// <param name="ctx">The call's context.</param>
        /// <exception cref="OverflowException"><c>Overflow encountered in string multiply</c>.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts of the loop.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void VStringLane(byte* lane, byte* cell, RepeatContext ctx)
        {
            var alloc = ctx.Out;
            if (alloc.load(cell, out byte* buf, out long size0) < 0)
                throw new OutOfMemoryException("Failed to load string in multiply");
            ulong size = (ulong)size0;
            for (long k = 1; k < ctx.N; k++)
            {
                // An empty text stays empty whatever the count (see the class remarks for NumPy's negative-count loop).
                if (size == 0)
                    break;
                // The count as NumPy's size_t: a negative int64 is 2^64 + count, so it overflows any non-empty text.
                ulong factor = (ulong)*(long*)(lane + k * ctx.CountStride);
                ulong high = Math.BigMul(size, factor, out ulong newsize);
                if (high != 0 || newsize > long.MaxValue)
                    throw new OverflowException("Overflow encountered in string multiply");
                // NumPy allocates each intermediate text (the in-place temporary): past every host's address space its
                // PyMem_RawMalloc fails at this step.
                if (newsize > MaxAllocatableTextSize)
                    throw new OutOfMemoryException("Failed to allocate string in multiply");
                size = newsize;
            }
            if (size == (ulong)size0)
                return;
            if (size == 0)
            {
                // A count of 0: the empty text (NpyString_free leaves an empty string in the cell).
                if (alloc.free(cell) < 0)
                    throw new OutOfMemoryException("Failed to deallocate string in multiply");
                return;
            }
            // The repetition is built in a temporary (the cell's own text may live in the arena the pack reallocates), then
            // packed: NumPy's in-place path, a temporary buffer and NpyString_pack.
            byte* tmp;
            try
            {
                tmp = (byte*)NativeMemory.Alloc((nuint)size);
            }
            catch (OutOfMemoryException)
            {
                throw new OutOfMemoryException("Failed to allocate string in multiply");
            }
            try
            {
                long total = (long)size, done = size0;
                Buffer.MemoryCopy(buf, tmp, total, size0);
                while (done < total)
                {
                    long chunk = Math.Min(done, total - done);
                    Buffer.MemoryCopy(tmp, tmp + done, total - done, chunk);
                    done += chunk;
                }
                if (alloc.pack(cell, tmp, total) < 0)
                    throw new OutOfMemoryException("Failed to pack string in multiply");
            }
            finally
            {
                NativeMemory.Free(tmp);
            }
        }
    }
}
