using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Sorting;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The per-call state of a <c>StringDType</c> reduction lane kernel: the input's and the output's allocators (locked by
    ///     the caller for the call), the NA rule of the loop, the lane geometry, and the size pass's accumulator. The line
    ///     kernels resolve it from their auxdata (<see cref="FromAux"/>), a pointer to a <see cref="GCHandle"/>.
    /// </summary>
    internal sealed unsafe class VStringReduceContext
    {
        /// <summary>The input cells' allocator.</summary>
        internal VStringAllocator In;

        /// <summary>The output's allocator (a <c>StringDType</c> result), or null (the int64 index of an arg-reduction).</summary>
        internal VStringAllocator Out;

        /// <summary>The NA rule of the loop's deciding descriptor (the accumulator's: the output's, NumPy's <c>descriptors[0]</c>).</summary>
        internal VStringNARule Rule;

        /// <summary>The deciding descriptor's default string (UTF-8, pinned by the caller), or null when empty.</summary>
        internal byte* DefaultText;

        /// <summary>The byte length of <see cref="DefaultText"/>.</summary>
        internal long DefaultSize;

        /// <summary>
        ///     True when the first element of a lane is CAST into the output rather than duplicated: the input has an NA
        ///     object and the loop's (output's) descriptor has none, so NumPy's <c>string_to_string</c> cast packs an NA
        ///     first element as the input's <c>na_name</c> TEXT (<c>'nan'</c>, <c>'None'</c>) — which the loop then reads as
        ///     a string, not as a missing value.
        /// </summary>
        internal bool FirstNAAsText;

        /// <summary>The input's <c>na_name</c> (UTF-8, pinned by the caller) for <see cref="FirstNAAsText"/>, or null.</summary>
        internal byte* FirstNAText;

        /// <summary>The byte length of <see cref="FirstNAText"/>.</summary>
        internal long FirstNASize;

        /// <summary>The number of elements along a lane (the reduced axis' extent).</summary>
        internal long N;

        /// <summary>The input's byte stride along the lane.</summary>
        internal long InStride;

        /// <summary>The output's byte stride along the lane (an accumulation writes a lane of outputs).</summary>
        internal long OutStride;

        /// <summary>The mask's byte stride along the lane (a <c>where=</c> mask), or unused.</summary>
        internal long MaskStride;

        /// <summary>True when a <c>where=</c> mask rides as the line operand after the output.</summary>
        internal bool HasMask;

        /// <summary>True for <c>maximum</c> / <c>argmax</c>, false for <c>minimum</c> / <c>argmin</c>.</summary>
        internal bool IsMax;

        /// <summary>True when every output cell is still the all-zero cell of a fresh array.</summary>
        internal bool FreshOut;

        /// <summary>The <c>initial=</c> value as a cell of <see cref="InitialAllocator"/>'s buffer, or null for none.</summary>
        internal byte* InitialCell;

        /// <summary>The allocator of <see cref="InitialCell"/>.</summary>
        internal VStringAllocator InitialAllocator;

        /// <summary>The ufunc NumPy names in its errors (<c>add</c>, <c>maximum</c>, <c>minimum</c>).</summary>
        internal string Name;

        /// <summary>The size pass's accumulator: the arena bytes the outputs will take.</summary>
        internal long Reserve;

        /// <summary>A growable scratch buffer (an accumulation's running text), reused across lanes.</summary>
        internal byte[] Scratch = Array.Empty<byte>();

        /// <summary>The context a line kernel's auxdata points at (a pointer to its handle).</summary>
        /// <param name="aux">The kernel's auxdata.</param>
        /// <returns>The context.</returns>
        internal static VStringReduceContext FromAux(void* aux) => (VStringReduceContext)GCHandle.FromIntPtr(*(IntPtr*)aux).Target;
    }

    /// <summary>
    ///     NumPy's reductions over NEP 55 <c>StringDType</c> text (strings plan D18): <c>np.sum</c> / <c>add.reduce</c>,
    ///     <c>np.cumsum</c> / <c>add.accumulate</c>, <c>np.max</c> / <c>np.min</c> (<c>maximum</c> / <c>minimum.reduce</c>,
    ///     <c>initial=</c>, <c>where=</c> and <c>out=</c> included), <c>np.argmax</c> / <c>np.argmin</c> — NumPy's results,
    ///     linear where NumPy's <c>add.reduce</c> is quadratic.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The lane walker.</b> A <c>StringDType</c> reduction has at most one axis (NumPy's <c>add</c> / <c>maximum</c> /
    ///     <c>minimum</c> loops are not reorderable; the caller's checks refuse more), so every output position reduces one
    ///     LANE of input cells along the axis. <see cref="AxisSort.DriveAllButAxis"/> visits the lanes (NumPy's
    ///     IterAllButAxis) and a C# line kernel reduces each one, reading the cells in place. NDIter's REDUCE operand is not
    ///     used: its contract accumulates through the output, which for text is the quadratic shape of NumPy's own loop.
    ///     </para>
    ///     <para>
    ///     <b>sum</b>: per lane one pass sums the sizes (NumPy's NA rules first), one allocation, one copy per element — the
    ///     string NumPy builds by re-copying the whole running text at every step. A lane of one element is that element
    ///     copied (the reduction's first-element copy, so an NA stays NA). A fresh output's arena is reserved exactly first
    ///     (a size pass over every lane).
    ///     </para>
    ///     <para>
    ///     <b>max / min</b>: the winner by comparisons only (<c>_compare(acc, x)</c>: the accumulator first, a tie keeps it;
    ///     a NaN-like NA as the greatest element), one duplicate at the end. <b>argmax / argmin</b>: the same comparisons in
    ///     NumPy's ArgFunc order (<c>compare(x, best)</c>: so a later NaN-like NA beats an earlier one, NumPy's
    ///     <c>argmax(['b', nan, 'c', nan]) == 3</c>).
    ///     </para>
    ///     <para>
    ///     <b>The first element.</b> NumPy copies each lane's first element into the output with the input-to-output cast
    ///     (<c>PyArray_CopyInitialReduceValues</c>, the accumulation's copy function): an NA stays NA between two descriptors
    ///     with NA objects, but becomes the input's <c>na_name</c> text in an output descriptor without one
    ///     (<see cref="VStringReduceContext.FirstNAAsText"/>) — <c>np.sum([[nan, 'a']], axis=1, out=T)</c> is
    ///     <c>'nana'</c>, while a later NA reads as the output descriptor's default string.
    ///     </para>
    ///     <para>
    ///     <b>Overlap.</b> An <c>out=</c> that may share memory with the input makes the input a copy first (NumPy's
    ///     <c>COPY_IF_OVERLAP</c>); an accumulation into ITSELF is NumPy's exact alias, kept as such (its first-element cast
    ///     frees a short first string before it copies it, so NumPy's <c>np.cumsum(a, out=a)</c> starts with <c>''</c>).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDVStringReduction
    {
        // =============================================================================================================
        // sum
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>add.reduce</c> over a <c>StringDType</c> array along at most one axis, after the caller's checks
        ///     (<see cref="NDStringUfuncs.VStringReduces(string, NDArray, int?, DType, NDArray)"/>) passed.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The axis (negative counts from the end), or null: a 0-d or 1-D input reduces flat (an N-D one
        ///     was refused by the caller).</param>
        /// <param name="keepdims">Keep the reduced axis with length 1.</param>
        /// <param name="out">The output (a <c>StringDType</c> array of the result's shape, or an <c>S</c> / <c>U</c> one that
        ///     receives it through NumPy's output cast), or null.</param>
        /// <returns><paramref name="out"/> itself, or a fresh <c>StringDType</c> array in NumPy's reduction layout — a
        ///     read-only 0-d array (NumPy's scalar) for a flat reduction without keepdims.</returns>
        /// <exception cref="ValueError">A lane of two or more elements with an NA under the <c>None</c>-like rule
        ///     (<c>Cannot add null that is not a nan-like value</c>).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static NDArray Sum(NDArray a, int? axis, bool keepdims, NDArray @out)
            => Reduce(ReduceKind.Sum, a, axis, keepdims, @out, null, null, "add");

        /// <summary>
        ///     NumPy's <c>add.reduce(a, axis, out=out, keepdims=keepdims)</c> with an <c>out=</c> argument, its checks in
        ///     <c>PyUFunc_ReduceWrapper</c>'s order once the caller parsed the arguments (the read-only out and the axis):
        ///     the loop resolution on (out, a), the reorderable check, the out's rank and shape, the missing identity of an
        ///     empty reduction — then <see cref="Sum"/>. <c>np.trace</c>'s reduction of the diagonal runs through here.
        /// </summary>
        /// <param name="op">The NumPy function name for the loop resolution (<c>"sum"</c>, <c>"trace"</c>).</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The normalized axis, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axis.</param>
        /// <param name="out">The output, or null.</param>
        /// <returns>See <see cref="Sum"/>.</returns>
        /// <exception cref="TypeError">No loop (an out of another class, two incompatible NA objects).</exception>
        /// <exception cref="ValueError">Two or more reduced axes, an out of the wrong rank or length, an empty reduction,
        ///     or <see cref="Sum"/>'s NA refusal.</exception>
        /// <exception cref="IncorrectShapeException">The out does not broadcast against the result.</exception>
        internal static NDArray SumNumPy(string op, NDArray a, int? axis, bool keepdims, NDArray @out)
        {
            if (NDStringUfuncs.VStringReduceLoopAnswer(op, a, null, @out) is { } refusal)
                throw refusal;
            int[] axes = axis is { } ax ? new[] { ax } : null;
            if (NDStringUfuncs.VStringNotReorderable("add", a, axes) is { } many)
                throw many;
            if (@out is not null)
                ReductionArguments.ValidateOperands(a, ReductionArguments.AxisFlags(a.ndim, axis), keepdims, @out, null, "add");
            if (NDStringUfuncs.VStringNoIdentity("add", a, axes) is { } empty)
                throw empty;
            return Sum(a, axis, keepdims, @out);
        }

        /// <summary>
        ///     <c>add.reduce(u, axis, out=out, keepdims=keepdims)</c> of a <c>U</c> input into a <c>StringDType</c> out: NumPy's
        ///     <c>StringDType</c> add promoter pairs <c>StringDType</c> with <c>StrDType</c>, so the loop is the
        ///     <c>StringDType</c> concatenation over the input cast to the default <c>StringDType()</c>
        ///     (<c>np.trace(np.array([['c', 'a'], ['x', 'b']]), out=T)</c> is <c>'cb'</c>, probed).
        ///     <c>PyUFunc_ReduceWrapper</c>'s checks (not reorderable, the out's rank and shape, the missing identity) read
        ///     only shapes, so they run on the <c>U</c> input in NumPy's order; the iterator then casts the input, which is
        ///     where an invalid code point would be reported.
        /// </summary>
        /// <param name="u">The <c>U</c> input (the caller resolved the loop:
        ///     <see cref="NDStringUfuncs.TextOutReduce.Concatenate"/>).</param>
        /// <param name="axis">The normalized axis, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axis.</param>
        /// <param name="out">The <c>StringDType</c> out (writeable).</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="ValueError">Two or more reduced axes, an out of the wrong rank or length, or an empty reduction.</exception>
        /// <exception cref="IncorrectShapeException">The out does not broadcast against the result.</exception>
        /// <exception cref="TypeError">A <c>U</c> unit that is not a valid code point (the cast's
        ///     <c>Invalid unicode code point found</c>).</exception>
        internal static NDArray SumNumPyOfCast(NDArray u, int? axis, bool keepdims, NDArray @out)
        {
            // The loop resolution cannot refuse here: a reduction (unlike an accumulation) has no uniform-descriptor check,
            // and the cast's StringDType() NA rule is compatible with every out's (NumPy computes into an out with an NA
            // object, probed).
            int[] axes = axis is { } ax ? new[] { ax } : null;
            if (NDStringUfuncs.VStringNotReorderable("add", u, axes) is { } many)
                throw many;
            ReductionArguments.ValidateOperands(u, ReductionArguments.AxisFlags(u.ndim, axis), keepdims, @out, null, "add");
            if (NDStringUfuncs.VStringNoIdentity("add", u, axes) is { } empty)
                throw empty;
            using var t = NDVStringComparison.ToDefaultVString(u);
            return Sum(t, axis, keepdims, @out);
        }

        /// <summary>The reductions <see cref="Reduce"/> runs.</summary>
        private enum ReduceKind
        {
            /// <summary><c>add.reduce</c>.</summary>
            Sum,

            /// <summary><c>maximum.reduce</c>.</summary>
            Max,

            /// <summary><c>minimum.reduce</c>.</summary>
            Min,
        }

        /// <summary>
        ///     The driver of the value reductions (sum, max, min): the result, the lanes, the overlap copy, the locks and the
        ///     line kernels.
        /// </summary>
        /// <param name="kind">The reduction.</param>
        /// <param name="a">The input.</param>
        /// <param name="axis">The axis, or null for a flat reduction of a 0-d / 1-D input.</param>
        /// <param name="keepdims">Keep the reduced axis.</param>
        /// <param name="out">The output, or null.</param>
        /// <param name="initial">The <c>initial=</c> value as a 0-d <c>StringDType</c> array of the loop's descriptor, or null
        ///     (max / min only).</param>
        /// <param name="where">A bool mask of <paramref name="a"/>'s shape (already broadcast), or null (max / min only;
        ///     requires <paramref name="initial"/>).</param>
        /// <param name="name">The ufunc NumPy names in its errors.</param>
        /// <returns>The result.</returns>
        /// <exception cref="Exception">The loop's errors.</exception>
        private static NDArray Reduce(ReduceKind kind, NDArray a, int? axis, bool keepdims, NDArray @out, NDArray initial,
                                      NDArray where, string name)
        {
            int nd = a.ndim;
            int ax = axis is { } x0 ? (x0 < 0 ? x0 + nd : x0) : 0;
            // A 0-d input reduces over no axis: its one element is the result (a lane of one).
            bool scalarInput = nd == 0;
            long[] dims = ResultDims(a, scalarInput ? -1 : ax, keepdims);
            bool castOut = @out is not null && !NDStringUfuncs.IsVString(@out);
            // The loop's descriptor (NumPy's descriptors[0]): the out's when it is StringDType, the cast's default
            // StringDType() for a U out (PyArray_CastDescrToDType: an NA first element is then written as the input's
            // na_name, and a later NA reads as ''), else the input's.
            DType loopDescr = castOut ? DTypeRegistry.VString.default_descr() : @out is not null ? @out.dtype : a.dtype;
            NDArray result = @out is not null && !castOut
                ? @out
                : new NDArray(DTypeRegistry.VString.Descr(loopDescr.VString.HasNAObject, loopDescr.VString.na_object, loopDescr.VString.coerce),
                              // NumPy's reduction iterator lays the result out over the input AND the where= mask
                              // (both vote on the axis order; a disagreement keeps C order).
                              dims.Length == 0 ? Shape.Scalar : (scalarInput ? Shape.Adopt(dims) : Shape.ReductionOutputShape(dims, a.Shape, ax, where?.Shape)),
                              true);
            // COPY_IF_OVERLAP: a caller's StringDType out that may share cells with the input reduces a copy of the input
            // (NumPy's iterator copies the operand; its first-element copy then reads the original, which is the same text).
            NDArray overlapCopy = ReferenceEquals(result, @out)
                                  && NDMemOverlap.SolveMayShareMemory(result, a, maxWork: 0) != MemOverlap.No
                ? np.copy(a)
                : null;
            try
            {
                var source = overlapCopy ?? a;
                if (result.size != 0 && source.size != 0)
                {
                    // The lane walker wants the output with the input's rank (the reduced axis kept at length 1); a 0-d input
                    // is one lane of one element along a new axis 0. The views are this call's own and disposed here.
                    using var keep = scalarInput || keepdims ? null : np.expand_dims(result, ax);
                    using var input1 = scalarInput ? source.reshape(1) : null;
                    using var out1 = scalarInput ? result.reshape(1) : null;
                    using var mask1 = scalarInput ? where?.reshape(1) : null;
                    NDArray input = input1 ?? source, outLanes = out1 ?? keep ?? result, mask = scalarInput ? mask1 : where;
                    Run(kind, input, outLanes, scalarInput ? 0 : ax, mask, initial, loopDescr, a.dtype, name,
                        fresh: !ReferenceEquals(result, @out));
                }
                else if (result.size != 0 && initial is not null)
                {
                    // An empty reduced axis with initial= (max / min): every output is the initial value.
                    FillInitial(result, initial);
                }
                if (!castOut)
                    return ReferenceEquals(result, @out) ? result : result.MarkReductionScalar();
                // The loop's StringDType result cast into the caller's U out (a reduction validates its output cast with
                // NumPy's historic unsafe casting).
                np.copyto(@out, result, "unsafe");
                return @out;
            }
            catch
            {
                if (!ReferenceEquals(result, @out))
                    result.Dispose();
                result = null;
                throw;
            }
            finally
            {
                overlapCopy?.Dispose();
                if (castOut && result is not null)
                    result.Dispose();
            }
        }

        /// <summary>
        ///     The dimensions of a single-axis reduction's result: the input's without the axis, or with it set to 1
        ///     (keepdims); a 0-d input's result is 0-d.
        /// </summary>
        /// <param name="a">The input.</param>
        /// <param name="axis">The normalized axis, or -1 for a 0-d input.</param>
        /// <param name="keepdims">Keep the axis.</param>
        /// <returns>A fresh dimension array.</returns>
        private static long[] ResultDims(NDArray a, int axis, bool keepdims)
        {
            int nd = a.ndim;
            if (nd == 0)
                return Array.Empty<long>();
            var src = a.Shape.dimensions;
            if (keepdims)
            {
                var k = (long[])src.Clone();
                k[axis] = 1;
                return k;
            }
            var r = new long[nd - 1];
            for (int i = 0, w = 0; i < nd; i++)
                if (i != axis)
                    r[w++] = src[i];
            return r;
        }

        /// <summary>
        ///     Locks the allocators, builds the context and drives the lanes: a size pass and the exact arena reservation for a
        ///     fresh output (sum), then the reducing pass.
        /// </summary>
        /// <param name="kind">The reduction.</param>
        /// <param name="input">The input (rank at least 1).</param>
        /// <param name="outLanes">The output with the input's rank (the reduced axis at length 1).</param>
        /// <param name="axis">The reduced axis.</param>
        /// <param name="mask">The bool mask of the input's shape, or null.</param>
        /// <param name="initial">The 0-d <c>initial=</c> array, or null.</param>
        /// <param name="loopDescr">The loop's deciding descriptor (the output's).</param>
        /// <param name="inputDescr">The input's descriptor (its <c>na_name</c> is the first element's cast text).</param>
        /// <param name="name">The ufunc name.</param>
        /// <param name="fresh">True when the output is a fresh (all-zero) array.</param>
        private static void Run(ReduceKind kind, NDArray input, NDArray outLanes, int axis, NDArray mask, NDArray initial,
                                DType loopDescr, DType inputDescr, string name, bool fresh)
        {
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var inAlloc = input.Storage.VStringSlice.Allocator;
            var outAlloc = outLanes.Storage.VStringSlice.Allocator;
            var initAlloc = initial?.Storage.VStringSlice.Allocator;
            byte[] defaultText = NDUtf8.Encode(loopDescr.VString.default_string);
            // NumPy's first-element cast: an NA becomes the input's na_name only when the output descriptor has no NA object.
            bool firstNAAsText = inputDescr.VString.HasNAObject && !loopDescr.VString.HasNAObject;
            byte[] naText = firstNAAsText ? NDUtf8.Encode(inputDescr.VString.na_name) : Array.Empty<byte>();
            var ctx = new VStringReduceContext
            {
                In = inAlloc,
                Out = outAlloc,
                Rule = NDVStringCells.RuleOf(loopDescr),
                DefaultSize = defaultText.Length,
                FirstNAAsText = firstNAAsText,
                FirstNASize = naText.Length,
                N = input.Shape.dimensions[axis],
                InStride = input.Shape.strides[axis] * (long)cell,
                MaskStride = mask is null ? 0 : mask.Shape.strides[axis],
                HasMask = mask is not null,
                IsMax = kind == ReduceKind.Max,
                FreshOut = fresh,
                InitialAllocator = initAlloc,
                Name = name,
            };
            var locks = new System.Collections.Generic.List<VStringAllocator> { inAlloc, outAlloc };
            if (initAlloc is not null)
                locks.Add(initAlloc);
            var handle = GCHandle.Alloc(ctx);
            VStringAllocator.acquire_allocators(locks);
            try
            {
                fixed (byte* def = defaultText)
                fixed (byte* na = naText)
                {
                    // The pinned texts stay valid for the whole call: the lanes read them through the context.
                    ctx.DefaultText = defaultText.Length == 0 ? null : def;
                    ctx.FirstNAText = naText.Length == 0 ? null : na;
                    if (initial is not null)
                        ctx.InitialCell = (byte*)initial.Address + initial.Shape.offset * (long)cell;
                    IntPtr h = GCHandle.ToIntPtr(handle);
                    if (kind == ReduceKind.Sum)
                    {
                        if (fresh)
                        {
                            AxisSort.DriveAllButAxis(new[] { input }, new[] { NDIterPerOpFlags.READONLY }, axis, s_sumSize, &h);
                            outAlloc.TryReserve(ctx.Reserve);
                        }
                        AxisSort.DriveAllButAxis(new[] { input, outLanes }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE },
                            axis, s_sumLane, &h);
                    }
                    else
                    {
                        var ops = mask is null ? new[] { input, outLanes } : new[] { input, outLanes, mask };
                        var flags = mask is null
                            ? new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE }
                            : new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE, NDIterPerOpFlags.READONLY };
                        AxisSort.DriveAllButAxis(ops, flags, axis, s_maxMinLane, &h);
                    }
                }
            }
            finally
            {
                VStringAllocator.release_allocators(locks);
                handle.Free();
            }
            // The initial value's cell is read through a raw pointer until the lanes are done.
            GC.KeepAlive(initial);
        }

        /// <summary>The size pass of <c>sum</c>, one call per lane (<see cref="SumLaneSize"/>).</summary>
        private static readonly NDInnerLoopFunc s_sumSize = static (p, s, c, a) => SumLaneSize((byte*)p[0], VStringReduceContext.FromAux(a));

        /// <summary>The reducing pass of <c>sum</c>, one call per lane (<see cref="SumLane"/>).</summary>
        private static readonly NDInnerLoopFunc s_sumLane = static (p, s, c, a) => SumLane((byte*)p[0], (byte*)p[1], VStringReduceContext.FromAux(a));

        /// <summary>The reducing pass of <c>max</c> / <c>min</c>, one call per lane (<see cref="MaxMinLane"/>).</summary>
        private static readonly NDInnerLoopFunc s_maxMinLane = static (p, s, c, a) =>
        {
            var ctx = VStringReduceContext.FromAux(a);
            MaxMinLane((byte*)p[0], (byte*)p[1], ctx.HasMask ? (byte*)p[2] : null, ctx);
        };

        /// <summary>
        ///     A cell's text size, or (for an NA) the deciding descriptor's default string size, reported through
        ///     <paramref name="na"/>.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="ctx">The context.</param>
        /// <param name="na">Receives whether the cell is an NA.</param>
        /// <returns>The size in bytes.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static long CellSize(byte* cell, VStringReduceContext ctx, out bool na)
        {
            ulong h = Unsafe.ReadUnaligned<ulong>(cell + 8);
            uint f = (uint)(h >> 56);
            na = (f & VStringAllocator.NPY_STRING_MISSING) != 0;
            if (na)
                return ctx.DefaultSize;
            return (f & VStringAllocator.NPY_STRING_FLAG_MASK) == (VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA)
                ? f & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK
                : (long)(h & NDVStringCells.SizeMask);
        }

        /// <summary>
        ///     A lane's FIRST cell as the loop sees it after NumPy's first-element cast: an NA cast to the input's
        ///     <c>na_name</c> text (<see cref="VStringReduceContext.FirstNAAsText"/>) is that text's size and no longer an NA;
        ///     anything else is <see cref="CellSize"/>.
        /// </summary>
        /// <param name="cell">The lane's first cell.</param>
        /// <param name="ctx">The context.</param>
        /// <param name="na">Receives whether the cast element is an NA.</param>
        /// <returns>The size in bytes.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static long FirstCellSize(byte* cell, VStringReduceContext ctx, out bool na)
        {
            long size = CellSize(cell, ctx, out na);
            if (na && ctx.FirstNAAsText)
            {
                na = false;
                return ctx.FirstNASize;
            }
            return size;
        }

        /// <summary>
        ///     NumPy's first-element copy of a lane into its output cell: the <c>string_to_string</c> cast — an NA packs the
        ///     input's <c>na_name</c> under <see cref="VStringReduceContext.FirstNAAsText"/>, a cell that shares its text
        ///     with the output cell is left alone (<c>NpyString_share_memory</c>), anything else is freed and duplicated.
        /// </summary>
        /// <param name="ctx">The context.</param>
        /// <param name="input">The lane's first input cell.</param>
        /// <param name="output">The output cell.</param>
        /// <exception cref="OutOfMemoryException">NumPy's cast <c>MemoryError</c> texts.</exception>
        private static void CopyFirst(VStringReduceContext ctx, byte* input, byte* output)
        {
            if (VStringAllocator.share_memory(input, ctx.In, output, ctx.Out))
                return;
            if (ctx.FirstNAAsText && VStringAllocator.isnull(input))
            {
                if (ctx.Out.free(output) < 0)
                    throw new OutOfMemoryException("Failed to deallocate string in string to string cast");
                if (ctx.Out.pack(output, ctx.FirstNAText, ctx.FirstNASize) < 0)
                    throw new OutOfMemoryException("Failed to pack string in string to string cast.");
                return;
            }
            VStringAllocator.FreeAndCopy(ctx.In, ctx.Out, input, output, "string to string cast");
        }

        /// <summary>
        ///     The arena bytes one lane's sum takes in a fresh output: the copy of a lone element, nothing for an NA result
        ///     or a result of at most 15 bytes, else the total's allocation.
        /// </summary>
        /// <param name="lane">The lane's first cell.</param>
        /// <param name="ctx">The context.</param>
        private static void SumLaneSize(byte* lane, VStringReduceContext ctx)
        {
            long n = ctx.N, stride = ctx.InStride;
            if (n == 1)
            {
                ctx.Reserve += ctx.FirstNAAsText && VStringAllocator.isnull(lane)
                    ? VStringAllocator.ArenaBytesForText(ctx.FirstNASize)
                    : VStringAllocator.ArenaBytesForDuplicate(lane);
                return;
            }
            long total = FirstCellSize(lane, ctx, out bool anyNA);
            for (long i = 1; i < n; i++)
            {
                total += CellSize(lane + i * stride, ctx, out bool na);
                anyNA |= na;
            }
            if (anyNA && ctx.Rule is VStringNARule.NaN or VStringNARule.Null)
                return;
            ctx.Reserve += VStringAllocator.ArenaBytesForText(total);
        }

        /// <summary>
        ///     One lane of NumPy's <c>add.reduce</c>: a lone element copied (the first-element cast); otherwise the NA rule (a
        ///     NaN-like NA makes the sum NA, a <c>None</c>-like NA raises at its first addition, a string NA or an NA under a
        ///     descriptor without NA object reads as the default string), the total size, one output string, one copy per
        ///     element.
        /// </summary>
        /// <param name="lane">The lane's first cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The context.</param>
        /// <exception cref="ValueError"><c>Cannot add null that is not a nan-like value</c>.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        private static void SumLane(byte* lane, byte* o, VStringReduceContext ctx)
        {
            long n = ctx.N, stride = ctx.InStride;
            if (n == 1)
            {
                // The reduction copies its first element into the output: an NA stays NA (or becomes its name, see CopyFirst).
                CopyFirst(ctx, lane, o);
                return;
            }
            long total = FirstCellSize(lane, ctx, out bool anyNA);
            for (long i = 1; i < n; i++)
            {
                total += CellSize(lane + i * stride, ctx, out bool na);
                anyNA |= na;
            }
            if (anyNA)
            {
                if (ctx.Rule == VStringNARule.NaN)
                {
                    if (ctx.Out.pack_null(o) < 0)
                        throw new OutOfMemoryException("Failed to deallocate string in add");
                    return;
                }
                if (ctx.Rule == VStringNARule.Null)
                    throw new ValueError("Cannot add null that is not a nan-like value");
            }
            if (ctx.Out.free(o) < 0)
                throw new OutOfMemoryException("Failed to deallocate string in add");
            if (ctx.Out.newemptysize(total, o) < 0)
                throw new OutOfMemoryException("Failed to allocate string in add");
            if (ctx.Out.load(o, out byte* dst, out _) < 0)
                throw new OutOfMemoryException("Failed to load string in add");
            long pos = 0;
            for (long i = 0; i < n; i++)
            {
                int isNull = ctx.In.load(lane + i * stride, out byte* buf, out long size);
                if (isNull < 0)
                    throw new OutOfMemoryException("Failed to load string in add");
                if (isNull != 0)
                {
                    // The first element's NA was cast to the input's na_name; a later one reads as the default string.
                    if (i == 0 && ctx.FirstNAAsText)
                    {
                        buf = ctx.FirstNAText;
                        size = ctx.FirstNASize;
                    }
                    else
                    {
                        buf = ctx.DefaultText;
                        size = ctx.DefaultSize;
                    }
                }
                Buffer.MemoryCopy(buf, dst + pos, total - pos, size);
                pos += size;
            }
        }

        // =============================================================================================================
        // max / min
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>maximum</c> / <c>minimum.reduce</c> over a <c>StringDType</c> array along at most one axis, after the
        ///     caller's checks passed.
        /// </summary>
        /// <param name="isMax">True for <c>maximum</c>.</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The axis, or null for a flat reduction of a 0-d / 1-D input.</param>
        /// <param name="keepdims">Keep the reduced axis.</param>
        /// <param name="out">The output, or null.</param>
        /// <param name="initial">The <c>initial=</c> value as a 0-d array of the loop's descriptor, or null.</param>
        /// <param name="where">The bool mask of <paramref name="a"/>'s shape, or null (requires <paramref name="initial"/>).</param>
        /// <returns><paramref name="out"/> itself, or the fresh result (a read-only 0-d array for a flat reduction without
        ///     keepdims).</returns>
        /// <exception cref="ValueError">A comparison with an NA under the <c>None</c>-like rule (<c>Cannot compare null that is
        ///     not a nan-like value</c>).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static NDArray MaxMin(bool isMax, NDArray a, int? axis, bool keepdims, NDArray @out, NDArray initial, NDArray where)
            => Reduce(isMax ? ReduceKind.Max : ReduceKind.Min, a, axis, keepdims, @out, initial, where, isMax ? "maximum" : "minimum");

        /// <summary>
        ///     <c>np.max</c> / <c>np.min</c> / <c>np.nanmax</c> / <c>np.nanmin</c> (and <c>ndarray.max</c> / <c>min</c>) with
        ///     NumPy's keywords when the input or the out is <c>StringDType</c>: <c>PyUFunc_ReduceWrapper</c>'s order once the
        ///     caller parsed the arguments (the read-only out, the mask's dtype, the axis) — the loop resolution on (out, a),
        ///     the reorderable check, the out's and the mask's shapes, <c>initial=</c> packed into the loop's descriptor
        ///     (<c>PyArray_Pack</c>), a mask without <c>initial=</c>, an empty reduction without <c>initial=</c> — then the
        ///     reduction.
        /// </summary>
        /// <param name="op">The NumPy function (<c>"max"</c>, <c>"min"</c>, <c>"nanmax"</c>, <c>"nanmin"</c>): fmax / fmin
        ///     have no <c>StringDType</c> loop.</param>
        /// <param name="a">The input (any dtype: a numeric or fixed-width input into a <c>StringDType</c> out has no loop).</param>
        /// <param name="axis">The axis (already validated), or null for every axis.</param>
        /// <param name="out">The output, or null.</param>
        /// <param name="keepdims">Keep the reduced axis.</param>
        /// <param name="initial">The <c>initial=</c> value (a str, a number under <c>coerce</c>, the NA object, a 0-d
        ///     array cast), or null for none (NumPy's <c>initial=None</c> is no initial value).</param>
        /// <param name="where">The bool mask broadcastable to <paramref name="a"/>, or null.</param>
        /// <returns><paramref name="out"/> itself, or the fresh result.</returns>
        /// <exception cref="TypeError">No loop: <c>fmax</c> / <c>fmin</c>, an out or input of another class, two incompatible
        ///     NA objects.</exception>
        /// <exception cref="ValueError">Two or more reduced axes, a misshapen out, a mask without <c>initial=</c>, an empty
        ///     reduction without <c>initial=</c>, an <c>initial=</c> that is not text under <c>coerce=False</c>, or a
        ///     comparison with a <c>None</c>-like NA.</exception>
        /// <exception cref="IncorrectShapeException">The out or the mask does not broadcast.</exception>
        internal static NDArray MaxMinNumPy(string op, NDArray a, int? axis, NDArray @out, bool keepdims, object initial, NDArray where)
        {
            string ufunc = NDStringUfuncs.ReduceUfuncOf(op, out _);
            // A numeric (or fixed-width) input into a StringDType out: no loop pairs them with StringDType.
            if (!NDStringUfuncs.IsVString(a))
                throw NDStringUfuncs.VStringNoLoop(ufunc, new[] { NDStringUfuncs.DispatchClass(@out), NDStringUfuncs.DispatchClass(a) });
            if (NDStringUfuncs.VStringReduceLoopAnswer(op, a, null, @out) is { } refusal)
                throw refusal;
            int nd = a.ndim;
            int[] axes = axis is { } ax0 && nd > 0 ? new[] { ax0 < 0 ? ax0 + nd : ax0 } : null;
            if (NDStringUfuncs.VStringNotReorderable(ufunc, a, axes) is { } many)
                throw many;
            ReductionArguments.ValidateOperands(a, ReductionArguments.AxisFlags(nd, axis), keepdims, @out, where, ufunc);

            // where=True (a 0-d true mask: C# spells NumPy's literal as one) is no mask at all, as on the numeric path.
            if (where is not null && where.ndim == 0 && where.GetTypeCode == NPTypeCode.Boolean && where.GetAtIndex<bool>(0))
                where = null;

            // initial= packed into the loop's descriptor (the out's when it is StringDType), after every shape error.
            DType loopDescr = NDStringUfuncs.IsVString(@out) ? @out.dtype : a.dtype;
            using var init = initial is null ? null : PackInitial(initial, loopDescr);
            if (init is null)
            {
                if (where is not null)
                    throw new ValueError(
                        $"reduction operation '{ufunc}' does not have an identity, so to use a where mask one has to specify 'initial'");
                if (NDStringUfuncs.VStringNoIdentity(ufunc, a, axes) is { } empty)
                    throw empty;
            }
            // The lanes read the mask along the reduced axis: a mask that broadcasts is viewed at the input's shape.
            using var mask = where is null || where.Shape.Equals(a.Shape) ? null : np.broadcast_to(where, a.Shape);
            return MaxMin(ufunc == "maximum", a, axis, keepdims, @out, init, mask ?? where);
        }

        /// <summary>
        ///     <c>PyArray_Pack(descr, initial)</c> for a <c>StringDType</c> loop: a 0-d array is CAST into the descriptor
        ///     (<c>cast_raw_scalar_item</c>); any other value is <c>stringdtype_setitem</c> — the descriptor's NA object packs
        ///     an NA cell, a str packs itself, anything else its <c>str()</c> under <c>coerce</c> (and a ValueError without).
        /// </summary>
        /// <param name="initial">The value.</param>
        /// <param name="descr">The loop's descriptor.</param>
        /// <returns>A fresh 0-d array of <paramref name="descr"/> holding the packed value (owned by the caller).</returns>
        /// <exception cref="ValueError">A value that is not text under <c>coerce=False</c>.</exception>
        private static NDArray PackInitial(object initial, DType descr)
        {
            var packed = new NDArray(descr, Shape.Scalar, true);
            try
            {
                if (initial is NDArray nd && nd.ndim == 0)
                    np.copyto(packed, nd, "unsafe");
                else
                    packed.Storage.VStringSlice.SetIndex(0, initial);
                return packed;
            }
            catch
            {
                packed.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     One lane of NumPy's <c>maximum</c> / <c>minimum.reduce</c>: the accumulator starts as the first element (after
        ///     the first-element cast) or the <c>initial=</c> value, each later (selected) element meets it through
        ///     <c>_compare(acc, x)</c> (a tie keeps the accumulator; <c>maximum</c> keeps it when <c>acc &gt;= x</c>,
        ///     <c>minimum</c> when <c>acc &lt; x</c>), and the winner is duplicated into the output once.
        /// </summary>
        /// <param name="lane">The lane's first cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="mask">The lane's first mask byte, or null.</param>
        /// <param name="ctx">The context.</param>
        /// <exception cref="ValueError">A comparison with a <c>None</c>-like NA.</exception>
        /// <exception cref="OutOfMemoryException">A failed duplicate.</exception>
        private static void MaxMinLane(byte* lane, byte* o, byte* mask, VStringReduceContext ctx)
        {
            long n = ctx.N, stride = ctx.InStride, mstride = ctx.MaskStride;
            byte* best = ctx.InitialCell;
            VStringAllocator bestAlloc = ctx.InitialAllocator;
            long i = 0;
            if (best == null)
            {
                // No initial: the first element (a mask always comes with initial=). An NA first element cast to its name is
                // text from then on, so it is materialized in the output cell and compared from there.
                if (ctx.FirstNAAsText && VStringAllocator.isnull(lane))
                {
                    CopyFirst(ctx, lane, o);
                    best = o;
                    bestAlloc = ctx.Out;
                }
                else
                {
                    best = lane;
                    bestAlloc = ctx.In;
                }
                i = 1;
            }
            for (; i < n; i++)
            {
                if (mask != null && mask[i * mstride] == 0)
                    continue;
                byte* x = lane + i * stride;
                int cmp = Compare(best, bestAlloc, x, ctx.In, ctx);
                // NumPy's `(cmp < 0) ^ invert`: maximum keeps the accumulator unless it is less, minimum keeps it only when
                // it is less; a tie keeps it (the accumulator IS the output).
                bool keep = ctx.IsMax ? cmp >= 0 : cmp < 0;
                if (cmp != 0 && !keep)
                {
                    best = x;
                    bestAlloc = ctx.In;
                }
            }
            // The winner duplicated once (the materialized first element is already in place).
            if (best != o)
                VStringAllocator.FreeAndCopy(bestAlloc, ctx.Out, best, o, ctx.Name);
        }

        /// <summary>
        ///     NumPy's <c>_compare</c> for two cells with the context's rule: the short fast path (both texts in their cells),
        ///     else <see cref="NDVStringArithmetic.CompareCells"/>.
        /// </summary>
        /// <param name="a">The first cell.</param>
        /// <param name="aa">Its allocator.</param>
        /// <param name="b">The second cell.</param>
        /// <param name="ab">Its allocator.</param>
        /// <param name="ctx">The context.</param>
        /// <returns>Negative, zero or positive.</returns>
        /// <exception cref="ValueError">An NA under the <c>None</c>-like rule.</exception>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static int Compare(byte* a, VStringAllocator aa, byte* b, VStringAllocator ab, VStringReduceContext ctx)
        {
            ulong ha = Unsafe.ReadUnaligned<ulong>(a + 8), hb = Unsafe.ReadUnaligned<ulong>(b + 8);
            uint fa = (uint)(ha >> 56), fb = (uint)(hb >> 56);
            const uint shortFlags = VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA;
            if ((fa & VStringAllocator.NPY_STRING_FLAG_MASK) == shortFlags && (fb & VStringAllocator.NPY_STRING_FLAG_MASK) == shortFlags)
                return NDVStringArithmetic.ShortOrder(a, fa, ha, b, fb, hb);
            // Two texts load through their allocators' arena bases inline (an allocator call per cell was most of a mixed
            // short / arena reduction); an NA or a corrupt cell takes the general compare, which applies the rule or raises.
            int na = NDVStringCells.Load(a, aa.ArenaBase, out byte* ba, out long sa);
            int nb = NDVStringCells.Load(b, ab.ArenaBase, out byte* bb, out long sb);
            if ((na | nb) == 0)
                return NDVStringCells.Order(ba, sa, bb, sb);
            // _compare's NaN-like rule inline (an NA first operand is greater, else an NA second operand is): once a
            // reduction's accumulator is that NA, every later element meets it here.
            if (na >= 0 && nb >= 0 && ctx.Rule == VStringNARule.NaN)
                return na != 0 ? 1 : -1;
            return NDVStringArithmetic.CompareCells(a, aa, b, ab, ctx.Rule, ctx.DefaultText, ctx.DefaultSize);
        }

        /// <summary>Writes the <c>initial=</c> value into every element of an output whose reduced axis is empty.</summary>
        /// <param name="result">The output.</param>
        /// <param name="initial">The 0-d initial array.</param>
        private static void FillInitial(NDArray result, NDArray initial) => np.copyto(result, initial, "unsafe");

        // =============================================================================================================
        // argmax / argmin
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.argmax</c> / <c>np.argmin</c> over a <c>StringDType</c> array (the dtype's ArgFunc after
        ///     <c>PyArray_ArgMax</c>'s ravel / transpose), after the caller's checks passed: per lane the first index for which
        ///     <c>compare(x[i], x[best])</c> is greater (argmax) / less (argmin) than the best so far.
        /// </summary>
        /// <param name="isMax">True for argmax.</param>
        /// <param name="a">The <c>StringDType</c> array (non-empty along the reduced axis: the caller raised NumPy's empty
        ///     sequence error).</param>
        /// <param name="axis">The axis, or null for the flattened array (C order).</param>
        /// <param name="keepdims">Keep the reduced axis (axis null: every axis at length 1).</param>
        /// <returns>A fresh int64 array (C order), a read-only 0-d array for a flat reduction without keepdims.</returns>
        /// <exception cref="ValueError">A comparison with an NA under the <c>None</c>-like rule.</exception>
        internal static NDArray ArgMaxMin(bool isMax, NDArray a, int? axis, bool keepdims)
        {
            NDArray flat = null, contiguous = null;
            try
            {
                NDArray input = a;
                int ax;
                long[] dims;
                if (axis is not { } given || a.ndim == 0)
                {
                    // The flattened array in C order (NumPy ravels; a non-C-contiguous input is copied, as NumPy copies it).
                    contiguous = a.Shape.IsContiguous ? null : np.ascontiguousarray(a);
                    flat = (contiguous ?? a).reshape(a.size);
                    input = flat;
                    ax = 0;
                    dims = keepdims ? Ones(a.ndim) : Array.Empty<long>();
                }
                else
                {
                    ax = given < 0 ? given + a.ndim : given;
                    dims = ResultDims(a, ax, keepdims);
                }
                var result = new NDArray(NPTypeCode.Int64, dims.Length == 0 ? Shape.Scalar : Shape.Adopt(dims), true);
                if (result.size == 0 || input.size == 0)
                    return result.MarkReductionScalar();
                // The lanes write into the result viewed with the input's rank (the reduced axis at length 1).
                NDArray outLanes;
                if (axis is null || a.ndim == 0)
                    outLanes = result.reshape(1);
                else
                    outLanes = keepdims ? result : np.expand_dims(result, ax);
                try
                {
                    RunArg(isMax, input, outLanes, ax);
                }
                finally
                {
                    if (!ReferenceEquals(outLanes, result))
                        outLanes.Dispose();
                }
                return result.MarkReductionScalar();
            }
            finally
            {
                flat?.Dispose();
                contiguous?.Dispose();
            }
        }

        /// <summary>A dimension array of <paramref name="nd"/> ones (keepdims of a flat reduction).</summary>
        /// <param name="nd">The rank.</param>
        /// <returns>The array.</returns>
        private static long[] Ones(int nd)
        {
            var r = new long[nd];
            for (int i = 0; i < nd; i++)
                r[i] = 1;
            return r;
        }

        /// <summary>Drives the arg lanes: one kernel call per lane, the index written as int64.</summary>
        /// <param name="isMax">True for argmax.</param>
        /// <param name="input">The input (rank at least 1).</param>
        /// <param name="outLanes">The int64 output with the input's rank (the axis at length 1).</param>
        /// <param name="axis">The reduced axis.</param>
        private static void RunArg(bool isMax, NDArray input, NDArray outLanes, int axis)
        {
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var inAlloc = input.Storage.VStringSlice.Allocator;
            byte[] defaultText = NDUtf8.Encode(input.dtype.VString.default_string);
            var ctx = new VStringReduceContext
            {
                In = inAlloc,
                Rule = NDVStringCells.RuleOf(input.dtype),
                DefaultSize = defaultText.Length,
                N = input.Shape.dimensions[axis],
                InStride = input.Shape.strides[axis] * (long)cell,
                IsMax = isMax,
            };
            var handle = GCHandle.Alloc(ctx);
            inAlloc.acquire_allocator();
            try
            {
                fixed (byte* def = defaultText)
                {
                    ctx.DefaultText = defaultText.Length == 0 ? null : def;
                    IntPtr h = GCHandle.ToIntPtr(handle);
                    AxisSort.DriveAllButAxis(new[] { input, outLanes }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY },
                        axis, s_argLane, &h);
                }
            }
            finally
            {
                inAlloc.release_allocator();
                handle.Free();
            }
        }

        /// <summary>The arg kernel, one call per lane (<see cref="ArgLane"/>).</summary>
        private static readonly NDInnerLoopFunc s_argLane = static (p, s, c, a) => ArgLane((byte*)p[0], (long*)p[1], VStringReduceContext.FromAux(a));

        /// <summary>
        ///     One lane of NumPy's StringDType <c>argmax</c> / <c>argmin</c> ArgFunc: <c>best = 0</c>, then index i replaces it
        ///     when <c>compare(x[i], x[best])</c> is positive (argmax) / negative (argmin).
        /// </summary>
        /// <param name="lane">The lane's first cell.</param>
        /// <param name="o">The int64 output.</param>
        /// <param name="ctx">The context.</param>
        /// <exception cref="ValueError">A comparison with a <c>None</c>-like NA.</exception>
        private static void ArgLane(byte* lane, long* o, VStringReduceContext ctx)
        {
            long n = ctx.N, stride = ctx.InStride, best = 0;
            byte* bestCell = lane;
            bool isMax = ctx.IsMax;
            for (long i = 1; i < n; i++)
            {
                byte* x = lane + i * stride;
                int c = Compare(x, ctx.In, bestCell, ctx.In, ctx);
                if (isMax ? c > 0 : c < 0)
                {
                    best = i;
                    bestCell = x;
                }
            }
            *o = best;
        }

        // =============================================================================================================
        // cumsum
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>add.accumulate</c> (<c>np.cumsum</c>) over a <c>StringDType</c> array, after the caller's checks
        ///     passed: <c>out[0] = in[0]</c> (the first-element cast: an NA stays NA, or becomes the input's <c>na_name</c> in
        ///     an out without NA object), <c>out[i] = out[i - 1] + in[i]</c> through the add loop's NA rules (a NaN-like NA
        ///     propagates to every later output, a <c>None</c>-like NA raises at its first addition, a string NA reads as its
        ///     string). The running text is kept once and each output is one copy of it.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The axis, or null for the flattened array (C order).</param>
        /// <param name="out">The output (a <c>StringDType</c> array, or a <c>U</c> one receiving the result through NumPy's
        ///     output cast), or null. The caller resolved the loop (<see cref="NDStringUfuncs.VStringReduceLoopAnswer"/>).</param>
        /// <returns><paramref name="out"/> itself, or a fresh array of the input's dimensions (1-D for a flattened input) in
        ///     the input's memory order (NumPy's KEEPORDER allocation).</returns>
        /// <exception cref="IncorrectShapeException">An out the accumulation cannot write: <c>provided out is the wrong size
        ///     for the accumulation.</c> on NumPy's direct path, the iterator's texts on its iterator path (see the remarks);
        ///     NumPy's ValueError texts, the house shape exception.</exception>
        /// <exception cref="ValueError">A <c>None</c>-like NA met by an addition.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        /// <remarks>
        ///     <para>
        ///         NumPy's <c>PyUFunc_Accumulate</c> takes one of two paths, and they check an <c>out=</c> differently. The
        ///         DIRECT path (a 1-D accumulation into a <c>StringDType</c> out that does not overlap the input, or no out)
        ///         returns at once for an empty input or an empty out and otherwise demands the input's exact dimensions
        ///         (<c>provided out is the wrong size for the accumulation.</c>). The ITERATOR path (more than one dimension,
        ///         an out of another dtype, or an out whose bounds overlap the input's) builds an NpyIter over (out, input)
        ///         with <c>op_axes</c> naming the input's axes: an out with fewer dimensions is the iterator's
        ///         <c>op_axes</c> error, an extent that does not broadcast is its remapped-shapes error, an out of extent 1
        ///         where the input is longer is its non-broadcastable-output error — and an out with MORE dimensions is
        ///         accepted, its extra trailing axes staying at index 0. The input broadcasts against the out on every
        ///         axis but the accumulated one, and along that axis only the input's extent is written (an input of
        ///         extent 1 against a longer out writes the first cell).
        ///     </para>
        ///     <para>
        ///         The loop's descriptor is the out's for a <c>StringDType</c> out, the cast's default <c>StringDType()</c>
        ///         for a <c>U</c> out (an NA first element is then written as the input's <c>na_name</c>, NumPy's lossy
        ///         string-to-string cast), and a fresh descriptor with the input's NA object and coerce without an out.
        ///         Divergence: a 0-d out over a 1-D input on the direct path crashes NumPy 2.4.2 (it reads the strides of a
        ///         0-d array, before even its empty-input exit); NumSharp raises the direct path's wrong-size error, an empty
        ///         input included, as its numeric accumulation does.
        ///     </para>
        /// </remarks>
        internal static NDArray CumSum(NDArray a, int? axis, NDArray @out) => CumSum(a, axis, @out, castInput: false);

        /// <summary>
        ///     <c>add.accumulate</c> of a <c>U</c> input into a <c>StringDType</c> out: NumPy's <c>StringDType</c> add promoter
        ///     pairs <c>StringDType</c> with <c>StrDType</c>, so the loop is the <c>StringDType</c> concatenation and the
        ///     input is cast to the default <c>StringDType()</c> (<c>np.cumsum(np.array(['c', 'a']), out=T)</c> is
        ///     <c>['c', 'ca']</c>, probed).
        /// </summary>
        /// <param name="u">The <c>U</c> input (the caller resolved the loop: <see cref="NDStringUfuncs.TextOutReduce.Concatenate"/>).</param>
        /// <param name="axis">The axis, or null for the flattened input.</param>
        /// <param name="out">The <c>StringDType</c> out (writeable).</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="TypeError">An out with an NA object: the accumulation's uniform-descriptor check refuses it over
        ///     the cast's <c>StringDType()</c> (<c>the resolved dtypes are not compatible with add.accumulate. Resolved
        ///     (StringDType(na_object=None), StringDType(), StringDType(na_object=None))</c>, probed), before anything is
        ///     cast.</exception>
        /// <exception cref="ValueError">The iterator's shape errors (<see cref="CumSum(NDArray, int?, NDArray)"/>).</exception>
        /// <exception cref="IncorrectShapeException">The iterator's op_axes error.</exception>
        internal static NDArray CumSumOfCast(NDArray u, int? axis, NDArray @out)
        {
            // reducelike_promote_and_resolve runs before the iterator: the resolved descriptors are (out, StringDType(), out)
            // and enforce_uniform_args compares the first two — NumPy's StringDType-to-StringDType cast from a descriptor
            // WITHOUT an NA object to one with it is safe, the reverse is not, so only an out with an NA object is refused.
            if (@out.dtype.VString.HasNAObject)
            {
                string o = @out.dtype.ToString(true);
                throw new TypeError($"the resolved dtypes are not compatible with add.accumulate. Resolved ({o}, StringDType(), {o})");
            }
            return CumSum(u, axis, @out, castInput: true);
        }

        /// <summary>
        ///     The body of <see cref="CumSum(NDArray, int?, NDArray)"/> and <see cref="CumSumOfCast"/>.
        /// </summary>
        /// <param name="a">The input: <c>StringDType</c>, or a <c>U</c> array when <paramref name="castInput"/> is set.</param>
        /// <param name="axis">The axis, or null for the flattened input.</param>
        /// <param name="out">The output, or null.</param>
        /// <param name="castInput">Whether <paramref name="a"/> is a <c>U</c> input NumPy casts to the default
        ///     <c>StringDType()</c>. Its descriptor differs from the loop's, so <c>PyUFunc_Accumulate</c> always takes the
        ///     ITERATOR path (its shape errors, a longer out accepted), and the cast runs after every shape check, on exactly
        ///     the cells the accumulation reads — the iterator's buffered cast.</param>
        /// <returns><paramref name="out"/> itself, or the fresh result.</returns>
        /// <exception cref="ValueError">The iterator's remapped-shapes error.</exception>
        /// <exception cref="IncorrectShapeException">The direct path's wrong size, or the iterator's op_axes error.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        private static NDArray CumSum(NDArray a, int? axis, NDArray @out, bool castInput)
        {
            NDArray flat = null, contiguous = null, overlapCopy = null, broadcastInput = null, outView = null, target = null;
            NDArray castSource = null;
            bool castOut = @out is not null && !NDStringUfuncs.IsVString(@out);
            NDArray result = null;
            try
            {
                NDArray input = a;
                int ax;
                if (axis is not { } given || a.ndim == 0)
                {
                    // NumPy ravels in C order (a copy for a non-contiguous input) and accumulates along axis 0.
                    contiguous = a.Shape.IsContiguous ? null : np.ascontiguousarray(a);
                    flat = (contiguous ?? a).reshape(a.size);
                    input = flat;
                    ax = 0;
                }
                else
                {
                    ax = given < 0 ? given + a.ndim : given;
                }
                long count = input.Shape.dimensions[ax];
                DType loopDescr = castOut ? DTypeRegistry.VString.default_descr()
                    : @out is not null ? @out.dtype
                    : DTypeRegistry.VString.Descr(a.dtype.VString.HasNAObject, a.dtype.VString.na_object, a.dtype.VString.coerce);
                if (@out is null)
                {
                    result = new NDArray(loopDescr, Shape.NpyIterOutputShape((long[])input.Shape.dimensions.Clone(), new[] { input.Shape }), true);
                    if (result.size != 0)
                        RunCumSum(input, result, ax, loopDescr, a.dtype, fresh: true);
                    return result;
                }
                // need_outer_iterator: more than one dimension, an out of another dtype (the iterator's UPDATEIFCOPY), an
                // input of another descriptor than the loop's (a cast U input), or an out whose bounds overlap the input's
                // (NPY_MAY_SHARE_BOUNDS).
                bool iterator = input.ndim > 1 || castOut || castInput
                                || NDMemOverlap.SolveMayShareMemory(@out, input, maxWork: 0) != MemOverlap.No;
                if (iterator && AccumulateIteratorShapeError(@out, input) is { } shapeError)
                    throw shapeError;
                // The direct path reads the out's stride along the axis before anything else (`PyArray_STRIDES(op[0])[axis]`),
                // which a 0-d out does not have: NumPy 2.4.2 crashes there, for every dtype and an empty input too. NumSharp
                // reports the wrong-size error instead, as its numeric accumulation does.
                if (!iterator && @out.ndim == 0)
                    throw new IncorrectShapeException("provided out is the wrong size for the accumulation.");
                // Nothing to accumulate (NumPy checks these before the direct path's dimensions).
                if (count == 0 || @out.size == 0)
                    return @out;
                if (!iterator && !SameDims(@out, input))
                    throw new IncorrectShapeException("provided out is the wrong size for the accumulation.");
                // The cells the accumulation writes: the out's leading input-rank axes (a longer out's trailing axes stay at
                // index 0) and, along the axis, as many as the input has.
                outView = LeadingAxesView(@out, input.ndim);
                target = outView.Shape.dimensions[ax] == count ? outView : AxisPrefixView(outView, ax, count);
                NDArray src = input;
                if (!SameDims(target, input))
                    src = broadcastInput = np.broadcast_to(input, new Shape((long[])target.Shape.dimensions.Clone()));
                // The iterator's buffered input cast: every shape check passed, so the cells the accumulation reads are cast
                // to the default StringDType() now (a U input and a StringDType out never share memory).
                DType inputDescr = a.dtype;
                if (castInput)
                {
                    src = castSource = NDVStringComparison.ToDefaultVString(src);
                    inputDescr = castSource.dtype;
                }
                // COPY_IF_OVERLAP: an out that may share cells with the input accumulates from a copy, unless it is the
                // input itself (NumPy's exact alias: the iterator copies nothing, the loop reads each input before it writes).
                if (!castOut && !castInput && NDMemOverlap.SolveMayShareMemory(@out, input, maxWork: 0) != MemOverlap.No
                    && !SameElements(@out, input))
                {
                    overlapCopy = np.copy(src);
                    src = overlapCopy;
                }
                if (!castOut)
                {
                    RunCumSum(src, target, ax, loopDescr, inputDescr, fresh: false);
                    return @out;
                }
                // A U out: the accumulation runs into a default StringDType() temporary of the written cells, then NumPy's
                // write-back casts it in (the iterator casts unsafely).
                result = new NDArray(loopDescr, new Shape((long[])target.Shape.dimensions.Clone()), true);
                RunCumSum(src, result, ax, loopDescr, inputDescr, fresh: true);
                np.copyto(target, result, "unsafe");
                return @out;
            }
            catch
            {
                if (result is not null && !ReferenceEquals(result, @out))
                    result.Dispose();
                result = null;
                throw;
            }
            finally
            {
                overlapCopy?.Dispose();
                castSource?.Dispose();
                broadcastInput?.Dispose();
                if (target is not null && !ReferenceEquals(target, outView))
                    target.Dispose();
                if (outView is not null && !ReferenceEquals(outView, @out))
                    outView.Dispose();
                flat?.Dispose();
                contiguous?.Dispose();
                if (castOut && result is not null)
                    result.Dispose();
            }
        }

        /// <summary>
        ///     The shape errors of <c>PyUFunc_Accumulate</c>'s iterator over (out, input) with <c>op_axes</c>
        ///     <c>[0 .. ndim-1]</c> for both, in NpyIter's order: the out's <c>op_axes</c> entries first (an out with fewer
        ///     dimensions than the input), then the broadcast of each operand's extents, then the out's NO_BROADCAST check.
        /// </summary>
        /// <param name="out">The out array.</param>
        /// <param name="input">The accumulated input (already flattened for <c>axis=None</c>).</param>
        /// <returns>The error, or null when the iterator accepts the operands.</returns>
        /// <remarks>
        ///     Every text is NumPy's as it prints it, quirks included: the <c>op_axes</c> error reports the entry index
        ///     <c>ndim - idim - 1</c> beside the axis value <c>idim</c>, and the non-broadcastable error's "remapped to" lists
        ///     the out's extents in REVERSED axis order (NumPy builds it from <c>axes[ndim - idim - 1]</c>).
        /// </remarks>
        private static IncorrectShapeException AccumulateIteratorShapeError(NDArray @out, NDArray input)
        {
            int ndim = input.ndim, ondim = @out.ndim;
            long[] outDims = @out.Shape.dimensions ?? Array.Empty<long>();
            long[] inDims = input.Shape.dimensions;
            var bshape = new long[ndim];
            Array.Fill(bshape, 1L);
            // Operand 0, the out: its op_axes name axes 0 .. ndim-1, and an axis past its rank is invalid.
            for (int idim = 0; idim < ndim; idim++)
            {
                if (idim >= ondim)
                    return new IncorrectShapeException(
                        $"Iterator input op_axes[0][{ndim - idim - 1}] (=={idim}) is not a valid axis of op[0], which has {ondim} dimensions ");
                if (bshape[idim] == 1)
                    bshape[idim] = outDims[idim];
            }
            // Operand 1, the input: an extent that neither matches nor is 1 does not broadcast.
            for (int idim = 0; idim < ndim; idim++)
            {
                long b = bshape[idim], s = inDims[idim];
                if (b == 1)
                    bshape[idim] = s;
                else if (b != s && s != 1)
                {
                    var sb = new StringBuilder("operands could not be broadcast together with remapped shapes [original->remapped]: ");
                    var outRemapped = new long[ndim];
                    Array.Copy(outDims, outRemapped, ndim);
                    sb.Append(Shape.ConvertShapeToString(outDims)).Append("->").Append(Shape.ConvertShapeToString(outRemapped)).Append(' ');
                    sb.Append(Shape.ConvertShapeToString(inDims)).Append("->").Append(Shape.ConvertShapeToString(inDims)).Append(' ');
                    return new IncorrectShapeException(sb.ToString());
                }
            }
            // The out never broadcasts: an extent of 1 where the broadcast is longer (NumPy fills the axes last first, and
            // the text does not depend on which axis fails).
            for (int idim = 0; idim < ndim; idim++)
            {
                if (bshape[idim] != 1 && outDims[idim] == 1)
                {
                    var reversed = new long[ndim];
                    for (int k = 0; k < ndim; k++)
                        reversed[k] = outDims[ndim - k - 1];
                    return new IncorrectShapeException(
                        $"non-broadcastable output operand with shape {Shape.ConvertShapeToString(outDims)} [remapped to " +
                        $"{Shape.ConvertShapeToString(reversed)}] doesn't match the broadcast shape {Shape.ConvertShapeToString(bshape)}");
                }
            }
            return null;
        }

        /// <summary>
        ///     The view of an out's leading <paramref name="ndim"/> axes with every later axis at index 0 — the cells NumPy's
        ///     accumulate iterator walks when the out has more dimensions than its <c>op_axes</c> name.
        /// </summary>
        /// <param name="out">The out array (non-empty).</param>
        /// <param name="ndim">The input's rank.</param>
        /// <returns><paramref name="out"/> itself when it has exactly <paramref name="ndim"/> axes, else a view (owned by the caller).</returns>
        private static NDArray LeadingAxesView(NDArray @out, int ndim)
        {
            if (@out.ndim == ndim)
                return @out;
            var slices = new Slice[@out.ndim];
            for (int i = 0; i < slices.Length; i++)
                slices[i] = i < ndim ? Slice.All : Slice.Index(0);
            return @out[slices];
        }

        /// <summary>The view of the first <paramref name="count"/> cells of an array along one axis.</summary>
        /// <param name="x">The array.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="count">How many leading cells to keep.</param>
        /// <returns>A view (owned by the caller).</returns>
        private static NDArray AxisPrefixView(NDArray x, int axis, long count)
        {
            var slices = new Slice[x.ndim];
            for (int i = 0; i < slices.Length; i++)
                slices[i] = i == axis ? new Slice(0, count) : Slice.All;
            return x[slices];
        }

        /// <summary>Whether two arrays have the same dimensions.</summary>
        /// <param name="x">The first array.</param>
        /// <param name="y">The second array.</param>
        /// <returns>True for equal ranks and extents.</returns>
        private static bool SameDims(NDArray x, NDArray y)
        {
            if (x.ndim != y.ndim)
                return false;
            for (int i = 0; i < x.ndim; i++)
                if (x.Shape.dimensions[i] != y.Shape.dimensions[i])
                    return false;
            return true;
        }

        /// <summary>Whether two arrays are the same cells: one address, one shape, one set of strides.</summary>
        /// <param name="x">The first array.</param>
        /// <param name="y">The second array.</param>
        /// <returns>True for an exact alias.</returns>
        private static bool SameElements(NDArray x, NDArray y)
        {
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            if ((byte*)x.Address + x.Shape.offset * (long)cell != (byte*)y.Address + y.Shape.offset * (long)cell || !SameDims(x, y))
                return false;
            for (int i = 0; i < x.ndim; i++)
                if (x.Shape.dimensions[i] != 1 && x.Shape.strides[i] != y.Shape.strides[i])
                    return false;
            return true;
        }

        /// <summary>Locks, builds the context and drives the accumulation lanes (a size pass and reservation for a fresh output).</summary>
        /// <param name="input">The input.</param>
        /// <param name="result">The output (the input's dimensions).</param>
        /// <param name="axis">The accumulated axis.</param>
        /// <param name="loopDescr">The deciding descriptor (the output's).</param>
        /// <param name="inputDescr">The input's descriptor (its <c>na_name</c> is the first element's cast text).</param>
        /// <param name="fresh">True for a fresh output.</param>
        private static void RunCumSum(NDArray input, NDArray result, int axis, DType loopDescr, DType inputDescr, bool fresh)
        {
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var inAlloc = input.Storage.VStringSlice.Allocator;
            var outAlloc = result.Storage.VStringSlice.Allocator;
            byte[] defaultText = NDUtf8.Encode(loopDescr.VString.default_string);
            bool firstNAAsText = inputDescr.VString.HasNAObject && !loopDescr.VString.HasNAObject;
            byte[] naText = firstNAAsText ? NDUtf8.Encode(inputDescr.VString.na_name) : Array.Empty<byte>();
            var ctx = new VStringReduceContext
            {
                In = inAlloc,
                Out = outAlloc,
                Rule = NDVStringCells.RuleOf(loopDescr),
                DefaultSize = defaultText.Length,
                FirstNAAsText = firstNAAsText,
                FirstNASize = naText.Length,
                N = input.Shape.dimensions[axis],
                InStride = input.Shape.strides[axis] * (long)cell,
                OutStride = result.Shape.strides[axis] * (long)cell,
                FreshOut = fresh,
                Name = "add",
            };
            var handle = GCHandle.Alloc(ctx);
            VStringAllocator.acquire_allocators(inAlloc, outAlloc);
            try
            {
                fixed (byte* def = defaultText)
                fixed (byte* na = naText)
                {
                    ctx.DefaultText = defaultText.Length == 0 ? null : def;
                    ctx.FirstNAText = naText.Length == 0 ? null : na;
                    IntPtr h = GCHandle.ToIntPtr(handle);
                    if (fresh)
                    {
                        AxisSort.DriveAllButAxis(new[] { input }, new[] { NDIterPerOpFlags.READONLY }, axis, s_cumSize, &h);
                        outAlloc.TryReserve(ctx.Reserve);
                    }
                    AxisSort.DriveAllButAxis(new[] { input, result }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE },
                        axis, s_cumLane, &h);
                }
            }
            finally
            {
                VStringAllocator.release_allocators(inAlloc, outAlloc);
                handle.Free();
            }
        }

        /// <summary>The size pass of <c>cumsum</c>, one call per lane (<see cref="CumSumLaneSize"/>).</summary>
        private static readonly NDInnerLoopFunc s_cumSize = static (p, s, c, a) => CumSumLaneSize((byte*)p[0], VStringReduceContext.FromAux(a));

        /// <summary>The accumulating pass of <c>cumsum</c>, one call per lane (<see cref="CumSumLane"/>).</summary>
        private static readonly NDInnerLoopFunc s_cumLane = static (p, s, c, a) => CumSumLane((byte*)p[0], (byte*)p[1], VStringReduceContext.FromAux(a));

        /// <summary>
        ///     The arena bytes one lane's accumulation takes in a fresh output: the first element's copy, then every prefix
        ///     text of more than 15 bytes (none once a NaN-like NA made the rest NA, none past a <c>None</c>-like NA, which raises).
        /// </summary>
        /// <param name="lane">The lane's first cell.</param>
        /// <param name="ctx">The context.</param>
        private static void CumSumLaneSize(byte* lane, VStringReduceContext ctx)
        {
            long n = ctx.N, stride = ctx.InStride;
            ctx.Reserve += ctx.FirstNAAsText && VStringAllocator.isnull(lane)
                ? VStringAllocator.ArenaBytesForText(ctx.FirstNASize)
                : VStringAllocator.ArenaBytesForDuplicate(lane);
            long running = FirstCellSize(lane, ctx, out bool accNA);
            for (long i = 1; i < n; i++)
            {
                running += CellSize(lane + i * stride, ctx, out bool na);
                if ((accNA || na) && ctx.Rule is VStringNARule.NaN or VStringNARule.Null)
                    return;
                accNA = false;
                ctx.Reserve += VStringAllocator.ArenaBytesForText(running);
            }
        }

        /// <summary>
        ///     One lane of NumPy's <c>add.accumulate</c> (see <see cref="CumSum"/>): the running text is appended to in a scratch
        ///     buffer and each output packs one copy of it.
        /// </summary>
        /// <param name="lane">The lane's first input cell.</param>
        /// <param name="outLane">The lane's first output cell.</param>
        /// <param name="ctx">The context.</param>
        /// <exception cref="ValueError"><c>Cannot add null that is not a nan-like value</c>.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        private static void CumSumLane(byte* lane, byte* outLane, VStringReduceContext ctx)
        {
            long n = ctx.N, istride = ctx.InStride, ostride = ctx.OutStride;
            // out[0] = in[0]: the accumulation's first-element cast. The running text is read from out[0] afterwards — for
            // NumPy's exact alias (np.cumsum(a, out=a)) the cast emptied a short first string, and the sum follows that.
            CopyFirst(ctx, lane, outLane);
            int first = ctx.Out.load(outLane, out byte* buf0, out long size0);
            if (first < 0)
                throw new OutOfMemoryException("Failed to load string in add");
            bool accNA = first != 0;
            long running = 0;
            if (!accNA)
                Append(ctx, ref running, buf0, size0);
            for (long i = 1; i < n; i++)
            {
                byte* x = lane + i * istride;
                byte* o = outLane + i * ostride;
                int isNull = ctx.In.load(x, out byte* buf, out long size);
                if (isNull < 0)
                    throw new OutOfMemoryException("Failed to load string in add");
                if (accNA || isNull != 0)
                {
                    switch (ctx.Rule)
                    {
                        case VStringNARule.NaN:
                            // NA + anything (and anything + NA) is NA, and stays so for every later output.
                            if (ctx.Out.pack_null(o) < 0)
                                throw new OutOfMemoryException("Failed to deallocate string in add");
                            accNA = true;
                            continue;
                        case VStringNARule.Null:
                            throw new ValueError("Cannot add null that is not a nan-like value");
                        default:
                            if (accNA)
                            {
                                // The NA first output reads as the default string from now on.
                                Append(ctx, ref running, ctx.DefaultText, ctx.DefaultSize);
                                accNA = false;
                            }
                            if (isNull != 0)
                            {
                                buf = ctx.DefaultText;
                                size = ctx.DefaultSize;
                            }
                            break;
                    }
                }
                // Appended before the output is freed: under NumPy's exact alias the input cell IS this output cell.
                Append(ctx, ref running, buf, size);
                fixed (byte* text = ctx.Scratch)
                {
                    if (ctx.Out.free(o) < 0)
                        throw new OutOfMemoryException("Failed to deallocate string in add");
                    if (ctx.Out.newsize(text, running, o) < 0)
                        throw new OutOfMemoryException("Failed to allocate string in add");
                }
            }
        }

        /// <summary>Appends <paramref name="size"/> bytes to the context's running text, growing the scratch by doubling.</summary>
        /// <param name="ctx">The context.</param>
        /// <param name="running">The running text's length (updated).</param>
        /// <param name="text">The bytes to append.</param>
        /// <param name="size">Their count.</param>
        /// <exception cref="OutOfMemoryException">The running text passes the largest managed array.</exception>
        private static void Append(VStringReduceContext ctx, ref long running, byte* text, long size)
        {
            long need = running + size;
            if (need > ctx.Scratch.Length)
            {
                if (need > Array.MaxLength)
                    throw new OutOfMemoryException("Failed to allocate string in add");
                var grown = new byte[Math.Min(Math.Max(need, Math.Max(64L, ctx.Scratch.Length * 2L)), Array.MaxLength)];
                Buffer.BlockCopy(ctx.Scratch, 0, grown, 0, (int)running);
                ctx.Scratch = grown;
            }
            if (size != 0)
            {
                fixed (byte* dst = ctx.Scratch)
                    Buffer.MemoryCopy(text, dst + running, ctx.Scratch.Length - running, size);
            }
            running = need;
        }
    }
}
