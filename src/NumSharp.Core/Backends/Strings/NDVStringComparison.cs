using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     NumPy's six comparison ufuncs over NEP 55 <c>StringDType</c> operands: <c>(T, T)</c>, <c>(T, U)</c> and
    ///     <c>(U, T)</c> — <c>string_comparison_strided_loop</c> of <c>stringdtype_ufuncs.cpp</c>, run by NDIter over the IL
    ///     kernels of <see cref="ILKernelGenerator.GetVStringCompareKernel"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Where it sits.</b> <see cref="NDStringUfuncs.BinaryGuard"/> runs every check NumPy makes before its loop
    ///     (<c>out=</c> / <c>where=</c> parsing, the loop resolution with its <c>dtype=</c> rule, the NA objects'
    ///     compatibility, the output cast into <c>out=</c>, the broadcast) and hands a call that computes to
    ///     <see cref="Execute"/>; an empty broadcast never reaches it.
    ///     </para>
    ///     <para>
    ///     <b>The promotion.</b> NumPy's <c>string_unicode_bool_output_promoter</c> makes both inputs <c>StringDType</c>: a
    ///     <c>U</c> operand (a Python <c>str</c> included) is cast to a DEFAULT <c>StringDType()</c>. The loop reads its NA
    ///     rule from its FIRST descriptor, so <c>np.equal(u, t)</c> reads an NA of <c>t</c> as <c>''</c> (the default
    ///     descriptor's default string) while <c>np.equal(t, u)</c> follows <c>t</c>'s own rule (probed against 2.4.2). The
    ///     C# operators with a string on the left keep Python's reflection (<c>"b" &lt; t</c> is <c>t &gt; "b"</c>,
    ///     <see cref="NDArray"/>'s text-comparison operators), so the array is the first operand there, as in NumPy.
    ///     </para>
    ///     <para>
    ///     <b>The NA rules</b> (<see cref="VStringNARule"/>, from the first descriptor): a NaN-like NA makes every comparison
    ///     with it <c>False</c> (<c>!=</c> too); a <c>None</c>-like NA reads as <c>''</c> under <c>==</c> / <c>!=</c> (NumPy
    ///     overwrites its null-ness answer with the compare of the loaded strings, so <c>'' == None</c> is True) and raises
    ///     NumPy's <c>ValueError</c> under an ordering; a string NA reads as that string; without an NA object an NA cell (of
    ///     the other operand) reads as <c>''</c>.
    ///     </para>
    ///     <para>
    ///     <b>Routes.</b> NumPy's trivial loop (one kernel call, no iterator) when every operand walks one linear order;
    ///     otherwise one NDIter pass with <c>EXTERNAL_LOOP</c> over the inputs and the bool output (a <c>where=</c> mask as
    ///     the iterator's ARRAYMASK operand, one kernel call per mask run). The iterator never buffers or copies a
    ///     <c>StringDType</c> input (no cast, no COPY_IF_OVERLAP: a bool output cannot share memory with cells), because the
    ///     kernel decodes the cells through their own buffer's arena. A non-bool <c>out=</c> receives the bool results
    ///     through <see cref="np.copyto(NDArray, NDArray, string, NDArray)"/> (NumPy's output cast: <c>True</c> → 1); the
    ///     bools are computed under the <c>where=</c> mask first, so a masked-off pair is never compared (NumPy's masked
    ///     inner loop runs only the selected runs — an NA there raises nothing).
    ///     </para>
    /// </remarks>
    internal static class NDVStringComparison
    {
        /// <summary>
        ///     Test hook: the kernel the last comparison with a non-empty result on the calling thread ran, so a test can prove
        ///     which form (<see cref="NDVStringCells.TakesPrefetchingKernel"/>) computed its values, not only which one the
        ///     choice named.
        /// </summary>
        /// <remarks>Thread-static (the test suite runs in parallel); production code never reads it.</remarks>
        [ThreadStatic]
        internal static NDInnerLoopFunc LastKernel;

        /// <summary>
        ///     Computes <c>ufunc(x1, x2, out=out, where=where)</c> for a pair with a <c>StringDType</c> operand, after
        ///     <see cref="NDStringUfuncs.BinaryGuard"/> accepted the call.
        /// </summary>
        /// <param name="ufunc">One of the six comparison names.</param>
        /// <param name="x1">The first operand (<c>T</c> or <c>U</c>; at least one operand is <c>T</c>).</param>
        /// <param name="x2">The second operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="out">The <c>out=</c> operand, or null (its cast and shape were checked by the guard; a text
        ///     <c>out=</c> is handled by the guard, which formats the bools into it).</param>
        /// <param name="where">The bool <c>where=</c> mask, or null.</param>
        /// <returns><paramref name="out"/> itself, or a fresh bool array of the broadcast shape in NumPy's output layout.</returns>
        /// <exception cref="ArgumentException"><paramref name="ufunc"/> is not a comparison, or no operand holds
        ///     <c>StringDType</c> text (a routing bug).</exception>
        /// <exception cref="ValueError">An ordering meets an NA of a <c>None</c>-like descriptor (NumPy's
        ///     <c>'less' not supported for null values that are not nan-like or strings.</c>).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell (NumPy's <c>Failed to load string in …</c>).</exception>
        internal static unsafe NDArray Execute(string ufunc, NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            if (!NDStringUfuncs.IsVString(x1) && !NDStringUfuncs.IsVString(x2))
                throw new ArgumentException($"'{ufunc}' over StringDType needs a StringDType operand, not {x1?.dtype} and {x2?.dtype}.");
            (TextCompareOp op, bool swap, byte code) = ufunc switch
            {
                "equal" => (TextCompareOp.Equal, false, (byte)0),
                "not_equal" => (TextCompareOp.NotEqual, false, (byte)1),
                "less" => (TextCompareOp.Less, false, (byte)2),
                "less_equal" => (TextCompareOp.LessEqual, false, (byte)3),
                "greater" => (TextCompareOp.Less, true, (byte)4),
                "greater_equal" => (TextCompareOp.LessEqual, true, (byte)5),
                _ => throw new ArgumentException($"'{ufunc}' is not a comparison ufunc.", nameof(ufunc)),
            };

            NDArray t1 = null, t2 = null, result = null;
            bool castOut = @out is not null && @out.typecode != NPTypeCode.Boolean;
            try
            {
                // NumPy's promoter: a U operand becomes a default StringDType() array (a fresh buffer, its own allocator).
                t1 = NDStringUfuncs.IsVString(x1) ? x1 : ToDefaultVString(x1);
                t2 = NDStringUfuncs.IsVString(x2) ? x2 : ToDefaultVString(x2);

                var dims = NDStringUfuncs.BroadcastDims(x1, x2, @out, where);
                // A non-bool out= is cast into from a bool temporary (no buffered iterator: the kernels read the cells
                // through their arenas, which a buffered StringDType operand would not have). The temporary is computed
                // under the where= mask too: NumPy's masked inner loop never compares a masked-off pair (a None-like NA
                // there must not raise), and np.copyto below writes only the selected slots, so the temporary's
                // masked-off bytes are never read.
                result = @out is not null && !castOut ? @out : new NDArray(NPTypeCode.Boolean, ResultShape(dims, x1, x2, where), false);
                if (result.size != 0)
                {
                    // An ordering of long texts whose first lines pass L2 takes the form that prefetches the texts ahead of
                    // the pair it compares (every such pair is a cache miss the hardware prefetchers do not predict); both
                    // forms give the same results.
                    bool prefetch = NDVStringCells.TakesPrefetchingKernel(op, t1.Storage.VStringSlice, t1.size, t2.Storage.VStringSlice, t2.size);
                    var kernel = ILKernelGenerator.GetVStringCompareKernel(op, swap ? 1 : 0, prefetch) ?? ManagedKernel(op, swap ? 1 : 0);
                    LastKernel = kernel;
                    RunLocked(kernel, t1, t2, result, where, swap, code);
                }
                if (!castOut)
                    return result;
                UnmanagedStorage.Copyto(@out, result, NPY_CASTING.NPY_UNSAFE_CASTING, where);
                return @out;
            }
            catch
            {
                // A fresh result never escapes a failed call; a caller's out= stays the caller's.
                if (result is not null && !ReferenceEquals(result, @out))
                    result.Dispose();
                result = null;
                throw;
            }
            finally
            {
                // The promoted temporaries, and the bool temporary of a cast out=, are this call's own.
                if (t1 is not null && !ReferenceEquals(t1, x1))
                    t1.Dispose();
                if (t2 is not null && !ReferenceEquals(t2, x2))
                    t2.Dispose();
                if (castOut && result is not null)
                    result.Dispose();
            }
        }

        /// <summary>
        ///     Casts a <c>U</c> operand to a fresh default <c>StringDType()</c> array of its dimensions (NumPy's
        ///     <c>unicode_to_string</c> cast, which the comparison promoter inserts).
        /// </summary>
        /// <param name="u">The <c>U</c> operand (any layout, 0-d included).</param>
        /// <returns>A new <c>StringDType</c> array the caller disposes.</returns>
        /// <exception cref="TypeError">A <c>U</c> unit that is not a valid code point (NumPy's <c>Invalid unicode code point found</c>).</exception>
        internal static NDArray ToDefaultVString(NDArray u)
        {
            var dims = (long[])u.Shape.dimensions.Clone();
            var t = new NDArray(DTypeRegistry.VString.default_descr(), dims.Length == 0 ? Shape.NewScalar() : Shape.Adopt(dims), true);
            try
            {
                NDVStringCasting.Copy(t.Storage, u.Storage);
                return t;
            }
            catch
            {
                t.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Locks both inputs' allocators (NumPy's <c>NpyString_acquire_allocators</c>), fills the pair context and runs the
        ///     comparison: the trivial loop when it applies, else the iterator.
        /// </summary>
        /// <param name="kernel">The comparison kernel.</param>
        /// <param name="t1">The first input, promoted (<c>StringDType</c>).</param>
        /// <param name="t2">The second input, promoted.</param>
        /// <param name="result">The bool output.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="swap">True when the kernel reads operand slot 1 as A (<c>greater</c> / <c>greater_equal</c>).</param>
        /// <param name="ufuncCode">The index of the call's ufunc in <see cref="NDVStringCells.ComparisonNames"/>.</param>
        /// <exception cref="ValueError">See <see cref="Execute"/>.</exception>
        private static unsafe void RunLocked(NDInnerLoopFunc kernel, NDArray t1, NDArray t2, NDArray result, NDArray where,
                                             bool swap, byte ufuncCode)
        {
            var a1 = t1.Storage.VStringSlice.Allocator;
            var a2 = t2.Storage.VStringSlice.Allocator;
            // The NA rule and the default string are the FIRST descriptor's (NumPy's descr1), whatever the swap.
            var descr1 = t1.dtype;
            byte[] defaultText = NDUtf8.Encode(descr1.VString.default_string);
            VStringAllocator.acquire_allocators(a1, a2);
            try
            {
                fixed (byte* def = defaultText)
                {
                    VStringPairAux aux;
                    aux.ArenaA = swap ? a2.ArenaBase : a1.ArenaBase;
                    aux.ArenaB = swap ? a1.ArenaBase : a2.ArenaBase;
                    aux.DefaultText = defaultText.Length == 0 ? null : def;
                    aux.DefaultSize = defaultText.Length;
                    aux.Rule = NDVStringCells.RuleOf(descr1);
                    aux.Ufunc = ufuncCode;
                    if (!TryTrivial(kernel, t1, t2, result, where, &aux))
                        RunIterator(kernel, t1, t2, result, where, &aux);
                }
            }
            finally
            {
                VStringAllocator.release_allocators(a1, a2);
            }
        }

        /// <summary>
        ///     The layout of a fresh result: NumPy's NpyIter output allocation over the two inputs and the mask (each votes
        ///     with the axes it really strides; C order wins ties).
        /// </summary>
        /// <param name="dims">The broadcast dimensions (adopted by the returned shape — a fresh array).</param>
        /// <param name="x1">The first input (as given: its layout votes, not its promoted copy's).</param>
        /// <param name="x2">The second input.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>A dense, offset-0 shape of <paramref name="dims"/>.</returns>
        internal static Shape ResultShape(long[] dims, NDArray x1, NDArray x2, NDArray where)
        {
            if (dims.Length == 0)
                return Shape.NewScalar();
            if (where is null)
                return Shape.NpyIterOutputShape(dims, new[] { x1.Shape, x2.Shape });
            return Shape.NpyIterOutputShape(dims, new[] { x1.Shape, x2.Shape, where.Shape });
        }

        /// <summary>
        ///     NumPy's trivial loop: one kernel call covers the whole result when there is no mask and each input is one element
        ///     (stride 0) or walks the result's linear order — the result's dimensions in a shared contiguous order, or, for a
        ///     0-d / 1-D result, any 1-D input at its own byte stride.
        /// </summary>
        /// <param name="kernel">The comparison kernel.</param>
        /// <param name="t1">The first input (<c>StringDType</c>).</param>
        /// <param name="t2">The second input.</param>
        /// <param name="result">The bool output.</param>
        /// <param name="where">The mask, or null (a masked call takes the iterator).</param>
        /// <param name="aux">The pair context.</param>
        /// <returns>True when the call was computed here.</returns>
        private static unsafe bool TryTrivial(NDInnerLoopFunc kernel, NDArray t1, NDArray t2, NDArray result, NDArray where,
                                              VStringPairAux* aux)
        {
            if (where is not null)
                return false;
            var rs = result.Shape;
            void** ptrs = stackalloc void*[3];
            long* strides = stackalloc long[3];
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            if (rs.NDim <= 1)
            {
                if (!Linear1D(t1, rs) || !Linear1D(t2, rs))
                    return false;
                strides[0] = t1.size == 1 ? 0 : t1.Shape.strides[0] * (long)cell;
                strides[1] = t2.size == 1 ? 0 : t2.Shape.strides[0] * (long)cell;
                strides[2] = rs.NDim == 0 || result.size == 1 ? 0 : rs.strides[0];
            }
            else
            {
                bool c = rs.IsContiguous, f = rs.IsFContiguous;
                if (!c && !f)
                    return false;
                if (!LinearOperand(t1, rs, ref c, ref f) || !LinearOperand(t2, rs, ref c, ref f) || !(c || f))
                    return false;
                strides[0] = t1.size == 1 ? 0 : cell;
                strides[1] = t2.size == 1 ? 0 : cell;
                strides[2] = 1;
            }
            ptrs[0] = (byte*)t1.Address + t1.Shape.offset * (long)cell;
            ptrs[1] = (byte*)t2.Address + t2.Shape.offset * (long)cell;
            ptrs[2] = (byte*)result.Address + result.Shape.offset;
            kernel(ptrs, strides, result.size, aux);
            GC.KeepAlive(t1);
            GC.KeepAlive(t2);
            GC.KeepAlive(result);
            return true;
        }

        /// <summary>
        ///     Whether an input can be walked linearly next to a multi-dimensional result: a single element always (stride 0),
        ///     otherwise it must have the result's dimensions and share one of the result's contiguous orders.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The result's shape.</param>
        /// <param name="c">In: the result (and earlier operands) are C-contiguous; out: and this one too.</param>
        /// <param name="f">In / out: the same for F order.</param>
        /// <returns>False when the input needs the iterator.</returns>
        private static bool LinearOperand(NDArray x, Shape rs, ref bool c, ref bool f)
        {
            if (x.size == 1)
                return true;
            var xs = x.Shape;
            if (!SameDimensions(xs.dimensions, rs.dimensions))
                return false;
            c &= xs.IsContiguous;
            f &= xs.IsFContiguous;
            return c || f;
        }

        /// <summary>
        ///     Whether an input can be walked linearly next to a 0-d or 1-D result: one element, or a 1-D operand of the result's
        ///     length at any stride.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The result's shape.</param>
        /// <returns>False when the input needs the iterator.</returns>
        private static bool Linear1D(NDArray x, Shape rs)
            => x.size == 1 || x.ndim == 1 && rs.NDim == 1 && x.Shape.dimensions[0] == rs.dimensions[0];

        /// <summary>Whether two dimension lists are equal.</summary>
        /// <param name="a">The first dimensions (null for 0-d).</param>
        /// <param name="b">The second dimensions (null for 0-d).</param>
        /// <returns>True when equal.</returns>
        private static bool SameDimensions(long[] a, long[] b)
        {
            a ??= Array.Empty<long>();
            b ??= Array.Empty<long>();
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     The NDIter pass: <c>EXTERNAL_LOOP</c> over (x1, x2, out[, mask]), each input with its own descriptor (no cast,
        ///     so no buffer: the kernel reads every cell in its own array), the mask as the ARRAYMASK operand.
        /// </summary>
        /// <param name="kernel">The comparison kernel.</param>
        /// <param name="t1">The first input.</param>
        /// <param name="t2">The second input.</param>
        /// <param name="result">The bool output.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="aux">The pair context.</param>
        private static unsafe void RunIterator(NDInnerLoopFunc kernel, NDArray t1, NDArray t2, NDArray result, NDArray where,
                                               VStringPairAux* aux)
        {
            var descrs = where is null ? new DType[3] : new DType[4];
            descrs[2] = DType.Boolean;
            var ops = where is null ? new[] { t1, t2, result } : new[] { t1, t2, result, where };
            var flags = where is null
                ? new[]
                {
                    NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST,
                }
                : new[]
                {
                    NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.WRITEMASKED | NDIterPerOpFlags.NO_BROADCAST,
                    NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK,
                };
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops, NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.REFS_OK,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING, flags, opDescrs: descrs);
            iter.ForEach(kernel, aux);
        }

        /// <summary>
        ///     The comparison as a managed <see cref="NDInnerLoopFunc"/> (hosts without dynamic code; the IL kernel is tested
        ///     against it): the same pair helpers in a C# loop.
        /// </summary>
        /// <param name="op">The comparison of A against B.</param>
        /// <param name="aIndex">The operand slot A is read from.</param>
        /// <returns>The kernel.</returns>
        internal static unsafe NDInnerLoopFunc ManagedKernel(TextCompareOp op, int aIndex)
            => (dataptrs, strides, count, auxdata) =>
            {
                byte* pa = (byte*)dataptrs[aIndex], pb = (byte*)dataptrs[1 - aIndex], po = (byte*)dataptrs[2];
                long sa = strides[aIndex], sb = strides[1 - aIndex], so = strides[2];
                var aux = (VStringPairAux*)auxdata;
                for (long i = 0; i < count; i++, pa += sa, pb += sb, po += so)
                    *po = op switch
                    {
                        TextCompareOp.Equal => NDVStringCells.Equal(pa, pb, aux, false),
                        TextCompareOp.NotEqual => NDVStringCells.Equal(pa, pb, aux, true),
                        TextCompareOp.Less => NDVStringCells.Less(pa, pb, aux, false),
                        _ => NDVStringCells.Less(pa, pb, aux, true),
                    };
            };
    }
}
