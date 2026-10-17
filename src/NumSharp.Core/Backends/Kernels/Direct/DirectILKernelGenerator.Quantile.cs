using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Statistics;
using NumSharp.Utilities;

// =============================================================================
// DirectILKernelGenerator.Quantile — quantile / percentile / median IL kernel
// =============================================================================
//
// Why this exists:
//   The old QuantileEngine path was a 15-arm switch (NPTypeCode → ComputeForType
//   <T>) followed by a hand-written C# row loop. That violated two of the
//   project's hard rules:
//       1. "No per-dtype switch — emit IL via DirectILKernelGenerator."
//       2. "Loops must be IL- or NDIter-driven."
//
// What this kernel does:
//   * Generates one `DynamicMethod` per `(srcDtype, outDtype, lerpDtype, qDtype,
//     QuantileMethod, ignoreNaN, medianMean)` tuple and caches it. The first call
//     for a given tuple pays the emit cost; subsequent calls are a
//     ConcurrentDictionary lookup + delegate invoke.
//   * The emitted method body contains the OUTER row-loop in IL — there is
//     no managed `for` driving the rows.
//   * Per row, the kernel calls a generic helper `ProcessQuantileRow<T, TOut, TM, TQ>`
//     which is JIT-specialized at call site. The `typeof(T) == typeof(X)`
//     chains inside the helper are JIT-folded to constant `true`/`false` per
//     specialization, so the dispatch happens once at first-call codegen,
//     not on every iteration.
//
// The four types (NumPy 2.4.2's _quantile / _lerp, numpy/lib/_function_base_impl.py):
//   * T    — the data's dtype D: the partition, the neighbours, and `diff_b_a = b - a` (which WRAPS for integers and
//            ROUNDS for float16/float32, before any promotion).
//   * TQ   — q's dtype G: NumPy computes the virtual index, the neighbours and the weight t in it (a float16 q array
//            rounds `(n - 1) * q` to float16). Python numbers are weak and became float64 / int64 q arrays, both exact
//            in double here. See Statistics/QuantileIndexes.cs.
//   * TM   — the lerp's dtype M: `diff_b_a * t` and `a + ...` are ufuncs of D and t's dtype (G for a q array, a weak
//            Python float for a Python q), so M = result_type(D, G), or D for float data under a weak q. long here is
//            the int64 lerp NumPy runs for an integer q ARRAY with an integer virtual index (weibull,
//            averaged_inverted_cdf, interpolated_inverted_cdf): its two neighbours are always one element.
//   * TOut — the buffer the kernel writes: M, or out's dtype O when the caller's out (a float or complex dtype other
//            than M) receives the lerp — NumPy's `subtract(b, diff_b_a * (1 - t), out=out, dtype=type(out.dtype))`
//            computes the t >= 0.5 branch in O itself, so it cannot be written in M and cast afterwards.
//
// Why not emit the whole partition body in IL:
//   Hoare partition has nontrivial control flow (median-of-three pivot,
//   left/right pointer dance, swaps, depth-limited heap-sort fallback).
//   Inlining all of it in raw IL would inflate the kernel by ~10x without a
//   speedup — the JIT-specialized generic helper is already inlined hot code.
//   This mirrors how `DirectILKernelGenerator.Scan` and `DirectILKernelGenerator.Reduction
//   .Arg` keep the inner algorithm as a typed C# helper while the outer
//   driver is IL-emitted.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Public API

        /// <summary>
        ///     The emitted quantile kernel: one call computes every row (reduced line) of the staged data.
        /// </summary>
        /// <param name="srcBase">The staged data: <paramref name="outer"/> contiguous lines of <paramref name="n"/>
        ///     elements of the data's dtype.</param>
        /// <param name="scratchBase">A scratch line of <paramref name="n"/> elements (the partition's working copy).</param>
        /// <param name="outer">The number of lines.</param>
        /// <param name="n">The line's length (at least 1).</param>
        /// <param name="kSorted">The ascending, distinct partition indexes (the plain path; unused by the nan path).</param>
        /// <param name="nKs">How many partition indexes.</param>
        /// <param name="cellIndex">The plain path's plan: per quantile, the previous and next index (in [0, n - 1]).</param>
        /// <param name="cellWeight">The plain path's plan: per quantile, the weight t and <c>1 - t</c> (values of q's
        ///     dtype).</param>
        /// <param name="q">The quantiles (values of q's dtype), read by the nan path to plan each line.</param>
        /// <param name="nQs">The number of quantiles.</param>
        /// <param name="dstBase">The result buffer (the kernel's output dtype), quantile-major.</param>
        /// <param name="dstOuterStride">The distance in elements between two quantiles' blocks (0 for a scalar q).</param>
        /// <param name="ignoreNaN">1 for the nan forms.</param>
        /// <param name="rowKScratch">The nan path's partition indexes (2 per quantile).</param>
        /// <param name="rowRaw">The nan path's neighbours (2 per quantile).</param>
        /// <param name="rowWeight">The nan path's weights (2 per quantile).</param>
        public unsafe delegate void QuantileKernel(
            void* srcBase, void* scratchBase,
            long outer, int n,
            int* kSorted, int nKs,
            int* cellIndex, double* cellWeight,
            double* q, int nQs,
            void* dstBase, long dstOuterStride,
            int ignoreNaN, int* rowKScratch, long* rowRaw, double* rowWeight);

        /// <summary>
        ///     The index plan of a quantile call (NumPy's <c>get_virtual_index</c> / <c>_get_indexes</c> / <c>_get_gamma</c>
        ///     in q's precision, see <see cref="QuantileIndexes"/>), for a line of <c>n</c> elements: per quantile, two raw
        ///     indexes (the selection method's index twice, or the previous and next neighbour as intp, before the
        ///     partition's negative adjustment) and two weights (t and <c>1 - t</c>; 0 and 1 for a selection method).
        /// </summary>
        /// <param name="methodInt">The <see cref="QuantileMethod"/> (a selection method uses the take index).</param>
        /// <param name="n">The line's length (0 for an empty line).</param>
        /// <param name="q">The quantiles (values of q's dtype).</param>
        /// <param name="nQs">The number of quantiles.</param>
        /// <param name="inexact">1 when the data is a float array (a NaN virtual index then reads the last element).</param>
        /// <param name="raw">Receives 2 * <paramref name="nQs"/> raw indexes.</param>
        /// <param name="weight">Receives 2 * <paramref name="nQs"/> weights.</param>
        internal unsafe delegate void QuantilePlanBuilder(int methodInt, long n, double* q, int nQs, int inexact,
            long* raw, double* weight);

        internal readonly struct QuantileKey : IEquatable<QuantileKey>
        {
            public readonly NPTypeCode SrcType;
            /// <summary>The buffer the kernel writes: the lerp's dtype, or the caller's out dtype when it receives the
            /// lerp (see the file header).</summary>
            public readonly NPTypeCode OutType;
            /// <summary>The lerp's dtype M (Int64 for the integer-weight lerp).</summary>
            public readonly NPTypeCode LerpType;
            /// <summary>q's precision (Half / Single / Double): the nan path plans each line in it.</summary>
            public readonly NPTypeCode QType;
            public readonly QuantileMethod Method;
            public readonly bool IgnoreNaN;
            /// <summary>Bakes NumPy's mean-of-middle <c>np.median</c> reduction into the kernel (see
            /// <see cref="QuantileEngine.Compute"/>'s <c>medianMean</c>). Part of the cache key so the
            /// median and quantile(0.5) kernels are distinct compiled bodies.</summary>
            public readonly bool MedianMean;

            public QuantileKey(NPTypeCode srcType, NPTypeCode outType, NPTypeCode lerpType, NPTypeCode qType,
                QuantileMethod method, bool ignoreNaN, bool medianMean)
            {
                SrcType = srcType; OutType = outType; LerpType = lerpType; QType = qType; Method = method;
                IgnoreNaN = ignoreNaN; MedianMean = medianMean;
            }

            public bool Equals(QuantileKey o) => SrcType == o.SrcType && OutType == o.OutType && LerpType == o.LerpType
                && QType == o.QType && Method == o.Method && IgnoreNaN == o.IgnoreNaN && MedianMean == o.MedianMean;
            public override bool Equals(object obj) => obj is QuantileKey o && Equals(o);
            public override int GetHashCode()
                => HashCode.Combine((int)SrcType, (int)OutType, (int)LerpType, (int)QType, (int)Method, IgnoreNaN, MedianMean);
        }

        internal static readonly ConcurrentDictionary<QuantileKey, QuantileKernel> _quantileKernelCache = new();

        /// <summary>The plan builders by q's precision (<see cref="BuildQuantilePlan{TQ}"/> instantiated once each).</summary>
        private static readonly ConcurrentDictionary<NPTypeCode, QuantilePlanBuilder> _quantilePlanCache = new();

        /// <summary>
        ///     Run the cached quantile kernel for the given dtypes. First call for a
        ///     tuple emits and caches the DynamicMethod; later calls jump straight into the
        ///     specialized native code.
        /// </summary>
        /// <param name="srcType">The data's dtype D.</param>
        /// <param name="outType">The buffer's dtype: <paramref name="lerpType"/>, or the caller's out dtype when it receives
        ///     the lerp.</param>
        /// <param name="lerpType">The lerp's dtype M (the data's dtype for a selection method).</param>
        /// <param name="qType">q's precision: Half, Single or Double.</param>
        /// <param name="method">The method (after the engine's remaps).</param>
        /// <param name="srcBase">The staged data.</param>
        /// <param name="scratchBase">A scratch line.</param>
        /// <param name="outer">The number of lines.</param>
        /// <param name="n">The line's length (at least 1).</param>
        /// <param name="kSorted">The plain path's ascending partition indexes.</param>
        /// <param name="nKs">How many.</param>
        /// <param name="cellIndex">The plain path's per-quantile neighbours (partition-ready).</param>
        /// <param name="cellWeight">The plain path's per-quantile weights.</param>
        /// <param name="q">The quantiles.</param>
        /// <param name="nQs">How many.</param>
        /// <param name="dstBase">The result buffer.</param>
        /// <param name="dstOuterStride">The distance between two quantiles' blocks (0 for a scalar q).</param>
        /// <param name="ignoreNaN">The nan forms.</param>
        /// <param name="rowKScratch">The nan path's partition indexes (2 per quantile).</param>
        /// <param name="rowRaw">The nan path's neighbours (2 per quantile).</param>
        /// <param name="rowWeight">The nan path's weights (2 per quantile).</param>
        /// <param name="medianMean">
        ///     Reduce even-size slices with NumPy's mean-of-middle <c>np.median</c> rule
        ///     (<c>(a+b)/2</c>) instead of the quantile <c>_lerp</c> at q=0.5 — the two differ
        ///     by up to 1 ULP. Set only by <c>np.median</c>/<c>np.nanmedian</c>; baked into the
        ///     cached kernel via <see cref="QuantileKey.MedianMean"/>.
        /// </param>
        /// <exception cref="ValueError">The nan path: a line whose plan holds an index the partition refuses (NumPy's
        ///     <c>kth(=k) out of bounds (m)</c>, m the line's NaN-free length) — only with a float16 / float32 q.</exception>
        public static unsafe void Quantile(
            NPTypeCode srcType, NPTypeCode outType, NPTypeCode lerpType, NPTypeCode qType, QuantileMethod method,
            void* srcBase, void* scratchBase, long outer, int n,
            int* kSorted, int nKs, int* cellIndex, double* cellWeight,
            double* q, int nQs,
            void* dstBase, long dstOuterStride,
            bool ignoreNaN = false, int* rowKScratch = null, long* rowRaw = null, double* rowWeight = null,
            bool medianMean = false)
        {
            var key = new QuantileKey(srcType, outType, lerpType, qType, method, ignoreNaN, medianMean);
            var kernel = _quantileKernelCache.GetOrAdd(key, k => EmitQuantileKernel(k));
            kernel(srcBase, scratchBase, outer, n, kSorted, nKs, cellIndex, cellWeight, q, nQs, dstBase, dstOuterStride,
                ignoreNaN ? 1 : 0, rowKScratch, rowRaw, rowWeight);
        }

        /// <summary>
        ///     Builds the plain path's index plan (see <see cref="QuantilePlanBuilder"/>) in q's precision: the builder for
        ///     that precision is <see cref="BuildQuantilePlan{TQ}"/> instantiated once and cached, so the engine never
        ///     branches on the dtype.
        /// </summary>
        /// <param name="qType">q's precision: Half, Single, Double, or an integer q's dtype (int8 to uint64; a bool q passes
        ///     Int64); any other dtype computes in double.</param>
        /// <param name="method">The method (after the engine's remaps).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantiles.</param>
        /// <param name="nQs">How many.</param>
        /// <param name="inexact">Whether the data is a float array.</param>
        /// <param name="raw">Receives the raw indexes (2 per quantile).</param>
        /// <param name="weight">Receives the weights (2 per quantile).</param>
        /// <exception cref="OverflowException">An integer q: NumPy's conversion of <c>n</c> / <c>n - 1</c> into its dtype, or
        ///     the setitem of -1 into an unsigned index array (see <see cref="QuantileIndexes.IntegerPlan{TQ}"/>).</exception>
        internal static unsafe void QuantilePlan(NPTypeCode qType, QuantileMethod method, long n, double* q, int nQs,
            bool inexact, long* raw, double* weight)
        {
            var builder = _quantilePlanCache.GetOrAdd(qType, static t =>
            {
                Type clr = QuantilePrecisionType(t);
                var generic = typeof(DirectILKernelGenerator).GetMethod(nameof(BuildQuantilePlan),
                    BindingFlags.NonPublic | BindingFlags.Static) ?? throw new MissingMethodException(nameof(BuildQuantilePlan));
                return (QuantilePlanBuilder)generic.MakeGenericMethod(clr).CreateDelegate(typeof(QuantilePlanBuilder));
            });
            builder((int)method, n, q, nQs, inexact ? 1 : 0, raw, weight);
        }

        #endregion

        #region IL Emission

        private static QuantileKernel EmitQuantileKernel(QuantileKey key)
        {
            // Emits:
            //
            //   for (long i = 0; i < outer; i++) {
            //       ProcessQuantileRow<T, TOut, TM, TQ>(
            //           (byte*)srcBase + i * n * sizeof(T),
            //           scratchBase,
            //           n, kSorted, nKs, cellIndex, cellWeight, q, nQs,
            //           (byte*)dstBase + i * sizeof(TOut),
            //           dstOuterStride,
            //           (int)method, ignoreNaN, medianMean,
            //           rowKScratch, rowRaw, rowWeight, &planLength, &planCount);
            //   }
            //
            // planLength / planCount are locals of the kernel (-1 / 0 at entry): the nan path keeps in them the NaN-free
            // length whose plan rowRaw / rowWeight / rowKScratch hold and that plan's partition count, so consecutive
            // rows of the same length reuse it (see ProcessQuantileRowNaN).
            //
            // The cast to `T*` / `TOut*` happens inside the JIT-specialized helper.
            // The loop variable, address arithmetic, and outer comparison are all
            // pure IL — no managed for-loop drives the rows.

            int srcSize = GetTypeSize(key.SrcType);
            int outSize = GetTypeSize(key.OutType);
            Type srcClr = GetClrType(key.SrcType);
            Type outClr = GetClrType(key.OutType);
            Type lerpClr = GetClrType(key.LerpType);
            // q's precision: float16 and float32 round, an integer q wraps in its dtype; every other q computes in double.
            Type qClr = QuantilePrecisionType(key.QType);

            var dm = new DynamicMethod(
                name: $"Quantile_{key.SrcType}_{key.OutType}_{key.LerpType}_{key.QType}_{key.Method}",
                returnType: typeof(void),
                parameterTypes: new[] {
                    typeof(void*),   // 0 srcBase
                    typeof(void*),   // 1 scratchBase
                    typeof(long),    // 2 outer
                    typeof(int),     // 3 n
                    typeof(int*),    // 4 kSorted
                    typeof(int),     // 5 nKs
                    typeof(int*),    // 6 cellIndex
                    typeof(double*), // 7 cellWeight
                    typeof(double*), // 8 q
                    typeof(int),     // 9 nQs
                    typeof(void*),   // 10 dstBase
                    typeof(long),    // 11 dstOuterStride
                    typeof(int),     // 12 ignoreNaN
                    typeof(int*),    // 13 rowKScratch
                    typeof(long*),   // 14 rowRaw
                    typeof(double*), // 15 rowWeight
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var helper = typeof(DirectILKernelGenerator)
                .GetMethod(nameof(ProcessQuantileRow), BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new MissingMethodException(nameof(ProcessQuantileRow));
            var helperSpecialized = helper.MakeGenericMethod(srcClr, outClr, lerpClr, qClr);

            var locI = il.DeclareLocal(typeof(long));
            var locPlanLength = il.DeclareLocal(typeof(long));
            var locPlanCount = il.DeclareLocal(typeof(int));
            var loopStart = il.DefineLabel();
            var loopCond = il.DefineLabel();

            // planLength = -1 (no plan yet: every NaN-free length that plans is 1 or more), planCount = 0
            il.Emit(OpCodes.Ldc_I8, -1L);
            il.Emit(OpCodes.Stloc, locPlanLength);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locPlanCount);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, loopCond);

            // --- loop body ---
            il.MarkLabel(loopStart);

            //   srcRow = (byte*)srcBase + i * n * sizeof(T)
            il.Emit(OpCodes.Ldarg_0);                  // srcBase
            il.Emit(OpCodes.Ldloc, locI);              // i  (long)
            il.Emit(OpCodes.Ldarg_3);                  // n  (int)
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Mul);                      // i * n
            il.Emit(OpCodes.Ldc_I8, (long)srcSize);
            il.Emit(OpCodes.Mul);                      // i * n * sizeof(T)
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);                      // srcBase + offset → void*

            //   scratchBase, n, kSorted, nKs, cellIndex, cellWeight, q, nQs
            il.Emit(OpCodes.Ldarg_1);                  // scratchBase
            il.Emit(OpCodes.Ldarg_3);                  // n
            il.Emit(OpCodes.Ldarg_S, (byte)4);         // kSorted
            il.Emit(OpCodes.Ldarg_S, (byte)5);         // nKs
            il.Emit(OpCodes.Ldarg_S, (byte)6);         // cellIndex
            il.Emit(OpCodes.Ldarg_S, (byte)7);         // cellWeight
            il.Emit(OpCodes.Ldarg_S, (byte)8);         // q
            il.Emit(OpCodes.Ldarg_S, (byte)9);         // nQs

            //   dstCell = (byte*)dstBase + i * sizeof(TOut)
            il.Emit(OpCodes.Ldarg_S, (byte)10);        // dstBase
            il.Emit(OpCodes.Ldloc, locI);              // i  (long)
            il.Emit(OpCodes.Ldc_I8, (long)outSize);
            il.Emit(OpCodes.Mul);                      // i * sizeof(TOut)
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);                      // dstBase + offset → void*

            //   dstOuterStride, methodInt, ignoreNaN (baked), medianMean (baked), rowKScratch, rowRaw, rowWeight
            il.Emit(OpCodes.Ldarg_S, (byte)11);              // dstOuterStride
            il.Emit(OpCodes.Ldc_I4, (int)key.Method);        // method baked in
            il.Emit(OpCodes.Ldc_I4, key.IgnoreNaN ? 1 : 0);  // ignoreNaN baked in
            il.Emit(OpCodes.Ldc_I4, key.MedianMean ? 1 : 0); // medianMean baked in
            il.Emit(OpCodes.Ldarg_S, (byte)13);              // rowKScratch
            il.Emit(OpCodes.Ldarg_S, (byte)14);              // rowRaw
            il.Emit(OpCodes.Ldarg_S, (byte)15);              // rowWeight
            // &planLength, &planCount: the locals live in this frame for the whole call (conv.u turns the managed
            // address of a stack local into the unmanaged pointer the helper takes; a stack slot never moves).
            il.Emit(OpCodes.Ldloca, locPlanLength);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Ldloca, locPlanCount);
            il.Emit(OpCodes.Conv_U);

            il.Emit(OpCodes.Call, helperSpecialized);

            //   i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            // --- condition: i < outer ---
            il.MarkLabel(loopCond);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);                  // outer
            il.Emit(OpCodes.Blt, loopStart);

            il.Emit(OpCodes.Ret);

            return (QuantileKernel)dm.CreateDelegate(typeof(QuantileKernel));
        }

        #endregion

        #region Index plan (q's precision)

        /// <summary>
        ///     <see cref="QuantilePlanBuilder"/> in q's precision TQ: per quantile, NumPy's take index (twice) for a
        ///     selection method, or its neighbours and weights for an interpolating one (<see cref="QuantileIndexes"/>).
        /// </summary>
        /// <typeparam name="TQ">q's precision (double, float or Half).</typeparam>
        /// <param name="methodInt">The <see cref="QuantileMethod"/>.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantiles.</param>
        /// <param name="nQs">How many.</param>
        /// <param name="inexact">1 when the data is a float array.</param>
        /// <param name="raw">Receives the raw indexes (2 per quantile).</param>
        /// <param name="weight">Receives the weights (2 per quantile).</param>
        internal static unsafe void BuildQuantilePlan<TQ>(int methodInt, long n, double* q, int nQs, int inexact,
            long* raw, double* weight) where TQ : unmanaged
        {
            var method = (QuantileMethod)methodInt;
            // An integer q computes its indexes in its integer dtype (JIT-folded: one branch survives per instantiation).
            if (QuantileIndexes.IsIntegerPrecision<TQ>())
            {
                QuantileIndexes.IntegerPlan<TQ>(method, n, q, nQs, inexact != 0, raw, weight);
                return;
            }
            bool take = QuantileEngine.IsDiscrete(method);
            for (int j = 0; j < nQs; j++)
            {
                if (take)
                {
                    long index = QuantileIndexes.TakeIndex<TQ>(method, n, q[j]);
                    raw[2 * j] = index;
                    raw[2 * j + 1] = index;
                    weight[2 * j] = 0;
                    weight[2 * j + 1] = 1;
                }
                else
                {
                    QuantileIndexes.Neighbours<TQ>(method, n, q[j], inexact != 0, out raw[2 * j], out raw[2 * j + 1],
                        out weight[2 * j], out weight[2 * j + 1]);
                }
            }
        }

        /// <summary>
        ///     The CLR type that stands for q's precision in the plan and the kernel: float16 / float32 (rounded), an integer
        ///     q's own integer type (wrapped), or double for every other q (a float64 q, a Python number, a decimal / char
        ///     q, whose index arithmetic is exact in double).
        /// </summary>
        /// <param name="qType">q's precision as the engine passes it.</param>
        /// <returns>The type argument of <see cref="BuildQuantilePlan{TQ}"/> and the kernel's row helper.</returns>
        private static Type QuantilePrecisionType(NPTypeCode qType)
            => qType is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16
                or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
                ? GetClrType(qType)
                : typeof(double);

        #endregion

        #region Generic per-row helper (JIT-specialized at call site)

        /// <summary>
        ///     Process one row: copy src → scratch, NaN-prescan (floats only), partition
        ///     at the requested k indices, then emit one output cell per quantile fraction.
        ///     The <c>typeof(T) == typeof(X)</c> guards are JIT-folded at specialization
        ///     time, so the body collapses to dtype-specific straight-line code on first
        ///     call. This is how compliance with the "no runtime dtype switch" rule is
        ///     achieved without hand-emitting the entire partition body in IL.
        /// </summary>
        /// <typeparam name="T">The data's element type.</typeparam>
        /// <typeparam name="TOut">The buffer's element type (the lerp's, or the caller's out dtype).</typeparam>
        /// <typeparam name="TM">The lerp's precision (double / float / Half; long for the integer-weight lerp).</typeparam>
        /// <typeparam name="TQ">q's precision (the nan path plans each line in it).</typeparam>
        /// <param name="srcRow">The row.</param>
        /// <param name="scratchBase">The scratch line.</param>
        /// <param name="n">The row's length.</param>
        /// <param name="kSorted">The plain path's partition indexes.</param>
        /// <param name="nKs">How many.</param>
        /// <param name="cellIndex">The plain path's per-quantile neighbours.</param>
        /// <param name="cellWeight">The plain path's per-quantile weights.</param>
        /// <param name="q">The quantiles (the nan path's plans).</param>
        /// <param name="nQs">How many quantiles.</param>
        /// <param name="dstCell">This row's first result cell.</param>
        /// <param name="dstOuterStride">The distance between two quantiles' cells.</param>
        /// <param name="methodInt">The method.</param>
        /// <param name="ignoreNaN">1 for the nan forms.</param>
        /// <param name="medianMean">1 for np.median's mean of the middle.</param>
        /// <param name="rowKScratch">The nan path's partition indexes.</param>
        /// <param name="rowRaw">The nan path's neighbours.</param>
        /// <param name="rowWeight">The nan path's weights.</param>
        /// <param name="planLength">The kernel's record of the NaN-free length whose plan the nan path's scratch holds (-1
        ///     before the first plan of the call).</param>
        /// <param name="planCount">The kernel's record of that plan's partition-index count.</param>
        /// <exception cref="ValueError">The nan path's partition refuses an index of this row's plan (see
        ///     <see cref="Quantile"/>).</exception>
        /// <exception cref="OverflowException">The nan path with an integer q: NumPy refuses the index arithmetic for this
        ///     row's NaN-free length (see <see cref="QuantileIndexes.IntegerPlan{TQ}"/>).</exception>
        internal static unsafe void ProcessQuantileRow<T, TOut, TM, TQ>(
            void* srcRow, void* scratchBase, int n,
            int* kSorted, int nKs, int* cellIndex, double* cellWeight,
            double* q, int nQs,
            void* dstCell, long dstOuterStride,
            int methodInt, int ignoreNaN, int medianMean, int* rowKScratch, long* rowRaw, double* rowWeight,
            long* planLength, int* planCount)
            where T : unmanaged, IComparable<T>
            where TOut : unmanaged
            where TM : unmanaged
            where TQ : unmanaged
        {
            T* src = (T*)srcRow;
            T* scratch = (T*)scratchBase;
            TOut* dst = (TOut*)dstCell;
            var method = (QuantileMethod)methodInt;
            bool medianMeanB = medianMean != 0;   // NumPy np.median mean-of-middle rule (baked)

            // 1. memcpy
            Buffer.MemoryCopy(src, scratch, (long)n * sizeof(T), (long)n * sizeof(T));

            // ── NaN-ignoring path (np.nanmedian / np.nanquantile / np.nanpercentile) ──
            // The engine only routes float dtypes here (ints carry no NaN), so the
            // compaction guards below collapse to a single typed loop per
            // specialization. Each row is independently compacted (NaNs dropped to a
            // local valid-count m), then the partition/interpolate runs against the
            // first m elements with per-row indices — m varies per row, so kSorted
            // (sized for n) cannot be reused; the row's plan is built into rowRaw/rowWeight.
            if (ignoreNaN != 0)
            {
                ProcessQuantileRowNaN<T, TOut, TM, TQ>(scratch, n, q, nQs, dst, dstOuterStride, method, medianMeanB,
                    rowKScratch, rowRaw, rowWeight, planLength, planCount);
                return;
            }

            // 2. NaN prescan — only floats can carry NaN. JIT folds the guards away
            //    for non-float specializations. float/double use a Vector<T> self-equality
            //    scan (a NaN lane fails v==v); Half has no Vector arithmetic so stays scalar.
            //    The first NaN's position is kept: NumPy fills a NaN row with `arr[-1, ...]`, the data's own NaN (its sign
            //    and payload, e.g. np.nan's positive quiet NaN), which an out= receives bit for bit.
            int nanAt = -1;
            if (typeof(T) == typeof(double))     nanAt = FirstNaN((double*)scratch, n);
            else if (typeof(T) == typeof(float)) nanAt = FirstNaN((float*)scratch, n);
            else if (typeof(T) == typeof(Half))
            {
                var p = (Half*)scratch;
                for (int i = 0; i < n; i++) { if (Half.IsNaN(p[i])) { nanAt = i; break; } }
            }
            bool hasNaN = nanAt >= 0;
            // The row's NaN as a double (a float32 / float16 NaN widens with its sign and payload), read before the
            // partition moves it.
            double rowNaN = hasNaN ? ToDouble(scratch[nanAt]) : 0;

            // 3. Partition unless NaN-tainted (saves work — the answer will be NaN).
            //    n==1 is a no-op (already sorted).
            if (!hasNaN && n > 1)
                QuickSelect.PartitionAtMany(scratch, n, kSorted, nKs);

            // 4. Emit one cell per q, from the call's plan (the same for every row: n is the same).
            for (int j = 0; j < nQs; j++)
            {
                TOut* outCell = dst + (long)j * dstOuterStride;
                if (hasNaN) { WriteNaNCell(outCell, rowNaN); continue; }
                WriteCell<T, TOut, TM>(scratch, cellIndex[2 * j], cellIndex[2 * j + 1], cellWeight[2 * j],
                    cellWeight[2 * j + 1], method, medianMeanB, outCell);
            }
        }

        /// <summary>
        ///     SIMD "where is this row's first NaN?" via <c>Vector&lt;T&gt;</c> self-equality —
        ///     a NaN lane fails <c>v == v</c>, so <see cref="System.Numerics.Vector.EqualsAll{T}"/>
        ///     turning false flags the vector, whose lanes are then scanned. Early-exits on the first NaN vector, scalar tail
        ///     for the remainder. This is the plain-path (non-<c>nan*</c>) prescan that gates
        ///     whether the row partitions at all; now that the partition itself is ~3× faster it
        ///     is a meaningful share of a float median, so it is vectorized to match.
        /// </summary>
        /// <param name="p">The row.</param>
        /// <param name="n">Its length.</param>
        /// <returns>The index of the first NaN, or -1 when there is none.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe int FirstNaN(float* p, int n)
        {
            int i = 0;
            int w = System.Numerics.Vector<float>.Count;
            if (System.Numerics.Vector.IsHardwareAccelerated && n >= w)
            {
                for (; i <= n - w; i += w)
                {
                    var v = Unsafe.ReadUnaligned<System.Numerics.Vector<float>>(p + i);
                    if (!System.Numerics.Vector.EqualsAll(v, v)) break;   // the scalar loop below finds the lane
                }
            }
            for (; i < n; i++) if (float.IsNaN(p[i])) return i;
            return -1;
        }

        /// <inheritdoc cref="FirstNaN(float*, int)"/>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe int FirstNaN(double* p, int n)
        {
            int i = 0;
            int w = System.Numerics.Vector<double>.Count;
            if (System.Numerics.Vector.IsHardwareAccelerated && n >= w)
            {
                for (; i <= n - w; i += w)
                {
                    var v = Unsafe.ReadUnaligned<System.Numerics.Vector<double>>(p + i);
                    if (!System.Numerics.Vector.EqualsAll(v, v)) break;   // the scalar loop below finds the lane
                }
            }
            for (; i < n; i++) if (double.IsNaN(p[i])) return i;
            return -1;
        }

        /// <summary>
        ///     Per-row NaN-ignoring quantile: NumPy's <c>_nanquantile_1d</c> — the NaNs removed (valid count m), an all-NaN
        ///     row NaN for every q (<c>np.full(q.shape, np.nan)</c>), otherwise <c>_quantile</c> of the m-element slice: its
        ///     plan in q's precision, the partition's kth validation, the partition and the cells.
        /// </summary>
        /// <typeparam name="T">The data's element type (a float type: the engine routes only those here).</typeparam>
        /// <typeparam name="TOut">The buffer's element type.</typeparam>
        /// <typeparam name="TM">The lerp's precision.</typeparam>
        /// <typeparam name="TQ">q's precision.</typeparam>
        /// <param name="scratch">A copy of the row (length n).</param>
        /// <param name="n">The row's length.</param>
        /// <param name="q">The quantiles.</param>
        /// <param name="nQs">How many.</param>
        /// <param name="dst">This row's first result cell.</param>
        /// <param name="dstOuterStride">The distance between two quantiles' cells.</param>
        /// <param name="method">The method.</param>
        /// <param name="medianMean">np.median's mean of the middle.</param>
        /// <param name="rowKScratch">Scratch for the partition indexes (2 per quantile).</param>
        /// <param name="rowRaw">Scratch for the neighbours (2 per quantile).</param>
        /// <param name="rowWeight">Scratch for the weights (2 per quantile).</param>
        /// <param name="planLength">The NaN-free length whose plan <paramref name="rowRaw"/>, <paramref name="rowWeight"/>
        ///     and <paramref name="rowKScratch"/> hold (-1 before the call's first plan); updated when this slice plans.</param>
        /// <param name="planCount">The number of partition indexes of that plan in <paramref name="rowKScratch"/>.</param>
        /// <exception cref="ValueError">The slice's partition refuses an index (NumPy's <c>kth(=k) out of bounds (m)</c>):
        ///     only a float16 / float32 q over a slice long enough for q's rounding to reach m (or NaN).</exception>
        /// <exception cref="OverflowException">An integer q whose index arithmetic NumPy refuses for this slice's length
        ///     (see <see cref="QuantileIndexes.IntegerPlan{TQ}"/>).</exception>
        /// <remarks>
        ///     A slice's plan depends on nothing but its NaN-free length m, so the slices of one call that share m share
        ///     the plan: it is built only when m differs from the previous plan's (<paramref name="planLength"/>). That
        ///     matters for many short slices with a float16 / float32 q, whose plan rounds several values per quantile and
        ///     costs as much as the partition of a few elements. A plan that throws is never recorded.
        /// </remarks>
        private static unsafe void ProcessQuantileRowNaN<T, TOut, TM, TQ>(
            T* scratch, int n, double* q, int nQs,
            TOut* dst, long dstOuterStride, QuantileMethod method, bool medianMean,
            int* rowKScratch, long* rowRaw, double* rowWeight, long* planLength, int* planCount)
            where T : unmanaged, IComparable<T>
            where TOut : unmanaged
            where TM : unmanaged
            where TQ : unmanaged
        {
            // Compact NaNs out (floats only — JIT folds the guard for int specializations).
            int m = n;
            if (typeof(T) == typeof(double))
            {
                var p = (double*)scratch; int w = 0;
                for (int i = 0; i < n; i++) { double v = p[i]; if (!double.IsNaN(v)) p[w++] = v; }
                m = w;
            }
            else if (typeof(T) == typeof(float))
            {
                var p = (float*)scratch; int w = 0;
                for (int i = 0; i < n; i++) { float v = p[i]; if (!float.IsNaN(v)) p[w++] = v; }
                m = w;
            }
            else if (typeof(T) == typeof(Half))
            {
                var p = (Half*)scratch; int w = 0;
                for (int i = 0; i < n; i++) { Half v = p[i]; if (!Half.IsNaN(v)) p[w++] = v; }
                m = w;
            }

            if (m == 0)
            {
                // `np.full(q.shape, np.nan, dtype=arr1d.dtype)`: Python's nan, the positive quiet NaN.
                for (int j = 0; j < nQs; j++) WriteNaNCell(dst + (long)j * dstOuterStride, PositiveQuietNaN);
                return;
            }

            int u;
            if (m == *planLength)
            {
                // The previous slice planned this length: its partition-ready indexes, weights and sorted partition list
                // are this slice's too.
                u = *planCount;
            }
            else
            {
                // The slice's plan (_quantile on the m-element slice, in q's precision; the slice is a float array, so a
                // NaN virtual index reads its last element), then partition_prep_kth_array's validation, in NumPy's list
                // order.
                BuildQuantilePlan<TQ>((int)method, m, q, nQs, 1, rowRaw, rowWeight);
                if (QuantileEngine.IsDiscrete(method))
                    QuantileIndexes.CheckTakeKth(rowRaw, nQs, 2, m);
                else
                    QuantileIndexes.CheckInterpolationKth(rowRaw, 2 * nQs, m);

                // Partition-ready indexes (NumPy's -1 is the last element), kept per quantile in rowRaw for the cells and
                // sorted + deduplicated into rowKScratch so PartitionAtMany places every needed index in one pass.
                int cnt = 0;
                for (int i = 0; i < 2 * nQs; i++)
                {
                    long k = rowRaw[i];
                    if (k < 0) k += m;
                    rowRaw[i] = k;
                    rowKScratch[cnt++] = (int)k;
                }
                new Span<int>(rowKScratch, cnt).Sort();
                u = 0;
                for (int i = 0; i < cnt; i++)
                    if (u == 0 || rowKScratch[u - 1] != rowKScratch[i]) rowKScratch[u++] = rowKScratch[i];

                // Recorded only now: a plan whose validation threw never reaches here.
                *planLength = m;
                *planCount = u;
            }

            if (m > 1)
                QuickSelect.PartitionAtMany(scratch, m, rowKScratch, u);

            for (int j = 0; j < nQs; j++)
                WriteCell<T, TOut, TM>(scratch, (int)rowRaw[2 * j], (int)rowRaw[2 * j + 1], rowWeight[2 * j],
                    rowWeight[2 * j + 1], method, medianMean, dst + (long)j * dstOuterStride);
        }

        // ── per-cell write (JIT-specialized) ────────────────────────────────────

        /// <summary>Python's <c>float('nan')</c> (NumPy's <c>np.nan</c>): the positive quiet NaN, <c>0x7FF8000000000000</c>.
        ///     .NET's <see cref="double.NaN"/> is the NEGATIVE one, which an out= would receive with the wrong sign.</summary>
        private static readonly double PositiveQuietNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

        /// <summary>
        ///     The cell of a slice holding NaN: NumPy's <c>np.copyto(result, arr[-1, ...], where=slices_having_nans)</c> — the
        ///     data's own NaN — or the nan forms' all-NaN slice (<c>np.nan</c>), converted to the buffer's dtype with its sign
        ///     and payload (a complex buffer receives <c>nan+0j</c>: a float NaN cast to complex).
        /// </summary>
        /// <typeparam name="TOut">The buffer's element type.</typeparam>
        /// <param name="dst">The cell.</param>
        /// <param name="nan">The NaN to write, as a double.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WriteNaNCell<TOut>(TOut* dst, double nan) where TOut : unmanaged
        {
            if (typeof(TOut) == typeof(double))       *(double*)dst  = nan;
            else if (typeof(TOut) == typeof(float))   *(float*)dst   = (float)nan;
            else if (typeof(TOut) == typeof(Half))    *(Half*)dst    = (Half)nan;
            else if (typeof(TOut) == typeof(Complex)) *(Complex*)dst = new Complex(nan, 0.0);
            else                                      *dst = default;   // ints / bool / char — NumPy stores 0
        }

        /// <summary>
        ///     One result cell: the selected sample (a selection method), or NumPy's <c>_lerp</c> of the two neighbours with
        ///     the weight t, in the lerp's precision TM, written as TOut.
        /// </summary>
        /// <typeparam name="T">The data's element type.</typeparam>
        /// <typeparam name="TOut">The buffer's element type: TM, or the caller's out dtype (float / complex) when it
        ///     receives the lerp (then the t &gt;= 0.5 branch is computed in it, as NumPy's
        ///     <c>subtract(..., dtype=type(out.dtype))</c> does).</typeparam>
        /// <typeparam name="TM">The lerp's precision: double, float or Half (each operation rounded to it), long for the
        ///     integer-weight lerp (whose neighbours are one element).</typeparam>
        /// <param name="scratch">The partitioned line.</param>
        /// <param name="prevIdx">The previous neighbour (the selected index for a selection method).</param>
        /// <param name="nextIdx">The next neighbour.</param>
        /// <param name="gamma">The weight t (a value of q's dtype; 0 for a selection method).</param>
        /// <param name="oneMinusGamma"><c>1 - t</c> in q's dtype (the plan's).</param>
        /// <param name="method">The method.</param>
        /// <param name="medianMean">
        ///     When true, a continuous even-size result (γ = 0.5, the two central samples) is the
        ///     <b>mean</b> <c>(a+b)/2</c> at the output precision — NumPy's <c>np.median</c> rule,
        ///     which differs from the quantile <c>_lerp</c> at q=0.5 by up to 1 ULP. γ = 0 (odd, a
        ///     single central sample) yields that sample, equal to NumPy's value. Set only by
        ///     <c>np.median</c>/<c>np.nanmedian</c>; the discrete methods never see it. Open gap: NumPy's
        ///     median is <c>np.mean</c> of the central sample(s), i.e. <c>(+0.0 + a [+ b]) / n</c>, so a
        ///     <c>-0.0</c> result there is <c>+0.0</c>; both branches here keep the <c>-0.0</c>
        ///     (pinned by <c>OpenBugsReductionTests.Median_OfNegativeZeros_IsPositiveZero</c>).
        /// </param>
        /// <param name="dst">The cell.</param>
        private static unsafe void WriteCell<T, TOut, TM>(
            T* scratch, int prevIdx, int nextIdx, double gamma, double oneMinusGamma,
            QuantileMethod method, bool medianMean, TOut* dst)
            where T : unmanaged
            where TOut : unmanaged
            where TM : unmanaged
        {
            // Discrete methods take a single sorted sample — preserves integer dtype on
            // integer input (TOut == T in the common case; cross-type cast goes via double).
            bool discrete =
                method == QuantileMethod.Lower ||
                method == QuantileMethod.Higher ||
                method == QuantileMethod.Nearest ||
                method == QuantileMethod.InvertedCdf ||
                method == QuantileMethod.ClosestObservation;

            if (discrete)
            {
                if (typeof(T) == typeof(TOut))
                {
                    *dst = ((TOut*)scratch)[prevIdx];   // same-dtype fast path
                    return;
                }
                WriteAsTOut(ToDouble(scratch[prevIdx]), dst);
                return;
            }

            // The integer-weight lerp (an integer q ARRAY under weibull / averaged_inverted_cdf /
            // interpolated_inverted_cdf over integer data): NumPy's lerp runs in result_type(data, q) — an integer dtype —
            // and its two neighbours are always the same element (q is 0 or 1, so the virtual index lies outside the line
            // and both indexes clamp to one end), so `a + (b - a) * t` is a exactly: written in that integer dtype (which
            // holds every value of the data's), without passing through a double.
            if (QuantileIndexes.IsIntegerPrecision<TM>())
            {
                WriteInt64AsTOut(ToInt64(scratch[prevIdx]), dst);
                return;
            }

            // Continuous (interpolating) methods (linear / midpoint / hazen / weibull / …).
            //
            // NumPy's _lerp is TWO-BRANCH, not the naive `a + (b-a)*t`
            // (numpy/lib/_function_base_impl.py::_lerp):
            //
            //     diff_b_a = b - a
            //     result   = a + diff_b_a * t            # for every element
            //     result   = b - diff_b_a * (1 - t)      # OVERWRITE where t >= 0.5
            //
            // The second branch is what keeps the interpolation monotone and correctly
            // rounded toward b; using only the first branch diverges from NumPy by up to
            // ~200 ULP on ~7 % of float64 inputs (the historical [known bug]). `gamma` is
            // the interpolation weight t in q's dtype and `oneMinusGamma` its `1 - t`, both
            // from the plan; the `t >= 0.5` test reads t itself (0.5 is exact in every float).
            //
            // The arithmetic PRECISION TM mirrors NumPy's ufunc promotion of
            // `_lerp(previous, next, gamma)` (all verified bit-exact against 2.4.2):
            //   * float32 + scalar/weak q  → float32 throughout (weak t adopts float32).
            //   * float16 + scalar/weak q  → float16 throughout; EVERY op rounds to f16
            //     and the weak t is rounded to f16 BEFORE the multiply (NumPy's f16 loops
            //     upcast to f32, compute, then round the result back to f16 per op).
            //   * a q ARRAY of dtype G → result_type(data, G): float64 for a float64 q;
            //     float32 / float16 where a float32 / float16 q leaves the data's
            //     precision enough room (int8 data with a float16 q lerps in float16).
            //     `diff_b_a = b - a` is computed in the INPUT dtype FIRST (rounding to
            //     f32/f16, wrapping for integers) and only then widened.
            //   * everything else (int / bool / char / float64 under a weak q) → float64.
            //
            // The JIT folds every `typeof(T)` / `typeof(TOut)` / `typeof(TM)` guard to a constant per
            // specialization, so each instantiation is one straight-line body.
            bool hi = gamma >= 0.5;    // NumPy's `where = t >= 0.5`

            if (typeof(T) == typeof(decimal))
            {
                // Decimal is NumSharp-only (no NumPy dtype). Its independent oracle
                // (test/oracle/gen_decimal_oracle.cs) is the sole source of truth and uses
                // the single-branch lerp, so decimal stays single-branch on purpose:
                // a two-branch change here has no parity benefit and would silently diverge
                // from that committed oracle. A weight that is not finite (a float16 q over a line of 65521 or more
                // elements) has no decimal value: the cell is 0, decimal's stand-in for NaN.
                decimal prev = ((decimal*)scratch)[prevIdx];
                decimal next = ((decimal*)scratch)[nextIdx];
                if (!medianMean && !double.IsFinite(gamma))
                {
                    WriteAsTOut(0.0, dst);
                    return;
                }
                decimal r = medianMean ? (hi ? (prev + next) / 2m : prev)     // np.median mean-of-middle
                                       : (gamma == 0) ? prev : prev + (next - prev) * (decimal)gamma;
                if (typeof(TOut) == typeof(decimal)) *(decimal*)dst = r;
                else WriteAsTOut((double)r, dst);
                return;
            }

            if (medianMean)
            {
                if (typeof(T) == typeof(float) && typeof(TOut) == typeof(float))
                {
                    // np.median: mean of the two central f32, in float32 (an overflowing sum stays inf, as NumPy's).
                    float prev = ((float*)scratch)[prevIdx];
                    float next = ((float*)scratch)[nextIdx];
                    *(float*)dst = hi ? (prev + next) / 2f : prev;
                    return;
                }
                if (typeof(T) == typeof(Half) && typeof(TOut) == typeof(Half))
                {
                    // np.median: mean of the two central f16. NumPy's f16 mean is
                    // sum-then-halve rounding to f16 twice — sum16 = (Half)(a+b), then /2 in f16.
                    float prev = (float)((Half*)scratch)[prevIdx];
                    float next = (float)((Half*)scratch)[nextIdx];
                    if (hi) { Half sum = (Half)(prev + next); *(Half*)dst = (Half)((float)sum / 2f); }
                    else    { *(Half*)dst = (Half)prev; }   // odd: the single central f16 sample
                    return;
                }
                LoadLerpOperands(scratch, prevIdx, nextIdx, out double mprev, out double mnext, out _);
                // np.median: mean of the two central values in float64
                WriteAsTOut(hi ? (mprev + mnext) / 2.0 : mprev, dst);
                return;
            }

            LoadLerpOperands(scratch, prevIdx, nextIdx, out double dprev, out double dnext, out double ddiff);

            // The operands in TM: a, b and diff are exact in TM (M is the promotion of D and q's dtype, which holds every
            // value of D — or float64, whose conversion of an int64 / uint64 is NumPy's own rounding), and t and 1 - t are
            // values of q's dtype, rounded to TM only under a weak q (a Python float meeting float16 / float32 data).
            double t = QuantileIndexes.Round<TM>(gamma);
            if (!hi)
            {
                // a + diff_b_a * t: two operations, each rounded to TM (computing one in double and rounding is exactly
                // TM's operation — see QuantileIndexes' header).
                double lo = QuantileIndexes.Round<TM>(dprev + QuantileIndexes.Round<TM>(ddiff * t));
                WriteLerpValue(lo, dst);
                return;
            }
            double product = QuantileIndexes.Round<TM>(ddiff * QuantileIndexes.Round<TM>(oneMinusGamma));
            if (typeof(TOut) == typeof(TM))
            {
                // b - diff_b_a * (1 - t), in TM.
                WriteAsTOut(QuantileIndexes.Round<TM>(dnext - product), dst);
                return;
            }
            // The caller's out of another dtype receives the lerp: NumPy's
            // `subtract(b, diff_b_a * (1 - t), out=out, dtype=type(out.dtype), casting='unsafe')` casts b (from the data's
            // dtype) and the TM product to out's dtype and subtracts there.
            WriteSubtractInOut(scratch[nextIdx], product, dst);
        }

        /// <summary>
        ///     The lerp's operands as doubles: the two neighbours <c>a</c> and <c>b</c>, and <c>diff_b_a = b - a</c> computed
        ///     in the DATA's dtype (NumPy subtracts before any promotion) and then widened.
        /// </summary>
        /// <typeparam name="T">The data's element type (not decimal: the cell handles it apart).</typeparam>
        /// <param name="scratch">The partitioned line.</param>
        /// <param name="prevIdx">The previous neighbour.</param>
        /// <param name="nextIdx">The next neighbour.</param>
        /// <param name="dprev">a, widened (int64 / uint64 round to the nearest double, as NumPy's cast to float64).</param>
        /// <param name="dnext">b, widened.</param>
        /// <param name="ddiff">b - a in the data's dtype (wrapped for integers, rounded for float16 / float32), widened.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void LoadLerpOperands<T>(T* scratch, int prevIdx, int nextIdx,
            out double dprev, out double dnext, out double ddiff) where T : unmanaged
        {
            // NumPy computes `diff_b_a = b - a` in the OPERAND'S OWN dtype and only then promotes
            // to the lerp's dtype for `a + diff*t`. For the narrow types that rounds/WRAPS the subtraction
            // before widening — a load-bearing divergence NumSharp must reproduce, not "fix":
            //   * float32 / float16 → the subtraction ROUNDS to f32/f16 first, then widens.
            //   * every integer width → the subtraction WRAPS (two's-complement / modular) at that
            //     width, so e.g. int16 `1 - (-32768)` overflows to -32767 and its percentile is a
            //     large POSITIVE value; computing the diff in double instead would give a
            //     sign-flipped answer (the historical W6-B "gross error"). NumPy raises a
            //     RuntimeWarning here but still returns the wrapped result, which is the contract.
            // The endpoints `a`/`b` themselves are just widened (no arithmetic), matching NumPy's
            // promotion of `previous`/`next`.
            if (typeof(T) == typeof(float))
            {
                float fp = ((float*)scratch)[prevIdx];
                float fn = ((float*)scratch)[nextIdx];
                ddiff = (double)(fn - fp);          // f32 subtraction, then widen
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(Half))
            {
                float fp = (float)((Half*)scratch)[prevIdx];
                float fn = (float)((Half*)scratch)[nextIdx];
                ddiff = (double)(Half)(fn - fp);    // f16 subtraction, then widen
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(long))
            {
                long fp = ((long*)scratch)[prevIdx], fn = ((long*)scratch)[nextIdx];
                ddiff = unchecked(fn - fp);         // int64 subtraction wraps, then widen
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(ulong))
            {
                ulong fp = ((ulong*)scratch)[prevIdx], fn = ((ulong*)scratch)[nextIdx];
                // uint64 subtraction wraps, then widens; every widening rounds ONCE, as NumPy's C cast does (.NET 8
                // converts from 2^63 on as signed and adds 2^64, rounding twice: 1 ulp off for large uint64 data).
                ddiff = Converts.ToDouble(unchecked(fn - fp));
                dprev = Converts.ToDouble(fp); dnext = Converts.ToDouble(fn);
            }
            else if (typeof(T) == typeof(int))
            {
                int fp = ((int*)scratch)[prevIdx], fn = ((int*)scratch)[nextIdx];
                ddiff = unchecked(fn - fp);         // int32 wraps
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(uint))
            {
                uint fp = ((uint*)scratch)[prevIdx], fn = ((uint*)scratch)[nextIdx];
                ddiff = unchecked(fn - fp);         // uint32 wraps
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(short))
            {
                short fp = ((short*)scratch)[prevIdx], fn = ((short*)scratch)[nextIdx];
                ddiff = (short)(fn - fp);           // int16 wraps
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(ushort))
            {
                ushort fp = ((ushort*)scratch)[prevIdx], fn = ((ushort*)scratch)[nextIdx];
                ddiff = (ushort)(fn - fp);          // uint16 wraps
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(sbyte))
            {
                sbyte fp = ((sbyte*)scratch)[prevIdx], fn = ((sbyte*)scratch)[nextIdx];
                ddiff = (sbyte)(fn - fp);           // int8 wraps
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(byte))
            {
                byte fp = ((byte*)scratch)[prevIdx], fn = ((byte*)scratch)[nextIdx];
                ddiff = (byte)(fn - fp);            // uint8 wraps
                dprev = fp; dnext = fn;
            }
            else if (typeof(T) == typeof(char))
            {
                char fp = ((char*)scratch)[prevIdx], fn = ((char*)scratch)[nextIdx];
                ddiff = (ushort)(fn - fp);          // Char is 16-bit — wrap like uint16
                dprev = fp; dnext = fn;
            }
            else
            {
                // double (and any future exact-widening type): subtract after widening.
                dprev = ToDouble(scratch[prevIdx]);
                dnext = ToDouble(scratch[nextIdx]);
                ddiff = dnext - dprev;
            }
        }

        /// <summary>
        ///     The <c>a + diff_b_a * t</c> branch's value (a value of the lerp's dtype) written as the buffer's dtype: NumPy's
        ///     <c>add(..., out=out)</c> casts its output into out (a complex out receives <c>value+0j</c>).
        /// </summary>
        /// <typeparam name="TOut">The buffer's element type.</typeparam>
        /// <param name="value">The lerp's value.</param>
        /// <param name="dst">The cell.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void WriteLerpValue<TOut>(double value, TOut* dst) where TOut : unmanaged
        {
            if (typeof(TOut) == typeof(Complex)) { *(Complex*)dst = new Complex(value, 0.0); return; }
            WriteAsTOut(value, dst);
        }

        /// <summary>
        ///     The <c>b - diff_b_a * (1 - t)</c> branch computed in out's dtype O: NumPy's
        ///     <c>subtract(b, product, out=out, dtype=type(out.dtype), casting='unsafe')</c> casts b from the data's dtype and
        ///     the product from the lerp's dtype to O, and subtracts in O (a complex O subtracts the real parts in double; its
        ///     imaginary part is <c>0 - 0</c>).
        /// </summary>
        /// <typeparam name="T">The data's element type.</typeparam>
        /// <typeparam name="TOut">Out's element type (double, float, Half or Complex).</typeparam>
        /// <param name="b">The next neighbour.</param>
        /// <param name="product"><c>diff_b_a * (1 - t)</c>, a value of the lerp's dtype.</param>
        /// <param name="dst">The cell.</param>
        private static unsafe void WriteSubtractInOut<T, TOut>(T b, double product, TOut* dst)
            where T : unmanaged where TOut : unmanaged
        {
            if (typeof(TOut) == typeof(Complex))
            {
                // complex128: re = (double)b - (double)product, im = 0.0 - 0.0 = +0.0.
                *(Complex*)dst = new Complex(ToDouble(b) - product, 0.0);
                return;
            }
            // NumPy's cast of b to O: an int64 / uint64 into float32 rounds once, straight from the integer (through a
            // double it could round twice); every other pair rounds correctly from the exact double.
            double bo;
            if (typeof(TOut) == typeof(float) && typeof(T) == typeof(long))
                bo = (float)Unsafe.As<T, long>(ref b);
            else if (typeof(TOut) == typeof(float) && typeof(T) == typeof(ulong))
                bo = Converts.ToSingle(Unsafe.As<T, ulong>(ref b));   // rounded once (.NET 8's conversion rounds twice)
            else
                bo = QuantileIndexes.Round<TOut>(ToDouble(b));
            WriteAsTOut(QuantileIndexes.Round<TOut>(bo - QuantileIndexes.Round<TOut>(product)), dst);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WriteAsTOut<TOut>(double v, TOut* dst) where TOut : unmanaged
        {
            if (typeof(TOut) == typeof(double))      { *(double*)dst  = v; return; }
            if (typeof(TOut) == typeof(float))       { *(float*)dst   = (float)v; return; }
            if (typeof(TOut) == typeof(Half))        { *(Half*)dst    = (Half)v; return; }
            if (typeof(TOut) == typeof(Complex))     { *(Complex*)dst = new Complex(v, 0.0); return; }
            if (typeof(TOut) == typeof(decimal))     { *(decimal*)dst = double.IsFinite(v) ? (decimal)v : 0m; return; }
            if (typeof(TOut) == typeof(long))        { *(long*)dst    = (long)v; return; }
            if (typeof(TOut) == typeof(ulong))       { *(ulong*)dst   = (ulong)v; return; }
            if (typeof(TOut) == typeof(int))         { *(int*)dst     = (int)v; return; }
            if (typeof(TOut) == typeof(uint))        { *(uint*)dst    = (uint)v; return; }
            if (typeof(TOut) == typeof(short))       { *(short*)dst   = (short)v; return; }
            if (typeof(TOut) == typeof(ushort))      { *(ushort*)dst  = (ushort)v; return; }
            if (typeof(TOut) == typeof(byte))        { *(byte*)dst    = (byte)v; return; }
            if (typeof(TOut) == typeof(sbyte))       { *(sbyte*)dst   = (sbyte)v; return; }
            if (typeof(TOut) == typeof(char))        { *(char*)dst    = (char)(ushort)v; return; }
            if (typeof(TOut) == typeof(bool))        { *(bool*)dst    = v != 0; return; }
            throw new NotSupportedException(typeof(TOut).Name);
        }

        /// <summary>
        ///     An integer written into an integer buffer, with C's cast (the integer-weight lerp's element, which the lerp's
        ///     integer dtype holds exactly).
        /// </summary>
        /// <typeparam name="TOut">The buffer's integer type.</typeparam>
        /// <param name="v">The value.</param>
        /// <param name="dst">The cell.</param>
        /// <exception cref="NotSupportedException">A non-integer buffer.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WriteInt64AsTOut<TOut>(long v, TOut* dst) where TOut : unmanaged
        {
            unchecked
            {
                if (typeof(TOut) == typeof(long))   { *(long*)dst   = v; return; }
                if (typeof(TOut) == typeof(ulong))  { *(ulong*)dst  = (ulong)v; return; }
                if (typeof(TOut) == typeof(int))    { *(int*)dst    = (int)v; return; }
                if (typeof(TOut) == typeof(uint))   { *(uint*)dst   = (uint)v; return; }
                if (typeof(TOut) == typeof(short))  { *(short*)dst  = (short)v; return; }
                if (typeof(TOut) == typeof(ushort)) { *(ushort*)dst = (ushort)v; return; }
                if (typeof(TOut) == typeof(sbyte))  { *(sbyte*)dst  = (sbyte)v; return; }
                if (typeof(TOut) == typeof(byte))   { *(byte*)dst   = (byte)v; return; }
            }
            throw new NotSupportedException(typeof(TOut).Name);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double ToDouble<T>(T value) where T : unmanaged
        {
            // Same JIT-folded chain pattern. The CLR ABI lets us re-interpret the
            // value via pointer-cast since T is unmanaged.
            unsafe
            {
                T* p = &value;
                if (typeof(T) == typeof(byte))    return *(byte*)p;
                if (typeof(T) == typeof(sbyte))   return *(sbyte*)p;
                if (typeof(T) == typeof(short))   return *(short*)p;
                if (typeof(T) == typeof(ushort))  return *(ushort*)p;
                if (typeof(T) == typeof(int))     return *(int*)p;
                if (typeof(T) == typeof(uint))    return *(uint*)p;
                if (typeof(T) == typeof(long))    return *(long*)p;
                // uint64 rounds ONCE, as NumPy's C cast does (.NET 8 converts from 2^63 on as signed and adds 2^64,
                // rounding twice: a 1-ulp difference in the lerp of large uint64 data).
                if (typeof(T) == typeof(ulong))   return Converts.ToDouble(*(ulong*)p);
                if (typeof(T) == typeof(char))    return *(char*)p;
                if (typeof(T) == typeof(double))  return *(double*)p;
                if (typeof(T) == typeof(float))   return *(float*)p;
                if (typeof(T) == typeof(Half))    return (double)*(Half*)p;
                if (typeof(T) == typeof(decimal)) return (double)*(decimal*)p;
                if (typeof(T) == typeof(bool))    return *(bool*)p ? 1.0 : 0.0;
            }
            throw new NotSupportedException(typeof(T).Name);
        }

        /// <summary>
        ///     An integer element as int64, exactly (NumPy's cast of the integer-weight lerp's operand to int64).
        /// </summary>
        /// <typeparam name="T">An integer element type (bool and uint64 never reach the int64 lerp).</typeparam>
        /// <param name="value">The element.</param>
        /// <returns>Its value.</returns>
        /// <exception cref="NotSupportedException">A non-integer type.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long ToInt64<T>(T value) where T : unmanaged
        {
            unsafe
            {
                T* p = &value;
                if (typeof(T) == typeof(byte))   return *(byte*)p;
                if (typeof(T) == typeof(sbyte))  return *(sbyte*)p;
                if (typeof(T) == typeof(short))  return *(short*)p;
                if (typeof(T) == typeof(ushort)) return *(ushort*)p;
                if (typeof(T) == typeof(int))    return *(int*)p;
                if (typeof(T) == typeof(uint))   return *(uint*)p;
                if (typeof(T) == typeof(long))   return *(long*)p;
                if (typeof(T) == typeof(char))   return *(char*)p;
            }
            throw new NotSupportedException(typeof(T).Name);
        }

        #endregion
    }
}
