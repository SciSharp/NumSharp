using System;
using System.Collections.Concurrent;
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
//   * Generates one `DynamicMethod` per `(srcDtype, outDtype, QuantileMethod)`
//     tuple and caches it. The first call for a given tuple pays the emit
//     cost; subsequent calls are a ConcurrentDictionary lookup + delegate
//     invoke.
//   * The emitted method body contains the OUTER row-loop in IL — there is
//     no managed `for` driving the rows.
//   * Per row, the kernel calls a generic helper `ProcessQuantileRow<T, TOut>`
//     which is JIT-specialized at call site. The `typeof(T) == typeof(X)`
//     chains inside the helper are JIT-folded to constant `true`/`false` per
//     specialization, so the dispatch happens once at first-call codegen,
//     not on every iteration.
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

        public unsafe delegate void QuantileKernel(
            void* srcBase, void* scratchBase,
            long outer, int n,
            int* kSorted, int nKs,
            double* q, int nQs,
            void* dstBase, long dstOuterStride,
            int ignoreNaN, int* rowKScratch);

        internal readonly struct QuantileKey : IEquatable<QuantileKey>
        {
            public readonly NPTypeCode SrcType;
            public readonly NPTypeCode OutType;
            public readonly QuantileMethod Method;
            public readonly bool IgnoreNaN;

            public QuantileKey(NPTypeCode srcType, NPTypeCode outType, QuantileMethod method, bool ignoreNaN)
            { SrcType = srcType; OutType = outType; Method = method; IgnoreNaN = ignoreNaN; }

            public bool Equals(QuantileKey o) => SrcType == o.SrcType && OutType == o.OutType && Method == o.Method && IgnoreNaN == o.IgnoreNaN;
            public override bool Equals(object obj) => obj is QuantileKey o && Equals(o);
            public override int GetHashCode() => ((int)SrcType << 17) | ((int)OutType << 9) | ((int)Method << 1) | (IgnoreNaN ? 1 : 0);
        }

        internal static readonly ConcurrentDictionary<QuantileKey, QuantileKernel> _quantileKernelCache = new();

        /// <summary>
        ///     Run the cached quantile kernel for the given dtype triple. First call for a
        ///     tuple emits and caches the DynamicMethod; later calls jump straight into the
        ///     specialized native code.
        /// </summary>
        public static unsafe void Quantile(
            NPTypeCode srcType, NPTypeCode outType, QuantileMethod method,
            void* srcBase, void* scratchBase, long outer, int n,
            int* kSorted, int nKs,
            double* q, int nQs,
            void* dstBase, long dstOuterStride,
            bool ignoreNaN = false, int* rowKScratch = null)
        {
            var key = new QuantileKey(srcType, outType, method, ignoreNaN);
            var kernel = _quantileKernelCache.GetOrAdd(key, k => EmitQuantileKernel(k));
            kernel(srcBase, scratchBase, outer, n, kSorted, nKs, q, nQs, dstBase, dstOuterStride,
                ignoreNaN ? 1 : 0, rowKScratch);
        }

        #endregion

        #region IL Emission

        private static QuantileKernel EmitQuantileKernel(QuantileKey key)
        {
            // Emits:
            //
            //   for (long i = 0; i < outer; i++) {
            //       ProcessQuantileRow<T, TOut>(
            //           (byte*)srcBase + i * n * sizeof(T),
            //           scratchBase,
            //           n, kSorted, nKs, q, nQs,
            //           (byte*)dstBase + i * sizeof(TOut),
            //           dstOuterStride,
            //           (int)method);
            //   }
            //
            // The cast to `T*` / `TOut*` happens inside the JIT-specialized helper.
            // The loop variable, address arithmetic, and outer comparison are all
            // pure IL — no managed for-loop drives the rows.

            int srcSize = GetTypeSize(key.SrcType);
            int outSize = GetTypeSize(key.OutType);
            Type srcClr = GetClrType(key.SrcType);
            Type outClr = GetClrType(key.OutType);

            var dm = new DynamicMethod(
                name: $"Quantile_{key.SrcType}_{key.OutType}_{key.Method}",
                returnType: typeof(void),
                parameterTypes: new[] {
                    typeof(void*),   // 0 srcBase
                    typeof(void*),   // 1 scratchBase
                    typeof(long),    // 2 outer
                    typeof(int),     // 3 n
                    typeof(int*),    // 4 kSorted
                    typeof(int),     // 5 nKs
                    typeof(double*), // 6 q
                    typeof(int),     // 7 nQs
                    typeof(void*),   // 8 dstBase
                    typeof(long),    // 9 dstOuterStride
                    typeof(int),     // 10 ignoreNaN
                    typeof(int*),    // 11 rowKScratch
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var helper = typeof(DirectILKernelGenerator)
                .GetMethod(nameof(ProcessQuantileRow), BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new MissingMethodException(nameof(ProcessQuantileRow));
            var helperSpecialized = helper.MakeGenericMethod(srcClr, outClr);

            var locI = il.DeclareLocal(typeof(long));
            var loopStart = il.DefineLabel();
            var loopCond = il.DefineLabel();

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

            //   scratchBase, n, kSorted, nKs, q, nQs
            il.Emit(OpCodes.Ldarg_1);                  // scratchBase
            il.Emit(OpCodes.Ldarg_3);                  // n
            il.Emit(OpCodes.Ldarg_S, (byte)4);         // kSorted
            il.Emit(OpCodes.Ldarg_S, (byte)5);         // nKs
            il.Emit(OpCodes.Ldarg_S, (byte)6);         // q
            il.Emit(OpCodes.Ldarg_S, (byte)7);         // nQs

            //   dstCell = (byte*)dstBase + i * sizeof(TOut)
            il.Emit(OpCodes.Ldarg_S, (byte)8);         // dstBase
            il.Emit(OpCodes.Ldloc, locI);              // i  (long)
            il.Emit(OpCodes.Ldc_I8, (long)outSize);
            il.Emit(OpCodes.Mul);                      // i * sizeof(TOut)
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);                      // dstBase + offset → void*

            //   dstOuterStride, methodInt, ignoreNaN (baked), rowKScratch
            il.Emit(OpCodes.Ldarg_S, (byte)9);              // dstOuterStride
            il.Emit(OpCodes.Ldc_I4, (int)key.Method);       // method baked in
            il.Emit(OpCodes.Ldc_I4, key.IgnoreNaN ? 1 : 0); // ignoreNaN baked in
            il.Emit(OpCodes.Ldarg_S, (byte)11);             // rowKScratch

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

        #region Generic per-row helper (JIT-specialized at call site)

        /// <summary>
        ///     Process one row: copy src → scratch, NaN-prescan (floats only), partition
        ///     at the requested k indices, then emit one output cell per quantile fraction.
        ///     The <c>typeof(T) == typeof(X)</c> guards are JIT-folded at specialization
        ///     time, so the body collapses to dtype-specific straight-line code on first
        ///     call. This is how compliance with the "no runtime dtype switch" rule is
        ///     achieved without hand-emitting the entire partition body in IL.
        /// </summary>
        internal static unsafe void ProcessQuantileRow<T, TOut>(
            void* srcRow, void* scratchBase, int n,
            int* kSorted, int nKs,
            double* q, int nQs,
            void* dstCell, long dstOuterStride,
            int methodInt, int ignoreNaN, int* rowKScratch)
            where T : unmanaged, IComparable<T>
            where TOut : unmanaged
        {
            T* src = (T*)srcRow;
            T* scratch = (T*)scratchBase;
            TOut* dst = (TOut*)dstCell;
            var method = (QuantileMethod)methodInt;

            // 1. memcpy
            Buffer.MemoryCopy(src, scratch, (long)n * sizeof(T), (long)n * sizeof(T));

            // ── NaN-ignoring path (np.nanmedian / np.nanquantile / np.nanpercentile) ──
            // The engine only routes float dtypes here (ints carry no NaN), so the
            // compaction guards below collapse to a single typed loop per
            // specialization. Each row is independently compacted (NaNs dropped to a
            // local valid-count m), then the partition/interpolate runs against the
            // first m elements with per-row indices — m varies per row, so kSorted
            // (sized for n) cannot be reused; indices are recomputed into rowKScratch.
            if (ignoreNaN != 0)
            {
                ProcessQuantileRowNaN<T, TOut>(scratch, n, q, nQs, dst, dstOuterStride, method, rowKScratch);
                return;
            }

            // 2. NaN prescan — only floats can carry NaN. JIT folds the guards away
            //    for non-float specializations.
            bool hasNaN = false;
            if (typeof(T) == typeof(double))
            {
                var p = (double*)scratch;
                for (int i = 0; i < n; i++) { if (double.IsNaN(p[i])) { hasNaN = true; break; } }
            }
            else if (typeof(T) == typeof(float))
            {
                var p = (float*)scratch;
                for (int i = 0; i < n; i++) { if (float.IsNaN(p[i])) { hasNaN = true; break; } }
            }
            else if (typeof(T) == typeof(Half))
            {
                var p = (Half*)scratch;
                for (int i = 0; i < n; i++) { if (Half.IsNaN(p[i])) { hasNaN = true; break; } }
            }

            // 3. Partition unless NaN-tainted (saves work — the answer will be NaN).
            //    n==1 is a no-op (already sorted).
            if (!hasNaN && n > 1)
                QuickSelect.PartitionAtMany(scratch, n, kSorted, nKs);

            // 4. Emit one cell per q
            for (int j = 0; j < nQs; j++)
            {
                TOut* outCell = dst + (long)j * dstOuterStride;
                if (hasNaN) { WriteNaNCell(outCell); continue; }

                ComputeIndex(n, q[j], method, out int prevIdx, out int nextIdx, out double gamma);
                WriteCell(scratch, prevIdx, nextIdx, gamma, method, outCell);
            }
        }

        /// <summary>
        ///     Per-row NaN-ignoring quantile. <paramref name="scratch"/> already holds a
        ///     copy of the row (length n). NaNs are compacted to the front (valid count
        ///     m); an all-NaN row writes NaN for every q (matches NumPy's "All-NaN slice").
        ///     Partition indices are derived from m per row and sorted/deduped into
        ///     <paramref name="rowKScratch"/> (capacity >= 2*nQs).
        /// </summary>
        private static unsafe void ProcessQuantileRowNaN<T, TOut>(
            T* scratch, int n, double* q, int nQs,
            TOut* dst, long dstOuterStride, QuantileMethod method, int* rowKScratch)
            where T : unmanaged, IComparable<T>
            where TOut : unmanaged
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
                for (int j = 0; j < nQs; j++) WriteNaNCell(dst + (long)j * dstOuterStride);
                return;
            }

            // Collect (prev,next) indices for every q against this row's valid count m.
            int cnt = 0;
            for (int j = 0; j < nQs; j++)
            {
                ComputeIndex(m, q[j], method, out int pi, out int ni, out _);
                rowKScratch[cnt++] = pi;
                rowKScratch[cnt++] = ni;
            }
            // Sort + dedup so PartitionAtMany places every needed index in one pass.
            new Span<int>(rowKScratch, cnt).Sort();
            int u = 0;
            for (int i = 0; i < cnt; i++)
                if (u == 0 || rowKScratch[u - 1] != rowKScratch[i]) rowKScratch[u++] = rowKScratch[i];

            if (m > 1)
                QuickSelect.PartitionAtMany(scratch, m, rowKScratch, u);

            for (int j = 0; j < nQs; j++)
            {
                ComputeIndex(m, q[j], method, out int prevIdx, out int nextIdx, out double gamma);
                WriteCell(scratch, prevIdx, nextIdx, gamma, method, dst + (long)j * dstOuterStride);
            }
        }

        // ── per-method index/gamma ──────────────────────────────────────────────

        /// <summary>
        ///     Computes (previous index, next index, lerp weight γ) for one quantile q in
        ///     <c>[0,1]</c> against a row of length n, per the selected method's
        ///     virtual-index formula (see numpy._function_base_impl._QuantileMethods).
        /// </summary>
        internal static void ComputeIndex(int n, double q, QuantileMethod method,
            out int prevIdx, out int nextIdx, out double gamma)
        {
            double vi;
            switch (method)
            {
                case QuantileMethod.Linear:
                    vi = (n - 1) * q;
                    prevIdx = (int)Math.Floor(vi);
                    nextIdx = prevIdx + 1;
                    gamma = vi - prevIdx;
                    break;
                case QuantileMethod.Lower:
                    prevIdx = (int)Math.Floor((n - 1) * q);
                    nextIdx = prevIdx;
                    gamma = 0;
                    break;
                case QuantileMethod.Higher:
                    prevIdx = (int)Math.Ceiling((n - 1) * q);
                    nextIdx = prevIdx;
                    gamma = 0;
                    break;
                case QuantileMethod.Nearest:
                    prevIdx = (int)Math.Round((n - 1) * q, MidpointRounding.ToEven);
                    nextIdx = prevIdx;
                    gamma = 0;
                    break;
                case QuantileMethod.Midpoint:
                    vi = (n - 1) * q;
                    {
                        double lo = Math.Floor(vi);
                        prevIdx = (int)lo;
                        nextIdx = (int)Math.Ceiling(vi);
                        gamma = (vi == lo) ? 0.0 : 0.5;
                    }
                    break;
                case QuantileMethod.InvertedCdf:
                {
                    double v = n * q - 1.0;
                    double fl = Math.Floor(v);
                    vi = (v - fl == 0) ? fl : fl + 1.0;
                    if (vi < 0) vi = 0;
                    prevIdx = (int)vi;
                    nextIdx = prevIdx;
                    gamma = 0;
                    break;
                }
                case QuantileMethod.ClosestObservation:
                {
                    double idx = n * q - 1.0 - 0.5;
                    double fl = Math.Floor(idx);
                    double frac = idx - fl;
                    if (frac == 0 && ((long)fl % 2 + 2) % 2 == 1) vi = fl; else vi = fl + 1.0;
                    if (vi < 0) vi = 0;
                    prevIdx = (int)vi;
                    nextIdx = prevIdx;
                    gamma = 0;
                    break;
                }
                case QuantileMethod.AveragedInvertedCdf:
                    vi = n * q - 1.0;
                    prevIdx = (int)Math.Floor(vi);
                    nextIdx = prevIdx + 1;
                    {
                        double g = vi - prevIdx;
                        gamma = (g == 0) ? 0.5 : 1.0;
                    }
                    break;
                case QuantileMethod.InterpolatedInvertedCdf:
                    vi = AB(n, q, 0.0, 1.0);
                    prevIdx = (int)Math.Floor(vi);
                    nextIdx = prevIdx + 1;
                    gamma = vi - prevIdx;
                    break;
                case QuantileMethod.Hazen:
                    vi = AB(n, q, 0.5, 0.5);
                    prevIdx = (int)Math.Floor(vi);
                    nextIdx = prevIdx + 1;
                    gamma = vi - prevIdx;
                    break;
                case QuantileMethod.Weibull:
                    vi = AB(n, q, 0.0, 0.0);
                    prevIdx = (int)Math.Floor(vi);
                    nextIdx = prevIdx + 1;
                    gamma = vi - prevIdx;
                    break;
                case QuantileMethod.MedianUnbiased:
                    vi = AB(n, q, 1.0 / 3.0, 1.0 / 3.0);
                    prevIdx = (int)Math.Floor(vi);
                    nextIdx = prevIdx + 1;
                    gamma = vi - prevIdx;
                    break;
                case QuantileMethod.NormalUnbiased:
                    vi = AB(n, q, 3.0 / 8.0, 3.0 / 8.0);
                    prevIdx = (int)Math.Floor(vi);
                    nextIdx = prevIdx + 1;
                    gamma = vi - prevIdx;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(method));
            }
            // Clamp into [0, n-1] (matches numpy._get_indexes).
            if (prevIdx < 0) prevIdx = 0;
            if (prevIdx > n - 1) prevIdx = n - 1;
            if (nextIdx < 0) nextIdx = 0;
            if (nextIdx > n - 1) nextIdx = n - 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double AB(int n, double q, double alpha, double beta) =>
            n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0;

        // ── per-cell write (JIT-specialized) ────────────────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WriteNaNCell<TOut>(TOut* dst) where TOut : unmanaged
        {
            if (typeof(TOut) == typeof(double))     *(double*)dst  = double.NaN;
            else if (typeof(TOut) == typeof(float)) *(float*)dst   = float.NaN;
            else if (typeof(TOut) == typeof(Half))  *(Half*)dst    = Half.NaN;
            else                                    *dst = default;   // ints / bool / char — NumPy stores 0
        }

        private static unsafe void WriteCell<T, TOut>(
            T* scratch, int prevIdx, int nextIdx, double gamma,
            QuantileMethod method, TOut* dst)
            where T : unmanaged
            where TOut : unmanaged
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

            // Continuous methods (linear / midpoint / hazen / weibull / etc.).
            //
            // We pick the arithmetic-precision type by input T:
            //   float  → float
            //   decimal → decimal
            //   anything else (int families, double, bool, char, Half lifted to float) → double
            //
            // The JIT collapses the typeof guards to constant true/false per
            // (T, TOut) specialization, so each instantiation is a single
            // arithmetic body with no dispatch.

            if (typeof(T) == typeof(float) && typeof(TOut) == typeof(float))
            {
                // Weak (scalar-q) float32 → float32: NumPy keeps the whole lerp in float32.
                float prev = ((float*)scratch)[prevIdx];
                float next = ((float*)scratch)[nextIdx];
                *(float*)dst = prev + (next - prev) * (float)gamma;
                return;
            }
            if (typeof(T) == typeof(decimal))
            {
                decimal prev = ((decimal*)scratch)[prevIdx];
                decimal next = ((decimal*)scratch)[nextIdx];
                decimal r = (gamma == 0) ? prev : prev + (next - prev) * (decimal)gamma;
                if (typeof(TOut) == typeof(decimal)) *(decimal*)dst = r;
                else WriteAsTOut((double)r, dst);
                return;
            }
            if (typeof(T) == typeof(Half) && typeof(TOut) == typeof(Half))
            {
                // Weak (scalar-q) float16 → float16.
                float prev = (float)((Half*)scratch)[prevIdx];
                float next = (float)((Half*)scratch)[nextIdx];
                *(Half*)dst = (Half)(prev + (next - prev) * (float)gamma);
                return;
            }

            // Default: lerp in double. Covers int / bool / char / double inputs and the
            // strong-q (array-q) float16/float32 → float64 promotion, where NumPy widens
            // to float64 before interpolating.
            double dprev = ToDouble(scratch[prevIdx]);
            double dnext = ToDouble(scratch[nextIdx]);
            double dr = dprev + (dnext - dprev) * gamma;
            WriteAsTOut(dr, dst);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WriteAsTOut<TOut>(double v, TOut* dst) where TOut : unmanaged
        {
            if (typeof(TOut) == typeof(double))      { *(double*)dst  = v; return; }
            if (typeof(TOut) == typeof(float))       { *(float*)dst   = (float)v; return; }
            if (typeof(TOut) == typeof(Half))        { *(Half*)dst    = (Half)v; return; }
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
                if (typeof(T) == typeof(ulong))   return *(ulong*)p;
                if (typeof(T) == typeof(char))    return *(char*)p;
                if (typeof(T) == typeof(double))  return *(double*)p;
                if (typeof(T) == typeof(float))   return *(float*)p;
                if (typeof(T) == typeof(Half))    return (double)*(Half*)p;
                if (typeof(T) == typeof(decimal)) return (double)*(decimal*)p;
                if (typeof(T) == typeof(bool))    return *(bool*)p ? 1.0 : 0.0;
            }
            throw new NotSupportedException(typeof(T).Name);
        }

        #endregion
    }
}
