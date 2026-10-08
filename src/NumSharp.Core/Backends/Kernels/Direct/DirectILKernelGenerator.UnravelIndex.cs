using System;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.UnravelIndex.cs — IL kernel for np.unravel_index
// =============================================================================
//
// RESPONSIBILITY:
//   np.unravel_index converts an arbitrary array of flat indices into a tuple
//   of per-axis coordinate arrays. The per-element work is `ndim` divmods,
//   which is divmod-bound regardless of layout. The kernel is dtype-agnostic
//   (input is cast to int64 by the caller) — a single DynamicMethod handles
//   any ndim + both C / F order.
//
//   The coordinates go into ONE interleaved buffer, as NumPy's arr_unravel_index
//   writes them: element i's coordinates are the row out[i*ndim .. i*ndim+ndim-1],
//   and the caller hands out each coordinate as a strided column VIEW of that
//   buffer (stride ndim, NumPy's `ret_arr[..., j]`). One row per element also keeps
//   every write of an element inside one cache line pair, where the per-axis
//   column buffers this kernel used to fill took ndim write streams.
//
//   Structural twin of <see cref="NonZeroPerDimKernel"/>, but without the
//   monotonic-index optimization: the input indices are arbitrary, so we
//   cannot use the carry-chain incremental advance — every element pays
//   `ndim` divmods.
//
// KERNEL (DynamicMethod-emitted, singleton):
//
//   * UnravelIndexKernel
//       (long* indices,        // contig int64 buffer (caller casts)
//        long count,
//        long* dims,           // shape, ndim entries
//        long unravelSize,     // product of dims, for OOB validation
//        long* out,            // count*ndim interleaved buffer: row i = element i's coordinates
//        long ndim,
//        long idxStart,        // ndim-1 for C-order, 0 for F-order
//        long idxStep)         // -1 for C-order, +1 for F-order
//        -> long: count on success, else the index of the first OOB element.
//
//   Caller reads indices[returned_index] to produce NumPy's diagnostic
//   "index N is out of bounds for array with size M" message.
//
// OPTIMIZATION: SKIP LAST DIVMOD
//   After ndim-1 divmods, `val < dims[final_idx]`, so coords[final_idx] = val
//   directly. Saves 1/ndim of the divmod cost — a 50% reduction for ndim==2,
//   33% for ndim==3, …
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted flat→multi-coord expander for <c>np.unravel_index</c>.
    /// Caller pre-casts indices to int64 and computes <paramref name="unravelSize"/>
    /// = product of <paramref name="dims"/>; the kernel reads <paramref name="indices"/>
    /// linearly, validates each value against <c>[0, unravelSize)</c>, and writes
    /// element <c>i</c>'s coordinates into the row <c>out[i*ndim .. i*ndim+ndim-1]</c>
    /// of the interleaved buffer <paramref name="out"/>, using the C-order or F-order
    /// extraction direction selected by <paramref name="idxStart"/> /
    /// <paramref name="idxStep"/>.
    /// </summary>
    /// <param name="indices">The int64 flat indices, read linearly (the caller lays them out in NumPy's iteration
    ///     order, so row <c>i</c> belongs to the <c>i</c>-th element NumPy visits).</param>
    /// <param name="count">The number of indices.</param>
    /// <param name="dims">The unravel shape, <paramref name="ndim"/> entries.</param>
    /// <param name="unravelSize">The product of <paramref name="dims"/>; an index outside <c>[0, unravelSize)</c> stops
    ///     the kernel.</param>
    /// <param name="out">The <c>count * ndim</c> interleaved output; rows past a failing element are not written.</param>
    /// <param name="ndim">The number of dimensions, at least 1 (the caller answers <c>shape=()</c> itself).</param>
    /// <param name="idxStart">The first dimension a divmod peels: <c>ndim - 1</c> for C order, 0 for F order.</param>
    /// <param name="idxStep"><c>-1</c> for C order, <c>+1</c> for F order.</param>
    /// <returns>
    /// <paramref name="count"/> on success. On failure, the row index of the first
    /// element with <c>val &lt; 0</c> or <c>val &gt;= unravelSize</c>; the caller
    /// reads <c>indices[returned]</c> to produce the error message.
    /// </returns>
    public unsafe delegate long UnravelIndexKernel(
        long* indices, long count, long* dims, long unravelSize,
        long* @out, long ndim, long idxStart, long idxStep);

    /// <summary>
    /// The reciprocal twin of <see cref="UnravelIndexKernel"/>: the same contract, but each peeled dimension is divided by
    /// a precomputed invariant reciprocal (Granlund &amp; Montgomery, "Division by Invariant Integers using
    /// Multiplication", Fig. 4.1) instead of the hardware divide. Exact for every non-negative index and every POSITIVE
    /// dimension; a negative dimension (C's truncating division of a negative running quotient) must take
    /// <see cref="UnravelIndexKernel"/>.
    /// </summary>
    /// <param name="indices">The int64 flat indices, read linearly in NumPy's iteration order.</param>
    /// <param name="count">The number of indices.</param>
    /// <param name="divisors">Four int64 per dimension, filled by
    ///     <see cref="DirectILKernelGenerator.FillUnravelDivisor"/>: the dimension, the magic multiplier, and the two shift
    ///     counts.</param>
    /// <param name="unravelSize">The product of the dimensions; an index outside <c>[0, unravelSize)</c> stops the
    ///     kernel.</param>
    /// <param name="out">The <c>count * ndim</c> interleaved output (row i = element i's coordinates).</param>
    /// <param name="ndim">The number of dimensions, at least 1.</param>
    /// <param name="idxStart"><c>ndim - 1</c> for C order, 0 for F order.</param>
    /// <param name="idxStep"><c>-1</c> for C order, <c>+1</c> for F order.</param>
    /// <returns><paramref name="count"/> on success, else the row index of the first out-of-range element.</returns>
    public unsafe delegate long UnravelIndexMagicKernel(
        long* indices, long count, long* divisors, long unravelSize,
        long* @out, long ndim, long idxStart, long idxStep);

    public static partial class DirectILKernelGenerator
    {
        private static UnravelIndexKernel _unravelIndexKernel;

        /// <summary>
        /// IL-emitted unravel kernel (singleton — same kernel handles any ndim and
        /// both C / F order via the runtime <c>idxStart</c> / <c>idxStep</c> args).
        /// Returns <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static UnravelIndexKernel GetUnravelIndexKernel()
        {
            if (!Enabled)
                return null;

            var cached = _unravelIndexKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateUnravelIndexKernelIL();
                Interlocked.CompareExchange(ref _unravelIndexKernel, k, null);
                return _unravelIndexKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetUnravelIndexKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the unravel kernel. Pseudocode:
        /// <code>
        /// long Unravel(long* indices, long count, long* dims, long unravelSize,
        ///              long* out, long ndim, long idxStart, long idxStep) {
        ///     long ndimMinusOne = ndim - 1;
        ///     long rowBytes = ndim * 8;
        ///     for (long i = 0; i &lt; count; i++) {
        ///         long val = indices[i];
        ///         if (val &lt; 0 || val &gt;= unravelSize) return i;
        ///         long* row = out + i * ndim;          // element i's interleaved coordinates
        ///         long idx = idxStart;
        ///         for (long k = 0; k &lt; ndimMinusOne; k++) {
        ///             long m = dims[idx];
        ///             row[idx] = val % m;
        ///             val = val / m;
        ///             idx += idxStep;
        ///         }
        ///         // Last coord: val &lt; dims[idx] already, no divmod needed.
        ///         row[idx] = val;
        ///     }
        ///     return count;
        /// }
        /// </code>
        /// </summary>
        private static UnravelIndexKernel GenerateUnravelIndexKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_UnravelIndex",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(long*),  // 0 indices
                    typeof(long),   // 1 count
                    typeof(long*),  // 2 dims
                    typeof(long),   // 3 unravelSize
                    typeof(long*),  // 4 out (interleaved rows)
                    typeof(long),   // 5 ndim
                    typeof(long),   // 6 idxStart
                    typeof(long),   // 7 idxStep
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locVal = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locK = il.DeclareLocal(typeof(long));
            var locM = il.DeclareLocal(typeof(long));
            var locRow = il.DeclareLocal(typeof(long*));
            var locNdimMinusOne = il.DeclareLocal(typeof(long));
            var locRowBytes = il.DeclareLocal(typeof(long));

            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblInnerHead = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();

            // ndimMinusOne = ndim - 1 (hoisted)
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locNdimMinusOne);

            // rowBytes = ndim * 8 (hoisted): the byte distance between consecutive elements' rows
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locRowBytes);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ----- Outer loop: for (i = 0; i < count; i++) -----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);   // count
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // val = indices[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locVal);

            // if (val < 0) goto fail
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblFail);

            // if (val >= unravelSize) goto fail
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Bge, lblFail);

            // row = out + i * rowBytes — element i's interleaved coordinates
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locRowBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locRow);

            // idx = idxStart
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Stloc, locIdx);

            // k = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locK);

            // ----- Inner loop: for (k = 0; k < ndim - 1; k++) -----
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locNdimMinusOne);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // m = dims[idx]
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locM);

            // row[idx] = val % m
            il.Emit(OpCodes.Ldloc, locRow);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stind_I8);

            // val = val / m
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locVal);

            // idx += idxStep
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdx);

            // k++
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // ----- Last coord: row[idx] = val (skip divmod) -----
            il.Emit(OpCodes.Ldloc, locRow);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Stind_I8);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuterHead);

            // ----- Fail path: return i -----
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ret);

            // ----- Success: return count -----
            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ret);

            return (UnravelIndexKernel)dm.CreateDelegate(typeof(UnravelIndexKernel));
        }

        private static UnravelIndexMagicKernel _unravelIndexMagicKernel;

        /// <summary>
        /// IL-emitted reciprocal unravel kernel (singleton — any ndim, both orders). Returns <c>null</c> only when
        /// <see cref="Enabled"/> is false or the emission fails (the caller then has no kernel at all, as with
        /// <see cref="GetUnravelIndexKernel"/>).
        /// </summary>
        /// <remarks>
        /// Why it exists: the hardware 64-bit divide costs ~15-40 cycles and RyuJIT emits one for the remainder and
        /// another for the quotient of the same operands, so the divmod kernel ran two divides per peeled dimension —
        /// NumPy's own speed (its C compiler fuses them into one, but it peels every dimension). A multiply-high, a
        /// subtract, two shifts and a multiply-subtract do the same work in a few cycles, independent elements
        /// overlapping in the pipeline.
        /// </remarks>
        public static UnravelIndexMagicKernel GetUnravelIndexMagicKernel()
        {
            if (!Enabled)
                return null;

            var cached = _unravelIndexMagicKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateUnravelIndexMagicKernelIL();
                Interlocked.CompareExchange(ref _unravelIndexMagicKernel, k, null);
                return _unravelIndexMagicKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetUnravelIndexMagicKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Fills one dimension's entry of the <see cref="UnravelIndexMagicKernel"/> divisor table: Granlund &amp;
        /// Montgomery's round-up reciprocal for N = 64-bit unsigned division by <paramref name="d"/>.
        /// </summary>
        /// <param name="d">The dimension: positive (a zero dimension never divides — its product refuses every index —
        ///     and a negative one takes the divmod kernel).</param>
        /// <param name="entry">Four int64: <c>[d, m', sh1, sh2]</c> with <c>l = ceil(log2 d)</c>,
        ///     <c>m' = floor(2^64 (2^l - d) / d) + 1</c>, <c>sh1 = min(l, 1)</c>, <c>sh2 = max(l - 1, 0)</c>; the quotient
        ///     of any <c>n &lt; 2^64</c> is then <c>(t + ((n - t) &gt;&gt; sh1)) &gt;&gt; sh2</c> with
        ///     <c>t = mulhi(m', n)</c> — <c>d = 1</c> included (<c>m' = 1</c>, both shifts 0, the quotient is n).</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="d"/> is not positive.</exception>
        internal static void FillUnravelDivisor(long d, Span<long> entry)
        {
            if (d <= 0)
                throw new ArgumentOutOfRangeException(nameof(d), d, "the reciprocal kernel divides by positive dimensions only");
            // l = ceil(log2 d): 0 for d = 1, else the bit width of d - 1.
            int l = d == 1 ? 0 : 64 - BitOperations.LeadingZeroCount((ulong)(d - 1));
            // 2^l - d fits 64 bits (d <= 2^l <= 2^63 for any positive long), and the 128-bit numerator keeps the floor exact.
            ulong twoL = l == 64 ? 0UL : 1UL << l;
            UInt128 numerator = (UInt128)(twoL - (ulong)d) << 64;
            ulong magic = (ulong)(numerator / (ulong)d) + 1UL;
            entry[0] = d;
            entry[1] = unchecked((long)magic);
            entry[2] = Math.Min(l, 1);
            entry[3] = Math.Max(l - 1, 0);
        }

        /// <summary>
        /// Emits the reciprocal unravel kernel. Pseudocode (unsigned arithmetic on the running quotient, which never goes
        /// negative: the index is checked non-negative and every dimension is positive):
        /// <code>
        /// long Unravel(long* indices, long count, long* divs, long unravelSize,
        ///              long* out, long ndim, long idxStart, long idxStep) {
        ///     long ndimMinusOne = ndim - 1;
        ///     long rowBytes = ndim * 8;
        ///     for (long i = 0; i &lt; count; i++) {
        ///         long val = indices[i];
        ///         if (val &lt; 0 || val &gt;= unravelSize) return i;
        ///         long* row = out + i * ndim;
        ///         long idx = idxStart;
        ///         for (long k = 0; k &lt; ndimMinusOne; k++) {
        ///             long* e = divs + idx * 4;                    // [d, m', sh1, sh2]
        ///             ulong t = Math.BigMul((ulong)val, (ulong)e[1], out _);
        ///             ulong q = (t + (((ulong)val - t) &gt;&gt; (int)e[2])) &gt;&gt; (int)e[3];
        ///             row[idx] = val - (long)q * e[0];
        ///             val = (long)q;
        ///             idx += idxStep;
        ///         }
        ///         row[idx] = val;                                 // val &lt; dims[idx] already
        ///     }
        ///     return count;
        /// }
        /// </code>
        /// </summary>
        /// <returns>The compiled delegate.</returns>
        private static UnravelIndexMagicKernel GenerateUnravelIndexMagicKernelIL()
        {
            var bigMul = typeof(Math).GetMethod(nameof(Math.BigMul),
                BindingFlags.Public | BindingFlags.Static, null,
                new[] { typeof(ulong), typeof(ulong), typeof(ulong).MakeByRefType() }, null)
                ?? throw new MissingMethodException(nameof(Math), "BigMul(ulong, ulong, out ulong)");

            var dm = new DynamicMethod(
                name: "IL_UnravelIndexMagic",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(long*),  // 0 indices
                    typeof(long),   // 1 count
                    typeof(long*),  // 2 divisors ([d, m', sh1, sh2] per dimension)
                    typeof(long),   // 3 unravelSize
                    typeof(long*),  // 4 out (interleaved rows)
                    typeof(long),   // 5 ndim
                    typeof(long),   // 6 idxStart
                    typeof(long),   // 7 idxStep
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locVal = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locK = il.DeclareLocal(typeof(long));
            var locRow = il.DeclareLocal(typeof(long*));
            var locEntry = il.DeclareLocal(typeof(long*));
            var locT = il.DeclareLocal(typeof(ulong));
            var locQ = il.DeclareLocal(typeof(ulong));
            var locLow = il.DeclareLocal(typeof(ulong));
            var locNdimMinusOne = il.DeclareLocal(typeof(long));
            var locRowBytes = il.DeclareLocal(typeof(long));

            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblInnerHead = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();

            // ndimMinusOne = ndim - 1; rowBytes = ndim * 8 (hoisted)
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locNdimMinusOne);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locRowBytes);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ----- Outer loop over indices -----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // val = indices[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locVal);

            // if (val < 0 || val >= unravelSize) goto fail
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblFail);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Bge, lblFail);

            // row = out + i * rowBytes
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locRowBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locRow);

            // idx = idxStart; k = 0
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Stloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locK);

            // ----- Inner loop: peel every dimension but the last -----
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locNdimMinusOne);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // entry = divs + idx * 32
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 32L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locEntry);

            // t = Math.BigMul((ulong)val, (ulong)entry[1], out low)
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldloc, locEntry);
            il.Emit(OpCodes.Ldc_I4, 8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldloca, locLow);
            il.Emit(OpCodes.Call, bigMul);
            il.Emit(OpCodes.Stloc, locT);

            // q = (t + ((val - t) >> sh1)) >> sh2   (unsigned shifts)
            il.Emit(OpCodes.Ldloc, locT);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldloc, locT);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldloc, locEntry);
            il.Emit(OpCodes.Ldc_I4, 16);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Conv_I4);
            il.Emit(OpCodes.Shr_Un);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locEntry);
            il.Emit(OpCodes.Ldc_I4, 24);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Conv_I4);
            il.Emit(OpCodes.Shr_Un);
            il.Emit(OpCodes.Stloc, locQ);

            // row[idx] = val - q * d
            il.Emit(OpCodes.Ldloc, locRow);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldloc, locQ);
            il.Emit(OpCodes.Ldloc, locEntry);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stind_I8);

            // val = q
            il.Emit(OpCodes.Ldloc, locQ);
            il.Emit(OpCodes.Stloc, locVal);

            // idx += idxStep; k++
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdx);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // ----- Last coord: row[idx] = val (already below its dimension) -----
            il.Emit(OpCodes.Ldloc, locRow);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Stind_I8);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuterHead);

            // ----- Fail path: return i -----
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ret);

            // ----- Success: return count -----
            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ret);

            return (UnravelIndexMagicKernel)dm.CreateDelegate(typeof(UnravelIndexMagicKernel));
        }
    }
}
