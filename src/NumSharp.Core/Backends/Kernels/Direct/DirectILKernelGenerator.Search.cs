using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

namespace NumSharp.Backends.Kernels
{
    // =============================================================================
    // DirectILKernelGenerator.Search.cs
    //   OWNERSHIP: searchsorted binary-search kernels, NumPy 2.5's batched bisection.
    //   RESPONSIBILITY:
    //     - Typed inner loop per (NPTypeCode, side, has_sorter), no boxing / no double conversion.
    //     - Batched bisection (NumPy 2.5, gh-30517): every key shares one interval length, halved once
    //       per pass, and each key's base lives in the result array between passes. The first pass
    //       compares every key with the same pivot (the median), the next with the quartiles, so
    //       pivots stay in cache and the keys' comparisons are independent (out-of-order friendly).
    //       It replaced 2.4's per-key bisection with the monotonic carry of the previous key's bound.
    //     - Caller must materialize v (keyPtr) to contiguous before calling; sorter must be int64
    //       and contiguous. a (arrPtr) may be strided via arrStrideBytes.
    //   PARITY WITH NUMPY:
    //     - Mirrors numpy/_core/src/npysort/binsearch.cpp template binsearch<Tag, side>, and
    //       argbinsearch<Tag, side> for a sorter: every sorter entry the bisection reaches is checked
    //       against [0, arrLen) before it addresses a (argbinsearch returns -1, which
    //       PyArray_SearchSorted raises as "Sorter index out of range."); an entry it never reaches is
    //       never read, so it is never checked. The middle entry sorter[arrLen >> 1] is reached
    //       before any key is read, so it is checked even when there is no key.
    //     - side='left'  cmp = (val <  target)  -> bisect_left  (returns first i with a[i] >= target)
    //     - side='right' cmp = (val <= target)  -> bisect_right (returns first i with a[i] >  target)
    //     - A sorted haystack has one answer per key, which both algorithms find. An UNSORTED one
    //       (np.unique's NaN probe over a masked array's raw data, a user's mistake) is answered by
    //       whichever pivots the algorithm visits, so the batched schedule is what makes the answer
    //       NumPy 2.5's. NumPy's generic search (text, object) keeps the 2.4 algorithm:
    //       Sorting.TextSearch / Sorting.VStringSearch.
    //     - Decimal (no NumPy dtype) and Char (NumPy's uint16 in the oracle) run the same numeric
    //       schedule as every other fixed-size dtype.
    // =============================================================================
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// SearchSorted kernel delegate.
        /// </summary>
        /// <param name="arrPtr">Pointer to a's data, already at a.Shape.offset.</param>
        /// <param name="arrLen">Number of elements in a.</param>
        /// <param name="arrStrideBytes">a's stride in bytes (= elemSize when contiguous).</param>
        /// <param name="keyPtr">Pointer to v's data; v MUST be contiguous (elemSize stride implicit). Not read when
        ///     <paramref name="keyLen"/> is 0.</param>
        /// <param name="keyLen">Number of elements in v; 0 is legal (only the sorter's middle entry is checked).</param>
        /// <param name="sorterPtr">Pointer to int64 sorter array (contiguous), or null for no sorter.</param>
        /// <param name="retPtr">Output int64* (contiguous, keyLen elements). It also holds each key's running base between
        ///     the bisection's passes, so it must not alias the keys or the haystack.</param>
        /// <returns>0 when every key was searched; -1 (NumPy's argbinsearch) when the bisection reached a sorter entry outside
        ///     <c>[0, arrLen)</c>: the search stops there, before that entry addresses <paramref name="arrPtr"/>, and the caller
        ///     raises NumPy's <c>ValueError: Sorter index out of range.</c> (the bases already written are discarded with the
        ///     result, as NumPy's are). The middle entry is reached before any key, so -1 can come with no key at all. Always 0
        ///     without a sorter.</returns>
        public unsafe delegate long SearchSortedKernel(
            void* arrPtr,
            long arrLen,
            long arrStrideBytes,
            void* keyPtr,
            long keyLen,
            void* sorterPtr,
            long* retPtr);

        internal readonly struct SearchKernelKey : IEquatable<SearchKernelKey>
        {
            public readonly NPTypeCode Type;
            public readonly bool LeftSide;
            public readonly bool HasSorter;
            public readonly bool ContiguousA;  // if true, arrStrideBytes ignored — use elemSize as constant

            public SearchKernelKey(NPTypeCode type, bool leftSide, bool hasSorter, bool contiguousA)
            { Type = type; LeftSide = leftSide; HasSorter = hasSorter; ContiguousA = contiguousA; }

            public bool Equals(SearchKernelKey other) => Type == other.Type && LeftSide == other.LeftSide && HasSorter == other.HasSorter && ContiguousA == other.ContiguousA;
            public override bool Equals(object obj) => obj is SearchKernelKey k && Equals(k);
            public override int GetHashCode() => ((int)Type * 8) | (LeftSide ? 4 : 0) | (HasSorter ? 2 : 0) | (ContiguousA ? 1 : 0);
            public override string ToString() => $"{Type}_{(LeftSide ? "Left" : "Right")}_{(HasSorter ? "Sort" : "NoSort")}_{(ContiguousA ? "Contig" : "Strided")}";
        }

        internal static readonly ConcurrentDictionary<SearchKernelKey, SearchSortedKernel> _searchCache = new();

        // Resolved once (fail-fast at type load) so kernel emission never pays the GetMethod cost.
        private static readonly MethodInfo _numpyComplexLess =
            typeof(DirectILKernelGenerator).GetMethod(nameof(NumpyComplexLess),
                BindingFlags.Static | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("NumpyComplexLess not found");

        /// <summary>
        /// NumPy's <c>CDOUBLE_LT</c> (npysort_common.h): NaN-aware lexicographic complex "less-than",
        /// returning 1/0 for the IL comparator convention. The real parts compare first; when they are
        /// equal (or BOTH real parts are NaN) the imaginary parts break the tie; and any NaN component
        /// sorts LAST. This is the SAME total order <c>np.sort</c> / <c>np.sort_complex</c> apply
        /// (see <c>AxisSort.ComplexCmp</c>), so <c>searchsorted</c> stays consistent with a
        /// complex-sorted array — a real-only compare did not.
        /// </summary>
        internal static int NumpyComplexLess(System.Numerics.Complex a, System.Numerics.Complex b)
        {
            double ar = a.Real, ai = a.Imaginary, br = b.Real, bi = b.Imaginary;
            bool aiN = double.IsNaN(ai), biN = double.IsNaN(bi);
            if (ar < br) return (!aiN || biN) ? 1 : 0;                        // a.im not NaN, OR b.im NaN
            if (ar > br) return (biN && !aiN) ? 1 : 0;                        // b.im NaN AND a.im not NaN
            if (ar == br || (double.IsNaN(ar) && double.IsNaN(br)))           // reals equal, or both real-NaN
                return (ai < bi || (biN && !aiN)) ? 1 : 0;
            return double.IsNaN(br) ? 1 : 0;                                  // b.real is NaN
        }

        /// <summary>
        /// Get or generate a searchsorted kernel.
        /// </summary>
        /// <param name="type">Element dtype of a (and contiguous v, after caller normalizes).</param>
        /// <param name="leftSide">true = side='left', false = side='right'.</param>
        /// <param name="hasSorter">true = sorter param non-null (kernel emits sort_idx indirection).</param>
        /// <param name="contiguousA">true = a is contiguous (arrStrideBytes == elemSize). Lets JIT
        /// use scaled-index addressing instead of imul, which closes the gap to NumPy on the random-key
        /// hot path. When false, arrStrideBytes is honored as a runtime parameter.</param>
        public static SearchSortedKernel GetSearchSortedKernel(NPTypeCode type, bool leftSide, bool hasSorter, bool contiguousA)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");
            return _searchCache.GetOrAdd(new SearchKernelKey(type, leftSide, hasSorter, contiguousA), GenerateSearchKernel);
        }

        private static SearchSortedKernel GenerateSearchKernel(SearchKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"SearchSorted_{key}",
                returnType: typeof(long),   // 0, or -1 for a sorter entry outside the haystack (SearchSortedKernel)
                parameterTypes: new[]
                {
                    typeof(void*),  // arrPtr        (arg 0)
                    typeof(long),   // arrLen        (arg 1)
                    typeof(long),   // arrStrideBytes(arg 2)
                    typeof(void*),  // keyPtr        (arg 3)
                    typeof(long),   // keyLen        (arg 4)
                    typeof(void*),  // sorterPtr     (arg 5)
                    typeof(long*),  // retPtr        (arg 6)
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            EmitSearchSortedLoop(dm.GetILGenerator(), key);
            return dm.CreateDelegate<SearchSortedKernel>();
        }

        /// <summary>
        ///     Emit NumPy 2.5's batched binary search in IL: <c>binsearch&lt;Tag, side&gt;</c> (no sorter) or
        ///     <c>argbinsearch&lt;Tag, side&gt;</c> (a sorter) of numpy/_core/src/npysort/binsearch.cpp, statement for
        ///     statement (gh-30517).
        /// </summary>
        /// <param name="il">The kernel's IL generator (the <see cref="SearchSortedKernel"/> signature).</param>
        /// <param name="key">The dtype, side, sorter and haystack-contiguity the kernel is specialized for.</param>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements, the candidates of key j being <c>[base_j, base_j + interval_length]</c>:
        ///     <code>
        ///     if (arr_len &lt;= 0) { every ret[i] = 0; return; }
        ///     interval_length = arr_len; half = interval_length &gt;&gt; 1; interval_length -= half;
        ///     mid_val = arr[sort ? sort[half] : half];                      // base is 0 for every key
        ///     for i: ret[i] = cmp(mid_val, key[i]) * half;
        ///     while (interval_length &gt; 1) {
        ///         half = interval_length &gt;&gt; 1; interval_length -= half;
        ///         for i: base = ret[i]; ret[i] = base + cmp(arr[sort ? sort[base + half] : base + half], key[i]) * half;
        ///     }
        ///     for i: base = ret[i]; ret[i] = base + cmp(arr[sort ? sort[base] : base], key[i]);
        ///     </code>
        ///     <c>cmp</c> is <c>Tag::less</c> for <c>side='left'</c> and <c>Tag::less_equal</c> for <c>'right'</c>
        ///     (<see cref="EmitCmpForSide"/>), and every sorter entry is range-checked before it addresses the haystack.
        ///     </para>
        ///     <para>
        ///     The pass order is what decides an UNSORTED haystack's answers (which pivots each key meets), so it is kept
        ///     exactly: a per-key bisection — NumPy 2.4's, which also carried the previous key's bound — finds the same index
        ///     only when the haystack is sorted. The first pivot is read, and with a sorter range-checked, before the first
        ///     key: with no key a bad middle entry still returns -1, NumPy 2.5's <c>Sorter index out of range.</c>.
        ///     </para>
        ///     <para>
        ///     The comparison feeds the base arithmetically (<c>cmp * half</c>, no branch), as NumPy's does: the keys of a
        ///     pass are independent, so the processor overlaps them instead of waiting on mispredicted branches.
        ///     </para>
        /// </remarks>
        private static void EmitSearchSortedLoop(ILGenerator il, SearchKernelKey key)
        {
            int elemSize = GetTypeSize(key.Type);

            // Compare in the element's OWN dtype. Complex loads the full 16-byte struct and compares
            // lexicographically (real, then imaginary) via NumPy's CDOUBLE_LT (see EmitLess), so
            // searchsorted agrees with np.sort / np.sort_complex — which use the same total order.
            // (A real-only compare diverged whenever two elements shared a real part but differed in
            //  imag, or a key carried a NaN imaginary component.)
            var compareType = key.Type;
            var compareClrType = GetClrType(key.Type);

            // ---- locals ----
            var locI      = il.DeclareLocal(typeof(long));   // key index of the current pass
            var locLen    = il.DeclareLocal(typeof(long));   // interval_length: key j's candidates are [base_j, base_j + length]
            var locHalf   = il.DeclareLocal(typeof(long));   // half = length >> 1, the pivot's distance from a base
            var locBase   = il.DeclareLocal(typeof(long));   // one key's base, read from ret[i] and written back to it
            var locMidIdx = il.DeclareLocal(typeof(long));   // the pivot's haystack index (after the sorter's indirection)
            var locKey    = il.DeclareLocal(compareClrType); // key[i]
            var locMidVal = il.DeclareLocal(compareClrType); // the pivot's value

            // ---- labels ----
            var lblReturn     = il.DefineLabel();
            var lblBadSorter  = il.DefineLabel();
            var lblNonEmpty   = il.DefineLabel();
            var lblPassCheck  = il.DefineLabel();
            var lblPassBody   = il.DefineLabel();

            // Pushes &ret[i] (the result doubles as the per-key base store between passes).
            void EmitRetAddress()
            {
                il.Emit(OpCodes.Ldarg, 6);             // retPtr
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 8L);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }

            // locKey = key[i]: the keys are contiguous, elemSize apart (the caller materializes them).
            void EmitLoadKey()
            {
                il.Emit(OpCodes.Ldarg, 3);             // keyPtr
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect(il, compareType);
                il.Emit(OpCodes.Stloc, locKey);
            }

            // locMidIdx holds a position in the sorted order; a sorter maps it to sorter[position]. argbinsearch's
            // `if (mid_idx < 0 || mid_idx >= arr_len) return -1;` runs before the entry addresses a: the caller's sorter
            // checks cover its rank, dtype and size, not its values. As unsigned numbers a negative index is above every
            // length, so one compare covers both ends.
            void EmitSorterIndirection()
            {
                if (!key.HasSorter)
                    return;
                il.Emit(OpCodes.Ldarg, 5);             // sorterPtr
                il.Emit(OpCodes.Ldloc, locMidIdx);
                il.Emit(OpCodes.Ldc_I8, 8L);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, locMidIdx);
                il.Emit(OpCodes.Ldloc, locMidIdx);
                il.Emit(OpCodes.Ldarg_1);              // arrLen
                il.Emit(OpCodes.Bge_Un, lblBadSorter);
            }

            // locMidVal = a[locMidIdx]. For a contiguous a the element size is baked in as a constant so the JIT can use
            // scaled-index addressing instead of a runtime imul (Complex's 16-byte struct included); a strided a reads its
            // byte stride from the arguments.
            void EmitLoadPivot()
            {
                il.Emit(OpCodes.Ldarg_0);              // arrPtr
                il.Emit(OpCodes.Ldloc, locMidIdx);
                if (key.ContiguousA)
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                else
                    il.Emit(OpCodes.Ldarg_2);          // arrStrideBytes (runtime)
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect(il, compareType);
                il.Emit(OpCodes.Stloc, locMidVal);
            }

            // for (i = 0; i < keyLen; i++) body(); — every pass visits the keys in order.
            void EmitKeyLoop(Action body)
            {
                var lblCheck = il.DefineLabel();
                var lblBody = il.DefineLabel();
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblCheck);
                il.MarkLabel(lblBody);
                body();
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.MarkLabel(lblCheck);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg, 4);             // keyLen
                il.Emit(OpCodes.Blt, lblBody);
            }

            // ---- if (arr_len <= 0): every insertion point is 0, and no sorter entry exists to check ----
            il.Emit(OpCodes.Ldarg_1);                  // arrLen
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bgt, lblNonEmpty);
            EmitKeyLoop(() =>
            {
                EmitRetAddress();
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stind_I8);
            });
            il.Emit(OpCodes.Br, lblReturn);

            il.MarkLabel(lblNonEmpty);

            // ---- interval_length = arr_len; half = interval_length >> 1; interval_length -= half (ceil(length / 2)) ----
            il.Emit(OpCodes.Ldarg_1);                  // arrLen
            il.Emit(OpCodes.Stloc, locLen);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Shr);
            il.Emit(OpCodes.Stloc, locHalf);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldloc, locHalf);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locLen);

            // ---- the unrolled first pass: every base is 0, so one pivot a[half] (read once) serves every key, and ret is
            //      written without being read (NumPy's saving of |keys| writes and reads). Its sorter entry is checked
            //      here, before any key: an empty key set can still fail. ----
            il.Emit(OpCodes.Ldloc, locHalf);
            il.Emit(OpCodes.Stloc, locMidIdx);
            EmitSorterIndirection();
            EmitLoadPivot();
            EmitKeyLoop(() =>
            {
                // ret[i] = cmp(mid_val, key[i]) * half
                EmitLoadKey();
                EmitRetAddress();
                EmitCmpForSide(il, compareType, key.LeftSide, locMidVal, locKey);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Ldloc, locHalf);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Stind_I8);
            });

            // ---- while (interval_length > 1): halve the shared length, then move every key's base ----
            il.Emit(OpCodes.Br, lblPassCheck);
            il.MarkLabel(lblPassBody);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Shr);
            il.Emit(OpCodes.Stloc, locHalf);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldloc, locHalf);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locLen);
            EmitKeyLoop(() =>
            {
                // base = ret[i]; mid = a[sorter ? sorter[base + half] : base + half]; ret[i] = base + cmp(mid, key[i]) * half
                EmitRetAddress();
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, locBase);
                il.Emit(OpCodes.Ldloc, locBase);
                il.Emit(OpCodes.Ldloc, locHalf);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locMidIdx);
                EmitSorterIndirection();
                EmitLoadPivot();
                EmitLoadKey();
                EmitRetAddress();
                il.Emit(OpCodes.Ldloc, locBase);
                EmitCmpForSide(il, compareType, key.LeftSide, locMidVal, locKey);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Ldloc, locHalf);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stind_I8);
            });
            il.MarkLabel(lblPassCheck);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bgt, lblPassBody);

            // ---- interval_length == 1: key j's insertion point is base_j or base_j + 1, decided by a[base_j] ----
            EmitKeyLoop(() =>
            {
                // base = ret[i]; ret[i] = base + cmp(a[sorter ? sorter[base] : base], key[i])
                EmitRetAddress();
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, locBase);
                il.Emit(OpCodes.Ldloc, locBase);
                il.Emit(OpCodes.Stloc, locMidIdx);
                EmitSorterIndirection();
                EmitLoadPivot();
                EmitLoadKey();
                EmitRetAddress();
                il.Emit(OpCodes.Ldloc, locBase);
                EmitCmpForSide(il, compareType, key.LeftSide, locMidVal, locKey);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stind_I8);
            });

            il.MarkLabel(lblReturn);
            il.Emit(OpCodes.Ldc_I8, 0L);               // every key searched
            il.Emit(OpCodes.Ret);

            // A sorter entry outside the haystack was reached (only a sorter kernel branches here).
            il.MarkLabel(lblBadSorter);
            il.Emit(OpCodes.Ldc_I8, -1L);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>
        /// Emit the comparison that moves a key's base for the requested side, reading the two operands
        /// from <paramref name="a"/> and <paramref name="b"/> locals (they are reloaded, which the
        /// NaN-aware form requires). Pushes int32 (0 or 1), which the caller multiplies by the half
        /// interval, as NumPy's <c>cmp(mid_val, key_val) * half</c> does.
        ///
        /// Mirrors NumPy binsearch.cpp + numpy_tag.h: both side comparators derive from a single
        /// <c>less</c> functor —
        ///   side='left'  cmp(a,b) = less(a, b)
        ///   side='right' cmp(a,b) = less_equal(a, b) = !less(b, a)
        /// For the float tags <c>less</c> is NaN-aware ("NaN sorts last"): finite &lt; NaN is true,
        /// NaN &lt; finite / NaN &lt; NaN are false. So a NaN key bisects to the end of a sorted haystack.
        /// Integer/char/bool have no NaN and collapse to ordered '&lt;'.
        /// </summary>
        /// <param name="il">The kernel's IL generator.</param>
        /// <param name="type">The compared dtype.</param>
        /// <param name="leftSide">True for <c>side='left'</c>.</param>
        /// <param name="a">The first operand's local (the haystack pivot).</param>
        /// <param name="b">The second operand's local (the key).</param>
        private static void EmitCmpForSide(ILGenerator il, NPTypeCode type, bool leftSide, LocalBuilder a, LocalBuilder b)
        {
            if (leftSide)
            {
                EmitLess(il, type, a, b);            // less(a, b)
            }
            else
            {
                EmitLess(il, type, b, a);            // less(b, a)
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Ceq);                // !less(b, a)  ==  less_equal(a, b)
            }
        }

        /// <summary>
        /// Emit NumPy's <c>less(x, y)</c> functor (numpy_tag.h), leaving int32 0/1 on the stack.
        /// Float tags: <c>(x &lt; y) | (x==x &amp; y!=y)</c> — the second term makes NaN sort last
        /// (finite &lt; NaN true, NaN &lt; finite false, NaN &lt; NaN false), matching
        /// FLOAT_LT / DOUBLE_LT / HALF_LT. Complex uses NumPy's CDOUBLE_LT (NaN-aware lexicographic
        /// real-then-imag) via <see cref="NumpyComplexLess"/>. Decimal has no NaN and uses its
        /// ordered operator; integer/char/bool use the ordered clt / clt.un.
        /// </summary>
        private static void EmitLess(ILGenerator il, NPTypeCode type, LocalBuilder x, LocalBuilder y)
        {
            switch (type)
            {
                case NPTypeCode.Complex:
                    il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ldloc, y);
                    il.EmitCall(OpCodes.Call, _numpyComplexLess, null);   // CDOUBLE_LT -> int 0/1
                    return;

                case NPTypeCode.Single:
                case NPTypeCode.Double:
                    // (x < y) | (x==x & y!=y)      [ceq/clt are ordered: NaN operand -> 0]
                    il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ldloc, y); il.Emit(OpCodes.Clt);
                    il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ceq);   // x==x  (0 if x NaN)
                    il.Emit(OpCodes.Ldloc, y); il.Emit(OpCodes.Ldloc, y); il.Emit(OpCodes.Ceq);
                    il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Ceq);                              // y!=y  (1 if y NaN)
                    il.Emit(OpCodes.And);
                    il.Emit(OpCodes.Or);
                    return;

                case NPTypeCode.Half:
                {
                    var hlt = ScalarMethodCache.BinaryOp(typeof(Half), "op_LessThan");
                    var heq = ScalarMethodCache.BinaryOp(typeof(Half), "op_Equality");
                    il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ldloc, y); il.EmitCall(OpCodes.Call, hlt, null);
                    il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ldloc, x); il.EmitCall(OpCodes.Call, heq, null);
                    il.Emit(OpCodes.Ldloc, y); il.Emit(OpCodes.Ldloc, y); il.EmitCall(OpCodes.Call, heq, null);
                    il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Ceq);
                    il.Emit(OpCodes.And);
                    il.Emit(OpCodes.Or);
                    return;
                }

                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ldloc, y);
                    il.EmitCall(OpCodes.Call, ScalarMethodCache.BinaryOp(typeof(decimal), "op_LessThan"), null);
                    return;

                default:
                    // integers, char, bool — ordered '<', no NaN.
                    bool isUnsigned = IsUnsigned(type) || type == NPTypeCode.Char || type == NPTypeCode.Boolean;
                    il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Ldloc, y);
                    il.Emit(isUnsigned ? OpCodes.Clt_Un : OpCodes.Clt);
                    return;
            }
        }
    }
}
