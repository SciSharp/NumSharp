using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

namespace NumSharp.Backends.Kernels
{
    // =============================================================================
    // DirectILKernelGenerator.Search.cs
    //   OWNERSHIP: searchsorted binary-search kernels with NumPy-style monotonic optimization.
    //   RESPONSIBILITY:
    //     - Typed inner loop per (NPTypeCode, side, has_sorter), no boxing / no double conversion.
    //     - Monotonic key bound tracking: when consecutive keys ascend (per the side's cmp), the
    //       lower bound L carries from the previous iteration — same trick NumPy uses (binsearch.cpp).
    //     - Caller must materialize v (keyPtr) to contiguous before calling; sorter must be int64
    //       and contiguous. a (arrPtr) may be strided via arrStrideBytes.
    //   PARITY WITH NUMPY:
    //     - Mirrors numpy/_core/src/npysort/binsearch.cpp template binsearch<Tag, side>.
    //     - side='left'  cmp = (val <  target)  -> bisect_left  (returns first i with a[i] >= target)
    //     - side='right' cmp = (val <= target)  -> bisect_right (returns first i with a[i] >  target)
    // =============================================================================
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// SearchSorted kernel delegate.
        /// </summary>
        /// <param name="arrPtr">Pointer to a's data, already at a.Shape.offset.</param>
        /// <param name="arrLen">Number of elements in a.</param>
        /// <param name="arrStrideBytes">a's stride in bytes (= elemSize when contiguous).</param>
        /// <param name="keyPtr">Pointer to v's data; v MUST be contiguous (elemSize stride implicit).</param>
        /// <param name="keyLen">Number of elements in v.</param>
        /// <param name="sorterPtr">Pointer to int64 sorter array (contiguous), or null for no sorter.</param>
        /// <param name="retPtr">Output int64* (contiguous, keyLen elements).</param>
        public unsafe delegate void SearchSortedKernel(
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
                returnType: typeof(void),
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
        /// Emit the searchsorted inner loop in IL.
        /// Mirrors NumPy's binsearch.cpp template:
        ///   while keys remain:
        ///     compute key bounds via monotonic check
        ///     bisect using cmp(midVal, key)
        ///     write result, carry L/R to next iteration
        /// </summary>
        private static void EmitSearchSortedLoop(ILGenerator il, SearchKernelKey key)
        {
            int elemSize = GetTypeSize(key.Type);

            // Complex: load only the Real (double) component for comparison.
            // System.Numerics.Complex memory layout puts Real first (m_real, m_imaginary),
            // so ldind.r8 on a Complex* yields Real directly while the element stride remains 16 bytes.
            // Matches the legacy NumSharp behavior (Converts.ToDouble(Complex) -> Real).
            bool isComplex = key.Type == NPTypeCode.Complex;
            var compareType = isComplex ? NPTypeCode.Double : key.Type;
            var compareClrType = isComplex ? typeof(double) : GetClrType(key.Type);

            // ---- locals ----
            var locI       = il.DeclareLocal(typeof(long));   // outer key index
            var locL       = il.DeclareLocal(typeof(long));   // bisect lower bound (current iteration)
            var locR       = il.DeclareLocal(typeof(long));   // bisect upper bound
            var locPrevL   = il.DeclareLocal(typeof(long));   // carried lower bound from prev iteration
            var locPrevR   = il.DeclareLocal(typeof(long));   // carried upper bound
            var locM       = il.DeclareLocal(typeof(long));   // bisect midpoint
            var locMidIdx  = il.DeclareLocal(typeof(long));   // post-sorter midpoint
            var locKey     = il.DeclareLocal(compareClrType); // current key value
            var locLastKey = il.DeclareLocal(compareClrType); // previous key value (for monotonic check)
            var locMidVal  = il.DeclareLocal(compareClrType); // value at midpoint

            // ---- labels ----
            var lblReturn      = il.DefineLabel();
            var lblOuter       = il.DefineLabel();
            var lblOuterBody   = il.DefineLabel();
            var lblBisect      = il.DefineLabel();
            var lblBisectDone  = il.DefineLabel();
            var lblAscending   = il.DefineLabel();
            var lblBoundsReady = il.DefineLabel();
            var lblMoveLeft    = il.DefineLabel();

            // ---- early-out: if (keyLen == 0) return ----
            il.Emit(OpCodes.Ldarg, 4);             // keyLen
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblReturn);       // keyLen <= 0 ? return

            // ---- init: prevL = 0; prevR = arrLen; lastKey = *(T*)keyPtr; i = 0; ----
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locPrevL);

            il.Emit(OpCodes.Ldarg, 1);             // arrLen
            il.Emit(OpCodes.Stloc, locPrevR);

            il.Emit(OpCodes.Ldarg, 3);             // keyPtr
            EmitLoadIndirect(il, compareType);     // Complex -> r8 (Real), others -> typed load
            il.Emit(OpCodes.Stloc, locLastKey);

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ---- outer loop ----
            il.MarkLabel(lblOuter);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg, 4);             // keyLen
            il.Emit(OpCodes.Blt, lblOuterBody);
            il.Emit(OpCodes.Br, lblReturn);

            il.MarkLabel(lblOuterBody);

            // ---- load key[i] ----
            // ptr = keyPtr + i * elemSize
            il.Emit(OpCodes.Ldarg, 3);             // keyPtr
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);               // long -> native int
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, compareType);
            il.Emit(OpCodes.Stloc, locKey);

            // ---- monotonic bound update ----
            // bool ascending = cmp(lastKey, key)
            //   side='left':  cmp = less(a, b)
            //   side='right': cmp = less_equal(a, b) = !less(b, a)
            // less is NaN-aware for float tags (NaN sorts last), so a NaN key never reports
            // "ascending" against a following finite key — it resets the carried bound (numpy parity).
            EmitCmpForSide(il, compareType, key.LeftSide, locLastKey, locKey);   // ascending? -> int 0/1
            il.Emit(OpCodes.Brtrue, lblAscending);

            // descending branch: L = 0; R = min(prevR + 1, arrLen)
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locL);

            il.Emit(OpCodes.Ldloc, locPrevR);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locR);
            // if (R > arrLen) R = arrLen
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Ldarg, 1);             // arrLen
            il.Emit(OpCodes.Ble, lblBoundsReady);
            il.Emit(OpCodes.Ldarg, 1);
            il.Emit(OpCodes.Stloc, locR);
            il.Emit(OpCodes.Br, lblBoundsReady);

            // ascending branch: L = prevL; R = arrLen
            il.MarkLabel(lblAscending);
            il.Emit(OpCodes.Ldloc, locPrevL);
            il.Emit(OpCodes.Stloc, locL);
            il.Emit(OpCodes.Ldarg, 1);             // arrLen
            il.Emit(OpCodes.Stloc, locR);

            il.MarkLabel(lblBoundsReady);

            // lastKey = key
            il.Emit(OpCodes.Ldloc, locKey);
            il.Emit(OpCodes.Stloc, locLastKey);

            // ---- bisect loop: while (L < R) { ... } ----
            il.MarkLabel(lblBisect);
            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Bge, lblBisectDone);

            // m = L + ((R - L) >> 1)
            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Shr);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locM);

            // midIdx = sorter ? sorter[m] : m
            if (key.HasSorter)
            {
                il.Emit(OpCodes.Ldarg, 5);         // sorterPtr
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Ldc_I8, 8L);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, locMidIdx);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Stloc, locMidIdx);
            }

            // midVal = *(T*)(arrPtr + midIdx * arrStrideBytes)
            // For contiguous a: bake elemSize as a constant so the JIT can use scaled-index addressing
            // and avoid the runtime imul. (Complex stride remains 16 because the struct is 16 bytes.)
            il.Emit(OpCodes.Ldarg_0);              // arrPtr
            il.Emit(OpCodes.Ldloc, locMidIdx);
            if (key.ContiguousA)
            {
                il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            }
            else
            {
                il.Emit(OpCodes.Ldarg_2);          // arrStrideBytes (runtime)
            }
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, compareType);
            il.Emit(OpCodes.Stloc, locMidVal);

            // if (cmp(midVal, key)) L = m + 1; else R = m;
            EmitCmpForSide(il, compareType, key.LeftSide, locMidVal, locKey);
            il.Emit(OpCodes.Brfalse, lblMoveLeft);

            // L = m + 1
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locL);
            il.Emit(OpCodes.Br, lblBisect);

            il.MarkLabel(lblMoveLeft);
            // R = m
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Stloc, locR);
            il.Emit(OpCodes.Br, lblBisect);

            il.MarkLabel(lblBisectDone);

            // *(retPtr + i*8) = L
            il.Emit(OpCodes.Ldarg, 6);             // retPtr
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Stind_I8);

            // prevL = L; prevR = R; i++
            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Stloc, locPrevL);
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Stloc, locPrevR);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuter);

            il.MarkLabel(lblReturn);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>
        /// Emit the comparison that drives bisect direction / carry for the requested side,
        /// reading the two operands from <paramref name="a"/> and <paramref name="b"/> locals
        /// (they are reloaded, which the NaN-aware form requires). Pushes int32 (0 or 1).
        ///
        /// Mirrors NumPy binsearch.cpp + numpy_tag.h: both side comparators derive from a single
        /// <c>less</c> functor —
        ///   side='left'  cmp(a,b) = less(a, b)
        ///   side='right' cmp(a,b) = less_equal(a, b) = !less(b, a)
        /// For the float tags <c>less</c> is NaN-aware ("NaN sorts last"): finite &lt; NaN is true,
        /// NaN &lt; finite / NaN &lt; NaN are false. So a NaN key bisects to the end and never carries
        /// a stale bound into a following key. Integer/char/bool have no NaN and collapse to ordered '&lt;'.
        /// </summary>
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
        /// FLOAT_LT / DOUBLE_LT / HALF_LT. Complex flows here via its Real (Double) component.
        /// Decimal has no NaN and uses its ordered operator; integer/char/bool use the ordered
        /// clt / clt.un.
        /// </summary>
        private static void EmitLess(ILGenerator il, NPTypeCode type, LocalBuilder x, LocalBuilder y)
        {
            switch (type)
            {
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
