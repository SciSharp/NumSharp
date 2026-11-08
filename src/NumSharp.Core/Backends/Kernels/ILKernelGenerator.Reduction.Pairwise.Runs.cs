using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// ILKernelGenerator.Reduction.Pairwise.Runs.cs — the pairwise fold's per-RUN-KIND
// SIMD folds (contiguous, reversed, every other element) and their two-leaf nodes
// =============================================================================
//
// NumPy's pairwise_sum (loops_utils.h.src) is ONE scalar routine for every stride:
// eight accumulators r[0..7] per leaf of at most 128 elements, r[k] summing elements
// k, k+8, k+16, …, a tree combine, and a split at n/2 rounded down to a multiple of 8.
// Its bits depend only on WHICH element reaches WHICH accumulator in WHICH order —
// not on how the elements are loaded — so any load that puts element j+k in lane k
// reproduces it exactly.
//
// The first emitted fold (ILKernelGenerator.Reduction.Pairwise.cs) vectorized only the
// contiguous leaf. Two costs remained, both measured on a 100K-element float64 run
// (Raptor Lake P-core, AVX2):
//
//   * LATENCY. A leaf's accumulators are 8 dependent add chains, i.e. 2 vector chains
//     for float64 and ONE for float32 on 256-bit vectors: 15 dependent adds per leaf.
//     Interleaving the two leaves of a node whose children are both leaves doubles the
//     independent chains without touching any chain's order: float64 7.6 -> 5.8 µs,
//     float32 6.9 -> 3.8 µs (a four-leaf interleave measured no better).
//   * STRIDED RUNS fell to the scalar eight-accumulator leaf. A reversed run (stride -1)
//     is contiguous memory read backwards: one load + one lane reversal per vector
//     (float64 16.0 -> 8.1 µs, float32 14.5 -> 7.7 µs). Every other element (stride 2)
//     is two loads + a blend + a lane permutation (float64 18.2 -> 12.1 µs, float32
//     33.2 -> 7.4 µs). An AVX2 gather measured no faster than the scalar leaf for any
//     stride, so the remaining strides keep it.
//
// The stride-2 block never reads past its last element: for float64 it loads p[0..3]
// and p[3..6] and blends lanes (p0, p4, p2, p6) out of them, where an unpack of p[0..3]
// and p[4..7] would read p[7] — one element beyond the run's end when the run is the
// last thing in its buffer (x[::2] of an odd-length array). float32 loads p[0..7] and
// p[7..14] the same way.
//
// complex128 (NumPy's CDOUBLE_pairwise_sum: four accumulators per part over the
// interleaved doubles, leaves of at most 64 complexes, split at (nc - nc%8)/2) gets the
// same two levers: two complexes per 256-bit register (2 vector chains in place of 4
// 128-bit ones) and the two-leaf node — 27.2 -> 12.2 µs for 100K complexes — for
// contiguous and reversed runs.
//
// Each kind is its own self-recursive DynamicMethod (no per-element stride arithmetic,
// no kind test in the loops) reached through the stride dispatch at the top of the
// generic fold, which keeps every caller's fold(a, n, stride) contract unchanged.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>
        ///     The element step a per-kind pairwise fold walks. The value IS the step in elements, which the IL uses
        ///     directly for its scalar tails and its recursion offsets.
        /// </summary>
        internal enum PairwiseRunKind
        {
            /// <summary>Stride +1: plain vector loads.</summary>
            Contiguous = 1,
            /// <summary>Stride -1: a vector load ending at the block's first element, lanes reversed.</summary>
            Reversed = -1,
            /// <summary>Stride +2: two overlapping vector loads, the even elements blended and permuted into place.</summary>
            Stride2 = 2,
            /// <summary>
            ///     Any stride, passed at run time (complex128 only: each complex is one 128-bit load wherever it sits, so a
            ///     register pair is two loads combined; the real folds keep their scalar leaf for such strides).
            /// </summary>
            AnyStride = 0,
        }

        /// <summary>
        ///     The per-kind folds, rooted here so the <c>call</c> sites in the dispatching fold (and each kind's own
        ///     self-recursion) stay alive for the life of the process — a DynamicMethod referenced only from another
        ///     DynamicMethod's IL is otherwise collectable.
        /// </summary>
        internal static readonly ConcurrentDictionary<(NPTypeCode Type, PairwiseRunKind Kind), DynamicMethod> _pwRunFolds = new();

        /// <summary>
        ///     Whether the reversed / stride-2 real folds can be emitted on this host: they are written against AVX2's
        ///     lane permutations at 256 bits (the lane count does not change any bit, so a wider host still uses them).
        /// </summary>
        private static bool PairwiseRunKindsSupported => Avx2.IsSupported;

        /// <summary>
        ///     Whether the pairwise fold of <paramref name="type"/> reads a run at element stride <paramref name="stride"/>
        ///     with SIMD loads: stride 1 always; on AVX2 hosts also -1 and 2 (float32 / float64 / float16 run kinds) and
        ///     every stride for complex128 (two complexes per register). Any other stride falls to the folds' scalar
        ///     8-accumulator path — the same bits, but a caller that could copy the run into a contiguous block first
        ///     (a strided copy and a SIMD fold) should then prefer the copy.
        /// </summary>
        /// <param name="type">The fold's element dtype (float16, float32, float64 or complex128).</param>
        /// <param name="stride">The run's element stride (may be negative).</param>
        /// <returns>True when folding the run in place keeps the SIMD fold.</returns>
        internal static bool PairwiseFoldIsVectorized(NPTypeCode type, long stride)
            => stride == 1
               || (PairwiseRunKindsSupported
                   && (type == NPTypeCode.Complex || stride == -1 || stride == 2));

        /// <summary>
        ///     The cached per-kind real fold <c>T fold(void* a, long n)</c> for <paramref name="tc"/>, emitted on first use.
        /// </summary>
        /// <param name="tc">Double or Single.</param>
        /// <param name="kind">The run kind.</param>
        /// <param name="clrType">The element CLR type.</param>
        /// <param name="elemSize">The element size in bytes.</param>
        /// <param name="simdBits">The vector width for <see cref="PairwiseRunKind.Contiguous"/> (the reversed and
        /// stride-2 kinds always use 256 bits).</param>
        /// <param name="laneCount">Elements per vector at <paramref name="simdBits"/> (must divide 8).</param>
        /// <returns>The DynamicMethod.</returns>
        private static DynamicMethod GetPairwiseRunFold(NPTypeCode tc, PairwiseRunKind kind, Type clrType, int elemSize, int simdBits, int laneCount)
            => _pwRunFolds.GetOrAdd((tc, kind), k =>
            {
                int bits = k.Kind == PairwiseRunKind.Contiguous ? simdBits : 256;
                int lanes = k.Kind == PairwiseRunKind.Contiguous ? laneCount : 32 / elemSize;
                return EmitPairwiseRunFold(k.Type, k.Kind, clrType, elemSize, bits, lanes);
            });

        /// <summary>
        ///     Emits NumPy's pairwise_sum over a run of kind <paramref name="kind"/>: <c>T fold(void* a, long n)</c>, the
        ///     element j at <c>a + j·step·elemSize</c> (step = the kind's value). The structure is NumPy's — n &lt; 8 a
        ///     sequential sum seeded with -0.0, n ≤ 128 one leaf of eight accumulators and the tree combine, a longer run
        ///     split at <c>n2 = n/2 - (n/2)%8</c> — plus a two-leaf node: when both halves are leaves they are computed in
        ///     one loop (each keeps its own accumulators and order, so the bits are the recursion's).
        /// </summary>
        /// <param name="tc">Double or Single.</param>
        /// <param name="kind">The run kind.</param>
        /// <param name="clrType">The element CLR type.</param>
        /// <param name="elemSize">The element size in bytes.</param>
        /// <param name="simdBits">The vector width.</param>
        /// <param name="laneCount">Elements per vector (divides 8).</param>
        /// <returns>The emitted method.</returns>
        /// <exception cref="InvalidOperationException">A vector method the kind needs is missing on this runtime.</exception>
        private static DynamicMethod EmitPairwiseRunFold(NPTypeCode tc, PairwiseRunKind kind, Type clrType, int elemSize, int simdBits, int laneCount)
        {
            Type pV = typeof(void).MakePointerType();
            var dm = new DynamicMethod($"PwFold_{tc}_{kind}", clrType, new[] { pV, typeof(long) },
                typeof(ILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();

            long step = (long)kind;
            int numAccVecs = 8 / laneCount;
            Type vecType = VectorMethodCache.V(simdBits, clrType);
            MethodInfo mLoad = VectorMethodCache.LoadX86(simdBits, clrType) ?? VectorMethodCache.Load(simdBits, clrType);
            MethodInfo mAdd = VectorMethodCache.BinaryX86(simdBits, "Add", clrType) ?? VectorMethodCache.Generic(simdBits, "Add", clrType, paramCount: 2);
            MethodInfo mGet = VectorMethodCache.GetElement(simdBits, clrType);
            OpCode ldT = LdindFloat(elemSize);

            // The lane-rearranging intrinsics of the reversed / stride-2 kinds (256-bit only, see GetPairwiseRunFold).
            bool isDouble = clrType == typeof(double);
            Type v256T = typeof(Vector256<>).MakeGenericType(clrType);
            MethodInfo mPermute4x64 = null, mPermuteVar = null, mBlend = null, mCreate8 = null;
            if (kind != PairwiseRunKind.Contiguous)
            {
                mBlend = typeof(Avx).GetMethod(nameof(Avx.Blend), new[] { v256T, v256T, typeof(byte) })
                         ?? throw new InvalidOperationException("Avx.Blend missing");
                if (isDouble)
                    mPermute4x64 = typeof(Avx2).GetMethod(nameof(Avx2.Permute4x64), new[] { typeof(Vector256<double>), typeof(byte) })
                                   ?? throw new InvalidOperationException("Avx2.Permute4x64 missing");
                else
                {
                    mPermuteVar = typeof(Avx2).GetMethod(nameof(Avx2.PermuteVar8x32), new[] { typeof(Vector256<float>), typeof(Vector256<int>) })
                                  ?? throw new InvalidOperationException("Avx2.PermuteVar8x32 missing");
                    mCreate8 = typeof(Vector256).GetMethod(nameof(Vector256.Create),
                                   new[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int) })
                               ?? throw new InvalidOperationException("Vector256.Create(int x8) missing");
                }
            }

            var locI = il.DeclareLocal(typeof(long));
            var locLim = il.DeclareLocal(typeof(long));
            var locN2 = il.DeclareLocal(typeof(long));
            var locNr = il.DeclareLocal(typeof(long));
            var locRes = il.DeclareLocal(clrType);
            var locResL = il.DeclareLocal(clrType);
            var locB = il.DeclareLocal(pV);
            var locCtl = il.DeclareLocal(typeof(Vector256<int>));
            var accL = new LocalBuilder[numAccVecs];
            var accR = new LocalBuilder[numAccVecs];
            for (int v = 0; v < numAccVecs; v++) { accL[v] = il.DeclareLocal(vecType); accR[v] = il.DeclareLocal(vecType); }

            var LBASE = il.DefineLabel();
            var LLEAF = il.DefineLabel();
            var LLEAF2 = il.DefineLabel();

            // push (basePtr + elemIndex·step·elemSize) where pushIndex leaves a long element index on the stack.
            void PushElemPtr(Action pushBase, Action pushIndex, long elemStep)
            {
                pushBase();
                pushIndex();
                il.Emit(OpCodes.Ldc_I8, elemStep * elemSize); il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }

            // The float32 control vector of a kind: the lane reversal, or the stride-2 permutation that turns the blend
            // (p0, p8, p2, p10, p4, p12, p6, p14) into (p0, p2, …, p14). Materialized once per leaf call.
            void EmitControlVector()
            {
                if (isDouble || kind == PairwiseRunKind.Contiguous)
                    return;
                int[] idx = kind == PairwiseRunKind.Reversed ? new[] { 7, 6, 5, 4, 3, 2, 1, 0 } : new[] { 0, 2, 4, 6, 1, 3, 5, 7 };
                foreach (int x in idx) il.Emit(OpCodes.Ldc_I4, x);
                il.Emit(OpCodes.Call, mCreate8);
                il.Emit(OpCodes.Stloc, locCtl);
            }

            // push the vector of the leaf's elements [off, off + laneCount) — element off + k in lane k — for the kind.
            // pushBase pushes the leaf's element 0, pushOff the (long) element offset within the leaf.
            void EmitBlock(Action pushBase, Action pushOff)
            {
                switch (kind)
                {
                    case PairwiseRunKind.Contiguous:
                        PushElemPtr(pushBase, pushOff, 1);
                        il.Emit(OpCodes.Call, mLoad);
                        return;
                    case PairwiseRunKind.Reversed:
                        // elements off .. off+L-1 sit at addresses base-(off+L-1) .. base-off: load from the lowest, reverse.
                        PushElemPtr(pushBase, () => { pushOff(); il.Emit(OpCodes.Ldc_I8, (long)(laneCount - 1)); il.Emit(OpCodes.Add); }, -1);
                        il.Emit(OpCodes.Call, mLoad);
                        if (isDouble) { il.Emit(OpCodes.Ldc_I4, 0b00_01_10_11); il.Emit(OpCodes.Call, mPermute4x64); }
                        else { il.Emit(OpCodes.Ldloc, locCtl); il.Emit(OpCodes.Call, mPermuteVar); }
                        return;
                    default:
                        // Stride2: elements off .. off+L-1 are p[0], p[2], …, p[2L-2] with p = base + 2·off. Load p[0..L)
                        // and p[L-1 .. 2L-1) — never past p[2L-2] — blend the odd lanes of the second over the first,
                        // then permute: float64 (p0,p4,p2,p6) -> (p0,p2,p4,p6); float32 (p0,p8,p2,p10,…) -> (p0,p2,…,p14).
                        PushElemPtr(pushBase, pushOff, 2);
                        il.Emit(OpCodes.Call, mLoad);
                        PushElemPtr(pushBase, () => { pushOff(); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Ldc_I8, (long)(laneCount - 1)); il.Emit(OpCodes.Add); }, 1);
                        il.Emit(OpCodes.Call, mLoad);
                        il.Emit(OpCodes.Ldc_I4, isDouble ? 0b1010 : 0b1010_1010);
                        il.Emit(OpCodes.Call, mBlend);
                        if (isDouble) { il.Emit(OpCodes.Ldc_I4, 0b11_01_10_00); il.Emit(OpCodes.Call, mPermute4x64); }
                        else { il.Emit(OpCodes.Ldloc, locCtl); il.Emit(OpCodes.Call, mPermuteVar); }
                        return;
                }
            }

            // push NumPy's tree combine of eight accumulators held in `acc`: ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)).
            void EmitCombine(LocalBuilder[] acc)
            {
                void LdR(int k) { il.Emit(OpCodes.Ldloc, acc[k / laneCount]); il.Emit(OpCodes.Ldc_I4, k % laneCount); il.Emit(OpCodes.Call, mGet); }
                LdR(0); LdR(1); il.Emit(OpCodes.Add); LdR(2); LdR(3); il.Emit(OpCodes.Add); il.Emit(OpCodes.Add);
                LdR(4); LdR(5); il.Emit(OpCodes.Add); LdR(6); LdR(7); il.Emit(OpCodes.Add); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Add);
            }

            // for (; i < end; i++) res += element i of the run at pushBase  (end pushed by pushEnd)
            void EmitScalarTail(Action pushBase, Action pushEnd)
            {
                var tL = il.DefineLabel(); var tE = il.DefineLabel();
                il.MarkLabel(tL);
                il.Emit(OpCodes.Ldloc, locI); pushEnd(); il.Emit(OpCodes.Bge, tE);
                il.Emit(OpCodes.Ldloc, locRes);
                PushElemPtr(pushBase, () => il.Emit(OpCodes.Ldloc, locI), step); il.Emit(ldT);
                il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locRes);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, tL);
                il.MarkLabel(tE);
            }

            void PushA() => il.Emit(OpCodes.Ldarg_0);
            void PushB() => il.Emit(OpCodes.Ldloc, locB);

            // ---- dispatch: n < 8 → BASE; n ≤ 128 → LEAF; both halves leaves → LEAF2; else recurse ----
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Blt, LBASE);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 128L); il.Emit(OpCodes.Ble, LLEAF);
            // n2 = n/2; n2 -= n2 % 8; nr = n - n2
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Div); il.Emit(OpCodes.Stloc, locN2);
            il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locN2);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locNr);
            // n2 ≤ nr always, so "nr ≤ 128" means both halves are leaves (n2 ≥ 57 > 8 for n > 128).
            il.Emit(OpCodes.Ldloc, locNr); il.Emit(OpCodes.Ldc_I8, 128L); il.Emit(OpCodes.Ble, LLEAF2);
            // REC: fold(a, n2) + fold(a + n2·step, nr)
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Call, dm);
            PushElemPtr(PushA, () => il.Emit(OpCodes.Ldloc, locN2), step);
            il.Emit(OpCodes.Ldloc, locNr); il.Emit(OpCodes.Call, dm);
            il.Emit(OpCodes.Add); il.Emit(OpCodes.Ret);

            // ---- BASE: res = -0; for i in [0, n): res += a[i·step] ----
            il.MarkLabel(LBASE);
            EmitNegZero(il, elemSize); il.Emit(OpCodes.Stloc, locRes);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            EmitScalarTail(PushA, () => il.Emit(OpCodes.Ldarg_1));
            il.Emit(OpCodes.Ldloc, locRes); il.Emit(OpCodes.Ret);

            // ---- LEAF: eight accumulators seeded with elements 0..7, blocks of 8 up to n - n%8, combine, tail ----
            il.MarkLabel(LLEAF);
            EmitControlVector();
            for (int v = 0; v < numAccVecs; v++)
            {
                int off = v * laneCount;
                EmitBlock(PushA, () => il.Emit(OpCodes.Ldc_I8, (long)off));
                il.Emit(OpCodes.Stloc, accL[v]);
            }
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locLim);
            il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Stloc, locI);
            {
                var lL = il.DefineLabel(); var lE = il.DefineLabel();
                il.MarkLabel(lL);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locLim); il.Emit(OpCodes.Bge, lE);
                for (int v = 0; v < numAccVecs; v++)
                {
                    int off = v * laneCount;
                    il.Emit(OpCodes.Ldloc, accL[v]);
                    EmitBlock(PushA, () => { il.Emit(OpCodes.Ldloc, locI); if (off != 0) { il.Emit(OpCodes.Ldc_I8, (long)off); il.Emit(OpCodes.Add); } });
                    il.Emit(OpCodes.Call, mAdd); il.Emit(OpCodes.Stloc, accL[v]);
                }
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lL);
                il.MarkLabel(lE);
            }
            EmitCombine(accL); il.Emit(OpCodes.Stloc, locRes);
            EmitScalarTail(PushA, () => il.Emit(OpCodes.Ldarg_1));
            il.Emit(OpCodes.Ldloc, locRes); il.Emit(OpCodes.Ret);

            // ---- LEAF2: the leaves (a, n2) and (b = a + n2·step, nr) in one loop. n2 is a multiple of 8, so the left
            //      leaf has no tail; the right one has at most one more block (nr - n2 < 16) and its tail. ----
            il.MarkLabel(LLEAF2);
            EmitControlVector();
            PushElemPtr(PushA, () => il.Emit(OpCodes.Ldloc, locN2), step); il.Emit(OpCodes.Stloc, locB);
            for (int v = 0; v < numAccVecs; v++)
            {
                int off = v * laneCount;
                EmitBlock(PushA, () => il.Emit(OpCodes.Ldc_I8, (long)off)); il.Emit(OpCodes.Stloc, accL[v]);
                EmitBlock(PushB, () => il.Emit(OpCodes.Ldc_I8, (long)off)); il.Emit(OpCodes.Stloc, accR[v]);
            }
            il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Stloc, locI);
            {
                var lL = il.DefineLabel(); var lE = il.DefineLabel();
                il.MarkLabel(lL);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locN2); il.Emit(OpCodes.Bge, lE);
                for (int v = 0; v < numAccVecs; v++)
                {
                    int off = v * laneCount;
                    il.Emit(OpCodes.Ldloc, accL[v]);
                    EmitBlock(PushA, () => { il.Emit(OpCodes.Ldloc, locI); if (off != 0) { il.Emit(OpCodes.Ldc_I8, (long)off); il.Emit(OpCodes.Add); } });
                    il.Emit(OpCodes.Call, mAdd); il.Emit(OpCodes.Stloc, accL[v]);
                    il.Emit(OpCodes.Ldloc, accR[v]);
                    EmitBlock(PushB, () => { il.Emit(OpCodes.Ldloc, locI); if (off != 0) { il.Emit(OpCodes.Ldc_I8, (long)off); il.Emit(OpCodes.Add); } });
                    il.Emit(OpCodes.Call, mAdd); il.Emit(OpCodes.Stloc, accR[v]);
                }
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lL);
                il.MarkLabel(lE);
            }
            // the right leaf's remaining whole blocks: lim = nr - nr % 8
            il.Emit(OpCodes.Ldloc, locNr); il.Emit(OpCodes.Ldloc, locNr); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locLim);
            {
                var rL = il.DefineLabel(); var rE = il.DefineLabel();
                il.MarkLabel(rL);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locLim); il.Emit(OpCodes.Bge, rE);
                for (int v = 0; v < numAccVecs; v++)
                {
                    int off = v * laneCount;
                    il.Emit(OpCodes.Ldloc, accR[v]);
                    EmitBlock(PushB, () => { il.Emit(OpCodes.Ldloc, locI); if (off != 0) { il.Emit(OpCodes.Ldc_I8, (long)off); il.Emit(OpCodes.Add); } });
                    il.Emit(OpCodes.Call, mAdd); il.Emit(OpCodes.Stloc, accR[v]);
                }
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, rL);
                il.MarkLabel(rE);
            }
            EmitCombine(accL); il.Emit(OpCodes.Stloc, locResL);
            EmitCombine(accR); il.Emit(OpCodes.Stloc, locRes);
            EmitScalarTail(PushB, () => il.Emit(OpCodes.Ldloc, locNr));
            // left + right, the recursion's own operand order
            il.Emit(OpCodes.Ldloc, locResL); il.Emit(OpCodes.Ldloc, locRes); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ret);

            return dm;
        }

        /// <summary>
        ///     The cached complex128 run fold <c>Vector128&lt;double&gt; fold(void* a, long nc)</c> for a contiguous
        ///     (<see cref="PairwiseRunKind.Contiguous"/>, 16-byte stride) or reversed (<see cref="PairwiseRunKind.Reversed"/>,
        ///     -16-byte stride) run of complexes, emitted on first use.
        /// </summary>
        /// <param name="kind">Contiguous or Reversed.</param>
        /// <returns>The DynamicMethod.</returns>
        private static DynamicMethod GetComplexPairwiseRunFold(PairwiseRunKind kind)
            => _pwRunFolds.GetOrAdd((NPTypeCode.Complex, kind), k => EmitComplexPairwiseRunFold(k.Kind, runtimeStride: false));

        /// <summary>
        ///     The rooted complex128 fold for ANY stride, <c>Vector128&lt;double&gt; fold(void* a, long nc, long strideBytes)</c>
        ///     — the same two-complexes-per-register leaves and two-leaf nodes as the contiguous kind, each register built
        ///     from two 128-bit loads (one per complex), so it reads only the complexes themselves.
        /// </summary>
        /// <returns>The DynamicMethod.</returns>
        private static DynamicMethod GetComplexPairwiseStridedFold()
            => _pwRunFolds.GetOrAdd((NPTypeCode.Complex, PairwiseRunKind.AnyStride),
                static _ => EmitComplexPairwiseRunFold(PairwiseRunKind.Contiguous, runtimeStride: true));

        /// <summary>
        ///     Emits NumPy's <c>CDOUBLE_pairwise_sum</c> over a contiguous or reversed run of complexes with TWO complexes
        ///     per 256-bit register: the four per-part accumulators v0..v3 (complexes 0..3 of each block) live in x0 = (v0,
        ///     v1) and x1 = (v2, v3), the leaf result is <c>(v0 + v1) + (v2 + v3)</c> = <c>(x0.lo + x0.hi) + (x1.lo +
        ///     x1.hi)</c>, the tail adds whole complexes, and a node whose halves (split at <c>(nc - nc%8)/2</c>) are both
        ///     leaves of at most 64 complexes computes them in one loop.
        /// </summary>
        /// <param name="kind">Contiguous (step +16 bytes) or Reversed (step -16 bytes); ignored when
        /// <paramref name="runtimeStride"/> is set.</param>
        /// <param name="runtimeStride">Emit the any-stride form instead: a third parameter carries the stride in bytes,
        /// and each register pair is two 128-bit loads (complex c, complex c+1) combined.</param>
        /// <returns>The emitted method.</returns>
        /// <exception cref="InvalidOperationException">A vector method is missing on this runtime.</exception>
        private static DynamicMethod EmitComplexPairwiseRunFold(PairwiseRunKind kind, bool runtimeStride)
        {
            Type pV = typeof(void).MakePointerType();
            Type d = typeof(double);
            Type v128 = VectorMethodCache.V(128, d);
            Type v256 = VectorMethodCache.V(256, d);
            var dm = runtimeStride
                ? new DynamicMethod("PwFoldC_Strided", v128, new[] { pV, typeof(long), typeof(long) }, typeof(ILKernelGenerator), skipVisibility: true)
                : new DynamicMethod($"PwFoldC_{kind}", v128, new[] { pV, typeof(long) }, typeof(ILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            MethodInfo mLoad128 = VectorMethodCache.LoadX86(128, d) ?? VectorMethodCache.Load(128, d);
            MethodInfo mAdd128 = VectorMethodCache.BinaryX86(128, "Add", d) ?? VectorMethodCache.Generic(128, "Add", d, paramCount: 2);
            MethodInfo mLoad256 = VectorMethodCache.LoadX86(256, d) ?? VectorMethodCache.Load(256, d);
            MethodInfo mAdd256 = VectorMethodCache.BinaryX86(256, "Add", d) ?? VectorMethodCache.Generic(256, "Add", d, paramCount: 2);
            MethodInfo mLower = VectorMethodCache.GetLower(256, d);
            MethodInfo mUpper = VectorMethodCache.GetUpper(256, d);
            MethodInfo mBcast = VectorMethodCache.CreateBroadcast(128, d);
            MethodInfo mPermute4x64 = !runtimeStride && kind == PairwiseRunKind.Reversed
                ? typeof(Avx2).GetMethod(nameof(Avx2.Permute4x64), new[] { typeof(Vector256<double>), typeof(byte) })
                  ?? throw new InvalidOperationException("Avx2.Permute4x64 missing")
                : null;
            MethodInfo mCreate256 = runtimeStride
                ? typeof(Vector256).GetMethod(nameof(Vector256.Create), new[] { typeof(Vector128<double>), typeof(Vector128<double>) })
                  ?? throw new InvalidOperationException("Vector256.Create(Vector128<double>, Vector128<double>) missing")
                : null;
            long stepBytes = 16L * (long)kind;

            var c = il.DeclareLocal(typeof(long));
            var lim = il.DeclareLocal(typeof(long));
            var left = il.DeclareLocal(typeof(long));
            var right = il.DeclareLocal(typeof(long));
            var b = il.DeclareLocal(pV);
            var res = il.DeclareLocal(v128);
            var resL = il.DeclareLocal(v128);
            var l0 = il.DeclareLocal(v256); var l1 = il.DeclareLocal(v256);
            var r0 = il.DeclareLocal(v256); var r1 = il.DeclareLocal(v256);

            void PushA() => il.Emit(OpCodes.Ldarg_0);
            void PushB() => il.Emit(OpCodes.Ldloc, b);

            // push (base + complexIndex · step), the step a constant or — any-stride form — the third argument (bytes)
            void PushPtr(Action pushBase, Action pushIdx)
            {
                pushBase(); pushIdx();
                if (runtimeStride) il.Emit(OpCodes.Ldarg_2);
                else il.Emit(OpCodes.Ldc_I8, stepBytes);
                il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            }

            // push (complex c, complex c+1) as one 256-bit register: a plain load, or — reversed — the load at complex
            // c+1 (the lower address) with its two halves swapped, or — any stride — two 128-bit loads combined.
            void EmitPair(Action pushBase, Action pushIdx)
            {
                if (runtimeStride)
                {
                    PushPtr(pushBase, pushIdx);
                    il.Emit(OpCodes.Call, mLoad128);
                    PushPtr(pushBase, () => { pushIdx(); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); });
                    il.Emit(OpCodes.Call, mLoad128);
                    il.Emit(OpCodes.Call, mCreate256);
                    return;
                }
                if (kind == PairwiseRunKind.Contiguous)
                {
                    PushPtr(pushBase, pushIdx);
                    il.Emit(OpCodes.Call, mLoad256);
                    return;
                }
                PushPtr(pushBase, () => { pushIdx(); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); });
                il.Emit(OpCodes.Call, mLoad256);
                il.Emit(OpCodes.Ldc_I4, 0b01_00_11_10);
                il.Emit(OpCodes.Call, mPermute4x64);
            }

            // push (x0.lo + x0.hi) + (x1.lo + x1.hi)
            void EmitCombine(LocalBuilder x0, LocalBuilder x1)
            {
                il.Emit(OpCodes.Ldloc, x0); il.Emit(OpCodes.Call, mLower);
                il.Emit(OpCodes.Ldloc, x0); il.Emit(OpCodes.Call, mUpper);
                il.Emit(OpCodes.Call, mAdd128);
                il.Emit(OpCodes.Ldloc, x1); il.Emit(OpCodes.Call, mLower);
                il.Emit(OpCodes.Ldloc, x1); il.Emit(OpCodes.Call, mUpper);
                il.Emit(OpCodes.Call, mAdd128);
                il.Emit(OpCodes.Call, mAdd128);
            }

            // for (; c < end; c++) res += complex c of the run at pushBase
            void EmitTail(Action pushBase, Action pushEnd)
            {
                var tL = il.DefineLabel(); var tE = il.DefineLabel();
                il.MarkLabel(tL);
                il.Emit(OpCodes.Ldloc, c); pushEnd(); il.Emit(OpCodes.Bge, tE);
                il.Emit(OpCodes.Ldloc, res);
                PushPtr(pushBase, () => il.Emit(OpCodes.Ldloc, c)); il.Emit(OpCodes.Call, mLoad128);
                il.Emit(OpCodes.Call, mAdd128); il.Emit(OpCodes.Stloc, res);
                il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, c);
                il.Emit(OpCodes.Br, tL);
                il.MarkLabel(tE);
            }

            // acc = acc + pair(base, c + off)
            void EmitAccumulate(LocalBuilder acc, Action pushBase, long off)
            {
                il.Emit(OpCodes.Ldloc, acc);
                EmitPair(pushBase, () => { il.Emit(OpCodes.Ldloc, c); if (off != 0) { il.Emit(OpCodes.Ldc_I8, off); il.Emit(OpCodes.Add); } });
                il.Emit(OpCodes.Call, mAdd256); il.Emit(OpCodes.Stloc, acc);
            }

            var LBASE = il.DefineLabel(); var LLEAF = il.DefineLabel(); var LLEAF2 = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Blt, LBASE);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 64L); il.Emit(OpCodes.Ble, LLEAF);
            // left = (nc - nc % 8) / 2; right = nc - left
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 8L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Div); il.Emit(OpCodes.Stloc, left);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldloc, left); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, right);
            il.Emit(OpCodes.Ldloc, right); il.Emit(OpCodes.Ldc_I8, 64L); il.Emit(OpCodes.Ble, LLEAF2);
            // REC (the any-stride form passes its stride down)
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldloc, left); if (runtimeStride) il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Call, dm);
            PushPtr(PushA, () => il.Emit(OpCodes.Ldloc, left)); il.Emit(OpCodes.Ldloc, right); if (runtimeStride) il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Call, dm);
            il.Emit(OpCodes.Call, mAdd128); il.Emit(OpCodes.Ret);

            // BASE nc < 4: res = (-0, -0); res += complex c for c in [0, nc)
            il.MarkLabel(LBASE);
            il.Emit(OpCodes.Ldc_R8, -0.0); il.Emit(OpCodes.Call, mBcast); il.Emit(OpCodes.Stloc, res);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, c);
            EmitTail(PushA, () => il.Emit(OpCodes.Ldarg_1));
            il.Emit(OpCodes.Ldloc, res); il.Emit(OpCodes.Ret);

            // LEAF nc ≤ 64
            il.MarkLabel(LLEAF);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, c);
            EmitPair(PushA, () => il.Emit(OpCodes.Ldc_I8, 0L)); il.Emit(OpCodes.Stloc, l0);
            EmitPair(PushA, () => il.Emit(OpCodes.Ldc_I8, 2L)); il.Emit(OpCodes.Stloc, l1);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, lim);
            il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Stloc, c);
            {
                var lL = il.DefineLabel(); var lE = il.DefineLabel();
                il.MarkLabel(lL);
                il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldloc, lim); il.Emit(OpCodes.Bge, lE);
                EmitAccumulate(l0, PushA, 0); EmitAccumulate(l1, PushA, 2);
                il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, c);
                il.Emit(OpCodes.Br, lL);
                il.MarkLabel(lE);
            }
            EmitCombine(l0, l1); il.Emit(OpCodes.Stloc, res);
            EmitTail(PushA, () => il.Emit(OpCodes.Ldarg_1));
            il.Emit(OpCodes.Ldloc, res); il.Emit(OpCodes.Ret);

            // LEAF2: leaves (a, left) and (b = a + left·step, right); left is a multiple of 4, so it has no tail, and the
            // right one has at most one more block of 4 (right - left = nc % 8 < 8) and its tail.
            il.MarkLabel(LLEAF2);
            PushPtr(PushA, () => il.Emit(OpCodes.Ldloc, left)); il.Emit(OpCodes.Stloc, b);
            EmitPair(PushA, () => il.Emit(OpCodes.Ldc_I8, 0L)); il.Emit(OpCodes.Stloc, l0);
            EmitPair(PushA, () => il.Emit(OpCodes.Ldc_I8, 2L)); il.Emit(OpCodes.Stloc, l1);
            EmitPair(PushB, () => il.Emit(OpCodes.Ldc_I8, 0L)); il.Emit(OpCodes.Stloc, r0);
            EmitPair(PushB, () => il.Emit(OpCodes.Ldc_I8, 2L)); il.Emit(OpCodes.Stloc, r1);
            il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Stloc, c);
            {
                var lL = il.DefineLabel(); var lE = il.DefineLabel();
                il.MarkLabel(lL);
                il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldloc, left); il.Emit(OpCodes.Bge, lE);
                EmitAccumulate(l0, PushA, 0); EmitAccumulate(l1, PushA, 2);
                EmitAccumulate(r0, PushB, 0); EmitAccumulate(r1, PushB, 2);
                il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, c);
                il.Emit(OpCodes.Br, lL);
                il.MarkLabel(lE);
            }
            il.Emit(OpCodes.Ldloc, right); il.Emit(OpCodes.Ldloc, right); il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, lim);
            {
                var rL = il.DefineLabel(); var rE = il.DefineLabel();
                il.MarkLabel(rL);
                il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldloc, lim); il.Emit(OpCodes.Bge, rE);
                EmitAccumulate(r0, PushB, 0); EmitAccumulate(r1, PushB, 2);
                il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, c);
                il.Emit(OpCodes.Br, rL);
                il.MarkLabel(rE);
            }
            EmitCombine(l0, l1); il.Emit(OpCodes.Stloc, resL);
            EmitCombine(r0, r1); il.Emit(OpCodes.Stloc, res);
            EmitTail(PushB, () => il.Emit(OpCodes.Ldloc, right));
            il.Emit(OpCodes.Ldloc, resL); il.Emit(OpCodes.Ldloc, res); il.Emit(OpCodes.Call, mAdd128); il.Emit(OpCodes.Ret);

            return dm;
        }
    }
}
