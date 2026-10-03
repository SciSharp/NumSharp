using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

// =============================================================================
// DirectILKernelGenerator.Clip.InnerLoop — clip bodies for the NDIter-driven route
// =============================================================================
//
// np.clip has two execution routes (Default.ClipNDArray.cs):
//   * the flat route, the whole-array kernel in DirectILKernelGenerator.Clip.cs, for
//     operands that are C-contiguous (after a transpose into the result's memory order);
//   * the iterator route, NumPy's own structure: an NDIter over (a, lo, hi, out[, where])
//     drives a Tier-3B inner-loop kernel, which serves broadcasting, every layout, out=
//     casting, where= masks and overlap (COPY_IF_OVERLAP).
//
// This file emits the per-element and per-vector BODIES of that inner-loop kernel. They
// reuse the flat kernel's own clamp emitters (EmitScalarClamp, EmitVectorMinOrMax) so both
// routes compute bit-identical results for the same semantics.
//
// Operand order on the evaluation stack follows the Tier-3B contract
// (DirectILKernelGenerator.InnerLoop.cs): input 0 deepest, the last input on top.
//   both bounds : [x, lo, hi] -> [r]   (NumPy's um.clip)
//   lower only  : [x, lo]     -> [r]   (um.maximum, where np.clip routes it)
//   upper only  : [x, hi]     -> [r]   (um.minimum)
//   no bound    : [x]         -> [x]   (um.positive, a copy)
//
// Two semantics exist for a two-sided float32/float64 clip, and NumPy picks between them
// per inner-loop call (clip.cpp _npy_clip): when both bound strides are 0 it runs the
// CONST path (x < lo ? lo : x, then x > hi ? hi : x; a NaN bound fills the call with that
// NaN), otherwise the general _NPY_MIN(_NPY_MAX(x, lo), hi). They differ on signed-zero
// ties and on which NaN a NaN bound produces. The caller decides which one NumPy would run
// (Default.ClipNDArray.ConstLoop.cs) and selects the matching body here.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Whether the clip inner-loop body has a vector form for <paramref name="dtype"/>. The
        /// vector body exists exactly for the dtypes the flat kernel vectorizes (hardware min/max
        /// lanes); Half, Complex, Decimal, Char and Boolean run the scalar body only.
        /// </summary>
        /// <param name="dtype">The loop dtype every operand carries.</param>
        /// <returns>True when <see cref="EmitClipInnerVector"/> may be used.</returns>
        internal static bool CanEmitClipInnerVector(NPTypeCode dtype)
            => CanUseSimd(dtype) && SupportsVectorMinMax(dtype);

        /// <summary>
        /// Emits the per-element clip body for the iterator route. Stack in: the inputs
        /// (<c>[x, lo, hi]</c>, <c>[x, lo]</c>, <c>[x, hi]</c> or <c>[x]</c> by
        /// <paramref name="mode"/>); stack out: one value of <paramref name="dtype"/>.
        /// </summary>
        /// <param name="il">The IL stream of the inner-loop kernel being generated.</param>
        /// <param name="dtype">The loop dtype; every operand has it (the caller casts first).</param>
        /// <param name="mode">Which bounds are present. <c>null</c> means neither: the body is the
        /// identity (NumPy's <c>positive</c>, a copy).</param>
        /// <param name="constSelect">True to emit NumPy's const fast path (float32/float64 only,
        /// two bounds only); false for the general <c>MIN(MAX(x, lo), hi)</c>.</param>
        /// <exception cref="ArgumentException"><paramref name="constSelect"/> is requested for a
        /// dtype or mode NumPy never runs the const path for.</exception>
        /// <remarks>
        /// The general body stores the upper bound in a local so that the clamp emitters, which
        /// consume two values, see <c>[x, lo]</c> first and <c>[t, hi]</c> second. The const body
        /// calls <see cref="FloatClipConst"/> / <see cref="DoubleClipConst"/>, which carry the
        /// NaN-bound rule along with the compare-select, because unlike the flat kernel's const
        /// path no dispatcher short-circuit has removed NaN bounds before an iterator-route kernel
        /// runs (a broadcast bound array can hold NaN in some rows only).
        /// </remarks>
        internal static void EmitClipInnerScalar(ILGenerator il, NPTypeCode dtype, ClipMode? mode, bool constSelect)
        {
            if (mode is null)
                return;   // identity: [x] -> [x]

            if (constSelect)
            {
                RequireConstSelectable(dtype, mode.Value);
                il.EmitCall(OpCodes.Call, GetHelper(dtype == NPTypeCode.Single ? nameof(FloatClipConst) : nameof(DoubleClipConst)), null);
                return;
            }

            switch (mode.Value)
            {
                case ClipMode.MinOnly:
                    EmitScalarClamp(il, dtype, isMax: true);    // maximum(x, lo)
                    return;
                case ClipMode.MaxOnly:
                    EmitScalarClamp(il, dtype, isMax: false);   // minimum(x, hi)
                    return;
                default:
                {
                    var hi = il.DeclareLocal(GetClrType(dtype));
                    il.Emit(OpCodes.Stloc, hi);                 // [x, lo]
                    EmitScalarClamp(il, dtype, isMax: true);    // [t]  t = MAX(x, lo)
                    il.Emit(OpCodes.Ldloc, hi);                 // [t, hi]
                    EmitScalarClamp(il, dtype, isMax: false);   // [r]  r = MIN(t, hi)
                    return;
                }
            }
        }

        /// <summary>
        /// Emits the per-vector clip body for the iterator route, the vector twin of
        /// <see cref="EmitClipInnerScalar"/>: same stack shape with <c>Vector{W}&lt;T&gt;</c> values,
        /// same results lane for lane.
        /// </summary>
        /// <param name="il">The IL stream of the inner-loop kernel being generated.</param>
        /// <param name="dtype">The loop dtype; <see cref="CanEmitClipInnerVector"/> must hold.</param>
        /// <param name="mode">Which bounds are present; <c>null</c> for the identity.</param>
        /// <param name="constSelect">True for NumPy's const fast path (float32/float64, two bounds).</param>
        /// <exception cref="ArgumentException"><paramref name="constSelect"/> is requested for a
        /// dtype or mode NumPy never runs the const path for.</exception>
        /// <remarks>
        /// The const body is the two compare-selects of <c>_npy_clip_const_minmax_</c> plus the
        /// NaN-bound rule, so a NaN lower bound wins over a NaN upper bound, as in NumPy's
        /// <c>npy_isnan(min_val) ? min_val : max_val</c>; a NaN element fails both compares and passes
        /// through with its own bits, as in NumPy. On x86 the compare-selects are the ordered
        /// MAX/MIN instructions (<see cref="TryGetX86OrderedMaxMin"/>) and the NaN-bound rule one
        /// blend against a fill computed from the bounds alone (hoistable when the bounds are
        /// broadcast); elsewhere they are four explicit selects — <c>lo</c>/<c>hi</c> where the
        /// compares hold, then <c>hi</c> where hi is NaN, then <c>lo</c> where lo is NaN. Both forms
        /// give the same lanes.
        /// </remarks>
        internal static void EmitClipInnerVector(ILGenerator il, NPTypeCode dtype, ClipMode? mode, bool constSelect)
        {
            if (mode is null)
                return;   // identity

            var clrType = GetClrType(dtype);
            var vecType = VectorMethodCache.V(VectorBits, clrType);

            if (constSelect && TryGetX86OrderedMaxMin(clrType, out var orderedMax, out var orderedMin))
            {
                RequireConstSelectable(dtype, mode.Value);
                var vHi = il.DeclareLocal(vecType);
                var vLo = il.DeclareLocal(vecType);
                var vX = il.DeclareLocal(vecType);
                var vOk = il.DeclareLocal(vecType);
                il.Emit(OpCodes.Stloc, vHi);
                il.Emit(OpCodes.Stloc, vLo);
                il.Emit(OpCodes.Stloc, vX);                                                                  // []

                // x = Max(lo, x) ; x = Min(hi, x) — x86's ordered MAX/MIN ARE the const path's two
                // compare-selects (TryGetX86OrderedMaxMin), one instruction per bound.
                il.Emit(OpCodes.Ldloc, vLo);
                il.Emit(OpCodes.Ldloc, vX);
                il.EmitCall(OpCodes.Call, orderedMax, null);
                il.Emit(OpCodes.Stloc, vX);
                il.Emit(OpCodes.Ldloc, vHi);
                il.Emit(OpCodes.Ldloc, vX);
                il.EmitCall(OpCodes.Call, orderedMin, null);
                il.Emit(OpCodes.Stloc, vX);

                // The NaN-bound rule as ONE blend: ok = (lo == lo) & (hi == hi) marks the lanes with no
                // NaN bound; fill = lo == lo ? hi : lo is the NaN that replaces the element otherwise
                // (the lower bound's when both are NaN). ok and fill depend only on the bounds, so when
                // the bounds are hoisted broadcasts (the const pattern's S inputs) the JIT hoists them
                // out of the loop too and the per-vector cost is max + min + one blend.
                il.Emit(OpCodes.Ldloc, vLo);
                il.Emit(OpCodes.Ldloc, vLo);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clrType), null);              // [lo==lo]
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Ldloc, vHi);
                il.Emit(OpCodes.Ldloc, vHi);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clrType), null);              // [lo==lo, lo==lo, hi==hi]
                il.EmitCall(OpCodes.Call, VectorMethodCache.Operator(VectorBits, clrType, "op_BitwiseAnd"), null);
                il.Emit(OpCodes.Stloc, vOk);                                                                 // [lo==lo]
                il.Emit(OpCodes.Ldloc, vHi);
                il.Emit(OpCodes.Ldloc, vLo);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);   // [fill]
                var vFill = il.DeclareLocal(vecType);
                il.Emit(OpCodes.Stloc, vFill);                                                               // []
                il.Emit(OpCodes.Ldloc, vOk);
                il.Emit(OpCodes.Ldloc, vX);
                il.Emit(OpCodes.Ldloc, vFill);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);   // [r]
                return;
            }

            if (constSelect)
            {
                RequireConstSelectable(dtype, mode.Value);
                var vHi = il.DeclareLocal(vecType);
                var vLo = il.DeclareLocal(vecType);
                var vX = il.DeclareLocal(vecType);
                il.Emit(OpCodes.Stloc, vHi);
                il.Emit(OpCodes.Stloc, vLo);
                il.Emit(OpCodes.Stloc, vX);                                                                  // []

                // x = x < lo ? lo : x
                il.Emit(OpCodes.Ldloc, vX);
                il.Emit(OpCodes.Ldloc, vLo);
                il.EmitCall(OpCodes.Call, VectorMethodCache.LessThan(VectorBits, clrType), null);
                il.Emit(OpCodes.Ldloc, vLo);
                il.Emit(OpCodes.Ldloc, vX);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);
                il.Emit(OpCodes.Stloc, vX);

                // x = x > hi ? hi : x
                il.Emit(OpCodes.Ldloc, vX);
                il.Emit(OpCodes.Ldloc, vHi);
                il.EmitCall(OpCodes.Call, VectorMethodCache.GreaterThan(VectorBits, clrType), null);
                il.Emit(OpCodes.Ldloc, vHi);
                il.Emit(OpCodes.Ldloc, vX);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);
                il.Emit(OpCodes.Stloc, vX);

                // x = hi == hi ? x : hi   (a NaN upper bound fills its lanes with itself)
                il.Emit(OpCodes.Ldloc, vHi);
                il.Emit(OpCodes.Ldloc, vHi);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clrType), null);
                il.Emit(OpCodes.Ldloc, vX);
                il.Emit(OpCodes.Ldloc, vHi);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);
                il.Emit(OpCodes.Stloc, vX);

                // x = lo == lo ? x : lo   (a NaN lower bound wins over a NaN upper bound)
                il.Emit(OpCodes.Ldloc, vLo);
                il.Emit(OpCodes.Ldloc, vLo);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clrType), null);
                il.Emit(OpCodes.Ldloc, vX);
                il.Emit(OpCodes.Ldloc, vLo);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null);   // [r]
                return;
            }

            switch (mode.Value)
            {
                case ClipMode.MinOnly:
                    EmitVectorMinOrMax(il, isMax: true, dtype, propagateNaN: true);
                    return;
                case ClipMode.MaxOnly:
                    EmitVectorMinOrMax(il, isMax: false, dtype, propagateNaN: true);
                    return;
                default:
                {
                    var vHi = il.DeclareLocal(vecType);
                    il.Emit(OpCodes.Stloc, vHi);                                        // [x, lo]
                    EmitVectorMinOrMax(il, isMax: true, dtype, propagateNaN: true);     // [t]
                    il.Emit(OpCodes.Ldloc, vHi);                                        // [t, hi]
                    EmitVectorMinOrMax(il, isMax: false, dtype, propagateNaN: true);    // [r]
                    return;
                }
            }
        }

        /// <summary>
        /// Guards the const body: NumPy runs it only for floating_point_tag dtypes (float32, float64
        /// and long double; NumSharp has the first two) and only in the two-bound clip loop.
        /// </summary>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="mode">The clip mode.</param>
        /// <exception cref="ArgumentException">Any other dtype or mode: a caller bug, since the
        /// integer and half/complex loops cannot tell the two forms apart and the one-sided routes
        /// are minimum/maximum, which have no const path.</exception>
        private static void RequireConstSelectable(NPTypeCode dtype, ClipMode mode)
        {
            if ((dtype != NPTypeCode.Single && dtype != NPTypeCode.Double) || mode != ClipMode.BothBounds)
                throw new ArgumentException($"clip's const path exists only for a two-bound float32/float64 loop (got {dtype}, {mode}).");
        }

        /// <summary>
        /// One element of NumPy's float32 clip CONST path (clip.cpp <c>_npy_clip_const_minmax_</c>,
        /// floating_point_tag), including its NaN-bound branch: a NaN bound replaces the element with
        /// that NaN — the lower bound's when both are NaN — whatever the element is.
        /// </summary>
        /// <param name="x">The element.</param>
        /// <param name="lo">The lower bound of this element's inner-loop call.</param>
        /// <param name="hi">The upper bound of this element's inner-loop call.</param>
        /// <returns>The clipped element; a tie with a bound keeps the element (its signed zero).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatClipConst(float x, float lo, float hi)
        {
            if (float.IsNaN(lo)) return lo;
            if (float.IsNaN(hi)) return hi;
            if (x < lo) x = lo;
            if (x > hi) x = hi;
            return x;
        }

        /// <summary>The float64 twin of <see cref="FloatClipConst"/>.</summary>
        /// <param name="x">The element.</param>
        /// <param name="lo">The lower bound of this element's inner-loop call.</param>
        /// <param name="hi">The upper bound of this element's inner-loop call.</param>
        /// <returns>The clipped element; a tie with a bound keeps the element (its signed zero).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleClipConst(double x, double lo, double hi)
        {
            if (double.IsNaN(lo)) return lo;
            if (double.IsNaN(hi)) return hi;
            if (x < lo) x = lo;
            if (x > hi) x = hi;
            return x;
        }
    }
}
