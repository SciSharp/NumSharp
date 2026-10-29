using System;
using System.Collections.Generic;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// ILKernelGenerator.Polynomial.Emitter.cs — the NumPy-typed IL emitter for the step trees
// =============================================================================
//
// VALUES
// ------
// A value is either SHARED — one scalar local in its own dtype, computed once for all points of a
// block (a broadcast coefficient, a weak constant, a 0-d x, or math on those only) — or PER-POINT:
// one local per interleaved chain, a scalar in scalar chains or a W-lane vector in vector chains.
// Shared math therefore costs nothing per point: NumPy's peeled coefficient-dtype steps and its
// Python-literal constants run once per block. A shared value is spread (converted + broadcast) only
// where it meets a per-point one.
//
// LANES
// -----
// In vector chains every per-point value carries the SAME lane count W (the loop dtype's lanes at the
// host width): a float32 x beside a float64 loop is a Vector128<float> of 4 lanes that widens to ONE
// Vector256<double> — lane-matched, no shuffles. float16 values live as 8 float32 lanes kept ON the
// f16 grid (every op narrows and widens back through the house HalfNarrow8V/HalfWiden8V, NumPy's
// HALF loop one op at a time), with the wheel's NaN priority re-imposed by an explicit blend.
//
// COMPLEX MULTIPLY: THREE NUMPY SEMANTICS
// ---------------------------------------
// NumPy 2.4.2 (win-amd64) multiplies complex128 three ways (PolyComplexProduct), picked by how the
// operands reach the loop, never by their values:
//   * Simd — the ufunc's vector body simd_cmul (loops_arithm_fp.dispatch.c.src):
//     re = fma(ar, br, -(ai*bi)), im = fma(ar, bi, ai*br). Every ARRAY op of any size, on the trivial
//     loop or on NpyIter, except the case below. The house EmitScalarOperation.
//   * Naive — scalarmath (np.complex128 * np.complex128): the four-product formula, no FMA.
//   * LoopScalar — the same ufunc's scalar fallback, which it jumps to when the OUTPUT stride is 0.
//     MSVC compiled that fallback in the AVX2+FMA dispatch target and CONTRACTED it:
//     re = fma(ar, br, -(ai*bi)) (Simd's), im = fma(ai, br, ar*bi) (the other product fused).
//     An output stride of 0 happens when NpyIter iterates a SINGLE element: nditer_constr.c gives
//     every length-1 axis stride 0. NpyIter (not the trivial loop) runs when the non-0-d operands
//     differ in ndim (try_trivial_single_output_loop returns -2). For {p}val that is an N-D series
//     at a per-point x whose result has one element (PolyUnitBroadcast, set by NDPolyEval).
// When the result is 0-d every value is a NumPy scalar, and an op is a ufunc only when one operand is
// the raw 0-d x array — so the emitter, in "scalar math" mode, emits the naive product except next to
// the x leaf. Measured against NumPy: chebval with a 0-d complex x differs from BOTH the array-x and
// the Python-scalar-x results; this rule reproduces all three. In PolyUnitBroadcast mode every value
// carries NumPy's ndim (PolyValue.Nd, representative values — only comparisons matter) and a product
// of two ARRAY values whose ndims differ is the LoopScalar one. Measured: 6,000 random pairs, every
// basis at 8 single-element shape configurations x 7 coefficient counts (the review plan,
// docs/plans/numpy-polynomial-review.md).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>
        ///     One float16 arithmetic op on 8 f16 values held as float32 lanes ON the f16 grid: the float32
        ///     op, the RTNE narrow + exact widen back (NumPy's <c>npy_float_to_half(float(a) OP float(b))</c>),
        ///     and the wheel's NaN priority re-imposed explicitly — add/multiply keep the SECOND operand's
        ///     quieted NaN, subtract/divide the FIRST's. Bit-identical per lane to the house scalar
        ///     <c>HalfArithBits</c>, which is what lets a block switch between vector and scalar chains.
        /// </summary>
        /// <param name="fa">Left operand lanes (exact f16 values as float32).</param>
        /// <param name="fb">Right operand lanes.</param>
        /// <param name="op">The <see cref="BinaryOp"/> as an int (Add, Subtract, Multiply or Divide).</param>
        /// <returns>The 8 results, as float32 lanes on the f16 grid.</returns>
        /// <remarks>
        ///     The NaN blend is in the float32 domain: a widened f16 NaN has its 10-bit payload in float bits
        ///     13..22, so <c>| 0x00400000</c> (float quiet bit) is exactly the half quiet bit <c>0x0200</c>
        ///     after the narrow — the same <c>quiet(x) = x | 0x0200</c> the house uses. It must be explicit:
        ///     RyuJIT may swap the operands of a commutative intrinsic, so x86's "first operand's NaN" rule
        ///     cannot be relied on. Requires AVX2 (callers gate on it).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<float> PolyHalfArith8(Vector256<float> fa, Vector256<float> fb, int op)
        {
            Vector256<float> r = (BinaryOp)op switch
            {
                BinaryOp.Add => Avx.Add(fa, fb),
                BinaryOp.Subtract => Avx.Subtract(fa, fb),
                BinaryOp.Multiply => Avx.Multiply(fa, fb),
                _ => Avx.Divide(fa, fb),
            };
            r = DirectILKernelGenerator.HalfWiden8V(DirectILKernelGenerator.HalfNarrow8V(r));
            var nanA = Avx.CompareUnordered(fa, fa);
            var nanB = Avx.CompareUnordered(fb, fb);
            var quiet = Vector256.Create(0x00400000).AsSingle();
            var qa = Avx.Or(fa, quiet);
            var qb = Avx.Or(fb, quiet);
            if ((BinaryOp)op is BinaryOp.Add or BinaryOp.Multiply)
            {
                r = Avx.BlendVariable(r, qa, nanA);
                r = Avx.BlendVariable(r, qb, nanB);   // in2 wins
            }
            else
            {
                r = Avx.BlendVariable(r, qb, nanB);
                r = Avx.BlendVariable(r, qa, nanA);   // in1 wins
            }
            return r;
        }

        /// <summary>
        ///     NumPy's SCALAR complex product (<c>scalarmath</c>): <c>(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)</c>,
        ///     each product rounded before the add — NOT the fused array multiply. Only the polynomial kernels'
        ///     0-d ("all NumPy scalars") mode calls it.
        /// </summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The naive product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyNaiveComplexMultiply(Complex a, Complex b)
            => new Complex(a.Real * b.Real - a.Imaginary * b.Imaginary, a.Real * b.Imaginary + a.Imaginary * b.Real);

        /// <summary>
        ///     NumPy's ufunc FALLBACK complex product — <c>CDOUBLE_multiply</c>'s <c>loop_scalar</c>
        ///     (<c>loops_arithm_fp.dispatch.c.src</c>), written <c>a_r*b_r - a_i*b_i</c>, <c>a_r*b_i + a_i*b_r</c> and
        ///     contracted by MSVC in the AVX2+FMA target: <c>re = fma(a.re, b.re, -(a.im*b.im))</c>,
        ///     <c>im = fma(a.im, b.re, a.re*b.im)</c>. The real part equals <c>simd_cmul</c>'s; the imaginary part
        ///     fuses the OTHER product, so it can differ from <c>simd_cmul</c>'s in the last bit, and the next
        ///     recurrence step carries that difference into both parts.
        /// </summary>
        /// <param name="a">Left operand (NumPy's first input).</param><param name="b">Right operand.</param>
        /// <returns>The product, as NumPy's fallback loop computes it.</returns>
        /// <remarks>
        ///     NumPy runs this loop only when the output stride is 0, i.e. NpyIter's single-element iteration of
        ///     two array operands that differ in ndim. Only <see cref="PolyUnitBroadcast"/> kernels call it.
        ///     The forms are MEASURED, not derived from the C source (which has no FMA): over 6,000 random pairs
        ///     NumPy's real part always matched <c>fma(a.re, b.re, -round(a.im*b.im))</c> and its imaginary part
        ///     always <c>fma(a.im, b.re, round(a.re*b.im))</c>, in exact rational arithmetic.
        ///     <see cref="Math.FusedMultiplyAdd(double,double,double)"/> is IEEE fma on every host (one
        ///     <c>vfmadd</c> with FMA3, the CRT's exact <c>fma</c> without), so the finite bits are host-independent;
        ///     which NaN payload survives is not (the oracle tokenizes NaN for binary ops).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyLoopScalarComplexMultiply(Complex a, Complex b)
            => new Complex(Math.FusedMultiplyAdd(a.Real, b.Real, -(a.Imaginary * b.Imaginary)),
                           Math.FusedMultiplyAdd(a.Imaginary, b.Real, a.Real * b.Imaginary));

        /// <summary><see cref="PolyHalfArith8"/>, called by the float16 lane kind's Bin.</summary>
        internal static readonly MethodInfo s_polyHalfArith8 = typeof(ILKernelGenerator).GetMethod(nameof(PolyHalfArith8),
            BindingFlags.NonPublic | BindingFlags.Static) ?? throw new MissingMethodException(nameof(PolyHalfArith8));

        /// <summary><see cref="PolyNaiveComplexMultiply"/>, called in scalar-math (0-d result) mode.</summary>
        internal static readonly MethodInfo s_polyNaiveComplexMultiply = typeof(ILKernelGenerator).GetMethod(nameof(PolyNaiveComplexMultiply),
            BindingFlags.NonPublic | BindingFlags.Static) ?? throw new MissingMethodException(nameof(PolyNaiveComplexMultiply));

        /// <summary><see cref="PolyLoopScalarComplexMultiply"/>, called by <see cref="PolyUnitBroadcast"/> kernels for a
        ///     product of two array values whose NumPy ndims differ.</summary>
        internal static readonly MethodInfo s_polyLoopScalarComplexMultiply = typeof(ILKernelGenerator).GetMethod(nameof(PolyLoopScalarComplexMultiply),
            BindingFlags.NonPublic | BindingFlags.Static) ?? throw new MissingMethodException(nameof(PolyLoopScalarComplexMultiply));
    }

    /// <summary>
    ///     Which of NumPy's three complex128 products one multiply reproduces (see the file header). Only a
    ///     SCALAR <see cref="NPTypeCode.Complex"/> multiply reads it. The kernel never emits vector chains where
    ///     NumPy takes <see cref="Naive"/> or <see cref="LoopScalar"/> (both are one-point kernels), so a vector
    ///     kind ignores it and always emits <c>simd_cmul</c>.
    /// </summary>
    internal enum PolyComplexProduct : byte
    {
        /// <summary>The ufunc vector body <c>simd_cmul</c>: <c>re = fma(ar, br, -(ai*bi))</c>,
        ///     <c>im = fma(ar, bi, ai*br)</c>. Every array op except the <see cref="LoopScalar"/> case.</summary>
        Simd,
        /// <summary><c>scalarmath</c> (<c>np.complex128 * np.complex128</c>): each product rounded, no FMA.</summary>
        Naive,
        /// <summary>The ufunc's fallback loop as MSVC contracted it: <c>re = fma(ar, br, -(ai*bi))</c>,
        ///     <c>im = fma(ai, br, ar*bi)</c>. NpyIter's single-element iteration (<see cref="PolyUnitBroadcast"/>).</summary>
        LoopScalar,
    }

    /// <summary>
    ///     How values of one dtype live in IL: a scalar of the CLR type, or a <c>Vector{bits}&lt;T&gt;</c>.
    ///     Scalar ops go through the house <c>EmitScalarOperation</c> (NumPy-exact per dtype: fused complex
    ///     multiply, float16 widen-compute-narrow with its NaN pin, decimal operators, int wrap-around);
    ///     vector ops through the vector type's own operator at THIS kind's width.
    /// </summary>
    internal class PolyValueKind
    {
        /// <summary>Vector (true) or scalar (false) values.</summary>
        public readonly bool Vec;
        /// <summary>The NumPy dtype.</summary>
        public readonly NPTypeCode T;
        /// <summary>Its CLR element type.</summary>
        public readonly Type Clr;
        /// <summary>Vector width in bits (0 for scalars).</summary>
        public readonly int Bits;
        /// <summary>The IL local type holding one value.</summary>
        public readonly Type LocalType;

        /// <summary>Creates a kind.</summary>
        /// <param name="vec">Vector or scalar.</param>
        /// <param name="t">The dtype.</param>
        /// <param name="bits">Vector width (128/256/512); ignored for scalars.</param>
        /// <exception cref="NotSupportedException">A vector width with no Vector type.</exception>
        public PolyValueKind(bool vec, NPTypeCode t, int bits = 0)
        {
            Vec = vec; T = t; Clr = DirectILKernelGenerator.GetClrType(t); Bits = vec ? bits : 0;
            LocalType = !vec ? Clr : bits switch
            {
                128 => typeof(Vector128<>).MakeGenericType(Clr),
                256 => typeof(Vector256<>).MakeGenericType(Clr),
                512 => typeof(Vector512<>).MakeGenericType(Clr),
                _ => throw new NotSupportedException($"no {bits}-bit vector of {t}"),
            };
        }

        /// <summary>[a, b] → [a op b].</summary>
        /// <param name="il">The generator.</param>
        /// <param name="op">The op.</param>
        /// <param name="product">Which of NumPy's complex products a scalar Complex multiply reproduces
        ///     (ignored by every other op, and by vector kinds — see <see cref="PolyComplexProduct"/>).</param>
        public virtual void Bin(ILGenerator il, BinaryOp op, PolyComplexProduct product = PolyComplexProduct.Simd)
        {
            if (Vec)
                PolyLanes.EmitVecOperator(il, op, LocalType);
            else if (product != PolyComplexProduct.Simd && op == BinaryOp.Multiply && T == NPTypeCode.Complex)
                il.EmitCall(OpCodes.Call, product == PolyComplexProduct.Naive
                    ? ILKernelGenerator.s_polyNaiveComplexMultiply
                    : ILKernelGenerator.s_polyLoopScalarComplexMultiply, null);
            else
                DirectILKernelGenerator.EmitScalarOperation(il, op, T);
        }

        /// <summary>[address] → [value]: one element, or one contiguous vector.</summary>
        /// <param name="il">The generator.</param>
        public virtual void Load(ILGenerator il)
        {
            if (Vec) il.EmitCall(OpCodes.Call, VectorMethodCache.Load(Bits, Clr), null);
            else DirectILKernelGenerator.EmitLoadIndirect(il, T);
        }

        /// <summary>[scalar of T] → [vector of this kind] (every lane the same value).</summary>
        /// <param name="il">The generator.</param>
        public virtual void BroadcastFromScalar(ILGenerator il) =>
            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(Bits, Clr), null);

        /// <summary>[value, address] → []: stores one element (scalar) or one vector.</summary>
        /// <param name="il">The generator.</param>
        public virtual void StoreValueFirst(ILGenerator il)
        {
            if (Vec) { il.EmitCall(OpCodes.Call, VectorMethodCache.Store(Bits, Clr), null); return; }
            // stind wants [address, value]: swap through temps.
            var addr = il.DeclareLocal(typeof(byte*)); il.Emit(OpCodes.Stloc, addr);
            var val = il.DeclareLocal(LocalType); il.Emit(OpCodes.Stloc, val);
            il.Emit(OpCodes.Ldloc, addr); il.Emit(OpCodes.Ldloc, val);
            DirectILKernelGenerator.EmitStoreIndirect(il, T);
        }

        /// <summary>Lanes per value (1 for scalars).</summary>
        public virtual int Lanes => Vec ? Bits / 8 / DirectILKernelGenerator.GetTypeSize(T) : 1;
    }

    /// <summary>
    ///     float16 as 8 float32 lanes kept ON the f16 grid (<see cref="ILKernelGenerator.PolyHalfArith8"/>),
    ///     <c>W</c> of them live: 8 in a float16/float32 loop, 4 in a float64 loop and 2 in a complex128 loop,
    ///     where float16 values appear as x or as the peeled steps' coefficient dtype. Because the lanes already
    ///     ARE float32 values, a float16 lane value converts to a float32 loop for free.
    /// </summary>
    internal sealed class PolyHalfLaneKind : PolyValueKind
    {
        private readonly int _w;

        /// <summary>Creates the float16 vector kind (<c>Vector256&lt;float&gt;</c> locals).</summary>
        /// <param name="w">Live lanes: 2, 4 or 8 (the loop dtype's lane count).</param>
        /// <exception cref="NotSupportedException">Another lane count.</exception>
        public PolyHalfLaneKind(int w) : base(true, NPTypeCode.Single, 256)
        {
            if (w is not (2 or 4 or 8)) throw new NotSupportedException($"no {w}-lane float16 kind");
            _w = w;
        }

        /// <inheritdoc/>
        /// <remarks>The dead lanes (w &lt; 8) hold zeros; an op on them produces a value no store reads.</remarks>
        public override void Bin(ILGenerator il, BinaryOp op, PolyComplexProduct product = PolyComplexProduct.Simd)
        {
            il.Emit(OpCodes.Ldc_I4, (int)op);
            il.EmitCall(OpCodes.Call, ILKernelGenerator.s_polyHalfArith8, null);
        }

        /// <inheritdoc/>
        /// <remarks>Reads exactly <c>w</c> float16 values: never past the chain's last point.</remarks>
        public override void Load(ILGenerator il)
        {
            if (_w != 8)
            {
                il.EmitCall(OpCodes.Call, _w == 4 ? PolyLaneOps.s_halfLoad4 : PolyLaneOps.s_halfLoad2, null);
                return;
            }
            il.EmitCall(OpCodes.Call, VectorMethodCache.Load(128, typeof(ushort)), null);
            il.EmitCall(OpCodes.Call, PolyLanes.s_halfWiden8V, null);
        }

        /// <inheritdoc/>
        public override void BroadcastFromScalar(ILGenerator il)
        {
            // [Half] -> float (exact) -> 8 lanes.
            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Half, NPTypeCode.Single);
            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(256, typeof(float)), null);
        }

        /// <inheritdoc/>
        /// <exception cref="NotSupportedException">Fewer than 8 live lanes: only a float16 loop (w = 8) stores float16.</exception>
        public override void StoreValueFirst(ILGenerator il)
        {
            if (_w != 8) throw new NotSupportedException("float16 results are stored by 8-lane (float16 loop) chains only");
            var ptr = il.DeclareLocal(typeof(byte*)); il.Emit(OpCodes.Stloc, ptr);
            il.EmitCall(OpCodes.Call, PolyLanes.s_halfNarrow8V, null);
            il.Emit(OpCodes.Ldloc, ptr);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Store(128, typeof(ushort)), null);
        }

        /// <inheritdoc/>
        public override int Lanes => _w;
    }

    /// <summary>
    ///     A value per chain. A SHARED value is ONE scalar local of its own dtype for all chains; a per-point
    ///     value is one local per chain (a scalar in scalar chains, a W-lane vector in vector chains).
    /// </summary>
    internal sealed class PolyValue
    {
        /// <summary>Per-chain locals (the same scalar local repeated when shared).</summary>
        public readonly LocalBuilder[] L;
        /// <summary>The value's NumPy dtype.</summary>
        public readonly NPTypeCode T;
        /// <summary>One scalar for all chains.</summary>
        public readonly bool Shared;
        /// <summary>
        ///     The ndim NumPy's array for this value has, in a <see cref="PolyUnitBroadcast"/> kernel (0 in every
        ///     other kernel, and for weak Python values). It picks the complex product: two array operands of
        ///     different ndim are NumPy's single-element NpyIter loop (<see cref="PolyComplexProduct.LoopScalar"/>).
        ///     The numbers are representative, not the caller's — only which operand is deeper matters.
        /// </summary>
        public readonly int Nd;

        /// <summary>Wraps locals.</summary>
        /// <param name="l">Locals.</param><param name="t">Dtype.</param><param name="shared">Shared flag.</param>
        /// <param name="nd">NumPy ndim tag (see <see cref="Nd"/>).</param>
        public PolyValue(LocalBuilder[] l, NPTypeCode t, bool shared, int nd = 0) { L = l; T = t; Shared = shared; Nd = nd; }
    }

    /// <summary>The values the step expressions may read, bound per block.</summary>
    internal sealed class PolyEnv
    {
        /// <summary>Symbol values (null when not bound in the current context).</summary>
        public PolyValue X, X2, C0, C1, Tmp, Ck;

        /// <summary>The bound value of <paramref name="s"/>.</summary>
        /// <param name="s">The symbol.</param><returns>Its value.</returns>
        /// <exception cref="InvalidOperationException">The symbol is not bound — a step table reads something
        ///     the kernel class never computed (a program-construction bug).</exception>
        public PolyValue Get(PolySym s) => (s switch
        {
            PolySym.X => X, PolySym.X2 => X2, PolySym.C0 => C0, PolySym.C1 => C1,
            PolySym.Tmp => Tmp, PolySym.Ck => Ck, _ => null,
        }) ?? throw new InvalidOperationException($"symbol {s} is not bound here");
    }

    /// <summary>
    ///     Emits step expressions for U interleaved chains. Each point's recurrence is one serial dependency
    ///     chain, so interleaving U of them hides the multiply/add latency (measured: ×4 is 1.2–1.9× over ×1).
    /// </summary>
    internal sealed class PolyEmitter
    {
        /// <summary>The generator.</summary>
        public readonly ILGenerator IL;
        /// <summary>Chain count.</summary>
        public readonly int U;
        /// <summary>Vector chains (per-point values are W-lane vectors) or scalar chains.</summary>
        public readonly bool Vec;
        /// <summary>Lanes per chain (1 for scalar chains).</summary>
        public readonly int W;
        /// <summary>
        ///     NumPy's ndim of a coefficient read (<see cref="PolySym.Ck"/>) and of the per-point x in a
        ///     <see cref="PolyUnitBroadcast"/> kernel: representative values (2 and 1, or 1 and 2 — only which side
        ///     is deeper matters). Both 0 in every other kernel, so no product there can become
        ///     <see cref="PolyComplexProduct.LoopScalar"/>.
        /// </summary>
        public readonly int CoefNd, PointNd;
        private readonly PolyConstPool _pool;
        private readonly LocalBuilder _table, _row;
        private readonly bool _scalarMath;
        private readonly Dictionary<NPTypeCode, PolyValueKind> _scalar = new(), _lane = new();

        /// <summary>Creates an emitter.</summary>
        /// <param name="il">Generator.</param>
        /// <param name="u">Chains.</param>
        /// <param name="vec">Vector chains.</param>
        /// <param name="lanes">Lanes per chain in vector chains (the loop dtype's).</param>
        /// <param name="pool">The kernel's constant pool (regions are registered while emitting).</param>
        /// <param name="table">Local holding the region table pointer.</param>
        /// <param name="row">Local holding the current row (<c>nd</c>) for per-row constants.</param>
        /// <param name="scalarMath">NumPy runs every op as scalar math (0-d result): complex multiplies not
        ///     touching the x leaf are the naive product.</param>
        /// <param name="unit">NumPy's single-element broadcast (<see cref="PolyUnitBroadcast"/>): values are tagged
        ///     with their ndim, and a product of two array values whose ndims differ is NumPy's fallback loop.</param>
        public PolyEmitter(ILGenerator il, int u, bool vec, int lanes, PolyConstPool pool, LocalBuilder table, LocalBuilder row,
            bool scalarMath, PolyUnitBroadcast unit)
        {
            IL = il; U = u; Vec = vec; W = vec ? lanes : 1; _pool = pool; _table = table; _row = row; _scalarMath = scalarMath;
            // The same representative ndims PeelCount plans with, so the planned fixpoint is the emitted one.
            (CoefNd, PointNd) = PolyTyping.UnitNds(unit);
        }

        /// <summary>The scalar kind of <paramref name="t"/> (shared values, scalar chains).</summary>
        /// <param name="t">The dtype.</param><returns>The kind.</returns>
        public PolyValueKind Scalar(NPTypeCode t)
        {
            if (!_scalar.TryGetValue(t, out var k)) _scalar[t] = k = new PolyValueKind(false, t);
            return k;
        }

        /// <summary>The per-point kind of <paramref name="t"/>: a W-lane vector in vector chains (the table in
        ///     <c>ILKernelGenerator.Polynomial.Lanes.cs</c>), else the scalar.</summary>
        /// <param name="t">The dtype.</param><returns>The kind.</returns>
        /// <exception cref="NotSupportedException">No W-lane kind exists for t — the block gate missed a
        ///     combination; the kernel compile catches this and emits scalar chains instead.</exception>
        public PolyValueKind Lane(NPTypeCode t)
        {
            if (!Vec) return Scalar(t);
            if (!_lane.TryGetValue(t, out var k)) _lane[t] = k = PolyLanes.CreateLaneKind(t, W);
            return k;
        }

        /// <summary>Declares locals for a value.</summary>
        /// <param name="t">Dtype.</param><param name="shared">One scalar for all chains.</param>
        /// <param name="nd">NumPy ndim tag (<see cref="PolyValue.Nd"/>; 0 outside <see cref="PolyUnitBroadcast"/> kernels).</param>
        /// <returns>The value.</returns>
        public PolyValue Fresh(NPTypeCode t, bool shared, int nd = 0)
        {
            var l = new LocalBuilder[U];
            if (shared) { var one = IL.DeclareLocal(Scalar(t).LocalType); for (int u = 0; u < U; u++) l[u] = one; }
            else { var type = Lane(t).LocalType; for (int u = 0; u < U; u++) l[u] = IL.DeclareLocal(type); }
            return new PolyValue(l, t, shared, nd);
        }

        /// <summary>The per-point form of a shared value: in vector chains a broadcast vector (built once for
        ///     all chains); in scalar chains the scalar itself.</summary>
        /// <param name="s">A shared value.</param><returns>A local every chain can read.</returns>
        private LocalBuilder Spread(PolyValue s)
        {
            if (!Vec) return s.L[0];
            var k = Lane(s.T);
            var v = IL.DeclareLocal(k.LocalType);
            IL.Emit(OpCodes.Ldloc, s.L[0]);
            k.BroadcastFromScalar(IL);
            IL.Emit(OpCodes.Stloc, v);
            return v;
        }

        /// <summary>Copies <paramref name="from"/> into the per-point <paramref name="to"/> chain by chain (a
        ///     shared source is spread): how loop-carried state is written back.</summary>
        /// <param name="from">Source of the same dtype.</param><param name="to">Per-point destination.</param>
        public void CopyTo(PolyValue from, PolyValue to)
        {
            var spread = from.Shared ? Spread(from) : null;
            for (int u = 0; u < U; u++) { IL.Emit(OpCodes.Ldloc, spread ?? from.L[u]); IL.Emit(OpCodes.Stloc, to.L[u]); }
        }

        /// <summary>Emits a tree at its own NumPy dtype.</summary>
        /// <param name="e">The tree.</param><param name="env">Bound symbols.</param><returns>The value.</returns>
        /// <exception cref="InvalidOperationException">A bare weak value (it has no dtype) or an unbound symbol.</exception>
        public PolyValue Emit(PolyExpr e, PolyEnv env)
        {
            switch (e)
            {
                case PolyLeaf l: return env.Get(l.S);
                case PolyBin b:
                {
                    var t = PolyTyping.LoopType(b, PolyTyping.TypeOf(b.A, s => env.Get(s).T), PolyTyping.TypeOf(b.B, s => env.Get(s).T));
                    var va = EmitAt(b.A, env, t);
                    var vb = EmitAt(b.B, env, t);
                    // A ufunc (array) op in NumPy's eyes iff one operand is the raw x array; see the file header.
                    bool touchesX = b.A is PolyLeaf { S: PolySym.X } || b.B is PolyLeaf { S: PolySym.X };
                    // The product NumPy runs (file header): scalar math when every value is a NumPy scalar; its
                    // single-element NpyIter loop when BOTH operands are arrays (Nd 0 = a weak Python value, which
                    // the trivial loop accepts as 0-d) of different ndim; simd_cmul otherwise.
                    var product = _scalarMath && !touchesX ? PolyComplexProduct.Naive
                        : va.Nd != 0 && vb.Nd != 0 && va.Nd != vb.Nd ? PolyComplexProduct.LoopScalar
                        : PolyComplexProduct.Simd;
                    return Op(b.Op, va, vb, t, product);
                }
                default: throw new InvalidOperationException("a weak value is typed by its partner operand");
            }
        }

        /// <summary>Emits a tree converted to <paramref name="t"/> (a weak value is materialized in it).</summary>
        /// <param name="e">The tree.</param><param name="env">Bound symbols.</param><param name="t">Target dtype.</param>
        /// <returns>The value.</returns>
        public PolyValue EmitAt(PolyExpr e, PolyEnv env, NPTypeCode t) => e is PolyWeak w ? Const(w.W, t) : Convert(Emit(e, env), t);

        /// <summary>Converts a value (NumPy's loop-input cast): a shared scalar through the house
        ///     <c>EmitConvertTo</c>, a per-point vector through the exact lane conversion.</summary>
        /// <param name="v">The value.</param><param name="t">Target dtype.</param><returns>The converted value (or v).</returns>
        public PolyValue Convert(PolyValue v, NPTypeCode t)
        {
            if (v.T == t) return v;
            var r = Fresh(t, v.Shared, v.Nd);   // a cast keeps the array's ndim
            int n = v.Shared ? 1 : U;
            for (int u = 0; u < n; u++)
            {
                IL.Emit(OpCodes.Ldloc, v.L[u]);
                if (Vec && !v.Shared) PolyLanes.EmitLaneConvert(IL, v.T, t, W);
                else DirectILKernelGenerator.EmitConvertTo(IL, v.T, t);
                IL.Emit(OpCodes.Stloc, r.L[u]);
            }
            return r;
        }

        /// <summary>Loads a weak value from its constant-pool region as a shared scalar (row = the current row
        ///     for per-row constants, 0 otherwise).</summary>
        /// <param name="w">The constant.</param><param name="t">Its dtype here.</param><returns>A shared value.</returns>
        private PolyValue Const(PolyWeakConst w, NPTypeCode t)
        {
            int region = _pool.Region(w, t);
            var r = Fresh(t, shared: true);
            IL.Emit(OpCodes.Ldloc, _table);
            IL.Emit(OpCodes.Ldc_I4, region * 8); IL.Emit(OpCodes.Conv_I); IL.Emit(OpCodes.Add);
            IL.Emit(OpCodes.Ldind_I);
            if (w.PerRow)
            {
                IL.Emit(OpCodes.Ldloc, _row);
                IL.Emit(OpCodes.Ldc_I8, (long)DirectILKernelGenerator.GetTypeSize(t));
                IL.Emit(OpCodes.Mul); IL.Emit(OpCodes.Conv_I); IL.Emit(OpCodes.Add);
            }
            Scalar(t).Load(IL);
            IL.Emit(OpCodes.Stloc, r.L[0]);
            return r;
        }

        /// <summary>Emits <c>a op b</c>: once as a scalar when both are shared, else per chain with any shared
        ///     operand spread once.</summary>
        /// <param name="op">The op.</param>
        /// <param name="a">Left (already in t).</param>
        /// <param name="b">Right (already in t).</param>
        /// <param name="t">Loop dtype.</param>
        /// <param name="product">Which NumPy complex product a Complex multiply reproduces (see the file header).</param>
        /// <returns>The result, tagged with NumPy's broadcast ndim (the larger operand ndim).</returns>
        private PolyValue Op(BinaryOp op, PolyValue a, PolyValue b, NPTypeCode t, PolyComplexProduct product)
        {
            int nd = Math.Max(a.Nd, b.Nd);
            if (a.Shared && b.Shared)
            {
                var s = Fresh(t, true, nd);
                IL.Emit(OpCodes.Ldloc, a.L[0]); IL.Emit(OpCodes.Ldloc, b.L[0]);
                Scalar(t).Bin(IL, op, product);
                IL.Emit(OpCodes.Stloc, s.L[0]);
                return s;
            }
            if (Vec && op == BinaryOp.Divide && t == NPTypeCode.Complex && b.Shared)
            {
                // A divisor shared by the whole block (the steps' `/nd`): CDOUBLE_divide's branch, rat and scl
                // depend on the divisor alone, so they are computed ONCE here — the same values NumPy recomputes
                // per element — and each chain runs only the per-point half (no division per point).
                var prep = IL.DeclareLocal(typeof(PolyCDivShared));
                IL.Emit(OpCodes.Ldloc, b.L[0]);
                IL.EmitCall(OpCodes.Call, PolyLaneOps.s_cDivPrep, null);
                IL.Emit(OpCodes.Stloc, prep);
                var q = Fresh(t, false, nd);
                for (int u = 0; u < U; u++)
                {
                    IL.Emit(OpCodes.Ldloc, a.L[u]);   // a is per point: two shared operands took the scalar path above
                    IL.Emit(OpCodes.Ldloca, prep);
                    IL.EmitCall(OpCodes.Call, PolyLaneOps.s_cDivBy, null);
                    IL.Emit(OpCodes.Stloc, q.L[u]);
                }
                return q;
            }
            var sa = a.Shared ? Spread(a) : null;
            var sb = b.Shared ? Spread(b) : null;
            var r = Fresh(t, false, nd);
            var k = Lane(t);
            for (int u = 0; u < U; u++)
            {
                IL.Emit(OpCodes.Ldloc, sa ?? a.L[u]);
                IL.Emit(OpCodes.Ldloc, sb ?? b.L[u]);
                // Scalar-math and single-element-broadcast kernels emit only scalar chains, whose loop-carried
                // state (Horner's c0, Clenshaw's c0/c1) is per-point: those ops run NumPy's scalar-math or fallback
                // product too. A vector kind ignores the choice (it only exists where NumPy runs simd_cmul).
                k.Bin(IL, op, product);
                IL.Emit(OpCodes.Stloc, r.L[u]);
            }
            return r;
        }
    }
}
