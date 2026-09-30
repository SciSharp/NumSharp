using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.PolyAlgebra.cs — kernel access for numpy.polynomial's series algebra (plan U2)
// =============================================================================
//
// The series-algebra engine (Polynomial/Package/NDPolyAlgebra*.cs) replays NumPy's Python — {p}mul, {p}div, {p}pow,
// {p}fromroots and the basis conversions — as a sequence of ARRAY operations over short intermediate series, the
// way NumPy itself runs them: `c[-i] * xs`, `(c1 * (nd - 1)) / nd`, `c1[i:j] -= c2 * c1[j]`, `zs + zs[::-1]`. Each of
// those is one ufunc call in NumPy, so each runs here through the SAME house kernel NumPy's ufunc runs through in
// NumSharp (GetMixedTypeKernel): the complex product is simd_cmul (NDComplexMath.Multiply, operand order kept by
// the scalar-left / scalar-right paths), the complex quotient CDOUBLE_divide's Smith algorithm, float16 NumPy's HALF
// loop (widen, float32 op, round), decimal the house decimal ops. What this file adds is only the LOOKUP: the
// engine issues a few dozen such calls per Python-level iteration, and GetMixedTypeKernel's dictionary lookup
// (~25 ns) would dominate a call that moves a dozen elements, so a flat slot array fronts it per (op, path, dtype) —
// the same pattern as the scalar-kernel slots of DirectILKernelGenerator.PolySeries.cs.
//
// The {p}mulx kernels are calculus kernels (DirectILKernelGenerator.PolyCalculus.cs: PolyCalcKey.Mulx) with the
// same slot front, because the engine calls mulx inside every Legendre / Laguerre / Hermite product step.
//
// THE FUSED CHEBMULX KERNEL
// -------------------------
// chebmulx is the one mulx that is not a Python loop but three array statements —
//
//     prd[0] = c[0]*0; prd[1] = c[0]; tmp = c[1:]/2; prd[2:] = tmp; prd[0:-2] += tmp
//
// — so NumPy streams it, and neither the calculus kernel (built to vectorize over the COLUMNS of an N-D series, it walks
// a 1-D series one row at a time with a true division per element: 0.43x NumPy on a 10000-term float16 series, 0.74x
// float32) nor the statements run one by one through the house kernels (three passes over an arena temporary: 1.05x
// complex128, 1.48x float64) keeps up. Their combined effect is element-wise:
//
//     prd[0]   = c[0]*0 + c[1]/2          prd[j] = c[j-1]/2 + c[j+1]/2   (2 <= j <= n-2)
//     prd[1]   = c[0] (+ c[2]/2, n > 2)   prd[n-1] = c[n-2]/2 (n > 2),   prd[n] = c[n-1]/2
//
// and GetPolyChebMulxKernel computes it in ONE forward pass, each element with the operations NumPy's statements apply
// to it, in their operand order: `c[0]*0` is scalarmath (a NumPy scalar times the Python int 0: the NAIVE complex
// product, so an infinite part keeps its NaN), `x/2` is the ufunc's true division by the weak int 2 in the series
// dtype, the add is the ufunc's (prd's element first). The vector stage halves by MULTIPLYING by 0.5 instead of
// dividing: for every binary float x, x/2 and x*0.5 are the same exact value rounded once, NaNs included — and vdivpd
// would bound the pass at one division per element. float16 lanes (exact float32 values) halve exactly in float32 and
// are then rounded to the float16 grid, which only a subnormal half can leave (PolyLaneOps.HalfHalveOnGrid): the two
// halves are float16 values before NumPy adds them, and the add is a plain float32 add rounded once by the store —
// HALF_add. A complex half is CDOUBLE_divide's Smith branch for the divisor 2+0j, whose rat (+0) and scl (0.5) are
// prepared once (PolyLaneOps.CDivPrep / CDivBy), so the stage divides nothing. Scalar steps (head, remainder, tail,
// and every element of a decimal series) divide literally. The kernel reads c anywhere — in place from the caller's
// contiguous series, or from prd[1:] where the caller converted it (each element is read before the pass writes its
// slot).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     One fused <c>chebmulx</c> call (see <see cref="DirectILKernelGenerator"/>'s PolyAlgebra file header): NumPy's
    ///     <c>prd</c> for a series of <paramref name="n"/> ≥ 2 terms of one dtype, computed in one forward pass.
    /// </summary>
    /// <param name="c">c[0]; the <paramref name="n"/> terms are contiguous. It may be <c>prd + itemsize</c> — the series
    ///     already converted into the result's slots 1 … n: every element is read before the pass writes its slot.</param>
    /// <param name="n">Terms (≥ 2; the one-term series is the calculus mulx kernel's).</param>
    /// <param name="prd">prd[0]: <paramref name="n"/> + 1 contiguous elements, every one written.</param>
    public unsafe delegate void PolyChebMulxKernel(byte* c, long n, byte* prd);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>Fast front of <see cref="GetPolyHouseBinaryKernel"/>: four ops × three paths × every NPTypeCode value.</summary>
        private static readonly MixedTypeKernel[] s_polyHouseBinarySlots = new MixedTypeKernel[4 * 3 * PolyDtypeSlots];

        /// <summary>Fast front of <see cref="GetPolyMulxKernel"/>: six bases × every NPTypeCode value.</summary>
        private static readonly PolyCalcKernel[] s_polyMulxSlots = new PolyCalcKernel[6 * PolyDtypeSlots];

        /// <summary>The fused chebmulx kernels (<see cref="GetPolyChebMulxKernel"/>), one per NPTypeCode value.</summary>
        private static readonly PolyChebMulxKernel[] s_polyChebMulxSlots = new PolyChebMulxKernel[PolyDtypeSlots];

        /// <summary>
        ///     The house same-dtype binary kernel NumPy's ufunc call runs through in NumSharp — <c>np.add</c>,
        ///     <c>np.subtract</c>, <c>np.multiply</c>, <c>np.true_divide</c> of two <paramref name="t"/> operands into a
        ///     <paramref name="t"/> result — for one contiguous layout: both operands arrays (<see cref="ExecutionPath.SimdFull"/>),
        ///     the right one a scalar (<see cref="ExecutionPath.SimdScalarRight"/>: <c>array OP scalar</c>) or the left one
        ///     (<see cref="ExecutionPath.SimdScalarLeft"/>: <c>scalar OP array</c>). Operand order is kept, which is what
        ///     makes a complex product NumPy's (simd_cmul is not symmetric in its last bit).
        /// </summary>
        /// <param name="op">Add, Subtract, Multiply or Divide (true division).</param>
        /// <param name="path">SimdFull, SimdScalarRight or SimdScalarLeft.</param>
        /// <param name="t">The operands' and result's dtype.</param>
        /// <returns>The cached kernel: <c>(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim, n)</c>, of which these
        ///     three paths read only the pointers and <c>n</c> (contiguous operands, a scalar operand read once).</returns>
        /// <exception cref="System.InvalidOperationException">IL generation is disabled (no dynamic code).</exception>
        /// <exception cref="System.NotSupportedException">The dtype has no house loop for <paramref name="op"/>.</exception>
        internal static MixedTypeKernel GetPolyHouseBinaryKernel(BinaryOp op, ExecutionPath path, NPTypeCode t)
        {
            int slot = (int)op <= (int)BinaryOp.Divide && (int)path <= (int)ExecutionPath.SimdScalarLeft && (int)t < PolyDtypeSlots
                ? ((int)op * 3 + (int)path) * PolyDtypeSlots + (int)t : -1;
            if (slot >= 0 && s_polyHouseBinarySlots[slot] is { } fast)
                return fast;
            var k = GetMixedTypeKernel(new MixedTypeKernelKey(t, t, t, op, path));
            if (slot >= 0)
                Volatile.Write(ref s_polyHouseBinarySlots[slot], k);
            return k;
        }

        /// <summary>
        ///     The <c>{p}mulx</c> kernel of <paramref name="basis"/> for a 1-D series of <paramref name="t"/>: an integral-layout
        ///     calculus kernel (<see cref="PolyCalcKey.Mulx"/>) called as <c>k(null, 0, 0, prd, size, 1, len(c), 1, 1, null)</c>
        ///     over a buffer holding <c>c</c> at rows 1 … len(c) — it computes NumPy's <c>prd</c> in place.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="t">The series dtype (np.common_type's: float16/float32/float64/complex128 or decimal).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="System.PlatformNotSupportedException">The runtime cannot emit dynamic code.</exception>
        internal static PolyCalcKernel GetPolyMulxKernel(PolyBasis basis, NPTypeCode t)
        {
            int slot = (int)t < PolyDtypeSlots ? (int)basis * PolyDtypeSlots + (int)t : -1;
            if (slot >= 0 && s_polyMulxSlots[slot] is { } fast)
                return fast;
            var k = GetPolyCalcKernel(new PolyCalcKey(basis, Integrate: true, T: t, Src: t, ScalarMath: true, Scale: false, Mulx: true));
            if (slot >= 0)
                Volatile.Write(ref s_polyMulxSlots[slot], k);
            return k;
        }

        /// <summary>
        ///     The fused <c>chebmulx</c> kernel of a series of <paramref name="t"/> (see the file header): one forward pass
        ///     computing NumPy's three array statements element by element, vector lanes over the output positions where the
        ///     dtype has a lane kind. Compiled once per dtype; a racing first call may compile twice, and either copy is the
        ///     same code.
        /// </summary>
        /// <param name="t">The series dtype (np.common_type's: float16/float32/float64/complex128 or decimal).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyChebMulxKernel GetPolyChebMulxKernel(NPTypeCode t)
        {
            int slot = (int)t < PolyDtypeSlots ? (int)t : -1;
            if (slot >= 0 && s_polyChebMulxSlots[slot] is { } fast)
                return fast;
            var k = CompilePolyChebMulx(t, allowVector: true);
            if (slot >= 0)
                Volatile.Write(ref s_polyChebMulxSlots[slot], k);
            return k;
        }

        /// <summary>
        ///     Compiles a fused chebmulx kernel (uncached — <see cref="GetPolyChebMulxKernel"/> caches). With
        ///     <paramref name="allowVector"/> the vector stage is emitted wherever the dtype has a lane kind; an op the lane
        ///     kind turns out to lack restarts the emission scalar-only (the same bits, slower) and counts in
        ///     <see cref="PolyCalcVectorFallbacks"/>, the calculus family's counter, so a test can prove it never happens.
        ///     Tests compile the scalar-only kernel directly to hold the vector stage to its bits.
        /// </summary>
        /// <param name="t">The series dtype.</param>
        /// <param name="allowVector">Emit the vector stage where a lane kind exists.</param>
        /// <returns>A new kernel.</returns>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyChebMulxKernel CompilePolyChebMulx(NPTypeCode t, bool allowVector)
        {
            if (!RuntimeFeature.IsDynamicCodeSupported)
                throw new PlatformNotSupportedException(
                    "numpy.polynomial's chebmulx compiles an IL kernel at runtime; this runtime (NativeAOT) cannot emit dynamic code.");
            if (allowVector)
            {
                try
                {
                    return EmitPolyChebMulx(t, vector: true);
                }
                catch (NotSupportedException)
                {
                    // A lane op the table lacks: nothing half-built survives (the DynamicMethods of the failed attempt are
                    // dropped), the scalar-only kernel below is complete on its own.
                    Interlocked.Increment(ref s_polyCalcVectorFallbacks);
                }
            }
            return EmitPolyChebMulx(t, vector: false);
        }

        /// <summary>
        ///     Emits one fused chebmulx kernel: a root (head, scalar remainder, tail — see <see cref="EmitPolyChebMulxRoot"/>)
        ///     calling a separate vector stage (<see cref="EmitPolyChebMulxVector"/>). The stage is its own DynamicMethod for
        ///     the reason the calculus and evaluation kernels split theirs: the JIT's inline budget is per method, and the
        ///     root's scalar complex/float16 helpers must not crowd the lane helpers out of the hot loop.
        /// </summary>
        /// <param name="t">The series dtype.</param>
        /// <param name="vector">Emit the vector stage when <paramref name="t"/> has a lane kind (decimal never does).</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="NotSupportedException">The lane kind lacks an op the vector stage emits.</exception>
        private static PolyChebMulxKernel EmitPolyChebMulx(NPTypeCode t, bool vector)
        {
            int size = GetTypeSize(t);
            PolyValueKind vk = null;
            if (vector && t != NPTypeCode.Decimal)
            {
                int lanes = PolyLanes.LoopLanes(t);
                if (lanes > 1 && PolyLanes.HasLaneKind(t, lanes))
                    vk = PolyLanes.CreateLaneKind(t, lanes);
            }
            string name = $"NDPolyChebMulx_{t}{(vk is null ? "_scalar" : "")}";
            var vec = vk is null ? null : EmitPolyChebMulxVector(t, vk, size, name + "_vec");
            var root = new DynamicMethod(name, typeof(void), new[] { typeof(byte*), typeof(long), typeof(byte*) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            EmitPolyChebMulxRoot(root.GetILGenerator(), t, size, vec);
            return root.CreateDelegate<PolyChebMulxKernel>();
        }

        /// <summary>
        ///     The root of a fused chebmulx kernel, arguments <c>(c, n, prd)</c> (<see cref="PolyChebMulxKernel"/>):
        ///     <c>prd[0] = c[0]*0 + c[1]/2</c> and <c>prd[1] = c[0] (+ c[2]/2)</c> from locals read first; the positions
        ///     <c>2 … n-2</c>, the vector stage's share first, then the rest one element at a time; then
        ///     <c>prd[n-1] = c[n-2]/2</c> (n &gt; 2) and <c>prd[n] = c[n-1]/2</c>. Every op is the scalar op of
        ///     <paramref name="t"/> NumPy's statement applies (the division is a real division here).
        /// </summary>
        /// <param name="il">The root's generator.</param>
        /// <param name="t">The series dtype.</param>
        /// <param name="size">Its itemsize.</param>
        /// <param name="vec">The vector stage <c>(c, prd, j, last) → next j</c>, or null for a scalar-only kernel.</param>
        private static void EmitPolyChebMulxRoot(ILGenerator il, NPTypeCode t, int size, DynamicMethod vec)
        {
            Type clr = GetClrType(t);
            var two = il.DeclareLocal(clr);
            var zero = il.DeclareLocal(clr);
            var c0 = il.DeclareLocal(clr);
            var v = il.DeclareLocal(clr);
            var cm = il.DeclareLocal(typeof(byte*));   // c - size: c[j-1] sits at cm + j*size
            var cp = il.DeclareLocal(typeof(byte*));   // c + size: c[j+1] sits at cp + j*size
            var prd = il.DeclareLocal(typeof(byte*));
            var j = il.DeclareLocal(typeof(long));
            var last = il.DeclareLocal(typeof(long));  // n - 2: the last position that receives both halves

            // [x] -> [x / 2]: NumPy's true division by the weak int 2, in t.
            void Halve()
            {
                il.Emit(OpCodes.Ldloc, two);
                EmitScalarOperation(il, BinaryOp.Divide, t);
            }

            // [] -> [p[idx]] for a byte* local p and a long local idx.
            void LoadAt(LocalBuilder p, LocalBuilder idx)
            {
                EmitPolyCalcAddr(il, p, idx, size);
                EmitLoadIndirect(il, t);
            }

            // The Python ints of NumPy's statements (`c[0]*0`, `c[1:]/2`) converted into t, NEP 50's weak-int rule.
            il.Emit(OpCodes.Ldc_I8, 2L);
            EmitPolyCalcWeakInt(il, t);
            il.Emit(OpCodes.Stloc, two);
            il.Emit(OpCodes.Ldc_I8, 0L);
            EmitPolyCalcWeakInt(il, t);
            il.Emit(OpCodes.Stloc, zero);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, cm);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, cp);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, prd);
            // c[0] into a local first: when c is prd + size, prd[1] IS c[0].
            il.Emit(OpCodes.Ldarg_0);
            EmitLoadIndirect(il, t);
            il.Emit(OpCodes.Stloc, c0);

            // prd[0] = c[0]*0 + c[1]/2 — prd[0] lies outside c, so it may be written before c[2] is read below.
            il.Emit(OpCodes.Ldloc, prd);
            il.Emit(OpCodes.Ldloc, c0);
            il.Emit(OpCodes.Ldloc, zero);
            EmitPolyCalcScalarBin(il, BinaryOp.Multiply, t, PolyComplexProduct.Naive);
            il.Emit(OpCodes.Ldloc, cp);
            EmitLoadIndirect(il, t);
            Halve();
            EmitScalarOperation(il, BinaryOp.Add, t);
            EmitStoreIndirect(il, t);

            // prd[1] = c[0], plus tmp[1] = c[2]/2 when there is a third term (prd[0:-2] then reaches index 1).
            il.Emit(OpCodes.Ldloc, c0);
            il.Emit(OpCodes.Stloc, v);
            var noThird = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Ble, noThird);
            il.Emit(OpCodes.Ldloc, c0);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, 2 * size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, t);
            Halve();
            EmitScalarOperation(il, BinaryOp.Add, t);
            il.Emit(OpCodes.Stloc, v);
            il.MarkLabel(noThird);
            il.Emit(OpCodes.Ldloc, prd);
            il.Emit(OpCodes.Ldc_I4, size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, v);
            EmitStoreIndirect(il, t);

            // Positions 2 … n-2: the vector stage's full vectors, then one element at a time.
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Stloc, j);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, last);
            if (vec is not null)
            {
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, prd);
                il.Emit(OpCodes.Ldloc, j);
                il.Emit(OpCodes.Ldloc, last);
                il.Emit(OpCodes.Call, vec);
                il.Emit(OpCodes.Stloc, j);
            }
            var top = il.DefineLabel();
            var done = il.DefineLabel();
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldloc, last);
            il.Emit(OpCodes.Bgt, done);
            // prd[j] = c[j-1]/2 + c[j+1]/2 — c[j-1] is prd[j] itself in the in-place layout, read before the store.
            EmitPolyCalcAddr(il, prd, j, size);
            LoadAt(cm, j);
            Halve();
            LoadAt(cp, j);
            Halve();
            EmitScalarOperation(il, BinaryOp.Add, t);
            EmitStoreIndirect(il, t);
            EmitPolyCalcBump(il, j, 1);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(done);

            // prd[n-1] = c[n-2]/2 when n > 2 (for n == 2, prd[1] is c[0]'s, written above); prd[n] = c[n-1]/2.
            var noTail = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Ble, noTail);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, j);
            EmitPolyCalcAddr(il, prd, j, size);
            LoadAt(cm, j);
            Halve();
            EmitStoreIndirect(il, t);
            il.MarkLabel(noTail);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, j);
            EmitPolyCalcAddr(il, prd, j, size);
            LoadAt(cm, j);
            Halve();
            EmitStoreIndirect(il, t);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>
        ///     The vector stage of a fused chebmulx kernel, a DynamicMethod <c>long (byte* c, byte* prd, long j, long last)</c>:
        ///     for every full vector of positions <c>j … j+W-1 ≤ last</c>, <c>prd[j…] = half(c[j-1…]) + half(c[j+1…])</c>,
        ///     returning the first position it did not process. A float64 / float32 lane halves by multiplying by 0.5 (exact —
        ///     see the file header), a float16 lane by <see cref="PolyLaneOps.HalfHalveOnGrid"/> (and adds in plain float32,
        ///     rounded once by the store), a complex lane by <see cref="PolyLaneOps.CDivBy"/> with the divisor 2+0j prepared once.
        ///     In the in-place layout (c = prd + size) each vector reads prd[j…j+W+1] before storing prd[j…j+W-1], and no
        ///     later vector reads below its own j, so the forward order is safe at any width.
        /// </summary>
        /// <param name="t">The series dtype (float16/float32/float64/complex128).</param>
        /// <param name="vk">Its lane kind (float16 as float32 lanes on the f16 grid; complex128 as [re, im] pairs).</param>
        /// <param name="size">The itemsize.</param>
        /// <param name="name">The stage's name (shows in JIT disassembly).</param>
        /// <returns>The stage.</returns>
        /// <exception cref="NotSupportedException">The lane kind lacks multiply / add / load / store.</exception>
        private static DynamicMethod EmitPolyChebMulxVector(NPTypeCode t, PolyValueKind vk, int size, string name)
        {
            var dm = new DynamicMethod(name, typeof(long), new[] { typeof(byte*), typeof(byte*), typeof(long), typeof(long) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            int w = vk.Lanes;
            var cm = il.DeclareLocal(typeof(byte*));
            var cp = il.DeclareLocal(typeof(byte*));
            var prd = il.DeclareLocal(typeof(byte*));
            var j = il.DeclareLocal(typeof(long));
            var stop = il.DeclareLocal(typeof(long));   // last - (W - 1): the last j a full vector fits at
            LocalBuilder half = null, prep = null;
            if (t == NPTypeCode.Complex)
            {
                // CDOUBLE_divide's branch, rat and scl depend on the divisor alone: for 2+0j, |2| >= |0|, rat = +0,
                // scl = 0.5 — the values NumPy recomputes per element, prepared once.
                prep = il.DeclareLocal(typeof(PolyCDivShared));
                il.Emit(OpCodes.Ldc_I8, 2L);
                EmitPolyCalcWeakInt(il, t);
                il.EmitCall(OpCodes.Call, PolyLaneOps.s_cDivPrep, null);
                il.Emit(OpCodes.Stloc, prep);
            }
            else if (t != NPTypeCode.Half)
            {
                // 0.5 in the lanes' element type (float64 or float32).
                half = il.DeclareLocal(vk.LocalType);
                if (vk.Clr == typeof(double)) il.Emit(OpCodes.Ldc_R8, 0.5);
                else il.Emit(OpCodes.Ldc_R4, 0.5f);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(vk.Bits, vk.Clr), null);
                il.Emit(OpCodes.Stloc, half);
            }

            // [lanes] -> [lanes / 2].
            void Halve()
            {
                if (prep is not null)
                {
                    il.Emit(OpCodes.Ldloca, prep);
                    il.EmitCall(OpCodes.Call, PolyLaneOps.s_cDivBy, null);
                }
                else if (t == NPTypeCode.Half)
                {
                    // HALF_divide by 2 straight onto the f16 grid (the halves must be float16 values before NumPy adds them).
                    il.EmitCall(OpCodes.Call, PolyLaneOps.s_halfHalveOnGrid, null);
                }
                else
                {
                    il.Emit(OpCodes.Ldloc, half);
                    vk.Bin(il, BinaryOp.Multiply);
                }
            }

            // [a, b] -> [a + b]. float16: a PLAIN float32 add — HALF_add is the float32 sum rounded once to float16, and the
            // store's narrow is that rounding (the kind's Bin would round to the grid here and again at the store).
            void Add()
            {
                if (t == NPTypeCode.Half) PolyLanes.EmitVecOperator(il, BinaryOp.Add, vk.LocalType);
                else vk.Bin(il, BinaryOp.Add);
            }

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, cm);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, cp);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, prd);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, j);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I8, (long)(w - 1));
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, stop);
            var top = il.DefineLabel();
            var done = il.DefineLabel();
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldloc, stop);
            il.Emit(OpCodes.Bgt, done);
            EmitPolyCalcAddr(il, cm, j, size);
            vk.Load(il);
            Halve();
            EmitPolyCalcAddr(il, cp, j, size);
            vk.Load(il);
            Halve();
            Add();
            EmitPolyCalcAddr(il, prd, j, size);
            vk.StoreValueFirst(il);
            EmitPolyCalcBump(il, j, w);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(done);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ret);
            return dm;
        }
    }
}
