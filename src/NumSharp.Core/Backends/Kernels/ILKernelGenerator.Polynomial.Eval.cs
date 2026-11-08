using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Polynomial.Eval.cs — the {p}val Tier-3A kernel (Horner / Clenshaw)
// =============================================================================
//
// CONTRACT (NDInnerLoopFunc — the per-chunk model)
// ------------------------------------------------
//   void(void** dataptrs, long* strides, long count, void* aux)
//
//   operands, in order, only those that exist for the call's form:
//     x   (tx)  — when x is a per-point array (PolyXMode.PerPoint)
//     c0  (tc)  — when the coefficients are N-D (a view of c[0]; c[k] is read at c0 + k*kstride)
//     y   (tl)  — the result
//   aux (long slots):
//     [0] kstride  coefficient byte stride along the series axis
//     [1] nc       coefficient count
//     [2] table    region table of the constant pool (PolyConstPool.Prepare)
//     [3] cbase    1-D coefficients: base address (the coefficients live in aux, not in an operand)
//     [4] cbaseL   1-D coefficients pre-converted to the loop dtype (PolyEvalKernel.Preconv)
//     [5] kstrideL their byte stride
//     [6] xaddr    PolyXMode.Shared: address of the 0-d x element
//
// DISPATCH (once per chunk)
// -------------------------
// Vector chains need x (when per point) and y element-contiguous in the chunk; the coefficients are
// then broadcast (one series for the chunk: 1-D c, or c0's inner stride 0) or per lane (contiguous
// c0: one series per point — tensor=False, _valnd). Everything else runs 4 interleaved scalar chains
// over the byte strides. The recurrence reads x ONCE and writes y ONCE per point; NumPy runs about three
// full-array ops, each allocating a temporary, per coefficient.
//
// The kernel is a small dispatcher (the NDInnerLoopFunc) that reads the chunk's strides and calls one
// PART — vector/scalar x broadcast/per-lane series — and every part is two DynamicMethods of its own:
// the U-chain stage, which hands the index it reached to the remainder stage (1-chain blocks and, for
// a vector part, the scalar tail). The split is for the JIT, not for structure: a method stops inlining
// once its per-method inline budget is spent (its local-variable limit and time budget), and one method
// holding every block left the lane helpers as calls (EmitPolyEval's remarks carry the measurements).
//
// WHY THE N-D FORM IS NEVER BUFFERED
// ----------------------------------
// The kernel reads c[k] at (c0 pointer + k*kstride), i.e. OUTSIDE the c0 operand the iterator knows
// about. A buffered iterator would hand the kernel a copy of c0 alone, and the k offsets would read the
// buffer's neighbours. The 1-D form carries the coefficients in aux, so it may buffer a strided x
// (measured 2.7-3.0x over strided scalar chains).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>How the caller's x reaches a polynomial evaluation kernel.</summary>
    internal enum PolyXMode : byte
    {
        /// <summary>An array with at least one dimension: a per-point NDIter operand.</summary>
        PerPoint,
        /// <summary>A 0-d array: a strong scalar read once from aux slot 6 (NumPy still sees an ndarray).</summary>
        Shared,
        /// <summary>A Python scalar (C# primitive): folded into weak constants; there is no x value in IL.</summary>
        Weak,
    }

    /// <summary>
    ///     NumPy's single-element broadcast: an N-D series at a per-point x whose result has ONE element, where a
    ///     coefficient read <c>c[k]</c> and x differ in ndim. NumPy's trivial ufunc loop refuses non-0-d operands
    ///     of different ndim, so each such multiply runs on NpyIter, whose one-element iteration gives every
    ///     operand stride 0 — and <c>CDOUBLE_multiply</c> then takes its MSVC-contracted fallback loop
    ///     (<see cref="PolyComplexProduct.LoopScalar"/>) instead of <c>simd_cmul</c>. Which products that hits
    ///     depends on which side is deeper, because a value that has absorbed x has the deeper ndim of the two.
    /// </summary>
    /// <remarks>
    ///     Only complex products are affected (a real product is one IEEE multiply on every path, and a
    ///     real-cast operand makes every form agree), so <c>NDPolyEval</c> sets a mode only for a complex loop.
    ///     Measured against NumPy 2.4.2 for every basis over 8 single-element shape configurations x 7
    ///     coefficient counts (docs/plans/numpy-polynomial-review.md).
    /// </remarks>
    internal enum PolyUnitBroadcast : byte
    {
        /// <summary>No single-element broadcast: every array product is <c>simd_cmul</c> (or scalar math).</summary>
        None,
        /// <summary><c>ndim(c[k]) &gt; x.ndim</c> — every <c>tensor=True</c> case, and <c>tensor=False</c> with a
        ///     deeper series: every product of a coefficient-derived value and an x-derived value takes the
        ///     fallback loop, at every step.</summary>
        CoefDeeper,
        /// <summary><c>ndim(c[k]) &lt; x.ndim</c> (<c>tensor=False</c>): only products whose coefficient-derived
        ///     operand has not absorbed x yet — Clenshaw's first step (or the final product for two coefficients).
        ///     Later values have x's ndim, and NumPy's trivial loop runs them.</summary>
        PointsDeeper,
    }

    /// <summary>
    ///     Which of NumPy's two multiply.reduce loops <c>polyvalfromroots</c>' <c>np.prod(x - r, axis=0)</c> runs — decided
    ///     by the reduction's iteration, never by the values. Only complex128 and float16 products differ between them (a
    ///     float32 / float64 / decimal product is one rounding per step on both), so the driver sets
    ///     <see cref="Reduce"/> only for those dtypes.
    /// </summary>
    /// <remarks>
    ///     <para>NumPy's reduction iterator sorts the axes of the <c>x - r</c> temporary by its strides (the allocated output
    ///     cannot vote). When the roots axis is the innermost non-unit axis, or the result has one element, the inner
    ///     loop walks the roots with the output's stride 0 — <c>IS_BINARY_REDUCE</c>: <c>CDOUBLE_multiply</c> jumps to its
    ///     MSVC-contracted fallback loop, and <c>HALF_multiply</c> accumulates the whole call in float32 and narrows ONCE
    ///     (<c>io1 *= npy_half_to_float(in2)</c>; one call holds every root of one output). Otherwise the inner loop walks
    ///     the outputs and each root is one elementwise <c>out = out * t[k]</c>: <c>simd_cmul</c>, and a float16 narrow per
    ///     step.</para>
    ///     <para>Measured against NumPy 2.4.2: float16 at a Python-scalar x (2,000 / 2,000 the float32 accumulation), at a
    ///     one-point x (500 / 500), tensor=False with an F-ordered 2-D r (1,000 / 1,000); at seven points (1,400 / 1,400)
    ///     and a C-ordered r (1,000 / 1,000) the per-step narrow.</para>
    /// </remarks>
    internal enum PolyRootForm : byte
    {
        /// <summary>Not a root product (<c>{p}val</c>'s Horner / Clenshaw).</summary>
        None,
        /// <summary>The inner loop walks the outputs: one elementwise multiply per root (<c>simd_cmul</c>; a float16
        ///     product narrowed every step).</summary>
        Slab,
        /// <summary>The inner loop walks the roots (the output's stride 0): the complex fallback loop, a float16 product
        ///     accumulated in float32 and narrowed once.</summary>
        Reduce,
    }

    /// <summary>
    ///     The identity of one compiled evaluation kernel. <see cref="Cls"/> is the coefficient-count class
    ///     (<see cref="PolyTyping.Class"/>): every count at or above <c>peel + 3</c> shares one kernel.
    /// </summary>
    /// <param name="Basis">The basis.</param>
    /// <param name="XMode">How x arrives.</param>
    /// <param name="Tx">x dtype (<see cref="NPTypeCode.Empty"/> for a weak x).</param>
    /// <param name="WeakKind">Python type of a weak x (<see cref="PyKind.Int"/> otherwise, unused).</param>
    /// <param name="Tc">Coefficient dtype.</param>
    /// <param name="Cls">Coefficient-count class.</param>
    /// <param name="CoefOperand">The coefficients are an N-D operand (else 1-D, in aux).</param>
    /// <param name="ScalarMath">0-d result: NumPy runs every op as scalar math (the naive complex product).</param>
    /// <param name="Unit">NumPy's single-element broadcast (the fallback complex product where operand ndims differ).</param>
    /// <param name="Root"><c>polyvalfromroots</c>' root product and its reduction loop (<see cref="PolyRootForm.None"/>
    ///     for <c>{p}val</c>); the "coefficients" are then the roots.</param>
    internal readonly record struct PolyEvalKey(PolyBasis Basis, PolyXMode XMode, NPTypeCode Tx, PyKind WeakKind,
        NPTypeCode Tc, int Cls, bool CoefOperand, bool ScalarMath, PolyUnitBroadcast Unit, PolyRootForm Root);

    /// <summary>A compiled evaluation kernel plus what its caller needs to run it.</summary>
    internal sealed class PolyEvalKernel
    {
        /// <summary>The Tier-3A inner loop.</summary>
        public NDInnerLoopFunc Fn;
        /// <summary>Its weak-value regions.</summary>
        public PolyConstPool Pool;
        /// <summary>NumPy's result dtype for this (x, coefficient dtype, count class).</summary>
        public NPTypeCode Tl;
        /// <summary>The caller must pass the 1-D coefficients pre-converted to <see cref="Tl"/> in aux [4]/[5].</summary>
        public bool Preconv;
    }

    public static partial class ILKernelGenerator
    {
        /// <summary>Interleaved chains per block — 4 hides the add/multiply latency of one recurrence step.</summary>
        private const int PolyUnroll = 4;

        private static readonly ConcurrentDictionary<(PolyBasis, PolyXMode, NPTypeCode, PyKind, NPTypeCode, PolyUnitBroadcast), int> s_polyPeel = new();
        private static readonly ConcurrentDictionary<PolyEvalKey, Lazy<PolyEvalKernel>> s_polyEval = new();

        /// <summary>
        ///     The evaluation program a call runs: the basis's own table (or <c>polyvalfromroots</c>' root product) for an
        ///     array x, the Python-folded one for a scalar x.
        /// </summary>
        /// <param name="basis">The basis.</param><param name="xMode">How x arrives.</param>
        /// <param name="weakKind">Python type of a weak x.</param>
        /// <param name="root">The root product's reduction form, or <see cref="PolyRootForm.None"/> for <c>{p}val</c>.</param>
        /// <returns>The program.</returns>
        internal static PolyEvalProgram PolyProgram(PolyBasis basis, PolyXMode xMode, PyKind weakKind, PolyRootForm root = PolyRootForm.None)
            => root != PolyRootForm.None
                ? (xMode == PolyXMode.Weak ? PolySteps.RootsForWeakX(weakKind) : PolySteps.Roots)
                : xMode == PolyXMode.Weak ? PolySteps.RewriteForWeakX(basis, weakKind) : PolySteps.Eval(basis);

        /// <summary>The peel count of a (basis, x, coefficient dtype, broadcast mode) combination, computed once.</summary>
        /// <param name="basis">The basis.</param><param name="xMode">How x arrives.</param>
        /// <param name="tx">x dtype (ignored when weak).</param><param name="weakKind">Python type of a weak x.</param>
        /// <param name="tc">Coefficient dtype.</param>
        /// <param name="unit">Single-element broadcast mode: its ndim state joins the dtype fixpoint.</param>
        /// <returns>The peel count.</returns>
        internal static int PolyPeel(PolyBasis basis, PolyXMode xMode, NPTypeCode tx, PyKind weakKind, NPTypeCode tc, PolyUnitBroadcast unit)
            => s_polyPeel.GetOrAdd((basis, xMode, tx, weakKind, tc, unit),
                k => PolyTyping.PeelCount(PolyProgram(k.Item1, k.Item2, k.Item4), k.Item3, k.Item5, k.Item6));

        /// <summary>
        ///     Returns (compiling once) the evaluation kernel for a call. Thread-safe: concurrent first calls
        ///     compile exactly one kernel (a <see cref="Lazy{T}"/> per key), so the kernel and its constant pool
        ///     can never come from two different emissions.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="xMode">How x arrives.</param>
        /// <param name="tx">x dtype (ignored when weak).</param>
        /// <param name="weakKind">Python type of a weak x (ignored otherwise).</param>
        /// <param name="tc">Coefficient dtype (never int-like: <c>{p}val</c> converts those to float64 first).</param>
        /// <param name="nc">Coefficient count (≥ 1; only its class matters).</param>
        /// <param name="coefOperand">N-D coefficients (operand form) instead of 1-D (aux form).</param>
        /// <param name="scalarMath">The result is 0-d (NumPy's scalar-math complex product).</param>
        /// <param name="unit">NumPy's single-element broadcast mode (<see cref="PolyUnitBroadcast"/>; None almost always).</param>
        /// <param name="root"><c>polyvalfromroots</c>: the root product and NumPy's reduction loop for it (the
        ///     "coefficients" are the roots, <paramref name="nc"/> their count, 0 allowed); <see cref="PolyRootForm.None"/>
        ///     for <c>{p}val</c>.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        /// <exception cref="ArgumentException"><paramref name="unit"/> is set for a call that is not an N-D series at
        ///     a per-point x — a caller bug (only that form can be a single-element NpyIter broadcast); or a root product
        ///     asked for scalar math or a single-element broadcast (its products are always np.prod's reduction loops).</exception>
        internal static PolyEvalKernel GetPolyEvalKernel(PolyBasis basis, PolyXMode xMode, NPTypeCode tx, PyKind weakKind,
            NPTypeCode tc, long nc, bool coefOperand, bool scalarMath, PolyUnitBroadcast unit, PolyRootForm root = PolyRootForm.None)
        {
            if (unit != PolyUnitBroadcast.None && !(coefOperand && xMode == PolyXMode.PerPoint))
                throw new ArgumentException("a single-element broadcast needs an N-D series at a per-point x", nameof(unit));
            if (root != PolyRootForm.None && (scalarMath || unit != PolyUnitBroadcast.None))
                throw new ArgumentException("a root product runs np.prod's reduction loops: no scalar math, no single-element broadcast", nameof(root));
            if (xMode == PolyXMode.Weak) tx = NPTypeCode.Empty; else weakKind = PyKind.Int;
            var prog = PolyProgram(basis, xMode, weakKind, root);
            // The root product needs no peel: every factor is x - r[k] in one dtype from the first root on.
            int peel = root != PolyRootForm.None ? 0 : PolyPeel(basis, xMode, tx, weakKind, tc, unit);
            int cls = PolyTyping.Class(prog, nc, peel);
            var key = new PolyEvalKey(basis, xMode, tx, weakKind, tc, cls, coefOperand, scalarMath, unit, root);
            return s_polyEval.GetOrAdd(key, k => new Lazy<PolyEvalKernel>(() => CompilePolyEval(k, prog, peel))).Value;
        }

        /// <summary>
        ///     Emits and compiles one evaluation kernel — with vector blocks wherever the lane gate allows them,
        ///     falling back to a scalar-chains-only kernel when a vector lane kind is missing.
        /// </summary>
        /// <param name="key">The kernel identity.</param><param name="prog">Its program.</param><param name="peel">Its peel count.</param>
        /// <returns>The kernel with its sealed constant pool.</returns>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code.</exception>
        /// <exception cref="NotSupportedException">The scalar path itself cannot handle a dtype.</exception>
        private static PolyEvalKernel CompilePolyEval(PolyEvalKey key, PolyEvalProgram prog, int peel)
        {
            if (!System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported)
                throw new PlatformNotSupportedException(
                    "numpy.polynomial evaluation compiles IL kernels at runtime; this runtime (NativeAOT) cannot emit dynamic code.");
            try
            {
                return EmitPolyEval(key, prog, peel, allowVector: true);
            }
            catch (NotSupportedException) when (!key.ScalarMath)
            {
                // A per-point dtype the lane table (PolyLanes) has no kind or conversion for: the vector blocks
                // cannot be emitted, so this kernel runs scalar chains only — the same bits, slower. The whole
                // emission restarts (a fresh DynamicMethod and constant pool), so nothing half-built survives.
                // Counted so a test can prove every NumPy dtype combination vectorizes (PolyVectorFallbacks).
                System.Threading.Interlocked.Increment(ref s_polyVectorFallbacks);
                return EmitPolyEval(key, prog, peel, allowVector: false);
            }
        }

        private static long s_polyVectorFallbacks;

        /// <summary>How many evaluation kernels fell back to scalar-only chains because a vector lane kind or
        ///     conversion was missing (see <see cref="CompilePolyEval"/>). Zero for every combination the lane
        ///     table covers — the tests assert it stays zero across NumPy's whole dtype matrix.</summary>
        internal static long PolyVectorFallbacks => System.Threading.Interlocked.Read(ref s_polyVectorFallbacks);

        /// <summary>The independently compiled parts of one evaluation kernel (see <see cref="EmitPolyEval"/>).</summary>
        private enum PolyPart
        {
            /// <summary>Vector chains over a chunk whose points share one series, then its scalar remainder.</summary>
            VecBcast,
            /// <summary>Vector chains over a chunk whose points each read their own series, then its remainder.</summary>
            VecPerLane,
            /// <summary>Scalar chains, one series for the chunk, any strides.</summary>
            ScalarBcast,
            /// <summary>Scalar chains, a series per point, any strides.</summary>
            ScalarPerLane,
        }

        /// <summary>
        ///     Emits and compiles one evaluation kernel: a small dispatcher that picks a part (<see cref="PolyPart"/>)
        ///     per chunk from the chunk's strides, and each part as TWO DynamicMethods of its own — the U-chain
        ///     stage over every complete group of U lanes-times-chains, which then calls the remainder stage
        ///     (1-chain vector blocks and the scalar tail) with the index it reached.
        /// </summary>
        /// <param name="key">The kernel identity.</param><param name="prog">Its program.</param><param name="peel">Its peel count.</param>
        /// <param name="allowVector">Emit vector parts where the gate allows them (false: scalar chains only).</param>
        /// <returns>The kernel with its sealed constant pool.</returns>
        /// <exception cref="NotSupportedException">A vector lane kind or conversion is missing (only when
        ///     <paramref name="allowVector"/>), or the scalar path cannot handle a dtype.</exception>
        /// <remarks>
        ///     Why so many methods: the JIT stops inlining in a method once its per-method inline budget is spent
        ///     — a local-variable limit (every inlined helper adds temporaries) and a time budget, AggressiveInlining
        ///     notwithstanding — so a single method holding every block of a peeled kernel leaves its hot helpers
        ///     as CALLS (which of the two limits bit was not isolated: the release JIT reports no inline reasons). Measured: the complex128-x / float64-series <c>legval</c>
        ///     kernel compiled to 23 KB with 340 calls left un-inlined — the lane helpers, <c>Vector256.Load/Store</c>
        ///     and the Complex operators among them — 3.3x slower than the same recurrence over complex128 series;
        ///     one method per part still left 95 calls in its float16-series twin (0.93x NumPy), whose peeled
        ///     float16 steps inline heavy narrow/widen helpers. One method per STAGE keeps every hot block inlined.
        ///     The hops cost two direct calls per chunk (dispatcher → stage, U-chain stage → remainder stage).
        ///     All stages register their weak values in the one shared <see cref="PolyConstPool"/>, sealed last.
        /// </remarks>
        private static PolyEvalKernel EmitPolyEval(PolyEvalKey key, PolyEvalProgram prog, int peel, bool allowVector)
        {
            var pool = new PolyConstPool();
            var tl = PolyTyping.ResultType(prog, key.Tx, key.Tc, key.Cls, peel);
            bool preconv = !key.CoefOperand && key.Tc != tl && !key.ScalarMath;
            bool xPerPoint = key.XMode == PolyXMode.PerPoint;
            // A 0-d result (ScalarMath) and a single-element broadcast (Unit) are one point: no vector part could
            // run, so none is emitted — which is also what keeps their non-simd products off the vector kinds.
            bool onePoint = key.ScalarMath || key.Unit != PolyUnitBroadcast.None;
            // A reduce-form root product keeps its fallback complex product and its float32 float16 accumulator on scalar
            // chains (a vector kind emits simd_cmul). Its many-point calls read the roots at a stride (the roots axis is
            // the innermost of the temporary, so one point's roots are never adjacent to the next point's) — the vector
            // gate would refuse those chunks anyway.
            bool reduceForm = key.Root == PolyRootForm.Reduce;
            bool vecB = allowVector && !onePoint && !reduceForm && PolyLanes.VectorBlockOk(key.Tx, xPerPoint, key.Tc, perLaneCoef: false, tl);
            bool vecL = allowVector && !onePoint && !reduceForm && key.CoefOperand && PolyLanes.VectorBlockOk(key.Tx, xPerPoint, key.Tc, perLaneCoef: true, tl);
            string name = $"NDPolyEval_{key.Basis}_{key.XMode}_{key.Tx}_{key.WeakKind}_{key.Tc}_c{key.Cls}_{(key.CoefOperand ? "op" : "aux")}"
                          + $"{(key.ScalarMath ? "_s" : "")}{(key.Unit != PolyUnitBroadcast.None ? "_u" + key.Unit : "")}"
                          + $"{(key.Root != PolyRootForm.None ? "_roots" + key.Root : "")}";

            PolyEvalCtx Ctx() => new PolyEvalCtx { Key = key, Prog = prog, Tl = tl, P = peel, Pool = pool, Preconv = preconv };
            DynamicMethod Part(PolyPart part)
            {
                bool vec = part is PolyPart.VecBcast or PolyPart.VecPerLane;
                bool bcast = part is PolyPart.VecBcast or PolyPart.ScalarBcast;
                var rest = NewPolyDm($"{name}_{part}_rest", stage: true);
                EmitPolyStage(rest.GetILGenerator(), Ctx(), vec, bcast, unrolled: false, next: null);
                if (onePoint) return rest;   // one point: no U-chain stage
                var main = NewPolyDm($"{name}_{part}", stage: true);
                EmitPolyStage(main.GetILGenerator(), Ctx(), vec, bcast, unrolled: true, next: rest);
                return main;
            }
            var scB = Part(PolyPart.ScalarBcast);
            var scL = key.CoefOperand ? Part(PolyPart.ScalarPerLane) : null;
            var vB = vecB ? Part(PolyPart.VecBcast) : null;
            var vL = vecL ? Part(PolyPart.VecPerLane) : null;
            var root = NewPolyDm(name, stage: false);
            EmitPolyDispatch(root.GetILGenerator(), key, tl, vB, vL, scB, scL);
            var fn = root.CreateDelegate<NDInnerLoopFunc>();
            pool.Seal();
            return new PolyEvalKernel { Fn = fn, Pool = pool, Tl = tl, Preconv = preconv };
        }

        /// <summary>
        ///     A DynamicMethod with the per-chunk inner-loop signature (<see cref="NDInnerLoopFunc"/>), or a STAGE:
        ///     the same four arguments plus the element index to start at (<c>long start</c>, argument 4).
        /// </summary>
        /// <param name="name">Its name (shows in JIT disassembly and profiles).</param>
        /// <param name="stage">A stage (five arguments) rather than the kernel's entry point.</param>
        /// <returns>The method.</returns>
        private static DynamicMethod NewPolyDm(string name, bool stage) => new DynamicMethod(
            name: name,
            returnType: typeof(void),
            parameterTypes: stage
                ? new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*), typeof(long) }
                : new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
            owner: typeof(ILKernelGenerator),
            skipVisibility: true);

        /// <summary>
        ///     The dispatcher: reads the chunk's strides and forwards the four arguments (and start 0) to the part
        ///     that serves them — a vector part when x (per point) and y are element-contiguous and the coefficients are
        ///     broadcast (stride 0) or one series per point (stride = element size); a scalar part otherwise.
        /// </summary>
        /// <param name="il">Generator.</param><param name="key">The kernel identity.</param><param name="tl">Loop dtype.</param>
        /// <param name="vB">Vector broadcast-series part, or null.</param><param name="vL">Vector per-point-series part, or null.</param>
        /// <param name="scB">Scalar broadcast-series part.</param><param name="scL">Scalar per-point-series part, or null.</param>
        private static void EmitPolyDispatch(ILGenerator il, PolyEvalKey key, NPTypeCode tl,
            DynamicMethod vB, DynamicMethod vL, DynamicMethod scB, DynamicMethod scL)
        {
            int op = 0;
            int xOp = key.XMode == PolyXMode.PerPoint ? op++ : -1;
            int cOp = key.CoefOperand ? op++ : -1;
            int yOp = op;
            var sx = il.DeclareLocal(typeof(long)); var sy = il.DeclareLocal(typeof(long)); var sc = il.DeclareLocal(typeof(long));
            if (xOp >= 0) PolyLdSlot(il, 1, xOp, sx);
            PolyLdSlot(il, 1, yOp, sy);
            if (cOp >= 0) PolyLdSlot(il, 1, cOp, sc);
            else { il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, sc); }

            void Forward(DynamicMethod target)
            {
                il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldc_I8, 0L);   // the chunk from its first element
                il.Emit(OpCodes.Call, target);
                il.Emit(OpCodes.Ret);
            }

            var scalar = il.DefineLabel();
            if (vB is not null || vL is not null)
            {
                if (xOp >= 0)
                {
                    il.Emit(OpCodes.Ldloc, sx); il.Emit(OpCodes.Ldc_I8, (long)DirectILKernelGenerator.GetTypeSize(key.Tx));
                    il.Emit(OpCodes.Bne_Un, scalar);
                }
                il.Emit(OpCodes.Ldloc, sy); il.Emit(OpCodes.Ldc_I8, (long)DirectILKernelGenerator.GetTypeSize(tl));
                il.Emit(OpCodes.Bne_Un, scalar);
                var perLane = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, sc); il.Emit(OpCodes.Brtrue, perLane);
                if (vB is not null) Forward(vB); else il.Emit(OpCodes.Br, scalar);
                il.MarkLabel(perLane);
                if (vL is not null)
                {
                    // Coefficient stride = element size: one contiguous series per point, lane-loadable.
                    il.Emit(OpCodes.Ldloc, sc); il.Emit(OpCodes.Ldc_I8, (long)DirectILKernelGenerator.GetTypeSize(key.Tc));
                    il.Emit(OpCodes.Bne_Un, scalar);
                    Forward(vL);
                }
                else il.Emit(OpCodes.Br, scalar);
            }
            il.MarkLabel(scalar);
            if (scL is not null)
            {
                var bcast = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, sc); il.Emit(OpCodes.Brfalse, bcast);
                Forward(scL);
                il.MarkLabel(bcast);
            }
            Forward(scB);
        }

        /// <summary>Locals and static facts shared by every block of one evaluation kernel.</summary>
        private sealed class PolyEvalCtx
        {
            /// <summary>The kernel identity.</summary>
            public PolyEvalKey Key;
            /// <summary>The step tables it runs.</summary>
            public PolyEvalProgram Prog;
            /// <summary>The loop (result) dtype.</summary>
            public NPTypeCode Tl;
            /// <summary>The peel count (Clenshaw steps emitted straight-line before the loop).</summary>
            public int P;
            /// <summary>The weak-value pool every stage of the kernel registers into.</summary>
            public PolyConstPool Pool;
            /// <summary>The steady loop reads the 1-D coefficients pre-converted to <see cref="Tl"/> (aux slots 4/5).</summary>
            public bool Preconv;
            /// <summary>Operand slots (-1 when absent).</summary>
            public int XOp, COp, YOp;
            /// <summary>Chunk pointers (x, coefficient base, y), the region table and the shared-x address.</summary>
            public LocalBuilder Xp, Cp, Yp, Table, XAddr;
            /// <summary>Chunk byte strides (x, coefficients, y), the series-axis byte stride and nc.</summary>
            public LocalBuilder Sx, Sc, Sy, KStride, Nc;
            /// <summary>The current recurrence row (nd) and the element index within the chunk.</summary>
            public LocalBuilder Row, I;
            /// <summary>The pre-converted 1-D coefficients' base and stride (when <see cref="Preconv"/>).</summary>
            public LocalBuilder CpL, KStrideL;
        }

        /// <summary>How the coefficients of one block are read.</summary>
        private sealed class PolyCoefCtx
        {
            /// <summary>One series for the whole chunk (1-D c, or c0's inner stride 0).</summary>
            public bool Broadcast;
            /// <summary>The chunk's coefficient pointer (the series of its first point).</summary>
            public LocalBuilder Cp;
            /// <summary>Per-lane mode: each chain's series base for the current block iteration.</summary>
            public LocalBuilder[] CBase;
            /// <summary>The coefficient dtype.</summary>
            public NPTypeCode Tc;
        }

        /// <summary>
        ///     One stage of a part (see <see cref="EmitPolyEval"/> and the file header for the operand/aux
        ///     contract): the prologue (starting at element <c>start</c>), then either the U-chain block followed by
        ///     a call of the remainder stage at the index it reached, or the remainder itself — the 1-chain vector
        ///     block and the scalar tail for a vector part, the 1-chain scalar block for a scalar part.
        /// </summary>
        /// <param name="il">Generator.</param><param name="ctx">Static facts (locals are declared here).</param>
        /// <param name="vec">A vector part.</param><param name="bcast">Broadcast (shared) series.</param>
        /// <param name="unrolled">The U-chain stage (true) or the remainder stage (false).</param>
        /// <param name="next">The remainder stage the U-chain stage hands over to (null for the remainder stage).</param>
        private static void EmitPolyStage(ILGenerator il, PolyEvalCtx ctx, bool vec, bool bcast, bool unrolled, DynamicMethod next)
        {
            EmitPolyPrologue(il, ctx);
            var end = il.DefineLabel();
            if (unrolled)
            {
                var rest = il.DefineLabel();
                EmitPolyEvalBlock(il, ctx, vec, bcast, PolyUnroll, rest);
                il.MarkLabel(rest);
                il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, ctx.I);
                il.Emit(OpCodes.Call, next);
            }
            else if (vec)
            {
                var tail = il.DefineLabel();
                EmitPolyEvalBlock(il, ctx, vec: true, bcast, 1, tail);
                il.MarkLabel(tail);
                EmitPolyEvalBlock(il, ctx, vec: false, bcast, 1, end);   // the < W remainder
            }
            else
            {
                EmitPolyEvalBlock(il, ctx, vec: false, bcast, 1, end);
            }
            il.MarkLabel(end);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>Declares a stage's locals and loads the operand pointers, strides and auxdata slots into them;
        ///     the element index starts at the stage's <c>start</c> argument.</summary>
        /// <param name="il">Generator.</param><param name="ctx">Static facts; its locals are set here.</param>
        private static void EmitPolyPrologue(ILGenerator il, PolyEvalCtx ctx)
        {
            var key = ctx.Key;
            int op = 0;
            ctx.XOp = key.XMode == PolyXMode.PerPoint ? op++ : -1;
            ctx.COp = key.CoefOperand ? op++ : -1;
            ctx.YOp = op;

            LocalBuilder Ptr() => il.DeclareLocal(typeof(byte*));
            LocalBuilder Lng() => il.DeclareLocal(typeof(long));
            ctx.Xp = Ptr(); ctx.Cp = Ptr(); ctx.Yp = Ptr(); ctx.Table = Ptr(); ctx.XAddr = Ptr();
            ctx.Sx = Lng(); ctx.Sc = Lng(); ctx.Sy = Lng(); ctx.KStride = Lng(); ctx.Nc = Lng(); ctx.Row = Lng(); ctx.I = Lng();
            if (ctx.XOp >= 0) { PolyLdSlot(il, 0, ctx.XOp, ctx.Xp); PolyLdSlot(il, 1, ctx.XOp, ctx.Sx); }
            if (ctx.COp >= 0) { PolyLdSlot(il, 0, ctx.COp, ctx.Cp); PolyLdSlot(il, 1, ctx.COp, ctx.Sc); }
            else { PolyLdSlot(il, 3, 3, ctx.Cp); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, ctx.Sc); }
            PolyLdSlot(il, 0, ctx.YOp, ctx.Yp); PolyLdSlot(il, 1, ctx.YOp, ctx.Sy);
            PolyLdSlot(il, 3, 0, ctx.KStride); PolyLdSlot(il, 3, 1, ctx.Nc); PolyLdSlot(il, 3, 2, ctx.Table);
            if (key.XMode == PolyXMode.Shared) PolyLdSlot(il, 3, 6, ctx.XAddr);
            if (ctx.Preconv)
            {
                ctx.CpL = Ptr(); ctx.KStrideL = Lng();
                PolyLdSlot(il, 3, 4, ctx.CpL); PolyLdSlot(il, 3, 5, ctx.KStrideL);
            }
            il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Stloc, ctx.I);
        }

        /// <summary>dst = ((long*)arg)[slot] — reads a dataptr, stride or auxdata slot into a local.</summary>
        /// <param name="il">Generator.</param><param name="arg">0 dataptrs, 1 strides, 3 auxdata.</param>
        /// <param name="slot">Slot index.</param><param name="dst">Pointer or long local.</param>
        private static void PolyLdSlot(ILGenerator il, int arg, int slot, LocalBuilder dst)
        {
            il.Emit(arg switch { 0 => OpCodes.Ldarg_0, 1 => OpCodes.Ldarg_1, 2 => OpCodes.Ldarg_2, _ => OpCodes.Ldarg_3 });
            il.Emit(OpCodes.Ldc_I4, slot * 8); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            if (dst.LocalType == typeof(byte*)) il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Stloc, dst);
        }

        /// <summary>Pushes <c>basePtr + (i + laneOffset) * stride</c> (stride from a local, or a constant when the
        ///     block runs on proven-contiguous data).</summary>
        /// <param name="il">Generator.</param><param name="basePtr">Base pointer local.</param><param name="i">Element index local.</param>
        /// <param name="laneOffset">Element offset of the chain.</param><param name="strideLocal">Runtime stride, or null.</param>
        /// <param name="strideConst">Constant stride when <paramref name="strideLocal"/> is null.</param>
        private static void PolyAddr(ILGenerator il, LocalBuilder basePtr, LocalBuilder i, long laneOffset, LocalBuilder strideLocal, long strideConst)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, i);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            if (strideLocal is not null) il.Emit(OpCodes.Ldloc, strideLocal); else il.Emit(OpCodes.Ldc_I8, strideConst);
            il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
        }

        /// <summary>One loop block: while U more lane groups fit, read U x values (or the shared x), run the
        ///     recurrence as U interleaved chains, store U results.</summary>
        /// <param name="il">Generator.</param><param name="ctx">Kernel facts.</param><param name="vec">Vector chains.</param>
        /// <param name="bcast">Broadcast coefficients.</param><param name="U">Chains.</param><param name="exit">Where to go when fewer remain.</param>
        /// <exception cref="InvalidOperationException">The emitted result dtype differs from the planned one (a typing bug).</exception>
        private static void EmitPolyEvalBlock(ILGenerator il, PolyEvalCtx ctx, bool vec, bool bcast, int U, Label exit)
        {
            var key = ctx.Key;
            var em = new PolyEmitter(il, U, vec, PolyLanes.LoopLanes(ctx.Tl), ctx.Pool, ctx.Table, ctx.Row, key.ScalarMath, key.Unit,
                reduceProduct: key.Root == PolyRootForm.Reduce);
            int W = em.W;
            int szC = DirectILKernelGenerator.GetTypeSize(key.Tc), szL = DirectILKernelGenerator.GetTypeSize(ctx.Tl);
            var top = il.DefineLabel();
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, ctx.I); il.Emit(OpCodes.Ldc_I8, (long)(U * W)); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bgt, exit);

            var cc = new PolyCoefCtx { Broadcast = bcast, Cp = ctx.Cp, Tc = key.Tc };
            if (!bcast)
            {
                cc.CBase = new LocalBuilder[U];
                for (int u = 0; u < U; u++)
                {
                    cc.CBase[u] = il.DeclareLocal(typeof(byte*));
                    PolyAddr(il, ctx.Cp, ctx.I, u * W, vec ? null : ctx.Sc, szC);
                    il.Emit(OpCodes.Stloc, cc.CBase[u]);
                }
            }

            var env = new PolyEnv();
            switch (key.XMode)
            {
                case PolyXMode.PerPoint:
                {
                    int szX = DirectILKernelGenerator.GetTypeSize(key.Tx);
                    env.X = em.Fresh(key.Tx, false, em.PointNd);   // x's ndim (single-element broadcast; 0 otherwise)
                    for (int u = 0; u < U; u++)
                    {
                        PolyAddr(il, ctx.Xp, ctx.I, u * W, vec ? null : ctx.Sx, szX);
                        em.Lane(key.Tx).Load(il);
                        il.Emit(OpCodes.Stloc, env.X.L[u]);
                    }
                    break;
                }
                case PolyXMode.Shared:
                    env.X = em.Fresh(key.Tx, true);
                    il.Emit(OpCodes.Ldloc, ctx.XAddr);
                    em.Scalar(key.Tx).Load(il);
                    il.Emit(OpCodes.Stloc, env.X.L[0]);
                    break;
            }

            // The steady loop's coefficient source: the pre-converted copy when there is one (1-D form only).
            var ccLoop = ctx.Preconv ? new PolyCoefCtx { Broadcast = true, Cp = ctx.CpL, Tc = ctx.Tl } : cc;
            var loopStride = ctx.Preconv ? ctx.KStrideL : ctx.KStride;
            var y = ctx.Prog.RootProduct ? EmitPolyRootProduct(em, ctx, ccLoop, loopStride, env)
                  : ctx.Prog.Horner ? EmitPolyHorner(em, ctx, cc, ccLoop, loopStride, env)
                  : EmitPolyClenshaw(em, ctx, cc, ccLoop, loopStride, env);
            if (y.T != ctx.Tl) throw new InvalidOperationException($"planned {ctx.Tl}, emitted {y.T}");
            if (y.Shared) { var yl = em.Fresh(ctx.Tl, false); em.CopyTo(y, yl); y = yl; }   // result independent of the points
            for (int u = 0; u < U; u++)
            {
                il.Emit(OpCodes.Ldloc, y.L[u]);
                PolyAddr(il, ctx.Yp, ctx.I, u * W, vec ? null : ctx.Sy, szL);
                em.Lane(ctx.Tl).StoreValueFirst(il);
            }
            il.Emit(OpCodes.Ldloc, ctx.I); il.Emit(OpCodes.Ldc_I8, (long)(U * W)); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, ctx.I);
            il.Emit(OpCodes.Br, top);
        }

        /// <summary>Loads coefficient <c>c[k]</c> (k given as a byte offset along the series axis): one SHARED
        ///     scalar for a broadcast series, or one lane load per chain from that chain's own series.</summary>
        /// <param name="em">Emitter.</param><param name="cc">Coefficient context.</param><param name="kOff">Byte offset local.</param>
        /// <returns>The coefficient value (dtype = the context's coefficient dtype).</returns>
        private static PolyValue PolyLoadCoef(PolyEmitter em, PolyCoefCtx cc, LocalBuilder kOff)
        {
            var il = em.IL;
            // Every read is an array c[k] of the series' ndim (0 outside a single-element broadcast kernel).
            if (cc.Broadcast)
            {
                var r = em.Fresh(cc.Tc, true, em.CoefNd);
                il.Emit(OpCodes.Ldloc, cc.Cp); il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                em.Scalar(cc.Tc).Load(il);
                il.Emit(OpCodes.Stloc, r.L[0]);
                return r;
            }
            var k = em.Lane(cc.Tc);
            var v = em.Fresh(cc.Tc, false, em.CoefNd);
            if (em.Vec && cc.Tc == NPTypeCode.Half && em.W < 8 && em.U * em.W % 8 == 0)
            {
                // Gang load: float16 series in a float64 (w = 4) or complex128 (w = 2) loop. A float16 lane value
                // is 8 float32 lanes of which only w are live, and its exact widen costs the same for 8 as for w;
                // the chains' series are adjacent (a vector part runs only when the coefficient stride is the
                // element size), so ONE 16-byte load + widen serves 8/w chains, each taking its w lanes by one
                // 64-bit-lane permute. Measured on complex128 x with float16 per-point series: the per-step
                // widen per chain was the difference between 0.93-1.47x and the float64-series speed.
                int per = 8 / em.W;
                var gang = il.DeclareLocal(typeof(Vector256<float>));
                for (int g = 0; g < em.U / per; g++)
                {
                    il.Emit(OpCodes.Ldloc, cc.CBase[g * per]); il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    il.EmitCall(OpCodes.Call, PolyLaneOps.s_halfLoad8, null);
                    il.Emit(OpCodes.Stloc, gang);
                    for (int j = 0; j < per; j++)
                    {
                        il.Emit(OpCodes.Ldloc, gang);
                        if (j > 0)
                        {
                            // Bring chain j's lanes [j*w, j*w + w) down to lane 0: 64-bit lane j (w = 2) or the
                            // upper 128 bits (w = 4, j = 1), whatever lands above them is dead.
                            il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, typeof(float), typeof(double)), null);
                            il.Emit(OpCodes.Ldc_I4, em.W == 2 ? j : 0b11_10_11_10);
                            il.EmitCall(OpCodes.Call, PolyLaneOps.s_permute4x64, null);
                            il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, typeof(double), typeof(float)), null);
                        }
                        il.Emit(OpCodes.Stloc, v.L[g * per + j]);
                    }
                }
                return v;
            }
            for (int u = 0; u < em.U; u++)
            {
                il.Emit(OpCodes.Ldloc, cc.CBase[u]); il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                k.Load(il);
                il.Emit(OpCodes.Stloc, v.L[u]);
            }
            return v;
        }

        /// <summary>kOff = (nc - fromEnd) * kstride.</summary>
        /// <param name="il">Generator.</param><param name="ctx">Kernel facts.</param>
        /// <param name="kOff">Destination.</param><param name="fromEnd">Distance from the end.</param>
        private static void PolySetKOffFromEnd(ILGenerator il, PolyEvalCtx ctx, LocalBuilder kOff, long fromEnd)
        {
            il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, fromEnd); il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldloc, ctx.KStride); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, kOff);
        }

        /// <summary>
        ///     Clenshaw for one block — NumPy's <c>{p}val</c> body: classes 1 and 2 are its <c>len(c) == 1 / 2</c>
        ///     branches; otherwise the first <c>min(nc-2, P)</c> steps are straight-line (their dtypes differ)
        ///     and the rest run as an IL loop at the fixpoint dtypes.
        /// </summary>
        /// <param name="em">Emitter.</param><param name="ctx">Kernel facts.</param>
        /// <param name="cc">Coefficient context of the peeled/branch reads (original dtype).</param>
        /// <param name="ccLoop">Coefficient context of the steady loop (possibly pre-converted).</param>
        /// <param name="loopStride">Byte stride of the steady loop's coefficients.</param>
        /// <param name="env">Bound x.</param><returns>The result value.</returns>
        /// <exception cref="InvalidOperationException">The loop body changed a carried dtype or NumPy ndim (the peel
        ///     count is wrong).</exception>
        private static PolyValue EmitPolyClenshaw(PolyEmitter em, PolyEvalCtx ctx, PolyCoefCtx cc, PolyCoefCtx ccLoop, LocalBuilder loopStride, PolyEnv env)
        {
            var il = em.IL; var prog = ctx.Prog; int cls = ctx.Key.Cls;
            var kOff = il.DeclareLocal(typeof(long));
            if (cls <= 2)
            {
                // len(c) == 1: c0 = c[0], c1 = 0 (a Python int); len(c) == 2: c0 = c[0], c1 = c[1].
                var final = cls == 1 ? prog.FinalLen1 : prog.Final;
                if (prog.Pre is not null && (prog.PreAlways || PolySteps.Uses(final, PolySym.X2))) env.X2 = em.Emit(prog.Pre, env);
                il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, kOff);
                env.C0 = PolyLoadCoef(em, cc, kOff);
                if (cls == 2)
                {
                    il.Emit(OpCodes.Ldloc, ctx.KStride); il.Emit(OpCodes.Stloc, kOff);
                    env.C1 = PolyLoadCoef(em, cc, kOff);
                }
                return em.Emit(final, env);
            }
            if (prog.Pre is not null) env.X2 = em.Emit(prog.Pre, env);
            bool usesRow = PolySteps.UsesRow(prog.StepC0) || PolySteps.UsesRow(prog.StepC1);
            PolySetKOffFromEnd(il, ctx, kOff, 2); var c0 = PolyLoadCoef(em, cc, kOff);
            PolySetKOffFromEnd(il, ctx, kOff, 1); var c1 = PolyLoadCoef(em, cc, kOff);
            int straight = cls <= ctx.P + 2 ? cls - 2 : ctx.P;
            for (int p = 0; p < straight; p++)
            {
                // Step p reads c[nc-3-p] with nd = nc-1-p (NumPy's nd = nd - 1 precedes the step's lines).
                PolySetKOffFromEnd(il, ctx, kOff, 3 + p);
                if (usesRow) { il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, 1L + p); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, ctx.Row); }
                env.Ck = PolyLoadCoef(em, cc, kOff); env.Tmp = c0; env.C1 = c1;
                var n0 = em.Emit(prog.StepC0, env);
                var n1 = em.Emit(prog.StepC1, env);
                c0 = n0; c1 = n1;
            }
            if (cls > ctx.P + 2)
            {
                // The carried pair keeps the dtype AND the ndim tag the straight steps reached (the fixpoint).
                var s0 = em.Fresh(c0.T, false, c0.Nd); var s1 = em.Fresh(c1.T, false, c1.Nd);
                em.CopyTo(c0, s0); em.CopyTo(c1, s1);
                var k = il.DeclareLocal(typeof(long));
                var top = il.DefineLabel(); var done = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, 3L + ctx.P); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
                // kOff walks down by one stride per step (no multiply in the loop).
                il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, kOff);
                il.MarkLabel(top);
                il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Blt, done);
                if (usesRow) { il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, ctx.Row); }
                env.Ck = PolyLoadCoef(em, ccLoop, kOff); env.Tmp = s0; env.C1 = s1;
                var n0 = em.Emit(prog.StepC0, env);
                var n1 = em.Emit(prog.StepC1, env);
                if (n0.T != s0.T || n1.T != s1.T || n0.Nd != s0.Nd || n1.Nd != s1.Nd)
                    throw new InvalidOperationException("loop body changed a carried dtype or ndim: peel count wrong");
                // tmp (= s0) was read above; only now overwrite the carried pair.
                em.CopyTo(n0, s0); em.CopyTo(n1, s1);
                il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
                il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, kOff);
                il.Emit(OpCodes.Br, top);
                il.MarkLabel(done);
                c0 = s0; c1 = s1;
            }
            env.C0 = c0; env.C1 = c1;
            return em.Emit(prog.Final, env);
        }

        /// <summary>Horner for one block (NumPy's <c>polyval</c>): the init line types the state once, so no peeling.</summary>
        /// <param name="em">Emitter.</param><param name="ctx">Kernel facts.</param>
        /// <param name="cc">Coefficient context of the init read.</param>
        /// <param name="ccLoop">Coefficient context of the loop (possibly pre-converted).</param>
        /// <param name="loopStride">Byte stride of the loop's coefficients.</param>
        /// <param name="env">Bound x.</param><returns>The result value.</returns>
        /// <exception cref="InvalidOperationException">The step changed the carried dtype or NumPy ndim (cannot happen:
        ///     promotion is idempotent here, and the init line already has x's and the series' broadcast ndim).</exception>
        private static PolyValue EmitPolyHorner(PolyEmitter em, PolyEvalCtx ctx, PolyCoefCtx cc, PolyCoefCtx ccLoop, LocalBuilder loopStride, PolyEnv env)
        {
            var il = em.IL; var prog = ctx.Prog;
            var kOff = il.DeclareLocal(typeof(long));
            PolySetKOffFromEnd(il, ctx, kOff, 1);
            env.Ck = PolyLoadCoef(em, cc, kOff);
            var c0 = em.Emit(prog.HornerInit, env);
            var s0 = em.Fresh(c0.T, false, c0.Nd); em.CopyTo(c0, s0);
            var k = il.DeclareLocal(typeof(long));
            var top = il.DefineLabel(); var done = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
            il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, kOff);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Blt, done);
            env.Ck = PolyLoadCoef(em, ccLoop, kOff); env.C0 = s0;
            var n0 = em.Emit(prog.HornerStep, env);
            if (n0.T != s0.T || n0.Nd != s0.Nd) throw new InvalidOperationException("Horner step changed the carried dtype or ndim");
            em.CopyTo(n0, s0);
            il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
            il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, kOff);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(done);
            return s0;
        }

        /// <summary>
        ///     The root product for one block — NumPy's <c>polyvalfromroots</c>, <c>np.prod(x - r, axis=0)</c>: the accumulator
        ///     starts at the multiply identity (NumPy's reduction assigns it before its loop) and is multiplied by
        ///     <c>x - r[k]</c> for every root in order, the accumulator first. No peel: every factor has the loop dtype from the
        ///     first root on.
        /// </summary>
        /// <param name="em">Emitter.</param><param name="ctx">Kernel facts.</param>
        /// <param name="ccLoop">The roots' context (pre-converted to the loop dtype when the driver passes a copy).</param>
        /// <param name="loopStride">Byte stride between consecutive roots.</param>
        /// <param name="env">Bound x.</param>
        /// <returns>The product, in the loop dtype.</returns>
        /// <exception cref="InvalidOperationException">A factor is not of the loop dtype, or a step changed the accumulator's
        ///     dtype (a typing bug).</exception>
        /// <remarks>
        ///     <para>In the <see cref="PolyRootForm.Reduce"/> form a float16 product is NumPy's <c>HALF_multiply</c> reduce
        ///     branch: each factor converted exactly to float32, the running product rounded to float32, one narrow at the end
        ///     (one inner-loop call holds every root of an output, so there is no narrow between). The complex fallback product
        ///     of that form comes from the emitter (<c>reduceProduct</c>).</para>
        ///     <para>The identity multiply is emitted, not skipped: for complex it is not exact — <c>(1+0j)*(re + inf j)</c> has
        ///     a NaN real part (<c>0*inf</c>), and a <c>-0.0</c> real part can come back <c>+0.0</c> — and NumPy runs it.</para>
        ///     <para>An empty r runs the loop zero times: the identity, NumPy's <c>np.prod</c> of an empty axis.</para>
        /// </remarks>
        private static PolyValue EmitPolyRootProduct(PolyEmitter em, PolyEvalCtx ctx, PolyCoefCtx ccLoop, LocalBuilder loopStride, PolyEnv env)
        {
            var il = em.IL; var prog = ctx.Prog;
            NPTypeCode t = ctx.Tl;
            NPTypeCode accT = ctx.Key.Root == PolyRootForm.Reduce && t == NPTypeCode.Half ? NPTypeCode.Single : t;
            // acc = 1: the multiply identity (a Python int materialized in the accumulator's dtype), carried per point.
            var one = em.EmitAt(PolySteps.W(PolySteps.I1), env, accT);
            var acc = em.Fresh(accT, false);
            em.CopyTo(one, acc);
            var kOff = il.DeclareLocal(typeof(long));
            var k = il.DeclareLocal(typeof(long));
            var top = il.DefineLabel(); var done = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, k);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, kOff);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Bge, done);
            env.Ck = PolyLoadCoef(em, ccLoop, kOff);
            // The factor x - r[k], in NumPy's subtract dtype (the loop dtype); then into the accumulator's dtype (exact: the
            // float16 reduce form widens to float32, every other form is a no-op).
            var f = em.Emit(prog.RootFactor, env);
            if (f.T != t) throw new InvalidOperationException($"root factor typed {f.T}, planned {t}");
            env.C0 = acc;
            env.Tmp = em.Convert(f, accT);
            var n = em.Emit(prog.RootStep, env);
            if (n.T != acc.T) throw new InvalidOperationException("root product step changed the accumulator's dtype");
            em.CopyTo(n, acc);
            il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, k);
            il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, kOff);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(done);
            // HALF_multiply's one narrow (round to nearest even) in the float16 reduce form; the accumulator itself otherwise.
            return em.Convert(acc, t);
        }
    }
}
