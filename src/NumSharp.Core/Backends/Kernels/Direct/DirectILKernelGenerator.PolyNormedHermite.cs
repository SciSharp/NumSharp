using System;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.PolyNormedHermite.cs — NumPy's normalised Hermite recurrences (plan U8,
// numpy-polynomial-remaining-compute.md §3.4: hermgauss / hermegauss)
// =============================================================================
//
// hermgauss and hermegauss evaluate the NORMALISED Hermite functions at the Gauss nodes three times each, with these
// private NumPy 2.4.2 helpers (numpy/polynomial/hermite.py, hermite_e.py):
//
//   def _normed_hermite_n(x, n):                         # _normed_hermite_e_n:
//       if n == 0:
//           return np.full(x.shape, 1 / np.sqrt(np.sqrt(np.pi)))          # np.sqrt(np.sqrt(2 * np.pi))
//       c0 = 0.
//       c1 = 1. / np.sqrt(np.sqrt(np.pi))                                  # np.sqrt(np.sqrt(2 * np.pi))
//       nd = float(n)
//       for i in range(n - 1):
//           tmp = c0
//           c0 = -c1 * np.sqrt((nd - 1.) / nd)
//           c1 = tmp + c1 * x * np.sqrt(2. / nd)                           # np.sqrt(1. / nd)
//           nd = nd - 1.0
//       return c0 + c1 * x * np.sqrt(2)                                    # c0 + c1 * x
//
// NumPy runs that loop in Python, five array ufunc calls (and two scalar np.sqrt calls) per step over the whole x; this
// file emits it as ONE pass per point (n - 1 steps in registers), every point independent: blocks of 4 vectors of
// points per step, so four dependency chains overlap, then at most one block of 2 vectors, one vector and a scalar
// tail. A step's chain — c1 * x, the factor, the NaN-ordered add — is ~18 cycles long, so the cascade is what the
// kernel's speed comes from (measured at 500 points, 499 steps: 2 vectors 194 us, 4 vectors 107 us, 6 vectors 100 us,
// where AVX2's 16 vector registers start to spill). The per-step factors sqrt((nd - 1)/nd) and sqrt(2/nd) (or 1/nd)
// are the same for every point, so a second kernel writes them once into a table of (s1, s2) rows; the degree-(n - 1)
// recurrence hermgauss also needs reads the same table from its second row (its nd starts one lower and falls by the
// same exact steps of 1.0).
//
// BITS. Every value is float64 (x is eigvalsh's float64 result), and each statement is one IEEE operation in NumPy's
// order: (-c1) * s1 (unary minus binds first), ((c1 * x) * s2) (Python evaluates left to right), tmp + that, and the
// constants are the same correctly rounded divisions and square roots (np.sqrt of a float64 is IEEE sqrt, as Math.Sqrt
// is). NumPy's loops never fuse a multiply and an add, and RyuJIT never contracts separate IL mul / add, so the finite
// bits agree by construction. The NaN bits follow NumPy's win-amd64 loop rule (DirectILKernelGenerator.NaNOperandOrder.cs):
// where BOTH operands of an add / multiply can be NaN — c1 * x and tmp + P, both array-array ufunc calls in NumPy — the
// SECOND operand's NaN wins, emitted order-independently (RyuJIT may swap a commutative operation's operands). Every
// other operation has a finite constant operand (C, s1, s2, sqrt(2), 0.0, and the first step's scalar c0 = -C * s1), so
// x86's one-NaN rule (the NaN operand, quieted) is NumPy's; and np.negative flips a NaN's sign, as IL `neg` does. NaN
// is reachable: from degree 741 the recurrence overflows to inf at the outer nodes and inf - inf / 0 * inf make NaN
// (probed against NumPy 2.4.2, the same for hermegauss: hermgauss(741) is the first degree with NaN nodes,
// hermgauss(1000) returns 68 of them).
//
// The steps are UNIFORM: NumPy's first step works on the scalars c0 = 0.0 and c1 = C, its second on the scalar
// c0 = -C * s1, and the ops are the same IEEE ops on scalars as on arrays; with c0 = +0.0 and c1 = C broadcast into
// every lane the kernel's first steps give the same bits (+0.0 + v is NumPy's `0. + array`, which turns -0.0 into
// +0.0, and the NaN-priority masks are inert against finite operands).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Which of NumPy's two normalised Hermite recurrences a <see cref="PolyNormedHermiteKernel"/> runs.
    /// </summary>
    internal enum PolyNormedHermite
    {
        /// <summary>
        ///     <c>numpy.polynomial.hermite._normed_hermite_n</c> (physicists' Hermite, weight <c>exp(-x²)</c>):
        ///     <c>c1 = 1/sqrt(sqrt(pi))</c>, step factor <c>sqrt(2/nd)</c>, result <c>c0 + (c1*x)*sqrt(2)</c>.
        /// </summary>
        Hermite = 0,

        /// <summary>
        ///     <c>numpy.polynomial.hermite_e._normed_hermite_e_n</c> (probabilists' Hermite, weight <c>exp(-x²/2)</c>):
        ///     <c>c1 = 1/sqrt(sqrt(2*pi))</c>, step factor <c>sqrt(1/nd)</c>, result <c>c0 + c1*x</c>.
        /// </summary>
        HermiteE = 1,
    }

    /// <summary>
    ///     Writes the step table of one normalised Hermite recurrence: row <c>j</c> (two doubles at <c>table + 2*j</c>) holds
    ///     NumPy's <c>np.sqrt((nd - 1.) / nd)</c> and <c>np.sqrt(2. / nd)</c> (<see cref="PolyNormedHermite.HermiteE"/>:
    ///     <c>np.sqrt(1. / nd)</c>) for <c>nd = first - j</c>, as NumPy's loop computes them before its <c>nd = nd - 1.0</c>.
    /// </summary>
    /// <param name="table">The first of <c>2 * <paramref name="rows"/></c> writable doubles.</param>
    /// <param name="rows">Rows to write (zero or less writes nothing).</param>
    /// <param name="first">NumPy's <c>nd = float(n)</c>: the degree of the recurrence the table starts at.</param>
    /// <remarks>
    ///     <c>nd</c> falls by exact steps of 1.0 (every integer below 2^53 is a double), so the table of degree
    ///     <c>n</c> holds the table of degree <c>n - 1</c> as its rows 1 onward: the caller writes it once.
    /// </remarks>
    public unsafe delegate void PolyNormedHermiteStepsKernel(double* table, long rows, double first);

    /// <summary>
    ///     One normalised Hermite recurrence of degree <c><paramref name="steps"/> + 1</c> (at least 1) at
    ///     <paramref name="count"/> contiguous float64 points: <c>dst[k] = h(x[k])</c> with NumPy's statements (see the file
    ///     header), reading row <c>j</c> of <paramref name="table"/> at step <c>j</c>.
    /// </summary>
    /// <param name="x">Element 0 of <paramref name="count"/> contiguous float64 points (only read).</param>
    /// <param name="dst">Element 0 of <paramref name="count"/> contiguous float64 slots, every one written; may equal
    ///     <paramref name="x"/> (each point is read before its slot is written, and no other point reads it).</param>
    /// <param name="count">Point count (zero or less writes nothing).</param>
    /// <param name="table">The step table (<see cref="PolyNormedHermiteStepsKernel"/>), at least <c>2 * steps</c> doubles.</param>
    /// <param name="steps">NumPy's loop count <c>n - 1</c> (zero runs no step: degree 1).</param>
    /// <remarks>
    ///     Degree 0 (NumPy's <c>np.full</c> branch) is not this kernel's: the caller fills the constant itself.
    /// </remarks>
    public unsafe delegate void PolyNormedHermiteKernel(double* x, double* dst, long count, double* table, long steps);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Vectors of points one step of the widest block loop advances together (independent dependency chains). Four
        ///     hide most of a step's ~18-cycle chain; six gain 7 % more on AVX2 but leave the remainder of every count below
        ///     24 points to the narrower loops, and spill registers.
        /// </summary>
        private const int PolyNormedHermiteUnroll = 4;

        /// <summary>The cached recurrence kernels, indexed by <see cref="PolyNormedHermite"/> (a racing first call may compile one twice — the same code).</summary>
        private static readonly PolyNormedHermiteKernel[] s_polyNormedHermite = new PolyNormedHermiteKernel[2];

        /// <summary>The cached step-table kernels, indexed by <see cref="PolyNormedHermite"/>.</summary>
        private static readonly PolyNormedHermiteStepsKernel[] s_polyNormedHermiteSteps = new PolyNormedHermiteStepsKernel[2];

        /// <summary>
        ///     NumPy's initial <c>c1</c> of the recurrence, which is also its degree-0 value:
        ///     <c>1. / np.sqrt(np.sqrt(np.pi))</c> (<see cref="PolyNormedHermite.Hermite"/>) or
        ///     <c>1. / np.sqrt(np.sqrt(2 * np.pi))</c> — two IEEE square roots and one IEEE division of NumPy's float64
        ///     <c>np.pi</c> (bit-equal to <see cref="Math.PI"/>; <c>2 * np.pi</c> is an exact doubling).
        /// </summary>
        /// <param name="kind">The recurrence.</param>
        /// <returns>The constant.</returns>
        internal static double PolyNormedHermiteC1(PolyNormedHermite kind)
            => 1.0 / Math.Sqrt(Math.Sqrt(kind == PolyNormedHermite.Hermite ? Math.PI : 2 * Math.PI));

        /// <summary>
        ///     The recurrence kernel for <paramref name="kind"/> (see <see cref="PolyNormedHermiteKernel"/>), compiled on first use.
        /// </summary>
        /// <param name="kind">The recurrence.</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="InvalidOperationException">IL generation is disabled (<see cref="Enabled"/> is false).</exception>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyNormedHermiteKernel GetPolyNormedHermiteKernel(PolyNormedHermite kind)
        {
            if (Volatile.Read(ref s_polyNormedHermite[(int)kind]) is { } fast)
                return fast;
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");
            var k = EmitPolyNormedHermite(kind);
            Volatile.Write(ref s_polyNormedHermite[(int)kind], k);
            return k;
        }

        /// <summary>
        ///     The step-table kernel for <paramref name="kind"/> (see <see cref="PolyNormedHermiteStepsKernel"/>), compiled on
        ///     first use.
        /// </summary>
        /// <param name="kind">The recurrence.</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="InvalidOperationException">IL generation is disabled (<see cref="Enabled"/> is false).</exception>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyNormedHermiteStepsKernel GetPolyNormedHermiteStepsKernel(PolyNormedHermite kind)
        {
            if (Volatile.Read(ref s_polyNormedHermiteSteps[(int)kind]) is { } fast)
                return fast;
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");
            var k = EmitPolyNormedHermiteSteps(kind);
            Volatile.Write(ref s_polyNormedHermiteSteps[(int)kind], k);
            return k;
        }

        /// <summary>Emits the <see cref="PolyNormedHermiteStepsKernel"/> for <paramref name="kind"/>.</summary>
        /// <param name="kind">The recurrence (decides the second factor's numerator, 2.0 or 1.0).</param>
        /// <returns>The kernel.</returns>
        private static unsafe PolyNormedHermiteStepsKernel EmitPolyNormedHermiteSteps(PolyNormedHermite kind)
        {
            // args: 0 table, 1 rows, 2 first
            var dm = new DynamicMethod("NDPolyNormedHermiteSteps_" + kind, typeof(void),
                new[] { typeof(double*), typeof(long), typeof(double) }, typeof(DirectILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            var p = il.DeclareLocal(typeof(double*));
            var left = il.DeclareLocal(typeof(long));
            var nd = il.DeclareLocal(typeof(double));
            var loop = il.DefineLabel();
            var done = il.DefineLabel();
            MethodInfo sqrt = ScalarMethodCache.MathFn1(typeof(double), "Sqrt");
            double numerator = kind == PolyNormedHermite.Hermite ? 2.0 : 1.0;

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, p);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, left);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, nd);

            // while (left > 0) — a SIGNED down-counter, so zero or fewer rows write nothing.
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, done);

            // p[0] = sqrt((nd - 1.) / nd): Python's float subtraction and true division, then np.sqrt (IEEE sqrt).
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldloc, nd);
            il.Emit(OpCodes.Ldc_R8, 1.0);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldloc, nd);
            il.Emit(OpCodes.Div);
            il.EmitCall(OpCodes.Call, sqrt, null);
            il.Emit(OpCodes.Stind_R8);

            // p[1] = sqrt(2. / nd) (hermite_e: sqrt(1. / nd)).
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldc_I4, sizeof(double));
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_R8, numerator);
            il.Emit(OpCodes.Ldloc, nd);
            il.Emit(OpCodes.Div);
            il.EmitCall(OpCodes.Call, sqrt, null);
            il.Emit(OpCodes.Stind_R8);

            // nd = nd - 1.0 (exact: nd is an integer below 2^53); p += one row; left--.
            il.Emit(OpCodes.Ldloc, nd);
            il.Emit(OpCodes.Ldc_R8, 1.0);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, nd);
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldc_I4, 2 * sizeof(double));
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, p);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, left);
            il.Emit(OpCodes.Br, loop);

            il.MarkLabel(done);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyNormedHermiteStepsKernel>();
        }

        /// <summary>
        ///     Emits the <see cref="PolyNormedHermiteKernel"/> for <paramref name="kind"/>: blocks of
        ///     <see cref="PolyNormedHermiteUnroll"/> vectors of points, then at most one block of half as many, then single
        ///     vectors, then a scalar tail (only the tail without SIMD hardware).
        /// </summary>
        /// <param name="kind">The recurrence (its initial c1, its final multiplier).</param>
        /// <returns>The kernel.</returns>
        private static unsafe PolyNormedHermiteKernel EmitPolyNormedHermite(PolyNormedHermite kind)
        {
            // args: 0 x, 1 dst, 2 count, 3 table, 4 steps
            var dm = new DynamicMethod("NDPolyNormedHermite_" + kind, typeof(void),
                new[] { typeof(double*), typeof(double*), typeof(long), typeof(double*), typeof(long) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            var xp = il.DeclareLocal(typeof(double*));
            var dp = il.DeclareLocal(typeof(double*));
            var left = il.DeclareLocal(typeof(long));
            var tp = il.DeclareLocal(typeof(double*));
            var j = il.DeclareLocal(typeof(long));
            double c1 = PolyNormedHermiteC1(kind);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, xp);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, dp);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, left);

            if (VectorBits != 0)
            {
                // Blocks of U vectors first (U independent chains hide the step's add / multiply latency), then the
                // remainder (fewer than U vectors) in a halving cascade — at most one block of U/2 vectors, then one
                // vector at a time — so a remainder of U/2 vectors or more still runs U/2 chains, not one, before the
                // scalar tail.
                EmitPolyNormedHermiteVectorLoop(il, kind, c1, PolyNormedHermiteUnroll, xp, dp, left, tp, j);
                EmitPolyNormedHermiteVectorLoop(il, kind, c1, PolyNormedHermiteUnroll / 2, xp, dp, left, tp, j);
                EmitPolyNormedHermiteVectorLoop(il, kind, c1, 1, xp, dp, left, tp, j);
            }
            EmitPolyNormedHermiteScalarLoop(il, kind, c1, xp, dp, left, tp, j);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyNormedHermiteKernel>();
        }

        /// <summary>
        ///     Emits <c>while (left &gt;= units * lanes)</c>: <paramref name="units"/> vectors of points loaded, the recurrence
        ///     run over every table row with each row's two factors broadcast once, the results stored, the pointers advanced.
        /// </summary>
        /// <param name="il">The IL stream.</param>
        /// <param name="kind">The recurrence.</param>
        /// <param name="c1">The initial c1 (<see cref="PolyNormedHermiteC1"/>).</param>
        /// <param name="units">Vectors per block (1 or more).</param>
        /// <param name="xp">The running point pointer.</param>
        /// <param name="dp">The running result pointer.</param>
        /// <param name="left">Points left.</param>
        /// <param name="tp">Scratch: the running table pointer.</param>
        /// <param name="j">Scratch: the steps left.</param>
        private static void EmitPolyNormedHermiteVectorLoop(ILGenerator il, PolyNormedHermite kind, double c1, int units,
            LocalBuilder xp, LocalBuilder dp, LocalBuilder left, LocalBuilder tp, LocalBuilder j)
        {
            int bits = VectorBits;
            Type d = typeof(double);
            Type vt = VectorMethodCache.V(bits, d);
            int lanes = bits / 64;
            int blockElems = units * lanes;
            MethodInfo load = VectorMethodCache.Load(bits, d);
            MethodInfo store = VectorMethodCache.Store(bits, d);
            MethodInfo broadcast = VectorMethodCache.CreateBroadcast(bits, d);
            MethodInfo zero = VectorMethodCache.Zero(bits, d);
            // np.negative as the house emits it (EmitVectorSignFlip): XOR with -0.0, which flips every lane's sign bit —
            // a NaN's included — and touches nothing else.
            MethodInfo xor = VectorMethodCache.BinaryX86(bits, "Xor", d) ?? VectorMethodCache.Generic(bits, "Xor", d, paramCount: 2);

            var vx = new LocalBuilder[units];
            var vc0 = new LocalBuilder[units];
            var vc1 = new LocalBuilder[units];
            for (int u = 0; u < units; u++)
            {
                vx[u] = il.DeclareLocal(vt);
                vc0[u] = il.DeclareLocal(vt);
                vc1[u] = il.DeclareLocal(vt);
            }
            var vs1 = il.DeclareLocal(vt);
            var vs2 = il.DeclareLocal(vt);
            var vInit = il.DeclareLocal(vt);
            var vFinal = il.DeclareLocal(vt);
            var vSign = il.DeclareLocal(vt);

            var block = il.DefineLabel();
            var step = il.DefineLabel();
            var stepsDone = il.DefineLabel();
            var done = il.DefineLabel();

            // Loop-invariant broadcasts: NumPy's initial c1, the sign mask of np.negative and (Hermite) its final np.sqrt(2).
            il.Emit(OpCodes.Ldc_R8, c1);
            il.EmitCall(OpCodes.Call, broadcast, null);
            il.Emit(OpCodes.Stloc, vInit);
            il.Emit(OpCodes.Ldc_R8, -0.0);
            il.EmitCall(OpCodes.Call, broadcast, null);
            il.Emit(OpCodes.Stloc, vSign);
            if (kind == PolyNormedHermite.Hermite)
            {
                il.Emit(OpCodes.Ldc_R8, Math.Sqrt(2.0));
                il.EmitCall(OpCodes.Call, broadcast, null);
                il.Emit(OpCodes.Stloc, vFinal);
            }

            // while (left >= blockElems)
            il.MarkLabel(block);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, (long)blockElems);
            il.Emit(OpCodes.Blt, done);

            // x[u] = the block's points; c0 = +0.0 (NumPy's Python float 0.); c1 = C.
            for (int u = 0; u < units; u++)
            {
                EmitElementPointer(il, xp, u * lanes);
                il.EmitCall(OpCodes.Call, load, null);
                il.Emit(OpCodes.Stloc, vx[u]);
                il.EmitCall(OpCodes.Call, zero, null);
                il.Emit(OpCodes.Stloc, vc0[u]);
                il.Emit(OpCodes.Ldloc, vInit);
                il.Emit(OpCodes.Stloc, vc1[u]);
            }

            // tp = table; j = steps
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Stloc, tp);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Stloc, j);

            // while (j > 0): one NumPy loop iteration for every vector of the block.
            il.MarkLabel(step);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, stepsDone);

            il.Emit(OpCodes.Ldloc, tp);
            il.Emit(OpCodes.Ldind_R8);
            il.EmitCall(OpCodes.Call, broadcast, null);
            il.Emit(OpCodes.Stloc, vs1);
            EmitElementPointer(il, tp, 1);
            il.Emit(OpCodes.Ldind_R8);
            il.EmitCall(OpCodes.Call, broadcast, null);
            il.Emit(OpCodes.Stloc, vs2);

            for (int u = 0; u < units; u++)
            {
                // c0' = (-c1) * s1 — np.negative (the sign-bit flip) times a finite factor.
                il.Emit(OpCodes.Ldloc, vc1[u]);
                il.Emit(OpCodes.Ldloc, vSign);
                il.EmitCall(OpCodes.Call, xor, null);
                il.Emit(OpCodes.Ldloc, vs1);
                EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Double);
                // c1' = tmp + (c1 * x) * s2 with tmp = c0: c1 * x and tmp + P are NumPy's array-array calls (the second
                // operand's NaN wins when both are NaN); the multiply by s2 has a finite operand.
                il.Emit(OpCodes.Ldloc, vc0[u]);
                il.Emit(OpCodes.Ldloc, vc1[u]);
                il.Emit(OpCodes.Ldloc, vx[u]);
                EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
                il.Emit(OpCodes.Ldloc, vs2);
                EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Double);
                EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
                // Stack [c0', c1']: both computed from the OLD c0 / c1, as NumPy's three assignments are.
                il.Emit(OpCodes.Stloc, vc1[u]);
                il.Emit(OpCodes.Stloc, vc0[u]);
            }

            // tp += one row; j--
            EmitElementPointer(il, tp, 2);
            il.Emit(OpCodes.Stloc, tp);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, j);
            il.Emit(OpCodes.Br, step);

            // dst = c0 + (c1 * x) * sqrt(2) (Hermite) / c0 + c1 * x (HermiteE), both adds and c1 * x array-array.
            il.MarkLabel(stepsDone);
            for (int u = 0; u < units; u++)
            {
                il.Emit(OpCodes.Ldloc, vc0[u]);
                il.Emit(OpCodes.Ldloc, vc1[u]);
                il.Emit(OpCodes.Ldloc, vx[u]);
                EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
                if (kind == PolyNormedHermite.Hermite)
                {
                    il.Emit(OpCodes.Ldloc, vFinal);
                    EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Double);
                }
                EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
                EmitElementPointer(il, dp, u * lanes);
                il.EmitCall(OpCodes.Call, store, null);
            }

            // xp += blockElems; dp += blockElems; left -= blockElems
            EmitElementPointer(il, xp, blockElems);
            il.Emit(OpCodes.Stloc, xp);
            EmitElementPointer(il, dp, blockElems);
            il.Emit(OpCodes.Stloc, dp);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, (long)blockElems);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, left);
            il.Emit(OpCodes.Br, block);

            il.MarkLabel(done);
        }

        /// <summary>
        ///     Emits the scalar tail <c>while (left &gt; 0)</c>: one point at a time, the same statements on doubles, the two
        ///     array-array operations through the scalar second-operand-NaN emitter.
        /// </summary>
        /// <param name="il">The IL stream.</param>
        /// <param name="kind">The recurrence.</param>
        /// <param name="c1">The initial c1 (<see cref="PolyNormedHermiteC1"/>).</param>
        /// <param name="xp">The running point pointer.</param>
        /// <param name="dp">The running result pointer.</param>
        /// <param name="left">Points left.</param>
        /// <param name="tp">Scratch: the running table pointer.</param>
        /// <param name="j">Scratch: the steps left.</param>
        private static void EmitPolyNormedHermiteScalarLoop(ILGenerator il, PolyNormedHermite kind, double c1,
            LocalBuilder xp, LocalBuilder dp, LocalBuilder left, LocalBuilder tp, LocalBuilder j)
        {
            var sx = il.DeclareLocal(typeof(double));
            var sc0 = il.DeclareLocal(typeof(double));
            var sc1 = il.DeclareLocal(typeof(double));
            var point = il.DefineLabel();
            var step = il.DefineLabel();
            var stepsDone = il.DefineLabel();
            var done = il.DefineLabel();

            // while (left > 0)
            il.MarkLabel(point);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, done);

            il.Emit(OpCodes.Ldloc, xp);
            il.Emit(OpCodes.Ldind_R8);
            il.Emit(OpCodes.Stloc, sx);
            il.Emit(OpCodes.Ldc_R8, 0.0);
            il.Emit(OpCodes.Stloc, sc0);
            il.Emit(OpCodes.Ldc_R8, c1);
            il.Emit(OpCodes.Stloc, sc1);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Stloc, tp);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Stloc, j);

            il.MarkLabel(step);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, stepsDone);

            // c0' = (-c1) * s1 (IL neg flips the sign bit, NaN's included, as np.negative does).
            il.Emit(OpCodes.Ldloc, sc1);
            il.Emit(OpCodes.Neg);
            il.Emit(OpCodes.Ldloc, tp);
            il.Emit(OpCodes.Ldind_R8);
            il.Emit(OpCodes.Mul);
            // c1' = c0 + (c1 * x) * s2
            il.Emit(OpCodes.Ldloc, sc0);
            il.Emit(OpCodes.Ldloc, sc1);
            il.Emit(OpCodes.Ldloc, sx);
            EmitScalarOperation(il, BinaryOp.Multiply, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
            EmitElementPointer(il, tp, 1);
            il.Emit(OpCodes.Ldind_R8);
            il.Emit(OpCodes.Mul);
            EmitScalarOperation(il, BinaryOp.Add, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
            il.Emit(OpCodes.Stloc, sc1);
            il.Emit(OpCodes.Stloc, sc0);

            EmitElementPointer(il, tp, 2);
            il.Emit(OpCodes.Stloc, tp);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, j);
            il.Emit(OpCodes.Br, step);

            // *dp = c0 + (c1 * x) * sqrt(2) / c0 + c1 * x
            il.MarkLabel(stepsDone);
            il.Emit(OpCodes.Ldloc, dp);
            il.Emit(OpCodes.Ldloc, sc0);
            il.Emit(OpCodes.Ldloc, sc1);
            il.Emit(OpCodes.Ldloc, sx);
            EmitScalarOperation(il, BinaryOp.Multiply, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
            if (kind == PolyNormedHermite.Hermite)
            {
                il.Emit(OpCodes.Ldc_R8, Math.Sqrt(2.0));
                il.Emit(OpCodes.Mul);
            }
            EmitScalarOperation(il, BinaryOp.Add, NPTypeCode.Double, NaNOperandOrder.NumPyUfunc);
            il.Emit(OpCodes.Stind_R8);

            // xp++; dp++; left--
            EmitElementPointer(il, xp, 1);
            il.Emit(OpCodes.Stloc, xp);
            EmitElementPointer(il, dp, 1);
            il.Emit(OpCodes.Stloc, dp);
            il.Emit(OpCodes.Ldloc, left);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, left);
            il.Emit(OpCodes.Br, point);

            il.MarkLabel(done);
        }

        /// <summary>
        ///     Pushes <c>ptr + elements</c> for a <c>double*</c> local (a byte offset of <c>8 * elements</c>).
        /// </summary>
        /// <param name="il">The IL stream; stack <c>[]</c> → <c>[double*]</c>.</param>
        /// <param name="ptr">The pointer local.</param>
        /// <param name="elements">The element offset (0 pushes the pointer itself).</param>
        private static void EmitElementPointer(ILGenerator il, LocalBuilder ptr, int elements)
        {
            il.Emit(OpCodes.Ldloc, ptr);
            if (elements == 0)
                return;
            il.Emit(OpCodes.Ldc_I4, elements * sizeof(double));
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }
    }
}
