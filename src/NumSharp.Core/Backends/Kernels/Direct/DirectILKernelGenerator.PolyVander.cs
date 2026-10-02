using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.PolyVander.cs — numpy.polynomial's Vandermonde family ({p}vander / {p}vander2d / {p}vander3d, plan U5)
// =============================================================================
//
// WHAT NUMPY DOES
// ---------------
// Every basis's pseudo-Vandermonde matrix is the basis's three-term recurrence run DOWN the degree axis, the same
// statements for every point (NumPy 2.4.2 numpy/polynomial/{polynomial,chebyshev,legendre,laguerre,hermite,hermite_e}.py):
//
//   x = np.array(x, copy=None, ndmin=1) + 0.0          # int/bool -> float64, -0.0 -> +0.0, a NaN quieted
//   v = np.empty((ideg + 1,) + x.shape, dtype=x.dtype)
//   v[0] = x * 0 + 1
//   if ideg > 0:  v[1] = <x | 1 - x | x * 2>;  for i in range(2, ideg + 1): v[i] = <recurrence of v[i-1], v[i-2], x>
//   return np.moveaxis(v, 0, -1)
//
// and the 2-D / 3-D forms (polyutils._vander_nd) stack the points (np.asarray((x, y[, z])) + 0.0), build each
// dimension's 1-D matrix and multiply them as an outer product along new trailing axes —
// ((V_x[..., :, None] * V_y[..., None, :]) [..., None] * V_z[..., None, None, :]) — flattening those axes last.
//
// Every statement is an ARRAY op (x is an ndarray from the first line on), so each is NumPy's ufunc loop: the complex
// product is simd_cmul (the house PolyComplexProduct.Simd — no vander operand ever reaches the ufunc's loop_scalar: the
// 1-D statements are trivially iterable at every size, and the outer products' operands are either broadcast, and
// then have two or more elements, or of equal shapes and trivially iterable), a division by a Python int is
// CDOUBLE_divide's Smith algorithm, float16 runs the HALF loop per op, and a Python int (0, 1, 2, 2*i - 1, i - 1, i,
// 2*(i - 1)) converts to the dtype NEP 50's way (exactly into decimal, else through a double and the house cast).
//
// THE KERNEL (DirectILKernelGenerator contract: it walks its own layout)
// ----------------------------------------------------------------------
// The degree axis is independent per point, exactly as the calculus family's series axis is per column
// (DirectILKernelGenerator.PolyCalculus.cs) — so the recurrences are the SAME data structures and the SAME step
// emitter: rows = degrees, columns = points (in C order of x's shape). One call walks the points in BLOCKS; per block:
//   1. LOAD: scratch slot 0 = T(x) + T(0) over the block's points (NumPy's `np.array(x) + 0.0`, the conversion and the
//      add in one pass; a contiguous source takes the house lane kinds, any other stride the scalar path);
//   2. RECUR: the routine's head (v[0], and v[1] when ideg > 0) and its loop i = 2..ideg, each step one vector loop
//      over the block's points plus a scalar tail. chebvander's `x2 = 2 * x` is written ONCE per block to scratch slot 1
//      (hermvander's x2 IS v[1], read back from that row). The 1-D form writes straight into the result's rows;
//   3. (2-D / 3-D) the per-dimension matrices live in scratch, and a PRODUCT stage writes each output row
//      r = (a*(dy+1) + b)[*(dz+1) + c] as V_x[a] * V_y[b] (* V_z[c], through a scratch row holding V_x[a]*V_y[b] — the
//      first outer product NumPy materializes, rounded to the dtype before the second multiply).
// Only the last two rows and the scratch slots are touched per step, so a block stays in L1/L2 while its rows stream
// out; NumPy materializes three to five full temporaries per degree.
//
// A 2-D / 3-D result past NDPolyVander.NonTemporalMinBytes (32 MiB) cannot stay cached for its consumer, so its product
// rows are written with NON-TEMPORAL stores (PolyVanderKey.NonTemporal): each row runs a scalar head up to the store's
// alignment, then aligned streaming vector stores, then the scalar tail, and the stage ends with an sfence. Same bytes;
// no read-for-ownership per line (a 96.8 MB result: 12 ms streamed vs 15-18 ms normal, ~10.4 ms of either being the
// fresh pages' demand-zero faults NumPy pays as well). The 1-D form never streams: its recurrence reads its rows back.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     One numpy.polynomial Vandermonde kernel call (see the file header): the pseudo-Vandermonde matrix of one basis
    ///     over <paramref name="cols"/> points, 1-D or the fused 2-D / 3-D outer product.
    /// </summary>
    /// <param name="srcs">Per dimension, the first point to LOAD (C order of the points' shape). A <see cref="PolyVanderKey.NDims"/>-long array.</param>
    /// <param name="srcCols">Per dimension, the byte stride between consecutive points of that source (any sign; contiguous
    ///     sources take the vector load).</param>
    /// <param name="dst">Output row 0, point 0: the C-contiguous <c>(rows, cols)</c> result buffer (<c>(ideg + 1, npts)</c> for
    ///     1-D, <c>(prod(ideg_k + 1), npts)</c> for 2-D / 3-D).</param>
    /// <param name="dstRow">Byte stride between output rows (<c>cols * itemsize</c>).</param>
    /// <param name="cols">Points (0 does nothing).</param>
    /// <param name="rows">Per dimension, <c>ideg_k + 1</c> (≥ 1).</param>
    /// <param name="block">Points per block (≥ 1): the scratch holds one block of every slot and per-dimension row.</param>
    /// <param name="scratch">Per dimension, the scratch area of that dimension: its <see cref="PolyVanderRoutines.Slots"/> slots,
    ///     then (2-D / 3-D) its <c>rows[k]</c> matrix rows, each slot / row <c>block * itemsize</c> bytes; for 3-D a
    ///     fourth entry, the product row. The caller owns the memory.</param>
    public unsafe delegate void PolyVanderKernel(byte** srcs, long* srcCols, byte* dst, long dstRow, long cols, long* rows, long block, byte** scratch);

    /// <summary>
    ///     The identity of one compiled Vandermonde kernel.
    /// </summary>
    /// <param name="Basis">The basis (selects the recurrence, <see cref="PolyVanderRoutines"/>); every dimension of a 2-D / 3-D
    ///     matrix uses it (NumPy's public vander2d/3d pass the same 1-D function for every axis).</param>
    /// <param name="T">The dtype NumPy computes in: <c>(x + 0.0).dtype</c> — float16/float32/float64/complex128, or NumSharp's
    ///     decimal (bool, integer and char points are float64).</param>
    /// <param name="NDims">1 (<c>{p}vander</c>), 2 or 3 (<c>{p}vander2d</c> / <c>{p}vander3d</c>, the fused outer product).</param>
    /// <param name="Src0">The dtype the load stage reads for dimension 0 (the points as given; converted to <see cref="T"/> with
    ///     the <c>+ 0.0</c> — exact to NumPy's two-step <c>stack + 0.0</c>, see <c>NDPolyVander</c>).</param>
    /// <param name="Src1">Dimension 1's source dtype (<see cref="NPTypeCode.Empty"/> for 1-D).</param>
    /// <param name="Src2">Dimension 2's source dtype (<see cref="NPTypeCode.Empty"/> below 3-D).</param>
    /// <param name="NonTemporal">Write the 2-D / 3-D product's OUTPUT rows with non-temporal (cache-bypassing) stores. The
    ///     driver asks for it when the result is too large to stay cached for its consumer
    ///     (<c>NDPolyVander.NonTemporalMinBytes</c>): such a result is evicted to memory anyway, and a non-temporal store
    ///     skips the read-for-ownership a normal store pays per cache line — measured on a 96.8 MB result, 13.4 vs 17.9 ms
    ///     on fresh pages and 1.7 vs 4.7 ms on reused ones. It is a request, not a promise: the emitter keeps normal stores
    ///     where the host (no AVX) or the lane kind (decimal, a non-256-bit kind) has no aligned non-temporal store, and
    ///     the 1-D form ignores it (its recurrence reads its output rows back, which a non-temporal store would push out of
    ///     cache first). The bits written are the same either way.</param>
    internal readonly record struct PolyVanderKey(PolyBasis Basis, NPTypeCode T, int NDims, NPTypeCode Src0,
        NPTypeCode Src1 = NPTypeCode.Empty, NPTypeCode Src2 = NPTypeCode.Empty, bool NonTemporal = false)
    {
        /// <summary>Dimension <paramref name="k"/>'s source dtype.</summary>
        /// <param name="k">0, 1 or 2.</param>
        /// <returns>The dtype.</returns>
        public NPTypeCode Src(int k) => k switch { 0 => Src0, 1 => Src1, _ => Src2 };
    }

    /// <summary>
    ///     NumPy 2.4.2's six Vandermonde recurrences, transcribed statement for statement into the calculus family's step
    ///     language (<see cref="PolyCalcRoutine"/>): row q of the buffer is <c>v[q]</c>, the loop variable is NumPy's <c>i</c>,
    ///     and <c>x</c> (and chebvander's <c>x2</c>) are the kernel's per-block scratch slots (<see cref="PolyCalcLoad.Scratch"/>).
    ///     Each routine's comment is the NumPy source it encodes; the trees are never re-associated (operand order decides
    ///     complex products and NaN priority, evaluation order decides rounding).
    /// </summary>
    internal static class PolyVanderRoutines
    {
        // The table spelling: short builders so each routine reads like the NumPy line it transcribes.

        /// <summary>A read of the step's row slot <paramref name="row"/> (an index into the step's <c>Rows</c>, NOT a buffer row).</summary>
        /// <param name="row">The slot.</param><returns>The expression.</returns>
        private static PolyCalcExpr L(int row) => new PolyCalcLoad(row);

        /// <summary>The step's constant <paramref name="c"/> (an index into its <c>Consts</c>: NumPy's Python int, converted to the
        ///     dtype per step the NEP 50 way).</summary>
        /// <param name="c">The constant slot.</param><returns>The expression.</returns>
        private static PolyCalcExpr K(int c) => new PolyCalcConst(c);

        /// <summary><c>a * b</c> as NumPy's array multiply, operands in NumPy's order (it decides complex products).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The expression.</returns>
        private static PolyCalcExpr Mul(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Multiply, a, b);

        /// <summary><c>a / b</c> as NumPy's array true divide (complex by a Python int: Smith's algorithm).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The expression.</returns>
        private static PolyCalcExpr Div(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Divide, a, b);

        /// <summary><c>a + b</c> as NumPy's array add.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The expression.</returns>
        private static PolyCalcExpr Add(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Add, a, b);

        /// <summary><c>a - b</c> as NumPy's array subtract.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The expression.</returns>
        private static PolyCalcExpr Sub(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Subtract, a, b);

        /// <summary>Stores <paramref name="e"/> into the step's row slot <paramref name="row"/>.</summary>
        /// <param name="row">The row slot written.</param><param name="e">The value.</param><returns>The statement.</returns>
        private static PolyCalcStmt Store(int row, PolyCalcExpr e) => new PolyCalcStore(row, e);

        /// <summary><c>mul * i + add</c> of the loop variable i (NumPy's <c>i</c>): a row index or a Python-int constant.</summary>
        /// <param name="mul">Multiplier.</param><param name="add">Offset.</param><returns>The affine form.</returns>
        private static PolyCalcAffine J(long mul, long add) => new PolyCalcAffine(mul, add);

        /// <summary>A fixed row index or constant <paramref name="v"/> (independent of the loop variable).</summary>
        /// <param name="v">The value.</param><returns>The affine form.</returns>
        private static PolyCalcAffine Fixed(long v) => new PolyCalcAffine(0, v);

        /// <summary>The scratch slot holding the block's converted points, NumPy's <c>x = np.array(x) + 0.0</c>.</summary>
        internal const int XSlot = 0;

        /// <summary>The scratch slot holding chebvander's <c>x2 = 2 * x</c> for the block.</summary>
        internal const int X2Slot = 1;

        /// <summary>NumPy's <c>x</c> (scratch slot <see cref="XSlot"/>).</summary>
        private static PolyCalcExpr X => new PolyCalcLoad(XSlot, scratch: true);

        /// <summary>chebvander's <c>x2</c> (scratch slot <see cref="X2Slot"/>).</summary>
        private static PolyCalcExpr X2 => new PolyCalcLoad(X2Slot, scratch: true);

        /// <summary>The six routines, indexed by <see cref="PolyBasis"/> (built once; immutable after).</summary>
        private static readonly PolyCalcRoutine[] s_routines = Build();

        /// <summary>The Vandermonde recurrence of one basis.</summary>
        /// <param name="basis">The basis.</param>
        /// <returns>The routine (head: v[0], v[1] with <see cref="PolyCalcStep.MinLen"/> 2 — NumPy's <c>if ideg &gt; 0</c>; loop
        ///     i = 2 … rows - 1).</returns>
        public static PolyCalcRoutine Get(PolyBasis basis) => s_routines[(int)basis];

        /// <summary>
        ///     How many per-block scratch slots the basis's routine reads: 2 for chebvander (<c>x</c> and <c>x2 = 2 * x</c>, which
        ///     NumPy keeps as its own array for the whole loop), 1 for every other basis (hermvander's <c>x2</c> is <c>v[1]</c>
        ///     itself, read back from that row).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <returns>1 or 2.</returns>
        public static int Slots(PolyBasis basis) => basis == PolyBasis.Chebyshev ? 2 : 1;

        /// <summary><c>v[0] = x * 0 + 1</c> — every basis's first row (row 0; the Python ints 0 and 1 are NEP 50 constants).</summary>
        /// <returns>The step.</returns>
        private static PolyCalcStep FirstRow() => new PolyCalcStep
        {
            Rows = new[] { Fixed(0) },
            Consts = new[] { Fixed(0), Fixed(1) },
            Body = new[] { Store(0, Add(Mul(X, K(0)), K(1))) },
        };

        /// <summary><c>v[1] = x</c> — the second row of poly/cheb/leg/hermevander (a copy: NumPy's slice assignment).</summary>
        /// <returns>The step (runs only when there are two rows or more).</returns>
        private static PolyCalcStep SecondRowIsX() => new PolyCalcStep
        {
            MinLen = 2,
            Rows = new[] { Fixed(1) },
            Body = new[] { Store(0, X) },
        };

        /// <summary>Builds the six routines — each one NumPy 2.4.2's vander body transcribed statement for statement (the
        ///     source line is the comment above each entry).</summary>
        /// <returns>The routines, indexed by <see cref="PolyBasis"/>.</returns>
        private static PolyCalcRoutine[] Build()
        {
            var r = new PolyCalcRoutine[6];
            // polyvander:  v[0] = x*0 + 1;  v[1] = x;  for i in range(2, ideg + 1): v[i] = v[i-1] * x
            r[(int)PolyBasis.Power] = new PolyCalcRoutine
            {
                Head = new[] { FirstRow(), SecondRowIsX() },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -1) },
                    Body = new[] { Store(0, Mul(L(1), X)) },
                },
            };
            // chebvander:  v[0] = x*0 + 1;  x2 = 2 * x;  v[1] = x;  for i in range(2, ideg + 1): v[i] = v[i-1] * x2 - v[i-2]
            // x2 is written once per block to scratch slot 1 (the same statement NumPy runs once for the whole array).
            r[(int)PolyBasis.Chebyshev] = new PolyCalcRoutine
            {
                Head = new[]
                {
                    FirstRow(),
                    new PolyCalcStep
                    {
                        MinLen = 2,
                        Rows = new[] { Fixed(1) },
                        Consts = new[] { Fixed(2) },
                        Body = new[] { new PolyCalcStore(X2Slot, Mul(K(0), X), scratch: true), Store(0, X) },
                    },
                },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -1), J(1, -2) },
                    Body = new[] { Store(0, Sub(Mul(L(1), X2), L(2))) },
                },
            };
            // legvander:   v[0] = x*0 + 1;  v[1] = x
            //              for i in range(2, ideg + 1): v[i] = (v[i-1] * x * (2*i - 1) - v[i-2] * (i - 1)) / i
            // (Python's left-to-right `v[i-1] * x * (2*i - 1)` is ((v[i-1] * x) * (2*i - 1)).)
            r[(int)PolyBasis.Legendre] = new PolyCalcRoutine
            {
                Head = new[] { FirstRow(), SecondRowIsX() },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -1), J(1, -2) },
                    Consts = new[] { J(2, -1), J(1, -1), J(1, 0) },
                    Body = new[] { Store(0, Div(Sub(Mul(Mul(L(1), X), K(0)), Mul(L(2), K(1))), K(2))) },
                },
            };
            // lagvander:   v[0] = x*0 + 1;  v[1] = 1 - x
            //              for i in range(2, ideg + 1): v[i] = (v[i-1] * (2*i - 1 - x) - v[i-2] * (i - 1)) / i
            // (`2*i - 1 - x` is ((2*i - 1) - x): the Python int 2*i - 1 first, then one array subtraction.)
            r[(int)PolyBasis.Laguerre] = new PolyCalcRoutine
            {
                Head = new[]
                {
                    FirstRow(),
                    new PolyCalcStep { MinLen = 2, Rows = new[] { Fixed(1) }, Consts = new[] { Fixed(1) }, Body = new[] { Store(0, Sub(K(0), X)) } },
                },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -1), J(1, -2) },
                    Consts = new[] { J(2, -1), J(1, -1), J(1, 0) },
                    Body = new[] { Store(0, Div(Sub(Mul(L(1), Sub(K(0), X)), Mul(L(2), K(1))), K(2))) },
                },
            };
            // hermvander:  v[0] = x*0 + 1;  x2 = x * 2;  v[1] = x2
            //              for i in range(2, ideg + 1): v[i] = (v[i-1] * x2 - v[i-2] * (2 * (i - 1)))
            // x2 IS v[1] (a copy of the same array), so the loop reads it from row 1 (row slot 3).
            r[(int)PolyBasis.Hermite] = new PolyCalcRoutine
            {
                Head = new[]
                {
                    FirstRow(),
                    new PolyCalcStep { MinLen = 2, Rows = new[] { Fixed(1) }, Consts = new[] { Fixed(2) }, Body = new[] { Store(0, Mul(X, K(0))) } },
                },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -1), J(1, -2), Fixed(1) },
                    Consts = new[] { J(2, -2) },
                    Body = new[] { Store(0, Sub(Mul(L(1), L(3)), Mul(L(2), K(0)))) },
                },
            };
            // hermevander: v[0] = x*0 + 1;  v[1] = x
            //              for i in range(2, ideg + 1): v[i] = (v[i-1] * x - v[i-2] * (i - 1))
            r[(int)PolyBasis.HermiteE] = new PolyCalcRoutine
            {
                Head = new[] { FirstRow(), SecondRowIsX() },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -1), J(1, -2) },
                    Consts = new[] { J(1, -1) },
                    Body = new[] { Store(0, Sub(Mul(L(1), X), Mul(L(2), K(0)))) },
                },
            };
            return r;
        }
    }

    /// <summary>
    ///     The Vandermonde product stage's float32 / float64 multiply with NumPy's NaN priority. NumPy's FLOAT/DOUBLE_multiply
    ///     (MSVC, win-amd64 — probed at every loop length, scalar and vector paths alike) returns the SECOND operand's NaN
    ///     when both are NaN, while x86's mulps / mulpd — and RyuJIT, which may swap a commutative multiply's operands —
    ///     return the first's, so the product blends explicitly: <c>b</c>'s quieted NaN wins, then <c>a</c>'s, else the product
    ///     (an invalid 0*inf still yields the default NaN, exactly as NumPy's). Only the 2-D / 3-D outer product needs it:
    ///     it is the one place two independently sourced NaNs meet (in a 1-D matrix every NaN of a point is x's own payload
    ///     or the one default NaN). float16 already imposes the priority in its lane kind; complex128 keeps the house
    ///     simd_cmul's NaN choice (NaN payloads are tokenized by the oracle for every complex product).
    /// </summary>
    internal static class PolyVanderOps
    {
        /// <summary>The float64 quiet-NaN bit (x86 quiets a signalling NaN operand by setting it).</summary>
        private const ulong DoubleQuiet = 0x0008000000000000UL;

        /// <summary>The float32 quiet-NaN bit.</summary>
        private const uint SingleQuiet = 0x00400000U;

        /// <summary>The quiet-bit vector of the lane type (folded by the JIT per instantiation).</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <returns>The mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<T> Quiet128<T>() where T : unmanaged
            => typeof(T) == typeof(double) ? Vector128.Create(DoubleQuiet).As<ulong, T>() : Vector128.Create(SingleQuiet).As<uint, T>();

        /// <summary>The quiet-bit vector of the lane type (folded by the JIT per instantiation).</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <returns>The mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<T> Quiet256<T>() where T : unmanaged
            => typeof(T) == typeof(double) ? Vector256.Create(DoubleQuiet).As<ulong, T>() : Vector256.Create(SingleQuiet).As<uint, T>();

        /// <summary>The quiet-bit vector of the lane type (folded by the JIT per instantiation).</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <returns>The mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<T> Quiet512<T>() where T : unmanaged
            => typeof(T) == typeof(double) ? Vector512.Create(DoubleQuiet).As<ulong, T>() : Vector512.Create(SingleQuiet).As<uint, T>();

        /// <summary><c>a * b</c> per lane with NumPy's NaN priority (see the class summary).</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="a">Left operand lanes (NumPy's first input).</param><param name="b">Right operand lanes.</param>
        /// <returns>The products.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<T> Mul128<T>(Vector128<T> a, Vector128<T> b) where T : unmanaged
        {
            var r = a * b;
            // A product with no NaN lane had no NaN operand (a NaN operand always yields a NaN product), so the plain
            // product IS NumPy's: the blends below run only for a vector holding a NaN — one compare on the common path.
            if (Vector128.EqualsAll(r, r)) return r;
            var q = Quiet128<T>();
            r = Vector128.ConditionalSelect(Vector128.Equals(a, a), r, a | q);   // a NaN a: quiet(a)
            return Vector128.ConditionalSelect(Vector128.Equals(b, b), r, b | q);  // a NaN b wins: quiet(b)
        }

        /// <summary><c>a * b</c> per lane with NumPy's NaN priority (see the class summary).</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="a">Left operand lanes (NumPy's first input).</param><param name="b">Right operand lanes.</param>
        /// <returns>The products.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<T> Mul256<T>(Vector256<T> a, Vector256<T> b) where T : unmanaged
        {
            var r = a * b;
            // A product with no NaN lane had no NaN operand (a NaN operand always yields a NaN product), so the plain
            // product IS NumPy's: the blends below run only for a vector holding a NaN — one compare on the common path.
            if (Vector256.EqualsAll(r, r)) return r;
            var q = Quiet256<T>();
            r = Vector256.ConditionalSelect(Vector256.Equals(a, a), r, a | q);
            return Vector256.ConditionalSelect(Vector256.Equals(b, b), r, b | q);
        }

        /// <summary><c>a * b</c> per lane with NumPy's NaN priority (see the class summary).</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="a">Left operand lanes (NumPy's first input).</param><param name="b">Right operand lanes.</param>
        /// <returns>The products.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector512<T> Mul512<T>(Vector512<T> a, Vector512<T> b) where T : unmanaged
        {
            var r = a * b;
            // A product with no NaN lane had no NaN operand (a NaN operand always yields a NaN product), so the plain
            // product IS NumPy's: the blends below run only for a vector holding a NaN — one compare on the common path.
            if (Vector512.EqualsAll(r, r)) return r;
            var q = Quiet512<T>();
            r = Vector512.ConditionalSelect(Vector512.Equals(a, a), r, a | q);
            return Vector512.ConditionalSelect(Vector512.Equals(b, b), r, b | q);
        }

        /// <summary>Scalar <c>a * b</c> with NumPy's NaN priority (the product stage's tail).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static double Mul(double a, double b)
        {
            if (double.IsNaN(b)) return BitConverter.UInt64BitsToDouble(BitConverter.DoubleToUInt64Bits(b) | DoubleQuiet);
            if (double.IsNaN(a)) return BitConverter.UInt64BitsToDouble(BitConverter.DoubleToUInt64Bits(a) | DoubleQuiet);
            return a * b;
        }

        /// <summary>Scalar <c>a * b</c> with NumPy's NaN priority (the product stage's tail).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float Mul(float a, float b)
        {
            if (float.IsNaN(b)) return BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(b) | SingleQuiet);
            if (float.IsNaN(a)) return BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(a) | SingleQuiet);
            return a * b;
        }

        /// <summary>The vector helper matching a lane kind's width, closed over <paramref name="clr"/>.</summary>
        /// <param name="bits">128, 256 or 512.</param><param name="clr"><see cref="float"/> or <see cref="double"/>.</param>
        /// <returns>The method.</returns>
        /// <exception cref="NotSupportedException">Another width.</exception>
        internal static MethodInfo VectorMul(int bits, Type clr)
        {
            string name = bits switch
            {
                128 => nameof(Mul128),
                256 => nameof(Mul256),
                512 => nameof(Mul512),
                _ => throw new NotSupportedException($"no {bits}-bit NaN-priority multiply"),
            };
            return typeof(PolyVanderOps).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!.MakeGenericMethod(clr);
        }

        /// <summary>The scalar helper for <paramref name="clr"/>.</summary>
        /// <param name="clr"><see cref="float"/> or <see cref="double"/>.</param>
        /// <returns>The method.</returns>
        internal static MethodInfo ScalarMul(Type clr)
            => typeof(PolyVanderOps).GetMethod(nameof(Mul), BindingFlags.NonPublic | BindingFlags.Static, new[] { clr, clr })!;

        /// <summary>
        ///     [value, address] → []: one 32-byte non-temporal store (<c>vmovntdq</c>) of a 256-bit lane value — float64,
        ///     float32, or complex128's <c>[re0, im0, re1, im1]</c>. The same bytes a normal store writes, but the cache line
        ///     goes to memory through the write-combining buffers without being read first.
        /// </summary>
        /// <typeparam name="T">The lane element type.</typeparam>
        /// <param name="value">The lanes.</param>
        /// <param name="address">The destination. MUST be 32-byte aligned (the product stage peels a scalar head per row to
        ///     get there); a misaligned address faults (#GP), it is not silently slower.</param>
        /// <remarks>Weakly ordered: the stage that issues these ends with <see cref="Sse.StoreFence"/>, so the result is
        ///     globally visible before the kernel returns (the issuing thread always sees its own stores).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void StoreNonTemporal256<T>(Vector256<T> value, byte* address) where T : unmanaged
            => Avx.StoreAlignedNonTemporal(address, value.AsByte());

        /// <summary>
        ///     [value, address] → []: the float16 lane kind's store (8 float32 lanes on the f16 grid, narrowed by the house RTNE
        ///     narrow to 8 float16 values) as one 16-byte non-temporal store (<c>movntdq</c>).
        /// </summary>
        /// <param name="lanes">The 8 float32 lanes (every one live: only a float16 loop stores float16).</param>
        /// <param name="address">The destination. MUST be 16-byte aligned (a misaligned address faults).</param>
        /// <remarks>Weakly ordered, like <see cref="StoreNonTemporal256{T}"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void StoreNonTemporalHalf8(Vector256<float> lanes, byte* address)
            => Sse2.StoreAlignedNonTemporal(address, DirectILKernelGenerator.HalfNarrow8V(lanes).AsByte());

        /// <summary>
        ///     The non-temporal store for a lane kind and the alignment it needs, or <c>(null, 0)</c> when there is none: the
        ///     host has no AVX (the 256-bit store) / SSE2, or the kind is not one of the product stage's stored kinds (a 256-bit
        ///     float64 / float32 / complex128 kind; the 8-lane float16 kind). Callers then keep the kind's normal store.
        /// </summary>
        /// <param name="vk">The loop dtype's lane kind (null: scalar kernel).</param>
        /// <returns>The helper taking <c>[value, address]</c>, and the byte alignment its address needs.</returns>
        internal static (MethodInfo Store, int Align) NonTemporalStore(PolyValueKind vk)
        {
            if (vk is null || !vk.Vec) return (null, 0);
            if (vk is PolyHalfLaneKind)
                return vk.Lanes == 8 && Sse2.IsSupported
                    ? (typeof(PolyVanderOps).GetMethod(nameof(StoreNonTemporalHalf8), BindingFlags.NonPublic | BindingFlags.Static)!, 16)
                    : (null, 0);
            if (vk.Bits != 256 || !Avx.IsSupported || vk.Clr != typeof(double) && vk.Clr != typeof(float)) return (null, 0);
            return (typeof(PolyVanderOps).GetMethod(nameof(StoreNonTemporal256), BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(vk.Clr), 32);
        }
    }

    public static partial class DirectILKernelGenerator
    {
        /// <summary>Compiled kernels by identity. A <see cref="Lazy{T}"/> per key so concurrent first calls compile once; the
        ///     entries live for the process (the key space is bounded: 6 bases × dtypes × source dtypes × 1-3 D × streamed).</summary>
        private static readonly ConcurrentDictionary<PolyVanderKey, Lazy<PolyVanderKernel>> s_polyVanderKernels = new();

        /// <summary>Backing counter of <see cref="PolyVanderVectorFallbacks"/> (Interlocked).</summary>
        private static long s_polyVanderVectorFallbacks;

        /// <summary>How many Vandermonde kernels fell back to scalar-only code because a vector lane kind or conversion
        ///     was missing. Zero for every combination the lane table covers; the tests pin it at zero.</summary>
        internal static long PolyVanderVectorFallbacks => Interlocked.Read(ref s_polyVanderVectorFallbacks);

        /// <summary>
        ///     Returns (compiling once) the Vandermonde kernel for <paramref name="key"/>. Thread-safe: concurrent first calls
        ///     compile exactly one kernel (a <see cref="Lazy{T}"/> per key).
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyVanderKernel GetPolyVanderKernel(in PolyVanderKey key)
            => s_polyVanderKernels.GetOrAdd(key, static k => new Lazy<PolyVanderKernel>(() => CompilePolyVander(k))).Value;

        /// <summary>Emits one kernel with vector loops where the lane table allows them, falling back to scalar-only code when
        ///     a lane kind or conversion turns out to be missing (the same bits, slower).</summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="PlatformNotSupportedException">No dynamic code.</exception>
        private static PolyVanderKernel CompilePolyVander(PolyVanderKey key)
        {
            if (!RuntimeFeature.IsDynamicCodeSupported)
                throw new PlatformNotSupportedException(
                    "numpy.polynomial Vandermonde matrices compile IL kernels at runtime; this runtime (NativeAOT) cannot emit dynamic code.");
            try
            {
                return EmitPolyVander(key, allowVector: true);
            }
            catch (NotSupportedException)
            {
                // A lane kind or conversion the table lacks: restart with scalar code only (a fresh set of DynamicMethods,
                // so nothing half-built survives). Counted so a test can prove it never happens.
                Interlocked.Increment(ref s_polyVanderVectorFallbacks);
                return EmitPolyVander(key, allowVector: false);
            }
        }

        /// <summary>
        ///     Emits and compiles one Vandermonde kernel: per distinct source dtype a vector load (contiguous points) and a
        ///     scalar load (any stride), the basis's recurrence stage, for 2-D / 3-D the product stage, and the root that
        ///     walks the point blocks. Each stage is its own DynamicMethod so the JIT's per-method inline budget covers one
        ///     stage's lane helpers (see the calculus kernels).
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <param name="allowVector">Emit vector loops where the lane table allows them.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="NotSupportedException">A lane kind or conversion is missing (only when vectors are allowed).</exception>
        private static PolyVanderKernel EmitPolyVander(PolyVanderKey key, bool allowVector)
        {
            // The recurrence reuses the calculus step emitter, which reads its arithmetic facts from a PolyCalcKey: the ops are
            // ufuncs in T (never scalarmath — every vander statement is an array op), with no scale.
            var e = new PolyCalcEmit
            {
                Key = new PolyCalcKey(key.Basis, Integrate: true, key.T, key.T, ScalarMath: false, Scale: false),
                Routine = PolyVanderRoutines.Get(key.Basis),
                Size = GetTypeSize(key.T),
                SrcSize = GetTypeSize(key.T),
            };
            if (allowVector && key.T != NPTypeCode.Decimal)
            {
                int lanes = PolyLanes.LoopLanes(key.T);
                if (lanes > 1 && PolyLanes.HasLaneKind(key.T, lanes))
                {
                    e.Vk = PolyLanes.CreateLaneKind(key.T, lanes);
                    e.W = lanes;
                }
            }

            int slots = PolyVanderRoutines.Slots(key.Basis);
            string name = $"NDPolyVander_{key.Basis}_{key.T}_{key.NDims}d_{key.Src0}"
                          + (key.NDims > 1 ? $"_{key.Src1}" : "") + (key.NDims > 2 ? $"_{key.Src2}" : "")
                          + (e.Vk is null ? "_scalar" : "") + (key.NonTemporal && key.NDims > 1 ? "_nt" : "");

            // One vector + one scalar load per DISTINCT source dtype (a 2-D / 3-D matrix of same-dtype points shares them).
            var loads = new Dictionary<NPTypeCode, (DynamicMethod vec, DynamicMethod scalar)>();
            var loadVec = new DynamicMethod[key.NDims];
            var loadScalar = new DynamicMethod[key.NDims];
            for (int k = 0; k < key.NDims; k++)
            {
                NPTypeCode src = key.Src(k);
                if (!loads.TryGetValue(src, out var pair))
                {
                    PolyValueKind srcVk = null;
                    if (e.Vk is not null)
                    {
                        // A bool source stays scalar: its lane kind would read the raw storage byte, where NumPy's cast (and
                        // the house scalar EmitConvertTo) reads True for ANY nonzero byte.
                        if (src == key.T) srcVk = e.Vk;
                        else if (src != NPTypeCode.Boolean && PolyLanes.LaneConvertible(src, key.T, e.W))
                            srcVk = PolyLanes.CreateLaneKind(src, e.W);
                    }
                    pair = (srcVk is null ? null : EmitPolyVanderLoad(e, src, srcVk, $"{name}_load{src}v"),
                            EmitPolyVanderLoad(e, src, null, $"{name}_load{src}s"));
                    loads[src] = pair;
                }
                loadVec[k] = pair.vec;
                loadScalar[k] = pair.scalar;
            }
            var recur = EmitPolyVanderRecur(e, slots, name + "_recur");
            DynamicMethod product = key.NDims switch
            {
                2 => EmitPolyVanderProduct2(e, name + "_prod", key.NonTemporal),
                3 => EmitPolyVanderProduct3(e, name + "_prod", key.NonTemporal),
                _ => null,
            };

            var root = new DynamicMethod(name, typeof(void),
                new[] { typeof(byte**), typeof(long*), typeof(byte*), typeof(long), typeof(long), typeof(long*), typeof(long), typeof(byte**) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            EmitPolyVanderRoot(root.GetILGenerator(), e, key, slots, loadVec, loadScalar, recur, product);
            return root.CreateDelegate<PolyVanderKernel>();
        }

        /// <summary>
        ///     The root: <c>for p0 in blocks: for each dimension: load (x + 0.0 into its scratch slot 0); recurrence; then
        ///     (2-D / 3-D) the product rows</c>. See <see cref="PolyVanderKernel"/> for the arguments (0 srcs, 1 srcCols, 2 dst,
        ///     3 dstRow, 4 cols, 5 rows, 6 block, 7 scratch).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="key">The kernel identity.</param>
        /// <param name="slots">Scratch slots per dimension.</param>
        /// <param name="loadVec">Per dimension, the vector load (contiguous points) or null.</param>
        /// <param name="loadScalar">Per dimension, the scalar load (any stride).</param>
        /// <param name="recur">The recurrence stage.</param>
        /// <param name="product">The product stage (2-D / 3-D), or null for 1-D.</param>
        private static void EmitPolyVanderRoot(ILGenerator il, PolyCalcEmit e, in PolyVanderKey key, int slots,
            DynamicMethod[] loadVec, DynamicMethod[] loadScalar, DynamicMethod recur, DynamicMethod product)
        {
            int nd = key.NDims;
            var p0 = il.DeclareLocal(typeof(long));
            var w = il.DeclareLocal(typeof(long));
            var scratchRow = il.DeclareLocal(typeof(long));
            var src = new LocalBuilder[nd];
            var sCol = new LocalBuilder[nd];
            var nRows = new LocalBuilder[nd];
            var area = new LocalBuilder[nd];
            var vBase = new LocalBuilder[nd];
            LocalBuilder pRow = null;

            // scratchRow = block * itemsize: every scratch slot / per-dimension row is one block of T.
            il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, scratchRow);

            // Hoist the per-dimension arguments out of their arrays (byte** / long*: element k at k * sizeof(element)).
            for (int k = 0; k < nd; k++)
            {
                src[k] = il.DeclareLocal(typeof(byte*));
                il.Emit(OpCodes.Ldarg_0); EmitPolyVanderElementOffset(il, k, IntPtr.Size); il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, src[k]);
                sCol[k] = il.DeclareLocal(typeof(long));
                il.Emit(OpCodes.Ldarg_1); EmitPolyVanderElementOffset(il, k, sizeof(long)); il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stloc, sCol[k]);
                nRows[k] = il.DeclareLocal(typeof(long));
                il.Emit(OpCodes.Ldarg_S, (byte)5); EmitPolyVanderElementOffset(il, k, sizeof(long)); il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stloc, nRows[k]);
                area[k] = il.DeclareLocal(typeof(byte*));
                il.Emit(OpCodes.Ldarg_S, (byte)7); EmitPolyVanderElementOffset(il, k, IntPtr.Size); il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, area[k]);
                if (nd > 1)
                {
                    // A 2-D / 3-D dimension's matrix rows follow its slots in its scratch area.
                    vBase[k] = il.DeclareLocal(typeof(byte*));
                    il.Emit(OpCodes.Ldloc, area[k]);
                    il.Emit(OpCodes.Ldc_I8, (long)slots); il.Emit(OpCodes.Ldloc, scratchRow); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Stloc, vBase[k]);
                }
            }
            if (nd == 3)
            {
                // The 3-D product row (V_x[a] * V_y[b] for the block) is the fourth scratch entry.
                pRow = il.DeclareLocal(typeof(byte*));
                il.Emit(OpCodes.Ldarg_S, (byte)7); EmitPolyVanderElementOffset(il, 3, IntPtr.Size); il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, pRow);
            }

            var pTop = il.DefineLabel(); var pEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, p0);
            il.MarkLabel(pTop);
            il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Bge, pEnd);

            // w = min(block, cols - p0)
            var wOk = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, w);
            il.Emit(OpCodes.Ldloc, w); il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Ble, wOk);
            il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Stloc, w);
            il.MarkLabel(wOk);

            for (int k = 0; k < nd; k++)
            {
                // load(src_k + p0 * sCol_k, sCol_k, area_k, w): scratch slot 0 of dimension k = T(x_k) + 0.0.
                void CallLoad(DynamicMethod stage)
                {
                    il.Emit(OpCodes.Ldloc, src[k]); il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldloc, sCol[k]); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldloc, sCol[k]); il.Emit(OpCodes.Ldloc, area[k]); il.Emit(OpCodes.Ldloc, w);
                    il.Emit(OpCodes.Call, stage);
                }
                var loaded = il.DefineLabel();
                if (loadVec[k] is not null)
                {
                    // Contiguous points take the vector load; any other stride (a reversed or strided view) the scalar one.
                    var strided = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, sCol[k]); il.Emit(OpCodes.Ldc_I8, (long)GetTypeSize(key.Src(k))); il.Emit(OpCodes.Bne_Un, strided);
                    CallLoad(loadVec[k]);
                    il.Emit(OpCodes.Br, loaded);
                    il.MarkLabel(strided);
                }
                CallLoad(loadScalar[k]);
                il.MarkLabel(loaded);

                // recur(base, bufRow, rows_k, w, area_k, scratchRow): 1-D writes the result rows (dst + p0 * itemsize, row stride
                // dstRow); 2-D / 3-D writes the dimension's own matrix rows in scratch (row stride scratchRow).
                if (nd == 1)
                {
                    il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldarg_3);
                }
                else
                {
                    il.Emit(OpCodes.Ldloc, vBase[k]);
                    il.Emit(OpCodes.Ldloc, scratchRow);
                }
                il.Emit(OpCodes.Ldloc, nRows[k]); il.Emit(OpCodes.Ldloc, w); il.Emit(OpCodes.Ldloc, area[k]); il.Emit(OpCodes.Ldloc, scratchRow);
                il.Emit(OpCodes.Call, recur);
            }

            if (product is not null)
            {
                // product(dst + p0 * itemsize, dstRow, V_0, V_1[, V_2], scratchRow, rows_0, rows_1[, rows_2], w[, pRow])
                il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldarg_3);
                for (int k = 0; k < nd; k++) il.Emit(OpCodes.Ldloc, vBase[k]);
                il.Emit(OpCodes.Ldloc, scratchRow);
                for (int k = 0; k < nd; k++) il.Emit(OpCodes.Ldloc, nRows[k]);
                il.Emit(OpCodes.Ldloc, w);
                if (nd == 3) il.Emit(OpCodes.Ldloc, pRow);
                il.Emit(OpCodes.Call, product);
            }

            il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, p0);
            il.Emit(OpCodes.Br, pTop);
            il.MarkLabel(pEnd);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>[pointer] → [pointer + k * elementSize]: the address of element <paramref name="k"/> of a pointer / long array
        ///     argument (nothing is added for element 0).</summary>
        /// <param name="il">The generator.</param><param name="k">The element index.</param><param name="elementSize">Bytes per element.</param>
        private static void EmitPolyVanderElementOffset(ILGenerator il, int k, int elementSize)
        {
            if (k == 0) return;
            il.Emit(OpCodes.Ldc_I4, k * elementSize);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        /// <summary>
        ///     A load stage <c>(byte* s, long sCol, byte* d, long w)</c>: <c>d[i] = T(s[i * sCol]) + T(0)</c> for the block's w points —
        ///     NumPy's <c>np.array(x) + 0.0</c>, the loop-input cast and the add of the Python float 0.0 (NEP 50: T's zero) in
        ///     one pass, operand order (x, 0.0): -0.0 becomes +0.0, a signalling NaN is quieted (its payload kept), every other
        ///     value is unchanged. With a vector kind the points must be contiguous (<c>sCol</c> = source element size; the root
        ///     dispatches on it); the scalar tail — or every point of a scalar stage — reads through <c>sCol</c>.
        /// </summary>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="from">The dtype read from <c>s</c>.</param>
        /// <param name="fromVk">Its lane kind for vector loads, or null for a scalar stage.</param>
        /// <param name="name">The stage name.</param>
        /// <returns>The stage.</returns>
        /// <exception cref="NotSupportedException">A lane conversion the table lacks.</exception>
        private static DynamicMethod EmitPolyVanderLoad(PolyCalcEmit e, NPTypeCode from, PolyValueKind fromVk, string name)
        {
            var t = e.Key.T;
            var dm = NewPolyCalcStage(name, typeof(byte*), typeof(long), typeof(byte*), typeof(long));
            var il = dm.GetILGenerator();
            int fromSize = GetTypeSize(from);
            bool vec = fromVk is not null && e.Vk is not null;
            var sp = il.DeclareLocal(typeof(byte*));
            var dp = il.DeclareLocal(typeof(byte*));
            var i = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Stloc, sp);
            il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Stloc, dp);

            // NumPy's weak Python float 0.0 in T (float16/float32/float64 +0, complex (0+0j), decimal 0).
            var zeroS = il.DeclareLocal(GetClrType(t));
            il.Emit(OpCodes.Ldc_R8, 0.0); EmitConvertTo(il, NPTypeCode.Double, t); il.Emit(OpCodes.Stloc, zeroS);
            LocalBuilder zeroV = null;
            if (vec)
            {
                zeroV = il.DeclareLocal(e.Vk.LocalType);
                il.Emit(OpCodes.Ldloc, zeroS); e.Vk.BroadcastFromScalar(il); il.Emit(OpCodes.Stloc, zeroV);
            }

            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);
            if (vec)
            {
                var vTop = il.DefineLabel(); var vEnd = il.DefineLabel();
                il.MarkLabel(vTop);
                il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, (long)e.W); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Bgt, vEnd);
                EmitPolyCalcAddr(il, sp, i, fromSize);
                fromVk.Load(il);
                if (from != t) PolyLanes.EmitLaneConvert(il, from, t, e.W);
                il.Emit(OpCodes.Ldloc, zeroV);
                e.Vk.Bin(il, BinaryOp.Add, PolyComplexProduct.Simd);
                EmitPolyCalcAddr(il, dp, i, e.Size);
                e.Vk.StoreValueFirst(il);
                EmitPolyCalcBump(il, i, e.W);
                il.Emit(OpCodes.Br, vTop);
                il.MarkLabel(vEnd);
            }

            // Scalar tail (every point of a scalar stage): d[i] = T(s[i * sCol]) + T(0).
            var tTop = il.DefineLabel(); var tEnd = il.DefineLabel();
            il.MarkLabel(tTop);
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Bge, tEnd);
            EmitPolyCalcAddr(il, dp, i, e.Size);
            il.Emit(OpCodes.Ldloc, sp); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, from);
            EmitConvertTo(il, from, t);
            il.Emit(OpCodes.Ldloc, zeroS);
            EmitScalarOperation(il, BinaryOp.Add, t);
            EmitStoreIndirect(il, t);
            EmitPolyCalcBump(il, i, 1);
            il.Emit(OpCodes.Br, tTop);
            il.MarkLabel(tEnd);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>
        ///     The recurrence stage <c>(byte* base, long bufRow, long rows, long w, byte* scratch, long scratchRow)</c>: the routine's
        ///     head steps (v[0]; v[1] when <c>rows &gt;= 2</c>) and its loop i = 2 … rows - 1 over the block's w points, with the
        ///     scratch slots bound at <c>scratch + slot * scratchRow</c>. Argument order 0-3 is the calculus recurrence stage's, so
        ///     its step emitter runs unchanged.
        /// </summary>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="slots">The routine's scratch slots.</param>
        /// <param name="name">The stage name.</param>
        /// <returns>The stage.</returns>
        /// <exception cref="NotSupportedException">A lane op the kind lacks.</exception>
        private static DynamicMethod EmitPolyVanderRecur(PolyCalcEmit e, int slots, string name)
        {
            var dm = NewPolyCalcStage(name, typeof(byte*), typeof(long), typeof(long), typeof(long), typeof(byte*), typeof(long));
            var il = dm.GetILGenerator();
            var r = e.Routine;

            var slot = new LocalBuilder[slots];
            for (int s = 0; s < slots; s++)
            {
                slot[s] = il.DeclareLocal(typeof(byte*));
                il.Emit(OpCodes.Ldarg_S, (byte)4);
                if (s != 0) { il.Emit(OpCodes.Ldc_I8, (long)s); il.Emit(OpCodes.Ldarg_S, (byte)5); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); }
                il.Emit(OpCodes.Stloc, slot[s]);
            }

            LocalBuilder negMask = null;
            if (e.Vk is not null && PolyCalcRoutines.UsesNeg(r))
            {
                // NumPy's negative is a sign flip (xor), never 0 - x (see the calculus recurrence stage). No vander routine
                // negates today; kept so a future table edit cannot silently emit the wrong negation.
                negMask = il.DeclareLocal(e.Vk.LocalType);
                if (e.Vk.Clr == typeof(float)) il.Emit(OpCodes.Ldc_R4, -0.0f);
                else il.Emit(OpCodes.Ldc_R8, -0.0);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(e.Vk.Bits, e.Vk.Clr), null);
                il.Emit(OpCodes.Stloc, negMask);
            }

            foreach (var s in r.Head)
            {
                if (s.MinLen > 0)
                {
                    // NumPy's `if ideg > 0:` — rows (= ideg + 1) below the step's minimum skip it.
                    var skip = il.DefineLabel();
                    il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldc_I8, s.MinLen); il.Emit(OpCodes.Blt, skip);
                    EmitPolyCalcStep(il, e, s, null, negMask, slot);
                    il.MarkLabel(skip);
                }
                else EmitPolyCalcStep(il, e, s, null, negMask, slot);
            }

            // for i in range(2, ideg + 1): rows = ideg + 1, so i runs LoopLo … rows - 1.
            var j = il.DeclareLocal(typeof(long));
            var top = il.DefineLabel(); var end = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, r.LoopLo); il.Emit(OpCodes.Stloc, j);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, end);
            EmitPolyCalcStep(il, e, r.Loop, j, negMask, slot);
            EmitPolyCalcBump(il, j, 1);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(end);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>
        ///     <c>dst[i] = a[i] * b[i]</c> over the block's points (argument <paramref name="wArg"/> of the stage): the vector loop
        ///     (the lane kind's multiply — simd_cmul for complex128, the HALF loop for float16, and for float32 / float64
        ///     <see cref="PolyVanderOps"/>' multiply with NumPy's NaN priority) and the scalar tail (the same product per
        ///     element). Operand order (a, b) is NumPy's.
        /// </summary>
        /// <param name="il">The generator.</param><param name="e">The kernel's shared facts.</param>
        /// <param name="dst">The destination row pointer local.</param>
        /// <param name="a">The left operand row pointer local.</param><param name="b">The right operand row pointer local.</param>
        /// <param name="wArg">The stage argument index holding w.</param>
        /// <param name="nonTemporal">Store the vector part with <see cref="PolyVanderOps.NonTemporalStore"/>'s cache-bypassing
        ///     store (the product's OUTPUT rows of a large result — never a row the kernel reads back). Each row first runs a
        ///     scalar head up to the store's alignment (row starts are only element-aligned: the output's row stride is
        ///     <c>npts * itemsize</c>); a lane kind without such a store keeps its normal one.</param>
        /// <returns>Whether non-temporal stores were emitted (the stage must then end with a store fence).</returns>
        private static bool EmitPolyVanderMulRow(ILGenerator il, PolyCalcEmit e, LocalBuilder dst, LocalBuilder a, LocalBuilder b, byte wArg,
            bool nonTemporal = false)
        {
            var t = e.Key.T;
            // float32 / float64 products take NumPy's NaN priority explicitly (the second operand's NaN wins —
            // PolyVanderOps): the outer product is where two independently sourced NaNs meet.
            bool nanPriority = t is NPTypeCode.Single or NPTypeCode.Double;
            var (ntStore, ntAlign) = nonTemporal ? PolyVanderOps.NonTemporalStore(e.Vk) : (null, 0);
            var i = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);

            // dst[i] = a[i] * b[i] for i up to the bound the delegate pushes (the head's local, or w): the scalar body.
            void ScalarLoop(Action<ILGenerator> loadBound)
            {
                var tTop = il.DefineLabel(); var tEnd = il.DefineLabel();
                il.MarkLabel(tTop);
                il.Emit(OpCodes.Ldloc, i); loadBound(il); il.Emit(OpCodes.Bge, tEnd);
                EmitPolyCalcAddr(il, dst, i, e.Size);
                EmitPolyCalcAddr(il, a, i, e.Size); EmitLoadIndirect(il, t);
                EmitPolyCalcAddr(il, b, i, e.Size); EmitLoadIndirect(il, t);
                if (nanPriority)
                    il.EmitCall(OpCodes.Call, PolyVanderOps.ScalarMul(GetClrType(t)), null);
                else
                    EmitPolyCalcScalarBin(il, BinaryOp.Multiply, t, PolyComplexProduct.Simd);
                EmitStoreIndirect(il, t);
                EmitPolyCalcBump(il, i, 1);
                il.Emit(OpCodes.Br, tTop);
                il.MarkLabel(tEnd);
            }

            if (e.Vk is not null)
            {
                if (ntStore is not null)
                {
                    // head = elements before dst reaches ntAlign: min(w, (ntAlign - (dst & (ntAlign-1))) / size) when dst is
                    // element-aligned (every NumSharp buffer is), else w — the whole row scalar, never a faulting NT store.
                    var head = il.DeclareLocal(typeof(long));
                    var mis = il.DeclareLocal(typeof(long));
                    var headDone = il.DefineLabel(); var allScalar = il.DefineLabel(); var clamp = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Conv_U8); il.Emit(OpCodes.Ldc_I8, (long)(ntAlign - 1)); il.Emit(OpCodes.And);
                    il.Emit(OpCodes.Stloc, mis);
                    il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, head);
                    il.Emit(OpCodes.Ldloc, mis); il.Emit(OpCodes.Brfalse, headDone);
                    il.Emit(OpCodes.Ldloc, mis); il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Rem_Un); il.Emit(OpCodes.Brtrue, allScalar);
                    il.Emit(OpCodes.Ldc_I8, (long)ntAlign); il.Emit(OpCodes.Ldloc, mis); il.Emit(OpCodes.Sub);
                    il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Div_Un); il.Emit(OpCodes.Stloc, head);
                    il.Emit(OpCodes.Br, clamp);
                    il.MarkLabel(allScalar);
                    il.Emit(OpCodes.Ldarg_S, wArg); il.Emit(OpCodes.Stloc, head);
                    il.MarkLabel(clamp);
                    il.Emit(OpCodes.Ldloc, head); il.Emit(OpCodes.Ldarg_S, wArg); il.Emit(OpCodes.Ble, headDone);
                    il.Emit(OpCodes.Ldarg_S, wArg); il.Emit(OpCodes.Stloc, head);
                    il.MarkLabel(headDone);
                    ScalarLoop(g => g.Emit(OpCodes.Ldloc, head));
                }

                var vTop = il.DefineLabel(); var vEnd = il.DefineLabel();
                il.MarkLabel(vTop);
                il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, (long)e.W); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ldarg_S, wArg); il.Emit(OpCodes.Bgt, vEnd);
                EmitPolyCalcAddr(il, a, i, e.Size); e.Vk.Load(il);
                EmitPolyCalcAddr(il, b, i, e.Size); e.Vk.Load(il);
                if (nanPriority)
                    il.EmitCall(OpCodes.Call, PolyVanderOps.VectorMul(e.Vk.Bits, e.Vk.Clr), null);
                else
                    e.Vk.Bin(il, BinaryOp.Multiply, PolyComplexProduct.Simd);
                EmitPolyCalcAddr(il, dst, i, e.Size);
                if (ntStore is not null) il.EmitCall(OpCodes.Call, ntStore, null);   // aligned by the head
                else e.Vk.StoreValueFirst(il);
                EmitPolyCalcBump(il, i, e.W);
                il.Emit(OpCodes.Br, vTop);
                il.MarkLabel(vEnd);
            }
            ScalarLoop(g => g.Emit(OpCodes.Ldarg_S, wArg));
            return ntStore is not null && e.Vk is not null;
        }

        /// <summary>[] → [ptr + idx * rowBytes] stored into <paramref name="dst"/>: a row pointer from a base pointer and a row index.</summary>
        /// <param name="il">The generator.</param><param name="dst">The byte* local to set.</param>
        /// <param name="basePtr">The base pointer argument index.</param><param name="idx">The row index local.</param>
        /// <param name="rowBytes">The row stride argument index.</param>
        private static void EmitPolyVanderRowPtr(ILGenerator il, LocalBuilder dst, byte basePtr, LocalBuilder idx, byte rowBytes)
        {
            il.Emit(OpCodes.Ldarg_S, basePtr);
            il.Emit(OpCodes.Ldloc, idx); il.Emit(OpCodes.Ldarg_S, rowBytes); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, dst);
        }

        /// <summary>
        ///     The 2-D product stage <c>(byte* o, long oRow, byte* v0, byte* v1, long vRow, long r0, long r1, long w)</c>: output row
        ///     <c>a * r1 + b</c> (in order, <c>o</c> advancing by <c>oRow</c>) is <c>V_x[a] * V_y[b]</c> over the block's points —
        ///     NumPy's <c>V_x[..., :, None] * V_y[..., None, :]</c> laid out as its reshape reads it.
        /// </summary>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="name">The stage name.</param>
        /// <param name="nonTemporal">Write the output rows with non-temporal stores (<see cref="PolyVanderKey.NonTemporal"/>).</param>
        /// <returns>The stage.</returns>
        /// <exception cref="NotSupportedException">A lane op the kind lacks.</exception>
        private static DynamicMethod EmitPolyVanderProduct2(PolyCalcEmit e, string name, bool nonTemporal)
        {
            var dm = NewPolyCalcStage(name, typeof(byte*), typeof(long), typeof(byte*), typeof(byte*), typeof(long), typeof(long), typeof(long), typeof(long));
            var il = dm.GetILGenerator();
            var orow = il.DeclareLocal(typeof(byte*));
            var a = il.DeclareLocal(typeof(long)); var b = il.DeclareLocal(typeof(long));
            var pa = il.DeclareLocal(typeof(byte*)); var pb = il.DeclareLocal(typeof(byte*));
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Stloc, orow);

            var aTop = il.DefineLabel(); var aEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, a);
            il.MarkLabel(aTop);
            il.Emit(OpCodes.Ldloc, a); il.Emit(OpCodes.Ldarg_S, (byte)5); il.Emit(OpCodes.Bge, aEnd);
            EmitPolyVanderRowPtr(il, pa, 2, a, 4);

            var bTop = il.DefineLabel(); var bEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, b);
            il.MarkLabel(bTop);
            il.Emit(OpCodes.Ldloc, b); il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Bge, bEnd);
            EmitPolyVanderRowPtr(il, pb, 3, b, 4);
            bool fenced = EmitPolyVanderMulRow(il, e, orow, pa, pb, 7, nonTemporal);
            // Next output row: the (a, b) order is the reshape's C order of the two degree axes.
            il.Emit(OpCodes.Ldloc, orow); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, orow);
            EmitPolyCalcBump(il, b, 1);
            il.Emit(OpCodes.Br, bTop);
            il.MarkLabel(bEnd);

            EmitPolyCalcBump(il, a, 1);
            il.Emit(OpCodes.Br, aTop);
            il.MarkLabel(aEnd);
            // Non-temporal stores are weakly ordered: fence them before returning, so every store of the block is globally
            // visible (to another thread handed the result) before the kernel is done — the idiom every streaming memcpy ends with.
            if (fenced) il.EmitCall(OpCodes.Call, s_polyVanderStoreFence, null);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary><see cref="Sse.StoreFence"/> (sfence), which ends a stage that issued non-temporal stores.</summary>
        private static readonly MethodInfo s_polyVanderStoreFence =
            typeof(Sse).GetMethod(nameof(Sse.StoreFence), Type.EmptyTypes) ?? throw new MissingMethodException(nameof(Sse), nameof(Sse.StoreFence));

        /// <summary>
        ///     The 3-D product stage <c>(byte* o, long oRow, byte* v0, byte* v1, byte* v2, long vRow, long r0, long r1, long r2, long w,
        ///     byte* p)</c>: for every (a, b), <c>P = V_x[a] * V_y[b]</c> (the first outer product NumPy materializes — rounded to the
        ///     dtype) into the scratch row <c>p</c>, then output rows <c>(a * r1 + b) * r2 + c = P * V_z[c]</c> in order.
        /// </summary>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="name">The stage name.</param>
        /// <param name="nonTemporal">Write the OUTPUT rows with non-temporal stores (<see cref="PolyVanderKey.NonTemporal"/>); the
        ///     scratch row P, which the stage reads back, always takes normal stores.</param>
        /// <returns>The stage.</returns>
        /// <exception cref="NotSupportedException">A lane op the kind lacks.</exception>
        private static DynamicMethod EmitPolyVanderProduct3(PolyCalcEmit e, string name, bool nonTemporal)
        {
            var dm = NewPolyCalcStage(name, typeof(byte*), typeof(long), typeof(byte*), typeof(byte*), typeof(byte*), typeof(long),
                typeof(long), typeof(long), typeof(long), typeof(long), typeof(byte*));
            var il = dm.GetILGenerator();
            var orow = il.DeclareLocal(typeof(byte*));
            var p = il.DeclareLocal(typeof(byte*));
            var a = il.DeclareLocal(typeof(long)); var b = il.DeclareLocal(typeof(long)); var c = il.DeclareLocal(typeof(long));
            var pa = il.DeclareLocal(typeof(byte*)); var pb = il.DeclareLocal(typeof(byte*)); var pc = il.DeclareLocal(typeof(byte*));
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Stloc, orow);
            il.Emit(OpCodes.Ldarg_S, (byte)10); il.Emit(OpCodes.Stloc, p);

            var aTop = il.DefineLabel(); var aEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, a);
            il.MarkLabel(aTop);
            il.Emit(OpCodes.Ldloc, a); il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Bge, aEnd);
            EmitPolyVanderRowPtr(il, pa, 2, a, 5);

            var bTop = il.DefineLabel(); var bEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, b);
            il.MarkLabel(bTop);
            il.Emit(OpCodes.Ldloc, b); il.Emit(OpCodes.Ldarg_S, (byte)7); il.Emit(OpCodes.Bge, bEnd);
            EmitPolyVanderRowPtr(il, pb, 3, b, 5);
            EmitPolyVanderMulRow(il, e, p, pa, pb, 9);

            var cTop = il.DefineLabel(); var cEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, c);
            il.MarkLabel(cTop);
            il.Emit(OpCodes.Ldloc, c); il.Emit(OpCodes.Ldarg_S, (byte)8); il.Emit(OpCodes.Bge, cEnd);
            EmitPolyVanderRowPtr(il, pc, 4, c, 5);
            bool fenced = EmitPolyVanderMulRow(il, e, orow, p, pc, 9, nonTemporal);
            // Next output row: (a, b, c) is the reshape's C order of the three degree axes.
            il.Emit(OpCodes.Ldloc, orow); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, orow);
            EmitPolyCalcBump(il, c, 1);
            il.Emit(OpCodes.Br, cTop);
            il.MarkLabel(cEnd);

            EmitPolyCalcBump(il, b, 1);
            il.Emit(OpCodes.Br, bTop);
            il.MarkLabel(bEnd);

            EmitPolyCalcBump(il, a, 1);
            il.Emit(OpCodes.Br, aTop);
            il.MarkLabel(aEnd);
            // See the 2-D stage: non-temporal stores are fenced before the kernel returns.
            if (fenced) il.EmitCall(OpCodes.Call, s_polyVanderStoreFence, null);
            il.Emit(OpCodes.Ret);
            return dm;
        }
    }
}
