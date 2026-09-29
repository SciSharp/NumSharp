using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.PolyCalculus.cs — numpy.polynomial's calculus family ({p}der / {p}int, plan U4)
// =============================================================================
//
// WHAT NUMPY DOES
// ---------------
// Every basis's derivative and integral is one Python loop over the SERIES axis, applied `m` times:
//
//   der:  n = n - 1;  c *= scl;  der = np.empty(n);  for j in range(n, lo, -1): <2-3 lines>;  c = der
//   int:  n = len(c); c *= scl;  tmp = np.empty(n + 1); tmp[0] = c[0]*0; tmp[1] = c[0]; <loop>;
//         tmp[0] += k[i] - {p}val(lbnd, tmp);  c = tmp
//
// (NumPy 2.4.2 numpy/polynomial/{polynomial,chebyshev,legendre,laguerre,hermite,hermite_e}.py). The
// statements are the same for every element of the other axes: the recurrence runs down the series axis,
// independently for every "column" (a position in c.shape[1:]). That independence is the whole engine: a
// column's values never meet another column's, so ANY column order computes NumPy's bits.
//
// THE KERNEL (DirectILKernelGenerator contract: it walks its own layout)
// ----------------------------------------------------------------------
// One call processes a whole working BUFFER: `rows x cols` elements of the coefficient dtype, C-contiguous
// (row q = coefficient q, every row the `cols` columns in C order of c.shape[1:]). Columns are taken in
// BLOCKS of `block` elements; for each block the kernel
//   1. LOADS the block's rows from the caller's source (any dtype the series converts from, any row stride,
//      a constant column stride) — converting to the coefficient dtype and applying `c *= scl` in the same
//      pass — or, when the rows are already in the buffer, scales them in place;
//   2. runs the RECURRENCE for the block: for every j, one vector loop over the block's columns (W lanes of
//      the coefficient dtype per vector, the house lane kinds of ILKernelGenerator.Polynomial.Lanes.cs), then
//      a scalar tail. The per-j Python ints (2*j, j-2, 2*(j+1), ...) are converted to the dtype ONCE per
//      (block, j) — NumPy's NEP 50 conversion of a weak int: exact to double, then the house cast — and the
//      complex divisor's Smith preparation is hoisted with them.
// A block keeps `rows x block` elements hot across the whole recurrence (and across all `orders` of a
// derivative), where NumPy streams the full arrays through three to five temporaries per j.
//
// IN PLACE, ONE BUFFER FOR THE WHOLE CALL
// ---------------------------------------
// The recurrences never need a value after it is overwritten, if the result is written one row over:
//   der: c[q] lives at row q of the order's window; der[q] is written to row q + 1 — der[j-1] lands on c[j]'s
//        row, which step j reads last (`der[j-1] = K*c[j]` and `c[j-2] += ...c[j]...` share ONE load of c[j]).
//        After `orders` orders the result is rows [orders, n).
//   int: tmp[q] is written to row q and c[q] lives at row q + 1 — tmp[j+1] lands on c[j]'s row, and tmp[0]
//        takes the fresh row above the window. Each order moves the window up one row, so an m-fold
//        integral needs m spare rows above the loaded series.
// So a call allocates exactly one buffer, whatever the order, and every value is still produced by NumPy's
// statement sequence (the statements are applied in NumPy's order to the same values).
//
// NUMPY'S THREE ARITHMETICS, PER STATEMENT
// ----------------------------------------
//   * `c *= scl` is always an ARRAY op (c is an ndarray): NumPy's ufunc loop, whose complex product is
//     simd_cmul (fused) — observable: `(inf+1j) *= 1` is `(inf+nanj)`. Emitted with the house scalar op /
//     the complex lane kind's CMul, operand order (c, scl).
//   * For a 1-D series, c[j] is a NumPy SCALAR: `(2*j) * c[j]`, `c[j] / K`, `c[j-2] += ...` are scalarmath,
//     whose complex product is the NAIVE four-product formula (ILKernelGenerator.PolyNaiveComplexMultiply).
//     Those kernels are one column (ScalarMath key) and emit no vector code.
//   * For an N-D series, c[j] is a row VIEW: every statement is a ufunc (simd_cmul) over the row.
//   Division (complex: CDOUBLE_divide's Smith algorithm, which scalarmath also calls), add, subtract and
//   negation have one form each. float16 runs NumPy's HALF loop per op (widen, float32 op, RTNE narrow).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     One numpy.polynomial calculus kernel call (see the file header): loads/scales and differentiates
    ///     (<c>orders</c> fused orders) or integrates (one order) a C-contiguous working buffer in place.
    /// </summary>
    /// <param name="src">The source element (row 0, column 0) to LOAD from, or null when the series already sits in
    ///     the buffer (then only <c>c *= scl</c> runs over it, when the kernel scales).</param>
    /// <param name="srcRow">Byte stride of the source along the series axis (any sign, 0 for a broadcast).</param>
    /// <param name="srcCol">Byte stride of the source between consecutive columns (C order of the other axes).</param>
    /// <param name="buf">Buffer row 0: c[0]'s row for a derivative, the row tmp[0] goes to for an integral
    ///     (c[0] then sits at row 1).</param>
    /// <param name="bufRow">Byte stride between buffer rows (<c>cols * itemsize</c>).</param>
    /// <param name="cols">Columns (the product of the other axes; 1 for a 1-D series, 0 does nothing).</param>
    /// <param name="n"><c>len(c)</c> — coefficients in the series before the first order.</param>
    /// <param name="orders">A derivative's order count, 1 ≤ orders &lt; n (each order shortens the series by one);
    ///     an integral kernel runs exactly one order and ignores it.</param>
    /// <param name="block">Columns per block (≥ 1): how many columns stay cache-resident through the recurrence.</param>
    /// <param name="scl">One element of the coefficient dtype holding NumPy's <c>scl</c> after its NEP 50
    ///     conversion (read only by a kernel compiled with <see cref="PolyCalcKey.Scale"/>).</param>
    public unsafe delegate void PolyCalcKernel(byte* src, long srcRow, long srcCol, byte* buf, long bufRow,
        long cols, long n, long orders, long block, byte* scl);

    /// <summary>
    ///     The identity of one compiled calculus kernel.
    /// </summary>
    /// <param name="Basis">The basis (selects the recurrence, <see cref="PolyCalcRoutines"/>).</param>
    /// <param name="Integrate">{p}int (true) or {p}der (false).</param>
    /// <param name="T">The coefficient dtype NumPy computes in: float16/float32/float64/complex128, or NumSharp's
    ///     decimal (integer and bool series are converted to float64 before any arithmetic).</param>
    /// <param name="Src">The dtype the load stage reads from <c>src</c> (T when the series is already in the buffer).</param>
    /// <param name="ScalarMath">The series is 1-D: every recurrence op is NumPy scalarmath (naive complex product)
    ///     and the kernel is one column (no vector code).</param>
    /// <param name="Scale">The kernel applies <c>c *= scl</c> itself (false when the caller has already scaled
    ///     the series, or scales it through the house ufunc for a promoting / array <c>scl</c>).</param>
    internal readonly record struct PolyCalcKey(PolyBasis Basis, bool Integrate, NPTypeCode T, NPTypeCode Src, bool ScalarMath, bool Scale);

    // -------------------------------------------------------------------------------------------------
    //  The recurrences as DATA
    // -------------------------------------------------------------------------------------------------

    /// <summary>
    ///     An affine function of the recurrence index, <c>JMul*j + Add</c>: a buffer row relative to the order's
    ///     window, or a Python int constant of the recurrence (every one NumPy writes is affine in j).
    /// </summary>
    /// <param name="JMul">The coefficient of j (0 for a fixed row / literal).</param>
    /// <param name="Add">The constant term.</param>
    internal readonly record struct PolyCalcAffine(long JMul, long Add);

    /// <summary>
    ///     One expression node of a calculus statement. The trees are NumPy's source expressions token for token
    ///     and must never be re-associated: operand order decides complex products and NaN priority, and the
    ///     evaluation order decides rounding.
    /// </summary>
    internal abstract class PolyCalcExpr { }

    /// <summary>The value at one of the step's rows (the current column).</summary>
    internal sealed class PolyCalcLoad : PolyCalcExpr
    {
        /// <summary>Index into the step's row table.</summary>
        public readonly int Row;

        /// <summary>Reads row slot <paramref name="row"/>.</summary>
        /// <param name="row">The row slot.</param>
        public PolyCalcLoad(int row) { Row = row; }
    }

    /// <summary>A value bound earlier in the step by <see cref="PolyCalcLet"/> (NumPy reads it twice, e.g. <c>c[j]</c>).</summary>
    internal sealed class PolyCalcVar : PolyCalcExpr
    {
        /// <summary>The variable slot.</summary>
        public readonly int Var;

        /// <summary>Reads variable <paramref name="v"/>.</summary>
        /// <param name="v">The variable slot.</param>
        public PolyCalcVar(int v) { Var = v; }
    }

    /// <summary>A Python int of the step (<c>2*j</c>, <c>j - 2</c>, <c>4</c>, ...), converted to the dtype NEP 50's way.</summary>
    internal sealed class PolyCalcConst : PolyCalcExpr
    {
        /// <summary>Index into the step's constant table.</summary>
        public readonly int Const;

        /// <summary>Reads constant slot <paramref name="c"/>.</summary>
        /// <param name="c">The constant slot.</param>
        public PolyCalcConst(int c) { Const = c; }
    }

    /// <summary>A binary op in NumPy's operand order. A Multiply is a RECURRENCE product (scalarmath naive for a 1-D
    ///     complex series, simd_cmul otherwise); a Divide by a constant is prepared once per (block, j).</summary>
    internal sealed class PolyCalcBin : PolyCalcExpr
    {
        /// <summary>Add, Subtract, Multiply or Divide.</summary>
        public readonly BinaryOp Op;
        /// <summary>Left and right operands, in NumPy's source order.</summary>
        public readonly PolyCalcExpr A, B;

        /// <summary>Creates <c>a op b</c>.</summary>
        /// <param name="op">The operator.</param><param name="a">Left operand.</param><param name="b">Right operand.</param>
        public PolyCalcBin(BinaryOp op, PolyCalcExpr a, PolyCalcExpr b) { Op = op; A = a; B = b; }
    }

    /// <summary>NumPy's negative (a sign flip: -0.0 and a NaN's sign included, both parts of a complex).</summary>
    internal sealed class PolyCalcNeg : PolyCalcExpr
    {
        /// <summary>The operand.</summary>
        public readonly PolyCalcExpr A;

        /// <summary>Creates <c>-a</c>.</summary>
        /// <param name="a">The operand.</param>
        public PolyCalcNeg(PolyCalcExpr a) { A = a; }
    }

    /// <summary>One statement of a step.</summary>
    internal abstract class PolyCalcStmt { }

    /// <summary>Binds a value to a variable slot (evaluated once, read by later statements of the step).</summary>
    internal sealed class PolyCalcLet : PolyCalcStmt
    {
        /// <summary>The variable slot.</summary>
        public readonly int Var;
        /// <summary>The value.</summary>
        public readonly PolyCalcExpr E;

        /// <summary>Creates <c>var = e</c>.</summary>
        /// <param name="v">The variable slot.</param><param name="e">The value.</param>
        public PolyCalcLet(int v, PolyCalcExpr e) { Var = v; E = e; }
    }

    /// <summary>Writes a value to one of the step's rows (the current column).</summary>
    internal sealed class PolyCalcStore : PolyCalcStmt
    {
        /// <summary>The row slot.</summary>
        public readonly int Row;
        /// <summary>The value.</summary>
        public readonly PolyCalcExpr E;

        /// <summary>Creates <c>row = e</c>.</summary>
        /// <param name="row">The row slot.</param><param name="e">The value.</param>
        public PolyCalcStore(int row, PolyCalcExpr e) { Row = row; E = e; }
    }

    /// <summary>
    ///     One step of a recurrence: the rows it touches and the Python ints it uses (both affine in j), and its
    ///     statements in NumPy's order. A step with <see cref="MinLen"/> runs only when the series length reaches it
    ///     (NumPy's <c>if n &gt; 1:</c>).
    /// </summary>
    internal sealed class PolyCalcStep
    {
        /// <summary>The rows, relative to the order's window (see the file header's in-place layout).</summary>
        public PolyCalcAffine[] Rows = Array.Empty<PolyCalcAffine>();
        /// <summary>The Python int constants.</summary>
        public PolyCalcAffine[] Consts = Array.Empty<PolyCalcAffine>();
        /// <summary>The statements.</summary>
        public PolyCalcStmt[] Body = Array.Empty<PolyCalcStmt>();
        /// <summary>The smallest series length (der: NumPy's decremented n; int: len(c)) the step runs at.</summary>
        public long MinLen;
        /// <summary>Variable slots the body binds.</summary>
        public int Vars;
    }

    /// <summary>
    ///     One basis's derivative or integral: fixed head steps (int), the j loop, fixed tail steps (der). A
    ///     derivative loops j from its length DOWN to <see cref="LoopLo"/>; an integral from <see cref="LoopLo"/> UP
    ///     to its length minus one — NumPy's <c>range</c> bounds.
    /// </summary>
    internal sealed class PolyCalcRoutine
    {
        /// <summary>Steps before the loop (int only).</summary>
        public PolyCalcStep[] Head = Array.Empty<PolyCalcStep>();
        /// <summary>The loop step.</summary>
        public PolyCalcStep Loop;
        /// <summary>The loop's last (der) or first (int) j.</summary>
        public long LoopLo;
        /// <summary>Steps after the loop (der only).</summary>
        public PolyCalcStep[] Tail = Array.Empty<PolyCalcStep>();
    }

    /// <summary>
    ///     NumPy 2.4.2's twelve calculus recurrences, transcribed statement for statement. Rows follow the in-place
    ///     layout of the file header: der — c[q] at row q, der[q] at row q + 1; int — tmp[q] at row q, c[q] at
    ///     row q + 1. Each routine's comment is the NumPy source it encodes.
    /// </summary>
    internal static class PolyCalcRoutines
    {
        private static PolyCalcExpr L(int row) => new PolyCalcLoad(row);
        private static PolyCalcExpr V(int v) => new PolyCalcVar(v);
        private static PolyCalcExpr K(int c) => new PolyCalcConst(c);
        private static PolyCalcExpr Mul(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Multiply, a, b);
        private static PolyCalcExpr Div(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Divide, a, b);
        private static PolyCalcExpr Add(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Add, a, b);
        private static PolyCalcExpr Sub(PolyCalcExpr a, PolyCalcExpr b) => new PolyCalcBin(BinaryOp.Subtract, a, b);
        private static PolyCalcExpr Neg(PolyCalcExpr a) => new PolyCalcNeg(a);
        private static PolyCalcStmt Let(int v, PolyCalcExpr e) => new PolyCalcLet(v, e);
        private static PolyCalcStmt Store(int row, PolyCalcExpr e) => new PolyCalcStore(row, e);
        private static PolyCalcAffine J(long mul, long add) => new PolyCalcAffine(mul, add);
        private static PolyCalcAffine Fixed(long v) => new PolyCalcAffine(0, v);

        private static readonly PolyCalcRoutine[] s_der = BuildDer();
        private static readonly PolyCalcRoutine[] s_int = BuildInt();

        /// <summary>The recurrence of one basis.</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="integrate">{p}int (true) or {p}der (false).</param>
        /// <returns>The routine.</returns>
        public static PolyCalcRoutine Get(PolyBasis basis, bool integrate) => (integrate ? s_int : s_der)[(int)basis];

        /// <summary>A derivative whose loop is the single statement <c>der[j - 1] = K(j) * c[j]</c>.</summary>
        /// <param name="k">NumPy's multiplier: <c>j</c> (polyder, hermeder) or <c>2 * j</c> (hermder).</param>
        /// <returns>The routine.</returns>
        private static PolyCalcRoutine ScaledDer(PolyCalcAffine k) => new PolyCalcRoutine
        {
            LoopLo = 1,
            Loop = new PolyCalcStep { Rows = new[] { J(1, 0) }, Consts = new[] { k }, Body = new[] { Store(0, Mul(K(0), L(0))) } },
        };

        private static PolyCalcRoutine[] BuildDer()
        {
            var r = new PolyCalcRoutine[6];
            // polyder:  for j in range(n, 0, -1): der[j - 1] = j * c[j]
            r[(int)PolyBasis.Power] = ScaledDer(J(1, 0));
            // hermder:  for j in range(n, 0, -1): der[j - 1] = (2 * j) * c[j]
            r[(int)PolyBasis.Hermite] = ScaledDer(J(2, 0));
            // hermeder: for j in range(n, 0, -1): der[j - 1] = j * c[j]
            r[(int)PolyBasis.HermiteE] = ScaledDer(J(1, 0));
            // chebder:  for j in range(n, 2, -1): der[j - 1] = (2 * j) * c[j]; c[j - 2] += (j * c[j]) / (j - 2)
            //           if n > 1: der[1] = 4 * c[2]
            //           der[0] = c[1]                  (der[0] is written onto c[1]'s own row: nothing to do)
            r[(int)PolyBasis.Chebyshev] = new PolyCalcRoutine
            {
                LoopLo = 3,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -2) },
                    Consts = new[] { J(2, 0), J(1, 0), J(1, -2) },
                    Vars = 1,
                    Body = new[] { Let(0, L(0)), Store(0, Mul(K(0), V(0))), Store(1, Add(L(1), Div(Mul(K(1), V(0)), K(2)))) },
                },
                Tail = new[] { new PolyCalcStep { MinLen = 2, Rows = new[] { Fixed(2) }, Consts = new[] { Fixed(4) }, Body = new[] { Store(0, Mul(K(0), L(0))) } } },
            };
            // legder:   for j in range(n, 2, -1): der[j - 1] = (2 * j - 1) * c[j]; c[j - 2] += c[j]
            //           if n > 1: der[1] = 3 * c[2]
            //           der[0] = c[1]
            r[(int)PolyBasis.Legendre] = new PolyCalcRoutine
            {
                LoopLo = 3,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -2) },
                    Consts = new[] { J(2, -1) },
                    Vars = 1,
                    Body = new[] { Let(0, L(0)), Store(0, Mul(K(0), V(0))), Store(1, Add(L(1), V(0))) },
                },
                Tail = new[] { new PolyCalcStep { MinLen = 2, Rows = new[] { Fixed(2) }, Consts = new[] { Fixed(3) }, Body = new[] { Store(0, Mul(K(0), L(0))) } } },
            };
            // lagder:   for j in range(n, 1, -1): der[j - 1] = -c[j]; c[j - 1] += c[j]
            //           der[0] = -c[1]
            r[(int)PolyBasis.Laguerre] = new PolyCalcRoutine
            {
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 0), J(1, -1) },
                    Vars = 1,
                    Body = new[] { Let(0, L(0)), Store(0, Neg(V(0))), Store(1, Add(L(1), V(0))) },
                },
                Tail = new[] { new PolyCalcStep { Rows = new[] { Fixed(1) }, Body = new[] { Store(0, Neg(L(0))) } } },
            };
            return r;
        }

        /// <summary><c>tmp[0] = c[0] * 0</c> (row 0) with <c>tmp[1] = c[0]</c> left on c[0]'s own row — the head of every
        ///     integral except hermint's (<c>c[0]/2</c>) and lagint's (no multiply).</summary>
        /// <returns>The step.</returns>
        private static PolyCalcStep ZeroHead() => new PolyCalcStep
        {
            Rows = new[] { Fixed(1), Fixed(0) },
            Consts = new[] { Fixed(0) },
            Body = new[] { Store(1, Mul(L(0), K(0))) },
        };

        /// <summary>An integral whose loop is the single statement <c>tmp[j + 1] = c[j] / K(j)</c>.</summary>
        /// <param name="head">The head steps.</param>
        /// <param name="k">NumPy's divisor.</param>
        /// <returns>The routine.</returns>
        private static PolyCalcRoutine DividedInt(PolyCalcStep[] head, PolyCalcAffine k) => new PolyCalcRoutine
        {
            Head = head,
            LoopLo = 1,
            Loop = new PolyCalcStep { Rows = new[] { J(1, 1) }, Consts = new[] { k }, Body = new[] { Store(0, Div(L(0), K(0))) } },
        };

        private static PolyCalcRoutine[] BuildInt()
        {
            var r = new PolyCalcRoutine[6];
            // polyint:  tmp[0] = c[0] * 0; tmp[1] = c[0]; for j in range(1, n): tmp[j + 1] = c[j] / (j + 1)
            r[(int)PolyBasis.Power] = DividedInt(new[] { ZeroHead() }, J(1, 1));
            // hermeint: tmp[0] = c[0] * 0; tmp[1] = c[0]; for j in range(1, n): tmp[j + 1] = c[j] / (j + 1)
            r[(int)PolyBasis.HermiteE] = DividedInt(new[] { ZeroHead() }, J(1, 1));
            // hermint:  tmp[0] = c[0] * 0; tmp[1] = c[0] / 2; for j in range(1, n): tmp[j + 1] = c[j] / (2 * (j + 1))
            r[(int)PolyBasis.Hermite] = DividedInt(new[]
            {
                new PolyCalcStep
                {
                    Rows = new[] { Fixed(1), Fixed(0) },
                    Consts = new[] { Fixed(0), Fixed(2) },
                    Vars = 1,
                    Body = new[] { Let(0, L(0)), Store(1, Mul(V(0), K(0))), Store(0, Div(V(0), K(1))) },
                },
            }, J(2, 2));
            // chebint:  tmp[0] = c[0] * 0; tmp[1] = c[0]; if n > 1: tmp[2] = c[1] / 4
            //           for j in range(2, n): tmp[j + 1] = c[j] / (2 * (j + 1)); tmp[j - 1] -= c[j] / (2 * (j - 1))
            r[(int)PolyBasis.Chebyshev] = new PolyCalcRoutine
            {
                Head = new[] { ZeroHead(), new PolyCalcStep { MinLen = 2, Rows = new[] { Fixed(2) }, Consts = new[] { Fixed(4) }, Body = new[] { Store(0, Div(L(0), K(0))) } } },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 1), J(1, -1) },
                    Consts = new[] { J(2, 2), J(2, -2) },
                    Vars = 1,
                    Body = new[] { Let(0, L(0)), Store(0, Div(V(0), K(0))), Store(1, Sub(L(1), Div(V(0), K(1)))) },
                },
            };
            // legint:   tmp[0] = c[0] * 0; tmp[1] = c[0]; if n > 1: tmp[2] = c[1] / 3
            //           for j in range(2, n): t = c[j] / (2 * j + 1); tmp[j + 1] = t; tmp[j - 1] -= t
            r[(int)PolyBasis.Legendre] = new PolyCalcRoutine
            {
                Head = new[] { ZeroHead(), new PolyCalcStep { MinLen = 2, Rows = new[] { Fixed(2) }, Consts = new[] { Fixed(3) }, Body = new[] { Store(0, Div(L(0), K(0))) } } },
                LoopLo = 2,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 1), J(1, -1) },
                    Consts = new[] { J(2, 1) },
                    Vars = 2,
                    Body = new[] { Let(0, L(0)), Let(1, Div(V(0), K(0))), Store(0, V(1)), Store(1, Sub(L(1), V(1))) },
                },
            };
            // lagint:   tmp[0] = c[0]; tmp[1] = -c[0]; for j in range(1, n): tmp[j] += c[j]; tmp[j + 1] = -c[j]
            r[(int)PolyBasis.Laguerre] = new PolyCalcRoutine
            {
                Head = new[]
                {
                    new PolyCalcStep { Rows = new[] { Fixed(1), Fixed(0) }, Vars = 1, Body = new[] { Let(0, L(0)), Store(1, V(0)), Store(0, Neg(V(0))) } },
                },
                LoopLo = 1,
                Loop = new PolyCalcStep
                {
                    Rows = new[] { J(1, 1), J(1, 0) },
                    Vars = 1,
                    Body = new[] { Let(0, L(0)), Store(1, Add(L(1), V(0))), Store(0, Neg(V(0))) },
                },
            };
            return r;
        }

        /// <summary>Whether any step of <paramref name="r"/> negates (the kernel then hoists a sign mask).</summary>
        /// <param name="r">The routine.</param>
        /// <returns>True when a <see cref="PolyCalcNeg"/> appears.</returns>
        public static bool UsesNeg(PolyCalcRoutine r)
        {
            static bool E(PolyCalcExpr e) => e switch
            {
                PolyCalcNeg => true,
                PolyCalcBin b => E(b.A) || E(b.B),
                _ => false,
            };
            static bool S(PolyCalcStep s)
            {
                foreach (var st in s.Body)
                    if (E(st is PolyCalcLet l ? l.E : ((PolyCalcStore)st).E)) return true;
                return false;
            }
            foreach (var s in r.Head) if (S(s)) return true;
            foreach (var s in r.Tail) if (S(s)) return true;
            return S(r.Loop);
        }
    }

    public static partial class DirectILKernelGenerator
    {
        private static readonly ConcurrentDictionary<PolyCalcKey, Lazy<PolyCalcKernel>> s_polyCalcKernels = new();
        private static long s_polyCalcVectorFallbacks;

        /// <summary>How many calculus kernels fell back to scalar-only code because a vector lane kind or conversion
        ///     was missing. Zero for every combination the lane table covers; the tests pin it at zero.</summary>
        internal static long PolyCalcVectorFallbacks => Interlocked.Read(ref s_polyCalcVectorFallbacks);

        /// <summary>
        ///     Returns (compiling once) the calculus kernel for <paramref name="key"/>. Thread-safe: concurrent first
        ///     calls compile exactly one kernel (a <see cref="Lazy{T}"/> per key).
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="PlatformNotSupportedException">The runtime cannot emit dynamic code (NativeAOT).</exception>
        internal static PolyCalcKernel GetPolyCalcKernel(in PolyCalcKey key)
            => s_polyCalcKernels.GetOrAdd(key, static k => new Lazy<PolyCalcKernel>(() => CompilePolyCalc(k))).Value;

        /// <summary>Emits one kernel with vector blocks where the lane table allows them, falling back to scalar-only code
        ///     when a lane kind or conversion turns out to be missing (the same bits, slower).</summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="PlatformNotSupportedException">No dynamic code.</exception>
        private static PolyCalcKernel CompilePolyCalc(PolyCalcKey key)
        {
            if (!RuntimeFeature.IsDynamicCodeSupported)
                throw new PlatformNotSupportedException(
                    "numpy.polynomial calculus compiles IL kernels at runtime; this runtime (NativeAOT) cannot emit dynamic code.");
            try
            {
                return EmitPolyCalc(key, allowVector: true);
            }
            catch (NotSupportedException) when (!key.ScalarMath)
            {
                // A lane kind or conversion the table lacks: restart with scalar code only (a fresh set of
                // DynamicMethods, so nothing half-built survives). Counted so a test can prove it never happens.
                Interlocked.Increment(ref s_polyCalcVectorFallbacks);
                return EmitPolyCalc(key, allowVector: false);
            }
        }

        /// <summary>What every stage of one calculus kernel shares.</summary>
        private sealed class PolyCalcEmit
        {
            /// <summary>The kernel identity.</summary>
            public PolyCalcKey Key;
            /// <summary>The recurrence.</summary>
            public PolyCalcRoutine Routine;
            /// <summary>Element sizes of T and of the source dtype.</summary>
            public int Size, SrcSize;
            /// <summary>The coefficient dtype's vector lane kind, or null when the kernel is scalar-only.</summary>
            public PolyValueKind Vk;
            /// <summary>Lanes per vector (1 when scalar-only).</summary>
            public int W = 1;
            /// <summary>The source dtype's lane kind (loaded then lane-converted to T), or null when the source load has
            ///     no vector path (scalar-only kernels, bool sources, pairs without an exact lane conversion).</summary>
            public PolyValueKind SrcVk;
        }

        /// <summary>
        ///     Emits and compiles one calculus kernel: the root (column-block loop, order loop), two load stages for the
        ///     source (vector over contiguous columns / scalar over any stride), in-place scaling stages for T, and the
        ///     recurrence stage. Each stage is its own DynamicMethod so the JIT's per-method inline budget covers the
        ///     lane helpers of one stage only (a single method left them as calls in the evaluation kernels).
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <param name="allowVector">Emit vector loops where the lane table allows them.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="NotSupportedException">A lane kind or conversion is missing (only when vectors are allowed).</exception>
        private static PolyCalcKernel EmitPolyCalc(PolyCalcKey key, bool allowVector)
        {
            var e = new PolyCalcEmit
            {
                Key = key,
                Routine = PolyCalcRoutines.Get(key.Basis, key.Integrate),
                Size = GetTypeSize(key.T),
                SrcSize = GetTypeSize(key.Src),
            };
            if (allowVector && !key.ScalarMath && key.T != NPTypeCode.Decimal)
            {
                int lanes = PolyLanes.LoopLanes(key.T);
                if (lanes > 1 && PolyLanes.HasLaneKind(key.T, lanes))
                {
                    e.Vk = PolyLanes.CreateLaneKind(key.T, lanes);
                    e.W = lanes;
                }
            }
            if (e.Vk is not null)
            {
                // A bool source stays scalar: its lane kind would read the raw storage byte, where NumPy's cast (and the
                // house scalar EmitConvertTo) reads True for ANY nonzero byte.
                if (key.Src == key.T) e.SrcVk = e.Vk;
                else if (key.Src != NPTypeCode.Boolean && PolyLanes.LaneConvertible(key.Src, key.T, e.W))
                    e.SrcVk = PolyLanes.CreateLaneKind(key.Src, e.W);
            }

            string name = $"NDPolyCalc_{key.Basis}_{(key.Integrate ? "int" : "der")}_{key.T}_{key.Src}"
                          + $"{(key.ScalarMath ? "_s" : "")}{(key.Scale ? "_scl" : "")}{(e.Vk is null ? "_scalar" : "")}";

            var loadSrcVec = e.SrcVk is null ? null : EmitPolyCalcLoad(e, key.Src, e.SrcVk, name + "_loadv");
            var loadSrcScalar = EmitPolyCalcLoad(e, key.Src, null, name + "_loads");
            DynamicMethod scaleT = null;
            if (key.Scale)
            {
                // In-place `c *= scl` of rows already in the buffer (contiguous columns of T).
                scaleT = key.Src == key.T ? (loadSrcVec ?? loadSrcScalar)
                       : e.Vk is not null ? EmitPolyCalcLoad(e, key.T, e.Vk, name + "_scalev")
                       : EmitPolyCalcLoad(e, key.T, null, name + "_scales");
            }
            var recur = EmitPolyCalcRecur(e, name + "_recur");

            var root = new DynamicMethod(name, typeof(void),
                new[] { typeof(byte*), typeof(long), typeof(long), typeof(byte*), typeof(long), typeof(long), typeof(long), typeof(long), typeof(long), typeof(byte*) },
                typeof(DirectILKernelGenerator), skipVisibility: true);
            EmitPolyCalcRoot(root.GetILGenerator(), e, loadSrcVec, loadSrcScalar, scaleT, recur);
            return root.CreateDelegate<PolyCalcKernel>();
        }

        /// <summary>A stage DynamicMethod owned by this class (skipVisibility: the lane helpers are internal).</summary>
        /// <param name="name">Its name (shows in JIT disassembly).</param>
        /// <param name="args">Parameter types.</param>
        /// <returns>The method.</returns>
        private static DynamicMethod NewPolyCalcStage(string name, params Type[] args)
            => new DynamicMethod(name, typeof(void), args, typeof(DirectILKernelGenerator), skipVisibility: true);

        /// <summary>Pushes <c>ptr + idx * size</c> (a byte pointer).</summary>
        /// <param name="il">The generator.</param><param name="ptr">A byte* local.</param>
        /// <param name="idx">A long local (element index).</param><param name="size">Bytes per element.</param>
        private static void EmitPolyCalcAddr(ILGenerator il, LocalBuilder ptr, LocalBuilder idx, int size)
        {
            il.Emit(OpCodes.Ldloc, ptr);
            il.Emit(OpCodes.Ldloc, idx);
            il.Emit(OpCodes.Ldc_I8, (long)size);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        /// <summary><c>local += step</c> for a long local.</summary>
        /// <param name="il">The generator.</param><param name="local">The local.</param><param name="step">The increment.</param>
        private static void EmitPolyCalcBump(ILGenerator il, LocalBuilder local, long step)
        {
            il.Emit(OpCodes.Ldloc, local);
            il.Emit(OpCodes.Ldc_I8, step);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, local);
        }

        /// <summary>
        ///     [T, T] → [T]: one scalar op of the coefficient dtype. A Multiply takes <paramref name="product"/>'s complex
        ///     form (every other dtype and op has one form: the house scalar op, NumPy's loop arithmetic).
        /// </summary>
        /// <param name="il">The generator.</param><param name="op">The op.</param><param name="t">The dtype.</param>
        /// <param name="product">Which NumPy complex product a Complex multiply reproduces.</param>
        private static void EmitPolyCalcScalarBin(ILGenerator il, BinaryOp op, NPTypeCode t, PolyComplexProduct product)
        {
            if (t == NPTypeCode.Complex && op == BinaryOp.Multiply && product == PolyComplexProduct.Naive)
                il.EmitCall(OpCodes.Call, ILKernelGenerator.s_polyNaiveComplexMultiply, null);
            else
                EmitScalarOperation(il, op, t);
        }

        /// <summary>
        ///     [long K] → [T]: NumPy's NEP 50 conversion of the Python int K into the coefficient dtype — exact into
        ///     decimal, otherwise through a double (<c>PyLong_AsDouble</c>, exact for every K a series can reach) and the
        ///     house cast (so a float16 constant rounds to nearest-even exactly as NumPy's <c>npy_double_to_half</c>, and
        ///     a complex one is <c>(K + 0j)</c>).
        /// </summary>
        /// <param name="il">The generator.</param><param name="t">The coefficient dtype.</param>
        private static void EmitPolyCalcWeakInt(ILGenerator il, NPTypeCode t)
        {
            if (t == NPTypeCode.Decimal)
            {
                EmitConvertTo(il, NPTypeCode.Int64, NPTypeCode.Decimal);
                return;
            }
            il.Emit(OpCodes.Conv_R8);
            EmitConvertTo(il, NPTypeCode.Double, t);
        }

        /// <summary>
        ///     The root: <c>for p0 in blocks: load (or scale in place); recurrence (all orders of a derivative)</c>. See
        ///     <see cref="PolyCalcKernel"/> for the arguments (0 src, 1 srcRow, 2 srcCol, 3 buf, 4 bufRow, 5 cols, 6 n,
        ///     7 orders, 8 block, 9 scl).
        /// </summary>
        /// <param name="il">The generator.</param><param name="e">The kernel's shared facts.</param>
        /// <param name="loadSrcVec">Vector source load (contiguous columns), or null.</param>
        /// <param name="loadSrcScalar">Scalar source load (any column stride).</param>
        /// <param name="scaleT">In-place scaling of T rows, or null for a non-scaling kernel.</param>
        /// <param name="recur">The recurrence stage.</param>
        private static void EmitPolyCalcRoot(ILGenerator il, PolyCalcEmit e, DynamicMethod loadSrcVec, DynamicMethod loadSrcScalar,
            DynamicMethod scaleT, DynamicMethod recur)
        {
            var key = e.Key;
            var p0 = il.DeclareLocal(typeof(long));
            var w = il.DeclareLocal(typeof(long));
            var cRow0 = il.DeclareLocal(typeof(byte*));
            var s0 = il.DeclareLocal(typeof(byte*));
            var o = il.DeclareLocal(typeof(long));
            var b = il.DeclareLocal(typeof(byte*));
            var pTop = il.DefineLabel(); var pEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, p0);
            il.MarkLabel(pTop);
            il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldarg_S, (byte)5); il.Emit(OpCodes.Bge, pEnd);

            // w = min(block, cols - p0)
            var wOk = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_S, (byte)5); il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, w);
            il.Emit(OpCodes.Ldloc, w); il.Emit(OpCodes.Ldarg_S, (byte)8); il.Emit(OpCodes.Ble, wOk);
            il.Emit(OpCodes.Ldarg_S, (byte)8); il.Emit(OpCodes.Stloc, w);
            il.MarkLabel(wOk);

            // cRow0 = buf (+ bufRow for an integral: c[0] sits one row below tmp[0]) + p0 * size
            il.Emit(OpCodes.Ldarg_3);
            if (key.Integrate) { il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, cRow0);

            // Load phase: from src when given, else scale the rows already in the buffer.
            var noSrc = il.DefineLabel(); var loaded = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Conv_U); il.Emit(OpCodes.Beq, noSrc);
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, s0);
            void CallLoad(DynamicMethod stage)
            {
                il.Emit(OpCodes.Ldloc, s0); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, cRow0); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Ldarg_S, (byte)6);
                il.Emit(OpCodes.Ldloc, w); il.Emit(OpCodes.Ldarg_S, (byte)9);
                il.Emit(OpCodes.Call, stage);
            }
            if (loadSrcVec is not null)
            {
                // Contiguous source columns take the vector load; any other column stride the scalar one.
                var strided = il.DefineLabel();
                il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldc_I8, (long)e.SrcSize); il.Emit(OpCodes.Bne_Un, strided);
                CallLoad(loadSrcVec);
                il.Emit(OpCodes.Br, loaded);
                il.MarkLabel(strided);
            }
            CallLoad(loadSrcScalar);
            il.Emit(OpCodes.Br, loaded);
            il.MarkLabel(noSrc);
            if (scaleT is not null)
            {
                // c *= scl over rows already in the buffer: source = destination, contiguous columns.
                il.Emit(OpCodes.Ldloc, cRow0); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Ldc_I8, (long)e.Size);
                il.Emit(OpCodes.Ldloc, cRow0); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Ldarg_S, (byte)6);
                il.Emit(OpCodes.Ldloc, w); il.Emit(OpCodes.Ldarg_S, (byte)9);
                il.Emit(OpCodes.Call, scaleT);
            }
            il.MarkLabel(loaded);

            if (key.Integrate)
            {
                // One order: tmp rows [0, n] from c rows [1, n].
                il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Ldloc, w);
                il.Emit(OpCodes.Call, recur);
            }
            else
            {
                // Order o: c = rows [o, n), n - o coefficients; its derivative (n - o - 1 of them) lands on rows [o+1, n).
                var oTop = il.DefineLabel(); var oEnd = il.DefineLabel();
                il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, o);
                il.MarkLabel(oTop);
                il.Emit(OpCodes.Ldloc, o); il.Emit(OpCodes.Ldarg_S, (byte)7); il.Emit(OpCodes.Bge, oEnd);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, o); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldc_I8, (long)e.Size); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, b);
                if (scaleT is not null)
                {
                    // The first order was scaled by the load phase; every later order scales its own (shorter) c.
                    var noScale = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, o); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Beq, noScale);
                    il.Emit(OpCodes.Ldloc, b); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Ldc_I8, (long)e.Size);
                    il.Emit(OpCodes.Ldloc, b); il.Emit(OpCodes.Ldarg_S, (byte)4);
                    il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Ldloc, o); il.Emit(OpCodes.Sub);
                    il.Emit(OpCodes.Ldloc, w); il.Emit(OpCodes.Ldarg_S, (byte)9);
                    il.Emit(OpCodes.Call, scaleT);
                    il.MarkLabel(noScale);
                }
                il.Emit(OpCodes.Ldloc, b); il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Ldloc, o); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Ldloc, w);
                il.Emit(OpCodes.Call, recur);
                EmitPolyCalcBump(il, o, 1);
                il.Emit(OpCodes.Br, oTop);
                il.MarkLabel(oEnd);
            }

            il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldarg_S, (byte)8); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, p0);
            il.Emit(OpCodes.Br, pTop);
            il.MarkLabel(pEnd);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>
        ///     A load stage <c>(byte* s, long sRow, long sCol, byte* d, long dRow, long rows, long w, byte* scl)</c>: for
        ///     every row, <c>d[i] = T(s[i])</c> (times <c>scl</c> for a scaling kernel — NumPy's <c>c *= scl</c>, an array
        ///     op, so the complex product is simd_cmul with c as the first operand) over the block's w columns. With a
        ///     vector kind the columns must be contiguous (<c>sCol</c> = source element size); the scalar tail — or the
        ///     whole row for a scalar stage — reads through <c>sCol</c>. <c>s</c> may equal <c>d</c> (in-place scaling):
        ///     every element is read before it is written.
        /// </summary>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="from">The dtype read from <c>s</c>.</param>
        /// <param name="fromVk">Its lane kind for vector loads, or null for a scalar stage.</param>
        /// <param name="name">The stage name.</param>
        /// <returns>The stage.</returns>
        /// <exception cref="NotSupportedException">A lane conversion the table lacks.</exception>
        private static DynamicMethod EmitPolyCalcLoad(PolyCalcEmit e, NPTypeCode from, PolyValueKind fromVk, string name)
        {
            var key = e.Key;
            var dm = NewPolyCalcStage(name, typeof(byte*), typeof(long), typeof(long), typeof(byte*), typeof(long), typeof(long), typeof(long), typeof(byte*));
            var il = dm.GetILGenerator();
            int fromSize = GetTypeSize(from);
            bool vec = fromVk is not null && e.Vk is not null;
            var q = il.DeclareLocal(typeof(long));
            var i = il.DeclareLocal(typeof(long));
            var sp = il.DeclareLocal(typeof(byte*));
            var dp = il.DeclareLocal(typeof(byte*));
            LocalBuilder sclS = null, sclV = null;
            if (key.Scale)
            {
                sclS = il.DeclareLocal(GetClrType(key.T));
                il.Emit(OpCodes.Ldarg_S, (byte)7); EmitLoadIndirect(il, key.T); il.Emit(OpCodes.Stloc, sclS);
                if (vec)
                {
                    sclV = il.DeclareLocal(e.Vk.LocalType);
                    il.Emit(OpCodes.Ldloc, sclS); e.Vk.BroadcastFromScalar(il); il.Emit(OpCodes.Stloc, sclV);
                }
            }

            var qTop = il.DefineLabel(); var qEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, q);
            il.MarkLabel(qTop);
            il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldarg_S, (byte)5); il.Emit(OpCodes.Bge, qEnd);
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, sp);
            il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, dp);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);

            if (vec)
            {
                var vTop = il.DefineLabel(); var vEnd = il.DefineLabel();
                il.MarkLabel(vTop);
                il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, (long)e.W); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Bgt, vEnd);
                EmitPolyCalcAddr(il, sp, i, fromSize);
                fromVk.Load(il);
                if (from != key.T) PolyLanes.EmitLaneConvert(il, from, key.T, e.W);
                if (key.Scale) { il.Emit(OpCodes.Ldloc, sclV); e.Vk.Bin(il, BinaryOp.Multiply, PolyComplexProduct.Simd); }
                EmitPolyCalcAddr(il, dp, i, e.Size);
                e.Vk.StoreValueFirst(il);
                EmitPolyCalcBump(il, i, e.W);
                il.Emit(OpCodes.Br, vTop);
                il.MarkLabel(vEnd);
            }

            // Scalar tail (the whole row for a scalar stage): d[i] = T(s[i * sCol]) [* scl].
            var tTop = il.DefineLabel(); var tEnd = il.DefineLabel();
            il.MarkLabel(tTop);
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_S, (byte)6); il.Emit(OpCodes.Bge, tEnd);
            EmitPolyCalcAddr(il, dp, i, e.Size);
            il.Emit(OpCodes.Ldloc, sp); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, from);
            EmitConvertTo(il, from, key.T);
            if (key.Scale) { il.Emit(OpCodes.Ldloc, sclS); EmitPolyCalcScalarBin(il, BinaryOp.Multiply, key.T, PolyComplexProduct.Simd); }
            EmitStoreIndirect(il, key.T);
            EmitPolyCalcBump(il, i, 1);
            il.Emit(OpCodes.Br, tTop);
            il.MarkLabel(tEnd);

            EmitPolyCalcBump(il, q, 1);
            il.Emit(OpCodes.Br, qTop);
            il.MarkLabel(qEnd);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>
        ///     The recurrence stage <c>(byte* base, long bufRow, long len, long w)</c>: the routine's head steps, its j loop
        ///     and its tail steps over the block's w columns. <c>len</c> is the routine's length variable — NumPy's
        ///     decremented n for a derivative (the loop runs j = len .. LoopLo), len(c) for an integral (j = LoopLo ..
        ///     len - 1).
        /// </summary>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="name">The stage name.</param>
        /// <returns>The stage.</returns>
        /// <exception cref="NotSupportedException">A lane op the kind lacks.</exception>
        private static DynamicMethod EmitPolyCalcRecur(PolyCalcEmit e, string name)
        {
            var dm = NewPolyCalcStage(name, typeof(byte*), typeof(long), typeof(long), typeof(long));
            var il = dm.GetILGenerator();
            var r = e.Routine;
            LocalBuilder negMask = null;
            if (e.Vk is not null && PolyCalcRoutines.UsesNeg(r))
            {
                // NumPy's negative is a sign flip (xor), never 0 - x: -(+0.0) is -0.0 and a NaN's sign flips too. The mask is
                // -0.0 in the kind's element type (float lanes also carry float16 values, double lanes complex parts).
                negMask = il.DeclareLocal(e.Vk.LocalType);
                if (e.Vk.Clr == typeof(float)) il.Emit(OpCodes.Ldc_R4, -0.0f);
                else il.Emit(OpCodes.Ldc_R8, -0.0);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(e.Vk.Bits, e.Vk.Clr), null);
                il.Emit(OpCodes.Stloc, negMask);
            }
            var j = il.DeclareLocal(typeof(long));

            void Fixed(PolyCalcStep s)
            {
                if (s.MinLen > 0)
                {
                    var skip = il.DefineLabel();
                    il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldc_I8, s.MinLen); il.Emit(OpCodes.Blt, skip);
                    EmitPolyCalcStep(il, e, s, null, negMask);
                    il.MarkLabel(skip);
                }
                else EmitPolyCalcStep(il, e, s, null, negMask);
            }

            foreach (var s in r.Head) Fixed(s);

            var top = il.DefineLabel(); var end = il.DefineLabel();
            if (e.Key.Integrate)
            {
                il.Emit(OpCodes.Ldc_I8, r.LoopLo); il.Emit(OpCodes.Stloc, j);
                il.MarkLabel(top);
                il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, end);
                EmitPolyCalcStep(il, e, r.Loop, j, negMask);
                EmitPolyCalcBump(il, j, 1);
            }
            else
            {
                il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Stloc, j);
                il.MarkLabel(top);
                il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldc_I8, r.LoopLo); il.Emit(OpCodes.Blt, end);
                EmitPolyCalcStep(il, e, r.Loop, j, negMask);
                EmitPolyCalcBump(il, j, -1);
            }
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(end);

            foreach (var s in r.Tail) Fixed(s);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>Pushes the affine value <c>a.JMul*j + a.Add</c> (a long).</summary>
        /// <param name="il">The generator.</param><param name="a">The affine function.</param><param name="j">The j local, or null for a fixed value.</param>
        private static void EmitPolyCalcAffine(ILGenerator il, PolyCalcAffine a, LocalBuilder j)
        {
            if (j is null || a.JMul == 0)
            {
                il.Emit(OpCodes.Ldc_I8, a.Add);
                return;
            }
            il.Emit(OpCodes.Ldloc, j);
            if (a.JMul != 1) { il.Emit(OpCodes.Ldc_I8, a.JMul); il.Emit(OpCodes.Mul); }
            if (a.Add != 0) { il.Emit(OpCodes.Ldc_I8, a.Add); il.Emit(OpCodes.Add); }
        }

        /// <summary>
        ///     One step for the current j (or a fixed step): the row pointers and the constants (scalar, vector broadcast and,
        ///     for a complex vector divisor, the Smith preparation) once, then the vector loop over the block's columns and
        ///     the scalar tail — each running the step's statements in NumPy's order.
        /// </summary>
        /// <param name="il">The generator (the recurrence stage: arg 0 base, 1 bufRow, 2 len, 3 w).</param>
        /// <param name="e">The kernel's shared facts.</param>
        /// <param name="s">The step.</param>
        /// <param name="j">The j local, or null for a fixed step.</param>
        /// <param name="negMask">The sign mask local (vector kernels that negate), or null.</param>
        private static void EmitPolyCalcStep(ILGenerator il, PolyCalcEmit e, PolyCalcStep s, LocalBuilder j, LocalBuilder negMask)
        {
            var t = e.Key.T;
            var rows = new LocalBuilder[s.Rows.Length];
            for (int r = 0; r < rows.Length; r++)
            {
                rows[r] = il.DeclareLocal(typeof(byte*));
                il.Emit(OpCodes.Ldarg_0);
                EmitPolyCalcAffine(il, s.Rows[r], j);
                il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, rows[r]);
            }

            var constS = new LocalBuilder[s.Consts.Length];
            var constV = new LocalBuilder[s.Consts.Length];
            var prep = new LocalBuilder[s.Consts.Length];
            bool[] divisor = new bool[s.Consts.Length];
            foreach (var st in s.Body) MarkDivisors(st is PolyCalcLet l ? l.E : ((PolyCalcStore)st).E, divisor);
            for (int c = 0; c < constS.Length; c++)
            {
                constS[c] = il.DeclareLocal(GetClrType(t));
                EmitPolyCalcAffine(il, s.Consts[c], j);
                EmitPolyCalcWeakInt(il, t);
                il.Emit(OpCodes.Stloc, constS[c]);
                if (e.Vk is null) continue;
                if (t == NPTypeCode.Complex && divisor[c])
                {
                    // CDOUBLE_divide's branch, rat and scl depend on the divisor alone: prepared once, the same values
                    // NumPy recomputes per element, so each vector runs only the per-point half.
                    prep[c] = il.DeclareLocal(typeof(PolyCDivShared));
                    il.Emit(OpCodes.Ldloc, constS[c]);
                    il.EmitCall(OpCodes.Call, PolyLaneOps.s_cDivPrep, null);
                    il.Emit(OpCodes.Stloc, prep[c]);
                }
                constV[c] = il.DeclareLocal(e.Vk.LocalType);
                il.Emit(OpCodes.Ldloc, constS[c]);
                e.Vk.BroadcastFromScalar(il);
                il.Emit(OpCodes.Stloc, constV[c]);
            }

            var i = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);
            if (e.Vk is not null)
            {
                var vTop = il.DefineLabel(); var vEnd = il.DefineLabel();
                il.MarkLabel(vTop);
                il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, (long)e.W); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Bgt, vEnd);
                var vars = new LocalBuilder[s.Vars];
                for (int v = 0; v < vars.Length; v++) vars[v] = il.DeclareLocal(e.Vk.LocalType);
                var ctx = new PolyCalcBodyCtx { E = e, Rows = rows, Consts = constV, Prep = prep, Vars = vars, I = i, Vector = true, NegMask = negMask };
                foreach (var st in s.Body) EmitPolyCalcStmt(il, ctx, st);
                EmitPolyCalcBump(il, i, e.W);
                il.Emit(OpCodes.Br, vTop);
                il.MarkLabel(vEnd);
            }

            var tTop = il.DefineLabel(); var tEnd = il.DefineLabel();
            il.MarkLabel(tTop);
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Bge, tEnd);
            {
                var vars = new LocalBuilder[s.Vars];
                for (int v = 0; v < vars.Length; v++) vars[v] = il.DeclareLocal(GetClrType(t));
                var ctx = new PolyCalcBodyCtx { E = e, Rows = rows, Consts = constS, Prep = prep, Vars = vars, I = i, Vector = false, NegMask = null };
                foreach (var st in s.Body) EmitPolyCalcStmt(il, ctx, st);
            }
            EmitPolyCalcBump(il, i, 1);
            il.Emit(OpCodes.Br, tTop);
            il.MarkLabel(tEnd);

            static void MarkDivisors(PolyCalcExpr x, bool[] divisor)
            {
                switch (x)
                {
                    case PolyCalcBin b:
                        if (b.Op == BinaryOp.Divide && b.B is PolyCalcConst k) divisor[k.Const] = true;
                        MarkDivisors(b.A, divisor);
                        MarkDivisors(b.B, divisor);
                        break;
                    case PolyCalcNeg n:
                        MarkDivisors(n.A, divisor);
                        break;
                }
            }
        }

        /// <summary>The emission state of one step body in one mode (vector loop or scalar tail).</summary>
        private sealed class PolyCalcBodyCtx
        {
            /// <summary>The kernel's shared facts.</summary>
            public PolyCalcEmit E;
            /// <summary>Row pointer locals (the block's first column of each row slot).</summary>
            public LocalBuilder[] Rows;
            /// <summary>Constant locals of this mode (scalar T, or the kind's broadcast vector).</summary>
            public LocalBuilder[] Consts;
            /// <summary>Complex divisor preparations (vector mode, divisor constants only).</summary>
            public LocalBuilder[] Prep;
            /// <summary>Variable locals of this mode.</summary>
            public LocalBuilder[] Vars;
            /// <summary>The column index local.</summary>
            public LocalBuilder I;
            /// <summary>Vector (true) or scalar (false) mode.</summary>
            public bool Vector;
            /// <summary>The sign mask (vector mode, negating routines).</summary>
            public LocalBuilder NegMask;
        }

        /// <summary>Emits one statement of a step body.</summary>
        /// <param name="il">The generator.</param><param name="ctx">The body state.</param><param name="st">The statement.</param>
        private static void EmitPolyCalcStmt(ILGenerator il, PolyCalcBodyCtx ctx, PolyCalcStmt st)
        {
            var e = ctx.E;
            switch (st)
            {
                case PolyCalcLet let:
                    EmitPolyCalcExpr(il, ctx, let.E);
                    il.Emit(OpCodes.Stloc, ctx.Vars[let.Var]);
                    break;
                case PolyCalcStore store:
                    if (ctx.Vector)
                    {
                        EmitPolyCalcExpr(il, ctx, store.E);
                        EmitPolyCalcAddr(il, ctx.Rows[store.Row], ctx.I, e.Size);
                        e.Vk.StoreValueFirst(il);
                    }
                    else
                    {
                        EmitPolyCalcAddr(il, ctx.Rows[store.Row], ctx.I, e.Size);
                        EmitPolyCalcExpr(il, ctx, store.E);
                        EmitStoreIndirect(il, e.Key.T);
                    }
                    break;
                default:
                    throw new InvalidOperationException("unknown calculus statement");
            }
        }

        /// <summary>Pushes the value of an expression in the body's mode.</summary>
        /// <param name="il">The generator.</param><param name="ctx">The body state.</param><param name="x">The expression.</param>
        private static void EmitPolyCalcExpr(ILGenerator il, PolyCalcBodyCtx ctx, PolyCalcExpr x)
        {
            var e = ctx.E;
            var t = e.Key.T;
            switch (x)
            {
                case PolyCalcLoad ld:
                    EmitPolyCalcAddr(il, ctx.Rows[ld.Row], ctx.I, e.Size);
                    if (ctx.Vector) e.Vk.Load(il);
                    else EmitLoadIndirect(il, t);
                    break;
                case PolyCalcVar v:
                    il.Emit(OpCodes.Ldloc, ctx.Vars[v.Var]);
                    break;
                case PolyCalcConst k:
                    il.Emit(OpCodes.Ldloc, ctx.Consts[k.Const]);
                    break;
                case PolyCalcBin b:
                    if (ctx.Vector && b.Op == BinaryOp.Divide && t == NPTypeCode.Complex && b.B is PolyCalcConst kd)
                    {
                        EmitPolyCalcExpr(il, ctx, b.A);
                        il.Emit(OpCodes.Ldloca, ctx.Prep[kd.Const]);
                        il.EmitCall(OpCodes.Call, PolyLaneOps.s_cDivBy, null);
                        break;
                    }
                    EmitPolyCalcExpr(il, ctx, b.A);
                    EmitPolyCalcExpr(il, ctx, b.B);
                    if (ctx.Vector) e.Vk.Bin(il, b.Op, PolyComplexProduct.Simd);
                    else EmitPolyCalcScalarBin(il, b.Op, t, e.Key.ScalarMath ? PolyComplexProduct.Naive : PolyComplexProduct.Simd);
                    break;
                case PolyCalcNeg n:
                    EmitPolyCalcExpr(il, ctx, n.A);
                    if (ctx.Vector)
                    {
                        var vt = e.Vk.LocalType;
                        il.Emit(OpCodes.Ldloc, ctx.NegMask);
                        il.EmitCall(OpCodes.Call, vt.GetMethod("op_ExclusiveOr", new[] { vt, vt })
                                                  ?? throw new MissingMethodException(vt.Name, "op_ExclusiveOr"), null);
                    }
                    else EmitUnaryScalarOperation(il, UnaryOp.Negate, t);
                    break;
                default:
                    throw new InvalidOperationException("unknown calculus expression");
            }
        }
    }
}
