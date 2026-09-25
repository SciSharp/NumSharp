using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Numerics;

// =============================================================================
// ILKernelGenerator.Polynomial.cs — numpy.polynomial evaluation as DATA
// =============================================================================
//
// WHAT THIS FILE HOLDS
// --------------------
// The six numpy.polynomial bases evaluate their series with two recurrences: Horner for the power
// basis (polyval) and Clenshaw for chebval/legval/lagval/hermval/hermeval. NumPy writes each one as
// a handful of Python lines over whole arrays. Here those lines are transcribed token for token into
// small expression trees (PolySteps.Eval) — the bases are DATA, and ONE typed IL emitter
// (ILKernelGenerator.Polynomial.Emitter.cs) turns any of them into a per-chunk Tier-3A kernel
// (ILKernelGenerator.Polynomial.Eval.cs). There is no per-basis and no per-dtype C#.
//
// WHY TREES AND NOT C# FORMULAS
// -----------------------------
// NumPy's bits are decided by three things a formula rewrite silently changes:
//   * operand ORDER — complex multiply is not commutative bit-for-bit, and float16 NaN priority
//     depends on which operand is first;
//   * evaluation ORDER — `c1*x*((2*nd - 1)/nd)` (legval: two array multiplies) and
//     `(c1*(nd - 1))/nd` (lagval: a multiply then a divide) are the same algebra and different bits;
//   * the dtype of EVERY intermediate — NumPy types each op on its own (NEP 50), so a float32 series
//     at a float64 x runs its first Clenshaw steps IN FLOAT32 (see PeelCount in the typing file).
// A tree preserves all three by construction, and the typing pass reads the same tree the emitter
// emits, so the planned dtypes and the emitted ones cannot drift apart.
//
// PYTHON SCALARS (NEP 50 "weak" values)
// -------------------------------------
// The literals in the source lines (2, 0, (nd - 1)/nd, 2*(nd - 1), ...) are Python ints/floats: they
// carry no dtype and adopt their partner operand's. When the caller's x is itself a Python scalar
// (a C# primitive here), NumPy never wraps it in an array: every x-only subexpression (`2*x`,
// `x*2`, `x*0`, `1 - x`, `(2*nd - 1) - x`) is computed by PYTHON, in Python's int/float/complex
// arithmetic, and only its result meets NumPy. PolySteps.RewriteForWeakX reproduces that by folding
// every all-weak subtree into one PolyWeakConst whose value is computed with PyScalar — CPython 3.12's
// own arithmetic — so the result is bit-identical, signed zeros and NaN included.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     The six <c>numpy.polynomial</c> bases. The kernel cache keys on it, and the facades pass it to
    ///     <c>NDPolyEval</c>; nothing else about a basis is code — its recurrence is the data in
    ///     <see cref="PolySteps"/>.
    /// </summary>
    internal enum PolyBasis : byte
    {
        /// <summary>Power series (<c>numpy.polynomial.polynomial</c>) — Horner.</summary>
        Power,
        /// <summary>Chebyshev series of the first kind (<c>numpy.polynomial.chebyshev</c>) — Clenshaw.</summary>
        Chebyshev,
        /// <summary>Legendre series (<c>numpy.polynomial.legendre</c>) — Clenshaw.</summary>
        Legendre,
        /// <summary>Laguerre series (<c>numpy.polynomial.laguerre</c>) — Clenshaw, two divisions per step.</summary>
        Laguerre,
        /// <summary>Physicists' Hermite series (<c>numpy.polynomial.hermite</c>) — Clenshaw.</summary>
        Hermite,
        /// <summary>Probabilists' Hermite series (<c>numpy.polynomial.hermite_e</c>) — Clenshaw.</summary>
        HermiteE,
    }

    /// <summary>
    ///     The values a step expression may read. <see cref="Tmp"/> is Clenshaw's saved <c>c0</c> (NumPy's
    ///     <c>tmp = c0</c>), <see cref="X2"/> the x-only pre-op (<c>x2 = 2*x</c> / <c>x*2</c>) and
    ///     <see cref="Ck"/> the coefficient read by the current step (<c>c[-i]</c>).
    /// </summary>
    internal enum PolySym : byte { X, X2, C0, C1, Tmp, Ck }

    /// <summary>
    ///     The Python type of a weak (NEP 50) value. <c>bool</c> is folded into <see cref="Int"/> at the
    ///     boundary: every Python-level operation the step tables perform on a bool (<c>2*x</c>,
    ///     <c>x*0</c>, <c>1 - x</c>) yields an int, and the one place a bare bool meets NumPy
    ///     (<c>c1*x</c>) the coefficients are always inexact, where a Python bool and a Python int of the
    ///     same value promote and convert identically.
    /// </summary>
    internal enum PyKind : byte
    {
        /// <summary>Python <c>int</c> (arbitrary precision).</summary>
        Int,
        /// <summary>Python <c>float</c> (IEEE binary64).</summary>
        Float,
        /// <summary>Python <c>complex</c> (two binary64 components).</summary>
        Complex,
    }

    /// <summary>
    ///     One Python scalar with CPython 3.12's arithmetic — the semantics NumPy 2.4.2's pure-Python
    ///     polynomial code runs on its literals and on a scalar <c>x</c>. Python ints are exact
    ///     (<see cref="BigInteger"/>), so <c>2*x</c> of a huge int never wraps (NumPy's own ints would).
    /// </summary>
    /// <remarks>
    ///     <para>Mixed-kind operations follow CPython 3.12 (NOT 3.14, which changed real+complex to be
    ///     componentwise): the narrower operand is converted first — int to float by correct rounding,
    ///     int/float to complex as <c>(value, +0.0)</c> — and the complex product is <c>_Py_c_prod</c>'s
    ///     naive formula. That conversion is observable: <c>2*(inf+1j)</c> is <c>(inf+nanj)</c> because
    ///     the int 2 becomes <c>(2+0j)</c> and <c>0*inf</c> is NaN.</para>
    ///     <para>Only +, -, * and true division exist, because the step tables use no other operator.</para>
    /// </remarks>
    internal readonly struct PyScalar
    {
        /// <summary>The Python type.</summary>
        public readonly PyKind Kind;
        /// <summary>The value when <see cref="Kind"/> is <see cref="PyKind.Int"/>.</summary>
        public readonly BigInteger I;
        /// <summary>The value when <see cref="Kind"/> is <see cref="PyKind.Float"/>.</summary>
        public readonly double F;
        /// <summary>The value when <see cref="Kind"/> is <see cref="PyKind.Complex"/>.</summary>
        public readonly Complex C;

        private PyScalar(PyKind kind, BigInteger i, double f, Complex c) { Kind = kind; I = i; F = f; C = c; }

        /// <summary>A Python int.</summary><param name="v">The value.</param><returns>The scalar.</returns>
        public static PyScalar Int(BigInteger v) => new PyScalar(PyKind.Int, v, 0, default);
        /// <summary>A Python float.</summary><param name="v">The value.</param><returns>The scalar.</returns>
        public static PyScalar Float(double v) => new PyScalar(PyKind.Float, default, v, default);
        /// <summary>A Python complex.</summary><param name="v">The value.</param><returns>The scalar.</returns>
        public static PyScalar Cplx(Complex v) => new PyScalar(PyKind.Complex, default, 0, v);

        /// <summary>
        ///     CPython's int-to-float conversion (<c>float(i)</c> / <c>PyLong_AsDouble</c>): correctly
        ///     rounded, raising for a magnitude past the float range instead of returning inf.
        /// </summary>
        /// <returns>The nearest double.</returns>
        /// <exception cref="OverflowException">The int does not fit a double — CPython's
        ///     <c>OverflowError: int too large to convert to float</c>.</exception>
        public double IntToDouble()
        {
            // |i| <= 2^53 converts exactly. Above that the BCL's explicit BigInteger->double conversion
            // TRUNCATES (measured: 2^64-1 -> 2^64-2048, where CPython's float(2**64-1) rounds to 2^64), so
            // the value goes through the decimal text instead — .NET Core 3.0+ parses correctly rounded
            // (round-half-even), which is CPython's rule. An out-of-range magnitude parses to ±inf, where
            // CPython raises.
            if (BigInteger.Abs(I) <= s_twoPow53)
                return (double)(long)I;
            double d = double.Parse(I.ToString(CultureInfo.InvariantCulture), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            if (double.IsInfinity(d))
                throw new OverflowException("int too large to convert to float");
            return d;
        }

        /// <summary>2^53, the largest magnitude every integer below which a double holds exactly.</summary>
        private static readonly BigInteger s_twoPow53 = BigInteger.One << 53;

        /// <summary>The value as a double (Int or Float only).</summary>
        /// <returns>The double.</returns>
        /// <exception cref="InvalidOperationException">A complex value (callers promote to complex first).</exception>
        public double AsDouble() => Kind switch
        {
            PyKind.Int => IntToDouble(),
            PyKind.Float => F,
            _ => throw new InvalidOperationException("a Python complex has no float value"),
        };

        /// <summary>CPython's <c>TO_COMPLEX</c>: an int/float becomes <c>(value, +0.0)</c>.</summary>
        /// <returns>The complex value.</returns>
        public Complex AsComplex() => Kind == PyKind.Complex ? C : new Complex(AsDouble(), 0.0);

        /// <summary>
        ///     <c>a op b</c> with Python semantics: int∘int stays an exact int (true division gives a
        ///     correctly rounded float), anything with a float is float arithmetic, anything with a complex
        ///     is complex arithmetic after <see cref="AsComplex"/>.
        /// </summary>
        /// <param name="op">Add, Subtract, Multiply or Divide.</param>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The Python result.</returns>
        /// <exception cref="DivideByZeroException">Division by a zero int/float (Python's ZeroDivisionError).</exception>
        /// <exception cref="NotSupportedException">Another operator, a complex division, or an int/int division
        ///     whose operands exceed 2^53 — none reachable from the step tables, whose only Python-level
        ///     divisions are <c>(nd-1)/nd</c> and <c>(2*nd-1)/nd</c>.</exception>
        /// <exception cref="OverflowException">An int too large for a float met a float or complex operand.</exception>
        public static PyScalar Apply(BinaryOp op, in PyScalar a, in PyScalar b)
        {
            if (a.Kind == PyKind.Complex || b.Kind == PyKind.Complex)
            {
                Complex x = a.AsComplex(), y = b.AsComplex();
                return Cplx(op switch
                {
                    BinaryOp.Add => new Complex(x.Real + y.Real, x.Imaginary + y.Imaginary),
                    BinaryOp.Subtract => new Complex(x.Real - y.Real, x.Imaginary - y.Imaginary),
                    // _Py_c_prod: the naive four-product formula, left-to-right (no FMA in CPython).
                    BinaryOp.Multiply => new Complex(x.Real * y.Real - x.Imaginary * y.Imaginary,
                                                     x.Real * y.Imaginary + x.Imaginary * y.Real),
                    // No step table divides a Python complex (lagval's divisions are array / nd), so
                    // _Py_c_quot and its IEEE edge recovery are deliberately not ported.
                    _ => throw new NotSupportedException($"Python complex {op} is not produced by the step tables"),
                });
            }

            if (a.Kind == PyKind.Float || b.Kind == PyKind.Float)
            {
                double x = a.AsDouble(), y = b.AsDouble();
                return Float(op switch
                {
                    BinaryOp.Add => x + y,
                    BinaryOp.Subtract => x - y,
                    BinaryOp.Multiply => x * y,
                    BinaryOp.Divide => y == 0 ? throw new DivideByZeroException("float division by zero") : x / y,
                    _ => throw new NotSupportedException(op.ToString()),
                });
            }

            switch (op)
            {
                case BinaryOp.Add: return Int(a.I + b.I);
                case BinaryOp.Subtract: return Int(a.I - b.I);
                case BinaryOp.Multiply: return Int(a.I * b.I);
                case BinaryOp.Divide:
                {
                    if (b.I.IsZero) throw new DivideByZeroException("division by zero");
                    // Python's int/int is the correctly rounded quotient. Both operands exact in a double
                    // make one IEEE division exactly that; larger ints never occur in the step tables.
                    var limit = new BigInteger(1L << 53);
                    if (BigInteger.Abs(a.I) > limit || BigInteger.Abs(b.I) > limit)
                        throw new NotSupportedException("int/int true division beyond 2^53 is not needed by the step tables");
                    return Float((double)a.I / (double)b.I);
                }
                default: throw new NotSupportedException(op.ToString());
            }
        }

        /// <summary>A Python-like rendering for diagnostics and kernel names.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => Kind switch
        {
            PyKind.Int => I.ToString(CultureInfo.InvariantCulture),
            PyKind.Float => F.ToString("R", CultureInfo.InvariantCulture),
            _ => C.ToString(CultureInfo.InvariantCulture),
        };
    }

    /// <summary>
    ///     One node of a step expression. The trees are written in NumPy's source order and are NEVER
    ///     re-associated or simplified: operand order decides complex-multiply bits and float16 NaN
    ///     priority, and evaluation order decides float rounding — a "harmless" refactor is a parity bug.
    /// </summary>
    internal abstract class PolyExpr { }

    /// <summary>A read of one <see cref="PolySym"/> (its dtype comes from the emission environment).</summary>
    internal sealed class PolyLeaf : PolyExpr
    {
        /// <summary>The symbol read.</summary>
        public readonly PolySym S;

        /// <summary>Creates a leaf reading <paramref name="s"/>.</summary>
        /// <param name="s">The symbol.</param>
        public PolyLeaf(PolySym s) { S = s; }
    }

    /// <summary>
    ///     A Python value (a literal such as <c>2</c>, a Python-computed one such as <c>(nd-1)/nd</c>, or a
    ///     folded x-only subexpression when <c>x</c> is a Python scalar). It has NO dtype of its own: NEP 50
    ///     types it by its partner operand, which is why it can only appear inside a <see cref="PolyBin"/>
    ///     whose other side is a real array value.
    /// </summary>
    internal sealed class PolyWeak : PolyExpr
    {
        /// <summary>The constant's identity and value function.</summary>
        public readonly PolyWeakConst W;

        /// <summary>Wraps <paramref name="w"/> as an expression node.</summary>
        /// <param name="w">The weak constant.</param>
        public PolyWeak(PolyWeakConst w) { W = w; }
    }

    /// <summary>A binary array op (<c>+ - * /</c>) exactly as NumPy spells it: <c>A op B</c>.</summary>
    internal sealed class PolyBin : PolyExpr
    {
        /// <summary>The house binary op the emitter lowers this node to.</summary>
        public readonly BinaryOp Op;
        /// <summary>Left and right operands, in NumPy's source order.</summary>
        public readonly PolyExpr A, B;

        /// <summary>Creates <c>a op b</c>.</summary>
        /// <param name="op">The operation (Add, Subtract, Multiply or Divide).</param>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        public PolyBin(BinaryOp op, PolyExpr a, PolyExpr b) { Op = op; A = a; B = b; }
    }

    /// <summary>
    ///     A weak (Python) value. <see cref="PerRow"/> values depend on the recurrence index (<c>nd</c>),
    ///     read by the kernel from row <c>nd</c> of their constant-pool region; <see cref="XDependent"/>
    ///     values depend on the caller's Python-scalar <c>x</c> and are materialized per call rather than
    ///     cached. Identity is REFERENCE identity: the constant pool keys its regions on the object, so a
    ///     program's constants must be created once and reused (they are, by the program caches).
    /// </summary>
    internal sealed class PolyWeakConst
    {
        private readonly Func<long, PyScalar, PyScalar> _value;

        /// <summary>A readable identity used in kernel names and diagnostics.</summary>
        public readonly string Name;
        /// <summary>The Python type of every value this constant takes.</summary>
        public readonly PyKind Kind;
        /// <summary>Whether the value depends on the recurrence row.</summary>
        public readonly bool PerRow;
        /// <summary>Whether the value depends on the call's Python-scalar x (such a region is never cached).</summary>
        public readonly bool XDependent;

        /// <summary>Declares a weak constant.</summary>
        /// <param name="name">Identity text.</param>
        /// <param name="kind">Python type of the value.</param>
        /// <param name="perRow">Row-dependent or fixed.</param>
        /// <param name="xDependent">Depends on the call's scalar x.</param>
        /// <param name="value">Value per (row, x); x is <c>default</c> when <paramref name="xDependent"/> is false.</param>
        public PolyWeakConst(string name, PyKind kind, bool perRow, bool xDependent, Func<long, PyScalar, PyScalar> value)
        {
            Name = name; Kind = kind; PerRow = perRow; XDependent = xDependent; _value = value;
        }

        /// <summary>The Python value at <paramref name="row"/> for the call's <paramref name="x"/>.</summary>
        /// <param name="row">Recurrence row (ignored for fixed constants).</param>
        /// <param name="x">The call's Python-scalar x (ignored unless <see cref="XDependent"/>).</param>
        /// <returns>The value.</returns>
        public PyScalar Value(long row, in PyScalar x) => _value(row, x);
    }

    /// <summary>
    ///     A basis's evaluation step table (NumPy 2.4.2 <c>{p}val</c>, source order), possibly rewritten for a
    ///     Python-scalar <c>x</c> (<see cref="PolySteps.RewriteForWeakX"/>).
    /// </summary>
    internal sealed class PolyEvalProgram
    {
        /// <summary>Horner (power basis) instead of the two-accumulator Clenshaw skeleton.</summary>
        public bool Horner;
        /// <summary>The x-only pre-op (<c>x2 = 2*x</c> / <c>x*2</c>), or null (none, or folded into weak constants).</summary>
        public PolyExpr Pre;
        /// <summary>Whether <see cref="Pre"/> runs for EVERY coefficient count (hermval) rather than only when
        ///     <c>len(c) &gt;= 3</c> (chebval). Observable only through the IL it costs; kept so the emitted
        ///     sequence mirrors the source.</summary>
        public bool PreAlways;
        /// <summary>Clenshaw: the new <c>c0</c> (reads Ck, C1) and new <c>c1</c> (reads Tmp, C1, X, X2).</summary>
        public PolyExpr StepC0, StepC1;
        /// <summary>Clenshaw: the returned expression (reads C0, C1, X, X2).</summary>
        public PolyExpr Final;
        /// <summary>Clenshaw's <c>len(c) == 1</c> return: <see cref="Final"/> with <c>c1 = 0</c> (a Python int).</summary>
        public PolyExpr FinalLen1;
        /// <summary>Horner: <c>c0 = c[-1] + x*0</c> and <c>c0 = c[-i] + c0*x</c>.</summary>
        public PolyExpr HornerInit, HornerStep;
        /// <summary>Whether x was folded away (the call's x is a Python scalar).</summary>
        public bool XIsWeak;
    }

    /// <summary>NumPy 2.4.2's step expressions for every basis, as data, plus the tree utilities.</summary>
    internal static class PolySteps
    {
        /// <summary>Leaf shorthands.</summary>
        public static readonly PolyExpr X = new PolyLeaf(PolySym.X), X2 = new PolyLeaf(PolySym.X2),
            C0 = new PolyLeaf(PolySym.C0), C1 = new PolyLeaf(PolySym.C1),
            Tmp = new PolyLeaf(PolySym.Tmp), Ck = new PolyLeaf(PolySym.Ck);

        /// <summary><c>a + b</c>.</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
        public static PolyExpr Add(PolyExpr a, PolyExpr b) => new PolyBin(BinaryOp.Add, a, b);
        /// <summary><c>a - b</c>.</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
        public static PolyExpr Sub(PolyExpr a, PolyExpr b) => new PolyBin(BinaryOp.Subtract, a, b);
        /// <summary><c>a * b</c>.</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
        public static PolyExpr Mul(PolyExpr a, PolyExpr b) => new PolyBin(BinaryOp.Multiply, a, b);
        /// <summary><c>a / b</c> (true division).</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
        public static PolyExpr Div(PolyExpr a, PolyExpr b) => new PolyBin(BinaryOp.Divide, a, b);
        /// <summary>A weak constant node.</summary><param name="w">The constant.</param><returns>The node.</returns>
        public static PolyExpr W(PolyWeakConst w) => new PolyWeak(w);

        /// <summary>Declares an x-independent constant.</summary>
        /// <param name="name">Identity text.</param><param name="kind">Python type.</param>
        /// <param name="perRow">Row-dependent.</param><param name="value">Value per row.</param>
        /// <returns>The constant.</returns>
        private static PolyWeakConst K(string name, PyKind kind, bool perRow, Func<long, PyScalar> value)
            => new PolyWeakConst(name, kind, perRow, xDependent: false, (r, _) => value(r));

        /// <summary>Fixed Python literals.</summary>
        public static readonly PolyWeakConst I0 = K("0", PyKind.Int, false, _ => PyScalar.Int(0)),
            I1 = K("1", PyKind.Int, false, _ => PyScalar.Int(1)),
            I2 = K("2", PyKind.Int, false, _ => PyScalar.Int(2));

        /// <summary>
        ///     Row-dependent Python values (row = NumPy's <c>nd</c> after its <c>nd = nd - 1</c>). The int ones
        ///     are exact in a long for every row a kernel can reach; the two quotients are Python's int/int
        ///     true division, correctly rounded — as is one IEEE double division of two exact small ints.
        /// </summary>
        public static readonly PolyWeakConst NdM1 = K("nd-1", PyKind.Int, true, r => PyScalar.Int(r - 1)),
            Nd = K("nd", PyKind.Int, true, r => PyScalar.Int(r)),
            TwoNdM1 = K("2nd-1", PyKind.Int, true, r => PyScalar.Int(2 * r - 1)),
            TwoNdM1b = K("2(nd-1)", PyKind.Int, true, r => PyScalar.Int(2 * (r - 1))),
            Q1 = K("(nd-1)/nd", PyKind.Float, true, r => PyScalar.Float((double)(r - 1) / r)),
            Q2 = K("(2nd-1)/nd", PyKind.Float, true, r => PyScalar.Float((double)(2 * r - 1) / r));

        /// <summary>
        ///     <c>{p}val</c> step tables. Each entry is the NumPy 2.4.2 source it transcribes; note
        ///     <c>legval</c>'s <c>c1*((nd-1)/nd)</c> (one Python float) against <c>lagval</c>'s
        ///     <c>(c1*(nd-1))/nd</c> (two array ops) — the same algebra, different bits.
        /// </summary>
        private static readonly PolyEvalProgram[] s_eval = BuildEval();

        /// <summary>The strong-x (array) step table of <paramref name="b"/>.</summary>
        /// <param name="b">The basis.</param><returns>The program.</returns>
        public static PolyEvalProgram Eval(PolyBasis b) => s_eval[(int)b];

        private static PolyEvalProgram[] BuildEval()
        {
            var p = new PolyEvalProgram[6];
            // polyval:  c0 = c[-1] + x*0 ;  c0 = c[-i] + c0*x
            p[(int)PolyBasis.Power] = new PolyEvalProgram
            {
                Horner = true,
                HornerInit = Add(Ck, Mul(X, W(I0))),
                HornerStep = Add(Ck, Mul(C0, X)),
            };
            // chebval:  x2 = 2*x (only when len(c) >= 3) ;  c0 = c[-i] - c1 ;  c1 = tmp + c1*x2 ;  return c0 + c1*x
            p[(int)PolyBasis.Chebyshev] = new PolyEvalProgram
            {
                Pre = Mul(W(I2), X),
                StepC0 = Sub(Ck, C1),
                StepC1 = Add(Tmp, Mul(C1, X2)),
                Final = Add(C0, Mul(C1, X)),
            };
            // legval:  c0 = c[-i] - c1*((nd - 1)/nd) ;  c1 = tmp + c1*x*((2*nd - 1)/nd) ;  return c0 + c1*x
            p[(int)PolyBasis.Legendre] = new PolyEvalProgram
            {
                StepC0 = Sub(Ck, Mul(C1, W(Q1))),
                StepC1 = Add(Tmp, Mul(Mul(C1, X), W(Q2))),
                Final = Add(C0, Mul(C1, X)),
            };
            // lagval:  c0 = c[-i] - (c1*(nd - 1))/nd ;  c1 = tmp + (c1*((2*nd - 1) - x))/nd ;  return c0 + c1*(1 - x)
            p[(int)PolyBasis.Laguerre] = new PolyEvalProgram
            {
                StepC0 = Sub(Ck, Div(Mul(C1, W(NdM1)), W(Nd))),
                StepC1 = Add(Tmp, Div(Mul(C1, Sub(W(TwoNdM1), X)), W(Nd))),
                Final = Add(C0, Mul(C1, Sub(W(I1), X))),
            };
            // hermval:  x2 = x*2 (ALWAYS, before the length test) ;  c0 = c[-i] - c1*(2*(nd - 1)) ;
            //           c1 = tmp + c1*x2 ;  return c0 + c1*x2
            p[(int)PolyBasis.Hermite] = new PolyEvalProgram
            {
                Pre = Mul(X, W(I2)),
                PreAlways = true,
                StepC0 = Sub(Ck, Mul(C1, W(TwoNdM1b))),
                StepC1 = Add(Tmp, Mul(C1, X2)),
                Final = Add(C0, Mul(C1, X2)),
            };
            // hermeval:  c0 = c[-i] - c1*(nd - 1) ;  c1 = tmp + c1*x ;  return c0 + c1*x
            p[(int)PolyBasis.HermiteE] = new PolyEvalProgram
            {
                StepC0 = Sub(Ck, Mul(C1, W(NdM1))),
                StepC1 = Add(Tmp, Mul(C1, X)),
                Final = Add(C0, Mul(C1, X)),
            };
            foreach (var prog in p)
                if (!prog.Horner)
                    prog.FinalLen1 = Subst(prog.Final, PolySym.C1, W(I0));   // len(c) == 1: c1 = 0 (a Python int)
            return p;
        }

        private static readonly ConcurrentDictionary<(PolyBasis, PyKind), PolyEvalProgram> s_weakX = new();

        /// <summary>
        ///     The step table of <paramref name="b"/> for a Python-scalar x of type <paramref name="kind"/>:
        ///     x becomes an x-dependent weak constant and every all-weak subtree (<c>x*0</c>, <c>2*x</c>,
        ///     <c>(2*nd - 1) - x</c>, ...) is folded into ONE weak constant computed with Python arithmetic —
        ///     exactly what CPython does before NumPy ever sees the value. Cached per (basis, kind) so the
        ///     folded constants keep a stable identity for the kernel cache and the constant pool.
        /// </summary>
        /// <param name="b">The basis.</param><param name="kind">Python type of x.</param>
        /// <returns>The rewritten program (no <see cref="PolySym.X"/>/<see cref="PolySym.X2"/> leaves remain).</returns>
        public static PolyEvalProgram RewriteForWeakX(PolyBasis b, PyKind kind)
            => s_weakX.GetOrAdd((b, kind), key =>
            {
                var src = Eval(key.Item1);
                var xw = new PolyWeak(new PolyWeakConst("x", key.Item2, perRow: false, xDependent: true, (_, x) => x));
                // x2 is itself Python arithmetic on x (2*x / x*2), so fold it first and splice it in.
                PolyExpr x2 = src.Pre is null ? null : Fold(Subst(src.Pre, PolySym.X, xw));
                PolyExpr R(PolyExpr e)
                {
                    if (e is null) return null;
                    var t = Subst(e, PolySym.X, xw);
                    if (x2 is not null) t = Subst(t, PolySym.X2, x2);
                    return Fold(t);
                }
                return new PolyEvalProgram
                {
                    Horner = src.Horner,
                    Pre = null,
                    PreAlways = false,
                    StepC0 = R(src.StepC0),
                    StepC1 = R(src.StepC1),
                    Final = R(src.Final),
                    FinalLen1 = R(src.FinalLen1),
                    HornerInit = R(src.HornerInit),
                    HornerStep = R(src.HornerStep),
                    XIsWeak = true,
                };
            });

        /// <summary>Returns <paramref name="e"/> with every read of <paramref name="s"/> replaced by <paramref name="with"/>.</summary>
        /// <param name="e">The tree.</param><param name="s">The symbol to replace.</param><param name="with">The replacement.</param>
        /// <returns>A new tree (shared subtrees are fine: trees are immutable).</returns>
        public static PolyExpr Subst(PolyExpr e, PolySym s, PolyExpr with) => e switch
        {
            PolyLeaf l when l.S == s => with,
            PolyBin b => new PolyBin(b.Op, Subst(b.A, s, with), Subst(b.B, s, with)),
            _ => e,
        };

        /// <summary>
        ///     Folds every binary node whose two operands are weak into one weak constant computed with
        ///     <see cref="PyScalar.Apply"/> — the subexpressions CPython evaluates before handing the result
        ///     to NumPy. Row- and x-dependence propagate, so a folded <c>(2*nd - 1) - x</c> is a per-row,
        ///     per-call region.
        /// </summary>
        /// <param name="e">The tree.</param><returns>The folded tree.</returns>
        public static PolyExpr Fold(PolyExpr e)
        {
            if (e is not PolyBin b) return e;
            var a = Fold(b.A);
            var c = Fold(b.B);
            if (a is PolyWeak wa && c is PolyWeak wb)
            {
                var ka = wa.W; var kb = wb.W;
                var op = b.Op;
                var kind = ResultKind(op, ka.Kind, kb.Kind);
                return new PolyWeak(new PolyWeakConst(
                    $"({ka.Name}{OpChar(op)}{kb.Name})", kind, ka.PerRow || kb.PerRow, ka.XDependent || kb.XDependent,
                    (r, x) => PyScalar.Apply(op, ka.Value(r, x), kb.Value(r, x))));
            }
            return ReferenceEquals(a, b.A) && ReferenceEquals(c, b.B) ? b : new PolyBin(b.Op, a, c);
        }

        /// <summary>The Python type of <c>a op b</c>: complex beats float beats int; int/int true division is a float.</summary>
        /// <param name="op">The operator.</param><param name="a">Left type.</param><param name="b">Right type.</param>
        /// <returns>The result type.</returns>
        public static PyKind ResultKind(BinaryOp op, PyKind a, PyKind b)
        {
            if (a == PyKind.Complex || b == PyKind.Complex) return PyKind.Complex;
            if (a == PyKind.Float || b == PyKind.Float) return PyKind.Float;
            return op == BinaryOp.Divide ? PyKind.Float : PyKind.Int;
        }

        /// <summary>The operator's source character, for folded-constant names.</summary>
        /// <param name="op">The operator.</param><returns>The character.</returns>
        private static char OpChar(BinaryOp op) => op switch
        {
            BinaryOp.Add => '+', BinaryOp.Subtract => '-', BinaryOp.Multiply => '*', _ => '/',
        };

        /// <summary>Whether <paramref name="e"/> reads a per-row weak constant — when no step does, the kernel never
        ///     maintains the row index (one store less per step).</summary>
        /// <param name="e">The tree.</param><returns>True when a per-row constant is read.</returns>
        public static bool UsesRow(PolyExpr e) => e switch
        {
            PolyWeak w => w.W.PerRow,
            PolyBin b => UsesRow(b.A) || UsesRow(b.B),
            _ => false,
        };

        /// <summary>Whether <paramref name="e"/> reads <paramref name="s"/>.</summary>
        /// <param name="e">The tree.</param><param name="s">The symbol.</param><returns>True when read.</returns>
        public static bool Uses(PolyExpr e, PolySym s) => e switch
        {
            PolyLeaf l => l.S == s,
            PolyBin b => Uses(b.A, s) || Uses(b.B, s),
            _ => false,
        };
    }
}
