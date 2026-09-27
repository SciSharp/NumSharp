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
    ///     <para>The operators are +, -, *, true division and negation, plus the two comparisons with the int 0
    ///     that <c>numpy.polynomial</c> performs on a Python value (<c>scl != 0</c> in <c>{p}line</c>,
    ///     <c>tol &lt; 0</c> in <c>trimcoef</c>). Division is CPython's exactly: int/int is
    ///     <c>long_true_divide</c> (correctly rounded at ANY magnitude, <c>OverflowError</c> past the float
    ///     range), and complex division is 3.12's <c>_Py_c_quot</c> (Smith's algorithm WITHOUT the C11 Annex G
    ///     infinity recovery 3.13 added — probed: <c>(1+1j)/(inf+infj)</c> is <c>(nan+nanj)</c> on 3.12.12).</para>
    ///     <para>A Python <c>bool</c> is an <see cref="PyKind.Int"/> tagged <see cref="IsBool"/>: CPython's bool
    ///     IS an int subclass, so every arithmetic result is a plain int (<c>True + True == 2</c>,
    ///     <c>-True == -1</c>) and only a LEAF carries the tag — which NEP 50 promotion (a weak bool keeps a
    ///     bool partner bool) and array creation (<c>np.array([True, True])</c> is bool) read.</para>
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
        /// <summary>
        ///     Whether this int is a Python <c>bool</c> (<see cref="I"/> is then 0 or 1). Never set on an arithmetic
        ///     result — CPython's bool arithmetic yields ints — so it is a property of a caller-supplied leaf only.
        /// </summary>
        public readonly bool IsBool;

        private PyScalar(PyKind kind, BigInteger i, double f, Complex c, bool isBool = false)
        {
            Kind = kind; I = i; F = f; C = c; IsBool = isBool;
        }

        /// <summary>A Python int.</summary><param name="v">The value.</param><returns>The scalar.</returns>
        public static PyScalar Int(BigInteger v) => new PyScalar(PyKind.Int, v, 0, default);
        /// <summary>A Python float.</summary><param name="v">The value.</param><returns>The scalar.</returns>
        public static PyScalar Float(double v) => new PyScalar(PyKind.Float, default, v, default);
        /// <summary>A Python complex.</summary><param name="v">The value.</param><returns>The scalar.</returns>
        public static PyScalar Cplx(Complex v) => new PyScalar(PyKind.Complex, default, 0, v);
        /// <summary>
        ///     A Python bool: an int 0/1 carrying the <see cref="IsBool"/> tag. Arithmetic on it (which always
        ///     yields a plain int) drops the tag, exactly as CPython's <c>True + 1</c> is the int 2.
        /// </summary>
        /// <param name="v">The value.</param><returns>The scalar.</returns>
        public static PyScalar Bool(bool v) => new PyScalar(PyKind.Int, v ? BigInteger.One : BigInteger.Zero, 0, default, isBool: true);

        /// <summary>
        ///     The Python type name (<c>type(x).__name__</c>) — the word CPython's error messages quote, e.g.
        ///     <c>'float' object is not subscriptable</c>.
        /// </summary>
        public string TypeName => IsBool ? "bool" : Kind switch
        {
            PyKind.Int => "int",
            PyKind.Float => "float",
            _ => "complex",
        };

        /// <summary>
        ///     CPython's unary minus: an int (or bool) negates exactly to an int, a float flips its sign (NaN
        ///     included), a complex negates both parts.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <returns><c>-a</c>.</returns>
        public static PyScalar Negate(in PyScalar a) => a.Kind switch
        {
            PyKind.Int => Int(-a.I),
            PyKind.Float => Float(-a.F),
            _ => Cplx(new Complex(-a.C.Real, -a.C.Imaginary)),
        };

        /// <summary>
        ///     CPython's <c>x != 0</c>: an int compares exactly, a float is nonzero unless it is ±0 (NaN IS
        ///     nonzero), a complex is nonzero unless both parts are ±0.
        /// </summary>
        public bool IsNonZero => Kind switch
        {
            PyKind.Int => !I.IsZero,
            PyKind.Float => F != 0.0,
            _ => C.Real != 0.0 || C.Imaginary != 0.0,
        };

        /// <summary>
        ///     CPython's <c>x &lt; 0</c> for an int or float (NaN compares false). A complex has no ordering in
        ///     Python, which raises <c>TypeError: '&lt;' not supported between instances of 'complex' and 'int'</c>.
        /// </summary>
        /// <returns>Whether the value is negative.</returns>
        /// <exception cref="TypeError">The value is a complex.</exception>
        public bool IsNegative() => Kind switch
        {
            PyKind.Int => I.Sign < 0,
            PyKind.Float => F < 0.0,
            _ => throw new TypeError("'<' not supported between instances of 'complex' and 'int'"),
        };

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
        /// <exception cref="DivideByZeroException">Division by zero — Python's ZeroDivisionError with CPython's
        ///     text for the operand kinds (<c>division by zero</c>, <c>float division by zero</c>,
        ///     <c>complex division by zero</c>).</exception>
        /// <exception cref="NotSupportedException">An operator other than the four.</exception>
        /// <exception cref="OverflowException">An int too large for a float met a float or complex operand
        ///     (<c>int too large to convert to float</c>), or an int/int quotient beyond the float range
        ///     (<c>integer division result too large for a float</c>).</exception>
        public static PyScalar Apply(BinaryOp op, in PyScalar a, in PyScalar b)
        {
            if (a.Kind == PyKind.Complex || b.Kind == PyKind.Complex)
            {
                Complex x = a.AsComplex(), y = b.AsComplex();
                return Cplx(op switch
                {
                    BinaryOp.Add => ComplexSum(x, y),
                    BinaryOp.Subtract => ComplexDiff(x, y),
                    BinaryOp.Multiply => ComplexProd(x, y),
                    BinaryOp.Divide => ComplexQuotient(x, y),
                    _ => throw new NotSupportedException(op.ToString()),
                });
            }

            if (a.Kind == PyKind.Float || b.Kind == PyKind.Float)
            {
                double x = a.AsDouble(), y = b.AsDouble();
                return Float(op switch
                {
                    BinaryOp.Add => FloatAdd(x, y),
                    BinaryOp.Subtract => FloatSub(x, y),
                    BinaryOp.Multiply => FloatMul(x, y),
                    BinaryOp.Divide => FloatDiv(x, y),
                    _ => throw new NotSupportedException(op.ToString()),
                });
            }

            switch (op)
            {
                case BinaryOp.Add: return Int(a.I + b.I);
                case BinaryOp.Subtract: return Int(a.I - b.I);
                case BinaryOp.Multiply: return Int(a.I * b.I);
                case BinaryOp.Divide: return Float(IntTrueDivide(a.I, b.I));
                default: throw new NotSupportedException(op.ToString());
            }
        }

        /// <summary>
        ///     CPython 3.12's <c>_Py_c_quot</c> (Objects/complexobject.c): Smith's algorithm, dividing top and
        ///     bottom by whichever of <c>b.real</c>/<c>b.imag</c> has the larger magnitude, with NO recovery of
        ///     infinities from a NaN result (that block arrived in 3.13). A zero divisor is Python's
        ///     <c>ZeroDivisionError: complex division by zero</c>; a NaN component of the divisor that defeats both
        ///     magnitude tests yields <c>(nan+nanj)</c>.
        /// </summary>
        /// <param name="a">Dividend.</param>
        /// <param name="b">Divisor.</param>
        /// <returns>The quotient, bit for bit as CPython 3.12 computes it (MSVC x64 build: SSE2, no FMA
        ///     contraction).</returns>
        /// <exception cref="DivideByZeroException">Both parts of <paramref name="b"/> are ±0.</exception>
        /// <remarks>
        ///     NOT NumPy's complex division (<c>ComplexDivideNumPy</c>): NumPy multiplies by a reciprocal
        ///     <c>scl = 1/denom</c> and turns division by zero into inf/nan, where CPython divides by
        ///     <c>denom</c> directly and raises. The two differ in the last bit and on every edge — the reason
        ///     Python-level arithmetic (a C# primitive) and NumPy-level arithmetic (an array element) are kept
        ///     apart throughout <c>numpy.polynomial</c>.
        /// </remarks>
        public static Complex ComplexQuotient(Complex a, Complex b)
        {
            double abs_breal = b.Real < 0 ? -b.Real : b.Real;
            double abs_bimag = b.Imaginary < 0 ? -b.Imaginary : b.Imaginary;
            if (abs_breal >= abs_bimag)
            {
                // Divide top and bottom by b.real. (A NaN abs_breal fails this test and the next one.)
                if (abs_breal == 0.0)
                    throw new DivideByZeroException("complex division by zero");
                double ratio = LeftDiv(b.Imaginary, b.Real);
                double denom = LeftAdd(b.Real, LeftMul(b.Imaginary, ratio));
                return new Complex(LeftDiv(LeftAdd(a.Real, LeftMul(a.Imaginary, ratio)), denom),
                                   LeftDiv(LeftSub(a.Imaginary, LeftMul(a.Real, ratio)), denom));
            }
            if (abs_bimag >= abs_breal)
            {
                // Divide top and bottom by b.imag.
                double ratio = LeftDiv(b.Real, b.Imaginary);
                double denom = LeftAdd(LeftMul(b.Real, ratio), b.Imaginary);
                return new Complex(LeftDiv(LeftAdd(LeftMul(a.Real, ratio), a.Imaginary), denom),
                                   LeftDiv(LeftSub(LeftMul(a.Imaginary, ratio), a.Real), denom));
            }
            // At least one of b.real / b.imag is a NaN: CPython's Py_NAN, which this build defines as the POSITIVE
            // quiet NaN (probed: (1+2j)/complex(nan, 1) is 0x7ff8000000000000 in both parts) — not .NET's
            // double.NaN, whose sign bit is set.
            return new Complex(s_pyNaN, s_pyNaN);
        }

        /// <summary>CPython's <c>Py_NAN</c> on win-amd64: the positive quiet NaN <c>0x7ff8000000000000</c>.</summary>
        private static readonly double s_pyNaN = BitConverter.Int64BitsToDouble(0x7ff8000000000000L);

        /// <summary>CPython's <c>_Py_c_sum</c>: componentwise addition, each part keeping the LEFT operand's NaN
        ///     (a Python int/float operand joins as <c>(value, +0.0)</c> — the caller's <see cref="AsComplex"/>).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The sum.</returns>
        public static Complex ComplexSum(Complex a, Complex b) => new Complex(LeftAdd(a.Real, b.Real), LeftAdd(a.Imaginary, b.Imaginary));

        /// <summary>CPython's <c>_Py_c_diff</c>: componentwise subtraction (left operand's NaN first).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The difference.</returns>
        public static Complex ComplexDiff(Complex a, Complex b) => new Complex(LeftSub(a.Real, b.Real), LeftSub(a.Imaginary, b.Imaginary));

        /// <summary>
        ///     CPython's <c>_Py_c_prod</c>: the naive four-product formula evaluated left to right —
        ///     <c>(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)</c> — with no fused multiply-add (the MSVC build
        ///     targets SSE2) and every operation keeping its left operand's NaN. NOT NumPy's array multiply (a
        ///     fused <c>simd_cmul</c>) — the two differ in the last bit.
        /// </summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The product.</returns>
        public static Complex ComplexProd(Complex a, Complex b)
            => new Complex(LeftSub(LeftMul(a.Real, b.Real), LeftMul(a.Imaginary, b.Imaginary)),
                           LeftAdd(LeftMul(a.Real, b.Imaginary), LeftMul(a.Imaginary, b.Real)));

        // ---------------------------------------------------------------------------------------------
        //  CPython's IEEE arithmetic, NaN operand priority included
        // ---------------------------------------------------------------------------------------------
        //
        // x86-64 SSE returns the FIRST source operand's NaN (quieted) when both operands are NaN, and any NaN
        // input propagates quieted; an invalid operation on non-NaN operands (inf-inf, 0*inf) yields the default
        // NaN (sign bit set). Which operand is the "first source" of a COMMUTATIVE operator is the compiler's
        // choice: RyuJIT swaps `a + b` / `a * b` whenever its register allocation prefers the other order, so a C#
        // operator does not pin the NaN a result carries — two call sites of the same expression can disagree.
        // CPython 3.12 as MSVC compiled it (probed on the corpus's interpreter, per operator and operand order):
        //   float_add ............ the RIGHT operand's NaN (generic and specialized BINARY_OP_ADD_FLOAT alike)
        //   float_sub, float_div . the LEFT operand's NaN
        //   float_mul ............ the LEFT operand's NaN once the call site is specialized
        //                          (BINARY_OP_MULTIPLY_FLOAT, the steady state of any site executed twice); a
        //                          site's FIRST, unspecialized execution (generic float_mul) returns the RIGHT
        //                          one's — interpreter state no library can reproduce, so the steady state is modelled
        //   _Py_c_sum/_diff/_prod/_quot: every operation the LEFT operand's NaN (source order)
        // These helpers pin the order explicitly; the cost is two predictable NaN tests per operation.

        /// <summary>A NaN as SSE propagates it: its quiet bit set (a signalling NaN keeps its payload and sign).</summary>
        /// <param name="v">A NaN.</param>
        /// <returns>The quieted NaN.</returns>
        private static double Quiet(double v) => BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(v) | 0x0008000000000000L);

        /// <summary>IEEE <c>a + b</c>, the LEFT operand's NaN winning when both are NaN.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The sum.</returns>
        public static double LeftAdd(double a, double b) => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : a + b;

        /// <summary>IEEE <c>a - b</c>, the LEFT operand's NaN winning (subtraction is never swapped, but a NaN
        ///     input is still re-quieted explicitly for symmetry with the other helpers).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The difference.</returns>
        public static double LeftSub(double a, double b) => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : a - b;

        /// <summary>IEEE <c>a * b</c>, the LEFT operand's NaN winning when both are NaN.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The product.</returns>
        public static double LeftMul(double a, double b) => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : a * b;

        /// <summary>IEEE <c>a / b</c>, the LEFT operand's NaN winning.</summary>
        /// <param name="a">Dividend.</param><param name="b">Divisor.</param><returns>The quotient.</returns>
        public static double LeftDiv(double a, double b) => double.IsNaN(a) ? Quiet(a) : double.IsNaN(b) ? Quiet(b) : a / b;

        /// <summary>CPython's <c>float + float</c>: the RIGHT operand's NaN wins when both are NaN.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The sum.</returns>
        public static double FloatAdd(double a, double b) => double.IsNaN(b) ? Quiet(b) : double.IsNaN(a) ? Quiet(a) : a + b;

        /// <summary>CPython's <c>float - float</c> (the left operand's NaN wins).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The difference.</returns>
        public static double FloatSub(double a, double b) => LeftSub(a, b);

        /// <summary>CPython's <c>float * float</c> at a specialized call site (the left operand's NaN wins).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>The product.</returns>
        public static double FloatMul(double a, double b) => LeftMul(a, b);

        /// <summary>CPython's <c>float / float</c>: a ±0 divisor is ZeroDivisionError, else the left operand's NaN wins.</summary>
        /// <param name="a">Dividend.</param><param name="b">Divisor.</param><returns>The quotient.</returns>
        /// <exception cref="DivideByZeroException"><paramref name="b"/> is ±0 (<c>float division by zero</c>).</exception>
        public static double FloatDiv(double a, double b)
        {
            if (b == 0)
                throw new DivideByZeroException("float division by zero");
            return LeftDiv(a, b);
        }

        /// <summary>2^53: an integer below this magnitude is exactly a double (CPython's "small" operand test).</summary>
        private static readonly BigInteger s_twoPow53Exclusive = BigInteger.One << 53;

        /// <summary>
        ///     CPython's <c>long_true_divide</c> (Objects/longobject.c): the correctly rounded (round-half-even)
        ///     quotient of two Python ints at ANY magnitude, subnormals included. Both operands below 2^53 take
        ///     CPython's fast path — one IEEE division of the two exact doubles; otherwise the quotient is formed
        ///     exactly with 55–56 significant bits, the sticky bit from the discarded remainder, and rounded once.
        /// </summary>
        /// <param name="a">Dividend.</param>
        /// <param name="b">Divisor.</param>
        /// <returns>The quotient; a zero result carries the XOR of the operand signs (<c>0 / -5</c> is <c>-0.0</c>).</returns>
        /// <exception cref="DivideByZeroException"><paramref name="b"/> is 0 (<c>division by zero</c>).</exception>
        /// <exception cref="OverflowException">The quotient exceeds the float range (<c>integer division result too
        ///     large for a float</c>).</exception>
        public static double IntTrueDivide(BigInteger a, BigInteger b)
        {
            if (b.IsZero)
                throw new DivideByZeroException("division by zero");
            // CPython: negate = (a < 0) != (b < 0), with a zero dividend never negative.
            bool negate = (a.Sign < 0) != (b.Sign < 0);
            if (a.IsZero)
                return negate ? -0.0 : 0.0;
            BigInteger aa = BigInteger.Abs(a), bb = BigInteger.Abs(b);
            if (aa < s_twoPow53Exclusive && bb < s_twoPow53Exclusive)
            {
                // Both exactly representable: one IEEE division IS the correctly rounded quotient.
                double q = (double)(long)aa / (double)(long)bb;
                return negate ? -q : q;
            }

            const int DblMantDig = 53, DblMaxExp = 1024, DblMinExp = -1021;
            long diff = (long)aa.GetBitLength() - (long)bb.GetBitLength();
            if (diff > DblMaxExp)
                throw new OverflowException("integer division result too large for a float");
            if (diff < DblMinExp - DblMantDig - 1)
                return negate ? -0.0 : 0.0;

            // x = |a| * 2^-shift, with shift chosen so x // |b| has DBL_MANT_DIG + 1 or + 2 bits (fewer for a
            // subnormal result); every bit shifted out, and a nonzero remainder, sets the sticky bit.
            long shift = Math.Max(diff, DblMinExp) - DblMantDig - 2;
            bool inexact = false;
            BigInteger x;
            if (shift <= 0)
                x = aa << (int)(-shift);
            else
            {
                x = aa >> (int)shift;
                if (!(aa & ((BigInteger.One << (int)shift) - 1)).IsZero)
                    inexact = true;
            }
            x = BigInteger.DivRem(x, bb, out BigInteger rem);
            if (!rem.IsZero)
                inexact = true;
            long xBits = (long)x.GetBitLength();

            // Round half-to-even at the bit that makes the result DBL_MANT_DIG bits (or the subnormal grid),
            // exactly CPython's "modify the low digit" step: mask is the half-ulp bit, 3*mask-1 the ulp bit
            // plus every bit below the half bit (with the sticky bit OR'd into bit 0).
            long extraBits = Math.Max(xBits, DblMinExp - shift) - DblMantDig;
            BigInteger mask = BigInteger.One << (int)(extraBits - 1);
            if (inexact)
                x |= BigInteger.One;
            if (!(x & mask).IsZero && !(x & (3 * mask - 1)).IsZero)
                x += mask;
            x &= ~(2 * mask - 1);

            // x now has at most DBL_MANT_DIG significant bits (plus trailing zeros): the conversion is exact.
            double dx = (double)x;
            if (shift + xBits >= DblMaxExp && (shift + xBits > DblMaxExp || dx == Math.ScaleB(1.0, (int)xBits)))
                throw new OverflowException("integer division result too large for a float");
            double result = Math.ScaleB(dx, (int)shift);
            return negate ? -result : result;
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
