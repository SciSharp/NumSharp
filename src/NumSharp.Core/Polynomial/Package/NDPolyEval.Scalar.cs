using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyEval.Scalar.cs — {p}val at a NumPy SCALAR x: scalarmath on every coefficient
// =============================================================================
//
// The series classes evaluate at a NumPy scalar far more often than at an array: `p(0.5)` maps 0.5 through
// `pu.mapdomain`, whose `off + scl*x` turns the Python float into a NumPy scalar of the domain's dtype, and
// integ's lbnd (`off + scl*lbnd`) is one too. NumPy's {p}val then sees an x that is neither a list nor an
// ndarray, so it never reshapes c and every statement of the recurrence is NumPy SCALAR arithmetic
// (scalarmath): the naive complex product, NEP 50 against the Python ints of the recurrence, IEEE inf/nan
// on a zero divisor. The evaluation kernel's 0-d mode answers for a 0-d ARRAY, whose operations NumPy runs
// as ufuncs (the fused complex product) — the same bits for real dtypes, other bits for complex ones.
//
// So this file interprets the U3 step trees (PolySteps.Eval, NumPy's source expressions, never
// re-associated) with PolyNumber arithmetic, statement for statement — the strong-x sibling of
// NDPolyCalc.PyVal1D, which interprets the trees rewritten for a Python-scalar x.
//
// =============================================================================

namespace NumSharp
{
    internal static unsafe partial class NDPolyEval
    {
        /// <summary>
        ///     <c>{p}val(x, c)</c> of a 1-D series at a NumPy SCALAR <paramref name="x"/>, as NumPy computes it: the basis's
        ///     strong step tree (<see cref="PolySteps.Eval"/>) interpreted with <see cref="PolyNumber"/> arithmetic, i.e. NumPy
        ///     scalarmath for every statement, Python ints of the recurrence weak against the coefficients.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">The point: a NumPy scalar (<see cref="PolyNumberKind.Scalar"/>, a complex64 one included).</param>
        /// <param name="c">Element 0 of the series.</param>
        /// <param name="n">len(c).</param>
        /// <param name="stride">The byte step between coefficients (the element size for a contiguous series).</param>
        /// <param name="t">The coefficient dtype — already NumPy's evaluation dtype (an integer or bool series converts to
        ///     float64 before this is called, as <c>{p}val</c>'s <c>c.astype(np.double)</c> does).</param>
        /// <returns>NumPy's value: a NumPy scalar.</returns>
        /// <exception cref="IndexError">An empty series: NumPy's first coefficient read (<c>c[-1]</c> for polyval,
        ///     <c>c[-2]</c> for the Clenshaw bases) fails.</exception>
        /// <remarks>
        ///     Statement order is NumPy's, including the x-only pre-step: chebval doubles x only when <c>len(c) &gt;= 3</c>,
        ///     hermval always (before its length test), so an overflow or NaN there happens exactly when NumPy's does. The
        ///     row counter <c>nd</c> starts at <c>len(c)</c> and drops before each step's statements, as in NumPy.
        /// </remarks>
        internal static PolyNumber ValAtScalar(PolyBasis basis, in PolyNumber x, byte* c, long n, long stride, NPTypeCode t)
        {
            var prog = PolySteps.Eval(basis);
            if (n == 0)
                throw new IndexError($"index {(prog.Horner ? -1 : -2)} is out of bounds for axis 0 with size 0");
            // NumPy's common case: a float64 series at a float64 scalar (p(0.5), integ's lbnd), every statement float64
            // scalarmath — the double lane, the same bits at a fraction of the cost.
            if (t == NPTypeCode.Double && x.Kind == PolyNumberKind.Scalar && x.Dtype == NPTypeCode.Double && F64LaneApplies(basis))
            {
                PolyRaw16 raw;
                x.WriteAs(NPTypeCode.Double, &raw);
                double r = ValAtScalarF64(prog, *(double*)&raw, c, n, stride);
                return PolyNumber.FromScalar(NPTypeCode.Double, (byte*)&r);
            }
            var xv = x;
            PolyNumber C(long i) => PolyNumber.FromScalar(t, c + i * stride);

            if (prog.Horner)
            {
                // polyval: c0 = c[-1] + x*0; for i in range(2, len(c) + 1): c0 = c[-i] + c0*x
                var h = Run(prog.HornerInit, xv, default, C(n - 1), default, default, default, 0);
                for (long i = 2; i <= n; i++)
                    h = Run(prog.HornerStep, xv, default, C(n - i), h, default, default, 0);
                return h;
            }

            // x2 = 2*x (chebval: inside its len(c) >= 3 branch; hermval: always, first).
            PolyNumber x2 = prog.Pre is not null && (prog.PreAlways || n >= 3)
                ? Run(prog.Pre, xv, default, default, default, default, default, 0)
                : default;
            if (n == 1)
                return Run(prog.FinalLen1, xv, x2, default, C(0), default, default, 0);
            if (n == 2)
                return Run(prog.Final, xv, x2, default, C(0), C(1), default, 0);
            // Clenshaw: c0 = c[-2]; c1 = c[-1]; for i: tmp = c0; nd -= 1; c0 = <StepC0>; c1 = <StepC1 of the OLD c1>
            PolyNumber a0 = C(n - 2), a1 = C(n - 1);
            long nd = n;
            for (long i = 3; i <= n; i++)
            {
                var tmp = a0;
                nd--;
                a0 = Run(prog.StepC0, xv, x2, C(n - i), default, a1, default, nd);
                a1 = Run(prog.StepC1, xv, x2, default, default, a1, tmp, nd);
            }
            return Run(prog.Final, xv, x2, default, a0, a1, default, 0);
        }

        /// <summary>
        ///     <see cref="ValAtScalar(PolyBasis, in PolyNumber, byte*, long, long, NPTypeCode)"/> for a series array (any
        ///     1-D layout): NumPy's <c>c.astype(np.double)</c> of an integer or bool series first, then the interpretation
        ///     over the elements in place.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">The point (a NumPy scalar).</param>
        /// <param name="c">The 1-D series.</param>
        /// <returns>NumPy's value (a NumPy scalar).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="c"/> is null.</exception>
        /// <exception cref="IndexError">An empty series.</exception>
        internal static PolyNumber ValAtScalar(PolyBasis basis, in PolyNumber x, NDArray c)
        {
            if (c is null) throw new ArgumentNullException(nameof(c));
            if (PolyTyping.IsIntLike(c.typecode))
            {
                // {p}val's `c.astype(np.double)` (polyval's `c + 0.0`: the same values for every integer and bool).
                using var cd = c.astype(NPTypeCode.Double);
                return ValAtScalar(basis, x, cd);
            }
            int size = c.dtypesize;
            byte* p = (byte*)c.Storage.Address + c.Shape.offset * size;
            long n = c.size;
            long stride = n > 1 ? c.Shape.strides[0] * size : size;
            return ValAtScalar(basis, x, p, n, stride, c.typecode);
        }

        /// <summary>
        ///     Evaluates one step expression with NumPy scalar arithmetic: the leaves read the current recurrence values, a
        ///     weak constant is the Python value of its row (<c>nd - 1</c>, <c>(nd - 1)/nd</c>, …), and every node is
        ///     <see cref="PolyNumber.Binary"/> — Python ∘ NumPy scalar under NEP 50, NumPy ∘ NumPy scalar in scalarmath.
        /// </summary>
        /// <param name="e">The expression.</param>
        /// <param name="x">x.</param><param name="x2">x2 (when computed).</param><param name="ck">The coefficient read.</param>
        /// <param name="c0">c0.</param><param name="c1">c1.</param><param name="tmp">tmp.</param>
        /// <param name="row">The row counter <c>nd</c> (read by per-row constants only).</param>
        /// <returns>The value.</returns>
        /// <exception cref="InvalidOperationException">A node kind the step tables never contain (a programming error).</exception>
        private static PolyNumber Run(PolyExpr e, in PolyNumber x, in PolyNumber x2, in PolyNumber ck, in PolyNumber c0,
            in PolyNumber c1, in PolyNumber tmp, long row) => e switch
        {
            PolyLeaf l => l.S switch
            {
                PolySym.X => x,
                PolySym.X2 => x2,
                PolySym.Ck => ck,
                PolySym.C0 => c0,
                PolySym.C1 => c1,
                PolySym.Tmp => tmp,
                _ => throw new InvalidOperationException($"symbol {l.S} has no value in a scalar evaluation"),
            },
            PolyWeak w => PolyNumber.FromPython(w.W.Value(row, default)),
            PolyBin b => PolyNumber.Binary(b.Op, Run(b.A, x, x2, ck, c0, c1, tmp, row), Run(b.B, x, x2, ck, c0, c1, tmp, row)),
            _ => throw new InvalidOperationException("unknown step-tree node"),
        };

        // ---------------------------------------------------------------------------------------------
        //  The float64 lane
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Per basis: whether its strong step table has no operation of two weak operands (CPython would compute such a
        ///     pair before NumPy sees it, which the double lane does not model). 0 = not yet computed, 1 = applies, 2 = not.
        /// </summary>
        private static readonly int[] s_f64Lane = new int[8];

        /// <summary>
        ///     Whether <see cref="ValAtScalarF64"/> reproduces <paramref name="basis"/>'s scalar recurrence: every operation of
        ///     its step table has at least one NumPy-scalar operand, so with a float64 series and a float64 x every one is
        ///     float64 scalarmath, a weak Python value converted into the float64 loop (NEP 50). The tables are static, so the
        ///     answer is computed once.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <returns>Whether the lane applies.</returns>
        private static bool F64LaneApplies(PolyBasis basis)
        {
            int i = (int)basis;
            int known = System.Threading.Volatile.Read(ref s_f64Lane[i]);
            if (known != 0)
                return known == 1;
            var p = PolySteps.Eval(basis);
            bool ok = NoWeakPair(p.Pre) && NoWeakPair(p.StepC0) && NoWeakPair(p.StepC1) && NoWeakPair(p.Final)
                      && NoWeakPair(p.FinalLen1) && NoWeakPair(p.HornerInit) && NoWeakPair(p.HornerStep);
            System.Threading.Volatile.Write(ref s_f64Lane[i], ok ? 1 : 2);
            return ok;
        }

        /// <summary>Whether no operation of <paramref name="e"/> combines two weak (Python) operands.</summary>
        /// <param name="e">The step expression, or null (absent).</param>
        /// <returns>True when every operation has a NumPy-scalar operand.</returns>
        private static bool NoWeakPair(PolyExpr e) => e switch
        {
            null => true,
            PolyBin { A: PolyWeak, B: PolyWeak } => false,
            PolyBin b => NoWeakPair(b.A) && NoWeakPair(b.B),
            _ => true,
        };

        /// <summary>
        ///     The float64 lane of <see cref="ValAtScalar(PolyBasis, in PolyNumber, byte*, long, long, NPTypeCode)"/>: a float64
        ///     series at a float64 NumPy scalar. Every statement is then NumPy's float64 scalarmath, so the steps run on plain
        ///     doubles through the SAME cached scalar kernels <see cref="PolyNumber.Binary"/> calls for float64 operands —
        ///     the same instructions, so the same bits, NaN payloads included — without a <see cref="PolyNumber"/> per operand
        ///     (that value's construction, the NEP 50 loop-dtype lookup and the conversions were most of every operation).
        /// </summary>
        /// <param name="prog">The basis's strong step table (checked by <see cref="F64LaneApplies"/>).</param>
        /// <param name="x">The point.</param>
        /// <param name="c">Element 0 of the float64 series.</param>
        /// <param name="n">len(c), at least 1.</param>
        /// <param name="stride">The byte step between coefficients.</param>
        /// <returns>NumPy's value.</returns>
        /// <exception cref="OverflowException">A weak Python int past the float range (CPython's int-to-float error; the
        ///     recurrence's constants are small, so never in practice).</exception>
        private static double ValAtScalarF64(PolyEvalProgram prog, double x, byte* c, long n, long stride)
        {
            double C(long i) => *(double*)(c + i * stride);
            if (prog.Horner)
            {
                double h = RunF64(prog.HornerInit, x, 0, C(n - 1), 0, 0, 0, 0);
                for (long i = 2; i <= n; i++)
                    h = RunF64(prog.HornerStep, x, 0, C(n - i), h, 0, 0, 0);
                return h;
            }
            double x2 = prog.Pre is not null && (prog.PreAlways || n >= 3) ? RunF64(prog.Pre, x, 0, 0, 0, 0, 0, 0) : 0;
            if (n == 1)
                return RunF64(prog.FinalLen1, x, x2, 0, C(0), 0, 0, 0);
            if (n == 2)
                return RunF64(prog.Final, x, x2, 0, C(0), C(1), 0, 0);
            double a0 = C(n - 2), a1 = C(n - 1);
            long nd = n;
            for (long i = 3; i <= n; i++)
            {
                double tmp = a0;
                nd--;
                a0 = RunF64(prog.StepC0, x, x2, C(n - i), 0, a1, 0, nd);
                a1 = RunF64(prog.StepC1, x, x2, 0, 0, a1, tmp, nd);
            }
            return RunF64(prog.Final, x, x2, 0, a0, a1, 0, 0);
        }

        /// <summary>
        ///     <see cref="Run"/> on doubles: a leaf reads the current float64 value, a weak constant is its Python value of the
        ///     row converted into the float64 loop as NEP 50 converts it (a Python int correctly rounded, a Python float as it
        ///     is), and every operation is the float64 scalar kernel of its operator.
        /// </summary>
        /// <param name="e">The expression.</param>
        /// <param name="x">x.</param><param name="x2">x2 (when computed).</param><param name="ck">The coefficient read.</param>
        /// <param name="c0">c0.</param><param name="c1">c1.</param><param name="tmp">tmp.</param>
        /// <param name="row">The row counter <c>nd</c>.</param>
        /// <returns>The value.</returns>
        /// <exception cref="InvalidOperationException">A node kind the step tables never contain (a programming error).</exception>
        private static double RunF64(PolyExpr e, double x, double x2, double ck, double c0, double c1, double tmp, long row)
        {
            switch (e)
            {
                case PolyLeaf l:
                    return l.S switch
                    {
                        PolySym.X => x,
                        PolySym.X2 => x2,
                        PolySym.Ck => ck,
                        PolySym.C0 => c0,
                        PolySym.C1 => c1,
                        PolySym.Tmp => tmp,
                        _ => throw new InvalidOperationException($"symbol {l.S} has no value in a scalar evaluation"),
                    };
                case PolyWeak w:
                    return w.W.Value(row, default).AsDouble();
                case PolyBin b:
                {
                    double a = RunF64(b.A, x, x2, ck, c0, c1, tmp, row), v = RunF64(b.B, x, x2, ck, c0, c1, tmp, row), r;
                    DirectILKernelGenerator.GetPolyScalarBinaryKernel(b.Op, NPTypeCode.Double, PolyComplexProduct.Naive)(
                        (byte*)&a, (byte*)&v, (byte*)&r);
                    return r;
                }
                default:
                    throw new InvalidOperationException("unknown step-tree node");
            }
        }
    }
}
