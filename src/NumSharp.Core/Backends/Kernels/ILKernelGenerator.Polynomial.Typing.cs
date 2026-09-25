using System;

// =============================================================================
// ILKernelGenerator.Polynomial.Typing.cs — NumPy's per-op dtype for the step trees
// =============================================================================
//
// Every node of a step tree is typed exactly as NumPy 2.4.2 types the matching Python line:
//   * two array operands          -> np.promote_types (the house NEP 50 table);
//   * array op Python scalar      -> NEP 50 weak promotion (WeakPromote): a Python int adopts an
//                                    integer partner (and must FIT it — OverflowError otherwise), turns
//                                    a bool partner into int64; a Python float turns bool/int partners
//                                    into float64; a Python complex makes everything complex;
//   * true division of an integer -> float64 (NumPy's 'dd->d' loop).
// The emitter and the planners (PeelCount, ResultType) share TypeOf, so what is planned is what is
// emitted.
//
// PEELING
// -------
// NumPy starts Clenshaw's accumulators at the COEFFICIENTS' dtype (c0 = c[-2], c1 = c[-1]) and each
// step promotes them with x's: c1 absorbs x in the first step, c0 absorbs c1 in the second. So for a
// float32 series at a float64 x the first one or two steps run IN FLOAT32 — measured: pre-casting the
// coefficients to float64 changes 100% of legval/lagval/hermval/hermeval results. PeelCount finds how
// many straight-line steps precede the fixpoint; the kernel emits those, then a loop at fixed dtypes.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>NumPy's dtype rules for the polynomial step trees (see the file header).</summary>
    internal static class PolyTyping
    {
        /// <summary>NumPy's strong-strong promotion (<c>np.result_type</c> of two array dtypes).</summary>
        /// <param name="a">Left dtype.</param><param name="b">Right dtype.</param><returns>The promoted dtype.</returns>
        public static NPTypeCode Promote(NPTypeCode a, NPTypeCode b) => a == b ? a : (NPTypeCode)np.promote_types(a, b);

        /// <summary>
        ///     Bool, every integer dtype and NumSharp's <c>Char</c> (a uint16-like integer) — the dtypes a Python
        ///     float promotes to float64, the coefficient dtypes <c>{p}val</c> converts to float64
        ///     (<c>c.dtype.char in '?bBhHiIlLqQpP'</c>), and the loops whose true division is float64.
        /// </summary>
        /// <param name="t">The dtype.</param><returns>True for bool/int/uint/char.</returns>
        public static bool IsIntLike(NPTypeCode t) => t is NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte
            or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64
            or NPTypeCode.UInt64 or NPTypeCode.Char;

        /// <summary>
        ///     NEP 50: the loop dtype of <c>array op python_scalar</c>. A Python int adopts an integer partner
        ///     (a value that does not fit raises when the constant pool materializes it, like NumPy) and makes a
        ///     bool partner int64; a Python float makes bool/integer partners float64 and keeps inexact ones;
        ///     a Python complex makes any real partner complex (NumPy's complex64 for a float16/float32 partner
        ///     collapses onto NumSharp's single complex128 — the library-wide #569 divergence).
        /// </summary>
        /// <param name="partner">The array operand's dtype.</param>
        /// <param name="kind">The Python scalar's type.</param>
        /// <returns>The loop dtype.</returns>
        public static NPTypeCode WeakPromote(NPTypeCode partner, PyKind kind)
        {
            switch (kind)
            {
                case PyKind.Int:
                    return partner == NPTypeCode.Boolean ? NPTypeCode.Int64 : partner;
                case PyKind.Float:
                    return IsIntLike(partner) ? NPTypeCode.Double : partner;
                default:
                    if (partner == NPTypeCode.Complex) return NPTypeCode.Complex;
                    // Decimal has no NumPy analog; route it through the house table like any strong pair.
                    return partner == NPTypeCode.Decimal ? Promote(NPTypeCode.Decimal, NPTypeCode.Complex) : NPTypeCode.Complex;
            }
        }

        /// <summary>
        ///     The loop dtype of one binary node given its operands' dtypes (null = weak Python value). True
        ///     division of an integer loop is float64, as NumPy's <c>'dd-&gt;d'</c> loop.
        /// </summary>
        /// <param name="b">The node.</param><param name="ta">Left dtype or null.</param><param name="tb">Right dtype or null.</param>
        /// <returns>The dtype both operands are converted to and the result has.</returns>
        /// <exception cref="InvalidOperationException">Both operands are weak — the tree was not folded
        ///     (<see cref="PolySteps.Fold"/>), a program-construction bug.</exception>
        public static NPTypeCode LoopType(PolyBin b, NPTypeCode? ta, NPTypeCode? tb)
        {
            if (ta is null && tb is null)
                throw new InvalidOperationException("two weak operands reached the typing pass: the tree was not folded");
            NPTypeCode t = ta is null ? WeakPromote(tb.Value, ((PolyWeak)b.A).W.Kind)
                         : tb is null ? WeakPromote(ta.Value, ((PolyWeak)b.B).W.Kind)
                         : Promote(ta.Value, tb.Value);
            if (b.Op == BinaryOp.Divide && IsIntLike(t)) t = NPTypeCode.Double;
            return t;
        }

        /// <summary>The dtype of a tree (null for a bare weak value), resolving leaves through <paramref name="sym"/>.</summary>
        /// <param name="e">The tree.</param><param name="sym">Leaf dtypes.</param><returns>The dtype, or null.</returns>
        /// <exception cref="InvalidOperationException">An unknown node type.</exception>
        public static NPTypeCode? TypeOf(PolyExpr e, Func<PolySym, NPTypeCode> sym) => e switch
        {
            PolyLeaf l => sym(l.S),
            PolyWeak => null,
            PolyBin b => LoopType(b, TypeOf(b.A, sym), TypeOf(b.B, sym)),
            _ => throw new InvalidOperationException("unknown step-tree node"),
        };

        /// <summary>The dtype of x2 (<c>2*x</c>/<c>x*2</c>), or x's when the program has no pre-op.</summary>
        /// <param name="prog">The program.</param><param name="tx">x dtype (unused when x is weak).</param>
        /// <returns>The dtype.</returns>
        public static NPTypeCode X2Type(PolyEvalProgram prog, NPTypeCode tx)
            => prog.Pre is null ? tx : TypeOf(prog.Pre, _ => tx).Value;

        /// <summary>
        ///     How many Clenshaw steps must be emitted straight-line before the carried <c>(c0, c1)</c> dtypes
        ///     stop changing (see the file header). A homogeneous call needs 0.
        /// </summary>
        /// <param name="prog">The program.</param><param name="tx">x dtype (unused when x is weak).</param>
        /// <param name="tc">Coefficient dtype.</param>
        /// <returns>The peel count (0 for Horner, whose state is typed once by its init line).</returns>
        /// <exception cref="InvalidOperationException">The dtypes never settle (cannot happen: promotion only widens).</exception>
        public static int PeelCount(PolyEvalProgram prog, NPTypeCode tx, NPTypeCode tc)
        {
            if (prog.Horner) return 0;
            NPTypeCode tx2 = X2Type(prog, tx);
            NPTypeCode t0 = tc, t1 = tc;
            for (int p = 0; p < 8; p++)
            {
                NPTypeCode s0 = t0, s1 = t1;
                NPTypeCode Sym(PolySym s) => s switch
                {
                    PolySym.X => tx, PolySym.X2 => tx2, PolySym.Ck => tc,
                    PolySym.Tmp => s0, PolySym.C0 => s0, PolySym.C1 => s1,
                    _ => throw new InvalidOperationException(),
                };
                NPTypeCode n0 = TypeOf(prog.StepC0, Sym).Value, n1 = TypeOf(prog.StepC1, Sym).Value;
                if (n0 == t0 && n1 == t1) return p;
                t0 = n0; t1 = n1;
            }
            throw new InvalidOperationException("Clenshaw dtypes never reach a fixpoint");
        }

        /// <summary>
        ///     The kernel class for a coefficient count: Horner has one; Clenshaw distinguishes NumPy's
        ///     <c>len(c) == 1</c> and <c>== 2</c> branches, the counts whose steps are all peeled, and
        ///     "peeled steps + loop" — <c>min(nc, P + 3)</c>.
        /// </summary>
        /// <param name="prog">The program.</param><param name="nc">Coefficient count (≥ 1).</param><param name="peel">Peel count.</param>
        /// <returns>The class.</returns>
        public static int Class(PolyEvalProgram prog, long nc, int peel) => prog.Horner ? 1 : (int)Math.Min(nc, peel + 3L);

        /// <summary>The dtype a kernel class returns — NumPy's result dtype for that coefficient count.</summary>
        /// <param name="prog">The program.</param><param name="tx">x dtype (unused when x is weak).</param>
        /// <param name="tc">Coefficient dtype.</param><param name="cls">Kernel class (<see cref="Class"/>).</param>
        /// <param name="peel">Peel count.</param>
        /// <returns>The result dtype.</returns>
        public static NPTypeCode ResultType(PolyEvalProgram prog, NPTypeCode tx, NPTypeCode tc, int cls, int peel)
        {
            if (prog.Horner) return TypeOf(prog.HornerInit, s => s == PolySym.X ? tx : tc).Value;
            NPTypeCode tx2 = X2Type(prog, tx);
            Func<PolySym, NPTypeCode> Env(NPTypeCode a, NPTypeCode b) => s => s switch
            {
                PolySym.X => tx, PolySym.X2 => tx2, PolySym.Ck => tc,
                PolySym.C0 => a, PolySym.Tmp => a, PolySym.C1 => b,
                _ => throw new InvalidOperationException(),
            };
            if (cls == 1) return TypeOf(prog.FinalLen1, Env(tc, tc)).Value;
            NPTypeCode t0 = tc, t1 = tc;
            int steps = cls == 2 ? 0 : (cls <= peel + 2 ? cls - 2 : peel);
            for (int p = 0; p < steps; p++)
            {
                var sym = Env(t0, t1);
                (t0, t1) = (TypeOf(prog.StepC0, sym).Value, TypeOf(prog.StepC1, sym).Value);
            }
            return TypeOf(prog.Final, Env(t0, t1)).Value;
        }
    }
}
