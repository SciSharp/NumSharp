#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
// =============================================================================
// polynomial_il_eval_probe.cs — the house-compliant ENGINE for numpy.polynomial evaluation (U3) and
// Vandermonde (U5): Tier-3A raw IL inner loops (DirectILKernelGenerator.CompileRawInnerLoop) driven by
// NDIter (NDIterRef.ForEach + auxdata), measured against NumPy 2.4.2.
//
// ONE emitter covers every basis and every dtype:
//   * the six bases are DATA — step tables written as small expression trees in NumPy 2.4.2's source
//     order, token for token (Steps below);
//   * every node is typed with NumPy's per-op promotion (strong operands via np.promote_types, Python
//     literals as NEP 50 weak scalars) and emitted through the house EmitScalarOperation /
//     EmitVectorOperation / EmitConvertTo, so there is no per-dtype C# code and no struct kernel;
//   * Clenshaw steps are PEELED until the carried dtypes reach a fixpoint (NumPy runs the first one or
//     two steps in the coefficients' own dtype), then loop;
//   * weak constants (2, (nd-1)/nd, 2*(nd-1), ...) are pre-converted ONCE per call shape into a constant
//     pool the kernel reads through auxdata, rounded to the dtype NumPy would round them to.
// Every cell is byte-compared with NumPy's own output (written by polynomial_il_numpy.py) before it is
// timed. The verdict and the tables live in docs/plans/numpy-polynomial.md §10.
//
// Usage (from this directory; generate data + NumPy timings first with the twin):
//   set NS_PROBE_AFFINITY=0xFF                (same mask as the twin; hybrid-core hosts)
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polynomial_il_numpy.py > numpy_il_times.tsv
//   dotnet run -c Release polynomial_il_eval_probe.cs -- data/il ABCDEFG numpy_il_times.tsv
// Output TSV: cell, variant, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check.
// Sections: A evaluation (6 bases x degree x size), B Tier-3A vs Tier-3B shell vs plain C#, C dtypes and
// mixed dtypes, D non-contiguous x, E BUFFERED on an already-contiguous x, F N-D coefficients
// (multi-series, tensor=False, val2d, grid2d), G Vandermonde (+ vander2d). Default: ABCDEFG.
// AssemblyName=NumSharp.DotNetRunScript grants the InternalsVisibleTo the IL helpers need (the repo's
// Directory.Build.props signs the script with the matching key). PublishAot=false is load-bearing:
// without dynamic code there is no DynamicMethod and every kernel here would throw.
// =============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
#nullable disable

// Pin before anything is JIT-ed so every timed call runs on the chosen cores. Several logical CPUs
// (e.g. 0xFF), not one: a single pinned CPU starves the tiered-JIT background compiler.
if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } affinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

Probe.Data = args.Length > 0 ? args[0] : Path.Combine("data", "il");
if (args.Length > 2) Probe.LoadNumPyTimes(args[2]);
Probe.Run(args.Length > 1 ? args[1] : "ABCDEFG");

/// <summary>The six numpy.polynomial bases. The lower-case names ARE the cell-label prefixes the NumPy
/// twin prints (<c>chebval</c>, <c>legvander</c>, ...), so the two TSVs join on the label.</summary>
enum Basis { poly, cheb, leg, lag, herm, herme }

/// <summary>The values a step expression may read. <c>Tmp</c> is Clenshaw's saved <c>c0</c>;
/// <c>Vm1</c>/<c>Vm2</c> are Vandermonde's two previous rows.</summary>
enum Sym { X, X2, C0, C1, Tmp, Ck, Vm1, Vm2 }

/// <summary>One node of a step expression. The trees are written in NumPy's source order and never
/// re-associated: operand order decides complex-multiply bits and the evaluation order decides float
/// rounding, so a "harmless" refactor of a tree is a parity bug.</summary>
abstract class E { }

/// <summary>A read of one <see cref="Sym"/> (its dtype comes from the emission environment).</summary>
sealed class Leaf : E
{
    /// <summary>The symbol read.</summary>
    public readonly Sym S;
    /// <summary>Creates a leaf reading <paramref name="s"/>.</summary>
    /// <param name="s">The symbol.</param>
    public Leaf(Sym s) { S = s; }
}

/// <summary>A Python literal (or a Python-computed value such as <c>(nd-1)/nd</c>). It has NO dtype of
/// its own: NEP 50 types it by its partner operand, which is why it can only appear inside a
/// <see cref="Bn"/>.</summary>
sealed class Wk : E
{
    /// <summary>The constant's identity and value function.</summary>
    public readonly WeakConst W;
    /// <summary>Wraps <paramref name="w"/> as an expression node.</summary>
    /// <param name="w">The weak constant.</param>
    public Wk(WeakConst w) { W = w; }
}

/// <summary>A binary array op (<c>+ - * /</c>) exactly as NumPy spells it: <c>A op B</c>.</summary>
sealed class Bn : E
{
    /// <summary>The house binary op the emitter lowers this node to.</summary>
    public readonly BinaryOp Op;
    /// <summary>Left and right operands, in NumPy's source order.</summary>
    public readonly E A, B;
    /// <summary>Creates <c>a op b</c>.</summary>
    /// <param name="op">The operation.</param>
    /// <param name="a">Left operand.</param>
    /// <param name="b">Right operand.</param>
    public Bn(BinaryOp op, E a, E b) { Op = op; A = a; B = b; }
}

/// <summary>
///     A weak (Python-literal) constant. <see cref="PerRow"/> constants depend on the recurrence index
///     (<c>nd</c> for Clenshaw, <c>i</c> for Vandermonde); the kernel reads them from row <c>nd</c> of
///     their pool region, the others from row 0.
/// </summary>
sealed class WeakConst
{
    /// <summary>Region identity together with the dtype it is materialized in.</summary>
    public readonly string Name;
    /// <summary>True for a Python float (<c>(nd-1)/nd</c>), false for a Python int (<c>2*(nd-1)</c>).</summary>
    public readonly bool IsFloat;
    /// <summary>Whether the value depends on the row index.</summary>
    public readonly bool PerRow;
    /// <summary>The exact Python value for a row (Python ints stay exact in a double for every row the
    /// kernels read; Python's int/int true division is correctly rounded, as is C#'s double division).</summary>
    public readonly Func<long, double> Value;

    /// <summary>Declares a weak constant.</summary>
    /// <param name="name">Identity (must be unique per value function).</param>
    /// <param name="isFloat">Python float vs Python int.</param>
    /// <param name="perRow">Row-dependent or fixed.</param>
    /// <param name="value">Value per row.</param>
    public WeakConst(string name, bool isFloat, bool perRow, Func<long, double> value)
    {
        Name = name; IsFloat = isFloat; PerRow = perRow; Value = value;
    }
}

/// <summary>A basis's evaluation step table (NumPy 2.4.2 <c>{p}val</c>, source order).</summary>
sealed class EvalProgram
{
    /// <summary>Horner (power basis) instead of the two-accumulator Clenshaw skeleton.</summary>
    public bool Horner;
    /// <summary>The x-only pre-op (<c>x2 = 2*x</c> / <c>x*2</c>), or null.</summary>
    public E Pre;
    /// <summary>Clenshaw: the new <c>c0</c> (reads Ck, C1) and new <c>c1</c> (reads Tmp, C1, X, X2).</summary>
    public E StepC0, StepC1;
    /// <summary>Clenshaw: the returned expression (reads C0, C1, X, X2).</summary>
    public E Final;
    /// <summary>Horner: <c>c0 = c[-1] + x*0</c> and <c>c0 = c[k] + c0*x</c>.</summary>
    public E HornerInit, HornerStep;
}

/// <summary>A basis's Vandermonde step table (NumPy 2.4.2 <c>{p}vander</c>, source order). Row 0 is
/// the shared <c>x*0 + 1</c>; X is already <c>x + 0.0</c>.</summary>
sealed class VanderProgram
{
    /// <summary>The x-only pre-op, or null.</summary>
    public E Pre;
    /// <summary>Row 1.</summary>
    public E V1;
    /// <summary>Row i from rows i-1 (Vm1) and i-2 (Vm2).</summary>
    public E Step;
}

/// <summary>NumPy 2.4.2's step expressions for every basis, as data.</summary>
static class Steps
{
    /// <summary>Leaf shorthands.</summary>
    public static readonly E X = new Leaf(Sym.X), X2 = new Leaf(Sym.X2), C0 = new Leaf(Sym.C0), C1 = new Leaf(Sym.C1),
        Tmp = new Leaf(Sym.Tmp), Ck = new Leaf(Sym.Ck), Vm1 = new Leaf(Sym.Vm1), Vm2 = new Leaf(Sym.Vm2);

    /// <summary><c>a + b</c>.</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
    public static E Add(E a, E b) => new Bn(BinaryOp.Add, a, b);
    /// <summary><c>a - b</c>.</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
    public static E Sub(E a, E b) => new Bn(BinaryOp.Subtract, a, b);
    /// <summary><c>a * b</c>.</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
    public static E Mul(E a, E b) => new Bn(BinaryOp.Multiply, a, b);
    /// <summary><c>a / b</c> (true division).</summary><param name="a">Left.</param><param name="b">Right.</param><returns>The node.</returns>
    public static E Div(E a, E b) => new Bn(BinaryOp.Divide, a, b);
    /// <summary>A weak constant node.</summary><param name="w">The constant.</param><returns>The node.</returns>
    public static E W(WeakConst w) => new Wk(w);

    /// <summary>Fixed Python literals.</summary>
    public static readonly WeakConst I0 = new("0", false, false, _ => 0), I1 = new("1", false, false, _ => 1),
        I2 = new("2", false, false, _ => 2), F0 = new("0.0", true, false, _ => 0.0);

    /// <summary>Row-dependent Python values (row = <c>nd</c> in Clenshaw, <c>i</c> in Vandermonde).</summary>
    public static readonly WeakConst NdM1 = new("nd-1", false, true, r => r - 1),
        Nd = new("nd", false, true, r => r),
        TwoNdM1 = new("2nd-1", false, true, r => 2 * r - 1),
        TwoNdM1b = new("2(nd-1)", false, true, r => 2 * (r - 1)),
        Q1 = new("(nd-1)/nd", true, true, r => (double)(r - 1) / r),
        Q2 = new("(2nd-1)/nd", true, true, r => (double)(2 * r - 1) / r);

    /// <summary>
    ///     <c>{p}val</c> step tables. Each line is the NumPy 2.4.2 source line it transcribes; note
    ///     <c>legval</c>'s <c>c1*((nd-1)/nd)</c> (one Python float) vs <c>lagval</c>'s <c>(c1*(nd-1))/nd</c>
    ///     (two array ops) — the same arithmetic, different bits.
    /// </summary>
    public static readonly Dictionary<Basis, EvalProgram> Eval = new()
    {
        // c0 = c[-1] + x*0 ; c0 = c[-i] + c0*x
        [Basis.poly] = new() { Horner = true, HornerInit = Add(Ck, Mul(X, W(I0))), HornerStep = Add(Ck, Mul(C0, X)) },
        // x2 = 2*x ; c0 = c[-i] - c1 ; c1 = tmp + c1*x2 ; return c0 + c1*x
        [Basis.cheb] = new() { Pre = Mul(W(I2), X), StepC0 = Sub(Ck, C1), StepC1 = Add(Tmp, Mul(C1, X2)), Final = Add(C0, Mul(C1, X)) },
        // c0 = c[-i] - c1*((nd - 1)/nd) ; c1 = tmp + c1*x*((2*nd - 1)/nd) ; return c0 + c1*x
        [Basis.leg] = new() { StepC0 = Sub(Ck, Mul(C1, W(Q1))), StepC1 = Add(Tmp, Mul(Mul(C1, X), W(Q2))), Final = Add(C0, Mul(C1, X)) },
        // c0 = c[-i] - (c1*(nd - 1))/nd ; c1 = tmp + (c1*((2*nd - 1) - x))/nd ; return c0 + c1*(1 - x)
        [Basis.lag] = new()
        {
            StepC0 = Sub(Ck, Div(Mul(C1, W(NdM1)), W(Nd))),
            StepC1 = Add(Tmp, Div(Mul(C1, Sub(W(TwoNdM1), X)), W(Nd))),
            Final = Add(C0, Mul(C1, Sub(W(I1), X))),
        },
        // x2 = x*2 ; c0 = c[-i] - c1*(2*(nd - 1)) ; c1 = tmp + c1*x2 ; return c0 + c1*x2
        [Basis.herm] = new() { Pre = Mul(X, W(I2)), StepC0 = Sub(Ck, Mul(C1, W(TwoNdM1b))), StepC1 = Add(Tmp, Mul(C1, X2)), Final = Add(C0, Mul(C1, X2)) },
        // c0 = c[-i] - c1*(nd - 1) ; c1 = tmp + c1*x ; return c0 + c1*x
        [Basis.herme] = new() { StepC0 = Sub(Ck, Mul(C1, W(NdM1))), StepC1 = Add(Tmp, Mul(C1, X)), Final = Add(C0, Mul(C1, X)) },
    };

    /// <summary><c>{p}vander</c> step tables (row 0 = <c>x*0 + 1</c> for every basis).</summary>
    public static readonly Dictionary<Basis, VanderProgram> Vander = new()
    {
        // v[1] = x ; v[i] = v[i-1]*x
        [Basis.poly] = new() { V1 = X, Step = Mul(Vm1, X) },
        // x2 = 2*x ; v[1] = x ; v[i] = v[i-1]*x2 - v[i-2]
        [Basis.cheb] = new() { Pre = Mul(W(I2), X), V1 = X, Step = Sub(Mul(Vm1, X2), Vm2) },
        // v[1] = x ; v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i
        [Basis.leg] = new() { V1 = X, Step = Div(Sub(Mul(Mul(Vm1, X), W(TwoNdM1)), Mul(Vm2, W(NdM1))), W(Nd)) },
        // v[1] = 1 - x ; v[i] = (v[i-1]*(2*i - 1 - x) - v[i-2]*(i - 1))/i
        [Basis.lag] = new() { V1 = Sub(W(I1), X), Step = Div(Sub(Mul(Vm1, Sub(W(TwoNdM1), X)), Mul(Vm2, W(NdM1))), W(Nd)) },
        // x2 = x*2 ; v[1] = x2 ; v[i] = (v[i-1]*x2 - v[i-2]*(2*(i - 1)))
        [Basis.herm] = new() { Pre = Mul(X, W(I2)), V1 = X2, Step = Sub(Mul(Vm1, X2), Mul(Vm2, W(TwoNdM1b))) },
        // v[1] = x ; v[i] = (v[i-1]*x - v[i-2]*(i - 1))
        [Basis.herme] = new() { V1 = X, Step = Sub(Mul(Vm1, X), Mul(Vm2, W(NdM1))) },
    };

    /// <summary>Returns <paramref name="e"/> with every read of <paramref name="s"/> replaced by
    /// <paramref name="with"/> (Clenshaw's <c>len(c) == 1</c> case sets <c>c1 = 0</c>, a Python int).</summary>
    /// <param name="e">The tree.</param><param name="s">The symbol to replace.</param><param name="with">The replacement.</param>
    /// <returns>A new tree (shared subtrees are fine: trees are immutable).</returns>
    public static E Subst(E e, Sym s, E with) => e switch
    {
        Leaf l when l.S == s => with,
        Bn b => new Bn(b.Op, Subst(b.A, s, with), Subst(b.B, s, with)),
        _ => e,
    };

    /// <summary>Whether <paramref name="e"/> reads a per-row weak constant — when neither step does, the kernel
    /// never maintains the row index (one store less per step).</summary>
    /// <param name="e">The tree.</param><returns>True when a per-row constant is read.</returns>
    public static bool UsesRow(E e) => e switch
    {
        Wk w => w.W.PerRow,
        Bn b => UsesRow(b.A) || UsesRow(b.B),
        _ => false,
    };

    /// <summary>Whether <paramref name="e"/> reads <paramref name="s"/> (decides whether x2 must be emitted
    /// before a class's body: chebval only computes it when <c>len(c) &gt;= 3</c>, hermval always).</summary>
    /// <param name="e">The tree.</param><param name="s">The symbol.</param><returns>True when read.</returns>
    public static bool Uses(E e, Sym s) => e switch
    {
        Leaf l => l.S == s,
        Bn b => Uses(b.A, s) || Uses(b.B, s),
        _ => false,
    };
}

/// <summary>
///     The probe: a NumPy-typed IL emitter for the polynomial recurrences, the Tier-3A kernels built on
///     it, NDIter drivers mirroring NumPy's <c>{p}val</c> / <c>_valnd</c> / <c>_gridnd</c> / <c>{p}vander</c> /
///     <c>_vander_nd</c>, and a harness that byte-checks and times every cell.
/// </summary>
static unsafe class Probe
{
    /// <summary>Directory holding the .npy inputs and NumPy references written by the twin.</summary>
    public static string Data = Path.Combine("data", "il");

    // ======================================================================== NumPy typing

    /// <summary>NumPy's strong-strong promotion (<c>np.result_type</c> of two array dtypes).</summary>
    /// <param name="a">Left dtype.</param><param name="b">Right dtype.</param><returns>The promoted dtype.</returns>
    static NPTypeCode Promote(NPTypeCode a, NPTypeCode b) => a == b ? a : (NPTypeCode)np.promote_types(a, b);

    /// <summary>Bool and every integer dtype (the dtypes a Python float promotes to float64).</summary>
    /// <param name="t">The dtype.</param><returns>True for bool/int/uint/char.</returns>
    static bool IsIntLike(NPTypeCode t) => t is NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16
        or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char;

    /// <summary>
    ///     NEP 50: the dtype of <c>array op python_scalar</c>. A Python int adopts an integer partner's
    ///     dtype (a value that does not fit raises OverflowError at pool-fill time, like NumPy), makes a
    ///     bool partner int64; a Python float makes any integer/bool partner float64; float, complex and
    ///     decimal partners are kept.
    /// </summary>
    /// <param name="partner">The array operand's dtype.</param>
    /// <param name="isFloat">Whether the literal is a Python float.</param>
    /// <returns>The loop dtype.</returns>
    static NPTypeCode WeakPromote(NPTypeCode partner, bool isFloat)
    {
        if (partner == NPTypeCode.Boolean) return isFloat ? NPTypeCode.Double : NPTypeCode.Int64;
        if (IsIntLike(partner)) return isFloat ? NPTypeCode.Double : partner;
        return partner;
    }

    /// <summary>The loop dtype of one binary node given its operands' dtypes (null = weak literal).
    /// <c>true_divide</c> of an integer loop is float64, as NumPy's <c>'dd-&gt;d'</c> loop.</summary>
    /// <param name="b">The node.</param><param name="ta">Left dtype or null.</param><param name="tb">Right dtype or null.</param>
    /// <returns>The dtype both operands are converted to and the result has.</returns>
    /// <exception cref="InvalidOperationException">Both operands are weak (a tree the port never builds:
    /// Python would fold such a subexpression before NumPy sees it).</exception>
    public static NPTypeCode LoopType(Bn b, NPTypeCode? ta, NPTypeCode? tb)
    {
        if (ta is null && tb is null) throw new InvalidOperationException("two weak operands: fold the constant expression instead");
        NPTypeCode t = ta is null ? WeakPromote(tb.Value, ((Wk)b.A).W.IsFloat)
                     : tb is null ? WeakPromote(ta.Value, ((Wk)b.B).W.IsFloat)
                     : Promote(ta.Value, tb.Value);
        if (b.Op == BinaryOp.Divide && IsIntLike(t)) t = NPTypeCode.Double;
        return t;
    }

    /// <summary>The dtype of a tree (null for a bare weak constant), resolving leaves through
    /// <paramref name="sym"/>. The emitter and the static planners share it, so the planned dtypes and
    /// the emitted ones cannot drift apart.</summary>
    /// <param name="e">The tree.</param><param name="sym">Leaf dtypes.</param><returns>The dtype, or null.</returns>
    public static NPTypeCode? TypeOf(E e, Func<Sym, NPTypeCode> sym) => e switch
    {
        Leaf l => sym(l.S),
        Wk => null,
        Bn b => LoopType(b, TypeOf(b.A, sym), TypeOf(b.B, sym)),
        _ => throw new InvalidOperationException(),
    };

    /// <summary>
    ///     How many Clenshaw steps must be emitted straight-line before the carried <c>(c0, c1)</c>
    ///     dtypes stop changing. NumPy starts both at the coefficient dtype; <c>c1</c> absorbs x's dtype
    ///     in the first step and <c>c0</c> absorbs <c>c1</c>'s in the second, so a float32 series at a
    ///     float64 x needs 2 (and those two steps compute IN FLOAT32, which changes the result bits —
    ///     pre-casting the coefficients is not equivalent). A homogeneous call needs 0.
    /// </summary>
    /// <param name="prog">The step table.</param><param name="tx">x dtype.</param><param name="tc">Coefficient dtype.</param>
    /// <returns>The peel count (0 for Horner, whose state is typed once by its init line).</returns>
    /// <exception cref="InvalidOperationException">The dtypes never settle (cannot happen for these tables:
    /// promotion only widens).</exception>
    static int PeelCount(EvalProgram prog, NPTypeCode tx, NPTypeCode tc)
    {
        if (prog.Horner) return 0;
        NPTypeCode tx2 = prog.Pre is null ? tx : TypeOf(prog.Pre, _ => tx).Value;
        NPTypeCode t0 = tc, t1 = tc;
        for (int p = 0; p < 4; p++)
        {
            NPTypeCode s0 = t0, s1 = t1;
            Func<Sym, NPTypeCode> sym = s => s switch { Sym.X => tx, Sym.X2 => tx2, Sym.Ck => tc, Sym.Tmp => s0, Sym.C0 => s0, Sym.C1 => s1, _ => throw new InvalidOperationException() };
            NPTypeCode n0 = TypeOf(prog.StepC0, sym).Value, n1 = TypeOf(prog.StepC1, sym).Value;
            if (n0 == t0 && n1 == t1) return p;
            t0 = n0; t1 = n1;
        }
        throw new InvalidOperationException("Clenshaw dtypes never reach a fixpoint");
    }

    /// <summary>The dtype a kernel class returns (NumPy's result dtype for that coefficient count).</summary>
    /// <param name="prog">Step table.</param><param name="tx">x dtype.</param><param name="tc">Coefficient dtype.</param>
    /// <param name="cls">Coefficient-count class (1, 2, or the straight-line/loop class).</param><param name="P">Peel count.</param>
    /// <returns>The result dtype.</returns>
    static NPTypeCode ResultType(EvalProgram prog, NPTypeCode tx, NPTypeCode tc, int cls, int P)
    {
        NPTypeCode tx2 = prog.Pre is null ? tx : TypeOf(prog.Pre, _ => tx).Value;
        if (prog.Horner) return TypeOf(prog.HornerInit, s => s == Sym.X ? tx : tc).Value;
        Func<NPTypeCode, NPTypeCode, Func<Sym, NPTypeCode>> env = (a, b) => s => s switch
        {
            Sym.X => tx, Sym.X2 => tx2, Sym.Ck => tc, Sym.C0 => a, Sym.Tmp => a, Sym.C1 => b, _ => throw new InvalidOperationException(),
        };
        if (cls == 1) return TypeOf(Steps.Subst(prog.Final, Sym.C1, Steps.W(Steps.I0)), env(tc, tc)).Value;
        NPTypeCode t0 = tc, t1 = tc;
        int steps = cls == 2 ? 0 : (cls <= P + 2 ? cls - 2 : P);
        for (int p = 0; p < steps; p++)
        {
            var sym = env(t0, t1);
            (t0, t1) = (TypeOf(prog.StepC0, sym).Value, TypeOf(prog.StepC1, sym).Value);
        }
        return TypeOf(prog.Final, env(t0, t1)).Value;
    }

    // ======================================================================== value kinds

    /// <summary>
    ///     How values of one dtype live in IL: a scalar of the CLR type, or a <c>Vector{bits}&lt;T&gt;</c>. In
    ///     vector chains every per-point value carries the SAME lane count W (the loop dtype's lanes at the
    ///     host width), so a float32 x beside a float64 loop is a <c>Vector128&lt;float&gt;</c> of 4 lanes that
    ///     widens to one <c>Vector256&lt;double&gt;</c> — lane-matched, no shuffles.
    /// </summary>
    class K
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
        public Type LocalType;

        /// <summary>Creates a kind.</summary>
        /// <param name="vec">Vector or scalar.</param><param name="t">The dtype.</param>
        /// <param name="bits">Vector width (128/256/512); ignored for scalars.</param>
        /// <exception cref="NotSupportedException">A vector width with no Vector type.</exception>
        public K(bool vec, NPTypeCode t, int bits = 0)
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

        /// <summary>[a, b] -> [a op b]. Scalars go through the house EmitScalarOperation (NumPy-exact per dtype:
        /// complex array-loop multiply, float16 widen-compute-narrow, decimal); vectors through the
        /// <c>Vector{bits}&lt;T&gt;</c> operator at THIS kind's width (the house EmitVectorOperation is fixed to the
        /// host width, and a lane-matched float32 x beside a float64 loop is narrower).</summary>
        /// <param name="il">The generator.</param><param name="op">The op.</param>
        public virtual void Bin(ILGenerator il, BinaryOp op)
        {
            if (Vec) EmitVecOperator(il, op, LocalType);
            else DirectILKernelGenerator.EmitScalarOperation(il, op, T);
        }

        /// <summary>[address] -> [value]: one element, or one contiguous vector.</summary>
        /// <param name="il">The generator.</param>
        public virtual void Load(ILGenerator il)
        {
            if (Vec) il.EmitCall(OpCodes.Call, VectorMethodCache.Load(Bits, Clr), null);
            else DirectILKernelGenerator.EmitLoadIndirect(il, T);
        }

        /// <summary>[scalar of T] -> [vector of this kind] (every lane the same value).</summary>
        /// <param name="il">The generator.</param>
        public virtual void BroadcastFromScalar(ILGenerator il) =>
            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(Bits, Clr), null);

        /// <summary>[value, address] -> []: stores one element (scalar) or one vector.</summary>
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
    ///     float16 as 8 float32 lanes kept ON the float16 grid: every op computes in float32 then narrows
    ///     (round-to-nearest-even) and widens back — exactly NumPy's HALF loop, one op at a time. The .NET
    ///     BCL has no vector float16 arithmetic, so this uses the house <c>HalfWiden8V</c>/<c>HalfNarrow8V</c>
    ///     AVX2 primitives (private; bound by reflection here, internal in a real implementation). Because the
    ///     lanes already ARE float32 values, converting a float16 lane value to a float32 loop costs nothing.
    /// </summary>
    sealed class HalfVecK : K
    {
        static readonly MethodInfo Widen = typeof(DirectILKernelGenerator).GetMethod("HalfWiden8V", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException("DirectILKernelGenerator.HalfWiden8V");
        static readonly MethodInfo Narrow = typeof(DirectILKernelGenerator).GetMethod("HalfNarrow8V", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException("DirectILKernelGenerator.HalfNarrow8V");

        /// <summary>Creates the float16 vector kind (8 lanes, Vector256&lt;float&gt; locals).</summary>
        public HalfVecK() : base(true, NPTypeCode.Single, 256) { }

        /// <inheritdoc/>
        public override void Bin(ILGenerator il, BinaryOp op)
        {
            EmitVecOperator(il, op, typeof(Vector256<float>));
            il.EmitCall(OpCodes.Call, Narrow, null);
            il.EmitCall(OpCodes.Call, Widen, null);
        }

        /// <inheritdoc/>
        public override void Load(ILGenerator il)
        {
            il.EmitCall(OpCodes.Call, VectorMethodCache.Load(128, typeof(ushort)), null);
            il.EmitCall(OpCodes.Call, Widen, null);
        }

        /// <inheritdoc/>
        public override void BroadcastFromScalar(ILGenerator il)
        {
            // [Half] -> float (exact) -> 8 lanes.
            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Half, NPTypeCode.Single);
            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(256, typeof(float)), null);
        }

        /// <inheritdoc/>
        public override void StoreValueFirst(ILGenerator il)
        {
            var ptr = il.DeclareLocal(typeof(byte*)); il.Emit(OpCodes.Stloc, ptr);
            il.EmitCall(OpCodes.Call, Narrow, null);
            il.Emit(OpCodes.Ldloc, ptr);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Store(128, typeof(ushort)), null);
        }

        /// <inheritdoc/>
        public override int Lanes => 8;
    }

    /// <summary>[a, b] -> [a op b] through the vector type's own operator (exact IEEE lane-wise for floats,
    /// wrapping for integers — the same as NumPy's loops).</summary>
    /// <param name="il">The generator.</param><param name="op">Add/Subtract/Multiply/Divide.</param><param name="vt">The vector type.</param>
    /// <exception cref="NotSupportedException">Another op.</exception>
    /// <exception cref="MissingMethodException">The vector type lacks that operator.</exception>
    static void EmitVecOperator(ILGenerator il, BinaryOp op, Type vt)
    {
        string name = op switch
        {
            BinaryOp.Add => "op_Addition", BinaryOp.Subtract => "op_Subtraction",
            BinaryOp.Multiply => "op_Multiply", BinaryOp.Divide => "op_Division",
            _ => throw new NotSupportedException(op.ToString()),
        };
        il.EmitCall(OpCodes.Call, vt.GetMethod(name, new[] { vt, vt }) ?? throw new MissingMethodException(vt.Name, name), null);
    }

    static readonly MethodInfo s_cvtps2pd = typeof(Avx).GetMethod(nameof(Avx.ConvertToVector256Double), new[] { typeof(Vector128<float>) });
    static readonly MethodInfo s_cvtdq2pd = typeof(Avx).GetMethod(nameof(Avx.ConvertToVector256Double), new[] { typeof(Vector128<int>) });
    static readonly MethodInfo s_cvtqq2pd = typeof(Vector256).GetMethod(nameof(Vector256.ConvertToDouble), new[] { typeof(Vector256<long>) });

    /// <summary>
    ///     Whether a per-point value of <paramref name="from"/> can be converted lane-for-lane into the
    ///     <paramref name="to"/> loop at W lanes — the gate for vector chains over MIXED dtypes. Every listed
    ///     conversion is exact (NumPy's loop-input cast is exact for these pairs too); anything else runs the
    ///     scalar chains. (A 512-bit host would add the Avx512F twins of the same instructions.)
    /// </summary>
    /// <param name="from">Per-point dtype.</param><param name="to">Loop dtype.</param><param name="w">Lanes.</param>
    /// <returns>True when <see cref="EmitLaneConvert"/> handles the pair.</returns>
    static bool LaneConvertible(NPTypeCode from, NPTypeCode to, int w) =>
        from == to
        || (from == NPTypeCode.Half && to == NPTypeCode.Single && w == 8 && Avx2.IsSupported)
        || (to == NPTypeCode.Double && w == 4 && Avx.IsSupported
            && (from == NPTypeCode.Single || from == NPTypeCode.Int32 || from == NPTypeCode.Int64));

    /// <summary>[lane vector of from] -> [lane vector of to] (see <see cref="LaneConvertible"/>).</summary>
    /// <param name="il">Generator.</param><param name="from">Source dtype.</param><param name="to">Target dtype.</param>
    /// <exception cref="NotSupportedException">A pair the gate rejects (the gate and the emitter disagree).</exception>
    static void EmitLaneConvert(ILGenerator il, NPTypeCode from, NPTypeCode to)
    {
        if (from == to) return;
        if (from == NPTypeCode.Half && to == NPTypeCode.Single) return;   // HalfVecK lanes already are float32
        var m = (from, to) switch
        {
            (NPTypeCode.Single, NPTypeCode.Double) => s_cvtps2pd,
            (NPTypeCode.Int32, NPTypeCode.Double) => s_cvtdq2pd,
            (NPTypeCode.Int64, NPTypeCode.Double) => s_cvtqq2pd,
            _ => throw new NotSupportedException($"no lane conversion {from} -> {to}"),
        };
        il.EmitCall(OpCodes.Call, m, null);
    }

    /// <summary>
    ///     Whether a block may run vector chains: the loop dtype vectorizes (float16 via AVX2 only when the
    ///     whole call is float16; complex and decimal never), x converts lane-for-lane into it, and — when every
    ///     point reads its own series — so do the coefficients. Broadcast coefficients and weak constants are
    ///     SHARED scalars in their own dtype (NumPy's exact semantics for coefficient-only math) and only get
    ///     converted + broadcast where they meet a per-point value, so they never limit the gate.
    /// </summary>
    /// <param name="tx">x dtype.</param><param name="tc">Coefficient dtype.</param><param name="tl">Loop dtype.</param>
    /// <param name="perLaneCoef">Every point reads its own series.</param><returns>True when vector chains apply.</returns>
    static bool VectorBlockOk(NPTypeCode tx, NPTypeCode tc, NPTypeCode tl, bool perLaneCoef)
    {
        if (tl == NPTypeCode.Half) return Avx2.IsSupported && tx == NPTypeCode.Half && tc == NPTypeCode.Half;
        if (tl is not (NPTypeCode.Single or NPTypeCode.Double) || !DirectILKernelGenerator.CanUseSimd(tl)) return false;
        int w = DirectILKernelGenerator.GetVectorCount(tl);
        return LaneConvertible(tx, tl, w) && (!perLaneCoef || LaneConvertible(tc, tl, w));
    }

    /// <summary>Lanes of the loop dtype in vector chains (8 for float16, which rides 8 float32 lanes).</summary>
    /// <param name="tl">Loop dtype.</param><returns>Lanes.</returns>
    static int LoopLanes(NPTypeCode tl) => tl == NPTypeCode.Half ? 8 : DirectILKernelGenerator.GetVectorCount(tl);

    // ======================================================================== constant pool

    /// <summary>
    ///     The weak constants one kernel reads, registered WHILE it is emitted (one region per
    ///     constant x dtype), and the per-row-count tables built from them. A region is an array of
    ///     <c>rows</c> elements of its dtype, filled with NumPy's own conversion (the house astype), so a
    ///     float16 kernel reads <c>(nd-1)/nd</c> rounded to float16 exactly as NumPy rounds the Python
    ///     float. Tables are cached per row count: the per-call cost is one dictionary lookup.
    /// </summary>
    sealed class ConstPool
    {
        /// <summary>Registered regions, in emission order (the index is baked into the IL).</summary>
        public readonly List<(WeakConst W, NPTypeCode T)> Regions = new();
        readonly Dictionary<long, IntPtr> _tables = new();
        readonly List<NDArray> _keep = new();

        /// <summary>Returns the region index for (<paramref name="w"/>, <paramref name="t"/>), adding it on first use.</summary>
        /// <param name="w">The constant.</param><param name="t">The dtype it is materialized in.</param><returns>The index.</returns>
        public int Region(WeakConst w, NPTypeCode t)
        {
            for (int r = 0; r < Regions.Count; r++)
                if (ReferenceEquals(Regions[r].W, w) && Regions[r].T == t) return r;
            Regions.Add((w, t));
            return Regions.Count - 1;
        }

        /// <summary>The region table for a call reading rows <c>[0, rows)</c>; built once per row count.</summary>
        /// <param name="rows">Row count (coefficient count for evaluation, degree + 1 for Vandermonde).</param>
        /// <returns>Address of a table of region base pointers.</returns>
        /// <exception cref="OverflowException">A Python int does not fit an integer x dtype — NumPy's
        /// <c>OverflowError: Python integer N out of bounds for int8</c> (e.g. lagval on int8 x with 70 coefficients).</exception>
        public long TableFor(long rows)
        {
            if (_tables.TryGetValue(rows, out var t)) return (long)t;
            var table = (long*)NativeMemory.Alloc((nuint)Math.Max(1, Regions.Count), 8);
            for (int r = 0; r < Regions.Count; r++)
            {
                var (w, dt) = Regions[r];
                long n = Math.Max(rows, 2);
                // Rows 0/1 of a per-row constant are never read (the recurrences start at nd = 2), so they
                // hold 0 instead of a value that could spuriously fail the integer range check.
                NDArray arr;
                if (w.IsFloat)
                {
                    var v = new double[n];
                    for (long i = 0; i < n; i++) v[i] = w.PerRow ? (i < 2 ? 0 : w.Value(i)) : w.Value(0);
                    arr = np.array(v).astype(dt);
                }
                else
                {
                    var v = new long[n];
                    for (long i = 0; i < n; i++) v[i] = (long)(w.PerRow ? (i < 2 ? 0 : w.Value(i)) : w.Value(0));
                    var src = np.array(v);
                    arr = src.astype(dt);
                    if (IsIntLike(dt))
                    {
                        // Dtype-agnostic range check: a value survives the round trip iff it fits.
                        var back = arr.astype(NPTypeCode.Int64);
                        for (long i = 0; i < n; i++)
                            if (back.GetInt64(i) != v[i])
                                throw new OverflowException($"Python integer {v[i]} out of bounds for {dt.AsNumpyDtypeName()}");
                    }
                }
                _keep.Add(arr);
                table[r] = (long)((byte*)arr.Storage.Address + arr.Shape.offset * arr.dtypesize);
            }
            _tables[rows] = (IntPtr)table;
            return (long)table;
        }
    }

    // ======================================================================== the typed emitter

    /// <summary>
    ///     A value per chain. A SHARED value (a broadcast coefficient, a weak constant, or math on those
    ///     only) is ONE scalar local of its own dtype, computed once for all chains — so coefficient-only
    ///     math (NumPy's peeled first steps in the coefficient dtype) costs nothing per point. A per-point
    ///     value is one local per chain: a scalar in scalar chains, a W-lane vector in vector chains.
    /// </summary>
    sealed class TVal
    {
        /// <summary>Per-chain locals (the same scalar local repeated when shared).</summary>
        public readonly LocalBuilder[] L;
        /// <summary>The value's NumPy dtype.</summary>
        public readonly NPTypeCode T;
        /// <summary>One scalar for all chains.</summary>
        public readonly bool Shared;
        /// <summary>Wraps locals.</summary><param name="l">Locals.</param><param name="t">Dtype.</param><param name="shared">Shared flag.</param>
        public TVal(LocalBuilder[] l, NPTypeCode t, bool shared) { L = l; T = t; Shared = shared; }
    }

    /// <summary>The values the step expressions may read, bound per block.</summary>
    sealed class Env
    {
        /// <summary>Symbol values (null when not bound in the current context).</summary>
        public TVal X, X2, C0, C1, Tmp, Ck, Vm1, Vm2;

        /// <summary>The bound value of <paramref name="s"/>.</summary>
        /// <param name="s">The symbol.</param><returns>Its value.</returns>
        /// <exception cref="InvalidOperationException">The symbol is not bound — a step table reads something
        /// the class never computed (e.g. x2 before it was emitted).</exception>
        public TVal Get(Sym s) => (s switch
        {
            Sym.X => X, Sym.X2 => X2, Sym.C0 => C0, Sym.C1 => C1, Sym.Tmp => Tmp, Sym.Ck => Ck, Sym.Vm1 => Vm1, Sym.Vm2 => Vm2,
            _ => null,
        }) ?? throw new InvalidOperationException($"symbol {s} is not bound here");
    }

    /// <summary>
    ///     Emits step expressions for U interleaved chains. Independent chains are what make the
    ///     recurrence fast: each point's Clenshaw is one serial dependency chain, so interleaving U of them
    ///     hides the multiply/add latency (measured: x4 is 1.3-2.4x faster than x1).
    /// </summary>
    sealed class Emitter
    {
        /// <summary>The generator.</summary>
        public readonly ILGenerator IL;
        /// <summary>Chain count.</summary>
        public readonly int U;
        /// <summary>Vector chains (per-point values are W-lane vectors) or scalar chains.</summary>
        public readonly bool Vec;
        /// <summary>Lanes per chain (1 for scalar chains).</summary>
        public readonly int W;
        readonly ConstPool _pool;
        readonly LocalBuilder _table, _row;
        readonly Dictionary<NPTypeCode, K> _scalar = new(), _lane = new();

        /// <summary>Creates an emitter.</summary>
        /// <param name="il">Generator.</param><param name="u">Chains.</param><param name="vec">Vector chains.</param>
        /// <param name="lanes">Lanes per chain in vector chains (the loop dtype's).</param>
        /// <param name="pool">The kernel's constant pool.</param><param name="table">Local holding the region table pointer.</param>
        /// <param name="row">Local holding the current row (nd or i) for per-row constants.</param>
        public Emitter(ILGenerator il, int u, bool vec, int lanes, ConstPool pool, LocalBuilder table, LocalBuilder row)
        {
            IL = il; U = u; Vec = vec; W = vec ? lanes : 1; _pool = pool; _table = table; _row = row;
        }

        /// <summary>The scalar kind of <paramref name="t"/> (shared values, scalar chains).</summary>
        /// <param name="t">The dtype.</param><returns>The kind.</returns>
        public K Scalar(NPTypeCode t)
        {
            if (!_scalar.TryGetValue(t, out var k)) _scalar[t] = k = new K(false, t);
            return k;
        }

        /// <summary>The per-point kind of <paramref name="t"/>: a W-lane vector in vector chains, else the scalar.</summary>
        /// <param name="t">The dtype.</param><returns>The kind.</returns>
        /// <exception cref="NotSupportedException">No W-lane vector exists for t (the block gate should have refused).</exception>
        public K Lane(NPTypeCode t)
        {
            if (!Vec) return Scalar(t);
            if (!_lane.TryGetValue(t, out var k))
            {
                if (t == NPTypeCode.Half)
                    k = W == 8 ? new HalfVecK() : throw new NotSupportedException($"float16 lanes need W = 8, got {W}");
                else
                    k = new K(true, t, W * 8 * DirectILKernelGenerator.GetTypeSize(t));
                _lane[t] = k;
            }
            return k;
        }

        /// <summary>Declares locals for a value.</summary>
        /// <param name="t">Dtype.</param><param name="shared">One scalar for all chains.</param><returns>The value.</returns>
        public TVal Fresh(NPTypeCode t, bool shared)
        {
            var l = new LocalBuilder[U];
            if (shared) { var one = IL.DeclareLocal(Scalar(t).LocalType); for (int u = 0; u < U; u++) l[u] = one; }
            else { var type = Lane(t).LocalType; for (int u = 0; u < U; u++) l[u] = IL.DeclareLocal(type); }
            return new TVal(l, t, shared);
        }

        /// <summary>The per-point form of a shared value: in vector chains a fresh broadcast vector (built once
        /// for all chains); in scalar chains the scalar itself.</summary>
        /// <param name="s">A shared value.</param><returns>A local every chain can read.</returns>
        LocalBuilder Spread(TVal s)
        {
            if (!Vec) return s.L[0];
            var k = Lane(s.T);
            var v = IL.DeclareLocal(k.LocalType);
            IL.Emit(OpCodes.Ldloc, s.L[0]);
            k.BroadcastFromScalar(IL);
            IL.Emit(OpCodes.Stloc, v);
            return v;
        }

        /// <summary>Copies <paramref name="from"/> into the per-point <paramref name="to"/> chain by chain (a shared
        /// source is spread): how loop-carried state is written back.</summary>
        /// <param name="from">Source of the same dtype.</param><param name="to">Per-point destination.</param>
        public void CopyTo(TVal from, TVal to)
        {
            var spread = from.Shared ? Spread(from) : null;
            for (int u = 0; u < U; u++) { IL.Emit(OpCodes.Ldloc, spread ?? from.L[u]); IL.Emit(OpCodes.Stloc, to.L[u]); }
        }

        /// <summary>Emits a tree at its own NumPy dtype.</summary>
        /// <param name="e">The tree.</param><param name="env">Bound symbols.</param><returns>The value.</returns>
        /// <exception cref="InvalidOperationException">A bare weak constant (it has no dtype) or an unbound symbol.</exception>
        public TVal Emit(E e, Env env)
        {
            switch (e)
            {
                case Leaf l: return env.Get(l.S);
                case Bn b:
                {
                    var t = LoopType(b, TypeOf(b.A, s => env.Get(s).T), TypeOf(b.B, s => env.Get(s).T));
                    var va = EmitAt(b.A, env, t);
                    var vb = EmitAt(b.B, env, t);
                    return Op(b.Op, va, vb, t);
                }
                default: throw new InvalidOperationException("a weak constant is typed by its partner operand");
            }
        }

        /// <summary>Emits a tree converted to <paramref name="t"/> (a weak constant is materialized in it).</summary>
        /// <param name="e">The tree.</param><param name="env">Bound symbols.</param><param name="t">Target dtype.</param><returns>The value.</returns>
        public TVal EmitAt(E e, Env env, NPTypeCode t) => e is Wk w ? Const(w.W, t) : Convert(Emit(e, env), t);

        /// <summary>Converts a value (NumPy's loop-input cast): a shared scalar through the house EmitConvertTo,
        /// a per-point vector through the exact lane conversion, a scalar chain through EmitConvertTo.</summary>
        /// <param name="v">The value.</param><param name="t">Target dtype.</param><returns>The converted value (or v).</returns>
        public TVal Convert(TVal v, NPTypeCode t)
        {
            if (v.T == t) return v;
            var r = Fresh(t, v.Shared);
            int n = v.Shared ? 1 : U;
            for (int u = 0; u < n; u++)
            {
                IL.Emit(OpCodes.Ldloc, v.L[u]);
                if (Vec && !v.Shared) EmitLaneConvert(IL, v.T, t);
                else DirectILKernelGenerator.EmitConvertTo(IL, v.T, t);
                IL.Emit(OpCodes.Stloc, r.L[u]);
            }
            return r;
        }

        /// <summary>Loads a weak constant from its pool region as a shared scalar (row = current row for
        /// per-row constants).</summary>
        /// <param name="w">The constant.</param><param name="t">Its dtype here.</param><returns>A shared value.</returns>
        TVal Const(WeakConst w, NPTypeCode t)
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
        /// operand spread once.</summary>
        /// <param name="op">The op.</param><param name="a">Left (already in t).</param><param name="b">Right (already in t).</param>
        /// <param name="t">Loop dtype.</param><returns>The result.</returns>
        TVal Op(BinaryOp op, TVal a, TVal b, NPTypeCode t)
        {
            if (a.Shared && b.Shared)
            {
                var s = Fresh(t, true);
                IL.Emit(OpCodes.Ldloc, a.L[0]); IL.Emit(OpCodes.Ldloc, b.L[0]);
                Scalar(t).Bin(IL, op);
                IL.Emit(OpCodes.Stloc, s.L[0]);
                return s;
            }
            var sa = a.Shared ? Spread(a) : null;
            var sb = b.Shared ? Spread(b) : null;
            var r = Fresh(t, false);
            var k = Lane(t);
            for (int u = 0; u < U; u++)
            {
                IL.Emit(OpCodes.Ldloc, sa ?? a.L[u]);
                IL.Emit(OpCodes.Ldloc, sb ?? b.L[u]);
                k.Bin(IL, op);
                IL.Emit(OpCodes.Stloc, r.L[u]);
            }
            return r;
        }
    }

    // ======================================================================== evaluation kernel (Tier 3A)

    /// <summary>A compiled evaluation kernel plus what its caller needs to run it.</summary>
    sealed class EvalKernel
    {
        /// <summary>The Tier-3A inner loop.</summary>
        public NDInnerLoopFunc Fn;
        /// <summary>Its weak-constant regions.</summary>
        public ConstPool Pool;
        /// <summary>NumPy's result dtype for this (x dtype, c dtype, coefficient class).</summary>
        public NPTypeCode Tl;
    }

    /// <summary>Locals and static facts shared by every block of one evaluation kernel.</summary>
    sealed class EvalCtx
    {
        /// <summary>The basis's step table.</summary>
        public EvalProgram Prog;
        /// <summary>x, coefficient and result dtypes.</summary>
        public NPTypeCode Tx, Tc, Tl;
        /// <summary>Coefficient-count class and peel count (see <see cref="PeelCount"/>).</summary>
        public int Cls, P;
        /// <summary>The kernel's constant pool.</summary>
        public ConstPool Pool;
        /// <summary>Chunk pointers (x, coefficient base, y) and the region table.</summary>
        public LocalBuilder Xp, Cp, Yp, Table;
        /// <summary>Chunk byte strides (x, coefficients, y), the series-axis byte stride and nc.</summary>
        public LocalBuilder Sx, Sc, Sy, KStride, Nc;
        /// <summary>The current recurrence row (nd) and the element index within the chunk.</summary>
        public LocalBuilder Row, I;
        /// <summary>
        ///     1-D coefficients of a dtype other than the loop's: the steady loop reads a copy PRE-CONVERTED to the
        ///     loop dtype (auxdata [4] base, [5] byte stride). Exact: in the steady state NumPy promotes <c>c[k]</c>
        ///     to the loop dtype inside the op, a widening conversion; only the peeled steps must see the
        ///     original dtype. Saves one scalar conversion per coefficient per block (a float16 series is a
        ///     software Half-to-float call each time).
        /// </summary>
        public bool Preconv;
        /// <summary>The pre-converted copy's base and stride (when <see cref="Preconv"/>).</summary>
        public LocalBuilder CpL, KStrideL;
    }

    /// <summary>How the coefficients of one block are read: broadcast (every point of the chunk shares a
    /// series — 1-D c, or a multi-series c whose series is constant along the chunk) or per lane
    /// (tensor=False / _valnd: every point has its own series).</summary>
    sealed class CoefCtx
    {
        /// <summary>One series for the whole chunk.</summary>
        public bool Broadcast;
        /// <summary>The chunk's coefficient pointer (the series of its first point).</summary>
        public LocalBuilder Cp;
        /// <summary>Per-lane mode: each chain's series base for the current block iteration.</summary>
        public LocalBuilder[] CBase;
        /// <summary>The coefficient dtype.</summary>
        public NPTypeCode Tc;
    }

    static readonly Dictionary<(Basis, NPTypeCode, NPTypeCode), int> s_peel = new();
    static readonly Dictionary<(Basis, NPTypeCode, NPTypeCode, int, bool, int), EvalKernel> s_eval = new();

    /// <summary>Returns (compiling once) the evaluation kernel for a basis, dtype pair and coefficient count.</summary>
    /// <param name="b">Basis.</param><param name="tx">x dtype.</param><param name="tc">Coefficient dtype.</param>
    /// <param name="nc">Coefficient count (only its class matters: 1, 2, the straight-line counts, or "loop").</param>
    /// <param name="coefOperand">Coefficients come from an NDIter operand (N-D c) instead of auxdata (1-D c).</param>
    /// <param name="unroll">Interleaved chains.</param><returns>The kernel.</returns>
    static EvalKernel GetEvalKernel(Basis b, NPTypeCode tx, NPTypeCode tc, int nc, bool coefOperand, int unroll)
    {
        var prog = Steps.Eval[b];
        if (!s_peel.TryGetValue((b, tx, tc), out int P)) s_peel[(b, tx, tc)] = P = PeelCount(prog, tx, tc);
        int cls = prog.Horner ? 1 : Math.Min(nc, P + 3);
        var key = (b, tx, tc, cls, coefOperand, unroll);
        if (s_eval.TryGetValue(key, out var k)) return k;
        var pool = new ConstPool();
        var tl = ResultType(prog, tx, tc, cls, P);
        var ctx = new EvalCtx { Prog = prog, Tx = tx, Tc = tc, Tl = tl, Cls = cls, P = P, Pool = pool, Preconv = !coefOperand && tc != tl };
        string name = $"polyprobe_eval_{b}_{tx}_{tc}_{cls}_{(coefOperand ? "op" : "aux")}_u{unroll}";
        var fn = DirectILKernelGenerator.CompileRawInnerLoop(il => EmitEvalKernel(il, ctx, coefOperand, unroll), name);
        return s_eval[key] = new EvalKernel { Fn = fn, Pool = pool, Tl = tl };
    }

    /// <summary>
    ///     The Tier-3A body. auxdata: [0] coefficient byte stride along the series axis, [1] nc, [2] region
    ///     table, [3] coefficient base (auxdata form only). Operands: (x, y) or (x, c[0]-view, y).
    ///     Dispatch happens ONCE per chunk: vector chains when x and y are contiguous and the coefficients
    ///     are broadcast or contiguous per lane; interleaved scalar chains otherwise (strided x, mixed
    ///     dtypes, complex, decimal).
    /// </summary>
    /// <param name="il">Generator.</param><param name="ctx">Static facts (locals are declared here).</param>
    /// <param name="coefOperand">Operand form.</param><param name="unroll">Chains.</param>
    static void EmitEvalKernel(ILGenerator il, EvalCtx ctx, bool coefOperand, int unroll)
    {
        LocalBuilder Ptr() => il.DeclareLocal(typeof(byte*));
        LocalBuilder Lng() => il.DeclareLocal(typeof(long));
        ctx.Xp = Ptr(); ctx.Cp = Ptr(); ctx.Yp = Ptr(); ctx.Table = Ptr();
        ctx.Sx = Lng(); ctx.Sc = Lng(); ctx.Sy = Lng(); ctx.KStride = Lng(); ctx.Nc = Lng(); ctx.Row = Lng(); ctx.I = Lng();
        LdSlot(il, 0, 0, ctx.Xp); LdSlot(il, 1, 0, ctx.Sx);
        if (coefOperand)
        {
            LdSlot(il, 0, 1, ctx.Cp); LdSlot(il, 1, 1, ctx.Sc);
            LdSlot(il, 0, 2, ctx.Yp); LdSlot(il, 1, 2, ctx.Sy);
        }
        else
        {
            LdSlot(il, 3, 3, ctx.Cp); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, ctx.Sc);
            LdSlot(il, 0, 1, ctx.Yp); LdSlot(il, 1, 1, ctx.Sy);
        }
        LdSlot(il, 3, 0, ctx.KStride); LdSlot(il, 3, 1, ctx.Nc); LdSlot(il, 3, 2, ctx.Table);
        if (ctx.Preconv)
        {
            ctx.CpL = Ptr(); ctx.KStrideL = Lng();
            LdSlot(il, 3, 4, ctx.CpL); LdSlot(il, 3, 5, ctx.KStrideL);
        }
        il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, ctx.I);

        int szX = DirectILKernelGenerator.GetTypeSize(ctx.Tx), szC = DirectILKernelGenerator.GetTypeSize(ctx.Tc),
            szL = DirectILKernelGenerator.GetTypeSize(ctx.Tl);
        var end = il.DefineLabel(); var scalarDispatch = il.DefineLabel();
        bool vecB = VectorBlockOk(ctx.Tx, ctx.Tc, ctx.Tl, perLaneCoef: false);
        bool vecL = coefOperand && VectorBlockOk(ctx.Tx, ctx.Tc, ctx.Tl, perLaneCoef: true);
        if (vecB || vecL)
        {
            var lblB = il.DefineLabel(); var lblL = il.DefineLabel();
            // Vector chains need x and y contiguous in this chunk; the coefficient stride picks broadcast
            // (0: one series for the chunk) or per-lane (contiguous: one series per point).
            il.Emit(OpCodes.Ldloc, ctx.Sx); il.Emit(OpCodes.Ldc_I8, (long)szX); il.Emit(OpCodes.Bne_Un, scalarDispatch);
            il.Emit(OpCodes.Ldloc, ctx.Sy); il.Emit(OpCodes.Ldc_I8, (long)szL); il.Emit(OpCodes.Bne_Un, scalarDispatch);
            il.Emit(OpCodes.Ldloc, ctx.Sc); il.Emit(OpCodes.Brfalse, vecB ? lblB : scalarDispatch);
            if (vecL) { il.Emit(OpCodes.Ldloc, ctx.Sc); il.Emit(OpCodes.Ldc_I8, (long)szC); il.Emit(OpCodes.Beq, lblL); }
            il.Emit(OpCodes.Br, scalarDispatch);
            foreach (bool bcast in new[] { true, false })
            {
                if (bcast ? !vecB : !vecL) continue;
                il.MarkLabel(bcast ? lblB : lblL);
                var l1 = il.DefineLabel(); var tail = il.DefineLabel();
                if (unroll > 1) EmitEvalBlock(il, ctx, vec: true, bcast, unroll, l1);
                il.MarkLabel(l1);
                EmitEvalBlock(il, ctx, vec: true, bcast, 1, tail);
                il.MarkLabel(tail);
                EmitEvalBlock(il, ctx, vec: false, bcast, 1, end);
                il.Emit(OpCodes.Br, end);
            }
            if (!vecB) il.MarkLabel(lblB);   // never branched to; marked so every defined label is placed
            if (!vecL) il.MarkLabel(lblL);
        }
        il.MarkLabel(scalarDispatch);
        var scalarL = il.DefineLabel();
        il.Emit(OpCodes.Ldloc, ctx.Sc); il.Emit(OpCodes.Brtrue, scalarL);
        foreach (bool bcast in new[] { true, false })
        {
            if (!bcast) il.MarkLabel(scalarL);
            var l1 = il.DefineLabel();
            if (unroll > 1) EmitEvalBlock(il, ctx, vec: false, bcast, unroll, l1);
            il.MarkLabel(l1);
            EmitEvalBlock(il, ctx, vec: false, bcast, 1, end);
            il.Emit(OpCodes.Br, end);
        }
        il.MarkLabel(end);
        il.Emit(OpCodes.Ret);
    }

    /// <summary>dst = ((long*)arg)[slot] — reads a dataptr, stride or auxdata slot into a local.</summary>
    /// <param name="il">Generator.</param><param name="arg">0 dataptrs, 1 strides, 3 auxdata.</param>
    /// <param name="slot">Slot index.</param><param name="dst">Pointer or long local.</param>
    static void LdSlot(ILGenerator il, int arg, int slot, LocalBuilder dst)
    {
        il.Emit(arg switch { 0 => OpCodes.Ldarg_0, 1 => OpCodes.Ldarg_1, 2 => OpCodes.Ldarg_2, _ => OpCodes.Ldarg_3 });
        il.Emit(OpCodes.Ldc_I4, slot * 8); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Ldind_I8);
        if (dst.LocalType == typeof(byte*)) il.Emit(OpCodes.Conv_U);
        il.Emit(OpCodes.Stloc, dst);
    }

    /// <summary>Pushes <c>basePtr + (i + laneOffset) * stride</c> (stride from a local, or a constant when the
    /// block runs on proven-contiguous data).</summary>
    /// <param name="il">Generator.</param><param name="basePtr">Base pointer local.</param><param name="i">Element index local.</param>
    /// <param name="laneOffset">Element offset of the chain.</param><param name="strideLocal">Runtime stride, or null.</param>
    /// <param name="strideConst">Constant stride when <paramref name="strideLocal"/> is null.</param>
    static void Addr(ILGenerator il, LocalBuilder basePtr, LocalBuilder i, long laneOffset, LocalBuilder strideLocal, long strideConst)
    {
        il.Emit(OpCodes.Ldloc, basePtr);
        il.Emit(OpCodes.Ldloc, i);
        if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
        if (strideLocal is not null) il.Emit(OpCodes.Ldloc, strideLocal); else il.Emit(OpCodes.Ldc_I8, strideConst);
        il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
    }

    /// <summary>One loop block: while U more lane groups fit, load U x values, run the recurrence as U
    /// interleaved chains, store U results.</summary>
    /// <param name="il">Generator.</param><param name="ctx">Kernel facts.</param><param name="vec">Vector chains.</param>
    /// <param name="bcast">Broadcast coefficients.</param><param name="U">Chains.</param><param name="exit">Where to go when fewer remain.</param>
    static void EmitEvalBlock(ILGenerator il, EvalCtx ctx, bool vec, bool bcast, int U, Label exit)
    {
        var em = new Emitter(il, U, vec, LoopLanes(ctx.Tl), ctx.Pool, ctx.Table, ctx.Row);
        int W = em.W;
        int szX = DirectILKernelGenerator.GetTypeSize(ctx.Tx), szC = DirectILKernelGenerator.GetTypeSize(ctx.Tc),
            szL = DirectILKernelGenerator.GetTypeSize(ctx.Tl);
        var top = il.DefineLabel();
        il.MarkLabel(top);
        il.Emit(OpCodes.Ldloc, ctx.I); il.Emit(OpCodes.Ldc_I8, (long)(U * W)); il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bgt, exit);

        var cc = new CoefCtx { Broadcast = bcast, Cp = ctx.Cp, Tc = ctx.Tc };
        if (!bcast)
        {
            cc.CBase = new LocalBuilder[U];
            for (int u = 0; u < U; u++)
            {
                cc.CBase[u] = il.DeclareLocal(typeof(byte*));
                Addr(il, ctx.Cp, ctx.I, u * W, vec ? null : ctx.Sc, szC);
                il.Emit(OpCodes.Stloc, cc.CBase[u]);
            }
        }
        var X = em.Fresh(ctx.Tx, false);
        for (int u = 0; u < U; u++)
        {
            Addr(il, ctx.Xp, ctx.I, u * W, vec ? null : ctx.Sx, szX);
            em.Lane(ctx.Tx).Load(il);
            il.Emit(OpCodes.Stloc, X.L[u]);
        }
        // The steady loop's coefficient source: the pre-converted copy when there is one (broadcast form only).
        var ccLoop = ctx.Preconv && bcast ? new CoefCtx { Broadcast = true, Cp = ctx.CpL, Tc = ctx.Tl } : cc;
        var loopStride = ctx.Preconv && bcast ? ctx.KStrideL : ctx.KStride;
        var y = ctx.Prog.Horner ? EmitHorner(em, ctx, cc, ccLoop, loopStride, X) : EmitClenshaw(em, ctx, cc, ccLoop, loopStride, X);
        if (y.T != ctx.Tl) throw new InvalidOperationException($"planned {ctx.Tl}, emitted {y.T}");
        if (y.Shared) { var yl = em.Fresh(ctx.Tl, false); em.CopyTo(y, yl); y = yl; }   // a result independent of x (never for these tables)
        for (int u = 0; u < U; u++)
        {
            il.Emit(OpCodes.Ldloc, y.L[u]);
            Addr(il, ctx.Yp, ctx.I, u * W, vec ? null : ctx.Sy, szL);
            em.Lane(ctx.Tl).StoreValueFirst(il);
        }
        il.Emit(OpCodes.Ldloc, ctx.I); il.Emit(OpCodes.Ldc_I8, (long)(U * W)); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, ctx.I);
        il.Emit(OpCodes.Br, top);
    }

    /// <summary>Loads coefficient <c>c[k]</c> (k given as a byte offset along the series axis): one SHARED
    /// scalar (broadcast series — it is spread only where it meets a per-point value), or one lane load per
    /// chain from that chain's own series.</summary>
    /// <param name="em">Emitter.</param><param name="cc">Coefficient context.</param><param name="kOff">Byte offset local.</param>
    /// <returns>The coefficient value (dtype = coefficient dtype).</returns>
    static TVal LoadCoef(Emitter em, CoefCtx cc, LocalBuilder kOff)
    {
        var il = em.IL;
        if (cc.Broadcast)
        {
            var r = em.Fresh(cc.Tc, true);
            il.Emit(OpCodes.Ldloc, cc.Cp); il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            em.Scalar(cc.Tc).Load(il);
            il.Emit(OpCodes.Stloc, r.L[0]);
            return r;
        }
        var k = em.Lane(cc.Tc);
        var v = em.Fresh(cc.Tc, false);
        for (int u = 0; u < em.U; u++)
        {
            il.Emit(OpCodes.Ldloc, cc.CBase[u]); il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            k.Load(il);
            il.Emit(OpCodes.Stloc, v.L[u]);
        }
        return v;
    }

    /// <summary>kOff = (nc - fromEnd) * kstride.</summary>
    /// <param name="il">Generator.</param><param name="ctx">Kernel facts.</param><param name="kOff">Destination.</param><param name="fromEnd">Distance from the end.</param>
    static void SetKOffFromEnd(ILGenerator il, EvalCtx ctx, LocalBuilder kOff, long fromEnd)
    {
        il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, fromEnd); il.Emit(OpCodes.Sub);
        il.Emit(OpCodes.Ldloc, ctx.KStride); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, kOff);
    }

    /// <summary>
    ///     Clenshaw for one block, NumPy's <c>{p}val</c> body: classes 1 and 2 are NumPy's
    ///     <c>len(c) == 1 / 2</c> branches; otherwise the first <c>min(nc-2, P)</c> steps are straight-line
    ///     (their dtypes differ) and the rest run as an IL loop at the fixpoint dtypes.
    /// </summary>
    /// <param name="em">Emitter.</param><param name="ctx">Kernel facts.</param><param name="cc">Coefficient context.</param>
    /// <param name="X">The loaded x values.</param><returns>The result value.</returns>
    /// <exception cref="InvalidOperationException">The loop body changed a carried dtype (the peel count is wrong).</exception>
    static TVal EmitClenshaw(Emitter em, EvalCtx ctx, CoefCtx cc, CoefCtx ccLoop, LocalBuilder loopStride, TVal X)
    {
        var il = em.IL; var prog = ctx.Prog; var env = new Env { X = X };
        var kOff = il.DeclareLocal(typeof(long));
        if (ctx.Cls <= 2)
        {
            // len(c) == 1: c0 = c[0], c1 = 0 (a Python int); len(c) == 2: c0 = c[0], c1 = c[1].
            var final = ctx.Cls == 1 ? Steps.Subst(prog.Final, Sym.C1, Steps.W(Steps.I0)) : prog.Final;
            if (prog.Pre is not null && Steps.Uses(final, Sym.X2)) env.X2 = em.Emit(prog.Pre, env);   // hermval: x2 always
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, kOff);
            env.C0 = LoadCoef(em, cc, kOff);
            if (ctx.Cls == 2)
            {
                il.Emit(OpCodes.Ldloc, ctx.KStride); il.Emit(OpCodes.Stloc, kOff);
                env.C1 = LoadCoef(em, cc, kOff);
            }
            return em.Emit(final, env);
        }
        if (prog.Pre is not null) env.X2 = em.Emit(prog.Pre, env);
        bool usesRow = Steps.UsesRow(prog.StepC0) || Steps.UsesRow(prog.StepC1);
        SetKOffFromEnd(il, ctx, kOff, 2); var c0 = LoadCoef(em, cc, kOff);
        SetKOffFromEnd(il, ctx, kOff, 1); var c1 = LoadCoef(em, cc, kOff);
        int straight = ctx.Cls <= ctx.P + 2 ? ctx.Cls - 2 : ctx.P;
        for (int p = 0; p < straight; p++)
        {
            // Step p reads c[nc-3-p] with nd = nc-1-p (NumPy's nd = nd - 1 before the step's lines).
            SetKOffFromEnd(il, ctx, kOff, 3 + p);
            if (usesRow) { il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, 1L + p); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, ctx.Row); }
            env.Ck = LoadCoef(em, cc, kOff); env.Tmp = c0; env.C1 = c1;
            var n0 = em.Emit(prog.StepC0, env);
            var n1 = em.Emit(prog.StepC1, env);
            c0 = n0; c1 = n1;
        }
        if (ctx.Cls > ctx.P + 2)
        {
            var s0 = em.Fresh(c0.T, false); var s1 = em.Fresh(c1.T, false);
            em.CopyTo(c0, s0); em.CopyTo(c1, s1);
            var k = il.DeclareLocal(typeof(long));
            var top = il.DefineLabel(); var done = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, 3L + ctx.P); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
            // kOff walks down by one stride per step (no multiply in the loop).
            il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, kOff);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Blt, done);
            if (usesRow) { il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, ctx.Row); }
            env.Ck = LoadCoef(em, ccLoop, kOff); env.Tmp = s0; env.C1 = s1;
            var n0 = em.Emit(prog.StepC0, env);
            var n1 = em.Emit(prog.StepC1, env);
            if (n0.T != s0.T || n1.T != s1.T) throw new InvalidOperationException("loop body changed a carried dtype: peel count wrong");
            // tmp was read above; only now overwrite the carried pair.
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
    /// <param name="em">Emitter.</param><param name="ctx">Kernel facts.</param><param name="cc">Coefficient context.</param>
    /// <param name="X">The loaded x values.</param><returns>The result value.</returns>
    /// <exception cref="InvalidOperationException">The step changed the carried dtype (cannot happen: promotion is idempotent here).</exception>
    static TVal EmitHorner(Emitter em, EvalCtx ctx, CoefCtx cc, CoefCtx ccLoop, LocalBuilder loopStride, TVal X)
    {
        var il = em.IL; var prog = ctx.Prog; var env = new Env { X = X };
        var kOff = il.DeclareLocal(typeof(long));
        SetKOffFromEnd(il, ctx, kOff, 1);
        env.Ck = LoadCoef(em, cc, kOff);
        var c0 = em.Emit(prog.HornerInit, env);
        var s0 = em.Fresh(c0.T, false); em.CopyTo(c0, s0);
        var k = il.DeclareLocal(typeof(long));
        var top = il.DefineLabel(); var done = il.DefineLabel();
        il.Emit(OpCodes.Ldloc, ctx.Nc); il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
        il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Stloc, kOff);
        il.MarkLabel(top);
        il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Blt, done);
        env.Ck = LoadCoef(em, ccLoop, kOff); env.C0 = s0;
        var n0 = em.Emit(prog.HornerStep, env);
        if (n0.T != s0.T) throw new InvalidOperationException("Horner step changed the carried dtype");
        em.CopyTo(n0, s0);
        il.Emit(OpCodes.Ldloc, k); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, k);
        il.Emit(OpCodes.Ldloc, kOff); il.Emit(OpCodes.Ldloc, loopStride); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, kOff);
        il.Emit(OpCodes.Br, top);
        il.MarkLabel(done);
        return s0;
    }

    // ======================================================================== Vandermonde kernel (Tier 3A)

    /// <summary>A compiled Vandermonde kernel.</summary>
    sealed class VanderKernel
    {
        /// <summary>The Tier-3A inner loop.</summary>
        public NDInnerLoopFunc Fn;
        /// <summary>Its weak-constant regions.</summary>
        public ConstPool Pool;
        /// <summary>The output dtype (<c>result_type(x, 0.0)</c>: float64 for bool/int x).</summary>
        public NPTypeCode Tv;
    }

    static readonly Dictionary<(Basis, NPTypeCode, int), VanderKernel> s_vander = new();

    /// <summary>Returns (compiling once) the Vandermonde kernel for a basis and x dtype.</summary>
    /// <param name="b">Basis.</param><param name="tx">x dtype.</param><param name="unroll">Chains.</param><returns>The kernel.</returns>
    static VanderKernel GetVanderKernel(Basis b, NPTypeCode tx, int unroll)
    {
        if (s_vander.TryGetValue((b, tx, unroll), out var k)) return k;
        var pool = new ConstPool();
        NPTypeCode tv = WeakPromote(tx, isFloat: true);
        string name = $"polyprobe_vander_{b}_{tx}_u{unroll}";
        var fn = DirectILKernelGenerator.CompileRawInnerLoop(il => EmitVanderKernel(il, Steps.Vander[b], tx, tv, pool, unroll), name);
        return s_vander[(b, tx, unroll)] = new VanderKernel { Fn = fn, Pool = pool, Tv = tv };
    }

    /// <summary>
    ///     Vandermonde body. Operands (x, v[0]-view); auxdata: [0] row byte stride of the C-contiguous
    ///     <c>(deg+1, *x.shape)</c> buffer, [1] deg, [2] region table. Each point is read once and each of
    ///     its deg+1 outputs written once (NumPy re-reads two full rows per row). Row 0 is <c>x*0 + 1</c>
    ///     on the <c>x + 0.0</c> copy NumPy makes, so <c>-0.0</c> becomes <c>+0.0</c> exactly as in NumPy.
    /// </summary>
    /// <param name="il">Generator.</param><param name="prog">Step table.</param><param name="tx">x dtype.</param>
    /// <param name="tv">Output dtype.</param><param name="pool">Constant pool.</param><param name="unroll">Chains.</param>
    static void EmitVanderKernel(ILGenerator il, VanderProgram prog, NPTypeCode tx, NPTypeCode tv, ConstPool pool, int unroll)
    {
        var xp = il.DeclareLocal(typeof(byte*)); var vp = il.DeclareLocal(typeof(byte*)); var table = il.DeclareLocal(typeof(byte*));
        var sx = il.DeclareLocal(typeof(long)); var sv = il.DeclareLocal(typeof(long)); var rowStride = il.DeclareLocal(typeof(long));
        var deg = il.DeclareLocal(typeof(long)); var row = il.DeclareLocal(typeof(long)); var i = il.DeclareLocal(typeof(long));
        LdSlot(il, 0, 0, xp); LdSlot(il, 1, 0, sx); LdSlot(il, 0, 1, vp); LdSlot(il, 1, 1, sv);
        LdSlot(il, 3, 0, rowStride); LdSlot(il, 3, 1, deg); LdSlot(il, 3, 2, table);
        il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);
        int szX = DirectILKernelGenerator.GetTypeSize(tx), szV = DirectILKernelGenerator.GetTypeSize(tv);
        var end = il.DefineLabel(); var scalar = il.DefineLabel();

        void Block(bool vec, int U, Label exit)
        {
            var em = new Emitter(il, U, vec, LoopLanes(tv), pool, table, row);
            int W = em.W;
            var top = il.DefineLabel();
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, (long)(U * W)); il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bgt, exit);
            var X = em.Fresh(tx, false);
            for (int u = 0; u < U; u++)
            {
                Addr(il, xp, i, u * W, vec ? null : sx, szX);
                em.Lane(tx).Load(il);
                il.Emit(OpCodes.Stloc, X.L[u]);
            }
            var env = new Env { X = X };
            env.X = em.Emit(Steps.Add(Steps.X, Steps.W(Steps.F0)), env);           // x = x + 0.0
            var rowOff = il.DeclareLocal(typeof(long));
            void Store(TVal v)
            {
                if (v.T != tv) throw new InvalidOperationException($"vander row typed {v.T}, expected {tv}");
                if (v.Shared) { var vl = em.Fresh(tv, false); em.CopyTo(v, vl); v = vl; }
                for (int u = 0; u < U; u++)
                {
                    il.Emit(OpCodes.Ldloc, v.L[u]);
                    Addr(il, vp, i, u * W, vec ? null : sv, szV);
                    il.Emit(OpCodes.Ldloc, rowOff); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    em.Lane(tv).StoreValueFirst(il);
                }
            }
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, rowOff);
            var v0 = em.Emit(Steps.Add(Steps.Mul(Steps.X, Steps.W(Steps.I0)), Steps.W(Steps.I1)), env);   // v[0] = x*0 + 1
            Store(v0);
            var next = il.DefineLabel();
            // deg == 0: only row 0 exists (NumPy's `if ideg > 0:`).
            il.Emit(OpCodes.Ldloc, deg); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Blt, next);
            if (prog.Pre is not null) env.X2 = em.Emit(prog.Pre, env);
            il.Emit(OpCodes.Ldloc, rowStride); il.Emit(OpCodes.Stloc, rowOff);
            var v1 = em.Emit(prog.V1, env);
            Store(v1);
            var s2 = em.Fresh(tv, false); var s1 = em.Fresh(tv, false);
            em.CopyTo(v0, s2); em.CopyTo(v1, s1);
            var loop = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 2L); il.Emit(OpCodes.Stloc, row);
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, row); il.Emit(OpCodes.Ldloc, deg); il.Emit(OpCodes.Bgt, next);
            il.Emit(OpCodes.Ldloc, rowOff); il.Emit(OpCodes.Ldloc, rowStride); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, rowOff);
            env.Vm1 = s1; env.Vm2 = s2;
            var vi = em.Emit(prog.Step, env);
            Store(vi);
            em.CopyTo(s1, s2); em.CopyTo(vi, s1);
            il.Emit(OpCodes.Ldloc, row); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, row);
            il.Emit(OpCodes.Br, loop);
            il.MarkLabel(next);
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, (long)(U * W)); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Br, top);
        }

        // Vector chains when x converts lane-for-lane into the output dtype (int64/int32/float32 x -> float64
        // included); the x + 0.0 conversion is then the lane conversion itself.
        if (VectorBlockOk(tx, tv, tv, perLaneCoef: false))
        {
            il.Emit(OpCodes.Ldloc, sx); il.Emit(OpCodes.Ldc_I8, (long)szX); il.Emit(OpCodes.Bne_Un, scalar);
            il.Emit(OpCodes.Ldloc, sv); il.Emit(OpCodes.Ldc_I8, (long)szV); il.Emit(OpCodes.Bne_Un, scalar);
            var l1 = il.DefineLabel(); var tail = il.DefineLabel();
            if (unroll > 1) Block(true, unroll, l1);
            il.MarkLabel(l1);
            Block(true, 1, tail);
            il.MarkLabel(tail);
            Block(false, 1, end);
            il.Emit(OpCodes.Br, end);
        }
        il.MarkLabel(scalar);
        var s1l = il.DefineLabel();
        if (unroll > 1) Block(false, unroll, s1l);
        il.MarkLabel(s1l);
        Block(false, 1, end);
        il.MarkLabel(end);
        il.Emit(OpCodes.Ret);
    }

    // ======================================================================== NDIter drivers (NumPy's Python layer)

    /// <summary>
    ///     NumPy's <c>{p}val(x, c, tensor)</c> over the IL kernel. 1-D c: operands (x, y) with the
    ///     coefficients in auxdata (safe to BUFFER a strided x). N-D c: NumPy's tensor reshape, then
    ///     (x, c[0]-view, y) with NDIter broadcasting — NEVER buffered, because the kernel reads
    ///     <c>c[0] + k*kstride</c>, beyond the operand a buffer would copy.
    /// </summary>
    /// <param name="b">Basis.</param><param name="x">Points (any layout/dtype).</param><param name="c">Coefficients, series along axis 0.</param>
    /// <param name="tensor">NumPy's tensor flag.</param><param name="unroll">Chains.</param>
    /// <param name="buffered">BUFFERED+CONTIG on x (1-D c only).</param><param name="forceOperand">Use the operand form for a 1-D c (overhead measurement).</param>
    /// <returns>The values, shape <c>c.shape[1:] + x.shape</c> (tensor) or the broadcast shape.</returns>
    static NDArray Val(Basis b, NDArray x, NDArray c, bool tensor = true, int unroll = 4, bool buffered = false, bool forceOperand = false)
    {
        NDArray own = null, ownL = null;
        if (IsIntLike(c.typecode)) c = own = c.astype(NPTypeCode.Double);   // chebval: astype(double); polyval: c + 0.0 — same values
        try
        {
            int nc = (int)c.shape[0];
            bool operand = forceOperand || c.ndim > 1;
            var k = GetEvalKernel(b, x.typecode, c.typecode, nc, operand, unroll);
            long* aux = stackalloc long[6];
            aux[0] = c.strides[0];
            aux[1] = nc;
            aux[2] = k.Pool.TableFor(nc);
            if (!operand)
            {
                aux[3] = (long)((byte*)c.Storage.Address + c.Shape.offset * c.dtypesize);
                if (c.typecode != k.Tl)
                {
                    // Steady-loop copy in the loop dtype (see EvalCtx.Preconv): nc elements, NumPy's exact widening.
                    ownL = c.astype(k.Tl);
                    aux[4] = (long)((byte*)ownL.Storage.Address + ownL.Shape.offset * ownL.dtypesize);
                    aux[5] = ownL.strides[0];
                }
                var y = new NDArray(k.Tl, new Shape(x.shape), false);   // dims only: never x.Shape (plan R8)
                var gflags = NDIterGlobalFlags.EXTERNAL_LOOP;
                var xflags = NDIterPerOpFlags.READONLY;
                if (buffered) { gflags |= NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER; xflags |= NDIterPerOpFlags.CONTIG; }
                using var iter = NDIterRef.MultiNew(2, new[] { x, y }, gflags, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                    new[] { xflags, NDIterPerOpFlags.WRITEONLY }, buffered ? new[] { x.typecode, k.Tl } : null);
                iter.ForEach(k.Fn, aux);
                return y;
            }
            var cr = c;
            if (tensor)
            {
                var dims = new long[c.ndim + x.ndim];
                c.shape.CopyTo(dims, 0);
                for (int d = c.ndim; d < dims.Length; d++) dims[d] = 1;
                cr = c.reshape(new Shape(dims));
            }
            var c0 = cr[0];
            var outShape = BroadcastDims(c0.shape, x.shape);
            var y2 = new NDArray(k.Tl, new Shape(outShape), false);
            using (var iter = NDIterRef.MultiNew(3, new[] { x, c0, y2 }, NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                       NPY_CASTING.NPY_SAFE_CASTING, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null))
                iter.ForEach(k.Fn, aux);
            return y2;
        }
        finally { own?.Dispose(); ownL?.Dispose(); }
    }

    /// <summary>NumPy's right-aligned broadcast of two shapes.</summary>
    /// <param name="a">First shape.</param><param name="b">Second shape.</param><returns>The broadcast shape.</returns>
    /// <exception cref="ArgumentException">Incompatible shapes.</exception>
    static long[] BroadcastDims(long[] a, long[] b)
    {
        int n = Math.Max(a.Length, b.Length);
        var r = new long[n];
        for (int d = 0; d < n; d++)
        {
            long da = d < n - a.Length ? 1 : a[d - (n - a.Length)], db = d < n - b.Length ? 1 : b[d - (n - b.Length)];
            if (da != db && da != 1 && db != 1) throw new ArgumentException("operands could not be broadcast together");
            r[d] = da == 1 ? db : da;
        }
        return r;
    }

    /// <summary>NumPy's <c>{p}val2d</c> = <c>pu._valnd</c>: tensor evaluation over x, then per-point
    /// (tensor=False) evaluation over y. Two kernel passes, NumPy's exact intermediate.</summary>
    /// <param name="b">Basis.</param><param name="x">First ordinate.</param><param name="y">Second ordinate (same shape).</param>
    /// <param name="c">2-D coefficients.</param><returns>The values, shape x.shape.</returns>
    /// <exception cref="ArgumentException">x and y differ in shape (NumPy: <c>ValueError: x, y are incompatible</c>).</exception>
    static NDArray Val2d(Basis b, NDArray x, NDArray y, NDArray c)
    {
        if (!x.shape.SequenceEqual(y.shape)) throw new ArgumentException("x, y are incompatible");
        using var c1 = Val(b, x, c);
        return Val(b, y, c1, tensor: false);
    }

    /// <summary>NumPy's <c>{p}grid2d</c> = <c>pu._gridnd</c>: two tensor evaluations.</summary>
    /// <param name="b">Basis.</param><param name="x">First axis points.</param><param name="y">Second axis points.</param>
    /// <param name="c">2-D coefficients.</param><returns>The grid, shape x.shape + y.shape.</returns>
    static NDArray Grid2d(Basis b, NDArray x, NDArray y, NDArray c)
    {
        using var c1 = Val(b, x, c);
        return Val(b, y, c1);
    }

    /// <summary>
    ///     NumPy's <c>{p}vander(x, deg)</c>: fills a C-contiguous <c>(deg+1, *x.shape)</c> buffer through
    ///     NDIter over (x, v[0]) and returns <c>np.moveaxis(v, 0, -1)</c> — NumPy's exact layout.
    /// </summary>
    /// <param name="b">Basis.</param><param name="x">Points.</param><param name="deg">Degree (≥ 0).</param>
    /// <param name="unroll">Chains.</param><param name="copyStrided">Copy a non-contiguous x first (NumPy's own
    /// <c>x + 0.0</c> copies it) so the vector path runs.</param><returns>The pseudo-Vandermonde matrix.</returns>
    static NDArray Vander(Basis b, NDArray x, int deg, int unroll = 4, bool copyStrided = false)
    {
        NDArray own = null;
        if (copyStrided && !x.Shape.IsContiguous) x = own = x.copy();
        try
        {
            var k = GetVanderKernel(b, x.typecode, unroll);
            var dims = new long[x.ndim + 1];
            dims[0] = deg + 1;
            x.shape.CopyTo(dims, 1);
            var v = new NDArray(k.Tv, new Shape(dims), false);
            using (var v0 = v[0])
            {
                long* aux = stackalloc long[3];
                aux[0] = x.size * v.dtypesize;
                aux[1] = deg;
                aux[2] = k.Pool.TableFor(deg + 1);
                using var iter = NDIterRef.MultiNew(2, new[] { x, v0 }, NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAFE_CASTING, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null);
                iter.ForEach(k.Fn, aux);
            }
            var r = np.moveaxis(v, 0, -1);
            v.Dispose();   // the view holds its own reference to the buffer
            return r;
        }
        finally { own?.Dispose(); }
    }

    /// <summary>NumPy's <c>{p}vander2d</c> = <c>pu._vander_nd_flat</c>: two 1-D Vandermondes, a broadcast
    /// multiply (<c>vx[..., :, None] * vy[..., None, :]</c>) and the flattening reshape.</summary>
    /// <param name="b">Basis.</param><param name="x">First ordinate.</param><param name="y">Second ordinate.</param>
    /// <param name="dx">x degree.</param><param name="dy">y degree.</param><returns>Shape <c>x.shape + ((dx+1)*(dy+1),)</c>.</returns>
    static NDArray Vander2d(Basis b, NDArray x, NDArray y, int dx, int dy)
    {
        using var vx = Vander(b, x, dx);
        using var vy = Vander(b, y, dy);
        using var ex = np.expand_dims(vx, -1);
        using var ey = np.expand_dims(vy, -2);
        var prod = np.multiply(ex, ey);
        var dims = new long[x.ndim + 1];
        x.shape.CopyTo(dims, 0);
        dims[^1] = (dx + 1L) * (dy + 1L);
        // A view when the product's layout allows (NumPy's case), else a copy; either way the result owns
        // its reference and the product can be released.
        var r = prod.reshape(new Shape(dims));
        prod.Dispose();
        return r;
    }

    // ======================================================================== _vander_nd product kernel (Tier 3A)

    static readonly Dictionary<NPTypeCode, NDInnerLoopFunc> s_vprod = new();

    /// <summary>Points per tile in the product kernel: 512 float64 = 4 KB per row, so a degree-3 x degree-3
    /// product keeps its 4 + 4 input rows (32 KB) in L1 while it writes the 16 output rows.</summary>
    const long ProdTile = 512;

    /// <summary>Returns (compiling once per dtype) the <c>_vander_nd</c> row-product kernel.</summary>
    /// <param name="t">The Vandermonde dtype.</param><returns>The kernel.</returns>
    static NDInnerLoopFunc GetVProdKernel(NPTypeCode t)
    {
        if (s_vprod.TryGetValue(t, out var k)) return k;
        return s_vprod[t] = DirectILKernelGenerator.CompileRawInnerLoop(il => EmitVProdKernel(il, t), $"polyprobe_vprod2_{t}");
    }

    /// <summary>
    ///     <c>_vander_nd</c> for two ordinates as ONE pass: <c>out[i*(dy+1)+j] = vx_i * vy_j</c> per point — the
    ///     single multiply NumPy's broadcast <c>vx[..., :, None] * vy[..., None, :]</c> performs, in its operand
    ///     order, written straight into a C-contiguous <c>((dx+1)(dy+1), *x.shape)</c> buffer whose
    ///     <c>moveaxis(0, -1)</c> view is NumPy's F-contiguous result. Operands (vx[..., 0], vy[..., 0],
    ///     out[0]); auxdata [0] dx, [1] dy, [2..4] the three row byte strides. Contiguous chunks run tiled
    ///     vector loops (the input tiles stay in L1 across all output rows); anything else runs a strided
    ///     scalar loop.
    /// </summary>
    /// <param name="il">Generator.</param><param name="t">Dtype.</param>
    static void EmitVProdKernel(ILGenerator il, NPTypeCode t)
    {
        LocalBuilder P() => il.DeclareLocal(typeof(byte*));
        LocalBuilder Ln() => il.DeclareLocal(typeof(long));
        LocalBuilder ax = P(), ay = P(), ao = P(), pa = P(), pb = P(), po = P();
        LocalBuilder sx = Ln(), sy = Ln(), so = Ln(), dx = Ln(), dy = Ln(), rx = Ln(), ry = Ln(), ro = Ln();
        LocalBuilder i = Ln(), j = Ln(), q = Ln(), t0 = Ln(), tn = Ln();
        LdSlot(il, 0, 0, ax); LdSlot(il, 0, 1, ay); LdSlot(il, 0, 2, ao);
        LdSlot(il, 1, 0, sx); LdSlot(il, 1, 1, sy); LdSlot(il, 1, 2, so);
        LdSlot(il, 3, 0, dx); LdSlot(il, 3, 1, dy); LdSlot(il, 3, 2, rx); LdSlot(il, 3, 3, ry); LdSlot(il, 3, 4, ro);
        int sz = DirectILKernelGenerator.GetTypeSize(t);
        var ks = new K(false, t);
        var end = il.DefineLabel(); var scalar = il.DefineLabel();

        // dst = basePtr + idx * rowStride (+ whatever `extra` adds on the stack)
        void RowPtr(LocalBuilder basePtr, LocalBuilder idx, LocalBuilder rowStride, LocalBuilder dst, Action extra)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, idx); il.Emit(OpCodes.Ldloc, rowStride); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
            extra?.Invoke();
            il.Emit(OpCodes.Stloc, dst);
        }
        void PushOutRow()   // pushes i*(dy+1) + j
        {
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, dy); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Add);
        }
        void ElemAt(LocalBuilder ptr, LocalBuilder idx, long size)   // pushes ptr + idx*size
        {
            il.Emit(OpCodes.Ldloc, ptr); il.Emit(OpCodes.Ldloc, idx); il.Emit(OpCodes.Ldc_I8, size); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
        }
        void Loop(LocalBuilder v, Action<Label> exitWhenDone, Action body, long step)   // for (v = 0; ; v += step) { exitWhenDone; body }
        {
            var top = il.DefineLabel(); var done = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, v);
            il.MarkLabel(top);
            exitWhenDone(done);
            body();
            il.Emit(OpCodes.Ldloc, v); il.Emit(OpCodes.Ldc_I8, step); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, v);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(done);
        }

        if (VectorBlockOk(t, t, t, perLaneCoef: false))
        {
            K kv = t == NPTypeCode.Half ? new HalfVecK() : new K(true, t, DirectILKernelGenerator.VectorBits);
            long W = kv.Lanes;
            il.Emit(OpCodes.Ldloc, sx); il.Emit(OpCodes.Ldc_I8, (long)sz); il.Emit(OpCodes.Bne_Un, scalar);
            il.Emit(OpCodes.Ldloc, sy); il.Emit(OpCodes.Ldc_I8, (long)sz); il.Emit(OpCodes.Bne_Un, scalar);
            il.Emit(OpCodes.Ldloc, so); il.Emit(OpCodes.Ldc_I8, (long)sz); il.Emit(OpCodes.Bne_Un, scalar);
            Loop(t0, done => { il.Emit(OpCodes.Ldloc, t0); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, done); }, () =>
            {
                // tn = min(TILE, count - t0)
                var small = il.DefineLabel();
                il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldloc, t0); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Ldc_I8, ProdTile); il.Emit(OpCodes.Ble, small);
                il.Emit(OpCodes.Pop); il.Emit(OpCodes.Ldc_I8, ProdTile);
                il.MarkLabel(small); il.Emit(OpCodes.Stloc, tn);
                Action tileOff = () => { il.Emit(OpCodes.Ldloc, t0); il.Emit(OpCodes.Ldc_I8, (long)sz); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); };
                Loop(i, done => { il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, dx); il.Emit(OpCodes.Bgt, done); }, () =>
                {
                    RowPtr(ax, i, rx, pa, tileOff);
                    Loop(j, done => { il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldloc, dy); il.Emit(OpCodes.Bgt, done); }, () =>
                    {
                        RowPtr(ay, j, ry, pb, tileOff);
                        il.Emit(OpCodes.Ldloc, ao); PushOutRow(); il.Emit(OpCodes.Ldloc, ro); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                        tileOff(); il.Emit(OpCodes.Stloc, po);
                        // vector body over the tile, then its scalar tail
                        var vtop = il.DefineLabel(); var vdone = il.DefineLabel(); var sdone = il.DefineLabel();
                        il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, q);
                        il.MarkLabel(vtop);
                        il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldc_I8, W); il.Emit(OpCodes.Add); il.Emit(OpCodes.Ldloc, tn); il.Emit(OpCodes.Bgt, vdone);
                        ElemAt(pa, q, sz); kv.Load(il); ElemAt(pb, q, sz); kv.Load(il); kv.Bin(il, BinaryOp.Multiply);
                        ElemAt(po, q, sz); kv.StoreValueFirst(il);
                        il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldc_I8, W); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, q);
                        il.Emit(OpCodes.Br, vtop);
                        il.MarkLabel(vdone);
                        var stop = il.DefineLabel();
                        il.MarkLabel(stop);
                        il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldloc, tn); il.Emit(OpCodes.Bge, sdone);
                        ElemAt(po, q, sz);
                        ElemAt(pa, q, sz); ks.Load(il); ElemAt(pb, q, sz); ks.Load(il); ks.Bin(il, BinaryOp.Multiply);
                        DirectILKernelGenerator.EmitStoreIndirect(il, t);
                        il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, q);
                        il.Emit(OpCodes.Br, stop);
                        il.MarkLabel(sdone);
                    }, 1);
                }, 1);
            }, ProdTile);
            il.Emit(OpCodes.Br, end);
        }
        il.MarkLabel(scalar);
        // Strided chunk: per point, per (i, j): out = vx_i * vy_j through the true strides.
        Loop(q, done => { il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, done); }, () =>
        {
            Loop(i, done => { il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, dx); il.Emit(OpCodes.Bgt, done); }, () =>
            {
                RowPtr(ax, i, rx, pa, () => { il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldloc, sx); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); });
                Loop(j, done => { il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldloc, dy); il.Emit(OpCodes.Bgt, done); }, () =>
                {
                    il.Emit(OpCodes.Ldloc, ao); PushOutRow(); il.Emit(OpCodes.Ldloc, ro); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldloc, so); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldloc, pa); ks.Load(il);
                    il.Emit(OpCodes.Ldloc, ay); il.Emit(OpCodes.Ldloc, j); il.Emit(OpCodes.Ldloc, ry); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldloc, q); il.Emit(OpCodes.Ldloc, sy); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                    ks.Load(il);
                    ks.Bin(il, BinaryOp.Multiply);
                    DirectILKernelGenerator.EmitStoreIndirect(il, t);
                }, 1);
            }, 1);
        }, 1);
        il.MarkLabel(end);
        il.Emit(OpCodes.Ret);
    }

    /// <summary>NumPy's <c>{p}vander2d</c> through the fused product kernel: two 1-D Vandermondes, then ONE pass
    /// writing every <c>vx_i * vy_j</c> into NumPy's final layout (no broadcast temporaries, no reshape copy).</summary>
    /// <param name="b">Basis.</param><param name="x">First ordinate.</param><param name="y">Second ordinate (same shape and dtype).</param>
    /// <param name="dx">x degree.</param><param name="dy">y degree.</param><returns>Shape <c>x.shape + ((dx+1)*(dy+1),)</c>.</returns>
    /// <exception cref="ArgumentException">x and y differ in shape.</exception>
    static NDArray Vander2dFused(Basis b, NDArray x, NDArray y, int dx, int dy)
    {
        if (!x.shape.SequenceEqual(y.shape)) throw new ArgumentException("x, y are incompatible");
        using var vx = Vander(b, x, dx);
        using var vy = Vander(b, y, dy);
        var t = vx.typecode;
        var dims = new long[x.ndim + 1];
        dims[0] = (dx + 1L) * (dy + 1L);
        x.shape.CopyTo(dims, 1);
        var o = new NDArray(t, new Shape(dims), false);
        using (var vx0 = vx["..., 0"])
        using (var vy0 = vy["..., 0"])
        using (var o0 = o[0])
        {
            long* aux = stackalloc long[5];
            aux[0] = dx; aux[1] = dy;
            aux[2] = vx.strides[^1]; aux[3] = vy.strides[^1];
            aux[4] = x.size * o.dtypesize;
            using var iter = NDIterRef.MultiNew(3, new[] { vx0, vy0, o0 }, NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null);
            iter.ForEach(GetVProdKernel(t), aux);
        }
        var r = np.moveaxis(o, 0, -1);
        o.Dispose();
        return r;
    }

    // ======================================================================== Tier-3B shell + plain C# (section B)

    static NDInnerLoopFunc s_cheb3B;

    /// <summary>
    ///     The SAME Chebyshev recurrence as per-element bodies inside the house Tier-3B shell (the standard
    ///     4x-unrolled loop the shell owns). One chain per body: the shell's unroll re-runs the body, it does
    ///     not interleave independent chains — which is why Tier 3A wins here.
    /// </summary>
    /// <param name="x">float64 points.</param><param name="c">float64 coefficients.</param><returns>chebval(x, c).</returns>
    static NDArray ChebTier3B(NDArray x, NDArray c)
    {
        var pool = s_cheb3BPool ??= new ConstPool();
        s_cheb3B ??= DirectILKernelGenerator.CompileInnerLoop(new[] { NPTypeCode.Double, NPTypeCode.Double },
            il => Body3B(il, false), il => Body3B(il, true), "polyprobe_cheb_tier3b_f64");
        var y = new NDArray(NPTypeCode.Double, new Shape(x.shape), false);
        long* aux = stackalloc long[4];
        aux[0] = c.strides[0]; aux[1] = c.size; aux[2] = pool.TableFor(c.size);
        aux[3] = (long)((byte*)c.Storage.Address + c.Shape.offset * c.dtypesize);
        using var iter = NDIterRef.MultiNew(2, new[] { x, y }, NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
            NPY_CASTING.NPY_SAFE_CASTING, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null);
        iter.ForEach(s_cheb3B, aux);
        return y;

        static void Body3B(ILGenerator il, bool vec)
        {
            // Tier-3B contract: [x] on the stack in, [y] on the stack out; auxdata is arg 3.
            var ctx = new EvalCtx { Prog = Steps.Eval[Basis.cheb], Tx = NPTypeCode.Double, Tc = NPTypeCode.Double, Tl = NPTypeCode.Double, Cls = 3, P = 0, Pool = s_cheb3BPool };
            var em = new Emitter(il, 1, vec, LoopLanes(NPTypeCode.Double), ctx.Pool, null, null);
            var X = em.Fresh(NPTypeCode.Double, false);
            il.Emit(OpCodes.Stloc, X.L[0]);
            ctx.Cp = il.DeclareLocal(typeof(byte*)); ctx.Table = il.DeclareLocal(typeof(byte*));
            ctx.KStride = il.DeclareLocal(typeof(long)); ctx.Nc = il.DeclareLocal(typeof(long)); ctx.Row = il.DeclareLocal(typeof(long));
            LdSlot(il, 3, 0, ctx.KStride); LdSlot(il, 3, 1, ctx.Nc); LdSlot(il, 3, 2, ctx.Table); LdSlot(il, 3, 3, ctx.Cp);
            em = new Emitter(il, 1, vec, LoopLanes(NPTypeCode.Double), ctx.Pool, ctx.Table, ctx.Row);
            var cc = new CoefCtx { Broadcast = true, Cp = ctx.Cp, Tc = NPTypeCode.Double };
            var y = EmitClenshaw(em, ctx, cc, cc, ctx.KStride, X);
            il.Emit(OpCodes.Ldloc, y.L[0]);
        }
    }

    static ConstPool s_cheb3BPool;

    /// <summary>A hand-written C# Chebyshev kernel with 4 interleaved Vector256 chains — the calibration point
    /// showing that emitted IL compiles to the same machine code as the equivalent C#.</summary>
    /// <param name="x">Points.</param><param name="c">Coefficients.</param><param name="nc">Count (≥ 3).</param>
    /// <param name="y">Output.</param><param name="n">Point count.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void ChebvalCs(double* x, double* c, int nc, double* y, long n)
    {
        long i = 0;
        var two = Vector256.Create(2.0);
        var i0 = Vector256.Create(c[nc - 2]); var i1 = Vector256.Create(c[nc - 1]);
        for (; i + 16 <= n; i += 16)
        {
            var xa = Vector256.Load(x + i); var xb = Vector256.Load(x + i + 4); var xc = Vector256.Load(x + i + 8); var xd = Vector256.Load(x + i + 12);
            var ta = two * xa; var tb = two * xb; var tc = two * xc; var td = two * xd;
            Vector256<double> a0 = i0, b0 = i0, c0v = i0, d0 = i0, a1 = i1, b1 = i1, c1v = i1, d1 = i1;
            for (int k = nc - 3; k >= 0; k--)
            {
                var ck = Vector256.Create(c[k]);
                var s0 = a0; a0 = ck - a1; a1 = s0 + a1 * ta; var s1 = b0; b0 = ck - b1; b1 = s1 + b1 * tb;
                var s2 = c0v; c0v = ck - c1v; c1v = s2 + c1v * tc; var s3 = d0; d0 = ck - d1; d1 = s3 + d1 * td;
            }
            (a0 + a1 * xa).Store(y + i); (b0 + b1 * xb).Store(y + i + 4); (c0v + c1v * xc).Store(y + i + 8); (d0 + d1 * xd).Store(y + i + 12);
        }
        for (; i < n; i++)
        {
            double xv = x[i], x2 = 2 * xv, c0 = c[nc - 2], c1 = c[nc - 1];
            for (int k = nc - 3; k >= 0; k--) { double t = c0; c0 = c[k] - c1; c1 = t + c1 * x2; }
            y[i] = c0 + c1 * xv;
        }
    }

    // ======================================================================== harness

    static readonly Dictionary<string, double> s_numpy = new();

    /// <summary>Reads the twin's TSV (<c>cell&lt;TAB&gt;us</c>) so every row can print NPY/NS directly.</summary>
    /// <param name="path">TSV path.</param>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    public static void LoadNumPyTimes(string path)
    {
        foreach (var line in File.ReadAllLines(path))
        {
            var p = line.Split('\t');
            if (p.Length == 2 && double.TryParse(p[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us)) s_numpy[p[0]] = us;
        }
    }

    /// <summary>Loads one .npy the twin wrote.</summary>
    /// <param name="name">File stem.</param><returns>The array.</returns>
    /// <exception cref="FileNotFoundException">The twin has not been run.</exception>
    static NDArray L(string name) => np.load_npy(Path.Combine(Data, name + ".npy"));

    /// <summary>Best-of-7 microseconds per call after a 300 ms warm-up (tier-1 JIT), each repeat sized to
    /// ~25 ms; results are disposed so the loop measures steady-state pooled allocation.</summary>
    /// <param name="f">The call.</param><returns>Microseconds.</returns>
    static double Bench(Func<NDArray> f)
    {
        var sw = Stopwatch.StartNew(); int w = 0;
        while (sw.ElapsedMilliseconds < 300 || w < 3) { f().Dispose(); w++; }
        sw.Restart(); f().Dispose();
        int iters = Math.Max(1, (int)(25.0 / Math.Max(sw.Elapsed.TotalMilliseconds, 1e-4)));
        double best = double.MaxValue;
        for (int r = 0; r < 7; r++)
        {
            sw.Restart();
            for (int j = 0; j < iters; j++) f().Dispose();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
        }
        return best * 1000.0;
    }

    /// <summary>Byte-compares a result (in logical C order) with a NumPy reference: dtype, shape, then every
    /// element's bytes. NaN payloads and signed zeros count.</summary>
    /// <param name="got">NumSharp result.</param><param name="refName">Reference stem, or null (timing-only cell).</param>
    /// <returns><c>exact</c>, or what differs.</returns>
    static string Check(NDArray got, string refName)
    {
        if (refName is null) return "-";
        using var want = L(refName);
        if (got.typecode != want.typecode) return $"DTYPE {got.typecode} vs {want.typecode}";
        if (!got.shape.SequenceEqual(want.shape)) return $"SHAPE ({string.Join(",", got.shape)}) vs ({string.Join(",", want.shape)})";
        var gb = np.ascontiguousarray(got);
        try
        {
            ReadOnlySpan<byte> a = gb.Unsafe.ReadOnlyBytes(), w = want.Unsafe.ReadOnlyBytes();
            int isz = got.dtypesize; long bad = 0;
            for (int e = 0; e < a.Length; e += isz)
                if (!a.Slice(e, isz).SequenceEqual(w.Slice(e, isz))) bad++;
            return bad == 0 ? "exact" : $"{bad}/{got.size} differ";
        }
        finally { if (!ReferenceEquals(gb, got)) gb.Dispose(); }
    }

    /// <summary>Checks, optionally times, and prints one TSV row (with the NumPy time and NPY/NS when the twin's
    /// TSV was loaded).</summary>
    /// <param name="cell">Join label (identical to the twin's).</param><param name="variant">Strategy label.</param>
    /// <param name="f">The call.</param><param name="refName">Reference stem or null.</param>
    /// <param name="timed">Time it (false: correctness-only row).</param><param name="layout">Optional extra check on the result.</param>
    static void Row(string cell, string variant, Func<NDArray> f, string refName, bool timed = true, Func<NDArray, string> layout = null)
    {
        string ok;
        using (var first = f())
        {
            ok = Check(first, refName);
            if (layout is not null) ok += " " + layout(first);
        }
        double us = timed ? Bench(f) : double.NaN;
        bool hasNp = s_numpy.TryGetValue(cell, out var npus);
        string ratio = timed && hasNp ? (npus / us).ToString("F2", CultureInfo.InvariantCulture) : "-";
        Console.WriteLine($"{cell}\t{variant}\t{(timed ? us.ToString("F3", CultureInfo.InvariantCulture) : "-")}\t" +
                          $"{(hasNp ? npus.ToString("F3", CultureInfo.InvariantCulture) : "-")}\t{ratio}\t{ok}");
    }

    /// <summary>The layout NumPy's <c>moveaxis</c> view has for a 1-D x: strides <c>(itemsize, itemsize*n)</c>.</summary>
    /// <param name="v">A Vandermonde result.</param><returns><c>layout-ok</c> or the strides found.</returns>
    static string VanderLayout(NDArray v)
    {
        long isz = v.dtypesize, n = v.shape[0];
        var s = v.strides;
        return s.Length == 2 && s[0] == isz && s[1] == isz * n ? "layout-ok" : $"LAYOUT ({string.Join(",", s)})";
    }

    // ======================================================================== sections

    /// <summary>Runs the requested sections.</summary>
    /// <param name="sections">Section letters.</param>
    public static void Run(string sections)
    {
        Console.WriteLine($"# VectorBits={DirectILKernelGenerator.VectorBits} AVX2={Avx2.IsSupported}");
        Console.WriteLine("cell\tvariant\tns_us\tnumpy_us\tNPY/NS\tcheck");
        var x = new Dictionary<int, NDArray>();
        foreach (int n in new[] { 1_000, 100_000, 10_000_000 }) x[n] = L($"x{n}");
        var c = new Dictionary<int, NDArray>();
        foreach (int d in new[] { 0, 1, 2, 3, 4, 10, 30 }) c[d] = L($"c{d}");
        var all = Enum.GetValues<Basis>();

        if (sections.Contains('A'))
        {
            foreach (var b in all)
                foreach (int d in new[] { 0, 1, 2, 3, 4, 10, 30 })
                    foreach (int n in new[] { 1_000, 100_000, 10_000_000 })
                    {
                        if (n == 100_000 && d is not (3 or 10 or 30)) continue;
                        if (n == 10_000_000 && d != 10) continue;
                        bool timed = d is 3 or 10 or 30;
                        string cell = $"A {b}val d{d} n{n}", rf = n == 10_000_000 ? null : $"ref_A_{b}_d{d}_n{n}";
                        var xn = x[n]; var cd = c[d];
                        Row(cell, "IL3A-x4", () => Val(b, xn, cd), rf, timed);
                        if (timed && d >= 10 && n != 10_000_000) Row(cell, "IL3A-x1", () => Val(b, xn, cd, unroll: 1), rf);
                        if (timed && d == 10 && n != 10_000_000) Row(cell, "IL3A-x4-operand", () => Val(b, xn, cd, forceOperand: true), rf);
                    }
        }
        if (sections.Contains('B'))
        {
            foreach (int n in new[] { 1_000, 100_000, 10_000_000 })
            {
                string cell = $"A chebval d10 n{n}", rf = n == 10_000_000 ? null : $"ref_A_cheb_d10_n{n}";
                var xn = x[n]; var c10 = c[10];
                Row(cell, "IL3A-x4", () => Val(Basis.cheb, xn, c10), rf);
                Row(cell, "IL3B-shell", () => ChebTier3B(xn, c10), rf);
                Row(cell, "Cs-direct", () =>
                {
                    var y = new NDArray(NPTypeCode.Double, new Shape(xn.shape), false);
                    ChebvalCs((double*)xn.Storage.Address, (double*)c10.Storage.Address, (int)c10.size, (double*)y.Storage.Address, xn.size);
                    return y;
                }, rf);
            }
        }
        if (sections.Contains('C'))
        {
            var x100 = x[100_000];
            var x32 = L("x32"); var x16 = L("x16"); var xc = L("xc"); var cc10 = L("cc10"); var xi = L("xi");
            var c10f32 = c[10].astype(NPTypeCode.Single); var c10f16 = c[10].astype(NPTypeCode.Half);
            foreach (var b in all)
            {
                Row($"C {b}val f32 d10 n100000", "IL3A-x4", () => Val(b, x32, c10f32), $"ref_C_{b}_f32");
                Row($"C {b}val c128 d10 n100000", "IL3A-x4", () => Val(b, xc, cc10), $"ref_C_{b}_c128");
            }
            foreach (var b in new[] { Basis.poly, Basis.cheb, Basis.leg, Basis.herm })
                Row($"C {b}val f16 d10 n100000", "IL3A-x4", () => Val(b, x16, c10f16), $"ref_C_{b}_f16");
            foreach (var b in new[] { Basis.cheb, Basis.leg })
                Row($"C {b}val x32/c64 d10 n100000", "IL3A-x4", () => Val(b, x32, c[10]), $"ref_C_{b}_x32c64");
            foreach (var b in new[] { Basis.poly, Basis.cheb, Basis.lag })
                Row($"C {b}val int64x d10 n100000", "IL3A-x4", () => Val(b, xi, c[10]), $"ref_C_{b}_xi");
            foreach (var b in new[] { Basis.cheb, Basis.leg, Basis.lag, Basis.herm, Basis.herme })
                foreach (int nc in new[] { 3, 4, 5, 11 })
                {
                    var cf = L($"cf32_{nc}");
                    string cell = $"C {b}val c32/x64 nc{nc} n100000", rf = $"ref_C_{b}_c32x64_nc{nc}";
                    Row(cell, "IL3A-x4-peeled", () => Val(b, x100, cf), rf, timed: nc == 11);
                    // The tempting shortcut: cast the series up front. Not NumPy (its first steps run in float32).
                    var cf64 = cf.astype(NPTypeCode.Double);
                    Row(cell, "precast-c (wrong)", () => Val(b, x100, cf64), rf, timed: false);
                }
            var cf16 = L("cf16");
            Row("C legval c16/x32 nc11 n100000", "IL3A-x4-peeled", () => Val(Basis.leg, x32, cf16), "ref_C_leg_c16x32");
            var e64 = L("edge64"); var e32 = L("edge32");
            foreach (var b in all) Row($"C {b}val edge64 d10", "IL3A-x4", () => Val(b, e64, c[10]), $"ref_C_{b}_edge64", timed: false);
            Row("C chebval edge32/c64 d10", "IL3A-x4", () => Val(Basis.cheb, e32, c[10]), "ref_C_cheb_edge32", timed: false);
        }
        if (sections.Contains('D'))
        {
            var x100 = x[100_000];
            foreach (var b in new[] { Basis.cheb, Basis.leg })
            {
                var layouts = new (string, NDArray, string)[]
                {
                    ($"D {b}val strided d10 n50000", x100["::2"], $"ref_D_{b}_strided"),
                    ($"D {b}val reversed d10 n100000", x100["::-1"], $"ref_D_{b}_reversed"),
                    ($"D {b}val 2dT d10 n100000", x100.reshape(250, 400).T, $"ref_D_{b}_2dT"),
                };
                foreach (var (cell, xv, rf) in layouts)
                {
                    Row(cell, "IL3A-x4 strided", () => Val(b, xv, c[10]), rf);
                    Row(cell, "IL3A-x4 BUFFERED+CONTIG", () => Val(b, xv, c[10], buffered: true), rf);
                }
            }
        }
        if (sections.Contains('E'))
        {
            foreach (int n in new[] { 1_000, 100_000 })
            {
                var xn = x[n];
                Row($"A chebval d10 n{n}", "IL3A-x4 +BUFFERED (contig x)", () => Val(Basis.cheb, xn, c[10], buffered: true), $"ref_A_cheb_d10_n{n}");
            }
        }
        if (sections.Contains('F'))
        {
            var x100 = x[100_000]; var y = L("y"); var cm = L("cm"); var cp = L("cp"); var c55 = L("c55"); var gx = L("gx"); var gy = L("gy");
            foreach (var b in new[] { Basis.poly, Basis.cheb, Basis.leg })
            {
                Row($"F {b}val multi8 d10 n100000", "IL3A-x4 operand(bcast)", () => Val(b, x100, cm), $"ref_F_{b}_multi");
                Row($"F {b}val tensorF d10 n100000", "IL3A-x4 operand(lane)", () => Val(b, x100, cp, tensor: false), $"ref_F_{b}_tensorF");
                Row($"F {b}val2d d5x5 n100000", "_valnd over IL3A", () => Val2d(b, x100, y, c55), $"ref_F_{b}_val2d");
                Row($"F {b}grid2d d5x5 300x300", "_gridnd over IL3A", () => Grid2d(b, gx, gy, c55), $"ref_F_{b}_grid2d");
            }
            var xs = x100["::2"]; var cps = cp[":, ::2"];
            Row("F chebval tensorF strided d10 n50000", "IL3A-x4 operand(lane, scalar)", () => Val(Basis.cheb, xs, cps, tensor: false), "ref_F_cheb_tensorF_strided");
        }
        if (sections.Contains('G'))
        {
            var xm = new Dictionary<int, NDArray> { [1_000] = x[1_000], [100_000] = x[100_000], [1_000_000] = x[10_000_000]["0:1000000"] };
            var e64 = L("edge64");
            foreach (var b in all)
            {
                foreach (int n in new[] { 1_000, 100_000, 1_000_000 })
                {
                    var xn = xm[n];
                    string cell = $"G {b}vander d10 n{n}", rf = n == 1_000_000 ? null : $"ref_G_{b}_d10_n{n}";
                    Row(cell, "IL3A-x4", () => Vander(b, xn, 10), rf, layout: VanderLayout);
                    if (n == 100_000) Row(cell, "IL3A-x1", () => Vander(b, xn, 10, unroll: 1), rf);
                }
                Row($"G {b}vander d0 n1000", "IL3A-x4", () => Vander(b, x[1_000], 0), $"ref_G_{b}_d0", timed: false);
                Row($"G {b}vander d1 n1000", "IL3A-x4", () => Vander(b, x[1_000], 1), $"ref_G_{b}_d1", timed: false);
                Row($"G {b}vander edge64 d10", "IL3A-x4", () => Vander(b, e64, 10), $"ref_G_{b}_edge64", timed: false);
            }
            var x100 = x[100_000];
            var x32 = L("x32"); var x16 = L("x16"); var xc = L("xc"); var xi = L("xi"); var y = L("y");
            Row("G chebvander f32 d10 n100000", "IL3A-x4", () => Vander(Basis.cheb, x32, 10), "ref_G_cheb_f32");
            Row("G chebvander f16 d10 n100000", "IL3A-x4", () => Vander(Basis.cheb, x16, 10), "ref_G_cheb_f16");
            Row("G chebvander c128 d10 n100000", "IL3A-x4", () => Vander(Basis.cheb, xc, 10), "ref_G_cheb_c128");
            Row("G chebvander int64x d10 n100000", "IL3A-x4", () => Vander(Basis.cheb, xi, 10), "ref_G_cheb_xi");
            var xs = x100["::2"]; var xt = x100.reshape(250, 400).T;
            Row("G chebvander strided d10 n50000", "IL3A-x4 strided", () => Vander(Basis.cheb, xs, 10), "ref_G_cheb_strided");
            Row("G chebvander strided d10 n50000", "IL3A-x4 copy-first", () => Vander(Basis.cheb, xs, 10, copyStrided: true), "ref_G_cheb_strided");
            Row("G chebvander 2dT d10 n100000", "IL3A-x4 strided", () => Vander(Basis.cheb, xt, 10), "ref_G_cheb_2dT");
            Row("G chebvander 2dT d10 n100000", "IL3A-x4 copy-first", () => Vander(Basis.cheb, xt, 10, copyStrided: true), "ref_G_cheb_2dT");
            foreach (var b in new[] { Basis.poly, Basis.cheb })
            {
                Row($"G {b}vander2d d3x3 n100000", "_vander_nd np.multiply", () => Vander2d(b, x100, y, 3, 3), $"ref_G_{b}_vander2d");
                Row($"G {b}vander2d d3x3 n100000", "_vander_nd IL product", () => Vander2dFused(b, x100, y, 3, 3), $"ref_G_{b}_vander2d",
                    layout: v => v.strides.Length == 2 && v.strides[0] == v.dtypesize && v.strides[1] == v.dtypesize * v.shape[0]
                        ? "layout-ok" : $"LAYOUT ({string.Join(",", v.strides)})");
            }
        }
    }
}
