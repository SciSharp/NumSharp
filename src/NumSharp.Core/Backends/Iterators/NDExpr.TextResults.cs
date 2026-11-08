using System;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.TextResults.cs — the stages of np.evaluate's phase E3 (docs/plans/evaluate-text.md)
// =============================================================================
//
// E1 and E2 computed on text only where NumPy's answer is a number or a bool. Phase E3 gives every other node with a
// text operand NumPy's own answer: a node whose NumPy statement computes text (np.add's concatenation, np.multiply's
// StringDType repeat, np.maximum / np.minimum over StringDType, np.where with text branches, a cast to text, np.sum /
// np.max / np.min over StringDType, np.real / np.imag) yields a TEXT stage result, which later stages and the
// evaluation's own result read; a node NumPy refuses for text raises NumPy's own error when the call runs, in NumPy's
// statement order.
//
// Each stage below calls the engine member that IS NumPy's statement for its node — the same member np.add, np.where,
// np.sum, ... call — so its values, its StringDType NA rules, its promotion and its error texts are the ones the string
// oracle already gates against NumPy, and no text rule is written twice. A stage never decides an answer itself: it maps
// its node to that member and passes the operands the lowering resolved (inputs, NumPy's Python scalars for literals,
// earlier stage results, computed numeric sub-trees in NumPy's temporary layout).
//
// The stage's ResultType is the dtype NumPy's statement has WHEN it computes: String for a text result (its class and
// width are read off the array per call, so one program serves every width), int64 for np.argmax / np.argmin, float64
// for np.mean / np.nanmean (a NaN-like NA's sum is NaN). A statement NumPy refuses for every text class is typed as a
// text result: no numeric remainder input is made for it, and it raises before anything could read it.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     E3: a binary ufunc with a text operand, through the engine's ufunc entry
    ///     (<see cref="DefaultEngine.ExecuteBinaryOp"/>, whose string guard is the one every np.* binary ufunc runs first):
    ///     np.add concatenates two strings of one class (<c>S+S → S(wa+wb)</c>, <c>U+U → U(wa+wb)</c>, a StringDType or a
    ///     StringDType with U / a Python str → StringDType), np.multiply repeats a StringDType string by an integer (an
    ///     <c>S</c> / <c>U</c> repeat needs <c>out=</c>, NumPy's own rule), and every other pair or ufunc raises NumPy's
    ///     no-loop error.
    /// </summary>
    internal sealed class NDExprBinaryStage : NDExprTextStage
    {
        /// <summary>The ufunc.</summary>
        internal readonly BinaryOp Op;

        /// <summary>Creates a binary stage.</summary>
        /// <param name="op">The ufunc.</param>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        internal NDExprBinaryStage(BinaryOp op, NDExprStageOperand left, NDExprStageOperand right)
            : base(DefaultEngine.UfuncName(op), NPTypeCode.String, left, right)
        {
            Op = op;
        }

        /// <inheritdoc/>
        internal override bool TakesOutAndWhere => true;

        /// <inheritdoc/>
        /// <exception cref="TypeError">NumPy has no loop for the operands (S with U, text with a number, any ufunc but add /
        ///     multiply), or an <c>S</c> / <c>U</c> repeat without <c>out=</c> ("The 'out' kwarg is necessary when using the
        ///     string multiply ufunc directly"), or a concatenation wider than 2^31 - 1 bytes: NumPy's texts.</exception>
        /// <exception cref="OverflowException">A StringDType repeat by a negative count or one past 2^63 - 1 characters
        ///     (NumPy's "Overflow encountered in string multiply").</exception>
        /// <exception cref="ValueError">A StringDType NA the ufunc's rule refuses (<c>Cannot add null that is not a nan-like
        ///     value</c>), or two StringDType operands with incompatible NA objects.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args) => engine.ExecuteBinaryOp(args[0], args[1], Op);

        /// <inheritdoc/>
        /// <exception cref="TypeError">The loop's text result cannot be cast into <paramref name="out"/> under
        ///     same_kind (NumPy's ufunc output-cast text).</exception>
        /// <exception cref="ArgumentException">The operands do not broadcast into <paramref name="out"/>.</exception>
        internal override NDArray ExecuteInto(DefaultEngine engine, NDArray[] args, NDArray @out, NDArray where)
            => engine.ExecuteBinaryOp(args[0], args[1], Op, @out, where);
    }

    /// <summary>
    ///     E3: <c>np.maximum</c> / <c>np.minimum</c> with a text operand, through the engine's ufunc entry: two StringDType
    ///     operands compare (a NaN-like NA wins, a None-like NA raises, a string NA compares as its text); every other pair —
    ///     two <c>U</c> or <c>S</c>, StringDType beside <c>U</c> or a Python str, text beside a number — is NumPy's no-loop
    ///     refusal.
    /// </summary>
    internal sealed class NDExprMinMaxStage : NDExprTextStage
    {
        /// <summary>True for np.minimum, false for np.maximum.</summary>
        internal readonly bool IsMin;

        /// <summary>Creates a maximum / minimum stage.</summary>
        /// <param name="isMin">True for np.minimum.</param>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        internal NDExprMinMaxStage(bool isMin, NDExprStageOperand left, NDExprStageOperand right)
            : base(isMin ? "minimum" : "maximum", NPTypeCode.String, left, right)
        {
            IsMin = isMin;
        }

        /// <inheritdoc/>
        internal override bool TakesOutAndWhere => true;

        /// <inheritdoc/>
        /// <exception cref="TypeError">NumPy has no loop for the operand classes (its texts name the descriptors or the
        ///     classes).</exception>
        /// <exception cref="ValueError">A None-like StringDType NA (<c>Cannot compare null that is not a nan-like value</c>), or
        ///     incompatible NA objects.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args)
            => engine.ExecuteBinaryOp(args[0], args[1], IsMin ? BinaryOp.Minimum : BinaryOp.Maximum);

        /// <inheritdoc/>
        /// <exception cref="TypeError">The result cannot be cast into <paramref name="out"/> under
        ///     same_kind.</exception>
        /// <exception cref="ArgumentException">The operands do not broadcast into <paramref name="out"/>.</exception>
        internal override NDArray ExecuteInto(DefaultEngine engine, NDArray[] args, NDArray @out, NDArray where)
            => engine.ExecuteBinaryOp(args[0], args[1], IsMin ? BinaryOp.Minimum : BinaryOp.Maximum, @out, where);
    }

    /// <summary>
    ///     E3: a two-sided <c>np.clip(x, lo, hi)</c> with a text operand, through the engine's clip
    ///     (<see cref="TensorEngine.ClipNDArray"/>): NumPy's clip ufunc has no text loop for any class, so this is NumPy's
    ///     refusal naming all three operands.
    /// </summary>
    internal sealed class NDExprClipStage : NDExprTextStage
    {
        /// <summary>Creates a clip stage.</summary>
        /// <param name="x">The values.</param>
        /// <param name="lo">The lower bound.</param>
        /// <param name="hi">The upper bound.</param>
        internal NDExprClipStage(NDExprStageOperand x, NDExprStageOperand lo, NDExprStageOperand hi)
            : base("clip", NPTypeCode.String, x, lo, hi)
        {
        }

        /// <inheritdoc/>
        /// <exception cref="TypeError">Always for a text operand: <c>ufunc 'clip' did not contain a loop with signature
        ///     matching types (...)</c>, NumPy's text.</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args) => engine.ClipNDArray(args[0], args[1], args[2]);
    }

    /// <summary>
    ///     E3: a unary node with a text operand, through the engine's member for NumPy's function (np.negative, np.sqrt,
    ///     np.logical_not, np.isinf, np.round, np.angle, ...): np.real of text is the text itself, np.imag of text an
    ///     all-empty array of its dtype, and every other one NumPy's refusal (the legacy resolver's "not supported for the
    ///     input types" for <c>S</c> / <c>U</c>, the no-loop text naming StringDType's class). IsNaN has its own stage
    ///     (<see cref="NDExprIsNanStage"/>), which computes over StringDType.
    /// </summary>
    internal sealed class NDExprUnaryStage : NDExprTextStage
    {
        /// <summary>The operation.</summary>
        internal readonly UnaryOp Op;

        /// <summary>Creates a unary stage.</summary>
        /// <param name="op">The operation (never IsNan).</param>
        /// <param name="x">The operand.</param>
        internal NDExprUnaryStage(UnaryOp op, NDExprStageOperand x) : base(FunctionName(op), NPTypeCode.String, x)
        {
            Op = op;
        }

        /// <summary>NumPy's function name of a unary node (the np.* call the node is), for diagnostics.</summary>
        /// <param name="op">The operation.</param>
        /// <returns>The name (<c>"negative"</c>, <c>"isposinf"</c>, <c>"real"</c>, ...).</returns>
        private static string FunctionName(UnaryOp op) => op switch
        {
            UnaryOp.Real => "real",
            UnaryOp.Imag => "imag",
            UnaryOp.Angle => "angle",
            UnaryOp.Round => "round",
            UnaryOp.Rint => "rint",
            UnaryOp.IsPosInf => "isposinf",
            UnaryOp.IsNegInf => "isneginf",
            UnaryOp.BitwiseCount => "bitwise_count",
            UnaryOp.Exp2 => "exp2",
            UnaryOp.Expm1 => "expm1",
            UnaryOp.Log2 => "log2",
            UnaryOp.Log10 => "log10",
            UnaryOp.Log1p => "log1p",
            UnaryOp.Deg2Rad => "deg2rad",
            UnaryOp.Rad2Deg => "rad2deg",
            UnaryOp.SignBit => "signbit",
            _ => DefaultEngine.UfuncName(op),
        };

        /// <inheritdoc/>
        /// <exception cref="TypeError">NumPy has no loop for the operand (every unary ufunc but isnan / real / imag over
        ///     text), with NumPy's text for its class.</exception>
        /// <exception cref="InvalidOperationException">An operation with no NumPy function over text here (IsNan, which has its
        ///     own stage) — a lowering bug.</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args)
        {
            var x = args[0];
            switch (Op)
            {
                case UnaryOp.Real:
                {
                    // NumPy's np.real of a non-complex array is the array itself; np.evaluate answers with an array of its own,
                    // so the text is copied in its own layout (K order), never handed out as the caller's input.
                    var real = NDArray.real_(x);
                    return ReferenceEquals(real, x) ? x.copy('K') : real;
                }
                case UnaryOp.Imag:
                {
                    // NumPy's np.imag of a non-complex array is a fresh READ-ONLY all-zero (here: all-empty) array of its dtype;
                    // np.evaluate's result is writeable, so a read-only answer is copied once and released.
                    var imag = NDArray.imag_(x);
                    if (imag.Shape.IsWriteable)
                        return imag;
                    using (imag)
                        return imag.copy('K');
                }
            }

            return Op switch
            {
                UnaryOp.Negate => engine.Negate(x),
                UnaryOp.Positive => engine.Positive(x),
                UnaryOp.Abs => engine.Abs(x),
                UnaryOp.Fabs => engine.Fabs(x),
                UnaryOp.Sqrt => engine.Sqrt(x),
                UnaryOp.Cbrt => engine.Cbrt(x),
                UnaryOp.Square => engine.Square(x),
                UnaryOp.Reciprocal => engine.Reciprocal(x),
                UnaryOp.Sign => engine.Sign(x),
                UnaryOp.Exp => engine.Exp(x),
                UnaryOp.Exp2 => engine.Exp2(x),
                UnaryOp.Expm1 => engine.Expm1(x),
                UnaryOp.Log => engine.Log(x),
                UnaryOp.Log2 => engine.Log2(x),
                UnaryOp.Log10 => engine.Log10(x),
                UnaryOp.Log1p => engine.Log1p(x),
                UnaryOp.Sin => engine.Sin(x),
                UnaryOp.Cos => engine.Cos(x),
                UnaryOp.Tan => engine.Tan(x),
                UnaryOp.Sinh => engine.Sinh(x),
                UnaryOp.Cosh => engine.Cosh(x),
                UnaryOp.Tanh => engine.Tanh(x),
                UnaryOp.ASin => engine.ASin(x),
                UnaryOp.ACos => engine.ACos(x),
                UnaryOp.ATan => engine.ATan(x),
                UnaryOp.Asinh => engine.ASinh(x),
                UnaryOp.Acosh => engine.ACosh(x),
                UnaryOp.Atanh => engine.ATanh(x),
                UnaryOp.Deg2Rad => engine.Deg2Rad(x),
                UnaryOp.Rad2Deg => engine.Rad2Deg(x),
                UnaryOp.Floor => engine.Floor(x),
                UnaryOp.Ceil => engine.Ceil(x),
                UnaryOp.Truncate => engine.Truncate(x),
                // NDExpr.Round(x) is np.round(x) (decimals 0): the engine's round, whose text refusal names NumPy's 'rint'.
                UnaryOp.Round => engine.Round(x),
                UnaryOp.Rint => engine.Rint(x),
                UnaryOp.BitwiseNot => engine.Invert(x),
                UnaryOp.BitwiseCount => engine.BitwiseCount(x),
                UnaryOp.LogicalNot => engine.LogicalNot(x),
                UnaryOp.IsFinite => engine.IsFinite(x),
                UnaryOp.IsInf => engine.IsInf(x),
                UnaryOp.IsPosInf => engine.IsPosInf(x),
                UnaryOp.IsNegInf => engine.IsNegInf(x),
                UnaryOp.SignBit => engine.SignBit(x),
                // NDExpr.Conjugate is the np.conjugate UFUNC (no text loop), not ndarray.conjugate's method.
                UnaryOp.Conjugate => engine.Conjugate(x),
                UnaryOp.Spacing => engine.Spacing(x),
                UnaryOp.Angle => engine.Angle(x),
                _ => throw new InvalidOperationException(
                    $"np.evaluate's unary text stage has no NumPy function for {Op}: a lowering bug."),
            };
        }
    }

    /// <summary>
    ///     E3: <c>np.round(x, decimals)</c> (a nonzero <c>decimals</c>) with a text operand, through the engine's round:
    ///     NumPy's refusal from its first step (<c>np.multiply</c> / <c>np.divide</c> by <c>10**|decimals|</c> has no text
    ///     loop).
    /// </summary>
    internal sealed class NDExprRoundStage : NDExprTextStage
    {
        /// <summary>The decimals.</summary>
        internal readonly int Decimals;

        /// <summary>Creates a round stage.</summary>
        /// <param name="decimals">The decimals (nonzero: a zero is the unary round node).</param>
        /// <param name="x">The operand.</param>
        internal NDExprRoundStage(int decimals, NDExprStageOperand x) : base("round", NPTypeCode.String, x)
        {
            Decimals = decimals;
        }

        /// <inheritdoc/>
        /// <exception cref="TypeError">Always for text: NumPy's no-loop text of the multiply / divide step.</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args) => engine.Round(args[0], Decimals);
    }

    /// <summary>
    ///     E3: <c>np.where(cond, a, b)</c> with a text branch, through the array object's <c>PyArray_Where</c>
    ///     (<see cref="NDArray.where(NDArray, NDArray, NDArray)"/>): the branches' NEP 50 promotion first (U with S → U,
    ///     U with int64 → <c>&lt;U21</c>, U with float64 → <c>&lt;U32</c>, U with StringDType → StringDType; a Python number
    ///     beside fixed-width text, StringDType beside a number and S beside StringDType are NumPy's DTypePromotionError),
    ///     then the broadcast, then the selection — a text condition read through its bool cast inside the call.
    /// </summary>
    internal sealed class NDExprWhereStage : NDExprTextStage
    {
        /// <summary>Creates a where stage.</summary>
        /// <param name="cond">The condition.</param>
        /// <param name="a">The values where it holds.</param>
        /// <param name="b">The values elsewhere.</param>
        internal NDExprWhereStage(NDExprStageOperand cond, NDExprStageOperand a, NDExprStageOperand b)
            : base("where", NPTypeCode.String, cond, a, b)
        {
        }

        /// <inheritdoc/>
        /// <exception cref="TypeError">The branches have no common dtype (NumPy's DTypePromotionError text), or two StringDType
        ///     branches carry incompatible NA objects.</exception>
        /// <exception cref="IncorrectShapeException">The three operands do not broadcast (NumPy's iterator text).</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args) => NDArray.where(args[0], args[1], args[2]);
    }

    /// <summary>
    ///     E3: a root reduction of text, through the engine's member for NumPy's function with the node's axis, keepdims and
    ///     ddof: np.argmax / np.argmin over <c>S</c> / <c>U</c> / StringDType (int64 indices; a NaN-like NA ranks last), np.sum
    ///     / np.max / np.min over StringDType (a concatenation and a selection; NumPy refuses a flat one over several axes,
    ///     <c>reduction operation 'add' is not reorderable</c>), np.mean / np.nanmean (float64 NaN when a NaN-like NA makes
    ///     the sum NaN, else NumPy's refusal), and NumPy's refusal for np.prod, np.ptp, np.var, np.std, np.nanmin, np.nanmax.
    ///     <c>NanSum</c> / <c>NanProd</c> of text arrive as np.sum / np.prod (NumPy's NaN replacement leaves text as it is).
    /// </summary>
    internal sealed class NDExprReduceStage : NDExprTextStage
    {
        /// <summary>The reduction.</summary>
        internal readonly NDExprReduceKind Kind;

        /// <summary>The axis, or null for a flat reduction.</summary>
        internal readonly int? Axis;

        /// <summary>Whether the reduced axes are kept as length 1.</summary>
        internal readonly bool Keepdims;

        /// <summary>Var / Std's delta degrees of freedom.</summary>
        internal readonly int Ddof;

        /// <summary>Creates a reduction stage.</summary>
        /// <param name="kind">The reduction (never Any / All: those read the truth through a truth stage).</param>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="keepdims">Whether the reduced axes stay as length 1.</param>
        /// <param name="ddof">Var / Std's delta degrees of freedom.</param>
        /// <param name="x">The reduced text.</param>
        internal NDExprReduceStage(NDExprReduceKind kind, int? axis, bool keepdims, int ddof, NDExprStageOperand x)
            : base(FunctionName(kind), ResultTypeOf(kind), x)
        {
            Kind = kind;
            Axis = axis;
            Keepdims = keepdims;
            Ddof = ddof;
        }

        /// <summary>NumPy's function of a reduction kind.</summary>
        /// <param name="kind">The kind.</param>
        /// <returns>The np.* name.</returns>
        private static string FunctionName(NDExprReduceKind kind) => kind switch
        {
            NDExprReduceKind.Sum => "sum",
            NDExprReduceKind.Prod => "prod",
            NDExprReduceKind.Min => "min",
            NDExprReduceKind.Max => "max",
            NDExprReduceKind.Mean => "mean",
            NDExprReduceKind.Ptp => "ptp",
            NDExprReduceKind.NanMin => "nanmin",
            NDExprReduceKind.NanMax => "nanmax",
            NDExprReduceKind.ArgMax => "argmax",
            NDExprReduceKind.ArgMin => "argmin",
            NDExprReduceKind.NanMean => "nanmean",
            NDExprReduceKind.Var => "var",
            NDExprReduceKind.Std => "std",
            _ => kind.ToString().ToLowerInvariant(),
        };

        /// <summary>
        ///     The dtype a reduction of text has when NumPy computes it: int64 for the index kinds, float64 for np.mean /
        ///     np.nanmean (a NaN-like NA's NaN), text for np.sum / np.max / np.min over StringDType — and text for the kinds
        ///     NumPy always refuses over text, so no numeric remainder input is made for them.
        /// </summary>
        /// <param name="kind">The kind.</param>
        /// <returns>The result dtype.</returns>
        private static NPTypeCode ResultTypeOf(NDExprReduceKind kind) => kind switch
        {
            NDExprReduceKind.ArgMax or NDExprReduceKind.ArgMin => NPTypeCode.Int64,
            NDExprReduceKind.Mean or NDExprReduceKind.NanMean => NPTypeCode.Double,
            _ => NPTypeCode.String,
        };

        /// <summary>The reduce ufunc NumPy names in an <c>out=</c> refusal (<c>add</c>, <c>maximum</c>, ...).</summary>
        internal string ReduceUfunc => Kind switch
        {
            NDExprReduceKind.Sum or NDExprReduceKind.Mean or NDExprReduceKind.NanMean
                or NDExprReduceKind.Var or NDExprReduceKind.Std => "add",
            NDExprReduceKind.Prod => "multiply",
            NDExprReduceKind.Min => "minimum",
            NDExprReduceKind.NanMin => "fmin",
            NDExprReduceKind.NanMax => "fmax",
            NDExprReduceKind.ArgMax => "argmax",
            NDExprReduceKind.ArgMin => "argmin",
            _ => "maximum",
        };

        /// <inheritdoc/>
        /// <exception cref="TypeError">NumPy refuses the reduction for the text class (<c>the resolved dtypes are not
        ///     compatible with add.reduce</c> for S / U sums, the no-loop texts of maximum / multiply / divide / subtract /
        ///     fmin / fmax), with NumPy's texts.</exception>
        /// <exception cref="ValueError">A flat StringDType sum / max / min over several axes (<c>reduction operation 'add' is
        ///     not reorderable, so at most one axis may be specified</c>), a None-like NA (<c>Cannot add null ...</c>), or an
        ///     empty reduction without identity.</exception>
        /// <exception cref="ArgumentException">An axis out of range (NumPy's AxisError text).</exception>
        /// <exception cref="InvalidOperationException">Any / All reached here — a lowering bug (they read the truth).</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args)
        {
            var reduced = Reduce(engine, args[0]);
            // A flat reduction's answer is NumPy's SCALAR (an np.int64, a Python str), which the engine's member models as a
            // READ-ONLY 0-d array; np.evaluate answers with a writeable array of its own, as its numeric reductions do.
            if (reduced.Shape.IsWriteable)
                return reduced;
            using (reduced)
                return reduced.copy('K');
        }

        /// <summary>The engine member for the kind, with the node's axis, keepdims and ddof (see <see cref="Execute"/>).</summary>
        /// <param name="engine">The engine.</param>
        /// <param name="x">The reduced text.</param>
        /// <returns>The member's answer (possibly a read-only 0-d stand-in for NumPy's scalar).</returns>
        /// <exception cref="InvalidOperationException">Any / All reached here — a lowering bug.</exception>
        private NDArray Reduce(DefaultEngine engine, NDArray x)
        {
            return Kind switch
            {
                NDExprReduceKind.Sum => engine.ReduceAdd(x, Axis, Keepdims),
                NDExprReduceKind.Prod => engine.ReduceProduct(x, Axis, Keepdims),
                NDExprReduceKind.Min => engine.ReduceAMin(x, Axis, Keepdims),
                NDExprReduceKind.Max => engine.ReduceAMax(x, Axis, Keepdims),
                NDExprReduceKind.Mean => engine.ReduceMean(x, Axis, Keepdims),
                NDExprReduceKind.Ptp => engine.Ptp(x, Axis, null, Keepdims),
                NDExprReduceKind.NanMin => engine.NanMin(x, Axis, Keepdims),
                NDExprReduceKind.NanMax => engine.NanMax(x, Axis, Keepdims),
                NDExprReduceKind.ArgMax => engine.ReduceArgMax(x, Axis, Keepdims),
                NDExprReduceKind.ArgMin => engine.ReduceArgMin(x, Axis, Keepdims),
                NDExprReduceKind.NanMean => engine.NanMean(x, Axis, null, Keepdims),
                NDExprReduceKind.Var => engine.ReduceVar(x, Axis, Keepdims, Ddof),
                NDExprReduceKind.Std => engine.ReduceStd(x, Axis, Keepdims, Ddof),
                _ => throw new InvalidOperationException(
                    $"np.evaluate's {Kind} of text reads the truth through a truth stage, not a reduction stage: a lowering bug."),
            };
        }
    }

    /// <summary>
    ///     E3: a weighted average with text values or weights, through the engine's np.average
    ///     (<see cref="TensorEngine.Average"/>): NumPy's dtype resolution of the values and the weights (a text beside float
    ///     weights is its DTypePromotionError) or its sum (the text's add.reduce refusal) answers, always a refusal.
    /// </summary>
    internal sealed class NDExprAverageStage : NDExprTextStage
    {
        /// <summary>The axis, or null for the whole array.</summary>
        internal readonly int? Axis;

        /// <summary>Whether the reduced axes are kept as length 1.</summary>
        internal readonly bool Keepdims;

        /// <summary>Creates an average stage.</summary>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="keepdims">Whether the reduced axes stay as length 1.</param>
        /// <param name="values">The values.</param>
        /// <param name="weights">The weights.</param>
        internal NDExprAverageStage(int? axis, bool keepdims, NDExprStageOperand values, NDExprStageOperand weights)
            : base("average", NPTypeCode.Double, values, weights)
        {
            Axis = axis;
            Keepdims = keepdims;
        }

        /// <inheritdoc/>
        /// <exception cref="TypeError">The values and weights have no common dtype, or the text sum is refused (NumPy's
        ///     texts).</exception>
        /// <exception cref="ArgumentException">The weights' shape does not fit the values (NumPy's texts).</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args)
            => engine.Average(args[0], Axis is int axis ? new[] { axis } : null, args[1], returned: false, keepdims: Keepdims).avg;
    }
}
