using System;
using System.Collections.Generic;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

// =============================================================================
// NDExpr.Text.cs — text operands in np.evaluate (docs/plans/evaluate-text.md, phase E1)
// =============================================================================
//
// A fused kernel computes on numbers: a text element is not a SIMD lane and has no value on the IL
// stack. So a node that computes ON text never enters the fused kernel. At program build, after the
// arrays are bound, the LOWERING pass below walks the tree in evaluation order and cuts every such
// node out as a STAGE: the engine's own text call over whole arrays (the comparison guard of the
// string ufuncs, exactly what np.equal runs), executed per call before the fused pass. The stage
// leaves a positional input of its result dtype in its place, so the REMAINDER is an ordinary
// numeric tree over the numeric inputs and the stage results — typed, vectorized, cached and run by
// the existing machinery with the call's keywords (trivial loop, NDIter, where=, out= casts, the
// streamed and folded reductions, Compile()).
//
// Why whole-array stages and not a per-block stage inside the fused kernel: measured in the plan
// (§3), the text kernel is 82-97 % of a typical filter, the two designs tie at 100K+, the whole-array
// stage keeps the text kernels' rows route (14-24 % faster on narrow rows) and the block stage wins
// only ~0.5 µs of fixed cost at 1K. The deciding property is that a stage runs code the string
// oracle already gates byte for byte — NumPy's errors, NA rules and descriptor resolution included —
// and that a program never bakes a text width or class: the per-call guard reads them, so the
// program cache needs no descriptor in its key.
//
// Phase E1 lowers the six comparisons. Every other node with a text operand raises
// StringComputationNotSupportedException naming the node when the program is built; each such site
// notes in a "plan:" comment the phase that lifts it (E2: the truth value of text, IsNaN over
// StringDType, the np.strings factories, text -> number casts; E3: text results, text reductions and
// NumPy's own error for every remaining node).
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public abstract partial class NDExpr
    {
        /// <summary>
        ///     A C# <see cref="string"/> as NumPy's Python <c>str</c>: the 0-d <c>&lt;U{len}</c> text array
        ///     <c>np.array(str)</c> builds (<see cref="NDArray.PythonStrLiteral"/>), embedded as an array leaf, so
        ///     <c>NDExpr.Equal(city, "paris")</c> and <c>(NDExpr)city &lt; "m"</c> compare against the text.
        /// </summary>
        /// <param name="text">The text (trailing NULs are kept in the element, as NumPy keeps them up to the width).</param>
        /// <returns>An array leaf over the 0-d text array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="text"/> is null.</exception>
        /// <remarks>
        ///     The text is an INPUT of the evaluation, not IL: one compiled program serves every text value of the same
        ///     tree, so a literal that varies per call compiles nothing new (unlike a numeric literal, which is baked into
        ///     the kernel). Its width is read by the comparison per call. A C# <c>string</c> is always <c>str</c>; compare
        ///     an <c>S</c> array against a 0-d <c>S</c> array (<c>np.array(bytes, dtype: "S")</c> reshaped to 0-d), since
        ///     NumPy's <c>bytes</c> and <c>str</c> never compare (<c>UFuncTypeError</c>).
        /// </remarks>
        public static implicit operator NDExpr(string text)
            => new ArrayNode(NDArray.PythonStrLiteral(text ?? throw new ArgumentNullException(nameof(text))));

        /// <summary>
        ///     Lowers this BOUND node (its array leaves already positional) for a program with text inputs: a node that
        ///     computes on text becomes a stage of <paramref name="ctx"/> and is replaced by the stage's result input; every
        ///     other node is rebuilt over its lowered children in the remainder's input numbering.
        /// </summary>
        /// <param name="ctx">The lowering of one program: its input signature, the stages found so far and the remainder's
        ///     numbering.</param>
        /// <returns>The lowered node, or a text value (an input that is text, which only a stage may consume).</returns>
        /// <exception cref="StringComputationNotSupportedException">A node takes a text operand that this phase does not
        ///     compute (see <see cref="NDExprTextLowering.NotYet"/>).</exception>
        internal abstract NDExprLowered LowerText(NDExprTextLowering ctx);
    }

    /// <summary>
    ///     What lowering one node produced: a node of the numeric remainder (in the remainder's input numbering), or a TEXT
    ///     value — an input of the original signature that holds text, which only a stage can consume.
    /// </summary>
    internal readonly struct NDExprLowered
    {
        /// <summary>The lowered remainder node; null for a text value.</summary>
        internal readonly NDExpr Node;

        /// <summary>The ORIGINAL input index of a text value; -1 for a remainder node.</summary>
        internal readonly int TextInput;

        /// <summary>Creates a lowered value.</summary>
        /// <param name="node">The remainder node, or null for a text value.</param>
        /// <param name="textInput">The original input index of a text value, or -1.</param>
        private NDExprLowered(NDExpr node, int textInput)
        {
            Node = node;
            TextInput = textInput;
        }

        /// <summary>Whether this is a text value (an input that holds text) rather than a remainder node.</summary>
        internal bool IsText => TextInput >= 0;

        /// <summary>A remainder node.</summary>
        /// <param name="node">The node, in the remainder's numbering.</param>
        /// <returns>The lowered value.</returns>
        internal static NDExprLowered Numeric(NDExpr node) => new(node, -1);

        /// <summary>A text value: the original input <paramref name="input"/>, which holds text.</summary>
        /// <param name="input">The original input index.</param>
        /// <returns>The lowered value.</returns>
        internal static NDExprLowered Text(int input) => new(null, input);
    }

    /// <summary>
    ///     The lowering of one program with text inputs (see the file header): the input signature it reads text-ness from,
    ///     the remainder's input numbering (the numeric inputs first, in input order, then one input per stage), and the
    ///     stages in evaluation order.
    /// </summary>
    internal sealed class NDExprTextLowering
    {
        /// <summary>The program's input signature; <see cref="NPTypeCode.String"/> marks a text input.</summary>
        private readonly NPTypeCode[] _inputTypes;

        /// <summary>Per original input: its index among the remainder's inputs, or -1 for a text input.</summary>
        private readonly int[] _remainderIndex;

        /// <summary>The original input index of every numeric input, in input order (the remainder's first inputs).</summary>
        internal readonly List<int> NumericInputs = new();

        /// <summary>The remainder's input dtypes: the numeric inputs', then each stage result's.</summary>
        internal readonly List<NPTypeCode> RemainderTypes = new();

        /// <summary>The stages, in evaluation order (a stage's operands never name a later stage).</summary>
        internal readonly List<NDExprTextStage> Stages = new();

        /// <summary>Creates the lowering of a program over <paramref name="inputTypes"/>.</summary>
        /// <param name="inputTypes">The program's input signature (embedded arrays in binding order, or the positional
        ///     operands).</param>
        internal NDExprTextLowering(NPTypeCode[] inputTypes)
        {
            _inputTypes = inputTypes;
            _remainderIndex = new int[inputTypes.Length];
            for (int i = 0; i < inputTypes.Length; i++)
            {
                if (inputTypes[i] == NPTypeCode.String)
                {
                    _remainderIndex[i] = -1;
                    continue;
                }

                _remainderIndex[i] = NumericInputs.Count;
                NumericInputs.Add(i);
                RemainderTypes.Add(inputTypes[i]);
            }
        }

        /// <summary>Whether original input <paramref name="index"/> holds text (S, U or StringDType).</summary>
        /// <param name="index">The original input index.</param>
        /// <returns>True for a text input.</returns>
        /// <exception cref="InvalidOperationException">The index is outside the signature (a tree / operand-list mismatch,
        ///     reported the way the typing pass reports it).</exception>
        internal bool IsTextInput(int index)
        {
            if ((uint)index >= (uint)_inputTypes.Length)
                throw new InvalidOperationException(
                    $"Input({index}) out of range; typing provided {_inputTypes.Length} inputs.");
            return _inputTypes[index] == NPTypeCode.String;
        }

        /// <summary>
        ///     The remainder's spelling of a numeric input leaf: the same node when its index is unchanged (no text input
        ///     precedes it), else a new <see cref="InputNode"/> at its remainder index.
        /// </summary>
        /// <param name="input">A leaf over a numeric input of the original signature.</param>
        /// <returns>The leaf in the remainder's numbering.</returns>
        internal InputNode Renumber(InputNode input)
        {
            int k = _remainderIndex[input.Index];
            return k == input.Index ? input : new InputNode(k);
        }

        /// <summary>
        ///     Records <paramref name="stage"/> and returns the remainder leaf that reads its result: a new input appended
        ///     after the numeric inputs and every earlier stage, typed <paramref name="resultType"/>.
        /// </summary>
        /// <param name="stage">The stage (its operands already resolved against earlier inputs and stages).</param>
        /// <param name="resultType">The dtype the stage's result always has (bool for a comparison).</param>
        /// <returns>The leaf of the stage's result in the remainder.</returns>
        internal NDExprLowered AddStage(NDExprTextStage stage, NPTypeCode resultType)
        {
            Stages.Add(stage);
            RemainderTypes.Add(resultType);
            return NDExprLowered.Numeric(new InputNode(RemainderTypes.Count - 1));
        }

        /// <summary>
        ///     How a stage reads one of its operands per call: a text value or a numeric input leaf is the input array
        ///     itself; a literal is NumPy's Python scalar (<see cref="ConstNode.ToOperandArray"/>); any other numeric sub-tree
        ///     is a 0-d stand-in of the sub-tree's dtype — enough for a comparison of text with a number, which NumPy refuses
        ///     by the operand CLASSES before reading a value.
        /// </summary>
        /// <param name="lowered">The operand after lowering.</param>
        /// <param name="original">The operand before lowering (its leaf index is in the original numbering).</param>
        /// <returns>The stage operand.</returns>
        internal NDExprStageOperand Operand(in NDExprLowered lowered, NDExpr original)
        {
            if (lowered.IsText)
                return NDExprStageOperand.Input(lowered.TextInput);
            if (original is InputNode leaf)
                return NDExprStageOperand.Input(leaf.Index);
            if (original is ConstNode literal)
                return NDExprStageOperand.Literal(literal);
            // A computed numeric operand: its NumPy dtype over the remainder's inputs typed so far (every stage result it
            // reads is already recorded, stages being found in evaluation order).
            var dtype = lowered.Node.ResolveNumPyTypes(RemainderTypes.ToArray(), out _);
            return NDExprStageOperand.StandIn(dtype);
        }

        /// <summary>
        ///     The refusal of a node that takes a text operand np.evaluate does not compute: names the node, states the rule
        ///     (text computes in the six comparisons) and points at the functions that compute it. Each call site notes, as a
        ///     <c>plan:</c> comment, the phase of docs/plans/evaluate-text.md that lifts it.
        /// </summary>
        /// <param name="node">The node as the factory spells it (<c>NDExpr.Add</c>, <c>NDExpr.Where's condition</c>, ...).</param>
        /// <returns>The exception (not thrown).</returns>
        internal static StringComputationNotSupportedException NotYet(string node)
            => NDStringUfuncs.EvaluateTextNotYet(node);
    }

    /// <summary>How a stage reads one operand per call.</summary>
    internal enum NDExprStageOperandKind : byte
    {
        /// <summary>An input array of the call (text or numeric).</summary>
        Input,

        /// <summary>A literal of the tree, as NumPy's Python scalar (a C# number literal is a weak operand).</summary>
        Literal,

        /// <summary>A computed numeric sub-tree, as a 0-d array of its dtype (only its class is ever read).</summary>
        StandIn,
    }

    /// <summary>One operand of a stage: an input array, a literal or a dtype stand-in (<see cref="NDExprStageOperandKind"/>).</summary>
    internal readonly struct NDExprStageOperand
    {
        /// <summary>How the operand is read.</summary>
        internal readonly NDExprStageOperandKind Kind;

        /// <summary>The original input index (<see cref="NDExprStageOperandKind.Input"/>).</summary>
        internal readonly int Index;

        /// <summary>The literal (<see cref="NDExprStageOperandKind.Literal"/>).</summary>
        internal readonly ConstNode Value;

        /// <summary>The stand-in's dtype (<see cref="NDExprStageOperandKind.StandIn"/>).</summary>
        internal readonly NPTypeCode Dtype;

        /// <summary>Creates an operand.</summary>
        /// <param name="kind">How it is read.</param>
        /// <param name="index">The input index, for an input.</param>
        /// <param name="value">The literal, for a literal.</param>
        /// <param name="dtype">The stand-in's dtype, for a stand-in.</param>
        private NDExprStageOperand(NDExprStageOperandKind kind, int index, ConstNode value, NPTypeCode dtype)
        {
            Kind = kind;
            Index = index;
            Value = value;
            Dtype = dtype;
        }

        /// <summary>An input array of the call.</summary>
        /// <param name="index">The original input index.</param>
        /// <returns>The operand.</returns>
        internal static NDExprStageOperand Input(int index) => new(NDExprStageOperandKind.Input, index, null, NPTypeCode.Empty);

        /// <summary>A literal of the tree.</summary>
        /// <param name="value">The literal node.</param>
        /// <returns>The operand.</returns>
        internal static NDExprStageOperand Literal(ConstNode value) => new(NDExprStageOperandKind.Literal, -1, value, NPTypeCode.Empty);

        /// <summary>A 0-d stand-in of a computed numeric sub-tree.</summary>
        /// <param name="dtype">The sub-tree's NumPy dtype.</param>
        /// <returns>The operand.</returns>
        internal static NDExprStageOperand StandIn(NPTypeCode dtype) => new(NDExprStageOperandKind.StandIn, -1, null, dtype);

        /// <summary>
        ///     The array this operand stands for in one call.
        /// </summary>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="owned">True when the array was made here (a literal's or stand-in's 0-d array), which the caller
        ///     must dispose; false for a caller's input.</param>
        /// <returns>The array.</returns>
        internal NDArray Resolve(NDArray[] inputs, out bool owned)
        {
            switch (Kind)
            {
                case NDExprStageOperandKind.Input:
                    owned = false;
                    return inputs[Index];
                case NDExprStageOperandKind.Literal:
                    owned = true;
                    return Value.ToOperandArray();
                default:
                    owned = true;
                    return new NDArray(Dtype, Shape.NewScalar(), true);
            }
        }
    }

    /// <summary>
    ///     One node that computes on text, run per call as the engine's own text call over whole arrays (phase E1: the six
    ///     comparisons, through <see cref="NDStringUfuncs.BinaryGuard"/> — the guard <c>np.equal</c> and its siblings run,
    ///     which computes NumPy's result or raises NumPy's error).
    /// </summary>
    internal sealed class NDExprTextStage
    {
        /// <summary>NumPy's ufunc name (<c>"equal"</c> … <c>"greater_equal"</c>).</summary>
        internal readonly string Ufunc;

        /// <summary>The first operand.</summary>
        internal readonly NDExprStageOperand Left;

        /// <summary>The second operand.</summary>
        internal readonly NDExprStageOperand Right;

        /// <summary>Creates a comparison stage.</summary>
        /// <param name="ufunc">NumPy's ufunc name of the comparison.</param>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        internal NDExprTextStage(string ufunc, NDExprStageOperand left, NDExprStageOperand right)
        {
            Ufunc = ufunc;
            Left = left;
            Right = right;
        }

        /// <summary>
        ///     Computes this stage for one call: <c>np.{ufunc}(x1, x2)</c> over the resolved operands, as a fresh array the
        ///     caller owns (NumPy's own result: dtype, shape and layout).
        /// </summary>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <returns>The stage's result (bool for a comparison).</returns>
        /// <exception cref="TypeError">NumPy has no loop for the operand pair (S with U, text with a number: its
        ///     <c>UFuncTypeError</c> text naming the operand classes).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ufunc text).</exception>
        /// <exception cref="ValueError">A StringDType ordering meets an NA of a <c>None</c>-like descriptor, or two
        ///     StringDType operands carry incompatible NA objects (NumPy's texts).</exception>
        /// <exception cref="InvalidOperationException">Neither operand is text — a lowering bug (a stage is only made for a
        ///     node with a text operand).</exception>
        internal NDArray Run(NDArray[] inputs)
        {
            var x1 = Left.Resolve(inputs, out bool own1);
            NDArray x2 = null;
            bool own2 = false;
            try
            {
                x2 = Right.Resolve(inputs, out own2);
                // The guard answers every pair with a text operand: the comparison of two strings (S/S, U/U, T/T, T/U, U/T,
                // any widths and NA objects), or NumPy's refusal; it returns null only when neither operand is text.
                return NDStringUfuncs.BinaryGuard(Ufunc, x1, x2)
                       ?? throw new InvalidOperationException(
                           $"np.evaluate's '{Ufunc}' stage has no text operand ({x1.dtype}, {x2.dtype}): a lowering bug.");
            }
            finally
            {
                // A literal's or stand-in's 0-d array is this call's own; the inputs are the caller's.
                if (own1)
                    x1.Dispose();
                if (own2)
                    x2?.Dispose();
            }
        }
    }

    /// <summary>
    ///     The text half of a program (see the file header): the stages in evaluation order and the numeric remainder over
    ///     the numeric inputs and the stage results. Immutable after construction and shared by every call of its program.
    /// </summary>
    internal sealed class NDExprTextPlan
    {
        /// <summary>The stages, in evaluation order.</summary>
        internal readonly NDExprTextStage[] Stages;

        /// <summary>The remainder tree: positional leaves, the numeric inputs first (<see cref="NumericInputs"/>), then one per
        ///     stage result in stage order. Its per-root program slot caches the remainder's program across calls.</summary>
        internal readonly NDExpr Remainder;

        /// <summary>The original input index of each of the remainder's numeric inputs, in order.</summary>
        internal readonly int[] NumericInputs;

        /// <summary>The remainder's input dtypes (the numeric inputs', then each stage result's).</summary>
        internal readonly NPTypeCode[] RemainderTypes;

        /// <summary>Creates the plan.</summary>
        /// <param name="stages">The stages in evaluation order.</param>
        /// <param name="remainder">The remainder tree.</param>
        /// <param name="numericInputs">The original indices of the remainder's numeric inputs.</param>
        /// <param name="remainderTypes">The remainder's input dtypes.</param>
        internal NDExprTextPlan(NDExprTextStage[] stages, NDExpr remainder, int[] numericInputs, NPTypeCode[] remainderTypes)
        {
            Stages = stages;
            Remainder = remainder;
            NumericInputs = numericInputs;
            RemainderTypes = remainderTypes;
        }

        /// <summary>Whether the remainder's root is a reduction (a <see cref="ReduceNode"/> or a weighted average).</summary>
        internal bool IsReduction => Remainder is ReduceNode or WeightedAverageNode;

        /// <summary>
        ///     Whether the remainder is nothing but one stage's result (<c>np.evaluate(NDExpr.Equal(a, b))</c>): that stage's
        ///     array is then the evaluation's result as it is, when no keyword shapes the result.
        /// </summary>
        /// <param name="stage">The stage whose result is the whole evaluation, when true.</param>
        /// <returns>True when the remainder is a bare stage-result leaf.</returns>
        internal bool IsBareStageResult(out int stage)
        {
            if (Remainder is InputNode leaf && leaf.Index >= NumericInputs.Length)
            {
                stage = leaf.Index - NumericInputs.Length;
                return true;
            }

            stage = -1;
            return false;
        }

        /// <summary>The remainder's inputs for one call: the numeric inputs, then the stage results.</summary>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="results">The stage results of this call, in stage order.</param>
        /// <returns>A new array of the remainder's inputs (the stage results stay owned by the caller).</returns>
        internal NDArray[] RemainderInputs(NDArray[] inputs, NDArray[] results)
        {
            var ops = new NDArray[NumericInputs.Length + results.Length];
            for (int i = 0; i < NumericInputs.Length; i++)
                ops[i] = inputs[NumericInputs[i]];
            Array.Copy(results, 0, ops, NumericInputs.Length, results.Length);
            return ops;
        }

        /// <summary>
        ///     Lowers a bound tree with text inputs into a plan.
        /// </summary>
        /// <param name="bound">The bound tree (positional leaves).</param>
        /// <param name="inputTypes">The program's input signature (at least one <see cref="NPTypeCode.String"/>).</param>
        /// <returns>The plan.</returns>
        /// <exception cref="StringComputationNotSupportedException">A node takes a text operand this phase does not compute,
        ///     or the root itself is a text input (a text result).</exception>
        internal static NDExprTextPlan Lower(NDExpr bound, NPTypeCode[] inputTypes)
        {
            var ctx = new NDExprTextLowering(inputTypes);
            var root = bound.LowerText(ctx);
            if (root.IsText)
                throw NDExprTextLowering.NotYet("a text result (a bare text leaf as the expression)" /* plan: E3, text results */);
            return new NDExprTextPlan(ctx.Stages.ToArray(), root.Node, ctx.NumericInputs.ToArray(), ctx.RemainderTypes.ToArray());
        }
    }

    // =========================================================================
    // Per-node lowering (mirrors BindArrays' child order: the stages are found in evaluation order)
    // =========================================================================

    public sealed partial class InputNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
            => ctx.IsTextInput(_index) ? NDExprLowered.Text(_index) : NDExprLowered.Numeric(ctx.Renumber(this));
    }

    public sealed partial class ConstNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx) => NDExprLowered.Numeric(this);

        /// <summary>
        ///     This literal as the 0-d operand NumPy's ufunc would see: a Python <c>bool</c> / <c>int</c> / <c>float</c> /
        ///     <c>complex</c> (marked weak), or the NumPy scalar a <c>Half</c> / <c>char</c> stands for, or a strong decimal —
        ///     the marks <see cref="NDArray.MarkAsCSharpLiteral{TArray}"/> gives a C# literal, so a refusal names the class
        ///     NumPy names (<c>_PyLongDType</c> for a Python int).
        /// </summary>
        /// <returns>A new 0-d array the caller owns.</returns>
        internal NDArray ToOperandArray() => _kind switch
        {
            NDExprLiteralKind.Bool => NDArray.MarkAsCSharpLiteral(NDArray.Scalar(_i != 0)),
            NDExprLiteralKind.Int => NDArray.MarkAsCSharpLiteral(NDArray.Scalar(_i)),
            NDExprLiteralKind.UInt64 => NDArray.MarkAsCSharpLiteral(NDArray.Scalar(_u)),
            NDExprLiteralKind.Float => NDArray.MarkAsCSharpLiteral(NDArray.Scalar(_f)),
            NDExprLiteralKind.Complex => NDArray.MarkAsCSharpLiteral(NDArray.Scalar(_c)),
            NDExprLiteralKind.Half => NDArray.MarkAsCSharpLiteral(NDArray.Scalar((Half)_f)),
            NDExprLiteralKind.Decimal => NDArray.MarkAsCSharpLiteral(NDArray.Scalar(_m)),
            _ => NDArray.MarkAsCSharpLiteral(NDArray.Scalar((char)_i)),
        };
    }

    public sealed partial class BinaryNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            // add (the concatenation) and multiply (the StringDType repeat) compute on text in NumPy, every other binary
            // ufunc refuses it; both answers come with E3.
            if (l.IsText || r.IsText)
                throw NDExprTextLowering.NotYet($"NDExpr.{_op}" /* plan: E3, text results and NumPy's own refusals */);
            return NDExprLowered.Numeric(ReferenceEquals(l.Node, _left) && ReferenceEquals(r.Node, _right)
                ? this
                : new BinaryNode(_op, l.Node, r.Node));
        }
    }

    public sealed partial class UnaryNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _child.LowerText(ctx);
            if (c.IsText)
                throw _op == UnaryOp.IsNan
                    ? NDExprTextLowering.NotYet("NDExpr.IsNaN" /* plan: E2, IsNaN over StringDType */)
                    : NDExprTextLowering.NotYet($"NDExpr.{_op}" /* plan: E3, NumPy's own refusals */);
            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child) ? this : new UnaryNode(_op, c.Node));
        }
    }

    public sealed partial class CastNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _child.LowerText(ctx);
            if (c.IsText)
                throw NDExprTextLowering.NotYet("NDExpr.Cast" /* plan: E2, text -> number casts */);
            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child) ? this : new CastNode(c.Node, _target));
        }
    }

    public sealed partial class RoundNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _child.LowerText(ctx);
            if (c.IsText)
                throw NDExprTextLowering.NotYet("NDExpr.Round" /* plan: E3, NumPy's own refusals */);
            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child) ? this : new RoundNode(c.Node, _decimals));
        }
    }

    public sealed partial class ComparisonNode
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     A comparison with a text operand is a stage whatever the other operand is: two strings compare, and text
        ///     against a number reaches NumPy's own refusal through the same guard per call. The factories' truth test
        ///     (<see cref="IsTruthTest"/>, the <c>x != 0</c> of <c>Any</c> / <c>All</c> / <c>CountNonzero</c>) is NOT a
        ///     comparison in NumPy (<c>np.count_nonzero(text)</c> reads the truth, <c>text != 0</c> has no loop), so it is
        ///     refused until E2.
        /// </remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            if (!l.IsText && !r.IsText)
                return NDExprLowered.Numeric(ReferenceEquals(l.Node, _left) && ReferenceEquals(r.Node, _right)
                    ? this
                    : new ComparisonNode(_op, l.Node, r.Node, _truthTest));
            if (_truthTest)
                throw NDExprTextLowering.NotYet("NDExpr.Any / All / CountNonzero" /* plan: E2, the truth value of text */);
            var stage = new NDExprTextStage(UfuncName(_op), ctx.Operand(l, _left), ctx.Operand(r, _right));
            return ctx.AddStage(stage, NPTypeCode.Boolean);
        }

        /// <summary>NumPy's ufunc name of a comparison.</summary>
        /// <param name="op">The comparison.</param>
        /// <returns><c>equal</c>, <c>not_equal</c>, <c>less</c>, <c>less_equal</c>, <c>greater</c> or <c>greater_equal</c>.</returns>
        internal static string UfuncName(ComparisonOp op) => op switch
        {
            ComparisonOp.Equal => "equal",
            ComparisonOp.NotEqual => "not_equal",
            ComparisonOp.Less => "less",
            ComparisonOp.LessEqual => "less_equal",
            ComparisonOp.Greater => "greater",
            _ => "greater_equal",
        };
    }

    public sealed partial class MinMaxNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            if (l.IsText || r.IsText)
                throw NDExprTextLowering.NotYet(_clipSelection ? "NDExpr.Clip" : _isMin ? "NDExpr.Minimum" : "NDExpr.Maximum" /* plan: E3, text results and NumPy's own refusals */);
            return NDExprLowered.Numeric(ReferenceEquals(l.Node, _left) && ReferenceEquals(r.Node, _right)
                ? this
                : new MinMaxNode(_isMin, l.Node, r.Node, _clipSelection));
        }
    }

    public sealed partial class WhereNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _cond.LowerText(ctx);
            var a = _a.LowerText(ctx);
            var b = _b.LowerText(ctx);
            if (c.IsText)
                throw NDExprTextLowering.NotYet("NDExpr.Where's condition" /* plan: E2, the truth value of text */);
            if (a.IsText || b.IsText)
                throw NDExprTextLowering.NotYet("NDExpr.Where with text branches" /* plan: E3, text results */);
            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _cond) && ReferenceEquals(a.Node, _a) && ReferenceEquals(b.Node, _b)
                ? this
                : new WhereNode(c.Node, a.Node, b.Node));
        }
    }

    public sealed partial class LogicalNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            if (l.IsText || r.IsText)
                throw NDExprTextLowering.NotYet($"NDExpr.Logical{_op}" /* plan: E2, the truth value of text */);
            return NDExprLowered.Numeric(ReferenceEquals(l.Node, _left) && ReferenceEquals(r.Node, _right)
                ? this
                : new LogicalNode(_op, l.Node, r.Node));
        }
    }

    public sealed partial class CallNode
    {
        /// <inheritdoc/>
        /// <remarks>A delegate's parameters are numbers, so a text argument is refused in every phase: compute the text
        ///     part first (a comparison stage, or the string functions) and pass its numeric result.</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            NDExpr[]? lowered = null;
            for (int i = 0; i < _args.Length; i++)
            {
                var a = _args[i].LowerText(ctx);
                if (a.IsText)
                    throw NDStringUfuncs.EvaluateTextCallArgument();
                if (!ReferenceEquals(a.Node, _args[i]) && lowered is null)
                {
                    lowered = new NDExpr[_args.Length];
                    Array.Copy(_args, lowered, i);
                }

                if (lowered is not null)
                    lowered[i] = a.Node;
            }

            return NDExprLowered.Numeric(lowered is null ? this : new CallNode(this, lowered));
        }
    }

    public sealed partial class ArrayNode
    {
        /// <inheritdoc/>
        /// <exception cref="InvalidOperationException">Always: lowering runs on a BOUND tree, where every array leaf is
        ///     already positional.</exception>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
            => throw new InvalidOperationException("ArrayNode must be bound before the text lowering — a program-build bug.");
    }

    public sealed partial class ReduceNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _child.LowerText(ctx);
            // A bare text child: argmax / argmin of S / U / T and max / min / sum of StringDType compute in NumPy, every other
            // reduction refuses text; both answers come with E3. (Any / All / CountNonzero never get here with a text child:
            // their truth test refuses first.)
            if (c.IsText)
                throw NDExprTextLowering.NotYet($"NDExpr.{_kind}" /* plan: E3, text reductions and NumPy's own refusals */);
            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child) ? this : new ReduceNode(_kind, c.Node, _axis, _keepdims, _ddof));
        }
    }

    public sealed partial class WeightedAverageNode
    {
        /// <inheritdoc/>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var v = _values.LowerText(ctx);
            var w = _weights.LowerText(ctx);
            if (v.IsText || w.IsText)
                throw NDExprTextLowering.NotYet("NDExpr.Average" /* plan: E3, NumPy's own refusals */);
            return NDExprLowered.Numeric(ReferenceEquals(v.Node, _values) && ReferenceEquals(w.Node, _weights)
                ? this
                : new WeightedAverageNode(v.Node, w.Node, _axis, _keepdims));
        }
    }
}
