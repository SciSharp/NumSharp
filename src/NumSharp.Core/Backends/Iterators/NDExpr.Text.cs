using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

// =============================================================================
// NDExpr.Text.cs — text in np.evaluate (docs/plans/evaluate-text.md, phases E1, E2 and E3)
// =============================================================================
//
// A fused kernel computes on numbers: a text element is not a SIMD lane and has no value on the IL
// stack. So a node that computes ON text never enters the fused kernel. At program build, after the
// arrays are bound, the LOWERING pass below walks the tree in evaluation order and cuts every such
// node out as a STAGE: the engine's own text call over whole arrays, executed per call before the
// fused pass. A stage with a numeric result leaves a positional input of its result dtype in its place,
// so the REMAINDER is an ordinary numeric tree over the numeric inputs and those stage results — typed,
// vectorized, cached and run by the existing machinery with the call's keywords (trivial loop, NDIter,
// where=, out= casts, the streamed and folded reductions, Compile()). A stage with a TEXT result (E3)
// leaves a text value instead, which only later stages read: the operand of a comparison, of a
// numpy.strings function, of another text node — or the evaluation's own result, the TEXT ROOT, cast
// and copied into the call's keywords by the engine (DefaultEngine.Evaluate.Text.cs).
//
// Why whole-array stages and not a per-block stage inside the fused kernel: measured in the plan
// (§3), the text kernel is 82-97 % of a typical filter, the two designs tie at 100K+, the whole-array
// stage keeps the text kernels' rows route (8-24 % faster on narrow rows) and the block stage wins
// only ~0.5 µs of fixed cost at 1K. The deciding property is that a stage runs code the string
// oracle already gates byte for byte — NumPy's errors, NA rules and descriptor resolution included —
// and that a program never bakes a text width or class: the per-call stage reads them, so the
// program cache needs no descriptor in its key.
//
// The stages (each one NumPy statement over whole arrays):
//   - E1, the six comparisons: NDStringUfuncs.BinaryGuard, np.equal's own guard.
//   - E2, the truth value of text: NumPy reads a string's truth through its bool CAST (the logical
//     ufuncs, np.where's condition, np.any / np.all) or through the dtype's NONZERO function
//     (np.count_nonzero without an axis); a TRUTH stage computes that bool array, and the logical node,
//     the where or the reduction stays in the fused remainder over it. NumPy casts inside its iterator,
//     so the truth array must not change the layout the original operand gives the result: a broadcast
//     operand's truth is computed once per distinct element and broadcast back (stride 0 where the text
//     had stride 0), not materialized densely.
//   - E2, IsNaN: the engine's np.isnan (StringDType's NaN-like NA; S / U raise NumPy's error).
//   - E2, Cast to a number: astype, NumPy's parse (`x.astype(int64)` is its own statement in NumPy).
//   - E2, the numpy.strings functions (NDExpr.Strings.cs): str_len, the nine predicates, the searches.
//   - E3, every other node with a text operand, as the engine's member for NumPy's own statement
//     (NDExpr.TextResults.cs): the binary ufuncs (np.add's concatenation, np.multiply's repeat, and
//     NumPy's refusal for the rest), np.maximum / np.minimum (StringDType), np.clip, np.where with text
//     branches (its promotion), the unary ufuncs (np.real / np.imag compute, the rest refuse), np.round,
//     a cast to or from text (astype), the root reductions (np.argmax / np.argmin over S / U / T,
//     np.sum / np.max / np.min over StringDType, NumPy's refusal or answer for every other kind) and
//     np.average; a bare text leaf as the whole expression is a copy of it.
// A stage operand is an input array, a literal (NumPy's Python scalar), an earlier stage's result (text
// or numbers), or a COMPUTED numeric sub-tree, evaluated per call into the temporary NumPy's unfused
// sequence would hold (its K-order layout), so the stage meets NumPy's operand — and the sub-tree's own
// errors come first, as in NumPy's statement order.
//
// Only a delegate (NDExpr.Call) refuses text, in every phase: its parameters are numbers, and NumPy has
// no such node to follow (StringComputationNotSupportedException). Every other answer, value or error,
// is NumPy's own, raised per call when the statement runs.
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
        ///     computes on text becomes a stage of <paramref name="ctx"/> and is replaced by the stage's result (a remainder
        ///     input for a numeric result, a text value for a text one); every other node is rebuilt over its lowered children
        ///     in the remainder's input numbering.
        /// </summary>
        /// <param name="ctx">The lowering of one program: its input signature, the stages found so far and the remainder's
        ///     numbering.</param>
        /// <returns>The lowered node, or a text value (a text input or a text stage result, which only a stage — or the
        ///     evaluation's own result — may consume).</returns>
        /// <exception cref="StringComputationNotSupportedException">A delegate (<c>NDExpr.Call</c>) gets a text argument: its
        ///     parameters are numbers in every phase.</exception>
        /// <exception cref="InvalidOperationException">An unbound array leaf or a truth test the reduction did not lower — a
        ///     program-build bug, never a user's error.</exception>
        /// <remarks>Every other answer about text — NumPy's value or NumPy's error — comes from the stage when the call runs,
        ///     in NumPy's statement order, never from the lowering.</remarks>
        internal abstract NDExprLowered LowerText(NDExprTextLowering ctx);
    }

    /// <summary>
    ///     What lowering one node produced: a node of the numeric remainder (in the remainder's input numbering), or a TEXT
    ///     value — an input of the original signature that holds text, or the result of a stage whose result is text — which
    ///     only a stage (or the evaluation's own result) can consume.
    /// </summary>
    internal readonly struct NDExprLowered
    {
        /// <summary>The lowered remainder node; null for a text value.</summary>
        internal readonly NDExpr Node;

        /// <summary>The ORIGINAL input index of a text input; -1 otherwise.</summary>
        internal readonly int TextInput;

        /// <summary>The index of the stage whose TEXT result this value is; -1 otherwise.</summary>
        internal readonly int TextStage;

        /// <summary>Creates a lowered value.</summary>
        /// <param name="node">The remainder node, or null for a text value.</param>
        /// <param name="textInput">The original input index of a text input, or -1.</param>
        /// <param name="textStage">The stage index of a text stage result, or -1.</param>
        private NDExprLowered(NDExpr node, int textInput, int textStage)
        {
            Node = node;
            TextInput = textInput;
            TextStage = textStage;
        }

        /// <summary>Whether this is a text value (a text input or a text stage result) rather than a remainder node.</summary>
        internal bool IsText => TextInput >= 0 || TextStage >= 0;

        /// <summary>A remainder node.</summary>
        /// <param name="node">The node, in the remainder's numbering.</param>
        /// <returns>The lowered value.</returns>
        internal static NDExprLowered Numeric(NDExpr node) => new(node, -1, -1);

        /// <summary>A text value: the original input <paramref name="input"/>, which holds text.</summary>
        /// <param name="input">The original input index.</param>
        /// <returns>The lowered value.</returns>
        internal static NDExprLowered Text(int input) => new(null, input, -1);

        /// <summary>A text value: the text result of stage <paramref name="stage"/>.</summary>
        /// <param name="stage">The stage index.</param>
        /// <returns>The lowered value.</returns>
        internal static NDExprLowered TextStageResult(int stage) => new(null, -1, stage);
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

        /// <summary>The remainder's input dtypes: the numeric inputs', then each NUMERIC stage result's, in stage order.</summary>
        internal readonly List<NPTypeCode> RemainderTypes = new();

        /// <summary>Per remainder input after the numeric inputs: the index of the stage whose result it is (a text stage
        ///     has no remainder input, so the two numberings part as soon as one stage yields text).</summary>
        internal readonly List<int> RemainderStages = new();

        /// <summary>The stages, in evaluation order (a stage's operands never name a later stage).</summary>
        internal readonly List<NDExprTextStage> Stages = new();

        /// <summary>The stage a root reduction became (a reduction or a weighted average of text, root-only like every
        ///     reduction), or -1: the call validates <c>out=</c> by the reduction's rules and refuses the elementwise
        ///     keywords for it.</summary>
        internal int RootReductionStage = -1;

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
        ///     Records <paramref name="stage"/> and returns what reads its result: for a numeric result, the remainder leaf of
        ///     a new input appended after the numeric inputs and every earlier numeric stage result, typed with the stage's
        ///     result dtype; for a TEXT result (E3), a text value only later stages — or the evaluation's own result — read.
        /// </summary>
        /// <param name="stage">The stage (its operands already resolved against earlier inputs and stages).</param>
        /// <returns>The leaf of the stage's numeric result in the remainder, or its text value.</returns>
        internal NDExprLowered AddStage(NDExprTextStage stage)
        {
            Stages.Add(stage);
            int index = Stages.Count - 1;
            if (stage.ResultIsText)
                return NDExprLowered.TextStageResult(index);
            RemainderTypes.Add(stage.ResultType);
            RemainderStages.Add(index);
            return NDExprLowered.Numeric(new InputNode(RemainderTypes.Count - 1));
        }

        /// <summary>
        ///     Records the stage a ROOT reduction became (a text reduction or weighted average): the call checks the
        ///     reduction keywords and validates <c>out=</c> the way a reduction does (<see cref="RootReductionStage"/>).
        /// </summary>
        /// <param name="stage">The reduction stage.</param>
        /// <returns>What reads its result (see <see cref="AddStage"/>).</returns>
        internal NDExprLowered AddRootReduction(NDExprTextStage stage)
        {
            RootReductionStage = Stages.Count;
            return AddStage(stage);
        }

        /// <summary>
        ///     The truth stage of a text value (see the file header): records it and returns the remainder leaf of its bool
        ///     result, which the logical node, the where or the reduction then reads like any bool operand.
        /// </summary>
        /// <param name="text">A lowered text value (a text input or an earlier stage's text result).</param>
        /// <param name="rule">The route NumPy's reader takes to the truth (<see cref="NDExprTruthRule"/>).</param>
        /// <returns>The bool leaf in the remainder.</returns>
        internal NDExpr Truth(in NDExprLowered text, NDExprTruthRule rule)
            => AddStage(new NDExprTruthStage(rule, Operand(text, null))).Node;

        /// <summary>
        ///     How a stage reads one of its operands per call: a text input or a numeric input leaf is the input array itself;
        ///     a literal is NumPy's Python scalar (<see cref="ConstNode.ToOperandArray"/>); an earlier stage's result (text or
        ///     numbers) is that array; any other numeric sub-tree is COMPUTED per call into the temporary NumPy's unfused
        ///     sequence would hold (so the stage reads its values and layout, and the sub-tree's own errors come first).
        /// </summary>
        /// <param name="lowered">The operand after lowering.</param>
        /// <param name="original">The operand before lowering (its leaf index is in the original numbering), or null for a
        ///     text value, which is read through <paramref name="lowered"/> alone.</param>
        /// <returns>The stage operand.</returns>
        internal NDExprStageOperand Operand(in NDExprLowered lowered, NDExpr original)
        {
            if (lowered.TextInput >= 0)
                return NDExprStageOperand.Input(lowered.TextInput);
            if (lowered.TextStage >= 0)
                return NDExprStageOperand.StageResult(lowered.TextStage);
            if (original is InputNode leaf)
                return NDExprStageOperand.Input(leaf.Index);
            if (original is ConstNode literal)
                return NDExprStageOperand.Literal(literal);
            // A node lowered into a stage-result leaf (a comparison, a truth test, a strings function) reads that result.
            if (lowered.Node is InputNode stageLeaf && stageLeaf.Index >= NumericInputs.Count)
                return NDExprStageOperand.StageResult(RemainderStages[stageLeaf.Index - NumericInputs.Count]);
            return NDExprStageOperand.Computed(lowered.Node);
        }

        /// <summary>
        ///     The operand of an optional argument (a search's <c>start</c> / <c>end</c>): absent when the factory got null,
        ///     else <see cref="Operand"/> of its lowering.
        /// </summary>
        /// <param name="original">The argument before lowering, or null.</param>
        /// <returns>The stage operand (<see cref="NDExprStageOperandKind.Absent"/> for null).</returns>
        /// <exception cref="StringComputationNotSupportedException">The argument holds a delegate with a text argument (its
        ///     lowering refuses).</exception>
        internal NDExprStageOperand OptionalOperand(NDExpr original)
        {
            if (original is null)
                return NDExprStageOperand.Absent;
            var lowered = original.LowerText(this);
            return Operand(lowered, original);
        }
    }

    /// <summary>How a stage reads one operand per call.</summary>
    internal enum NDExprStageOperandKind : byte
    {
        /// <summary>An input array of the call (text or numeric), by its original input index.</summary>
        Input,

        /// <summary>A literal of the tree, as NumPy's Python scalar (a C# number literal is a weak operand).</summary>
        Literal,

        /// <summary>An optional argument the factory left out (NumPy's default: <c>start=0</c>, <c>end=None</c>).</summary>
        Absent,

        /// <summary>The result of an earlier stage of the same call, by stage index.</summary>
        StageResult,

        /// <summary>A numeric sub-tree, evaluated per call over the remainder's inputs it reads (see
        ///     <see cref="NDExprStageOperand.Computed"/>).</summary>
        Computed,
    }

    /// <summary>
    ///     One operand of a stage (<see cref="NDExprStageOperandKind"/>). Plain data shared by every call of its program;
    ///     the engine resolves it per call (DefaultEngine.Evaluate.Text.cs), because a computed operand needs the evaluator.
    /// </summary>
    internal readonly struct NDExprStageOperand
    {
        /// <summary>How the operand is read.</summary>
        internal readonly NDExprStageOperandKind Kind;

        /// <summary>The original input index (<see cref="NDExprStageOperandKind.Input"/>) or the stage index
        ///     (<see cref="NDExprStageOperandKind.StageResult"/>).</summary>
        internal readonly int Index;

        /// <summary>The literal (<see cref="NDExprStageOperandKind.Literal"/>).</summary>
        internal readonly ConstNode Value;

        /// <summary>The computed sub-tree over positional leaves <c>0 .. Inputs.Length - 1</c>
        ///     (<see cref="NDExprStageOperandKind.Computed"/>).</summary>
        internal readonly NDExpr Tree;

        /// <summary>For <see cref="Tree"/>: the remainder input behind each of its leaves (a numeric input's remainder index,
        ///     or <c>NumericInputs + j</c> for the j-th numeric stage result, <see cref="NDExprTextPlan.RemainderStages"/>), in
        ///     leaf order. Empty for a constant-only tree.</summary>
        internal readonly int[] Inputs;

        /// <summary>For a constant-only <see cref="Tree"/>: whether its NumPy value is a WEAK Python scalar (an arithmetic
        ///     expression of literals: NumPy sees the Python number it computes), so the evaluated 0-d is marked as one.</summary>
        internal readonly bool WeakConstant;

        /// <summary>Stores the parts; only the factories build an operand.</summary>
        /// <param name="kind">How it is read.</param>
        /// <param name="index">The input or stage index, or -1.</param>
        /// <param name="value">The literal, or null.</param>
        /// <param name="tree">The computed sub-tree, or null.</param>
        /// <param name="inputs">The computed sub-tree's input map, or null.</param>
        /// <param name="weakConstant">Whether a constant-only computed tree is a weak Python scalar.</param>
        private NDExprStageOperand(NDExprStageOperandKind kind, int index, ConstNode value, NDExpr tree, int[] inputs,
                                   bool weakConstant)
        {
            Kind = kind;
            Index = index;
            Value = value;
            Tree = tree;
            Inputs = inputs;
            WeakConstant = weakConstant;
        }

        /// <summary>An input array of the call.</summary>
        /// <param name="index">The original input index.</param>
        /// <returns>The operand.</returns>
        internal static NDExprStageOperand Input(int index) => new(NDExprStageOperandKind.Input, index, null, null, null, false);

        /// <summary>A literal of the tree.</summary>
        /// <param name="value">The literal node.</param>
        /// <returns>The operand.</returns>
        internal static NDExprStageOperand Literal(ConstNode value) => new(NDExprStageOperandKind.Literal, -1, value, null, null, false);

        /// <summary>An omitted optional argument (resolves to null: the engine function applies NumPy's default).</summary>
        internal static NDExprStageOperand Absent => new(NDExprStageOperandKind.Absent, -1, null, null, null, false);

        /// <summary>An earlier stage's result.</summary>
        /// <param name="stage">The stage index.</param>
        /// <returns>The operand.</returns>
        internal static NDExprStageOperand StageResult(int stage) => new(NDExprStageOperandKind.StageResult, stage, null, null, null, false);

        /// <summary>
        ///     A computed numeric sub-tree: renumbered so that its leaves are <c>0 .. m-1</c> in first-reference order, and
        ///     evaluated per call over exactly those <c>m</c> remainder inputs (an input it does not read joins neither its
        ///     broadcast nor its iterator, as NumPy's statement over its own operands).
        /// </summary>
        /// <param name="lowered">The sub-tree in the remainder's numbering (numeric only: it was lowered).</param>
        /// <returns>The operand.</returns>
        internal static NDExprStageOperand Computed(NDExpr lowered)
        {
            var map = new List<int>();
            var tree = lowered.BindArrays(new NDExprBindContext
            {
                InputRewrite = leaf =>
                {
                    int k = map.IndexOf(leaf.Index);
                    if (k < 0)
                    {
                        map.Add(leaf.Index);
                        k = map.Count - 1;
                    }
                    return NDExpr.Input(k);
                },
            });
            // A constant-only sub-tree (an expression of literals, `-(NDExpr)3`) has no array: NumPy computes a Python
            // number from it, weak unless a cast pinned its dtype — read off the tree's own typing.
            bool weak = map.Count == 0
                        && tree.InferType(Array.Empty<NPTypeCode>(), new Dictionary<NDExpr, NPTypeCode>()).IsWeak;
            return new(NDExprStageOperandKind.Computed, -1, null, tree, map.ToArray(), weak);
        }
    }

    /// <summary>
    ///     One node that computes on text, run per call as one NumPy statement over whole arrays (see the file header): the
    ///     engine resolves its operands, then <see cref="Execute"/> runs the statement. Immutable, shared by every call.
    /// </summary>
    internal abstract class NDExprTextStage
    {
        /// <summary>NumPy's name of the statement (<c>equal</c>, <c>str_len</c>, <c>isnan</c>, <c>astype</c>, the truth
        ///     route's reader), for diagnostics.</summary>
        internal readonly string Name;

        /// <summary>The operands, in the statement's argument order.</summary>
        internal readonly NDExprStageOperand[] Operands;

        /// <summary>The dtype the statement's result has whenever it computes (the remainder's input type for a numeric
        ///     result); <see cref="NPTypeCode.String"/> for a text result, whose class and width the call reads off the array.
        ///     A statement NumPy refuses for every operand class (np.negative of text) is typed as a text result, so no
        ///     remainder input is made for it: it raises before anything reads it.</summary>
        internal readonly NPTypeCode ResultType;

        /// <summary>Stores the parts.</summary>
        /// <param name="name">NumPy's name of the statement.</param>
        /// <param name="resultType">The result dtype.</param>
        /// <param name="operands">The operands.</param>
        protected NDExprTextStage(string name, NPTypeCode resultType, params NDExprStageOperand[] operands)
        {
            Name = name;
            ResultType = resultType;
            Operands = operands;
        }

        /// <summary>Whether the statement's result is text (E3): only later stages and the evaluation's own result read it,
        ///     never the numeric remainder.</summary>
        internal bool ResultIsText => ResultType == NPTypeCode.String;

        /// <summary>
        ///     Whether the statement is a ufunc call that takes the evaluation's <c>out=</c> and <c>where=</c> itself when it is
        ///     the whole expression (np.add / np.multiply / np.maximum / np.minimum over text): NumPy's ufunc writes its loop's
        ///     result into <c>out=</c> under the mask, which for the string repeat is the only way an <c>S</c> / <c>U</c>
        ///     result exists (<see cref="ExecuteInto"/>).
        /// </summary>
        internal virtual bool TakesOutAndWhere => false;

        /// <summary>
        ///     Runs the statement over the resolved operands for one call.
        /// </summary>
        /// <param name="engine">The engine evaluating the call (its members are NumPy's statements).</param>
        /// <param name="args">The operand arrays, parallel to <see cref="Operands"/> (null for an absent one). The caller owns
        ///     them and disposes the ones it made after the call.</param>
        /// <returns>A fresh array the caller owns (never one of <paramref name="args"/>), of <see cref="ResultType"/> (a text
        ///     result's own descriptor).</returns>
        internal abstract NDArray Execute(DefaultEngine engine, NDArray[] args);

        /// <summary>
        ///     Runs the statement with the evaluation's <c>out=</c> / <c>where=</c> as the ufunc's own keywords (only for a
        ///     stage that <see cref="TakesOutAndWhere"/>).
        /// </summary>
        /// <param name="engine">The engine evaluating the call.</param>
        /// <param name="args">The operand arrays (see <see cref="Execute"/>).</param>
        /// <param name="out">The caller's result array, or null for a fresh result.</param>
        /// <param name="where">The write mask, or null.</param>
        /// <returns><paramref name="out"/>, or a fresh result whose masked-off elements NumPy leaves uninitialized.</returns>
        /// <exception cref="InvalidOperationException">The stage takes no keywords — a driver bug.</exception>
        internal virtual NDArray ExecuteInto(DefaultEngine engine, NDArray[] args, NDArray @out, NDArray where)
            => throw new InvalidOperationException($"np.evaluate's '{Name}' stage takes no out= / where= of its own: a driver bug.");
    }

    /// <summary>
    ///     E1: a comparison with a text operand, through <see cref="NDStringUfuncs.BinaryGuard"/> — the guard <c>np.equal</c>
    ///     and its siblings run, which computes NumPy's result or raises NumPy's error.
    /// </summary>
    internal sealed class NDExprComparisonStage : NDExprTextStage
    {
        /// <summary>Creates a comparison stage.</summary>
        /// <param name="ufunc">NumPy's ufunc name of the comparison (<c>"equal"</c> … <c>"greater_equal"</c>).</param>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        internal NDExprComparisonStage(string ufunc, NDExprStageOperand left, NDExprStageOperand right)
            : base(ufunc, NPTypeCode.Boolean, left, right)
        {
        }

        /// <inheritdoc/>
        /// <exception cref="TypeError">NumPy has no loop for the operand pair (S with U, text with a number: its
        ///     <c>UFuncTypeError</c> text naming the operand classes).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ufunc text).</exception>
        /// <exception cref="ValueError">A StringDType ordering meets an NA of a <c>None</c>-like descriptor, or two
        ///     StringDType operands carry incompatible NA objects (NumPy's texts).</exception>
        /// <exception cref="InvalidOperationException">Neither operand is text — a lowering bug (a stage is only made for a
        ///     node with a text operand).</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args)
            // The guard answers every pair with a text operand: the comparison of two strings (S/S, U/U, T/T, T/U, U/T, any
            // widths and NA objects), or NumPy's refusal; it returns null only when neither operand is text.
            => NDStringUfuncs.BinaryGuard(Name, args[0], args[1])
               ?? throw new InvalidOperationException(
                   $"np.evaluate's '{Name}' stage has no text operand ({args[0].dtype}, {args[1].dtype}): a lowering bug.");
    }

    /// <summary>The route a NumPy reader takes to a string's truth value (<see cref="NDStringTruth.Flavor"/>).</summary>
    internal enum NDExprTruthRule : byte
    {
        /// <summary>The bool cast: the logical ufuncs, <c>np.where</c>'s condition, <c>np.any</c> / <c>np.all</c>.</summary>
        Cast,

        /// <summary>The dtype's nonzero function: <c>np.count_nonzero(a)</c> without an axis or keepdims.</summary>
        Nonzero,

        /// <summary>
        ///     <c>np.count_nonzero(a, axis, keepdims)</c> with an axis or keepdims, which NumPy computes as
        ///     <c>a_bool.sum(axis)</c>: <c>S</c> / <c>U</c> through <c>a != ''</c> (the nonzero answer), <c>StringDType</c>
        ///     through <c>a.astype(bool)</c> (the cast — an NA of an empty-string NA object counts there and not without an
        ///     axis; probed 2.4.2).
        /// </summary>
        CountNonzeroAxis,
    }

    /// <summary>
    ///     E2: the truth value of a text operand as a bool array, for a logical node, a where's condition or an
    ///     Any / All / CountNonzero reduction that then reads it in the fused remainder (NumPy's structure: the cast, then
    ///     the bool loop).
    /// </summary>
    internal sealed class NDExprTruthStage : NDExprTextStage
    {
        /// <summary>The truth route.</summary>
        internal readonly NDExprTruthRule Rule;

        /// <summary>Creates a truth stage.</summary>
        /// <param name="rule">The route of the reader.</param>
        /// <param name="text">The text operand.</param>
        internal NDExprTruthStage(NDExprTruthRule rule, NDExprStageOperand text)
            : base(rule == NDExprTruthRule.Cast ? "astype(bool)" : "nonzero", NPTypeCode.Boolean, text)
        {
            Rule = rule;
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy casts inside its iterator, so the reader's result takes its layout from the ORIGINAL operand, in which a
        ///     broadcast axis abstains: a dense truth array would vote instead (a row-broadcast text array's K-order cast is
        ///     F-contiguous, probed). So a broadcast operand's truth is computed over the operand with every stride-0 axis
        ///     cut to one element and broadcast back: the same stride-0 pattern, every distinct element read once. Any other
        ///     layout keeps its memory order (<see cref="NDStringTruth.Truth"/> lays the result out like astype(order='K')).
        /// </remarks>
        /// <exception cref="SystemError">The cast route over a <c>U</c> element that is one unit above U+10FFFF (NumPy's
        ///     <c>PyUnicode_New</c> refusal).</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args)
        {
            var text = args[0];
            var flavor = Rule switch
            {
                NDExprTruthRule.Cast => NDStringTruth.Flavor.Cast,
                NDExprTruthRule.Nonzero => NDStringTruth.Flavor.Nonzero,
                _ => NDStringUfuncs.IsVString(text) ? NDStringTruth.Flavor.Cast : NDStringTruth.Flavor.Nonzero,
            };
            if (!text.Shape.IsBroadcasted)
                return NDStringTruth.Truth(text, flavor);

            // One element along every stride-0 axis (the slice keeps the stride; the axis is now length 1).
            var dims = text.Shape.dimensions;
            var strides = text.Shape.strides;
            var key = new StringBuilder();
            for (int k = 0; k < dims.Length; k++)
                key.Append(k == 0 ? "" : ",").Append(strides[k] == 0 && dims[k] > 1 ? "0:1" : ":");
            using var distinct = text[key.ToString()];
            using var truth = NDStringTruth.Truth(distinct, flavor);
            // The broadcast view holds its own reference on the truth buffer, so the compact array can be released here.
            return NDArray.broadcast_to(truth, text.Shape);
        }
    }

    /// <summary>
    ///     E2: <c>np.isnan</c> of a text operand through the engine's ufunc: True for a <c>StringDType</c> NA whose NA object
    ///     is NaN-like; <c>S</c> / <c>U</c> raise NumPy's own refusal.
    /// </summary>
    internal sealed class NDExprIsNanStage : NDExprTextStage
    {
        /// <summary>Creates an isnan stage.</summary>
        /// <param name="x">The operand.</param>
        internal NDExprIsNanStage(NDExprStageOperand x) : base("isnan", NPTypeCode.Boolean, x)
        {
        }

        /// <inheritdoc/>
        /// <exception cref="TypeError">An <c>S</c> / <c>U</c> operand (NumPy: <c>ufunc 'isnan' not supported for the input
        ///     types ...</c>).</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args) => engine.IsNan(args[0], null, null, null);
    }

    /// <summary>
    ///     A cast with text on either side, <c>x.astype(target)</c> — a statement of its own in NumPy, so its result is the
    ///     K-order array astype returns. E2: text to a number (NumPy's parse: <c>int(str)</c>, <c>float(str)</c>, ...). E3: a
    ///     number to text (NumPy's <c>str()</c> of each element, cut to a fixed width: <c>int64 → &lt;U21</c>, <c>U2</c> keeps two
    ///     characters) and text to text (a recast between classes, widths and NA objects).
    /// </summary>
    internal sealed class NDExprCastStage : NDExprTextStage
    {
        /// <summary>The target dtype, a whole descriptor (a text one carries its class, width and NA object).</summary>
        internal readonly DType Target;

        /// <summary>Creates a cast stage.</summary>
        /// <param name="x">The operand (text, or a number for a text target).</param>
        /// <param name="target">The target dtype: a number, a bool, or a text descriptor.</param>
        internal NDExprCastStage(NDExprStageOperand x, DType target) : base("astype", target.typecode, x)
        {
            Target = target;
        }

        /// <inheritdoc/>
        /// <exception cref="ValueError">An element that does not parse (NumPy's <c>invalid literal for int() ...</c>, <c>could
        ///     not convert string to float: ...</c>), or a <c>StringDType</c> NA cast to a non-nullable type.</exception>
        /// <exception cref="OverflowException">A parsed integer out of the target's range (NumPy's
        ///     <c>OverflowError</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> / StringDType element outside ASCII cast to <c>S</c> (NumPy's
        ///     ascii codec refusal).</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args) => args[0].astype(Target, copy: true);
    }

    /// <summary>
    ///     E2: a numpy.strings function (<see cref="StringsNode"/>) through the engine's own implementation: the checks,
    ///     the NA rules and the errors of <c>np.strings.{name}</c>.
    /// </summary>
    internal sealed class NDExprStringsStage : NDExprTextStage
    {
        /// <summary>The function.</summary>
        internal readonly NDExprStringsFunction Function;

        /// <summary>Creates a strings stage.</summary>
        /// <param name="fn">The function.</param>
        /// <param name="operands">Its operands: <c>x</c>, or <c>a, sub, start, end</c> for a search.</param>
        internal NDExprStringsStage(NDExprStringsFunction fn, NDExprStageOperand[] operands)
            : base(StringsNode.NumPyNameOf(fn), StringsNode.ResultTypeOf(fn), operands)
        {
            Function = fn;
        }

        /// <inheritdoc/>
        /// <exception cref="TypeError">No loop for the operand classes (a non-text <c>x</c> / <c>a</c>, an <c>S</c> array with a
        ///     <c>str</c> substring, <c>isdecimal</c> / <c>isnumeric</c> over <c>S</c>), or two <c>StringDType</c> operands
        ///     with different NA objects.</exception>
        /// <exception cref="TypeError">A <c>start</c> / <c>end</c> that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA the function's rule refuses, or <c>index</c> / <c>rindex</c>
        ///     without a match.</exception>
        internal override NDArray Execute(DefaultEngine engine, NDArray[] args)
        {
            int predicate = StringsNode.PredicateKindOf(Function);
            if (predicate >= 0)
                return NDStringMethods.Predicate(predicate, args[0], null, null, null);
            return Function switch
            {
                NDExprStringsFunction.StrLen => NDStringMethods.StrLen(args[0], null, null, null),
                NDExprStringsFunction.Find => NDStringMethods.FindLike(Name, TextMethodOp.Find, args[0], args[1], args[2], args[3]),
                NDExprStringsFunction.RFind => NDStringMethods.FindLike(Name, TextMethodOp.RFind, args[0], args[1], args[2], args[3]),
                NDExprStringsFunction.Count => NDStringMethods.FindLike(Name, TextMethodOp.Count, args[0], args[1], args[2], args[3]),
                // index / rindex are find / rfind whose ufunc name makes a missing match raise (np.strings.index's own call).
                NDExprStringsFunction.Index => NDStringMethods.FindLike(Name, TextMethodOp.Find, args[0], args[1], args[2], args[3]),
                NDExprStringsFunction.RIndex => NDStringMethods.FindLike(Name, TextMethodOp.RFind, args[0], args[1], args[2], args[3]),
                NDExprStringsFunction.StartsWith => NDStringMethods.FindLike(Name, TextMethodOp.StartsWith, args[0], args[1], args[2], args[3]),
                _ => NDStringMethods.FindLike(Name, TextMethodOp.EndsWith, args[0], args[1], args[2], args[3]),
            };
        }
    }

    /// <summary>
    ///     The text half of a program (see the file header): the stages in evaluation order and either the numeric remainder
    ///     over the numeric inputs and the numeric stage results, or the TEXT ROOT — the text value that is the evaluation's
    ///     whole result. Immutable after construction and shared by every call of its program.
    /// </summary>
    internal sealed class NDExprTextPlan
    {
        /// <summary>The stages, in evaluation order.</summary>
        internal readonly NDExprTextStage[] Stages;

        /// <summary>The remainder tree: positional leaves, the numeric inputs first (<see cref="NumericInputs"/>), then one per
        ///     numeric stage result in stage order (<see cref="RemainderStages"/>). Its per-root program slot caches the
        ///     remainder's program across calls. Null for a plan whose result is text (<see cref="HasTextRoot"/>).</summary>
        internal readonly NDExpr Remainder;

        /// <summary>The original input index of each of the remainder's numeric inputs, in order.</summary>
        internal readonly int[] NumericInputs;

        /// <summary>The remainder's input dtypes (the numeric inputs', then each numeric stage result's).</summary>
        internal readonly NPTypeCode[] RemainderTypes;

        /// <summary>Per remainder input after <see cref="NumericInputs"/>: the stage whose result it is.</summary>
        internal readonly int[] RemainderStages;

        /// <summary>The evaluation's result when it is text: a text input of the call (a bare text leaf as the whole
        ///     expression) or a text stage's result; <see cref="NDExprStageOperandKind.Absent"/> when the result is the numeric
        ///     remainder's.</summary>
        internal readonly NDExprStageOperand TextRoot;

        /// <summary>The stage a root reduction (or weighted average) of text became, or -1 (see
        ///     <see cref="NDExprTextLowering.RootReductionStage"/>).</summary>
        internal readonly int RootReductionStage;

        /// <summary>Creates the plan.</summary>
        /// <param name="stages">The stages in evaluation order.</param>
        /// <param name="remainder">The remainder tree, or null for a text result.</param>
        /// <param name="numericInputs">The original indices of the remainder's numeric inputs.</param>
        /// <param name="remainderTypes">The remainder's input dtypes.</param>
        /// <param name="remainderStages">The stage behind each remainder input after the numeric inputs.</param>
        /// <param name="textRoot">The text result's operand, or <see cref="NDExprStageOperand.Absent"/>.</param>
        /// <param name="rootReductionStage">The root reduction's stage, or -1.</param>
        internal NDExprTextPlan(NDExprTextStage[] stages, NDExpr remainder, int[] numericInputs, NPTypeCode[] remainderTypes,
                                int[] remainderStages, NDExprStageOperand textRoot, int rootReductionStage)
        {
            Stages = stages;
            Remainder = remainder;
            NumericInputs = numericInputs;
            RemainderTypes = remainderTypes;
            RemainderStages = remainderStages;
            TextRoot = textRoot;
            RootReductionStage = rootReductionStage;
        }

        /// <summary>Whether the evaluation's result is text (<see cref="TextRoot"/>) rather than the remainder's.</summary>
        internal bool HasTextRoot => TextRoot.Kind != NDExprStageOperandKind.Absent;

        /// <summary>Whether the root is a reduction: the remainder's (a <see cref="ReduceNode"/> or a weighted average), or a
        ///     reduction of text that became a stage.</summary>
        internal bool IsReduction => Remainder is ReduceNode or WeightedAverageNode || RootReductionStage >= 0;

        /// <summary>
        ///     The program's result dtype: <see cref="NPTypeCode.String"/> for a text result (the class and width are the
        ///     call's), else the remainder's NumPy result dtype over the nominal input types.
        /// </summary>
        /// <returns>The result dtype.</returns>
        internal NPTypeCode ResolveResultType()
            => HasTextRoot ? NPTypeCode.String : Remainder.ResolveNumPyTypes(RemainderTypes, out _);

        /// <summary>
        ///     Whether the remainder is nothing but one stage's result (<c>np.evaluate(NDExpr.Equal(a, b))</c>,
        ///     <c>np.evaluate(NDExpr.StrLen(a))</c>): that stage's array — NumPy's own statement result — is then the
        ///     evaluation's result as it is, when no keyword shapes the result.
        /// </summary>
        /// <param name="stage">The stage whose result is the whole evaluation, when true.</param>
        /// <returns>True when the remainder is a bare stage-result leaf.</returns>
        internal bool IsBareStageResult(out int stage)
        {
            if (Remainder is InputNode leaf && leaf.Index >= NumericInputs.Length)
            {
                stage = RemainderStages[leaf.Index - NumericInputs.Length];
                return true;
            }

            stage = -1;
            return false;
        }

        /// <summary>The remainder's inputs for one call: the numeric inputs, then the numeric stage results.</summary>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="results">The stage results of this call, in stage order.</param>
        /// <returns>A new array of the remainder's inputs (the stage results stay owned by the caller).</returns>
        internal NDArray[] RemainderInputs(NDArray[] inputs, NDArray[] results)
        {
            var ops = new NDArray[NumericInputs.Length + RemainderStages.Length];
            for (int i = 0; i < NumericInputs.Length; i++)
                ops[i] = inputs[NumericInputs[i]];
            for (int j = 0; j < RemainderStages.Length; j++)
                ops[NumericInputs.Length + j] = results[RemainderStages[j]];
            return ops;
        }

        /// <summary>
        ///     One remainder input for one call: numeric input <paramref name="index"/>, or the result of the stage behind it
        ///     (a computed stage operand reads only earlier stages, already run).
        /// </summary>
        /// <param name="index">The remainder input index.</param>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="results">The stage results of this call so far.</param>
        /// <returns>The array (owned by the caller or by the call's stage results).</returns>
        internal NDArray RemainderInput(int index, NDArray[] inputs, NDArray[] results)
            => index < NumericInputs.Length
                ? inputs[NumericInputs[index]]
                : results[RemainderStages[index - NumericInputs.Length]];

        /// <summary>
        ///     Lowers a bound tree with text inputs (or a node only a stage runs: a numpy.strings function, a cast to text)
        ///     into a plan.
        /// </summary>
        /// <param name="bound">The bound tree (positional leaves).</param>
        /// <param name="inputTypes">The program's input signature.</param>
        /// <returns>The plan.</returns>
        /// <exception cref="StringComputationNotSupportedException">A delegate (<c>NDExpr.Call</c>) gets a text argument.</exception>
        internal static NDExprTextPlan Lower(NDExpr bound, NPTypeCode[] inputTypes)
        {
            var ctx = new NDExprTextLowering(inputTypes);
            var root = bound.LowerText(ctx);
            // A text value at the root is the evaluation's result itself: a text stage's array, or a copy of a text input
            // (np.evaluate always answers with an array of its own); the call casts it into the keywords.
            var textRoot = root.IsText ? ctx.Operand(root, null) : NDExprStageOperand.Absent;
            return new NDExprTextPlan(ctx.Stages.ToArray(), root.IsText ? null : root.Node, ctx.NumericInputs.ToArray(),
                ctx.RemainderTypes.ToArray(), ctx.RemainderStages.ToArray(), textRoot, ctx.RootReductionStage);
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
#pragma warning disable NDW012 // not a leak: MarkAsCSharpLiteral hands back the 0-d it was given (marked), which is returned to the caller that owns it
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
#pragma warning restore NDW012
    }

    public sealed partial class BinaryNode
    {
        /// <inheritdoc/>
        /// <remarks>A binary ufunc with a text operand is a stage running NumPy's own call (<see cref="NDExprBinaryStage"/>):
        ///     np.add concatenates two strings of one class (StringDType with U or a Python str too), np.multiply repeats a
        ///     StringDType string an integer number of times, and every other pair — S with U, text with a number, any other
        ///     ufunc — raises NumPy's no-loop error when the call runs.</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            if (l.IsText || r.IsText)
                return ctx.AddStage(new NDExprBinaryStage(_op, ctx.Operand(l, _left), ctx.Operand(r, _right)));
            return NDExprLowered.Numeric(ReferenceEquals(l.Node, _left) && ReferenceEquals(r.Node, _right)
                ? this
                : new BinaryNode(_op, l.Node, r.Node));
        }

        /// <summary>The operation (the stage's ufunc).</summary>
        internal BinaryOp TextOp => _op;
    }

    public sealed partial class UnaryNode
    {
        /// <summary>The operation.</summary>
        internal UnaryOp Op => _op;

        /// <summary>The operand.</summary>
        internal NDExpr Child => _child;

        /// <inheritdoc/>
        /// <remarks>IsNaN of text is the engine's np.isnan (StringDType's NaN-like NA; NumPy's refusal for S / U). np.real /
        ///     np.imag of text compute (the array itself, an all-empty array of its dtype); every other unary ufunc —
        ///     logical_not included — is NumPy's own refusal, raised by the engine's member when the call runs
        ///     (<see cref="NDExprUnaryStage"/>).</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _child.LowerText(ctx);
            if (c.IsText)
            {
                var x = ctx.Operand(c, _child);
                return ctx.AddStage(_op == UnaryOp.IsNan ? new NDExprIsNanStage(x) : new NDExprUnaryStage(_op, x));
            }

            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child) ? this : new UnaryNode(_op, c.Node));
        }
    }

    public sealed partial class CastNode
    {
        /// <inheritdoc/>
        /// <remarks>A cast with text on either side is <c>x.astype(target)</c>, NumPy's own statement: text to a number is
        ///     NumPy's parse, a number to text NumPy's formatting cut to the width, text to text a recast
        ///     (<see cref="NDExprCastStage"/>). A cast of numbers to numbers stays fused.</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _child.LowerText(ctx);
            if (c.IsText || _textTarget is not null)
                return ctx.AddStage(new NDExprCastStage(ctx.Operand(c, _child), _textTarget ?? (DType)_target));

            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child) ? this : new CastNode(c.Node, _target));
        }
    }

    public sealed partial class RoundNode
    {
        /// <inheritdoc/>
        /// <remarks><c>np.round(text, decimals)</c> is NumPy's refusal (its first step, <c>np.multiply</c> by 10^decimals,
        ///     has no text loop), raised by the engine's own round when the call runs (<see cref="NDExprRoundStage"/>).</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _child.LowerText(ctx);
            if (c.IsText)
                return ctx.AddStage(new NDExprRoundStage(_decimals, ctx.Operand(c, _child)));
            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child) ? this : new RoundNode(c.Node, _decimals));
        }
    }

    public sealed partial class ComparisonNode
    {
        /// <summary>The first operand (a truth test's <c>x</c>, read by the reduction that lowers it).</summary>
        internal NDExpr Left => _left;

        /// <summary>The second operand (a truth test's literal zero).</summary>
        internal NDExpr Right => _right;

        /// <inheritdoc/>
        /// <remarks>
        ///     A comparison with a text operand is a stage whatever the other operand is: two strings compare, and text
        ///     against a number reaches NumPy's own refusal through the same guard per call (the number computed first, as
        ///     NumPy's statement computes it). The factories' truth test (<see cref="IsTruthTest"/>, the <c>x != 0</c> of
        ///     <c>Any</c> / <c>All</c> / <c>CountNonzero</c>) is NOT a comparison in NumPy (<c>np.count_nonzero(text)</c>
        ///     reads the truth, <c>text != 0</c> has no loop): its reduction lowers it into a truth stage before this runs.
        /// </remarks>
        /// <exception cref="InvalidOperationException">A truth test over text reached here, which only a lowering bug can
        ///     cause (its reduction lowers it).</exception>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            if (!l.IsText && !r.IsText)
                return NDExprLowered.Numeric(ReferenceEquals(l.Node, _left) && ReferenceEquals(r.Node, _right)
                    ? this
                    : new ComparisonNode(_op, l.Node, r.Node, _truthTest));
            if (_truthTest)
                throw new InvalidOperationException(
                    "np.evaluate's truth test over text was not lowered by its reduction: a program-build bug.");
            var stage = new NDExprComparisonStage(UfuncName(_op), ctx.Operand(l, _left), ctx.Operand(r, _right));
            return ctx.AddStage(stage);
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
        /// <remarks>
        ///     <para>
        ///     np.maximum / np.minimum with a text operand are stages running NumPy's own ufunc
        ///     (<see cref="NDExprMinMaxStage"/>): two StringDType operands compare (their NA rules included); every other pair
        ///     is NumPy's no-loop refusal (U, S, StringDType beside U, text beside a number), raised when the call runs.
        ///     </para>
        ///     <para>
        ///     <c>Clip</c>'s two halves (<c>Min(Max(x, lo), hi)</c>, both flagged as clip halves) are ONE NumPy statement,
        ///     <c>np.clip(x, lo, hi)</c>, whose refusal names all three operands: when any of the three is text they lower
        ///     together into a clip stage (<see cref="NDExprClipStage"/>), not into a maximum stage inside a minimum one. A
        ///     one-sided clip is np.maximum / np.minimum in NumPy too, and lowers as one.
        ///     </para>
        /// </remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            if (_clipSelection && _isMin && _left is MinMaxNode { _clipSelection: true, _isMin: false } inner)
            {
                // NumPy evaluates np.clip's arguments in order: x, lo, hi (the bound tree's child order too).
                var x = inner._left.LowerText(ctx);
                var lo = inner._right.LowerText(ctx);
                var hi = _right.LowerText(ctx);
                if (x.IsText || lo.IsText || hi.IsText)
                    return ctx.AddStage(new NDExprClipStage(
                        ctx.Operand(x, inner._left), ctx.Operand(lo, inner._right), ctx.Operand(hi, _right)));

                var innerNode = ReferenceEquals(x.Node, inner._left) && ReferenceEquals(lo.Node, inner._right)
                    ? inner
                    : new MinMaxNode(isMin: false, x.Node, lo.Node, clipSelection: true);
                return NDExprLowered.Numeric(ReferenceEquals(innerNode, _left) && ReferenceEquals(hi.Node, _right)
                    ? this
                    : new MinMaxNode(isMin: true, innerNode, hi.Node, clipSelection: true));
            }

            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            if (l.IsText || r.IsText)
                return ctx.AddStage(new NDExprMinMaxStage(_isMin, ctx.Operand(l, _left), ctx.Operand(r, _right)));
            return NDExprLowered.Numeric(ReferenceEquals(l.Node, _left) && ReferenceEquals(r.Node, _right)
                ? this
                : new MinMaxNode(_isMin, l.Node, r.Node, _clipSelection));
        }
    }

    public sealed partial class WhereNode
    {
        /// <summary>The condition.</summary>
        internal NDExpr Condition => _cond;

        /// <summary>The values where the condition holds.</summary>
        internal NDExpr Then => _a;

        /// <summary>The values elsewhere.</summary>
        internal NDExpr Else => _b;

        /// <inheritdoc/>
        /// <remarks>
        ///     <para>
        ///     Text BRANCHES make the where a stage: NumPy's own <c>np.where(cond, a, b)</c> (<see cref="NDExprWhereStage"/>),
        ///     whose result dtype is the promotion of the branches (U with S is U, U with int64 <c>&lt;U21</c>, U with
        ///     StringDType is StringDType; a Python number beside fixed-width text, StringDType beside a number or S beside
        ///     StringDType raise NumPy's DTypePromotionError). A text condition is passed as it is: np.where casts it to bool
        ///     inside its own call.
        ///     </para>
        ///     <para>
        ///     With numeric branches a text condition is read through its bool cast (np.where casts it inside its iterator): a
        ///     truth stage, recorded after the branches' own stages because NumPy computes the branches before np.where runs,
        ///     and the where stays fused over its bool result.
        ///     </para>
        /// </remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var c = _cond.LowerText(ctx);
            var a = _a.LowerText(ctx);
            var b = _b.LowerText(ctx);
            if (a.IsText || b.IsText)
                return ctx.AddStage(new NDExprWhereStage(ctx.Operand(c, _cond), ctx.Operand(a, _a), ctx.Operand(b, _b)));
            var cond = c.IsText ? ctx.Truth(c, NDExprTruthRule.Cast) : c.Node;
            return NDExprLowered.Numeric(ReferenceEquals(cond, _cond) && ReferenceEquals(a.Node, _a) && ReferenceEquals(b.Node, _b)
                ? this
                : new WhereNode(cond, a.Node, b.Node));
        }
    }

    public sealed partial class LogicalNode
    {
        /// <inheritdoc/>
        /// <remarks>logical_and / logical_or / logical_xor read every text operand through its bool cast (NumPy's
        ///     <c>logical_ufunc_promoter</c>), for any mix of S, U, StringDType and numbers: each text operand becomes a truth
        ///     stage, and the logical node stays fused over the bool results.</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var l = _left.LowerText(ctx);
            var r = _right.LowerText(ctx);
            var left = l.IsText ? ctx.Truth(l, NDExprTruthRule.Cast) : l.Node;
            var right = r.IsText ? ctx.Truth(r, NDExprTruthRule.Cast) : r.Node;
            return NDExprLowered.Numeric(ReferenceEquals(left, _left) && ReferenceEquals(right, _right)
                ? this
                : new LogicalNode(_op, left, right));
        }
    }

    public sealed partial class CallNode
    {
        /// <inheritdoc/>
        /// <remarks>A delegate's parameters are numbers, so a text argument is refused in every phase: compute the text
        ///     part first (a stage in the same tree, or the string functions) and pass its numeric result.</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            NDExpr[] lowered = null;   // allocated at the first argument the lowering changes
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
        /// <remarks>
        ///     <para>
        ///     <c>Any</c> / <c>All</c> / <c>CountNonzero</c> over text read its truth value (the factories' truth test,
        ///     <see cref="ComparisonNode.IsTruthTest"/>): the operand becomes a truth stage by NumPy's route for that reader
        ///     (<see cref="NDExprTruthRule"/>), and the reduction stays in the remainder over the bool result — the same
        ///     reduction with the same axis, keepdims and keywords as over numbers (a 0-d operand with axis 0 / -1 included,
        ///     which NumPy lets through).
        ///     </para>
        ///     <para>
        ///     <c>NanSum</c> / <c>NanProd</c> (<see cref="IsNanFamily"/>) are np.nansum / np.nanprod of their operand: over
        ///     text, NumPy's NaN replacement leaves the array as it is and the call is np.sum / np.prod of it (a reduction
        ///     stage); over numbers the replacement stays as built, its operand lowered ONCE and shared by the test and the
        ///     selection (a strings function inside runs one stage, not two).
        ///     </para>
        ///     <para>
        ///     Every other reduction of a text child is a stage running NumPy's own call (<see cref="NDExprReduceStage"/>):
        ///     np.argmax / np.argmin over S / U / StringDType and np.sum / np.max / np.min over StringDType compute, the rest
        ///     answer as NumPy does (np.mean of a NaN-like NA's sum is NaN, everything else NumPy's refusal), raised when the
        ///     call runs.
        ///     </para>
        /// </remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            if (_child is ComparisonNode { IsTruthTest: true } truthTest)
            {
                // The truth test's operand is lowered once: its own stages (a strings function, a comparison) are recorded
                // here, and a text value takes the truth stage instead of the test.
                var x = truthTest.Left.LowerText(ctx);
                if (x.IsText)
                {
                    var rule = _kind is NDExprReduceKind.Any or NDExprReduceKind.All
                        ? NDExprTruthRule.Cast
                        : _axis is null && !_keepdims ? NDExprTruthRule.Nonzero : NDExprTruthRule.CountNonzeroAxis;
                    var truth = ctx.Truth(x, rule);
                    return NDExprLowered.Numeric(new ReduceNode(_kind, ComparisonNode.TruthTest(truth), _axis, _keepdims, _ddof));
                }

                var test = ReferenceEquals(x.Node, truthTest.Left)
                    ? truthTest
                    : ComparisonNode.TruthTest(x.Node);
                return NDExprLowered.Numeric(ReferenceEquals(test, _child)
                    ? this
                    : new ReduceNode(_kind, test, _axis, _keepdims, _ddof));
            }

            if (TryGetNanReplacement(out var nanOperand, out var fill))
            {
                // Lowered once: the replacement reads the same operand twice (IsNaN's and the selection's), which binding made
                // into two equal sub-trees; lowering both would record their stages twice.
                var x = nanOperand.LowerText(ctx);
                if (x.IsText)
                    return ctx.AddRootReduction(new NDExprReduceStage(_kind, _axis, _keepdims, _ddof, ctx.Operand(x, nanOperand)));
                var replaced = ReferenceEquals(x.Node, nanOperand)
                    ? _child
                    : new WhereNode(new UnaryNode(UnaryOp.IsNan, x.Node), fill, x.Node);
                return NDExprLowered.Numeric(ReferenceEquals(replaced, _child)
                    ? this
                    : new ReduceNode(_kind, replaced, _axis, _keepdims, _ddof, nanFamily: true));
            }

            var c = _child.LowerText(ctx);
            if (c.IsText)
                return ctx.AddRootReduction(new NDExprReduceStage(_kind, _axis, _keepdims, _ddof, ctx.Operand(c, _child)));
            return NDExprLowered.Numeric(ReferenceEquals(c.Node, _child)
                ? this
                : new ReduceNode(_kind, c.Node, _axis, _keepdims, _ddof, _nanFamily));
        }
    }

    public sealed partial class WeightedAverageNode
    {
        /// <inheritdoc/>
        /// <remarks>A weighted average with text values or weights is NumPy's own <c>np.average(values, axis, weights,
        ///     keepdims=...)</c> (<see cref="NDExprAverageStage"/>): its dtype resolution refuses text beside the float weights
        ///     (NumPy's DTypePromotionError) or the sum refuses the text, raised when the call runs.</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var v = _values.LowerText(ctx);
            var w = _weights.LowerText(ctx);
            if (v.IsText || w.IsText)
                return ctx.AddRootReduction(new NDExprAverageStage(_axis, _keepdims, ctx.Operand(v, _values), ctx.Operand(w, _weights)));
            return NDExprLowered.Numeric(ReferenceEquals(v.Node, _values) && ReferenceEquals(w.Node, _weights)
                ? this
                : new WeightedAverageNode(v.Node, w.Node, _axis, _keepdims));
        }
    }

    internal sealed partial class StringsNode
    {
        /// <inheritdoc/>
        /// <remarks>Always a stage: the engine's numpy.strings function over the operands, in NumPy's argument order (an
        ///     omitted <c>start</c> / <c>end</c> is NumPy's default). Every operand lowers first, so a stage inside one (a
        ///     comparison, another strings function) runs before this one, as NumPy evaluates the arguments first.</remarks>
        internal override NDExprLowered LowerText(NDExprTextLowering ctx)
        {
            var a = _a.LowerText(ctx);
            var x = ctx.Operand(a, _a);
            if (_sub is null)
                return ctx.AddStage(new NDExprStringsStage(_fn, new[] { x }));

            var sub = _sub.LowerText(ctx);
            var subOperand = ctx.Operand(sub, _sub);
            var start = ctx.OptionalOperand(_start);
            var end = ctx.OptionalOperand(_end);
            return ctx.AddStage(new NDExprStringsStage(_fn, new[] { x, subOperand, start, end }));
        }
    }
}
