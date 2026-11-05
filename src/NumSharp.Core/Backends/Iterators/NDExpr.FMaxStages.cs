using System;
using System.Collections.Generic;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.FMaxStages.cs — float32 / float64 fmax / fmin nodes run as NumPy's own np.fmax / np.fmin (parity QA-20)
// =============================================================================
//
// NumPy's float fmax / fmin give a ±0 tie and a NaN pair one answer inside whole SIMD vectors of an inner-loop call and
// the C library's answer elsewhere (DefaultEngine, Default.FMaxFMin.Positions.cs), so an element's bits depend on where
// it falls in the calls NumPy makes for that ufunc call — on the operands' layouts, the broadcast, NpyIter's buffering.
// A fused kernel has no such calls: it would answer every element by one rule. np.evaluate promises the answer of the
// unfused NumPy sequence, in which each fmax / fmin is its own ufunc call over fresh temporaries, so such a node is
// LOWERED here into a stage: per call, its two children are evaluated on their own (a leaf as it is, a literal as the
// Python scalar NumPy would see, a computed subtree into the fresh array NumPy's ufunc would allocate — its K-order
// layout), the engine's np.fmax / np.fmin runs over them (NumPy's positions), and the result joins the rest of the tree
// as a positional input. The rest stays fused. Nested nodes are staged inner first, so a stage's children never hold an
// fmax / fmin of their own.
//
// Only float32 / float64 nodes are staged (float16, the integers and complex have position-independent loops), and only
// over at least one array: a constant-only fmax keeps its weak NEP 50 type inside the fused tree.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public sealed partial class BinaryNode
    {
        /// <summary>The binary op this node applies (read by the fmax / fmin staging, which stages only those ops).</summary>
        internal BinaryOp Op => _op;
    }

    /// <summary>
    /// One operand of an <see cref="NDExprFMaxStage"/>: a positional input of the staged evaluation (an original input or an
    /// earlier stage's result), a literal, or a computed subtree evaluated over just the inputs it reads.
    /// </summary>
    internal sealed class NDExprFMaxStageChild
    {
        /// <summary>The input index when the child is a bare positional leaf; -1 otherwise.</summary>
        internal readonly int Input;

        /// <summary>The literal when the child is a constant (the Python scalar NumPy would see); null otherwise.</summary>
        internal readonly ConstNode Literal;

        /// <summary>The computed subtree over positional leaves 0..<c>Inputs.Length - 1</c>; null for a leaf or a literal.</summary>
        internal readonly NDExpr Tree;

        /// <summary>For <see cref="Tree"/>: the staged evaluation's input index behind each of the subtree's positional
        /// leaves, in leaf order.</summary>
        internal readonly int[] Inputs;

        /// <summary>Stores the parts; only <see cref="Of"/> builds a child.</summary>
        /// <param name="input">The leaf's input index, or -1.</param>
        /// <param name="literal">The literal, or null.</param>
        /// <param name="tree">The renumbered subtree, or null.</param>
        /// <param name="inputs">The subtree's input map, or null.</param>
        private NDExprFMaxStageChild(int input, ConstNode literal, NDExpr tree, int[] inputs)
        {
            Input = input;
            Literal = literal;
            Tree = tree;
            Inputs = inputs;
        }

        /// <summary>
        /// Classifies a stage's (already rewritten) child: a positional leaf, a literal, or a subtree, which is renumbered
        /// so that its leaves are 0..m-1 in first-reference order — evaluated later over exactly those m operands, so an
        /// unread input neither joins its broadcast nor its iterator.
        /// </summary>
        /// <param name="child">The child, over the staged evaluation's positional inputs.</param>
        /// <returns>The classified child.</returns>
        internal static NDExprFMaxStageChild Of(NDExpr child)
        {
            if (child is InputNode leaf)
                return new NDExprFMaxStageChild(leaf.Index, null, null, null);
            if (child is ConstNode literal)
                return new NDExprFMaxStageChild(-1, literal, null, null);

            var map = new List<int>();
            var renumber = new NDExprBindContext
            {
                InputRewrite = node =>
                {
                    int k = map.IndexOf(node.Index);
                    if (k < 0)
                    {
                        map.Add(node.Index);
                        k = map.Count - 1;
                    }
                    return NDExpr.Input(k);
                },
            };
            var tree = child.BindArrays(renumber);
            return new NDExprFMaxStageChild(-1, null, tree, map.ToArray());
        }
    }

    /// <summary>
    /// One staged node: <c>np.fmax</c> / <c>np.fmin</c> over two children at the node's float loop dtype.
    /// </summary>
    internal sealed class NDExprFMaxStage
    {
        /// <summary><see cref="BinaryOp.FMax"/> or <see cref="BinaryOp.FMin"/>.</summary>
        internal readonly BinaryOp Op;

        /// <summary>The first operand.</summary>
        internal readonly NDExprFMaxStageChild Left;

        /// <summary>The second operand.</summary>
        internal readonly NDExprFMaxStageChild Right;

        /// <summary>The node's NEP 50 result dtype (float32 or float64), passed as the ufunc's <c>dtype=</c> so a 0-d array
        /// operand promotes as the fused typing says (strong), not as a Python scalar.</summary>
        internal readonly NPTypeCode Dtype;

        /// <summary>Stores a stage.</summary>
        /// <param name="op">The op.</param>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        /// <param name="dtype">The loop dtype.</param>
        internal NDExprFMaxStage(BinaryOp op, NDExprFMaxStageChild left, NDExprFMaxStageChild right, NPTypeCode dtype)
        {
            Op = op;
            Left = left;
            Right = right;
            Dtype = dtype;
        }
    }

    /// <summary>
    /// A program's staged fmax / fmin nodes (see the file header): the stages in evaluation order and the remainder tree
    /// over the original inputs followed by one positional input per stage result. Immutable and shared by every call of
    /// its program; the remainder's and each computed child's programs cache on their own roots.
    /// </summary>
    internal sealed class NDExprFMaxPlan
    {
        /// <summary>The stages, inner first (a stage reads only original inputs and earlier stages' results).</summary>
        internal readonly NDExprFMaxStage[] Stages;

        /// <summary>The tree with every staged node replaced by the positional leaf of its result, renumbered over just
        /// the operands it reads (<see cref="RemainderInputs"/>), so an input only the stages read joins neither its
        /// broadcast nor its iterator.</summary>
        internal readonly NDExpr Remainder;

        /// <summary>The number of original inputs (stage <c>k</c>'s result is staged input <c>InputCount + k</c>).</summary>
        internal readonly int InputCount;

        /// <summary>For each of the remainder's positional leaves, the staged evaluation's input behind it (an original
        /// input index, or <c>InputCount + k</c> for stage <c>k</c>'s result).</summary>
        internal readonly int[] RemainderInputs;

        /// <summary>The remainder's input dtypes, in its leaf order.</summary>
        internal readonly NPTypeCode[] RemainderTypes;

        /// <summary>Stores a plan; only <see cref="TryLower"/> builds one.</summary>
        /// <param name="stages">The stages.</param>
        /// <param name="remainder">The renumbered remainder tree.</param>
        /// <param name="inputCount">The original input count.</param>
        /// <param name="remainderInputs">The remainder's input map.</param>
        /// <param name="remainderTypes">The remainder's input dtypes.</param>
        private NDExprFMaxPlan(NDExprFMaxStage[] stages, NDExpr remainder, int inputCount, int[] remainderInputs,
                               NPTypeCode[] remainderTypes)
        {
            Stages = stages;
            Remainder = remainder;
            InputCount = inputCount;
            RemainderInputs = remainderInputs;
            RemainderTypes = remainderTypes;
        }

        /// <summary>Whether the remainder's root is a reduction (a staged fmax inside a reduction's child).</summary>
        internal bool IsReduction => Remainder is ReduceNode or WeightedAverageNode;

        /// <summary>
        /// Whether the remainder is just the LAST stage's result — the expression was one float fmax / fmin at its root, so
        /// that stage can take the call's out= / where= itself (NumPy's masked or out= call has its own positions).
        /// </summary>
        internal bool RootIsLastStage
            => Remainder is InputNode leaf && RemainderInputs[leaf.Index] == InputCount + Stages.Length - 1;

        /// <summary>
        /// Lowers a bound tree's float32 / float64 fmax / fmin nodes into stages, or returns null when it has none.
        /// </summary>
        /// <param name="bound">The bound tree (positional leaves; a reduction, if any, at its root only).</param>
        /// <param name="inputTypes">The program's input dtypes.</param>
        /// <returns>The plan, or null.</returns>
        internal static NDExprFMaxPlan TryLower(NDExpr bound, NPTypeCode[] inputTypes)
        {
            bound.ResolveNumPyTypes(inputTypes, out var nodeTypes);
            bool any = false;
            foreach (var (node, type) in nodeTypes)
                if (node is BinaryNode b && b.Op is BinaryOp.FMax or BinaryOp.FMin
                    && type is NPTypeCode.Single or NPTypeCode.Double)
                {
                    any = true;
                    break;
                }
            if (!any)
                return null;

            var stages = new List<NDExprFMaxStage>();
            var types = new List<NPTypeCode>(inputTypes);
            var lower = new NDExprBindContext
            {
                BinaryRewrite = (node, l, r) =>
                {
                    // A child must be something NumPy's unfused call holds: an array (it reads an input) or a bare literal
                    // (a Python scalar). An arithmetic expression of literals has no temporary, and a constant-only fmax
                    // keeps its weak NEP 50 type in the fused tree: neither is staged.
                    bool lReads = ReadsInput(l), rReads = ReadsInput(r);
                    if (node.Op is not (BinaryOp.FMax or BinaryOp.FMin)
                        || !nodeTypes.TryGetValue(node, out var t) || t is not (NPTypeCode.Single or NPTypeCode.Double)
                        || (!lReads && !rReads)
                        || (!lReads && l is not ConstNode) || (!rReads && r is not ConstNode))
                        return null;
                    stages.Add(new NDExprFMaxStage(node.Op, NDExprFMaxStageChild.Of(l), NDExprFMaxStageChild.Of(r), t));
                    types.Add(t);
                    return NDExpr.Input(inputTypes.Length + stages.Count - 1);
                },
            };
            var remainder = bound.BindArrays(lower);
            if (stages.Count == 0)
                return null;

            // Renumber the remainder over the inputs it reads, in first-reference order.
            var map = new List<int>();
            remainder = remainder.BindArrays(new NDExprBindContext
            {
                InputRewrite = node =>
                {
                    int k = map.IndexOf(node.Index);
                    if (k < 0)
                    {
                        map.Add(node.Index);
                        k = map.Count - 1;
                    }
                    return NDExpr.Input(k);
                },
            });
            var remainderTypes = new NPTypeCode[map.Count];
            for (int i = 0; i < map.Count; i++)
                remainderTypes[i] = types[map[i]];
            return new NDExprFMaxPlan(stages.ToArray(), remainder, inputTypes.Length, map.ToArray(), remainderTypes);
        }

        /// <summary>Whether a (rewritten) subtree reads any positional input — false for a constant-only subtree.</summary>
        /// <param name="tree">The subtree.</param>
        /// <returns>True when a positional leaf appears in it.</returns>
        private static bool ReadsInput(NDExpr tree)
        {
            if (tree is InputNode)
                return true;
            if (tree is ConstNode)
                return false;
            bool found = false;
            tree.BindArrays(new NDExprBindContext
            {
                InputRewrite = _ =>
                {
                    found = true;
                    return null;
                },
            });
            return found;
        }
    }
}
