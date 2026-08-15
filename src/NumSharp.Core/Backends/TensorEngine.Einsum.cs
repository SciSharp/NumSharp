using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     <c>np.einsum</c> — the contraction itself, reached once the subscripts have parsed and
        ///     every operand validated.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     A composition, not a new low-level kernel — which is exactly how NumPy computes it on
        ///     its <c>optimize=</c> path: <c>numpy/_core/einsumfunc.py</c>'s <c>bmm_einsum</c>
        ///     ("batched-matrix-multiply einsum") reduces every pairwise contraction to a
        ///     <c>matmul</c> (plus a single-operand einsum to take diagonals, sum, and transpose the
        ///     terms first), or to a broadcast <c>multiply</c> when nothing is contracted. This is a
        ///     port of that, over NumSharp's own <see cref="np.matmul"/> / <see cref="np.multiply"/> /
        ///     <see cref="np.sum"/> / <see cref="np.transpose"/> / <see cref="np.diagonal"/>.
        ///     </para>
        ///     <para>
        ///     <b>This is what makes einsum "go through OpenBLAS like other functions".</b> The
        ///     products land on <see cref="Matmul"/> (2-D <c>gemm</c> and batched), which already
        ///     dispatches on <see cref="Blas"/> — so with <c>NumSharp.Interop.OpenBLAS</c> referenced,
        ///     the float32/float64/complex128 contractions are byte-identical to NumPy's, and without
        ///     it they fall back to the managed GEMM. There is deliberately no <c>TryEinsum</c> seam:
        ///     einsum reaches a backend the same indirect way <c>np.tensordot</c> and
        ///     <c>np.linalg.multi_dot</c> do, through the matrix product.
        ///     </para>
        ///     <para>
        ///     <b>Value parity.</b> For every contraction that reduces to a matrix product — matmul,
        ///     batched matmul, matvec, <c>ij,kj-&gt;ik</c>, tensor contractions, inner/outer — the
        ///     result equals <c>np.einsum(..., optimize=True)</c> (which is itself the matmul path),
        ///     hence byte-identical to NumPy when the products are. Integer and boolean contractions
        ///     are byte-exact throughout (modular / logical reduction is order-independent). The one
        ///     place a last-ULP difference can appear is a FLOAT summation done outside a product —
        ///     a pure reduction such as <c>ij-&gt;i</c>, a diagonal-then-sum, or the accumulation
        ///     across three or more float operands — because that summation order follows NumSharp's
        ///     <see cref="np.sum"/> and a left-to-right pairwise fold rather than NumPy's einsum
        ///     iterator; the results stay <c>allclose</c>.
        ///     </para>
        /// </remarks>
        public virtual NDArray Einsum(string subscripts, NDArray[] operands, NDArray @out,
            NPTypeCode? dtype, char order, string casting, object optimize)
        {
            // Parse AND validate — rank, ellipsis grammar, operand count, output labels, out='s rank,
            // every diagonal and every label extent. Nothing past this line is reachable by an
            // expression NumPy would reject, and an impossible diagonal is reported here (with NumPy's
            // wording) rather than surfacing later from np.diagonal.
            var plan = EinsumSubscripts.Parse(subscripts, operands, @out);
            var (inputTerms, outputTerm) = plan.BuildTerms();

            // einsum accumulates in the operands' promoted dtype (NEP50 result_type), NOT np.sum's
            // widened accumulator: einsum('ij->i', int32) is int32, where np.sum would give int64.
            // A dtype= override forces it (subject to the casting rule below).
            NPTypeCode computeType = dtype ?? np.result_type(operands);
            if (dtype.HasValue)
                RequireDtypeCastable(operands, dtype.Value, casting);

            var terms = new List<(NDArray Array, string Term)>(operands.Length);
            for (int i = 0; i < operands.Length; i++)
            {
                var a = operands[i].typecode == computeType ? operands[i] : operands[i].astype(computeType);
                terms.Add((a, inputTerms[i]));
            }

            NDArray result;
            if (terms.Count == 1)
            {
                result = SingleTermEinsum(terms[0].Array, terms[0].Term, outputTerm, computeType);
            }
            else
            {
                // Left-to-right pairwise fold. Each step contracts the first two operands into one,
                // targeting the labels that must still survive — the output's, plus any label a
                // later operand still needs. The final step targets the real output term.
                while (terms.Count > 1)
                {
                    var (a, ta) = terms[0];
                    var (b, tb) = terms[1];
                    string target = terms.Count == 2 ? outputTerm : SurvivingTerm(ta, tb, outputTerm, terms, 2);
                    var contracted = PairwiseContract(a, ta, b, tb, target, computeType);
                    terms.RemoveRange(0, 2);
                    terms.Insert(0, (contracted, target));
                }

                result = terms[0].Array;
            }

            result = ApplyOrder(result, order, operands);

            if (@out is null)
                return result;

            if (!ShapesEqual(@out.Shape.dimensions, result.Shape.dimensions))
                throw new ValueError(
                    $"einsum() output parameter has shape ({string.Join(",", @out.Shape.dimensions)}) " +
                    $"but the contraction produces ({string.Join(",", result.Shape.dimensions)})");

            np.copyto(@out, result, casting);
            return @out;
        }

        /// <summary>
        ///     The labels of a pairwise contraction that must survive it: those in the final output,
        ///     plus any still needed by a later operand — in first-appearance order across the two
        ///     terms. bmm handles arranging the result into this order.
        /// </summary>
        private static string SurvivingTerm(string ta, string tb, string outputTerm,
            List<(NDArray Array, string Term)> terms, int fromIndex)
        {
            var survivors = new HashSet<char>(outputTerm);
            for (int i = fromIndex; i < terms.Count; i++)
                foreach (char c in terms[i].Term)
                    survivors.Add(c);

            var seen = new HashSet<char>();
            var sb = new StringBuilder();
            foreach (char c in ta)
                if (survivors.Contains(c) && seen.Add(c))
                    sb.Append(c);
            foreach (char c in tb)
                if (survivors.Contains(c) && seen.Add(c))
                    sb.Append(c);
            return sb.ToString();
        }

        /// <summary>
        ///     One operand's einsum — port of the <c>c_einsum(eq, a)</c> prep in <c>bmm_einsum</c>:
        ///     collapse each repeated label to a diagonal, sum away every label absent from
        ///     <paramref name="dstTerm"/>, and transpose what is left into <paramref name="dstTerm"/>'s
        ///     order. Also the whole computation for a single-operand einsum (trace, diagonal, axis
        ///     sum, transpose).
        /// </summary>
        private static NDArray SingleTermEinsum(NDArray a, string srcTerm, string dstTerm, NPTypeCode computeType)
        {
            var term = new List<char>(srcTerm);
            var arr = a;

            // 1) Diagonals — collapse the first repeated label until none remain. np.diagonal drops
            //    its two axes and appends the diagonal as the last axis, so the term follows suit.
            while (true)
            {
                int p1 = -1, p2 = -1;
                for (int i = 0; i < term.Count && p1 < 0; i++)
                for (int j = i + 1; j < term.Count; j++)
                {
                    if (term[i] != term[j])
                        continue;
                    p1 = i;
                    p2 = j;
                    break;
                }

                if (p1 < 0)
                    break;

                char label = term[p1];
                arr = np.diagonal(arr, 0, p1, p2);
                term.RemoveAt(p2);
                term.RemoveAt(p1);
                term.Add(label);
            }

            // 2) Sum away every label not kept — highest axis first so the lower indices stay valid.
            //    The dtype is forced to the compute type so the reduction does not widen it the way
            //    np.sum's default accumulator would.
            var toSum = new List<int>();
            for (int i = 0; i < term.Count; i++)
                if (dstTerm.IndexOf(term[i]) < 0)
                    toSum.Add(i);
            for (int k = toSum.Count - 1; k >= 0; k--)
            {
                arr = np.sum(arr, toSum[k], computeType);
                term.RemoveAt(toSum[k]);
            }

            // 3) Transpose the remaining axes into the destination order.
            if (!SameOrder(term, dstTerm))
            {
                var perm = new int[dstTerm.Length];
                for (int i = 0; i < dstTerm.Length; i++)
                    perm[i] = term.IndexOf(dstTerm[i]);
                arr = np.transpose(arr, perm);
            }

            if (arr.typecode != computeType)
                arr = arr.astype(computeType);

            return arr;
        }

        /// <summary>
        ///     A single pairwise contraction <c>a[aTerm], b[bTerm] -&gt; outTerm</c> — port of
        ///     <c>bmm_einsum</c> + <c>_parse_eq_to_batch_matmul</c>. Every index is classified from
        ///     the operands' ACTUAL shapes (so size-1 broadcasting is handled exactly as NumPy does),
        ///     then the contraction runs as one <see cref="np.matmul"/> (batched when there are shared
        ///     output indices) or, when nothing is contracted, one broadcast <see cref="np.multiply"/>.
        /// </summary>
        private static NDArray PairwiseContract(NDArray a, string aTerm, NDArray b, string bTerm,
            string outTerm, NPTypeCode computeType)
        {
            var shapeA = a.Shape.dimensions;
            var shapeB = b.Shape.dimensions;
            var outSet = new HashSet<char>(outTerm);

            // Unique labels with extent > 1, in first-appearance order; size-1 labels are singletons
            // (broadcast placeholders) tracked apart. sizes records each >1 label's extent.
            var sizes = new Dictionary<char, long>();
            var singletons = new HashSet<char>();

            var leftOrder = new List<char>();
            var leftSet = new HashSet<char>();
            for (int i = 0; i < aTerm.Length; i++)
            {
                char ix = aTerm[i];
                long d = shapeA[i];
                if (d == 1)
                {
                    singletons.Add(ix);
                    continue;
                }

                if (sizes.TryGetValue(ix, out long prev))
                {
                    if (prev != d)
                        throw new ValueError($"einsum: label '{Describe(ix)}' has mismatched sizes {prev} and {d}.");
                }
                else
                {
                    sizes[ix] = d;
                }

                if (leftSet.Add(ix))
                    leftOrder.Add(ix);
            }

            var rightOrder = new List<char>();
            var rightSet = new HashSet<char>();
            for (int i = 0; i < bTerm.Length; i++)
            {
                char ix = bTerm[i];
                long d = shapeB[i];
                if (d == 1)
                {
                    if (!leftSet.Contains(ix))
                        singletons.Add(ix);
                    continue;
                }

                singletons.Remove(ix);
                if (sizes.TryGetValue(ix, out long prev))
                {
                    if (prev != d)
                        throw new ValueError($"einsum: label '{Describe(ix)}' has mismatched sizes {prev} and {d}.");
                }
                else
                {
                    sizes[ix] = d;
                }

                if (rightSet.Add(ix))
                    rightOrder.Add(ix);
            }

            // Classify the >1 labels: batch (both, kept), contracted (both, dropped), a-kept, b-kept.
            var bat = new List<char>();
            var con = new List<char>();
            var aKeep = new List<char>();
            var bKeep = new List<char>();
            foreach (char ix in leftOrder)
            {
                if (rightSet.Remove(ix))            // NumPy's right.pop(ix) — shared with b
                    (outSet.Contains(ix) ? bat : con).Add(ix);
                else if (outSet.Contains(ix))
                    aKeep.Add(ix);
            }

            foreach (char ix in rightOrder)         // b-only labels that were not popped
                if (rightSet.Contains(ix) && outSet.Contains(ix))
                    bKeep.Add(ix);

            if (con.Count == 0)
                return PureMultiplication(a, aTerm, shapeA, b, bTerm, shapeB, outTerm, computeType);

            // Only the size-1 output labels matter — they are re-introduced as length-1 axes at the end.
            var singletonsOut = new List<char>();
            foreach (char c in outTerm)
                if (singletons.Contains(c))
                    singletonsOut.Add(c);

            string desiredA = Concat(bat, aKeep, con);
            string desiredB = Concat(bat, con, bKeep);

            List<char>[] lgroups, rgroups, ogroups;
            if (bat.Count > 0)
            {
                lgroups = new[] {bat, aKeep, con};
                rgroups = new[] {bat, con, bKeep};
                ogroups = new[] {bat, aKeep, bKeep};
            }
            else
            {
                lgroups = new[] {aKeep, con};
                rgroups = new[] {con, bKeep};
                ogroups = new[] {aKeep, bKeep};
            }

            long[] newShapeA = FuseShape(lgroups, sizes);
            long[] newShapeB = FuseShape(rgroups, sizes);
            long[] newShapeAB = FuseOutputShape(ogroups, singletonsOut.Count, sizes);

            // Where the matmul leaves each label, before the final permutation.
            var producedSb = new StringBuilder();
            foreach (char c in singletonsOut) producedSb.Append(c);
            foreach (char c in bat) producedSb.Append(c);
            foreach (char c in aKeep) producedSb.Append(c);
            foreach (char c in bKeep) producedSb.Append(c);
            string outProduced = producedSb.ToString();

            int[] permAB = null;
            if (outProduced != outTerm)
            {
                permAB = new int[outTerm.Length];
                for (int i = 0; i < outTerm.Length; i++)
                    permAB[i] = outProduced.IndexOf(outTerm[i]);
            }

            var left = desiredA != aTerm ? SingleTermEinsum(a, aTerm, desiredA, computeType) : a;
            if (newShapeA != null)
                left = np.reshape(left, newShapeA);

            var right = desiredB != bTerm ? SingleTermEinsum(b, bTerm, desiredB, computeType) : b;
            if (newShapeB != null)
                right = np.reshape(right, newShapeB);

            var ab = np.matmul(left, right);

            if (newShapeAB != null)
                ab = np.reshape(ab, newShapeAB);
            if (permAB != null)
                ab = np.transpose(ab, permAB);

            return ab;
        }

        /// <summary>
        ///     The no-contraction case — an outer / Hadamard / broadcast product. Each operand is
        ///     transposed to the output's order and reshaped with length-1 axes wherever it lacks an
        ///     output label, then a single broadcast <see cref="np.multiply"/> forms the result.
        /// </summary>
        private static NDArray PureMultiplication(NDArray a, string aTerm, long[] shapeA,
            NDArray b, string bTerm, long[] shapeB, string outTerm, NPTypeCode computeType)
        {
            var desiredA = new StringBuilder();
            var desiredB = new StringBuilder();
            var newShapeA = new long[outTerm.Length];
            var newShapeB = new long[outTerm.Length];

            for (int i = 0; i < outTerm.Length; i++)
            {
                char ix = outTerm[i];
                int ai = aTerm.IndexOf(ix);
                if (ai >= 0)
                {
                    desiredA.Append(ix);
                    newShapeA[i] = shapeA[ai];
                }
                else
                {
                    newShapeA[i] = 1;
                }

                int bi = bTerm.IndexOf(ix);
                if (bi >= 0)
                {
                    desiredB.Append(ix);
                    newShapeB[i] = shapeB[bi];
                }
                else
                {
                    newShapeB[i] = 1;
                }
            }

            var left = desiredA.ToString() != aTerm ? SingleTermEinsum(a, aTerm, desiredA.ToString(), computeType) : a;
            left = np.reshape(left, newShapeA);

            var right = desiredB.ToString() != bTerm ? SingleTermEinsum(b, bTerm, desiredB.ToString(), computeType) : b;
            right = np.reshape(right, newShapeB);

            return np.multiply(left, right);
        }

        /// <summary>Product of each group's label extents; null when every group is already a single axis.</summary>
        private static long[] FuseShape(List<char>[] groups, Dictionary<char, long> sizes)
        {
            bool needsFuse = false;
            foreach (var g in groups)
            {
                if (g.Count != 1)
                {
                    needsFuse = true;
                    break;
                }
            }

            if (!needsFuse)
                return null;

            var shape = new long[groups.Length];
            for (int i = 0; i < groups.Length; i++)
            {
                long p = 1;
                foreach (char c in groups[i])
                    p *= sizes[c];
                shape[i] = p;
            }

            return shape;
        }

        /// <summary>The output shape after the matmul — length-1 axes for the size-1 output labels, then the un-fused groups.</summary>
        private static long[] FuseOutputShape(List<char>[] groups, int singletonCount, Dictionary<char, long> sizes)
        {
            bool needs = singletonCount > 0;
            int total = singletonCount;
            foreach (var g in groups)
            {
                if (g.Count != 1)
                    needs = true;
                total += g.Count;
            }

            if (!needs)
                return null;

            var shape = new long[total];
            int w = 0;
            for (int i = 0; i < singletonCount; i++)
                shape[w++] = 1;
            foreach (var g in groups)
                foreach (char c in g)
                    shape[w++] = sizes[c];

            return shape;
        }

        private static NDArray ApplyOrder(NDArray result, char order, NDArray[] operands)
        {
            switch (order)
            {
                case 'C':
                    return np.ascontiguousarray(result);
                case 'F':
                    return np.asfortranarray(result);
                case 'A':
                    bool allF = true;
                    foreach (var o in operands)
                    {
                        if (o.Shape.IsFContiguous)
                            continue;
                        allF = false;
                        break;
                    }

                    return allF ? np.asfortranarray(result) : np.ascontiguousarray(result);
                default:                 // 'K' — keep the natural layout (may be a view)
                    return result;
            }
        }

        /// <summary>Enforces <c>dtype=</c>'s casting rule the way NumPy's einsum loop does.</summary>
        private static void RequireDtypeCastable(NDArray[] operands, NPTypeCode target, string casting)
        {
            for (int i = 0; i < operands.Length; i++)
            {
                if (!np.can_cast(operands[i].typecode, target, casting))
                    throw new TypeError(
                        $"Cannot cast einsum operand {i} from dtype('{operands[i].typecode.AsNumpyDtypeName()}') " +
                        $"to dtype('{target.AsNumpyDtypeName()}') according to the rule '{casting}'");
            }
        }

        private static string Concat(List<char> a, List<char> b, List<char> c)
        {
            var sb = new StringBuilder(a.Count + b.Count + c.Count);
            foreach (char x in a) sb.Append(x);
            foreach (char x in b) sb.Append(x);
            foreach (char x in c) sb.Append(x);
            return sb.ToString();
        }

        private static bool SameOrder(List<char> term, string dstTerm)
        {
            if (term.Count != dstTerm.Length)
                return false;
            for (int i = 0; i < term.Count; i++)
                if (term[i] != dstTerm[i])
                    return false;
            return true;
        }

        private static bool ShapesEqual(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>Renders a label for a message — a real ASCII letter as itself, a reserved ellipsis slot as "...".</summary>
        private static string Describe(char label) => label >= 0xE000 ? "..." : label.ToString();
    }
}
