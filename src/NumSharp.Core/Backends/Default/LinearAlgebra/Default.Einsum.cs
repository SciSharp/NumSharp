// NumPy's numpy/_core/einsumfunc.py (np.einsum's optimize= path, bmm_einsum) as DefaultEngine's Einsum: the argument
// checks, the parse, the view path, then a left-to-right pairwise fold over the matrix products. NumPy's statements call
// np.result_type, np.can_cast, np.copyto, np.diagonal, np.sum, np.transpose, np.reshape, np.matmul, np.multiply,
// np.ascontiguousarray and np.asfortranarray; here they are the dtype layer's, the array object's and the operands'
// engine members. The facade is LinearAlgebra/np.einsum.cs.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // reclaims the cast operands, every pairwise step's terms and products, and the layout copy
        public override NDArray Einsum(string subscripts, NDArray[] operands, NDArray @out = null,
            DType dtype = null, char order = 'K', string casting = "safe", object optimize = null)
        {
            if (subscripts is null)
                throw new ArgumentNullException(nameof(subscripts));

            // NumPy's einsum hands any optimize= but False to einsum_path, whose path_type check runs before c_einsum
            // looks at the operands, order= or casting=.
            RequireOptimize(optimize);

            if (operands is null || operands.Length == 0)
                throw new ValueError(
                    "must specify the einstein sum subscripts string and at least one operand");

            order = Backends.GufuncGuard.RequireOrder(order);   // a lowercase letter is NumPy's too
            Backends.GufuncGuard.RequireCasting(casting);

            // Parse AND validate — rank, ellipsis grammar, operand count, output labels, out='s rank,
            // every diagonal and every label extent. Nothing past this line is reachable by an
            // expression NumPy would reject, and an impossible diagonal is reported here (with NumPy's
            // wording) rather than surfacing later from np.diagonal. The grammar half of the parse is
            // cached by (subscripts, operand ranks), as NumPy caches its own equation parsing.
            var plan = EinsumSubscripts.Bind(subscripts, operands, @out);
            string[] inputTerms = plan.Grammar.InputTerms;
            string outputTerm = plan.Grammar.OutputTerm;

            // THE VIEW PATH — one operand, no out=, nothing summed. NumPy answers with a VIEW of the
            // operand (diagonals + transpose only, get_combined_dims_view), and on this path it
            // ignores order= ENTIRELY and even discards a dtype= request (probed on 2.4.2:
            // einsum('ii->i', a, order='F'/'C') is the same non-contiguous view, and
            // einsum('ii->i', int32, dtype=int64) returns the int32 view). The view is writeable iff
            // the operand is — np.einsum('ii->i', a)[:] = 1 sets a's diagonal — where np.diagonal's
            // contract is read-only, so the WRITEABLE flag is restored here, never in np.diagonal.
            if (plan.SingleOperandView)
            {
                var view = SingleTermEinsum(operands[0], inputTerms[0], outputTerm, operands[0].typecode);

                // Never hand back the operand INSTANCE itself (einsum('ij->ij', a)): NumPy returns a
                // distinct view object, and returning `a` would let e.g. result.resize() mutate the
                // operand where NumPy's view refuses.
                if (ReferenceEquals(view, operands[0]))
                    view = new NDArray(operands[0].Storage.Alias(operands[0].Shape)) { TensorEngine = operands[0].TensorEngine };
                else if (!view.Shape.IsWriteable && operands[0].Shape.IsWriteable)
                {
                    // Re-alias from the OPERAND's storage, not the view's: Alias() inherits
                    // read-onlyness from the storage it aliases (so a view of the read-only
                    // np.diagonal stays read-only), and the operand's storage is the writeable one.
                    // The view's Shape addresses the same buffer — offset and strides are absolute.
                    view = new NDArray(operands[0].Storage.Alias(view.Shape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE)))
                        { TensorEngine = operands[0].TensorEngine };
                }

                return view;
            }

            // From here NumPy's c_einsum runs ONE iterator over the operands and out= (NpyIter_AdvancedNew with
            // NPY_ITER_COMMON_DTYPE, out READWRITE). It takes in the out's write flag first, before any dtype is
            // resolved: a read-only out raises NpyIter's text, not np.copyto's at the end.
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "operand array with iterator write flag set");

            // Text (an operand, a string dtype= request, or a text out=) has no sum-of-products: after the parse and
            // the view path above (which hands text back as a view, as NumPy does), NumPy's common dtype, its
            // operand casts, the out's casts, its bool-zero output fill and its loop lookup refuse it, in that order.
            Backends.Strings.NDStringUfuncs.ThrowIfEinsum(operands, dtype, casting, @out);

            // einsum accumulates in the operands' promoted dtype (NEP50 result_type), NOT np.sum's
            // widened accumulator: einsum('ij->i', int32) is int32, where np.sum would give int64.
            // A dtype= override forces it. With out= and no dtype= the iterator's common dtype takes the out in
            // too (NPY_ITER_COMMON_DTYPE over every operand): int8 operands with an int64 out are summed in int64.
            // EVERY operand must reach the loop dtype under the casting rule — which is what makes casting='no'
            // reject mixed dtypes — with NumPy's iterator wording verbatim.
            // np.result_type(arrays)'s statement: DTypePromotion.ResultType over the arrays (one array's canonical
            // descriptor, which is np.result_type's one-array shortcut), as ThrowIfEinsum's text check computes it.
            NPTypeCode computeType = dtype?.GetTypeCode()
                ?? (@out is null ? DTypePromotion.ResultType(operands) : DTypePromotion.ResultType(WithOut(operands, @out)));

            // np.can_cast(from, to, casting)'s statement at the call sites below, where the two lanes always differ (its
            // from == to shortcut never fires): the rule string is parsed per call, as the facade did.
            bool CanCast(NPTypeCode from, NPTypeCode to)
                => DTypeCasting.CanCastTypeTo(DType.From(from), DType.From(to), DTypeCasting.ParseCasting(casting));

            for (int i = 0; i < operands.Length; i++)
            {
                if (operands[i].typecode != computeType && !CanCast(operands[i].typecode, computeType))
                    throw new TypeError(
                        $"Iterator operand {i} dtype could not be cast from " +
                        $"dtype('{operands[i].typecode.AsNumpyDtypeName()}') to " +
                        $"dtype('{computeType.AsNumpyDtypeName()}') according to the rule '{casting}'");
            }

            // The out is the iterator's last operand, READWRITE: NpyIter checks the read cast (out -> loop) and then
            // the write cast (loop -> out) under the same rule, before anything is computed.
            if (@out is not null && @out.typecode != computeType)
            {
                if (!CanCast(@out.typecode, computeType))
                    throw new TypeError(
                        $"Iterator operand {operands.Length} dtype could not be cast from " +
                        $"dtype('{@out.typecode.AsNumpyDtypeName()}') to " +
                        $"dtype('{computeType.AsNumpyDtypeName()}') according to the rule '{casting}'");
                if (!CanCast(computeType, @out.typecode))
                    throw new TypeError(
                        $"Iterator requested dtype could not be cast from " +
                        $"dtype('{computeType.AsNumpyDtypeName()}') to " +
                        $"dtype('{@out.typecode.AsNumpyDtypeName()}'), the operand {operands.Length} dtype, " +
                        $"according to the rule '{casting}'");
            }

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

            // Rank was validated at parse; a same-rank extent mismatch is reported here. NumPy leaks
            // NpyIter's "remapped shapes" text for this (the same iterator wording the label-extent
            // mismatch gets), so — like that case — NumSharp words it about the contraction instead.
            if (!ShapesEqual(@out.Shape.dimensions, result.Shape.dimensions))
                throw new ValueError(
                    $"einsum() output parameter has shape ({string.Join(",", @out.Shape.dimensions)}) " +
                    $"but the contraction produces ({string.Join(",", result.Shape.dimensions)})");

            // The result must reach out='s dtype under the SAME casting rule as the inputs. NumPy's
            // iterator names out as operand <nop> in this message.
            if (@out.typecode != result.typecode && !CanCast(result.typecode, @out.typecode))
                throw new TypeError(
                    $"Iterator requested dtype could not be cast from " +
                    $"dtype('{result.typecode.AsNumpyDtypeName()}') to " +
                    $"dtype('{@out.typecode.AsNumpyDtypeName()}'), the operand {operands.Length} dtype, " +
                    $"according to the rule '{casting}'");

            UnmanagedStorage.Copyto(@out, result, NPY_CASTING.NPY_UNSAFE_CASTING, null);   // np.copyto(out, result, 'unsafe')
            return @out;
        }

        /// <summary>
        ///     The labels of a pairwise contraction that must survive it: those in the final output,
        ///     plus any still needed by a later operand — in first-appearance order across the two
        ///     terms. bmm handles arranging the result into this order.
        /// </summary>
        /// <param name="ta">The first operand's labels.</param>
        /// <param name="tb">The second operand's labels.</param>
        /// <param name="outputTerm">The final output's labels.</param>
        /// <param name="terms">The fold's pending operands with their labels.</param>
        /// <param name="fromIndex">The first pending entry still to be contracted (the two being contracted come before
        ///     it).</param>
        /// <returns>The labels this step keeps, in first-appearance order across <paramref name="ta"/> then
        ///     <paramref name="tb"/>.</returns>
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
        /// <param name="a">The operand.</param>
        /// <param name="srcTerm">Its labels, one per axis.</param>
        /// <param name="dstTerm">The labels to produce, in their order.</param>
        /// <param name="computeType">The loop dtype: the sums run in it and the result is cast to it.</param>
        /// <returns>The diagonals, sums and transpose of <paramref name="a"/> in <paramref name="dstTerm"/>'s order, in
        ///     <paramref name="computeType"/> (a view of <paramref name="a"/> when nothing is summed or
        ///     cast).</returns>
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
                arr = NDArray.Diagonal(arr, 0, p1, p2);
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
                arr = arr.TensorEngine.Sum(arr, axis: toSum[k], dtype: computeType, keepdims: false);
                term.RemoveAt(toSum[k]);
            }

            // 3) Transpose the remaining axes into the destination order.
            if (!SameOrder(term, dstTerm))
            {
                var perm = new int[dstTerm.Length];
                for (int i = 0; i < dstTerm.Length; i++)
                    perm[i] = term.IndexOf(dstTerm[i]);
                arr = arr.TensorEngine.Transpose(arr, perm);
            }

            if (arr.typecode != computeType)
                arr = arr.astype(computeType);

            return arr;
        }

        /// <summary>
        ///     A single pairwise contraction <c>a[aTerm], b[bTerm] -&gt; outTerm</c> — port of
        ///     <c>bmm_einsum</c> + <c>_parse_eq_to_batch_matmul</c>. Every index is classified from
        ///     the operands' ACTUAL shapes (so size-1 broadcasting is handled exactly as NumPy does),
        ///     then the contraction runs as one <c>np.matmul</c> (batched when there are shared
        ///     output indices) or, when nothing is contracted, one broadcast <c>np.multiply</c>.
        /// </summary>
        /// <param name="a">The first operand, already in the loop dtype.</param>
        /// <param name="aTerm">Its labels.</param>
        /// <param name="b">The second operand, already in the loop dtype.</param>
        /// <param name="bTerm">Its labels.</param>
        /// <param name="outTerm">The labels the step keeps, in their order.</param>
        /// <param name="computeType">The loop dtype.</param>
        /// <returns>The contraction, labelled by <paramref name="outTerm"/>.</returns>
        /// <exception cref="ValueError">A label whose extents (both above 1) differ between the operands; the parse
        ///     rejects that first, so it is a guard.</exception>
        [NDScopedCovered] // only reached from [NDScoped] Einsum's fold (superseded left/right/ab stages are scope-reclaimed)
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
                left = left.reshape(newShapeA);

            var right = desiredB != bTerm ? SingleTermEinsum(b, bTerm, desiredB, computeType) : b;
            if (newShapeB != null)
                right = right.reshape(newShapeB);

            var ab = left.TensorEngine.Matmul(left, right, null, null, null, null, null, "same_kind", 'K');   // np.matmul's defaults

            if (newShapeAB != null)
                ab = ab.reshape(newShapeAB);
            if (permAB != null)
                ab = ab.TensorEngine.Transpose(ab, permAB);

            return ab;
        }

        /// <summary>
        ///     The no-contraction case — an outer / Hadamard / broadcast product. Each operand is
        ///     transposed to the output's order and reshaped with length-1 axes wherever it lacks an
        ///     output label, then a single broadcast <c>np.multiply</c> forms the result.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="aTerm">Its labels.</param>
        /// <param name="shapeA">Its dimensions.</param>
        /// <param name="b">The second operand.</param>
        /// <param name="bTerm">Its labels.</param>
        /// <param name="shapeB">Its dimensions.</param>
        /// <param name="outTerm">The output labels, in their order.</param>
        /// <param name="computeType">The loop dtype.</param>
        /// <returns>The broadcast product, labelled by <paramref name="outTerm"/>.</returns>
        [NDScopedCovered] // only reached from PairwiseContract ← [NDScoped] Einsum (superseded left/right stages are scope-reclaimed)
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
            left = left.reshape(newShapeA);

            var right = desiredB.ToString() != bTerm ? SingleTermEinsum(b, bTerm, desiredB.ToString(), computeType) : b;
            right = right.reshape(newShapeB);

            return left.TensorEngine.Multiply(left, right);
        }

        /// <summary>Product of each group's label extents; null when every group is already a single axis.</summary>
        /// <param name="groups">The label groups, one per target axis.</param>
        /// <param name="sizes">Each label's extent.</param>
        /// <returns>One extent per group (the product of its labels' extents), or null when no group needs
        ///     fusing.</returns>
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

        /// <summary>The output shape after the matmul — length-1 axes for the size-1 output labels, then the un-fused
        ///     groups.</summary>
        /// <param name="groups">The matmul's output groups (batch, then each operand's kept labels).</param>
        /// <param name="singletonCount">How many size-1 output labels lead the shape.</param>
        /// <param name="sizes">Each label's extent.</param>
        /// <returns>The un-fused output shape, or null when the matmul's result already has it.</returns>
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

        /// <summary>
        ///     Resolves the computed result's memory layout. Probed against 2.4.2: <c>'A'</c> AND
        ///     <c>'K'</c> both come back F-contiguous when EVERY input is F-contiguous (a matmul,
        ///     hadamard or 3-op chain over all-F operands is F even at the default <c>order='K'</c>),
        ///     and C otherwise — where <c>'K'</c> leaves an already-computed C result untouched.
        ///     The single-operand VIEW path never reaches this method (order is ignored there).
        /// </summary>
        /// <param name="result">The computed result.</param>
        /// <param name="order">The checked order letter: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <param name="operands">The einsum operands (their F-contiguity decides <c>'A'</c> and <c>'K'</c>).</param>
        /// <returns><paramref name="result"/> itself, or a C- or F-ordered copy of it.</returns>
        private static NDArray ApplyOrder(NDArray result, char order, NDArray[] operands)
        {
            // A scalar result has no physical axis order. The PUBLIC ascontiguousarray /
            // asfortranarray APIs intentionally promote 0-D to shape (1,), per NumPy, but using
            // them here would corrupt einsum('i,i->')'s contractual scalar shape ().
            if (result.ndim == 0)
                return result;

            // np.ascontiguousarray's statement is NDArray.AsContiguousArray; np.asfortranarray's is the conversion in F
            // order, then a reshape to (1,) of a 0-d result, which cannot fire here (a 0-d result returned above).
            switch (order)
            {
                case 'C':
                    return NDArray.AsContiguousArray(result, null);
                case 'F':
                    return NDArray.ArrayFromObject(result, null, 'F', null);
                default:                 // 'A' and 'K'
                    bool allF = true;
                    foreach (var o in operands)
                    {
                        if (o.Shape.IsFContiguous)
                            continue;
                        allF = false;
                        break;
                    }

                    if (allF)
                        return NDArray.ArrayFromObject(result, null, 'F', null);
                    return order == 'A' ? NDArray.AsContiguousArray(result, null) : result;
            }
        }

        /// <summary>Joins three label groups into one term, in order (a matmul operand's desired label
        ///     order).</summary>
        /// <param name="a">The first group.</param>
        /// <param name="b">The second group.</param>
        /// <param name="c">The third group.</param>
        /// <returns>The labels of <paramref name="a"/>, then <paramref name="b"/>, then <paramref name="c"/>.</returns>
        private static string Concat(List<char> a, List<char> b, List<char> c)
        {
            var sb = new StringBuilder(a.Count + b.Count + c.Count);
            foreach (char x in a) sb.Append(x);
            foreach (char x in b) sb.Append(x);
            foreach (char x in c) sb.Append(x);
            return sb.ToString();
        }

        /// <summary>Whether the remaining labels already are the destination term, so no transpose is needed.</summary>
        /// <param name="term">The remaining labels.</param>
        /// <param name="dstTerm">The destination term.</param>
        /// <returns>True when both hold the same labels in the same order.</returns>
        private static bool SameOrder(List<char> term, string dstTerm)
        {
            if (term.Count != dstTerm.Length)
                return false;
            for (int i = 0; i < term.Count; i++)
                if (term[i] != dstTerm[i])
                    return false;
            return true;
        }

        /// <summary>Whether two dimension lists are equal (<c>out=</c>'s check against the result).</summary>
        /// <param name="a">The first dimensions.</param>
        /// <param name="b">The second dimensions.</param>
        /// <returns>True when both have the same rank and extents.</returns>
        private static bool ShapesEqual(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>Renders a label for a message — a real ASCII letter as itself, a reserved ellipsis slot as
        ///     "...".</summary>
        /// <param name="label">The label: an ASCII letter, or a reserved slot of the ellipsis (from U+E000).</param>
        /// <returns>The letter, or <c>...</c>.</returns>
        private static string Describe(char label) => label >= 0xE000 ? "..." : label.ToString();

        /// <summary>
        ///     NumPy's check of <c>optimize=</c>: <c>einsum</c> calls <c>c_einsum</c> directly for <c>False</c> (NumSharp's
        ///     null default too) and hands anything else to <c>einsum_path</c>, whose <c>path_type</c> branches accept
        ///     <c>True</c>, any string, an explicit path (an <see cref="EinsumPath"/> or a list headed
        ///     <c>'einsum_path'</c>) and a <c>(string, int | float)</c> pair (a bool is an int there), and refuse the rest:
        ///     the same resolution <c>np.einsum_path</c> runs (<c>EinsumPathPlanner.ResolveOptimize</c>). NumSharp contracts left to right
        ///     whatever the path says, so the resolved directive is dropped.
        /// </summary>
        /// <param name="optimize">The <c>optimize=</c> argument.</param>
        /// <exception cref="TypeError">A number (<c>object of type 'int' has no len()</c>: NumPy's <c>len(path_type)</c>),
        ///     or any other value NumPy does not understand (<c>Did not understand the path: ...</c>).</exception>
        private static void RequireOptimize(object optimize)
        {
            if (optimize is null || optimize is false)
                return;
            EinsumPathPlanner.ResolveOptimize(optimize);
        }

        /// <summary>
        ///     The operands followed by <c>out=</c>: the arrays NumPy's einsum iterator resolves its common dtype over
        ///     when no <c>dtype=</c> is given.
        /// </summary>
        /// <param name="operands">The einsum operands.</param>
        /// <param name="out">The <c>out=</c> array (not null).</param>
        /// <returns>A new array of the operands and the out, in that order.</returns>
        private static NDArray[] WithOut(NDArray[] operands, NDArray @out)
        {
            var all = new NDArray[operands.Length + 1];
            Array.Copy(operands, all, operands.Length);
            all[operands.Length] = @out;
            return all;
        }
    }
}
