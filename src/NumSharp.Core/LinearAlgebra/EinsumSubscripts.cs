using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    /// <summary>
    ///     A parsed <c>np.einsum</c> subscripts string: one label per dimension of every operand,
    ///     the output's labels, and the shape the contraction would produce.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Port of <c>PyArray_EinsteinSum</c>'s parsing block plus its two helpers
    ///     <c>parse_operand_subscripts</c> and <c>parse_output_subscripts</c>
    ///     (<c>numpy/_core/src/multiarray/einsum.cpp</c>) — the parser <c>np.einsum</c> reaches with
    ///     its DEFAULT <c>optimize=False</c>. NumPy has a second, independent parser in
    ///     <c>numpy/_core/einsumfunc.py</c> for the <c>optimize</c> path, and the two word their
    ///     rejections differently; this reproduces the C one, since that is what a default call hits.
    ///     </para>
    ///     <para>
    ///     <b>The label encoding is NumPy's, not a convenience.</b> Each operand gets one
    ///     <see cref="sbyte"/> per dimension:
    ///     <list type="bullet">
    ///     <item>a POSITIVE value is the ASCII code of the label, at its first occurrence;</item>
    ///     <item>a NEGATIVE value is the offset back to that first occurrence — a repeated label,
    ///     i.e. a diagonal to collapse;</item>
    ///     <item><c>0</c> is a broadcast dimension contributed by an ellipsis.</item>
    ///     </list>
    ///     So <c>"abbcbc"</c> over 6 dimensions becomes <c>[97, 98, -1, 99, -3, -2]</c> and
    ///     <c>"ab...bc"</c> over 6 becomes <c>[97, 98, 0, 0, -3, 99]</c>.
    ///     </para>
    /// </remarks>
    internal sealed class EinsumSubscripts
    {
        /// <summary>NumPy's <c>NPY_MAXDIMS</c>.</summary>
        private const int MaxDims = 64;

        /// <summary>One entry per dimension of each operand; see the encoding note on the class.</summary>
        internal sbyte[][] OperandLabels;

        /// <summary>The output's labels — ASCII codes, with <c>0</c> for each broadcast dimension.</summary>
        internal sbyte[] OutputLabels;

        /// <summary>How many dimensions the ellipsis contributes, across all operands.</summary>
        internal int BroadcastNdim;

        /// <summary>The shape the contraction produces.</summary>
        internal long[] OutputShape;

        /// <summary>True when the subscripts carried no <c>-&gt;</c> and the output was inferred.</summary>
        internal bool ImplicitOutput;

        /// <summary>
        ///     Parses and fully validates a subscripts string against its operands.
        /// </summary>
        /// <param name="out">
        ///     The caller's <c>out=</c>, checked for rank only — exactly where NumPy checks it,
        ///     which is after the output labels are known and before any operand is touched.
        /// </param>
        internal static EinsumSubscripts Parse(string subscripts, NDArray[] operands, NDArray @out)
        {
            int nop = operands.Length;

            // NumPy's own bounds. The lower one is unreachable through np.einsum — its argument
            // parser rejects an empty operand list first — but the upper one is exactly what a
            // 70-operand contraction hits.
            if (nop < 1)
                throw new ValueError("not enough operands provided to einstein sum function");
            if (nop > 32)
                throw new ValueError("too many operands");

            var labelCounts = new int[128];
            int minLabel = 127;
            int maxLabel = 0;

            var operandLabels = new sbyte[nop][];
            int pos = 0;

            for (int iop = 0; iop < nop; iop++)
            {
                // strcspn(subscripts, ",-") — the chunk for this operand ends at the next comma or
                // at the '-' that opens "->". A stray '>' is therefore INSIDE the chunk and is
                // rejected as an invalid subscript rather than as a malformed arrow.
                int length = 0;
                while (pos + length < subscripts.Length
                       && subscripts[pos + length] != ','
                       && subscripts[pos + length] != '-')
                    length++;

                char delimiter = pos + length < subscripts.Length ? subscripts[pos + length] : '\0';

                // Both texts are NumPy's, and both read BACKWARDS: the string is what has more (or
                // fewer) terms, not the operand list. Reproduced verbatim anyway — a caller
                // grepping for NumPy's message must find it.
                if (iop == nop - 1 && delimiter == ',')
                    throw new ValueError(
                        "more operands provided to einstein sum function than specified in the subscripts string");
                if (iop < nop - 1 && delimiter != ',')
                    throw new ValueError(
                        "fewer operands provided to einstein sum function than specified in the subscripts string");

                operandLabels[iop] = ParseOperandLabels(
                    subscripts, pos, length, operands[iop].ndim, iop, labelCounts, ref minLabel, ref maxLabel);

                pos += length;
                if (iop < nop - 1)
                    pos++;
            }

            // The number of broadcast dimensions is the widest ellipsis any one operand contributed.
            int broadcastNdim = 0;
            foreach (var labels in operandLabels)
            {
                int zeros = 0;
                foreach (sbyte label in labels)
                {
                    if (label == 0)
                        zeros++;
                }

                if (zeros > broadcastNdim)
                    broadcastNdim = zeros;
            }

            sbyte[] outputLabels;
            bool implicitOutput = pos >= subscripts.Length;

            if (implicitOutput)
            {
                // No "->": broadcast dimensions first, then every label used EXACTLY ONCE across all
                // operands, in ASCII order — so an upper-case label sorts before a lower-case one.
                var inferred = new List<sbyte>();
                for (int i = 0; i < broadcastNdim; i++)
                    inferred.Add(0);

                for (int label = minLabel; label <= maxLabel; label++)
                {
                    if (labelCounts[label] != 1)
                        continue;
                    if (inferred.Count >= MaxDims)
                        throw new ValueError("einstein sum subscript string has too many distinct labels");
                    inferred.Add((sbyte)label);
                }

                outputLabels = inferred.ToArray();
            }
            else
            {
                if (subscripts[pos] != '-' || pos + 1 >= subscripts.Length || subscripts[pos + 1] != '>')
                    throw new ValueError(
                        "einstein sum subscript string does not contain proper '->' output specified");

                outputLabels = ParseOutputLabels(subscripts, pos + 2, broadcastNdim, labelCounts);
            }

            if (@out is not null && @out.ndim != outputLabels.Length)
                throw new ValueError(
                    $"out parameter does not have the correct number of dimensions, has {@out.ndim} " +
                    $"but should have {outputLabels.Length}");

            var plan = new EinsumSubscripts
            {
                OperandLabels = operandLabels,
                OutputLabels = outputLabels,
                BroadcastNdim = broadcastNdim,
                ImplicitOutput = implicitOutput
            };

            plan.OutputShape = plan.ResolveShape(operands, @out);
            return plan;
        }

        /// <summary>
        ///     Port of <c>parse_operand_subscripts</c>.
        /// </summary>
        private static sbyte[] ParseOperandLabels(string subscripts, int start, int length, int ndim, int iop,
            int[] labelCounts, ref int minLabel, ref int maxLabel)
        {
            var labels = new sbyte[ndim];
            int idim = 0;
            int ellipsis = -1;

            for (int i = 0; i < length; i++)
            {
                char c = subscripts[start + i];

                if (IsAsciiLetter(c))
                {
                    if (idim >= ndim)
                        throw new ValueError(
                            $"einstein sum subscripts string contains too many subscripts for operand {iop}");

                    labels[idim++] = (sbyte)c;
                    if (c < minLabel)
                        minLabel = c;
                    if (c > maxLabel)
                        maxLabel = c;
                    labelCounts[c]++;
                }
                else if (c == '.')
                {
                    // One ellipsis per operand, exactly three dots, and room for all three.
                    if (ellipsis != -1 || i + 2 >= length
                        || subscripts[start + i + 1] != '.' || subscripts[start + i + 2] != '.')
                        throw new ValueError(
                            "einstein sum subscripts string contains a '.' that is not part of an " +
                            $"ellipsis ('...') in operand {iop}");

                    i += 2;
                    ellipsis = idim;
                }
                else if (c != ' ')
                {
                    throw new ValueError(
                        $"invalid subscript '{c}' in einstein sum subscripts string, subscripts must be letters");
                }
            }

            if (ellipsis == -1)
            {
                // Without an ellipsis the labels must cover every dimension. Too MANY was already
                // caught in the loop, so this can only be too few.
                if (idim != ndim)
                    throw new ValueError(
                        "operand has more dimensions than subscripts given in einstein sum, but no " +
                        "'...' ellipsis provided to broadcast the extra dimensions.");
            }
            else if (idim < ndim)
            {
                // Slide the labels that followed the ellipsis to the end, and zero the gap it opened.
                for (int i = 0; i < idim - ellipsis; i++)
                    labels[ndim - i - 1] = labels[idim - i - 1];
                for (int i = 0; i < ndim - idim; i++)
                    labels[ellipsis + i] = 0;
            }

            // Rewrite every repeat as the (negative) offset back to its first occurrence — which is
            // what turns a repeated label into a diagonal later on.
            for (idim = 0; idim < ndim - 1; idim++)
            {
                sbyte label = labels[idim];
                if (label <= 0)
                    continue;

                for (int next = idim + 1; next < ndim; next++)
                {
                    if (labels[next] == label)
                        labels[next] = (sbyte)(idim - next);
                }
            }

            return labels;
        }

        /// <summary>
        ///     Port of <c>parse_output_subscripts</c>.
        /// </summary>
        private static sbyte[] ParseOutputLabels(string subscripts, int start, int broadcastNdim, int[] labelCounts)
        {
            int length = subscripts.Length - start;
            var labels = new List<sbyte>();
            bool ellipsis = false;

            for (int i = 0; i < length; i++)
            {
                char c = subscripts[start + i];

                if (IsAsciiLetter(c))
                {
                    if (subscripts.IndexOf(c, start + i + 1) >= 0)
                        throw new ValueError(
                            $"einstein sum subscripts string includes output subscript '{c}' multiple times");

                    if (labelCounts[c] == 0)
                        throw new ValueError(
                            $"einstein sum subscripts string included output subscript '{c}' which never " +
                            "appeared in an input");

                    if (labels.Count >= MaxDims)
                        throw new ValueError(
                            "einstein sum subscripts string contains too many subscripts in the output");

                    labels.Add((sbyte)c);
                }
                else if (c == '.')
                {
                    if (ellipsis || i + 2 >= length
                        || subscripts[start + i + 1] != '.' || subscripts[start + i + 2] != '.')
                        throw new ValueError(
                            "einstein sum subscripts string contains a '.' that is not part of an " +
                            "ellipsis ('...') in the output");

                    if (labels.Count + broadcastNdim > MaxDims)
                        throw new ValueError(
                            "einstein sum subscripts string contains too many subscripts in the output");

                    i += 2;
                    ellipsis = true;
                    for (int b = 0; b < broadcastNdim; b++)
                        labels.Add(0);
                }
                else if (c != ' ')
                {
                    throw new ValueError(
                        $"invalid subscript '{c}' in einstein sum subscripts string, subscripts must be letters");
                }
            }

            if (!ellipsis && broadcastNdim > 0)
                throw new ValueError(
                    "output has more dimensions than subscripts given in einstein sum, but no '...' " +
                    "ellipsis provided to broadcast the extra dimensions.");

            return labels.ToArray();
        }

        /// <summary>
        ///     Collapses each operand's diagonals, resolves every label's extent across operands,
        ///     and builds the output shape.
        /// </summary>
        private long[] ResolveShape(NDArray[] operands, NDArray @out)
        {
            // NumPy attempts a no-copy view when there is a single operand, no out=, and nothing is
            // summed away. That attempt has its OWN wording for a bad diagonal, so which of the two
            // messages a caller sees depends on all three conditions.
            bool singleOperandView = operands.Length == 1 && @out is null && NothingIsSummed();

            var extents = new long[128];
            for (int i = 0; i < extents.Length; i++)
                extents[i] = -1;

            var broadcastShape = new long[BroadcastNdim];
            for (int i = 0; i < BroadcastNdim; i++)
                broadcastShape[i] = 1;

            // PASS ONE — collapse every operand's diagonals. NumPy runs this for ALL operands
            // (get_combined_dims_view, once per operand) before the iterator ever compares extents
            // ACROSS operands, so an impossible diagonal in operand 1 is reported as a diagonal even
            // when operand 0 already fixed a conflicting extent for the same label. Merging the two
            // passes reverses that and reports the wrong error.
            for (int iop = 0; iop < operands.Length; iop++)
            {
                var labels = OperandLabels[iop];
                var shape = operands[iop].Shape.dimensions;

                for (int idim = 0; idim < labels.Length; idim++)
                {
                    sbyte label = labels[idim];
                    if (label >= 0)
                        continue;

                    int first = idim + label;
                    sbyte original = labels[first];
                    long expected = shape[first];
                    long dim = shape[idim];
                    if (expected == dim)
                        continue;

                    throw new ValueError(singleOperandView
                        ? $"dimensions in single operand for collapsing index '{(char)original}' " +
                          $"don't match ({expected} != {dim})"
                        : $"dimensions in operand {iop} for collapsing index '{(char)original}' " +
                          $"don't match ({expected} != {dim})");
                }
            }

            // PASS TWO — resolve each label's extent across operands, and broadcast the ellipsis.
            for (int iop = 0; iop < operands.Length; iop++)
            {
                var labels = OperandLabels[iop];
                var shape = operands[iop].Shape.dimensions;
                int broadcastSeen = 0;
                int operandBroadcastNdim = 0;

                foreach (sbyte label in labels)
                {
                    if (label == 0)
                        operandBroadcastNdim++;
                }

                for (int idim = 0; idim < labels.Length; idim++)
                {
                    sbyte label = labels[idim];
                    long dim = shape[idim];

                    if (label < 0)
                        continue;   // already folded into its first occurrence

                    if (label == 0)
                    {
                        // Broadcast dimensions align to the RIGHT of the global block, as always.
                        int slot = BroadcastNdim - operandBroadcastNdim + broadcastSeen;
                        broadcastShape[slot] = BroadcastExtent(broadcastShape[slot], dim, iop);
                        broadcastSeen++;
                        continue;
                    }

                    long previous = extents[label];
                    if (previous < 0 || previous == 1)
                    {
                        extents[label] = dim;
                    }
                    else if (dim != 1 && dim != previous)
                    {
                        // NumPy's DEFAULT path leaks NpyIter's "remapped shapes" text here, which
                        // describes the iterator's axis bookkeeping rather than the contraction. This
                        // is NumPy's OWN wording for the identical error, from its einsumfunc.py
                        // parser — and note the two sizes read swapped against the sentence: the
                        // first is the size already recorded, the second the one just seen.
                        throw new ValueError(
                            $"Size of label '{(char)label}' for operand {iop} ({previous}) " +
                            $"does not match previous terms ({dim}).");
                    }
                }
            }

            var outputShape = new long[OutputLabels.Length];
            int broadcastIndex = 0;
            for (int i = 0; i < OutputLabels.Length; i++)
            {
                sbyte label = OutputLabels[i];
                outputShape[i] = label == 0 ? broadcastShape[broadcastIndex++] : extents[label];
            }

            return outputShape;
        }

        /// <summary>
        ///     True when every label an operand carries also appears in the output — the condition
        ///     under which NumPy can answer with a view instead of contracting.
        /// </summary>
        private bool NothingIsSummed()
        {
            foreach (var labels in OperandLabels)
            {
                foreach (sbyte label in labels)
                {
                    if (label <= 0)
                        continue;

                    bool inOutput = false;
                    foreach (sbyte outLabel in OutputLabels)
                    {
                        if (outLabel != label)
                            continue;
                        inOutput = true;
                        break;
                    }

                    if (!inOutput)
                        return false;
                }
            }

            return true;
        }

        private long BroadcastExtent(long current, long candidate, int iop)
        {
            if (current == 1 || current == candidate)
                return candidate;
            if (candidate == 1)
                return current;

            throw new ValueError(
                $"operands could not be broadcast together: the ellipsis dimensions of operand {iop} " +
                $"give {candidate} where an earlier operand gave {current}");
        }

        private static bool IsAsciiLetter(char c) => (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');

        /// <summary>
        ///     Renders NumPy's <c>einsum(op0, sublist0, op1, sublist1, …, [sublistout])</c> spelling
        ///     as the equivalent subscripts string, which is what NumPy itself does before parsing.
        /// </summary>
        /// <remarks>
        ///     The alphabet is NumPy's <c>einsum_symbols</c>, and its order is load-bearing: index
        ///     0-25 are <c>A-Z</c> and 26-51 are <c>a-z</c> — UPPER case first. Because that makes
        ///     index order and ASCII order the same, an inferred output comes out in the caller's
        ///     numbering. Reversing the two halves (the intuitive guess) is silently wrong rather
        ///     than an error: <c>einsum(a, [0, 26])</c> on a <c>(2,3)</c> operand would return
        ///     <c>(3,2)</c> instead of <c>(2,3)</c>.
        /// </remarks>
        internal static string FromSublists(object[] arguments, out NDArray[] operands)
        {
            const string symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

            var collected = new List<NDArray>();
            var terms = new List<string>();
            string outputTerm = null;

            int i = 0;
            while (i < arguments.Length)
            {
                if (arguments[i] is not NDArray operand)
                {
                    // A trailing sublist with no operand before it is the OUTPUT specification.
                    if (i == arguments.Length - 1 && collected.Count > 0)
                    {
                        outputTerm = RenderSublist(arguments[i], symbols);
                        break;
                    }

                    throw new TypeError("each subscript must be either an integer or an ellipsis");
                }

                if (i + 1 >= arguments.Length)
                    throw new TypeError("each subscript must be either an integer or an ellipsis");

                collected.Add(operand);
                terms.Add(RenderSublist(arguments[i + 1], symbols));
                i += 2;
            }

            operands = collected.ToArray();

            var text = new StringBuilder(string.Join(",", terms));
            if (outputTerm is not null)
                text.Append("->").Append(outputTerm);

            return text.ToString();
        }

        private static string RenderSublist(object sublist, string symbols)
        {
            IEnumerable<object> entries = sublist switch
            {
                int[] ints => Box(ints),
                object[] objects => objects,
                System.Collections.IEnumerable sequence and not string => Enumerate(sequence),
                _ => throw new TypeError("each subscript must be either an integer or an ellipsis")
            };

            var text = new StringBuilder();
            foreach (object entry in entries)
            {
                if (entry is Slice slice && slice.IsEllipsis)
                {
                    text.Append("...");
                    continue;
                }

                if (entry is not int index)
                {
                    try
                    {
                        index = Convert.ToInt32(entry);
                    }
                    catch (Exception e) when (e is FormatException or InvalidCastException or OverflowException
                                                  or ArgumentNullException)
                    {
                        throw new TypeError("each subscript must be either an integer or an ellipsis");
                    }
                }

                if (index < 0 || index >= symbols.Length)
                    throw new ValueError("subscript is not within the valid range [0, 52)");

                text.Append(symbols[index]);
            }

            return text.ToString();
        }

        private static IEnumerable<object> Box(int[] values)
        {
            foreach (int value in values)
                yield return value;
        }

        private static IEnumerable<object> Enumerate(System.Collections.IEnumerable sequence)
        {
            foreach (object item in sequence)
                yield return item;
        }
    }
}
