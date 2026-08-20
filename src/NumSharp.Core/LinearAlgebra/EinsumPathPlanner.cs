using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace NumSharp
{
    /// <summary>
    ///     The contraction planner behind <see cref="np.einsum_path(string, NDArray[])"/> — a
    ///     route-for-route port of NumPy 2.4.2's <c>numpy/_core/einsumfunc.py</c>: its
    ///     <c>_parse_einsum_input</c> parser, the <c>_greedy_path</c>/<c>_optimal_path</c> searches
    ///     with their <c>_flop_count</c> cost model, and the <c>einsum_path</c> info-string builder.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     This is DELIBERATELY not the C parser NumSharp uses for the default <c>np.einsum</c>
    ///     (<see cref="EinsumSubscripts"/>, a port of <c>einsum.cpp</c>). NumPy carries two independent
    ///     einsum parsers and <c>einsum_path</c> uses the Python one, which words its rejections
    ///     differently and expands each <c>...</c> into concrete letters. Both agree on the resolved
    ///     terms for every VALID expression; only the error taxonomy and the displayed ellipsis
    ///     letters differ, so the path itself is identical whichever parser runs.
    ///     </para>
    ///     <para>
    ///     <b>One inherent divergence, ellipsis-only.</b> NumPy draws the letters that replace a
    ///     <c>...</c> from <c>list(einsum_symbols_set - set(used))</c> — Python SET iteration order,
    ///     which is per-process hash-randomized, so NumPy's own info string shows DIFFERENT placeholder
    ///     letters on every run (probed: <c>I</c>/<c>f</c>/<c>P</c> across three processes). The
    ///     contraction PATH and every numeric metric are invariant to that choice, so NumSharp draws
    ///     the letters deterministically (einsum_symbols order, taken from the end) — the path and the
    ///     numbers match NumPy exactly; only the placeholder letters an ellipsis expands to differ,
    ///     which NumPy itself does not pin.
    ///     </para>
    /// </remarks>
    internal static class EinsumPathPlanner
    {
        // NumPy's einsum_symbols — UPPER case first, so index order equals ASCII order.
        private const string EinsumSymbols =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

        /// <summary>How the caller's <c>optimize=</c> resolved (see <c>np.einsum_path</c>).</summary>
        internal readonly struct Directive
        {
            internal readonly bool NoOpt;          // path_type is False
            internal readonly string Algorithm;    // "greedy" / "optimal" / any other string
            internal readonly int[][] ExplicitPath; // non-null => path_type[1:]
            internal readonly long? MemoryLimit;   // from the (str, N) tuple form

            internal Directive(bool noOpt, string algorithm, int[][] explicitPath, long? memoryLimit)
            {
                NoOpt = noOpt;
                Algorithm = algorithm;
                ExplicitPath = explicitPath;
                MemoryLimit = memoryLimit;
            }
        }

        /// <summary>
        ///     The port of <c>einsum_path</c>. Takes the raw subscripts and each operand's shape
        ///     (a 0-d/scalar operand is an empty <c>long[]</c>) and returns the contraction path and
        ///     its printable representation.
        /// </summary>
        internal static (int[][] path, string repr) Compute(string subscripts, long[][] shapes, Directive opt)
        {
            var (inputSubscripts, outputSubscript) = ParseEinsumInput(subscripts, shapes);

            string[] inputList = inputSubscripts.Split(',');
            int numInputs = inputList.Length;

            var inputSets = new List<HashSet<char>>(numInputs);
            foreach (string term in inputList)
                inputSets.Add(new HashSet<char>(term));

            var outputSet = new HashSet<char>(outputSubscript);
            var indices = new HashSet<char>(inputSubscripts.Replace(",", ""));
            int numIndices = indices.Count;

            // dimension_dict — resolve each label's extent, taking the largest for broadcasting.
            var dimensionDict = new Dictionary<char, long>();
            for (int tnum = 0; tnum < inputList.Length; tnum++)
            {
                string term = inputList[tnum];
                long[] sh = shapes[tnum];
                if (sh.Length != term.Length)
                    throw new ValueError(
                        $"Einstein sum subscript {inputSubscripts[tnum]} does not contain the correct " +
                        $"number of indices for operand {tnum}.");

                for (int cnum = 0; cnum < term.Length; cnum++)
                {
                    char ch = term[cnum];
                    long dim = sh[cnum];
                    if (dimensionDict.TryGetValue(ch, out long previous))
                    {
                        // For broadcasting cases we always want the largest dim size.
                        if (previous == 1)
                            dimensionDict[ch] = dim;
                        else if (dim != 1 && dim != previous)
                            throw new ValueError(
                                $"Size of label '{ch}' for operand {tnum} ({previous}) " +
                                $"does not match previous terms ({dim}).");
                    }
                    else
                    {
                        dimensionDict[ch] = dim;
                    }
                }
            }

            // Size of each input array plus the output array.
            var sizeList = new List<long>(numInputs + 1);
            foreach (string term in inputList)
                sizeList.Add(ComputeSize(term, dimensionDict));
            sizeList.Add(ComputeSize(outputSubscript, dimensionDict));
            long maxSize = sizeList.Max();

            long memoryArg = opt.MemoryLimit ?? maxSize;

            // Compute the path.
            int[][] path;
            if (opt.ExplicitPath != null)
            {
                path = opt.ExplicitPath;
            }
            else if (opt.NoOpt || numInputs == 1 || numInputs == 2 || indices.SetEquals(outputSet))
            {
                // Nothing to be optimized, leave it to einsum.
                path = new[] { RangeTuple(numInputs) };
            }
            else if (opt.Algorithm == "greedy")
            {
                path = GreedyPath(inputSets, outputSet, dimensionDict, memoryArg);
            }
            else if (opt.Algorithm == "optimal")
            {
                path = OptimalPath(inputSets, outputSet, dimensionDict, memoryArg);
            }
            else
            {
                // NumPy's verbatim (buggy) leak: raise KeyError("Path name %s not found", path_type).
                throw new KeyError($"('Path name %s not found', '{opt.Algorithm}')");
            }

            // Build the contraction list — mutates working copies of input_sets / input_list.
            var costList = new List<long>();
            var scaleList = new List<int>();
            var outSizeList = new List<long>();
            var contractionList = new List<(string einsumStr, string[] remaining)>();

            var workingSets = inputSets;
            var workingList = new List<string>(inputList);

            for (int cnum = 0; cnum < path.Length; cnum++)
            {
                // Remove inds from right to left.
                int[] contractInds = (int[])path[cnum].Clone();
                Array.Sort(contractInds);
                Array.Reverse(contractInds);

                var (outInds, newSets, idxRemoved, idxContract) =
                    FindContraction(contractInds, workingSets, outputSet);
                workingSets = newSets;

                long cost = FlopCount(idxContract, idxRemoved.Count > 0, contractInds.Length, dimensionDict);
                costList.Add(cost);
                scaleList.Add(idxContract.Count);
                outSizeList.Add(ComputeSize(outInds, dimensionDict));

                var tmpInputs = new List<string>(contractInds.Length);
                foreach (int x in contractInds)
                {
                    tmpInputs.Add(workingList[x]);
                    workingList.RemoveAt(x);
                }

                string idxResult;
                if (cnum - path.Length == -1)
                {
                    // Last contraction.
                    idxResult = outputSubscript;
                }
                else
                {
                    var sortResult = new List<(long size, char ch)>(outInds.Count);
                    foreach (char ind in outInds)
                        sortResult.Add((dimensionDict[ind], ind));
                    sortResult.Sort((p, q) => p.size != q.size ? p.size.CompareTo(q.size) : p.ch.CompareTo(q.ch));
                    var rsb = new StringBuilder(sortResult.Count);
                    foreach (var pair in sortResult)
                        rsb.Append(pair.ch);
                    idxResult = rsb.ToString();
                }

                workingList.Add(idxResult);
                string einsumStr = string.Join(",", tmpInputs) + "->" + idxResult;
                contractionList.Add((einsumStr, workingList.ToArray()));
            }

            if (workingList.Count != 1)
                throw new RuntimeError(
                    $"Invalid einsum_path is specified: {workingList.Count - 1} more operands has to be contracted.");

            // The printable representation.
            string overallContraction = inputSubscripts + "->" + outputSubscript;

            long sumUnique = 0;
            foreach (string x in inputList)
                sumUnique += new HashSet<char>(x).Count;
            bool innerProduct = sumUnique - numIndices > 0;
            long naiveCost = FlopCount(indices, innerProduct, numInputs, dimensionDict);

            long optCost = costList.Sum() + 1;
            double speedup = (double)naiveCost / optCost;
            long maxIntermediate = outSizeList.Max();

            var print = new StringBuilder();
            print.Append("  Complete contraction:  ").Append(overallContraction).Append('\n');
            print.Append("         Naive scaling:  ").Append(numIndices).Append('\n');
            print.Append("     Optimized scaling:  ").Append(scaleList.Max()).Append('\n');
            print.Append("      Naive FLOP count:  ").Append(Sci(naiveCost)).Append('\n');
            print.Append("  Optimized FLOP count:  ").Append(Sci(optCost)).Append('\n');
            print.Append("   Theoretical speedup:  ").Append(speedup.ToString("0.000", CultureInfo.InvariantCulture)).Append('\n');
            print.Append("  Largest intermediate:  ").Append(Sci(maxIntermediate)).Append(" elements\n");
            print.Append(new string('-', 74)).Append('\n');
            print.Append("scaling".PadLeft(6)).Append(' ').Append("current".PadLeft(24)).Append(' ')
                .Append("remaining".PadLeft(40)).Append('\n');
            print.Append(new string('-', 74));

            for (int n = 0; n < contractionList.Count; n++)
            {
                var (einsumStr, remaining) = contractionList[n];
                string remainingStr = string.Join(",", remaining) + "->" + outputSubscript;
                print.Append('\n');
                print.Append(scaleList[n].ToString(CultureInfo.InvariantCulture).PadLeft(4)).Append("    ")
                    .Append(einsumStr.PadLeft(24)).Append(' ').Append(remainingStr.PadLeft(40));
            }

            return (path, print.ToString());
        }

        // ---------------------------------------------------------------------------------------
        //  Parser — port of _parse_einsum_input's string branch.
        // ---------------------------------------------------------------------------------------

        private static (string inputSubscripts, string outputSubscript) ParseEinsumInput(
            string rawSubscripts, long[][] shapes)
        {
            string subscripts = rawSubscripts.Replace(" ", "");

            // Ensure all characters are valid.
            foreach (char s in subscripts)
            {
                if (s == '.' || s == ',' || s == '-' || s == '>')
                    continue;
                if (EinsumSymbols.IndexOf(s) < 0)
                    throw new ValueError($"Character {s} is not a valid symbol.");
            }

            // Check for proper "->".
            if (subscripts.IndexOf('-') >= 0 || subscripts.IndexOf('>') >= 0)
            {
                bool invalid = CountChar(subscripts, '-') > 1 || CountChar(subscripts, '>') > 1;
                if (invalid || CountSubstring(subscripts, "->") != 1)
                    throw new ValueError("Subscripts can only contain one '->'.");
            }

            // Parse ellipses.
            if (subscripts.IndexOf('.') >= 0)
                subscripts = ExpandEllipses(subscripts, shapes);

            string inputSubscripts;
            string outputSubscript;
            int arrow = subscripts.IndexOf("->", StringComparison.Ordinal);
            if (arrow >= 0)
            {
                inputSubscripts = subscripts.Substring(0, arrow);
                outputSubscript = subscripts.Substring(arrow + 2);
            }
            else
            {
                inputSubscripts = subscripts;
                outputSubscript = BuildImplicitOutput(subscripts.Replace(",", ""));
            }

            // Make sure output subscripts are in the input.
            foreach (char ch in outputSubscript)
            {
                if (CountChar(outputSubscript, ch) != 1)
                    throw new ValueError($"Output character {ch} appeared more than once in the output.");
                if (inputSubscripts.IndexOf(ch) < 0)
                    throw new ValueError($"Output character {ch} did not appear in the input");
            }

            // Make sure number of operands is equivalent to the number of terms.
            if (inputSubscripts.Split(',').Length != shapes.Length)
                throw new ValueError(
                    "Number of einsum subscripts must be equal to the number of operands.");

            return (inputSubscripts, outputSubscript);
        }

        /// <summary>Port of the "->"-less implicit output build: labels used exactly once, ASCII-sorted.</summary>
        private static string BuildImplicitOutput(string joinedInputs)
        {
            var sb = new StringBuilder();
            foreach (char s in SortedUnique(joinedInputs))
            {
                if (EinsumSymbols.IndexOf(s) < 0)
                    throw new ValueError($"Character {s} is not a valid symbol.");
                if (CountChar(joinedInputs, s) == 1)
                    sb.Append(s);
            }

            return sb.ToString();
        }

        /// <summary>
        ///     Port of <c>_parse_einsum_input</c>'s ellipsis block — expands each <c>...</c> to
        ///     concrete letters and returns the full "in-&gt;out" subscripts string. The chosen letters
        ///     are deterministic (einsum_symbols order); NumPy's are per-process random (see the class
        ///     remarks), but the resolved shape and the path are identical either way.
        /// </summary>
        private static string ExpandEllipses(string subscripts, long[][] shapes)
        {
            string used = subscripts.Replace(".", "").Replace(",", "").Replace("->", "");
            var usedSet = new HashSet<char>(used);
            var pool = new StringBuilder();
            foreach (char c in EinsumSymbols)
            {
                if (!usedSet.Contains(c))
                    pool.Append(c);
            }

            string ellipseInds = pool.ToString();
            int longest = 0;

            string[] splitSubscripts;
            string outputSub = null;
            bool outSub;
            int arrow = subscripts.IndexOf("->", StringComparison.Ordinal);
            if (arrow >= 0)
            {
                splitSubscripts = subscripts.Substring(0, arrow).Split(',');
                outputSub = subscripts.Substring(arrow + 2);
                outSub = true;
            }
            else
            {
                splitSubscripts = subscripts.Split(',');
                outSub = false;
            }

            for (int num = 0; num < splitSubscripts.Length; num++)
            {
                string sub = splitSubscripts[num];
                if (sub.IndexOf('.') < 0)
                    continue;

                if (CountChar(sub, '.') != 3 || CountSubstring(sub, "...") != 1)
                    throw new ValueError("Invalid Ellipses.");

                int ellipseCount;
                if (shapes[num].Length == 0)
                {
                    ellipseCount = 0;
                }
                else
                {
                    ellipseCount = Math.Max(shapes[num].Length, 1);
                    ellipseCount -= sub.Length - 3;
                }

                if (ellipseCount > longest)
                    longest = ellipseCount;

                if (ellipseCount < 0)
                    throw new ValueError("Ellipses lengths do not match.");
                if (ellipseCount == 0)
                    splitSubscripts[num] = sub.Replace("...", "");
                else
                    splitSubscripts[num] = sub.Replace("...", ellipseInds.Substring(ellipseInds.Length - ellipseCount));
            }

            string joined = string.Join(",", splitSubscripts);
            string outEllipse = longest == 0 ? "" : ellipseInds.Substring(ellipseInds.Length - longest);

            if (outSub)
                return joined + "->" + outputSub.Replace("...", outEllipse);

            // Special care for outputless ellipses.
            string outputSubscript = BuildImplicitOutput(joined.Replace(",", ""));
            var outEllipseSet = new HashSet<char>(outEllipse);
            var normal = new StringBuilder();
            foreach (char s in SortedUnique(outputSubscript))
            {
                if (!outEllipseSet.Contains(s))
                    normal.Append(s);
            }

            return joined + "->" + outEllipse + normal;
        }

        // ---------------------------------------------------------------------------------------
        //  Cost model — _compute_size_by_dict, _flop_count, _find_contraction.
        // ---------------------------------------------------------------------------------------

        private static long ComputeSize(IEnumerable<char> indices, Dictionary<char, long> idxDict)
        {
            long ret = 1;
            foreach (char c in indices)
                ret *= idxDict[c];
            return ret;
        }

        private static long FlopCount(IEnumerable<char> idxContraction, bool inner, int numTerms,
            Dictionary<char, long> sizeDictionary)
        {
            long overallSize = ComputeSize(idxContraction, sizeDictionary);
            long opFactor = Math.Max(1, numTerms - 1);
            if (inner)
                opFactor += 1;
            return overallSize * opFactor;
        }

        private static (HashSet<char> newResult, List<HashSet<char>> remaining, HashSet<char> idxRemoved,
            HashSet<char> idxContract) FindContraction(
            IReadOnlyList<int> positions, List<HashSet<char>> inputSets, HashSet<char> outputSet)
        {
            var idxContract = new HashSet<char>();
            var idxRemain = new HashSet<char>(outputSet);
            var remaining = new List<HashSet<char>>();

            for (int ind = 0; ind < inputSets.Count; ind++)
            {
                if (Contains(positions, ind))
                {
                    idxContract.UnionWith(inputSets[ind]);
                }
                else
                {
                    remaining.Add(inputSets[ind]);
                    idxRemain.UnionWith(inputSets[ind]);
                }
            }

            var newResult = new HashSet<char>(idxRemain);
            newResult.IntersectWith(idxContract);
            var idxRemoved = new HashSet<char>(idxContract);
            idxRemoved.ExceptWith(newResult);
            remaining.Add(newResult);

            return (newResult, remaining, idxRemoved, idxContract);
        }

        // ---------------------------------------------------------------------------------------
        //  Greedy path — port of _greedy_path + _parse_possible_contraction + _update_other_results.
        // ---------------------------------------------------------------------------------------

        private readonly struct Candidate
        {
            internal readonly long SortRemovedNeg; // -removed_size
            internal readonly long SortCost;       // cost
            internal readonly int[] Positions;
            internal readonly List<HashSet<char>> NewSets;

            internal Candidate(long sortRemovedNeg, long sortCost, int[] positions, List<HashSet<char>> newSets)
            {
                SortRemovedNeg = sortRemovedNeg;
                SortCost = sortCost;
                Positions = positions;
                NewSets = newSets;
            }

            internal bool Less(Candidate other) =>
                SortRemovedNeg != other.SortRemovedNeg
                    ? SortRemovedNeg < other.SortRemovedNeg
                    : SortCost < other.SortCost;
        }

        private static int[][] GreedyPath(List<HashSet<char>> inputSets, HashSet<char> outputSet,
            Dictionary<char, long> idxDict, long memoryLimit)
        {
            if (inputSets.Count == 1)
                return new[] { new[] { 0 } };
            if (inputSets.Count == 2)
                return new[] { new[] { 0, 1 } };

            var naiveContraction = FindContraction(RangeTuple(inputSets.Count), inputSets, outputSet);
            long naiveCost = FlopCount(naiveContraction.idxContract, naiveContraction.idxRemoved.Count > 0,
                inputSets.Count, idxDict);

            IEnumerable<int[]> combIter = Combinations2(inputSets.Count);
            var knownContractions = new List<Candidate>();
            long pathCost = 0;
            var path = new List<int[]>();

            var curInputSets = inputSets;
            int iterations = curInputSets.Count - 1;

            for (int iteration = 0; iteration < iterations; iteration++)
            {
                foreach (int[] positions in combIter)
                {
                    if (!curInputSets[positions[0]].Overlaps(curInputSets[positions[1]]))
                        continue; // always initially ignore outer products
                    if (TryParseContraction(positions, curInputSets, outputSet, idxDict, memoryLimit,
                            pathCost, naiveCost, out Candidate candidate))
                        knownContractions.Add(candidate);
                }

                if (knownContractions.Count == 0)
                {
                    foreach (int[] positions in Combinations2(curInputSets.Count))
                    {
                        if (TryParseContraction(positions, curInputSets, outputSet, idxDict, memoryLimit,
                                pathCost, naiveCost, out Candidate candidate))
                            knownContractions.Add(candidate);
                    }

                    if (knownContractions.Count == 0)
                    {
                        path.Add(RangeTuple(curInputSets.Count));
                        break;
                    }
                }

                Candidate best = knownContractions[0];
                for (int i = 1; i < knownContractions.Count; i++)
                {
                    if (knownContractions[i].Less(best))
                        best = knownContractions[i];
                }

                knownContractions = UpdateOtherResults(knownContractions, best);
                curInputSets = best.NewSets;
                int newTensorPos = curInputSets.Count - 1;
                combIter = PairsWith(newTensorPos);
                path.Add(best.Positions);
                pathCost += best.SortCost;
            }

            return path.ToArray();
        }

        private static bool TryParseContraction(int[] positions, List<HashSet<char>> inputSets,
            HashSet<char> outputSet, Dictionary<char, long> idxDict, long memoryLimit, long pathCost,
            long naiveCost, out Candidate candidate)
        {
            candidate = default;
            var (idxResult, newInputSets, idxRemoved, idxContract) =
                FindContraction(positions, inputSets, outputSet);

            long newSize = ComputeSize(idxResult, idxDict);
            if (newSize > memoryLimit)
                return false;

            long oldSizes = 0;
            foreach (int p in positions)
                oldSizes += ComputeSize(inputSets[p], idxDict);
            long removedSize = oldSizes - newSize;

            long cost = FlopCount(idxContract, idxRemoved.Count > 0, positions.Length, idxDict);
            if (pathCost + cost > naiveCost)
                return false;

            candidate = new Candidate(-removedSize, cost, positions, newInputSets);
            return true;
        }

        private static List<Candidate> UpdateOtherResults(List<Candidate> results, Candidate best)
        {
            int bx = best.Positions[0], by = best.Positions[1];
            HashSet<char> bestNew = best.NewSets[best.NewSets.Count - 1];
            var mod = new List<Candidate>();

            foreach (Candidate result in results)
            {
                int x = result.Positions[0], y = result.Positions[1];
                if (x == bx || x == by || y == bx || y == by)
                    continue; // ignore results involving tensors just contracted

                List<HashSet<char>> conSets = result.NewSets;
                conSets.RemoveAt(by - (by > x ? 1 : 0) - (by > y ? 1 : 0));
                conSets.RemoveAt(bx - (bx > x ? 1 : 0) - (bx > y ? 1 : 0));
                conSets.Insert(conSets.Count - 1, bestNew);

                int modX = x - (x > bx ? 1 : 0) - (x > by ? 1 : 0);
                int modY = y - (y > bx ? 1 : 0) - (y > by ? 1 : 0);
                mod.Add(new Candidate(result.SortRemovedNeg, result.SortCost, new[] { modX, modY }, conSets));
            }

            return mod;
        }

        // ---------------------------------------------------------------------------------------
        //  Optimal path — port of _optimal_path.
        // ---------------------------------------------------------------------------------------

        private static int[][] OptimalPath(List<HashSet<char>> inputSets, HashSet<char> outputSet,
            Dictionary<char, long> idxDict, long memoryLimit)
        {
            int nInputs = inputSets.Count;
            var fullResults = new List<(long cost, List<int[]> positions, List<HashSet<char>> remaining)>
            {
                (0, new List<int[]>(), inputSets)
            };

            for (int iteration = 0; iteration < nInputs - 1; iteration++)
            {
                var iterResults = new List<(long, List<int[]>, List<HashSet<char>>)>();

                foreach (var (cost, positions, remaining) in fullResults)
                {
                    foreach (int[] con in Combinations2(nInputs - iteration))
                    {
                        var (newResult, newInputSets, idxRemoved, idxContract) =
                            FindContraction(con, remaining, outputSet);

                        long newSize = ComputeSize(newResult, idxDict);
                        if (newSize > memoryLimit)
                            continue;

                        long totalCost = cost + FlopCount(idxContract, idxRemoved.Count > 0, con.Length, idxDict);
                        var newPos = new List<int[]>(positions) { con };
                        iterResults.Add((totalCost, newPos, newInputSets));
                    }
                }

                if (iterResults.Count > 0)
                {
                    fullResults = iterResults;
                }
                else
                {
                    var path = MinByCost(fullResults).positions;
                    path.Add(RangeTuple(nInputs - iteration));
                    return path.ToArray();
                }
            }

            if (fullResults.Count == 0)
                return new[] { RangeTuple(nInputs) };

            return MinByCost(fullResults).positions.ToArray();
        }

        private static (long cost, List<int[]> positions, List<HashSet<char>> remaining) MinByCost(
            List<(long cost, List<int[]> positions, List<HashSet<char>> remaining)> results)
        {
            var best = results[0];
            for (int i = 1; i < results.Count; i++)
            {
                if (results[i].cost < best.cost)
                    best = results[i];
            }

            return best;
        }

        // ---------------------------------------------------------------------------------------
        //  Small helpers.
        // ---------------------------------------------------------------------------------------

        private static int[] RangeTuple(int n)
        {
            var r = new int[n];
            for (int i = 0; i < n; i++)
                r[i] = i;
            return r;
        }

        private static IEnumerable<int[]> Combinations2(int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                    yield return new[] { i, j };
            }
        }

        private static IEnumerable<int[]> PairsWith(int newTensorPos)
        {
            for (int i = 0; i < newTensorPos; i++)
                yield return new[] { i, newTensorPos };
        }

        private static bool Contains(IReadOnlyList<int> positions, int value)
        {
            for (int i = 0; i < positions.Count; i++)
            {
                if (positions[i] == value)
                    return true;
            }

            return false;
        }

        private static int CountChar(string s, char c)
        {
            int count = 0;
            foreach (char ch in s)
            {
                if (ch == c)
                    count++;
            }

            return count;
        }

        private static int CountSubstring(string s, string sub)
        {
            int count = 0, idx = 0;
            while ((idx = s.IndexOf(sub, idx, StringComparison.Ordinal)) >= 0)
            {
                count++;
                idx += sub.Length;
            }

            return count;
        }

        private static IEnumerable<char> SortedUnique(string s)
        {
            var set = new SortedSet<char>();
            foreach (char c in s)
                set.Add(c);
            return set;
        }

        /// <summary>
        ///     Formats a non-negative integer the way Python's <c>f"{v:.3e}"</c> does — 4 significant
        ///     figures, ROUND-HALF-TO-EVEN. Done in integer arithmetic on purpose: .NET's
        ///     <c>"0.000e+00"</c> custom format rounds half-AWAY (13825 → 1.383e+04) where Python rounds
        ///     to even (→ 1.382e+04), which diverges on exact-halfway mantissas. Every value fed here
        ///     (FLOP counts, intermediate sizes) is an integer, so this is exact.
        /// </summary>
        private static string Sci(long value)
        {
            if (value == 0)
                return "0.000e+00";

            bool negative = value < 0;
            ulong v = negative ? (ulong)(-value) : (ulong)value;

            int digits = 0;
            for (ulong t = v; t > 0; t /= 10)
                digits++;
            int exponent = digits - 1;

            ulong mantissa4; // the leading 4 significant digits, 1000..9999
            if (digits <= 4)
            {
                ulong mul = 1;
                for (int i = 0; i < 4 - digits; i++)
                    mul *= 10;
                mantissa4 = v * mul;
            }
            else
            {
                ulong scale = 1;
                for (int i = 0; i < digits - 4; i++)
                    scale *= 10;
                ulong q = v / scale, r = v % scale, half = scale / 2;
                if (r > half || (r == half && (q & 1) == 1)) // round half to even
                    q++;
                if (q == 10000) // carry (e.g. 9999.5 -> 10000 -> 1.000e+(E+1))
                {
                    q = 1000;
                    exponent++;
                }

                mantissa4 = q;
            }

            char expSign = exponent >= 0 ? '+' : '-';
            return string.Create(CultureInfo.InvariantCulture,
                $"{(negative ? "-" : "")}{mantissa4 / 1000}.{mantissa4 % 1000:D3}e{expSign}{Math.Abs(exponent):D2}");
        }
    }
}
