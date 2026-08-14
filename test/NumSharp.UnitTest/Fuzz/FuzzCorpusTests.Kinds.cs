using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     The result-kind and error-parity half of the differential matrix.
    ///
    ///     The original corpus could only express ONE comparable thing: a single NDArray, checked
    ///     as (dtype, shape, C-contiguous bytes). That shape excluded three whole classes of op
    ///     from the gate — ones returning a tuple of arrays, ones returning a dtype or a scalar,
    ///     and ones returning text — and it reduced every raising case to "something was thrown".
    ///     <see cref="FuzzCorpus.Expected.Kind"/> and <see cref="FuzzCorpus.Case.Error"/> close
    ///     that gap; the comparators live here so the shared harness file stays thin.
    /// </summary>
    public partial class FuzzCorpusTests
    {
        // ---- new tiers ---------------------------------------------------------------------

        /// <summary>
        ///     Iterator TRACES. np.ndindex / np.ndenumerate / np.nditer / np.broadcast produce no
        ///     array, so they were deliberately absent from this corpus — but their MATERIALIZED
        ///     iteration is an ordinary array artifact, and it is the one that matters: traversal
        ///     ORDER. NumPy is the only oracle for how C/F/A/K resolve, how external_loop chunks,
        ///     and what a strided/broadcast/F-contiguous operand yields. Every other tier depends
        ///     on NDIter agreeing with NumPy here, so nothing else in the corpus can catch a drift
        ///     in it.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Iter() => RunCorpus("iter.jsonl");

        /// <summary>
        ///     The non-array result kinds: dtype-returning promotion helpers (result_type /
        ///     promote_types / min_scalar_type), scalar-returning predicates (can_cast / isscalar /
        ///     iscomplexobj / isrealobj / size), text-returning printing (array_str / array_repr),
        ///     and tuple-returning selection (nonzero / meshgrid) where ARITY is asserted too.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DtypeText() => RunCorpus("dtype_text.jsonl");

        /// <summary>
        ///     Error parity with NumPy's ACTUAL message. The deterministic op matrices are re-run
        ///     by the generator keeping only the cells where NumPy raises — cells the value tiers
        ///     silently skip — and each records the exception type and text verbatim. This is a
        ///     separate corpus on purpose: it leaves the (large, shared) value tiers byte-identical
        ///     rather than rewriting 87K committed lines to interleave error rows.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void ErrorsFull() => RunCorpus("errors_full.jsonl");

        /// <summary>
        ///     ufunc <c>out=</c> / <c>where=</c>. Previously reached only through maximum_out /
        ///     minimum_out / clip_out (11 cases each, contiguous out, no mask), so nothing these
        ///     parameters actually promise was gated. Every case records TWO slots — what the call
        ///     returned, and the ENTIRE base buffer behind <c>out</c> — because:
        ///
        ///     <list type="bullet">
        ///       <item><c>where</c> masking is defined by what does NOT change, so the out array's
        ///             prior contents are an operand and are re-checked afterwards.</item>
        ///       <item>a strided / offset / negstride / F-order / transposed <c>out</c> is where a
        ///             kernel that walks the buffer instead of the view corrupts elements OUTSIDE
        ///             the window — which a view-shaped comparison cannot see.</item>
        ///     </list>
        ///
        ///     A read-only (broadcast) <c>out</c> must be refused; those ride the error machinery.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void OutWhere() => RunCorpus("out_where.jsonl");

        // ---- comparators -------------------------------------------------------------------

        /// <summary>
        ///     The original (dtype, shape, bytes) comparison, factored out so tuple slots reuse it
        ///     verbatim. <paramref name="slot"/> labels which tuple slot is being compared (null
        ///     for a plain result) so a failure names it.
        /// </summary>
        private static void CompareArray(
            FuzzCorpus.Case c, NDArray result, FuzzCorpus.Expected exp, string slot,
            List<string> failures, Dictionary<string, int> documented)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            var tc = FuzzCorpus.DtypeToTC(exp.Dtype);
            string at = slot == null ? "" : $" {slot}";

            // NEP50: result dtype must match NumPy exactly (the headline promotion failure).
            if (result.typecode != tc)
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Dtype, null, null, tc, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{c.Id} [{c.Layout}]{at}: result dtype {result.typecode} != NumPy {exp.Dtype}");
                return;
            }
            // Broadcasting: result shape must match NumPy.
            if (!ShapeEquals(result.Shape.dimensions, exp.Shape))
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Shape, null, null, tc, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{c.Id} [{c.Layout}]{at}: result shape [{string.Join(",", result.Shape.dimensions)}] " +
                                  $"!= NumPy [{string.Join(",", exp.Shape)}]");
                return;
            }

            var actual = FuzzCorpus.ResultBytes(result);
            var expected = FuzzCorpus.FromHex(exp.Buffer);

            // Bit-exact to NumPy ("precise") passes HERE, before truth is ever read — matching
            // NumPy's bytes is the contract, and truth can never turn a precise result red.
            var diffs = BitDiff.Compare(expected, actual, tc);
            if (diffs.Count == 0)
                return;

            var truth = exp.Truth == null ? null : FuzzCorpus.FromHex(exp.Truth);
            var vreason = MisalignedRegistry.Classify(c, DivergenceKind.Value, expected, actual, tc, diffs, truth);
            if (vreason != null)
            {
                Bump(documented, vreason);
                return;
            }

            // Shrinking rebuilds the case as a 1-element repro, which only makes sense for the
            // elementwise single-array shape — skip it for tuple slots.
            var shrunk = slot == null ? Shrinker.ShrinkElementwise(c, diffs[0].Index) : null;
            failures.Add($"{c.Id} [{c.Layout}]{at}: " +
                string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}" +
                    TruthNote(expected, actual, truth, d.Index, tc))) +
                (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : "") +
                (shrunk != null ? $"\n      minimal repro: {shrunk}" : ""));
        }

        /// <summary>
        ///     For a truth-bearing failure, say WHO lost precision right in the failure line:
        ///     NumSharp's and NumPy's ULP distances to the correctly-rounded reference. An
        ///     untruthful divergence then reads e.g. "(truth-ulp NS=512 NPY=2)" — the precision
        ///     loss quantified — instead of two opaque hex tokens.
        /// </summary>
        private static string TruthNote(byte[] expected, byte[] actual, byte[] truth, int index, NPTypeCode tc)
        {
            if (truth == null || truth.Length != expected.Length)
                return "";
            long dNS = BitDiff.UlpDistance(actual, truth, index, tc);
            long dNPY = BitDiff.UlpDistance(expected, truth, index, tc);
            static string F(long d) => d == long.MaxValue ? "max" : d.ToString();
            return $" (truth-ulp NS={F(dNS)} NPY={F(dNPY)})";
        }

        /// <summary>
        ///     Multi-output ops: compare the ARITY first, then every slot. The older which/piece
        ///     params record one slot per case and so can never catch a tuple that came back with
        ///     the wrong number of entries (np.nonzero on a 2-D array being the standing example).
        /// </summary>
        private static void CompareTuple(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var got = OpRegistry.ApplyTuple(c.Op, c.Params, operands);
            var want = c.Expected.Slots ?? Array.Empty<FuzzCorpus.Expected>();

            if (got.Length != want.Length)
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Arity, null, null, default,
                                                         Array.Empty<BitDiff.Diff>());
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{c.Id} [{c.Layout}]: tuple arity {got.Length} != NumPy {want.Length}");
                return;
            }

            for (int i = 0; i < want.Length; i++)
                CompareArray(c, got[i], want[i], $"slot[{i}]", failures, documented);
        }

        /// <summary>
        ///     Promotion helpers return a dtype, not an array — compare NumPy's dtype NAME. Routed
        ///     through the registry as a Dtype divergence so the documented NEP50 weak-scalar
        ///     difference covers result_type over a 0-D operand, exactly as it covers the result
        ///     dtype of a binary op over the same operands.
        /// </summary>
        private static void CompareDtypeResult(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var got = OpRegistry.ApplyDtype(c.Op, c.Params, operands).AsNumpyDtypeName();
            if (string.Equals(got, c.Expected.Value, StringComparison.Ordinal))
                return;

            var reason = MisalignedRegistry.Classify(c, DivergenceKind.Dtype, null, null, default,
                                                     Array.Empty<BitDiff.Diff>());
            if (reason != null) Bump(documented, reason);
            else failures.Add($"{c.Id} [{c.Layout}]: dtype result '{got}' != NumPy '{c.Expected.Value}'");
        }

        /// <summary>Printing returns text — compare verbatim (this is a byte-exact port claim).</summary>
        private static void CompareTextResult(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var got = OpRegistry.ApplyText(c.Op, c.Params, operands);
            if (string.Equals(got, c.Expected.Value, StringComparison.Ordinal))
                return;

            var reason = MisalignedRegistry.Classify(c, DivergenceKind.Text, null, null, default,
                                                     Array.Empty<BitDiff.Diff>());
            if (reason != null) Bump(documented, reason);
            else failures.Add($"{c.Id} [{c.Layout}]: text differs\n" +
                              $"      numpy:    {Show(c.Expected.Value)}\n" +
                              $"      numsharp: {Show(got)}");
        }

        // ---- error parity ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's exception classes and the .NET types NumSharp is allowed to answer them
        ///     with. The taxonomies genuinely do not align 1:1 (NumPy's ValueError covers what .NET
        ///     splits across ArgumentException / InvalidOperationException / FormatException, and
        ///     NumSharp deliberately raises its own IncorrectShapeException / AxisOutOfRangeException),
        ///     so the TYPE check is a class-level sanity net and the MESSAGE carries the contract.
        /// </summary>
        private static readonly Dictionary<string, string[]> ErrorTypeMap = new()
        {
            ["ValueError"] = new[]
            {
                "ArgumentException", "ArgumentNullException", "ArgumentOutOfRangeException",
                "IncorrectShapeException", "AxisOutOfRangeException", "InvalidOperationException",
                "FormatException", "OverflowException", "NotSupportedException"
            },
            ["TypeError"] = new[]
            {
                "NotSupportedException", "InvalidCastException", "ArgumentException",
                "ArgumentNullException", "InvalidOperationException", "UFuncTypeException"
            },
            ["IndexError"] = new[]
            {
                "IndexOutOfRangeException", "ArgumentOutOfRangeException", "ArgumentException",
                "AxisOutOfRangeException"
            },
            ["AxisError"] = new[]
            {
                "AxisOutOfRangeException", "ArgumentOutOfRangeException", "ArgumentException",
                "IndexOutOfRangeException"
            },
            ["OverflowError"] = new[] { "OverflowException", "ArgumentException" },
            ["ZeroDivisionError"] = new[] { "DivideByZeroException", "ArithmeticException" },
            ["MemoryError"] = new[] { "OutOfMemoryException" },
            ["NotImplementedError"] = new[] { "NotImplementedException", "NotSupportedException" },
        };

        /// <summary>
        ///     Assert error parity. Three tiers, weakest first:
        ///     (1) it must throw at all — a silent result where NumPy raises is always a failure;
        ///     (2) if NumPy's exception was recorded, the .NET type must be a plausible counterpart;
        ///     (3) and the message must match NumPy's verbatim.
        ///     Tiers 2-3 route through <see cref="MisalignedRegistry"/> so a documented wording
        ///     difference is excused-but-printed rather than silently accepted.
        /// </summary>
        private static void CheckError(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            Exception caught = null;
            try
            {
                _ = OpRegistry.Invoke(c.Expected?.KindOrArray ?? "array", c.Op, c.Params, operands);
            }
            catch (Exception e)
            {
                caught = e;
            }

            if (caught == null)
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Value, null, null, default, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{c.Id} [{c.Layout}]: NumPy raises " +
                                  $"{c.Error?.Type ?? "an error"} but NumSharp produced a result (error-parity gap)");
                return;
            }

            // Legacy cases carry no recorded exception — "it threw" is all they ever asserted.
            if (c.Error == null)
                return;

            string gotType = caught.GetType().Name;
            string gotText = NormalizeMessage(caught.Message);
            string wantText = c.Error.Text ?? "";

            // NumSharp names several exceptions exactly after NumPy's (Exceptions/{ValueError,
            // TypeError,IndexError,AxisError}.cs), so an identical name is the strongest possible
            // match and is accepted before the taxonomy map is consulted at all.
            bool typeOk = gotType == c.Error.Type
                          || !ErrorTypeMap.TryGetValue(c.Error.Type, out var allowed)
                          || allowed.Contains(gotType);
            bool textOk = string.Equals(gotText, wantText, StringComparison.Ordinal);

            if (typeOk && textOk)
                return;

            var er = MisalignedRegistry.Classify(c, DivergenceKind.ErrorText, null, null, default, empty);
            if (er != null)
            {
                Bump(documented, er);
                return;
            }

            failures.Add($"{c.Id} [{c.Layout}]: error mismatch\n" +
                         $"      numpy:    {c.Error.Type}: {Show(wantText)}\n" +
                         $"      numsharp: {gotType}: {Show(gotText)}");
        }

        /// <summary>
        ///     Strip .NET's framing so the comparison is about the MESSAGE, not the platform:
        ///     ArgumentException appends " (Parameter 'name')" and multi-line messages carry the
        ///     parameter on its own line. NumPy's text is otherwise compared byte-for-byte.
        /// </summary>
        private static string NormalizeMessage(string m)
        {
            if (string.IsNullOrEmpty(m))
                return "";

            int p = m.IndexOf(" (Parameter '", StringComparison.Ordinal);
            if (p >= 0)
                m = m.Substring(0, p);

            int nl = m.IndexOf("\nParameter name:", StringComparison.Ordinal);
            if (nl >= 0)
                m = m.Substring(0, nl);

            return m.Trim();
        }

        /// <summary>Render a string for a one-line failure report: escaped and length-capped.</summary>
        private static string Show(string s)
        {
            if (s == null)
                return "<null>";
            s = s.Replace("\r", "\\r").Replace("\n", "\\n");
            return s.Length <= 200 ? $"\"{s}\"" : $"\"{s.Substring(0, 200)}…\" ({s.Length} chars)";
        }
    }
}
