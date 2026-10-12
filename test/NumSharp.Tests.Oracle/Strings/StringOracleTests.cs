using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     Replays the NumPy 2.4.2 string oracle (<c>Strings/corpus/strings.jsonl</c>) against NumSharp's fixed-width
    ///     string dtypes, one test per corpus section. Every case is compared byte for byte: an array result by its
    ///     descriptor, shape, C-order bytes, strides, flags and memory sharing; an in-place call by its target AND the whole
    ///     base buffer around it; a value, text, bool, descriptor or attribute set exactly; an error by NumPy's exception
    ///     class and verbatim message.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Phases.</b> A phase-1 case must match NumPy now. A phase-2 case is one NumPy answers by COMPUTING on text
    ///     (sorting, searching and the rest of the computational phase): it passes as "deferred" while NumSharp throws
    ///     <see cref="StringComputationNotSupportedException"/>, and is held to NumPy's answer the moment NumSharp returns
    ///     a result or throws anything else — so the computational phase is gated by this same corpus without editing it.
    ///     </para>
    ///     <para>
    ///     <b>Deferral ceilings.</b> A section may not defer more cases than <see cref="DeferralCeilings"/> allows: a
    ///     computation that NumSharp answers today (a comparison, a conversion between text and numbers) cannot slide back
    ///     into a deferral without failing the gate, even where the generator labels its case phase 2 (a computing call
    ///     whose phase follows NumPy's outcome). Lower a ceiling when a later phase removes deferrals.
    ///     </para>
    ///     <para>
    ///     <b>Comparison details.</b> Strides and contiguity are compared only on axes of more than one element of a
    ///     non-empty result (NumPy's strides elsewhere are arbitrary); memory sharing is <c>np.shares_memory</c> with each
    ///     operand.
    ///     </para>
    /// </remarks>
    [TestClass]
    [TestCategory("StringOracle")]
    public class StringOracleTests
    {
        private static List<StringOracleCorpus.Case> _cases;

        /// <summary>The corpus, loaded once per test run.</summary>
        private static List<StringOracleCorpus.Case> Cases => _cases ??= StringOracleCorpus.Load();

        /// <summary>The descriptors: attributes, promotion, result_type, can_cast, issubdtype.</summary>
        [TestMethod] public void Dtypes() => Run("dtype");

        /// <summary>Creation from C# values and from shapes, the *_like functions, asarray and frombuffer.</summary>
        [TestMethod] public void Creation() => Run("creation");

        /// <summary>Casts between string dtypes of every width, class, order and casting rule.</summary>
        [TestMethod] public void Astype() => Run("astype");

        /// <summary>Views: transposes, reshapes, flips, broadcasts, reinterpreting views, basic and advanced indexing.</summary>
        [TestMethod] public void Views() => Run("views");

        /// <summary>Copies and element movement: copy, take, repeat, pad, concatenate, where, split and the rest.</summary>
        [TestMethod] public void Movement() => Run("movement");

        /// <summary>In-place writes: setitem, fill, put, place, putmask, copyto, put_along_axis.</summary>
        [TestMethod] public void Mutation() => Run("mutation");

        /// <summary>Element access: item, tolist, iteration.</summary>
        [TestMethod] public void Access() => Run("access");

        /// <summary>Printing: str / repr / array2string / array_repr / array_str and print options.</summary>
        [TestMethod] public void Printing() => Run("printing");

        /// <summary>Ufuncs and comparison operators with string operands.</summary>
        [TestMethod] public void Ufuncs() => Run("ufuncs");

        /// <summary>Reductions and the other functions NumPy refuses or computes for text.</summary>
        [TestMethod] public void Reductions() => Run("reductions");

        /// <summary>
        ///     An unsized string <c>dtype=</c> request (<c>'U'</c>, <c>'S'</c>) on every ufunc, reduction, scan, gufunc,
        ///     <c>np.clip</c>, <c>np.einsum</c> and <c>np.trace</c>, over both string classes and every numeric dtype: NumPy's
        ///     refusal, which names the requested CLASS, or its string loop (the next phase).
        /// </summary>
        [TestMethod] public void StringDTypeRequests() => Run("strdtype");

        /// <summary>
        ///     The order NumPy reports a text ufunc call's problems in: a read-only <c>out=</c>, a non-bool <c>where=</c>,
        ///     the loop answer, a computing loop's output cast and broadcast, and <c>np.divmod</c>'s two outputs.
        /// </summary>
        [TestMethod] public void TextCallPrecedence() => Run("strprec");

        /// <summary>
        ///     The axis rules of every reduction-style function over text (ufunc.reduce, the counting statistics, the
        ///     ravelling arg / scan / sort family, tuple axes), the quantile family's q / axis / method / empty-axis order,
        ///     <c>np.average</c>'s weights, partition's kth bounds and the in-place sort's read-only check.
        /// </summary>
        [TestMethod] public void TextReductionAxes() => Run("straxis");

        /// <summary>
        ///     <c>np.cov</c> / <c>np.corrcoef</c> over text: the descriptor the observations convert to, the text-to-text
        ///     conversion errors, the empty and <c>y</c> shapes, and the weight checks, before the first reduction.
        /// </summary>
        [TestMethod] public void TextCovariance() => Run("strcov");

        /// <summary>
        ///     The text calls whose answer the SHAPE decides — truth tests, counts, index searches, sorts, arg-reductions,
        ///     <c>np.unique</c>, the set routines and the equality tests over empty inputs, lines of one element, or a pair the
        ///     comparison operators cannot compare — answered now (phase 1); the same calls over text NumPy compares stay
        ///     deferred.
        /// </summary>
        [TestMethod] public void ShapeDecidedAnswers() => Run("strshape");

        /// <summary>
        ///     The checks NumPy makes before it converts or compares text, in its order: <c>np.interp</c>, <c>np.digitize</c>,
        ///     <c>np.repeat</c> with array repeats, conversions of nothing (astype / copyto / setitem / where / concatenate /
        ///     choose), place / putmask with ndarray values, insert / delete with an array obj, searchsorted's arguments.
        /// </summary>
        [TestMethod] public void TextValidationOrder() => Run("strvalid");

        /// <summary>
        ///     The rest of NumPy's surface over text operands: <c>np.linalg</c>, the products and sliding windows, random
        ///     seeding (<c>SeedSequence</c>, <c>default_rng</c>, Philox counters) and integer bounds, the histogram family,
        ///     quantile <c>q</c>, reductions and arg-reductions with <c>out=</c>, polynomials, <c>np.emath</c>, index
        ///     conversions, dtype predicates, <c>np.ediff1d</c>, text <c>where=</c> masks, 0-d conversions to C# scalars and
        ///     <c>fill</c> with a sequence — NumPy's refusals in its order, its answers where dtypes and shapes decide them.
        /// </summary>
        [TestMethod] public void TextSurface() => Run("strsurface");

        /// <summary>
        ///     <c>numpy.ma</c> over text: every module function over masked text arrays (1-D, 2-D, empty, 0-d, bytes, all
        ///     masked, a fill value), the binary functions over text / number / bytes pairs, and the constructors' text masks
        ///     and fill values — the fill machinery (<c>'N/A'</c> / <c>b'N/A'</c>, conversion to the data's width), the
        ///     masks' forms, NumPy's refusals in its order, and its answers where dtypes and shapes decide them.
        /// </summary>
        [TestMethod] public void TextMaskedArrays() => Run("strma");

        /// <summary>
        ///     Section striter: <c>np.nditer</c> and <c>np.nested_iters</c> over fixed-width strings, as traces — the value
        ///     stream of every layout under each order and flag set, buffered width / class conversions under every casting
        ///     rule (NumPy's Unicode errors included), the copy routes and their write-backs, writes through the views,
        ///     allocated outputs, mixed pairs, NumPy's refusals and ranges; text-number conversions are the computational
        ///     phase (deferred).
        /// </summary>
        [TestMethod] public void TextIteration() => Run("striter");

        /// <summary>
        ///     Section strcomp (strings phase 2, P2.1): the six comparison ufuncs over two strings of one class across every
        ///     layout kind, cross-layout broadcasts and width pairs (past one vector and past the kernels' unrolled chunk plan),
        ///     special code units (bytes on both sides of 0x80, U units above U+10FFFF), the C# operators with a string operand,
        ///     <c>out=</c> / <c>where=</c> / <c>dtype=</c>, and the truth readers (the bool cast and the nonzero route, along
        ///     axes too). Every case is strict: NumSharp computes all of them.
        /// </summary>
        [TestMethod] public void TextComparisonsAndTruth() => Run("strcomp");

        /// <summary>
        ///     Section strarith (strings phase 2, P2.3): <c>np.add</c> over two strings of one class across every layout kind,
        ///     cross-layout broadcasts and width pairs, special code units, <c>out=</c> (exact, narrower, wider, S into U,
        ///     refused) / <c>where=</c> / <c>dtype=</c>, the byte-width limit; the string repeat (<c>np.multiply</c> into
        ///     <c>out=</c>: every count dtype, both operand orders, cut and padded outputs, overflow, the refusals); the C#
        ///     <c>+</c> / <c>*</c> operators with a string operand; <c>np.strings.add</c> and NumPy's two-pass
        ///     <c>np.strings.multiply</c>. Every case is strict: NumSharp computes all of them.
        /// </summary>
        [TestMethod] public void TextConcatenationAndRepetition() => Run("strarith");

        /// <summary>
        ///     Section strconv (strings phase 1, P1.1): conversion between text and numbers through every route that
        ///     converts — <c>astype</c> both ways (every numeric dtype and value class, every target width, layout,
        ///     casting rule and order; every text grammar NumPy's <c>int()</c> / <c>float()</c> / <c>complex()</c> accept or
        ///     refuse, Unicode digits and spaces, the 4300-digit limit, the first failing element in NumPy's order), element
        ///     stores (setitem, fill, full, ones, a text array into a numeric one), <c>np.copyto</c> with every casting rule
        ///     and a mask, <c>np.array</c> over mixed values, numeric ufuncs into a text <c>out=</c> (with <c>where=</c>
        ///     reading the out's text first), the string-request <c>np.add</c> over numbers, numeric-dtype reductions and
        ///     <c>np.cov</c> over text, <c>np.concatenate</c> / <c>np.where</c> / <c>np.pad</c> /
        ///     <c>np.linalg.matrix_power(a, 0)</c> mixing text and numbers. Every case is strict.
        /// </summary>
        [TestMethod] public void TextNumberConversions() => Run("strconv");

        /// <summary>
        ///     Section tdtype (strings phase 1, P1.2): the NEP 55 <c>StringDType</c> descriptor without arrays — every
        ///     attribute for every kind of <c>na_object</c> (None, NaN, text, numbers, bytes, <c>np.float16</c>, complex, big
        ///     ints, tuples, lists; a lone surrogate is NumPy's encode error) and <c>coerce</c>, <c>np.dtype</c>'s spellings,
        ///     equality (and against a dtype string), promotion with every class, common instances and sequences, the
        ///     casting-rule table both ways under every rule, the scalar hierarchy, <c>np.isdtype</c>'s refusals,
        ///     <c>newbyteorder</c> and the ufunc <c>dtype=</c> refusal. Every case is strict.
        /// </summary>
        [TestMethod] public void StringDTypeDescriptors() => Run("tdtype");

        /// <summary>
        ///     The most cases each section may defer to the computational phase (see the class remarks): the deferrals
        ///     left are sorting, searching, set routines, histograms and quantiles over text. A section not listed may
        ///     defer nothing.
        /// </summary>
        private static readonly Dictionary<string, int> DeferralCeilings = new()
        {
            ["reductions"] = 32, ["straxis"] = 130, ["strshape"] = 190, ["strvalid"] = 5, ["strsurface"] = 67,
            ["strma"] = 113,
        };

        /// <summary>The corpus is complete: every section has cases and every case belongs to a section above.</summary>
        [TestMethod]
        public void Corpus_EverySectionIsReplayed()
        {
            var sections = new HashSet<string>
            {
                "dtype", "creation", "astype", "views", "movement", "mutation", "access", "printing", "ufuncs", "reductions",
                "strdtype", "strprec", "straxis", "strcov", "strshape", "strvalid", "strsurface", "strma", "striter",
                "strcomp", "strarith", "strconv", "tdtype",
            };
            var bySection = Cases.GroupBy(c => c.Id.Split('/')[0]).ToDictionary(g => g.Key, g => g.Count());
            foreach (var s in sections)
                Assert.IsTrue(bySection.TryGetValue(s, out var n) && n > 0, $"section '{s}' has no cases");
            foreach (var s in bySection.Keys)
                Assert.IsTrue(sections.Contains(s), $"section '{s}' has no test method");
            Assert.IsTrue(Cases.Count >= 6000, $"the corpus has {Cases.Count} cases (truncated regeneration?)");
        }

        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The environment variable that lists every deferred case of a section on the test's standard output, with
        ///     NumSharp's refusal text (any non-empty value but <c>0</c> turns it on). A development aid: a phase of the
        ///     computational work reads it to see which NumPy answers are still refused, and that none of the calls it just
        ///     implemented still is. Off by default, because a section can defer hundreds of cases.
        /// </summary>
        internal const string ListDeferredVariable = "NUMSHARP_STRING_ORACLE_DEFERRED";

        /// <summary>Replays every case of one section and fails with the list of divergences.</summary>
        /// <param name="section">The section prefix of the case ids.</param>
        private static void Run(string section)
        {
            var failures = new List<string>();
            var deferredCases = new List<string>();
            int deferred = 0, total = 0;
            foreach (var c in Cases)
            {
                if (!c.Id.StartsWith(section + "/", StringComparison.Ordinal))
                    continue;
                total++;
                var outcome = ReplayOne(c, out var message);
                if (outcome == Outcome.Deferred)
                {
                    deferred++;
                    deferredCases.Add($"{c.Id} [{c.Layout}]: {message}");
                }
                else if (outcome == Outcome.Failed)
                    failures.Add($"{c.Id} [{c.Layout}]: {message}");
            }
            int excused = Cases.Count(c => c.Id.StartsWith(section + "/", StringComparison.Ordinal) && ExcusedLayouts.ContainsKey(c.Id));
            Console.WriteLine($"[strings/{section}] {total} cases, {deferred} deferred to the computational phase, " +
                              $"{excused} with an excused layout, {failures.Count} failed");
            var listDeferred = Environment.GetEnvironmentVariable(ListDeferredVariable);
            if (!string.IsNullOrEmpty(listDeferred) && listDeferred != "0")
                foreach (var d in deferredCases)
                    Console.WriteLine("  deferred " + d);
            // A computation answered today must not slide back into a deferral (the class remarks): more deferrals than
            // the section's ceiling fail the section.
            int ceiling = DeferralCeilings.TryGetValue(section, out var cap) ? cap : 0;
            if (deferred > ceiling)
                failures.Add($"{deferred} cases deferred to the computational phase, more than the section's ceiling of " +
                             $"{ceiling} (set {ListDeferredVariable}=1 to list them)");
            // Every divergence goes to the test's standard output (the assertion message keeps the first 80).
            foreach (var f in failures)
                Console.WriteLine("  " + f);
            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{total} cases diverged from NumPy:\n  " + string.Join("\n  ", failures.Take(80)));
        }

        /// <summary>The verdict for one case.</summary>
        private enum Outcome
        {
            /// <summary>NumSharp matched NumPy.</summary>
            Passed,

            /// <summary>A phase-2 case NumSharp refused with <see cref="StringComputationNotSupportedException"/>.</summary>
            Deferred,

            /// <summary>A divergence.</summary>
            Failed,
        }

        /// <summary>Replays one case and compares it.</summary>
        /// <param name="c">The case.</param>
        /// <param name="message">The divergence, when the outcome is <see cref="Outcome.Failed"/>; NumSharp's refusal text,
        ///     when it is <see cref="Outcome.Deferred"/>.</param>
        /// <returns>The verdict.</returns>
        private static Outcome ReplayOne(StringOracleCorpus.Case c, out string message)
        {
            message = null;
            NDArray[] ops;
            try
            {
                ops = c.Operands.Select(StringOracleCorpus.Reconstruct).ToArray();
            }
            catch (Exception e)
            {
                message = $"operand reconstruction threw {e.GetType().Name}: {e.Message}";
                return Outcome.Failed;
            }

            object result;
            var layoutExcuse = LayoutExcuse(c);
            if (layoutExcuse != null)
                ExcusedLayouts.TryAdd(c.Id, layoutExcuse);
            try
            {
                result = StringOracleOps.Invoke(c, ops);
            }
            catch (Exception e)
            {
                if (c.Phase == 2 && e is StringComputationNotSupportedException)
                {
                    message = Shorten(e.Message);
                    return Outcome.Deferred;
                }
                if (c.Error is null)
                {
                    message = $"NumPy returned a result, NumSharp threw {e.GetType().Name}: {Shorten(e.Message)}";
                    return Outcome.Failed;
                }
                return CompareError(c.Error, e, out message) ? Outcome.Passed : Outcome.Failed;
            }

            if (c.Error is not null)
            {
                message = $"NumPy raises {c.Error.Type}: {Shorten(c.Error.Text)} — NumSharp returned {Describe(result)}";
                return Outcome.Failed;
            }
            try
            {
                var diffs = new List<string>();
                Compare(c.Expected, result, layoutExcuse is null ? ops : null, diffs, compareLayout: layoutExcuse is null);
                if (diffs.Count == 0)
                    return Outcome.Passed;
                message = string.Join("; ", diffs);
                return Outcome.Failed;
            }
            catch (Exception e)
            {
                message = $"comparison threw {e.GetType().Name}: {e.Message}";
                return Outcome.Failed;
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // Errors

        /// <summary>The .NET exception classes that stand for each NumPy exception class (besides a same-named one).</summary>
        private static readonly Dictionary<string, string[]> ErrorTypes = new()
        {
            // The house equivalents the general oracle accepts too (FuzzCorpusTests.ErrorTypeMap): NumSharp keeps its
            // long-standing .NET types for these NumPy classes and matches the MESSAGE verbatim.
            ["TypeError"] = new[]
            {
                "TypeError", "IncorrectTypeException", "InvalidCastException", "UFuncTypeException", "ArgumentException",
                "NotSupportedException",
            },
            ["ValueError"] = new[] { "ValueError", "ArgumentException", "IncorrectShapeException", "IncorrectSizeException" },
            ["IndexError"] = new[]
            {
                "IndexError", "IndexOutOfRangeException", "ArgumentOutOfRangeException", "ArgumentException",
            },
            ["AxisError"] = new[] { "AxisError", "AxisOutOfRangeException" },
            ["OverflowError"] = new[] { "OverflowException" },
            ["DTypePromotionError"] = new[] { "DTypePromotionError" },
            ["UnicodeEncodeError"] = new[] { "UnicodeEncodeError" },
            ["UnicodeDecodeError"] = new[] { "UnicodeDecodeError" },
            ["SystemError"] = new[] { "SystemError" },
            // numpy.ma's rank checks (compress_rowcols / mask_rowcols on a non-2-D array) raise NotImplementedError; the
            // .NET class with that meaning is NotImplementedException (the message is matched verbatim).
            ["NotImplementedError"] = new[] { "NotImplementedException" },
        };

        /// <summary>Compares NumSharp's exception with NumPy's: a corresponding class and the identical message.</summary>
        /// <param name="want">NumPy's exception.</param>
        /// <param name="got">NumSharp's exception.</param>
        /// <param name="message">The divergence, when false.</param>
        /// <returns>True when they correspond.</returns>
        private static bool CompareError(StringOracleCorpus.ErrorInfo want, Exception got, out string message)
        {
            message = null;
            string gotType = got.GetType().Name;
            bool typeOk = gotType == want.Type || (ErrorTypes.TryGetValue(want.Type, out var allowed) && allowed.Contains(gotType));
            string gotText = Normalize(got.Message);
            bool textOk = gotText == want.Text.Trim();
            if (typeOk && textOk)
                return true;
            message = $"error mismatch\n      numpy:    {want.Type}: {Shorten(want.Text)}\n      numsharp: {gotType}: {Shorten(gotText)}";
            return false;
        }

        /// <summary>Strips .NET's " (Parameter 'x')" suffix so only the message itself is compared.</summary>
        private static string Normalize(string m)
        {
            if (string.IsNullOrEmpty(m))
                return "";
            int p = m.IndexOf(" (Parameter '", StringComparison.Ordinal);
            if (p >= 0)
                m = m.Substring(0, p);
            return m.Trim();
        }

        // ------------------------------------------------------------------------------------------------------------
        // Results

        /// <summary>
        ///     The cases whose LAYOUT (strides, contiguity, ownership, memory sharing) is a known NumSharp divergence that
        ///     is not about strings, keyed by case id with the reason; their values are still compared byte for byte.
        /// </summary>
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string> ExcusedLayouts = new();

        /// <summary>
        ///     The known layout divergences, all numeric-path behavior every dtype shares:
        ///     <list type="bullet">
        ///       <item><description>an integer-array or boolean index BEHIND a slice / newaxis / ellipsis
        ///       (<c>a[:, [3, 1]]</c>) — NumPy returns its MapIter buffer as a transposed VIEW, NumSharp a fresh C-ordered
        ///       copy (pinned in <c>OpenBugs.FancyIndex.cs</c>; plan <c>docs/plans/view-shape-open-bugs.md</c>);</description></item>
        ///       <item><description><c>np.ma.compress_nd</c> / <c>compress_rowcols</c> / <c>compress_cols</c>, whose column
        ///       selection is such an index.</description></item>
        ///     </list>
        ///     <c>np.argwhere</c> / <c>np.flatnonzero</c> (reached by an empty string array, whose index searches run the
        ///     numeric path on a bool stand-in) and <c>np.unravel_index</c> (section M's bool index array) were excused here
        ///     until the open-bugs rundown's bug 8 made them return NumPy's views; their layout is compared now.
        /// </summary>
        /// <param name="c">The case.</param>
        /// <returns>The reason, or null when the layout must match.</returns>
        private static string LayoutExcuse(StringOracleCorpus.Case c)
        {
            // (argwhere / flatnonzero / unravel_index were excused here until the open-bugs rundown's bug 8: they now
            // return NumPy's views of nonzero's / the interleaved index buffer, and their layout is compared.)
            // numpy.ma.compress_nd drops the masked columns of a 2-D array with `data[:, keep]`: a boolean index behind a
            // slice, which NumPy returns as its transposed MapIter VIEW and NumSharp as a fresh C-ordered copy.
            if (c.Op == "ma1" && c.Layout == "MU2" && c.Params is not null && c.Params.TryGetValue("name", out var maName)
                && maName.GetString() is "compress_nd" or "compress_rowcols" or "compress_cols")
                return "compress_nd's column selection is a fancy index behind a slice: NumPy's transposed MapIter view (OpenBugs.FancyIndex)";
            if (c.Op != "getitem" || c.Params is null || !c.Params.TryGetValue("index", out var index))
                return null;
            bool basicSeen = false;
            foreach (var e in index.EnumerateArray())
            {
                if (e.TryGetProperty("slice", out _) || e.TryGetProperty("none", out _) || e.TryGetProperty("ellipsis", out _))
                    basicSeen = true;
                else if (basicSeen && (e.TryGetProperty("arr", out _) || e.TryGetProperty("mask", out _) || e.TryGetProperty("mask2", out _)))
                    return "fancy index behind a slice: NumPy's transposed MapIter view (OpenBugs.FancyIndex)";
            }
            return null;
        }

        /// <summary>Dispatches on the expectation kind.</summary>
        /// <param name="exp">The expectation.</param>
        /// <param name="result">NumSharp's result.</param>
        /// <param name="ops">The operands (for memory sharing and mutation checks), or null to skip memory sharing.</param>
        /// <param name="diffs">Receives the divergences.</param>
        /// <param name="compareLayout">False to compare values only (a <see cref="LayoutExcuse"/>d case).</param>
        private static void Compare(JsonElement exp, object result, NDArray[] ops, List<string> diffs, bool compareLayout = true)
        {
            string kind = exp.GetProperty("kind").GetString();
            switch (kind)
            {
                case "array":
                    CompareArray(exp, AsArray(result), ops, diffs, "", compareLayout);
                    break;
                case "tuple":
                {
                    var items = AsArrays(result);
                    var want = exp.GetProperty("items").EnumerateArray().ToArray();
                    if (items.Length != want.Length)
                    {
                        diffs.Add($"tuple of {items.Length}, NumPy {want.Length}");
                        return;
                    }
                    // A layout excuse covers every item of a tuple result (np.unravel_index's coordinates).
                    for (int i = 0; i < items.Length; i++)
                        CompareArray(want[i], items[i], ops, diffs, $"[{i}] ", compareLayout);
                    break;
                }
                case "mutated":
                {
                    int k = exp.GetProperty("operand").GetInt32();
                    CompareArray(exp.GetProperty("view"), ops[k], null, diffs, "view ");
                    var baseHex = Convert.ToHexString(StringOracleCorpus.BaseBytes(ops[k])).ToLowerInvariant();
                    if (baseHex != exp.GetProperty("base").GetString())
                        diffs.Add($"base buffer {Shorten(baseHex)} != NumPy {Shorten(exp.GetProperty("base").GetString())}");
                    break;
                }
                case "value":
                {
                    var want = StringOracleCorpus.Decode(exp.GetProperty("value"));
                    var got = AsValue(result);
                    if (!ValueEquals(want, got))
                        diffs.Add($"value {Show(got)} != NumPy {Show(want)}");
                    break;
                }
                case "text":
                {
                    // The text travels as UTF-32 hex (a printed lone surrogate is not valid JSON text).
                    var want = StringOracleCorpus.Utf32HexToString(exp.GetProperty("u").GetString());
                    if (result is not string got || got != want)
                        diffs.Add($"text {Show(result)} != NumPy {Show(want)}");
                    break;
                }
                case "bool":
                {
                    bool want = exp.GetProperty("value").GetBoolean();
                    // A 0-d result of a masked reduction is an unmasked 0-d masked array (AsArray unwraps it to its data).
                    bool? got = result switch
                    {
                        bool bo => bo,
                        _ => AsArray(result) is NDArray nd && nd.ndim == 0 ? (bool)nd : null,
                    };
                    if (got != want)
                        diffs.Add($"bool {Show(result)} != NumPy {want}");
                    break;
                }
                case "dtype":
                {
                    if (result is not DType d)
                    {
                        diffs.Add($"expected a dtype, got {Describe(result)}");
                        return;
                    }
                    if (d.str != exp.GetProperty("str").GetString())
                        diffs.Add($"dtype {d.str} != NumPy {exp.GetProperty("str").GetString()}");
                    if (d.ToString(true) != exp.GetProperty("repr").GetString())
                        diffs.Add($"repr {d.ToString(true)} != NumPy {exp.GetProperty("repr").GetString()}");
                    break;
                }
                case "props":
                {
                    if (result is not DType d)
                    {
                        diffs.Add($"expected a dtype, got {Describe(result)}");
                        return;
                    }
                    void Check(string name, object got, object want)
                    {
                        if (!Equals(got, want))
                            diffs.Add($"{name} {got} != NumPy {want}");
                    }
                    Check("str", d.str, exp.GetProperty("str").GetString());
                    Check("name", d.name, exp.GetProperty("name").GetString());
                    Check("char", d.@char.ToString(), exp.GetProperty("char").GetString());
                    Check("kind", d.kind.ToString(), exp.GetProperty("dkind").GetString());
                    Check("num", d.num, exp.GetProperty("num").GetInt32());
                    Check("itemsize", d.itemsize, exp.GetProperty("itemsize").GetInt32());
                    Check("byteorder", d.byteorder.ToString(), exp.GetProperty("byteorder").GetString());
                    Check("isbuiltin", d.isbuiltin, exp.GetProperty("isbuiltin").GetInt32());
                    Check("isnative", d.isnative, exp.GetProperty("isnative").GetBoolean());
                    Check("alignment", d.alignment, exp.GetProperty("alignment").GetInt32());
                    Check("repr", d.ToString(true), exp.GetProperty("repr").GetString());
                    Check("str()", d.ToString(), exp.GetProperty("strfmt").GetString());
                    // Section tdtype records what a StringDType adds (and hasobject / flags for every descriptor).
                    if (exp.TryGetProperty("hasobject", out var hasobject))
                        Check("hasobject", d.hasobject, hasobject.GetBoolean());
                    if (exp.TryGetProperty("flags", out var flags))
                        Check("flags", d.flags, flags.GetInt32());
                    if (exp.TryGetProperty("coerce", out var coerce))
                        Check("coerce", d.coerce, coerce.GetBoolean());
                    if (exp.TryGetProperty("has_na", out var hasNA))
                        Check("has na_object", d.HasNAObject, hasNA.GetBoolean());
                    if (exp.TryGetProperty("na_repr", out var naRepr))
                        Check("repr(na_object)", NumSharp.Backends.Strings.NDPyObject.Repr(d.na_object), naRepr.GetString());
                    break;
                }
                case "masked":
                    CompareMasked(exp, result, diffs);
                    break;
                case "masked_constant":
                    if (!ReferenceEquals(result, np.ma.masked))
                        diffs.Add($"expected numpy.ma's masked constant, got {Describe(result)}");
                    break;
                case "none":
                    if (result is not null)
                        diffs.Add($"expected None, got {Describe(result)}");
                    break;
                case "seq":
                {
                    // A tuple holding masked arrays (np.ma.average(..., returned=True)): each item by its own kind.
                    var want = exp.GetProperty("items").EnumerateArray().ToArray();
                    var items = result is ITuple t
                        ? Enumerable.Range(0, t.Length).Select(i => t[i]).ToArray()
                        : result as object[] ?? Array.Empty<object>();
                    if (items.Length != want.Length)
                    {
                        diffs.Add($"sequence of {items.Length}, NumPy {want.Length}");
                        return;
                    }
                    for (int i = 0; i < items.Length; i++)
                        Compare(want[i], items[i], null, diffs, compareLayout: false);
                    break;
                }
                default:
                    diffs.Add($"unknown expectation kind '{kind}'");
                    break;
            }
        }

        /// <summary>
        ///     Compares a masked result with NumPy's: the data's descriptor, shape and C-order bytes (masked slots included),
        ///     the mask's FORM (<c>nomask</c> or an array) and bytes, the fill value its getter returns and the hard-mask flag.
        /// </summary>
        /// <param name="exp">The masked expectation.</param>
        /// <param name="result">NumSharp's result.</param>
        /// <param name="diffs">Receives the divergences.</param>
        private static void CompareMasked(JsonElement exp, object result, List<string> diffs)
        {
            if (result is not NDMaskedArray m || ReferenceEquals(result, np.ma.masked))
            {
                diffs.Add($"expected a masked array, got {Describe(result)}");
                return;
            }
            string wantDtype = exp.GetProperty("dtype").GetString();
            if (m.data.dtype.str != wantDtype)
            {
                diffs.Add($"data dtype {m.data.dtype.str} != NumPy {wantDtype}");
                return;
            }
            var wantShape = exp.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            if (!m.data.Shape.dimensions.SequenceEqual(wantShape))
            {
                diffs.Add($"shape ({string.Join(",", m.data.Shape.dimensions)}) != NumPy ({string.Join(",", wantShape)})");
                return;
            }
            // A null buffer is data NumPy left uninitialized (an S np.empty_like / masked_all): only its descriptor and
            // shape are NumPy's.
            var wantBuffer = exp.GetProperty("buffer");
            var gotHex = Convert.ToHexString(StringOracleCorpus.CBytes(m.data)).ToLowerInvariant();
            if (wantBuffer.ValueKind != JsonValueKind.Null && gotHex != wantBuffer.GetString())
                diffs.Add($"data bytes {Shorten(gotHex)} != NumPy {Shorten(wantBuffer.GetString())} (as text: {Show(Text(m.data))})");
            var wantMask = exp.GetProperty("mask");
            bool gotNomask = ReferenceEquals(m.mask, np.ma.nomask);
            if (wantMask.ValueKind == JsonValueKind.Null)
            {
                if (!gotNomask)
                    diffs.Add($"mask is an array ({Show(m.mask.tolist())}), NumPy nomask");
            }
            else if (gotNomask)
                diffs.Add("mask is nomask, NumPy an array");
            else
            {
                var gotMask = Convert.ToHexString(StringOracleCorpus.CBytes(m.mask)).ToLowerInvariant();
                if (gotMask != wantMask.GetString())
                    diffs.Add($"mask bytes {Shorten(gotMask)} != NumPy {Shorten(wantMask.GetString())}");
            }
            CompareFill(exp.GetProperty("fill"), m.fill_value, diffs);
            if (m.hardmask != exp.GetProperty("hard").GetBoolean())
                diffs.Add($"hardmask {m.hardmask} != NumPy {exp.GetProperty("hard").GetBoolean()}");
        }

        /// <summary>
        ///     Compares a fill value with NumPy's getter result: text ordinally, bytes by content, numbers by value (a
        ///     Python int against any C# integer, a float against any C# float), a complex both parts, an array fill as an
        ///     array.
        /// </summary>
        /// <param name="want">NumPy's encoded fill.</param>
        /// <param name="got">NumSharp's fill value.</param>
        /// <param name="diffs">Receives the divergences.</param>
        private static void CompareFill(JsonElement want, object got, List<string> diffs)
        {
            if (want.ValueKind == JsonValueKind.Object && want.TryGetProperty("arr", out var arr))
            {
                CompareArray(arr, got as NDArray, null, diffs, "fill ", compareLayout: false);
                return;
            }
            if (want.ValueKind == JsonValueKind.Object && want.TryGetProperty("complex", out var c))
            {
                var wc = new System.Numerics.Complex(c[0].GetDouble(), c[1].GetDouble());
                if (got is not System.Numerics.Complex gc || !gc.Equals(wc))
                    diffs.Add($"fill {Show(got)} != NumPy {wc}");
                return;
            }
            var w = StringOracleCorpus.Decode(want);
            bool same = (w, got) switch
            {
                (string ws, string gs) => string.Equals(ws, gs, StringComparison.Ordinal),
                (byte[] wb, byte[] gb) => wb.AsSpan().SequenceEqual(gb),
                (bool wb, bool gb) => wb == gb,
                (long wl, char ch) => ch == wl,
                (long wl, long or int or short or sbyte or byte or ushort or uint or ulong) => Convert.ToDecimal(got) == wl,
                (double wd, Half h) => (double)h == wd || double.IsNaN((double)h) && double.IsNaN(wd),
                (double wd, double or float) => Convert.ToDouble(got) == wd || double.IsNaN(Convert.ToDouble(got)) && double.IsNaN(wd),
                _ => false,
            };
            if (!same)
                diffs.Add($"fill {Show(got)} ({got?.GetType().Name}) != NumPy {Show(w)}");
        }

        /// <summary>Compares an array result with an array expectation.</summary>
        /// <param name="exp">The array expectation.</param>
        /// <param name="r">NumSharp's array.</param>
        /// <param name="ops">The operands for the memory-sharing check (null to skip it).</param>
        /// <param name="diffs">Receives the divergences.</param>
        /// <param name="prefix">A label for tuple items and mutation views.</param>
        /// <param name="compareLayout">False to compare the descriptor, shape and bytes only.</param>
        private static void CompareArray(JsonElement exp, NDArray r, NDArray[] ops, List<string> diffs, string prefix,
                                         bool compareLayout = true)
        {
            if (r is null)
            {
                diffs.Add(prefix + "expected an array, got null");
                return;
            }
            string wantDtype = exp.GetProperty("dtype").GetString();
            if (r.dtype.str != wantDtype)
            {
                diffs.Add($"{prefix}dtype {r.dtype.str} != NumPy {wantDtype}");
                return;
            }
            var wantShape = exp.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            var gotShape = r.Shape.dimensions;
            if (!gotShape.SequenceEqual(wantShape))
            {
                diffs.Add($"{prefix}shape ({string.Join(",", gotShape)}) != NumPy ({string.Join(",", wantShape)})");
                return;
            }
            var gotHex = Convert.ToHexString(StringOracleCorpus.CBytes(r)).ToLowerInvariant();
            var wantHex = exp.GetProperty("buffer").GetString();
            if (gotHex != wantHex)
                diffs.Add($"{prefix}bytes {Shorten(gotHex)} != NumPy {Shorten(wantHex)} (as text: {Show(Text(r))})");
            if (!compareLayout)
                return;

            // Layout: strides on the axes that have more than one element of a non-empty result, then the flags.
            if (r.size > 0 && exp.TryGetProperty("strides", out var ws))
            {
                var wantStrides = ws.EnumerateArray().Select(e => e.GetInt64()).ToArray();
                var gotStrides = r.Shape.strides;
                for (int i = 0; i < gotShape.Length; i++)
                    if (gotShape[i] > 1 && gotStrides[i] != wantStrides[i])
                    {
                        diffs.Add($"{prefix}strides ({string.Join(",", gotStrides)}) != NumPy ({string.Join(",", wantStrides)})");
                        break;
                    }
            }
            if (exp.TryGetProperty("c", out var wc) && r.flags.c_contiguous != wc.GetBoolean())
                diffs.Add($"{prefix}C_CONTIGUOUS {r.flags.c_contiguous} != NumPy {wc.GetBoolean()}");
            if (exp.TryGetProperty("f", out var wf) && r.flags.f_contiguous != wf.GetBoolean())
                diffs.Add($"{prefix}F_CONTIGUOUS {r.flags.f_contiguous} != NumPy {wf.GetBoolean()}");
            if (exp.TryGetProperty("writeable", out var ww) && r.flags.writeable != ww.GetBoolean())
                diffs.Add($"{prefix}WRITEABLE {r.flags.writeable} != NumPy {ww.GetBoolean()}");
            if (exp.TryGetProperty("owndata", out var wo) && r.flags.owndata != wo.GetBoolean())
                diffs.Add($"{prefix}OWNDATA {r.flags.owndata} != NumPy {wo.GetBoolean()}");
            if (ops is not null && exp.TryGetProperty("shares", out var sh))
            {
                var want = sh.EnumerateArray().Select(e => e.GetBoolean()).ToArray();
                for (int i = 0; i < want.Length && i < ops.Length; i++)
                {
                    bool got = np.shares_memory(r, ops[i]);
                    if (got != want[i])
                        diffs.Add($"{prefix}shares_memory(result, operand {i}) {got} != NumPy {want[i]}");
                }
            }
        }

        /// <summary>A result as one array: an NDArray, or a C# scalar NumSharp returns where NumPy returns a 0-d value.</summary>
        private static NDArray AsArray(object r) => r switch
        {
            NDArray nd => nd,
            bool b => NDArray.Scalar(b),
            long l => NDArray.Scalar(l),
            int i => NDArray.Scalar((long)i),
            double d => NDArray.Scalar(d),
            np.UniqueResult u => u,
            // numpy.ma returns a NumPy scalar for an unmasked 0-d result; NumSharp keeps a 0-d masked array (the module's
            // type-consistency choice), whose data is that scalar.
            NDMaskedArray m when !ReferenceEquals(m, np.ma.masked) && m.data.ndim == 0 && ReferenceEquals(m.mask, np.ma.nomask) => m.data,
            _ => null,
        };

        /// <summary>A tuple result as arrays: an <c>NDArray[]</c>, a C# value tuple of arrays, or a result struct.</summary>
        private static NDArray[] AsArrays(object r)
        {
            switch (r)
            {
                case NDArray[] arr: return arr;
                case ITuple t:
                {
                    var items = new NDArray[t.Length];
                    for (int i = 0; i < t.Length; i++)
                        items[i] = AsArray(t[i]);
                    return items;
                }
                case IEnumerable<NDArray> e: return e.ToArray();
            }
            return Array.Empty<NDArray>();
        }

        /// <summary>
        ///     A result as a Python value: a 0-d array is its element, a list stays a list (its 0-d arrays turned into
        ///     elements), a string / byte[] stays itself.
        /// </summary>
        private static object AsValue(object r)
        {
            switch (r)
            {
                case NDArray nd when nd.ndim == 0:
                    return nd.Storage.GetAtIndex(0);
                case NDArray nd:
                    return nd.tolist();
                case IList list when r is not byte[]:
                {
                    var items = new List<object>();
                    foreach (var x in list)
                        items.Add(AsValue(x));
                    return items;
                }
                default:
                    return r;
            }
        }

        /// <summary>Deep equality of decoded values: strings ordinally, bytes by content, lists element by element.</summary>
        private static bool ValueEquals(object want, object got)
        {
            switch (want)
            {
                case null: return got is null;
                case string ws: return got is string gs && string.Equals(ws, gs, StringComparison.Ordinal);
                case byte[] wb: return got is byte[] gb && wb.AsSpan().SequenceEqual(gb);
                case List<object> wl:
                {
                    if (got is not IList gl || got is byte[] || gl.Count != wl.Count)
                        return false;
                    for (int i = 0; i < wl.Count; i++)
                        if (!ValueEquals(wl[i], gl[i]))
                            return false;
                    return true;
                }
                default: return Equals(want, got);
            }
        }

        /// <summary>The decoded text of a string array (diagnostics), or its bytes otherwise.</summary>
        private static object Text(NDArray r)
        {
            try
            {
                return r.typecode == NPTypeCode.String ? r.tolist() : r.ToString();
            }
            catch (Exception e)
            {
                return $"<{e.GetType().Name}>";
            }
        }

        /// <summary>A short description of a result for a failure message.</summary>
        private static string Describe(object r) => r switch
        {
            null => "null",
            NDArray nd => $"an array {nd.dtype.str} ({string.Join(",", nd.Shape.dimensions)})",
            _ => $"{r.GetType().Name} {Show(r)}",
        };

        /// <summary>A value rendered for a failure message: escapes control characters, lists items, caps the length.</summary>
        private static string Show(object v)
        {
            var sb = new StringBuilder();
            void Render(object x)
            {
                switch (x)
                {
                    case null: sb.Append("None"); break;
                    case string s:
                        sb.Append('"');
                        foreach (var ch in s)
                            sb.Append(ch < 0x20 || ch > 0x7E ? $"\\u{(int)ch:x4}" : ch.ToString());
                        sb.Append('"');
                        break;
                    case byte[] b: sb.Append("b'").Append(Convert.ToHexString(b)).Append('\''); break;
                    case IList l:
                        sb.Append('[');
                        for (int i = 0; i < l.Count; i++)
                        {
                            if (i > 0) sb.Append(", ");
                            Render(l[i]);
                        }
                        sb.Append(']');
                        break;
                    default: sb.Append(x); break;
                }
            }
            Render(v);
            return Shorten(sb.ToString());
        }

        /// <summary>Caps a string for a failure line.</summary>
        private static string Shorten(string s) => s is null ? "null" : s.Length <= 300 ? s : s.Substring(0, 300) + "…";
    }
}
