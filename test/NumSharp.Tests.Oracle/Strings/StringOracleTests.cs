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
    ///     (comparisons, concatenation, truth tests, sorting and searching, text ↔ number conversion): it passes as
    ///     "deferred" while NumSharp throws <see cref="StringComputationNotSupportedException"/>, and is held to NumPy's
    ///     answer the moment NumSharp returns a result or throws anything else — so the computational phase is gated by
    ///     this same corpus without editing it.
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

        /// <summary>The corpus is complete: every section has cases and every case belongs to a section above.</summary>
        [TestMethod]
        public void Corpus_EverySectionIsReplayed()
        {
            var sections = new HashSet<string>
            {
                "dtype", "creation", "astype", "views", "movement", "mutation", "access", "printing", "ufuncs", "reductions",
                "strdtype", "strprec", "straxis", "strcov",
            };
            var bySection = Cases.GroupBy(c => c.Id.Split('/')[0]).ToDictionary(g => g.Key, g => g.Count());
            foreach (var s in sections)
                Assert.IsTrue(bySection.TryGetValue(s, out var n) && n > 0, $"section '{s}' has no cases");
            foreach (var s in bySection.Keys)
                Assert.IsTrue(sections.Contains(s), $"section '{s}' has no test method");
            Assert.IsTrue(Cases.Count >= 6000, $"the corpus has {Cases.Count} cases (truncated regeneration?)");
        }

        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Replays every case of one section and fails with the list of divergences.</summary>
        /// <param name="section">The section prefix of the case ids.</param>
        private static void Run(string section)
        {
            var failures = new List<string>();
            int deferred = 0, total = 0;
            foreach (var c in Cases)
            {
                if (!c.Id.StartsWith(section + "/", StringComparison.Ordinal))
                    continue;
                total++;
                var outcome = ReplayOne(c, out var message);
                if (outcome == Outcome.Deferred)
                    deferred++;
                else if (outcome == Outcome.Failed)
                    failures.Add($"{c.Id} [{c.Layout}]: {message}");
            }
            int excused = Cases.Count(c => c.Id.StartsWith(section + "/", StringComparison.Ordinal) && ExcusedLayouts.ContainsKey(c.Id));
            Console.WriteLine($"[strings/{section}] {total} cases, {deferred} deferred to the computational phase, " +
                              $"{excused} with an excused layout, {failures.Count} failed");
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
        /// <param name="message">The divergence, when the outcome is <see cref="Outcome.Failed"/>.</param>
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
                    return Outcome.Deferred;
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
        ///     The known layout divergences. One today: an integer-array or boolean index BEHIND a slice / newaxis /
        ///     ellipsis (<c>a[:, [3, 1]]</c>) — NumPy returns its MapIter buffer as a transposed VIEW, NumSharp a fresh
        ///     C-ordered copy, for every dtype (pinned in <c>OpenBugs.FancyIndex.cs</c>; plan
        ///     <c>docs/plans/view-shape-open-bugs.md</c>).
        /// </summary>
        /// <param name="c">The case.</param>
        /// <returns>The reason, or null when the layout must match.</returns>
        private static string LayoutExcuse(StringOracleCorpus.Case c)
        {
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
                    for (int i = 0; i < items.Length; i++)
                        CompareArray(want[i], items[i], ops, diffs, $"[{i}] ");
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
                    bool? got = result switch { bool bo => bo, NDArray nd when nd.ndim == 0 => (bool)nd, _ => null };
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
                    break;
                }
                default:
                    diffs.Add($"unknown expectation kind '{kind}'");
                    break;
            }
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
