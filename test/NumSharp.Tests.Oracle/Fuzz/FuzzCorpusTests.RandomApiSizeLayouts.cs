using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Tests.Fuzz.RandomApi;

namespace NumSharp.Tests.Fuzz
{
    public partial class FuzzCorpusTests
    {
        // ---- the random-API oracle under view-layout sizes ----------------------------------------------------------------
        //
        // NumPy's `size` is a tuple of dimensions; NumSharp's is a Shape, which can also carry a view's strides, offset and
        // buffer size. A sampler that allocated np.empty(size) from the Shape as given returned heap memory for a stepped
        // view (default_rng(42).integers(0, 100, arange(20)["::4"].Shape) gave 8589934592 and 140710211227288) and
        // transposed draws for an F-ordered one. The corpus encodes only dimensions, so the ordinary replays cannot reach
        // that path; these replays hand the SAME dimensions over as a view's Shape (ViewSizeLayouts) and require the answer
        // not to change.

        /// <summary>
        ///     Cases replayed per signature and tier by <see cref="RandomApiSizeLayouts"/>: each takes the next layout for its
        ///     size's rank in turn, so 14 cases give every rank-1 layout and the rank-0 one at least twice per overload while
        ///     the replay stays a fraction of the tier's own time.
        /// </summary>
        private const int SizeLayoutCasesPerSignature = 14;

        /// <summary>
        ///     Random-API oracle, VIEW-LAYOUT sizes against NumPy: for every overload taking a <see cref="Shape"/>, its corpus
        ///     cases are replayed with the size handed over as a view's Shape of the same dimensions (stepped, offset,
        ///     reversed, broadcast, contiguous slice, transposed, permuted, 0-d with an offset), and each must still give
        ///     NumPy's exact result, error and engine state. Then <see cref="ReplaySizeLayoutsMetamorphic"/> checks every such
        ///     overload at a 3-D+ size, where the dense F and permuted layouts exist.
        /// </summary>
        /// <remarks>
        ///     The portable tier always; the host-libm tier on Windows (its expected values are the win-amd64 CRT's), the LP64
        ///     tier on x64 Windows/Linux, as the tiers themselves are gated. The multivariate-normal tier needs the pinned
        ///     OpenBLAS (process-global), so it joins only the metamorphic part, which compares NumSharp with itself.
        /// </remarks>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void RandomApiSizeLayouts()
        {
            bool windows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows);
            bool linux = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux);
            bool x64 = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture == System.Runtime.InteropServices.Architecture.X64;

            var failures = new List<string>();
            var covered = new HashSet<string>(StringComparer.Ordinal);
            int replays = ReplaySizeLayoutsAgainstNumPy("random_api.jsonl", failures, covered);
            if (windows)
                replays += ReplaySizeLayoutsAgainstNumPy("random_api_host.jsonl", failures, covered);
            if (x64 && (windows || linux))
                replays += ReplaySizeLayoutsAgainstNumPy("random_api_lp64.jsonl", failures, covered);

            int metamorphic = ReplaySizeLayoutsMetamorphic(
                new[] { "random_api.jsonl", "random_api_host.jsonl", "random_api_lp64.jsonl", "random_api_mvn.jsonl" }, failures, out var gaps);

            Console.WriteLine($"[RandomApiSizeLayouts] {replays} view-layout replays against NumPy ({covered.Count} signature x layout pairs), " +
                              $"{metamorphic} metamorphic replays");
            Assert.AreEqual(0, gaps.Count, "overloads taking a Shape with no value case to replay at a view layout:\n  " + string.Join("\n  ", gaps));
            Assert.IsTrue(replays > 0, "no case carried a size to replay");
            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} view-layout size replays diverged:\n  " + string.Join("\n  ", failures.Take(60)));
        }

        /// <summary>
        ///     Replays a tier's cases that pass a size, each with its size given as a view layout (the next layout for the
        ///     size's rank, in turn per signature), through the same NumPy comparisons as the tier's own replay.
        /// </summary>
        /// <param name="file">The corpus tier.</param>
        /// <param name="failures">Receives one line per divergence.</param>
        /// <param name="covered">Receives each <c>signature | layout</c> pair replayed.</param>
        /// <returns>The number of replays.</returns>
        private static int ReplaySizeLayoutsAgainstNumPy(string file, List<string> failures, HashSet<string> covered)
        {
            using var cases = FuzzCorpus.Open(file);
            var perSignature = new Dictionary<string, int>(StringComparer.Ordinal);
            int replays = 0;
            foreach (var c in cases)
            {
                var dims = ViewSizeLayouts.SizeDims(c);
                if (dims == null)
                    continue;
                string sig = c.Params["sig"].GetString();
                int n = perSignature.TryGetValue(sig, out var seen) ? seen : 0;
                if (n >= SizeLayoutCasesPerSignature)
                    continue;
                perSignature[sig] = n + 1;

                var layouts = ViewSizeLayouts.Applicable(dims.Length);
                string layout = layouts[n % layouts.Count];
                covered.Add(sig + " | " + layout);
                var operands = new NDArray[c.Operands.Length];
                RandomApiHarness.SizeLayout = d => ViewSizeLayouts.Build(d, layout);
                try
                {
                    for (int i = 0; i < operands.Length; i++)
                        operands[i] = FuzzCorpus.Reconstruct(c.Operands[i]);
                    string problem = c.Expects_Throw ? CheckRandomApiError(c, operands) : CheckRandomApiValue(c, operands, out _);
                    if (problem != null)
                        failures.Add($"{c.Id} [size as {layout}]: {problem}");
                }
                catch (Exception e)
                {
                    failures.Add($"{c.Id} [size as {layout}]: HARNESS {e.GetType().Name}: {e.Message}");
                }
                finally
                {
                    RandomApiHarness.SizeLayout = null;
                    foreach (var o in operands)
                        o?.Dispose();
                }
                replays++;
            }
            return replays;
        }

        /// <summary>
        ///     Argument patterns replayed per signature by <see cref="ReplaySizeLayoutsMetamorphic"/>: distinct dtypes, methods,
        ///     omitted parameters and special values (a zero shape parameter, an empty size) reach different allocation sites
        ///     inside one overload — <c>random(size, float32)</c> fills through another helper than float64, <c>weibull(0)</c>
        ///     returns zeros without drawing, an empty size returns before the bounds are read.
        /// </summary>
        private const int SizeLayoutPatternsPerSignature = 6;

        /// <summary>
        ///     Every overload taking a <see cref="Shape"/>, at a size of rank 3 or more: up to
        ///     <see cref="SizeLayoutPatternsPerSignature"/> value cases with different argument patterns
        ///     (<see cref="ArgumentPattern"/>) are replayed with <c>(2, 3)</c> prepended to the size (a fresh C Shape), then once
        ///     per view layout of the same dimensions — including the transposed and permuted ones a 1-D corpus size cannot
        ///     have — and each must give the C-Shape replay's exact result, layout and engine state (or its exact error).
        /// </summary>
        /// <param name="files">The tiers whose signatures to cover (values are compared with NumSharp itself, so no tier
        ///     needs its host).</param>
        /// <param name="failures">Receives one line per divergence.</param>
        /// <param name="gaps">Receives the Shape-taking surface members that no tier gave a value case with a size.</param>
        /// <returns>The number of replays.</returns>
        /// <remarks>
        ///     Prepending keeps every call valid that was valid before (a broadcast against parameters still broadcasts); a call
        ///     the larger size makes invalid (a choice without replacement past its population, an <c>out=</c> of the old
        ///     shape) raises the same error under every layout, which is still a comparison.
        /// </remarks>
        private static int ReplaySizeLayoutsMetamorphic(string[] files, List<string> failures, out List<string> gaps)
        {
            var done = new HashSet<string>(StringComparer.Ordinal);
            var patterns = new HashSet<string>(StringComparer.Ordinal);
            var perSignature = new Dictionary<string, int>(StringComparer.Ordinal);
            int replays = 0;
            foreach (var file in files)
            {
                using var cases = FuzzCorpus.Open(file);
                foreach (var c in cases)
                {
                    if (c.Expects_Throw || ViewSizeLayouts.SizeDims(c) is not { } dims)
                        continue;
                    string sig = c.Params["sig"].GetString();
                    done.Add(sig);
                    int taken = perSignature.TryGetValue(sig, out var t) ? t : 0;
                    if (taken >= SizeLayoutPatternsPerSignature || !patterns.Add(sig + " " + ArgumentPattern(c)))
                        continue;
                    perSignature[sig] = taken + 1;

                    long[] Grown(long[] d) => new long[] { 2, 3 }.Concat(d).ToArray();
                    var reference = RunSizeLayout(c, d => new Shape(Grown(d)));
                    replays++;
                    foreach (var layout in ViewSizeLayouts.Applicable(dims.Length + 2))
                    {
                        var got = RunSizeLayout(c, d => ViewSizeLayouts.Build(Grown(d), layout));
                        replays++;
                        string diff = reference.Compare(got);
                        if (diff != null)
                            failures.Add($"{c.Id} [{sig}, size (2, 3) + {FormatDims(dims)} as {layout}]: {diff}");
                    }
                }
            }

            // Every Shape-taking member must have been reached: an overload whose cases never pass a size would otherwise go
            // unchecked here without anyone noticing.
            gaps = RandomApiSurface.Reflect()
                .Where(m => m.Parameters.Any(p => p.Type == "Shape" || p.Type == "Shape?"))
                .Select(m => m.Sig)
                .Where(sig => !done.Contains(sig))
                .OrderBy(s => s, StringComparer.Ordinal)
                .ToList();
            return replays;
        }

        /// <summary>One replay's observable outcome: the result, its layout and the receiver state, or the error.</summary>
        /// <param name="Result">The result observation (null when the call threw).</param>
        /// <param name="Layout">The result array's layout text (C/F flags, offset, strides, writeable, owndata), or null.</param>
        /// <param name="State">The receiver's state text after the call.</param>
        /// <param name="Error">The exception type and message, or null.</param>
        private sealed record SizeLayoutOutcome(string Result, string Layout, string State, string Error)
        {
            /// <summary>The first difference from <paramref name="other"/>, or null when every part matches.</summary>
            /// <param name="other">The replay at a view layout.</param>
            /// <returns>A description of the difference, or null.</returns>
            internal string Compare(SizeLayoutOutcome other)
            {
                if (Error != other.Error)
                    return $"outcome differs: C size {Error ?? "returned"}, view size {other.Error ?? "returned"}";
                if (Result != other.Result)
                    return $"result differs:\n      C size:    {Cap(Result)}\n      view size: {Cap(other.Result)}";
                if (Layout != other.Layout)
                    return $"result layout differs: C size {Layout}, view size {other.Layout}";
                if (State != other.State)
                    return $"engine state after the call differs:\n      C size:    {Cap(State)}\n      view size: {Cap(other.State)}";
                return null;
            }

            /// <summary>Shortens a long text for a failure line.</summary>
            /// <param name="s">The text.</param>
            /// <returns>At most 200 characters.</returns>
            private static string Cap(string s) => s == null || s.Length <= 200 ? s : s.Substring(0, 200) + "…";
        }

        /// <summary>Replays a case with its sizes built by <paramref name="layout"/> and captures the outcome.</summary>
        /// <param name="c">The case (a value case).</param>
        /// <param name="layout">Builds each size from its corpus dimensions.</param>
        /// <returns>The outcome.</returns>
        private static SizeLayoutOutcome RunSizeLayout(FuzzCorpus.Case c, Func<long[], Shape> layout)
        {
            var operands = new NDArray[c.Operands.Length];
            RandomApiHarness.SizeLayout = layout;
            RandomApiHarness.Outcome outcome = default;
            try
            {
                for (int i = 0; i < operands.Length; i++)
                    operands[i] = FuzzCorpus.Reconstruct(c.Operands[i]);
                try
                {
                    outcome = RandomApiHarness.Run(c.Params, operands);
                }
                catch (Exception e) when (e is not NotSupportedException || !e.Message.StartsWith("the random-API", StringComparison.Ordinal))
                {
                    return new SizeLayoutOutcome(null, null, null, e.GetType().Name + ": " + e.Message);
                }

                var mask = c.Params.TryGetValue("state_mask", out var m)
                    ? m.EnumerateArray().Select(x => x.GetString()).ToList()
                    : new List<string>();
                string observedType = outcome.Member.Kind == "ctor" ? outcome.Member.DeclaringType : outcome.Member.Returns;
                JsonNode obs = RandomApiObservation.MaskObs(RandomApiObservation.Observe(outcome.Result, observedType, c.Expected.Result), mask);
                string layoutText = outcome.Result is NDArray r
                    ? $"C={r.Shape.IsContiguous} F={r.Shape.IsFContiguous} offset={r.Shape.offset} strides=({string.Join(",", r.Shape.strides)}) " +
                      $"writeable={r.Shape.IsWriteable} owndata={r.Storage.OwnsData}"
                    : null;
                string state = RandomApiObservation.MaskText(RandomApiObservation.StateText(outcome.Receiver), mask);
                return new SizeLayoutOutcome(obs.ToJsonString(), layoutText, state, null);
            }
            finally
            {
                RandomApiHarness.SizeLayout = null;
                if (outcome.Result is NDArray r && !IsOperand(operands, r))
                    r.Dispose();
                foreach (var o in operands)
                    o?.Dispose();
            }
        }

        /// <summary>Formats dimensions as a Python tuple for messages.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The text.</returns>
        private static string FormatDims(long[] dims) => "(" + string.Join(", ", dims) + (dims.Length == 1 ? ",)" : ")");

        /// <summary>
        ///     A case's argument PATTERN: per argument, what decides which code path runs rather than its exact value — omitted,
        ///     null, an operand, an object, a dtype name, a string (a method name), a size's rank and emptiness, and a number's
        ///     class (zero, negative, NaN, infinite, other). Two cases with one pattern exercise the same allocation sites.
        /// </summary>
        /// <param name="c">The case.</param>
        /// <returns>The pattern text.</returns>
        private static string ArgumentPattern(FuzzCorpus.Case c)
        {
            var parts = new List<string>();
            foreach (var a in c.Params["args"].EnumerateArray())
            {
                string kind;
                if (a.TryGetProperty("omit", out _)) kind = "omit";
                else if (a.TryGetProperty("null", out _)) kind = "null";
                else if (a.TryGetProperty("none", out _)) kind = "none";
                else if (a.TryGetProperty("op", out _)) kind = "op";
                else if (a.TryGetProperty("obj", out _)) kind = "obj";
                else if (a.TryGetProperty("name", out var n)) kind = "dtype " + n.GetString();
                else if (a.TryGetProperty("dims", out var d))
                    kind = $"size rank {d.GetArrayLength()}{(d.EnumerateArray().Any(x => x.ToString() == "0") ? " empty" : "")}";
                else if (a.TryGetProperty("bits", out var b))
                {
                    double v = BitConverter.Int64BitsToDouble(unchecked((long)ulong.Parse(b.GetString(), System.Globalization.NumberStyles.HexNumber)));
                    kind = double.IsNaN(v) ? "nan" : double.IsInfinity(v) ? "inf" : v == 0 ? "zero" : v < 0 ? "negative" : "number";
                }
                else if (a.TryGetProperty("v", out var s))
                    kind = s.ValueKind is JsonValueKind.True or JsonValueKind.False ? s.ValueKind.ToString().ToLowerInvariant()
                        : s.ValueKind == JsonValueKind.String && !long.TryParse(s.GetString(), out _) ? "text " + s.GetString()
                        : s.ToString() == "0" ? "zero" : s.ToString().StartsWith("-", StringComparison.Ordinal) ? "negative" : "value";
                else kind = "other";
                parts.Add(a.GetProperty("n").GetString() + "=" + kind);
            }
            return string.Join("; ", parts);
        }
    }
}
