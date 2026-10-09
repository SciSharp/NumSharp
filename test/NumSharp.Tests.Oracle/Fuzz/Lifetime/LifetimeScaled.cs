using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using NumSharp.Backends;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     The SCALED lifetime tier: every op key of the ordinary and masked-array corpus replayed with its operands as large
    ///     temporaries under the lifetime stress thread, compared with the same call over held operands. It needs no NumPy
    ///     answer, so it reaches every op key those tiers reach (554 ordinary and 150 <c>ma.*</c> at 2026-10-09), where
    ///     the hand-written lifetime corpus reaches 229.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why scale.</b> A lost GC anchor shows only when a collection lands while the kernel still uses a freed
    ///     operand; a corpus operand of a few dozen elements gives the collection nanoseconds. Each selected case is
    ///     re-run with its operands enlarged until the call takes about <see cref="TargetCallMs"/> or the largest operand
    ///     reaches <see cref="TargetBytes"/>, in one of two ways (<see cref="Strategy"/>): a new leading axis of length K
    ///     (each slice keeps the corpus layout: strides, offset, broadcast axes) or, for ops that take one rank only, K
    ///     C-ordered copies along axis 0. 0-d and empty operands stay as they are (fresh, temporary). A strategy that makes
    ///     the op raise, or makes its result grow faster than linearly (an outer product, a full convolution's square),
    ///     is not used; a case no strategy scales is recorded as unscalable.
    ///     </para>
    ///     <para>
    ///     <b>The verdict.</b> The reference runs twice over held operands; two different answers mean the op is not
    ///     deterministic (an address, a random draw) and the case is skipped. Then <c>reps</c> stressed runs build every
    ///     operand inside the call expression (a temporary array of temporaries), exactly as the lifetime tier does: an
    ///     answer that differs from the reference, an exception, or a dead child is a lost anchor — never noise, since the
    ///     stress thread only poisons buffers it took from the pool.
    ///     </para>
    ///     <para>
    ///     <b>Where it runs.</b> In the lifetime child (<see cref="LifetimeChild"/>): optimized code, crash isolation. The
    ///     parent selects the cases (<see cref="Select"/>) and hands the child a selection file, so a child that restarts
    ///     after a crash does not parse the corpus again.
    ///     </para>
    ///     <para>
    ///     <b>Masked arrays.</b> A case of an <c>ma_*</c> tier builds each operand as an <see cref="NDMaskedArray"/> over the
    ///     scaled data and the scaled mask (K copies of the mask's C-order bytes, the shape the data takes), exactly as the
    ///     masked replay builds it (<c>np.ma.masked_array(data, mask)</c>), and runs through
    ///     <see cref="OpRegistry.ApplyMasked"/> / <see cref="OpRegistry.ApplyMaskedTuple"/>. The masked array is a temporary
    ///     too: its data, its mask and the arrays numpy.ma derives from them can all be lost mid-kernel.
    ///     </para>
    /// </remarks>
    internal static class LifetimeScaled
    {
        /// <summary>The prefix of the child's file argument that selects this mode: <c>scaled:&lt;selection file&gt;</c>.</summary>
        public const string ModePrefix = "scaled:";

        /// <summary>Cases selected per op key (distinct layouts first).</summary>
        public const int CasesPerOp = 2;

        /// <summary>The largest operand's target size: big enough that a call lasts long enough for a collection.</summary>
        public const long TargetBytes = 1L << 21;

        /// <summary>The call time the scaling stops at (a quadratic op reaches it long before its operands grow large).</summary>
        public const double TargetCallMs = 2.0;

        /// <summary>The largest scale factor tried.</summary>
        public const int MaxK = 1 << 16;

        /// <summary>
        ///     Ordinary-tier files the tier skips: their calls are not reproducible run to run (random streams), their
        ///     results are traces of objects (iterators) or text over huge arrays, or they only restate other tiers'
        ///     calls (relayout variants) or only raise.
        /// </summary>
        private static readonly string[] ExcludedFilePrefixes =
        {
            "random", "generator", "rnd_", "iter", "dtype_text", "relayout", "errors", "creation", "surface", "nditer",
        };

        /// <summary>How a case's operands are enlarged.</summary>
        public enum Strategy
        {
            /// <summary>A new leading axis of length K; each slice keeps the corpus operand's layout.</summary>
            LeadingAxis,

            /// <summary>K C-ordered copies along axis 0 (rank kept; for ops that take one rank only).</summary>
            Tile,
        }

        /// <summary>One selected case.</summary>
        /// <param name="Index">Its position in the selection (the child's resume key).</param>
        /// <param name="Id">The corpus case id.</param>
        /// <param name="Op">The op key.</param>
        /// <param name="File">The corpus file.</param>
        /// <param name="Case">The corpus case.</param>
        /// <param name="Gap">The known lifetime gap the case reaches (<c>LifetimeOracleTests.KnownScaledGaps</c>), or null for
        ///     a gate case.</param>
        internal sealed record ScaledCase(int Index, string Id, string Op, string File, FuzzCorpus.Case Case, string Gap);

        /// <summary>
        ///     Selects the cases: per op key of the ordinary and masked-array tiers (the index tiers use another case schema),
        ///     up to <see cref="CasesPerOp"/> success cases of distinct layouts with at least one operand that holds memory.
        /// </summary>
        /// <param name="gapOf">Names the known gap a case belongs to (null for a gate case). It is called while the corpus
        ///     stream still holds the case's line, so it may read the case's parameters and operands.</param>
        /// <returns>The selection, in a deterministic order (corpus files by name, then line order).</returns>
        /// <exception cref="IOException">The corpus directory cannot be read.</exception>
        /// <remarks>Parsing every ordinary tier takes seconds; the tests share one selection
        ///     (<c>LifetimeOracleTests.ScaledSelection</c>).</remarks>
        public static List<ScaledCase> Select(Func<FuzzCorpus.Case, string> gapOf)
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "Fuzz", "corpus");
            var files = Directory.GetFiles(dir, "*.jsonl").Select(Path.GetFileName)
                .Where(f => !f.StartsWith("index_", StringComparison.Ordinal))
                .Where(f => !f.EndsWith(".host.jsonl", StringComparison.Ordinal))
                .Where(f => !ExcludedFilePrefixes.Any(p => f.StartsWith(p, StringComparison.Ordinal)))
                .OrderBy(f => f, StringComparer.Ordinal).ToList();

            var picked = new Dictionary<string, List<string>>(StringComparer.Ordinal);   // op -> layouts taken
            var list = new List<ScaledCase>();
            foreach (var file in files)
            {
                using var corpus = FuzzCorpus.Open(file);
                foreach (var c in corpus)
                {
                    if (c.Expects_Throw || c.Operands.Length == 0 || c.Op is "grnd" or "random_api" or "rnd")
                        continue;
                    if (!c.Operands.Any(HoldsMemory))
                        continue;
                    var layouts = picked.TryGetValue(c.Op, out var l) ? l : picked[c.Op] = new List<string>();
                    string layout = c.Layout ?? "?";
                    if (layouts.Count >= CasesPerOp || layouts.Contains(layout))
                        continue;
                    layouts.Add(layout);
                    // gapOf reads the streamed case before the next line recycles its JSON elements.
                    string gap = gapOf(c);
                    list.Add(new ScaledCase(list.Count, c.Id, c.Op, file, Detach(c), gap));
                }
            }
            return list;
        }

        /// <summary>
        ///     A copy of a streamed case that outlives the stream: what the replay needs (the op, its parameters, the operands,
        ///     the result kind), with every <see cref="JsonElement"/> cloned out of the line's document.
        /// </summary>
        /// <param name="c">A case the corpus stream yielded (its JSON elements die with the stream's next line).</param>
        /// <returns>The standalone copy.</returns>
        /// <remarks>The streaming reader recycles each line's document, so a kept element throws on its next read ("not
        ///     valid due to the current state of the object"); the NumPy answer is not needed here and is dropped.</remarks>
        private static FuzzCorpus.Case Detach(FuzzCorpus.Case c) => new()
        {
            Id = c.Id,
            Op = c.Op,
            Params = c.Params?.ToDictionary(kv => kv.Key, kv => kv.Value.Clone(), StringComparer.Ordinal),
            Operands = c.Operands,
            Expected = new FuzzCorpus.Expected { Kind = c.Expected?.Kind },
            Layout = c.Layout,
            Alias = c.Alias,
        };

        /// <summary>True when an operand holds memory (not empty).</summary>
        /// <param name="o">The operand.</param>
        /// <returns>Whether every dimension is non-zero and it has a buffer.</returns>
        private static bool HoldsMemory(FuzzCorpus.Operand o) => o.BufferSize > 0 && o.Shape.All(d => d > 0);

        /// <summary>Writes a selection for the child (one serialized case per line, in selection order).</summary>
        /// <param name="path">The file.</param>
        /// <param name="selection">The cases.</param>
        /// <exception cref="IOException">The file cannot be written.</exception>
        public static void WriteSelection(string path, List<ScaledCase> selection)
        {
            // An explicit record per line: the case's Expected carries JsonElement fields that are default (undefined) for
            // most cases, and a default JsonElement cannot be serialized.
            using var stream = File.Create(path);
            foreach (var s in selection)
            {
                using (var w = new Utf8JsonWriter(stream))
                {
                    w.WriteStartObject();
                    w.WriteNumber("index", s.Index);
                    w.WriteString("id", s.Id);
                    w.WriteString("op", s.Op);
                    w.WriteString("file", s.File);
                    if (s.Gap is null) w.WriteNull("gap"); else w.WriteString("gap", s.Gap);
                    if (s.Case.Expected?.Kind is null) w.WriteNull("kind"); else w.WriteString("kind", s.Case.Expected.Kind);
                    if (s.Case.Layout is null) w.WriteNull("layout"); else w.WriteString("layout", s.Case.Layout);
                    w.WriteBoolean("alias", s.Case.Alias);
                    w.WritePropertyName("params");
                    w.WriteStartObject();
                    if (s.Case.Params != null)
                        foreach (var kv in s.Case.Params)
                        {
                            w.WritePropertyName(kv.Key);
                            kv.Value.WriteTo(w);
                        }
                    w.WriteEndObject();
                    w.WritePropertyName("operands");
                    JsonSerializer.Serialize(w, s.Case.Operands);
                    w.WriteEndObject();
                }
                stream.WriteByte((byte)'\n');
            }
        }

        /// <summary>Reads a selection written by <see cref="WriteSelection"/>.</summary>
        /// <param name="path">The file.</param>
        /// <returns>The cases.</returns>
        /// <exception cref="IOException">The file cannot be read.</exception>
        /// <exception cref="JsonException">A line is malformed.</exception>
        public static List<ScaledCase> ReadSelection(string path)
        {
            var list = new List<ScaledCase>();
            static string Str(JsonElement e) => e.ValueKind == JsonValueKind.String ? e.GetString() : null;
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                using var doc = JsonDocument.Parse(line);
                var r = doc.RootElement;
                var c = new FuzzCorpus.Case
                {
                    Id = r.GetProperty("id").GetString(),
                    Op = r.GetProperty("op").GetString(),
                    // Deserialized elements own their data: they outlive this line's document.
                    Params = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(r.GetProperty("params").GetRawText()),
                    Operands = JsonSerializer.Deserialize<FuzzCorpus.Operand[]>(r.GetProperty("operands").GetRawText()),
                    Expected = new FuzzCorpus.Expected { Kind = Str(r.GetProperty("kind")) },
                    Layout = Str(r.GetProperty("layout")),
                    Alias = r.GetProperty("alias").GetBoolean(),
                };
                list.Add(new ScaledCase(r.GetProperty("index").GetInt32(), c.Id, c.Op, r.GetProperty("file").GetString(), c,
                    Str(r.GetProperty("gap"))));
            }
            return list;
        }

        /// <summary>
        ///     The child mode: replays the selection from <paramref name="start"/>, the control first.
        /// </summary>
        /// <param name="selectionPath">The selection file.</param>
        /// <param name="start">The first selection index.</param>
        /// <param name="reps">Stressed runs per case.</param>
        /// <param name="report">The report file.</param>
        /// <param name="gap">Replay only this gap's cases, or null.</param>
        /// <returns>0 when every case was replayed.</returns>
        /// <exception cref="IOException">The selection or the report cannot be accessed.</exception>
        public static int RunChild(string selectionPath, int start, int reps, string report, string gap)
        {
            var cases = ReadSelection(selectionPath);
            using var thief = new LifetimeThief();
            if (start == 0)
                LifetimeChild.Append(report, new { control = LifetimeChild.RunControl(thief), reps = LifetimeChild.ControlReps });
            for (int i = start; i < cases.Count; i++)
            {
                var sc = cases[i];
                if (gap != null && sc.Gap != gap)
                    continue;
                Console.Out.WriteLine($"BEGIN {sc.Index} {sc.Id}");
                Console.Out.Flush();
                LifetimeChild.Append(report, RunCase(sc, reps, thief));
            }
            LifetimeChild.Append(report, new { done = true });
            return 0;
        }

        /// <summary>
        ///     Replays one case: scale it, check it is deterministic, then run it stressed.
        /// </summary>
        /// <param name="sc">The case.</param>
        /// <param name="reps">Stressed runs.</param>
        /// <param name="thief">The stress thread.</param>
        /// <returns>The outcome line.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static object RunCase(ScaledCase sc, int reps, LifetimeThief thief)
        {
            var c = sc.Case;
            bool masked = IsMaskedFile(sc.File);
            if (!TryScale(c, masked, out var strategy, out int k, out string why))
                return Outcome(sc, "skipped: " + why, 0, 0, null, 0, k, null);

            string reference, again;
            try
            {
                reference = HeldRun(c, masked, strategy, k);
                again = HeldRun(c, masked, strategy, k);
            }
            catch (Exception e)
            {
                return Outcome(sc, "skipped: the scaled reference raised " + LifetimeChild.Describe(e), 0, 0, null, 0, k, strategy);
            }
            if (reference != again)
                return Outcome(sc, "skipped: not deterministic (two held runs differ)", 0, 0, null, 0, k, strategy);

            // The buckets the operands take: the data and, for a masked case, the scaled masks (they can be lost too).
            var sizes = c.Operands.Where(HoldsMemory)
                .SelectMany(o => masked && o.Mask != null
                    ? new[] { ScaledBytes(o, strategy, k), MaskBytes(o, strategy, k) }
                    : new[] { ScaledBytes(o, strategy, k) })
                .Distinct().ToArray();
            thief.SetCaseSizes(sizes);
            int wrong = 0, threw = 0;
            string first = null;
            int collectionsBefore = thief.Collections;
            for (int r = 0; r < reps; r++)
            {
                object res;
                thief.Enter();
                try
                {
                    res = InvokeTemporaries(c, masked, strategy, k);
                }
                catch (Exception e)
                {
                    thief.Exit();
                    threw++;
                    first ??= $"rep {r}: " + LifetimeChild.Describe(e);
                    continue;
                }
                thief.Exit();
                string fp = Fingerprint(res);
                DisposeResult(res, Array.Empty<NDArray>());
                if (fp != reference)
                {
                    wrong++;
                    first ??= $"rep {r}: the answer differs from the held run's";
                }
            }
            return Outcome(sc, "ok", wrong, threw, first, thief.Collections - collectionsBefore, k, strategy);
        }

        /// <summary>Builds an outcome line in the lifetime report's format (plus the scale used).</summary>
        /// <param name="sc">The case.</param>
        /// <param name="refStatus"><c>ok</c>, or why the case was skipped.</param>
        /// <param name="wrong">Stressed runs that answered differently.</param>
        /// <param name="threw">Stressed runs that threw.</param>
        /// <param name="first">The first divergence, or null.</param>
        /// <param name="collections">Collections forced during the stressed runs.</param>
        /// <param name="k">The scale factor.</param>
        /// <param name="strategy">The scaling strategy, or null.</param>
        /// <returns>The line object.</returns>
        private static object Outcome(ScaledCase sc, string refStatus, int wrong, int threw, string first, int collections, int k, Strategy? strategy)
            => new { index = sc.Index, id = sc.Id, gap = sc.Gap, @ref = refStatus, wrong, threw, first, collections, op = sc.Op, k,
                     strategy = strategy?.ToString() };

        /// <summary>
        ///     Finds a scale: for each strategy, grows K by four from 1 while the call stays deterministic in shape, its
        ///     result grows linearly, its time stays under <see cref="TargetCallMs"/> and the largest operand under
        ///     <see cref="TargetBytes"/>.
        /// </summary>
        /// <param name="c">The case.</param>
        /// <param name="masked">The case is a masked-array tier's (its operands are masked arrays).</param>
        /// <param name="strategy">The chosen strategy.</param>
        /// <param name="k">The chosen scale factor (the last tried, for the report, when none works).</param>
        /// <param name="why">Why no strategy works, when the method returns false.</param>
        /// <returns>True when a scale was found.</returns>
        private static bool TryScale(FuzzCorpus.Case c, bool masked, out Strategy strategy, out int k, out string why)
        {
            why = null;
            k = 1;
            foreach (Strategy s in Enum.GetValues(typeof(Strategy)))
            {
                strategy = s;
                long size1;
                try
                {
                    size1 = TimedHeldSize(c, masked, s, 1, out _);
                }
                catch (Exception e)
                {
                    why = $"{s} at K=1 raised {LifetimeChild.Describe(e)}";
                    continue;
                }

                int kk = 1;
                long sizeK = size1;
                bool ok = true;
                while (kk < MaxK && LargestOperandBytes(c, s, kk) < TargetBytes)
                {
                    int next = kk * 4;
                    long sizeNext;
                    double ms;
                    try
                    {
                        sizeNext = TimedHeldSize(c, masked, s, next, out ms);
                    }
                    catch (Exception e)
                    {
                        why = $"{s} at K={next} raised {LifetimeChild.Describe(e)}";
                        ok = false;
                        break;
                    }
                    // Linear growth: four times the scale may give at most about four times the result (plus a constant
                    // for results that do not grow at all, such as reductions to a scalar).
                    if (sizeNext > 5 * Math.Max(sizeK, 1) + 64)
                    {
                        why = $"{s}: the result grows faster than linearly ({sizeK} -> {sizeNext} elements for K {kk} -> {next})";
                        ok = false;
                        break;
                    }
                    kk = next;
                    sizeK = sizeNext;
                    if (ms >= TargetCallMs)
                        break;
                }
                k = kk;
                if (ok && kk > 1)
                    return true;
                if (ok)
                    why = $"{s}: no growth was possible";
            }
            strategy = Strategy.LeadingAxis;
            return false;
        }

        /// <summary>Runs the case once over held operands at scale <paramref name="k"/> and measures it.</summary>
        /// <param name="c">The case.</param>
        /// <param name="masked">The case is a masked-array tier's.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <param name="ms">The call's wall time in milliseconds.</param>
        /// <returns>The result's total element count (1 for a non-array result).</returns>
        /// <exception cref="Exception">Whatever the op raises.</exception>
        private static long TimedHeldSize(FuzzCorpus.Case c, bool masked, Strategy s, int k, out double ms)
        {
            using var held = HeldOperands.Build(c, masked, s, k);
            var sw = Stopwatch.StartNew();
            var res = Invoke(c, held.Operands);
            ms = sw.Elapsed.TotalMilliseconds;
            long size = ResultSize(res);
            DisposeResult(res, held.Owned);
            return size;
        }

        /// <summary>Runs the case over held operands and fingerprints its answer.</summary>
        /// <param name="c">The case.</param>
        /// <param name="masked">The case is a masked-array tier's.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>The fingerprint.</returns>
        /// <exception cref="Exception">Whatever the op raises.</exception>
        private static string HeldRun(FuzzCorpus.Case c, bool masked, Strategy s, int k)
        {
            using var held = HeldOperands.Build(c, masked, s, k);
            var res = Invoke(c, held.Operands);
            string fp = Fingerprint(res);
            DisposeResult(res, held.Owned);
            return fp;
        }

        /// <summary>
        ///     A case's operands built for a HELD run (the reference, the scaling probes): this object keeps them reachable
        ///     for the whole call, and <see cref="Dispose"/> releases every array the harness built for them.
        /// </summary>
        private sealed class HeldOperands : IDisposable
        {
            /// <summary>The operand array the registry takes: <c>NDArray[]</c>, or <c>NDMaskedArray[]</c> for a masked case.</summary>
            public object Operands { get; private init; }

            /// <summary>
            ///     Every array built for the operands (data, masks, and the arrays a masked array wraps them in): what the
            ///     result disposer must keep, and what <see cref="Dispose"/> releases. The np.ma singletons are never listed.
            /// </summary>
            public List<NDArray> Owned { get; } = new();

            /// <summary>Builds a case's operands at scale <paramref name="k"/>, holding them.</summary>
            /// <param name="c">The case.</param>
            /// <param name="masked">Build masked arrays (a masked-array tier's case).</param>
            /// <param name="s">The strategy.</param>
            /// <param name="k">The scale.</param>
            /// <returns>The held operands; dispose them after the call.</returns>
            public static HeldOperands Build(FuzzCorpus.Case c, bool masked, Strategy s, int k)
            {
                if (masked)
                {
                    var owned = new List<NDArray>();
                    var ops = BuildAllMasked(c, s, k, owned);
                    var held = new HeldOperands { Operands = ops };
                    held.Owned.AddRange(owned);
                    return held;
                }
                var arrays = BuildAll(c, s, k);
                var plain = new HeldOperands { Operands = arrays };
                plain.Owned.AddRange(arrays);
                return plain;
            }

            /// <summary>Disposes every array built for the operands, once each (an alias case lists one array twice).</summary>
            public void Dispose()
            {
                foreach (var a in Owned.Distinct<NDArray>(ReferenceEqualityComparer.Instance))
                    a.Dispose();
            }
        }

        /// <summary>Runs the case's op over built operands through the registry entry its tier uses.</summary>
        /// <param name="c">The case.</param>
        /// <param name="operands"><c>NDArray[]</c> for an ordinary case, <c>NDMaskedArray[]</c> for a masked one.</param>
        /// <returns>The op's result.</returns>
        /// <exception cref="Exception">Whatever the op raises.</exception>
        private static object Invoke(FuzzCorpus.Case c, object operands)
            => operands is NDMaskedArray[] masked
                ? InvokeMasked(c, masked)
                : OpRegistry.Invoke(c.Expected.KindOrArray, c.Op, c.Params, (NDArray[])operands);

        /// <summary>Runs a masked-array case's op: a <c>masked_tuple</c> case through the tuple entry, any other through
        ///     <see cref="OpRegistry.ApplyMasked"/> (as the masked replay does).</summary>
        /// <param name="c">The case.</param>
        /// <param name="operands">The masked operands.</param>
        /// <returns>The op's result.</returns>
        /// <exception cref="Exception">Whatever the op raises.</exception>
        private static object InvokeMasked(FuzzCorpus.Case c, NDMaskedArray[] operands)
            => c.Expected.KindOrArray == "masked_tuple"
                ? OpRegistry.ApplyMaskedTuple(c.Op, c.Params, operands)
                : OpRegistry.ApplyMasked(c.Op, c.Params, operands);

        /// <summary>True for a case of a masked-array tier (its operands are built as <see cref="NDMaskedArray"/>s).</summary>
        /// <param name="file">The corpus file the case came from.</param>
        /// <returns>Whether the file is an <c>ma_*</c> tier.</returns>
        private static bool IsMaskedFile(string file) => file.StartsWith("ma_", StringComparison.Ordinal);

        /// <summary>
        ///     Runs the case with every operand built inside the call expression: the operand array is a temporary passed
        ///     straight into the registry, and nothing in this frame references an operand.
        /// </summary>
        /// <param name="c">The case.</param>
        /// <param name="masked">Build masked arrays (a masked-array tier's case): the masked arrays, their data and their
        ///     masks are all temporaries then.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>The op's result.</returns>
        /// <exception cref="Exception">Whatever the op raises.</exception>
        /// <remarks><c>NoInlining</c> keeps the temporary out of the caller's frame (an inlined copy could let the caller's
        ///     locals hold the array).</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static object InvokeTemporaries(FuzzCorpus.Case c, bool masked, Strategy s, int k)
            => masked
                ? InvokeMasked(c, BuildAllMasked(c, s, k, null))
                : OpRegistry.Invoke(c.Expected.KindOrArray, c.Op, c.Params, BuildAll(c, s, k));

        /// <summary>Builds every operand of a case at scale <paramref name="k"/> (an alias case repeats one array).</summary>
        /// <param name="c">The case.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>A fresh operand array.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray[] BuildAll(FuzzCorpus.Case c, Strategy s, int k)
        {
            var ops = new NDArray[c.Operands.Length];
            for (int i = 0; i < ops.Length; i++)
                ops[i] = Build(c.Operands[i], s, k);
            return c.Alias && ops.Length == 1 ? new[] { ops[0], ops[0] } : ops;
        }

        /// <summary>
        ///     Builds every operand of a masked-array case at scale <paramref name="k"/>: the scaled data and the scaled mask
        ///     wrapped by <c>np.ma.masked_array(data, mask)</c>, as the masked replay builds them.
        /// </summary>
        /// <param name="c">The case (its operands carry the mask as <see cref="FuzzCorpus.Operand.Mask"/>).</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <param name="owned">Collects every array built (for a held run to keep and dispose), or null for a stressed run,
        ///     whose operands must be referenced by nothing but the call.</param>
        /// <returns>A fresh masked operand array.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDMaskedArray[] BuildAllMasked(FuzzCorpus.Case c, Strategy s, int k, List<NDArray> owned)
        {
            var ops = new NDMaskedArray[c.Operands.Length];
            for (int i = 0; i < ops.Length; i++)
            {
                var data = Build(c.Operands[i], s, k);
                var mask = BuildMask(c.Operands[i], s, k);
                var m = np.ma.masked_array(data, mask);
                if (owned != null)
                {
                    // The masked array may wrap the data and the mask in arrays of its own; all of them are the harness's.
                    // np.ma.nomask is shared by the whole process: never listed (so never disposed).
                    foreach (var a in new[] { data, mask, m._data, m._mask })
                        if (a is not null && !Ownership.SharedArrays.Contains(a))
                            owned.Add(a);
                }
                ops[i] = m;
            }
            return c.Alias && ops.Length == 1 ? new[] { ops[0], ops[0] } : ops;
        }

        /// <summary>
        ///     Builds one operand's mask at scale <paramref name="k"/>, matching <see cref="Build"/>'s data: K copies of the
        ///     mask's C-order bytes, under a new leading axis (<see cref="Strategy.LeadingAxis"/>) or along axis 0
        ///     (<see cref="Strategy.Tile"/>). The result is pooled memory, so the stress thread can poison it when it is lost.
        /// </summary>
        /// <param name="o">The corpus operand.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>The scaled mask, or null for <c>nomask</c> (the operand carries none).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe NDArray BuildMask(FuzzCorpus.Operand o, Strategy s, int k)
        {
            var small = FuzzCorpus.ReconstructMask(o.Shape, o.Mask);
            // nomask, an empty or 0-d operand, and the unscaled tile keep the corpus mask (Build keeps their data too).
            if (small is null || !HoldsMemory(o) || o.Shape.Length == 0 || k == 1 && s == Strategy.Tile)
                return small;

            long one = small.size;   // one byte per element, C order (ReconstructMask builds a fresh C-ordered mask)
            long[] dims;
            if (s == Strategy.LeadingAxis)
            {
                dims = new long[o.Shape.Length + 1];
                dims[0] = k;
                Array.Copy(o.Shape, 0, dims, 1, o.Shape.Length);
            }
            else
            {
                dims = (long[])o.Shape.Clone();
                dims[0] *= k;
            }
            try
            {
                var big = np.empty(new Shape(dims), np.@bool);
                for (long t = 0; t < k; t++)
                    Buffer.MemoryCopy(small.Address, (byte*)big.Address + t * one, one, one);
                return big;
            }
            finally
            {
                small.Dispose();
            }
        }

        /// <summary>The byte size an operand's mask has at scale <paramref name="k"/> (its pooled buffer's size).</summary>
        /// <param name="o">The operand (it carries a mask).</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>The bytes: one per logical element, K times unless the operand stays as it is.</returns>
        private static long MaskBytes(FuzzCorpus.Operand o, Strategy s, int k)
        {
            long logical = o.Shape.Aggregate(1L, (a, d) => a * d);
            return !HoldsMemory(o) || o.Shape.Length == 0 || k == 1 && s == Strategy.Tile ? logical : k * logical;
        }

        /// <summary>
        ///     Builds one operand at scale <paramref name="k"/>, in pooled memory (the stress thread poisons buffers the pool
        ///     gets back).
        /// </summary>
        /// <param name="o">The corpus operand.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>A fresh array the caller owns.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe NDArray Build(FuzzCorpus.Operand o, Strategy s, int k)
        {
            // 0-d and empty operands are not enlarged (a 0-d bound or scalar must stay one), only rebuilt fresh.
            if (!HoldsMemory(o) || o.Shape.Length == 0 || k == 1 && s == Strategy.Tile)
                return FuzzCorpus.Reconstruct(o);

            var tc = FuzzCorpus.DtypeToTC(o.Dtype);
            int itemSize = tc.SizeOf();
            if (s == Strategy.LeadingAxis)
            {
                byte[] bytes = FuzzCorpus.FromHex(o.Buffer);
                long one = o.BufferSize * itemSize;
                var root = np.empty(new Shape(k * o.BufferSize), tc);
                fixed (byte* src = bytes)
                    for (long t = 0; t < k; t++)
                        Buffer.MemoryCopy(src, (byte*)root.Address + t * one, one, one);
                var dims = new long[o.Shape.Length + 1];
                var strides = new long[o.Shape.Length + 1];
                dims[0] = k;
                strides[0] = o.BufferSize;
                Array.Copy(o.Shape, 0, dims, 1, o.Shape.Length);
                Array.Copy(o.Strides, 0, strides, 1, o.Strides.Length);
                var view = new NDArray(root.Storage, new Shape(dims, strides, o.Offset, k * o.BufferSize));
                root.Dispose();
                if (o.Writeable == false)
                    view.setflags(write: false);
                return view;
            }

            // Tile: the operand's logical C-order elements, K times along axis 0.
            var small = FuzzCorpus.Reconstruct(o);
            NDArray contiguous = null;
            try
            {
                contiguous = np.ascontiguousarray(small);
                long one = small.size * itemSize;
                var dims = (long[])o.Shape.Clone();
                dims[0] *= k;
                var tiled = np.empty(new Shape(dims), tc);
                for (long t = 0; t < k; t++)
                    Buffer.MemoryCopy(contiguous.Address, (byte*)tiled.Address + t * one, one, one);
                if (o.Writeable == false)
                    tiled.setflags(write: false);
                return tiled;
            }
            finally
            {
                if (contiguous is not null && !ReferenceEquals(contiguous, small))
                    contiguous.Dispose();
                small.Dispose();
            }
        }

        /// <summary>The byte size an operand has at scale <paramref name="k"/> (its pooled buffer's size).</summary>
        /// <param name="o">The operand.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>The bytes.</returns>
        private static long ScaledBytes(FuzzCorpus.Operand o, Strategy s, int k)
        {
            int itemSize = FuzzCorpus.DtypeToTC(o.Dtype).SizeOf();
            if (!HoldsMemory(o) || o.Shape.Length == 0)
                return o.BufferSize * itemSize;
            long logical = o.Shape.Aggregate(1L, (a, d) => a * d);
            return s == Strategy.LeadingAxis ? k * o.BufferSize * itemSize : k * logical * itemSize;
        }

        /// <summary>The largest operand's byte size at scale <paramref name="k"/>.</summary>
        /// <param name="c">The case.</param>
        /// <param name="s">The strategy.</param>
        /// <param name="k">The scale.</param>
        /// <returns>The bytes.</returns>
        private static long LargestOperandBytes(FuzzCorpus.Case c, Strategy s, int k)
            => c.Operands.Select(o => ScaledBytes(o, s, k)).DefaultIfEmpty(0).Max();

        /// <summary>The element count of a result (arrays summed over a tuple; 1 for anything else).</summary>
        /// <param name="res">The result.</param>
        /// <returns>The count.</returns>
        private static long ResultSize(object res)
        {
            long total = 0;
            foreach (var o in UndisposedIntermediateTests.CollectDisposables(res))
                if (o is NDArray a)
                    total += a.size;
            return Math.Max(total, 1);
        }

        /// <summary>
        ///     A content fingerprint of a result, for comparing a stressed run with the held run: every array's dtype,
        ///     shape and C-order bytes (hashed), recursively through tuples, arrays of arrays and masked arrays; any other
        ///     value through its text. A masked array contributes its mask and its data at the UNMASKED slots only.
        /// </summary>
        /// <param name="res">The result.</param>
        /// <returns>The fingerprint.</returns>
        /// <remarks>
        ///     The data under a mask is undefined by numpy.ma's contract: <c>ma.masked_all_like</c> hands out
        ///     <c>np.empty_like</c> memory, so a held run and a stressed run (whose pool buffers the stress thread has
        ///     poisoned) differ there with no lost anchor at all. Hashing those slots made it fail 24 of 24 runs.
        /// </remarks>
        private static string Fingerprint(object res)
        {
            var sb = new StringBuilder();

            // The data of a masked array with its masked slots zeroed (a full mask of the data's size), or the whole data
            // when the mask is nomask or of another size (numpy.ma's masks match their data; anything else is hashed as is).
            void AddMaskedData(NDMaskedArray m)
            {
                var data = m._data;
                var mask = m._mask;
                if (data is null || mask is null || Ownership.SharedArrays.Contains(mask) || mask.size != data.size || data.size == 0)
                {
                    Add(data, 1);
                    return;
                }
                byte[] bytes = data.tobytes();
                byte[] flags = mask.tobytes();   // one byte per element, C order (a bool array)
                int item = (int)(bytes.Length / data.size);
                for (long i = 0; i < flags.Length; i++)
                    if (flags[i] != 0)
                        Array.Clear(bytes, (int)(i * item), item);
                sb.Append(data.dtype.str).Append('(').Append(string.Join(",", data.shape)).Append(")m")
                  .Append(Convert.ToHexString(SHA256.HashData(bytes))).Append(';');
            }

            void Add(object o, int depth)
            {
                if (depth > 8)
                    return;
                switch (o)
                {
                    case null:
                        sb.Append("null;");
                        return;
                    case NDArray a:
                        sb.Append(a.dtype.str).Append('(').Append(string.Join(",", a.shape)).Append(')')
                          .Append(Convert.ToHexString(SHA256.HashData(a.tobytes()))).Append(';');
                        return;
                    case NDMaskedArray m:
                        sb.Append("ma[");
                        AddMaskedData(m);
                        Add(m._mask, depth + 1);
                        sb.Append("];");
                        return;
                    case string str:
                        sb.Append('"').Append(str).Append("\";");
                        return;
                    case NDArray[] arr:
                        sb.Append('[');
                        foreach (var x in arr)
                            Add(x, depth + 1);
                        sb.Append("];");
                        return;
                    case System.Runtime.CompilerServices.ITuple t:
                        sb.Append('(');
                        for (int i = 0; i < t.Length; i++)
                            Add(t[i], depth + 1);
                        sb.Append(");");
                        return;
                    case object[] objs:
                        sb.Append('{');
                        foreach (var x in objs)
                            Add(x, depth + 1);
                        sb.Append("};");
                        return;
                    case double d:
                        sb.Append(d.ToString("R", System.Globalization.CultureInfo.InvariantCulture)).Append(';');
                        return;
                    case float f:
                        sb.Append(f.ToString("R", System.Globalization.CultureInfo.InvariantCulture)).Append(';');
                        return;
                    case IFormattable f:
                        sb.Append(f.ToString(null, System.Globalization.CultureInfo.InvariantCulture)).Append(';');
                        return;
                    default:
                        sb.Append(o.GetType().Name).Append(':').Append(o).Append(';');
                        return;
                }
            }
            Add(res, 0);
            return sb.ToString();
        }

        /// <summary>Disposes a result's arrays and disposable objects, except the given operands and shared arrays.</summary>
        /// <param name="res">The result.</param>
        /// <param name="keep">Held operands (never disposed here).</param>
        private static void DisposeResult(object res, IReadOnlyCollection<NDArray> keep)
        {
            var keepSet = new HashSet<object>(keep, ReferenceEqualityComparer.Instance);
            UndisposedIntermediateTests.DisposeAny(res, keepSet);
        }
    }
}
