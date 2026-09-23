using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The two corpus families the leak sweep used to SKIP because their case schemas differ from
    ///     <see cref="FuzzCorpus.Case"/>-through-<see cref="OpRegistry.Invoke"/>: the masked-array tiers
    ///     (<c>ma_*.jsonl</c>) and the index tiers (<c>index_*.jsonl</c>). Both run the sweep's exact
    ///     protocol (warm invocation, then <see cref="ScopeAudit.MeasureConfirmedTraffic"/>) and feed the
    ///     same <see cref="SweepAccumulator"/>, so their escapes/bypasses are verdicts of
    ///     <see cref="Corpus_AllOps_LeaveNoUndisposedIntermediates"/> and their measured op keys count
    ///     toward <see cref="LeakSurfaceCoverageTests"/>.
    /// </summary>
    public partial class UndisposedIntermediateTests
    {
        // ---------------------------------------------------------------------------------
        // Masked-array tiers (np.ma, 149 op keys / 68,860 cases at 2026-09-23).
        // ---------------------------------------------------------------------------------

        /// <summary>
        ///     Replays one masked-array tier: each operand is rebuilt as an <see cref="NDMaskedArray"/>
        ///     (data view + optional contiguous bool mask, exactly as FuzzCorpusTests.Ma does), the op
        ///     runs through <see cref="OpRegistry.ApplyMasked"/>/<see cref="OpRegistry.ApplyMaskedTuple"/>,
        ///     and the result's arrays are disposed — except any array the operands still reference.
        /// </summary>
        /// <remarks>
        ///     <para><b>Why the keep-set is recomputed per execution.</b> An <see cref="NDMaskedArray"/>
        ///     does not own its arrays, and several ma ops MUTATE an operand in place and return it
        ///     (<c>put</c>/<c>putmask</c> — possibly promoting a <c>nomask</c> operand to a fresh mask). The
        ///     arrays an operand references can therefore change between executions of the region, so
        ///     "what must not be disposed" is read from the operands AFTER each invocation, never cached:
        ///     disposing an operand's live mask would corrupt the next execution (a use-after-free, not a
        ///     leak verdict).</para>
        ///     <para><b>The masked constant.</b> A fully-masked scalar result is the process-wide
        ///     <see cref="NDMaskedConstant"/> singleton; its arrays are never disposed.</para>
        /// </remarks>
        /// <param name="file">The <c>ma_*.jsonl</c> file name.</param>
        /// <param name="includeOp">Optional op-key filter.</param>
        /// <param name="acc">The sweep's tallies.</param>
        private static void SweepMaskedFile(string file, Func<string, bool> includeOp, SweepAccumulator acc)
        {
            foreach (var c in FuzzCorpus.Load(file))
            {
                if (includeOp != null && !includeOp(c.Op))
                    continue;

                // Every array the harness built for this case (data views, masks, and whatever
                // masked_array wraps them in) — disposed once, after the case, never inside the region.
                var built = new List<NDArray>();
                NDMaskedArray[] ops;
                try
                {
                    ops = new NDMaskedArray[c.Operands.Length];
                    for (int i = 0; i < ops.Length; i++)
                    {
                        var data = FuzzCorpus.Reconstruct(c.Operands[i]);
                        built.Add(data);
                        var mask = FuzzCorpus.ReconstructMask(c.Operands[i].Shape, c.Operands[i].Mask);
                        if (mask is not null)
                            built.Add(mask);
                        ops[i] = np.ma.masked_array(data, mask);
                        built.Add(ops[i]._data);
                        if (ops[i]._mask is not null)
                            built.Add(ops[i]._mask);
                    }
                }
                catch
                {
                    acc.Threw(c.Op);
                    DisposeDistinct(built);
                    continue;
                }

                object Run() => c.Expected.KindOrArray == "masked_tuple"
                    ? OpRegistry.ApplyMaskedTuple(c.Op, c.Params, ops)
                    : OpRegistry.ApplyMasked(c.Op, c.Params, ops);

                // Arrays that must survive the region: everything the harness built, whatever the
                // operands reference NOW (a mutator may have attached a new mask to one of them), and
                // the np.ma process-wide singletons (mask_or/getmask/.mask hand back `nomask` itself).
                HashSet<object> Keep()
                {
                    var keep = new HashSet<object>(built, ReferenceEqualityComparer.Instance);
                    foreach (var singleton in MaskedSingletons())
                        keep.Add(singleton);
                    foreach (var m in ops)
                    {
                        if (m is null)
                            continue;
                        keep.Add(m._data);
                        if (m._mask is not null)
                            keep.Add(m._mask);
                    }
                    return keep;
                }

                void Region()
                {
                    object res = Run();
                    DisposeAny(res, Keep());
                }

                try
                {
                    if (c.Expects_Throw)
                    {
                        void ErrorRegion()
                        {
                            try
                            {
                                Region();
                            }
                            catch
                            {
                                // expected: the case is a NumPy error path
                            }
                        }

                        ErrorRegion();   // warm
                        var errTraffic = ScopeAudit.MeasureConfirmedTraffic(ErrorRegion);
                        if (errTraffic == null)
                        {
                            acc.GcInconclusive++;
                            continue;
                        }
                        acc.Record(c.Op, null, c.Layout ?? "?", errTraffic.Value, 0, errorPath: true, c.Id, file);
                        continue;
                    }

                    try
                    {
                        Region();        // warm (one-time caches, the first nomask->mask promotion)
                    }
                    catch
                    {
                        acc.Threw(c.Op);
                        continue;
                    }

                    var traffic = ScopeAudit.MeasureConfirmedTraffic(Region);
                    if (traffic == null)
                    {
                        acc.GcInconclusive++;
                        continue;
                    }

                    acc.Masked++;
                    // No freshness inspection here: masked results legitimately wrap operand VIEWS and
                    // the masked constant, and the ordinary sweep already gates the bypass class for the
                    // np.* kernels every ma op delegates to (freshBytes 0 = no bypass verdict).
                    acc.Record(c.Op, null, c.Layout ?? "?", traffic.Value, 0, errorPath: false, c.Id, file);
                }
                finally
                {
                    // Operands may now reference arrays the harness did not build (a mutator's new mask).
                    foreach (var m in ops)
                    {
                        if (m is null)
                            continue;
                        built.Add(m._data);
                        if (m._mask is not null)
                            built.Add(m._mask);
                    }
                    DisposeDistinct(built);
                }
            }
        }

        // ---------------------------------------------------------------------------------
        // Index tiers (the indexer get/set surface, 12,426 cases at 2026-09-23).
        // ---------------------------------------------------------------------------------

        /// <summary>
        ///     Replays one index tier through the <see cref="NDArray"/> indexer, the way IndexOracleTests
        ///     drives it: <c>get</c> cases index a base (<see cref="IndexOracleTests.Base"/> /
        ///     <see cref="IndexOracleTests.DtypeBase"/>) and dispose the result; <c>set</c> cases assign
        ///     into a pre-made copy of the base (the in-place write is re-executable — it rewrites the
        ///     same values). Cases where NumPy raised (<c>np.ok == false</c>) are ERROR paths.
        /// </summary>
        /// <remarks>
        ///     The base and the index tokens are built OUTSIDE the measured region under a harness
        ///     <see cref="NDScope"/>: the bases are views of throwaway parents (<c>arange(12).reshape</c>,
        ///     <c>.T</c>, slices) and the tokens are <c>np.array(...).reshape(...)</c> chains, so yielding
        ///     the final objects out of the scope releases every parent while ARC keeps the shared
        ///     buffers alive — no harness litter feeds the finalizer backlog the protocol has to drain.
        /// </remarks>
        /// <param name="file">The <c>index_*.jsonl</c> file name.</param>
        /// <param name="includeOp">Optional op-key filter (matched against <c>index.get</c>/<c>index.set</c>).</param>
        /// <param name="acc">The sweep's tallies.</param>
        private static void SweepIndexFile(string file, Func<string, bool> includeOp, SweepAccumulator acc)
        {
            // index_dtype / index_setter_dtype carry a "dtype" and no "base"/"op"; the curated and
            // random tiers carry "op" (get/set) + "base".
            foreach (var c in IndexOracleTests.LoadLines(file))
            {
                bool isDtypeTier = c.TryGetProperty("dtype", out var dtEl);
                string op = c.TryGetProperty("op", out var opEl)
                    ? opEl.GetString()
                    : (file.StartsWith("index_setter", StringComparison.Ordinal) ? "set" : "get");
                string key = "index." + op;
                if (includeOp != null && !includeOp(key))
                    continue;
                string id = c.GetProperty("id").GetString();
                bool errorPath = !c.GetProperty("np").GetProperty("ok").GetBoolean();
                string layout = isDtypeTier ? "dtype:" + dtEl.GetString() : "base:" + c.GetProperty("base").GetString();

                NDArray target;
                object[] idx;
                NDArray value = null;
                var built = new List<NDArray>();
                try
                {
                    using (var scope = NDScope.Open())
                    {
                        var b = isDtypeTier ? IndexOracleTests.DtypeBase(dtEl.GetString()) : IndexOracleTests.Base(c.GetProperty("base").GetString());
                        if (op == "set")
                        {
                            // The setter writes an independent copy, as IndexOracleTests does; order='K'
                            // (when the case records it) keeps a view base's F-contig/strided layout so
                            // the SET runs into a genuinely non-contiguous destination.
                            char ord = c.TryGetProperty("order", out var oe) ? oe.GetString()[0] : 'C';
                            target = scope.Returns(b.copy(ord));
                            value = scope.Returns(isDtypeTier
                                ? IndexOracleTests.BuildTypedValue(c.GetProperty("value"))
                                : IndexOracleTests.BuildValue(c.GetProperty("value")));
                        }
                        else
                        {
                            target = scope.Returns(b);
                        }
                        idx = IndexOracleTests.BuildIndex(c.GetProperty("tokens"));
                        foreach (var t in idx)
                            if (t is NDArray nd)
                                built.Add(scope.Returns(nd));
                    }
                    built.Add(target);
                    if (value is not null)
                        built.Add(value);
                }
                catch
                {
                    acc.Threw(key);
                    DisposeDistinct(built);
                    continue;
                }

                var keep = new HashSet<object>(built, ReferenceEqualityComparer.Instance);
                void Region()
                {
                    if (op == "set")
                    {
                        target[idx] = value;
                    }
                    else
                    {
                        var r = target[idx];
                        if (!keep.Contains(r))
                            r.Dispose();
                    }
                }

                try
                {
                    if (errorPath)
                    {
                        void ErrorRegion()
                        {
                            try
                            {
                                Region();
                            }
                            catch
                            {
                                // expected: NumPy raised on this index
                            }
                        }

                        ErrorRegion();   // warm
                        var errTraffic = ScopeAudit.MeasureConfirmedTraffic(ErrorRegion);
                        if (errTraffic == null)
                        {
                            acc.GcInconclusive++;
                            continue;
                        }
                        acc.Record(key, null, layout, errTraffic.Value, 0, errorPath: true, id, file);
                        continue;
                    }

                    try
                    {
                        Region();        // warm
                    }
                    catch
                    {
                        acc.Threw(key);  // an index divergence is IndexOracleTests' verdict, not ours
                        continue;
                    }

                    var traffic = ScopeAudit.MeasureConfirmedTraffic(Region);
                    if (traffic == null)
                    {
                        acc.GcInconclusive++;
                        continue;
                    }

                    acc.Index++;
                    // Index results are views or fancy copies of the base; no freshness verdict (the
                    // fancy-copy kernels are the take/put kernels the ordinary sweep already gates).
                    acc.Record(key, null, layout, traffic.Value, 0, errorPath: false, id, file);
                }
                finally
                {
                    DisposeDistinct(built);
                }
            }
        }

        // ---------------------------------------------------------------------------------
        // Shape-agnostic result disposal (masked tiers, catalogue).
        // ---------------------------------------------------------------------------------

        /// <summary>
        ///     Disposes every <see cref="NDArray"/> a result object carries exactly once — bare arrays,
        ///     array tuples, <see cref="NDMaskedArray"/> results (their data AND mask), masked tuples,
        ///     C# tuples of any of those, and <see cref="IDisposable"/> result objects (iterators) —
        ///     skipping anything in <paramref name="keep"/> and the <see cref="NDMaskedConstant"/>
        ///     singleton. Non-array results (scalars, strings, dtypes) carry nothing and are ignored.
        /// </summary>
        /// <remarks>
        ///     A result may legitimately CONTAIN kept arrays — a masked result sharing an operand's
        ///     mask, a view-returning op handing back an operand — so the keep-set check is per array,
        ///     not per result. Views of kept arrays are disposed normally: ARC releases the view's
        ///     reference only, the base keeps the buffer, and no pool return is counted (a view took
        ///     nothing).
        /// </remarks>
        /// <param name="result">The op result, of any shape.</param>
        /// <param name="keep">Arrays that must survive (operands, fixtures), compared by reference.</param>
        internal static void DisposeAny(object result, HashSet<object> keep)
        {
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);

            void One(NDArray a)
            {
                if (a is null || !seen.Add(a) || keep.Contains(a))
                    return;
                a.Dispose();
            }

            void Visit(object o, int depth)
            {
                if (o is null || depth > 8)
                    return;   // depth guard: a pathological self-referencing tuple cannot recurse forever
                switch (o)
                {
                    case NDArray nd:
                        One(nd);
                        return;
                    case NDMaskedConstant:
                        return;   // the process-wide masked singleton: never disposed
                    case NDMaskedArray m:
                        One(m._data);
                        One(m._mask);
                        return;
                    case NDArray[] arr:
                        foreach (var a in arr)
                            One(a);
                        return;
                    case NDMaskedArray[] ms:
                        foreach (var m in ms)
                            Visit(m, depth + 1);
                        return;
                    case System.Runtime.CompilerServices.ITuple tuple:
                        for (int i = 0; i < tuple.Length; i++)
                            Visit(tuple[i], depth + 1);
                        return;
                    case string:
                        return;
                    case IEnumerable<NDArray> nds when o is not IDisposable:
                        // A materialized collection of arrays (List<NDArray>, a LINQ-free array view).
                        // Only NDArray-typed sequences are walked: enumerating an ARBITRARY IEnumerable
                        // could consume a lazy iterator result (np.ndindex, an ndenumerate) or run
                        // user code, which a disposer must never do.
                        foreach (var a in nds)
                            One(a);
                        return;
                    case object[] objs:
                        foreach (var item in objs)
                            Visit(item, depth + 1);
                        return;
                    case IDisposable d:
                        if (seen.Add(d) && !keep.Contains(d))
                            d.Dispose();
                        return;
                }
            }

            Visit(result, 0);
        }

        /// <summary>
        ///     The np.ma module's process-wide <see cref="NDArray"/> singletons — <c>np.ma.nomask</c> and
        ///     the masked constant's 0-d data/mask. Several members hand them back AS their result
        ///     (<c>mask_or</c>/<c>make_mask</c> shrinking to nomask, <c>getmask</c> and
        ///     <see cref="NDMaskedArray.mask"/> of an unmasked array), so every never-dispose set must
        ///     carry them: disposing one would release a SHARED scalar slot that the next allocation
        ///     reuses, silently rewriting nomask for the whole process.
        /// </summary>
        /// <returns>The singleton arrays.</returns>
        internal static IEnumerable<object> MaskedSingletons()
        {
            yield return np.ma.nomask;
            yield return np.ma.NDMasked._data;
            if (np.ma.NDMasked._mask is not null)
                yield return np.ma.NDMasked._mask;
        }

        /// <summary>Disposes each distinct array in <paramref name="arrays"/> once (null entries skipped).</summary>
        /// <param name="arrays">Harness-built arrays, possibly with duplicates.</param>
        private static void DisposeDistinct(IEnumerable<NDArray> arrays)
        {
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
            foreach (var a in arrays)
                if (a is not null && seen.Add(a))
                    a.Dispose();
        }
    }
}
