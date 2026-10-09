using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp.Tests.Fuzz.Ownership;

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
        ///     The masked-array op keys that MUTATE an operand in place (<c>put</c> / <c>putmask</c>), whose measured runs
        ///     may raise where the case's first run succeeded: the warm run changes the target's state (it may promote a
        ///     <c>nomask</c> target to a mask, which sends a hard-masked put into NumPy's hard-mask branch). The sweep
        ///     audits the pool traffic of whichever path those runs take instead of crashing on the exception; the corpus
        ///     tiers gate the outcome on fresh operands.
        /// </summary>
        private static readonly HashSet<string> MaskedMutators = new(StringComparer.Ordinal) { "ma.put", "ma.putmask" };

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
            // Streamed (see CorpusFile): one parsed case alive at a time instead of the whole tier's list.
            using var corpus = FuzzCorpus.Open(file);
            foreach (var c in corpus)
            {
                if (includeOp != null && !includeOp(c.Op))
                    continue;

                // Every array the harness built for this case (data views, masks, and whatever
                // masked_array wraps them in) — disposed once, after the case, never inside the region.
                var built = new List<NDArray>();
                // The same arrays with their roles: what the op does not own (an NDMaskedArray does not own its arrays
                // either), snapshotted for the ownership checks.
                var owned = new List<(NDArray, string)>();
                NDMaskedArray[] ops;
                try
                {
                    ops = new NDMaskedArray[c.Operands.Length];
                    for (int i = 0; i < ops.Length; i++)
                    {
                        var data = FuzzCorpus.Reconstruct(c.Operands[i]);
                        built.Add(data);
                        owned.Add((data, $"data {i}"));
                        var mask = FuzzCorpus.ReconstructMask(c.Operands[i].Shape, c.Operands[i].Mask);
                        if (mask is not null)
                        {
                            built.Add(mask);
                            owned.Add((mask, $"mask {i}"));
                        }
                        ops[i] = np.ma.masked_array(data, mask);
                        built.Add(ops[i]._data);
                        owned.Add((ops[i]._data, $"masked {i} _data"));
                        if (ops[i]._mask is not null)
                        {
                            built.Add(ops[i]._mask);
                            owned.Add((ops[i]._mask, $"masked {i} _mask"));
                        }
                    }
                }
                catch
                {
                    acc.Threw(c.Op);
                    DisposeDistinct(built);
                    continue;
                }

                var ledger = OwnershipLedger.Of(owned);

                // The dispatcher of the case's result kind (an error case keeps its kind in ma_surface.jsonl): the text
                // and plain-array tuple ops of the numpy.ma surface tier are not registered in ApplyMasked.
                object Run() => c.Expected.KindOrArray switch
                {
                    "masked_tuple" => OpRegistry.ApplyMaskedTuple(c.Op, c.Params, ops),
                    "tuple" => OpRegistry.ApplyMaskedArrayTuple(c.Op, c.Params, ops),
                    "text" => OpRegistry.ApplyMaskedText(c.Op, c.Params, ops),
                    _ => OpRegistry.ApplyMasked(c.Op, c.Params, ops),
                };

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
                        // A fill an op materialized on the operand (its getter's default) is the operand's; a result
                        // sharing it must not release it.
                        if (m._fill_value is not null)
                            keep.Add(m._fill_value);
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
                        // A damaged operand array ends the case (see SweepOrdinaryFile).
                        if (acc.CheckOwnership(c.Op, ledger, c.Id, file, "error path"))
                            continue;
                        var errTraffic = ScopeAudit.MeasureConfirmedTraffic(ErrorRegion);
                        acc.CheckOwnership(c.Op, ledger, c.Id, file, "error path, measured");
                        if (errTraffic == null)
                        {
                            acc.Inconclusive(c.Op, errorPath: true);
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
                        acc.CheckOwnership(c.Op, ledger, c.Id, file, "threw");
                        continue;
                    }

                    if (acc.CheckOwnership(c.Op, ledger, c.Id, file, "warm"))
                        continue;

                    // A mutator's measured runs start from the state its warm run left, not the case's: the warm ma.put of
                    // masked values into a nomask hard-masked target creates a mask, so the next run takes NumPy's
                    // hard-mask branch and raises its ValueError. The measured region then audits that error path's pool
                    // traffic instead of crashing the sweep (the ma_put / ma_select tiers gate the outcome itself).
                    void MutatorRegion()
                    {
                        try
                        {
                            Region();
                        }
                        catch when (MaskedMutators.Contains(c.Op))
                        {
                            // expected: see above
                        }
                    }

                    var traffic = ScopeAudit.MeasureConfirmedTraffic(MutatorRegion);
                    acc.CheckOwnership(c.Op, ledger, c.Id, file, "measured");
                    if (traffic == null)
                    {
                        acc.Inconclusive(c.Op, errorPath: false);
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
                        if (m._fill_value is not null)
                            built.Add(m._fill_value);
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
                // What the indexer does not own: the target (a setter writes its values, never its lifetime), the
                // assigned value and every index array.
                var owned = new List<(NDArray, string)> { (target, "target") };
                if (value is not null)
                    owned.Add((value, "value"));
                for (int t = 0; t < idx.Length; t++)
                    if (idx[t] is NDArray ind)
                        owned.Add((ind, $"index {t}"));
                var ledger = OwnershipLedger.Of(owned);

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
                        // A damaged target, value or index ends the case (see SweepOrdinaryFile).
                        if (acc.CheckOwnership(key, ledger, id, file, "error path"))
                            continue;
                        var errTraffic = ScopeAudit.MeasureConfirmedTraffic(ErrorRegion);
                        acc.CheckOwnership(key, ledger, id, file, "error path, measured");
                        if (errTraffic == null)
                        {
                            acc.Inconclusive(key, errorPath: true);
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
                        acc.CheckOwnership(key, ledger, id, file, "threw");
                        continue;
                    }

                    if (acc.CheckOwnership(key, ledger, id, file, "warm"))
                        continue;

                    var traffic = ScopeAudit.MeasureConfirmedTraffic(Region);
                    acc.CheckOwnership(key, ledger, id, file, "measured");
                    if (traffic == null)
                    {
                        acc.Inconclusive(key, errorPath: false);
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
        ///     array tuples, <see cref="NDMaskedArray"/> results (their data, mask AND stored fill), masked tuples,
        ///     C# tuples of any of those, and <see cref="IDisposable"/> result objects (iterators) —
        ///     skipping anything in <paramref name="keep"/>, the <see cref="NDMaskedConstant"/>
        ///     singleton and every process-wide shared array (<see cref="SharedArrays"/>). Non-array results
        ///     (scalars, strings, dtypes) carry nothing and are ignored.
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
            // The process-wide shared arrays (np.ma's singletons, the numpy.polynomial constants) are never a result's to
            // dispose, whatever the caller's keep set lists: no caller owns them.
            foreach (var o in CollectDisposables(result))
                if (!keep.Contains(o) && !SharedArrays.Contains(o))
                    ((IDisposable)o).Dispose();
        }

        /// <summary>
        ///     Every distinct disposable object a result carries — each <see cref="NDArray"/> (bare, in
        ///     array/tuple/<see cref="object"/>[] containers, or a <see cref="NDMaskedArray"/>'s data, mask
        ///     and stored fill) and each other <see cref="IDisposable"/> result object (iterators, archives) — in
        ///     first-visit order, WITHOUT disposing anything. The single walk every result consumer
        ///     shares (<see cref="DisposeAny"/>, the catalogue's freshness check, the read gate's
        ///     stable-part detection), so "what a result holds" means the same thing to all of them.
        /// </summary>
        /// <remarks>
        ///     A lazy <see cref="IEnumerable{NDArray}"/> is enumerated exactly ONCE, here: a caller that
        ///     walked it again would get a second, different batch of arrays (the first batch then
        ///     unaccounted for), which is why consumers iterate this materialized list instead of
        ///     re-walking the result. Only NDArray-typed sequences are enumerated at all — walking an
        ///     ARBITRARY <see cref="System.Collections.IEnumerable"/> could consume a stateful iterator
        ///     result (np.ndindex, ndenumerate) or run user code, which inspecting a result must never do.
        ///     The <see cref="NDMaskedConstant"/> singleton and strings carry nothing.
        /// </remarks>
        /// <param name="result">The result object, of any shape (null yields nothing).</param>
        /// <returns>The distinct disposables, compared by reference.</returns>
        internal static List<object> CollectDisposables(object result)
        {
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
            var found = new List<object>();

            void One(NDArray a)
            {
                if (a is not null && seen.Add(a))
                    found.Add(a);
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
                        // The stored fill is a pooled 0-d array the masked array owns (shared with the arrays derived
                        // from it, which a keep set protects), part of the result like its data and mask.
                        One(m._fill_value);
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
                        // A materialized collection of arrays (List<NDArray>, a LINQ-free array view) —
                        // enumerated once, here (see remarks).
                        foreach (var a in nds)
                            One(a);
                        return;
                    case object[] objs:
                        foreach (var item in objs)
                            Visit(item, depth + 1);
                        return;
                    case IDisposable d:
                        if (seen.Add(d))
                            found.Add(d);
                        return;
                }
            }

            Visit(result, 0);
            return found;
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
