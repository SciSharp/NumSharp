using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The reflective READ gate: every property and field of the NumPy-facing surface
    ///     (<see cref="LeakSurface"/>, <see cref="SurfaceKind.Property"/>/<see cref="SurfaceKind.Field"/>) is read
    ///     — and every settable property round-trip WRITTEN with the value just read — on real targets, under
    ///     the sweep's measurement protocol. A getter is code like any other: <c>ndarray.flat</c> copies a
    ///     non-contiguous array, <c>ndarray.imag</c> of a real array allocates zeros, iterator properties hand
    ///     out views, <c>poly1d.roots</c> runs an eigensolver. None of that is reachable from the corpus, which
    ///     drives functions.
    /// </summary>
    /// <remarks>
    ///     <para><b>The target is built INSIDE the measured region.</b> Each execution constructs a fresh
    ///     target (array, masked array, iterator, archive, …), reads, and disposes the target again, so the
    ///     region's balance covers the target's whole lifecycle. That is what catches a getter that CACHES a
    ///     fresh allocation on its owner: measured against a long-lived target, only the first read allocates
    ///     (and it would be un-measured warm-up); measured over a lifecycle, the cache must be released when
    ///     the owner is, or the region escapes it. Targets that are process singletons (the module objects,
    ///     static members) are simply shared.</para>
    ///     <para><b>Owner-held vs fresh parts.</b> Each region reads the member TWICE (writing the first
    ///     value back in between for a settable property). An object both reads return by reference is held
    ///     by the target (a cached view, the target itself, a stored value) or is a process singleton — it is
    ///     never disposed by the harness (the target's own disposal releases what it holds). Every other
    ///     array/disposable either read returned is FRESH and is disposed, exactly as a caller would.</para>
    ///     <para><b>Error paths.</b> A getter that throws on a target (e.g. <c>mT</c> of a 1-D array,
    ///     <c>multi_index</c> without the flag) is measured as an error path — it must throw without stranding
    ///     a buffer. A read that needs a LAPACK backend (<see cref="MissingBackendException"/>) is re-measured
    ///     with OpenBLAS installed; on a host where none loads it is skipped and reported.</para>
    /// </remarks>
    public partial class UndisposedIntermediateTests
    {
        /// <summary>
        ///     The one read-gate run of the process, shared by the gate test and the completeness gate
        ///     (which credits a property/field only when this run actually measured it).
        /// </summary>
        internal static readonly Lazy<DirectRunResult> SharedPropertyReads =
            new(RunPropertyReads, System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        ///     The read gate: every surface property/field is read (and settable ones written back) on its
        ///     targets without stranding a buffer; unconstructible targets and unresolvable accessors are
        ///     harness defects and fail.
        /// </summary>
        /// <exception cref="AssertFailedException">A read could not be set up, or an escape family is unclassified.</exception>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void EveryPropertyAndField_Read_LeavesNoUndisposedIntermediates()
        {
            var run = SharedPropertyReads.Value;
            var r = run.Sweep;

            // Members that were ONLY ever measured throwing: legitimate for by-design throwers
            // (np.complex64, np.chars), but a member landing here because no target makes it succeed
            // would have its success path unaudited — listed so the target plan can be widened.
            var errorOnly = r.ErrorMeasuredByOp.Keys.Where(k => r.MeasuredByOp.GetValueOrDefault(k) == 0)
                             .OrderBy(k => k, StringComparer.Ordinal).ToList();
            Console.WriteLine($"[scope-audit/properties] measured={r.DirectMeasured} errorPaths={r.ErrorPathsMeasured} " +
                              $"members={r.MeasuredByOp.Keys.Union(r.ErrorMeasuredByOp.Keys).Count()} " +
                              $"gcInconclusive={r.GcInconclusive} harnessErrors={run.HarnessErrors.Count}");
            if (errorOnly.Count > 0)
                Console.WriteLine($"[scope-audit/properties] {errorOnly.Count} members measured on error paths only: " +
                                  string.Join(", ", errorOnly));
            if (run.BackendSkipReason != null)
                Console.WriteLine($"[scope-audit/properties] {run.BackendSkipped.Count} backend-only reads skipped: {run.BackendSkipReason}");
            PrintPerOpRollup(r.Groups);
            PrintBypassRollup(r.Bypasses);

            if (run.HarnessErrors.Count > 0)
                Assert.Fail($"{run.HarnessErrors.Count} property/field reads could not be set up:\n  " +
                            string.Join("\n  ", run.HarnessErrors));

            AssertNoUnclassifiedEscapes(r, "scope-audit/properties");
        }

        /// <summary>
        ///     One way to obtain an object a surface member is read on. <see cref="Make"/> runs INSIDE the
        ///     measured region (see the class remarks), so it must build a complete, balanced lifecycle:
        ///     every array it allocates is released by the returned cleanup.
        /// </summary>
        /// <param name="Label">The target's label (the escape family's layout, and the diagnostic sample id).</param>
        /// <param name="Make">Builds the target and returns it with the cleanup that releases everything
        /// the build allocated (a no-op for shared singletons and managed-only objects).</param>
        private sealed record ReadTarget(string Label, Func<LeakFixture, (object Target, Action Cleanup)> Make);

        /// <summary>The cleanup of a target that allocated nothing (a singleton, a managed-only object).</summary>
        private static readonly Action Noop = () => { };

        /// <summary>
        ///     Runs every surface property/field read: the managed pass, then the reads that needed a
        ///     matrix backend again with OpenBLAS installed.
        /// </summary>
        /// <returns>The run's observations.</returns>
        private static DirectRunResult RunPropertyReads()
        {
            var acc = new SweepAccumulator();
            var errors = new List<string>();
            var skipped = new HashSet<string>(StringComparer.Ordinal);
            var backendRetry = new List<(SurfaceMember member, ReadTarget target)>();
            string backendSkip = null;

            using var fx = new LeakFixture();
            ScopeAudit.Settle();

            foreach (var member in LeakSurface.Enumerate().Where(m => m.Kind is SurfaceKind.Property or SurfaceKind.Field))
            {
                var targets = ReadTargetsFor(member);
                if (targets == null)
                {
                    errors.Add($"{member.Id}: no read target is defined for instance members of owner '{member.Owner}' " +
                               "(add one to ReadTargetsFor)");
                    continue;
                }
                foreach (var target in targets)
                    MeasureRead(member, target, fx, acc, errors, backendRetry);
            }

            if (backendRetry.Count > 0)
            {
                backendSkip = TryEnableLeakBackend();
                if (backendSkip == null)
                {
                    try
                    {
                        foreach (var (member, target) in backendRetry)
                            MeasureRead(member, target, fx, acc, errors, backendRetry: null);
                    }
                    finally
                    {
                        OpenBlasEngine.Disable();
                    }
                }
                else
                {
                    foreach (var (member, _) in backendRetry)
                        skipped.Add(member.Id);
                }
            }

            return new DirectRunResult(acc.ToResult(0), errors, backendSkip, skipped);
        }

        /// <summary>
        ///     Measures one (member, target) read. The warm execution decides the path: success, error
        ///     (the read throws — measured with the exception swallowed), or backend-deferred (the read
        ///     threw <see cref="MissingBackendException"/> and a backend pass follows).
        /// </summary>
        /// <param name="member">The property/field.</param>
        /// <param name="target">The target to read it on.</param>
        /// <param name="fx">The shared fixture.</param>
        /// <param name="acc">The run's tallies.</param>
        /// <param name="errors">Where set-up defects (unconstructible target, unresolvable accessor) are reported.</param>
        /// <param name="backendRetry">Where a backend-needing read is deferred; null during the backend pass
        /// itself, where a backend miss is simply an error path.</param>
        private static void MeasureRead(SurfaceMember member, ReadTarget target, LeakFixture fx, SweepAccumulator acc,
                                        List<string> errors, List<(SurfaceMember, ReadTarget)> backendRetry)
        {
            // Set-up validation, OUTSIDE any measurement: a target that cannot be built, or an accessor that
            // cannot be resolved against it, is a harness defect — never a property error path.
            Func<object, object> get;
            Action<object, object> set;
            try
            {
                var (t, cleanup) = target.Make(fx);
                try
                {
                    (get, set) = ResolveAccessor(member, t);
                }
                finally
                {
                    cleanup();
                }
            }
            catch (Exception e)
            {
                errors.Add($"{member.Id} @ {target.Label}: set-up failed — {e.GetType().Name}: {FirstLine(e.Message)}");
                return;
            }

            void Body()
            {
                var (t, cleanup) = target.Make(fx);
                object v1 = null, v2 = null;
                bool haveSecond = false;
                try
                {
                    v1 = get(t);
                    if (set != null)
                        set(t, v1);   // round-trip write: exercises the setter with a value it must accept
                    v2 = get(t);
                    haveSecond = true;
                }
                finally
                {
                    // Runs on the error path too: a setter that REJECTS the value (imag of a real array)
                    // throws after the getter already allocated — the harness still owns that fresh read and
                    // must release it, or the exception itself would read as an escape. Without a second
                    // read, one more (possibly throwing) read supplies the owner-held comparison.
                    if (!haveSecond)
                    {
                        try
                        {
                            v2 = get(t);
                        }
                        catch
                        {
                            v2 = null;   // the getter itself throws: nothing further to compare
                        }
                    }
                    // The target's own arrays are never a read's fresh part (a getter may return the target).
                    var keep = new HashSet<object>(fx.Keep, ReferenceEqualityComparer.Instance);
                    foreach (var own in CollectDisposables(t))
                        keep.Add(own);
                    DisposeFreshReadParts(v1, v2, keep);
                    cleanup();
                }
            }

            bool errorPath = false;
            try
            {
                Body();   // warm + path decision
            }
            catch (Exception e)
            {
                if (e is MissingBackendException && backendRetry != null)
                {
                    backendRetry.Add((member, target));
                    return;
                }
                errorPath = true;
            }

            void ErrorBody()
            {
                try
                {
                    Body();
                }
                catch
                {
                    // expected: this read throws on this target — measured as an error path
                }
            }

            var traffic = ScopeAudit.MeasureConfirmedTraffic(errorPath ? ErrorBody : Body);
            if (traffic == null)
            {
                acc.GcInconclusive++;
                return;
            }
            if (!errorPath)
                acc.Direct++;
            acc.Record(member.Id, null, target.Label, traffic.Value, 0, errorPath, $"{member.Id}@{target.Label}", "property-read");
        }

        /// <summary>
        ///     Disposes the FRESH parts of a read: every array/disposable one read returned that the other
        ///     did not (by reference), except fixture arrays and process singletons. Parts both reads
        ///     returned are held by the target (or are singletons) and are left for the target's cleanup.
        /// </summary>
        /// <param name="v1">The first read's value.</param>
        /// <param name="v2">The second read's value (after the round-trip write, when settable).</param>
        /// <param name="keep">The fixture's never-dispose set (includes the np.ma singletons).</param>
        private static void DisposeFreshReadParts(object v1, object v2, HashSet<object> keep)
        {
            // Materialize each read's parts once (a lazy sequence must not be enumerated twice — see
            // CollectDisposables), then classify by reference.
            var r1 = CollectDisposables(v1);
            var r2 = CollectDisposables(v2);
            var held = new HashSet<object>(r1, ReferenceEqualityComparer.Instance);
            held.IntersectWith(r2);

            var done = new HashSet<object>(ReferenceEqualityComparer.Instance);
            foreach (var o in r1.Concat(r2))
                if (done.Add(o) && !held.Contains(o) && !keep.Contains(o))
                    ((IDisposable)o).Dispose();
        }

        /// <summary>
        ///     Resolves a member's reflective getter (and setter, for a settable property) against a concrete
        ///     target. Open generic owners (<c>NDArray&lt;T&gt;</c>) are re-resolved on the target's closed
        ///     runtime type; invocation exceptions are unwrapped so the real exception type decides the path.
        /// </summary>
        /// <param name="member">The property/field.</param>
        /// <param name="target">The target (null for static members).</param>
        /// <returns>The getter, and the setter or null.</returns>
        /// <exception cref="MissingMemberException">The member cannot be found on the target's type.</exception>
        private static (Func<object, object> get, Action<object, object> set) ResolveAccessor(SurfaceMember member, object target)
        {
            var info = member.Members[0];
            if (info.DeclaringType is { IsGenericTypeDefinition: true } && target != null)
            {
                // Re-resolve on the closed runtime type: an open generic's members cannot be invoked.
                const BindingFlags Declared = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
                var closed = target.GetType();
                while (closed != null && !(closed.IsGenericType && closed.GetGenericTypeDefinition() == info.DeclaringType))
                    closed = closed.BaseType;
                info = (MemberInfo)closed?.GetProperty(member.Name, Declared) ?? closed?.GetField(member.Name, Declared)
                       ?? throw new MissingMemberException(member.OwnerType.Name, member.Name);
            }

            switch (info)
            {
                case PropertyInfo p:
                {
                    // A hiding `new` property yields several same-name members; the most-derived one is what
                    // a C# caller binds.
                    if (member.Members.Length > 1 && target != null)
                        p = member.Members.OfType<PropertyInfo>()
                                  .Where(x => x.DeclaringType!.IsAssignableFrom(target.GetType()))
                                  .OrderByDescending(x => Depth(x.DeclaringType)).FirstOrDefault() ?? p;
                    var getter = p.GetGetMethod() ?? throw new MissingMemberException(member.OwnerType.Name, member.Name + " (no public getter)");
                    var setter = p.GetSetMethod();
                    return (t => Invoke(() => getter.Invoke(t, null)),
                            setter == null ? null : (t, v) => Invoke(() => setter.Invoke(t, new[] { v })));
                }
                case FieldInfo f:
                    return (t => f.GetValue(t), null);
                default:
                    throw new MissingMemberException(member.OwnerType.Name, member.Name);
            }
        }

        /// <summary>Inheritance depth of a type (used to pick the most-derived hiding property).</summary>
        /// <param name="t">The type.</param>
        /// <returns>The number of base types above it.</returns>
        private static int Depth(Type t)
        {
            int d = 0;
            for (var b = t.BaseType; b != null; b = b.BaseType)
                d++;
            return d;
        }

        /// <summary>Invokes a reflective call and rethrows its REAL exception (not the
        /// <see cref="TargetInvocationException"/> wrapper) with the original stack.</summary>
        /// <param name="call">The reflective call.</param>
        /// <returns>The call's return value.</returns>
        private static object Invoke(Func<object> call)
        {
            try
            {
                return call();
            }
            catch (TargetInvocationException e) when (e.InnerException != null)
            {
                ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                throw;   // unreachable: Throw() never returns
            }
        }

        /// <summary>
        ///     The targets a member is read on. Static members read on null; module instance members on the
        ///     module singleton (or a fresh instance where the member MUTATES state — the RandomState seed);
        ///     object members on fresh instances in the layouts/states that drive their distinct code paths.
        /// </summary>
        /// <param name="m">The surface member.</param>
        /// <returns>The targets, or null when no target is defined for the owner (a harness defect).</returns>
        private static IReadOnlyList<ReadTarget> ReadTargetsFor(SurfaceMember m)
        {
            bool isStatic = m.Members[0] switch
            {
                PropertyInfo p => (p.GetGetMethod() ?? p.GetSetMethod())!.IsStatic,
                FieldInfo f => f.IsStatic,
                _ => false,
            };
            if (isStatic)
                return StaticTarget;

            return m.Owner switch
            {
                "ndarray" => NdarrayTargets,
                // A fresh RandomState, not the global singleton: the settable Seed REseeds its instance.
                "np.random" => new[] { new ReadTarget("RandomState(1234)", _ => (np.random.RandomState(1234), Noop)) },
                "np.ma" => new[] { new ReadTarget("np.ma", _ => (np.ma, Noop)) },
                "np.fft" => new[] { new ReadTarget("np.fft", _ => (np.fft, Noop)) },
                "Generator" => new[] { new ReadTarget("default_rng(7)", _ => (np.random.default_rng(7), Noop)) },
                "SeedSequence" => new[] { new ReadTarget("SeedSequence(5)", _ => (new SeedSequence(5), Noop)) },
                "BitGenerator" or "PCG64" => new[] { new ReadTarget("PCG64(5)", _ => (new PCG64(5), Noop)) },
                "MT19937" => new[] { new ReadTarget("MT19937(3)", _ => (new MT19937(3), Noop)) },
                "NDIterator" => NDIteratorTargets,
                "FlatIterator" => new[]
                {
                    new ReadTarget("flatiter of a transposed view", f => (f.MT.flatiter, Noop)),
                    new ReadTarget("flatiter advanced 5", f =>
                    {
                        var it = f.MT.flatiter;
                        for (int i = 0; i < 5; i++)
                            it.next();
                        return (it, Noop);
                    }),
                },
                "NDIndex" => new[]
                {
                    new ReadTarget("ndindex(3,4) fresh", _ => Disposable(np.ndindex(3, 4))),
                    new ReadTarget("ndindex(3,4) advanced", _ =>
                    {
                        var it = np.ndindex(3, 4);
                        it.MoveNext();
                        return Disposable(it);
                    }),
                },
                "NDEnumerate" => new[]
                {
                    new ReadTarget("ndenumerate fresh", f => Disposable(np.ndenumerate(f.MT))),
                    new ReadTarget("ndenumerate advanced", f =>
                    {
                        var it = np.ndenumerate(f.MT);
                        it.MoveNext();
                        return Disposable(it);
                    }),
                },
                "Broadcast" => new[]
                {
                    new ReadTarget("broadcast fresh", f => Disposable(np.broadcast(f.M, f.Zero))),
                    new ReadTarget("broadcast advanced", f =>
                    {
                        var b = np.broadcast(f.M, f.Zero);
                        b.MoveNext();
                        return Disposable(b);
                    }),
                },
                "DType" => new[]
                {
                    new ReadTarget("float64", _ => (np.float64, Noop)),
                    new ReadTarget("complex128", _ => (np.complex128, Noop)),
                    new ReadTarget(">i4", _ => (DType.From(">i4"), Noop)),
                    new ReadTarget("M8[ns]", _ => (DType.From("M8[ns]"), Noop)),
                },
                "finfo" => new[] { new ReadTarget("float32", _ => (np.finfo(np.float32), Noop)) },
                "iinfo" => new[] { new ReadTarget("int16", _ => (np.iinfo(np.int16), Noop)) },
                "NDArrayFlags" => new[]
                {
                    new ReadTarget("flags of an owning array", _ =>
                    {
                        var x = np.ones(new Shape(3, 4));
                        return (x.flags, x.Dispose);
                    }),
                    new ReadTarget("flags of a transposed view", _ =>
                    {
                        var x = np.ones(new Shape(3, 4));
                        var t = x.T;
                        return (t.flags, () => { t.Dispose(); x.Dispose(); });
                    }),
                },
                "poly1d" => new[]
                {
                    new ReadTarget("poly1d([1,-3,2])", _ =>
                    {
                        var c = np.array(new double[] { 1, -3, 2 });
                        var p = new poly1d(c);
                        return (p, () => { p.Dispose(); c.Dispose(); });
                    }),
                },
                "NpzFile" => new[] { new ReadTarget("load_npz(bytes)", f => Disposable(np.load_npz(f.NpzBytes))) },
                "NDMaskedArray" => MaskedTargets,
                "NDArray<T>" => new[]
                {
                    new ReadTarget("NDArray<double> (3,4)", _ =>
                    {
                        var x = np.ones(new Shape(3, 4));
                        var g = x.MakeGeneric<double>();
                        return (g, () => { g.Dispose(); x.Dispose(); });
                    }),
                },
                _ => null,
            };
        }

        /// <summary>The single target of a static member: no instance.</summary>
        private static readonly IReadOnlyList<ReadTarget> StaticTarget = new[] { new ReadTarget("static", _ => (null, Noop)) };

        /// <summary>
        ///     ndarray targets: an owning C-contiguous float64 matrix, a transposed (non-contiguous) VIEW
        ///     — <c>flat</c> copies there — a complex128 matrix (the real/imag setters' success path) and a
        ///     1-D int32 vector (the rank-guarded members' error paths, the integer layout).
        /// </summary>
        private static readonly IReadOnlyList<ReadTarget> NdarrayTargets = new[]
        {
            new ReadTarget("float64 (3,4) owning", _ =>
            {
                var x = np.ones(new Shape(3, 4));
                return (x, x.Dispose);
            }),
            new ReadTarget("float64 transposed view", _ =>
            {
                var x = np.ones(new Shape(3, 4));
                var t = x.T;
                return (t, () => { t.Dispose(); x.Dispose(); });
            }),
            new ReadTarget("complex128 (2,3) owning", _ =>
            {
                var x = np.ones(new Shape(2, 3), np.complex128);
                return (x, x.Dispose);
            }),
            new ReadTarget("int32 (5,) owning", _ =>
            {
                var x = np.ones(new Shape(5), np.int32);
                return (x, x.Dispose);
            }),
        };

        /// <summary>
        ///     NDIterator targets: plain, multi_index-tracking, c_index-tracking, ranged (the iterrange
        ///     setter's success path), and advanced one step (Current/value mid-walk). Built over fixture
        ///     operands (never disposed); the cleanup closes the iterator's unmanaged state.
        /// </summary>
        private static readonly IReadOnlyList<ReadTarget> NDIteratorTargets = new[]
        {
            new ReadTarget("nditer plain", f => Disposable(np.nditer(f.M))),
            new ReadTarget("nditer multi_index", f => Disposable(np.nditer(f.M, new[] { "multi_index" }))),
            new ReadTarget("nditer c_index", f => Disposable(np.nditer(f.MT, new[] { "c_index" }))),
            new ReadTarget("nditer ranged", f => Disposable(np.nditer(f.M, new[] { "ranged" }))),
            new ReadTarget("nditer advanced", f =>
            {
                var it = np.nditer(f.M);
                it.MoveNext();
                return Disposable(it);
            }),
        };

        /// <summary>
        ///     NDMaskedArray targets: a real mask and nomask. Both own fresh data (and mask) arrays the
        ///     cleanup releases; a masked array does not own its arrays, so the cleanup disposes them
        ///     explicitly, skipping the np.ma singletons a nomask array references.
        /// </summary>
        private static readonly IReadOnlyList<ReadTarget> MaskedTargets = new[]
        {
            new ReadTarget("masked (3,4)", _ =>
            {
                var d = np.ones(new Shape(3, 4));
                var k = np.zeros(new Shape(3, 4), np.bool_);
                k.SetBoolean(true, 0, 1);
                var m = np.ma.masked_array(d, k);
                return (m, () => DisposeMaskedTarget(m, d, k));
            }),
            new ReadTarget("nomask (3,4)", _ =>
            {
                var d = np.ones(new Shape(3, 4));
                var m = np.ma.masked_array(d);
                return (m, () => DisposeMaskedTarget(m, d, null));
            }),
        };

        /// <summary>Releases a masked target's arrays — its data/mask wrappers and the inputs they were
        /// built from — once each, never an np.ma process singleton.</summary>
        /// <param name="m">The masked target.</param>
        /// <param name="data">The data array it was built from.</param>
        /// <param name="mask">The mask array it was built from, or null.</param>
        private static void DisposeMaskedTarget(NDMaskedArray m, NDArray data, NDArray mask)
        {
            var singletons = new HashSet<object>(MaskedSingletons(), ReferenceEqualityComparer.Instance);
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
            foreach (var a in new[] { m._data, m._mask, data, mask })
                if (a is not null && seen.Add(a) && !singletons.Contains(a))
                    a.Dispose();
        }

        /// <summary>Pairs a disposable target with its own disposal as the cleanup.</summary>
        /// <param name="target">The target object.</param>
        /// <returns>The target and the cleanup (a no-op when it is not <see cref="IDisposable"/>).</returns>
        private static (object, Action) Disposable(object target)
            => (target, target is IDisposable d ? d.Dispose : Noop);
    }
}
