using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The lifetime and garbage-collection safety of the ONNX Runtime bridge (the analysis is
    ///     <c>docs/plans/interop-lifetime-gc-analysis.md</c>). Three properties, each pinned here:
    ///     <list type="bullet">
    ///         <item><b>Finalizers never free under a reachable alias, in any order.</b> A bridge object's finalizer gives its
    ///         reference back with <c>Abandon</c> (no eager free), so a buffer an alias still reaches stays valid; every
    ///         finalization order of every scenario is run by hand (<see cref="LifetimeHarness"/>).</item>
    ///         <item><b>Reachable means valid.</b> What a caller can still reach — the handle's <c>OrtValue</c>, its
    ///         <c>DenseTensor</c>, its owned conversion, a pin — is never disposed or freed by a collection alone, and the
    ///         buffer's reference count keeps telling the truth about it (<c>resize(refcheck)</c> trusts it).</item>
    ///         <item><b>No read outlives its owner's root.</b> A copy that reads memory through a raw pointer keeps the object
    ///         owning that memory reachable until the read is done.</item>
    ///     </list>
    /// </summary>
    [TestClass]
    public class LifetimeGcTests : OnnxTestBase
    {
        /// <summary>The element count of every buffer the scenarios share (float64): small, and one pool bucket.</summary>
        private const int N = 4096;

        /// <summary>One finalization scenario: the objects that become unreachable together, and an alias the caller keeps.</summary>
        private sealed class Scenario
        {
            /// <summary>A reachable alias of the buffer that holds NO counted reference (a bare slice, as <c>GetData()</c> hands out).</summary>
            internal IArraySlice Alias;

            /// <summary>The objects that become unreachable together; only those with a finalizer of their own are run.</summary>
            internal object[] Participants;
        }

        /// <summary>The scenarios, each built fresh for every order it is run in.</summary>
        private static readonly (string Name, Func<Scenario> Build)[] Scenarios =
        {
            ("AsOrtValue", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, N);
                OrtTensor h = src.AsOrtValue();
                return new Scenario { Alias = src.Storage.InternalArray, Participants = new object[] { src, h, h.Value } };
            }),
            ("AsOrtValue of Decimal (the handle owns the float64 conversion)", () =>
            {
                NDArray dec = Arange(NPTypeCode.Decimal, N);
                OrtTensor h = dec.AsOrtValue();
                NDArray converted = h.Source;
                return new Scenario { Alias = converted.Storage.InternalArray, Participants = new object[] { dec, converted, h, h.Value } };
            }),
            ("AsDenseTensor", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, N);
                OrtTensor<double> h = src.AsDenseTensor<double>();
                return new Scenario { Alias = src.Storage.InternalArray, Participants = new object[] { src, h } };
            }),
            ("AsDenseTensor of Decimal (the handle owns the float64 conversion)", () =>
            {
                NDArray dec = Arange(NPTypeCode.Decimal, N);
                OrtTensor<double> h = dec.AsDenseTensor<double>();
                NDArray converted = h.Source;
                return new Scenario { Alias = converted.Storage.InternalArray, Participants = new object[] { dec, converted, h } };
            }),
            ("AsDenseTensor, viewed back with Tensor.AsNDArray()", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, N);
                OrtTensor<double> h = src.AsDenseTensor<double>();
                NDArray view = h.Tensor.AsNDArray();
                return new Scenario
                {
                    Alias = src.Storage.InternalArray,
                    Participants = new object[] { src, h, view, LifetimeHarness.DisposerOf(view.Storage.InternalArray) },
                };
            }),
            ("AsDenseTensor, wrapped by an OrtValue over its Memory that is never disposed", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, N);
                OrtTensor<double> h = src.AsDenseTensor<double>();
                OrtValue value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, h.Memory, new long[] { N });
                return new Scenario { Alias = src.Storage.InternalArray, Participants = new[] { src, h, value, PinnableOf(value) } };
            }),
            ("AsDenseTensor, with a pin of its Memory dropped undisposed", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, N);
                OrtTensor<double> h = src.AsDenseTensor<double>();
                MemoryHandle pin = h.Memory.Pin();
                return new Scenario { Alias = src.Storage.InternalArray, Participants = new[] { src, h, PinnableOf(pin) } };
            }),
            // The one route by which a FINALIZER disposes a pin deterministically: an owning import view's memory block,
            // finalized, runs its lease, which disposes the OrtValue it owns with ORT's Dispose(true), which disposes the pin
            // the value holds. The pin's release must not claim the eager free there, whatever it would do on a live dispose.
            ("AsDenseTensor's Memory wrapped by an OrtValue that an owning view (AsNDArray(ownsValue: true)) disposes", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, N);
                OrtTensor<double> h = src.AsDenseTensor<double>();
                OrtValue value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, h.Memory, new long[] { N });
                h.Dispose();   // the value's pin outlives the handle (documented): it is now one of the buffer's two references
                NDArray view = value.AsNDArray(ownsValue: true);
                return new Scenario
                {
                    Alias = src.Storage.InternalArray,
                    Participants = new[] { src, view, LifetimeHarness.DisposerOf(view.Storage.InternalArray), value, PinnableOf(value) },
                };
            }),
        };

        /// <summary>
        ///     Every scenario, finalized in every order of its participants: a buffer a caller still reaches through an
        ///     uncounted alias must keep its values, must not be freed, and must end with every counted reference given back
        ///     exactly once (count 0, not -1 = freed, not &gt; 0 = leaked); the export / import counters return to where they
        ///     were.
        /// </summary>
        /// <remarks>
        ///     <para>Each order is one REAL collection of the scenario's objects (<see cref="LifetimeHarness.StageCollection"/>):
        ///     they are made unreachable, the collection clears their weak handles, and their finalizers run on the finalizer
        ///     thread in the chosen order.</para>
        ///     <para>Before the fix every bridge finalizer called <c>Release</c>, whose decrement to zero frees the buffer at
        ///     once: whenever the source array's finalizer (an <c>Abandon</c>) ran first, the bridge's finalizer freed a buffer
        ///     the alias still read, and the next allocation of that size overwrote it. A real collection runs only the order the
        ///     runtime picks; the probe saw the failing order for both handles, the passing one for the ML.NET view. Two
        ///     scenarios failed in every order the other way: a pin ORT never disposes (its <c>OrtValue</c> finalizer skips
        ///     it) kept its reference forever, so the count stayed at 1. The import-back scenario failed in 10 of its 120
        ///     orders: a finalizer disposed the pin through ORT's <c>Dispose(true)</c>, and the pin's release freed the buffer.</para>
        /// </remarks>
        [TestMethod]
        public void Finalizers_InEveryOrder_NeverFreeUnderAReachableAlias_AndGiveEachReferenceBackOnce()
        {
            var failures = new List<string>();
            int runs = 0, broken = 0;
            foreach ((string name, Func<Scenario> build) in Scenarios)
            {
                int participantCount = CountFinalizable(build);
                foreach (int[] order in LifetimeHarness.Permutations(participantCount))
                {
                    runs++;
                    int exports = NDArrayOnnxInterop.LiveExports, imports = NDArrayOnnxInterop.LiveImports;
                    (IArraySlice alias, string orderText, LifetimeHarness.FinalizationReport report) = StageScenario(build, order);
                    LifetimeHarness.RunStagedCollection(report);

                    List<NDArray> poison = LifetimeHarness.Poison(N, 777.0);
                    double[] seen = LifetimeHarness.ReadDoubles(alias, N);
                    LifetimeHarness.DisposeAll(poison);

                    int before = failures.Count;
                    long count = LifetimeHarness.RefCount(alias);
                    for (int i = 0; i < N; i++)
                    {
                        if (seen[i] != i)
                        {
                            failures.Add($"{name} [{orderText}]: the alias reads {seen[i]} at {i} — the buffer was freed under it");
                            break;
                        }
                    }

                    if (alias.IsReleased)
                        failures.Add($"{name} [{orderText}]: the buffer was freed while an alias still reaches it");
                    if (count != 0)
                        failures.Add($"{name} [{orderText}]: reference count {count} after every holder was finalized (0 expected: -1 = freed, > 0 = a reference never given back)");
                    if (NDArrayOnnxInterop.LiveExports != exports || NDArrayOnnxInterop.LiveImports != imports)
                        failures.Add($"{name} [{orderText}]: counters exports {exports} -> {NDArrayOnnxInterop.LiveExports}, imports {imports} -> {NDArrayOnnxInterop.LiveImports}");
                    if (failures.Count > before)
                        broken++;
                }
            }

            Assert.AreEqual(92 + 120, runs, "the scenarios' orders were not all run");
            if (failures.Count > 0)
                Assert.Fail($"{broken} of {runs} finalization orders broke the lifetime contract:\n" + string.Join("\n", failures.Take(40)));
        }

        /// <summary>
        ///     A handle the caller dropped while still holding its <see cref="OrtTensor.Value"/>: after a collection the value
        ///     is neither disposed nor pointing at freed memory, and the buffer's count still includes the handle's reference
        ///     (so <c>resize(refcheck: true)</c> on the source refuses to reallocate under the native tensor). Once the value
        ///     is dropped too, the pair is collected and the reference given back.
        /// </summary>
        [TestMethod]
        public void HeldOrtValue_KeepsItsHandle_SoACollectionNeitherDisposesItNorFreesItsMemory()
        {
            NDArray src = Arange(NPTypeCode.Double, N);
            int baseline = NDArrayOnnxInterop.LiveExports;
            WeakReference handle = HoldOnlyTheValueThroughACollection(src);

            // The value is out of scope now (the helper returned): handle and value must be collected together.
            Assert.IsTrue(GcQuiescence.WaitCollected(handle), "the handle was not collected once its OrtValue was dropped too");
            Assert.IsTrue(WaitFor(() => NDArrayOnnxInterop.LiveExports == baseline, 10_000), "the collected handle did not give its export back");
            src.resize(new Shape(2 * N));
            Assert.AreEqual(2L * N, src.size, "the reference the collected handle held was not given back");
            src.Dispose();
        }

        /// <summary>
        ///     A <see cref="OrtTensor{T}"/> handle the caller dropped while still holding its <see cref="OrtTensor{T}.Tensor"/>:
        ///     the tensor stays readable after a collection (it used to throw <see cref="ObjectDisposedException"/> because the
        ///     handle's finalizer disposed its memory manager) and keeps the buffer's count honest; dropped too, the handle is
        ///     collected and its reference given back.
        /// </summary>
        [TestMethod]
        public void HeldDenseTensor_KeepsItsHandle_SoACollectionNeitherDisablesItNorFreesItsMemory()
        {
            NDArray src = Arange(NPTypeCode.Double, N);
            int baseline = NDArrayOnnxInterop.LiveExports;
            WeakReference handle = HoldOnlyTheTensorThroughACollection(src);

            // The tensor is out of scope now (the helper returned): handle and tensor must be collected together.
            Assert.IsTrue(GcQuiescence.WaitCollected(handle), "the handle was not collected once its tensor was dropped too");
            Assert.IsTrue(WaitFor(() => NDArrayOnnxInterop.LiveExports == baseline, 10_000), "the collected handle did not give its export back");
            src.resize(new Shape(2 * N));
            src.Dispose();
        }

        /// <summary>
        ///     The float64 conversion a handle owns for a Decimal source is reachable through <see cref="OrtTensor.Source"/>;
        ///     a caller that keeps only it must find it undisposed and intact after the handle is collected (the handle's
        ///     finalizer used to dispose it, and the next allocation overwrote it).
        /// </summary>
        [TestMethod]
        public void HeldSource_OfACollectedHandle_IsNeitherDisposedNorFreed()
        {
            foreach (bool dense in new[] { false, true })
            {
                NDArray converted = ExportKeepingOnlyTheConversion(dense);
                GcQuiescence.CollectFull();
                GcQuiescence.CollectFull();
                List<NDArray> poison = LifetimeHarness.Poison(N, 777.0);
                Assert.IsFalse(converted.IsDisposed, $"{(dense ? "AsDenseTensor" : "AsOrtValue")}: the collected handle disposed the conversion the caller still holds");
                Assert.AreEqual(N - 1.0, converted.GetDouble(N - 1), $"{(dense ? "AsDenseTensor" : "AsOrtValue")}: the conversion reads freed memory");
                LifetimeHarness.DisposeAll(poison);
                converted.Dispose();
            }
        }

        /// <summary>
        ///     Pins that are never disposed give their reference back once collected: an <see cref="OrtValue"/> made over the
        ///     handle's Memory and dropped (ORT's own finalizer does not dispose the pin it holds) and a bare
        ///     <see cref="MemoryHandle"/> dropped. Before the fix each kept the buffer's count raised for the life of the
        ///     process, so <c>resize(refcheck: true)</c> refused forever.
        /// </summary>
        [TestMethod]
        public void DroppedPins_GiveTheirReferenceBack_WhenCollected()
        {
            using NDArray nd = Arange(NPTypeCode.Double, N);
            DropPinsUndisposed(nd);
            Assert.IsTrue(WaitFor(() => nd.Storage.InternalArray.IsUniquelyReferenced, 10_000),
                $"the dropped pins still hold references (count {LifetimeHarness.RefCount(nd.Storage.InternalArray)}, 1 expected)");
            nd.resize(new Shape(2 * N));
            Assert.AreEqual(2L * N, nd.size);
        }

        /// <summary>
        ///     <see cref="MemoryHandle"/> is a struct: disposing two copies of one pin must give its reference back ONCE. The
        ///     manager used to be every pin's <see cref="IPinnable"/>, so each copy released a reference — the second one
        ///     someone else's — and here the source array's own buffer was freed while the array was alive.
        /// </summary>
        [TestMethod]
        public void TwoCopiesOfOnePin_Disposed_GiveOneReferenceBack()
        {
            using NDArray nd = Arange(NPTypeCode.Double, N);
            OrtTensor<double> h = nd.AsDenseTensor<double>();
            MemoryHandle pin = h.Memory.Pin();
            MemoryHandle copy = pin;
            pin.Dispose();
            copy.Dispose();
            h.Dispose();

            Assert.IsFalse(nd.Storage.InternalArray.IsReleased, "a second dispose of a copied pin released the source array's own reference");
            Assert.AreEqual(1L, LifetimeHarness.RefCount(nd.Storage.InternalArray), "only the source array's reference should remain");
            List<NDArray> poison = LifetimeHarness.Poison(N, 777.0);
            Assert.AreEqual(N - 1.0, nd.GetDouble(N - 1), "the source array reads freed memory");
            LifetimeHarness.DisposeAll(poison);
        }

        /// <summary>
        ///     A NumSharp view of a tensor over NumSharp's own memory (<c>h.Tensor.AsNDArray()</c>) is an ordinary alias of the
        ///     buffer: it holds one counted reference on it (no import lease — the import counter does not move) and gives it
        ///     back on Dispose; its values and writes are the source's.
        /// </summary>
        [TestMethod]
        public void TensorView_OverNumSharpMemory_IsACountedAlias_NotAForeignLease()
        {
            using NDArray nd = Arange(NPTypeCode.Double, N);
            using OrtTensor<double> h = nd.AsDenseTensor<double>();
            int imports = NDArrayOnnxInterop.LiveImports;
            long before = LifetimeHarness.RefCount(nd.Storage.InternalArray);

            NDArray view = h.Tensor.AsNDArray();
            Assert.AreEqual(imports, NDArrayOnnxInterop.LiveImports, "NumSharp's own memory was leased like foreign memory");
            Assert.AreEqual(before + 1, LifetimeHarness.RefCount(nd.Storage.InternalArray), "the view does not hold a counted reference on the buffer");
            view.SetDouble(-5.0, 7);
            Assert.AreEqual(-5.0, nd.GetDouble(7), "the view does not share the buffer");
            view.Dispose();
            Assert.AreEqual(before, LifetimeHarness.RefCount(nd.Storage.InternalArray), "disposing the view did not give its reference back");
        }

        /// <summary>
        ///     <see cref="NDArrayOnnxInterop.ToNDArray(OrtValue)"/> reads ORT's memory through a raw span; the OrtValue owning it
        ///     must stay reachable until the copy is done. Under GC pressure, with a value nobody else holds (made just for the
        ///     call), the copy before the fix read freed memory — in the probe the process died within seconds of a fatal access
        ///     violation inside the copy. A regression here therefore ends the test host rather than failing one assertion.
        /// </summary>
        /// <remarks>
        ///     The values are 16 MB: freeing a block that size returns it to the operating system, so a read of it faults at
        ///     once instead of seeing stale bytes, and the copy lasts long enough for a collection to land in it. The first
        ///     version of this test used 128 KB values and could not fail: with the KeepAlive removed it passed 1,335 rounds,
        ///     where 16 MB values crashed the process in 3 of 3 three-second runs. The code is warmed up to tier 1 first (see
        ///     <see cref="UnderCollectionPressure"/>), the only code in which the window exists.
        /// </remarks>
        [TestMethod]
        public void InFlightCopy_OfAnOrtValueNobodyElseHolds_IsNeverFreedUnderTheRead()
        {
            if (!LifetimeHarness.IsOptimized(typeof(NDArrayOnnxInterop).Assembly) || !LifetimeHarness.IsOptimized(typeof(LifetimeGcTests).Assembly))
                Assert.Inconclusive("a Debug build keeps every local alive until its method returns, so the window this test probes does not exist; run it in Release (as CI does).");

            (long rounds, long wrong) = UnderCollectionPressure(TimeSpan.FromSeconds(3), 1 << 21, (round, elements) =>
            {
                using NDArray copy = AllocatedValue(elements, round).ToNDArray();
                return IsRamp(copy, round, elements);
            });

            Assert.AreEqual(0L, wrong, $"{wrong} of {rounds} copies read memory freed under them");
            Assert.IsTrue(rounds > 0);
        }

        /// <summary>
        ///     <see cref="NDArrayOnnxInterop.ToNDArray{T}(DenseTensor{T})"/> of a LEGACY run's output: the tensor's memory is
        ///     ORT's (an <c>OrtValueTensor&lt;T&gt;</c> whose pin is a bare pointer that keeps nothing alive), so the copy must
        ///     keep the tensor — and through it the native value — reachable until it is done. The session runs without ORT's
        ///     memory arena, so an output freed under the copy goes back to the system allocator (a 16 MB block is returned to
        ///     the OS) rather than staying mapped in the arena. A regression here may end the test host.
        /// </summary>
        /// <remarks>
        ///     <b>This window could not be opened on the current runtime</b>, KeepAlive or not: with the KeepAlive removed the
        ///     test passed every run (about 30 rounds each at 16 MB, the same pressure under which the OrtValue copy above
        ///     crashed). The optimized code keeps the tensor's <c>Memory&lt;T&gt;</c> — which holds ORT's memory manager, which
        ///     holds the OrtValue — in a stack slot whose address it hands to the <c>Buffer</c> getter, and the JIT reports such
        ///     an address-taken slot for the whole method. That is an implementation choice the runtime does not promise; the
        ///     KeepAlive is the guarantee, and this test stays as the net for the day the slot is tracked precisely.
        /// </remarks>
        [TestMethod]
        public void InFlightCopy_OfALegacyOutputTensorNobodyElseHolds_IsNeverFreedUnderTheRead()
        {
            if (!LifetimeHarness.IsOptimized(typeof(NDArrayOnnxInterop).Assembly) || !LifetimeHarness.IsOptimized(typeof(LifetimeGcTests).Assembly))
                Assert.Inconclusive("a Debug build keeps every local alive until its method returns, so the window this test probes does not exist; run it in Release (as CI does).");

            using var options = new SessionOptions { EnableCpuMemArena = false, IntraOpNumThreads = 1, InterOpNumThreads = 1 };
            using var session = new InferenceSession(Model("identity_float64"), options);
            (long rounds, long wrong) = UnderCollectionPressure(TimeSpan.FromSeconds(3), 1 << 21, (round, elements) =>
            {
                using NDArray copy = LegacyOutput(session, elements, round).ToNDArray();
                return IsRamp(copy, round, elements);
            });

            Assert.AreEqual(0L, wrong, $"{wrong} of {rounds} copies read memory freed under them");
            Assert.IsTrue(rounds > 0);
        }

        // ---- helpers ------------------------------------------------------------------------------------------

        /// <summary>The number of participants of a scenario that have a finalizer (its permutations are over those).</summary>
        /// <param name="build">The scenario's builder (one throwaway instance is built, then collected).</param>
        /// <returns>The count.</returns>
        private static int CountFinalizable(Func<Scenario> build)
        {
            (int count, LifetimeHarness.FinalizationReport report) = StageThrowaway(build);
            // Collect the throwaway now: left to the GC, its finalizers would run during a later order's run and move the
            // export counters that run checks.
            LifetimeHarness.RunStagedCollection(report);
            return count;
        }

        /// <summary>Builds a throwaway instance of a scenario and stages its collection, returning only its participant count.</summary>
        /// <param name="build">The scenario's builder.</param>
        /// <returns>The number of participants with a finalizer, and the staged collection's report.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static (int, LifetimeHarness.FinalizationReport) StageThrowaway(Func<Scenario> build)
        {
            object[] finalizable = Finalizable(build());
            return (finalizable.Length, LifetimeHarness.StageCollection(finalizable));
        }

        /// <summary>
        ///     Builds a scenario and stages the collection of its participants in <paramref name="order"/>, returning only what
        ///     the test keeps — the alias, the order's text and the report — so that once this returns nothing on the test's
        ///     frame reaches a participant (a Debug build keeps every local of a method alive until it returns, hence a method
        ///     of its own).
        /// </summary>
        /// <param name="build">The scenario's builder.</param>
        /// <param name="order">The finalization order, as indices into the participants that have a finalizer.</param>
        /// <returns>The alias, the order's text, and the staged collection's report.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static (IArraySlice, string, LifetimeHarness.FinalizationReport) StageScenario(Func<Scenario> build, int[] order)
        {
            Scenario s = build();
            object[] finalizable = Finalizable(s);
            object[] ordered = Array.ConvertAll(order, i => finalizable[i]);
            string text = string.Join(" → ", Array.ConvertAll(ordered, o => o.GetType().Name));
            return (s.Alias, text, LifetimeHarness.StageCollection(ordered));
        }

        /// <summary>The participants of <paramref name="s"/> whose type has a finalizer of its own, in declaration order.</summary>
        /// <param name="s">The scenario.</param>
        /// <returns>Those participants.</returns>
        private static object[] Finalizable(Scenario s) =>
            s.Participants.Where(p => p is not null && LifetimeHarness.HasFinalizer(p.GetType())).ToArray();

        /// <summary>The <see cref="IPinnable"/> a <see cref="MemoryHandle"/> carries (its private field), boxed.</summary>
        /// <param name="handle">The pin.</param>
        /// <returns>The pinnable — a NumSharp pin token, or (before the fix) the memory manager.</returns>
        private static object PinnableOf(MemoryHandle handle) =>
            typeof(MemoryHandle).GetField("_pinnable", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(handle);

        /// <summary>The pinnable of the pin an <see cref="OrtValue"/> made over managed or NumSharp memory keeps (ORT's private <c>_memHandle</c>).</summary>
        /// <param name="value">A value made by <c>OrtValue.CreateTensorValueFromMemory</c>.</param>
        /// <returns>The pinnable.</returns>
        private static object PinnableOf(OrtValue value)
        {
            object held = typeof(OrtValue).GetField("_memHandle", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(value);
            return PinnableOf((MemoryHandle)held);
        }

        /// <summary>Whether ORT considers <paramref name="value"/> disposed (its private flag; no native call is made).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The flag.</returns>
        private static bool OrtValueDisposed(OrtValue value) =>
            (bool)typeof(OrtValue).GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(value);

        /// <summary>Exports <paramref name="src"/> and returns only the OrtValue (and a weak reference to the dropped handle).</summary>
        /// <param name="src">The array to export.</param>
        /// <returns>The value the caller keeps, and a weak reference to the handle it does not.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static (OrtValue, WeakReference) ExportKeepingOnlyTheValue(NDArray src)
        {
            OrtTensor h = src.AsOrtValue();
            return (h.Value, new WeakReference(h));
        }

        /// <summary>Exports <paramref name="src"/> as a DenseTensor and returns only the tensor (and a weak reference to the dropped handle).</summary>
        /// <param name="src">The array to export.</param>
        /// <returns>The tensor the caller keeps, and a weak reference to the handle it does not.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static (DenseTensor<double>, WeakReference) ExportKeepingOnlyTheTensor(NDArray src)
        {
            OrtTensor<double> h = src.AsDenseTensor<double>();
            return (h.Tensor, new WeakReference(h));
        }

        /// <summary>
        ///     Exports <paramref name="src"/>, keeps ONLY the handle's OrtValue, and checks — after a full collection — that the
        ///     value is undisposed, reads the source's values (with the pool poisoned) and keeps the buffer's count honest (the
        ///     source refuses a reallocating resize). Runs in its own frame so that, once it returns, nothing of the export is
        ///     reachable: a Debug build keeps every local and temporary of a method alive until the method returns.
        /// </summary>
        /// <param name="src">A float64 ramp of <see cref="N"/> elements.</param>
        /// <returns>A weak reference to the handle, for the caller to see it collected.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference HoldOnlyTheValueThroughACollection(NDArray src)
        {
            (OrtValue value, WeakReference handle) = ExportKeepingOnlyTheValue(src);
            GcQuiescence.CollectFull();
            Assert.IsTrue(handle.IsAlive, "the handle was collected while its OrtValue was still reachable");
            Assert.IsFalse(OrtValueDisposed(value), "the OrtValue was disposed by a collection while the caller held it (a native call on it faults)");
            List<NDArray> poison = LifetimeHarness.Poison(N, 777.0);
            double[] read = value.GetTensorDataAsSpan<double>().ToArray();
            LifetimeHarness.DisposeAll(poison);
            CollectionAssert.AreEqual(Enumerable.Range(0, N).Select(i => (double)i).ToArray(), read, "the OrtValue reads freed memory");
            Assert.ThrowsException<IncorrectShapeException>(() => src.resize(new Shape(2 * N)),
                "resize reallocated a buffer the held OrtValue still points at (the handle's reference was not counted)");
            GC.KeepAlive(value);
            return handle;
        }

        /// <summary>
        ///     Exports <paramref name="src"/> as a DenseTensor, keeps ONLY the tensor, and checks — after a full collection —
        ///     that the tensor is still readable (it threw <see cref="ObjectDisposedException"/> after a mere collection before
        ///     the fix), reads the source's values with the pool poisoned, and keeps the buffer's count honest. Its own frame,
        ///     for the reason <see cref="HoldOnlyTheValueThroughACollection"/> gives.
        /// </summary>
        /// <param name="src">A float64 ramp of <see cref="N"/> elements.</param>
        /// <returns>A weak reference to the handle, for the caller to see it collected.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference HoldOnlyTheTensorThroughACollection(NDArray src)
        {
            (DenseTensor<double> tensor, WeakReference handle) = ExportKeepingOnlyTheTensor(src);
            GcQuiescence.CollectFull();
            Assert.IsTrue(handle.IsAlive, "the handle was collected while its tensor was still reachable");
            List<NDArray> poison = LifetimeHarness.Poison(N, 777.0);
            double[] read = tensor.Buffer.Span.ToArray();
            LifetimeHarness.DisposeAll(poison);
            CollectionAssert.AreEqual(Enumerable.Range(0, N).Select(i => (double)i).ToArray(), read, "the tensor reads freed memory");
            Assert.ThrowsException<IncorrectShapeException>(() => src.resize(new Shape(2 * N)),
                "resize reallocated a buffer the held tensor still points at");
            GC.KeepAlive(tensor);
            return handle;
        }

        /// <summary>Exports a Decimal ramp and returns only the float64 conversion the dropped handle owns.</summary>
        /// <param name="dense"><c>true</c> to export with <c>AsDenseTensor</c>, <c>false</c> with <c>AsOrtValue</c>.</param>
        /// <returns>The conversion (<c>Source</c>).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray ExportKeepingOnlyTheConversion(bool dense)
        {
            NDArray dec = Arange(NPTypeCode.Decimal, N);
            return dense ? dec.AsDenseTensor<double>().Source : dec.AsOrtValue().Source;
        }

        /// <summary>Pins <paramref name="nd"/>'s exported memory twice and drops both pins (and the OrtValue holding one) undisposed.</summary>
        /// <param name="nd">The array to export.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void DropPinsUndisposed(NDArray nd)
        {
            using OrtTensor<double> h = nd.AsDenseTensor<double>();
            _ = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, h.Memory, new long[] { N });
            _ = h.Memory.Pin();
        }

        /// <summary>
        ///     Warms <paramref name="round"/> up to tier-1 code on small inputs, then runs it on <paramref name="elements"/>
        ///     elements again and again for <paramref name="duration"/> while another thread collects gen 0 once a millisecond
        ///     (the objects a round drops are young, so a gen-0 collection finds them unreachable and queues their finalizers).
        /// </summary>
        /// <remarks>
        ///     <para><b>Why warm up.</b> An in-flight window exists only in optimized code: tier-0 code reports every local and
        ///     argument live for the whole method, which keeps the owner reachable across the read whatever the source says.
        ///     This suite's test host runs with tiered compilation off (see its csproj), so its code is optimized from the
        ///     first call; for a host that does tier, the round is called on small inputs first, with pauses that let a
        ///     background compile finish (the size does not matter to tiering; it only costs time).</para>
        ///     <para><b>Why pace the collector.</b> Collecting back to back suspends the round's thread so often that it barely
        ///     runs (a 16 MB legacy round managed 4 rounds in 3 s that way); one collection a millisecond still lands several
        ///     times inside every multi-millisecond copy.</para>
        /// </remarks>
        /// <param name="duration">How long to run under pressure.</param>
        /// <param name="elements">The element count of each pressured round.</param>
        /// <param name="round">One round, given its number and element count; returns whether what it read was right.</param>
        /// <returns>The number of pressured rounds run and the number that read wrong values.</returns>
        private static (long Rounds, long Wrong) UnderCollectionPressure(TimeSpan duration, int elements, Func<long, int, bool> round)
        {
            for (int cycle = 0; cycle < 3; cycle++)
            {
                for (int i = 0; i < 40; i++)
                    round(i, 1024);
                Thread.Sleep(150);
            }

            using var stop = new CancellationTokenSource(duration);
            var collector = new Thread(() =>
            {
                while (!stop.IsCancellationRequested)
                {
                    GC.Collect(0);
                    Thread.Sleep(1);
                }
            }) { IsBackground = true };
            collector.Start();
            long rounds = 0, wrong = 0;
            try
            {
                while (!stop.IsCancellationRequested)
                {
                    if (!round(rounds, elements))
                        wrong++;
                    rounds++;
                }
            }
            finally
            {
                stop.Cancel();
                collector.Join();
            }

            return (rounds, wrong);
        }

        /// <summary>
        ///     Whether <paramref name="copy"/> holds the ramp <c>start, start + 1, …</c> — checked at a prime stride and at the
        ///     last element, enough to see a block of freed memory that another allocation overwrote.
        /// </summary>
        /// <param name="copy">The float64 copy to check.</param>
        /// <param name="start">The ramp's first value.</param>
        /// <param name="elements">The element count.</param>
        /// <returns>Whether every sampled element is right.</returns>
        private static bool IsRamp(NDArray copy, long start, int elements)
        {
            if (copy.size != elements)
                return false;
            for (int i = 0; i < elements; i += 997)
            {
                if (copy.GetDouble(i) != start + i)
                    return false;
            }

            return copy.GetDouble(elements - 1) == start + elements - 1;
        }

        /// <summary>
        ///     The output tensor of a legacy run (<c>session.Run(NamedOnnxValue…)</c> → <c>AsTensor&lt;T&gt;()</c>) of the
        ///     float64 identity model over <c>start, start + 1, …</c>; the results collection is dropped, so the returned tensor
        ///     — whose memory is ORT's — is referenced by nobody but the caller's argument.
        /// </summary>
        /// <param name="session">An identity_float64 session.</param>
        /// <param name="elements">The element count.</param>
        /// <param name="start">The first value.</param>
        /// <returns>The output tensor.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static DenseTensor<double> LegacyOutput(InferenceSession session, int elements, long start)
        {
            var input = new DenseTensor<double>(new[] { elements });
            for (int i = 0; i < elements; i++)
                input[i] = start + i;
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("X", input) });
            return (DenseTensor<double>)results.First().AsTensor<double>();
        }

        /// <summary>An ORT-allocated float64 tensor holding <c>start, start + 1, …</c>, referenced by nobody but the caller's argument.</summary>
        /// <param name="elements">The element count.</param>
        /// <param name="start">The first value.</param>
        /// <returns>The value (the caller passes it on without keeping it).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static OrtValue AllocatedValue(int elements, long start)
        {
            OrtValue value = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Double, new long[] { elements });
            Span<double> span = value.GetTensorMutableDataAsSpan<double>();
            for (int i = 0; i < elements; i++)
                span[i] = start + i;
            return value;
        }
    }
}
