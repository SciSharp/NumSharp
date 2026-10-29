using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The lifetime and garbage-collection safety of the ML.NET bridge (the analysis is
    ///     <c>docs/plans/interop-lifetime-gc-analysis.md</c>): the view's and the cursors' finalizers never free a buffer a
    ///     reachable alias still reads, in any finalization order, and give each reference back exactly once; and a getter
    ///     keeps its cursor — and through it the pins — reachable until its read is complete.
    /// </summary>
    [TestClass]
    public class LifetimeGcTests : MLNetTestBase
    {
        /// <summary>The rows and columns of every scenario's array (float64): small, and one pool bucket.</summary>
        private const int Rows = 512, Cols = 8, N = Rows * Cols;

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
            ("AsDataView (one vector column)", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, Rows, Cols);
                NDArrayDataView view = src.AsDataView("x");
                return new Scenario { Alias = src.Storage.InternalArray, Participants = new object[] { src, view } };
            }),
            ("AsDataView (a scalar column per feature)", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, Rows, Cols);
                NDArrayDataView view = src.AsDataView(Enumerable.Range(0, Cols).Select(c => $"c{c}").ToArray());
                return new Scenario { Alias = src.Storage.InternalArray, Participants = new object[] { src, view } };
            }),
            ("AsDataView of Decimal (the view owns the float64 conversion)", () =>
            {
                NDArray dec = Arange(NPTypeCode.Decimal, Rows, Cols);
                NDArrayDataView view = dec.AsDataView("x");
                NDArray converted = FedOf(view);
                return new Scenario { Alias = converted.Storage.InternalArray, Participants = new object[] { dec, converted, view } };
            }),
            ("ToDataView (the view owns the snapshot)", () =>
            {
                using NDArray src = Arange(NPTypeCode.Double, Rows, Cols);
                NDArrayDataView view = src.ToDataView("x");
                NDArray snapshot = FedOf(view);
                return new Scenario { Alias = snapshot.Storage.InternalArray, Participants = new object[] { snapshot, view } };
            }),
            ("AsDataView with an undisposed cursor", () =>
            {
                NDArray src = Arange(NPTypeCode.Double, Rows, Cols);
                NDArrayDataView view = src.AsDataView("x");
                DataViewRowCursor cursor = view.GetRowCursor(view.Schema);
                return new Scenario { Alias = src.Storage.InternalArray, Participants = new object[] { src, view, cursor } };
            }),
            ("AsDataView of Decimal with an undisposed cursor", () =>
            {
                NDArray dec = Arange(NPTypeCode.Decimal, Rows, Cols);
                NDArrayDataView view = dec.AsDataView("x");
                DataViewRowCursor cursor = view.GetRowCursor(view.Schema);
                NDArray converted = FedOf(view);
                return new Scenario { Alias = converted.Storage.InternalArray, Participants = new object[] { dec, converted, view, cursor } };
            }),
        };

        /// <summary>
        ///     Every scenario, finalized in every order of its participants: a buffer a caller still reaches through an
        ///     uncounted alias keeps its values, is not freed, and ends with every counted reference given back exactly once;
        ///     the export counter returns to where it was.
        /// </summary>
        /// <remarks>
        ///     <para>Each order is one REAL collection of the scenario's objects (<see cref="LifetimeHarness.StageCollection"/>):
        ///     they are made unreachable and their finalizers run on the finalizer thread in the chosen order.</para>
        ///     <para>The view's and the cursor's finalizers used <c>Release</c>, whose decrement to zero frees the buffer at
        ///     once, so whenever the source array's own finalizer (an <c>Abandon</c>) ran first, they freed a buffer an alias
        ///     still read. A real collection happened to pick the passing order for this bridge; the failing ones were reachable
        ///     all the same.</para>
        /// </remarks>
        [TestMethod]
        public void Finalizers_InEveryOrder_NeverFreeUnderAReachableAlias_AndGiveEachReferenceBackOnce()
        {
            var failures = new List<string>();
            int runs = 0, broken = 0;
            foreach ((string name, Func<Scenario> build) in Scenarios)
            {
                int participants = CountFinalizable(build);
                foreach (int[] order in LifetimeHarness.Permutations(participants))
                {
                    runs++;
                    int exports = NDArrayMLNetInterop.LiveExports;
                    (IArraySlice alias, string orderText, LifetimeHarness.FinalizationReport report) = StageScenario(build, order);
                    LifetimeHarness.RunStagedCollection(report);

                    List<NDArray> poison = LifetimeHarness.Poison(N, 777.0);
                    double[] seen = LifetimeHarness.ReadDoubles(alias, N);
                    LifetimeHarness.DisposeAll(poison);
                    long count = LifetimeHarness.RefCount(alias);

                    int before = failures.Count;
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
                    if (NDArrayMLNetInterop.LiveExports != exports)
                        failures.Add($"{name} [{orderText}]: exports {exports} -> {NDArrayMLNetInterop.LiveExports}");
                    if (failures.Count > before)
                        broken++;
                }
            }

            Assert.AreEqual(2 + 2 + 6 + 2 + 6 + 24, runs, "the scenarios' orders were not all run");
            if (failures.Count > 0)
                Assert.Fail($"{broken} of {runs} finalization orders broke the lifetime contract:\n" + string.Join("\n", failures.Take(40)));
        }

        /// <summary>
        ///     A getter keeps its cursor reachable until its read is done. The getter delegate is often all a caller holds (an
        ///     ML.NET transform keeps getters, not cursors), and the JIT does not report the delegate live while it runs; past
        ///     the getter's last field read nothing else roots the cursor, and a collection landing while the row is copied
        ///     finalized the cursor, whose release could free the buffer mid-copy (here the view and the source were already
        ///     disposed, so the cursor's pin was the only reference). Before the fix a probe saw 117 of 279 such cursors
        ///     finalized during their own getter's call.
        /// </summary>
        /// <remarks>
        ///     <para><b>How a round decides.</b> A reader thread calls the getter of a 32 MB row (milliseconds of copying, in a
        ///     native copy the collector does not wait for) from a frame that holds nothing else; this thread collects and
        ///     runs finalizers meanwhile and watches the export counter. The cursor is the only export alive (the previous
        ///     round's is finalized first), so a drop is this cursor's finalizer — and the round counts it only if the getter
        ///     had not yet COMMITTED its row (the caller's buffer still empty). The getter's KeepAlive comes after that commit,
        ///     so with it no finalization can precede the commit: the verdict has no false positive, not even for a collection
        ///     that suspends the reader just after the getter returns.</para>
        ///     <para><b>Why optimized code.</b> The window exists only in optimized code: tier-0 code reports every local and
        ///     argument live for the whole method, which keeps the cursor reachable across the copy. This suite's test host
        ///     runs with tiered compilation off (see its csproj), so every method is optimized from its first call; the
        ///     reader's own frames are optimized regardless (<see cref="MethodImplOptions.AggressiveOptimization"/>), and
        ///     <see cref="InFlightReader.TierUp"/> calls the getters first for a host that does tier. That warm-up alone was
        ///     not enough: in a whole-suite run with tiering on, the getters never reached tier 1 and the control was never
        ///     caught. The first version of this test had neither and could not fail: a mutation that removed the KeepAlive
        ///     survived it.</para>
        ///     <para><b>Proof that the window is open.</b> A control getter of the same shape WITHOUT a KeepAlive runs through
        ///     the same harness first. If the control is never caught, the window could not be opened in this process and the
        ///     run proves nothing, so the test is inconclusive rather than green.</para>
        /// </remarks>
        [TestMethod]
        public void Getter_KeepsItsCursorReachable_UntilItsReadIsDone()
        {
            if (!LifetimeHarness.IsOptimized(typeof(NDArrayDataView).Assembly) || !LifetimeHarness.IsOptimized(typeof(LifetimeGcTests).Assembly))
                Assert.Inconclusive("a Debug build keeps every local alive until its method returns, so the window this test probes does not exist; run it in Release (as CI does).");

            const int Width = 1 << 22, Rounds = 12;
            using NDArray controlRow = np.arange((double)Width);
            unsafe
            {
                InFlightReader.ControlMemory = (IntPtr)controlRow.Storage.InternalArray.Address;
            }
            try
            {
                InFlightReader.Prepare(Width);
                InFlightReader.TierUp();

                int controlCaught = 0;
                for (int round = 0; round < Rounds; round++)
                {
                    if (FinalizedBeforeCommit(InFlightReader.ReadControlRow, () => Volatile.Read(ref ControlOwner.Live), Width, out _))
                        controlCaught++;
                }

                // The probe's sensitivity in this run, kept in the test's output as the evidence that a green result means something.
                Console.WriteLine($"control getter (no KeepAlive) caught in {controlCaught} of {Rounds} rounds");
                if (controlCaught == 0)
                    Assert.Inconclusive($"the control getter (no KeepAlive) was never caught in {Rounds} rounds: the in-flight window could not be opened in this process, so this run proves nothing.");

                int productCaught = 0, wrongReads = 0;
                for (int round = 0; round < Rounds; round++)
                {
                    if (FinalizedBeforeCommit(InFlightReader.ReadProductRow, () => NDArrayMLNetInterop.LiveExports, Width, out bool readOk))
                        productCaught++;
                    if (!readOk)
                        wrongReads++;
                }

                Assert.AreEqual(0, productCaught, $"a cursor was finalized before its own getter had committed its row in {productCaught} of {Rounds} rounds (the control was caught in {controlCaught})");
                Assert.AreEqual(0, wrongReads, $"{wrongReads} of {Rounds} getter calls read the wrong values");
            }
            finally
            {
                InFlightReader.ControlMemory = IntPtr.Zero;
                InFlightReader.Release();
            }
        }

        // ---- helpers ------------------------------------------------------------------------------------------

        /// <summary>The number of participants of a scenario that have a finalizer (its permutations are over those).</summary>
        /// <param name="build">The scenario's builder (one throwaway instance is built, then collected).</param>
        /// <returns>The count.</returns>
        private static int CountFinalizable(Func<Scenario> build)
        {
            (int count, LifetimeHarness.FinalizationReport report) = StageThrowaway(build);
            // Collect the throwaway now: left to the GC, its finalizers would run during a later order's run and move the
            // export counter that run checks.
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

        /// <summary>The array a view reads (its private <c>_fed</c>: the source, a conversion or a snapshot).</summary>
        /// <param name="view">The view.</param>
        /// <returns>The array.</returns>
        private static NDArray FedOf(NDArrayDataView view) =>
            (NDArray)typeof(NDArrayDataView).GetField("_fed", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(view);

        /// <summary>
        ///     A getter whose cursor nothing else references: the source array and the view are disposed, so the cursor's own
        ///     pin is the buffer's only reference and the getter's closure the cursor's only root.
        /// </summary>
        /// <param name="width">The row width (elements of the one-row vector column).</param>
        /// <returns>The getter of the first row.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static ValueGetter<VBuffer<double>> GetterOfAnOtherwiseUnreachableCursor(int width)
        {
            using NDArray ramp = np.arange((double)width);
            NDArray nd = ramp.reshape(1, width);
            NDArrayDataView view = nd.AsDataView("x");
            DataViewRowCursor cursor = view.GetRowCursor(view.Schema);
            cursor.MoveNext();
            ValueGetter<VBuffer<double>> getter = cursor.GetGetter<VBuffer<double>>(view.Schema[0]);
            view.Dispose();
            nd.Dispose();
            return getter;   // the ramp is disposed on the way out: the cursor's pin is then the buffer's only reference
        }

        /// <summary>
        ///     One round of the in-flight probe: a reader thread runs <paramref name="read"/> (which calls a getter its frame no
        ///     longer roots) while this thread collects and runs finalizers; the round is caught when the getter's owner was
        ///     finalized (its live counter dropped) before the getter committed its row.
        /// </summary>
        /// <param name="read">The reader: <see cref="InFlightReader.ReadProductRow"/> or <see cref="InFlightReader.ReadControlRow"/>.</param>
        /// <param name="live">The counter the owner's finalizer decrements (the export counter, or the control's).</param>
        /// <param name="width">The row width.</param>
        /// <param name="readOk">Whether the getter returned the row's values.</param>
        /// <returns><c>true</c> when the owner was finalized before the commit.</returns>
        private static bool FinalizedBeforeCommit(Action<int> read, Func<int> live, int width, out bool readOk)
        {
            Settle();   // the previous round's owner is finalized now, not during this round's call
            InFlightReader.Reset();
            var reader = new Thread(() => read(width));
            reader.Start();
            while (Volatile.Read(ref InFlightReader.InCall) == 0 && reader.IsAlive)
                Thread.Yield();

            bool caught = false;
            while (Volatile.Read(ref InFlightReader.InCall) == 1)
            {
                // YOUNG collections: the owner and its closure are fresh, and a young collection takes well under a
                // millisecond whatever else the heap holds. A full one did not — in a whole-suite run, where earlier tests
                // leave a large heap, each lasted longer than the 32 MB copy, the copy finished inside it, the getter then
                // committed before the finalizer thread ran, and even the control was never caught.
                GcQuiescence.CollectYoung();
                if (live() < Volatile.Read(ref InFlightReader.LiveAtCall))
                {
                    // Seen the owner's finalizer: decide by whether the getter had committed by then. The commit was written
                    // before the owner could become unreachable (if the getter keeps it), and the collection in between
                    // orders the memory, so a commit that happened is visible here.
                    Interlocked.MemoryBarrier();
                    caught = InFlightReader.CommittedLength() == 0;
                    break;
                }
            }

            reader.Join();
            readOk = InFlightReader.ReadOk;
            return caught;
        }

        /// <summary>
        ///     The reader side of the in-flight probe. Its frames are compiled optimized from the first call
        ///     (<see cref="MethodImplOptions.AggressiveOptimization"/>), so that, as in production code that has run a while, a
        ///     getter passed to a call is not kept alive by the caller during that call — tier-0 code would keep it, and with it
        ///     the cursor, for the whole method. The getter writes into a static buffer so the watcher can see the commit.
        /// </summary>
        private static class InFlightReader
        {
            /// <summary>1 while the getter call is in progress.</summary>
            internal static int InCall;

            /// <summary>The owner counter's value just before the call (the round's owner included).</summary>
            internal static int LiveAtCall;

            /// <summary>Whether the last call returned the expected row.</summary>
            internal static bool ReadOk;

            /// <summary>The address of the control's row (an array the test keeps alive, so the control never reads freed memory).</summary>
            internal static IntPtr ControlMemory;

            /// <summary>The caller's buffer: empty (length 0) until the getter commits the row into it.</summary>
            private static VBuffer<double> s_buffer;

            /// <summary>
            ///     The array the getters fill, allocated once: a getter reuses its destination's array when it is large
            ///     enough, so no 32 MB allocation — which can start a full collection of its own — lands in the window.
            /// </summary>
            private static double[] s_values;

            /// <summary>Allocates the destination array the rounds reuse.</summary>
            /// <param name="width">The widest row a round reads.</param>
            internal static void Prepare(int width)
            {
                s_values = new double[width];
                s_buffer = new VBuffer<double>(0, s_values);
            }

            /// <summary>Drops the destination array (the test's last step).</summary>
            internal static void Release()
            {
                s_buffer = default;
                s_values = null;
            }

            /// <summary>Empties the buffer — keeping its array — and clears the flags for a new round.</summary>
            internal static void Reset()
            {
                s_buffer = new VBuffer<double>(0, s_values);
                ReadOk = false;
                Volatile.Write(ref InCall, 0);
            }

            /// <summary>The committed row's length: 0 until the getter's <c>editor.Commit()</c> result is assigned.</summary>
            /// <returns>The length.</returns>
            internal static int CommittedLength() => Volatile.Read(ref Unsafe.AsRef(in s_buffer.Length));

            /// <summary>
            ///     Calls both getters often enough that their code reaches tier 1 (30 calls, then a background compile the
            ///     sleeps leave time for); the window the probe looks for exists only in that code.
            /// </summary>
            internal static void TierUp()
            {
                for (int cycle = 0; cycle < 4; cycle++)
                {
                    for (int i = 0; i < 60; i++)
                    {
                        ReadProductRow(16);
                        ReadControlRow(16);
                    }

                    Thread.Sleep(150);
                }
            }

            /// <summary>Reads one row through the product's getter of a cursor nothing else references.</summary>
            /// <param name="width">The row width.</param>
            [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
            internal static void ReadProductRow(int width) =>
                Read(GetterOfAnOtherwiseUnreachableCursor(width), NDArrayMLNetInterop.LiveExports, width);

            /// <summary>Reads one row through the control getter (no KeepAlive) of an owner nothing else references.</summary>
            /// <param name="width">The row width (at most the control row's).</param>
            [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
            internal static void ReadControlRow(int width) =>
                Read(new ControlOwner(ControlMemory, width).Getter(), Volatile.Read(ref ControlOwner.Live), width);

            /// <summary>Calls <paramref name="getter"/> into the static buffer, flagging the call, then checks the row.</summary>
            /// <param name="getter">The getter (its owner is reachable only through it).</param>
            /// <param name="liveAtCall">The owner counter just before the call.</param>
            /// <param name="width">The row width.</param>
            [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
            private static void Read(ValueGetter<VBuffer<double>> getter, int liveAtCall, int width)
            {
                Volatile.Write(ref LiveAtCall, liveAtCall);
                Volatile.Write(ref InCall, 1);
                getter(ref s_buffer);   // the getter's last use: from here on this frame roots neither it nor its owner
                Volatile.Write(ref InCall, 0);
                ReadOnlySpan<double> v = s_buffer.GetValues();
                ReadOk = v.Length == width && v[0] == 0 && v[width / 2] == width / 2 && v[width - 1] == width - 1;
            }
        }

        /// <summary>
        ///     The control of the in-flight probe: a getter of the product's shape — it reads its owner's fields, then copies a
        ///     row from a raw address and commits it — but WITHOUT a KeepAlive, so its owner is unreachable during the copy.
        ///     Catching it shows that the probe can see the window in this process.
        /// </summary>
        private sealed class ControlOwner
        {
            /// <summary>The number of control owners not yet finalized (the control's export counter).</summary>
            internal static int Live;

            private readonly IntPtr _address;
            private readonly int _width;

            /// <summary>An owner of a row at <paramref name="address"/>.</summary>
            /// <param name="address">The row's first element (memory the test keeps alive).</param>
            /// <param name="width">The row width.</param>
            internal ControlOwner(IntPtr address, int width)
            {
                _address = address;
                _width = width;
                Interlocked.Increment(ref Live);
            }

            /// <summary>Counts the owner gone.</summary>
            ~ControlOwner() => Interlocked.Decrement(ref Live);

            /// <summary>The getter: it captures only <c>this</c>, whose last use is the field reads before the copy.</summary>
            /// <returns>The getter.</returns>
            internal ValueGetter<VBuffer<double>> Getter() => (ref VBuffer<double> dst) =>
            {
                int width = _width;
                IntPtr address = _address;
                VBufferEditor<double> editor = VBufferEditor.Create(ref dst, width);
                unsafe
                {
                    new ReadOnlySpan<double>((void*)address, width).CopyTo(editor.Values);
                }

                dst = editor.Commit();
            };
        }
    }
}
