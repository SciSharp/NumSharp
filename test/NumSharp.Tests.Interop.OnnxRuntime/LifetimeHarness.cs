using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Tools for the lifetime and garbage-collection proofs of the interop bridges: run finalizers in a chosen order,
    ///     enumerate every order, read a buffer's reference count, and poison the buffer pool so that a read through a freed
    ///     alias shows. Shared by the interop test projects (this file is linked into the ML.NET suite), which is why it lives
    ///     in the common <c>NumSharp.Tests</c> namespace.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why the finalization order is chosen.</b> When a group of objects becomes unreachable together, the
    ///     runtime queues all of their finalizers and runs them in an order it does not specify. A safety argument must hold
    ///     for EVERY order, and a collection exercises only the one the runtime happens to pick (the same one, run after
    ///     run). <see cref="StageCollection"/> + <see cref="RunStagedCollection"/> reproduce one REAL collection of the group
    ///     with the order chosen: the participants are made genuinely unreachable, so the collection clears every short weak
    ///     handle to them exactly as it would (code that tells finalization from a live call by such a handle — the ONNX
    ///     bridge's pin token does — sees what it would see), and ONE finalizable conductor, unreachable with them, runs
    ///     their finalizers on the finalizer thread in the chosen order. Each participant's own finalizer is suppressed
    ///     first, so the runtime never runs it a second time.</para>
    ///     <para>Only objects that are genuinely unreachable in the scenario may be staged: finalizing an object the scenario
    ///     still uses would test an impossible state.</para>
    /// </remarks>
    internal static class LifetimeHarness
    {
        /// <summary>
        ///     What one staged collection did, filled in from the finalizer thread: whether the participants' finalizers have
        ///     all run, and the exception one of them threw, if any. The test keeps it; it references no participant.
        /// </summary>
        internal sealed class FinalizationReport
        {
            private int _done;

            /// <summary><c>true</c> once the staged finalizers have all run (or one threw).</summary>
            internal bool Done => Volatile.Read(ref _done) != 0;

            /// <summary>The exception a staged finalizer threw (in production a throwing finalizer ends the process), or <c>null</c>.</summary>
            internal Exception Failure { get; private set; }

            /// <summary>Records the outcome; called once, by the conductor's finalizer.</summary>
            /// <param name="failure">The exception a finalizer threw, or <c>null</c>.</param>
            internal void Complete(Exception failure)
            {
                Failure = failure;
                Volatile.Write(ref _done, 1);
            }
        }

        /// <summary>
        ///     The one finalizable object of a staged collection: it holds the participants (so they stay unreachable with
        ///     it, and are resurrected with it when the collection queues it) and its finalizer runs theirs in the chosen
        ///     order on the finalizer thread — what the runtime does with a group, minus the choice of order.
        /// </summary>
        private sealed class Conductor
        {
            private readonly object[] _order;
            private readonly FinalizationReport _report;

            /// <summary>Holds <paramref name="order"/> for the collection to come.</summary>
            /// <param name="order">The participants, in the order their finalizers are to run.</param>
            /// <param name="report">Where the outcome goes.</param>
            internal Conductor(object[] order, FinalizationReport report)
            {
                _order = order;
                _report = report;
            }

            /// <summary>Runs each participant's finalizer in order, then reports; a throwing finalizer is reported, not rethrown.</summary>
            ~Conductor()
            {
                Exception failure = null;
                try
                {
                    foreach (object o in _order)
                        DeclaredFinalizer(o.GetType())?.Invoke(o, null);
                }
                catch (Exception e)
                {
                    // Rethrowing from a finalizer ends the process; the test asserts on the report instead.
                    failure = e;
                }
                finally
                {
                    _report.Complete(failure);
                }
            }
        }

        /// <summary>
        ///     Prepares one collection of <paramref name="order"/> with their finalizers run in that order: suppresses each
        ///     participant's own finalizer and hands them all to a conductor nothing references. The caller must then drop
        ///     every reference it holds to the participants — build them in a method that returns only the report and what the
        ///     test keeps (an alias, a text) — and call <see cref="RunStagedCollection"/>.
        /// </summary>
        /// <param name="order">The participants, in the order their finalizers are to run; types without a finalizer are skipped.</param>
        /// <returns>The report the conductor fills in.</returns>
        internal static FinalizationReport StageCollection(params object[] order)
        {
            foreach (object o in order)
                GC.SuppressFinalize(o);
            var report = new FinalizationReport();
            _ = new Conductor(order, report);
            return report;
        }

        /// <summary>
        ///     Runs the collection <see cref="StageCollection"/> prepared: a young collection first (the participants are
        ///     usually fresh), then up to five full ones, until the conductor has run.
        /// </summary>
        /// <param name="report">The report <see cref="StageCollection"/> returned.</param>
        /// <exception cref="InvalidOperationException">The conductor never ran: something still reaches a participant.</exception>
        /// <exception cref="TargetInvocationException">A participant's finalizer threw.</exception>
        internal static void RunStagedCollection(FinalizationReport report)
        {
            GcQuiescence.CollectYoung();
            for (int attempt = 0; !report.Done && attempt < 5; attempt++)
                GcQuiescence.CollectFull();
            if (!report.Done)
                throw new InvalidOperationException("the staged collection never ran: a participant (or the conductor) is still reachable from the test");
            if (report.Failure is not null)
                throw new TargetInvocationException("a staged finalizer threw", report.Failure);
        }

        /// <summary><c>true</c> when an object of <paramref name="type"/> has a finalizer of its own (not just <see cref="object"/>'s).</summary>
        /// <param name="type">The object's type.</param>
        /// <returns>Whether the runtime would run a finalizer for it.</returns>
        internal static bool HasFinalizer(Type type) => DeclaredFinalizer(type) is not null;

        /// <summary>The most derived finalizer of <paramref name="type"/>, or <c>null</c> when only <see cref="object"/> declares one.</summary>
        /// <param name="type">The type to search, then its base types.</param>
        /// <returns>The finalizer method (a C# destructor compiles to a <c>Finalize</c> override).</returns>
        private static MethodInfo DeclaredFinalizer(Type type)
        {
            for (Type t = type; t is not null && t != typeof(object); t = t.BaseType)
            {
                MethodInfo m = t.GetMethod("Finalize", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
                if (m is not null)
                    return m;
            }

            return null;
        }

        /// <summary>
        ///     Every order of <paramref name="n"/> participants (Heap's algorithm), as index permutations of
        ///     <c>0..n-1</c>; each yielded array is a fresh copy.
        /// </summary>
        /// <param name="n">The number of participants (small: the count is n!).</param>
        /// <returns>The n! permutations.</returns>
        internal static IEnumerable<int[]> Permutations(int n)
        {
            var a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = i;
            var c = new int[n];
            yield return (int[])a.Clone();
            int k = 0;
            while (k < n)
            {
                if (c[k] < k)
                {
                    int j = (k % 2 == 0) ? 0 : c[k];
                    (a[j], a[k]) = (a[k], a[j]);
                    yield return (int[])a.Clone();
                    c[k]++;
                    k = 0;
                }
                else
                {
                    c[k] = 0;
                    k++;
                }
            }
        }

        /// <summary>
        ///     The reference count of the memory block behind <paramref name="slice"/>, read from its private ARC state:
        ///     the number of counted references still held, or -1 once the block claimed its free.
        /// </summary>
        /// <param name="slice">A storage slice (an <c>ArraySlice&lt;T&gt;</c>).</param>
        /// <returns>The block's <c>_refCount</c>.</returns>
        /// <exception cref="InvalidOperationException">The ARC fields were renamed (the harness must follow).</exception>
        internal static long RefCount(IArraySlice slice)
        {
            object block = slice.GetType().GetField("MemoryBlock")?.GetValue(slice)
                ?? throw new InvalidOperationException("ArraySlice<T>.MemoryBlock not found");
            object disposer = block.GetType().GetField("_disposer", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(block)
                ?? throw new InvalidOperationException("UnmanagedMemoryBlock<T>._disposer not found");
            FieldInfo count = disposer.GetType().GetField("_refCount", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("Disposer._refCount not found");
            return (long)count.GetValue(disposer);
        }

        /// <summary>The memory block's disposer behind <paramref name="slice"/> — a finalizable participant of its own.</summary>
        /// <param name="slice">A storage slice (an <c>ArraySlice&lt;T&gt;</c>).</param>
        /// <returns>The disposer object.</returns>
        /// <exception cref="InvalidOperationException">The fields were renamed (the harness must follow).</exception>
        internal static object DisposerOf(IArraySlice slice)
        {
            object block = slice.GetType().GetField("MemoryBlock")?.GetValue(slice)
                ?? throw new InvalidOperationException("ArraySlice<T>.MemoryBlock not found");
            return block.GetType().GetField("_disposer", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(block)
                ?? throw new InvalidOperationException("UnmanagedMemoryBlock<T>._disposer not found");
        }

        /// <summary>
        ///     Takes same-size buffers back out of the pool and fills them with <paramref name="sentinel"/>, so that a buffer
        ///     freed under a live alias is handed out again and overwritten — the alias then reads the sentinel instead of its
        ///     own values. The caller disposes the returned arrays once it has read the alias.
        /// </summary>
        /// <param name="count">The element count of the buffer to catch (float64 elements).</param>
        /// <param name="sentinel">The fill value.</param>
        /// <returns>The poison arrays.</returns>
        internal static List<NDArray> Poison(long count, double sentinel)
        {
            var list = new List<NDArray>();
            for (int i = 0; i < 64; i++)
                list.Add(np.full(new Shape(count), sentinel));
            return list;
        }

        /// <summary>Disposes every array of <paramref name="arrays"/>.</summary>
        /// <param name="arrays">The arrays (typically from <see cref="Poison"/>).</param>
        internal static void DisposeAll(IEnumerable<NDArray> arrays)
        {
            foreach (NDArray a in arrays)
                a.Dispose();
        }

        /// <summary>
        ///     Reads <paramref name="count"/> float64 elements at <paramref name="slice"/>'s address — the way an alias that
        ///     holds no counted reference reads the buffer (no liveness check, no reference taken).
        /// </summary>
        /// <param name="slice">The alias.</param>
        /// <param name="count">The number of elements to read.</param>
        /// <returns>The elements.</returns>
        internal static unsafe double[] ReadDoubles(IArraySlice slice, long count)
        {
            var result = new double[count];
            double* p = (double*)slice.Address;
            for (long i = 0; i < count; i++)
                result[i] = p[i];
            return result;
        }

        /// <summary>
        ///     <c>true</c> when <paramref name="assembly"/> was compiled with the JIT optimizer enabled (a Release build). In a
        ///     Debug build the JIT keeps every local of a method alive until it returns, so a window in which an object is
        ///     already unreachable while code still uses memory it owns does not exist there, and a test of such a window
        ///     proves nothing.
        /// </summary>
        /// <param name="assembly">The assembly whose code runs the window.</param>
        /// <returns>Whether the window can exist.</returns>
        internal static bool IsOptimized(Assembly assembly)
        {
            var debuggable = assembly.GetCustomAttribute<DebuggableAttribute>();
            return debuggable is null || !debuggable.IsJITOptimizerDisabled;
        }
    }
}
