using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     A hand-written <see cref="NDScope"/> disposed on a thread other than the one that opened it - the shape of a
    ///     scope held across an <c>await</c>, whose end runs on the continuation's thread - is REFUSED in every build
    ///     (F83 / P20 of the bridge QA, 2026-11-03). Before, a Debug-only assert guarded it: in Release the foreign thread
    ///     disposed the scope's arrays, pooled the scope in ITS slot and reused it for its next scope, while the opening
    ///     thread still had it as its current scope and tracked its later arrays into it - so the foreign thread's next
    ///     close freed the opening thread's arrays (an array filled with 7.0 read back 9.0 through the bridge).
    /// </summary>
    /// <remarks>
    ///     Each test runs the opening thread as a <see cref="DedicatedThread"/> that stays alive between steps, so its
    ///     <c>[ThreadStatic]</c> scope chain persists exactly as a pool thread's would after the <c>await</c> moved the
    ///     method away; the steps are sequenced by joins, so every interleaving is deterministic. Only threads the test
    ///     creates are involved, so no process-wide state is read and the class runs in parallel with the suite.
    /// </remarks>
    [TestClass]
    public class NDScopeForeignDisposeTests
    {
        /// <summary>
        ///     One dedicated thread that runs submitted work items one at a time, in order, and outlives each of them -
        ///     the scope's OPENING thread, whose thread-static chain must persist between the steps of a test.
        /// </summary>
        private sealed class DedicatedThread : IDisposable
        {
            private readonly BlockingCollection<(Action Work, TaskCompletionSource<bool> Done)> _queue = new();
            private readonly Thread _thread;

            /// <summary>Starts the thread; it waits for work until <see cref="Dispose"/>.</summary>
            public DedicatedThread()
            {
                _thread = new Thread(Loop) { IsBackground = true, Name = nameof(NDScopeForeignDisposeTests) };
                _thread.Start();
            }

            /// <summary>Runs every queued item on this one thread, reporting each outcome to its submitter.</summary>
            private void Loop()
            {
                foreach (var (work, done) in _queue.GetConsumingEnumerable())
                {
                    try
                    {
                        work();
                        done.SetResult(true);
                    }
                    catch (Exception e)
                    {
                        done.SetException(e);
                    }
                }
            }

            /// <summary>Runs <paramref name="work"/> on the dedicated thread and waits for it.</summary>
            /// <param name="work">The step to run there.</param>
            /// <exception cref="Exception">Whatever <paramref name="work"/> threw, rethrown here (an assertion included).</exception>
            public void Run(Action work)
            {
                var done = new TaskCompletionSource<bool>();
                _queue.Add((work, done));
                done.Task.GetAwaiter().GetResult();
            }

            /// <summary>Stops the thread after the queued work and waits for it to end.</summary>
            public void Dispose()
            {
                _queue.CompleteAdding();
                _thread.Join();
                _queue.Dispose();
            }
        }

        /// <summary>Runs <paramref name="action"/> on a fresh thread, joins it, and returns what it threw (or null).</summary>
        /// <param name="action">The step to run on a thread that never saw the scope opened.</param>
        /// <returns>The exception the step threw, or <c>null</c>.</returns>
        private static Exception OnNewThread(Action action)
        {
            Exception error = null;
            var thread = new Thread(() =>
            {
                try
                {
                    action();
                }
                catch (Exception e)
                {
                    error = e;
                }
            });
            thread.Start();
            thread.Join();
            return error;
        }

        /// <summary>
        ///     The refusal is an <see cref="InvalidOperationException"/> that names the cause (a scope held across an
        ///     await) and both remedies, and a second Dispose - on either thread - is a no-op.
        /// </summary>
        [TestMethod]
        public void Dispose_OnAnotherThread_Throws_NamingTheAwaitRule()
        {
            using var opener = new DedicatedThread();
            NDScope scope = null;
            opener.Run(() => scope = NDScope.Open());

            Exception error = OnNewThread(scope.Dispose);
            Assert.IsInstanceOfType(error, typeof(InvalidOperationException));
            StringAssert.Contains(error.Message, "different thread than the one that opened the scope");
            StringAssert.Contains(error.Message, "'await'");
            StringAssert.Contains(error.Message, "[NDScopedAsync]");

            Assert.IsNull(OnNewThread(scope.Dispose), "a second Dispose on a foreign thread is a no-op");
            opener.Run(scope.Dispose);   // and on the opening thread too: the refused scope stays closed
        }

        /// <summary>
        ///     On its opening thread a refused scope is SKIPPED: it is no longer current, a later array is tracked by
        ///     the scope below it, and that scope still reclaims it. Without the skip every later array on that thread
        ///     would be tracked into a scope nobody can dispose - held forever.
        /// </summary>
        [TestMethod]
        public void RefusedScope_IsSkipped_OnItsOpeningThread()
        {
            using var opener = new DedicatedThread();
            NDScope outer = null, inner = null;
            opener.Run(() =>
            {
                outer = NDScope.Open();
                inner = NDScope.Open();
            });

            Assert.IsInstanceOfType(OnNewThread(inner.Dispose), typeof(InvalidOperationException));

            NDArray later = null;
            opener.Run(() =>
            {
                // The array FIRST: reading NDScope.Current would itself pop the refused scope, and then Track's own
                // skip - the path every construction takes - would go untested.
                later = np.arange(3);
                Assert.AreSame(outer, later.TrackingScope, "a later array is tracked by the live scope below the refused one");
                Assert.AreSame(outer, NDScope.Current, "the refused inner scope is skipped: the outer one is current again");
                outer.Dispose();
            });
            Assert.IsTrue(later.IsDisposed, "the live scope still reclaims what it tracked");

            opener.Run(() => Assert.IsNull(NDScope.Current, "nothing is left open on the opening thread"));
        }

        /// <summary>
        ///     The refusing thread never pools the scope: its next <see cref="NDScope.Open"/> is a different object. The
        ///     old Dispose pooled it in the foreign thread's single-slot free list, which is how the next request on that
        ///     thread came to share one scope with the opening thread.
        /// </summary>
        [TestMethod]
        public void RefusedScope_IsNeverPooled_OnTheRefusingThread()
        {
            using var opener = new DedicatedThread();
            NDScope refused = null;
            opener.Run(() => refused = NDScope.Open());

            bool reused = true;
            Exception error = OnNewThread(() =>
            {
                try
                {
                    refused.Dispose();
                }
                catch (InvalidOperationException)
                {
                    // the refusal under test; the step goes on to open the thread's next scope
                }
                using var next = NDScope.Open();
                reused = ReferenceEquals(refused, next);
            });
            Assert.IsNull(error);
            Assert.IsFalse(reused, "the refusing thread must not hand the refused scope out again");
        }

        /// <summary>
        ///     The F83 sequence, end to end. Thread A opens a scope and the method "awaits"; the continuation's thread B
        ///     ends the scope (refused); A goes on with other work and builds an array of 7.0; B opens, fills and closes
        ///     its next scope. A's array must still be alive and read 7.0 - before the fix B's close disposed it, because
        ///     B had pooled and reused A's still-current scope.
        /// </summary>
        [TestMethod]
        public void ScopeHeldAcrossAnAwait_NeverFreesTheOpeningThreadsLaterArrays()
        {
            using var threadA = new DedicatedThread();
            using var threadB = new DedicatedThread();
            NDScope handlerScope = null;
            NDArray beforeAwait = null;
            threadA.Run(() =>
            {
                handlerScope = NDScope.Open();               // using var scope = NDScope.Open();
                beforeAwait = np.full(new Shape(64), 1.0);   // a temp the handler built before its await
            });

            Exception refused = null;
            threadB.Run(() =>
            {
                try
                {
                    handlerScope.Dispose();                  // the using's end, on the continuation's thread
                }
                catch (InvalidOperationException e)
                {
                    refused = e;
                }
            });
            Assert.IsNotNull(refused, "the foreign-thread Dispose is refused");
            Assert.IsFalse(beforeAwait.IsDisposed, "a refused scope frees nothing: its arrays go to the finalizer");

            NDArray later = null;
            threadA.Run(() =>
            {
                later = np.full(new Shape(64), 7.0);         // thread A's next, unrelated work
                Assert.IsNull(later.TrackingScope, "the refused scope tracks nothing on its opening thread");
            });

            threadB.Run(() =>
            {
                using var nextRequest = NDScope.Open();      // thread B's next request
                Assert.AreNotSame(handlerScope, nextRequest);
                using NDArray nine = np.full(new Shape(64), 9.0);
            });

            Assert.IsFalse(later.IsDisposed, "thread B's close must not reach thread A's array");
            Assert.AreEqual(7.0, later.GetDouble(0));
            Assert.AreEqual(7.0, later.GetDouble(63));
            later.Dispose();
            beforeAwait.Dispose();
        }
    }
}
