using System.Threading;
using System.Threading.Tasks;

namespace NumSharp.Tests.Backends.Printing
{
    /// <summary>
    ///     Where the print options apply: NumPy 2.4.2 keeps them in a <c>ContextVar</c>, so under the oracle's Python 3.12 a
    ///     value set with <c>np.set_printoptions</c> or <c>with np.printoptions(...)</c> is seen by the setting flow and by
    ///     the asynchronous work it starts (an <c>await</c> continuation, <c>asyncio.to_thread</c>), while a new
    ///     <c>threading.Thread</c> starts at the defaults and a worker's own setting stays in the worker (parity QA-40).
    ///     NumSharp's <c>PrintOptions.Current</c> rides an <c>NDContextVar</c> with those semantics: an <c>await</c>,
    ///     <c>Task.Run</c> and the thread pool inherit, a <c>new Thread</c> does not.
    /// </summary>
    /// <remarks>
    ///     Every expected string is NumPy 2.4.2's <c>np.array2string(np.array([1.23456789]))</c> under the same setting
    ///     (probed on win-amd64 with Python 3.12: <c>[1.23456789]</c> at the default precision 8, <c>[1.23]</c> at 2,
    ///     <c>[1.235]</c> at 3, <c>[1.2346]</c> at 4). Each test changes the options only inside a <c>np.printoptions</c>
    ///     scope or on a thread it owns, so the suite's parallel workers never see another test's setting.
    /// </remarks>
    [TestClass]
    public class PrintOptionsContextTests
    {
        /// <summary>The probe value printed by every test.</summary>
        /// <returns>The array's <c>str</c> under the options the calling thread and flow see.</returns>
        private static string Show()
        {
            using var value = np.array(new[] { 1.23456789 });
            return value.ToString();
        }

        /// <summary>Runs <paramref name="body"/> on a new dedicated thread and waits for it.</summary>
        /// <param name="body">The work, which records what it saw.</param>
        private static void OnNewThread(System.Action body)
        {
            var worker = new Thread(() => body());
            worker.Start();
            worker.Join();
        }

        /// <summary>
        ///     NumPy: <c>threading.Thread</c> starts with an empty context, so a worker started inside
        ///     <c>with np.printoptions(precision=4)</c> (or after <c>np.set_printoptions(precision=2)</c>) prints
        ///     <c>[1.23456789]</c>.
        /// </summary>
        [TestMethod]
        public void NewThread_StartsAtTheDefaults()
        {
            string inScope = null, worker = null;
            using (np.printoptions(precision: 4))
            {
                inScope = Show();
                OnNewThread(() => worker = Show());
            }

            inScope.Should().Be("[1.2346]");
            worker.Should().Be("[1.23456789]", "a new thread starts with NumPy's default print options");
        }

        /// <summary>
        ///     NumPy: <c>asyncio.to_thread</c> copies the caller's context, so the worker prints <c>[1.23]</c> under
        ///     precision 2. <c>Task.Run</c> is its .NET twin.
        /// </summary>
        [TestMethod]
        public void TaskRun_InheritsTheCallersOptions()
        {
            string worker;
            using (np.printoptions(precision: 2))
                worker = Task.Run(Show).GetAwaiter().GetResult();

            worker.Should().Be("[1.23]");
        }

        /// <summary>
        ///     NumPy: an <c>await</c> continuation runs in the task's copied context, so <c>[1.23]</c> under precision 2,
        ///     whichever thread resumes it here (the continuation after <c>Task.Yield</c> runs on the thread pool).
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        public async Task AwaitContinuation_KeepsTheOptions()
        {
            string resumed;
            using (np.printoptions(precision: 2))
            {
                await Task.Yield();
                resumed = Show();
            }

            resumed.Should().Be("[1.23]");
        }

        /// <summary>
        ///     NumPy: a worker's own <c>np.set_printoptions(precision=3)</c> changes only the worker's context; the starting
        ///     thread still prints <c>[1.23456789]</c>.
        /// </summary>
        [TestMethod]
        public void WorkerSetting_StaysInTheWorker()
        {
            string worker = null;
            OnNewThread(() =>
            {
                np.set_printoptions(precision: 3);
                worker = Show();
            });

            worker.Should().Be("[1.235]");
            Show().Should().Be("[1.23456789]", "the worker's setting never reaches the thread that started it");
        }

        /// <summary>
        ///     A scope restores the options it replaced in the same flow, also when the flow resumed on another thread
        ///     inside it.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        public async Task Scope_RestoresThePreviousOptions()
        {
            string outer, inner, after;
            using (np.printoptions(precision: 4))
            {
                using (np.printoptions(precision: 2))
                {
                    await Task.Yield();
                    inner = Show();
                }

                outer = Show();
            }

            after = Show();
            inner.Should().Be("[1.23]");
            outer.Should().Be("[1.2346]");
            after.Should().Be("[1.23456789]");
        }

        /// <summary>
        ///     The one place .NET has no Python twin: the thread pool serves both <c>asyncio.to_thread</c> (which copies the
        ///     context) and a <c>concurrent.futures.ThreadPoolExecutor</c> (which does not), so a queued work item inherits
        ///     the options, where NumPy's executor worker prints <c>[1.23456789]</c>. A caller that wants the defaults queues
        ///     without the execution context (<see cref="ThreadPool.UnsafeQueueUserWorkItem(WaitCallback, object)"/>).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void ThreadPoolWorkItem_InheritsWhereThreadPoolExecutorStartsFresh()
        {
            string queued = null, unsafeQueued = null;
            using (var done = new ManualResetEventSlim())
            using (var doneUnsafe = new ManualResetEventSlim())
            using (np.printoptions(precision: 2))
            {
                ThreadPool.QueueUserWorkItem(_ =>
                {
                    queued = Show();
                    done.Set();
                });
                ThreadPool.UnsafeQueueUserWorkItem(_ =>
                {
                    unsafeQueued = Show();
                    doneUnsafe.Set();
                }, null);
                done.Wait();
                doneUnsafe.Wait();
            }

            queued.Should().Be("[1.23]", "a pool work item inherits, as asyncio.to_thread does (NumPy's executor: [1.23456789])");
            unsafeQueued.Should().Be("[1.23456789]", "without the execution context a work item starts at the defaults");
        }
    }
}
