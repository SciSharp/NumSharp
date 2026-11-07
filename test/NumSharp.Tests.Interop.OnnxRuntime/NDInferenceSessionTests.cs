using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     <see cref="NDInferenceSession"/> (the owner's decision of 2026-11-07): an <see cref="InferenceSession"/> whose
    ///     Dispose releases the native session exactly once, however many threads call it at once, and only after every
    ///     run of this package's verbs on it returned; a run after the Dispose began is refused. Measured on a plain
    ///     InferenceSession (ORT 1.31.1): two Disposes corrupted the heap, and a Dispose during the bridge's Run was an
    ///     access violation in RunImpl.
    /// </summary>
    /// <remarks>
    ///     The release count is <c>NDInferenceSession.Releases</c>, process-wide: the suite runs without parallelism and
    ///     every session a test makes is disposed in the test. A defect here is usually a crash of the test host.
    /// </remarks>
    [TestClass]
    public class NDInferenceSessionTests : OnnxTestBase
    {
        /// <summary>Races per test.</summary>
        private const int Trials = 100;

        /// <summary>One shared set of single-threaded options (what <see cref="OnnxTestBase.OpenSession"/> uses), for the sessions these tests make.</summary>
        private static readonly Lazy<SessionOptions> OneThread = new(() => new SessionOptions { IntraOpNumThreads = 1, InterOpNumThreads = 1 });

        /// <summary>Two intra-op threads: the least an asynchronous run accepts.</summary>
        private static readonly Lazy<SessionOptions> TwoThreads = new(() => new SessionOptions { IntraOpNumThreads = 2, InterOpNumThreads = 1 });

        /// <summary>
        ///     All eight of ONNX Runtime's constructors open the model, the session runs, and a double Dispose releases it
        ///     once.
        /// </summary>
        [TestMethod]
        public void EveryConstructor_OpensTheModel_AndADoubleDisposeReleasesOnce()
        {
            string path = Model("identity_float32");
            byte[] bytes = File.ReadAllBytes(path);
            using var weights = new PrePackedWeightsContainer();
            var factories = new Func<NDInferenceSession>[]
            {
                () => new NDInferenceSession(path),
                () => new NDInferenceSession(path, weights),
                () => new NDInferenceSession(path, OneThread.Value),
                () => new NDInferenceSession(path, OneThread.Value, weights),
                () => new NDInferenceSession(bytes),
                () => new NDInferenceSession(bytes, weights),
                () => new NDInferenceSession(bytes, OneThread.Value),
                () => new NDInferenceSession(bytes, OneThread.Value, weights),
            };
            using NDArray src = np.full(new Shape(4, 4), 2f);
            long releases = NDInferenceSession.Releases;
            foreach (Func<NDInferenceSession> open in factories)
            {
                NDInferenceSession session = open();
                using (NDArray y = session.Run(src))
                    y.GetSingle(3, 3).Should().Be(2f);
                session.Dispose();
                session.Dispose();
                session.IsDisposed.Should().BeTrue();
            }
            (NDInferenceSession.Releases - releases).Should().Be(factories.Length, "one release per session, the second Dispose does nothing");
        }

        /// <summary>
        ///     Eight threads dispose one session at the same instant, 100 times: each session is released exactly once. ORT's
        ///     own Dispose freed its name buffers once per thread that passed its unguarded flag (heap corruption).
        /// </summary>
        [TestMethod]
        public void EightThreadsDisposeOneSession_ReleaseItOnce()
        {
            long releases = NDInferenceSession.Releases;
            for (int trial = 0; trial < Trials; trial++)
            {
                var session = new NDInferenceSession(Model("identity_float32"), OneThread.Value);
                OrtValueArcTests.RunTogether(8, _ => session.Dispose()).Should().OnlyContain(e => e == null, "a Dispose never throws");
            }
            (NDInferenceSession.Releases - releases).Should().Be(Trials);
        }

        /// <summary>
        ///     Eight threads each run their own session, then dispose a neighbour's (possibly mid-run) and their own (possibly
        ///     already disposed): every run returns the right values or is refused, and every session is released once.
        /// </summary>
        [TestMethod]
        public void ManySessions_DisposedByTheirThreadAndANeighbour_EachReleasedOnce()
        {
            using NDArray src = np.full(new Shape(8, 8), 2f);
            long releases = NDInferenceSession.Releases;
            int wrong = 0;
            for (int trial = 0; trial < 25; trial++)
            {
                var sessions = new NDInferenceSession[8];
                for (int i = 0; i < sessions.Length; i++)
                    sessions[i] = new NDInferenceSession(Model("identity_float32"), OneThread.Value);
                Exception[] errors = OrtValueArcTests.RunTogether(8, k =>
                {
                    try
                    {
                        using NDArray y = sessions[k].Run(src);
                        if (y.GetSingle(7, 7) != 2f)
                            Interlocked.Increment(ref wrong);
                    }
                    finally
                    {
                        sessions[(k + 1) % sessions.Length].Dispose();
                        sessions[k].Dispose();
                    }
                });
                OrtValueArcTests.CountRefusals(errors);
            }
            wrong.Should().Be(0);
            (NDInferenceSession.Releases - releases).Should().Be(25 * 8, "every session released once, whoever disposed it first");
        }

        /// <summary>
        ///     One thread runs the bridge's Run in a loop while another disposes the session: every run returns the right
        ///     values or is refused, and the session is released once - after the run in flight.
        /// </summary>
        [TestMethod]
        public void DisposedDuringTheBridgesRuns_TheRunInFlightFinishes_ThenOneRelease()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            long releases = NDInferenceSession.Releases;
            var random = new Random(21);
            int wrong = 0, refused = 0;
            for (int trial = 0; trial < Trials; trial++)
            {
                var session = new NDInferenceSession(Model("identity_float32"), OneThread.Value);
                int delay = random.Next(0, 200_000);
                Exception[] errors = OrtValueArcTests.RunTogether(2, side =>
                {
                    if (side == 1)
                    {
                        OrtValueArcTests.Delay(delay);
                        session.Dispose();
                        return;
                    }
                    for (int i = 0; i < 50; i++)
                    {
                        using NDArray y = session.Run(src);
                        if (y.GetSingle(0, 0) != 2f || y.GetSingle(63, 63) != 2f)
                            Interlocked.Increment(ref wrong);
                    }
                });
                refused += OrtValueArcTests.CountRefusals(errors);
                session.Dispose();   // the loop may have finished before the disposer: a second Dispose, which does nothing
            }
            wrong.Should().Be(0);
            refused.Should().BeGreaterThan(0, "some Disposes land before the runner is done (otherwise the test raced nothing)");
            (NDInferenceSession.Releases - releases).Should().Be(Trials, "one release per session, after its last run");
        }

        /// <summary>
        ///     The rule itself, deterministically: with a run in flight, Dispose marks the session disposed and refuses new
        ///     runs but leaves the native session alive; the run's end releases it, once. A run refused because the Dispose
        ///     began gives itself back without a release of its own.
        /// </summary>
        [TestMethod]
        public void Dispose_WaitsForTheRunInFlight_AndTheLastRunReleases()
        {
            using NDArray src = np.full(new Shape(4, 4), 2f);
            long releases = NDInferenceSession.Releases;
            var session = new NDInferenceSession(Model("identity_float32"), OneThread.Value);
            session.TryEnterRun().Should().BeTrue("a run in flight");

            session.Dispose();
            session.IsDisposed.Should().BeTrue();
            NDInferenceSession.Releases.Should().Be(releases, "the run in flight keeps the native session");
            Action another = () => session.Run(src);
            another.Should().Throw<ObjectDisposedException>().WithMessage("Run: the NDInferenceSession has been disposed.*");
            NDInferenceSession.Releases.Should().Be(releases, "the refused run released nothing: the first run is still in flight");
            session.TryEnterRun().Should().BeFalse("a new run is refused once Dispose began");

            session.ExitRun(onOrtThread: false);
            NDInferenceSession.Releases.Should().Be(releases + 1, "the last run released the session, on its own thread");
            session.Dispose();
            NDInferenceSession.Releases.Should().Be(releases + 1, "and nothing releases it again");
        }

        /// <summary>
        ///     An asynchronous run in flight when the session is disposed completes with the right values; the release it
        ///     then makes runs on the .NET thread pool, not on ONNX Runtime's intra-op thread that completes the run (where
        ///     releasing the session never returns), and a run after it is refused.
        /// </summary>
        [TestMethod]
        public void AsyncRunInFlight_CompletesBeforeTheRelease_WhichRunsOnThePool()
        {
            if (!InferenceSessionExtensions.IsRunAsyncSupported)
                Assert.Inconclusive($"ONNX Runtime {OrtEnv.Instance().GetVersionString()}: the asynchronous run needs 1.30.0 or newer. The suite's current-ORT leg runs this test.");
            using NDArray src = np.full(new Shape(64, 64), 2f);
            long releases = NDInferenceSession.Releases;
            for (int trial = 0; trial < 20; trial++)
            {
                var session = new NDInferenceSession(Model("identity_float32"), TwoThreads.Value);
                string input = session.InputNames[0], output = session.OutputNames[0];
                using NDArray y = np.zeros(new Shape(64, 64), NPTypeCode.Single);
                Task run = session.RunIntoAsync(new Dictionary<string, NDArray> { [input] = src }, new Dictionary<string, NDArray> { [output] = y });
                session.Dispose();   // usually while the run is in flight; it defers the release when it is
                run.Wait(TimeSpan.FromSeconds(30)).Should().BeTrue("the run completes although its session was disposed meanwhile");
                y.GetSingle(63, 63).Should().Be(2f);
                WaitFor(() => NDInferenceSession.Releases - releases == trial + 1, 10_000)
                    .Should().BeTrue("the release after the run happens on the thread pool and returns (on ORT's own thread it never would)");
                Action again = () => session.RunIntoAsync(new Dictionary<string, NDArray> { [input] = src }, new Dictionary<string, NDArray> { [output] = y });
                again.Should().Throw<ObjectDisposedException>().WithMessage("RunIntoAsync: the NDInferenceSession has been disposed.*");
            }
        }

        /// <summary>
        ///     It IS an InferenceSession: ONNX Runtime's own Run works on it, and after Dispose the bridge's verbs refuse it
        ///     with the NDInferenceSession message (a plain session keeps ONNX Runtime's disposal check).
        /// </summary>
        [TestMethod]
        public void IsAnInferenceSession_OrtsOwnRunWorks_AndADisposedOneIsRefused()
        {
            using NDArray src = np.full(new Shape(4, 4), 2f);
            using OrtValue input = src.ToOrtValue();
            var session = new NDInferenceSession(Model("identity_float32"), OneThread.Value);
            InferenceSession plain = session;
            using (var options = new RunOptions())
            using (IDisposableReadOnlyCollection<OrtValue> results = plain.Run(options, new[] { plain.InputNames[0] }, new[] { input }, new[] { plain.OutputNames[0] }))
                results[0].GetTensorDataAsSpan<float>()[15].Should().Be(2f);

            session.Dispose();
            using NDArray buffer = np.zeros(new Shape(4, 4), NPTypeCode.Single);
            Action run = () => plain.Run(src);
            run.Should().Throw<ObjectDisposedException>().WithMessage("Run: the NDInferenceSession has been disposed.*");
            Action runInto = () => plain.RunInto(new Dictionary<string, NDArray> { [plain.InputNames[0]] = src },
                new Dictionary<string, NDArray> { [plain.OutputNames[0]] = buffer });
            runInto.Should().Throw<ObjectDisposedException>().WithMessage("RunInto: the NDInferenceSession has been disposed.*");
        }
    }
}
