using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
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
    ///     The asynchronous forms of the <c>Run</c> tier, <c>RunIntoAsync</c> and <c>RunAsync</c>, on ONNX Runtime's own
    ///     native asynchronous run.
    ///     <para><b>Two legs.</b> The forms need ONNX Runtime 1.30.0 or newer. On the floor version (1.16.0) the first
    ///     test checks the refusal and the others report Inconclusive with the reason; the suite's ORT 1.31.0 leg runs
    ///     them all.</para>
    ///     <para><b>Sessions.</b> An asynchronous run needs 2 or more intra-op threads, so these tests open their own
    ///     sessions (<see cref="OpenAsyncSession"/>); the suite's shared options use 1.</para>
    ///     <para><b>No test may hang.</b> The defects these tests guard against are deadlocks, so every wait has a
    ///     deadline and every test a timeout.</para>
    /// </summary>
    [TestClass]
    public class SessionRunAsyncTests : OnnxTestBase
    {
        /// <summary>The deadline of a wait that must end at once when the code is right.</summary>
        private static readonly TimeSpan Patience = TimeSpan.FromSeconds(20);

        private static readonly Lazy<SessionOptions> TwoThreads = new(() => new SessionOptions
        {
            IntraOpNumThreads = 2,
            InterOpNumThreads = 1,
            // FATAL: several tests make ONNX Runtime fail a run on purpose, and it logs each failure at ERROR.
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL,
        });

        /// <summary>A session over a committed model with 2 intra-op threads, the least an asynchronous run accepts.</summary>
        /// <param name="model">The model's name without the extension.</param>
        /// <returns>A new session the caller disposes.</returns>
        private static InferenceSession OpenAsyncSession(string model) => new(Model(model), TwoThreads.Value);

        /// <summary>Reports the test Inconclusive, with the reason, when the loaded ONNX Runtime cannot run asynchronously.</summary>
        private static void RequireAsyncRuns()
        {
            if (!InferenceSessionExtensions.IsRunAsyncSupported)
                Assert.Inconclusive($"ONNX Runtime {OrtEnv.Instance().GetVersionString()}: the asynchronous run needs 1.30.0 or newer. The suite's ORT 1.31.0 leg runs this test.");
        }

        /// <summary>A C-contiguous owning float32 matrix with small whole values, so every product of two of them is exact in any order of summation.</summary>
        /// <param name="rows">The row count.</param>
        /// <param name="columns">The column count.</param>
        /// <param name="seed">Shifts the pattern, so two matrices differ.</param>
        /// <returns>A new owning array.</returns>
        private static NDArray SmallWholes(int rows, int columns, int seed)
        {
            var values = new float[rows * columns];
            for (int i = 0; i < values.Length; i++)
                values[i] = (i * 7 + seed) % 4;
            NDArray matrix = np.array(values);
            matrix.resize(rows, columns);   // in place: the array keeps owning its buffer
            return matrix;
        }

        /// <summary>The cast_fanout model's output name for a dtype (<c>as_float32</c>, <c>as_bool</c>, …).</summary>
        /// <param name="code">A dtype that crosses zero-copy.</param>
        /// <returns>The output name.</returns>
        private static string FanoutName(NPTypeCode code) => "as_" + IdentityModel(code).Substring("identity_".Length);

        // ---- the version rule --------------------------------------------------------------------------

        /// <summary>
        ///     Both forms need ONNX Runtime 1.30.0, the first whose managed code pins the arrays the native run reads.
        ///     On an older one they throw <see cref="NotSupportedException"/> from the call itself (not through a task),
        ///     before any other check of the arguments but the null checks, and prepare nothing.
        ///     <see cref="InferenceSessionExtensions.IsRunAsyncSupported"/> tells which applies.
        /// </summary>
        [TestMethod]
        [Timeout(120_000)]
        public void AsyncForms_AreRefusedBelowOrt130_AndSupportedFromIt()
        {
            Version managed = typeof(InferenceSession).Assembly.GetName().Version ?? new Version(0, 0);
            bool expected = managed >= new Version(1, 30);
            InferenceSessionExtensions.IsRunAsyncSupported.Should().Be(expected, $"the managed ONNX Runtime assembly reports {managed}");

            using InferenceSession session = OpenAsyncSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            using NDArray y = np.zeros(new Shape(3), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "X", x } };
            var outputs = new Dictionary<string, NDArray> { { "Y", y } };

            // a null argument is a null argument on every version
            new Action(() => ((InferenceSession)null).RunIntoAsync(inputs, outputs)).Should().Throw<ArgumentNullException>();
            new Action(() => session.RunIntoAsync(null, outputs)).Should().Throw<ArgumentNullException>();
            new Action(() => session.RunIntoAsync(inputs, null)).Should().Throw<ArgumentNullException>();
            new Action(() => ((InferenceSession)null).RunAsync(inputs)).Should().Throw<ArgumentNullException>();
            new Action(() => session.RunAsync(null)).Should().Throw<ArgumentNullException>();
            if (expected)
                return;   // what the forms do is the subject of every other test of this class

            new Action(() => session.RunIntoAsync(inputs, outputs)).Should().Throw<NotSupportedException>()
                .WithMessage("RunIntoAsync:*1.30.0*#32015*synchronous RunInto*IsRunAsyncSupported*");
            new Action(() => session.RunAsync(inputs)).Should().Throw<NotSupportedException>()
                .WithMessage("RunAsync:*1.30.0*#32015*synchronous Run,*");
            // the refusal is about the loaded library, so it comes before the checks of the other arguments
            new Action(() => session.RunIntoAsync(inputs, outputs, casting: "bogus")).Should().Throw<NotSupportedException>();
            new Action(() => session.RunAsync(inputs, new[] { "nope" })).Should().Throw<NotSupportedException>();
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a refused call prepares nothing");
            y.ToArray<float>().Should().Equal(new float[] { 0, 0, 0 });
        }

        // ---- RunIntoAsync ------------------------------------------------------------------------------

        /// <summary>
        ///     <c>RunIntoAsync</c> writes the caller's arrays as <c>RunInto</c> does, for a model with two inputs, and
        ///     every handle of the run is released before the task completes: right after the <c>await</c> no export
        ///     is live and the arrays hold no reference of the run.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task RunIntoAsync_WritesTheCallersArrays_AndReleasesTheRunBeforeItCompletes()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("matmul_f32");
            using NDArray a = SmallWholes(2, 3, 1);
            using NDArray b = SmallWholes(3, 4, 2);
            using NDArray c = np.zeros(new Shape(2, 4), typeof(float));
            using NDArray expected = np.zeros(new Shape(2, 4), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "A", a }, { "B", b } };
            session.RunInto(inputs, new Dictionary<string, NDArray> { { "C", expected } });

            await session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "C", c } });

            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the handles are released before the task completes");
            a.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("the run gave its reference on the input back");
            c.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("and its reference on the output");
            BytesOf(c).Should().Equal(BytesOf(expected));
            expected.ToArray<float>().Should().Contain(v => v != 0f, "the model computed something");
        }

        /// <summary>Every dtype that crosses zero-copy runs asynchronously into a pre-allocated array of that dtype, 0-d and empty tensors included.</summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task RunIntoAsync_EveryDtype_RoundTripsThroughItsIdentityModel()
        {
            RequireAsyncRuns();
            foreach (NPTypeCode code in ZeroCopyDtypes)
            {
                using InferenceSession session = OpenAsyncSession(IdentityModel(code));
                foreach (int[] shape in new[] { new[] { 2, 3 }, Array.Empty<int>(), new[] { 0, 4 } })
                {
                    using NDArray x = Arange(code, shape);
                    using NDArray y = np.zeros(x.Shape.AsClean(), code);
                    await session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", y } });
                    BytesOf(y).Should().Equal(BytesOf(x), $"{code} {x.Shape}");
                }
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     ONNX Runtime completes its own task on an intra-op thread of the session, and a continuation that runs
        ///     there and disposes the session never returns (measured). The task of the bridge runs a caller's
        ///     continuation on the thread pool, or inline on the caller's thread when the run had already ended, never
        ///     on ONNX Runtime's thread; the session is then disposed from the continuation.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task Continuation_DoesNotRunOnOrtsThread_SoItCanDisposeTheSession()
        {
            RequireAsyncRuns();
            InferenceSession session = OpenAsyncSession("matmul_f32");
            try
            {
                // A product large enough (216 million multiply-adds) for the run to be in flight at the await below.
                using NDArray a = SmallWholes(600, 600, 1);
                using NDArray b = SmallWholes(600, 600, 2);
                using NDArray c = np.zeros(new Shape(600, 600), typeof(float));
                int caller = Environment.CurrentManagedThreadId;

                await session.RunIntoAsync(new Dictionary<string, NDArray> { { "A", a }, { "B", b } }, new Dictionary<string, NDArray> { { "C", c } });

                bool onPool = Thread.CurrentThread.IsThreadPoolThread;
                bool inline = Environment.CurrentManagedThreadId == caller;
                (onPool || inline).Should().BeTrue(
                    "the continuation must run on the thread pool (or inline, when the run had ended), not on a thread of ONNX Runtime's intra-op pool");
                c.GetSingle(0, 0).Should().BeGreaterThan(0f);

                // The reason for the rule: on ONNX Runtime's own thread this call waits for that thread to end.
                session.Dispose();
                session = null;
            }
            finally
            {
                session?.Dispose();
            }
        }

        /// <summary>
        ///     ONNX Runtime awaits its completion source without <c>ConfigureAwait(false)</c>, so it captures the caller's
        ///     synchronization context: a thread that owns a context and blocks on the task waited forever for ORT's own
        ///     continuation (measured). The bridge hides the context from ORT for the call, posts nothing to it, and
        ///     leaves it current afterwards.
        /// </summary>
        [TestMethod]
        [Timeout(120_000)]
        public void BlockingWait_OnAThreadThatOwnsASynchronizationContext_IsReleased()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            using NDArray y = np.zeros(new Shape(3), typeof(float));
            var context = new NeverPumpedContext();
            bool finished = false, contextKept = false;
            Exception error = null;

            var thread = new Thread(() =>
            {
                SynchronizationContext.SetSynchronizationContext(context);
                try
                {
                    Task task = session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", y } });
                    contextKept = ReferenceEquals(SynchronizationContext.Current, context);
                    finished = task.Wait(Patience);
                }
                catch (Exception e)
                {
                    error = e;
                }
            });
            thread.Start();
            thread.Join();

            error.Should().BeNull();
            contextKept.Should().BeTrue("the caller's context is current again when the call returns");
            finished.Should().BeTrue("nothing the run needs waits for the caller's context");
            context.Posts.Should().Be(0, "neither ONNX Runtime nor the bridge posts to the caller's context");
            y.ToArray<float>().Should().Equal(new float[] { 0, 1, 2 });
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     The same hazard through a task scheduler: an <c>await</c> without a synchronization context captures the
        ///     current task scheduler when it is not the default one. From inside a task of a single-thread scheduler,
        ///     a blocking wait on the run would wait for that scheduler to run ORT's continuation on the very thread
        ///     that waits. The bridge makes ORT's call from the default scheduler.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task BlockingWait_InsideATaskOfASingleThreadScheduler_IsReleased()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            using NDArray y = np.zeros(new Shape(3), typeof(float));
            using var scheduler = new SingleThreadScheduler();

            (bool OnScheduler, bool Finished) result = await Task.Factory.StartNew(() =>
            {
                bool onScheduler = ReferenceEquals(TaskScheduler.Current, scheduler);
                Task task = session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", y } });
                return (onScheduler, task.Wait(Patience));
            }, CancellationToken.None, TaskCreationOptions.None, scheduler);

            result.OnScheduler.Should().BeTrue("the call was made from a task of the custom scheduler");
            result.Finished.Should().BeTrue("ONNX Runtime's continuation must not be queued to the caller's scheduler");
            y.ToArray<float>().Should().Equal(new float[] { 0, 1, 2 });
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     The run holds its own reference on every buffer it reads and writes, so the caller may dispose the input
        ///     and output arrays while the run is in flight: ONNX Runtime keeps reading and writing live memory, and the
        ///     result is complete in the output's buffer (read here through a second view of it).
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task InputsAndOutputs_MayBeDisposed_WhileTheRunIsInFlight()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("matmul_f32");
            NDArray a = SmallWholes(600, 600, 1);
            NDArray b = SmallWholes(600, 600, 2);
            NDArray c = np.zeros(new Shape(600, 600), typeof(float));
            using NDArray keep = c["..."];   // a second view: the only way to read the buffer once c is disposed
            using NDArray expected = np.zeros(new Shape(600, 600), typeof(float));
            session.RunInto(new Dictionary<string, NDArray> { { "A", a }, { "B", b } }, new Dictionary<string, NDArray> { { "C", expected } });

            Task task = session.RunIntoAsync(new Dictionary<string, NDArray> { { "A", a }, { "B", b } }, new Dictionary<string, NDArray> { { "C", c } });
            a.Dispose();
            b.Dispose();
            c.Dispose();
            await task;

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            BytesOf(keep).Should().Equal(BytesOf(expected), "the run wrote the whole result into memory it kept alive");
        }

        /// <summary>The release of a run does not depend on the caller: a task that is dropped, never awaited, still ends its run and gives every handle back.</summary>
        [TestMethod]
        [Timeout(120_000)]
        public void ADroppedTask_StillReleasesTheRun()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            using NDArray y = np.zeros(new Shape(3), typeof(float));
            _ = session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", y } });
            WaitFor(() => NDArrayOnnxInterop.LiveExports == 0 && x.Storage.InternalArray.IsUniquelyReferenced, 20_000).Should().BeTrue();
            y.ToArray<float>().Should().Equal(new float[] { 0, 1, 2 });
        }

        /// <summary>
        ///     What the method can check, it checks before the run starts, and it throws from the call itself: a caller
        ///     sees a usage error at the call site, not in a faulted task. Every handle prepared before the refusal is
        ///     released.
        /// </summary>
        [TestMethod]
        [Timeout(120_000)]
        public void ArgumentErrors_ThrowFromTheCall_NotThroughTheTask()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("fixed_shape_f32");
            using NDArray data = Arange(NPTypeCode.Single, 1, 3, 4, 4);
            using NDArray relu = np.zeros(new Shape(1, 3, 4, 4), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "data", data } };
            var outputs = new Dictionary<string, NDArray> { { "relu", relu } };

            new Action(() => session.RunIntoAsync(inputs, outputs, casting: "bogus")).Should().Throw<ValueError>();
            new Action(() => session.RunIntoAsync(inputs, outputs, mode: (OnnxCodecMode)42)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => session.RunIntoAsync(inputs, new Dictionary<string, NDArray>())).Should().Throw<ArgumentException>().WithMessage("*no output was given*");
            new Action(() => session.RunIntoAsync(new Dictionary<string, NDArray> { { "nope", data } }, outputs)).Should().Throw<ArgumentException>().WithMessage("*no input named 'nope'*");
            new Action(() => session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "nope", relu } })).Should().Throw<ArgumentException>().WithMessage("*no output named 'nope'*");
            new Action(() => session.RunIntoAsync(new Dictionary<string, NDArray> { { "data", null } }, outputs)).Should().Throw<ArgumentNullException>();

            using (NDArray wide = data.astype(NPTypeCode.Double))
                new Action(() => session.RunIntoAsync(new Dictionary<string, NDArray> { { "data", wide } }, outputs)).Should().Throw<InvalidCastException>("float64 into float32 is not a safe cast");
            using (NDArray wrongShape = Arange(NPTypeCode.Single, 1, 3, 4, 5))
                new Action(() => session.RunIntoAsync(new Dictionary<string, NDArray> { { "data", wrongShape } }, outputs)).Should().Throw<ArgumentException>().WithMessage("*'data'*");
            using (NDArray wrongType = np.zeros(new Shape(1, 3, 4, 4), typeof(double)))
                new Action(() => session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "relu", wrongType } })).Should().Throw<InvalidOperationException>();
            using (NDArray transposed = Arange(NPTypeCode.Single, 4, 4, 3, 1).T)
                new Action(() => session.RunIntoAsync(new Dictionary<string, NDArray> { { "data", transposed } }, outputs, mode: OnnxCodecMode.View))
                    .Should().Throw<InvalidOperationException>().WithMessage("*'data'*not C-contiguous*");

            // an input and an output in one buffer: the strict mode refuses, and the message names this verb
            new Action(() => session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "relu", data } }, mode: OnnxCodecMode.View))
                .Should().Throw<InvalidOperationException>().WithMessage("RunIntoAsync(mode: OnnxCodecMode.View)*overlap*");

            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a refused call releases what it prepared");
            relu.ToArray<float>().Should().OnlyContain(v => v == 0f, "no run started");
        }

        /// <summary>
        ///     An input and an output in one buffer under the default mode: the input is fed as a copy, as in the
        ///     synchronous form, so the result is the one computed from the values before the run.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task OutputThatOverlapsAnInput_IsComputedFromTheValuesBeforeTheRun()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("matmul_f32");
            using NDArray a = SmallWholes(64, 64, 1);
            using NDArray b = SmallWholes(64, 64, 2);
            using NDArray expected = np.zeros(new Shape(64, 64), typeof(float));
            session.RunInto(new Dictionary<string, NDArray> { { "A", a }, { "B", b } }, new Dictionary<string, NDArray> { { "C", expected } });

            await session.RunIntoAsync(new Dictionary<string, NDArray> { { "A", a }, { "B", b } }, new Dictionary<string, NDArray> { { "C", a } });
            BytesOf(a).Should().Equal(BytesOf(expected), "A @ B written into A");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     A failure inside ONNX Runtime faults the task with ORT's own exception, and the run is released all the
        ///     same: an output of the wrong shape, a run terminated through its <see cref="RunOptions"/> (the way to
        ///     cancel), and a session with a single intra-op thread, which ONNX Runtime refuses to run asynchronously.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task AFailureInsideOnnxRuntime_FaultsTheTask_AndReleasesTheRun()
        {
            RequireAsyncRuns();
            using NDArray x = Arange(NPTypeCode.Single, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            using (InferenceSession session = OpenAsyncSession("identity_float32"))
            {
                using NDArray tooLong = np.zeros(new Shape(4), typeof(float));
                Task wrongShape = session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "Y", tooLong } });
                (await Awaiting(wrongShape)).Should().BeOfType<OnnxRuntimeException>("the model declares no shape, so only ORT can see that (4,) is wrong");
                tooLong.ToArray<float>().Should().Equal(new float[] { 0, 0, 0, 0 });
                NDArrayOnnxInterop.LiveExports.Should().Be(0, "a failed run releases its handles before its task completes");
                x.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue();

                using NDArray y = np.zeros(new Shape(3), typeof(float));
                using var terminate = new RunOptions { Terminate = true };
                Task terminated = session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "Y", y } }, terminate);
                Exception stopped = await Awaiting(terminated);
                stopped.Should().BeOfType<OnnxRuntimeException>();
                stopped.Message.Should().Contain("terminate");
                NDArrayOnnxInterop.LiveExports.Should().Be(0);
            }

            using (InferenceSession oneThread = OpenSession("identity_float32"))   // the suite's shared options: 1 intra-op thread
            {
                using NDArray y = np.zeros(new Shape(3), typeof(float));
                Task refused = oneThread.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "Y", y } });
                Exception error = await Awaiting(refused);
                error.Should().BeOfType<OnnxRuntimeException>();
                error.Message.Should().Contain("intra op thread pool");
                NDArrayOnnxInterop.LiveExports.Should().Be(0);
                y.ToArray<float>().Should().Equal(new float[] { 0, 0, 0 });
            }
        }

        /// <summary>Many runs in flight on one session at once, each with its own arrays: every result is right and nothing stays pinned.</summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task ManyRunsInFlight_OnOneSession_EachWritesItsOwnArrays()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("identity_float32");
            const int Runs = 64;
            var xs = new NDArray[Runs];
            var ys = new NDArray[Runs];
            var tasks = new Task[Runs];
            for (int i = 0; i < Runs; i++)
            {
                xs[i] = np.array(new float[] { i, i + 1, i + 2 });
                ys[i] = np.zeros(new Shape(3), typeof(float));
                tasks[i] = session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", xs[i] } }, new Dictionary<string, NDArray> { { "Y", ys[i] } });
            }

            await Task.WhenAll(tasks);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            for (int i = 0; i < Runs; i++)
            {
                ys[i].ToArray<float>().Should().Equal(new float[] { i, i + 1, i + 2 }, $"run {i}");
                xs[i].Dispose();
                ys[i].Dispose();
            }
        }

        /// <summary>
        ///     <see cref="OnnxCodecMode.Copy"/> feeds a copy of every input taken during the call, so the caller's array
        ///     is free at once (no reference of the run on it) and may change while the run is in flight: the result is
        ///     the one of the values at the call.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task CopyMode_FeedsASnapshot_SoTheInputMayChangeAtOnce()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("matmul_f32");
            // A product large enough for the run to be in flight while the inputs are overwritten below.
            using NDArray a = SmallWholes(600, 600, 1);
            using NDArray b = SmallWholes(600, 600, 2);
            using NDArray c = np.zeros(new Shape(600, 600), typeof(float));
            using NDArray expected = np.zeros(new Shape(600, 600), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "A", a }, { "B", b } };
            session.RunInto(inputs, new Dictionary<string, NDArray> { { "C", expected } });

            Task task = session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "C", c } }, mode: OnnxCodecMode.Copy);
            a.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("the run reads its own copy, not the caller's buffer");
            b.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue();
            a.fill(-1f);
            b.fill(-1f);
            await task;

            BytesOf(c).Should().Equal(BytesOf(expected), "the product of the values at the call");
        }

        /// <summary>An output that is a fed input passed through is completed by the bridge after an asynchronous run too (ONNX Runtime does not write it).</summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task RunIntoAsync_OutputThatIsAFedInput_ReceivesTheFedValues()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("passthrough");
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            using NDArray outX = np.array(new float[] { -777, -777, -777 });
            using NDArray outY = np.array(new float[] { -777, -777, -777 });
            await session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "X", outX }, { "Y", outY } });
            outX.ToArray<float>().Should().Equal(new float[] { 1, 2, 3 });
            outY.ToArray<float>().Should().Equal(new float[] { -1, -2, -3 });

            using NDArray tooLong = np.zeros(new Shape(4), typeof(float));
            new Action(() => session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "X", tooLong }, { "Y", outY } }))
                .Should().Throw<ArgumentException>().WithMessage("*'X'*passed through*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        // ---- RunAsync ----------------------------------------------------------------------------------

        /// <summary>
        ///     <c>RunAsync</c> allocates every output from the model's declared shape and dtype: one model with an output
        ///     of each of the 12 element types, each <c>[2, 3]</c>. Every result is a new owning C-contiguous array of
        ///     the declared dtype, with the values ONNX Runtime's Cast produces (NumSharp's own cast is the reference).
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task RunAsync_AllocatesEveryDeclaredOutput_OfEveryDtype()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("cast_fanout");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);

            IReadOnlyDictionary<string, NDArray> results = await session.RunAsync(new Dictionary<string, NDArray> { { "X", x } });
            try
            {
                NDArrayOnnxInterop.LiveExports.Should().Be(0, "the results are plain arrays: no handle outlives the run");
                NDArrayOnnxInterop.LiveImports.Should().Be(0, "and no lease on ORT memory: ORT wrote NumSharp's own buffers");
                results.Count.Should().Be(ZeroCopyDtypes.Length);
                foreach (NPTypeCode code in ZeroCopyDtypes)
                {
                    NDArray got = results[FanoutName(code)];
                    using NDArray want = x.astype(code);
                    got.typecode.Should().Be(code);
                    got.shape.Should().Equal(2, 3);
                    got.flags.owndata.Should().BeTrue(code.ToString());
                    got.flags.c_contiguous.Should().BeTrue(code.ToString());
                    got.flags.writeable.Should().BeTrue(code.ToString());
                    BytesOf(got).Should().Equal(BytesOf(want), code.ToString());
                }
            }
            finally
            {
                foreach (NDArray array in results.Values)
                    array.Dispose();
            }
        }

        /// <summary>
        ///     The names asked are the names fetched: a subset, in any order, and a name asked more than once is fetched
        ///     once. A null name, no name and an unknown name are usage errors thrown from the call.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task RunAsync_FetchesTheNamesAsked_EachOnce()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("cast_fanout");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            IReadOnlyDictionary<string, NDArray> results = await session.RunAsync(inputs, new[] { "as_int64", "as_bool", "as_int64" });
            try
            {
                results.Keys.Should().BeEquivalentTo(new[] { "as_int64", "as_bool" });
                results["as_int64"].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
                results["as_bool"].ToArray<bool>().Should().Equal(false, true, true, true, true, true);
            }
            finally
            {
                foreach (NDArray array in results.Values)
                    array.Dispose();
            }

            new Action(() => session.RunAsync(inputs, new string[] { null })).Should().Throw<ArgumentNullException>();
            new Action(() => session.RunAsync(inputs, Array.Empty<string>())).Should().Throw<ArgumentException>().WithMessage("*no output name was given*");
            new Action(() => session.RunAsync(inputs, new[] { "nope" })).Should().Throw<ArgumentException>().WithMessage("*no output named 'nope'*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     An output that is a fed input passed through is allocated from the FED tensor (its declared shape may be
        ///     symbolic) and holds the fed values: a model input, and a fed overridable initializer next to an ordinary
        ///     output of a fixed shape.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task RunAsync_OutputThatIsAFedInput_TakesTheFedShapeAndValues()
        {
            RequireAsyncRuns();
            using (InferenceSession session = OpenAsyncSession("passthrough"))
            using (NDArray x = np.array(new float[] { 1, 2, 3, 4, 5 }))
            {
                // X is declared ['N']: only the fed tensor tells its shape.
                IReadOnlyDictionary<string, NDArray> results = await session.RunAsync(new Dictionary<string, NDArray> { { "X", x } }, new[] { "X" });
                using NDArray got = results["X"];
                got.ToArray<float>().Should().Equal(new float[] { 1, 2, 3, 4, 5 });
                got.flags.owndata.Should().BeTrue("a new array, not a view of the input");
                x.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue();
            }

            using (InferenceSession session = OpenAsyncSession("overridable_out"))
            using (NDArray x = np.array(new float[] { 1, 2, 3 }))
            using (NDArray w = np.array(new float[] { 100, 200, 300 }))
            {
                IReadOnlyDictionary<string, NDArray> fed = await session.RunAsync(new Dictionary<string, NDArray> { { "x", x }, { "w", w } });
                using (NDArray gotW = fed["w"])
                using (NDArray gotY = fed["y"])
                {
                    gotW.ToArray<float>().Should().Equal(new float[] { 100, 200, 300 }, "the fed weight, not the default");
                    gotY.ToArray<float>().Should().Equal(new float[] { 101, 202, 303 });
                }

                IReadOnlyDictionary<string, NDArray> byDefault = await session.RunAsync(new Dictionary<string, NDArray> { { "x", x } });
                using (NDArray gotW = byDefault["w"])
                using (NDArray gotY = byDefault["y"])
                {
                    gotW.ToArray<float>().Should().Equal(new float[] { 10, 20, 30 }, "the default weight, which ONNX Runtime copies");
                    gotY.ToArray<float>().Should().Equal(new float[] { 11, 22, 33 });
                }
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     An output whose array cannot be allocated before the run is refused from the call, with what to do
        ///     instead, and before an input is read (the refusal takes no buffer from the pool, and a released input is
        ///     not even looked at): a dimension fixed only at run time, a declaration without dimensions, a string
        ///     tensor, a sequence.
        /// </summary>
        [TestMethod]
        [Timeout(120_000)]
        public void RunAsync_RefusesAnOutputItCannotAllocate_BeforeItReadsAnInput()
        {
            RequireAsyncRuns();
            using (InferenceSession session = OpenAsyncSession("matmul_f32"))
            using (NDArray a = SmallWholes(2, 3, 1))
            using (NDArray b = SmallWholes(3, 4, 2))
            {
                var inputs = new Dictionary<string, NDArray> { { "A", a }, { "B", b } };
                new Action(() => session.RunAsync(inputs)).Should().Throw<NotSupportedException>()
                    .WithMessage("RunAsync: output 'C'*('M', 'N')*fixes only at run time*RunIntoAsync*");
                Pool.Takes(() => { try { session.RunAsync(inputs); } catch (NotSupportedException) { } })
                    .Should().Be(0, "the refusal is decided from the model's declarations alone");

                NDArray released = SmallWholes(2, 3, 1);
                released.Dispose();
                new Action(() => session.RunAsync(new Dictionary<string, NDArray> { { "A", released }, { "B", b } }))
                    .Should().Throw<NotSupportedException>("the output is refused before the released input is read");
            }

            using (InferenceSession session = OpenAsyncSession("identity_float32"))
            using (NDArray x = Arange(NPTypeCode.Single, 3))
                new Action(() => session.RunAsync(new Dictionary<string, NDArray> { { "X", x } })).Should().Throw<NotSupportedException>()
                    .WithMessage("RunAsync: output 'Y'*without dimensions*scalar or a tensor of unknown rank*RunIntoAsync*");

            using (InferenceSession session = OpenAsyncSession("string_io"))
            using (NDArray text = np.array(new[] { "a", "b" }))
                new Action(() => session.RunAsync(new Dictionary<string, NDArray> { { "X", text } })).Should().Throw<NotSupportedException>()
                    .WithMessage("RunAsync: output 'Y' is a string tensor*synchronous Run*");

            using (InferenceSession session = OpenAsyncSession("sequence_out"))
            using (NDArray x = Arange(NPTypeCode.Single, 3))
                new Action(() => session.RunAsync(new Dictionary<string, NDArray> { { "X", x } })).Should().Throw<NotSupportedException>()
                    .WithMessage("RunAsync: output 'seq'*not a tensor*");

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     An input error of <c>RunAsync</c> is thrown from the call, and the inputs that were already prepared are
        ///     released: with two inputs of which the second does not cast, no export of the first stays live and its
        ///     array holds no reference of the run.
        /// </summary>
        [TestMethod]
        [Timeout(120_000)]
        public void RunAsync_AnInputError_ReleasesTheInputsAlreadyPrepared()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("overridable_out");
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            using NDArray w64 = np.array(new double[] { 100, 200, 300 });
            // A dictionary enumerates in insertion order: x is prepared (and shared) before w is refused.
            var inputs = new Dictionary<string, NDArray> { { "x", x }, { "w", w64 } };

            new Action(() => session.RunAsync(inputs)).Should().Throw<InvalidCastException>().WithMessage("*'w'*Double*Single*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the handle of the first input is released by the refusal");
            x.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue();
            new Action(() => session.RunAsync(new Dictionary<string, NDArray> { { "nope", x } })).Should().Throw<ArgumentException>().WithMessage("*no input named 'nope'*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     A run that fails after <c>RunAsync</c> allocated its outputs disposes them: the caller never receives
        ///     them, so nobody else could. Measured on the buffer pool: a failed call gives back every buffer it took.
        /// </summary>
        [TestMethod]
        [Timeout(120_000)]
        public void RunAsync_AFailedRun_DisposesTheArraysItAllocated()
        {
            RequireAsyncRuns();
            using InferenceSession session = OpenAsyncSession("cast_fanout");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using var terminate = new RunOptions { Terminate = true };
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            long balance = Pool.Balance(() =>
            {
                Task<IReadOnlyDictionary<string, NDArray>> task = session.RunAsync(inputs, runOptions: terminate);
                new Action(() => task.GetAwaiter().GetResult()).Should().Throw<OnnxRuntimeException>();
            });
            balance.Should().Be(0, "the 12 output arrays of a failed run go back to the pool before its task completes");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        // ---- helpers -----------------------------------------------------------------------------------

        /// <summary>Awaits a task that must fail and returns its exception; fails the test when the task succeeds.</summary>
        /// <param name="task">The task.</param>
        /// <returns>The exception the await threw.</returns>
        private static async Task<Exception> Awaiting(Task task)
        {
            try
            {
                await task;
            }
            catch (Exception e)
            {
                return e;
            }

            Assert.Fail("the task completed without an error");
            return null;
        }

        /// <summary>
        ///     A synchronization context that never runs what is posted to it, like a UI thread that is blocked: it only
        ///     counts the posts. Work that depends on it never completes, which is what makes a captured context visible.
        /// </summary>
        private sealed class NeverPumpedContext : SynchronizationContext
        {
            private int _posts;

            /// <summary>How many callbacks were posted or sent.</summary>
            public int Posts => Volatile.Read(ref _posts);

            /// <summary>Counts the post and drops it.</summary>
            /// <param name="d">The callback (never run).</param>
            /// <param name="state">Its state.</param>
            public override void Post(SendOrPostCallback d, object state) => Interlocked.Increment(ref _posts);

            /// <summary>Counts the send and drops it.</summary>
            /// <param name="d">The callback (never run).</param>
            /// <param name="state">Its state.</param>
            public override void Send(SendOrPostCallback d, object state) => Interlocked.Increment(ref _posts);

            /// <summary>The same context for a copy, so that a flowed copy still counts here.</summary>
            /// <returns>This instance.</returns>
            public override SynchronizationContext CreateCopy() => this;
        }

        /// <summary>
        ///     A task scheduler with ONE thread and no inlining from other threads: a task queued to it while its thread
        ///     is busy (or blocked) waits. A continuation that an <c>await</c> captured for this scheduler therefore
        ///     cannot run while the scheduler's thread waits for it.
        /// </summary>
        private sealed class SingleThreadScheduler : TaskScheduler, IDisposable
        {
            private readonly BlockingCollection<Task> _queue = new BlockingCollection<Task>();
            private readonly Thread _thread;

            /// <summary>Starts the scheduler's thread.</summary>
            public SingleThreadScheduler()
            {
                _thread = new Thread(() =>
                {
                    foreach (Task task in _queue.GetConsumingEnumerable())
                        TryExecuteTask(task);
                }) { IsBackground = true, Name = "SingleThreadScheduler" };
                _thread.Start();
            }

            /// <summary>Queues a task for the scheduler's thread.</summary>
            /// <param name="task">The task.</param>
            protected override void QueueTask(Task task) => _queue.Add(task);

            /// <summary>Runs a task inline only on the scheduler's own thread.</summary>
            /// <param name="task">The task.</param>
            /// <param name="taskWasPreviouslyQueued">Whether it is already in the queue (then it is not run a second time).</param>
            /// <returns><c>true</c> when the task ran inline.</returns>
            protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) =>
                Thread.CurrentThread == _thread && !taskWasPreviouslyQueued && TryExecuteTask(task);

            /// <summary>The queued tasks, for the debugger.</summary>
            /// <returns>A snapshot of the queue.</returns>
            protected override IEnumerable<Task> GetScheduledTasks() => _queue.ToArray();

            /// <summary>Ends the scheduler's thread once its queue is empty.</summary>
            public void Dispose()
            {
                _queue.CompleteAdding();
                _thread.Join(TimeSpan.FromSeconds(5));
                _queue.Dispose();
            }
        }
    }
}
