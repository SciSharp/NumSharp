using System.Collections.Generic;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The synchronous <c>Run</c> tier's own handles are TRANSIENT (see <see cref="OrtTensor"/>'s constructor): created and
    ///     disposed within one call on one thread and never handed out, they skip the ARC token, the owner table and the
    ///     disposed-value table. The asynchronous verbs, whose handles are disposed on a continuation, and the public export
    ///     verbs keep the full machinery.
    /// </summary>
    /// <remarks>
    ///     Pinned through <c>OrtValueBooking.TakenOverReleases</c>, which counts one takeover per disposed value that carried a
    ///     token: a synchronous run must not move it (before the transient handles every fed input moved it by one), an
    ///     asynchronous run and an <see cref="NDArrayOnnxInterop.AsOrtValue"/> handle must. The suite runs sequentially
    ///     (<c>[assembly: DoNotParallelize]</c>), so the process-wide counter moves only with the test's own calls. On an ONNX
    ///     Runtime without the members the takeover needs (<c>OrtValueBooking.CanTakeOver</c> false) no value carries a token at
    ///     all and the tests report Inconclusive rather than pass for the wrong reason.
    /// </remarks>
    [TestClass]
    public class RunTierTransientHandleTests : OnnxTestBase
    {
        /// <summary>Reports the test Inconclusive when the loaded ONNX Runtime gives no value a token (nothing to observe).</summary>
        private static void RequireTokens()
        {
            if (!OrtValueBooking.CanTakeOver)
                Assert.Inconclusive($"ONNX Runtime {OrtEnv.Instance().GetVersionString()} lacks the members the ARC takeover needs; no value carries a token to observe.");
        }

        /// <summary>A synchronous run feeds its inputs through transient handles: no token is taken over, and the run computes.</summary>
        [TestMethod]
        public void Run_FeedsInputsThroughTransientHandles()
        {
            RequireTokens();
            using InferenceSession session = OpenSession("add_f32");
            using NDArray a = Arange(NPTypeCode.Single, 2, 3);
            using NDArray b = Arange(NPTypeCode.Single, 2, 3);
            var inputs = new Dictionary<string, NDArray> { { "A", a }, { "B", b } };

            long before = OrtValueBooking.TakenOverReleases;
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(inputs);
            using NDArray c = outputs["C"];

            OrtValueBooking.TakenOverReleases.Should().Be(before, "the run's two input handles are transient and carry no token");
            c.GetSingle(1, 2).Should().Be(10f, "C = A + B, and A[1, 2] = B[1, 2] = 5");
            a.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("the transient handle gave its reference on A back");
            b.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("and on B");
        }

        /// <summary>
        ///     The decoder idiom: owning zero-copy output views fed back as the next run's inputs, step after step. Every
        ///     handle is transient, the values stay right, and every view and handle is released.
        /// </summary>
        [TestMethod]
        public void Run_ZeroCopyOutputsFedBack_StepAfterStep()
        {
            RequireTokens();
            using InferenceSession session = OpenSession("identity_float32");
            NDArray x = Arange(NPTypeCode.Single, 3, 4);
            long before = OrtValueBooking.TakenOverReleases;
            for (int step = 0; step < 5; step++)
            {
                NDArray y = session.Run(new Dictionary<string, NDArray> { { "X", x } }, zeroCopyOutputs: true)["Y"];
                x.Dispose();   // the previous step's view: the run that read it has returned
                x = y;
            }

            OrtValueBooking.TakenOverReleases.Should().Be(before, "five steps of transient input handles take no token over");
            x.GetSingle(2, 3).Should().Be(11f, "the identity chain keeps every value");
            x.Dispose();
        }

        /// <summary>A synchronous run into pre-allocated outputs: the input and the output handles are both transient.</summary>
        [TestMethod]
        public void RunInto_InputAndOutputHandlesAreTransient()
        {
            RequireTokens();
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 2);
            using NDArray y = np.zeros(new Shape(2, 2), typeof(float));

            long before = OrtValueBooking.TakenOverReleases;
            session.RunInto(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", y } });

            OrtValueBooking.TakenOverReleases.Should().Be(before, "RunInto's input and output handles are transient");
            BytesOf(y).Should().Equal(BytesOf(x));
            y.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("the output handle gave its reference back");
        }

        /// <summary>
        ///     An asynchronous run disposes its handles on a continuation, so they keep the full machinery: the input's and the
        ///     output's tokens are each taken over once.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async Task RunIntoAsync_KeepsTheTokenOfEveryHandle()
        {
            RequireTokens();
            if (!InferenceSessionExtensions.IsRunAsyncSupported)
                Assert.Inconclusive($"ONNX Runtime {OrtEnv.Instance().GetVersionString()}: the asynchronous run needs 1.30.0 or newer.");
            using var options = new SessionOptions { IntraOpNumThreads = 2, InterOpNumThreads = 1, LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
            using var session = new InferenceSession(Model("identity_float32"), options);
            using NDArray x = Arange(NPTypeCode.Single, 2, 2);
            using NDArray y = np.zeros(new Shape(2, 2), typeof(float));

            long before = OrtValueBooking.TakenOverReleases;
            await session.RunIntoAsync(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", y } });

            (OrtValueBooking.TakenOverReleases - before).Should().Be(2, "the asynchronous input and output handles are not transient");
            BytesOf(y).Should().Equal(BytesOf(x));
        }

        /// <summary>A public export handle is never transient: its value carries a token from birth, taken over on dispose.</summary>
        [TestMethod]
        public void AsOrtValue_PublicHandleKeepsItsToken()
        {
            RequireTokens();
            using NDArray a = Arange(NPTypeCode.Single, 4);
            long before = OrtValueBooking.TakenOverReleases;
            using (OrtTensor handle = a.AsOrtValue())
                OrtValueBooking.HasToken(handle.Value).Should().BeTrue("a value the caller can reach carries its ARC");

            (OrtValueBooking.TakenOverReleases - before).Should().Be(1, "disposing the handle took its value's token over once");
        }
    }
}
