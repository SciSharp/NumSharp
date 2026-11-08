using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The README's worked examples, asserted: issue #512 (BGRA bytes → NCHW float tensor with no per-pixel
    ///     loop) and the Stable-Diffusion-style step (the <c>TensorHelper</c> the ORT samples hand-write, as
    ///     NDArray ops feeding the UNet zero-copy); and the snippets of the documents for the modes, the Run tier,
    ///     text and the asynchronous forms, each run as it is written.
    /// </summary>
    [TestClass]
    public class DocExampleTests : OnnxTestBase
    {
        [TestMethod]
        public void Issue512_BgraBytesToNchwFloat_NoPixelLoop_ZeroCopyIntoOrt()
        {
            const int H = 4, W = 6;
            var data = new byte[H * W * 4];   // interleaved B,G,R,A per pixel
            for (int p = 0; p < H * W; p++)
            {
                data[p * 4 + 0] = (byte)(p);           // B
                data[p * 4 + 1] = (byte)(p * 2);       // G
                data[p * 4 + 2] = (byte)(p * 3);       // R
                data[p * 4 + 3] = 255;                 // A
            }

            // --- the README snippet -----------------------------------------------------------------
            using NDArray bgra = np.frombuffer(data, NPTypeCode.Byte).reshape(H, W, 4);
            using NDArray planar = np.stack(new[] { bgra[":,:,2"], bgra[":,:,1"], bgra[":,:,0"] }, axis: 0);   // (3,H,W) RGB
            using NDArray asFloat = planar.astype(NPTypeCode.Single);
            using NDArray chw = asFloat / 255f;
            using NDArray input = chw.reshape(1, 3, H, W);                                                     // NCHW, C-contiguous

            using InferenceSession session = OpenSession("identity_float32");
            using OrtTensor t = input.AsOrtValue(OnnxCodecMode.View);                                          // ZERO COPY, or it throws
            using var runOptions = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> results = session.Run(runOptions, new[] { "X" }, new[] { t.Value }, new[] { "Y" });
            using NDArray output = results[0].ToNDArray();
            // ----------------------------------------------------------------------------------------

            input.Shape.IsContiguous.Should().BeTrue("the fused cast/divide produced a fresh contiguous NCHW buffer");
            output.shape.Should().Equal(1, 3, H, W);
            // spot-check against the per-pixel formula the issue hand-wrote
            for (int y = 0; y < H; y += 3)
            for (int x = 0; x < W; x += 5)
            {
                int p = y * W + x;
                output.GetSingle(0, 0, y, x).Should().Be(data[p * 4 + 2] / 255f, "R plane");
                output.GetSingle(0, 1, y, x).Should().Be(data[p * 4 + 1] / 255f, "G plane");
                output.GetSingle(0, 2, y, x).Should().Be(data[p * 4 + 0] / 255f, "B plane");
            }
        }

        [TestMethod]
        public void StableDiffusionStep_TensorHelperReplacedByNDArrayOps_ZeroCopyUnetFeed()
        {
            // the "UNet" is the identity model here; the arithmetic is the sample's own
            using InferenceSession unet = OpenSession("identity_float32");
            np.random.seed(0);
            using NDArray latents = np.random.randn(1, 4, 8, 8).astype(NPTypeCode.Single);
            float sigma = 0.75f;
            float guidanceScale = 7.5f;

            using NDArray batch = np.concatenate(new[] { latents, latents }, axis: 0);          // TensorHelper.Duplicate
            using NDArray scaled = batch / MathF.Sqrt(sigma * sigma + 1f);                     // TensorHelper.DivideTensorByFloat
            batch.shape.Should().Equal(2, 4, 8, 8);

            using NDArray noisePred = unet.Run(scaled);                                          // zero-copy in, copy out
            noisePred.shape.Should().Equal(2, 4, 8, 8);

            using NDArray uncond = noisePred["0:1"];                                             // TensorHelper.SplitTensor
            using NDArray text = noisePred["1:2"];
            using NDArray diff = text - uncond;                                                  // Subtract
            using NDArray weighted = diff * guidanceScale;                                       // MultipleTensorByFloat
            using NDArray guided = uncond + weighted;                                            // AddTensors
            guided.shape.Should().Equal(1, 4, 8, 8);
            np.allclose(guided, uncond, 1e-6, 1e-6).Should().BeTrue("identity UNet: both halves equal, so guidance adds nothing");

            // VAE decode post-processing: clip(x/2+0.5, 0, 1)*255 -> bytes, the per-pixel loop of the sample
            using NDArray half = guided / 2f;
            using NDArray shifted = half + 0.5f;
            using NDArray clipped = np.clip(shifted, 0f, 1f);
            using NDArray scaled255 = clipped * 255f;
            using NDArray img = scaled255.astype(NPTypeCode.Byte);
            img.typecode.Should().Be(NPTypeCode.Byte);
            img.shape.Should().Equal(1, 4, 8, 8);

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void ResnetStyle_TopKOverSoftmax_FromAnOutput()
        {
            using InferenceSession session = OpenSession("identity_float32");
            np.random.seed(1);
            using NDArray logits = np.random.randn(1, 1000).astype(NPTypeCode.Single);
            using NDArray output = session.Run(logits);

            using NDArray probs = Postprocess.Softmax(output);
            (NDArray top, NDArray classes) = Postprocess.TopK(probs, 10);
            using (top)
            using (classes)
            {
                classes.shape.Should().Equal(1, 10);
                using NDArray bestClass = Postprocess.Argmax(output);
                classes.GetInt64(0, 0).Should().Be(bestClass.GetInt64(0), "top-1 of softmax is the argmax of the logits");
                for (int i = 1; i < 10; i++)
                    top.GetSingle(0, i).Should().BeLessThanOrEqualTo(top.GetSingle(0, i - 1), "descending");
            }
        }

        /// <summary>
        ///     The mode table of the documents, as code: <see cref="OnnxCodecMode.Auto"/> shares an array ORT can read
        ///     in place and copies one it cannot, <see cref="OnnxCodecMode.View"/> throws instead of copying,
        ///     <see cref="OnnxCodecMode.Copy"/> always copies, and <c>IsView</c> says which a handle holds.
        /// </summary>
        [TestMethod]
        public void Modes_AutoSharesOrCopies_ViewThrows_CopyCopies_AsTheDocumentsSay()
        {
            using NDArray contiguous = np.array(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            using NDArray transposed = contiguous.T;

            using (OrtTensor shared = contiguous.AsOrtValue())                           // Auto: a view
                shared.IsView.Should().BeTrue();
            using (OrtTensor copied = transposed.AsOrtValue())                           // Auto: a C-order copy the handle owns
            {
                copied.IsView.Should().BeFalse();
                copied.Value.GetTensorDataAsSpan<float>().ToArray().Should().Equal(1f, 4f, 2f, 5f, 3f, 6f);
            }

            new Action(() => transposed.AsOrtValue(OnnxCodecMode.View)).Should().Throw<InvalidOperationException>();   // View: never a silent copy
            using (OrtTensor always = contiguous.AsOrtValue(OnnxCodecMode.Copy))         // Copy: independent
                always.IsView.Should().BeFalse();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>The Run tier's snippet of the documents, form by form, on models that have the names it uses.</summary>
        [TestMethod]
        public void RunTier_EveryFormOfTheSnippet_RunsAsWritten()
        {
            using NDArray x = np.array(new float[] { 1, 2, 3 });

            using (InferenceSession session = OpenSession("identity_float32"))
            using (NDArray y = session.Run(x))                                           // one input, one output: names from metadata
                y.ToArray<float>().Should().Equal(1f, 2f, 3f);

            using (InferenceSession session = OpenSession("two_outputs_f32"))
            {
                using (NDArray y2 = session.Run("X", x, "Y2"))                           // named
                    y2.ToArray<float>().Should().Equal(-1f, -2f, -3f);

                var inputs = new Dictionary<string, NDArray> { { "X", x } };
                IReadOnlyDictionary<string, NDArray> views = session.Run(inputs, zeroCopyOutputs: true);   // views that OWN their OrtValue
                using (NDArray y1 = views["Y1"])
                using (NDArray y2 = views["Y2"])
                {
                    y1.flags.owndata.Should().BeFalse();
                    y2.ToArray<float>().Should().Equal(-1f, -2f, -3f);
                }

                using NDArray y = np.zeros(new Shape(3), typeof(float));
                session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y2", y } });                  // ORT writes INTO y
                y.ToArray<float>().Should().Equal(-1f, -2f, -3f);
            }

            using (InferenceSession session = OpenSession("add_f32"))
            using (NDArray a = np.array(new float[] { 1, 2, 3 }))
            using (NDArray b = np.array(new float[] { 10, 20, 30 }))
            {
                IReadOnlyDictionary<string, NDArray> outs = session.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
                using NDArray c = outs["C"];
                c.ToArray<float>().Should().Equal(11f, 22f, 33f);
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>The text section of the documents: a text array feeds a string input and a string output comes back as a text array.</summary>
        [TestMethod]
        public void Text_FeedsAStringInput_AndComesBackAsATextArray()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray words = np.array(new[] { "héllo", "wörld" });
            using NDArray echo = session.Run(words);
            echo.dtype.kind.Should().Be('U');
            NDArray.AsStringArray(echo).Should().Equal("héllo", "wörld");
            new Action(() => session.Run(words, mode: OnnxCodecMode.View)).Should().Throw<InvalidOperationException>("text is always a copy");
        }

        /// <summary>
        ///     The asynchronous snippet of the documents. It needs ONNX Runtime 1.30.0 (the test is Inconclusive below
        ///     it) and a session with 2 intra-op threads.
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        [Timeout(120_000)]
        public async System.Threading.Tasks.Task AsyncSnippet_RunsAsWritten()
        {
            if (!InferenceSessionExtensions.IsRunAsyncSupported)
                Assert.Inconclusive("the asynchronous run needs ONNX Runtime 1.30.0 or newer; the suite's ORT 1.31.0 leg runs this test.");
            using var options = new SessionOptions { IntraOpNumThreads = 2 };
            using var session = new InferenceSession(Model("fixed_shape_f32"), options);
            using NDArray data = Arange(NPTypeCode.Single, 1, 3, 4, 4);
            using NDArray relu = np.zeros(new Shape(1, 3, 4, 4), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "data", data } };
            var outputs = new Dictionary<string, NDArray> { { "relu", relu } };

            await session.RunIntoAsync(inputs, outputs);                                 // ORT writes into your arrays; no thread waits
            BytesOf(relu).Should().Equal(BytesOf(data), "Relu of non-negative values");

            IReadOnlyDictionary<string, NDArray> outs = await session.RunAsync(inputs);  // outputs allocated for you
            using NDArray allocated = outs["relu"];
            allocated.flags.owndata.Should().BeTrue();
            BytesOf(allocated).Should().Equal(BytesOf(data));
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }
    }
}
