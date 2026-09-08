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
    ///     NDArray ops feeding the UNet zero-copy).
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
            using OrtTensor t = input.AsOrtValue();                                                            // ZERO COPY
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
    }
}
