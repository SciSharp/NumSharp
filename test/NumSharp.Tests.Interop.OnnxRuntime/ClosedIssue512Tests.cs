using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Backs the answer that closed GitHub #512, "how to change to Microsoft.ML.OnnxRuntime.Tensors.DenseTensor&lt;float&gt;?"
    ///     (https://github.com/SciSharp/NumSharp/issues/512): NumSharp.Interop.OnnxRuntime converts between <c>NDArray</c> and
    ///     <c>DenseTensor&lt;T&gt;</c>, and the reporter's BGRA -&gt; planar-RGB float loop becomes one ufunc pass written
    ///     straight into the tensor, with the loop's exact bits. The answer also says that pass is not faster than the
    ///     hand-fused loop; that is a timing, measured for the answer, not asserted here.
    /// </summary>
    [TestClass]
    public class ClosedIssue512Tests : OnnxTestBase
    {
        /// <summary>
        ///     The answer's three lines (a <c>DenseTensor</c>, the BGRA bytes as an (H, W, 4) array, and
        ///     <c>np.multiply</c> over the reversed-channel, transposed view into <c>tensor.AsNDArray()["0"]</c>) produce exactly
        ///     the floats of the reporter's <c>ExtractPixelsArgb</c> loop, on an odd 5 x 7 image.
        /// </summary>
        [TestMethod]
        public void AnswerSnippet_WritesTheLoopsExactBits_IntoTheTensor()
        {
            const int h = 5, w = 7, n = h * w;
            var pixels = new byte[n * 4];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = (byte)(i * 7 + 3);

            var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });
            using (var bgra = np.frombuffer(pixels, np.uint8).reshape(h, w, 4))
            using (var view = tensor.AsNDArray())
            {
                np.multiply(bgra[":, :, 2::-1"].transpose(new[] { 2, 0, 1 }), // BGR -> RGB and HWC -> CHW are views
                            1f / 255f, @out: view["0"], dtype: np.float32);
            }

            // The reporter's loop, verbatim apart from the tensor it fills.
            var expected = new DenseTensor<float>(new[] { 1, 3, h, w });
            Span<float> r = expected.Buffer.Span, g = r[n..], b = g[n..];
            for (int i = 0, s = 0; i < n; i++, s += 4)
            {
                r[i] = pixels[s + 2] * 0.0039215686274509803921568627451f;
                g[i] = pixels[s + 1] * 0.0039215686274509803921568627451f;
                b[i] = pixels[s] * 0.0039215686274509803921568627451f;
            }

            tensor.Buffer.Span.SequenceEqual(expected.Buffer.Span).Should().BeTrue("the answer's one pass gives the loop's exact float bits");
        }

        /// <summary>
        ///     The four verbs the answer names convert both ways: <c>ToDenseTensor</c> copies an <c>NDArray</c> out,
        ///     <c>AsDenseTensor</c> shares it, and <c>ToNDArray</c> / <c>AsNDArray</c> bring a <c>DenseTensor</c> back, all with the
        ///     same values and shape.
        /// </summary>
        [TestMethod]
        public void FourVerbs_ConvertBothWays()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 2, 3);

            DenseTensor<float> copy = nd.ToDenseTensor<float>();
            copy.Dimensions.ToArray().Should().Equal(2, 3);
            copy[1, 2].Should().Be(5f);

            using (OrtTensor<float> shared = nd.AsDenseTensor<float>())
                shared.Tensor[1, 2].Should().Be(5f);

            using (var back = copy.ToNDArray())
                np.array_equal(back, nd).Should().BeTrue();
            using (var viewBack = copy.AsNDArray())
                viewBack.shape.Should().Equal(2L, 3L);
        }
    }
}
