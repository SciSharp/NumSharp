using System.Numerics.Tensors;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>The README's examples, as runnable tests (so the docs cannot rot).</summary>
    [TestClass]
    public class DocExampleTests : TensorsTestBase
    {
        [TestMethod]
        public void Example_AsTensorSpan_InPlaceMutation_WritesThrough()
        {
            using var nd = Arange(NPTypeCode.Single, 2, 3);       // [[0,1,2],[3,4,5]]

            using (var h = nd.AsTensorSpan<float>())
            {
                TensorSpan<float> span = h.Span;                  // zero-copy over nd's buffer
                for (nint i = 0; i < span.Lengths[0]; i++)
                    for (nint j = 0; j < span.Lengths[1]; j++)
                        span[new[] { i, j }] += 10f;              // ref-return indexer -> writes through
            }

            nd.GetSingle(0, 0).Should().Be(10f);
            nd.GetSingle(1, 2).Should().Be(15f);
        }

        [TestMethod]
        public void Example_ToTensor_ThenBack()
        {
            using var nd = Arange(NPTypeCode.Double, 4);
            Tensor<double> t = nd.ToTensor<double>();             // independent dense copy
            t[new nint[] { 0 }] = 99.0;

            using NDArray back = t.ToNDArray();                   // owning copy back
            back.GetDouble(0).Should().Be(99.0);
            nd.GetDouble(0).Should().Be(0.0, "ToTensor did not alias nd");
        }

        [TestMethod]
        public void Example_AsNDArray_ReadModelOutputZeroCopy()
        {
            // A tensor produced elsewhere (e.g. a compute pipeline) read back as an NDArray with no copy.
            var output = Tensor.Create(new float[] { 0.1f, 0.7f, 0.2f }, new nint[] { 3 });
            using NDArray probs = output.AsNDArray();

            probs.GetSingle(1).Should().Be(0.7f);
            long best = np.argmax(probs);
            best.Should().Be(1, "NumSharp reductions run straight on the shared buffer");
        }

        [TestMethod]
        public void Example_StridedView_ZeroCopy_NoMaterialize()
        {
            using var image = Arange(NPTypeCode.Single, 4, 4);
            using var channel = image["1:3, 1:3"];                // a strided crop — no copy
            using var h = channel.AsTensorSpan<float>();

            h.ReadOnlySpan[new nint[] { 0, 0 }].Should().Be(image.GetSingle(1, 1));
            h.ReadOnlySpan[new nint[] { 1, 1 }].Should().Be(image.GetSingle(2, 2));
        }
    }
}
