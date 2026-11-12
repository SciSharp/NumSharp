using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>The package README's examples, kept compiling and true.</summary>
    [TestClass]
    public class DocExampleTests : TorchTestBase
    {
        /// <summary>The README's opening example: preprocess in NumSharp, compute in libtorch, read the result back, no copies.</summary>
        [TestMethod]
        public void Readme_OpeningExample()
        {
            np.random.seed(7);
            using NDArray batch = np.random.rand(64, 3, 32, 32).astype(np.float32);   // NumSharp preprocessing
            NDArray result;
            using (Tensor x = batch.AsTensor())                                           // zero-copy: x is a view of batch
            using (Tensor y = nn.functional.relu(x))                                      // libtorch computes
            {
                batch.MayShareMemory(x).Should().BeTrue("x is a view of batch");
                result = y.AsNDArray();                                                   // zero-copy: a view of y's storage
                result.MayShareMemory(y).Should().BeTrue();
            }
            using (result)
            {
                result.shape.Should().Equal(new long[] { 64, 3, 32, 32 });
                using NDArray expected = np.maximum(batch, 0f);
                ((bool)np.array_equal(result, expected)).Should().BeTrue("relu of non-negative data is the data, and the view outlived y");
            }
        }

        /// <summary>The README's dtype rules: NEP 50 by default, an explicit downcast on request.</summary>
        [TestMethod]
        public void Readme_DtypeRules()
        {
            using NDArray ids = np.array(new ulong[] { 1, (1UL << 63) + 5 });
            using (Tensor f = ids.AsTensor())
                f.dtype.Should().Be(ScalarType.Float64, "NEP 50 promotes uint64 with int8 to float64");
            using (Tensor i = ids.AsTensor(dtype: ScalarType.Int64))
                i.data<long>()[1].Should().Be(-9223372036854775803L, "the explicit downcast wraps from 2^63, as astype does");
        }
    }
}
