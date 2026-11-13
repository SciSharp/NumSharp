using System;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     The examples of the package README and of the docs page (<c>docs/website-src/docs/interop/torchsharp.md</c>), kept
    ///     compiling and true: each <c>Page_</c> test runs one code block of the page as written and asserts what its comments
    ///     claim. The page's "From zero" and dtype blocks are the README's, so <see cref="Readme_OpeningExample"/> and
    ///     <see cref="Readme_DtypeRules"/> cover them too. Edit a block on the page and its test together.
    /// </summary>
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

        /// <summary>
        ///     The page's modes block ("Views, copies and TorchCodecMode"): Auto shares a C-contiguous array, ToTensor copies it,
        ///     and a reversed view copies under Auto and is refused under View with the very message the page quotes.
        /// </summary>
        [TestMethod]
        public void Page_Modes()
        {
            using NDArray a = np.arange(12).reshape(3, 4).astype(np.float32);
            using (Tensor shared = a.AsTensor())
            using (Tensor copy = a.ToTensor())
            {
                a.MayShareMemory(shared).Should().BeTrue("Auto shares a C-contiguous array");
                a.MayShareMemory(copy).Should().BeFalse("ToTensor is always independent");
            }

            using NDArray reversed = a["::-1"];
            using (Tensor r = reversed.AsTensor())
            {
                reversed.MayShareMemory(r).Should().BeFalse("Auto copies a view torch has no strides for");
                SameBits(reversed, r).Should().BeTrue("the copy holds the rows in reversed (logical) order");
            }
            // The page quotes this message in full, so the assertion is exact rather than a wildcard: a reworded refusal must
            // update the page too.
            Action view = () => reversed.AsTensor(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage(
                "AsTensor(View): the array has a negative stride (a reversed view such as a[::-1]), and torch has no " +
                "negative strides. Auto and Copy cross it as a copy.");
        }

        /// <summary>
        ///     The page's verbs block ("The verbs"), over a concrete array and tensor: AsTensor(View) shares, ToTensor(dtype:) is a
        ///     converted copy, AsNDArray views a CPU tensor's storage and ToNDArray gives an owning copy.
        /// </summary>
        [TestMethod]
        public void Page_Verbs()
        {
            using NDArray nd = np.arange(6).reshape(2, 3).astype(np.float64);
            using Tensor flat = torch.arange(6, dtype: ScalarType.Float32);
            using Tensor tensor = flat.reshape(2, 3);

            using (Tensor t = nd.AsTensor(TorchCodecMode.View))
                nd.MayShareMemory(t).Should().BeTrue("View is zero copy, or it throws");
            using (Tensor t2 = nd.ToTensor(dtype: ScalarType.Float32))
            {
                t2.dtype.Should().Be(ScalarType.Float32, "the requested dtype (float64 -> float32 is same_kind)");
                nd.MayShareMemory(t2).Should().BeFalse("a conversion is always a copy");
            }
            using (NDArray v = tensor.AsNDArray())
                v.MayShareMemory(tensor).Should().BeTrue("AsNDArray views a CPU tensor's storage");
            using (NDArray c = tensor.ToNDArray())
            {
                c.MayShareMemory(tensor).Should().BeFalse("ToNDArray copies");
                c.flags.owndata.Should().BeTrue("the copy owns its buffer");
            }
        }

        /// <summary>
        ///     The page's broadcast block ("NumSharp → torch"): a broadcast of writeable memory crosses as torch's expand, strides
        ///     (0, 1) over the row's own memory; libtorch refuses an in-place op over the repeated lanes, and a write to one lane
        ///     changes the one row all four repeat.
        /// </summary>
        [TestMethod]
        public void Page_BroadcastAsExpand()
        {
            using NDArray row = np.arange(5).astype(np.float32);
            using NDArray grid = np.broadcast_to(row, new Shape(4, 5));
            using Tensor g = grid.AsTensor();
            g.stride().Should().Equal(new long[] { 0, 1 }, "torch's expand layout");
            row.MayShareMemory(g).Should().BeTrue("no copy: the row's memory is the tensor's");

            // libtorch checks for internal overlap before it writes anything, so the refused add_ leaves the row as it was.
            Action inPlace = () => g.add_(1);
            inPlace.Should().Throw<ExternalException>("libtorch refuses an in-place op over overlapping lanes")
                .WithMessage("*more than one element*");
            row.ToArray<float>().Should().Equal(new[] { 0f, 1f, 2f, 3f, 4f }, "the refused op wrote nothing");

            // g[0] is a torch view of the exported storage: the scope disposes it, so the leak gate sees only g.
            using (var scope = Scope())
                g[0].fill_(7);
            row.ToArray<float>().Should().Equal(new[] { 7f, 7f, 7f, 7f, 7f }, "the one row all four lanes repeat");
        }

        /// <summary>
        ///     The page's import block ("torch → NumSharp"): a transposed torch view comes back as an F-contiguous NumSharp view
        ///     of the same storage, a tensor that requires grad as a read-only view, and its detach() as a writeable one.
        /// </summary>
        [TestMethod]
        public void Page_ImportViews()
        {
            using Tensor t = torch.arange(12, dtype: ScalarType.Float32).reshape(3, 4);
            using Tensor tt = t.t();
            using (NDArray v = tt.AsNDArray())
            {
                v.flags.f_contiguous.Should().BeTrue("strides (1, 4) over a (4, 3) view are Fortran order, as NumPy reports");
                v.flags.c_contiguous.Should().BeFalse();
                v.MayShareMemory(t).Should().BeTrue("a view of the storage, not a copy");
                SameBits(v, tt).Should().BeTrue();
            }

            using Tensor w = torch.randn(3, 3, requires_grad: true);
            using (NDArray wv = w.AsNDArray())
                wv.flags.writeable.Should().BeFalse("a tensor that requires grad is viewed read-only");
            using Tensor wd = w.detach();
            using (NDArray wdv = wd.AsNDArray())
                wdv.flags.writeable.Should().BeTrue("detach() opts out of autograd: a writeable view");
        }
    }
}
