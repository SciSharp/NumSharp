using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     NumSharp → torch across layouts: every non-negative-stride layout shares (as_strided), broadcasts over writeable memory
    ///     share as torch's expand, and reversed, read-only and misaligned arrays copy under Auto and throw under View.
    /// </summary>
    [TestClass]
    public class ExportLayoutTests : TorchTestBase
    {
        /// <summary>Views of a 3-D base, each a layout NumSharp makes without copying, named for the messages.</summary>
        /// <param name="b">The base array (shape 4, 5, 6).</param>
        /// <returns>The named views; the caller disposes them.</returns>
        private static List<(string name, NDArray view)> SharedLayouts(NDArray b) => new List<(string, NDArray)>
        {
            ("contiguous", b["..."]),
            ("offset slice", b["1:3"]),
            ("stepped", b[":, ::2"]),
            ("inner stepped", b["..., 1::3"]),
            ("transposed", b.T),
            ("swapaxes", np.swapaxes(b, 0, 2)),
            ("2-D of 3-D", b["2"]),
            ("column", b[":, :, 3"]),
            ("length-1 reversed axis", b[":, 1:2, :"][":, ::-1, :"]),
            ("F-order copy", np.asfortranarray(b)),
        };

        /// <summary>Every non-negative-stride layout of every shared dtype is a view in Auto and View, with the array's bits and strides.</summary>
        [TestMethod]
        public void NonNegativeStrides_ShareInAutoAndView()
        {
            foreach (NPTypeCode code in SharedDtypes)
            {
                using NDArray b = Ramp(code, 4, 5, 6);
                foreach (var (name, view) in SharedLayouts(b))
                {
                    using (view)
                    {
                        foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View })
                        {
                            using Tensor t = view.AsTensor(mode);
                            SameBits(view, t).Should().BeTrue($"{code} {name} {mode}");
                            view.MayShareMemory(t).Should().BeTrue($"{code} {name} {mode} shares");
                            long[] tStrides = t.stride();
                            for (int i = 0; i < view.ndim; i++)
                                if (view.shape[i] > 1)
                                    tStrides[i].Should().Be(view.Shape.strides[i], $"{code} {name}: torch's element strides are NumSharp's");
                        }
                    }
                }
            }
        }

        /// <summary>A write through a strided export lands on the element it names in the NumSharp base.</summary>
        [TestMethod]
        public void StridedView_WritesTheRightElements()
        {
            using NDArray b = np.arange(24).astype(np.float64).reshape(4, 6);
            using NDArray view = b.T["1:4, ::2"];   // shape (3, 2), strides (1, 12): element [i, j] is b[2j, i+1]
            using (Tensor t = view.AsTensor(TorchCodecMode.View))
            using (var scope = Scope())
                t[2, 1].fill_(-7);
            b.GetDouble(2, 3).Should().Be(-7, "t[2, 1] is view[2, 1] = b.T[3, 2] = b[2, 3]");
            using NDArray changed = np.not_equal(b, np.arange(24).astype(np.float64).reshape(4, 6));
            ((int)np.count_nonzero(changed)).Should().Be(1, "only that element changed");
        }

        /// <summary>A reversed view copies under Auto (same values) and is refused under View, naming the negative stride.</summary>
        [TestMethod]
        public void NegativeStrides_CopyInAuto_ThrowInView()
        {
            using NDArray b = np.arange(12).astype(np.int32).reshape(3, 4);
            using NDArray reversed = b["::-1, ::2"];
            using (Tensor t = reversed.AsTensor())
            {
                SameBits(reversed, t).Should().BeTrue("the copy holds the reversed values in C order");
                reversed.MayShareMemory(t).Should().BeFalse();
                t.is_contiguous().Should().BeTrue();
            }
            Action view = () => reversed.AsTensor(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*negative stride*");
        }

        /// <summary>
        ///     A broadcast over writeable memory crosses as torch's own expand: zero-copy, stride 0, libtorch's overlap check refuses
        ///     an in-place op over the repeated lanes, and a row write lands in the one row the broadcast repeats.
        /// </summary>
        [TestMethod]
        public void BroadcastOfWriteableMemory_CrossesAsExpand()
        {
            using NDArray row = np.arange(4).astype(np.float32);
            using NDArray batch = np.broadcast_to(row, new Shape(3, 4));
            batch.Shape.IsWriteable.Should().BeFalse("NumPy's rule: a broadcast is read-only");
            foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View })
            {
                using Tensor t = batch.AsTensor(mode);
                t.stride().Should().Equal(new long[] { 0, 1 }, $"{mode}: torch's expand layout");
                SameBits(batch, t).Should().BeTrue();
                row.MayShareMemory(t).Should().BeTrue($"{mode}: the memory under the broadcast is shared");
                Action inPlace = () => t.add_(1);
                inPlace.Should().Throw<ExternalException>("libtorch refuses an in-place op over overlapping lanes")
                    .WithMessage("*more than one element*");
            }
            using (Tensor t = batch.AsTensor())
            using (var scope = Scope())
                t[1].fill_(9);
            row.ToArray<float>().Should().Equal(9f, 9f, 9f, 9f);
        }

        /// <summary>A broadcast of a NumPy scalar (read-only memory) copies under Auto and is refused under View.</summary>
        [TestMethod]
        public void BroadcastOfReadOnlyMemory_CopyInAuto_ThrowInView()
        {
            using NDArray scalar = NDArray.Scalar(2.5);
            using NDArray b = np.broadcast_to(scalar, new Shape(4));
            using (Tensor t = b.AsTensor())
            {
                SameBits(b, t).Should().BeTrue();
                b.MayShareMemory(t).Should().BeFalse("the scalar's memory is immutable, so it is copied");
            }
            Action view = () => b.AsTensor(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*broadcast of read-only memory*");
        }

        /// <summary>A NumPy scalar (read-only 0-d) copies under Auto; a writeable 0-d array is a 0-d view.</summary>
        [TestMethod]
        public void ZeroD_ReadOnlyCopies_WriteableShares()
        {
            using NDArray scalar = NDArray.Scalar(3.25);
            using (Tensor t = scalar.AsTensor())
            {
                t.ndim.Should().Be(0);
                t.item<double>().Should().Be(3.25);
                scalar.MayShareMemory(t).Should().BeFalse();
            }
            Action view = () => scalar.AsTensor(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*read-only*");

            using NDArray zeroD = np.zeros(new Shape(), np.float64);
            zeroD.Shape.IsWriteable.Should().BeTrue();
            using (Tensor t = zeroD.AsTensor(TorchCodecMode.View))
            {
                t.ndim.Should().Be(0);
                t.fill_(4.5);
            }
            zeroD.GetDouble().Should().Be(4.5, "the 0-d tensor wrote the 0-d array");
        }

        /// <summary>Empty arrays keep their shape in every mode.</summary>
        [TestMethod]
        public void EmptyArrays_KeepTheirShape()
        {
            using NDArray empty = np.zeros(new Shape(0, 3), np.float32);
            foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
            {
                using Tensor t = empty.AsTensor(mode);
                t.shape.Should().Equal(new long[] { 0, 3 }, $"{mode}");
                t.numel().Should().Be(0);
            }
            using NDArray emptySlice = np.arange(10).astype(np.float64)["5:5"];
            using (Tensor t = emptySlice.AsTensor(TorchCodecMode.View))
                t.shape.Should().Equal(new long[] { 0 });
        }

        /// <summary>Data off its element alignment (a byte-reinterpreting view at an odd offset) copies under Auto, throws under View.</summary>
        [TestMethod]
        public unsafe void MisalignedData_CopyInAuto_ThrowInView()
        {
            using NDArray bytes = np.arange(24).astype(np.uint8);
            // The offset that is misaligned for THIS buffer: the pool aligns buffers, but its guard-page mode ends each one at
            // a page boundary instead, so a fixed offset of 1 is aligned there for some lengths.
            long address = (long)bytes.Storage.InternalArray.Address;
            int k = 1;
            while ((address + k) % 8 == 0)
                k++;
            using NDArray odd = bytes[$"{k}:{k + 16}"];
            NDArray misaligned;
            try
            {
                misaligned = odd.view(np.float64);
            }
            catch (Exception ex)
            {
                Assert.Inconclusive($"NumSharp refuses the reinterpreting view this test needs: {ex.Message}");
                return;
            }
            using (misaligned)
            {
                using (Tensor t = misaligned.AsTensor())
                {
                    SameBits(misaligned, t).Should().BeTrue();
                    misaligned.MayShareMemory(t).Should().BeFalse("misaligned data is copied");
                }
                Action view = () => misaligned.AsTensor(TorchCodecMode.View);
                view.Should().Throw<InvalidOperationException>().WithMessage("*not aligned to 8 bytes*");
            }
        }

        /// <summary>
        ///     Strided views that END at their buffer's last element, exported and read whole by libtorch's kernels (clone, sum,
        ///     contiguous). A reach computed one element too far, or a kernel reading past the view, is invisible to a value test
        ///     and crashes the host under <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c>, where every pooled buffer ends at an inaccessible page.
        /// </summary>
        [TestMethod]
        public void StridedViewsEndingAtTheBufferEnd_ReadInBounds()
        {
            foreach (NPTypeCode code in SharedDtypes)
            {
                using NDArray flat = Ramp(code, 9);
                using NDArray grid = Ramp(code, 5, 7);
                using NDArray cube = Ramp(code, 3, 4, 5);
                NDArray[] views =
                {
                    flat["::2"],             // 0, 2, 4, 6, 8: the last element is the buffer's last
                    flat["8:"],              // the last element alone
                    grid["1:, ::3"],         // ends at [4, 6], the last element
                    grid.T,                  // a transposed view's last logical element is the buffer's last
                    cube[":, 1:, 2:"]["..., ::2"],
                };
                foreach (NDArray view in views)
                {
                    using (view)
                    using (Tensor t = view.AsTensor(TorchCodecMode.View))
                    using (var scope = Scope())
                    {
                        Tensor dense = t.contiguous();   // libtorch's strided copy kernel reads every element the view names
                        SameBits(view, t).Should().BeTrue($"{code} {view.Shape}");
                        if (code == NPTypeCode.Complex)
                            continue;   // a real sum of complex values would drop the imaginary parts on both sides
                        double torchSum = t.to_type(ScalarType.Float64).sum().item<double>();   // libtorch's reduction kernel
                        using NDArray asDouble = view.astype(np.float64);
                        torchSum.Should().BeApproximately((double)np.sum(asDouble), 1e-9, $"{code} {view.Shape}");
                    }
                }
            }
        }

        /// <summary>Copy mode reads every layout - reversed, broadcast, transposed - into an independent C-contiguous tensor.</summary>
        [TestMethod]
        public void Copy_ReadsEveryLayoutIntoCOrder()
        {
            using NDArray b = np.arange(24).astype(np.int64).reshape(2, 3, 4);
            using NDArray row = np.arange(4).astype(np.int64);
            var layouts = new List<NDArray> { b["::-1, :, ::-2"], b.T, np.broadcast_to(row, new Shape(5, 4)), b[":, 1"] };
            foreach (NDArray layout in layouts)
            {
                using (layout)
                using (Tensor t = layout.ToTensor())
                {
                    t.is_contiguous().Should().BeTrue();
                    SameBits(layout, t).Should().BeTrue();
                    layout.MayShareMemory(t).Should().BeFalse();
                }
            }
        }
    }
}
