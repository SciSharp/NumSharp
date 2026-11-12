using System;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>NumSharp → torch: every dtype in every mode, the NEP 50 conversions, explicit dtypes with NumPy's casting, refusals.</summary>
    [TestClass]
    public class ExportTests : TorchTestBase
    {
        /// <summary>Auto and View share a C-contiguous array of each shared dtype; Copy never does; all three hold the same bits.</summary>
        [TestMethod]
        public void SharedDtypes_ViewInAutoAndView_CopyInCopy()
        {
            foreach (NPTypeCode code in SharedDtypes)
            {
                using NDArray nd = Ramp(code, 2, 3);
                foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
                {
                    using Tensor t = nd.AsTensor(mode);
                    t.dtype.Should().Be(NDArrayTorchInterop.ToScalarType(code), $"{code} {mode}");
                    t.shape.Should().Equal(new long[] { 2, 3 }, $"{code} {mode}");
                    SameBits(nd, t).Should().BeTrue($"{code} {mode}: the tensor holds the array's bits");
                    nd.MayShareMemory(t).Should().Be(mode != TorchCodecMode.Copy, $"{code} {mode}");
                }
            }
        }

        /// <summary>A view writes through both ways; a copy does not.</summary>
        [TestMethod]
        public void View_WritesThroughBothWays_CopyDoesNot()
        {
            using NDArray nd = np.arange(12).astype(np.float32).reshape(3, 4);
            using (Tensor view = nd.AsTensor())
            using (Tensor copy = nd.ToTensor())
            {
                view.mul_(2);
                nd.GetSingle(1, 2).Should().Be(12f, "torch's in-place multiply wrote the NumSharp buffer");
                nd.SetSingle(-1f, 0, 0);
                At(view, 0, 0).Should().Be(-1, "a NumSharp write shows in the tensor");
                At(copy, 0, 0).Should().Be(0, "the copy kept the values it was made with");
                At(copy, 1, 2).Should().Be(6);
            }
        }

        /// <summary>uint16, uint32, char, uint64 and decimal cross as NEP 50's dtypes with exact values; View refuses them.</summary>
        [TestMethod]
        public void ConvertedDtypes_CrossAsNep50Copies_ViewRefuses()
        {
            (NDArray nd, ScalarType st, double[] expect)[] cases =
            {
                (np.array(new ushort[] { 0, 1, 65535 }), ScalarType.Int32, new double[] { 0, 1, 65535 }),
                (np.array(new uint[] { 0, 7, uint.MaxValue }), ScalarType.Int64, new double[] { 0, 7, uint.MaxValue }),
                (np.array(new char[] { 'A', (char)0xFFFF, '\0' }), ScalarType.Int32, new double[] { 65, 65535, 0 }),
                (np.array(new ulong[] { 1, 1UL << 53, ulong.MaxValue }), ScalarType.Float64, new double[] { 1, 9007199254740992, 18446744073709551615.0 }),
                (np.array(new decimal[] { 0.5m, -2m, 1e10m }), ScalarType.Float64, new double[] { 0.5, -2, 1e10 }),
            };
            foreach (var (nd, st, expect) in cases)
            {
                using (nd)
                {
                    foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.Copy })
                    {
                        using Tensor t = nd.AsTensor(mode);
                        t.dtype.Should().Be(st, $"{nd.typecode} {mode}");
                        for (int i = 0; i < expect.Length; i++)
                            At(t, i).Should().Be(expect[i], $"{nd.typecode} {mode} [{i}]");
                        nd.MayShareMemory(t).Should().BeFalse("a converted array is always a copy");
                    }
                    Action view = () => nd.AsTensor(TorchCodecMode.View);
                    view.Should().Throw<InvalidOperationException>().WithMessage($"*{nd.typecode}*NEP 50*not a view*");
                }
            }
        }

        /// <summary>An explicit dtype converts through np.copyto's casting check: same_kind by default, unsafe on request.</summary>
        [TestMethod]
        public void ExplicitDtype_ConvertsWithNumPysCasting()
        {
            using NDArray f64 = np.array(new double[] { 1.5, -2.25, 3 });
            using (Tensor f32 = f64.AsTensor(dtype: ScalarType.Float32))
            {
                f32.dtype.Should().Be(ScalarType.Float32, "float64 to float32 is same_kind");
                At(f32, 1).Should().Be(-2.25);
            }

            Action toInt = () => f64.AsTensor(dtype: ScalarType.Int32);
            toInt.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'same_kind'");
            using (Tensor i32 = f64.AsTensor(dtype: ScalarType.Int32, casting: "unsafe"))
                At(i32, 1).Should().Be(-2, "an unsafe cast truncates, as astype does");

            Action safe = () => f64.AsTensor(dtype: ScalarType.Float32, casting: "safe");
            safe.Should().Throw<InvalidCastException>().WithMessage("*'safe'*");

            Action own = () => f64.AsTensor(TorchCodecMode.View, dtype: ScalarType.Float32);
            own.Should().Throw<InvalidOperationException>().WithMessage("*requested dtype converts*not a view*");
            using (Tensor same = f64.AsTensor(TorchCodecMode.View, dtype: ScalarType.Float64))
                f64.MayShareMemory(same).Should().BeTrue("naming the array's own dtype converts nothing");
        }

        /// <summary>uint64 downcasts to int64 on request, wrapping from 2^63 as astype(np.int64) does.</summary>
        [TestMethod]
        public void UInt64_DowncastsToInt64OnRequest()
        {
            using NDArray ids = np.array(new ulong[] { 1, (1UL << 63) + 5, ulong.MaxValue });
            using Tensor t = ids.AsTensor(dtype: ScalarType.Int64);
            t.dtype.Should().Be(ScalarType.Int64);
            t.data<long>().ToArray().Should().Equal(1L, -9223372036854775803L, -1L);
            using NDArray astype = ids.astype(np.int64);
            astype.ToArray<long>().Should().Equal(t.data<long>().ToArray(), "the bridge's downcast is NumSharp's astype");
        }

        /// <summary>A torch dtype NumSharp cannot write and an unknown casting rule are argument errors.</summary>
        [TestMethod]
        public void BadArguments_AreRefused()
        {
            using NDArray nd = np.ones(new Shape(3), np.float32);
            Action bf16 = () => nd.AsTensor(dtype: ScalarType.BFloat16);
            bf16.Should().Throw<ArgumentException>().WithMessage("*no BFloat16*torch side*");
            Action c64 = () => nd.AsTensor(dtype: ScalarType.ComplexFloat32);
            c64.Should().Throw<ArgumentException>().WithMessage("*ComplexFloat32*");
            Action casting = () => nd.AsTensor(casting: "kind");
            casting.Should().Throw<ArgumentException>().WithMessage("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'*");
            Action mode = () => nd.AsTensor((TorchCodecMode)3);
            mode.Should().Throw<ArgumentOutOfRangeException>();
            Action nul = () => ((NDArray)null).AsTensor();
            nul.Should().Throw<ArgumentNullException>();
        }

        /// <summary>Text cannot cross in any mode.</summary>
        [TestMethod]
        public void Text_IsRefusedInEveryMode()
        {
            using NDArray text = np.array(new[] { "a", "bc" });
            foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
            {
                Action act = () => text.AsTensor(mode);
                act.Should().Throw<NotSupportedException>().WithMessage("*no string tensors*");
            }
        }

        /// <summary>A disposed array is refused before anything reads its (possibly reused) buffer.</summary>
        [TestMethod]
        public void DisposedArray_IsRefused()
        {
            NDArray nd = np.arange(4).astype(np.float64);
            nd.Dispose();
            foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
            {
                Action act = () => nd.AsTensor(mode);
                act.Should().Throw<ObjectDisposedException>($"{mode}");
            }
        }

        /// <summary>The tensor is registered on the caller's DisposeScope, like any tensor TorchSharp makes.</summary>
        [TestMethod]
        public void Export_JoinsTheCurrentDisposeScope()
        {
            int before = NDArrayTorchInterop.LiveExports;
            using NDArray nd = np.ones(new Shape(8), np.float32);
            using (var scope = Scope())
            {
                Tensor t = nd.AsTensor();
                NDArrayTorchInterop.LiveExports.Should().Be(before + 1);
                scope.DisposablesCount.Should().BeGreaterThan(0, "the exported tensor (and nothing the bridge made in between) is on the scope");
            }
            NDArrayTorchInterop.LiveExports.Should().Be(before, "the scope disposed the exported tensor and libtorch released the buffer");
        }

        /// <summary>The bridge never leaves its own intermediates on the caller's scope (strided and broadcast exports make some).</summary>
        [TestMethod]
        public void Export_LeavesOnlyItsResultOnTheScope()
        {
            using NDArray nd = np.arange(12).astype(np.float64).reshape(3, 4);
            using NDArray transposed = nd.T;
            using NDArray row = np.arange(4).astype(np.float64);
            using NDArray broadcast = np.broadcast_to(row, new Shape(3, 4));
            using var scope = Scope();
            int count = scope.DisposablesCount;
            Tensor a = transposed.AsTensor();
            scope.DisposablesCount.Should().Be(count + 1, "from_blob's flat tensor is disposed after as_strided");
            Tensor b = broadcast.AsTensor();
            scope.DisposablesCount.Should().Be(count + 2, "the compact tensor under the broadcast is disposed after expand");
        }
    }
}
