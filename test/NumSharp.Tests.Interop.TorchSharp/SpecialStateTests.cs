using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static NumSharp.Tests.Interop.TorchSharp.StateMatrixSupport;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     NDArray states with no NumPy recipe to record (NumSharp's own scalars, scopes, typed wrappers and lifetimes) and
    ///     flag changes made after a crossing, each held to the bridge's rules.
    /// </summary>
    [TestClass]
    public class SpecialStateTests : TorchTestBase
    {
        /// <summary>A NumPy scalar (<c>NDArray.Scalar</c>: read-only 0-d by NumPy's scalar rule) copies under Auto, throws under View.</summary>
        [TestMethod]
        public void NumPyScalar_IsReadOnly_CopiesOrThrows()
        {
            using NDArray s = NDArray.Scalar(2.5f);
            s.flags.writeable.Should().BeFalse();
            using (Tensor t = s.AsTensor())
            {
                t.ndim.Should().Be(0);
                t.item<float>().Should().Be(2.5f);
                s.MayShareMemory(t).Should().BeFalse();
            }
            Action view = () => s.AsTensor(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*read-only*");
        }

        /// <summary>A broadcast of a NumPy scalar has read-only memory under it: copied, not expanded.</summary>
        [TestMethod]
        public void BroadcastOfNumPyScalar_Copies()
        {
            using NDArray s = NDArray.Scalar(4L);
            using NDArray b = np.broadcast_to(s, new Shape(3, 2));
            StateCatalog.ExportBlocker(b).Should().Be("broadcast of read-only memory");
            using Tensor t = b.AsTensor();
            t.stride().Should().Equal(new long[] { 2, 1 }, "a copy, not an expand");
            SameBits(b, t).Should().BeTrue();
        }

        /// <summary>Foreign memory (<c>np.frombuffer(IntPtr)</c>: OWNDATA false, writeable) is shared; libtorch writes reach it.</summary>
        [TestMethod]
        public unsafe void ForeignMemory_IsShared()
        {
            double* raw = (double*)NativeMemory.AlignedAlloc(8 * sizeof(double), 64);
            try
            {
                for (int i = 0; i < 8; i++)
                    raw[i] = i;
                using NDArray nd = np.frombuffer((IntPtr)raw, 8 * sizeof(double), np.float64);
                nd.flags.owndata.Should().BeFalse();
                using (Tensor t = nd.AsTensor(TorchCodecMode.View))
                {
                    TorchWriteFirst(t, 42);
                    raw[0].Should().Be(42, "the tensor wrote the foreign memory itself");
                    raw[3] = -3;
                    FirstValue(t[3]).Should().Be(-3);
                }
            }
            finally
            {
                Settle();   // every tensor over the memory is gone before it is freed
                NativeMemory.AlignedFree(raw);
            }
        }

        /// <summary>A typed wrapper (<c>NDArray&lt;T&gt;</c>) crosses like the array it wraps.</summary>
        [TestMethod]
        public void TypedWrapper_Crosses()
        {
            using NDArray plain = np.arange(6).astype(np.float32);
            NDArray<float> typed = plain.MakeGeneric<float>();
            using Tensor t = typed.AsTensor(TorchCodecMode.View);
            typed.MayShareMemory(t).Should().BeTrue();
            SameBits(typed, t).Should().BeTrue();
        }

        /// <summary>A matrix (<c>NDMatrix</c>) crosses as a 2-D tensor, and comes back as a plain NDArray (torch has no matrix class).</summary>
        [TestMethod]
        public void Matrix_CrossesAndComesBackPlain()
        {
            using NDMatrix m = np.asmatrix(np.arange(6).astype(np.float64).reshape(2, 3));
            using Tensor t = m.AsTensor(TorchCodecMode.View);
            t.shape.Should().Equal(new long[] { 2, 3 });
            using NDArray back = t.AsNDArray();
            back.Should().NotBeOfType<NDMatrix>();
            np.may_share_memory(back, m).Should().BeTrue();
        }

        /// <summary>An array an NDScope disposes after the export: the tensor's own reference keeps the buffer.</summary>
        [TestMethod]
        public void ScopedArray_OutlivesItsScopeInTheTensor()
        {
            Tensor t;
            using (NDScope.Open())
            {
                NDArray scoped = np.arange(1 << 18).astype(np.float32);   // disposed when the scope ends
                t = scoped.AsTensor(TorchCodecMode.View);
            }
            using (t)
            using (NDArray thief = np.full(new Shape(1 << 18), 9f, np.float32))
                FirstValue(t[12345]).Should().Be(12345, "the scope's dispose released the array's reference, not the tensor's");
        }

        /// <summary>A view whose owner was disposed: the view's own reference keeps the buffer, and the export shares it.</summary>
        [TestMethod]
        public void ViewOfDisposedOwner_Shares()
        {
            NDArray owner = np.arange(12).astype(np.int32).reshape(3, 4);
            using NDArray view = owner["1:, ::2"];
            owner.Dispose();
            using Tensor t = view.AsTensor(TorchCodecMode.View);
            SameBits(view, t).Should().BeTrue();
            view.MayShareMemory(t).Should().BeTrue();
        }

        /// <summary>
        ///     Read-only set AFTER a shared export does not reach back into torch (NumPy's flags are per array object; PyTorch
        ///     keeps writing the memory it was given), and a re-enabled owner exports as a view again.
        /// </summary>
        [TestMethod]
        public void WriteableFlag_ChangedAfterOrBeforeTheExport()
        {
            using NDArray nd = np.arange(4).astype(np.float64);
            using (Tensor t = nd.AsTensor(TorchCodecMode.View))
            {
                nd.setflags(write: false);
                TorchWriteFirst(t, 5);
                nd.GetDouble(0).Should().Be(5, "the tensor still shares the memory it was given");
                Action view = () => nd.AsTensor(TorchCodecMode.View);
                view.Should().Throw<InvalidOperationException>().WithMessage("*read-only*", "a new export reads the new flag");
            }
            nd.setflags(write: true);   // an owner may be made writeable again (NumPy's rule)
            using (Tensor again = nd.AsTensor(TorchCodecMode.View))
                nd.MayShareMemory(again).Should().BeTrue();
        }

        /// <summary>ALIGNED cleared on aligned memory is NumPy's flag, not a hazard: the bridge checks the address and shares.</summary>
        [TestMethod]
        public void AlignedFlagCleared_StillSharesAlignedMemory()
        {
            using NDArray nd = np.arange(6).astype(np.complex128);
            nd.setflags(align: false);
            nd.flags.aligned.Should().BeFalse();
            using Tensor t = nd.AsTensor(TorchCodecMode.View);
            nd.MayShareMemory(t).Should().BeTrue();
        }

        /// <summary>A complex array's .real and .imag (stride-2 float64 views, .imag one element in) share and write through.</summary>
        [TestMethod]
        public void ComplexParts_ShareAndWriteThrough()
        {
            using NDArray c = np.array(new System.Numerics.Complex[] { new(1, 2), new(3, 4) });
            using NDArray re = c.real;
            using NDArray im = c.imag;
            using (Tensor tr = re.AsTensor(TorchCodecMode.View))
            using (Tensor ti = im.AsTensor(TorchCodecMode.View))
            {
                tr.stride().Should().Equal(new long[] { 2 });
                ti.stride().Should().Equal(new long[] { 2 });
                TorchWriteFirst(ti, -9);
            }
            c.ToArray<System.Numerics.Complex>()[0].Should().Be(new System.Numerics.Complex(1, -9));
        }

        /// <summary>A reinterpreting view (<c>view(np.int64)</c> of float64) shares the same bits with torch as int64.</summary>
        [TestMethod]
        public void ReinterpretingView_SharesTheBits()
        {
            using NDArray f = np.array(new[] { 1.5, -2.0 });
            using NDArray bits = f.view(np.int64);
            using Tensor t = bits.AsTensor(TorchCodecMode.View);
            t.dtype.Should().Be(ScalarType.Int64);
            t.data<long>()[0].Should().Be(BitConverter.DoubleToInt64Bits(1.5));
        }

        /// <summary>complex128 data at 8 mod 16 (a reinterpreting view one float64 in) is copied: libtorch assumes 16-byte alignment.</summary>
        [TestMethod]
        public unsafe void Complex128Off16ByteBoundary_Copies()
        {
            using NDArray f = np.arange(10).astype(np.float64);
            using NDArray tail = f["1:9"];
            using NDArray c = tail.view(np.complex128);
            long address = (long)((byte*)c.Storage.InternalArray.Address + c.Shape.offset * 16);
            if (address % 16 == 0)
                Assert.Inconclusive("the pool placed this buffer so that one float64 in is 16-byte aligned (guard-page mode does that)");
            c.flags.aligned.Should().BeTrue("NumPy's alignment for complex128 is 8 bytes");
            using (Tensor t = c.AsTensor())
                c.MayShareMemory(t).Should().BeFalse("the bridge needs 16 bytes for libtorch's c10::complex<double>");
            Action view = () => c.AsTensor(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*not aligned to 16 bytes*");
        }
    }
}
