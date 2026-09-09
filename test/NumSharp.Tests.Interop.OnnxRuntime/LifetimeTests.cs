using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>The lifetime contract (plan §7): ARC pins, deterministic release, the finalizer safety nets, resize refusal.</summary>
    [TestClass]
    public class LifetimeTests : OnnxTestBase
    {
        [TestMethod]
        public void SourceDisposed_WhileExported_TheBufferStaysValid_UntilTheHandleGoes()
        {
            NDArray nd = Arange(NPTypeCode.Int32, 6);
            OrtTensor t = nd.AsOrtValue();
            NDArrayOnnxInterop.LiveExports.Should().Be(1);

            nd.Dispose();   // the array is gone; the handle's ARC reference keeps the memory
            nd.IsDisposed.Should().BeTrue();
            Settle();
            t.Value.GetTensorDataAsSpan<int>().ToArray().Should().Equal(new[] { 0, 1, 2, 3, 4, 5 }, "ORT still reads valid data");

            t.Dispose();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void Dispose_IsIdempotent_AndValueThrowsAfterwards()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 3);
            OrtTensor t = nd.AsOrtValue();
            t.IsDisposed.Should().BeFalse();
            t.Dispose();
            t.Dispose();
            t.IsDisposed.Should().BeTrue();
            new Action(() => _ = t.Value).Should().Throw<ObjectDisposedException>();
            t.Source.Should().BeSameAs(nd, "metadata stays readable");
            t.Shape.Should().Equal(3L);
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a double dispose releases once");

            using NDArray nd2 = Arange(NPTypeCode.Single, 3);
            OrtTensor<float> h = nd2.AsDenseTensor<float>();
            h.Dispose();
            h.Dispose();
            new Action(() => _ = h.Tensor).Should().Throw<ObjectDisposedException>();
            new Action(() => _ = h.Memory).Should().Throw<ObjectDisposedException>();
        }

        [TestMethod]
        public void MemoryOutlivingItsHandle_FailsLoudly_InsteadOfReadingFreedMemory()
        {
            using NDArray nd = Arange(NPTypeCode.Double, 4);
            Memory<double> escaped;
            using (OrtTensor<double> h = nd.AsDenseTensor<double>())
                escaped = h.Memory;
            new Action(() => _ = escaped.Span).Should().Throw<ObjectDisposedException>();
        }

        [TestMethod]
        public void Resize_WithRefcheck_RefusesWhileExported_AndWorksAgainAfterwards()
        {
            using NDArray nd = Arange(NPTypeCode.Int64, 4);
            using (OrtTensor t = nd.AsOrtValue())
            {
                new Action(() => nd.resize(8)).Should().Throw<Exception>()
                    .WithMessage("*referenc*", "the exported buffer is not uniquely referenced — NumPy's refcheck rule");
                t.Value.GetTensorDataAsSpan<long>().Length.Should().Be(4, "nothing moved under ORT");
            }

            nd.resize(8);
            nd.size.Should().Be(8, "once the export is released the array is uniquely referenced again");
        }

        [TestMethod]
        public void ForgottenHandle_IsReleasedByTheFinalizer_SafetyNet()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 16);
            LeakAHandle(nd);
            NDArrayOnnxInterop.LiveExports.Should().Be(1, "not yet collected");

            WaitFor(() => NDArrayOnnxInterop.LiveExports == 0, 10_000).Should().BeTrue("the finalizer dropped the ARC pin");
            nd.GetSingle(15).Should().Be(15f, "the source is untouched");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void LeakAHandle(NDArray nd)
        {
            OrtTensor t = nd.AsOrtValue();
            GC.KeepAlive(t);
        }

        [TestMethod]
        public void ForgottenView_IsReleasedByCollection_SafetyNet()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray nd = Arange(NPTypeCode.Single, 8);
            using OrtTensor input = nd.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "Y" });

            LeakAView(outputs[0]);
            NDArrayOnnxInterop.LiveImports.Should().Be(1);
            WaitFor(() => NDArrayOnnxInterop.LiveImports == 0, 10_000).Should().BeTrue("the memory-block finalizer fired the lease's release hook");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void LeakAView(OrtValue value)
        {
            NDArray view = value.AsNDArray();
            GC.KeepAlive(view);
        }

        [TestMethod]
        public void ManyHandlesOnOneBuffer_EachHoldsItsOwnReference()
        {
            using NDArray nd = Arange(NPTypeCode.Byte, 10);
            var handles = new OrtTensor[5];
            for (int i = 0; i < handles.Length; i++)
                handles[i] = nd.AsOrtValue();
            NDArrayOnnxInterop.LiveExports.Should().Be(5);
            nd.GetData().IsUniquelyReferenced.Should().BeFalse();

            for (int i = 0; i < handles.Length; i++)
            {
                handles[i].Dispose();
                NDArrayOnnxInterop.LiveExports.Should().Be(handles.Length - 1 - i);
            }

            nd.GetData().IsUniquelyReferenced.Should().BeTrue("all pins released; the array owns its buffer alone again");
            nd.GetValue<byte>(9).Should().Be((byte)9);
        }

        [TestMethod]
        public void DisposedSource_CannotBeExported()
        {
            NDArray nd = Arange(NPTypeCode.Single, 3);
            nd.Dispose();
            new Action(() => nd.AsOrtValue()).Should().Throw<ObjectDisposedException>();
            new Action(() => nd.ToOrtValue()).Should().Throw<ObjectDisposedException>();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }
    }
}
