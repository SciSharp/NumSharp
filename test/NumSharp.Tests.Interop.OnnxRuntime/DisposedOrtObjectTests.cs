using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     ONNX Runtime objects the CALLER disposed: a session and a bare value. ONNX Runtime checks neither, and a native
    ///     call on a released one is an access violation that ends the process (measured on 1.31.1: <c>session.Run</c>
    ///     after <c>Dispose</c>, raw ORT and through this package alike; <c>ToNDArray</c> of a disposed
    ///     <c>ToOrtValue</c> result in <c>GetTensorTypeAndShape</c>). <see cref="OrtDisposalProbe"/> reads ORT's own
    ///     private disposal flag, so every Run form and every import verb throws <see cref="ObjectDisposedException"/>
    ///     instead. A test that regressed would not fail but END the test host, so each one first asks whether the loaded
    ///     ONNX Runtime has the flag and is Inconclusive when it does not.
    /// </summary>
    [TestClass]
    public class DisposedOrtObjectTests : OnnxTestBase
    {
        /// <summary>
        ///     Every Run form on a disposed session throws <see cref="ObjectDisposedException"/> before anything is
        ///     exported (the single and named forms through the dictionary form; the asynchronous forms from the call,
        ///     where ONNX Runtime 1.30.0 or newer is loaded). Nothing was exported or leased.
        /// </summary>
        [TestMethod]
        public void EveryRunForm_OnADisposedSession_ThrowsObjectDisposed_InsteadOfEndingTheProcess()
        {
            if (!OrtDisposalProbe.CanTellSessions)
                Assert.Inconclusive("the loaded ONNX Runtime's InferenceSession has no private bool _disposed: the guard is off, and a run would end the test host");
            InferenceSession session = OpenSession("identity_float32");
            session.Dispose();
            using NDArray x = Arange(NPTypeCode.Single, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            new Action(() => session.Run(x)).Should().Throw<ObjectDisposedException>().WithMessage("Run: the InferenceSession has been disposed*");
            new Action(() => session.Run("X", x, "Y")).Should().Throw<ObjectDisposedException>();
            new Action(() => session.Run(inputs)).Should().Throw<ObjectDisposedException>();
            new Action(() => session.Run(inputs, zeroCopyOutputs: true)).Should().Throw<ObjectDisposedException>();
            using (NDArray y = np.zeros(new Shape(3), NPTypeCode.Single))
            {
                new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", y } })).Should().Throw<ObjectDisposedException>()
                    .WithMessage("RunInto: the InferenceSession has been disposed*");
                if (InferenceSessionExtensions.IsRunAsyncSupported)
                {
                    new Action(() => session.RunIntoAsync(inputs, new Dictionary<string, NDArray> { { "Y", y } })).Should().Throw<ObjectDisposedException>()
                        .WithMessage("RunIntoAsync: the InferenceSession has been disposed*");
                    new Action(() => session.RunAsync(inputs)).Should().Throw<ObjectDisposedException>().WithMessage("RunAsync: *");
                }
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the session is refused before any input is exported");
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>
        ///     Every import verb refuses a bare value the caller disposed - a <c>ToOrtValue</c> result, a sequence, a string
        ///     tensor - with <see cref="ObjectDisposedException"/> before its first native call, whatever kind the verb
        ///     expects (the check comes before the kind check). A live value still reads.
        /// </summary>
        [TestMethod]
        public void EveryImportVerb_OfADisposedBareValue_ThrowsObjectDisposed_InsteadOfEndingTheProcess()
        {
            if (!OrtDisposalProbe.CanTellValues)
                Assert.Inconclusive("the loaded ONNX Runtime's OrtValue has no private bool _disposed: the guard is off, and a read would end the test host");
            using NDArray x = Arange(NPTypeCode.Single, 4);

            OrtValue tensor = x.ToOrtValue();
            using (NDArray live = tensor.ToNDArray())
                live.ToArray<float>().Should().Equal(0f, 1f, 2f, 3f);
            tensor.Dispose();
            new Action(() => tensor.ToNDArray()).Should().Throw<ObjectDisposedException>().WithMessage("ToNDArray: this OrtValue has been disposed*");
            new Action(() => tensor.AsNDArray()).Should().Throw<ObjectDisposedException>().WithMessage("AsNDArray: *");
            new Action(() => tensor.AsNDArray(ownsValue: true)).Should().Throw<ObjectDisposedException>();
            new Action(() => tensor.ToNDArrays()).Should().Throw<ObjectDisposedException>().WithMessage("ToNDArrays: *");
            new Action(() => tensor.ToMap()).Should().Throw<ObjectDisposedException>().WithMessage("ToMap: *");
            new Action(() => tensor.ToMaps()).Should().Throw<ObjectDisposedException>().WithMessage("ToMaps: *");
            new Action(() => tensor.GetMapKeys()).Should().Throw<ObjectDisposedException>().WithMessage("GetMapKeys: *");
            new Action(() => tensor.ReadStringTensor()).Should().Throw<ObjectDisposedException>().WithMessage("ReadStringTensor: *");

            // a sequence of tensors and a string tensor, disposed by the caller
            OrtValue first = x.ToOrtValue();
            OrtValue sequence = OrtValue.CreateSequence(new List<OrtValue> { first });   // the sequence owns its element
            sequence.Dispose();
            new Action(() => sequence.ToNDArrays()).Should().Throw<ObjectDisposedException>();

            OrtValue text = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 2 });
            text.Dispose();
            new Action(() => text.ReadStringTensor()).Should().Throw<ObjectDisposedException>();
            new Action(() => text.ToNDArray()).Should().Throw<ObjectDisposedException>();

            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>
        ///     The probe reads ONNX Runtime's own flag: false for a live session and value, true after their
        ///     <c>Dispose</c>. This is what the two guards stand on; were the field renamed, the getters would be null and
        ///     the tests above Inconclusive, never a crash.
        /// </summary>
        [TestMethod]
        public void Probe_ReadsOnnxRuntimesOwnDisposalFlag()
        {
            if (!OrtDisposalProbe.CanTellSessions || !OrtDisposalProbe.CanTellValues)
                Assert.Inconclusive("the loaded ONNX Runtime has no private bool _disposed on InferenceSession / OrtValue");
            InferenceSession session = OpenSession("identity_float32");
            OrtDisposalProbe.IsDisposed(session).Should().BeFalse();
            session.Dispose();
            OrtDisposalProbe.IsDisposed(session).Should().BeTrue();

            using NDArray x = Arange(NPTypeCode.Single, 2);
            OrtValue value = x.ToOrtValue();
            OrtDisposalProbe.IsDisposed(value).Should().BeFalse();
            value.Dispose();
            OrtDisposalProbe.IsDisposed(value).Should().BeTrue();
        }
    }
}
