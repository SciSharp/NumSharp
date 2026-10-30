using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     Tests for the Array-API <c>device</c> surface — verified 1-to-1 against NumPy 2.4.2.
    ///     NumSharp, like NumPy, is single-device (CPU-only), so <c>device</c> is a conformance shim:
    ///     <c>ndarray.device</c> is always <c>"cpu"</c>, <c>ndarray.to_device</c> accepts only <c>"cpu"</c>
    ///     (returning the SAME array), and the blessed creation functions accept <c>device="cpu"</c>/<c>null</c>
    ///     while rejecting anything else with NumPy's verbatim messages. The two rejection messages are
    ///     deliberately different (single- vs double-quoted <c>cpu</c>), matching NumPy.
    /// </summary>
    [TestClass]
    public class np_device_Test
    {
        // ─── ndarray.device is always "cpu" ─────────────────────────────────

        [TestMethod]
        public void Device_IsCpu_ForFreshArray()
            => np.zeros(new Shape(3)).device.Should().Be("cpu");

        [TestMethod]
        public void Device_IsCpu_ForScalar()
            => NDArray.Scalar(5.0).device.Should().Be("cpu");

        [TestMethod]
        public void Device_IsCpu_ForView()
        {
            // A slice / transposed / broadcast view is still CPU-resident.
            var a = np.arange(12.0).reshape(3, 4);
            a["1:3, ::2"].device.Should().Be("cpu");
            a.T.device.Should().Be("cpu");
            np.broadcast_to(np.array(new[] { 1.0, 2.0, 3.0 }), new Shape(4, 3)).device.Should().Be("cpu");
        }

        // ─── ndarray.to_device ──────────────────────────────────────────────

        [TestMethod]
        public void ToDevice_Cpu_ReturnsSameInstance()
        {
            // NumPy returns `self` (Py_INCREF) — no copy. NumSharp returns `this`.
            var a = np.arange(5.0);
            a.to_device("cpu").Should().BeSameAs(a);
        }

        /// <summary>NumPy 2.4.2: <c>np.zeros(3).to_device('gpu')</c> raises ValueError, <c>Unsupported device: gpu. Only
        ///     'cpu' is accepted.</c> (ArgumentException before the device errors were made NumPy's type).</summary>
        [TestMethod]
        public void ToDevice_BadDevice_Throws_SingleQuoted()
            => new Action(() => np.zeros(new Shape(3)).to_device("gpu")).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("Unsupported device: gpu. Only 'cpu' is accepted.");

        /// <summary>NumPy 2.4.2: <c>to_device('cuda')</c> and the case-sensitive <c>to_device('CPU')</c> raise the same
        ///     ValueError text.</summary>
        [TestMethod]
        public void ToDevice_Cuda_Throws()
        {
            new Action(() => np.zeros(new Shape(3)).to_device("cuda")).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("Unsupported device: cuda. Only 'cpu' is accepted.");
            new Action(() => np.zeros(new Shape(3)).to_device("CPU")).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("Unsupported device: CPU. Only 'cpu' is accepted.");
        }

        [TestMethod]
        public void ToDevice_Null_Throws()
            // NumPy raises TypeError ("argument 1 must be str, not None"); the C# analog is ArgumentNullException.
            => new Action(() => np.zeros(new Shape(3)).to_device(null)).Should()
                .Throw<ArgumentNullException>();

        /// <summary>NumPy 2.4.2: <c>to_device('cpu', stream=1)</c> raises ValueError, <c>The stream argument in to_device()
        ///     is not supported</c>.</summary>
        [TestMethod]
        public void ToDevice_Stream_Throws()
            => new Action(() => np.zeros(new Shape(3)).to_device("cpu", stream: new object())).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("The stream argument in to_device() is not supported");

        [TestMethod]
        public void ToDevice_BadStream_BeatsBadDevice()
            // NumPy checks the stream before the device VALUE, so a bad-device + bad-stream call
            // reports the stream error first (both non-cpu; stream wins).
            => new Action(() => np.zeros(new Shape(3)).to_device("gpu", stream: new object())).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("The stream argument in to_device() is not supported");

        // ─── creation device= — the double-quoted rejection message ─────────

        static Action Reject(Action call) => call;

        /// <summary>NumPy 2.4.2's text for <c>device='gpu'</c> on every creation function (probed: zeros, ones, empty,
        ///     full, eye, arange, linspace, asarray, asanyarray, zeros_like, np.fft.fftfreq).</summary>
        const string BadDeviceText = "Device not understood. Only \"cpu\" is allowed, but received: gpu";

        /// <summary>NumPy 2.4.2: <c>np.zeros(3, device='gpu')</c> raises ValueError with the text alone. NumSharp raised an
        ///     ArgumentException whose message .NET extended with <c> (Parameter 'device')</c>.</summary>
        [TestMethod]
        public void Creation_BadDevice_Throws_DoubleQuoted()
            => new Action(() => np.zeros(new Shape(3), typeof(double), device: "gpu")).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("Device not understood. Only \"cpu\" is allowed, but received: gpu");

        /// <summary>NumPy 2.4.2: the device name is compared exactly, so <c>'CPU'</c> is refused, and an empty name is
        ///     quoted as nothing (<c>... but received: </c>).</summary>
        [TestMethod]
        public void Creation_BadDevice_CaseAndEmpty_AreNumPysText()
        {
            new Action(() => np.zeros(new Shape(3), typeof(double), device: "CPU")).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("Device not understood. Only \"cpu\" is allowed, but received: CPU");
            new Action(() => np.zeros(new Shape(3), typeof(double), device: "")).Should()
                .ThrowExactly<ValueError>()
                .WithMessage("Device not understood. Only \"cpu\" is allowed, but received: ");
        }

        [TestMethod]
        public void Creation_BadDevice_Throws_AcrossBlessedConstructors()
        {
            var a = np.arange(6.0).reshape(2, 3);

            Reject(() => np.zeros(new Shape(3), typeof(double), device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.ones(new Shape(3), typeof(double), device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.empty(new Shape(3), typeof(double), device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.full(new Shape(3), 1.0, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.eye(3, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.asanyarray(new[] { 1, 2, 3 }, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.asarray(a, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);

            Reject(() => np.zeros_like(a, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.ones_like(a, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.empty_like(a, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.full_like(a, 1.0, device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);

            Reject(() => np.arange(5.0, typeof(int), device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
            Reject(() => np.linspace(0.0, 1.0, 5, endpoint: true, dtype: typeof(double), device: "gpu")).Should().ThrowExactly<ValueError>().WithMessage(BadDeviceText);
        }

        // ─── creation device="cpu"/null are accepted and produce the same array ──

        [TestMethod]
        public void Creation_CpuAndNull_Accepted_Zeros()
        {
            np.zeros(new Shape(2, 3), typeof(double), device: "cpu").Should().NotBeNull();
            np.zeros(new Shape(2, 3), typeof(double), device: null).Should().NotBeNull();
            // device has no effect on the values/shape/dtype.
            var withCpu = np.zeros(new Shape(4), typeof(int), device: "cpu");
            withCpu.shape.Should().Equal(new long[] { 4 });
            withCpu.typecode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Creation_Cpu_Accepted_AcrossBlessedConstructors()
        {
            var a = np.arange(6.0).reshape(2, 3);

            np.ones(new Shape(3), typeof(double), device: "cpu").Should().NotBeNull();
            np.empty(new Shape(3), typeof(double), device: "cpu").Should().NotBeNull();
            np.full(new Shape(3), 7.0, device: "cpu").Should().NotBeNull();
            np.eye(3, device: "cpu").Should().NotBeNull();
            np.asanyarray(new[] { 1, 2, 3 }, device: "cpu").Should().NotBeNull();

            np.zeros_like(a, device: "cpu").shape.Should().Equal(a.shape);
            np.ones_like(a, device: "cpu").shape.Should().Equal(a.shape);
            np.empty_like(a, device: "cpu").shape.Should().Equal(a.shape);
            np.full_like(a, 1.0, device: "cpu").shape.Should().Equal(a.shape);

            np.arange(5.0, typeof(int), device: "cpu").size.Should().Be(5);
            np.linspace(0.0, 1.0, 5, endpoint: true, dtype: typeof(double), device: "cpu").size.Should().Be(5);
        }

        [TestMethod]
        public void Creation_Device_HasNoEffectOnResult()
        {
            // A device="cpu" result is byte-identical to the deviceless call.
            var withDevice = np.arange(0.0, 5.0, 1.0, typeof(int), device: "cpu");
            var without = np.arange(0.0, 5.0, 1.0, typeof(int));
            withDevice.array_equal(without).Should().BeTrue();
        }
    }
}
