using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.linspace / np.logspace / np.geomspace at the edges the snapshot pools and the older tests did not reach,
    ///     found when the np-layer dtype plan's P3 batch 9 made them engine members and replayed them against NumPy 2.4.2.
    ///     Every expectation is NumPy 2.4.2's output (values as raw float64 bytes where a NaN's bits or a zero's sign is
    ///     the point) or its error text.
    /// </summary>
    [TestClass]
    public class np_spacing_numpy_edges_tests
    {
        // ------------------------------------------------------------------------------------------- the sample count

        /// <summary>
        ///     NumPy's linspace starts with <c>if num &lt; 0: raise ValueError("Number of samples, ...")</c>, before it
        ///     reads the device (the ramp's np.arange does) and before anything is allocated - NumSharp reported the
        ///     allocation's <c>negative dimensions are not allowed</c>, or the device error when both were bad.
        /// </summary>
        [TestMethod]
        public void Linspace_NegativeCount_IsNumPysValueError_First()
        {
            // np.linspace(0, 1, -5) -> ValueError: Number of samples, -5, must be non-negative.
            new Action(() => np.linspace(0.0, 1.0, -5))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -5, must be non-negative.");
            new Action(() => np.linspace(0.0, 1.0, -1L, endpoint: false))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
            new Action(() => np.linspace(0f, 1f, -2))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -2, must be non-negative.");
            // np.linspace(0, 1, -1, device='cuda') -> the count's ValueError, not the device's
            new Action(() => np.linspace(0.0, 1.0, -1, device: "cuda"))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
            // np.linspace(0, 1, -1, dtype=np.int32) -> the count's ValueError
            new Action(() => np.linspace(0.0, 1.0, -1, dtype: np.int32))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
        }

        /// <summary>With a valid count, a device other than "cpu" is refused (NumPy's ValueError text).</summary>
        [TestMethod]
        public void Linspace_BadDevice_AfterAValidCount()
        {
            // np.linspace(0, 1, 3, device='cuda') -> ValueError: Device not understood. Only "cpu" is allowed, but received: cuda
            new Action(() => np.linspace(0.0, 1.0, 3, device: "cuda"))
                .Should().Throw<ArgumentException>().WithMessage("Device not understood. Only \"cpu\" is allowed, but received: cuda*");
            np.linspace(0.0, 1.0, 3, device: "cpu").Should().BeShaped(3);
        }
    }
}
