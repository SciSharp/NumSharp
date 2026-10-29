using System;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.empty_like, np.zeros_like, np.ones_like and np.full_like check the order's letter before the device: NumPy's
    ///     empty_like (which the other three call) converts its arguments in order - dtype, order, subok, shape, then device -
    ///     and allocates last, so the device is still checked before a negative dimension. Probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_like_OrderBeforeDevice_Test
    {
        /// <summary>An unknown order with an unknown device raises the order's ValueError, for each of the four forms.</summary>
        [TestMethod]
        public void UnknownOrder_IsReportedBeforeTheDevice()
        {
            using var a = np.zeros(new Shape(2));
            const string text = "order must be one of 'C', 'F', 'A', or 'K' (got 'Z')";
            Action empty = () => np.empty_like(a, null, default, 'Z', "cuda");
            empty.Should().Throw<ValueError>().WithMessage(text);
            Action zeros = () => np.zeros_like(a, null, 'Z', "cuda");
            zeros.Should().Throw<ValueError>().WithMessage(text);
            Action ones = () => np.ones_like(a, null, 'Z', "cuda");
            ones.Should().Throw<ValueError>().WithMessage(text);
            Action full = () => np.full_like(a, 1, null, 'Z', "cuda");
            full.Should().Throw<ValueError>().WithMessage(text);
        }

        /// <summary>A valid order lets the device be checked, and the device is still checked before the dimensions.</summary>
        [TestMethod]
        public void Device_IsCheckedBeforeANegativeDimension()
        {
            using var a = np.zeros(new Shape(2));
            Action order = () => np.zeros_like(a, null, 'C', "cuda");
            order.Should().Throw<ArgumentException>().WithMessage("Device not understood. Only \"cpu\" is allowed, but received: cuda*");
            Action shape = () => np.empty_like(a, null, new Shape(-1L), 'K', "cuda");
            shape.Should().Throw<ArgumentException>().WithMessage("Device not understood. Only \"cpu\" is allowed, but received: cuda*");
        }
    }
}
