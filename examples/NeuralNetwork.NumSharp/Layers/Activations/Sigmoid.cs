using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Element-wise sigmoid activation: sigma(x) = 1 / (1 + exp(-x)).
    ///
    /// Forward uses the "pseudo-stable" form where exp(-x) is clamped at
    /// the far ends by the saturation of sigma itself — exp(-large x)
    /// underflows to 0 (giving 1.0) and exp(-very-negative) overflows to
    /// +inf (giving 0.0). Both are correct limits, so no extra clipping
    /// is required for standard float32 inputs.
    ///
    /// Backward uses the closed-form derivative that re-uses the cached
    /// forward output:
    ///   d sigma(x)/dx = sigma(x) * (1 - sigma(x))
    ///   dL/dx = dL/dy * sigma * (1 - sigma)
    /// </summary>
    public class Sigmoid : BaseActivation
    {
        public Sigmoid() : base("sigmoid") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = (NDArray)1.0 / ((NDArray)1.0 + np.exp(-x));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * Output * ((NDArray)1.0 - Output);
        }
    }
}
