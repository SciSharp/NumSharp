using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Hyperbolic tangent activation: y = tanh(x), range (-1, 1).
    ///
    /// Backward uses the closed-form derivative on the cached forward output:
    ///   d tanh(x)/dx = 1 - tanh(x)^2
    ///   dL/dx = dL/dy * (1 - y^2)
    /// </summary>
    public class Tanh : BaseActivation
    {
        public Tanh() : base("tanh") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = np.tanh(x);
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * ((NDArray)1f - Output * Output);
        }
    }
}
