using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Softplus activation: y = ln(1 + exp(x)) — a smooth ReLU.
    ///
    /// Forward uses the overflow-safe decomposition
    ///   softplus(x) = max(x, 0) + log1p(exp(-|x|))
    /// so exp never sees a large positive argument (exp(700f) would be +inf
    /// and the naive ln(1+exp(x)) would return +inf instead of ~x).
    ///
    /// Backward: d softplus(x)/dx = sigmoid(x).
    /// </summary>
    public class Softplus : BaseActivation
    {
        public Softplus() : base("softplus") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = np.maximum(x, (NDArray)0f) + np.log1p(np.exp(-np.abs(x)));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * ((NDArray)1f / ((NDArray)1f + np.exp(-Input)));
        }
    }
}
