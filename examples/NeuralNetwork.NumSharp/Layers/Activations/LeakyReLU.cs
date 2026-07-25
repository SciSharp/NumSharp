using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Leaky rectified linear unit: y = x for x &gt; 0, alpha * x otherwise.
    ///
    /// The default slope alpha = 0.3 matches the Keras LeakyReLU layer default
    /// (PyTorch's nn.LeakyReLU uses 0.01 — pass it explicitly if you want that).
    ///
    /// Backward: dL/dx = dL/dy * (1 for x &gt; 0, alpha otherwise).
    /// </summary>
    public class LeakyReLU : BaseActivation
    {
        public float Alpha { get; }

        public LeakyReLU(float alpha = 0.3f) : base("leaky_relu")
        {
            Alpha = alpha;
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = np.where(x > 0, x, x * Alpha);
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * np.where(Input > 0, (NDArray)1f, (NDArray)Alpha);
        }
    }
}
