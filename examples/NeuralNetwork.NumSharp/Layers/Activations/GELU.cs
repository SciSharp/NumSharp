using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Gaussian error linear unit (Hendrycks &amp; Gimpel, 2016), tanh
    /// approximation — the form used by GPT/BERT-family models:
    ///
    ///   y = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    ///
    /// Keras's gelu(approximate=False) and PyTorch's default use the exact
    /// erf formulation; NumSharp has no np.erf yet, so this class implements
    /// approximate=True. The two agree to ~1e-3 absolute over the useful range.
    ///
    /// Backward differentiates the approximation directly, reusing the cached
    /// tanh term t:
    ///   u  = sqrt(2/pi) * (x + 0.044715 x^3)
    ///   dy/dx = 0.5 (1 + t) + 0.5 x (1 - t^2) * sqrt(2/pi) * (1 + 3 * 0.044715 x^2)
    /// </summary>
    public class GELU : BaseActivation
    {
        private const float K = 0.7978845608028654f;  // sqrt(2/pi)
        private const float C = 0.044715f;

        private NDArray _tanhCache;

        public GELU() : base("gelu") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            NDArray inner = (x + x * x * x * C) * K;
            _tanhCache = np.tanh(inner);
            Output = x * ((NDArray)1f + _tanhCache) * 0.5f;
        }

        public override void Backward(NDArray grad)
        {
            NDArray x = Input;
            NDArray t = _tanhCache;
            NDArray sech2 = (NDArray)1f - t * t;                       // 1 - tanh^2(u)
            NDArray du = ((NDArray)1f + x * x * (3f * C)) * K;         // du/dx
            NDArray dydx = ((NDArray)1f + t) * 0.5f + x * sech2 * du * 0.5f;
            InputGrad = grad * dydx;
        }
    }
}
