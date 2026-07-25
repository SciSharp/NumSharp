using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Sigmoid linear unit, a.k.a. Swish (Ramachandran et al., 2017):
    ///   y = x * sigmoid(x)
    ///
    /// Backward reuses the cached sigmoid s:
    ///   dy/dx = s + x * s * (1 - s)
    ///   dL/dx = dL/dy * dy/dx
    /// </summary>
    public class SiLU : BaseActivation
    {
        private NDArray _sigmoidCache;

        public SiLU() : base("silu") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            _sigmoidCache = (NDArray)1f / ((NDArray)1f + np.exp(-x));
            Output = x * _sigmoidCache;
        }

        public override void Backward(NDArray grad)
        {
            NDArray s = _sigmoidCache;
            InputGrad = grad * (s + Input * s * ((NDArray)1f - s));
        }
    }
}
