using System;
using NeuralNetwork.NumSharp.Activations;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Fully connected (dense) layer with a bias term and an optional
    /// activation applied after the affine transform:
    ///
    ///   y = activation(x @ W + b)
    ///
    /// Weights are initialized with He-normal when the attached activation
    /// is ReLU (preserves variance through the non-linearity) and Xavier/
    /// Glorot otherwise. Both weights and bias are float32 to stay on the
    /// SIMD-capable fast paths in NumSharp.
    ///
    /// The layer populates the standard <see cref="BaseLayer"/> slots —
    /// Parameters["w"], Parameters["b"], Grads["w"], Grads["b"] — so the
    /// stock Adam / SGD optimizers iterate it unchanged.
    /// </summary>
    public class FullyConnected : BaseLayer
    {
        public int InputDim  { get; set; }
        public int OutNeurons { get; set; }
        public bool UseBias { get; set; }
        public BaseActivation Activation { get; set; }

        public FullyConnected(int input_dim, int output_neurons, string act = "", bool useBias = true)
            : base("fc")
        {
            InputDim   = input_dim;
            OutNeurons = output_neurons;
            UseBias    = useBias;
            Activation = BaseActivation.Get(act);

            // He init for ReLU; Xavier for everything else (linear, sigmoid, softmax, ...).
            bool isReLU = string.Equals(act, "relu", StringComparison.OrdinalIgnoreCase);
            double stddev = isReLU
                ? Math.Sqrt(2.0 /  input_dim)
                : Math.Sqrt(2.0 / (input_dim + output_neurons));

            Parameters["w"] = np.random.normal(0.0, stddev, new Shape(input_dim, output_neurons))
                                       .astype(NPTypeCode.Single);
            if (UseBias)
                Parameters["b"] = np.zeros(new Shape(output_neurons), NPTypeCode.Single);
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            NDArray preact = np.dot(x, Parameters["w"]);
            if (UseBias)
                preact = preact + Parameters["b"];

            if (Activation != null)
            {
                Activation.Forward(preact);
                Output = Activation.Output;
            }
            else
            {
                Output = preact;
            }
        }

        public override void Backward(NDArray grad)
        {
            if (Activation != null)
            {
                Activation.Backward(grad);
                grad = Activation.InputGrad;
            }

            NDArray W = Parameters["w"];

            // np.dot ships a stride-aware GEMM (BLIS-style packing), so the
            // transposed views go through the SIMD fast path directly — no
            // need to materialize contiguous copies.
            Grads["w"] = np.dot(Input.transpose(), grad);
            if (UseBias)
                Grads["b"] = np.sum(grad, axis: 0);

            InputGrad = np.dot(grad, W.transpose());
        }
    }
}
