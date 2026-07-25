using System;
using NeuralNetwork.NumSharp.Activations;
using NeuralNetwork.NumSharp.Initializers;
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

        /// <summary>
        /// The activation as NAMED at construction. <see cref="Activation"/> is
        /// the resolved instance and is null for the linear case, so the name has
        /// to be kept separately for <see cref="GetConfig"/> to round-trip.
        /// </summary>
        public string ActivationName { get; }

        public FullyConnected(int input_dim, int output_neurons, string act = "", bool useBias = true,
                              BaseInitializer kernelInitializer = null, BaseInitializer biasInitializer = null)
            : base("fc")
        {
            InputDim   = input_dim;
            OutNeurons = output_neurons;
            UseBias    = useBias;
            ActivationName = act ?? "";
            Activation = BaseActivation.Get(act);

            if (kernelInitializer != null)
            {
                Parameters["w"] = kernelInitializer.Initialize(new Shape(input_dim, output_neurons));
            }
            else
            {
                // Historical default kept bit-for-bit (seeded runs stay reproducible):
                // UNtruncated He normal for ReLU, Xavier/Glorot normal otherwise.
                // Pass an Initializers.* instance for the Keras-exact (truncated)
                // variants.
                bool isReLU = string.Equals(act, "relu", StringComparison.OrdinalIgnoreCase);
                double stddev = isReLU
                    ? Math.Sqrt(2.0 /  input_dim)
                    : Math.Sqrt(2.0 / (input_dim + output_neurons));

                Parameters["w"] = np.random.normal(0.0, stddev, new Shape(input_dim, output_neurons))
                                           .astype(NPTypeCode.Single);
            }

            if (UseBias)
                Parameters["b"] = biasInitializer != null
                    ? biasInitializer.Initialize(new Shape(output_neurons))
                    : np.zeros(new Shape(output_neurons), NPTypeCode.Single);
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

        public override Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig("FullyConnected")
                .Set("input_dim", InputDim)
                .Set("units", OutNeurons)
                .Set("activation", ActivationName)
                .Set("use_bias", UseBias);
    }
}
