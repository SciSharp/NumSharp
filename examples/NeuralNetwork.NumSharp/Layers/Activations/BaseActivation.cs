using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.NumSharp.Layers;

namespace NeuralNetwork.NumSharp.Activations
{
    public class BaseActivation : BaseLayer
    {
        public BaseActivation(string name) : base(name)
        {

        }

        /// <summary>
        /// Resolves an activation by its Keras-style name (case-insensitive).
        /// ""/null/"linear"/"none" mean "no activation" and return null; an
        /// unknown name throws instead of silently producing a linear layer
        /// (the historical behavior — "softmax" used to return null because it
        /// was never registered here).
        /// </summary>
        public static BaseActivation Get(string name)
        {
            switch (name?.Trim().ToLowerInvariant())
            {
                case null:
                case "":
                case "linear":
                case "none":
                    return null;
                case "relu":
                    return new ReLU();
                case "sigmoid":
                    return new Sigmoid();
                case "softmax":
                    return new Softmax();
                case "tanh":
                    return new Tanh();
                case "leaky_relu":
                case "leakyrelu":
                    return new LeakyReLU();
                case "elu":
                    return new ELU();
                case "gelu":
                    return new GELU();
                case "silu":
                case "swish":
                    return new SiLU();
                case "softplus":
                    return new Softplus();
                case "selu":
                    return new SELU();
                default:
                    throw new ArgumentException(
                        $"Unknown activation '{name}'. Supported: relu, sigmoid, softmax, tanh, " +
                        "leaky_relu, elu, gelu, silu/swish, softplus, selu " +
                        "(or ''/'linear'/'none' for no activation).", nameof(name));
            }
        }
    }
}
