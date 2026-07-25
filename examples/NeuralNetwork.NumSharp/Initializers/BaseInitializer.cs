using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Initializers
{
    /// <summary>
    /// Base class for weight initializers (Keras keras.initializers model).
    /// An initializer maps a parameter shape to a freshly allocated float32
    /// NDArray. All randomness flows through np.random, so results are
    /// deterministic under np.random.seed — same seed, same weights.
    /// </summary>
    public abstract class BaseInitializer
    {
        public string Name { get; }

        protected BaseInitializer(string name)
        {
            Name = name;
        }

        /// <summary>Produce a float32 tensor of the given shape.</summary>
        public abstract NDArray Initialize(Shape shape);

        /// <summary>
        /// Keras's _compute_fans: rank 0 → (1, 1); rank 1 → (n, n);
        /// rank 2 → (rows, cols); rank &gt; 2 (conv kernels) → the two trailing
        /// dims scaled by the receptive-field size prod(shape[:-2]).
        /// </summary>
        protected static (double fanIn, double fanOut) ComputeFans(Shape shape)
        {
            int ndim = shape.NDim;
            if (ndim == 0)
                return (1, 1);
            if (ndim == 1)
                return (shape[0], shape[0]);
            if (ndim == 2)
                return (shape[0], shape[1]);

            double receptive = 1;
            for (int i = 0; i < ndim - 2; i++)
                receptive *= shape[i];
            return (shape[ndim - 2] * receptive, shape[ndim - 1] * receptive);
        }

        /// <summary>
        /// Resolves an initializer by its Keras-style name (case-insensitive).
        /// ""/null return null (meaning "layer default"); an unknown name throws.
        /// </summary>
        public static BaseInitializer Get(string name)
        {
            switch (name?.Trim().ToLowerInvariant())
            {
                case null:
                case "":
                    return null;
                case "zeros": return new Zeros();
                case "ones": return new Ones();
                case "random_normal": return new RandomNormal();
                case "random_uniform": return new RandomUniform();
                case "glorot_uniform":
                case "xavier_uniform": return new GlorotUniform();
                case "glorot_normal":
                case "xavier_normal": return new GlorotNormal();
                case "he_uniform": return new HeUniform();
                case "he_normal": return new HeNormal();
                case "lecun_uniform": return new LecunUniform();
                case "lecun_normal": return new LecunNormal();
                case "orthogonal": return new Orthogonal();
                default:
                    throw new ArgumentException(
                        $"Unknown initializer '{name}'. Supported: zeros, ones, random_normal, " +
                        "random_uniform, glorot_uniform/xavier_uniform, glorot_normal/xavier_normal, " +
                        "he_uniform, he_normal, lecun_uniform, lecun_normal, orthogonal.", nameof(name));
            }
        }
    }
}
