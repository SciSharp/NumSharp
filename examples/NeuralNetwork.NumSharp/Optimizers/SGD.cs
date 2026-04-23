using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.NumSharp.Layers;
using NumSharp;

namespace NeuralNetwork.NumSharp.Optimizers
{
    /// <summary>
    /// Stochastic gradient descent with optional classical momentum.
    ///
    ///   Without momentum (the default):
    ///     param &lt;- param - lr * grad
    ///
    ///   With momentum mu &gt; 0 (heavy-ball):
    ///     v     &lt;- mu * v - lr * grad
    ///     param &lt;- param + v
    ///
    /// <see cref="BaseOptimizer.DecayRate"/> applies an inverse-time decay
    /// to the learning rate (lr_t = lr / (1 + decay * iteration)) matching
    /// the Adam optimizer's convention.
    /// </summary>
    public class SGD : BaseOptimizer
    {
        private readonly Dictionary<string, NDArray> velocities = new Dictionary<string, NDArray>();

        public SGD(float lr = 0.01f, float momentum = 0f, float decayRate = 0f)
            : base(lr, "sgd")
        {
            Momentum  = momentum;
            DecayRate = decayRate;
        }

        public override void Update(int iteration, BaseLayer layer)
        {
            if (DecayRate > 0)
                LearningRate = LearningRate * (1f / (1f + DecayRate * iteration));

            foreach (var p in layer.Parameters.ToList())
            {
                string paramName = p.Key;
                string varName   = layer.Name + "_" + paramName;
                NDArray param    = p.Value;
                NDArray grad     = layer.Grads[paramName];

                if (Momentum > 0f)
                {
                    if (!velocities.ContainsKey(varName))
                        velocities[varName] = np.zeros(param.Shape, param.dtype);

                    velocities[varName] = Momentum * velocities[varName] - LearningRate * grad;
                    layer.Parameters[paramName] = param + velocities[varName];
                }
                else
                {
                    layer.Parameters[paramName] = param - LearningRate * grad;
                }
            }
        }
    }
}
