using NeuralNetwork.NumSharp.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.NumSharp.Optimizers
{
    public abstract class BaseOptimizer
    {
        public float Epsilon = 1e-7f;

        /// <summary>
        /// Gets or sets the name of the optimizer function
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the learning rate for the optimizer.
        /// </summary>
        /// <value>
        /// The learning rate.
        /// </value>
        public float LearningRate { get; set; }

        /// <summary>
        /// Parameter that accelerates SGD in the relevant direction and dampens oscillations.
        /// </summary>
        /// <value>
        /// The momentum.
        /// </value>
        public float Momentum { get; set; }

        /// <summary>
        /// Learning rate decay over each update.
        /// </summary>
        /// <value>
        /// The decay rate.
        /// </value>
        public float DecayRate { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseOptimizer"/> class.
        /// </summary>
        /// <param name="lr">The lr.</param>
        /// <param name="name">The name.</param>
        public BaseOptimizer(float lr, string name)
        {
            LearningRate = lr;
            Name = name;
        }

        /// <summary>
        /// Updates the specified iteration.
        /// </summary>
        /// <param name="iteration">The iteration.</param>
        /// <param name="layer">The layer.</param>
        public abstract void Update(int iteration, BaseLayer layer);

        /// <summary>
        /// Gets the specified optimizer type.
        /// </summary>
        /// <param name="optimizerType">Type of the optimizer.</param>
        /// <returns></returns>
        public static BaseOptimizer Get(string name)
        {
            BaseOptimizer opt = null;
            switch (name)
            {
                case "sgd":
                    break;
                case "adam":
                    opt = new Adam();
                    break;
                default:
                    break;
            }

            return opt;
        }
    }
}
