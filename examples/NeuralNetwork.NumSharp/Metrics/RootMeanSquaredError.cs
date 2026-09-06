using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>
    /// Root mean squared error: sqrt(mean((preds - labels)^2)).
    /// </summary>
    public class RootMeanSquaredError : BaseMetric
    {
        public RootMeanSquaredError() : base("root_mean_squared_error") { }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            NDArray e = preds - labels;
            return np.sqrt(np.mean(e * e));
        }
    }
}
