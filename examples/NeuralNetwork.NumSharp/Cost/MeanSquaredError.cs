using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    public class MeanSquaredError : BaseCost
    {
        public MeanSquaredError() : base("mean_squared_error")
        {
            
        }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            var error = preds - labels;
            return np.mean(np.power(error, 2));
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            float norm = 2 / (float)preds.shape[0];
            return norm * (preds - labels);
        }
    }
}
