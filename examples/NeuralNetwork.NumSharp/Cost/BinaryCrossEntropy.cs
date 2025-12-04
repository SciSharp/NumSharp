using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    public class BinaryCrossEntropy : BaseCost
    {
        public BinaryCrossEntropy() : base("binary_crossentropy")
        {
            
        }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            var output = np.clip(preds, Epsilon, 1 - Epsilon);
            output = np.mean(-(labels * np.log(output) + (1 - labels) * np.log(1 - output)));
            return output;
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            var output = np.clip(preds, Epsilon, 1 - Epsilon);
            output = preds;
            return (output - labels) / (output * (1 - output));
        }
    }
}
