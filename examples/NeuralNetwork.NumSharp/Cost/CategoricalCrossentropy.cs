using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    public class CategoricalCrossentropy : BaseCost
    {
        public CategoricalCrossentropy() : base("categorical_crossentropy")
        {
            
        }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            //var output = Clip(preds, Epsilon, 1 - Epsilon);
            var output = preds;
            output = np.mean(-(labels * np.log(output)));
            return output;
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            //var output = Clip(preds, Epsilon, 1 - Epsilon);
            var output = preds;
            return (output - labels) / output;
        }
    }
}
