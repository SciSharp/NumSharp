using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Cost
{
    /// <summary>
    /// Categorical cross-entropy taking INTEGER class labels instead of a
    /// one-hot matrix (Keras SparseCategoricalCrossentropy). Avoids
    /// materializing the (batch, numClasses) one-hot — the memory win grows
    /// with the class count (vocabularies, large label spaces).
    ///
    /// Expects:
    ///   preds  — probabilities (post-softmax), float32 (batch, numClasses)
    ///   labels — integer class indices (batch,) of dtype Byte, Int32 or Int64
    ///            (the three this project's loaders and np.argmax produce)
    ///
    /// Forward:  L = -mean_i( log(clip(preds[i, labels[i]], eps, 1-eps)) )
    /// Backward: dL/dpreds[i, j] = -1 / (clip(preds[i, labels[i]]) * batch)  for j == labels[i]
    ///                             0                                          otherwise
    ///
    /// The gather runs as an explicit strided loop (np.take_along_axis is not
    /// in NumSharp core yet — see ROADMAP core backlog); batch loops of this
    /// size are noise next to the matmuls.
    /// </summary>
    public class SparseCategoricalCrossentropy : BaseCost
    {
        public SparseCategoricalCrossentropy() : base("sparse_categorical_crossentropy") { }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            RequireSingle(preds);
            int batch = (int)preds.shape[0];
            int classes = (int)preds.shape[1];

            double sum = 0.0;
            for (int i = 0; i < batch; i++)
            {
                int li = LabelAt(labels, i, classes);
                float p = Clip(preds.GetSingle(i, li));
                sum += Math.Log(p);
            }

            return NDArray.Scalar((float)(-sum / batch));
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            RequireSingle(preds);
            int batch = (int)preds.shape[0];
            int classes = (int)preds.shape[1];

            NDArray grad = np.zeros(new Shape(batch, classes), NPTypeCode.Single);
            unsafe
            {
                float* g = (float*)grad.Unsafe.Address;
                for (int i = 0; i < batch; i++)
                {
                    int li = LabelAt(labels, i, classes);
                    float p = Clip(preds.GetSingle(i, li));
                    g[i * classes + li] = -1f / (p * batch);
                }
            }

            return grad;
        }

        private float Clip(float p)
        {
            if (p < Epsilon) return Epsilon;
            float hi = 1f - Epsilon;
            return p > hi ? hi : p;
        }

        private static void RequireSingle(NDArray preds)
        {
            if (preds.typecode != NPTypeCode.Single)
                throw new NotSupportedException(
                    $"SparseCategoricalCrossentropy expects float32 predictions (framework convention), got {preds.typecode}.");
        }

        private static int LabelAt(NDArray labels, int i, int numClasses)
        {
            int label = labels.typecode switch
            {
                NPTypeCode.Byte  => labels.GetByte(i),
                NPTypeCode.Int32 => labels.GetInt32(i),
                NPTypeCode.Int64 => (int)labels.GetInt64(i),
                _ => throw new NotSupportedException(
                    $"SparseCategoricalCrossentropy doesn't support label dtype {labels.typecode}."),
            };
            if ((uint)label >= (uint)numClasses)
                throw new ArgumentOutOfRangeException(nameof(labels),
                    $"label at index {i} = {label} is outside [0,{numClasses}).");
            return label;
        }
    }
}
