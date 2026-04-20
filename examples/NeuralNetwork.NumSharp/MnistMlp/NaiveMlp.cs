using NumSharp;

namespace NeuralNetwork.NumSharp.MnistMlp
{
    /// <summary>
    /// Baseline 2-layer MLP using ordinary np.* composition — no fused kernel.
    /// Each operation allocates a fresh output NDArray and runs its own
    /// iteration, so a forward pass costs:
    ///
    ///   Layer 1: np.dot + np.add (preact,b1) + np.maximum(...,0)    = 3 ops, 2 intermediates
    ///   Layer 2: np.dot + np.add (preact,b2)                        = 2 ops, 1 intermediate
    ///
    /// Fused version compresses layer 1 into np.dot + ONE NpyIter and layer 2
    /// into np.dot + ONE NpyIter, saving an allocation and an iteration pass
    /// per layer. The fused kernel also keeps (preact + b) in registers
    /// across the Max — no round-trip through DRAM for the intermediate.
    /// </summary>
    public static class NaiveMlp
    {
        public static NDArray Forward(NDArray x, NDArray W1, NDArray b1, NDArray W2, NDArray b2)
        {
            // Layer 1
            NDArray preact1 = np.dot(x, W1);
            NDArray sum1    = np.add(preact1, b1);
            NDArray hidden  = np.maximum(sum1, (NDArray)0f);

            // Layer 2
            NDArray preact2 = np.dot(hidden, W2);
            NDArray logits  = np.add(preact2, b2);

            return logits;
        }
    }
}
