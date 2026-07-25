using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Base class for the layers with predefined variables and functions
    /// </summary>
    public abstract class BaseLayer
    {
        /// <summary>
        /// Name of the layer
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Input for the layer
        /// </summary>
        public NDArray Input { get; set; }

        /// <summary>
        /// Output after forwarding the input across the neurons
        /// </summary>
        public NDArray Output { get; set; }

        /// <summary>
        /// Trainable parameters list, eg, weight, bias
        /// </summary>
        public Dictionary<string, NDArray> Parameters { get; set; }

        /// <summary>
        /// Non-trainable state that belongs to the layer but must NEVER reach an
        /// optimizer — BatchNormalization's running mean/variance being the
        /// motivating case (Keras `non_trainable_variables`).
        ///
        /// Optimizers iterate <see cref="Parameters"/> and demand a matching
        /// <see cref="Grads"/> entry for every key, so anything without a
        /// gradient has to live here instead. Serialization
        /// (<c>Serialization.ModelWeights</c>) walks BOTH dictionaries, so a
        /// checkpoint carries running statistics and reloads a model that
        /// evaluates identically.
        /// </summary>
        public Dictionary<string, NDArray> NonTrainable { get; set; }

        /// <summary>
        /// Gradient of the Input
        /// </summary>
        public NDArray InputGrad { get; set; }

        /// <summary>
        /// List of all parameters gradients calculated during back propagation.
        /// </summary>
        public Dictionary<string, NDArray> Grads { get; set; }

        /// <summary>
        /// Whether the layer is running inside a training step — the Keras
        /// <c>training=</c> argument / PyTorch <c>model.train()</c> vs
        /// <c>model.eval()</c> distinction, as a flag rather than a parameter.
        ///
        /// <para>Default <b>false</b>, so anything that just calls
        /// <c>Forward</c> gets inference behavior. <c>MlpTrainer</c> sets it true
        /// around the training forward pass and false for
        /// <c>Evaluate</c>/<c>EvaluateFull</c>; a hand-rolled loop must do the
        /// same or Dropout will not drop and BatchNorm will normalize with stale
        /// running statistics.</para>
        ///
        /// <para>Read by <see cref="Dropout"/> and
        /// <see cref="BatchNormalization"/>. Every other layer ignores it.
        /// A flag was chosen over changing the <c>Forward(x)</c> signature
        /// because the signature change would break every existing layer,
        /// activation and verification script for no behavioral gain.</para>
        /// </summary>
        public bool Training { get; set; }

        /// <summary>
        /// Optional weight penalties, keyed by the <see cref="Parameters"/> entry
        /// they apply to (Keras <c>kernel_regularizer</c> /
        /// <c>bias_regularizer</c>). Empty by default.
        ///
        /// <para>These are NOT applied by <c>Backward</c>. The trainer calls
        /// <see cref="ApplyRegularizerGradients"/> and
        /// <see cref="RegularizationPenalty"/> after the backward sweep, so a
        /// layer author cannot forget to honour a regularizer someone attached to
        /// their layer.</para>
        /// </summary>
        public Dictionary<string, Regularizers.BaseRegularizer> Regularizers { get; set; }

        /// <summary>
        /// Adds each regularizer's <c>dR/dw</c> into the matching
        /// <see cref="Grads"/> entry. No-op when nothing is attached.
        /// </summary>
        public void ApplyRegularizerGradients()
        {
            if (Regularizers.Count == 0)
                return;

            foreach (var kv in Regularizers)
            {
                if (kv.Value == null || !Parameters.TryGetValue(kv.Key, out NDArray w))
                    continue;
                if (!Grads.TryGetValue(kv.Key, out NDArray g) || g is null)
                    continue;

                Grads[kv.Key] = g + kv.Value.Gradient(w);
            }
        }

        /// <summary>
        /// Total penalty this layer contributes to the reported loss. 0 when
        /// nothing is attached.
        /// </summary>
        public float RegularizationPenalty()
        {
            if (Regularizers.Count == 0)
                return 0f;

            float total = 0f;
            foreach (var kv in Regularizers)
                if (kv.Value != null && Parameters.TryGetValue(kv.Key, out NDArray w))
                    total += kv.Value.Penalty(w);
            return total;
        }

        /// <summary>
        /// Base layer instance
        /// </summary>
        /// <param name="name"></param>
        public BaseLayer(string name)
        {
            Name = name + Util.GetNext();
            Parameters = new Dictionary<string, NDArray>();
            Grads = new Dictionary<string, NDArray>();
            NonTrainable = new Dictionary<string, NDArray>();
            Regularizers = new Dictionary<string, Regularizers.BaseRegularizer>();
        }

        /// <summary>
        /// Serializable description of this layer, Keras
        /// <c>{"class_name": ..., "config": {...}}</c> shape. Layers that can be
        /// rebuilt from a config override this AND register a factory with
        /// <c>Serialization.ModelArchitecture</c>; the default returns a config
        /// carrying only the type name, which round-trips as "unsupported".
        /// </summary>
        public virtual Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig(GetType().Name);

        /// <summary>
        /// Virtual forward method to perform calculation and move the input to next layer
        /// </summary>
        /// <param name="x"></param>
        public virtual void Forward(NDArray x)
        {
            Input = x;
        }

        /// <summary>
        /// Calculate the gradient of the layer. Usually a prtial derivative implemenation of the forward algorithm
        /// </summary>
        /// <param name="grad"></param>
        public virtual void Backward(NDArray grad)
        {
            
        }

        public void PrintParams(bool printGrads = true)
        {
            foreach (var item in Parameters)
            {
                Console.WriteLine(item.Value.ToString());
                if(printGrads && Grads.ContainsKey(item.Key))
                {
                    Console.WriteLine(Grads[item.Key].ToString());
                }
            }
        }
    }
}
