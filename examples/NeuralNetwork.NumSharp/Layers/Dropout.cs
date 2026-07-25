using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Inverted dropout — <c>keras.layers.Dropout</c>.
    ///
    /// <para>At training time each element is zeroed with probability
    /// <see cref="Rate"/> and the survivors are scaled by
    /// <c>1/(1-rate)</c>, so the layer's expected output equals its input and
    /// inference needs no compensation at all. (Probed against Keras 3.15:
    /// <c>Dropout(0.5)</c> on a tensor of ones emits exactly {0, 2} in training
    /// and passes through unchanged in inference.) The alternative — scale at
    /// test time — is what "inverted" is inverted relative to; nobody ships it
    /// any more because it makes inference depend on a training hyper-parameter.</para>
    ///
    /// <para>Reads <see cref="BaseLayer.Training"/>. Outside a training step this
    /// layer is the identity, which is also what a <c>rate</c> of 0 gives at any
    /// time.</para>
    ///
    /// <para>The mask is drawn from <c>np.random.bernoulli</c> (MT19937), so a
    /// seeded run reproduces exactly. The SCALED mask is cached, because backward
    /// must apply the identical scaling to the gradient — a run that re-drew or
    /// re-scaled here would silently train against a different network than the
    /// one it evaluated.</para>
    /// </summary>
    public class Dropout : BaseLayer
    {
        /// <summary>Fraction of elements to zero. In [0, 1).</summary>
        public float Rate { get; }

        /// <summary>The scaled mask from the last training forward pass, or null.</summary>
        private NDArray _mask;

        public Dropout(float rate) : base("dropout")
        {
            if (!(rate >= 0f) || rate >= 1f)
                throw new ArgumentOutOfRangeException(nameof(rate),
                    $"Dropout rate must be in [0, 1), got {rate}. A rate of 1 would zero every activation.");
            Rate = rate;
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            if (!Training || Rate == 0f)
            {
                _mask = null;
                Output = x;
                return;
            }

            // bernoulli(p) yields 1 with probability p, so the KEEP probability
            // is 1-rate. Returns float64; the framework is float32 throughout.
            float keep = 1f - Rate;
            NDArray mask = np.random.bernoulli(keep, x.Shape).astype(NPTypeCode.Single);

            // Fold the 1/(1-rate) compensation into the mask itself: one array to
            // cache, and backward is then a plain multiply by the same thing.
            _mask = mask * (1f / keep);
            Output = x * _mask;
        }

        public override void Backward(NDArray grad)
        {
            // Dropout has no parameters; the gradient passes through the same
            // mask (and the same scale) the forward pass used.
            InputGrad = _mask is null ? grad : grad * _mask;
        }

        public override Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig("Dropout").Set("rate", Rate);
    }
}
