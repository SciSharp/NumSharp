using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Layer normalization over the feature axis of a 2-D
    /// <c>(batch, features)</c> input — <c>keras.layers.LayerNormalization</c>
    /// with the default <c>axis=-1</c>.
    ///
    /// <code>
    ///   xhat = (x - mean_features) / sqrt(var_features + eps);  y = gamma*xhat + beta
    /// </code>
    ///
    /// <para>Statistics are per SAMPLE, over the features — the transpose of what
    /// <see cref="BatchNormalization"/> does. Three consequences follow and are
    /// the reason transformers use it: there are no running statistics, the
    /// layer behaves identically in training and inference (so it ignores
    /// <see cref="BaseLayer.Training"/>), and a sample's normalization is
    /// independent of the other samples in the batch — batch size 1 works.</para>
    ///
    /// <para><c>epsilon</c> defaults to Keras's <b>0.001</b> (probed against
    /// 3.15), not the 1e-5/1e-6 used by PyTorch and most transformer papers.
    /// Pass it explicitly when porting weights from elsewhere.</para>
    /// </summary>
    public class LayerNormalization : BaseLayer
    {
        public int Features { get; }
        public float Epsilon { get; }

        /// <summary>Learn <c>gamma</c> (Keras <c>scale</c>).</summary>
        public bool Scale { get; }

        /// <summary>Learn <c>beta</c> (Keras <c>center</c>).</summary>
        public bool Center { get; }

        private NDArray _xhat;     // (batch, features)
        private NDArray _invStd;   // (batch, 1) — one per SAMPLE, unlike BatchNorm

        public LayerNormalization(int features, float epsilon = 1e-3f, bool scale = true, bool center = true)
            : base("layernorm")
        {
            if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features));
            if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));

            Features = features;
            Epsilon = epsilon;
            Scale = scale;
            Center = center;

            if (Scale)
                Parameters["gamma"] = np.ones(new Shape(features), NPTypeCode.Single);
            if (Center)
                Parameters["beta"] = np.zeros(new Shape(features), NPTypeCode.Single);
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            RequireShape(x);

            NDArray mean = np.mean(x, axis: 1, keepdims: true);      // (N, 1)
            NDArray variance = np.var(x, axis: 1, keepdims: true);   // (N, 1), population

            _invStd = 1f / np.sqrt(variance + Epsilon);
            _xhat = (x - mean) * _invStd;

            NDArray y = _xhat;
            if (Scale) y = y * Parameters["gamma"];
            if (Center) y = y + Parameters["beta"];
            Output = y;
        }

        /// <summary>
        /// Same folded chain as <see cref="BatchNormalization"/>, but the means
        /// reduce over the FEATURE axis:
        /// <code>
        ///   dxhat = grad * gamma
        ///   dx    = invStd * ( dxhat - mean_D(dxhat) - xhat * mean_D(dxhat*xhat) )
        /// </code>
        /// <para>The parameter gradients still reduce over the BATCH, because
        /// gamma and beta are shared across samples — mixing those two axes up is
        /// the classic LayerNorm bug, and it produces gradients of the right
        /// SHAPE, so only a numeric check catches it.</para>
        /// </summary>
        public override void Backward(NDArray grad)
        {
            if (Scale)
                Grads["gamma"] = np.sum(grad * _xhat, axis: 0);
            if (Center)
                Grads["beta"] = np.sum(grad, axis: 0);

            NDArray dxhat = Scale ? grad * Parameters["gamma"] : grad;

            NDArray meanDxhat = np.mean(dxhat, axis: 1, keepdims: true);              // (N, 1)
            NDArray meanDxhatXhat = np.mean(dxhat * _xhat, axis: 1, keepdims: true);  // (N, 1)
            InputGrad = _invStd * (dxhat - meanDxhat - _xhat * meanDxhatXhat);
        }

        private void RequireShape(NDArray x)
        {
            if (x.ndim != 2)
                throw new NotSupportedException(
                    $"LayerNormalization currently supports 2-D (batch, features) inputs; got {x.ndim}-D.");
            if (x.shape[1] != Features)
                throw new ArgumentException(
                    $"LayerNormalization was built for {Features} features but received {x.shape[1]}.", nameof(x));
        }

        public override Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig("LayerNormalization")
                .Set("features", Features)
                .Set("epsilon", Epsilon)
                .Set("scale", Scale)
                .Set("center", Center);
    }
}
