using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Batch normalization over the feature axis of a 2-D <c>(batch, features)</c>
    /// input — <c>keras.layers.BatchNormalization</c> with the default
    /// <c>axis=-1</c>.
    ///
    /// <code>
    ///   training : xhat = (x - mean_batch) / sqrt(var_batch + eps);  y = gamma*xhat + beta
    ///   inference: xhat = (x - moving_mean) / sqrt(moving_var + eps); y = gamma*xhat + beta
    /// </code>
    ///
    /// <para><b>Defaults are Keras's, probed rather than assumed</b> (Keras 3.15):
    /// <c>momentum = 0.99</c>, <c>epsilon = 0.001</c> — note that is 1e-3, not the
    /// 1e-5 most other frameworks use — and the variance is the <b>population</b>
    /// (biased, ddof=0) one, for BOTH the normalization and the running-variance
    /// update. Using the sample variance instead changes the output by ~20% on a
    /// batch of 3 and is invisible on a batch of 256, which is exactly the kind of
    /// bug that survives casual testing.</para>
    ///
    /// <para>Running statistics update as
    /// <c>moving = momentum·moving + (1-momentum)·batch</c> and live in
    /// <see cref="BaseLayer.NonTrainable"/>, NOT in <c>Parameters</c> — an
    /// optimizer iterates <c>Parameters</c> and demands a <c>Grads</c> entry for
    /// every key, so a running mean parked there would either crash the step or,
    /// worse, get "optimized".</para>
    ///
    /// <para>Only the 2-D case is implemented; a 4-D convolutional input needs
    /// reduction over (N,H,W) and arrives with the P5 conv stack.</para>
    /// </summary>
    public class BatchNormalization : BaseLayer
    {
        public int Features { get; }
        public float Momentum { get; }
        public float Epsilon { get; }

        /// <summary>Learn <c>gamma</c> (Keras <c>scale</c>).</summary>
        public bool Scale { get; }

        /// <summary>Learn <c>beta</c> (Keras <c>center</c>).</summary>
        public bool Center { get; }

        // Cached from Forward for the backward pass.
        private NDArray _xhat;      // (batch, features)
        private NDArray _invStd;    // (1, features) — 1/sqrt(var + eps)
        private bool _trainedForward;

        public BatchNormalization(int features, float momentum = 0.99f, float epsilon = 1e-3f,
                                  bool scale = true, bool center = true)
            : base("batchnorm")
        {
            if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features));
            if (momentum < 0f || momentum > 1f) throw new ArgumentOutOfRangeException(nameof(momentum));
            if (epsilon <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon));

            Features = features;
            Momentum = momentum;
            Epsilon = epsilon;
            Scale = scale;
            Center = center;

            if (Scale)
                Parameters["gamma"] = np.ones(new Shape(features), NPTypeCode.Single);
            if (Center)
                Parameters["beta"] = np.zeros(new Shape(features), NPTypeCode.Single);

            NonTrainable["moving_mean"] = np.zeros(new Shape(features), NPTypeCode.Single);
            NonTrainable["moving_variance"] = np.ones(new Shape(features), NPTypeCode.Single);
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            RequireShape(x);

            NDArray mean, variance;
            if (Training)
            {
                // Population statistics of THIS batch (np.var is ddof=0).
                mean = np.mean(x, axis: 0, keepdims: true);       // (1, F)
                variance = np.var(x, axis: 0, keepdims: true);    // (1, F)

                // moving = momentum*moving + (1-momentum)*batch, in float32.
                NDArray flatMean = np.reshape(mean, new Shape(Features));
                NDArray flatVar = np.reshape(variance, new Shape(Features));
                NonTrainable["moving_mean"] = Momentum * NonTrainable["moving_mean"] + (1f - Momentum) * flatMean;
                NonTrainable["moving_variance"] = Momentum * NonTrainable["moving_variance"] + (1f - Momentum) * flatVar;
            }
            else
            {
                mean = np.reshape(NonTrainable["moving_mean"], new Shape(1, Features));
                variance = np.reshape(NonTrainable["moving_variance"], new Shape(1, Features));
            }

            _invStd = 1f / np.sqrt(variance + Epsilon);   // (1, F)
            _xhat = (x - mean) * _invStd;
            _trainedForward = Training;

            NDArray y = _xhat;
            if (Scale) y = y * Parameters["gamma"];
            if (Center) y = y + Parameters["beta"];
            Output = y;
        }

        /// <summary>
        /// <para>The training-mode input gradient is the standard chain through
        /// mean AND variance, written in its folded form:</para>
        /// <code>
        ///   dxhat = grad * gamma
        ///   dx    = invStd * ( dxhat - mean_N(dxhat) - xhat * mean_N(dxhat*xhat) )
        /// </code>
        /// <para>which is algebraically identical to the textbook
        /// <c>dvar</c>/<c>dmean</c> chain
        /// (<c>dvar = Σ dxhat·(x-μ)·(-½)(σ²+ε)^(-3/2)</c>,
        /// <c>dmean = Σ dxhat·(-1/σ) + dvar·Σ(-2(x-μ))/N</c>,
        /// <c>dx = dxhat/σ + dvar·2(x-μ)/N + dmean/N</c>) once
        /// <c>Σ(x-μ) = 0</c> is substituted into <c>dmean</c>'s second term. It
        /// stays exact with a non-zero epsilon, since epsilon only ever enters
        /// through <c>σ = sqrt(var+ε)</c>. Four temporaries instead of nine.</para>
        ///
        /// <para>In inference mode the statistics are constants, so the batch
        /// coupling disappears and <c>dx = grad·gamma·invStd</c>.</para>
        /// </summary>
        public override void Backward(NDArray grad)
        {
            int n = (int)grad.shape[0];

            if (Scale)
                Grads["gamma"] = np.sum(grad * _xhat, axis: 0);
            if (Center)
                Grads["beta"] = np.sum(grad, axis: 0);

            NDArray dxhat = Scale ? grad * Parameters["gamma"] : grad;

            if (!_trainedForward)
            {
                InputGrad = dxhat * _invStd;
                return;
            }

            NDArray meanDxhat = np.mean(dxhat, axis: 0, keepdims: true);              // (1, F)
            NDArray meanDxhatXhat = np.mean(dxhat * _xhat, axis: 0, keepdims: true);  // (1, F)
            InputGrad = _invStd * (dxhat - meanDxhat - _xhat * meanDxhatXhat);
        }

        private void RequireShape(NDArray x)
        {
            if (x.ndim != 2)
                throw new NotSupportedException(
                    $"BatchNormalization currently supports 2-D (batch, features) inputs; got {x.ndim}-D. " +
                    "Convolutional (N,C,H,W) normalization arrives with the P5 conv stack.");
            if (x.shape[1] != Features)
                throw new ArgumentException(
                    $"BatchNormalization was built for {Features} features but received {x.shape[1]}.", nameof(x));
        }

        public override Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig("BatchNormalization")
                .Set("features", Features)
                .Set("momentum", Momentum)
                .Set("epsilon", Epsilon)
                .Set("scale", Scale)
                .Set("center", Center);
    }
}
