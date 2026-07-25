using NeuralNetwork.NumSharp.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Optimizers
{
    public abstract class BaseOptimizer
    {
        public float Epsilon = 1e-7f;

        private float _clipNorm;
        private float _globalClipNorm;

        /// <summary>
        /// Gets or sets the name of the optimizer function
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the learning rate for the optimizer.
        /// </summary>
        /// <value>
        /// The learning rate.
        /// </value>
        public float LearningRate { get; set; }

        /// <summary>
        /// Parameter that accelerates SGD in the relevant direction and dampens oscillations.
        /// </summary>
        /// <value>
        /// The momentum.
        /// </value>
        public float Momentum { get; set; }

        /// <summary>
        /// Learning rate decay over each update.
        /// </summary>
        /// <value>
        /// The decay rate.
        /// </value>
        public float DecayRate { get; set; }

        // =================================================================
        // Gradient clipping (Keras `clipnorm` / `global_clipnorm` / `clipvalue`,
        // PyTorch `clip_grad_norm_` / `clip_grad_value_`)
        //
        // Keras applies exactly ONE of the three, in this precedence order, and
        // rejects clipnorm+global_clipnorm together. All three default to 0
        // (off). The formulas below are ports of keras.src.optimizers.base_
        // optimizer._clip_gradients and keras.src.ops.clip_by_norm — including
        // their multiply-then-divide shape, which is NOT the same floating-point
        // expression as "scale only when the norm is exceeded".
        // =================================================================

        /// <summary>
        /// Clip each parameter's gradient so its L2 norm is at most this value.
        ///
        /// <para><b>PER-PARAMETER, matching Keras.</b> Each tensor is normalized
        /// against its own norm — a model's weight matrix and its bias vector are
        /// clipped independently. PyTorch's <c>clip_grad_norm_</c> instead uses
        /// one norm over the whole model; that is <see cref="GlobalClipNorm"/>.
        /// The two give different updates whenever more than one parameter
        /// exceeds the threshold, so the distinction is not cosmetic.</para>
        ///
        /// 0 disables. Cannot be combined with <see cref="GlobalClipNorm"/>.
        /// </summary>
        public float ClipNorm
        {
            get => _clipNorm;
            set
            {
                if (value > 0f && _globalClipNorm > 0f)
                    throw new ArgumentException("Only one of ClipNorm and GlobalClipNorm can be set.", nameof(ClipNorm));
                _clipNorm = value;
            }
        }

        /// <summary>
        /// Clip all gradients by ONE norm taken over every parameter of the model
        /// at once (Keras <c>global_clipnorm</c>, PyTorch <c>clip_grad_norm_</c>).
        /// Preserves the gradient's direction in full-model space, which
        /// per-parameter clipping does not.
        ///
        /// <para>Applied by <see cref="ApplyGlobalClipNorm"/>, which the trainer
        /// calls once per step BEFORE the per-layer <see cref="Update"/> calls —
        /// a single <see cref="Update"/> only sees one layer and cannot compute a
        /// model-wide norm.</para>
        ///
        /// 0 disables. Cannot be combined with <see cref="ClipNorm"/>.
        /// </summary>
        public float GlobalClipNorm
        {
            get => _globalClipNorm;
            set
            {
                if (value > 0f && _clipNorm > 0f)
                    throw new ArgumentException("Only one of ClipNorm and GlobalClipNorm can be set.", nameof(GlobalClipNorm));
                _globalClipNorm = value;
            }
        }

        /// <summary>
        /// Clamp every gradient ELEMENT into [-ClipValue, +ClipValue]
        /// (Keras <c>clipvalue</c>). 0 disables. Ignored when a norm clip is set —
        /// Keras's <c>_clip_gradients</c> is an if/elif chain, not a composition.
        /// </summary>
        public float ClipValue { get; set; }

        /// <summary>
        /// Applies per-parameter <see cref="ClipNorm"/> or <see cref="ClipValue"/>
        /// to one gradient tensor. Subclasses call this on every
        /// <c>layer.Grads[key]</c> before using it.
        ///
        /// <para>Returns the input unchanged when no clip applies, so the
        /// no-clipping path allocates nothing.</para>
        /// </summary>
        protected NDArray ClipGradient(NDArray grad)
        {
            if (grad is null)
                return null;

            // Keras precedence: clipnorm, then global_clipnorm (already applied
            // model-wide by ApplyGlobalClipNorm), then clipvalue.
            if (ClipNorm > 0f)
                return ClipByNorm(grad, ClipNorm);

            if (GlobalClipNorm > 0f)
                return grad;

            if (ClipValue > 0f)
                return np.clip(grad, -ClipValue, ClipValue);

            return grad;
        }

        /// <summary>
        /// Rescales every gradient in the model by one shared factor derived from
        /// the model-wide L2 norm. No-op unless <see cref="GlobalClipNorm"/> is
        /// set. Must run after backward and before the <see cref="Update"/> calls.
        /// </summary>
        public void ApplyGlobalClipNorm(IReadOnlyList<BaseLayer> layers)
        {
            if (GlobalClipNorm <= 0f || layers == null)
                return;

            // global_norm = sqrt(sum over all tensors of sum(g^2))
            double sumSquares = 0.0;
            foreach (var layer in layers)
                foreach (var key in layer.Parameters.Keys)
                    if (layer.Grads.TryGetValue(key, out NDArray g) && g is not null)
                        sumSquares += (double)(float)np.sum(g * g);

            float useNorm = (float)Math.Sqrt(sumSquares);

            // keras.src.ops.clip_by_global_norm:
            //   scale = clip_norm * min(1/use_norm, 1/clip_norm)
            // which is EXACTLY 1 while use_norm <= clip_norm (so an under-budget
            // step is bit-for-bit untouched), and clip_norm/use_norm above it.
            float scale = GlobalClipNorm * Math.Min(1f / useNorm, 1f / GlobalClipNorm);

            // Keras adds (use_norm - use_norm) so a non-finite norm poisons the
            // scale instead of silently producing a finite update from inf/NaN
            // gradients.
            scale += useNorm - useNorm;

            if (scale == 1f)
                return;

            foreach (var layer in layers)
                foreach (var key in layer.Parameters.Keys.ToList())
                    if (layer.Grads.TryGetValue(key, out NDArray g) && g is not null)
                        layer.Grads[key] = g * scale;
        }

        /// <summary>
        /// Port of <c>keras.src.ops.clip_by_norm</c> with <c>axes=None</c>.
        ///
        /// <para>The shape of the expression is load-bearing:
        /// <c>values * clip_norm / max(l2norm, clip_norm)</c> multiplies FIRST and
        /// divides by the larger of the two, rather than branching on
        /// <c>norm &gt; clip</c>. Under the threshold this is
        /// <c>v * c / c</c> — algebraically the identity, and exactly the
        /// identity in IEEE arithmetic for every finite v and c since the
        /// multiply and divide use the same c. The zero-norm guard mirrors
        /// Keras's <c>where(l2sum &gt; 0, ...)</c>, which exists so
        /// <c>sqrt</c> never sees a value it would differentiate at 0.</para>
        /// </summary>
        protected static NDArray ClipByNorm(NDArray values, float clipNorm)
        {
            float l2sum = (float)np.sum(values * values);
            float l2norm = l2sum > 0f ? (float)Math.Sqrt(l2sum) : l2sum;
            return values * clipNorm / Math.Max(l2norm, clipNorm);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseOptimizer"/> class.
        /// </summary>
        /// <param name="lr">The lr.</param>
        /// <param name="name">The name.</param>
        public BaseOptimizer(float lr, string name)
        {
            LearningRate = lr;
            Name = name;
        }

        /// <summary>
        /// Updates the specified iteration.
        /// </summary>
        /// <param name="iteration">The iteration.</param>
        /// <param name="layer">The layer.</param>
        public abstract void Update(int iteration, BaseLayer layer);

        /// <summary>
        /// Gets the specified optimizer type.
        /// </summary>
        /// <param name="optimizerType">Type of the optimizer.</param>
        /// <returns></returns>
        public static BaseOptimizer Get(string name)
        {
            BaseOptimizer opt = null;
            switch (name)
            {
                case "sgd":
                    opt = new SGD();
                    break;
                case "adam":
                    opt = new Adam();
                    break;
                default:
                    break;
            }

            return opt;
        }
    }
}
