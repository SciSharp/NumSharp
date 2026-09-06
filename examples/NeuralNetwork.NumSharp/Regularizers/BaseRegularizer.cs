using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Regularizers
{
    /// <summary>
    /// Weight penalty attached to a layer parameter — the
    /// <c>keras.regularizers</c> family.
    ///
    /// <para>A regularizer contributes two things: a scalar <see cref="Penalty"/>
    /// added to the reported loss, and a <see cref="Gradient"/> added to the
    /// parameter's gradient before the optimizer step. Keras gets the second for
    /// free from autograd; with hand-written backward passes it has to be stated,
    /// and the two must be consistent — <see cref="Gradient"/> is exactly
    /// <c>d(Penalty)/dw</c>.</para>
    ///
    /// <para><b>Keras's scaling has no ½.</b> <c>L2(l2)</c> is
    /// <c>l2·Σw²</c>, not <c>½·l2·Σw²</c>, so the gradient is <c>2·l2·w</c> and
    /// not <c>l2·w</c>. (Probed against Keras 3.15: <c>L2(0.1)</c> on
    /// <c>[[1,-2],[3,-4]]</c> returns 3.0 = 0.1·30.) Papers and PyTorch's
    /// <c>weight_decay</c> often use the ½ convention, which makes the same
    /// nominal coefficient twice as strong here.</para>
    /// </summary>
    public abstract class BaseRegularizer
    {
        protected BaseRegularizer(string name) => Name = name;

        public string Name { get; }

        /// <summary>Scalar penalty added to the loss.</summary>
        public abstract float Penalty(NDArray w);

        /// <summary>
        /// <c>d(Penalty)/dw</c>, shaped like <paramref name="w"/>, added to the
        /// parameter's gradient.
        /// </summary>
        public abstract NDArray Gradient(NDArray w);

        /// <summary>
        /// Resolves "l1" / "l2" / "l1l2" to a default-strength instance
        /// (Keras's 0.01), mirroring the other <c>Get</c> resolvers in this
        /// project: "" / null returns null, an unknown name throws.
        /// </summary>
        public static BaseRegularizer Get(string name)
        {
            switch (name?.Trim().ToLowerInvariant())
            {
                case null:
                case "":
                case "none":
                    return null;
                case "l1":
                    return new L1();
                case "l2":
                    return new L2();
                case "l1l2":
                case "l1_l2":
                    return new L1L2();
                default:
                    throw new ArgumentException(
                        $"Unknown regularizer '{name}'. Supported: l1, l2, l1l2 (or '' for none).", nameof(name));
            }
        }
    }

    /// <summary>Lasso: <c>l1·Σ|w|</c>; gradient <c>l1·sign(w)</c>.</summary>
    public class L1 : BaseRegularizer
    {
        public float Strength { get; }

        public L1(float l1 = 0.01f) : base("l1")
        {
            if (l1 < 0f) throw new ArgumentOutOfRangeException(nameof(l1));
            Strength = l1;
        }

        public override float Penalty(NDArray w) => Strength * (float)np.sum(np.abs(w));

        public override NDArray Gradient(NDArray w) => Strength * SubGradient(w);

        /// <summary>
        /// <c>d|w|/dw</c> with <b>Keras/JAX's subgradient choice at the kink</b>:
        /// the branch is <c>w &gt;= 0</c>, so the derivative at exactly 0 — and at
        /// -0.0 — is <b>+1</b>, not 0.
        ///
        /// <para>This is NOT <c>np.sign</c>, which returns 0 there
        /// (<c>jnp.sign(0.0)</c> is 0 too). Keras's autodiff never routes
        /// <c>abs</c> through <c>sign</c>; <c>jax.grad(jnp.abs)</c> at ±0.0
        /// returns 1.0, which the oracle caught. Same convention as
        /// <see cref="Activations.LeakyReLU"/>'s gradient at 0, and the same
        /// class of bug — a zero weight is exactly where L1 spends its time, so
        /// getting it wrong stalls every weight the penalty has already driven
        /// to zero.</para>
        /// </summary>
        internal static NDArray SubGradient(NDArray w)
            => np.where(w >= (NDArray)0f, (NDArray)1f, (NDArray)(-1f)).astype(NPTypeCode.Single);
    }

    /// <summary>Ridge: <c>l2·Σw²</c>; gradient <c>2·l2·w</c> (note the 2).</summary>
    public class L2 : BaseRegularizer
    {
        public float Strength { get; }

        public L2(float l2 = 0.01f) : base("l2")
        {
            if (l2 < 0f) throw new ArgumentOutOfRangeException(nameof(l2));
            Strength = l2;
        }

        public override float Penalty(NDArray w) => Strength * (float)np.sum(w * w);

        public override NDArray Gradient(NDArray w) => (2f * Strength) * w;
    }

    /// <summary>Elastic net: <c>l1·Σ|w| + l2·Σw²</c>.</summary>
    public class L1L2 : BaseRegularizer
    {
        public float L1Strength { get; }
        public float L2Strength { get; }

        public L1L2(float l1 = 0.01f, float l2 = 0.01f) : base("l1l2")
        {
            if (l1 < 0f) throw new ArgumentOutOfRangeException(nameof(l1));
            if (l2 < 0f) throw new ArgumentOutOfRangeException(nameof(l2));
            L1Strength = l1;
            L2Strength = l2;
        }

        public override float Penalty(NDArray w)
            => L1Strength * (float)np.sum(np.abs(w)) + L2Strength * (float)np.sum(w * w);

        public override NDArray Gradient(NDArray w)
            => L1Strength * L1.SubGradient(w) + (2f * L2Strength) * w;
    }
}
