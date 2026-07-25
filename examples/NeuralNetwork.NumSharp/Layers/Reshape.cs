using System;
using System.Linq;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Collapses every axis after the batch into one —
    /// <c>keras.layers.Flatten</c>: <c>(N, d1, d2, ...) -&gt; (N, d1*d2*...)</c>.
    ///
    /// <para>Backward restores the input's shape. Both directions are pure
    /// <c>np.reshape</c>, which is metadata-only for a contiguous array — no
    /// element ever moves, and there are no parameters.</para>
    /// </summary>
    public class Flatten : BaseLayer
    {
        public Flatten() : base("flatten") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            int batch = x.ndim == 0 ? 1 : (int)x.shape[0];
            int rest = batch == 0 ? 0 : (int)(x.size / Math.Max(batch, 1));
            Output = np.reshape(x, new Shape(batch, rest));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = np.reshape(grad, Input.Shape);
        }

        public override Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig("Flatten");
    }

    /// <summary>
    /// Reshapes each sample to a fixed target — <c>keras.layers.Reshape</c>.
    ///
    /// <para><b>The target shape excludes the batch dimension</b>, as in Keras:
    /// <c>new Reshape(3, 4)</c> turns <c>(N, 12)</c> into <c>(N, 3, 4)</c>. The
    /// batch size is read from the input at every call, so one instance handles a
    /// partial final batch without reconfiguration.</para>
    /// </summary>
    public class Reshape : BaseLayer
    {
        /// <summary>Per-sample target shape, batch dimension excluded.</summary>
        public int[] TargetShape { get; }

        public Reshape(params int[] targetShape) : base("reshape")
        {
            if (targetShape == null || targetShape.Length == 0)
                throw new ArgumentException("Reshape needs at least one target dimension (batch excluded).", nameof(targetShape));
            if (targetShape.Any(d => d <= 0))
                throw new ArgumentException(
                    $"Reshape target dimensions must be positive; got ({string.Join(", ", targetShape)}). " +
                    "Inferred (-1) dimensions are not supported — state the shape.", nameof(targetShape));

            TargetShape = (int[])targetShape.Clone();
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            int batch = (int)x.shape[0];
            long perSample = TargetShape.Aggregate(1L, (a, d) => a * d);
            long incoming = batch == 0 ? 0 : x.size / batch;
            if (batch > 0 && incoming != perSample)
                throw new ArgumentException(
                    $"Reshape cannot map {incoming} values per sample onto ({string.Join(", ", TargetShape)}) " +
                    $"= {perSample} values.", nameof(x));

            var dims = new int[TargetShape.Length + 1];
            dims[0] = batch;
            Array.Copy(TargetShape, 0, dims, 1, TargetShape.Length);
            Output = np.reshape(x, new Shape(dims));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = np.reshape(grad, Input.Shape);
        }

        public override Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig("Reshape").Set("target_shape", TargetShape);
    }
}
