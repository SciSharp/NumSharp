using System;
using NeuralNetwork.NumSharp.Initializers;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Lookup table mapping integer indices to dense vectors —
    /// <c>keras.layers.Embedding</c>.
    ///
    /// <code>
    ///   input  (N,)   int indices  ->  output (N, D)
    ///   input  (N, T) int indices  ->  output (N, T, D)
    /// </code>
    ///
    /// <para>Weights default to Keras's <c>'uniform'</c>, i.e.
    /// <c>RandomUniform(-0.05, 0.05)</c> (probed against 3.15 — the string
    /// resolves to <c>RandomUniform</c>, not to Glorot).</para>
    ///
    /// <para><b>Backward is a scatter-ADD, and the ADD is the whole point.</b>
    /// A row appearing twice in a batch receives gradient from both occurrences;
    /// an assignment loop would keep only the last and silently under-train every
    /// repeated token — which in a language model is every common word. NumSharp
    /// core has no <c>np.add.at</c> (ROADMAP core backlog), so the accumulation
    /// runs as an explicit unsafe loop over the gradient buffer. That is also
    /// why the gradient is materialized DENSE at <c>(inputDim, D)</c>: real
    /// frameworks keep a sparse gradient for large vocabularies, which needs
    /// optimizer support this project does not have.</para>
    ///
    /// <para><see cref="BaseLayer.InputGrad"/> stays <b>null</b>: the input is a
    /// set of integer indices and there is nothing to differentiate with respect
    /// to. An Embedding is therefore always the first layer of a stack; the
    /// trainer's backward sweep tolerates the null because it only feeds it to a
    /// PREVIOUS layer, of which there is none.</para>
    /// </summary>
    public class Embedding : BaseLayer
    {
        /// <summary>Vocabulary size — valid indices are [0, InputDim).</summary>
        public int InputDim { get; }

        /// <summary>Embedding width.</summary>
        public int OutputDim { get; }

        /// <summary>Flattened copy of the indices the last Forward saw.</summary>
        private int[] _flatIndices;

        public Embedding(int inputDim, int outputDim, BaseInitializer embeddingsInitializer = null)
            : base("embedding")
        {
            if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
            if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

            InputDim = inputDim;
            OutputDim = outputDim;

            // Keras's default embeddings_initializer is the string 'uniform',
            // which resolves to RandomUniform(-0.05, 0.05).
            Parameters["w"] = (embeddingsInitializer ?? new RandomUniform(-0.05f, 0.05f))
                .Initialize(new Shape(inputDim, outputDim));
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            _flatIndices = ReadIndices(x);
            NDArray w = Parameters["w"];

            // Gather rows. A raw index array as the SOLE index is FANCY indexing
            // in NumSharp, so this selects rows and copies — which is what an
            // embedding lookup is.
            NDArray gathered = np.take(w, np.array(_flatIndices), axis: 0);   // (count, D)

            Output = x.ndim == 1
                ? gathered
                : np.reshape(gathered, BuildOutputShape(x));
        }

        public override void Backward(NDArray grad)
        {
            NDArray w = Parameters["w"];

            // Dense zero gradient, then accumulate each occurrence into its row.
            NDArray dw = np.zeros(w.Shape, NPTypeCode.Single);

            // Flatten the incoming gradient to (count, D) so the loop indexes it
            // the same way regardless of whether the input was (N,) or (N,T).
            NDArray flatGrad = grad.ndim == 2
                ? grad
                : np.reshape(grad, new Shape(_flatIndices.Length, OutputDim));

            if (flatGrad.typecode != NPTypeCode.Single)
                flatGrad = flatGrad.astype(NPTypeCode.Single);
            if (!flatGrad.Shape.IsContiguous)
                flatGrad = flatGrad.copy();

            unsafe
            {
                float* dst = (float*)dw.Unsafe.Address;
                float* src = (float*)flatGrad.Unsafe.Address;

                for (int i = 0; i < _flatIndices.Length; i++)
                {
                    // += , never = : duplicate indices MUST accumulate.
                    float* row = dst + (long)_flatIndices[i] * OutputDim;
                    float* g = src + (long)i * OutputDim;
                    for (int d = 0; d < OutputDim; d++)
                        row[d] += g[d];
                }
            }

            Grads["w"] = dw;

            // Indices are not differentiable — see the class remarks.
            InputGrad = null;
        }

        /// <summary>
        /// Reads the index tensor into a flat int[], validating the range. Accepts
        /// the integer dtypes this project's loaders, np.argmax and hand-built
        /// arrays actually produce.
        /// </summary>
        private int[] ReadIndices(NDArray x)
        {
            if (x.ndim < 1 || x.ndim > 2)
                throw new NotSupportedException(
                    $"Embedding expects (batch,) or (batch, timesteps) indices; got {x.ndim}-D.");

            int count = (int)x.size;
            var flat = np.reshape(x, new Shape(count));
            var indices = new int[count];

            for (int i = 0; i < count; i++)
            {
                int idx;
                switch (flat.typecode)
                {
                    case NPTypeCode.Byte: idx = flat.GetByte(i); break;
                    case NPTypeCode.SByte: idx = flat.GetSByte(i); break;
                    case NPTypeCode.Int16: idx = flat.GetInt16(i); break;
                    case NPTypeCode.UInt16: idx = flat.GetUInt16(i); break;
                    case NPTypeCode.Int32: idx = flat.GetInt32(i); break;
                    case NPTypeCode.UInt32: idx = (int)flat.GetUInt32(i); break;
                    case NPTypeCode.Int64: idx = (int)flat.GetInt64(i); break;
                    case NPTypeCode.UInt64: idx = (int)flat.GetUInt64(i); break;
                    default:
                        throw new NotSupportedException(
                            $"Embedding indices must be an integer dtype; got {flat.typecode}.");
                }

                if ((uint)idx >= (uint)InputDim)
                    throw new ArgumentOutOfRangeException(nameof(x),
                        $"index at position {i} = {idx} is outside [0, {InputDim}).");

                indices[i] = idx;
            }

            return indices;
        }

        private Shape BuildOutputShape(NDArray x)
        {
            var dims = new int[x.ndim + 1];
            for (int i = 0; i < x.ndim; i++)
                dims[i] = (int)x.shape[i];
            dims[x.ndim] = OutputDim;
            return new Shape(dims);
        }

        public override Serialization.LayerConfig GetConfig()
            => new Serialization.LayerConfig("Embedding")
                .Set("input_dim", InputDim)
                .Set("output_dim", OutputDim);
    }
}
