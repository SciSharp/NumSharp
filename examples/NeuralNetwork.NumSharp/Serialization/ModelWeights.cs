using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.NumSharp.Layers;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Serialization
{
    /// <summary>
    /// Model weight persistence — Keras <c>save_weights</c> / PyTorch
    /// <c>state_dict</c> analog, backed by NumSharp core's byte-exact <c>.npz</c>
    /// writer. A checkpoint written here is a genuine NumPy archive: real
    /// <c>numpy.load()</c> opens it and every entry comes back as an ndarray of
    /// the same dtype and shape, so checkpoints are inspectable and portable
    /// without a NumSharp dependency.
    ///
    /// <para><b>Keys are positional, not by layer Name.</b> <see cref="BaseLayer.Name"/>
    /// is assigned from <see cref="Util.GetNext"/>, a process-global counter that
    /// never resets — construct the same architecture twice in one process and
    /// the second copy is <c>fc_fused2</c>/<c>fc_fused3</c>, not
    /// <c>fc_fused0</c>/<c>fc_fused1</c>. Name-keyed checkpoints would therefore
    /// fail to load into a freshly-built model, which is the whole point of a
    /// checkpoint. Entries are keyed by layer INDEX instead:</para>
    ///
    /// <code>
    ///   layer0/param/w      trainable  (optimizer-updated)
    ///   layer0/param/b
    ///   layer1/state/running_mean    non-trainable (BatchNorm running stats)
    /// </code>
    ///
    /// <para>Loading is positional too: entry <c>i</c> goes into
    /// <c>layers[i]</c>, so the caller must rebuild the SAME architecture in the
    /// SAME order. Every mismatch (missing key, wrong shape, wrong layer count)
    /// is a hard error naming the offending slot rather than a silent partial
    /// load.</para>
    /// </summary>
    public static class ModelWeights
    {
        /// <summary>Key segment for optimizer-updated tensors.</summary>
        private const string ParamPrefix = "param";

        /// <summary>Key segment for <see cref="BaseLayer.NonTrainable"/> tensors.</summary>
        private const string StatePrefix = "state";

        // =================================================================
        // File round-trip
        // =================================================================

        /// <summary>
        /// Writes every layer's trainable parameters and non-trainable state to
        /// a single <c>.npz</c> archive.
        /// </summary>
        /// <param name="layers">The model, in forward order.</param>
        /// <param name="path">Destination file. Overwritten if it exists.</param>
        /// <param name="compressed">
        /// Use deflate (<c>np.savez_compressed</c>) instead of stored entries.
        /// Weights are dense float32 and compress poorly — the default is off.
        /// </param>
        public static void Save(IReadOnlyList<BaseLayer> layers, string path, bool compressed = false)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("path is required", nameof(path));

            var entries = ToDictionary(layers);
            if (compressed)
                np.savez_compressed(path, entries);
            else
                np.savez(path, entries);
        }

        /// <summary>
        /// Loads an archive written by <see cref="Save"/> into an
        /// already-constructed model, positionally.
        /// </summary>
        /// <param name="layers">
        /// The rebuilt model. Must have the same layer count, the same parameter
        /// keys and the same shapes as the model that was saved.
        /// </param>
        /// <param name="path">Archive to read.</param>
        public static void Load(IReadOnlyList<BaseLayer> layers, string path)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("path is required", nameof(path));

            // NpzFile is lazy and holds the archive open — it MUST be disposed.
            using (var npz = np.load_npz(path))
            {
                var available = new HashSet<string>(npz.Files);
                var loaded = new Dictionary<string, NDArray>();

                for (int i = 0; i < layers.Count; i++)
                {
                    foreach (var key in TensorKeys(layers[i], i))
                    {
                        if (!available.Contains(key))
                            throw new InvalidOperationException(
                                $"Checkpoint '{path}' has no entry '{key}'. It holds " +
                                $"[{string.Join(", ", npz.Files)}] — the archive was written from a " +
                                "different architecture, or the layers are in a different order.");

                        loaded[key] = npz[key];
                    }
                }

                // Validate EVERY tensor before writing ANY of them. Interleaving
                // the two would leave the model half overwritten when the
                // mismatch is in a later layer — layer 0 from the checkpoint,
                // layer 1 from the initializer — which trains to garbage without
                // ever raising anything.
                Validate(layers, loaded, path);

                for (int i = 0; i < layers.Count; i++)
                    Assign(layers[i], i, loaded, path);
            }
        }

        // =================================================================
        // In-memory snapshots (EarlyStopping restore_best_weights,
        // ModelCheckpoint's "keep the best in RAM" mode)
        // =================================================================

        /// <summary>
        /// Deep-copies the model's tensors into a detached snapshot. The copies
        /// are independent of the live arrays — subsequent optimizer steps do
        /// not disturb them, and the snapshot can be restored more than once.
        /// </summary>
        public static Dictionary<string, NDArray> Capture(IReadOnlyList<BaseLayer> layers)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));

            var snapshot = new Dictionary<string, NDArray>();
            for (int i = 0; i < layers.Count; i++)
            {
                foreach (var kv in layers[i].Parameters)
                    snapshot[Key(i, ParamPrefix, kv.Key)] = kv.Value.copy();
                foreach (var kv in layers[i].NonTrainable)
                    snapshot[Key(i, StatePrefix, kv.Key)] = kv.Value.copy();
            }
            return snapshot;
        }

        /// <summary>
        /// Writes a <see cref="Capture"/> snapshot back into the model. Each
        /// tensor is copied again on the way in, so the snapshot stays reusable.
        /// </summary>
        public static void Restore(IReadOnlyList<BaseLayer> layers, IReadOnlyDictionary<string, NDArray> snapshot)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));
            if (snapshot == null) throw new ArgumentNullException(nameof(snapshot));

            Validate(layers, snapshot, "<snapshot>");
            for (int i = 0; i < layers.Count; i++)
                Assign(layers[i], i, snapshot, "<snapshot>");
        }

        // =================================================================
        // Dictionary view (what actually goes into the archive)
        // =================================================================

        /// <summary>
        /// The exact <c>{key → tensor}</c> map <see cref="Save"/> writes. Exposed
        /// so callers can push a checkpoint through a stream or a byte[] using
        /// core's other <c>np.savez</c> overloads.
        /// </summary>
        public static Dictionary<string, NDArray> ToDictionary(IReadOnlyList<BaseLayer> layers)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));

            var entries = new Dictionary<string, NDArray>();
            for (int i = 0; i < layers.Count; i++)
            {
                foreach (var kv in layers[i].Parameters)
                    entries[Key(i, ParamPrefix, kv.Key)] = kv.Value;
                foreach (var kv in layers[i].NonTrainable)
                    entries[Key(i, StatePrefix, kv.Key)] = kv.Value;
            }
            return entries;
        }

        // =================================================================
        // Helpers
        // =================================================================

        private static string Key(int layerIndex, string bucket, string name)
            => $"layer{layerIndex}/{bucket}/{name}";

        /// <summary>Every archive key layer <paramref name="index"/> owns, params then state.</summary>
        private static IEnumerable<string> TensorKeys(BaseLayer layer, int index)
            => layer.Parameters.Keys.Select(k => Key(index, ParamPrefix, k))
               .Concat(layer.NonTrainable.Keys.Select(k => Key(index, StatePrefix, k)));

        /// <summary>
        /// Checks that <paramref name="source"/> carries a correctly-shaped tensor
        /// for every slot of every layer. Pure inspection — it writes nothing, so
        /// a caller can prove a load will succeed before starting one.
        /// </summary>
        private static void Validate(IReadOnlyList<BaseLayer> layers, IReadOnlyDictionary<string, NDArray> source, string origin)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                ValidateBucket(layers[i].Parameters, layers[i], i, ParamPrefix, source, origin);
                ValidateBucket(layers[i].NonTrainable, layers[i], i, StatePrefix, source, origin);
            }
        }

        private static void ValidateBucket(Dictionary<string, NDArray> target, BaseLayer layer, int index,
                                           string bucket, IReadOnlyDictionary<string, NDArray> source, string origin)
        {
            foreach (var kv in target)
            {
                string key = Key(index, bucket, kv.Key);
                if (!source.TryGetValue(key, out NDArray incoming))
                    throw new InvalidOperationException(
                        $"'{origin}' has no entry '{key}' for layer {index} ({layer.Name}).");

                if (!ShapeMatches(kv.Value, incoming))
                    throw new InvalidOperationException(
                        $"'{origin}' entry '{key}' has shape ({DescribeShape(incoming)}) but layer {index} " +
                        $"({layer.Name}) expects ({DescribeShape(kv.Value)}).");
            }
        }

        /// <summary>
        /// Copies one layer's tensors out of <paramref name="source"/>. Assumes
        /// <see cref="Validate"/> has already run. Values are cast to the live
        /// tensor's dtype — an archive round-trips float32 as float32, but a
        /// hand-built dictionary or a checkpoint written by another tool may not.
        /// </summary>
        private static void Assign(BaseLayer layer, int index, IReadOnlyDictionary<string, NDArray> source, string origin)
        {
            AssignBucket(layer.Parameters, index, ParamPrefix, source);
            AssignBucket(layer.NonTrainable, index, StatePrefix, source);
        }

        private static void AssignBucket(Dictionary<string, NDArray> target, int index,
                                         string bucket, IReadOnlyDictionary<string, NDArray> source)
        {
            // Snapshot the keys: the loop reassigns dictionary values, which
            // invalidates a live key enumerator.
            foreach (string name in target.Keys.ToList())
            {
                NDArray incoming = source[Key(index, bucket, name)];
                NDArray current = target[name];

                target[name] = incoming.dtype == current.dtype
                    ? incoming.copy()
                    : incoming.astype(current.typecode);
            }
        }

        private static bool ShapeMatches(NDArray a, NDArray b)
        {
            if (a.ndim != b.ndim) return false;
            for (int i = 0; i < a.ndim; i++)
                if (a.shape[i] != b.shape[i]) return false;
            return true;
        }

        private static string DescribeShape(NDArray a)
        {
            var dims = new string[a.ndim];
            for (int i = 0; i < a.ndim; i++)
                dims[i] = a.shape[i].ToString();
            return string.Join(", ", dims);
        }
    }
}
