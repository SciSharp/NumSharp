using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.MnistMlp;

namespace NeuralNetwork.NumSharp.Serialization
{
    /// <summary>
    /// Architecture persistence — the Keras <c>model.to_json()</c> /
    /// <c>model_from_json()</c> pair. A model is a JSON array of
    /// <c>{"class_name", "config"}</c> objects in forward order:
    ///
    /// <code>
    /// [
    ///  { "class_name": "FullyConnectedFused", "config": { "input_dim": 784, "units": 128, "activation": "relu" } },
    ///  { "class_name": "FullyConnectedFused", "config": { "input_dim": 128, "units": 10,  "activation": "linear" } }
    /// ]
    /// </code>
    ///
    /// <para>Architecture and weights are deliberately separate files, as in
    /// Keras: <see cref="FromJson"/> rebuilds freshly-initialized layers and
    /// <see cref="ModelWeights.Load"/> then fills them. Because the weight
    /// archive is keyed by layer index, a JSON round-trip produces a model whose
    /// slots line up with any checkpoint taken from the original.</para>
    ///
    /// <para>Layer types are resolved through a registry. Every built-in that
    /// can be rebuilt from hyper-parameters is registered below; user layers
    /// call <see cref="Register"/> with their own factory.</para>
    /// </summary>
    public static class ModelArchitecture
    {
        private static readonly Dictionary<string, Func<LayerConfig, BaseLayer>> Factories =
            new Dictionary<string, Func<LayerConfig, BaseLayer>>(StringComparer.OrdinalIgnoreCase);

        static ModelArchitecture()
        {
            // The built-ins know themselves; a central registry keeps the layer
            // classes free of a serialization dependency and makes the set of
            // rebuildable types greppable in one place.
            Register("FullyConnected", c => new FullyConnected(
                c.GetInt("input_dim"),
                c.GetInt("units"),
                c.GetString("activation", ""),
                c.GetBool("use_bias", true)));

            Register("FullyConnectedFused", c => new FullyConnectedFused(
                c.GetInt("input_dim"),
                c.GetInt("units"),
                c.GetString("activation", "")));

            Register("Dropout", c => new Dropout(c.GetFloat("rate")));

            Register("BatchNormalization", c => new BatchNormalization(
                c.GetInt("features"),
                c.GetFloat("momentum", 0.99f),
                c.GetFloat("epsilon", 1e-3f),
                c.GetBool("scale", true),
                c.GetBool("center", true)));

            Register("LayerNormalization", c => new LayerNormalization(
                c.GetInt("features"),
                c.GetFloat("epsilon", 1e-3f),
                c.GetBool("scale", true),
                c.GetBool("center", true)));

            Register("Embedding", c => new Embedding(
                c.GetInt("input_dim"),
                c.GetInt("output_dim")));

            Register("Flatten", c => new Flatten());

            Register("Reshape", c => new Reshape(c.GetIntArray("target_shape")));
        }

        /// <summary>
        /// Registers (or replaces) the factory that rebuilds
        /// <paramref name="className"/> from its config. Names are compared
        /// case-insensitively.
        /// </summary>
        public static void Register(string className, Func<LayerConfig, BaseLayer> factory)
        {
            if (string.IsNullOrWhiteSpace(className)) throw new ArgumentException("className is required", nameof(className));
            Factories[className] = factory ?? throw new ArgumentNullException(nameof(factory));
        }

        /// <summary>Layer type names that <see cref="FromJson"/> can rebuild.</summary>
        public static IEnumerable<string> RegisteredTypes => Factories.Keys;

        // =================================================================
        // Serialize
        // =================================================================

        public static string ToJson(IReadOnlyList<BaseLayer> layers, bool indented = true)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));

            var payload = new List<Dictionary<string, object>>(layers.Count);
            for (int i = 0; i < layers.Count; i++)
            {
                LayerConfig cfg = layers[i].GetConfig();
                if (!Factories.ContainsKey(cfg.ClassName))
                    throw new InvalidOperationException(
                        $"Layer {i} ({layers[i].Name}) reports class '{cfg.ClassName}', which has no registered " +
                        $"factory — it cannot be rebuilt from JSON. Override BaseLayer.GetConfig() and call " +
                        $"ModelArchitecture.Register(\"{cfg.ClassName}\", ...). Registered: [{string.Join(", ", Factories.Keys)}].");

                payload.Add(new Dictionary<string, object>
                {
                    ["class_name"] = cfg.ClassName,
                    ["config"] = cfg.Config,
                });
            }

            return JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = indented });
        }

        public static void Save(IReadOnlyList<BaseLayer> layers, string path)
            => File.WriteAllText(path, ToJson(layers));

        // =================================================================
        // Deserialize
        // =================================================================

        /// <summary>
        /// Rebuilds the layer stack. Layers come back freshly initialized —
        /// load a checkpoint with <see cref="ModelWeights.Load"/> to restore the
        /// trained values.
        /// </summary>
        public static List<BaseLayer> FromJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json)) throw new ArgumentException("json is required", nameof(json));

            using (var doc = JsonDocument.Parse(json))
            {
                if (doc.RootElement.ValueKind != JsonValueKind.Array)
                    throw new InvalidOperationException(
                        $"Expected a JSON array of layer descriptors, got {doc.RootElement.ValueKind}.");

                var layers = new List<BaseLayer>();
                int index = 0;
                foreach (var element in doc.RootElement.EnumerateArray())
                {
                    string className = element.TryGetProperty("class_name", out var cn)
                        ? cn.GetString()
                        : throw new InvalidOperationException($"Layer {index} has no 'class_name'.");

                    if (!Factories.TryGetValue(className, out var factory))
                        throw new InvalidOperationException(
                            $"Layer {index} has unknown class '{className}'. Registered: " +
                            $"[{string.Join(", ", Factories.Keys)}]. Call ModelArchitecture.Register to add it.");

                    var cfg = new LayerConfig(className);
                    if (element.TryGetProperty("config", out var configElement) &&
                        configElement.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var prop in configElement.EnumerateObject())
                            cfg.Set(prop.Name, ReadJsonValue(prop.Value));
                    }

                    layers.Add(factory(cfg));
                    index++;
                }

                return layers;
            }
        }

        public static List<BaseLayer> Load(string path) => FromJson(File.ReadAllText(path));

        /// <summary>
        /// JsonElement → CLR primitive. Numbers come back as <c>double</c>
        /// regardless of how they were written, which is why
        /// <see cref="LayerConfig"/>'s getters go through <see cref="Convert"/>.
        /// </summary>
        private static object ReadJsonValue(JsonElement e)
        {
            switch (e.ValueKind)
            {
                case JsonValueKind.Number: return e.GetDouble();
                case JsonValueKind.String: return e.GetString();
                case JsonValueKind.True: return true;
                case JsonValueKind.False: return false;
                case JsonValueKind.Null: return null;
                case JsonValueKind.Array:
                {
                    var items = new List<object>();
                    foreach (var child in e.EnumerateArray())
                        items.Add(ReadJsonValue(child));
                    return items;
                }
                default:
                    throw new InvalidOperationException($"Unsupported config value kind {e.ValueKind}.");
            }
        }
    }
}
