using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace NeuralNetwork.NumSharp.Serialization
{
    /// <summary>
    /// Serializable description of one layer — the Keras
    /// <c>{"class_name": "Dense", "config": {"units": 128, ...}}</c> shape.
    ///
    /// <para>Values are restricted to JSON primitives (integers, reals, strings,
    /// booleans) and flat arrays of them, which is all any layer's hyper-
    /// parameters need. Weights are NOT part of a config — they travel in the
    /// <c>.npz</c> written by <see cref="ModelWeights"/>, exactly as Keras keeps
    /// <c>to_json</c> and <c>save_weights</c> separate.</para>
    ///
    /// <para>The typed getters accept anything <see cref="Convert"/> can widen,
    /// because a value written as a C# <c>int</c> comes back from
    /// <see cref="System.Text.Json"/> as a <c>double</c> — the round-trip must
    /// not care.</para>
    /// </summary>
    public sealed class LayerConfig
    {
        /// <summary>Type name used to look up the factory on the way back in.</summary>
        public string ClassName { get; }

        /// <summary>Hyper-parameters, in insertion order.</summary>
        public Dictionary<string, object> Config { get; }

        public LayerConfig(string className)
        {
            ClassName = className ?? throw new ArgumentNullException(nameof(className));
            Config = new Dictionary<string, object>();
        }

        /// <summary>Fluent setter — <c>new LayerConfig("Dense").Set("units", 128)</c>.</summary>
        public LayerConfig Set(string key, object value)
        {
            Config[key] = value;
            return this;
        }

        public bool Has(string key) => Config.ContainsKey(key);

        // =================================================================
        // Typed reads (used by the factories in ModelArchitecture)
        // =================================================================

        public int GetInt(string key) => Convert.ToInt32(Require(key), CultureInfo.InvariantCulture);

        public int GetInt(string key, int fallback)
            => Config.TryGetValue(key, out var v) && v != null
                ? Convert.ToInt32(v, CultureInfo.InvariantCulture)
                : fallback;

        public float GetFloat(string key) => Convert.ToSingle(Require(key), CultureInfo.InvariantCulture);

        public float GetFloat(string key, float fallback)
            => Config.TryGetValue(key, out var v) && v != null
                ? Convert.ToSingle(v, CultureInfo.InvariantCulture)
                : fallback;

        public bool GetBool(string key, bool fallback)
            => Config.TryGetValue(key, out var v) && v != null
                ? Convert.ToBoolean(v, CultureInfo.InvariantCulture)
                : fallback;

        public string GetString(string key, string fallback = null)
            => Config.TryGetValue(key, out var v) && v != null
                ? Convert.ToString(v, CultureInfo.InvariantCulture)
                : fallback;

        /// <summary>
        /// Flat integer array (shapes). Accepts <c>int[]</c> as written and the
        /// <c>List&lt;object&gt;</c> System.Text.Json hands back.
        /// </summary>
        public int[] GetIntArray(string key)
        {
            object v = Require(key);
            if (v is int[] direct) return direct;
            if (v is IEnumerable seq && !(v is string))
                return seq.Cast<object>().Select(o => Convert.ToInt32(o, CultureInfo.InvariantCulture)).ToArray();
            throw new InvalidOperationException($"config['{key}'] is {v.GetType().Name}, not an array of integers.");
        }

        private object Require(string key)
        {
            if (!Config.TryGetValue(key, out var v) || v == null)
                throw new InvalidOperationException(
                    $"Layer config for '{ClassName}' is missing required key '{key}' " +
                    $"(has: [{string.Join(", ", Config.Keys)}]).");
            return v;
        }
    }
}
