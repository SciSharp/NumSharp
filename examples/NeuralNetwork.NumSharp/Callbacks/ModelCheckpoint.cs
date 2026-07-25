using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NeuralNetwork.NumSharp.Serialization;

namespace NeuralNetwork.NumSharp.Callbacks
{
    /// <summary>
    /// Writes the model's weights to disk at the end of an epoch — a port of
    /// <c>keras.callbacks.ModelCheckpoint</c> restricted to epoch frequency.
    ///
    /// <para>The output is one <c>.npz</c> per save, produced by
    /// <see cref="ModelWeights.Save"/> — a genuine NumPy archive, so a checkpoint
    /// can be inspected with <c>numpy.load()</c> without any .NET involvement.
    /// There is no "save the whole model" mode: this framework's architecture
    /// lives in <see cref="ModelArchitecture"/> as JSON, so
    /// <c>save_weights_only</c> is effectively always true and is not exposed as
    /// an option.</para>
    ///
    /// <para><see cref="Filepath"/> may contain <c>{epoch}</c> and any log key in
    /// braces, with an optional .NET format specifier after a colon —
    /// <c>"ckpt_{epoch:D3}_{val_loss:F4}.npz"</c>. As in Keras, <c>{epoch}</c>
    /// expands 1-BASED even though the callback API is 0-based. A path with no
    /// placeholders is overwritten every save, which is the usual
    /// <c>SaveBestOnly</c> setup.</para>
    /// </summary>
    public class ModelCheckpoint : BaseCallback
    {
        /// <summary>Destination path; may contain <c>{epoch}</c> / <c>{metric}</c> placeholders.</summary>
        public string Filepath { get; }

        public string Monitor { get; }

        /// <summary>Only overwrite when the monitored metric improved.</summary>
        public bool SaveBestOnly { get; }

        /// <summary>"auto" (default), "min" or "max".</summary>
        public string Mode { get; }

        /// <summary>0 = silent, 1 = one line per save / skip.</summary>
        public int Verbose { get; }

        /// <summary>Store the archive with deflate rather than as-is.</summary>
        public bool Compressed { get; }

        /// <summary>Best monitored value written so far.</summary>
        public float Best { get; private set; }

        /// <summary>Path of the most recent successful save, or null.</summary>
        public string LastSavedPath { get; private set; }

        /// <summary>Number of archives written this run.</summary>
        public int SaveCount { get; private set; }

        private readonly bool _maximize;

        public ModelCheckpoint(string filepath, string monitor = "val_loss", bool saveBestOnly = false,
                               string mode = "auto", int verbose = 0, bool compressed = false)
        {
            if (string.IsNullOrWhiteSpace(filepath)) throw new ArgumentException("filepath is required", nameof(filepath));

            Filepath = filepath;
            Monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
            SaveBestOnly = saveBestOnly;
            Mode = mode;
            Verbose = verbose;
            Compressed = compressed;

            _maximize = ResolveMaximize(monitor, mode);
        }

        public override void OnTrainBegin()
        {
            Best = _maximize ? float.NegativeInfinity : float.PositiveInfinity;
            LastSavedPath = null;
            SaveCount = 0;
        }

        public override void OnEpochEnd(int epoch, IDictionary<string, float> logs)
        {
            string path = FormatPath(Filepath, epoch, logs);

            if (!SaveBestOnly)
            {
                Write(path, epoch);
                return;
            }

            if (!TryGetMonitorValue(logs, Monitor, out float current))
            {
                // Keras warns and skips rather than failing the run.
                if (Verbose > 0)
                    Console.WriteLine($"Epoch {epoch + 1}: ModelCheckpoint can't save best model, " +
                                      $"metric '{Monitor}' is not available. Skipping.");
                return;
            }

            bool improved = _maximize ? current > Best : current < Best;
            if (improved)
            {
                if (Verbose > 0)
                    Console.WriteLine($"Epoch {epoch + 1}: {Monitor} improved from {Best:F5} to {current:F5}, " +
                                      $"saving model to {path}");
                Best = current;
                Write(path, epoch);
            }
            else if (Verbose > 0)
            {
                Console.WriteLine($"Epoch {epoch + 1}: {Monitor} did not improve from {Best:F5}");
            }
        }

        private void Write(string path, int epoch)
        {
            string dir = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            ModelWeights.Save(Context.Layers, path, Compressed);
            LastSavedPath = path;
            SaveCount++;

            if (Verbose > 0 && !SaveBestOnly)
                Console.WriteLine($"Epoch {epoch + 1}: saving model to {path}");
        }

        /// <summary>
        /// Expands <c>{name}</c> and <c>{name:format}</c> placeholders from the
        /// logs, plus the synthetic 1-based <c>{epoch}</c>. An unknown name is
        /// left verbatim rather than throwing — a filename is not worth failing a
        /// training run over, and the literal braces make the mistake obvious.
        /// </summary>
        internal static string FormatPath(string template, int epoch, IDictionary<string, float> logs)
        {
            if (template.IndexOf('{') < 0)
                return template;

            var sb = new StringBuilder(template.Length + 16);
            for (int i = 0; i < template.Length; i++)
            {
                if (template[i] != '{')
                {
                    sb.Append(template[i]);
                    continue;
                }

                int close = template.IndexOf('}', i + 1);
                if (close < 0)
                {
                    sb.Append(template, i, template.Length - i);
                    break;
                }

                string token = template.Substring(i + 1, close - i - 1);
                int colon = token.IndexOf(':');
                string key = colon < 0 ? token : token.Substring(0, colon);
                string fmt = colon < 0 ? null : token.Substring(colon + 1);

                if (string.Equals(key, "epoch", StringComparison.Ordinal))
                {
                    // Keras formats epoch 1-based in filenames.
                    int shown = epoch + 1;
                    sb.Append(fmt == null ? shown.ToString(CultureInfo.InvariantCulture)
                                          : shown.ToString(fmt, CultureInfo.InvariantCulture));
                }
                else if (logs != null && logs.TryGetValue(key, out float value))
                {
                    sb.Append(fmt == null ? value.ToString(CultureInfo.InvariantCulture)
                                          : value.ToString(fmt, CultureInfo.InvariantCulture));
                }
                else
                {
                    sb.Append('{').Append(token).Append('}');
                }

                i = close;
            }

            return sb.ToString();
        }
    }
}
