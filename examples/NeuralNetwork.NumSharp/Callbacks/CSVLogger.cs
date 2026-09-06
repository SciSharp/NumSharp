using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace NeuralNetwork.NumSharp.Callbacks
{
    /// <summary>
    /// Streams per-epoch metrics to a CSV file — a port of
    /// <c>keras.callbacks.CSVLogger</c>.
    ///
    /// <para>Keras quirks preserved on purpose:</para>
    /// <list type="bullet">
    ///   <item><b>The column set is frozen at the first epoch</b> and taken as the
    ///     SORTED log keys. A metric that only appears later gets no column; a
    ///     metric that disappears is written as <c>NA</c>. This makes the file a
    ///     rectangle, which is the point of a CSV.</item>
    ///   <item><b>The <c>epoch</c> column is 0-BASED</b> — Keras writes the raw
    ///     callback index here, unlike the 1-based epoch it formats into
    ///     checkpoint filenames and console output.</item>
    /// </list>
    ///
    /// <para>The file is flushed after every row, so a run killed mid-training
    /// still leaves a readable log. It is closed in
    /// <see cref="BaseCallback.OnTrainEnd"/> — which the trainer invokes even when
    /// training stopped early or threw.</para>
    /// </summary>
    public class CSVLogger : BaseCallback
    {
        public string Filename { get; }

        /// <summary>Field delimiter. Default ",".</summary>
        public string Separator { get; }

        /// <summary>Append to an existing file (and skip the header if it has content).</summary>
        public bool Append { get; }

        /// <summary>Columns after <c>epoch</c>, fixed at the first epoch.</summary>
        public IReadOnlyList<string> Keys => _keys;

        private string[] _keys;
        private StreamWriter _writer;
        private bool _headerWritten;

        public CSVLogger(string filename, string separator = ",", bool append = false)
        {
            if (string.IsNullOrWhiteSpace(filename)) throw new ArgumentException("filename is required", nameof(filename));
            Filename = filename;
            Separator = separator ?? ",";
            Append = append;
        }

        public override void OnTrainBegin()
        {
            string dir = Path.GetDirectoryName(Path.GetFullPath(Filename));
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            // Appending to a file that already has rows must not repeat the
            // header — Keras checks the existing file's length for exactly this.
            bool existingHasContent = Append && File.Exists(Filename) && new FileInfo(Filename).Length > 0;

            _writer = new StreamWriter(new FileStream(Filename, Append ? FileMode.Append : FileMode.Create,
                                                      FileAccess.Write, FileShare.Read));
            _headerWritten = existingHasContent;
            _keys = null;
        }

        public override void OnEpochEnd(int epoch, IDictionary<string, float> logs)
        {
            if (_writer == null)
                return;

            if (_keys == null)
                _keys = (logs?.Keys ?? Enumerable.Empty<string>()).OrderBy(k => k, StringComparer.Ordinal).ToArray();

            if (!_headerWritten)
            {
                _writer.WriteLine(string.Join(Separator, new[] { "epoch" }.Concat(_keys)));
                _headerWritten = true;
            }

            var row = new StringBuilder();
            // Keras writes the raw 0-based callback epoch index here.
            row.Append(epoch.ToString(CultureInfo.InvariantCulture));
            foreach (string key in _keys)
            {
                row.Append(Separator);
                row.Append(logs != null && logs.TryGetValue(key, out float v)
                    ? v.ToString("R", CultureInfo.InvariantCulture)
                    : "NA");
            }

            _writer.WriteLine(row.ToString());
            _writer.Flush();
        }

        public override void OnTrainEnd()
        {
            _writer?.Dispose();
            _writer = null;
        }
    }
}
