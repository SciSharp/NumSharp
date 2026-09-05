using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     The identity of the bundled OpenBLAS binaries BY CONTENT: the sha256 of every RID's pinned
    ///     library from <c>tools/openblas-manifest.json</c>, embedded into this assembly at build time.
    /// </summary>
    /// <remarks>
    ///     "Is this file the one the package bundles?" cannot be answered from where the file sits.
    ///     A PackageReference restore keeps <c>runtimes/&lt;rid&gt;/native/</c>, a RID-specific or
    ///     single-file publish flattens the bundle into the app root beside everything else, and a
    ///     ProjectReference build has neither the package nor the manifest on disk — so a layout
    ///     heuristic reported the flattened bundle as "not bundled" and would have reported any
    ///     hand-dropped OpenBLAS in the right folder as "bundled". Hashing the file and comparing
    ///     against the pin is the same identity rule the host-pinned parity corpus uses, and it is
    ///     strictly stronger. The hash is computed once per file (path, length, mtime) — ~50 ms for
    ///     the 25 MB library — and only when <c>OpenBlasEngine.IsBundledLibrary</c> is asked.
    /// </remarks>
    internal static class BundledAssetPin
    {
        /// <summary>The embedded resource name the csproj assigns to the manifest copy.</summary>
        internal const string ResourceName = "NumSharp.Interop.OpenBLAS.openblas-manifest.json";

        private static readonly Lazy<HashSet<string>> PinnedHashes =
            new Lazy<HashSet<string>>(LoadPinnedHashes, System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

        private static readonly object Sync = new object();
        private static string _cachedPath;
        private static long _cachedLength;
        private static DateTime _cachedWriteUtc;
        private static bool _cachedResult;

        /// <summary>True when the assembly carries the manifest pin (a normal build does).</summary>
        internal static bool Available => PinnedHashes.Value.Count > 0;

        /// <summary>The pinned sha256 values (lowercase hex), one per RID — for diagnostics and tests.</summary>
        internal static IReadOnlyCollection<string> Hashes => PinnedHashes.Value;

        /// <summary>
        ///     True when the file at <paramref name="path"/> hashes to one of the pinned bundled
        ///     libraries. False for a missing/unreadable file or a bare loader name.
        /// </summary>
        /// <remarks>Never throws.</remarks>
        internal static bool IsPinnedBinary(string path)
        {
            if (string.IsNullOrEmpty(path) || !Available)
                return false;

            try
            {
                var info = new FileInfo(path);
                if (!info.Exists)
                    return false;

                lock (Sync)
                {
                    if (_cachedPath != null &&
                        string.Equals(_cachedPath, info.FullName, StringComparison.OrdinalIgnoreCase) &&
                        _cachedLength == info.Length && _cachedWriteUtc == info.LastWriteTimeUtc)
                        return _cachedResult;
                }

                string hash;
                using (var stream = new FileStream(info.FullName, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan))
                    hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();

                bool result = PinnedHashes.Value.Contains(hash);
                lock (Sync)
                {
                    _cachedPath = info.FullName;
                    _cachedLength = info.Length;
                    _cachedWriteUtc = info.LastWriteTimeUtc;
                    _cachedResult = result;
                }

                return result;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
            {
                return false;
            }
        }

        private static HashSet<string> LoadPinnedHashes()
        {
            var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                using var stream = typeof(BundledAssetPin).Assembly.GetManifestResourceStream(ResourceName);
                if (stream == null)
                    return set;

                using var doc = JsonDocument.Parse(stream);
                if (doc.RootElement.ValueKind != JsonValueKind.Object ||
                    !doc.RootElement.TryGetProperty("runtimes", out var runtimes) ||
                    runtimes.ValueKind != JsonValueKind.Object)
                    return set;

                foreach (var rid in runtimes.EnumerateObject())
                {
                    if (rid.Value.ValueKind == JsonValueKind.Object &&
                        rid.Value.TryGetProperty("sha256", out var sha) &&
                        sha.ValueKind == JsonValueKind.String &&
                        !string.IsNullOrWhiteSpace(sha.GetString()))
                        set.Add(sha.GetString().Trim().ToLowerInvariant());
                }
            }
            catch (Exception e) when (e is IOException or JsonException or BadImageFormatException or InvalidOperationException or NotSupportedException)
            {
                // No pin → IsBundledLibrary falls back to the layout heuristic; never throws here.
            }

            return set;
        }
    }
}
