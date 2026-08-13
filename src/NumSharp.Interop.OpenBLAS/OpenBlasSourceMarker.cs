using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     The source marker (<c>openblas.source.json</c>) the BUILD phase writes next to a staged
    ///     OpenBLAS binary, read back by the RUNTIME phase to learn where that binary came from.
    /// </summary>
    /// <remarks>
    ///     The marker exists because the delivery design's "same binary" invariant makes a
    ///     build-staged version override and the package's bundled asset CONTENT-IDENTICAL, dropped
    ///     into the same <c>runtimes/&lt;rid&gt;/native/</c> layout — indistinguishable by
    ///     inspection. Yet they carry opposite contracts: a version override is a hard requirement
    ///     (its folder is probed ahead of everything but an explicit path, and a miss THROWS rather
    ///     than falling through), while the bundle is the zero-config parity default (probed after
    ///     the overrides, and a miss falls through to machine tooling). The marker is the one bit of
    ///     metadata that flips the folder between those roles. See
    ///     <c>docs/OPENBLAS_DELIVERY_DESIGN.md</c> §8.
    ///     <para>Schema (all keys optional except <c>mode</c>):</para>
    ///     <code>
    ///     { "mode": "version|path|none", "distribution": "scipy-openblas64",
    ///       "version": "0.3.34.106.0", "sha256": "…", "required": true, "path": "…" }
    ///     </code>
    ///     <para>
    ///     <c>mode="version"</c>: the staged folder holds a build-downloaded pinned version; loading
    ///     from it is REQUIRED (unless the marker explicitly says <c>"required": false</c>), and
    ///     when <c>sha256</c> is present each candidate file is hash-verified before it is loaded —
    ///     a mismatched file is not the pinned binary and is never silently substituted.
    ///     <c>mode="path"</c>: <c>path</c> is a read-in-place directory (the build-time analog of
    ///     <c>NUMSHARP_OPENBLAS_PATH</c>), non-binding — a miss falls through.
    ///     <c>mode="none"</c> or no marker at all: the folder is the ordinary bundle.
    ///     </para>
    /// </remarks>
    internal sealed class OpenBlasSourceMarker
    {
        internal const string FileName = "openblas.source.json";

        /// <summary>"version", "path" or "none".</summary>
        internal string Mode { get; private set; }

        /// <summary>scipy-openblas distribution the override pinned (informational).</summary>
        internal string Distribution { get; private set; }

        /// <summary>The pinned scipy-openblas version (informational + error text).</summary>
        internal string Version { get; private set; }

        /// <summary>Expected sha256 of the staged library, lowercase hex, or null to skip the check.</summary>
        internal string Sha256 { get; private set; }

        /// <summary>
        ///     Whether a version-mode miss is fatal. Defaults to true for <c>mode="version"</c>;
        ///     a build may write <c>"required": false</c> to stage a preferred-but-optional binary.
        /// </summary>
        internal bool Required { get; private set; }

        /// <summary>Read directory (absolute, or relative to the marker's own directory), or null.</summary>
        internal string Path { get; private set; }

        /// <summary>Full path of the marker file itself.</summary>
        internal string MarkerPath { get; private set; }

        internal bool IsVersionMode => string.Equals(Mode, "version", StringComparison.OrdinalIgnoreCase);

        internal bool IsPathMode => string.Equals(Mode, "path", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        ///     Locates and parses the first marker reachable from the app's asset directories:
        ///     <c>&lt;base&gt;/openblas.source.json</c> (the flattened / output-root spot), then
        ///     <c>&lt;base&gt;/runtimes/&lt;rid&gt;/native/openblas.source.json</c> (next to a staged
        ///     binary in the NuGet layout).
        /// </summary>
        /// <remarks>
        ///     Never throws — this runs on the module-initializer path, where an unreadable file must
        ///     not become a <c>TypeInitializationException</c>. A marker that EXISTS but cannot be
        ///     parsed is reported to stderr (it was written by a build and its intent is lost, which
        ///     is worth one loud line) and then ignored.
        /// </remarks>
        internal static OpenBlasSourceMarker TryFind()
        {
            foreach (var baseDir in CBlasNative.ProbeBases())
            {
                var m = TryRead(System.IO.Path.Combine(baseDir, FileName));
                if (m != null)
                    return m;

                foreach (var rid in CBlasNative.RuntimeIdentifierCandidates())
                {
                    m = TryRead(System.IO.Path.Combine(baseDir, "runtimes", rid, "native", FileName));
                    if (m != null)
                        return m;
                }
            }

            return null;
        }

        private static OpenBlasSourceMarker TryRead(string file)
        {
            try
            {
                if (!File.Exists(file))
                    return null;

                using var doc = JsonDocument.Parse(File.ReadAllText(file));
                var root = doc.RootElement;
                if (root.ValueKind != JsonValueKind.Object)
                    return null;

                var marker = new OpenBlasSourceMarker
                {
                    MarkerPath = file,
                    Mode = GetString(root, "mode") ?? "none",
                    Distribution = GetString(root, "distribution"),
                    Version = GetString(root, "version"),
                    Sha256 = GetString(root, "sha256")?.ToLowerInvariant(),
                    Path = GetString(root, "path"),
                };
                marker.Required = root.TryGetProperty("required", out var req) &&
                                  (req.ValueKind == JsonValueKind.True || req.ValueKind == JsonValueKind.False)
                    ? req.GetBoolean()
                    : marker.IsVersionMode;
                return marker;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException)
            {
                if (!(e is IOException) || File.Exists(file))
                    Console.Error.WriteLine(
                        $"NumSharp.Interop.OpenBLAS: ignoring unreadable source marker '{file}': {e.Message}");
                return null;
            }
        }

        private static string GetString(JsonElement root, string name)
        {
            return root.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String
                ? v.GetString()
                : null;
        }

        /// <summary>
        ///     The directories this marker says the staged/override binary should be read from.
        /// </summary>
        /// <remarks>
        ///     An explicit <c>path</c> wins (relative resolves against the marker's own directory —
        ///     the build wrote both, so they travel together); with none, the marker sits next to
        ///     the staged binary and its own directory IS the read location.
        /// </remarks>
        internal IEnumerable<string> ReadDirectories()
        {
            var markerDir = System.IO.Path.GetDirectoryName(MarkerPath);
            if (!string.IsNullOrEmpty(Path))
            {
                string resolved = Path;
                try
                {
                    if (!System.IO.Path.IsPathRooted(resolved) && !string.IsNullOrEmpty(markerDir))
                        resolved = System.IO.Path.GetFullPath(System.IO.Path.Combine(markerDir, resolved));
                }
                catch (ArgumentException)
                {
                    yield break; // malformed path in the marker — nothing to read
                }

                yield return resolved;
                yield break;
            }

            if (!string.IsNullOrEmpty(markerDir))
                yield return markerDir;
        }

        /// <summary>
        ///     True when a marker declares <paramref name="directory"/> an override read-location —
        ///     i.e. a library loaded from there is a build-staged override, NOT the bundle, even
        ///     though the folder layout (and, by the design's invariant, the bytes) may be identical.
        /// </summary>
        internal static bool DeclaresOverrideFor(string directory)
        {
            if (string.IsNullOrEmpty(directory))
                return false;

            var marker = TryFind();
            if (marker == null || !(marker.IsVersionMode || marker.IsPathMode))
                return false;

            foreach (var dir in marker.ReadDirectories())
            {
                if (PathsEqual(dir, directory))
                    return true;

                // A marker's path may legally name the library FILE rather than its folder
                // (Expand accepts both) — a library loaded from that file's directory is still
                // the declared override.
                if (File.Exists(dir))
                {
                    string parent;
                    try
                    {
                        parent = System.IO.Path.GetDirectoryName(dir);
                    }
                    catch (Exception e) when (e is ArgumentException or PathTooLongException)
                    {
                        continue;
                    }

                    if (!string.IsNullOrEmpty(parent) && PathsEqual(parent, directory))
                        return true;
                }
            }

            return false;
        }

        private static bool PathsEqual(string a, string b)
        {
            try
            {
                a = System.IO.Path.GetFullPath(a).TrimEnd(System.IO.Path.DirectorySeparatorChar, System.IO.Path.AltDirectorySeparatorChar);
                b = System.IO.Path.GetFullPath(b).TrimEnd(System.IO.Path.DirectorySeparatorChar, System.IO.Path.AltDirectorySeparatorChar);
            }
            catch (Exception e) when (e is ArgumentException or PathTooLongException or NotSupportedException)
            {
                return false;
            }

            return string.Equals(a, b, StringComparison.OrdinalIgnoreCase);
        }
    }
}
