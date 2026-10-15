using System;
using System.IO;
using System.IO.Compression;

namespace NumSharp.IO
{
    /// <summary>
    ///     The parts of NumPy's <c>numpy/lib/_datasource.py</c> that <c>np.loadtxt</c> and <c>np.savetxt</c> reach for a file
    ///     NAME: which file a name finds (a missing <c>data.txt</c> falls back to an existing <c>data.txt.gz</c>, …) and
    ///     which opener its extension picks (<c>_file_openers</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The extension is Python's <c>os.path.splitext</c> — the text from the last dot of the file name, compared
    ///     case-sensitively, so <c>data.GZ</c> is read as plain text, as NumPy reads it, and a name of dots only
    ///     (<c>.gz</c>) has no extension.
    ///     </para>
    ///     <para>
    ///     <c>.gz</c> is gzip. NumPy also opens <c>.bz2</c> with <c>bz2.open</c> and <c>.xz</c> / <c>.lzma</c> with
    ///     <c>lzma.open</c>; .NET has no decompressor for either, so NumSharp refuses those names
    ///     (<see cref="NotSupportedException"/>) rather than read compressed bytes as text. NumPy's further fall-backs (a
    ///     relative copy under the current directory, and URLs, which it downloads) are not ported.
    ///     </para>
    /// </remarks>
    internal static class PyDataSource
    {
        /// <summary>The compressed-file extensions of <c>_file_openers</c>, in its order (the order of the fall-backs).</summary>
        private static readonly string[] CompressedExtensions = { ".bz2", ".gz", ".xz", ".lzma" };

        /// <summary>Python's <c>os.path.splitext(path)[1]</c>: from the last dot of the file name, unless only dots come
        ///     before it (<c>.bashrc</c> has none); both <c>'/'</c> and <c>'\\'</c> separate directories, as on Windows.</summary>
        /// <param name="path">The path.</param>
        /// <returns>The extension with its dot, or the empty string.</returns>
        internal static string SplitExt(string path)
        {
            int sep = Math.Max(path.LastIndexOf('/'), path.LastIndexOf('\\'));
            int dot = path.LastIndexOf('.');
            if (dot > sep)
            {
                // Skip all leading dots of the file name: an extension needs a non-dot character before its dot.
                for (int i = sep + 1; i < dot; i++)
                    if (path[i] != '.')
                        return path.Substring(dot);
            }
            return "";
        }

        /// <summary>Whether <paramref name="path"/>'s extension is one of the compressed ones (<c>_iszip</c>).</summary>
        /// <param name="path">The path.</param>
        /// <returns>True for <c>.bz2</c>, <c>.gz</c>, <c>.xz</c>, <c>.lzma</c>.</returns>
        private static bool IsCompressedName(string path) => Array.IndexOf(CompressedExtensions, SplitExt(path)) >= 0;

        /// <summary>
        ///     <c>DataSource._findfile</c> for a local name: the name itself, then — unless it already names a compressed
        ///     file — the name with each compressed extension appended; the first that exists.
        /// </summary>
        /// <param name="path">The name as given.</param>
        /// <returns>The path found, or null (NumPy's <c>FileNotFoundError: &lt;path&gt; not found.</c>).</returns>
        internal static string FindFile(string path)
        {
            if (Exists(path))
                return path;
            if (IsCompressedName(path))
                return null;
            foreach (string ext in CompressedExtensions)
            {
                string candidate = path + ext;
                if (Exists(candidate))
                    return candidate;
            }
            return null;
        }

        /// <summary>Python's <c>os.path.exists</c>: a file or a directory (opening a directory then fails, as in NumPy).</summary>
        /// <param name="path">The path.</param>
        /// <returns>True when something exists there.</returns>
        private static bool Exists(string path) => File.Exists(path) || Directory.Exists(path);

        /// <summary>
        ///     Refuses a name whose opener NumSharp lacks (<c>.bz2</c>, <c>.xz</c>, <c>.lzma</c>), before anything is opened
        ///     or created.
        /// </summary>
        /// <param name="path">The file name.</param>
        /// <exception cref="NotSupportedException">A bz2 or LZMA file name.</exception>
        internal static void ThrowIfUnsupportedCompression(string path)
        {
            string ext = SplitExt(path);
            if (ext == ".bz2" || ext == ".xz" || ext == ".lzma")
                throw new NotSupportedException(
                    $"NumSharp cannot open '{path}': NumPy opens a '{ext}' name with {(ext == ".bz2" ? "bz2" : "lzma")}.open, " +
                    "and .NET has no decompressor for it (gzip '.gz' names are supported).");
        }

        /// <summary>Whether the opener of <paramref name="path"/> is <c>gzip.open</c> (the extension is exactly <c>.gz</c>).</summary>
        /// <param name="path">The file name.</param>
        /// <returns>True for gzip.</returns>
        internal static bool IsGzip(string path) => SplitExt(path) == ".gz";

        /// <summary>Opens a found file for reading with its opener: the raw bytes, or a gzip file's decompressed bytes.</summary>
        /// <param name="found">The path <see cref="FindFile"/> returned.</param>
        /// <returns>The byte stream (the caller owns it).</returns>
        /// <exception cref="NotSupportedException">A bz2 or LZMA file.</exception>
        /// <exception cref="IOException">The file cannot be opened.</exception>
        /// <exception cref="UnauthorizedAccessException">The path is a directory, or access is denied.</exception>
        internal static Stream OpenRead(string found)
        {
            ThrowIfUnsupportedCompression(found);
            Stream fs = new FileStream(found, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
            return IsGzip(found) ? new GZipStream(fs, CompressionMode.Decompress) : fs;
        }
    }
}
