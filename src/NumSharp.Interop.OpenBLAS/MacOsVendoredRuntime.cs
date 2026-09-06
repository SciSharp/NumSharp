using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     Makes the bundled macOS OpenBLAS dylib loadable by materializing its vendored Fortran
    ///     runtime where the dylib's own load commands expect it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The problem.</b> scipy-openblas' macOS wheel is post-processed by <c>delocate</c>,
    ///     which parks <c>libgfortran</c>/<c>libquadmath</c>/<c>libgcc_s</c> in a
    ///     <c>.dylibs/</c> folder BESIDE the wheel's <c>lib/</c> and rewrites the main dylib's
    ///     <c>LC_LOAD_DYLIB</c> entries to <c>@loader_path/../.dylibs/&lt;name&gt;</c>. NuGet, however,
    ///     delivers native assets only from UNDER <c>runtimes/&lt;rid&gt;/native/</c>: a sibling
    ///     folder is silently dropped from every consumer, and a nested one
    ///     (<c>native/.dylibs/</c>, where this package now stages the deps) is delivered but sits
    ///     one level too deep for the load command — and a RID-specific publish flattens it away
    ///     entirely, leaving the deps beside the main dylib in the app root, where
    ///     <c>@loader_path/../.dylibs</c> points OUTSIDE the app. So a plain <c>dlopen</c> of the
    ///     bundled dylib fails on every macOS consumer with "Library not loaded".
    ///     </para>
    ///     <para>
    ///     <b>Why not pre-load the deps or relink.</b> dyld matches an already-loaded image against a
    ///     requested path by that path or by the image's install name — and delocate's deps carry
    ///     the placeholder install name <c>/DLC/scipy_openblas64/.dylibs/…</c>, so loading them
    ///     first by absolute path satisfies nothing. Relinking the main dylib
    ///     (<c>install_name_tool</c>) needs macOS tooling plus re-signing, and the assets are staged
    ///     on a Linux runner; it would also change the bytes the parity pin hashes.
    ///     </para>
    ///     <para>
    ///     <b>What this does instead.</b> It reads the main dylib's load commands
    ///     (<see cref="MachOImage"/>), computes which <c>@loader_path/</c>-relative dependencies are
    ///     missing at the path dyld will expand, finds each beside the dylib
    ///     (<c>&lt;dir&gt;/.dylibs/&lt;name&gt;</c> — the delivered layout — then
    ///     <c>&lt;dir&gt;/&lt;name&gt;</c> — the flattened one), follows THEIR
    ///     <c>@loader_path/</c> dependencies the same way, and then puts the whole set where the
    ///     load commands point, by one of two strategies:
    ///     </para>
    ///     <list type="number">
    ///       <item><description>
    ///       <b>In place</b>, when the dylib sits in the NuGet layout
    ///       (<c>runtimes/&lt;rid&gt;/native/</c>) and every target stays inside that RID folder: a
    ///       relative directory symlink <c>runtimes/&lt;rid&gt;/.dylibs → native/.dylibs</c> (or a
    ///       per-file copy when the link cannot be made). Nothing is written outside the app's own
    ///       <c>runtimes/</c> tree. This is the normal outcome for a <c>dotnet build</c>/portable
    ///       publish, and it is free after the first process.
    ///       </description></item>
    ///       <item><description>
    ///       <b>A per-user cache copy</b>, when the folder is read-only or the layout is flattened
    ///       (a RID-specific/self-contained/single-file publish, where the expected path is the
    ///       PARENT of the app folder — never written to): the main dylib and its dependencies are
    ///       copied to <c>&lt;cache&gt;/dyld/&lt;sha256 of the main&gt;/native/…</c> +
    ///       <c>…/.dylibs/…</c> — the exact relative layout the load commands encode — and the copy
    ///       is what dyld maps. The entry is keyed by the main dylib's CONTENT hash, so a different
    ///       binary never aliases it, and every hit re-compares each file to its source byte for
    ///       byte before it is trusted (a stale or tampered entry is rebuilt). Writes go to a
    ///       temp directory renamed into place, so a killed process leaves no half-built entry.
    ///       The cache root is <c>NUMSHARP_OPENBLAS_CACHE_DIR</c>, else the same location the
    ///       build-time override cache uses (<c>%LOCALAPPDATA%</c> / <c>$XDG_CACHE_HOME</c> /
    ///       <c>~/.cache</c>, then <c>NumSharp/openblas</c>), else the temp directory.
    ///       </description></item>
    ///     </list>
    ///     <para>
    ///     Either way <c>OpenBlasEngine.LibraryPath</c> keeps naming the file discovery chose (the
    ///     parity pin hashes THAT file, and the relocated copy is verified identical to it), while
    ///     <c>OpenBlasEngine.LoadedImagePath</c> reports the file dyld actually mapped. Every
    ///     failure is a silent decline back to the caller — nothing here may throw on the
    ///     module-initializer path — and the reason travels in the returned note so an explicit
    ///     <c>OpenBlasEngine.Enable()</c> can report it.
    ///     </para>
    /// </remarks>
    internal static class MacOsVendoredRuntime
    {
        /// <summary>Sub-folder of the cache root holding relocated images: <c>&lt;root&gt;/dyld/&lt;sha256&gt;/</c>.</summary>
        internal const string CacheSubdirectory = "dyld";

        /// <summary>
        ///     Folder inside a cache entry holding the main dylib, so that its
        ///     <c>@loader_path/../.dylibs</c> stays INSIDE the entry.
        /// </summary>
        internal const string ImageSubdirectory = "native";

        /// <summary>A vendored file the image needs at a path relative to its own directory, and the staged file that satisfies it.</summary>
        internal readonly struct VendoredFile
        {
            /// <summary>Forward-slash path relative to the main dylib's directory, as dyld will expand it (e.g. <c>../.dylibs/libgfortran.5.dylib</c>).</summary>
            internal readonly string RelativeTarget;

            /// <summary>Absolute path of the staged file to place there.</summary>
            internal readonly string Source;

            internal VendoredFile(string relativeTarget, string source)
            {
                RelativeTarget = relativeTarget;
                Source = source;
            }
        }

        /// <summary>
        ///     Attempts to make <paramref name="library"/> loadable; on success
        ///     <paramref name="imagePath"/> is the file to hand to <c>dlopen</c> (the library itself
        ///     when the deps were materialized in place, or the cache copy).
        /// </summary>
        /// <param name="library">Path of the dylib that failed to load.</param>
        /// <param name="imagePath">The file to load instead, or null.</param>
        /// <param name="note">
        ///     What happened, for diagnostics — set on success (which strategy) and on any decline
        ///     that had a reason; null when the library simply is not a Mach-O image or needs
        ///     nothing (dyld's failure had another cause).
        /// </param>
        /// <returns>True when something was materialized and a retry is worth making.</returns>
        /// <remarks>Never throws.</remarks>
        internal static bool TryPrepare(string library, out string imagePath, out string note)
        {
            imagePath = null;
            note = null;
            try
            {
                if (string.IsNullOrEmpty(library) || !File.Exists(library))
                    return false;

                if (!MachOImage.TryRead(library, out var image))
                    return false; // not a Mach-O file: not ours to fix

                string full = Path.GetFullPath(library);
                string dir = Path.GetDirectoryName(full);
                if (string.IsNullOrEmpty(dir))
                    return false;

                var plan = Plan(dir, image, out string missing);
                if (plan == null)
                {
                    note = $"vendored dependency '{missing}' is not staged beside the library " +
                           "(looked in its .dylibs/ sub-folder and its own folder)";
                    return false;
                }

                if (plan.Count == 0)
                    return false; // every @loader_path dependency already resolves — not a layout problem

                string inPlaceReason = null;
                if (IsNuGetNativeLayout(dir) && AllTargetsInside(Path.GetDirectoryName(dir), dir, plan))
                {
                    if (TryMaterializeInPlace(dir, plan, out string how, out inPlaceReason))
                    {
                        imagePath = full;
                        note = "vendored runtime materialized in place: " + how;
                        return true;
                    }
                }
                else
                {
                    inPlaceReason = IsNuGetNativeLayout(dir)
                        ? "a dependency resolves outside the runtimes/<rid> folder"
                        : "not the runtimes/<rid>/native layout (flattened publish?), so ../.dylibs " +
                          "would be outside the application";
                }

                if (TryMaterializeInCache(full, plan, out imagePath, out string cacheNote))
                {
                    note = $"relocated to {imagePath} ({cacheNote}; in place not possible: {inPlaceReason})";
                    return true;
                }

                note = $"could not materialize the vendored runtime (in place: {inPlaceReason}; cache: {cacheNote})";
                imagePath = null;
                return false;
            }
            catch (Exception e) when (IsBenign(e))
            {
                note = "could not materialize the vendored runtime: " + FirstLine(e.Message);
                imagePath = null;
                return false;
            }
        }

        // ------------------------------------------------------------------ planning

        /// <summary>
        ///     The closure of <c>@loader_path/</c>-relative dependencies that are MISSING at the
        ///     path dyld will expand, each paired with the staged file that can satisfy it.
        /// </summary>
        /// <returns>
        ///     The plan (empty when nothing is missing), or null when a missing dependency cannot be
        ///     found beside the library — <paramref name="missing"/> names it.
        /// </returns>
        internal static List<VendoredFile> Plan(string dir, MachOImage image, out string missing)
        {
            missing = null;
            var plan = new List<VendoredFile>();
            var seenTargets = new HashSet<string>(StringComparer.Ordinal);
            var parsed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var queue = new Queue<KeyValuePair<string, MachOImage>>(); // (dir of the image, relative to `dir`; "" = the main)
            queue.Enqueue(new KeyValuePair<string, MachOImage>(string.Empty, image));

            while (queue.Count > 0)
            {
                var item = queue.Dequeue();
                string imageRelDir = item.Key;
                foreach (var dep in item.Value.Dependencies)
                {
                    if (!dep.StartsWith(MachOImage.LoaderPathPrefix, StringComparison.Ordinal))
                        continue; // absolute (/usr/lib/libSystem.B.dylib) or @rpath — not a vendored-layout matter

                    string relOfDep = dep.Substring(MachOImage.LoaderPathPrefix.Length);
                    string relTarget = NormalizeRelative(
                        string.IsNullOrEmpty(imageRelDir) ? relOfDep : imageRelDir + "/" + relOfDep);
                    if (string.IsNullOrEmpty(relTarget) || !seenTargets.Add(relTarget))
                        continue;

                    string expected = Path.GetFullPath(Path.Combine(dir, relTarget.Replace('/', Path.DirectorySeparatorChar)));
                    string fileToParse;
                    if (File.Exists(expected))
                    {
                        fileToParse = expected; // dyld will find it there; still follow ITS dependencies
                    }
                    else
                    {
                        string name = Path.GetFileName(relOfDep);
                        string source = FindStagedDependency(dir, name);
                        if (source == null)
                        {
                            missing = name;
                            return null;
                        }

                        plan.Add(new VendoredFile(relTarget, source));
                        fileToParse = source;
                    }

                    if (parsed.Add(fileToParse) && MachOImage.TryRead(fileToParse, out var depImage))
                    {
                        int slash = relTarget.LastIndexOf('/');
                        string depRelDir = slash < 0 ? string.Empty : relTarget.Substring(0, slash);
                        queue.Enqueue(new KeyValuePair<string, MachOImage>(depRelDir, depImage));
                    }
                }
            }

            return plan;
        }

        /// <summary>
        ///     Where a vendored dependency named <paramref name="name"/> is staged relative to the
        ///     main dylib: <c>.dylibs/</c> under its folder (the NuGet-delivered layout), else its
        ///     own folder (a RID-specific publish flattens the sub-folder away).
        /// </summary>
        internal static string FindStagedDependency(string dir, string name)
        {
            if (string.IsNullOrEmpty(name) || name.IndexOfAny(new[] { '/', '\\' }) >= 0)
                return null;

            string nested = Path.Combine(dir, ".dylibs", name);
            if (File.Exists(nested))
                return nested;

            string flat = Path.Combine(dir, name);
            return File.Exists(flat) ? flat : null;
        }

        /// <summary>
        ///     Collapses <c>.</c> and <c>..</c> segments of a forward-slash relative path, keeping any
        ///     leading <c>..</c> that climbs above the origin (<c>../.dylibs/../x</c> → <c>../x</c>).
        /// </summary>
        internal static string NormalizeRelative(string relative)
        {
            if (string.IsNullOrEmpty(relative))
                return string.Empty;

            var stack = new List<string>();
            foreach (var segment in relative.Replace('\\', '/').Split('/'))
            {
                if (segment.Length == 0 || segment == ".")
                    continue;
                if (segment == "..")
                {
                    if (stack.Count > 0 && stack[stack.Count - 1] != "..")
                        stack.RemoveAt(stack.Count - 1);
                    else
                        stack.Add("..");
                    continue;
                }

                stack.Add(segment);
            }

            return string.Join("/", stack);
        }

        /// <summary>True when <paramref name="dir"/> is a <c>runtimes/&lt;rid&gt;/native</c> folder (the NuGet layout, bundle or staged override).</summary>
        internal static bool IsNuGetNativeLayout(string dir)
        {
            if (string.IsNullOrEmpty(dir) ||
                !string.Equals(Path.GetFileName(dir), "native", StringComparison.OrdinalIgnoreCase))
                return false;

            string rid = Path.GetDirectoryName(dir);
            if (string.IsNullOrEmpty(rid))
                return false;

            string runtimes = Path.GetDirectoryName(rid);
            return !string.IsNullOrEmpty(runtimes) &&
                   string.Equals(Path.GetFileName(runtimes), "runtimes", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>True when every planned target, expanded from <paramref name="dir"/>, lies under <paramref name="root"/>.</summary>
        internal static bool AllTargetsInside(string root, string dir, List<VendoredFile> plan)
        {
            if (string.IsNullOrEmpty(root))
                return false;

            string rootFull = WithTrailingSeparator(Path.GetFullPath(root));
            foreach (var f in plan)
            {
                string target = Path.GetFullPath(Path.Combine(dir, f.RelativeTarget.Replace('/', Path.DirectorySeparatorChar)));
                if (!target.StartsWith(rootFull, StringComparison.OrdinalIgnoreCase))
                    return false;
            }

            return true;
        }

        // ------------------------------------------------------------------ in place

        /// <summary>
        ///     Materializes the plan next to the library: one relative directory symlink when every
        ///     dependency comes from one staged folder and lands in one target folder, else a
        ///     per-file copy (temp + rename, so a concurrent process never sees a partial file).
        /// </summary>
        internal static bool TryMaterializeInPlace(string dir, List<VendoredFile> plan, out string how, out string reason)
        {
            how = null;
            reason = null;
            try
            {
                if (TrySymlinkWholeFolder(dir, plan, out string linkNote))
                {
                    how = linkNote;
                    return true;
                }

                int copied = 0;
                foreach (var f in plan)
                {
                    string target = Path.GetFullPath(Path.Combine(dir, f.RelativeTarget.Replace('/', Path.DirectorySeparatorChar)));
                    if (File.Exists(target) && FilesIdentical(target, f.Source))
                        continue;

                    Directory.CreateDirectory(Path.GetDirectoryName(target));
                    CopyAtomically(f.Source, target);
                    copied++;
                }

                foreach (var f in plan)
                {
                    string target = Path.GetFullPath(Path.Combine(dir, f.RelativeTarget.Replace('/', Path.DirectorySeparatorChar)));
                    if (!File.Exists(target))
                    {
                        reason = "copied file did not appear at " + target;
                        return false;
                    }
                }

                how = $"copied {copied} file(s) beside the library" +
                      (string.IsNullOrEmpty(linkNote) ? string.Empty : " (" + linkNote + ")");
                return true;
            }
            catch (Exception e) when (IsBenign(e))
            {
                reason = FirstLine(e.Message);
                return false;
            }
        }

        /// <summary>
        ///     The cheap path: <c>runtimes/&lt;rid&gt;/.dylibs → native/.dylibs</c>. A directory
        ///     symlink keeps the two folders one thing (nothing to go stale) and costs no copy.
        /// </summary>
        private static bool TrySymlinkWholeFolder(string dir, List<VendoredFile> plan, out string note)
        {
            note = null;

            string sourceFolder = null, targetRelDir = null;
            foreach (var f in plan)
            {
                string s = Path.GetDirectoryName(Path.GetFullPath(f.Source));
                int slash = f.RelativeTarget.LastIndexOf('/');
                string t = slash < 0 ? string.Empty : f.RelativeTarget.Substring(0, slash);
                if (sourceFolder == null)
                {
                    sourceFolder = s;
                    targetRelDir = t;
                }
                else if (!string.Equals(sourceFolder, s, StringComparison.OrdinalIgnoreCase) ||
                         !string.Equals(targetRelDir, t, StringComparison.Ordinal))
                {
                    note = "dependencies span several folders";
                    return false;
                }
            }

            if (sourceFolder == null || string.IsNullOrEmpty(targetRelDir))
                return false;

            // Never link the library's OWN folder (the flattened layout) — that would expose the
            // whole application directory under another name.
            if (string.Equals(sourceFolder, Path.GetFullPath(dir), StringComparison.OrdinalIgnoreCase))
            {
                note = "dependencies sit in the library's own folder";
                return false;
            }

            string link = Path.GetFullPath(Path.Combine(dir, targetRelDir.Replace('/', Path.DirectorySeparatorChar)));
            if (Directory.Exists(link) || File.Exists(link))
            {
                note = "target folder already exists";
                return false; // an older real folder (or a foreign link): let the per-file copy complete it
            }

            string linkParent = Path.GetDirectoryName(link);
            if (string.IsNullOrEmpty(linkParent))
                return false;
            Directory.CreateDirectory(linkParent);

            string relativeTarget = Path.GetRelativePath(linkParent, sourceFolder);
            try
            {
                Directory.CreateSymbolicLink(link, relativeTarget);
            }
            catch (Exception e) when (IsBenign(e))
            {
                note = "symlink refused: " + FirstLine(e.Message);
                return false;
            }

            foreach (var f in plan)
            {
                string target = Path.GetFullPath(Path.Combine(dir, f.RelativeTarget.Replace('/', Path.DirectorySeparatorChar)));
                if (!File.Exists(target))
                {
                    note = "symlink created but " + target + " does not resolve";
                    return false; // the per-file copy will not help either while the link is there; report it
                }
            }

            note = $"symlinked {link} -> {relativeTarget}";
            return true;
        }

        // ------------------------------------------------------------------ cache

        /// <summary>
        ///     Builds (or re-uses) the per-user cache entry holding the main dylib plus the plan in the
        ///     relative layout its load commands encode, and returns the cached main to load.
        /// </summary>
        internal static bool TryMaterializeInCache(string library, List<VendoredFile> plan, out string imagePath, out string note)
        {
            imagePath = null;
            note = null;
            try
            {
                string root = ResolveCacheRoot();
                if (string.IsNullOrEmpty(root))
                {
                    note = "no cache directory could be resolved";
                    return false;
                }

                string hash = HashFile(library);
                string entry = Path.Combine(root, CacheSubdirectory, hash);
                string mainName = Path.GetFileName(library);
                string cachedMain = Path.Combine(entry, ImageSubdirectory, mainName);

                // Every target must stay inside the entry — a load command climbing further up than
                // ImageSubdirectory allows would otherwise write outside the cache.
                string entryFull = WithTrailingSeparator(Path.GetFullPath(entry));
                foreach (var f in plan)
                {
                    string t = Path.GetFullPath(Path.Combine(entry, ImageSubdirectory, f.RelativeTarget.Replace('/', Path.DirectorySeparatorChar)));
                    if (!t.StartsWith(entryFull, StringComparison.OrdinalIgnoreCase))
                    {
                        note = "dependency path escapes the cache entry: " + f.RelativeTarget;
                        return false;
                    }
                }

                if (EntryIsValid(entry, library, cachedMain, plan))
                {
                    imagePath = cachedMain;
                    note = "cache hit, every file verified against its source";
                    return true;
                }

                // Build in a temp sibling, then rename into place: a killed process cannot leave a
                // half-built entry under the content-hash name, and two processes racing the same
                // entry cannot clobber each other — the loser re-validates the winner's copy.
                string temp = entry + "." + Guid.NewGuid().ToString("N") + ".tmp";
                try
                {
                    Directory.CreateDirectory(Path.Combine(temp, ImageSubdirectory));
                    File.Copy(library, Path.Combine(temp, ImageSubdirectory, mainName), true);
                    foreach (var f in plan)
                    {
                        string t = Path.GetFullPath(Path.Combine(temp, ImageSubdirectory, f.RelativeTarget.Replace('/', Path.DirectorySeparatorChar)));
                        Directory.CreateDirectory(Path.GetDirectoryName(t));
                        File.Copy(f.Source, t, true);
                    }

                    if (Directory.Exists(entry))
                        TryDeleteDirectory(entry); // stale or incomplete — rebuilt below

                    try
                    {
                        Directory.Move(temp, entry);
                    }
                    catch (IOException) when (Directory.Exists(entry))
                    {
                        // Lost a race: another process installed the entry first.
                    }
                }
                finally
                {
                    if (Directory.Exists(temp))
                        TryDeleteDirectory(temp);
                }

                if (!EntryIsValid(entry, library, cachedMain, plan))
                {
                    note = "cache entry could not be verified after writing it";
                    return false;
                }

                imagePath = cachedMain;
                note = "cache entry created";
                return true;
            }
            catch (Exception e) when (IsBenign(e))
            {
                note = FirstLine(e.Message);
                return false;
            }
        }

        /// <summary>An entry is valid only when the main AND every planned file are byte-identical to their sources.</summary>
        private static bool EntryIsValid(string entry, string library, string cachedMain, List<VendoredFile> plan)
        {
            if (!File.Exists(cachedMain) || !FilesIdentical(cachedMain, library))
                return false;

            foreach (var f in plan)
            {
                string t = Path.GetFullPath(Path.Combine(entry, ImageSubdirectory, f.RelativeTarget.Replace('/', Path.DirectorySeparatorChar)));
                if (!File.Exists(t) || !FilesIdentical(t, f.Source))
                    return false;
            }

            return true;
        }

        /// <summary>
        ///     The cache root: <c>NUMSHARP_OPENBLAS_CACHE_DIR</c>, else the location the build-time
        ///     override cache uses (same precedence as the package's buildTransitive props:
        ///     <c>%LOCALAPPDATA%</c>, <c>$XDG_CACHE_HOME</c>, <c>~/.cache</c>, each + <c>NumSharp/openblas</c>),
        ///     else the temp directory.
        /// </summary>
        internal static string ResolveCacheRoot()
        {
            string explicitDir = EnvVars.OpenBlasCacheDir;
            if (!string.IsNullOrWhiteSpace(explicitDir))
                return Path.GetFullPath(explicitDir);

            string localAppData = Environment.GetEnvironmentVariable("LOCALAPPDATA");
            if (!string.IsNullOrWhiteSpace(localAppData))
                return Path.Combine(localAppData, "NumSharp", "openblas");

            string xdg = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
            if (!string.IsNullOrWhiteSpace(xdg))
                return Path.Combine(xdg, "NumSharp", "openblas");

            string home = Environment.GetEnvironmentVariable("HOME");
            if (!string.IsNullOrWhiteSpace(home))
                return Path.Combine(home, ".cache", "NumSharp", "openblas");

            string temp = Path.GetTempPath();
            return string.IsNullOrWhiteSpace(temp) ? null : Path.Combine(temp, "NumSharp", "openblas");
        }

        // ------------------------------------------------------------------ helpers

        internal static string HashFile(string file)
        {
            using var stream = File.OpenRead(file);
            return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }

        /// <summary>Length + streaming byte comparison — cheaper than hashing both sides.</summary>
        internal static bool FilesIdentical(string a, string b)
        {
            var fa = new FileInfo(a);
            var fb = new FileInfo(b);
            if (!fa.Exists || !fb.Exists || fa.Length != fb.Length)
                return false;

            using var sa = new FileStream(a, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
            using var sb = new FileStream(b, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
            var ba = new byte[1 << 16];
            var bb = new byte[1 << 16];
            while (true)
            {
                int na = ReadBlock(sa, ba);
                int nb = ReadBlock(sb, bb);
                if (na != nb)
                    return false;
                if (na == 0)
                    return true;
                if (!new ReadOnlySpan<byte>(ba, 0, na).SequenceEqual(new ReadOnlySpan<byte>(bb, 0, nb)))
                    return false;
            }
        }

        private static int ReadBlock(Stream s, byte[] buffer)
        {
            int total = 0;
            while (total < buffer.Length)
            {
                int n = s.Read(buffer, total, buffer.Length - total);
                if (n <= 0)
                    break;
                total += n;
            }

            return total;
        }

        /// <summary>Copy via a unique temp name in the target folder, then an overwriting rename.</summary>
        private static void CopyAtomically(string source, string target)
        {
            string tmp = target + "." + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                File.Copy(source, tmp, true);
                File.Move(tmp, target, true);
            }
            finally
            {
                if (File.Exists(tmp))
                {
                    try { File.Delete(tmp); }
                    catch (Exception e) when (IsBenign(e)) { }
                }
            }
        }

        private static void TryDeleteDirectory(string dir)
        {
            try
            {
                Directory.Delete(dir, true);
            }
            catch (Exception e) when (IsBenign(e))
            {
            }
        }

        private static string WithTrailingSeparator(string path)
            => path.EndsWith(Path.DirectorySeparatorChar) || path.EndsWith(Path.AltDirectorySeparatorChar)
                ? path
                : path + Path.DirectorySeparatorChar;

        private static bool IsBenign(Exception e)
            => e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException ||
               e is System.Security.SecurityException;

        private static string FirstLine(string message)
        {
            if (string.IsNullOrEmpty(message))
                return string.Empty;
            int i = message.IndexOfAny(new[] { '\r', '\n' });
            return i < 0 ? message : message.Substring(0, i);
        }
    }
}
