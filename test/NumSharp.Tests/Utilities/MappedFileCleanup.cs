using System.IO;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Best-effort deletion of a test's temporary directory whose files may still be memory-mapped by arrays or
    ///     views the test left to the garbage collector (<c>np.load(path, mmap_mode=…)</c> results).
    /// </summary>
    /// <remarks>
    ///     A mapped file stays open — and cannot be deleted on Windows — until its array and every view of it are
    ///     released. The views a test did not dispose are young garbage, so a young collection with a finalizer
    ///     drain (<see cref="GcQuiescence.CollectYoung"/>) releases them in well under a millisecond; the full
    ///     collection the callers used to force every time marks the whole test run's heap (tens of milliseconds
    ///     in-suite, see <see cref="GcQuiescence"/>) and is now paid only when the directory is still held after
    ///     the young pass — a view that a collection mid-test promoted to gen 2. Deletion never throws: a leftover
    ///     temp directory must not fail a test.
    /// </remarks>
    internal static class MappedFileCleanup
    {
        /// <summary>
        ///     Releases undisposed mappings and deletes <paramref name="dir"/> recursively, young collection first,
        ///     full collection only if the first attempt fails.
        /// </summary>
        /// <param name="dir">The temporary directory to remove (absent is fine).</param>
        public static void DeleteDirectory(string dir)
        {
            GcQuiescence.CollectYoung();
            if (TryDelete(dir))
                return;
            GcQuiescence.CollectFull();
            TryDelete(dir);
        }

        /// <summary>
        ///     Deletes <paramref name="dir"/> recursively, reporting instead of throwing when a file in it is still
        ///     open or mapped.
        /// </summary>
        /// <param name="dir">The directory to delete.</param>
        /// <returns>True when the directory is gone (deleted now, or already absent); false when deletion failed.</returns>
        public static bool TryDelete(string dir)
        {
            if (!Directory.Exists(dir))
                return true;
            try
            {
                Directory.Delete(dir, recursive: true);
                return true;
            }
            catch
            {
                // Sharing violation (IOException) or a still-mapped file (UnauthorizedAccessException): the caller
                // decides whether a full collection is worth one more attempt.
                return false;
            }
        }
    }
}
