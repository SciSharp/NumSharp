using System.IO;

namespace NumSharp.IO
{
    /// <summary>
    ///     The one place NumSharp decides how a file it opens by PATH is shared with other handles: Python's <c>open()</c>
    ///     share mode, which every NumPy file function inherits (<c>np.load</c> / <c>np.save</c> / <c>np.savez</c> /
    ///     <c>np.loadtxt</c> / <c>np.savetxt</c> / <c>ndarray.tofile</c> / <c>np.fromfile</c>, and the memmap
    ///     <c>np.load(mmap_mode=...)</c> builds over its own file object).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         On Windows CPython's <c>open()</c> reaches <c>CreateFileW</c> through the C runtime's <c>_wopen</c> with
    ///         <c>_SH_DENYNO</c>, i.e. <c>FILE_SHARE_READ | FILE_SHARE_WRITE</c>: a file NumPy holds open admits other
    ///         readers AND writers, and NumPy opens a file another handle holds read-write. .NET's defaults are stricter
    ///         (<see cref="FileShare.Read"/> to read, <see cref="FileShare.None"/> to write), so a live <c>r+</c> memmap
    ///         locked out every other load of its file and the readers / writers refused a file an editor or a logger had
    ///         open (parity QA-39). <see cref="FileShare.ReadWrite"/> is the same share mode; Python asks for no
    ///         <c>FILE_SHARE_DELETE</c>, so neither does NumSharp.
    ///     </para>
    ///     <para>
    ///         Footgun, the same as NumPy's: sharing is not coordination. Two writers interleave, a reader can see a file
    ///         another handle is still writing, and the operating system still refuses some combinations by itself
    ///         (truncating a file that has a live mapping fails for NumPy too). A new file-opening site must take its
    ///         stream from here, so every path keeps one rule.
    ///     </para>
    /// </remarks>
    internal static class NDFileShare
    {
        /// <summary>Python's <c>open()</c> share mode on Windows: other handles may read and write the file.</summary>
        internal const FileShare Python = FileShare.ReadWrite;

        /// <summary>
        ///     Opens an existing file for reading, shared as Python shares it (<c>open(path, 'rb')</c>).
        /// </summary>
        /// <param name="path">The file's path.</param>
        /// <param name="bufferSize">The stream's buffer size in bytes (.NET's default 4,096 unless a reader asks for
        ///     more).</param>
        /// <returns>A read-only stream the caller owns and must dispose.</returns>
        /// <exception cref="FileNotFoundException">The file does not exist.</exception>
        /// <exception cref="IOException">The operating system refuses the open (a handle that does not share reading,
        ///     for example).</exception>
        /// <exception cref="UnauthorizedAccessException">Access is denied.</exception>
        internal static FileStream OpenRead(string path, int bufferSize = 4096)
            => new FileStream(path, FileMode.Open, FileAccess.Read, Python, bufferSize);

        /// <summary>
        ///     Creates or truncates a file for writing, shared as Python shares it (<c>open(path, 'wb')</c>).
        /// </summary>
        /// <param name="path">The file's path.</param>
        /// <returns>A write-only stream the caller owns and must dispose.</returns>
        /// <exception cref="IOException">The operating system refuses the open (another handle that does not share
        ///     writing, or a live mapping of the file, which truncation cannot pass).</exception>
        /// <exception cref="UnauthorizedAccessException">Access is denied.</exception>
        internal static FileStream Create(string path)
            => new FileStream(path, FileMode.Create, FileAccess.Write, Python);

        /// <summary>
        ///     Opens an existing file with the given access, shared as Python shares it: the memmap's handle
        ///     (<c>open(path, 'rb')</c> for <c>'r'</c> / <c>'c'</c>, <c>open(path, 'r+b')</c> for <c>'r+'</c>).
        /// </summary>
        /// <param name="path">The file's path.</param>
        /// <param name="access">Read, or read and write.</param>
        /// <returns>A stream the caller owns and must dispose (a memory-mapped file built over it may take it over).</returns>
        /// <exception cref="FileNotFoundException">The file does not exist.</exception>
        /// <exception cref="IOException">The operating system refuses the open.</exception>
        /// <exception cref="UnauthorizedAccessException">Access is denied (a read-only file opened for writing).</exception>
        internal static FileStream Open(string path, FileAccess access)
            => new FileStream(path, FileMode.Open, access, Python);
    }
}
