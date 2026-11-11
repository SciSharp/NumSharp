using System;
using System.IO;

namespace NumSharp.IO
{
    /// <summary>
    ///     The one place NumSharp opens a file by PATH, as Python's <c>open()</c> does for every NumPy file function
    ///     (<c>np.load</c> / <c>np.save</c> / <c>np.savez</c> / <c>np.loadtxt</c> / <c>np.savetxt</c> /
    ///     <c>ndarray.tofile</c> / <c>np.fromfile</c>, and the memmap <c>np.load(mmap_mode=...)</c> builds over its own file
    ///     object): Python's share mode, and Python's errors when the open fails.
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
    ///         A failed open raises what CPython's <c>open()</c> raises, with its errno, strerror and the path as given:
    ///         a missing file or directory <see cref="FileNotFoundError"/> (<c>[Errno 2] No such file or directory:
    ///         'x.npy'</c>), a denied one <see cref="PermissionError"/> (<c>[Errno 13] Permission denied</c>, also a
    ///         directory on Windows and a file another process holds without sharing), a directory on Linux and macOS
    ///         <see cref="IsADirectoryError"/> (<c>[Errno 21] Is a directory</c>), an empty path a
    ///         <see cref="FileNotFoundError"/> and a NUL in the path a <see cref="ValueError"/> (<c>embedded null
    ///         character</c>). The .NET exception stays as the <see cref="Exception.InnerException"/>; any other .NET
    ///         failure (a path too long, a broken device) is rethrown as it is.
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
        /// <exception cref="FileNotFoundError">The file, or a directory on its path, does not exist; the path is
        ///     empty.</exception>
        /// <exception cref="PermissionError">Access is denied, the file is held without sharing, or (Windows) the path is a
        ///     directory.</exception>
        /// <exception cref="IsADirectoryError">The path is a directory (Linux and macOS).</exception>
        /// <exception cref="ValueError">The path holds a NUL (<c>embedded null character</c>).</exception>
        /// <exception cref="IOException">Any other refusal of the operating system, as .NET reports it.</exception>
        internal static FileStream OpenRead(string path, int bufferSize = 4096)
        {
            try
            {
                return new FileStream(path, FileMode.Open, FileAccess.Read, Python, bufferSize);
            }
            catch (Exception e)
            {
                // Rethrown as is (stack trace kept) when CPython would not name the failure.
                Exception py = OpenError(e, path);
                if (py is null)
                    throw;
                throw py;
            }
        }

        /// <summary>
        ///     Creates or truncates a file for writing, shared as Python shares it (<c>open(path, 'wb')</c>).
        /// </summary>
        /// <param name="path">The file's path.</param>
        /// <returns>A write-only stream the caller owns and must dispose.</returns>
        /// <exception cref="FileNotFoundError">A directory on the path does not exist; the path is empty.</exception>
        /// <exception cref="PermissionError">Access is denied (a read-only file), the file is held without sharing, or
        ///     (Windows) the path is a directory.</exception>
        /// <exception cref="IsADirectoryError">The path is a directory (Linux and macOS).</exception>
        /// <exception cref="ValueError">The path holds a NUL (<c>embedded null character</c>).</exception>
        /// <exception cref="IOException">Any other refusal (a live mapping of the file, which truncation cannot pass), as
        ///     .NET reports it.</exception>
        internal static FileStream Create(string path)
        {
            try
            {
                return new FileStream(path, FileMode.Create, FileAccess.Write, Python);
            }
            catch (Exception e)
            {
                Exception py = OpenError(e, path);
                if (py is null)
                    throw;
                throw py;
            }
        }

        /// <summary>
        ///     Opens an existing file with the given access, shared as Python shares it: the memmap's handle
        ///     (<c>open(path, 'rb')</c> for <c>'r'</c> / <c>'c'</c>, <c>open(path, 'r+b')</c> for <c>'r+'</c>).
        /// </summary>
        /// <param name="path">The file's path.</param>
        /// <param name="access">Read, or read and write.</param>
        /// <returns>A stream the caller owns and must dispose (a memory-mapped file built over it may take it over).</returns>
        /// <exception cref="FileNotFoundError">The file, or a directory on its path, does not exist; the path is
        ///     empty.</exception>
        /// <exception cref="PermissionError">Access is denied (a read-only file opened for writing), the file is held
        ///     without sharing, or (Windows) the path is a directory.</exception>
        /// <exception cref="IsADirectoryError">The path is a directory (Linux and macOS).</exception>
        /// <exception cref="ValueError">The path holds a NUL (<c>embedded null character</c>).</exception>
        /// <exception cref="IOException">Any other refusal of the operating system, as .NET reports it.</exception>
        internal static FileStream Open(string path, FileAccess access)
        {
            try
            {
                return new FileStream(path, FileMode.Open, access, Python);
            }
            catch (Exception e)
            {
                Exception py = OpenError(e, path);
                if (py is null)
                    throw;
                throw py;
            }
        }

        /// <summary>
        ///     What CPython's <c>open(path)</c> raises for the failure .NET reported as <paramref name="e"/>, or null when
        ///     CPython has no specific error for it (the caller rethrows .NET's).
        /// </summary>
        /// <param name="e">.NET's exception from the open.</param>
        /// <param name="path">The path as the caller gave it (CPython's <c>filename</c> is never resolved).</param>
        /// <returns>The Python error, its <see cref="Exception.InnerException"/> <paramref name="e"/>; or null.</returns>
        internal static Exception OpenError(Exception e, string path)
        {
            switch (e)
            {
                // ENOENT for the file and for a missing directory on its path alike (both are FileNotFoundError).
                case FileNotFoundException:
                case DirectoryNotFoundException:
                    return Errno(2, "No such file or directory", path, e);
                // .NET reports a directory opened as a file, a read-only file opened for writing and a denied ACL all as
                // UnauthorizedAccessException; CPython tells the directory apart off Windows (EISDIR), while the Windows
                // CRT maps all of them to EACCES.
                case UnauthorizedAccessException:
                    if (!OperatingSystem.IsWindows() && path != null && Directory.Exists(path))
                        return Errno(21, "Is a directory", path, e);
                    return Errno(13, "Permission denied", path, e);
                // A sharing or lock violation (Win32 32 / 33 in a FACILITY_WIN32 HRESULT): the CRT's EACCES.
                case IOException io when OperatingSystem.IsWindows()
                                         && (io.HResult & unchecked((int)0xFFFF0000)) == unchecked((int)0x80070000)
                                         && (io.HResult & 0xFFFF) is 32 or 33:
                    return Errno(13, "Permission denied", path, e);
                // .NET refuses these before the operating system sees them; CPython's open() reports an empty name as
                // ENOENT and a NUL as ValueError.
                case ArgumentException when path != null && path.Length == 0:
                    return Errno(2, "No such file or directory", path, e);
                case ArgumentException when path != null && path.IndexOf('\0') >= 0:
                    return new ValueError("embedded null character", e);
                default:
                    return null;
            }
        }

        /// <summary>
        ///     CPython's <c>PyErr_SetFromErrnoWithFilenameObject</c>: <c>OSError(errno, strerror, filename)</c>, whose
        ///     class the errno picks (<see cref="OSError.Create(int?, string, string)"/>), with the .NET exception as its
        ///     cause.
        /// </summary>
        /// <param name="errno">The errno (the same number on Windows, Linux and macOS for 2, 13 and 21).</param>
        /// <param name="strerror">The C runtime's text for it (the same on all three).</param>
        /// <param name="path">The file name.</param>
        /// <param name="inner">.NET's exception.</param>
        /// <returns>The error.</returns>
        private static OSError Errno(int errno, string strerror, string path, Exception inner)
            => OSError.Create(OSErrorArgs.Parse(3, errno, strerror, path, null, null, inner));
    }
}
