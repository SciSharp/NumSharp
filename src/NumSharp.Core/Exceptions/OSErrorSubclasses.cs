using System;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>FileNotFoundError</c> (errno <c>ENOENT</c>, 2): a file or directory that does not exist. NumSharp's
    ///     file functions raise it as CPython's <c>open()</c> does: <c>np.load</c> / <c>np.fromfile</c> / <c>np.save</c> into
    ///     a missing directory read <c>[Errno 2] No such file or directory: 'x.npy'</c>, and <c>np.loadtxt</c>'s
    ///     <c>DataSource</c> raises <c>x.txt not found.</c> (its own text, without an errno).
    /// </summary>
    /// <remarks>
    ///     A subclass of <see cref="OSError"/> and so of .NET's <see cref="System.IO.IOException"/>. Footgun: it is NOT .NET's
    ///     <see cref="System.IO.FileNotFoundException"/>, which a <c>catch (FileNotFoundException)</c> written for older
    ///     NumSharp expects; catch <see cref="FileNotFoundError"/>, or <see cref="System.IO.IOException"/> for both. The
    ///     .NET exception behind it is its <see cref="Exception.InnerException"/> (with the full path .NET resolved).
    /// </remarks>
    public class FileNotFoundError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public FileNotFoundError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public FileNotFoundError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public FileNotFoundError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public FileNotFoundError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public FileNotFoundError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public FileNotFoundError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public FileNotFoundError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> and NumSharp's raise sites use.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal FileNotFoundError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>PermissionError</c> (errno <c>EACCES</c> 13, <c>EPERM</c> 1): an operation the operating system
    ///     denies. NumSharp's file functions raise it as CPython's <c>open()</c> does: a read-only or locked file opened for
    ///     writing, a file another process holds without sharing, and on Windows a directory opened as a file
    ///     (<c>[Errno 13] Permission denied: 'data'</c>).
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>; footgun: NOT .NET's <see cref="UnauthorizedAccessException"/>, which is
    ///     not an <see cref="System.IO.IOException"/> either. The .NET exception behind it is its
    ///     <see cref="Exception.InnerException"/>.</remarks>
    public class PermissionError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public PermissionError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public PermissionError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public PermissionError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public PermissionError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public PermissionError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public PermissionError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public PermissionError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> and NumSharp's raise sites use.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal PermissionError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>IsADirectoryError</c> (errno <c>EISDIR</c>, 21): a file operation on a directory. NumSharp's file
    ///     functions raise it as CPython's <c>open()</c> does on Linux and macOS (<c>[Errno 21] Is a directory: 'data'</c>);
    ///     on Windows the same open is a <see cref="PermissionError"/>, as in CPython.
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>.</remarks>
    public class IsADirectoryError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public IsADirectoryError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public IsADirectoryError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public IsADirectoryError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public IsADirectoryError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public IsADirectoryError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public IsADirectoryError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public IsADirectoryError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> and NumSharp's raise sites use.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal IsADirectoryError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>NotADirectoryError</c> (errno <c>ENOTDIR</c>, 20): a directory operation on something that is not a
    ///     directory (a path that runs through a file: <c>data.npy/x</c>).
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>. NumSharp does not raise it itself; <see cref="OSError.Create(int?, string)"/>
    ///     builds it for errno 20.</remarks>
    public class NotADirectoryError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public NotADirectoryError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public NotADirectoryError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public NotADirectoryError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public NotADirectoryError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public NotADirectoryError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public NotADirectoryError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public NotADirectoryError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal NotADirectoryError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>FileExistsError</c> (errno <c>EEXIST</c>, 17): creating a file or directory that already exists
    ///     (<c>open(path, 'x')</c>, <c>os.mkdir</c>).
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>. NumSharp does not raise it itself (its writers truncate, as NumPy's
    ///     do); <see cref="OSError.Create(int?, string)"/> builds it for errno 17.</remarks>
    public class FileExistsError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public FileExistsError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public FileExistsError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public FileExistsError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public FileExistsError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public FileExistsError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public FileExistsError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public FileExistsError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal FileExistsError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>BlockingIOError</c> (errno <c>EAGAIN</c>, <c>EALREADY</c>, <c>EWOULDBLOCK</c>,
    ///     <c>EINPROGRESS</c>): an operation on a non-blocking object that would have blocked. A buffered write reports how
    ///     much it wrote first in <see cref="OSError.characters_written"/>.
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>. NumSharp does not raise it itself; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class BlockingIOError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public BlockingIOError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public BlockingIOError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public BlockingIOError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public BlockingIOError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public BlockingIOError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <summary>
        ///     Python's <c>BlockingIOError(errno, strerror, characters_written)</c>: a numeric third argument is the count
        ///     of characters written before the call would have blocked, not a file name; the message is
        ///     <c>[Errno 11] strerror</c>.
        /// </summary>
        /// <param name="errno">The error number, or null for <c>None</c>.</param>
        /// <param name="strerror">Its text, or null for <c>None</c>.</param>
        /// <param name="characters_written">The characters written (Python's <c>characters_written</c>).</param>
        public BlockingIOError(int? errno, string strerror, long characters_written)
            : base(OSErrorArgs.Parse(3, errno, strerror, null, null, null, null, characters_written)) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public BlockingIOError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public BlockingIOError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal BlockingIOError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>ChildProcessError</c> (errno <c>ECHILD</c>, 10): an operation on a child process that does not
    ///     exist (<c>os.waitpid</c> with no children).
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>. NumSharp does not raise it itself; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class ChildProcessError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public ChildProcessError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public ChildProcessError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public ChildProcessError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public ChildProcessError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public ChildProcessError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public ChildProcessError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public ChildProcessError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal ChildProcessError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>ConnectionError</c>: the base of the connection failures, <see cref="BrokenPipeError"/>,
    ///     <see cref="ConnectionAbortedError"/>, <see cref="ConnectionRefusedError"/> and
    ///     <see cref="ConnectionResetError"/>. No errno names it directly.
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>, so <c>catch (ConnectionError)</c> catches its four subclasses, as
    ///     <c>except ConnectionError</c> does. NumSharp does not raise it; the family exists for ports.</remarks>
    public class ConnectionError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public ConnectionError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public ConnectionError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public ConnectionError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public ConnectionError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public ConnectionError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public ConnectionError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public ConnectionError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form the subclasses pass on.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal ConnectionError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>BrokenPipeError</c> (errno <c>EPIPE</c>, <c>ESHUTDOWN</c>): a write to a pipe whose reader is gone,
    ///     or to a socket shut down for writing.
    /// </summary>
    /// <remarks>A subclass of <see cref="ConnectionError"/>. NumSharp does not raise it; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class BrokenPipeError : ConnectionError
    {
        /// <inheritdoc cref="OSError()"/>
        public BrokenPipeError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public BrokenPipeError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public BrokenPipeError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public BrokenPipeError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public BrokenPipeError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public BrokenPipeError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public BrokenPipeError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal BrokenPipeError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>ConnectionAbortedError</c> (errno <c>ECONNABORTED</c>): a connection the peer (or the local stack)
    ///     aborted.
    /// </summary>
    /// <remarks>A subclass of <see cref="ConnectionError"/>. NumSharp does not raise it; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class ConnectionAbortedError : ConnectionError
    {
        /// <inheritdoc cref="OSError()"/>
        public ConnectionAbortedError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public ConnectionAbortedError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public ConnectionAbortedError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public ConnectionAbortedError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public ConnectionAbortedError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public ConnectionAbortedError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public ConnectionAbortedError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal ConnectionAbortedError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>ConnectionRefusedError</c> (errno <c>ECONNREFUSED</c>): a connection the peer refused.
    /// </summary>
    /// <remarks>A subclass of <see cref="ConnectionError"/>. NumSharp does not raise it; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class ConnectionRefusedError : ConnectionError
    {
        /// <inheritdoc cref="OSError()"/>
        public ConnectionRefusedError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public ConnectionRefusedError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public ConnectionRefusedError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public ConnectionRefusedError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public ConnectionRefusedError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public ConnectionRefusedError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public ConnectionRefusedError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal ConnectionRefusedError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>ConnectionResetError</c> (errno <c>ECONNRESET</c>): a connection the peer reset.
    /// </summary>
    /// <remarks>A subclass of <see cref="ConnectionError"/>. NumSharp does not raise it; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class ConnectionResetError : ConnectionError
    {
        /// <inheritdoc cref="OSError()"/>
        public ConnectionResetError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public ConnectionResetError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public ConnectionResetError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public ConnectionResetError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public ConnectionResetError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public ConnectionResetError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public ConnectionResetError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal ConnectionResetError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>InterruptedError</c> (errno <c>EINTR</c>, 4): a system call a signal interrupted (CPython retries
    ///     most of them itself since PEP 475).
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>. NumSharp does not raise it; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class InterruptedError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public InterruptedError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public InterruptedError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public InterruptedError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public InterruptedError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public InterruptedError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public InterruptedError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public InterruptedError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal InterruptedError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>ProcessLookupError</c> (errno <c>ESRCH</c>, 3): a process that does not exist (<c>os.kill</c> of a
    ///     finished process).
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>. NumSharp does not raise it; the class exists for ports and for
    ///     <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class ProcessLookupError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public ProcessLookupError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public ProcessLookupError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public ProcessLookupError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public ProcessLookupError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public ProcessLookupError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public ProcessLookupError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public ProcessLookupError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal ProcessLookupError(OSErrorArgs args) : base(args) { }
    }

    /// <summary>
    ///     Python's <c>TimeoutError</c> (errno <c>ETIMEDOUT</c>): a system-level timeout (a socket operation, a lock).
    /// </summary>
    /// <remarks>A subclass of <see cref="OSError"/>, as in Python; footgun: NOT .NET's <see cref="TimeoutException"/>.
    ///     NumSharp does not raise it; the class exists for ports and for <see cref="OSError.Create(int?, string)"/>.</remarks>
    public class TimeoutError : OSError
    {
        /// <inheritdoc cref="OSError()"/>
        public TimeoutError() : base() { }

        /// <inheritdoc cref="OSError(string)"/>
        public TimeoutError(string message) : base(message) { }

        /// <inheritdoc cref="OSError(string, Exception)"/>
        public TimeoutError(string message, Exception innerException) : base(message, innerException) { }

        /// <inheritdoc cref="OSError(int?, string)"/>
        public TimeoutError(int? errno, string strerror) : base(errno, strerror) { }

        /// <inheritdoc cref="OSError(int?, string, string)"/>
        public TimeoutError(int? errno, string strerror, string filename) : base(errno, strerror, filename) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?)"/>
        public TimeoutError(int? errno, string strerror, string filename, int? winerror)
            : base(errno, strerror, filename, winerror) { }

        /// <inheritdoc cref="OSError(int?, string, string, int?, string)"/>
        public TimeoutError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : base(errno, strerror, filename, winerror, filename2) { }

        /// <summary>The parsed form <see cref="OSError.Create(int?, string)"/> uses.</summary>
        /// <param name="args">The parsed arguments.</param>
        internal TimeoutError(OSErrorArgs args) : base(args) { }
    }
}
