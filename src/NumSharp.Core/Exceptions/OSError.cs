using System;
using System.Globalization;
using System.IO;
using NumSharp.Backends.Printing;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible <c>OSError</c>: an operating-system level failure — a file that is missing or cannot be opened
    ///     (raised as one of its subclasses, <see cref="FileNotFoundError"/>, <see cref="PermissionError"/>, …), and the
    ///     file operations NumPy refuses itself (<c>ndarray.tofile</c>'s <c>cannot write object arrays to a file in binary
    ///     mode</c> for a <c>StringDType</c> array, <c>problem writing element 3 to file</c>). NumSharp raises it in the same
    ///     places with NumPy's (and CPython's) text.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Unlike most Python-named errors (<see cref="TypeError"/>, <see cref="SystemError"/>, which derive from
    ///     <see cref="NumSharpException"/>), it derives from <see cref="IOException"/>: Python's <c>OSError</c> IS the I/O
    ///     error class (<c>IOError</c> is its alias), so a .NET caller that catches <see cref="IOException"/> around a file
    ///     operation catches it and every subclass — the same reason <see cref="ValueError"/> derives from
    ///     <see cref="ArgumentException"/>.
    ///     </para>
    ///     <para>
    ///     Like Python's, it carries <see cref="errno"/>, <see cref="strerror"/>, <see cref="filename"/>,
    ///     <see cref="filename2"/> and, on Windows, <see cref="winerror"/>, and its text is CPython's: <c>[Errno 2] No such
    ///     file or directory: 'x.npy'</c>. NumSharp's file functions open their files through one place that turns .NET's
    ///     open failures into these errors as CPython's <c>open()</c> reports them (a missing file is a
    ///     <see cref="FileNotFoundError"/>, a denied one a <see cref="PermissionError"/>); other .NET I/O failures (a full
    ///     disk, a broken device) keep their own .NET exception types.
    ///     </para>
    ///     <para>
    ///     Python's <c>OSError(errno, strerror)</c> returns the subclass its errno names (<c>OSError(2, ...)</c> is a
    ///     <see cref="FileNotFoundError"/>); a C# constructor cannot return a subclass, so <c>new OSError(2, ...)</c> builds
    ///     this class with that errno, and <see cref="Create(int?, string)"/> (and its overloads) is Python's call. The errno
    ///     numbers are the operating system's own (CPython's <c>errno</c> module), so the same number can name different
    ///     subclasses on Windows, Linux and macOS.
    ///     </para>
    /// </remarks>
    public class OSError : IOException, INumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise OSError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(OSError())</c> is <c>''</c> (not .NET's
        ///     <see cref="IOException"/> default, "I/O error occurred."); every attribute is null.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>OSError</c>; prefer a message that names the file and the operation.</remarks>
        public OSError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's message text; every attribute is null, as for Python's
        ///     <c>OSError('message')</c>.</summary>
        /// <param name="message">The verbatim message (<c>cannot write object arrays to a file in binary mode</c>).</param>
        public OSError(string message) : base(message) { }

        /// <summary>Creates the error with NumPy's message text and the exception that caused it; every attribute is
        ///     null.</summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The cause (the .NET I/O exception behind a failed write).</param>
        public OSError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>
        ///     Python's <c>OSError(errno, strerror)</c>: the message is CPython's <c>[Errno 2] No such file or
        ///     directory</c>.
        /// </summary>
        /// <param name="errno">The operating system's error number, or null for <c>None</c>.</param>
        /// <param name="strerror">Its text (<c>strerror(errno)</c>), or null for <c>None</c>.</param>
        /// <remarks>Builds THIS class whatever the errno; <see cref="Create(int?, string)"/> picks the subclass the errno
        ///     names, as Python's call does.</remarks>
        public OSError(int? errno, string strerror)
            : this(OSErrorArgs.Parse(2, errno, strerror, null, null, null)) { }

        /// <summary>
        ///     Python's <c>OSError(errno, strerror, filename)</c>: the message is CPython's <c>[Errno 2] No such file or
        ///     directory: 'x.npy'</c>, the file name as Python's <c>repr()</c> quotes it.
        /// </summary>
        /// <param name="errno">The operating system's error number, or null for <c>None</c>.</param>
        /// <param name="strerror">Its text, or null for <c>None</c>.</param>
        /// <param name="filename">The file the operation was on, or null for none (the message then omits it).</param>
        public OSError(int? errno, string strerror, string filename)
            : this(OSErrorArgs.Parse(3, errno, strerror, filename, null, null)) { }

        /// <summary>
        ///     Python's <c>OSError(errno, strerror, filename, winerror)</c>: on Windows, <paramref name="winerror"/> (a Windows
        ///     error code, or an HRESULT carrying one) decides <see cref="errno"/> through CPython's
        ///     <c>winerror_to_errno</c>, and the message reads <c>[WinError 5] Access is denied: 'f'</c>.
        /// </summary>
        /// <param name="errno">The error number, or null for <c>None</c>; replaced on Windows when
        ///     <paramref name="winerror"/> is given.</param>
        /// <param name="strerror">Its text, or null for <c>None</c>.</param>
        /// <param name="filename">The file the operation was on, or null for none.</param>
        /// <param name="winerror">The Windows error code, or null for <c>None</c>.</param>
        /// <remarks>
        ///     Footgun, CPython's own: on Windows a four- or five-argument call prints <c>[WinError None]</c> even when
        ///     <paramref name="winerror"/> is null; elsewhere <paramref name="winerror"/> is ignored (CPython parses it and
        ///     drops it) and <see cref="winerror"/> stays null.
        /// </remarks>
        public OSError(int? errno, string strerror, string filename, int? winerror)
            : this(OSErrorArgs.Parse(4, errno, strerror, filename, winerror, null)) { }

        /// <summary>
        ///     Python's <c>OSError(errno, strerror, filename, winerror, filename2)</c>: an operation on two files, the message
        ///     naming both (<c>[Errno 17] File exists: 'a' -&gt; 'b'</c>; on Windows <c>[WinError …]</c>).
        /// </summary>
        /// <param name="errno">The error number, or null for <c>None</c>.</param>
        /// <param name="strerror">Its text, or null for <c>None</c>.</param>
        /// <param name="filename">The first file, or null for none (the second is then dropped, as in CPython).</param>
        /// <param name="winerror">The Windows error code, or null for <c>None</c> (see the four-argument form).</param>
        /// <param name="filename2">The second file, or null for none.</param>
        public OSError(int? errno, string strerror, string filename, int? winerror, string filename2)
            : this(OSErrorArgs.Parse(5, errno, strerror, filename, winerror, filename2)) { }

        /// <summary>
        ///     The parsed form every errno constructor and <see cref="Create(int?, string)"/> end in (and NumSharp's own raise
        ///     sites, which also keep the .NET exception behind the failure as <see cref="Exception.InnerException"/>).
        /// </summary>
        /// <param name="args">The parsed arguments, the message already formatted.</param>
        internal OSError(OSErrorArgs args) : base(args.Text, args.Inner)
        {
            errno = args.Errno;
            strerror = args.Strerror;
            filename = args.Filename;
            filename2 = args.Filename2;
            winerror = args.Winerror;
            characters_written = args.CharactersWritten;
        }

        /// <summary>
        ///     The operating system's error number (Python's <c>errno</c>; on Windows the one <see cref="winerror"/> maps
        ///     to), or null when the error was created without one.
        /// </summary>
        public int? errno { get; }

        /// <summary>The error number's text (Python's <c>strerror</c>: <c>No such file or directory</c>), or null.</summary>
        public string strerror { get; }

        /// <summary>The file the operation was on (Python's <c>filename</c>), as given, or null.</summary>
        public string filename { get; }

        /// <summary>The second file of a two-file operation (Python's <c>filename2</c>), or null; never set without
        ///     <see cref="filename"/>.</summary>
        public string filename2 { get; }

        /// <summary>
        ///     The Windows error code (Python's <c>winerror</c>, which exists on Windows only): the four- or five-argument
        ///     constructors' value on Windows, null otherwise and on every other operating system.
        /// </summary>
        public int? winerror { get; }

        /// <summary>
        ///     The characters a non-blocking write wrote before it would have blocked (Python's
        ///     <c>BlockingIOError.characters_written</c>), set only by
        ///     <see cref="BlockingIOError(int?, string, long)"/>; null otherwise (Python raises <c>AttributeError</c> there).
        /// </summary>
        public long? characters_written { get; }

        /// <summary>
        ///     Python's <c>OSError(errno, strerror)</c> call: the subclass <paramref name="errno"/> names on this operating
        ///     system (<c>2</c>: <see cref="FileNotFoundError"/>), or <see cref="OSError"/> itself.
        /// </summary>
        /// <param name="errno">The error number, or null for <c>None</c> (no subclass).</param>
        /// <param name="strerror">Its text, or null.</param>
        /// <returns>The new error, of the class CPython would build.</returns>
        public static OSError Create(int? errno, string strerror)
            => Create(OSErrorArgs.Parse(2, errno, strerror, null, null, null));

        /// <summary>Python's <c>OSError(errno, strerror, filename)</c> call: the subclass the errno names.</summary>
        /// <param name="errno">The error number, or null.</param>
        /// <param name="strerror">Its text, or null.</param>
        /// <param name="filename">The file, or null.</param>
        /// <returns>The new error, of the class CPython would build.</returns>
        public static OSError Create(int? errno, string strerror, string filename)
            => Create(OSErrorArgs.Parse(3, errno, strerror, filename, null, null));

        /// <summary>
        ///     Python's <c>OSError(errno, strerror, filename, winerror)</c> call: on Windows the subclass the errno
        ///     <paramref name="winerror"/> maps to names (<c>winerror=5</c>: <see cref="PermissionError"/>).
        /// </summary>
        /// <param name="errno">The error number, or null.</param>
        /// <param name="strerror">Its text, or null.</param>
        /// <param name="filename">The file, or null.</param>
        /// <param name="winerror">The Windows error code, or null (ignored off Windows).</param>
        /// <returns>The new error, of the class CPython would build.</returns>
        public static OSError Create(int? errno, string strerror, string filename, int? winerror)
            => Create(OSErrorArgs.Parse(4, errno, strerror, filename, winerror, null));

        /// <summary>Python's <c>OSError(errno, strerror, filename, winerror, filename2)</c> call.</summary>
        /// <param name="errno">The error number, or null.</param>
        /// <param name="strerror">Its text, or null.</param>
        /// <param name="filename">The first file, or null.</param>
        /// <param name="winerror">The Windows error code, or null (ignored off Windows).</param>
        /// <param name="filename2">The second file, or null.</param>
        /// <returns>The new error, of the class CPython would build.</returns>
        public static OSError Create(int? errno, string strerror, string filename, int? winerror, string filename2)
            => Create(OSErrorArgs.Parse(5, errno, strerror, filename, winerror, filename2));

        /// <summary>
        ///     CPython's <c>OSError_new</c> class choice over parsed arguments: its <c>errnomap</c>, built from the operating
        ///     system's errno values.
        /// </summary>
        /// <param name="a">The parsed arguments.</param>
        /// <returns>The error, of the subclass the errno names or of <see cref="OSError"/>.</returns>
        internal static OSError Create(OSErrorArgs a)
        {
            switch (a.Errno is int e ? ErrnoMap.ClassOf(e) : ErrnoMap.Kind.None)
            {
                case ErrnoMap.Kind.BlockingIO: return new BlockingIOError(a);
                case ErrnoMap.Kind.BrokenPipe: return new BrokenPipeError(a);
                case ErrnoMap.Kind.ChildProcess: return new ChildProcessError(a);
                case ErrnoMap.Kind.ConnectionAborted: return new ConnectionAbortedError(a);
                case ErrnoMap.Kind.ConnectionRefused: return new ConnectionRefusedError(a);
                case ErrnoMap.Kind.ConnectionReset: return new ConnectionResetError(a);
                case ErrnoMap.Kind.FileExists: return new FileExistsError(a);
                case ErrnoMap.Kind.FileNotFound: return new FileNotFoundError(a);
                case ErrnoMap.Kind.IsADirectory: return new IsADirectoryError(a);
                case ErrnoMap.Kind.NotADirectory: return new NotADirectoryError(a);
                case ErrnoMap.Kind.Interrupted: return new InterruptedError(a);
                case ErrnoMap.Kind.Permission: return new PermissionError(a);
                case ErrnoMap.Kind.ProcessLookup: return new ProcessLookupError(a);
                case ErrnoMap.Kind.Timeout: return new TimeoutError(a);
                default: return new OSError(a);
            }
        }
    }

    /// <summary>
    ///     CPython's parsed <c>OSError</c> arguments (<c>oserror_parse_args</c> / <c>oserror_init</c>) and the message its
    ///     <c>OSError.__str__</c> builds from them, for the operating system NumSharp runs on.
    /// </summary>
    internal readonly struct OSErrorArgs
    {
        /// <summary>The error number after the Windows winerror mapping, or null.</summary>
        internal readonly int? Errno;

        /// <summary>The error text, or null.</summary>
        internal readonly string Strerror;

        /// <summary>The first file, or null.</summary>
        internal readonly string Filename;

        /// <summary>The second file (only kept beside a first one), or null.</summary>
        internal readonly string Filename2;

        /// <summary>The Windows error code (Windows only, four or more arguments), or null.</summary>
        internal readonly int? Winerror;

        /// <summary><see cref="BlockingIOError"/>'s written-character count, or null.</summary>
        internal readonly long? CharactersWritten;

        /// <summary>The .NET exception behind the failure, or null.</summary>
        internal readonly Exception Inner;

        /// <summary>CPython's <c>str()</c> of the error.</summary>
        internal readonly string Text;

        /// <summary>Stores the parsed values.</summary>
        /// <param name="errno">The resolved error number.</param>
        /// <param name="strerror">The error text.</param>
        /// <param name="filename">The first file.</param>
        /// <param name="filename2">The second file.</param>
        /// <param name="winerror">The Windows error code.</param>
        /// <param name="charactersWritten">The written-character count.</param>
        /// <param name="inner">The .NET cause.</param>
        /// <param name="text">The message.</param>
        private OSErrorArgs(int? errno, string strerror, string filename, string filename2, int? winerror,
            long? charactersWritten, Exception inner, string text)
        {
            Errno = errno;
            Strerror = strerror;
            Filename = filename;
            Filename2 = filename2;
            Winerror = winerror;
            CharactersWritten = charactersWritten;
            Inner = inner;
            Text = text;
        }

        /// <summary>
        ///     CPython's parsing of <c>OSError(*args)</c> with 2 to 5 arguments, and its <c>OSError_str</c>.
        /// </summary>
        /// <param name="count">How many arguments Python's call passed (2 to 5): on Windows four or more print
        ///     <c>[WinError …]</c>.</param>
        /// <param name="errno">The error number, or null.</param>
        /// <param name="strerror">The error text, or null.</param>
        /// <param name="filename">The first file, or null.</param>
        /// <param name="winerror">The Windows error code, or null.</param>
        /// <param name="filename2">The second file, or null.</param>
        /// <param name="inner">The .NET exception behind the failure, or null.</param>
        /// <param name="charactersWritten"><see cref="BlockingIOError"/>'s count (its numeric third argument), or null.</param>
        /// <returns>The parsed arguments.</returns>
        internal static OSErrorArgs Parse(int count, int? errno, string strerror, string filename, int? winerror,
            string filename2, Exception inner = null, long? charactersWritten = null)
        {
            bool windows = OperatingSystem.IsWindows();
            // CPython keeps winerror only on Windows (elsewhere it parses and drops it), and there a given int decides the
            // errno through winerror_to_errno, overriding the errno argument.
            bool winerrorGiven = windows && count >= 4;
            int? win = winerrorGiven ? winerror : null;
            if (win is int w)
                errno = WinErrorToErrno(w);
            // filename2 is kept only beside a filename (oserror_init sets it inside the filename branch).
            if (filename is null)
                filename2 = null;

            string e = errno is int n ? n.ToString(CultureInfo.InvariantCulture) : "None";
            string s = strerror ?? "None";
            string text;
            if (winerrorGiven && filename != null)
                text = $"[WinError {(win is int wv ? wv.ToString(CultureInfo.InvariantCulture) : "None")}] {s}: {PyRepr.Str(filename)}" +
                       (filename2 != null ? " -> " + PyRepr.Str(filename2) : string.Empty);
            else if (winerrorGiven)
                text = $"[WinError {(win is int wv ? wv.ToString(CultureInfo.InvariantCulture) : "None")}] {s}";
            else if (filename != null)
                text = $"[Errno {e}] {s}: {PyRepr.Str(filename)}" + (filename2 != null ? " -> " + PyRepr.Str(filename2) : string.Empty);
            else
                text = $"[Errno {e}] {s}"; // two or more arguments always set errno and strerror (None prints "None")
            return new OSErrorArgs(errno, strerror, filename, filename2, win, charactersWritten, inner, text);
        }

        /// <summary>
        ///     CPython's <c>winerror_to_errno</c> (<c>PC/errmap.h</c>): the errno a Windows error code maps to. An HRESULT that
        ///     carries a Win32 code (<c>0x8007xxxx</c>) is unwrapped first; Winsock codes (10000-11999) are errno values
        ///     themselves, except the six Winsock spellings of C errno values; everything unlisted is <c>EINVAL</c> (22).
        /// </summary>
        /// <param name="winerror">The Windows error code or HRESULT.</param>
        /// <returns>The errno (Windows CRT numbering).</returns>
        internal static int WinErrorToErrno(int winerror)
        {
            // Unwrap FACILITY_WIN32 HRESULTs (.NET's IOException.HResult values are of this form).
            if ((winerror & unchecked((int)0xFFFF0000)) == unchecked((int)0x80070000))
                winerror &= 0xFFFF;
            if (winerror >= 10000 && winerror < 12000)
            {
                switch (winerror)
                {
                    case 10004: // WSAEINTR
                    case 10009: // WSAEBADF
                    case 10013: // WSAEACCES
                    case 10014: // WSAEFAULT
                    case 10022: // WSAEINVAL
                    case 10024: // WSAEMFILE
                        return winerror - 10000;
                    default:
                        return winerror;
                }
            }
            switch (winerror)
            {
                case 2: case 3: case 15: case 18: case 53: case 67: case 161: case 206:
                    return 2; // ENOENT: file / path not found, invalid drive, no more files, bad net path or name
                case 10:
                    return 7; // E2BIG: bad environment
                case 11: case 188: case 189: case 190: case 191: case 192: case 193: case 194: case 195: case 196:
                case 197: case 198: case 199: case 200: case 201: case 202:
                    return 8; // ENOEXEC: bad executable format
                case 6: case 114: case 130:
                    return 9; // EBADF: invalid handle
                case 128: case 129:
                    return 10; // ECHILD
                case 89: case 164: case 215:
                    return 11; // EAGAIN: no process slots, too many threads, nesting not allowed
                case 7: case 8: case 9: case 1816:
                    return 12; // ENOMEM
                case 5: case 16: case 19: case 20: case 21: case 22: case 23: case 24: case 25: case 26: case 27: case 28:
                case 29: case 30: case 31: case 32: case 33: case 34: case 35: case 36: case 65: case 82: case 83: case 108:
                case 132: case 158: case 167:
                    return 13; // EACCES: access denied, sharing / lock violations, device errors
                case 80: case 183:
                    return 17; // EEXIST
                case 17:
                    return 18; // EXDEV: not the same device
                case 267:
                    return 20; // ENOTDIR: ERROR_DIRECTORY
                case 4:
                    return 24; // EMFILE: too many open files
                case 112:
                    return 28; // ENOSPC: disk full
                case 109: case 232:
                    return 32; // EPIPE: broken pipe, no data
                case 145:
                    return 41; // ENOTEMPTY: directory not empty
                case 1113:
                    return 42; // EILSEQ: no Unicode translation
                default:
                    return 22; // EINVAL
            }
        }
    }

    /// <summary>
    ///     CPython's <c>errnomap</c> (<c>Objects/exceptions.c</c>): which <see cref="OSError"/> subclass an errno value
    ///     names, with the operating system's errno numbering (Windows: the CRT's numbers, with CPython's Winsock values
    ///     for the socket errors; Linux and Android: glibc's; macOS, iOS and FreeBSD: BSD's).
    /// </summary>
    internal static class ErrnoMap
    {
        /// <summary>The subclass an errno names.</summary>
        internal enum Kind
        {
            /// <summary>No subclass: <see cref="OSError"/> itself.</summary>
            None,
            /// <summary><see cref="BlockingIOError"/> (EAGAIN, EALREADY, EWOULDBLOCK, EINPROGRESS).</summary>
            BlockingIO,
            /// <summary><see cref="BrokenPipeError"/> (EPIPE, ESHUTDOWN).</summary>
            BrokenPipe,
            /// <summary><see cref="ChildProcessError"/> (ECHILD).</summary>
            ChildProcess,
            /// <summary><see cref="ConnectionAbortedError"/> (ECONNABORTED).</summary>
            ConnectionAborted,
            /// <summary><see cref="ConnectionRefusedError"/> (ECONNREFUSED).</summary>
            ConnectionRefused,
            /// <summary><see cref="ConnectionResetError"/> (ECONNRESET).</summary>
            ConnectionReset,
            /// <summary><see cref="FileExistsError"/> (EEXIST).</summary>
            FileExists,
            /// <summary><see cref="FileNotFoundError"/> (ENOENT).</summary>
            FileNotFound,
            /// <summary><see cref="IsADirectoryError"/> (EISDIR).</summary>
            IsADirectory,
            /// <summary><see cref="NotADirectoryError"/> (ENOTDIR).</summary>
            NotADirectory,
            /// <summary><see cref="InterruptedError"/> (EINTR).</summary>
            Interrupted,
            /// <summary><see cref="PermissionError"/> (EACCES, EPERM, ENOTCAPABLE).</summary>
            Permission,
            /// <summary><see cref="ProcessLookupError"/> (ESRCH).</summary>
            ProcessLookup,
            /// <summary><see cref="TimeoutError"/> (ETIMEDOUT).</summary>
            Timeout,
        }

        /// <summary>The subclass <paramref name="errno"/> names on this operating system.</summary>
        /// <param name="errno">The error number.</param>
        /// <returns>The subclass, or <see cref="Kind.None"/>.</returns>
        /// <remarks>Unix systems other than Linux, Android and the BSD family (Darwin, FreeBSD) use Linux's numbers.</remarks>
        internal static Kind ClassOf(int errno)
        {
            // The numbers every platform shares (POSIX's first block and the C runtime's).
            switch (errno)
            {
                case 1: case 13: return Kind.Permission;     // EPERM, EACCES
                case 2: return Kind.FileNotFound;            // ENOENT
                case 3: return Kind.ProcessLookup;           // ESRCH
                case 4: return Kind.Interrupted;             // EINTR
                case 10: return Kind.ChildProcess;           // ECHILD
                case 17: return Kind.FileExists;             // EEXIST
                case 20: return Kind.NotADirectory;          // ENOTDIR
                case 21: return Kind.IsADirectory;           // EISDIR
                case 32: return Kind.BrokenPipe;             // EPIPE
            }
            if (OperatingSystem.IsWindows())
            {
                // CPython replaces the CRT's own socket errnos with the Winsock values (WSAE*).
                switch (errno)
                {
                    case 11: case 10035: case 10036: case 10037: return Kind.BlockingIO; // EAGAIN, WSAEWOULDBLOCK, WSAEINPROGRESS, WSAEALREADY
                    case 10058: return Kind.BrokenPipe;        // WSAESHUTDOWN
                    case 10053: return Kind.ConnectionAborted; // WSAECONNABORTED
                    case 10061: return Kind.ConnectionRefused; // WSAECONNREFUSED
                    case 10054: return Kind.ConnectionReset;   // WSAECONNRESET
                    case 10060: return Kind.Timeout;           // WSAETIMEDOUT
                    default: return Kind.None;
                }
            }
            if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS() || OperatingSystem.IsTvOS() ||
                OperatingSystem.IsWatchOS() || OperatingSystem.IsMacCatalyst() || OperatingSystem.IsFreeBSD())
            {
                switch (errno)
                {
                    case 35: case 36: case 37: return Kind.BlockingIO; // EAGAIN (= EWOULDBLOCK), EINPROGRESS, EALREADY
                    case 58: return Kind.BrokenPipe;        // ESHUTDOWN
                    case 53: return Kind.ConnectionAborted; // ECONNABORTED
                    case 61: return Kind.ConnectionRefused; // ECONNREFUSED
                    case 54: return Kind.ConnectionReset;   // ECONNRESET
                    case 60: return Kind.Timeout;           // ETIMEDOUT
                    case 93 when OperatingSystem.IsFreeBSD(): return Kind.Permission; // ENOTCAPABLE (Capsicum)
                    default: return Kind.None;
                }
            }
            switch (errno)
            {
                case 11: case 114: case 115: return Kind.BlockingIO; // EAGAIN (= EWOULDBLOCK), EALREADY, EINPROGRESS
                case 108: return Kind.BrokenPipe;        // ESHUTDOWN
                case 103: return Kind.ConnectionAborted; // ECONNABORTED
                case 111: return Kind.ConnectionRefused; // ECONNREFUSED
                case 104: return Kind.ConnectionReset;   // ECONNRESET
                case 110: return Kind.Timeout;           // ETIMEDOUT
                default: return Kind.None;
            }
        }
    }
}
