using System;
using System.IO;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-compatible <c>OSError</c>: an operating-system level file operation NumPy refuses or that fails. NumPy raises
    ///     it from <c>ndarray.tofile</c> (<c>cannot write object arrays to a file in binary mode</c> for a
    ///     <c>StringDType</c> array, whose 16-byte cells point into an allocator a file cannot hold) and from its file
    ///     writes (<c>problem writing element 3 to file</c>), and NumSharp raises it in the same places with NumPy's
    ///     verbatim text.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Unlike the other Python-named errors (<see cref="TypeError"/>, <see cref="SystemError"/>, which derive from
    ///     <see cref="NumSharpException"/>), it derives from <see cref="IOException"/>: Python's <c>OSError</c> IS the I/O
    ///     error class (<c>IOError</c> is its alias), so a .NET caller that catches <see cref="IOException"/> around a file
    ///     operation catches this refusal too — the same reason <see cref="ValueError"/> derives from
    ///     <see cref="ArgumentException"/>.
    ///     </para>
    ///     <para>
    ///     Plain .NET I/O failures (a missing file, a full disk) keep their own .NET exception types; this class carries only
    ///     the errors NumPy itself raises with its own text.
    ///     </para>
    /// </remarks>
    public class OSError : IOException, INumSharpException
    {
        /// <summary>Creates the error with NumPy's message text.</summary>
        /// <param name="message">The verbatim message (<c>cannot write object arrays to a file in binary mode</c>).</param>
        public OSError(string message) : base(message) { }

        /// <summary>Creates the error with NumPy's message text and the exception that caused it.</summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The cause (the .NET I/O exception behind a failed write).</param>
        public OSError(string message, Exception innerException) : base(message, innerException) { }
    }
}
