using System;

namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's AttributeError.
    /// Raised when an attribute reference or assignment fails.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's AttributeError for API compatibility. In NumSharp this is raised where NumPy
    ///     itself raises <c>AttributeError</c> — notably <c>np.savetxt(fname, X, fmt=[...])</c> when the
    ///     list/tuple of formats is not the same length as the number of columns, which reaches
    ///     <c>_npyio_impl.py</c>'s <c>raise AttributeError(f'fmt has wrong shape.  {str(fmt)}')</c>.
    ///     <para>
    ///     Like Python's, it carries <see cref="name"/> and <see cref="obj"/>: the attribute that was looked up and the
    ///     object it was looked up on. CPython fills both when an attribute lookup fails (<c>'NoneType' object has no
    ///     attribute 'join'</c>), and NumSharp's raise sites that stand for such a lookup fill them too; an explicit
    ///     <c>raise AttributeError("...")</c> in NumPy's Python code leaves both None, and so does NumSharp's twin.
    ///     </para>
    /// </remarks>
    public class AttributeError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise AttributeError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(AttributeError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>AttributeError</c>; prefer a message that names the attribute.</remarks>
        public AttributeError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message; <see cref="name"/> and <see cref="obj"/> stay
        ///     null.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public AttributeError(string message) : base(message) { }

        /// <summary>
        ///     Python's <c>AttributeError(message, name=name, obj=obj)</c>: the message, the attribute's name and the object
        ///     the lookup ran on, which a caller reads from <see cref="name"/> / <see cref="obj"/> without parsing the text.
        /// </summary>
        /// <param name="message">The text, verbatim (<c>'NoneType' object has no attribute 'join'</c>).</param>
        /// <param name="name">The attribute that was looked up, or null for none (Python's <c>name=None</c>).</param>
        /// <param name="obj">The object it was looked up on, or null (Python's <c>None</c>: both "no object given" and a
        ///     lookup on <c>None</c> itself, as in Python, where the two read the same).</param>
        /// <remarks>Python's <c>name</c> and <c>obj</c> are keyword-only; the C# form takes both positionally (pass null for
        ///     the one you lack), so it never collides with <see cref="AttributeError(string, Exception)"/>.</remarks>
        public AttributeError(string message, string name, object obj) : base(message)
        {
            this.name = name;
            this.obj = obj;
        }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     AttributeError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as
        ///     Python keeps it in <c>__cause__</c>; <see cref="name"/> and <see cref="obj"/> stay null.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public AttributeError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>
        ///     The attribute that was looked up (Python's <c>name</c>), or null when the raiser did not give it.
        /// </summary>
        public string name { get; }

        /// <summary>
        ///     The object the attribute was looked up on (Python's <c>obj</c>), or null when the raiser did not give it or
        ///     the object was <c>None</c>.
        /// </summary>
        /// <remarks>Footgun: it holds a live reference to that object (an <see cref="NDArray"/>, for example) for as long
        ///     as the exception is kept, as Python's does; drop a caught error you store instead of keeping its
        ///     object alive by accident.</remarks>
        public object obj { get; }
    }
}
