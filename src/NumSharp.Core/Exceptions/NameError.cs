using System;

namespace NumSharp
{
    /// <summary>
    ///     Exception that corresponds to Python/NumPy's <c>NameError</c>.
    ///     Raised when a name referenced by value is not defined.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's <c>NameError</c> for API compatibility. In NumSharp this is raised where
    ///     NumPy itself raises <c>NameError</c> — notably <c>np.bmat("A,B; C,D", …)</c>, whose string
    ///     form resolves each whitespace/comma-separated token as a variable name and reaches
    ///     <c>defmatrix.py</c>'s <c>raise NameError(f"name {col!r} is not defined")</c> when a token is
    ///     absent from the supplied name dictionaries (this also fires for numeric literals, since a
    ///     token such as <c>"1"</c> is treated as a name).
    ///     <para>
    ///     Like Python's, it carries <see cref="name"/>: the undefined name when the raiser supplies it (CPython's own
    ///     lookup failures do; NumPy's <c>np.bmat</c> passes only the text, so its <see cref="name"/> is null, as
    ///     <c>e.name</c> is <c>None</c> there). Python's subclass <see cref="UnboundLocalError"/> derives from it.
    ///     </para>
    /// </remarks>
    public class NameError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise NameError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(NameError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>NameError</c>; prefer a message that names the undefined name.</remarks>
        public NameError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message; <see cref="name"/> stays null.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public NameError(string message) : base(message) { }

        /// <summary>
        ///     Python's <c>NameError(message, name=name)</c>: the message and the undefined name, which a caller reads from
        ///     <see cref="name"/> without parsing the text.
        /// </summary>
        /// <param name="message">The text, verbatim (<c>name 'x' is not defined</c>).</param>
        /// <param name="name">The name that is not defined, or null for none (Python's <c>name=None</c>).</param>
        /// <remarks>
        ///     Python's <c>name</c> is keyword-only: <c>NameError('m', 'x')</c> in Python is a two-argument error whose text is
        ///     <c>('m', 'x')</c>, while this constructor's second argument IS the name. Footgun: a literal <c>null</c> second
        ///     argument is ambiguous with <see cref="NameError(string, Exception)"/> (CS0121); cast it
        ///     (<c>(string)null</c>) or use the one-argument form.
        /// </remarks>
        public NameError(string message, string name) : base(message)
        {
            this.name = name;
        }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise NameError(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>; <see cref="name"/> stays null.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public NameError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>
        ///     The name that was not defined (Python's <c>name</c> attribute), or null when the raiser did not give it (every
        ///     constructor but <see cref="NameError(string, string)"/>).
        /// </summary>
        public string name { get; }
    }

    /// <summary>
    ///     Python's <c>UnboundLocalError</c>: a local variable read before any value was bound to it
    ///     (<c>cannot access local variable 'x' where it is not associated with a value</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A subclass of <see cref="NameError"/>, as in Python, so <c>catch (NameError)</c> catches it and it carries
    ///     <see cref="NameError.name"/>.
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it: C#'s definite-assignment rules make the situation a compile error. The class exists so
    ///     a port of code that raises or catches it compiles and behaves as in Python.
    ///     </para>
    /// </remarks>
    public class UnboundLocalError : NameError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise UnboundLocalError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(UnboundLocalError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>UnboundLocalError</c>.</remarks>
        public UnboundLocalError() : base(string.Empty) { }

        /// <summary>Creates the error with the interpreter's (or the port's) message; <see cref="NameError.name"/> stays
        ///     null.</summary>
        /// <param name="message">The text, verbatim (<c>cannot access local variable 'x' where it is not associated with a
        ///     value</c>).</param>
        public UnboundLocalError(string message) : base(message) { }

        /// <summary>Python's <c>UnboundLocalError(message, name=name)</c>: the message and the unbound name.</summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="name">The unbound local's name, or null for none.</param>
        /// <remarks>Footgun, as for <see cref="NameError(string, string)"/>: a literal <c>null</c> second argument is
        ///     ambiguous with the cause form; cast it (<c>(string)null</c>).</remarks>
        public UnboundLocalError(string message, string name) : base(message, name) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     UnboundLocalError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>;
        ///     <see cref="NameError.name"/> stays null.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public UnboundLocalError(string message, Exception innerException) : base(message, innerException) { }
    }
}
