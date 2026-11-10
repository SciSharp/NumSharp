using System;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>ImportError</c>: a module or a name in it could not be imported. NumSharp itself raises none (.NET
    ///     binds its assemblies at build time); the class exists so code ported from Python keeps Python's
    ///     <c>raise ImportError(...)</c> and <c>except ImportError</c> - a plugin or registry that loads by name, a missing
    ///     optional dependency - with Python's attributes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Like CPython's (3.12), it carries <see cref="msg"/>, <see cref="name"/> and <see cref="path"/>; Python's
    ///     keyword-only <c>name=</c> / <c>path=</c> are the <see cref="ImportError(string, string, string)"/> constructor.
    ///     CPython 3.12's undocumented <c>name_from</c> (filled by the import system for its "Did you mean" hints) is not
    ///     modelled.
    ///     </para>
    ///     <para>
    ///     It derives from <see cref="NumSharpException"/> like the other Python-named errors with no .NET counterpart, and
    ///     Python's own subclass <see cref="ModuleNotFoundError"/> derives from it, so <c>catch (ImportError)</c> catches
    ///     both as <c>except ImportError</c> does.
    ///     </para>
    /// </remarks>
    public class ImportError : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ImportError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ImportError())</c> is <c>''</c>; <see cref="msg"/>,
        ///     <see cref="name"/> and <see cref="path"/> are null.
        /// </summary>
        public ImportError() : base(string.Empty) { }

        /// <summary>Creates the error with a message, Python's <c>ImportError(msg)</c>: <see cref="msg"/> is the message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public ImportError(string message) : base(message) => msg = message;

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise ImportError(msg) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ImportError(string message, Exception innerException) : base(message, innerException) => msg = message;

        /// <summary>
        ///     Python's <c>ImportError(msg, name=name, path=path)</c>: the message plus the module that was being imported and
        ///     the file that triggered the error.
        /// </summary>
        /// <param name="message">The text (Python's <c>msg</c>), or null for Python's <c>ImportError(name=..., path=...)</c>
        ///     without one: <see cref="Exception.Message"/> is then empty, as <c>str()</c> of that error is <c>''</c>.</param>
        /// <param name="name">The module name (Python's <c>name</c>), or null.</param>
        /// <param name="path">The file path (Python's <c>path</c>), or null.</param>
        /// <remarks>Both keywords are required here (pass null for the one Python's call leaves out): an optional
        ///     <paramref name="name"/> would make <c>new ImportError(text, null)</c> ambiguous with the inner-exception
        ///     form.</remarks>
        public ImportError(string message, string name, string path) : base(message ?? string.Empty)
        {
            msg = message;
            this.name = name;
            this.path = path;
        }

        /// <summary>
        ///     Python's <c>raise ImportError(msg, name=name, path=path) from innerException</c>: the keyword form with the
        ///     exception that caused it.
        /// </summary>
        /// <param name="message">The text (Python's <c>msg</c>), or null for none (an empty <see cref="Exception.Message"/>).</param>
        /// <param name="name">The module name (Python's <c>name</c>), or null.</param>
        /// <param name="path">The file path (Python's <c>path</c>), or null.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ImportError(string message, string name, string path, Exception innerException) : base(message ?? string.Empty, innerException)
        {
            msg = message;
            this.name = name;
            this.path = path;
        }

        /// <summary>The message the error was created with (Python's <c>msg</c>), or null when it was created without one.</summary>
        /// <remarks>Python sets <c>msg</c> only from a lone positional argument; here every message-taking constructor sets
        ///     it.</remarks>
        public string msg { get; }

        /// <summary>The name of the module that was being imported (Python's <c>name</c>), or null.</summary>
        public string name { get; }

        /// <summary>The path of the file that triggered the error (Python's <c>path</c>), or null.</summary>
        public string path { get; }
    }

    /// <summary>
    ///     Python's <c>ModuleNotFoundError</c>, the <see cref="ImportError"/> raised when a module cannot be located
    ///     (<c>No module named 'x'</c>). NumSharp itself raises none; a port raises it where its Python source does - a
    ///     registry that resolves <c>"module:Name"</c> entry points, for example.
    /// </summary>
    /// <remarks>A subclass of <see cref="ImportError"/>, as in Python, so <c>catch (ImportError)</c> catches it. Python's
    ///     import system fills <see cref="ImportError.name"/> with the missing module; a port does the same through the
    ///     keyword constructor.</remarks>
    public class ModuleNotFoundError : ImportError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ModuleNotFoundError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ModuleNotFoundError())</c> is <c>''</c>.
        /// </summary>
        public ModuleNotFoundError() : base() { }

        /// <summary>Creates the error with a message (<c>No module named 'x'</c>).</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public ModuleNotFoundError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     ModuleNotFoundError(msg) from innerException</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ModuleNotFoundError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>Python's <c>ModuleNotFoundError(msg, name=name, path=path)</c>.</summary>
        /// <param name="message">The text (Python's <c>msg</c>), or null for none (an empty <see cref="Exception.Message"/>).</param>
        /// <param name="name">The missing module's name (Python's <c>name</c>), or null.</param>
        /// <param name="path">The file path (Python's <c>path</c>), or null.</param>
        public ModuleNotFoundError(string message, string name, string path) : base(message, name, path) { }

        /// <summary>Python's <c>raise ModuleNotFoundError(msg, name=name, path=path) from innerException</c>.</summary>
        /// <param name="message">The text (Python's <c>msg</c>), or null for none (an empty <see cref="Exception.Message"/>).</param>
        /// <param name="name">The missing module's name (Python's <c>name</c>), or null.</param>
        /// <param name="path">The file path (Python's <c>path</c>), or null.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ModuleNotFoundError(string message, string name, string path, Exception innerException)
            : base(message, name, path, innerException) { }
    }
}
