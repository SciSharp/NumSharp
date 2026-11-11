using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>SystemExit</c>: <c>sys.exit(code)</c>, a request to end the program with an exit status
    ///     (<see cref="code"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     In Python it sits beside <c>Exception</c> under <c>BaseException</c>, so <c>except Exception</c> does not catch
    ///     it. C# has one root: NumSharp's derives from <see cref="NumSharpException"/>, and footgun, a
    ///     <c>catch (Exception)</c> DOES catch it. Code that must let it through filters it out
    ///     (<c>catch (Exception e) when (e is not SystemExit)</c>).
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it, and throwing it does not end the process: a port that wants Python's behaviour catches
    ///     it at the top and calls <see cref="Environment.Exit(int)"/> with <see cref="code"/> (an int as is, None as 0,
    ///     anything else printed and 1, as CPython does).
    ///     </para>
    /// </remarks>
    public class SystemExit : NumSharpException
    {
        /// <summary>
        ///     Python's <c>sys.exit()</c>: an empty <see cref="Exception.Message"/> and a null <see cref="code"/>, as
        ///     <c>str(SystemExit())</c> is <c>''</c> and its <c>code</c> is <c>None</c> (exit status 0).
        /// </summary>
        public SystemExit() : base(string.Empty) { }

        /// <summary>
        ///     Python's <c>sys.exit(text)</c>: the text is both the message and <see cref="code"/> (CPython prints it to
        ///     stderr and exits with status 1).
        /// </summary>
        /// <param name="message">The text, or null for Python's <c>SystemExit(None)</c> (message <c>None</c>, code
        ///     null).</param>
        public SystemExit(string message) : base(message ?? "None")
        {
            code = message;
        }

        /// <summary>
        ///     Python's <c>sys.exit(code)</c>: the exit status (an int) or any other object. The message is Python's
        ///     <c>str(code)</c> (<c>3</c>, <c>True</c>, <c>None</c>).
        /// </summary>
        /// <param name="code">The exit status or object, or null for <c>None</c>.</param>
        /// <exception cref="ValueError"><paramref name="code"/> is a Python int (<see cref="System.Numerics.BigInteger"/>) of
        ///     more than 4300 decimal digits, whose <c>str()</c> CPython refuses.</exception>
        public SystemExit(object code) : base(NDPyObject.Str(code))
        {
            this.code = code;
        }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise SystemExit(message) from
        ///     innerException</c>, the cause kept in <see cref="Exception.InnerException"/>; the message is also
        ///     <see cref="code"/>.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public SystemExit(string message, Exception innerException) : base(message ?? "None", innerException)
        {
            code = message;
        }

        /// <summary>
        ///     The exit status or object (Python's <c>code</c>): the constructor's argument, or null after the bare
        ///     constructor (Python's <c>None</c>, exit status 0).
        /// </summary>
        public object code { get; }
    }

    /// <summary>
    ///     Python's <c>KeyboardInterrupt</c>: the user pressed the interrupt key (Ctrl+C), which CPython turns into this
    ///     error in the running code.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     In Python it sits beside <c>Exception</c> under <c>BaseException</c>, so <c>except Exception</c> does not catch
    ///     it. C# has one root: NumSharp's derives from <see cref="NumSharpException"/>, and footgun, a
    ///     <c>catch (Exception)</c> DOES catch it.
    ///     </para>
    ///     <para>
    ///     NumSharp never raises it (.NET reports Ctrl+C through <see cref="Console.CancelKeyPress"/>, not an exception); a
    ///     port that wants Python's behaviour throws it from its own cancellation check.
    ///     </para>
    /// </remarks>
    public class KeyboardInterrupt : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise KeyboardInterrupt</c> (what CPython raises): an
        ///     empty <see cref="Exception.Message"/>, as <c>str(KeyboardInterrupt())</c> is <c>''</c>.
        /// </summary>
        public KeyboardInterrupt() : base(string.Empty) { }

        /// <summary>Creates the error with the port's message.</summary>
        /// <param name="message">The text, verbatim.</param>
        public KeyboardInterrupt(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     KeyboardInterrupt(message) from innerException</c> (an <see cref="OperationCanceledException"/>, for
        ///     example).
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public KeyboardInterrupt(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Python's <c>GeneratorExit</c>: raised inside a generator when it is closed (<c>generator.close()</c>), so its
    ///     <c>finally</c> blocks run.
    /// </summary>
    /// <remarks>
    ///     In Python it sits beside <c>Exception</c> under <c>BaseException</c>; NumSharp's derives from
    ///     <see cref="NumSharpException"/>, so, footgun, a <c>catch (Exception)</c> catches it. NumSharp never raises it (a
    ///     C# iterator's <c>finally</c> runs on <see cref="IDisposable.Dispose"/>); the class exists for ports.
    /// </remarks>
    public class GeneratorExit : NumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise GeneratorExit</c> (what <c>close()</c> raises): an
        ///     empty <see cref="Exception.Message"/>, as <c>str(GeneratorExit())</c> is <c>''</c>.
        /// </summary>
        public GeneratorExit() : base(string.Empty) { }

        /// <summary>Creates the error with the port's message.</summary>
        /// <param name="message">The text, verbatim.</param>
        public GeneratorExit(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise GeneratorExit(message)
        ///     from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The text, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public GeneratorExit(string message, Exception innerException) : base(message, innerException) { }
    }
}
