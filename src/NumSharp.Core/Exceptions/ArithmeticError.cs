using System;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>ArithmeticError</c>: the base of the numeric failures <see cref="OverflowError"/>,
    ///     <see cref="ZeroDivisionError"/> and <see cref="FloatingPointError"/>. Catch it to handle all three at once, as
    ///     <c>except ArithmeticError</c> does.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from .NET's <see cref="ArithmeticException"/>, the exception with the same meaning, so a
    ///     <c>catch (ArithmeticException)</c> written for .NET code catches NumSharp's Python arithmetic errors too.
    ///     </para>
    ///     <para>
    ///     Footgun: its subclasses are NOT .NET's <see cref="OverflowException"/> / <see cref="DivideByZeroException"/>
    ///     (both are .NET siblings of this class under <see cref="ArithmeticException"/>), so a <c>catch
    ///     (OverflowException)</c> does not see an <see cref="OverflowError"/>. NumSharp raises the Python classes wherever
    ///     NumPy or CPython raises them; .NET's own exceptions stay for what .NET itself refuses (a <c>checked</c>
    ///     conversion inside NumSharp, a <see cref="Char8"/> conversion out of range).
    ///     </para>
    /// </remarks>
    public class ArithmeticError : ArithmeticException, INumSharpException
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ArithmeticError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ArithmeticError())</c> is <c>''</c> (not .NET's
        ///     <see cref="ArithmeticException"/> default, "Overflow or underflow in the arithmetic operation.").
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>ArithmeticError</c>.</remarks>
        public ArithmeticError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or the port's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        public ArithmeticError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     ArithmeticError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as
        ///     Python keeps it in <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public ArithmeticError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Python's <c>OverflowError</c>: a number too large for the type it must become. NumSharp raises it where NumPy
    ///     and CPython do, with their text: a Python int that does not fit the target (<c>Python integer 300 out of bounds
    ///     for uint8</c>, <c>Python int too large to convert to C long</c>), a float infinity converted to an integer
    ///     (<c>cannot convert float infinity to integer</c>), an int too large for a float (<c>int too large to convert to
    ///     float</c>), a string repeat or pad whose result cannot be sized (<c>Overflow encountered in string multiply</c>),
    ///     and the random generators' range checks (<c>Range exceeds valid bounds</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A subclass of <see cref="ArithmeticError"/> (Python's MRO: <c>OverflowError → ArithmeticError → Exception</c>),
    ///     so <c>catch (ArithmeticError)</c> and <c>catch (ArithmeticException)</c> catch it.
    ///     </para>
    ///     <para>
    ///     Footgun: it is NOT .NET's <see cref="OverflowException"/>, and a <c>catch (OverflowException)</c> written before
    ///     NumSharp raised Python's class misses it; catch <see cref="OverflowError"/> (or
    ///     <see cref="ArithmeticException"/> for both). A <c>checked</c> .NET conversion still throws
    ///     <see cref="OverflowException"/>; NumSharp's own size limits (a managed array past <see cref="int.MaxValue"/>,
    ///     a <see cref="Char8"/> conversion) keep theirs.
    ///     </para>
    /// </remarks>
    public class OverflowError : ArithmeticError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise OverflowError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(OverflowError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>OverflowError</c>; prefer a message that names the value and its target.</remarks>
        public OverflowError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or CPython's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error (<c>Python integer 300 out of
        ///     bounds for uint8</c>).</param>
        public OverflowError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise OverflowError(message)
        ///     from innerException</c>, the cause kept in <see cref="Exception.InnerException"/> as Python keeps it in
        ///     <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one (a .NET <see cref="OverflowException"/> from a
        ///     <c>checked</c> conversion, for example), or null for none.</param>
        public OverflowError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Python's <c>ZeroDivisionError</c>: a division or modulo whose divisor is zero, raised where Python's scalar
    ///     arithmetic raises it. NumSharp raises it where NumPy does: <c>np.average</c> over weights that sum to zero
    ///     (<c>Weights sum to zero, can't be normalized</c>), <c>np.fft.fftfreq</c> with <c>d=0</c> (<c>float division by
    ///     zero</c>), the object-dtype and polynomial paths that compute with Python scalars (<c>division by zero</c>,
    ///     <c>integer division or modulo by zero</c>, <c>complex division by zero</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Array division never raises it: NumPy's ufuncs divide by zero into <c>inf</c> / <c>nan</c> (floats) or 0
    ///     (integers), and so do NumSharp's. It comes only from code that NumPy runs on Python scalars.
    ///     </para>
    ///     <para>
    ///     A subclass of <see cref="ArithmeticError"/>; footgun: NOT .NET's <see cref="DivideByZeroException"/>, which a
    ///     <c>catch (DivideByZeroException)</c> written for older NumSharp expects. Catch <see cref="ZeroDivisionError"/>,
    ///     or <see cref="ArithmeticException"/> for both.
    ///     </para>
    /// </remarks>
    public class ZeroDivisionError : ArithmeticError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise ZeroDivisionError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(ZeroDivisionError())</c> is <c>''</c> (NumPy's polynomial
        ///     division by a zero series raises exactly this).
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>ZeroDivisionError</c>.</remarks>
        public ZeroDivisionError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's (or CPython's) message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error (<c>float division by
        ///     zero</c>).</param>
        public ZeroDivisionError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     ZeroDivisionError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>
        ///     as Python keeps it in <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one (a .NET <see cref="DivideByZeroException"/>,
        ///     for example), or null for none.</param>
        public ZeroDivisionError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Python's <c>FloatingPointError</c>: the error NumPy raises for a floating-point exception when
    ///     <c>np.seterr(all='raise')</c> (or <c>np.errstate</c>) turns warnings into errors, with NumPy's text
    ///     (<c>divide by zero encountered in divide</c>, <c>overflow encountered in exp</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumSharp itself never raises it: it has no floating-point error state (no <c>np.seterr</c> /
    ///     <c>np.errstate</c>) and computes IEEE results (<c>inf</c>, <c>nan</c>) silently, as NumPy does under its
    ///     default <c>'warn'</c> mode minus the warning. The class exists so a port of code that raises or catches it
    ///     compiles and behaves as in Python.
    ///     </para>
    ///     <para>A subclass of <see cref="ArithmeticError"/>, as in Python.</para>
    /// </remarks>
    public class FloatingPointError : ArithmeticError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise FloatingPointError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(FloatingPointError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks><see cref="Exception.ToString"/> then starts with the class name alone, as Python's traceback prints a
        ///     bare <c>FloatingPointError</c>.</remarks>
        public FloatingPointError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's message.</summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error (<c>divide by zero encountered in
        ///     divide</c>).</param>
        public FloatingPointError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with a message and the exception that caused it: Python's <c>raise
        ///     FloatingPointError(message) from innerException</c>, the cause kept in <see cref="Exception.InnerException"/>
        ///     as Python keeps it in <c>__cause__</c>.
        /// </summary>
        /// <param name="message">The text, verbatim: Python's <c>str()</c> of the error.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public FloatingPointError(string message, Exception innerException) : base(message, innerException) { }
    }
}
