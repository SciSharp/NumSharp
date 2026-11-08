namespace NumSharp
{
    /// <summary>
    ///     Exception that corresponds to Python's <c>SyntaxError</c>: text that is not valid Python source. Catch it
    ///     where a string argument is parsed as Python literals and the caller wants to tell malformed text from a
    ///     well-formed value that is merely not allowed (that one is a <see cref="ValueError"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Raised by the string form of <see cref="np.matrix(string, DType, bool)"/> / <see cref="np.asmatrix(string, DType)"/>,
    ///     where NumPy's <c>_convert_from_string</c> hands every whitespace-separated token to
    ///     <c>ast.literal_eval</c>. A token that does not parse (<c>"1 + 2"</c> splits into the lone token
    ///     <c>+</c>, <c>"1L"</c> is not a number) is CPython's <c>SyntaxError</c> there, with CPython's own text
    ///     (<c>invalid syntax (&lt;unknown&gt;, line 1)</c>, <c>invalid decimal literal (&lt;unknown&gt;, line 1)</c>).
    ///     </para>
    ///     <para>
    ///     It is NOT a <see cref="ValueError"/>, exactly as in Python, so a <c>catch (ValueError)</c> written for
    ///     "Rows not the same size." does not swallow a typo in the literal.
    ///     </para>
    /// </remarks>
    public class SyntaxError : NumSharpException
    {
        /// <summary>Creates the error with Python's bare class name as its message. Prefer the overload that
        ///     carries the parser's text: a bare <c>SyntaxError</c> tells the user nothing about the token.</summary>
        public SyntaxError() : base("SyntaxError") { }

        /// <summary>Creates the error with the parser's message.</summary>
        /// <param name="message">CPython's message, including its <c>(&lt;unknown&gt;, line 1)</c> location suffix
        ///     where CPython prints one.</param>
        public SyntaxError(string message) : base(message) { }
    }
}
