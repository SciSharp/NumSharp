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
    /// </remarks>
    public class NameError : NumSharpException
    {
        public NameError() : base("NameError") { }
        public NameError(string message) : base(message) { }
    }
}
