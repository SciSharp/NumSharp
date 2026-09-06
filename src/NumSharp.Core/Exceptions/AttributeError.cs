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
    /// </remarks>
    public class AttributeError : NumSharpException
    {
        public AttributeError() : base("AttributeError") { }
        public AttributeError(string message) : base(message) { }
    }
}
