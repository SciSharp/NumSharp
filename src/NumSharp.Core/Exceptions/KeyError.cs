namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's KeyError.
    /// Raised when a mapping (dict) key is not found.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's KeyError for API compatibility. In NumSharp this is raised where NumPy
    ///     itself raises <c>KeyError</c> — notably <c>np.einsum_path(..., optimize="&lt;unknown&gt;")</c>,
    ///     which reaches <c>einsumfunc.py</c>'s <c>raise KeyError("Path name %s not found", path_type)</c>.
    /// </remarks>
    public class KeyError : NumSharpException
    {
        public KeyError() : base("KeyError") { }
        public KeyError(string message) : base(message) { }
    }
}
