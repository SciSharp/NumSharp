namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's RuntimeError.
    /// Raised when an error is detected that does not fall into any of the other categories.
    /// </summary>
    /// <remarks>
    ///     Mirrors Python's RuntimeError for API compatibility. Common cases:
    ///     - np.cov: "cannot handle multidimensional fweights"
    ///     - np.cov: "incompatible numbers of samples and fweights"
    /// </remarks>
    public class RuntimeError : NumSharpException
    {
        public RuntimeError() : base("RuntimeError") { }
        public RuntimeError(string message) : base(message) { }
    }
}
