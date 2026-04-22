namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's IndexError.
    /// Raised when a sequence subscript is out of range, or when an index type is invalid
    /// (e.g. float/complex index on an ndarray).
    /// </summary>
    public class IndexError : NumSharpException
    {
        public IndexError() : base("IndexError") { }
        public IndexError(string message) : base(message) { }
    }
}
