namespace NumSharp
{
    /// <summary>
    /// Exception that corresponds to Python/NumPy's TypeError.
    /// Raised when an operation or function receives an argument of inappropriate type.
    /// </summary>
    public class TypeError : NumSharpException
    {
        public TypeError() : base("TypeError") { }
        public TypeError(string message) : base(message) { }
    }
}
