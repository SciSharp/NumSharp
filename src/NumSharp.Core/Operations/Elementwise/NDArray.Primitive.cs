namespace NumSharp
{
    /// <summary>
    ///     Arithmetic operators for NDArray.
    ///     Uses the object pattern matching NumPy's PyArray_FromAny behavior:
    ///     any type is accepted and converted via np.asanyarray.
    /// </summary>
    public partial class NDArray
    {
        // Core NDArray × NDArray operators
        public static NDArray operator +(NDArray x, NDArray y) => x.TensorEngine.Add(x, y);
        public static NDArray operator -(NDArray x, NDArray y) => x.TensorEngine.Subtract(x, y);
        public static NDArray operator *(NDArray x, NDArray y) => x.TensorEngine.Multiply(x, y);
        public static NDArray operator /(NDArray x, NDArray y) => x.TensorEngine.Divide(x, y);
        public static NDArray operator %(NDArray x, NDArray y) => x.TensorEngine.Mod(x, y);

        // Unary operators
        public static NDArray operator -(NDArray x) => x.TensorEngine.Negate(x);
        public static NDArray operator +(NDArray x) => x.copy(); // NumPy returns a copy for +arr

        // Binary operators with object: accepts any scalar or array-like, converts via np.asanyarray
        // This matches NumPy's internal PyArray_FromAny behavior in ufuncs

        // Add
        public static NDArray operator +(NDArray left, object right) => left + np.asanyarray(right);
        public static NDArray operator +(object left, NDArray right) => np.asanyarray(left) + right;

        // Subtract
        public static NDArray operator -(NDArray left, object right) => left - np.asanyarray(right);
        public static NDArray operator -(object left, NDArray right) => np.asanyarray(left) - right;

        // Multiply
        public static NDArray operator *(NDArray left, object right) => left * np.asanyarray(right);
        public static NDArray operator *(object left, NDArray right) => np.asanyarray(left) * right;

        // Divide
        public static NDArray operator /(NDArray left, object right) => left / np.asanyarray(right);
        public static NDArray operator /(object left, NDArray right) => np.asanyarray(left) / right;

        // Modulo
        public static NDArray operator %(NDArray left, object right) => left % np.asanyarray(right);
        public static NDArray operator %(object left, NDArray right) => np.asanyarray(left) % right;
    }
}
