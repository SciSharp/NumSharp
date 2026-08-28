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
        // This matches NumPy's internal PyArray_FromAny behavior in ufuncs.
        //
        // Scope: np.asanyarray(x) MINTS a fresh 0-d/array temp for a scalar or array-like operand —
        // a leftover otherwise reclaimable only by the finalizer. A plain `using` would be a BUG
        // (asanyarray returns the SAME array when x is already an NDArray, so it would dispose the
        // caller's input, rule R2), so each overload is [NDScoped]: the weaver tracks only the
        // freshly-constructed temp and reclaims it at exit, leaving an input passthrough untouched
        // and yielding the result. The core NDArray×NDArray operators above are deliberately NOT
        // scoped — they own no temp (the engine disposes its own scalar-cast temp on the hot path).

        // Add
        [NDScoped] public static NDArray operator +(NDArray left, object right) => left + np.asanyarray(right);
        [NDScoped] public static NDArray operator +(object left, NDArray right) => np.asanyarray(left) + right;

        // Subtract
        [NDScoped] public static NDArray operator -(NDArray left, object right) => left - np.asanyarray(right);
        [NDScoped] public static NDArray operator -(object left, NDArray right) => np.asanyarray(left) - right;

        // Multiply
        [NDScoped] public static NDArray operator *(NDArray left, object right) => left * np.asanyarray(right);
        [NDScoped] public static NDArray operator *(object left, NDArray right) => np.asanyarray(left) * right;

        // Divide
        [NDScoped] public static NDArray operator /(NDArray left, object right) => left / np.asanyarray(right);
        [NDScoped] public static NDArray operator /(object left, NDArray right) => np.asanyarray(left) / right;

        // Modulo
        [NDScoped] public static NDArray operator %(NDArray left, object right) => left % np.asanyarray(right);
        [NDScoped] public static NDArray operator %(object left, NDArray right) => np.asanyarray(left) % right;
    }
}
