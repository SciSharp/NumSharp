namespace NumSharp
{
    /// <summary>
    ///     Arithmetic operators for NDArray.
    ///     The <c>object</c> overloads accept any scalar or array-like: a boxed C# number is the weak Python scalar NumPy's
    ///     operator hands the ufunc (<see cref="OperatorOperand"/>), anything else is converted via np.asanyarray.
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
        /// <summary>
        ///     NumPy's unary <c>+arr</c>, the <c>positive</c> ufunc: a fresh copy of <paramref name="x"/> in its own dtype.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns>A copy of <paramref name="x"/> (never <paramref name="x"/> itself, as NumPy's ufunc allocates its
        ///     output), laid out as the ufunc allocates it (an F-ordered operand gives an F-ordered result).</returns>
        /// <exception cref="TypeError"><paramref name="x"/> is a bool array (positive has no bool loop: NumPy's
        ///     <c>ufunc 'positive' did not contain a loop with signature matching types &lt;class
        ///     'numpy.dtypes.BoolDType'&gt; -&gt; None</c>), or holds text (no string loop: the class-named text).</exception>
        /// <remarks>The whole call is the engine's <see cref="Backends.TensorEngine.Positive"/>, as for <c>np.positive</c>
        ///     (the np-layer dtype plan, P1c). The operator used to check text itself and return <c>x.copy()</c>: a bool
        ///     operand was copied where NumPy raises, and an F-ordered one came back C-ordered.</remarks>
        public static NDArray operator +(NDArray x) => x.TensorEngine.Positive(x);

        // Binary operators with object: accepts any scalar or array-like through OperatorOperand — a boxed C# number is the
        // WEAK Python scalar (as typed literals are through the implicit conversions), anything else np.asanyarray. This
        // matches NumPy, whose operators pass the Python object to the ufunc, which applies NEP 50 to it.
        //
        // Scope: OperatorOperand(x) MINTS a fresh 0-d/array temp for a scalar or array-like operand —
        // a leftover otherwise reclaimable only by the finalizer. A plain `using` would be a BUG
        // (it returns the SAME array when x is already an NDArray, so it would dispose the
        // caller's input, rule R2), so each overload is [NDScoped]: the weaver tracks only the
        // freshly-constructed temp and reclaims it at exit, leaving an input passthrough untouched
        // and yielding the result. The core NDArray×NDArray operators above are deliberately NOT
        // scoped — they own no temp (the engine disposes its own scalar-cast temp on the hot path).

        // Add
        [NDScoped] public static NDArray operator +(NDArray left, object right) => left + OperatorOperand(right);
        [NDScoped] public static NDArray operator +(object left, NDArray right) => OperatorOperand(left) + right;

        // Subtract
        [NDScoped] public static NDArray operator -(NDArray left, object right) => left - OperatorOperand(right);
        [NDScoped] public static NDArray operator -(object left, NDArray right) => OperatorOperand(left) - right;

        // Multiply
        [NDScoped] public static NDArray operator *(NDArray left, object right) => left * OperatorOperand(right);
        [NDScoped] public static NDArray operator *(object left, NDArray right) => OperatorOperand(left) * right;

        // Divide
        [NDScoped] public static NDArray operator /(NDArray left, object right) => left / OperatorOperand(right);
        [NDScoped] public static NDArray operator /(object left, NDArray right) => OperatorOperand(left) / right;

        // Modulo
        [NDScoped] public static NDArray operator %(NDArray left, object right) => left % OperatorOperand(right);
        [NDScoped] public static NDArray operator %(object left, NDArray right) => OperatorOperand(left) % right;
    }
}
