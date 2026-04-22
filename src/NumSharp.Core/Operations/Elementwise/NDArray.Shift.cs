namespace NumSharp
{
    /// <summary>
    ///     Bitwise shift operators for NDArray.
    ///
    ///     NumPy alignment: mirrors <c>__lshift__</c> / <c>__rshift__</c> on <c>ndarray</c>.
    ///     Wires straight into <see cref="Backends.TensorEngine.LeftShift"/> and
    ///     <see cref="Backends.TensorEngine.RightShift"/>, which validate integer dtype.
    ///
    ///     C# shift operator constraints:
    ///     <list type="bullet">
    ///       <item>First operand must be the declaring type (<see cref="NDArray"/>). So
    ///             <c>object &lt;&lt; NDArray</c> is impossible — use <c>np.left_shift(lhs, rhs)</c>
    ///             instead, or cast <c>lhs</c> to NDArray explicitly.</item>
    ///       <item>Second operand can be any type (C# 11+; LangVersion=latest, net8/net10 support this).</item>
    ///       <item>Compound <c>&lt;&lt;=</c> / <c>&gt;&gt;=</c> are synthesized by the compiler from these.</item>
    ///     </list>
    /// </summary>
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise left shift. Integer dtypes only.
        /// Shifts bits of <paramref name="lhs"/> left by <paramref name="rhs"/>.
        /// Broadcast-aware.
        /// </summary>
        public static NDArray operator <<(NDArray lhs, NDArray rhs)
        {
            return lhs.TensorEngine.LeftShift(lhs, rhs);
        }

        /// <summary>
        /// Element-wise left shift with any scalar or array-like on RHS.
        /// Converts RHS via <see cref="np.asanyarray(object)"/> (matches NumPy's PyArray_FromAny).
        /// </summary>
        public static NDArray operator <<(NDArray lhs, object rhs)
        {
            return lhs << np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise right shift. Integer dtypes only.
        /// Shifts bits of <paramref name="lhs"/> right by <paramref name="rhs"/>.
        /// Logical shift for unsigned types, arithmetic shift for signed types.
        /// Broadcast-aware.
        /// </summary>
        public static NDArray operator >>(NDArray lhs, NDArray rhs)
        {
            return lhs.TensorEngine.RightShift(lhs, rhs);
        }

        /// <summary>
        /// Element-wise right shift with any scalar or array-like on RHS.
        /// Converts RHS via <see cref="np.asanyarray(object)"/> (matches NumPy's PyArray_FromAny).
        /// </summary>
        public static NDArray operator >>(NDArray lhs, object rhs)
        {
            return lhs >> np.asanyarray(rhs);
        }
    }
}
