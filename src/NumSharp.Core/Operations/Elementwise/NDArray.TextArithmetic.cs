namespace NumSharp
{
    /// <summary>
    ///     The <c>+</c> and <c>*</c> operators with a C# <see cref="string"/> operand. A string array reads the string as
    ///     NumPy reads a Python <c>str</c>: a 0-d <c>U</c> scalar of its length, so <c>u + "xy"</c> is
    ///     <c>np.add(u, np.str_('xy'))</c> (NumPy's string concatenation, width <c>width(u) + 2</c>). Every other array keeps
    ///     NumSharp's legacy implicit <c>string → NDArray</c> conversion, exactly as before these overloads existed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why overloads.</b> Without them C# binds <c>u + "xy"</c> to <c>+(NDArray, NDArray)</c> through the implicit
    ///     conversion (a char array), which no string loop accepts. An exact <see cref="string"/> parameter wins.
    ///     </para>
    ///     <para>
    ///     <b>A string on the LEFT keeps the operand order.</b> Python's <c>"p" + u</c> asks <c>str.__add__</c> first, which
    ///     declines an array, then runs the array's <c>__radd__</c>, which is <c>np.add("p", u)</c> — the string stays the
    ///     FIRST operand (<c>'p' + ['ab']</c> is <c>['pab']</c>, probed 2.4.2). Unlike the comparisons, no operator is
    ///     swapped. A C# user-defined operator beats the language's own string concatenation here, so <c>"p" + u</c> never
    ///     becomes <c>"p" + u.ToString()</c>.
    ///     </para>
    ///     <para>
    ///     <b>The repeat needs <c>out=</c>.</b> NumPy's string <c>multiply</c> loop cannot size its result, so <c>u * 3</c>
    ///     and <c>3 * u</c> raise its TypeError ("The 'out' kwarg is necessary …", probed); use
    ///     <c>np.strings.multiply(u, 3)</c>. <c>u * "ab"</c> is NumPy's no-loop TypeError for two strings.
    ///     </para>
    /// </remarks>
    public partial class NDArray
    {
        /// <summary>Concatenation with a C# string (see the type remarks for how the string is read).</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The result of <c>np.add(lhs, rhs)</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array (NumPy refuses bytes plus str: "ufunc 'add' cannot use operands
        ///     with types …"), or a numeric array the legacy conversion cannot add to.</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray operator +(NDArray lhs, string rhs) => lhs + TextComparand(lhs, rhs);

        /// <summary>Concatenation of a C# string with an array: Python's <c>__radd__</c>, <c>np.add(lhs, rhs)</c> (see the type remarks).</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The result of <c>np.add(lhs, rhs)</c>, the string first.</returns>
        /// <exception cref="TypeError">An <c>S</c> array, or a numeric array the legacy conversion cannot add to.</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray operator +(string lhs, NDArray rhs) => TextComparand(rhs, lhs) + rhs;

        /// <summary>Multiplication by a C# string (see the type remarks): for a string array, NumPy's no-loop TypeError.</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The result of <c>np.multiply(lhs, rhs)</c> for a non-text array (the legacy conversion).</returns>
        /// <exception cref="TypeError">A string array (NumPy has no loop for two strings).</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray operator *(NDArray lhs, string rhs) => lhs * TextComparand(lhs, rhs);

        /// <summary>Multiplication of a C# string by an array (see the type remarks): Python's <c>__rmul__</c>, the string first.</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The result of <c>np.multiply(lhs, rhs)</c> for a non-text array (the legacy conversion).</returns>
        /// <exception cref="TypeError">A string array.</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray operator *(string lhs, NDArray rhs) => TextComparand(rhs, lhs) * rhs;
    }
}
