using NumSharp.Generic;

namespace NumSharp
{
    /// <summary>
    ///     The comparison operators with a C# <see cref="string"/> operand. A string array compares against the string as
    ///     NumPy compares against a Python <c>str</c>: a 0-d <c>U</c> scalar of its length (<c>u == "abc"</c> is
    ///     <c>np.equal(u, np.str_('abc'))</c>). Every other array keeps NumSharp's legacy implicit <c>string → NDArray</c>
    ///     conversion (a char array, or a parsed <c>"[1, 2]"</c> matrix), exactly as before these overloads existed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why overloads.</b> Without them C# binds <c>u == "abc"</c> to <c>==(NDArray, NDArray)</c> through the implicit
    ///     conversion (a user-defined conversion to the more specific target beats the reference conversion to
    ///     <see cref="object"/>), so the text would never reach the string comparison as NumPy's scalar. An exact
    ///     <see cref="string"/> parameter wins over both.
    ///     </para>
    ///     <para>
    ///     <b>A string on the LEFT is Python's reflection.</b> <c>"b" &lt; a</c> in Python first asks <c>str.__lt__</c>, which
    ///     returns NotImplemented for an array, so Python calls the array's reflected operator: <c>a &gt; "b"</c>, i.e.
    ///     <c>np.greater(a, "b")</c>. The overloads with the string first evaluate exactly that (<c>&lt;</c> ↔ <c>&gt;</c>,
    ///     <c>&lt;=</c> ↔ <c>&gt;=</c>, <c>==</c> and <c>!=</c> unchanged). The values are the same either way; what the
    ///     reflection decides is the error: an <c>S</c> array reports NumPy's own text, which names the reflected ufunc and
    ///     the array's dtype first (<c>"" &lt; s</c> raises "ufunc 'greater' did not contain a loop with signature matching
    ///     types (BytesDType, StrDType)", probed 2.4.2).
    ///     </para>
    ///     <para>
    ///     <b>Footguns.</b> A <c>null</c> literal now binds these overloads (<see cref="string"/> is the most specific target)
    ///     and keeps the old answer (<c>null</c> compares as a null array: <c>Scalar(false)</c>, or <c>true</c> for two nulls
    ///     where the NDArray overload says so). A <c>byte[]</c> is still a numeric <c>uint8</c> array, not NumPy's
    ///     <c>bytes</c>: build an <c>S</c> scalar with <see cref="Scalar(object)"/> to compare an <c>S</c> array. An
    ///     <c>S</c> array against a C# string is NumPy's bytes-against-str: <c>==</c> is all False, <c>!=</c> all True, the
    ///     ordering operators raise NumPy's no-loop TypeError.
    ///     </para>
    /// </remarks>
    public partial class NDArray
    {
        /// <summary>
        ///     The NDArray a C# string stands for when compared with <paramref name="array"/>: NumPy's <c>str</c> scalar (a
        ///     fresh 0-d <c>U</c> array) for a string array, the legacy implicit conversion otherwise, null for a null string.
        /// </summary>
        /// <param name="array">The other operand (any; null keeps the legacy conversion).</param>
        /// <param name="text">The C# string, or null.</param>
        /// <returns>The comparand (a temporary the calling operator's scope reclaims), or null.</returns>
        private static NDArray TextComparand(NDArray array, string text)
        {
            if (text is null)
                return null;
            if (array is not null && array.typecode == NPTypeCode.String)
                return Scalar(text);
            return text;   // NumSharp's legacy implicit string → NDArray conversion, unchanged for non-text operands
        }

        /// <summary>Element-wise equality with a C# string (see the type remarks for how the string is read).</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The bool result of <c>lhs == rhs</c>.</returns>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast with
        ///     <paramref name="lhs"/> (a non-text array).</exception>
        [NDScoped]
        public static NDArray<bool> operator ==(NDArray lhs, string rhs) => lhs == TextComparand(lhs, rhs);

        /// <summary>Element-wise equality of a C# string with an array: Python's reflection <c>rhs == lhs</c> (see the type remarks).</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The bool result of <c>lhs == rhs</c>.</returns>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator ==(string lhs, NDArray rhs) => rhs == TextComparand(rhs, lhs);

        /// <summary>Element-wise inequality with a C# string (see the type remarks).</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The bool result of <c>lhs != rhs</c>.</returns>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator !=(NDArray lhs, string rhs) => lhs != TextComparand(lhs, rhs);

        /// <summary>Element-wise inequality of a C# string with an array: Python's reflection <c>rhs != lhs</c> (see the type remarks).</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The bool result of <c>lhs != rhs</c>.</returns>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator !=(string lhs, NDArray rhs) => rhs != TextComparand(rhs, lhs);

        /// <summary>Element-wise less-than against a C# string (see the type remarks).</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The bool result of <c>lhs &lt; rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array (NumPy has no less loop for bytes against str).</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator <(NDArray lhs, string rhs) => lhs < TextComparand(lhs, rhs);

        /// <summary>Element-wise less-than of a C# string against an array: Python's reflection <c>rhs &gt; lhs</c> (see the type remarks).</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The bool result of <c>lhs &lt; rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array (NumPy's text names <c>greater</c>, the reflected ufunc).</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator <(string lhs, NDArray rhs) => rhs > TextComparand(rhs, lhs);

        /// <summary>Element-wise less-or-equal against a C# string (see the type remarks).</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The bool result of <c>lhs &lt;= rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array.</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator <=(NDArray lhs, string rhs) => lhs <= TextComparand(lhs, rhs);

        /// <summary>Element-wise less-or-equal of a C# string against an array: Python's reflection <c>rhs &gt;= lhs</c> (see the type remarks).</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The bool result of <c>lhs &lt;= rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array (NumPy's text names <c>greater_equal</c>, the reflected ufunc).</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator <=(string lhs, NDArray rhs) => rhs >= TextComparand(rhs, lhs);

        /// <summary>Element-wise greater-than against a C# string (see the type remarks).</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The bool result of <c>lhs &gt; rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array.</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator >(NDArray lhs, string rhs) => lhs > TextComparand(lhs, rhs);

        /// <summary>Element-wise greater-than of a C# string against an array: Python's reflection <c>rhs &lt; lhs</c> (see the type remarks).</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The bool result of <c>lhs &gt; rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array (NumPy's text names <c>less</c>, the reflected ufunc).</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator >(string lhs, NDArray rhs) => rhs < TextComparand(rhs, lhs);

        /// <summary>Element-wise greater-or-equal against a C# string (see the type remarks).</summary>
        /// <param name="lhs">The array.</param>
        /// <param name="rhs">The string, or null.</param>
        /// <returns>The bool result of <c>lhs &gt;= rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array.</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="rhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator >=(NDArray lhs, string rhs) => lhs >= TextComparand(lhs, rhs);

        /// <summary>Element-wise greater-or-equal of a C# string against an array: Python's reflection <c>rhs &lt;= lhs</c> (see the type remarks).</summary>
        /// <param name="lhs">The string, or null.</param>
        /// <param name="rhs">The array.</param>
        /// <returns>The bool result of <c>lhs &gt;= rhs</c>.</returns>
        /// <exception cref="TypeError">An <c>S</c> array (NumPy's text names <c>less_equal</c>, the reflected ufunc).</exception>
        /// <exception cref="IncorrectShapeException">The legacy conversion of <paramref name="lhs"/> does not broadcast.</exception>
        [NDScoped]
        public static NDArray<bool> operator >=(string lhs, NDArray rhs) => rhs <= TextComparand(rhs, lhs);
    }
}
