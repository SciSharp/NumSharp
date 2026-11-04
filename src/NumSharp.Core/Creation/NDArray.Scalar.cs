using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Creates a 0-d <see cref="NDArray"/> holding <paramref name="value"/> as <paramref name="dtype"/> — a NumPy
        ///     scalar of that dtype (<c>np.float32(2.5)</c>), a STRONG operand under NEP 50.
        /// </summary>
        /// <param name="value">The value of the scalar.</param>
        /// <param name="dtype">
        ///     The dtype of the scalar — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        /// </param>
        /// <returns>A new 0-d array the caller owns.</returns>
        /// <remarks>
        ///     <para>
        ///     A numeric <paramref name="dtype"/> converts <paramref name="value"/> with <see cref="Converts"/> (C#'s
        ///     conversions). A text <paramref name="dtype"/> stores it as NumPy's setitem does: a fixed-width <c>S</c> /
        ///     <c>U</c> dtype writes its text (a number's <c>str()</c>, cut to the width), and a NEP 55 <c>StringDType</c>
        ///     stores it by <c>stringdtype_setitem</c> (<c>np.array(5, dtype='T')</c> is <c>'5'</c>; the NA object, or
        ///     null for <c>None</c>, packs an NA element).
        ///     </para>
        ///     <para>
        ///     The result promotes like an array in every ufunc: <c>float32_array * NDArray.Scalar(2.0)</c> is float64, as
        ///     NumPy's <c>float32_array * np.float64(2.0)</c> is. A C# literal (<c>float32_array * 2.0</c>) is NumPy's weak
        ///     Python scalar instead and keeps float32 (parity QA-05).
        ///     </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">A non-text value into a <c>coerce=False</c> <c>StringDType</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8 into a <c>StringDType</c>.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate into a <c>StringDType</c>, or non-ASCII text
        ///     into <c>S</c>.</exception>
        /// <exception cref="InvalidCastException">A value <see cref="Converts"/> cannot convert to the dtype.</exception>
        [MethodImpl(Inline)]
        public static NDArray Scalar(object value, DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            return UnmanagedStorage.ScalarFromObject(value, dtype);
        }

        /// <summary>
        ///     Creates a 0-d <see cref="NDArray"/> holding <paramref name="value"/> in the dtype of its own C# type — a NumPy
        ///     scalar (a strong operand under NEP 50); a <see cref="string"/> is a <c>U</c> scalar, a <c>byte[]</c> an
        ///     <c>S</c> one.
        /// </summary>
        /// <param name="value">The value of the scalar: a C# number, a <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <returns>A new 0-d array the caller owns.</returns>
        /// <remarks>The result promotes like an array in every ufunc (<c>uint8_array + NDArray.Scalar(300L)</c> is int64);
        ///     a C# literal operand is NumPy's weak Python scalar instead (<c>uint8_array + 100</c> stays uint8).</remarks>
        /// <exception cref="NotSupportedException">A value of a type NumSharp has no dtype for.</exception>
        [MethodImpl(Inline)]
        public static NDArray Scalar(object value)
        {
            return UnmanagedStorage.ScalarFromObject(value);
        }

        /// <summary>
        ///     Creates a 0-d <see cref="NDArray"/> of <typeparamref name="T"/>'s dtype holding <paramref name="value"/> — a
        ///     NumPy scalar, a STRONG operand under NEP 50 (the library's computed scalars are built this way).
        /// </summary>
        /// <typeparam name="T">The element type (one of the storage types).</typeparam>
        /// <param name="value">The value of the scalar.</param>
        /// <returns>A new 0-d array the caller owns.</returns>
        /// <remarks>The result promotes like an array in every ufunc: <c>float16_array * NDArray.Scalar(0.5f)</c> is float32.
        ///     For an operand NumPy writes as a Python number (weak: it keeps the other operand's dtype), library code uses
        ///     <see cref="PythonLiteral{T}(T)"/>, the 0-d array a C# literal becomes through the implicit conversion.</remarks>
        [MethodImpl(Inline)]
        public static NDArray Scalar<T>(T value) where T : unmanaged
        {
            return new NDArray(UnmanagedStorage.Scalar(value));
        }

        /// <summary>
        ///     Creates a 0-d <see cref="NDArray"/> of <typeparamref name="T"/>'s dtype from <paramref name="value"/>, converted
        ///     to <typeparamref name="T"/> when it is another type — a NumPy scalar, a STRONG operand under NEP 50.
        /// </summary>
        /// <typeparam name="T">The element type (one of the storage types).</typeparam>
        /// <param name="value">The value of the scalar; a value of another type is converted with
        ///     <see cref="Converts.ChangeType{T}(object)"/>.</param>
        /// <returns>A new 0-d array the caller owns.</returns>
        /// <remarks>The result promotes like an array in every ufunc, as <see cref="Scalar{T}(T)"/>'s does.</remarks>
        /// <exception cref="InvalidCastException">A value <see cref="Converts"/> cannot convert to <typeparamref name="T"/>.</exception>
        [MethodImpl(Inline)]
        public static NDArray Scalar<T>(object value) where T : unmanaged
        {
            return new NDArray(UnmanagedStorage.Scalar(value as T? ?? Converts.ChangeType<T>(value)));
        }
    }
}
