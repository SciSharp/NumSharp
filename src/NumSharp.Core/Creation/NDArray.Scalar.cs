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
        /// <returns>A new read-only 0-d array the caller owns: NumPy's scalar, which is immutable (see
        ///     <see cref="PyArray_Return"/>).</returns>
        /// <remarks>
        ///     <para>
        ///     A numeric <paramref name="dtype"/> converts <paramref name="value"/> with <see cref="Converts"/> (C#'s
        ///     conversions). A text <paramref name="dtype"/> stores it as NumPy's setitem does: a fixed-width <c>S</c> /
        ///     <c>U</c> dtype writes its text (a number's <c>str()</c>, cut to the width), and a NEP 55 <c>StringDType</c>
        ///     stores it by <c>stringdtype_setitem</c> (<c>np.array(5, dtype='T')</c> is <c>'5'</c>; the NA object, or
        ///     null for <c>None</c>, packs an NA element).
        ///     </para>
        ///     <para>
        ///     The result is read-only, as NumPy's scalar is: a write into it raises NumPy's
        ///     <c>ValueError: assignment destination is read-only</c>. Code that needs a writable 0-d buffer allocates one
        ///     (<c>np.zeros(Shape.Scalar, dtype)</c>, <c>np.array(x)</c>).
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
            // The construction is NumPy's PyArray_FromAny of one value; the fresh 0-d array becomes the scalar.
            return WritableScalar(value, dtype).PyArray_Return();
        }

        /// <summary>
        ///     Creates a 0-d <see cref="NDArray"/> holding <paramref name="value"/> in the dtype of its own C# type — a NumPy
        ///     scalar (a strong operand under NEP 50); a <see cref="string"/> is a <c>U</c> scalar, a <c>byte[]</c> an
        ///     <c>S</c> one.
        /// </summary>
        /// <param name="value">The value of the scalar: a C# number, a <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <returns>A new read-only 0-d array the caller owns: NumPy's scalar, which is immutable (see
        ///     <see cref="PyArray_Return"/>).</returns>
        /// <remarks>The result promotes like an array in every ufunc (<c>uint8_array + NDArray.Scalar(300L)</c> is int64);
        ///     a C# literal operand is NumPy's weak Python scalar instead (<c>uint8_array + 100</c> stays uint8). A write into
        ///     the result raises NumPy's <c>ValueError: assignment destination is read-only</c>.</remarks>
        /// <exception cref="NotSupportedException">A value of a type NumSharp has no dtype for.</exception>
        [MethodImpl(Inline)]
        public static NDArray Scalar(object value)
        {
            return WritableScalar(value).PyArray_Return();
        }

        /// <summary>
        ///     Creates a 0-d <see cref="NDArray"/> of <typeparamref name="T"/>'s dtype holding <paramref name="value"/> — a
        ///     NumPy scalar, a STRONG operand under NEP 50 (the library's computed scalars are built this way).
        /// </summary>
        /// <typeparam name="T">The element type (one of the storage types).</typeparam>
        /// <param name="value">The value of the scalar.</param>
        /// <returns>A new read-only 0-d array the caller owns: NumPy's scalar, which is immutable (see
        ///     <see cref="PyArray_Return"/>).</returns>
        /// <remarks>The result promotes like an array in every ufunc: <c>float16_array * NDArray.Scalar(0.5f)</c> is float32.
        ///     For an operand NumPy writes as a Python number (weak: it keeps the other operand's dtype), library code uses
        ///     <see cref="PythonLiteral{T}(T)"/>, the 0-d array a C# literal becomes through the implicit conversion. A write
        ///     into the result raises NumPy's <c>ValueError: assignment destination is read-only</c>; code that needs a
        ///     writable 0-d buffer allocates one (<c>new NDArray(typecode, Shape.Scalar)</c>).</remarks>
        [MethodImpl(Inline)]
        public static NDArray Scalar<T>(T value) where T : unmanaged
        {
            return WritableScalar(value).PyArray_Return();
        }

        /// <summary>
        ///     Creates a 0-d <see cref="NDArray"/> of <typeparamref name="T"/>'s dtype from <paramref name="value"/>, converted
        ///     to <typeparamref name="T"/> when it is another type — a NumPy scalar, a STRONG operand under NEP 50.
        /// </summary>
        /// <typeparam name="T">The element type (one of the storage types).</typeparam>
        /// <param name="value">The value of the scalar; a value of another type is converted with
        ///     <see cref="Converts.ChangeType{T}(object)"/>.</param>
        /// <returns>A new read-only 0-d array the caller owns: NumPy's scalar, which is immutable (see
        ///     <see cref="PyArray_Return"/>).</returns>
        /// <remarks>The result promotes like an array in every ufunc, as <see cref="Scalar{T}(T)"/>'s does, and is
        ///     read-only like it.</remarks>
        /// <exception cref="InvalidCastException">A value <see cref="Converts"/> cannot convert to <typeparamref name="T"/>.</exception>
        [MethodImpl(Inline)]
        public static NDArray Scalar<T>(object value) where T : unmanaged
        {
            return WritableScalar<T>(value).PyArray_Return();
        }

        // ---------------------------------------------------------------------------------------------------------------
        // The writable twins. NumPy builds a scalar in two steps: a fresh 0-d ARRAY first (PyArray_FromAny, a reduction's
        // output), then PyArray_Return / PyArray_ToScalar hands out the scalar. Library code that needs the first step
        // alone - an engine's 0-d intermediate that a keepdims exit reshapes into (1, 1) or a kernel writes into, the 0-d
        // ARRAY np.array(5) / np.asarray(5) returns, a sampler's size=() result - builds it with these, never with the
        // read-only Scalar(...) above.
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The writable twin of <see cref="Scalar(object, DType)"/>: a new 0-d array holding <paramref name="value"/>
        ///     as <paramref name="dtype"/>, still WRITEABLE - the fresh array NumPy's scalar is built in, not the scalar.
        /// </summary>
        /// <param name="value">The value (converted as <see cref="Scalar(object, DType)"/> converts it).</param>
        /// <param name="dtype">The dtype descriptor.</param>
        /// <returns>A new writeable 0-d array the caller owns; <see cref="IsNumPyScalar"/> is true for it, but it becomes
        ///     NumPy's immutable scalar only through <see cref="PyArray_Return"/>.</returns>
        /// <remarks>Footgun: never hand this out where NumPy returns a scalar (the caller could write into "the scalar");
        ///     call <see cref="PyArray_Return"/> at that exit, or use <see cref="Scalar(object, DType)"/>.</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">A non-text value into a <c>coerce=False</c> <c>StringDType</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8 into a <c>StringDType</c>.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate into a <c>StringDType</c>, or non-ASCII text
        ///     into <c>S</c>.</exception>
        /// <exception cref="InvalidCastException">A value <see cref="Converts"/> cannot convert to the dtype.</exception>
        [MethodImpl(Inline)]
        internal static NDArray WritableScalar(object value, DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            return UnmanagedStorage.ScalarFromObject(value, dtype);
        }

        /// <summary>
        ///     The writable twin of <see cref="Scalar(object)"/>: a new 0-d array of <paramref name="value"/>'s own dtype,
        ///     still WRITEABLE (see <see cref="WritableScalar(object, DType)"/> for when to use it).
        /// </summary>
        /// <param name="value">The value: a C# number, a <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <returns>A new writeable 0-d array the caller owns.</returns>
        /// <exception cref="NotSupportedException">A value of a type NumSharp has no dtype for.</exception>
        [MethodImpl(Inline)]
        internal static NDArray WritableScalar(object value)
        {
            return UnmanagedStorage.ScalarFromObject(value);
        }

        /// <summary>
        ///     The writable twin of <see cref="Scalar{T}(T)"/>: a new 0-d array of <typeparamref name="T"/>'s dtype, still
        ///     WRITEABLE (see <see cref="WritableScalar(object, DType)"/> for when to use it).
        /// </summary>
        /// <typeparam name="T">The element type (one of the storage types).</typeparam>
        /// <param name="value">The value.</param>
        /// <returns>A new writeable 0-d array the caller owns.</returns>
        [MethodImpl(Inline)]
        internal static NDArray WritableScalar<T>(T value) where T : unmanaged
        {
            return new NDArray(UnmanagedStorage.Scalar(value));
        }

        /// <summary>
        ///     The writable twin of <see cref="Scalar{T}(object)"/>: <paramref name="value"/> converted to
        ///     <typeparamref name="T"/> into a new 0-d array, still WRITEABLE (see <see cref="WritableScalar(object, DType)"/>
        ///     for when to use it).
        /// </summary>
        /// <typeparam name="T">The element type (one of the storage types).</typeparam>
        /// <param name="value">The value; a value of another type is converted with
        ///     <see cref="Converts.ChangeType{T}(object)"/>.</param>
        /// <returns>A new writeable 0-d array the caller owns.</returns>
        /// <exception cref="InvalidCastException">A value <see cref="Converts"/> cannot convert to <typeparamref name="T"/>.</exception>
        [MethodImpl(Inline)]
        internal static NDArray WritableScalar<T>(object value) where T : unmanaged
        {
            return new NDArray(UnmanagedStorage.Scalar(value as T? ?? Converts.ChangeType<T>(value)));
        }
    }
}
