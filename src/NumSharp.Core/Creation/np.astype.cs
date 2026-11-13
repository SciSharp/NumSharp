using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Copies an array to a specified data type — NumPy 2.1's Array-API function <c>np.astype(x, dtype, /, *,
        ///     copy=True, device=None)</c>, the function form of <see cref="NDArray.astype(DType, char, string, bool)"/>.
        /// </summary>
        /// <param name="x">The array to cast. NumPy takes an ndarray or a NumPy scalar only (a Python list is its
        ///     TypeError); the <see cref="NDArray"/> parameter says so at compile time.</param>
        /// <param name="dtype">The target dtype — a <see cref="DType"/>, a C# <see cref="System.Type"/>, an
        ///     <see cref="NPTypeCode"/> or a dtype string (<c>"f4"</c>) convert implicitly; null is NumPy's
        ///     <c>dtype=None</c>, the default float64.</param>
        /// <param name="copy">True (the default) always returns a new array; false returns <paramref name="x"/> ITSELF when
        ///     its dtype already is <paramref name="dtype"/>, and a new array otherwise (a cast must copy).</param>
        /// <param name="device">NumPy's Array-API device: null or <c>"cpu"</c>; anything else is NumPy's ValueError.</param>
        /// <returns><paramref name="x"/> cast to <paramref name="dtype"/> with the unsafe rule (a float into an int
        ///     truncates), in <paramref name="x"/>'s own layout (the method's <c>order='K'</c>); <paramref name="x"/> itself
        ///     for a satisfied <c>copy: false</c>. A subclass operand (<c>numpy.matrix</c>) keeps its class, as NumPy's
        ///     <c>x.astype(...)</c> dispatch does.</returns>
        /// <exception cref="TypeError"><paramref name="x"/> is null: NumPy's <c>Input should be a NumPy array or scalar. It
        ///     is a &lt;class 'NoneType'&gt; instead.</c></exception>
        /// <exception cref="ValueError"><paramref name="device"/> is not <c>"cpu"</c>: <c>Device not understood. Only "cpu"
        ///     is allowed, but received: gpu</c> (checked before the cast, as in NumPy).</exception>
        /// <exception cref="System.NotSupportedException">A dtype NumSharp has no storage for (the method's refusal).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.astype.html — NumPy's body is the two checks, then
        ///     <c>x.astype(dtype, copy=copy)</c>: the method's defaults <c>order='K'</c> and <c>casting='unsafe'</c> apply, so
        ///     this function has neither keyword. Added in API-parity batch 7.
        ///     <para>NumPy's method allocates with <c>subok=True</c>, so the result is created as <paramref name="x"/>'s class
        ///     (<c>np.astype(matrix, float)</c> is a matrix): the cast runs in <see cref="NDArray.AsTypeComputation"/>, whose
        ///     <c>[NewSubtype("x")]</c> tells NumSharp.Build to create a subclass operand's result as its class (the weaver
        ///     reads the declaration on the computation the public function calls).</para>
        /// </remarks>
        public static NDArray astype(NDArray x, DType dtype, bool copy = true, string device = null)
        {
            // NumPy's checks in its order: the input type first (only None can fail here; C#'s type refuses the rest),
            // then the Array-API device, and only then the cast.
            if (x is null)
                throw new TypeError("Input should be a NumPy array or scalar. It is a <class 'NoneType'> instead.");
            if (device is not null && device != "cpu")
                throw new ValueError($"Device not understood. Only \"cpu\" is allowed, but received: {device}");
            return NDArray.AsTypeComputation(x, dtype, copy);
        }
    }

    public partial class NDArray
    {
        /// <summary>
        ///     <see cref="np.astype(NDArray, DType, bool, string)"/>'s computation: <c>x.astype(dtype, copy=copy)</c> with the
        ///     method's defaults (<c>order='K'</c>, <c>casting='unsafe'</c>). Its <c>[NewSubtype("x")]</c> is NumPy's
        ///     <c>subok=True</c> allocation: NumSharp.Build hands it the plain view of a subclass operand and creates the
        ///     result as the operand's class.
        /// </summary>
        /// <param name="x">The array to cast (a plain array when the public function's operand is a subclass).</param>
        /// <param name="dtype">The target dtype; null is float64.</param>
        /// <param name="copy">False returns <paramref name="x"/> itself when its dtype already matches.</param>
        /// <returns>The cast array, or <paramref name="x"/> for a satisfied <c>copy: false</c>.</returns>
        /// <exception cref="NotSupportedException">A dtype NumSharp has no storage for.</exception>
        [NewSubtype("x")]
        internal static NDArray AsTypeComputation(NDArray x, DType dtype, bool copy)
            => x.astype(dtype, copy: copy);
    }
}
