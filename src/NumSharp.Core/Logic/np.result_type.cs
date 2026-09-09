using System;
using System.Linq;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the arguments — NumPy's <c>np.result_type(*arrays_and_dtypes)</c> (<c>PyArray_ResultType</c>) under
        /// NEP 50.
        /// </summary>
        /// <param name="arrays_and_dtypes">
        /// Arrays and/or dtype arguments: any mix of <see cref="NDArray"/>, <see cref="DType"/>, <see cref="DTypeMeta"/>
        /// (a bare class), <see cref="NPTypeCode"/>, <see cref="Type"/>, dtype string, and C# scalars.
        /// </param>
        /// <returns>The result descriptor.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.result_type.html
        ///
        /// <para>
        /// <b>NEP 50 in C#.</b> Every <see cref="NDArray"/> — 0-d included — contributes its dtype fully ("strong"), as do
        /// <see cref="DType"/>, <see cref="NPTypeCode"/>, <see cref="Type"/>, dtype strings, <c>bool</c>, <c>char</c>,
        /// <c>Half</c> and <c>decimal</c> values. A C# integer literal (<c>int</c>, <c>long</c>, …), a <c>float</c>/<c>double</c>
        /// or a <c>Complex</c> is WEAK — the analog of a Python <c>int</c>/<c>float</c>/<c>complex</c> — and only contributes its
        /// category: <c>result_type(int8_array, 300)</c> is <c>int8</c> (no value-based inspection), <c>result_type(int8_array,
        /// 1.5)</c> is <c>float64</c>, <c>result_type(float32_array, 1e300)</c> is <c>float32</c>; a lone literal falls back to
        /// <c>int64</c> / <c>float64</c> / <c>complex128</c>.
        /// </para>
        /// <para>
        /// The reduction over three or more operands is NumPy's order-independent <c>PyArray_PromoteDTypeSequence</c>
        /// (<c>result_type(i1, i1, f8, i1)</c> is <c>float64</c> in every order). Parametric operands resolve their
        /// <c>common_instance</c> (<c>result_type("M8[s]", "m8[ms]", "m8[us]")</c> is <c>datetime64[us]</c>).
        /// </para>
        /// </remarks>
        /// <exception cref="ValueError"><c>at least one array or dtype is required</c> (no operands).</exception>
        /// <exception cref="DTypePromotionError">No common dtype exists — NumPy's verbatim text, listing every operand class.</exception>
        /// <example>
        /// <code>
        /// np.result_type(NPTypeCode.Int32, NPTypeCode.Int64)    // int64
        /// np.result_type(a, 5)                                  // a.dtype (weak int)
        /// np.result_type(a, 5.0)                                // float64 for an integer a
        /// np.result_type("M8[s]", "m8[ms]")                     // datetime64[ms]
        /// </code>
        /// </example>
        public static DType result_type(params object[] arrays_and_dtypes)
        {
            if (arrays_and_dtypes == null || arrays_and_dtypes.Length == 0)
                throw new ValueError("at least one array or dtype is required");
            return DTypePromotion.ResultType(arrays_and_dtypes);
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the given descriptors (all strong).
        /// </summary>
        /// <param name="dtypes">One or more descriptors.</param>
        /// <returns>The result descriptor.</returns>
        public static DType result_type(params DType[] dtypes)
        {
            if (dtypes == null || dtypes.Length == 0)
                throw new ValueError("at least one array or dtype is required");
            return DTypePromotion.ResultType(dtypes);
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the given dtype strings (<c>np.result_type("i1", "f8")</c>). Exists so that a string argument binds the
        /// dtype grammar rather than NumSharp's string→<see cref="NDArray"/> (character array) conversion.
        /// </summary>
        public static DType result_type(params string[] dtypes)
        {
            if (dtypes == null || dtypes.Length == 0)
                throw new ValueError("at least one array or dtype is required");
            return DTypePromotion.ResultType(dtypes.Select(s => (object)dtype(s)).ToArray());
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the arguments.
        /// </summary>
        /// <param name="types">One or more NPTypeCode values.</param>
        /// <returns>The result type from combining the inputs.</returns>
        public static DType result_type(params NPTypeCode[] types)
        {
            if (types == null || types.Length == 0)
                throw new ArgumentException("At least one type must be provided", nameof(types));

            if (types.Length == 1)
                return types[0];

            return DTypePromotion.ResultType(types.Select(t => (object)DType.From(t)).ToArray()).GetTypeCode();
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the arguments. Every array — 0-d included — is a full (strong) participant, as in NumPy 2.x.
        /// </summary>
        /// <param name="arrays">One or more NDArray objects.</param>
        /// <returns>The result type from combining the array dtypes.</returns>
        public static DType result_type(params NDArray[] arrays)
        {
            if (arrays == null || arrays.Length == 0)
                throw new ArgumentException("At least one array must be provided", nameof(arrays));

            if (arrays.Length == 1)
                return arrays[0].GetTypeCode;

            return DTypePromotion.ResultType(arrays.Cast<object>().ToArray()).GetTypeCode();
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the two type codes. Convenience overload to avoid params array allocation.
        /// </summary>
        /// <param name="type1">First type code.</param>
        /// <param name="type2">Second type code.</param>
        /// <returns>The result type from combining the inputs.</returns>
        public static DType result_type(NPTypeCode type1, NPTypeCode type2)
        {
            if (type1 == type2)
                return type1;
            return DTypePromotion.PromoteTypes(DType.From(type1), DType.From(type2)).GetTypeCode();
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the two CLR types. Convenience overload to avoid params array allocation.
        /// </summary>
        /// <param name="type1">First CLR type.</param>
        /// <param name="type2">Second CLR type.</param>
        /// <returns>The result type from combining the inputs.</returns>
        public static DType result_type(Type type1, Type type2)
        {
            return result_type(type1.GetTypeCode(), type2.GetTypeCode());
        }

        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules
        /// to the two arrays. Convenience overload to avoid params array allocation.
        /// </summary>
        /// <param name="arr1">First array.</param>
        /// <param name="arr2">Second array.</param>
        /// <returns>The result type from combining the array dtypes.</returns>
        public static DType result_type(NDArray arr1, NDArray arr2)
        {
            if (arr1 is null)
                throw new ArgumentNullException(nameof(arr1));
            if (arr2 is null)
                throw new ArgumentNullException(nameof(arr2));
            return result_type(arr1.GetTypeCode, arr2.GetTypeCode);
        }
    }
}
