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
        /// The descriptor arguments are promoted before the arrays and scalars, as NumPy's <c>array_result_type</c> does: the
        /// result does not depend on it, a refusal's text does (<c>np.result_type(text_array, np.float64)</c> names
        /// <c>Float64DType</c> as the DType that "could not be promoted by" <c>StringDType</c>).
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
            // array_result_type promotes the descriptor arguments before the arrays and scalars: only a refusal's text
            // shows the order (which DType "could not be promoted by" which, and the list of DTypes).
            return DTypePromotion.ResultType(DTypePromotion.InArrayResultTypeOrder(arrays_and_dtypes));
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
        /// <returns>
        /// The result descriptor — its CLASS included: an array of a C-type class contributes that class
        /// (<c>np.result_type(np.zeros(3, np.longdouble))</c> is <c>longdouble</c>), and NumPy's promotion can name one
        /// for two canonical classes (Windows' <c>np.result_type(int16_array, uint16_array)</c> is the C int class
        /// <c>'i'</c>, equal to int32).
        /// </returns>
        /// <exception cref="ArgumentException">No array.</exception>
        /// <exception cref="DTypePromotionError">No common dtype exists.</exception>
        public static DType result_type(params NDArray[] arrays)
        {
            if (arrays == null || arrays.Length == 0)
                throw new ArgumentException("At least one array must be provided", nameof(arrays));

            // One array: NumPy returns its descriptor made canonical (a native one is itself) — the class is kept, a
            // fixed-width string keeps its width.
            if (arrays.Length == 1)
            {
                var only = arrays[0].dtype;
                return only.Meta.ensure_canonical(only);
            }

            // Promote by DESCRIPTOR: a string's width (np.result_type(U3, S5) is <U5) and a lane-sharing C-type class
            // (np.longdouble) are not in the storage lane, so a type-code round trip would lose them.
            return DTypePromotion.ResultType(arrays.Cast<object>().ToArray());
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
        /// <returns>The result descriptor, its class included (see <see cref="result_type(NDArray[])"/>).</returns>
        /// <exception cref="ArgumentNullException">An array is null.</exception>
        /// <exception cref="DTypePromotionError">No common dtype exists.</exception>
        public static DType result_type(NDArray arr1, NDArray arr2)
        {
            if (arr1 is null)
                throw new ArgumentNullException(nameof(arr1));
            if (arr2 is null)
                throw new ArgumentNullException(nameof(arr2));
            var m1 = arr1.dtype.Meta;
            var m2 = arr2.dtype.Meta;
            // Two arrays of non-parametric classes: NumPy's result is the common class's default descriptor (its
            // singleton), which is what the sequence promotion below computes for two operands — without its
            // allocations, for every numeric pair. (Two arrays of NumSharp classes always have a common class.)
            if (!m1._parametric && !m2._parametric)
            {
                var common = ReferenceEquals(m1, m2) ? m1 : DTypePromotion.CommonDType(m1, m2);
                if (!common._parametric)
                    return common.default_descr();
            }
            // Promote by descriptor: a string's width and a lane-sharing C-type class are not in the storage lane.
            return DTypePromotion.ResultType(arr1, arr2);
        }
    }
}
