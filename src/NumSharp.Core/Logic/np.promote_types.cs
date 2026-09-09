using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns the data type with the smallest size and smallest scalar kind to which
        /// both type1 and type2 can be safely cast — NumPy's <c>np.promote_types(type1, type2)</c>
        /// (<c>PyArray_PromoteTypes</c>, NEP 42): the common DType CLASS of the two operands decides, a
        /// non-parametric class returns its default descriptor and a parametric one (datetime64 / timedelta64)
        /// returns the <c>common_instance</c> — the unit GCD (<c>promote_types('M8[s]', 'm8[ms]')</c> is <c>M8[ms]</c>,
        /// <c>promote_types('m8[10s]', 'm8[15s]')</c> is <c>m8[5s]</c>).
        /// </summary>
        /// <param name="type1">First data type (any spelling that converts to <see cref="DType"/>).</param>
        /// <param name="type2">Second data type.</param>
        /// <returns>The promoted descriptor.</returns>
        /// <exception cref="DTypePromotionError">No common dtype exists (<c>promote_types('M8[s]', 'f8')</c>) — NumPy's verbatim text.</exception>
        /// <exception cref="TypeError">Incompatible non-linear datetime units (<c>promote_types('m8[Y]', 'm8[D]')</c>) — verbatim.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.promote_types.html
        ///
        /// Unlike result_type, promote_types only considers types (not values),
        /// and always returns the smallest safe type. A non-native byte order is dropped (<c>promote_types('&gt;i4', '&gt;i4')</c>
        /// is the native <c>int32</c>), exactly as in NumPy.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.promote_types(DType.Int32, DType.Single)       // float64
        /// np.promote_types("M8[s]", "m8[ms]")               // datetime64[ms]
        /// np.promote_types("i1", "u1")                      // int16
        /// </code>
        /// </example>
        public static DType promote_types(DType type1, DType type2)
        {
            if (type1 is null) throw new ArgumentNullException(nameof(type1));
            if (type2 is null) throw new ArgumentNullException(nameof(type2));
            return DTypePromotion.PromoteTypes(type1, type2);
        }

        /// <summary>
        /// Returns the data type with the smallest size and smallest scalar kind to which
        /// both type1 and type2 can be safely cast.
        /// </summary>
        /// <param name="type1">First data type.</param>
        /// <param name="type2">Second data type.</param>
        /// <returns>The promoted type.</returns>
        /// <remarks>
        /// The <see cref="NPTypeCode"/> spelling of <see cref="promote_types(DType, DType)"/>: the same NEP 42 engine,
        /// whose answers for the storage-backed types ARE NumSharp's frozen promotion table.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.promote_types(NPTypeCode.Int32, NPTypeCode.Single)  // Double
        /// np.promote_types(NPTypeCode.Int16, NPTypeCode.UInt16)  // Int32
        /// np.promote_types(NPTypeCode.Int8, NPTypeCode.Int8)     // Int8
        /// </code>
        /// </example>
        public static DType promote_types(NPTypeCode type1, NPTypeCode type2)
        {
            if (type1 == type2)
                return type1;

            return DTypePromotion.PromoteTypes(DType.From(type1), DType.From(type2)).GetTypeCode();
        }

        /// <summary>
        /// Returns the data type with the smallest size and smallest scalar kind to which
        /// both type1 and type2 can be safely cast.
        /// </summary>
        /// <param name="type1">First CLR type.</param>
        /// <param name="type2">Second CLR type.</param>
        /// <returns>The promoted type as NPTypeCode.</returns>
        public static DType promote_types(Type type1, Type type2)
        {
            return promote_types(type1.GetTypeCode(), type2.GetTypeCode());
        }

        /// <summary>
        /// Returns the data type with the smallest size and smallest scalar kind to which
        /// both T1 and T2 can be safely cast.
        /// </summary>
        /// <typeparam name="T1">First type.</typeparam>
        /// <typeparam name="T2">Second type.</typeparam>
        /// <returns>The promoted type as NPTypeCode.</returns>
        /// <example>
        /// <code>
        /// np.promote_types&lt;int, long&gt;()      // Int64
        /// np.promote_types&lt;float, double&gt;()  // Double
        /// </code>
        /// </example>
        public static DType promote_types<T1, T2>()
            where T1 : struct
            where T2 : struct
        {
            return promote_types(typeof(T1).GetTypeCode(), typeof(T2).GetTypeCode());
        }
    }
}
