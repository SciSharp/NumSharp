using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns the data type with the smallest size and smallest scalar kind to which
        /// both type1 and type2 can be safely cast.
        /// </summary>
        /// <param name="type1">First data type.</param>
        /// <param name="type2">Second data type.</param>
        /// <returns>The promoted type.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.promote_types.html
        ///
        /// Unlike result_type, promote_types only considers types (not values),
        /// and always returns the smallest safe type.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.promote_types(NPTypeCode.Int32, NPTypeCode.Single)  // Double
        /// np.promote_types(NPTypeCode.Int16, NPTypeCode.UInt16)  // Int32
        /// np.promote_types(NPTypeCode.Int8, NPTypeCode.Int8)     // Int8
        /// </code>
        /// </example>
        public static NPTypeCode promote_types(NPTypeCode type1, NPTypeCode type2)
        {
            if (type1 == type2)
                return type1;

            // Use the existing type promotion infrastructure
            return _FindCommonArrayType(type1, type2);
        }

        /// <summary>
        /// Returns the data type with the smallest size and smallest scalar kind to which
        /// both type1 and type2 can be safely cast.
        /// </summary>
        /// <param name="type1">First CLR type.</param>
        /// <param name="type2">Second CLR type.</param>
        /// <returns>The promoted type as NPTypeCode.</returns>
        public static NPTypeCode promote_types(Type type1, Type type2)
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
        public static NPTypeCode promote_types<T1, T2>()
            where T1 : struct
            where T2 : struct
        {
            return promote_types(typeof(T1).GetTypeCode(), typeof(T2).GetTypeCode());
        }
    }
}
