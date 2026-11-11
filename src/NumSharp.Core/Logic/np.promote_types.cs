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
        /// <exception cref="TypeError">
        ///     Incompatible non-linear datetime units (<c>promote_types('m8[Y]', 'm8[D]')</c>) — verbatim; or a null type
        ///     (NumPy's <c>None</c>): "did not understand one of the types".
        /// </exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.promote_types.html
        ///
        /// Unlike result_type, promote_types only considers types (not values),
        /// and always returns the smallest safe type. A non-native byte order is dropped (<c>promote_types('&gt;i4', '&gt;i4')</c>
        /// is the native <c>int32</c>), exactly as in NumPy.
        /// <para>One signature, NumPy's (2026-11-11): an <see cref="NPTypeCode"/>, a CLR <see cref="Type"/> and a dtype
        /// string convert to <see cref="DType"/>, so the former NPTypeCode / Type / generic <c>&lt;T1, T2&gt;</c> overloads
        /// are gone (<c>promote_types&lt;int, long&gt;()</c> is <c>promote_types(typeof(int), typeof(long))</c>); every
        /// NPTypeCode pair answers as the frozen table did (a 15 x 15 differential).</para>
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
            // array_promote_types converts both with PyArray_DescrConverter2, which turns None into NULL, then refuses a
            // NULL with this TypeError — None is not float64 here, unlike np.dtype(None).
            if (type1 is null || type2 is null)
                throw new TypeError("did not understand one of the types");
            return DTypePromotion.PromoteTypes(type1, type2);
        }
    }
}
