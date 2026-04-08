using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Determines whether the given object represents a scalar dtype.
        /// </summary>
        /// <param name="rep">The object to check.</param>
        /// <returns>True if rep represents a scalar dtype.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.issctype.html
        /// </remarks>
        /// <example>
        /// <code>
        /// np.issctype(typeof(int))       // True
        /// np.issctype(NPTypeCode.Int32)  // True
        /// np.issctype(typeof(NDArray))   // False
        /// </code>
        /// </example>
        public static bool issctype(object rep)
        {
            if (rep == null)
                return false;

            return rep switch
            {
                NPTypeCode tc => tc != NPTypeCode.Empty && tc != NPTypeCode.String,
                Type t => t.GetTypeCode() != NPTypeCode.Empty,
                _ => false
            };
        }

        /// <summary>
        /// Returns True if dtype is of a specified category.
        /// </summary>
        /// <param name="dtype">The dtype to check.</param>
        /// <param name="kind">
        /// The dtype category. Can be:
        /// - "bool" - boolean
        /// - "integral" - integer types (signed and unsigned)
        /// - "real floating" - floating point types
        /// - "complex floating" - complex types
        /// - "numeric" - any numeric type
        /// </param>
        /// <returns>True if dtype belongs to the specified category.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.isdtype.html
        /// This is a NumPy 2.0+ function.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.isdtype(NPTypeCode.Int32, "integral")      // True
        /// np.isdtype(NPTypeCode.Double, "real floating") // True
        /// np.isdtype(NPTypeCode.Int32, "numeric")       // True
        /// </code>
        /// </example>
        public static bool isdtype(NPTypeCode dtype, string kind)
        {
            return kind.ToLowerInvariant() switch
            {
                "bool" or "boolean" => dtype == NPTypeCode.Boolean,
                "integral" or "integer" => dtype.IsInteger(),
                "real floating" or "floating" or "float" => dtype.IsFloatingPoint(),
                "complex floating" or "complex" => dtype == NPTypeCode.Complex,
                "numeric" => dtype.IsNumerical() && dtype != NPTypeCode.Boolean,
                _ => false
            };
        }

        /// <summary>
        /// Returns True if dtype is of any of the specified categories.
        /// </summary>
        /// <param name="dtype">The dtype to check.</param>
        /// <param name="kinds">Array of dtype categories to check.</param>
        /// <returns>True if dtype belongs to any of the specified categories.</returns>
        public static bool isdtype(NPTypeCode dtype, string[] kinds)
        {
            foreach (var kind in kinds)
            {
                if (isdtype(dtype, kind))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Determine if a class is a subclass of a second class.
        /// </summary>
        /// <param name="arg1">The dtype to check.</param>
        /// <param name="arg2">The dtype to compare against.</param>
        /// <returns>True if arg1 is a subtype of arg2.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.issubsctype.html
        /// </remarks>
        public static bool issubsctype(NPTypeCode arg1, NPTypeCode arg2)
        {
            return issubdtype(arg1, arg2);
        }

        /// <summary>
        /// Return the string representation of a scalar dtype.
        /// </summary>
        /// <param name="sctype">A scalar type.</param>
        /// <returns>The character code for the type.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.sctype2char.html
        ///
        /// Character codes:
        /// 'b' - boolean
        /// 'B' - unsigned byte
        /// 'h' - short (int16)
        /// 'H' - unsigned short
        /// 'i' or 'l' - int32
        /// 'I' or 'L' - uint32
        /// 'q' - int64
        /// 'Q' - uint64
        /// 'f' - float32
        /// 'd' - float64
        /// </remarks>
        /// <example>
        /// <code>
        /// np.sctype2char(NPTypeCode.Int32)   // 'i'
        /// np.sctype2char(NPTypeCode.Double)  // 'd'
        /// </code>
        /// </example>
        public static char sctype2char(NPTypeCode sctype)
        {
            return sctype switch
            {
                NPTypeCode.Boolean => 'b',
                NPTypeCode.Byte => 'B',
                NPTypeCode.Int16 => 'h',
                NPTypeCode.UInt16 => 'H',
                NPTypeCode.Int32 => 'i',
                NPTypeCode.UInt32 => 'I',
                NPTypeCode.Int64 => 'q',
                NPTypeCode.UInt64 => 'Q',
                NPTypeCode.Char => 'H',  // Char treated as uint16
                NPTypeCode.Single => 'f',
                NPTypeCode.Double => 'd',
                NPTypeCode.Decimal => 'd',  // Closest approximation
                NPTypeCode.Complex => 'D',
                _ => '?'
            };
        }

        /// <summary>
        /// Return the scalar type of highest precision of the same kind as the input.
        /// </summary>
        /// <param name="t">The input scalar type.</param>
        /// <returns>The highest precision type of the same kind.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.maximum_sctype.html
        /// </remarks>
        /// <example>
        /// <code>
        /// np.maximum_sctype(NPTypeCode.Int32)   // Int64
        /// np.maximum_sctype(NPTypeCode.Single)  // Double (or Decimal)
        /// </code>
        /// </example>
        public static NPTypeCode maximum_sctype(NPTypeCode t)
        {
            return t switch
            {
                // Boolean stays boolean
                NPTypeCode.Boolean => NPTypeCode.Boolean,

                // All signed integers -> int64
                NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64 => NPTypeCode.Int64,

                // All unsigned integers -> uint64
                NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64 => NPTypeCode.UInt64,

                // Char treated as unsigned integer
                NPTypeCode.Char => NPTypeCode.UInt64,

                // All floats -> highest precision (Decimal in NumSharp)
                NPTypeCode.Single or NPTypeCode.Double => NPTypeCode.Double,  // or Decimal for max precision
                NPTypeCode.Decimal => NPTypeCode.Decimal,

                // Complex stays complex
                NPTypeCode.Complex => NPTypeCode.Complex,

                _ => t
            };
        }
    }
}
