using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns True if first argument is a typecode lower/equal in type hierarchy.
        /// </summary>
        /// <param name="arg1">dtype or string - dtype or string representing a typecode.</param>
        /// <param name="arg2">dtype or string - dtype or string representing a typecode.</param>
        /// <returns>True if arg1 is a subtype of arg2.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html
        ///
        /// Type hierarchy in NumPy:
        /// - generic → number → integer/inexact
        /// - integer → signedinteger/unsignedinteger
        /// - inexact → floating/complexfloating
        ///
        /// Note: In NumPy 2.x, bool is NOT a subtype of integer.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.issubdtype(NPTypeCode.Int32, "integer")     // True
        /// np.issubdtype(NPTypeCode.Double, "floating")   // True
        /// np.issubdtype(NPTypeCode.Boolean, "integer")   // False (NumPy 2.x)
        /// </code>
        /// </example>
        public static bool issubdtype(NPTypeCode arg1, string arg2)
        {
            return arg2.ToLowerInvariant() switch
            {
                "generic" => true,  // All types are subtypes of generic
                "number" => IsNumber(arg1),
                "integer" => IsIntegerCategory(arg1),
                "signedinteger" or "signed" => IsSignedInteger(arg1),
                "unsignedinteger" or "unsigned" => IsUnsignedInteger(arg1),
                "inexact" => IsInexact(arg1),
                "floating" or "float" => IsFloating(arg1),
                "complexfloating" or "complex" => arg1 == NPTypeCode.Complex,
                "bool" or "boolean" => arg1 == NPTypeCode.Boolean,
                _ => false
            };
        }

        /// <summary>
        /// Returns True if first argument is a typecode lower/equal in type hierarchy.
        /// </summary>
        /// <param name="arg1">dtype - dtype representing a typecode.</param>
        /// <param name="arg2">dtype - dtype representing a typecode.</param>
        /// <returns>True if arg1 is equal to or a subtype of arg2.</returns>
        public static bool issubdtype(NPTypeCode arg1, NPTypeCode arg2)
        {
            if (arg1 == arg2)
                return true;

            // Check if arg1 is in the same category or subcategory as arg2
            return (arg2, arg1) switch
            {
                // Integer hierarchy
                (NPTypeCode.Int64, _) when IsSignedInteger(arg1) => true,
                (NPTypeCode.UInt64, _) when IsUnsignedInteger(arg1) => true,

                // Floating hierarchy
                (NPTypeCode.Double, NPTypeCode.Single) => true,
                (NPTypeCode.Decimal, NPTypeCode.Double) => true,
                (NPTypeCode.Decimal, NPTypeCode.Single) => true,

                _ => false
            };
        }

        /// <summary>
        /// Returns True if first argument is a typecode lower/equal in type hierarchy.
        /// </summary>
        /// <param name="arg1">Type - CLR type representing a typecode.</param>
        /// <param name="arg2">string - string representing a typecode category.</param>
        /// <returns>True if arg1 is a subtype of arg2 category.</returns>
        public static bool issubdtype(Type arg1, string arg2)
        {
            return issubdtype(arg1.GetTypeCode(), arg2);
        }

        // Helper methods for type categorization

        /// <summary>
        /// Check if type is a number (excludes bool, char, string).
        /// </summary>
        private static bool IsNumber(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Byte => true,
                NPTypeCode.Int16 or NPTypeCode.UInt16 => true,
                NPTypeCode.Int32 or NPTypeCode.UInt32 => true,
                NPTypeCode.Int64 or NPTypeCode.UInt64 => true,
                NPTypeCode.Single or NPTypeCode.Double => true,
                NPTypeCode.Decimal => true,
                NPTypeCode.Complex => true,
                _ => false  // Boolean, Char, String are not numbers
            };
        }

        /// <summary>
        /// Check if type is an integer category (excludes bool in NumPy 2.x).
        /// </summary>
        private static bool IsIntegerCategory(NPTypeCode type)
        {
            return IsSignedInteger(type) || IsUnsignedInteger(type);
        }

        /// <summary>
        /// Check if type is a signed integer.
        /// </summary>
        private static bool IsSignedInteger(NPTypeCode type)
        {
            return type is NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64;
        }

        /// <summary>
        /// Check if type is an unsigned integer (includes Byte but not Boolean/Char).
        /// </summary>
        private static bool IsUnsignedInteger(NPTypeCode type)
        {
            return type is NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64;
        }

        /// <summary>
        /// Check if type is inexact (floating or complex).
        /// </summary>
        private static bool IsInexact(NPTypeCode type)
        {
            return IsFloating(type) || type == NPTypeCode.Complex;
        }

        /// <summary>
        /// Check if type is floating point.
        /// </summary>
        private static bool IsFloating(NPTypeCode type)
        {
            return type is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal;
        }
    }
}
