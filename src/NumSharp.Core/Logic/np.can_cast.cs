using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns True if cast between data types can occur according to the casting rule.
        /// </summary>
        /// <param name="from">Data type to cast from.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur:
        /// - "no" means the data types should not be cast at all.
        /// - "equiv" means only byte-order changes are allowed.
        /// - "safe" means only casts which can preserve values are allowed.
        /// - "same_kind" means only safe casts or casts within a kind (int to int, float to float) are allowed.
        /// - "unsafe" means any data conversions may be done.
        /// </param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html
        /// </remarks>
        /// <example>
        /// <code>
        /// np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64)           // True
        /// np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32)           // False
        /// np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind")  // False (different kind)
        /// np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "unsafe") // True
        /// </code>
        /// </example>
        public static bool can_cast(NPTypeCode from, NPTypeCode to, string casting = "safe")
        {
            if (from == to)
                return true;

            return casting.ToLowerInvariant() switch
            {
                "no" => false,  // No casting allowed except same type
                "equiv" => false,  // Only byte-order changes (not applicable in NumSharp)
                "safe" => CanCastSafe(from, to),
                "same_kind" => CanCastSameKind(from, to),
                "unsafe" => true,  // Any cast allowed
                _ => throw new ArgumentException($"Invalid casting rule: {casting}", nameof(casting))
            };
        }

        /// <summary>
        /// Returns True if cast between data types can occur according to the casting rule.
        /// </summary>
        /// <param name="from">CLR type to cast from.</param>
        /// <param name="to">CLR type to cast to.</param>
        /// <param name="casting">Controls what kind of data casting may occur.</param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        public static bool can_cast(Type from, Type to, string casting = "safe")
        {
            return can_cast(from.GetTypeCode(), to.GetTypeCode(), casting);
        }

        /// <summary>
        /// Returns True if the int value can be cast to the data type according to the casting rule.
        /// </summary>
        /// <param name="value">Int value to check.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">Controls what kind of data casting may occur.</param>
        /// <returns>True if the value can be cast to the target type.</returns>
        public static bool can_cast(int value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the long value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(long value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the double value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(double value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the scalar value can be cast to the data type according to the casting rule.
        /// </summary>
        /// <param name="value">Scalar value to check.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">Controls what kind of data casting may occur.</param>
        /// <returns>True if the value can be cast to the target type.</returns>
        /// <remarks>
        /// Scalar values can often be cast to smaller types if the value fits.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.can_cast(100, NPTypeCode.Byte)    // True (100 fits in byte)
        /// np.can_cast(1000, NPTypeCode.Byte)   // False (1000 > 255)
        /// </code>
        /// </example>
        public static bool can_cast(object value, NPTypeCode to, string casting = "safe")
        {
            if (value == null)
                return false;

            var from = value.GetType().GetTypeCode();

            // For "unsafe" casting, any value can be cast
            if (casting.ToLowerInvariant() == "unsafe")
                return true;

            // Check if the value fits in the target type
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if cast from array dtype can occur according to the casting rule.
        /// </summary>
        /// <param name="from">Array to cast from.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">Controls what kind of data casting may occur.</param>
        /// <returns>True if the array can be cast to the target type.</returns>
        public static bool can_cast(NDArray from, NPTypeCode to, string casting = "safe")
        {
            // Arrays use type-based casting, not value-based
            return can_cast(from.GetTypeCode, to, casting);
        }

        /// <summary>
        /// Check if casting from one type to another preserves values (safe casting).
        /// </summary>
        private static bool CanCastSafe(NPTypeCode from, NPTypeCode to)
        {
            // Same type always safe
            if (from == to)
                return true;

            // Safe casting table based on NumPy rules
            return (from, to) switch
            {
                // Boolean can safely cast to any integer or float
                (NPTypeCode.Boolean, NPTypeCode.Byte) => true,
                (NPTypeCode.Boolean, NPTypeCode.Int16) => true,
                (NPTypeCode.Boolean, NPTypeCode.UInt16) => true,
                (NPTypeCode.Boolean, NPTypeCode.Int32) => true,
                (NPTypeCode.Boolean, NPTypeCode.UInt32) => true,
                (NPTypeCode.Boolean, NPTypeCode.Int64) => true,
                (NPTypeCode.Boolean, NPTypeCode.UInt64) => true,
                (NPTypeCode.Boolean, NPTypeCode.Single) => true,
                (NPTypeCode.Boolean, NPTypeCode.Double) => true,
                (NPTypeCode.Boolean, NPTypeCode.Decimal) => true,

                // Byte (uint8) safe casts
                (NPTypeCode.Byte, NPTypeCode.Int16) => true,
                (NPTypeCode.Byte, NPTypeCode.UInt16) => true,
                (NPTypeCode.Byte, NPTypeCode.Int32) => true,
                (NPTypeCode.Byte, NPTypeCode.UInt32) => true,
                (NPTypeCode.Byte, NPTypeCode.Int64) => true,
                (NPTypeCode.Byte, NPTypeCode.UInt64) => true,
                (NPTypeCode.Byte, NPTypeCode.Single) => true,
                (NPTypeCode.Byte, NPTypeCode.Double) => true,
                (NPTypeCode.Byte, NPTypeCode.Decimal) => true,

                // Int16 safe casts
                (NPTypeCode.Int16, NPTypeCode.Int32) => true,
                (NPTypeCode.Int16, NPTypeCode.Int64) => true,
                (NPTypeCode.Int16, NPTypeCode.Single) => true,
                (NPTypeCode.Int16, NPTypeCode.Double) => true,
                (NPTypeCode.Int16, NPTypeCode.Decimal) => true,

                // UInt16 safe casts
                (NPTypeCode.UInt16, NPTypeCode.Int32) => true,
                (NPTypeCode.UInt16, NPTypeCode.UInt32) => true,
                (NPTypeCode.UInt16, NPTypeCode.Int64) => true,
                (NPTypeCode.UInt16, NPTypeCode.UInt64) => true,
                (NPTypeCode.UInt16, NPTypeCode.Single) => true,
                (NPTypeCode.UInt16, NPTypeCode.Double) => true,
                (NPTypeCode.UInt16, NPTypeCode.Decimal) => true,

                // Int32 safe casts
                (NPTypeCode.Int32, NPTypeCode.Int64) => true,
                (NPTypeCode.Int32, NPTypeCode.Double) => true,  // Float64 can represent all int32
                (NPTypeCode.Int32, NPTypeCode.Decimal) => true,

                // UInt32 safe casts
                (NPTypeCode.UInt32, NPTypeCode.Int64) => true,
                (NPTypeCode.UInt32, NPTypeCode.UInt64) => true,
                (NPTypeCode.UInt32, NPTypeCode.Double) => true,  // Float64 can represent all uint32
                (NPTypeCode.UInt32, NPTypeCode.Decimal) => true,

                // Int64 safe casts
                (NPTypeCode.Int64, NPTypeCode.Decimal) => true,
                // Note: Int64 to Double is NOT safe (precision loss for large values)

                // UInt64 safe casts
                (NPTypeCode.UInt64, NPTypeCode.Decimal) => true,
                // Note: UInt64 to Double is NOT safe (precision loss for large values)

                // Single (float32) safe casts
                (NPTypeCode.Single, NPTypeCode.Double) => true,
                (NPTypeCode.Single, NPTypeCode.Decimal) => true,

                // Double safe casts
                (NPTypeCode.Double, NPTypeCode.Decimal) => true,

                // Char (treated as uint16)
                (NPTypeCode.Char, NPTypeCode.Int32) => true,
                (NPTypeCode.Char, NPTypeCode.UInt32) => true,
                (NPTypeCode.Char, NPTypeCode.Int64) => true,
                (NPTypeCode.Char, NPTypeCode.UInt64) => true,
                (NPTypeCode.Char, NPTypeCode.Single) => true,
                (NPTypeCode.Char, NPTypeCode.Double) => true,
                (NPTypeCode.Char, NPTypeCode.Decimal) => true,

                _ => false
            };
        }

        /// <summary>
        /// Check if casting is allowed within the same kind (integers to integers, floats to floats).
        /// </summary>
        private static bool CanCastSameKind(NPTypeCode from, NPTypeCode to)
        {
            // Safe casts are always allowed
            if (CanCastSafe(from, to))
                return true;

            // Allow downcasting within the same kind
            var fromKind = GetTypeKind(from);
            var toKind = GetTypeKind(to);

            return fromKind == toKind && fromKind != TypeKind.Other;
        }

        private enum TypeKind
        {
            SignedInteger,
            UnsignedInteger,
            Floating,
            Boolean,
            Other
        }

        private static TypeKind GetTypeKind(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => TypeKind.Boolean,
                NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64 => TypeKind.UnsignedInteger,
                NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64 => TypeKind.SignedInteger,
                NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => TypeKind.Floating,
                NPTypeCode.Char => TypeKind.UnsignedInteger,  // Char treated as unsigned integer
                _ => TypeKind.Other
            };
        }

        /// <summary>
        /// Check if a scalar value fits in the target type.
        /// </summary>
        private static bool ValueFitsInType(object value, NPTypeCode to)
        {
            try
            {
                switch (value)
                {
                    case bool b:
                        return true;  // Bool can fit in any numeric type

                    case byte by:
                        return to switch
                        {
                            NPTypeCode.Boolean => by == 0 || by == 1,
                            NPTypeCode.Byte => true,
                            _ => CanCastSafe(NPTypeCode.Byte, to)
                        };

                    case sbyte sb:
                        return to switch
                        {
                            NPTypeCode.Boolean => sb == 0 || sb == 1,
                            NPTypeCode.Byte => sb >= 0,
                            NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64 => true,
                            NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64 => sb >= 0,
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false
                        };

                    case short s:
                        return to switch
                        {
                            NPTypeCode.Boolean => s == 0 || s == 1,
                            NPTypeCode.Byte => s >= 0 && s <= 255,
                            NPTypeCode.Int16 => true,
                            NPTypeCode.UInt16 => s >= 0,
                            NPTypeCode.Int32 or NPTypeCode.Int64 => true,
                            NPTypeCode.UInt32 or NPTypeCode.UInt64 => s >= 0,
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false
                        };

                    case ushort us:
                        return to switch
                        {
                            NPTypeCode.Boolean => us == 0 || us == 1,
                            NPTypeCode.Byte => us <= 255,
                            NPTypeCode.Int16 => us <= short.MaxValue,
                            NPTypeCode.UInt16 => true,
                            NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 => true,
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false
                        };

                    case int i:
                        return to switch
                        {
                            NPTypeCode.Boolean => i == 0 || i == 1,
                            NPTypeCode.Byte => i >= 0 && i <= 255,
                            NPTypeCode.Int16 => i >= short.MinValue && i <= short.MaxValue,
                            NPTypeCode.UInt16 => i >= 0 && i <= ushort.MaxValue,
                            NPTypeCode.Int32 => true,
                            NPTypeCode.UInt32 => i >= 0,
                            NPTypeCode.Int64 or NPTypeCode.UInt64 => i >= 0 || to == NPTypeCode.Int64,
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false
                        };

                    case uint ui:
                        return to switch
                        {
                            NPTypeCode.Boolean => ui == 0 || ui == 1,
                            NPTypeCode.Byte => ui <= 255,
                            NPTypeCode.Int16 => ui <= (uint)short.MaxValue,
                            NPTypeCode.UInt16 => ui <= ushort.MaxValue,
                            NPTypeCode.Int32 => ui <= int.MaxValue,
                            NPTypeCode.UInt32 => true,
                            NPTypeCode.Int64 or NPTypeCode.UInt64 => true,
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false
                        };

                    case long l:
                        return to switch
                        {
                            NPTypeCode.Boolean => l == 0 || l == 1,
                            NPTypeCode.Byte => l >= 0 && l <= 255,
                            NPTypeCode.Int16 => l >= short.MinValue && l <= short.MaxValue,
                            NPTypeCode.UInt16 => l >= 0 && l <= ushort.MaxValue,
                            NPTypeCode.Int32 => l >= int.MinValue && l <= int.MaxValue,
                            NPTypeCode.UInt32 => l >= 0 && l <= uint.MaxValue,
                            NPTypeCode.Int64 => true,
                            NPTypeCode.UInt64 => l >= 0,
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false
                        };

                    case ulong ul:
                        return to switch
                        {
                            NPTypeCode.Boolean => ul == 0 || ul == 1,
                            NPTypeCode.Byte => ul <= 255,
                            NPTypeCode.Int16 => ul <= (ulong)short.MaxValue,
                            NPTypeCode.UInt16 => ul <= ushort.MaxValue,
                            NPTypeCode.Int32 => ul <= int.MaxValue,
                            NPTypeCode.UInt32 => ul <= uint.MaxValue,
                            NPTypeCode.Int64 => ul <= long.MaxValue,
                            NPTypeCode.UInt64 => true,
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false
                        };

                    case float f:
                        return to switch
                        {
                            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                            _ => false  // Float to int requires explicit cast
                        };

                    case double d:
                        return to switch
                        {
                            NPTypeCode.Double or NPTypeCode.Decimal => true,
                            NPTypeCode.Single => d >= float.MinValue && d <= float.MaxValue,
                            _ => false  // Double to int requires explicit cast
                        };

                    case decimal m:
                        return to switch
                        {
                            NPTypeCode.Decimal => true,
                            _ => false  // Decimal to other types requires explicit cast
                        };

                    default:
                        return false;
                }
            }
            catch
            {
                return false;
            }
        }
    }
}
