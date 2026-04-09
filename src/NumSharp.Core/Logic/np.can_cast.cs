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
        /// Returns True if the byte value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(byte value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the short value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(short value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the ushort value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(ushort value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the uint value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(uint value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the ulong value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(ulong value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the float value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(float value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the decimal value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(decimal value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if the bool value can be cast to the data type according to the casting rule.
        /// </summary>
        public static bool can_cast(bool value, NPTypeCode to, string casting = "safe")
        {
            if (casting.ToLowerInvariant() == "unsafe")
                return true;
            return ValueFitsInType(value, to);
        }

        /// <summary>
        /// Returns True if cast between data types can occur according to the casting rule.
        /// </summary>
        /// <typeparam name="TFrom">Source type.</typeparam>
        /// <typeparam name="TTo">Target type.</typeparam>
        /// <param name="casting">Controls what kind of data casting may occur.</param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        /// <example>
        /// <code>
        /// np.can_cast&lt;int, long&gt;()           // True
        /// np.can_cast&lt;long, int&gt;()           // False
        /// np.can_cast&lt;int, float&gt;("safe")    // True (int to float is safe)
        /// </code>
        /// </example>
        public static bool can_cast<TFrom, TTo>(string casting = "safe")
            where TFrom : struct
            where TTo : struct
        {
            return can_cast(typeof(TFrom).GetTypeCode(), typeof(TTo).GetTypeCode(), casting);
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
        /// <remarks>
        /// A cast is safe if and only if the target type is the common type of both types.
        /// This leverages the type promotion tables in <see cref="_FindCommonType_Array"/>
        /// which encode NumPy's type hierarchy.
        ///
        /// Mathematically: can_cast(A, B) ⟺ promote_types(A, B) == B
        /// </remarks>
        private static bool CanCastSafe(NPTypeCode from, NPTypeCode to)
        {
            if (from == to)
                return true;

            // Safe cast iff the target type is the common type of both
            // This reuses the type promotion tables, avoiding duplicate type knowledge
            return _FindCommonType_Array(from, to) == to;
        }

        /// <summary>
        /// Check if casting is allowed within the same kind (integers to integers, floats to floats).
        /// </summary>
        /// <remarks>
        /// Uses NPTypeHierarchy for consistent type categorization across all typing functions.
        /// same_kind allows downcasting within:
        /// - integers (both signed and unsigned, e.g., int64 -> int32, uint32 -> int16)
        /// - floating point (e.g., float64 -> float32)
        /// - complex (e.g., complex128 -> complex64, if we had it)
        /// </remarks>
        private static bool CanCastSameKind(NPTypeCode from, NPTypeCode to)
        {
            // Safe casts are always allowed
            if (CanCastSafe(from, to))
                return true;

            // Allow downcasting within the same kind
            return NPTypeHierarchy.IsSameKind(from, to);
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
