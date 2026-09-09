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
                Type t => t.GetTypeCode() is var tc && tc != NPTypeCode.Empty && tc != NPTypeCode.String,
                _ => false
            };
        }

        /// <summary>
        /// Returns True if the descriptor is of a specified category — NumPy 2.x's <c>np.isdtype(dtype, kind)</c> over a
        /// <see cref="DType"/>: <paramref name="kind"/> is one of <c>"bool"</c>, <c>"signed integer"</c>,
        /// <c>"unsigned integer"</c>, <c>"integral"</c>, <c>"real floating"</c>, <c>"complex floating"</c>, <c>"numeric"</c>
        /// (case-sensitive, as in NumPy). The datetime classes belong to none of them (NumPy builds the categories from
        /// <c>sctypes</c>, which exclude <c>datetime64</c>/<c>timedelta64</c>).
        /// <para>
        /// This is the ONE entry for every dtype spelling: a C# <see cref="Type"/> (<c>typeof(int)</c>), an
        /// <see cref="NPTypeCode"/> (<c>NPTypeCode.Int32</c>) or a dtype string all convert implicitly to <see cref="DType"/>
        /// (the earlier <c>NPTypeCode</c>/<c>Type</c> twins accepted <c>issubdtype</c>'s looser vocabulary — <c>"floating"</c>,
        /// <c>"integer"</c> — which NumPy's <c>isdtype</c> rejects; they were folded into this overload).
        /// </para>
        /// </summary>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.isdtype.html — a NumPy 2.0+ function.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.isdtype(NPTypeCode.Int32, "integral")      // True
        /// np.isdtype(typeof(double), "real floating")   // True
        /// np.isdtype(np.int32, "numeric")               // True
        /// </code>
        /// </example>
        /// <exception cref="ValueError"><c>kind argument is a string, but '…' is not a known kind name.</c> — verbatim NumPy.</exception>
        public static bool isdtype(DType dtype, string kind)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            if (kind is null) throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a null.");
            switch (kind)
            {
                case "bool":
                case "signed integer":
                case "unsigned integer":
                case "integral":
                case "real floating":
                case "complex floating":
                case "numeric":
                    break;
                default:
                    throw new ValueError($"kind argument is a string, but '{kind}' is not a known kind name.");
            }

            if (!dtype.Meta.HasStorage)
                return false; // datetime64 / timedelta64: not in NumPy's sctypes categories
            return NPTypeHierarchy.IsSubType(dtype.typecode, kind);
        }

        /// <summary>
        /// Returns True if the descriptor is of any of the specified categories (NumPy's tuple form).
        /// </summary>
        public static bool isdtype(DType dtype, params string[] kinds)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            if (kinds is null) throw new ArgumentNullException(nameof(kinds));
            bool result = false;
            foreach (var kind in kinds)
                result |= isdtype(dtype, kind); // every kind is validated, as NumPy processes the whole tuple first
            return result;
        }

        /// <summary>
        /// Returns True if the two descriptors are instances of the same DType class — NumPy's <c>np.isdtype(dtype, other_dtype)</c>
        /// compares the scalar TYPES, so <c>isdtype('M8[s]', 'M8[ns]')</c> is True while <c>isdtype(float32, float64)</c> is False.
        /// </summary>
        public static bool isdtype(DType dtype, DType kind)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            if (kind is null) throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a null.");
            return ReferenceEquals(dtype.Meta, kind.Meta);
        }

        /// <summary>
        /// <see cref="isdtype(DType, string)"/> for a dtype STRING (<c>np.isdtype("i4", "integral")</c>). Exists so that a
        /// string first argument binds the dtype grammar rather than NumSharp's string→<see cref="NDArray"/> conversion.
        /// </summary>
        public static bool isdtype(string dtype, string kind)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            return isdtype(np.dtype(dtype), kind);
        }

        /// <summary><see cref="isdtype(DType, string[])"/> for a dtype STRING.</summary>
        public static bool isdtype(string dtype, params string[] kinds)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            return isdtype(np.dtype(dtype), kinds);
        }

        /// <summary>
        /// Returns True if the array's dtype is of a specified category (NumSharp convenience: NumPy's <c>isdtype</c> takes
        /// only a dtype, so this is <c>np.isdtype(arr.dtype, kind)</c>).
        /// </summary>
        /// <param name="arr">The NDArray to check.</param>
        /// <param name="kind">The dtype category.</param>
        /// <returns>True if array's dtype belongs to the specified category.</returns>
        /// <exception cref="ArgumentNullException">Thrown if arr is null.</exception>
        public static bool isdtype(NDArray arr, string kind)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return isdtype(arr.dtype, kind);
        }

        /// <summary>
        /// Returns True if the array's dtype is of any of the specified categories.
        /// </summary>
        /// <param name="arr">The NDArray to check.</param>
        /// <param name="kinds">Array of dtype categories to check.</param>
        /// <returns>True if array's dtype belongs to any of the specified categories.</returns>
        /// <exception cref="ArgumentNullException">Thrown if arr is null.</exception>
        public static bool isdtype(NDArray arr, string[] kinds)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return isdtype(arr.dtype, kinds);
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
        /// Character codes (NumPy):
        /// '?' - boolean
        /// 'b' - int8 (signed byte)
        /// 'B' - uint8 (unsigned byte)
        /// 'h' - int16 (short)
        /// 'H' - uint16 (unsigned short)
        /// 'i' - int32
        /// 'I' - uint32
        /// 'q' - int64
        /// 'Q' - uint64
        /// 'e' - float16 (Half)
        /// 'f' - float32
        /// 'd' - float64
        /// 'D' - complex128
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
                NPTypeCode.Boolean => '?',
                NPTypeCode.Byte => 'B',
                NPTypeCode.SByte => 'b',
                NPTypeCode.Int16 => 'h',
                NPTypeCode.UInt16 => 'H',
                NPTypeCode.Int32 => 'i',
                NPTypeCode.UInt32 => 'I',
                NPTypeCode.Int64 => 'q',
                NPTypeCode.UInt64 => 'Q',
                NPTypeCode.Char => 'H',  // Char treated as uint16
                NPTypeCode.Half => 'e',
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
        ///
        /// Uses NPTypeHierarchy for consistent type categorization across all typing functions.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.maximum_sctype(NPTypeCode.Int32)   // Int64
        /// np.maximum_sctype(NPTypeCode.Single)  // Double (or Decimal)
        /// </code>
        /// </example>
        public static DType maximum_sctype(NPTypeCode t)
        {
            return NPTypeHierarchy.GetMaximumType(t);
        }
    }
}
