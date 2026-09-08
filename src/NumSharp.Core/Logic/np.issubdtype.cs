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
        /// Implementation mirrors NumPy's issubdtype which uses issubclass() on the
        /// type hierarchy defined in numpy/_core/src/multiarray/multiarraymodule.c.
        ///
        /// Type hierarchy in NumPy:
        /// - generic → number → integer/inexact
        /// - integer → signedinteger/unsignedinteger
        /// - inexact → floating/complexfloating
        ///
        /// Note: In NumPy 2.x, bool is NOT a subtype of integer (it's directly under generic).
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
            return NPTypeHierarchy.IsSubType(arg1, arg2);
        }

        /// <summary>
        /// Returns True if first argument is a typecode lower/equal in type hierarchy.
        /// </summary>
        /// <param name="arg1">dtype - dtype representing a typecode.</param>
        /// <param name="arg2">dtype - dtype representing a typecode.</param>
        /// <returns>True if arg1 is equal to or a subtype of arg2.</returns>
        /// <remarks>
        /// When comparing two concrete types, returns true only if they are the same type.
        /// For hierarchy checks, use the (NPTypeCode, string) overload.
        /// </remarks>
        public static bool issubdtype(NPTypeCode arg1, NPTypeCode arg2)
        {
            // Concrete type comparison: only same type matches
            // This matches NumPy: np.issubdtype(np.int32, np.int64) returns False
            return arg1 == arg2;
        }

        /// <summary>
        /// Returns True if the scalar type of <paramref name="arg1"/> is the scalar type of <paramref name="arg2"/> or a
        /// subclass of it — NumPy's <c>issubdtype(dtype1, dtype2)</c> over two DESCRIPTORS, which compares their
        /// <c>.type</c>: instances of one class compare equal whatever their parameters
        /// (<c>issubdtype('M8[s]', 'M8[ns]')</c> is True, as is <c>issubdtype('&gt;i4', 'i4')</c>), and distinct concrete
        /// classes are never related (<c>issubdtype(int32, int64)</c> is False).
        /// </summary>
        public static bool issubdtype(DType arg1, DType arg2)
        {
            if (arg1 is null) throw new ArgumentNullException(nameof(arg1));
            if (arg2 is null) throw new ArgumentNullException(nameof(arg2));
            return ReferenceEquals(arg1.Meta, arg2.Meta);
        }

        /// <summary>
        /// Returns True if the descriptor's scalar type is under the abstract category <paramref name="arg2"/>
        /// (<c>"generic"</c>, <c>"number"</c>, <c>"integer"</c>, <c>"signedinteger"</c>, <c>"unsignedinteger"</c>,
        /// <c>"inexact"</c>, <c>"floating"</c>, <c>"complexfloating"</c>, <c>"bool"</c>) — NumPy's scalar hierarchy, in
        /// which <c>datetime64</c> sits directly under <c>generic</c> ("Datetime doesn't fit in any category") and
        /// <c>timedelta64</c> under <c>signedinteger</c> ("Timedelta is an integer with an associated unit").
        /// </summary>
        public static bool issubdtype(DType arg1, string arg2)
        {
            if (arg1 is null) throw new ArgumentNullException(nameof(arg1));
            if (arg2 is null) throw new ArgumentNullException(nameof(arg2));

            if (arg1.Meta is DatetimeDTypeMeta dt)
            {
                if (!NPTypeHierarchy.TryParseKind(arg2, out var category))
                    return false;
                if (category == NPTypeKind.Generic)
                    return true;
                /* SINGLE_INHERIT(Timedelta, SignedInteger); SINGLE_INHERIT(Datetime, Generic) */
                return dt.IsTimedelta && !string.Equals(arg2, "numeric", StringComparison.OrdinalIgnoreCase)
                       && NPTypeHierarchy.IsSubType(NPTypeKind.SignedInteger, category);
            }

            if (!arg1.Meta.HasStorage)
                return string.Equals(arg2, "generic", StringComparison.OrdinalIgnoreCase);
            return NPTypeHierarchy.IsSubType(arg1.typecode, arg2);
        }

        /// <summary>
        /// Returns True if the dtype named by <paramref name="arg1"/> (a NumPy dtype string such as <c>"i4"</c> or
        /// <c>"M8[ns]"</c>) is under the abstract category <paramref name="arg2"/>. Exists so that a string first argument
        /// binds the dtype grammar rather than NumSharp's string→<see cref="NDArray"/> (character array) conversion.
        /// </summary>
        public static bool issubdtype(string arg1, string arg2)
        {
            if (arg1 is null) throw new ArgumentNullException(nameof(arg1));
            return issubdtype(dtype(arg1), arg2);
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

        /// <summary>
        /// Returns True if first argument is a typecode lower/equal in type hierarchy.
        /// </summary>
        /// <param name="arr">NDArray - array whose dtype to check.</param>
        /// <param name="arg2">string - string representing a typecode category.</param>
        /// <returns>True if array's dtype is a subtype of arg2 category.</returns>
        /// <exception cref="ArgumentNullException">Thrown if arr is null.</exception>
        public static bool issubdtype(NDArray arr, string arg2)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return issubdtype(arr.GetTypeCode, arg2);
        }

        /// <summary>
        /// Returns True if first argument is a typecode lower/equal in type hierarchy.
        /// </summary>
        /// <param name="arg1">Type - first CLR type.</param>
        /// <param name="arg2">Type - second CLR type to compare against.</param>
        /// <returns>True if arg1 is equal to or a subtype of arg2.</returns>
        public static bool issubdtype(Type arg1, Type arg2)
        {
            return issubdtype(arg1.GetTypeCode(), arg2.GetTypeCode());
        }
    }
}
