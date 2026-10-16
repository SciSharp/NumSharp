using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace NumSharp
{
    public static partial class np
    {
        // =========================================================================================================
        // NumPy's abstract scalar classes (numpy/_core/numerictypes.py: allTypes' "_abstract_type_names").
        //
        //   generic
        //     +-> bool
        //     +-> number
        //     |   +-> integer
        //     |   |   +-> signedinteger     int8 int16 int32 int64 timedelta64
        //     |   |   \-> unsignedinteger   uint8 uint16 uint32 uint64 (char)
        //     |   \-> inexact
        //     |       +-> floating          float16 float32 float64 (decimal)
        //     |       \-> complexfloating   complex128
        //     +-> flexible
        //     |   \-> character
        //     \-> datetime64
        //
        // Each is the one NDScalarType object of its class (identity matters: issubclass/isinstance compare references),
        // so the properties return the shared static, never a copy.
        // =========================================================================================================

        /// <summary>
        ///     <c>numpy.generic</c>: the base of every NumPy scalar class — <c>np.issubdtype(x, np.generic)</c> holds for
        ///     every dtype.
        /// </summary>
        /// <remarks>Abstract: <c>np.dtype(np.generic)</c> raises NumPy's <see cref="TypeError"/> (<c>Converting `np.generic` to a dtype is not allowed.</c>).</remarks>
        public static NDScalarType generic => NDScalarType.Generic;

        /// <summary>
        ///     <c>numpy.number</c>: every numeric class — the integers, floats and complex numbers. bool is NOT a number in
        ///     NumPy 2.x (<c>np.issubdtype(np.bool_, np.number)</c> is False).
        /// </summary>
        /// <remarks>Abstract: converting it to a dtype raises NumPy's <c>Converting `np.inexact` or `np.floating` to a dtype not allowed</c> (NumPy's own text names those two classes for <c>np.number</c> too).</remarks>
        public static NDScalarType number => NDScalarType.Number;

        /// <summary>
        ///     <c>numpy.integer</c>: the signed and unsigned integer classes — NumPy's most common type test,
        ///     <c>np.issubdtype(a.dtype, np.integer)</c>. timedelta64 is an integer (a signed integer with a unit); bool is
        ///     not.
        /// </summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed</c>.</remarks>
        public static NDScalarType integer => NDScalarType.Integer;

        /// <summary><c>numpy.signedinteger</c>: int8, int16, int32, int64 and timedelta64.</summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed</c>.</remarks>
        public static NDScalarType signedinteger => NDScalarType.SignedInteger;

        /// <summary><c>numpy.unsignedinteger</c>: uint8, uint16, uint32, uint64, and NumSharp's char (stored as a 16-bit unsigned integer).</summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting `np.unsignedinteger` to a dtype is not allowed</c>.</remarks>
        public static NDScalarType unsignedinteger => NDScalarType.UnsignedInteger;

        /// <summary><c>numpy.inexact</c>: the floating-point and complex classes (the dtypes with a <c>np.finfo</c>).</summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting `np.inexact` or `np.floating` to a dtype not allowed</c>.</remarks>
        public static NDScalarType inexact => NDScalarType.Inexact;

        /// <summary><c>numpy.floating</c>: float16, float32, float64, and NumSharp's decimal.</summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting `np.inexact` or `np.floating` to a dtype not allowed</c>.</remarks>
        public static NDScalarType floating => NDScalarType.Floating;

        /// <summary><c>numpy.complexfloating</c>: complex128 (NumSharp has no complex64).</summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting `np.complex` to a dtype is not allowed.</c></remarks>
        public static NDScalarType complexfloating => NDScalarType.ComplexFloating;

        /// <summary><c>numpy.flexible</c>: the classes whose instances carry a length — the string classes (under <see cref="character"/>) and <c>void</c>.</summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting `np.generic` to a dtype is not allowed.</c> (NumPy's text for both classes).</remarks>
        public static NDScalarType flexible => NDScalarType.Flexible;

        /// <summary><c>numpy.character</c>: the fixed-width string classes <c>bytes_</c> and <c>str_</c>.</summary>
        /// <remarks>Abstract: converting it to a dtype raises <c>Converting `np.character` to a dtype is not allowed</c>.</remarks>
        public static NDScalarType character => NDScalarType.Character;

        // =========================================================================================================
        // The relatives: NumPy's bool constants and its name tables (numpy/_core/numerictypes.py, _type_aliases.py)
        // =========================================================================================================

        /// <summary>
        ///     <c>numpy.True_</c>: NumPy's bool scalar <c>True</c>. NumSharp's bool scalar is the C# <see cref="bool"/>, so
        ///     this is <see langword="true"/> (the same rule as <see cref="nan"/> being a <see cref="double"/>).
        /// </summary>
        public static bool True_ => true;

        /// <summary><c>numpy.False_</c>: NumPy's bool scalar <c>False</c> — the C# <see langword="false"/> (see <see cref="True_"/>).</summary>
        public static bool False_ => false;

        /// <summary>
        ///     <c>numpy.typecodes</c>: NumPy's one-character type codes by group, verbatim —
        ///     <c>{'Character': 'c', 'Integer': 'bhilqnp', 'UnsignedInteger': 'BHILQNP', 'Float': 'efdg', 'Complex': 'FDG',
        ///     'AllInteger': 'bBhHiIlLqQnNpP', 'AllFloat': 'efdgFDG', 'Datetime': 'Mm', 'All': '?bhilqnpBHILQNPefdgFDGSUVOMm'}</c>,
        ///     in that order. Use it the way NumPy code does: <c>np.typecodes["AllFloat"].Contains(a.dtype.@char)</c>.
        /// </summary>
        /// <remarks>
        ///     The strings are NumPy's and describe NumPy's codes, including the ones NumSharp has no dtype for (<c>g</c>,
        ///     <c>F</c>, <c>G</c>, <c>S</c>, <c>U</c>, <c>V</c>, <c>O</c>). Read-only, where NumPy's is a plain dict: the
        ///     table is shared process-wide, and a write would change every reader's answer.
        /// </remarks>
        public static IReadOnlyDictionary<string, string> typecodes => ScalarTypeTables.TypeCodes;

        /// <summary>
        ///     <c>numpy.sctypeDict</c>: every name NumPy resolves to a scalar class, mapped to NumSharp's spelling of that
        ///     class — the dtype descriptor (<c>np.sctypeDict["float64"]</c> is <see cref="float64"/> itself, as NumPy's is
        ///     <c>np.float64</c>), so a value can be passed as <c>dtype=</c> directly or tested with
        ///     <c>np.issubdtype(value, np.floating)</c>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Keys in NumPy 2.4.2's order (win-amd64), values resolved on this platform: <c>"long"</c> / <c>"ulong"</c>
        ///     follow the C <c>long</c> (32-bit on Windows, 64-bit on LP64 Linux and macOS), <c>"intp"</c>, <c>"int_"</c>,
        ///     <c>"int"</c>, <c>"uintp"</c> and <c>"uint"</c> the pointer size.
        ///     </para>
        ///     <para>
        ///     Differences from NumPy, all from NumSharp's dtype set: the names of classes NumSharp lacks are absent
        ///     (<c>complex64</c>, <c>csingle</c>, <c>bytes_</c>, <c>str_</c>, <c>void</c>, <c>object_</c>, and the aliases
        ///     <c>object</c>, <c>bytes</c>, <c>a</c>, <c>str</c>, <c>unicode</c>); the names of classes NumSharp collapses
        ///     map to the class they collapse onto (<c>"intc"</c> is <see cref="int32"/>, <c>"longdouble"</c> is
        ///     <see cref="float64"/>, <c>"clongdouble"</c> is <see cref="complex128"/>, where Windows NumPy keeps distinct
        ///     classes). <c>"datetime64"</c> / <c>"timedelta64"</c> map to the generic-unit descriptors (<c>dtype('&lt;M8')</c>,
        ///     the object <c>np.dtype("M")</c> returns). Read-only, for the reason <see cref="typecodes"/> is.
        ///     </para>
        /// </remarks>
        public static IReadOnlyDictionary<string, DType> sctypeDict => ScalarTypeTables.SctypeDict;

        /// <summary>
        ///     The tables behind <see cref="typecodes"/> and <see cref="sctypeDict"/>, built when one of them is first read.
        ///     A nested class so that building them (which reads <see cref="DTypeRegistry"/>) is not part of <c>np</c>'s own
        ///     static initialization, which the registry must never be entangled with (the np ↔ registry init-order trap).
        /// </summary>
        private static class ScalarTypeTables
        {
            /// <summary>NumPy's <c>numpy._core.numerictypes.typecodes</c>, verbatim and in NumPy's order.</summary>
            internal static readonly IReadOnlyDictionary<string, string> TypeCodes = new ReadOnlyDictionary<string, string>(
                new Dictionary<string, string>
                {
                    ["Character"] = "c",
                    ["Integer"] = "bhilqnp",
                    ["UnsignedInteger"] = "BHILQNP",
                    ["Float"] = "efdg",
                    ["Complex"] = "FDG",
                    ["AllInteger"] = "bBhHiIlLqQnNpP",
                    ["AllFloat"] = "efdgFDG",
                    ["Datetime"] = "Mm",
                    ["All"] = "?bhilqnpBHILQNPefdgFDGSUVOMm",
                });

            /// <summary>NumPy's <c>sctypeDict</c> over NumSharp's descriptors (see <see cref="sctypeDict"/> for the differences).</summary>
            internal static readonly IReadOnlyDictionary<string, DType> SctypeDict = BuildSctypeDict();

            /// <summary>
            ///     Builds <see cref="SctypeDict"/> in NumPy's key order (<c>_type_aliases.py</c>: the <c>typeinfo</c> names,
            ///     then the <c>_aliases</c>, then the <c>_extra_aliases</c>), skipping the names of classes NumSharp lacks.
            ///     A <see cref="Dictionary{TKey,TValue}"/> filled without removals enumerates in insertion order, which is
            ///     what keeps NumPy's order.
            /// </summary>
            /// <returns>The read-only table.</returns>
            private static IReadOnlyDictionary<string, DType> BuildSctypeDict()
            {
                DType cLong = DTypeRegistry.CLongIs32Bit ? DType.Int32 : DType.Int64;
                DType cULong = DTypeRegistry.CLongIs32Bit ? DType.UInt32 : DType.UInt64;
                // NumPy 2.x: intp / int_ / int and uintp / uint are pointer-sized (np.intp / np.uintp). Spelled from the
                // pointer size here rather than read from np's fields, so this table never depends on np's initialization.
                DType intp = System.IntPtr.Size == 8 ? DType.Int64 : DType.Int32;
                DType uintp = System.IntPtr.Size == 8 ? DType.UInt64 : DType.UInt32;
                return new ReadOnlyDictionary<string, DType>(new Dictionary<string, DType>
                {
                    ["bool"] = DType.Boolean,
                    ["float16"] = DType.Half,
                    ["float32"] = DType.Single,
                    ["float64"] = DType.Double,
                    ["longdouble"] = DType.Double,
                    ["complex128"] = DType.Complex,
                    ["clongdouble"] = DType.Complex,
                    ["datetime64"] = DTypeRegistry.DateTime64.singleton,
                    ["timedelta64"] = DTypeRegistry.TimeDelta64.singleton,
                    ["int8"] = DType.SByte,
                    ["byte"] = DType.SByte,
                    ["uint8"] = DType.Byte,
                    ["ubyte"] = DType.Byte,
                    ["int16"] = DType.Int16,
                    ["short"] = DType.Int16,
                    ["uint16"] = DType.UInt16,
                    ["ushort"] = DType.UInt16,
                    ["intc"] = DType.Int32,
                    ["uintc"] = DType.UInt32,
                    ["int32"] = DType.Int32,
                    ["long"] = cLong,
                    ["uint32"] = DType.UInt32,
                    ["ulong"] = cULong,
                    ["int64"] = DType.Int64,
                    ["longlong"] = DType.Int64,
                    ["uint64"] = DType.UInt64,
                    ["ulonglong"] = DType.UInt64,
                    ["intp"] = intp,
                    ["uintp"] = uintp,
                    ["double"] = DType.Double,
                    ["cdouble"] = DType.Complex,
                    ["single"] = DType.Single,
                    ["half"] = DType.Half,
                    ["bool_"] = DType.Boolean,
                    ["int_"] = intp,
                    ["uint"] = uintp,
                    ["float"] = DType.Double,
                    ["complex"] = DType.Complex,
                    ["int"] = intp,
                });
            }
        }
    }
}
