using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        // =========================================================================================================
        // NumPy's abstract scalar classes (numpy/_core/numerictypes.py: allTypes' "_abstract_type_names").
        //
        //   object (Python's)
        //   \-> generic
        //     +-> bool
        //     +-> number
        //     |   +-> integer
        //     |   |   +-> signedinteger     int8 int16 int32 int64 timedelta64
        //     |   |   \-> unsignedinteger   uint8 uint16 uint32 uint64 (char)
        //     |   \-> inexact
        //     |       +-> floating          float16 float32 float64 (+ Python's float) (decimal)
        //     |       \-> complexfloating   complex64* complex128 (+ Python's complex)
        //     +-> flexible
        //     |   +-> character         bytes_ (+ Python's bytes) str_ (+ Python's str)
        //     |   \-> void*
        //     +-> datetime64
        //     \-> object_*
        //   (* a NumPy class NumSharp has no dtype for: np.dtype(cls) refuses)
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
        // NumPy's concrete classes NumSharp has no dtype for. They are classes all the same: the hierarchy, np.ScalarType
        // and every type test see them (np.issubdtype(np.complex64, np.complexfloating) is True); np.dtype(cls) and the
        // explicit (DType) conversion refuse with a NotSupportedException.
        // =========================================================================================================

        /// <summary>
        ///     <c>numpy.complex64</c> (two float32 parts): a class under <see cref="complexfloating"/> with no NumSharp dtype —
        ///     NumSharp's one complex width is <see cref="complex128"/>. Type tests work (<c>np.issubdtype(np.complex64,
        ///     np.inexact)</c> is True, <c>np.isdtype(np.complex64, "complex floating")</c> True), <c>np.finfo(np.complex64)</c>
        ///     is float32's, and <c>np.dtype(np.complex64)</c> raises <see cref="System.NotSupportedException"/>.
        /// </summary>
        /// <remarks>
        ///     <c>np.zeros(3, np.complex64)</c> does not compile: a class converts to a descriptor only explicitly, and
        ///     NumSharp cannot allocate complex64 storage. Use <see cref="complex128"/>.
        /// </remarks>
        public static NDScalarType complex64 => NDScalarType.Complex64;

        /// <summary><c>numpy.csingle</c>: NumPy's other name for <see cref="complex64"/> — the same class object.</summary>
        public static NDScalarType csingle => NDScalarType.Complex64;

        /// <summary>
        ///     <c>numpy.object_</c>: a class directly under <see cref="generic"/> with no NumSharp dtype (NumSharp has no
        ///     arrays of Python objects); <c>np.dtype(np.object_)</c> raises <see cref="System.NotSupportedException"/>.
        /// </summary>
        public static NDScalarType object_ => NDScalarType.Object_;

        /// <summary>
        ///     <c>numpy.void</c>: a class under <see cref="flexible"/> with no NumSharp dtype (NumSharp has no raw-byte or
        ///     structured arrays); <c>np.dtype(np.void)</c> raises <see cref="System.NotSupportedException"/>.
        /// </summary>
        public static NDScalarType @void => NDScalarType.Void;

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
        ///     The strings are NumPy's and describe NumPy's codes, including the ones NumSharp has no dtype for (<c>F</c>,
        ///     <c>V</c>, <c>O</c>). Read-only, where NumPy's is a plain dict: the table is shared process-wide, and a write
        ///     would change every reader's answer.
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
        ///     Keys in NumPy 2.4.2's order for the host's C <c>long</c> model (the C integer names: Windows lists
        ///     <c>intc, uintc, int32, long, …</c>, Linux <c>int32, intc, uint32, uintc, …</c>), values resolved on this
        ///     platform: <c>"long"</c> / <c>"ulong"</c> follow the C <c>long</c> (32-bit on Windows, 64-bit on LP64 Linux and
        ///     macOS), <c>"intp"</c>, <c>"int_"</c>, <c>"int"</c>, <c>"uintp"</c> and <c>"uint"</c> the pointer size. Linux
        ///     NumPy's <c>"float128"</c> / <c>"complex256"</c> (its 16-byte long double) are absent: NumSharp's long double
        ///     is 8 bytes, as on Windows.
        ///     </para>
        ///     <para>
        ///     The string names map to the UNSIZED string descriptors, the objects <see cref="bytes_"/> and
        ///     <see cref="str_"/> are (<c>"bytes_"</c>, <c>"bytes"</c>, <c>"a"</c> to <c>dtype('S')</c>; <c>"str_"</c>,
        ///     <c>"str"</c>, <c>"unicode"</c> to <c>dtype('&lt;U')</c>), as NumPy's <c>np.dtype(np.sctypeDict["str"])</c> is.
        ///     </para>
        ///     <para>
        ///     Differences from NumPy, all from NumSharp's dtype set: the names of the classes NumSharp has no dtype for are
        ///     absent (<c>complex64</c>, <c>csingle</c>, <c>void</c>, <c>object_</c>, <c>object</c>) — the values here are
        ///     descriptors, and those classes (<see cref="complex64"/>, <see cref="object_"/>, <see cref="@void"/>) have none.
        ///     The C-type names map to their own classes, as in NumPy (<c>"longdouble"</c> is <see cref="longdouble"/>,
        ///     equal to but not the class of <see cref="float64"/>; <c>"intc"</c> is <see cref="intc"/>, its own class on
        ///     Windows). <c>"datetime64"</c> / <c>"timedelta64"</c> map to the generic-unit descriptors
        ///     (<c>dtype('&lt;M8')</c>, the object <c>np.dtype("M")</c> returns). Read-only, for the reason
        ///     <see cref="typecodes"/> is.
        ///     </para>
        /// </remarks>
        public static IReadOnlyDictionary<string, DType> sctypeDict => ScalarTypeTables.SctypeDict;

        /// <summary>
        ///     <c>numpy.ScalarType</c>: every scalar class NumPy knows, as one read-only list — Python's <c>int</c>,
        ///     <c>float</c>, <c>complex</c>, <c>bool</c>, <c>bytes</c>, <c>str</c> and <c>memoryview</c> first, then each class
        ///     of NumPy's <c>sctypeDict</c> once, sorted by kind and item size the way NumPy sorts them (31 classes on Windows:
        ///     <c>[int, float, complex, bool, bytes, str, memoryview, numpy.bool, numpy.complex64, numpy.complex128,
        ///     numpy.clongdouble, numpy.float16, …, numpy.uint64, numpy.void]</c>).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy's construction, reproduced: <c>ScalarType = [int, float, complex, bool, bytes, str, memoryview] +
        ///     sorted(dict.fromkeys(sctypeDict.values()), key=lambda t: (dtype(t).kind.lower(), dtype(t).itemsize))</c> — a
        ///     STABLE sort, so classes of one kind and size keep their <c>sctypeDict</c> order (<c>intc</c> before
        ///     <c>int32</c> on Windows). The classes NumSharp has no dtype for (<see cref="complex64"/>, <see cref="object_"/>,
        ///     <see cref="@void"/>) are in it, as in NumPy; NumSharp's own decimal and char classes are not (they are no NumPy
        ///     name). The C-integer classes follow the host's C <c>long</c>: Windows lists <c>intc</c> and <c>int32</c>,
        ///     Linux <c>int64</c> and <c>longlong</c>.
        ///     </para>
        ///     <para>
        ///     A port of <c>isinstance(x, np.ScalarType)</c> is <c>np.ScalarType.Any(t =&gt; t.__instancecheck__(x))</c>, and of
        ///     <c>type(x) in np.ScalarType</c> a membership test of the class (<see cref="NDScalarType"/>s compare by identity).
        ///     </para>
        /// </remarks>
        public static IReadOnlyList<NDScalarType> ScalarType => ScalarTypeTables.ScalarTypes;

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

            /// <summary>
            ///     NumPy 2.4.2's <c>sctypeDict</c> keys in NumPy's order on this host's C <c>long</c> model, the dtype-less
            ///     classes' names included — the order <see cref="SctypeDict"/> enumerates in and the order
            ///     <c>dict.fromkeys(sctypeDict.values())</c> visits the classes in, which decides <c>ScalarType</c>'s order
            ///     among classes of one kind and size. Declared before both tables, which read it: static fields initialise
            ///     in declaration order.
            /// </summary>
            private static readonly string[] NumPySctypeDictKeys = BuildNumPySctypeDictKeys();

            /// <summary>NumPy's <c>sctypeDict</c> over NumSharp's descriptors (see <see cref="sctypeDict"/> for the differences).</summary>
            internal static readonly IReadOnlyDictionary<string, DType> SctypeDict = BuildSctypeDict();

            /// <summary>
            ///     Builds <see cref="NumPySctypeDictKeys"/>. NumPy lists the names of its classes in type-number order, each
            ///     class's sized name first when it is the canonical class of its width, so the ten C integer names come in
            ///     the host model's order (probed on Windows and Linux NumPy 2.4.2): Windows' C int class is only
            ///     <c>intc</c> (int32 is the C long, <c>int32</c> then <c>long</c>) and its C long long class is int64
            ///     (<c>int64</c> then <c>longlong</c>); Linux's C int class IS int32 (<c>int32</c> then <c>intc</c>), its C long
            ///     class int64 (<c>int64</c> then <c>long</c>), and its C long long class only <c>longlong</c>.
            /// </summary>
            /// <returns>The keys, in order.</returns>
            /// <remarks>
            ///     Linux NumPy also ends with <c>float128</c> and <c>complex256</c>, the sized names of its 16- and 32-byte
            ///     long double pair. NumPy adds them only when <c>float{bits}</c> of the long double is a new name; NumSharp's
            ///     long double is 8 bytes on every host (<c>float64</c>, as on Windows), so the rule adds nothing here.
            /// </remarks>
            private static string[] BuildNumPySctypeDictKeys()
            {
                string[] cIntegers = DTypeRegistry.CLongIs32Bit
                    ? new[] { "intc", "uintc", "int32", "long", "uint32", "ulong", "int64", "longlong", "uint64", "ulonglong" }
                    : new[] { "int32", "intc", "uint32", "uintc", "int64", "long", "uint64", "ulong", "longlong", "ulonglong" };
                var keys = new List<string>
                {
                    "bool", "float16", "float32", "float64", "longdouble", "complex64", "complex128", "clongdouble", "bytes_",
                    "str_", "void", "object_", "datetime64", "timedelta64", "int8", "byte", "uint8", "ubyte", "int16", "short",
                    "uint16", "ushort",
                };
                keys.AddRange(cIntegers);
                keys.AddRange(new[]
                {
                    "intp", "uintp", "double", "cdouble", "single", "csingle", "half", "bool_", "int_", "uint", "float",
                    "complex", "object", "bytes", "a", "int", "str", "unicode",
                });
                return keys.ToArray();
            }

            /// <summary>
            ///     NumPy's <c>ScalarType</c> (see <see cref="ScalarType"/>). Declared after <see cref="SctypeDict"/> and
            ///     <see cref="NumPySctypeDictKeys"/>, which it reads: static fields initialise in declaration order.
            /// </summary>
            internal static readonly IReadOnlyList<NDScalarType> ScalarTypes = BuildScalarType();

            /// <summary>
            ///     Builds <see cref="ScalarTypes"/>: Python's seven classes, then the classes of
            ///     <see cref="NumPySctypeDictKeys"/> once each (first occurrence), stably sorted by (lower-cased kind, item
            ///     size) — NumPy's <c>_scalar_type_key</c>.
            /// </summary>
            /// <returns>The read-only list.</returns>
            private static IReadOnlyList<NDScalarType> BuildScalarType()
            {
                var seen = new HashSet<NDScalarType>(ReferenceEqualityComparer.Instance);
                var classes = new List<NDScalarType>();
                foreach (var key in NumPySctypeDictKeys)
                {
                    // The dtype-less classes have no sctypeDict entry in NumSharp (its values are descriptors); every other
                    // name's class is the class of its descriptor.
                    NDScalarType cls = key switch
                    {
                        "complex64" or "csingle" => NDScalarType.Complex64,
                        "void" => NDScalarType.Void,
                        "object_" or "object" => NDScalarType.Object_,
                        _ => SctypeDict[key],
                    };
                    if (seen.Add(cls))
                        classes.Add(cls);
                }

                // LINQ's OrderBy is a stable sort, as Python's sorted() is.
                var sorted = classes.OrderBy(c => char.ToLowerInvariant(c.NumPyKind)).ThenBy(c => c.NumPyItemSize);
                var result = new List<NDScalarType>
                {
                    NDScalarType.PyInt, NDScalarType.PyFloat, NDScalarType.PyComplex, NDScalarType.PyBool,
                    NDScalarType.PyBytes, NDScalarType.PyStr, NDScalarType.PyMemoryView,
                };
                result.AddRange(sorted);
                return result.AsReadOnly();
            }

            /// <summary>
            ///     Builds <see cref="SctypeDict"/> in NumPy's key order for this host's model (<see cref="NumPySctypeDictKeys"/>,
            ///     after <c>_type_aliases.py</c>: the <c>typeinfo</c> names, then the <c>_aliases</c>, then the
            ///     <c>_extra_aliases</c>), skipping the names of classes NumSharp lacks. A
            ///     <see cref="Dictionary{TKey,TValue}"/> filled without removals enumerates in insertion order, which is what
            ///     keeps NumPy's order.
            /// </summary>
            /// <returns>The read-only table.</returns>
            private static IReadOnlyDictionary<string, DType> BuildSctypeDict()
            {
                var byName = SctypeDictEntries();
                var ordered = new Dictionary<string, DType>(byName.Count);
                foreach (var key in NumPySctypeDictKeys)
                {
                    // The dtype-less classes' names (complex64, csingle, void, object_, object) have no entry.
                    if (byName.TryGetValue(key, out var descr))
                        ordered.Add(key, descr);
                }
                // A name with an entry but missing from the key list would vanish silently: the two lists must agree.
                System.Diagnostics.Debug.Assert(ordered.Count == byName.Count, "every sctypeDict entry must be in NumPySctypeDictKeys");
                return new ReadOnlyDictionary<string, DType>(ordered);
            }

            /// <summary>
            ///     Every <c>sctypeDict</c> name NumSharp has a descriptor for, with that descriptor (in no particular order:
            ///     <see cref="BuildSctypeDict"/> orders them).
            /// </summary>
            /// <returns>The name → descriptor map.</returns>
            private static Dictionary<string, DType> SctypeDictEntries()
            {
                DType cLong = DTypeRegistry.CLongIs32Bit ? DType.Int32 : DType.Int64;
                DType cULong = DTypeRegistry.CLongIs32Bit ? DType.UInt32 : DType.UInt64;
                // NumPy 2.x: intp / int_ / int and uintp / uint are pointer-sized (np.intp / np.uintp). Spelled from the
                // pointer size here rather than read from np's fields, so this table never depends on np's initialization.
                DType intp = System.IntPtr.Size == 8 ? DType.Int64 : DType.Int32;
                DType uintp = System.IntPtr.Size == 8 ? DType.UInt64 : DType.UInt32;
                // np.dtype(np.bytes_) / np.dtype(np.str_) are the UNSIZED builtins (dtype('S'), dtype('<U')): the class
                // singletons, the same objects np.bytes_ / np.str_ are.
                DType bytes = DTypeRegistry.Bytes.singleton;
                DType str = DTypeRegistry.Str.singleton;
                return new Dictionary<string, DType>
                {
                    ["bool"] = DType.Boolean,
                    ["float16"] = DType.Half,
                    ["float32"] = DType.Single,
                    ["float64"] = DType.Double,
                    ["longdouble"] = DTypeRegistry.LongDouble.singleton,
                    ["complex128"] = DType.Complex,
                    ["clongdouble"] = DTypeRegistry.CLongDouble.singleton,
                    ["bytes_"] = bytes,
                    ["str_"] = str,
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
                    ["intc"] = DTypeRegistry.CInt.singleton,
                    ["uintc"] = DTypeRegistry.CUInt.singleton,
                    ["int32"] = DType.Int32,
                    ["long"] = cLong,
                    ["uint32"] = DType.UInt32,
                    ["ulong"] = cULong,
                    ["int64"] = DType.Int64,
                    ["longlong"] = DTypeRegistry.CLongLong.singleton,
                    ["uint64"] = DType.UInt64,
                    ["ulonglong"] = DTypeRegistry.CULongLong.singleton,
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
                    ["bytes"] = bytes,
                    ["a"] = bytes,
                    ["int"] = intp,
                    ["str"] = str,
                    ["unicode"] = str,
                };
            }
        }
    }
}
