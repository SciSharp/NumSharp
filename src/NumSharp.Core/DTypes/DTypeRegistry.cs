using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     The table of live <see cref="DTypeMeta"/> classes and their casting implementations — NumPy's
    ///     <c>_builtin_descrs</c> / <c>typenum_to_dtypemeta</c> / <c>_PyArray_MapPyTypeToDType</c> rolled into one static
    ///     registry. Lookups by <see cref="NPTypeCode"/>, C# scalar <see cref="Type"/>, NumPy type number and class name
    ///     (aliases included); <see cref="Register"/> admits a new class (a Stage C/D dtype, a user dtype) without
    ///     touching the kernel switches — the class registers its code and its casts, and the promotion/casting
    ///     engines pick it up.
    /// </summary>
    /// <remarks>
    ///     Static initialisation builds the 15 numeric storage classes, the two fixed-width string classes
    ///     (<see cref="Bytes"/> and <see cref="Str"/>, sharing the <see cref="NPTypeCode.String"/> lane), the two datetime
    ///     classes, the three NEP 50 scalar classes, their singleton descriptors, and the casting implementations NumPy
    ///     registers in <c>PyArray_InitializeNumericCasts</c> / <c>PyArray_InitializeDatetimeCasts</c> /
    ///     <c>PyArray_InitializeStringCasts</c>. It touches no <c>np</c> static state, so it cannot cycle with the
    ///     <c>np</c> static constructor; the promotion TABLE (<see cref="np._nptypemap_arr_arr"/>) is consulted lazily at
    ///     call time.
    ///     <para>
    ///     <b>Threading.</b> No lookup takes a lock. The three keyed tables are
    ///     <see cref="ConcurrentDictionary{TKey,TValue}"/>s, the code table is read with <see cref="Volatile.Read{T}"/>,
    ///     and <see cref="All"/> copies an array that each registration replaces instead of changing. Only
    ///     <see cref="Register"/> takes <see cref="_lock"/>: it checks the type number, the name, every alias and the
    ///     storage code before it writes any table, and two registrations must not interleave between their checks and
    ///     their writes. A lookup that runs during a registration sees each table either before or after its write, so a
    ///     new class can be found by its type number a moment before it can be found by its name.
    ///     </para>
    /// </remarks>
    public static class DTypeRegistry
    {
        /// <summary>
        ///     Serializes registrations and nothing else: <see cref="Register"/> holds it across <see cref="Add"/>'s checks
        ///     and writes, so two registrations cannot both pass the checks and then collide in the tables. No lookup takes
        ///     it.
        /// </summary>
        private static readonly object _lock = new object();

        /// <summary>
        ///     The class of each storage code, indexed by <see cref="NPTypeCode"/> (the largest code, <c>Complex</c>, is 128).
        ///     A slot is written at most once, from null to its class, with <see cref="Volatile.Write{T}"/>, and read with
        ///     <see cref="Volatile.Read{T}"/>, so a lookup sees null or a fully constructed class.
        /// </summary>
        private static readonly DTypeMeta[] _byCode = new DTypeMeta[129];

        /// <summary>
        ///     The class of each NumPy type number, read without a lock by <see cref="FromTypeNum"/> while
        ///     <see cref="Register"/> may be adding to it. <c>concurrencyLevel: 1</c> because <see cref="_lock"/> already
        ///     serializes every write, so the dictionary's own write lock is never contended; <c>capacity: 31</c> (37 buckets
        ///     after the round-up to a prime) holds the 20 builtin type numbers without a resize.
        /// </summary>
        private static readonly ConcurrentDictionary<int, DTypeMeta> _byTypeNum = new(concurrencyLevel: 1, capacity: 31);

        /// <summary>
        ///     The class of each class name and alias (ordinal), read without a lock by <see cref="FromName"/>.
        ///     <c>capacity: 61</c> (71 buckets) holds the 35 builtin names and aliases without a resize.
        /// </summary>
        private static readonly ConcurrentDictionary<string, DTypeMeta> _byName = new(concurrencyLevel: 1, capacity: 61, comparer: StringComparer.Ordinal);

        /// <summary>
        ///     The class of each C# scalar type, read without a lock by <see cref="FromScalarType"/>, which every
        ///     <c>Type → DType</c> conversion runs (<see cref="DType.From(Type)"/>, <c>np.zeros(3, typeof(int))</c>). The
        ///     first class registered for a type keeps it. <c>capacity: 31</c> holds the builtin scalar types without a resize.
        /// </summary>
        private static readonly ConcurrentDictionary<Type, DTypeMeta> _byScalarType = new(concurrencyLevel: 1, capacity: 31);

        /// <summary>
        ///     Every registered class, in registration order. Nothing writes into an array once it is published:
        ///     <see cref="AppendToAll"/> publishes a longer copy with <see cref="Volatile.Write{T}"/>, and <see cref="All"/>
        ///     copies whichever array it reads, so it needs no lock and its caller can never change the registry's array.
        /// </summary>
        private static DTypeMeta[] _all = Array.Empty<DTypeMeta>();

        /// <summary>True when the platform C <c>long</c> is 32 bits (MSVC on Windows, any 32-bit process) — decides which class <c>LongDType</c> aliases.</summary>
        internal static readonly bool CLongIs32Bit = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || IntPtr.Size != 8;

        // ---- the classes ----
        public static LegacyBuiltinDTypeMeta Bool { get; }
        public static LegacyBuiltinDTypeMeta Int8 { get; }
        public static LegacyBuiltinDTypeMeta UInt8 { get; }
        public static LegacyBuiltinDTypeMeta Int16 { get; }
        public static LegacyBuiltinDTypeMeta UInt16 { get; }
        public static LegacyBuiltinDTypeMeta Int32 { get; }
        public static LegacyBuiltinDTypeMeta UInt32 { get; }
        public static LegacyBuiltinDTypeMeta Int64 { get; }
        public static LegacyBuiltinDTypeMeta UInt64 { get; }
        public static LegacyBuiltinDTypeMeta Half { get; }
        public static LegacyBuiltinDTypeMeta Single { get; }
        public static LegacyBuiltinDTypeMeta Double { get; }
        public static LegacyBuiltinDTypeMeta Complex128 { get; }
        /// <summary>
        ///     <c>np.dtypes.LongDoubleDType</c> — NumPy's C <c>long double</c> class (type number 13, char <c>'g'</c>,
        ///     scalar class <c>numpy.longdouble</c>), a lane-sharing class over the <see cref="NPTypeCode.Double"/> lane: 8
        ///     bytes on every platform, as on NumPy's win-amd64 build (NumSharp has no extended precision; Linux NumPy's is an
        ///     80-bit type in 16 bytes). Equivalent to <see cref="Double"/> (<c>np.dtype('g') == np.dtype('d')</c>), but its
        ///     own class: <c>np.issubdtype(np.float64, np.longdouble)</c> is False.
        /// </summary>
        public static LegacyBuiltinDTypeMeta LongDouble { get; }

        /// <summary>
        ///     <c>np.dtypes.CLongDoubleDType</c> — NumPy's complex <c>long double</c> class (type number 16, char <c>'G'</c>,
        ///     scalar class <c>numpy.clongdouble</c>), sharing the <see cref="NPTypeCode.Complex"/> lane (two 8-byte parts);
        ///     equivalent to <see cref="Complex128"/>.
        /// </summary>
        public static LegacyBuiltinDTypeMeta CLongDouble { get; }

        /// <summary>
        ///     The class of NumPy's C <c>int</c> (<c>np.intc</c>, char <c>'i'</c>, type number 5): on LLP64 (Windows) its own
        ///     lane-sharing class <c>IntDType</c> (scalar class <c>numpy.intc</c>) beside <see cref="Int32"/>, which is the C
        ///     <c>long</c> there; on LP64 the <see cref="Int32"/> class itself (int32 IS the C <c>int</c>).
        /// </summary>
        public static LegacyBuiltinDTypeMeta CInt { get; }

        /// <summary>The class of NumPy's C <c>unsigned int</c> (<c>np.uintc</c>, char <c>'I'</c>, type number 6): <c>UIntDType</c> on LLP64, <see cref="UInt32"/> on LP64 (see <see cref="CInt"/>).</summary>
        public static LegacyBuiltinDTypeMeta CUInt { get; }

        /// <summary>
        ///     The class of NumPy's C <c>long long</c> (<c>np.longlong</c>, char <c>'q'</c>, type number 9): on LP64 (Linux,
        ///     macOS) its own lane-sharing class <c>LongLongDType</c> (scalar class <c>numpy.longlong</c>) beside
        ///     <see cref="Int64"/>, which is the C <c>long</c> there; on LLP64 the <see cref="Int64"/> class itself.
        /// </summary>
        public static LegacyBuiltinDTypeMeta CLongLong { get; }

        /// <summary>The class of NumPy's C <c>unsigned long long</c> (<c>np.ulonglong</c>, char <c>'Q'</c>, type number 10): <c>ULongLongDType</c> on LP64, <see cref="UInt64"/> on LLP64 (see <see cref="CLongLong"/>).</summary>
        public static LegacyBuiltinDTypeMeta CULongLong { get; }

        /// <summary>
        ///     Every numeric storage class in registration order — the lane owners, the lane-sharing C-type classes, and
        ///     NumSharp's Decimal and Char: the classes every numeric cast table (numeric ↔ numeric, ↔ datetime, ↔ strings,
        ///     ↔ StringDType) is registered over.
        /// </summary>
        internal static IReadOnlyList<LegacyBuiltinDTypeMeta> NumericStorage { get; }

        /// <summary>NumSharp-only: <see cref="System.Decimal"/> (type number 256, the user-defined range).</summary>
        public static LegacyBuiltinDTypeMeta Decimal { get; }
        /// <summary>NumSharp-only: <see cref="System.Char"/> (type number 257, the user-defined range; promotes as uint16).</summary>
        public static LegacyBuiltinDTypeMeta Char { get; }
        /// <summary>
        ///     <c>np.dtypes.StrDType</c> — fixed-width Unicode (<c>U</c>, type number 19, one UCS-4 code point per character).
        ///     The canonical class of the <see cref="NPTypeCode.String"/> storage lane and of <see cref="string"/>.
        /// </summary>
        public static StringDTypeMeta Str { get; }
        /// <summary>
        ///     <c>np.dtypes.BytesDType</c> — fixed-width bytes (<c>S</c>, type number 18, one byte per character). Shares the
        ///     <see cref="NPTypeCode.String"/> storage lane with <see cref="Str"/> but owns neither that code nor a C# type,
        ///     so it is reached by name only (<c>"S5"</c>, <see cref="np.bytes_"/>).
        /// </summary>
        public static StringDTypeMeta Bytes { get; }
        /// <summary>
        ///     <c>np.dtypes.StringDType</c> — the NEP 55 variable-width string class (kind <c>'T'</c>, type number 2056). It
        ///     shares the <see cref="NPTypeCode.String"/> lane with <see cref="Str"/> / <see cref="Bytes"/> but owns neither the
        ///     code nor a C# type, and it has no singleton: descriptors come from <see cref="np.dtypes.StringDType(bool)"/>,
        ///     <see cref="np.dtypes.StringDType(object, bool)"/> and <c>np.dtype("T")</c>, a fresh one each time.
        /// </summary>
        public static VStringDTypeMeta VString { get; }
        /// <summary><c>np.dtypes.DateTime64DType</c>.</summary>
        public static DatetimeDTypeMeta DateTime64 { get; }
        /// <summary><c>np.dtypes.TimeDelta64DType</c>.</summary>
        public static DatetimeDTypeMeta TimeDelta64 { get; }
        /// <summary>NEP 50's <c>_PyLongDType</c> — the class of a weak C# integer literal.</summary>
        public static PyScalarDTypeMeta PyLong { get; }
        /// <summary>NEP 50's <c>_PyFloatDType</c> — the class of a weak C# floating literal.</summary>
        public static PyScalarDTypeMeta PyFloat { get; }
        /// <summary>NEP 50's <c>_PyComplexDType</c> — the class of a weak C# <see cref="Complex"/> literal.</summary>
        public static PyScalarDTypeMeta PyComplex { get; }

        static DTypeRegistry()
        {
            const string numpy = "numpy.dtypes";
            const string numsharp = "numsharp.dtypes";
            const DTypeFlags builtin = DTypeFlags.NPY_DT_LEGACY | DTypeFlags.NPY_DT_NUMERIC;

            // The C-name aliases of the canonical 32/64-bit classes: LongDType names whichever is the C `long`, and the C
            // type of the other width that has NO class of its own on this platform is an alias too — IntDType on LP64
            // (int32 IS the C int), LongLongDType on LLP64 (int64 IS the C long long). The remaining C type of each width
            // is a lane-sharing class of its own below (IntDType on LLP64, LongLongDType on LP64), as in NumPy.
            string[] longAliases32 = CLongIs32Bit ? new[] { "LongDType" } : new[] { "IntDType" };
            string[] ulongAliases32 = CLongIs32Bit ? new[] { "ULongDType" } : new[] { "UIntDType" };
            string[] longAliases64 = CLongIs32Bit ? new[] { "LongLongDType" } : new[] { "LongDType" };
            string[] ulongAliases64 = CLongIs32Bit ? new[] { "ULongLongDType" } : new[] { "ULongDType" };

            // NumPy's type_num AND type char for the fixed-width 32/64-bit integers are PLATFORM-DEPENDENT, because
            // `np.dtype('int32')`/`np.dtype('int64')` resolve to whichever C integer is that width (verified against
            // numpy 2.4.2). On LP64 (Linux/macOS) C `long` is 64-bit: int32 == NPY_INT (5,'i'), int64 == NPY_LONG (7,'l').
            // On LLP64 (Windows / any 32-bit process) C `long` is 32-bit: int32 == NPY_LONG (7,'l'), int64 ==
            // NPY_LONGLONG (9,'q'). We register NumSharp's single Int32/Int64/UInt32/UInt64 with the LOCAL platform's
            // (num, char) so `a.dtype.num`/`.type_num`/`.char` are byte-identical to `np.dtype(...)` on the host — the
            // same `CLongIs32Bit` switch the LongDType/LongLongDType name aliases above already use (this also makes the
            // nums CONSISTENT with those aliases, which they were not while the nums were hard-coded LP64).
            int int32Num = CLongIs32Bit ? 7 : 5;   char int32Char = CLongIs32Bit ? 'l' : 'i';
            int uint32Num = CLongIs32Bit ? 8 : 6;   char uint32Char = CLongIs32Bit ? 'L' : 'I';
            int int64Num = CLongIs32Bit ? 9 : 7;   char int64Char = CLongIs32Bit ? 'q' : 'l';
            int uint64Num = CLongIs32Bit ? 10 : 8;  char uint64Char = CLongIs32Bit ? 'Q' : 'L';

            // NumPy type numbers (NPY_TYPES), kinds and chars per docs/plans/dtype-system.md §2.2 (the four C-integer
            // types are platform-resolved above).
            Bool = Legacy("BoolDType", numpy, 0, NPTypeCode.Boolean, typeof(bool), "bool", 'b', '?', 1, 1, builtin, null);
            Int8 = Legacy("Int8DType", numpy, 1, NPTypeCode.SByte, typeof(sbyte), "int8", 'i', 'b', 1, 1, builtin, new[] { "ByteDType" });
            UInt8 = Legacy("UInt8DType", numpy, 2, NPTypeCode.Byte, typeof(byte), "uint8", 'u', 'B', 1, 1, builtin, new[] { "UByteDType" });
            Int16 = Legacy("Int16DType", numpy, 3, NPTypeCode.Int16, typeof(short), "int16", 'i', 'h', 2, 2, builtin, new[] { "ShortDType" });
            UInt16 = Legacy("UInt16DType", numpy, 4, NPTypeCode.UInt16, typeof(ushort), "uint16", 'u', 'H', 2, 2, builtin, new[] { "UShortDType" });
            // NumPy's C-type classes that share a storage lane with the canonical class of their width
            // (LegacyBuiltinDTypeMeta's remarks), registered in type-number order beside the owners. Each is only reached
            // by name ("intc", "longdouble"), type character ('i', 'g') and type number: the owner keeps the NPTypeCode
            // and the C# element type.
            if (CLongIs32Bit)
            {
                // LLP64: int and long are both 32-bit, and int32 is the C long; the C int keeps its own class.
                CInt = Legacy("IntDType", numpy, 5, NPTypeCode.Int32, typeof(int), "intc", 'i', 'i', 4, 4, builtin, null, ownsLane: false);
                CUInt = Legacy("UIntDType", numpy, 6, NPTypeCode.UInt32, typeof(uint), "uintc", 'u', 'I', 4, 4, builtin, null, ownsLane: false);
            }
            Int32 = Legacy("Int32DType", numpy, int32Num, NPTypeCode.Int32, typeof(int), "int32", 'i', int32Char, 4, 4, builtin, longAliases32);
            UInt32 = Legacy("UInt32DType", numpy, uint32Num, NPTypeCode.UInt32, typeof(uint), "uint32", 'u', uint32Char, 4, 4, builtin, ulongAliases32);
            Int64 = Legacy("Int64DType", numpy, int64Num, NPTypeCode.Int64, typeof(long), "int64", 'i', int64Char, 8, 8, builtin, longAliases64);
            UInt64 = Legacy("UInt64DType", numpy, uint64Num, NPTypeCode.UInt64, typeof(ulong), "uint64", 'u', uint64Char, 8, 8, builtin, ulongAliases64);
            if (CLongIs32Bit)
            {
                CLongLong = Int64;
                CULongLong = UInt64;
            }
            else
            {
                // LP64: long and long long are both 64-bit, and int64 is the C long; the C long long keeps its own class.
                CInt = Int32;
                CUInt = UInt32;
                CLongLong = Legacy("LongLongDType", numpy, 9, NPTypeCode.Int64, typeof(long), "longlong", 'i', 'q', 8, 8, builtin, null, ownsLane: false);
                CULongLong = Legacy("ULongLongDType", numpy, 10, NPTypeCode.UInt64, typeof(ulong), "ulonglong", 'u', 'Q', 8, 8, builtin, null, ownsLane: false);
            }
            Single = Legacy("Float32DType", numpy, 11, NPTypeCode.Single, typeof(float), "float32", 'f', 'f', 4, 4, builtin, null);
            Double = Legacy("Float64DType", numpy, 12, NPTypeCode.Double, typeof(double), "float64", 'f', 'd', 8, 8, builtin, null);
            // NumSharp has no extended precision: long double is the 8-byte double of NumPy's win-amd64 build everywhere.
            LongDouble = Legacy("LongDoubleDType", numpy, 13, NPTypeCode.Double, typeof(double), "longdouble", 'f', 'g', 8, 8, builtin, null, ownsLane: false);
            Complex128 = Legacy("Complex128DType", numpy, 15, NPTypeCode.Complex, typeof(Complex), "complex128", 'c', 'D', 16, 8, builtin, null);
            CLongDouble = Legacy("CLongDoubleDType", numpy, 16, NPTypeCode.Complex, typeof(Complex), "clongdouble", 'c', 'G', 16, 8, builtin, null, ownsLane: false);
            Half = Legacy("Float16DType", numpy, 23, NPTypeCode.Half, typeof(System.Half), "float16", 'f', 'e', 2, 2, builtin, null);
            Decimal = Legacy("DecimalDType", numsharp, 256, NPTypeCode.Decimal, typeof(decimal), "decimal", 'f', (char)NPY_TYPECHAR.NPY_LONGLONGLTR, 16, 8, builtin, null);
            Char = Legacy("CharDType", numsharp, 257, NPTypeCode.Char, typeof(char), "char", 'u', (char)NPY_TYPECHAR.NPY_CHARLTR, 2, 2, builtin, null);

            // Every numeric storage class, the lane-sharing C-type classes included (in type-number order beside the
            // owners): each numeric cast table below spans them all, as NumPy's loops over its legacy type numbers do.
            var numericStorage = new List<LegacyBuiltinDTypeMeta> { Bool, Int8, UInt8, Int16, UInt16 };
            if (CLongIs32Bit)
            {
                numericStorage.Add(CInt);
                numericStorage.Add(CUInt);
            }
            numericStorage.AddRange(new[] { Int32, UInt32, Int64, UInt64 });
            if (!CLongIs32Bit)
            {
                numericStorage.Add(CLongLong);
                numericStorage.Add(CULongLong);
            }
            numericStorage.AddRange(new[] { Half, Single, Double, LongDouble, Complex128, CLongDouble, Decimal, Char });
            NumericStorage = numericStorage.AsReadOnly();

            // The two fixed-width string classes. Their singletons are NumPy's UNSIZED builtins `S0` (byte order '|') and
            // `U0` (native order): what np.dtype('S') / np.dtype('U') / np.dtype(str) return, isbuiltin == 1. An allocation
            // never uses an unsized descriptor — it becomes the class default (S1 / U1) or a discovered length.
            Bytes = new StringDTypeMeta(isUnicode: false);
            Str = new StringDTypeMeta(isUnicode: true);
            Add(Bytes);
            Add(Str);
            Bytes.singleton = new DType(Bytes, DType.NPY_IGNORE, null, 0, Bytes.TypeChar);
            Str.singleton = new DType(Str, DType.NPY_NATIVE, null, 0, Str.TypeChar);

            DateTime64 = new DatetimeDTypeMeta(isTimedelta: false);
            TimeDelta64 = new DatetimeDTypeMeta(isTimedelta: true);
            Add(DateTime64);
            Add(TimeDelta64);
            DateTime64.singleton = DateTime64.Descr(DatetimeMetaData.Generic);
            TimeDelta64.singleton = TimeDelta64.Descr(DatetimeMetaData.Generic);

            // NEP 55 StringDType: a NEW-STYLE class (no builtin singleton — each descriptor instance is its own, as in
            // NumPy where every instance carries a string allocator). It shares the NPTypeCode.String lane with S / U
            // (decision D4) but owns neither the lane nor the C# string type, so `(DType)NPTypeCode.String` and
            // `np.dtype(typeof(string))` keep meaning `<U`.
            VString = new VStringDTypeMeta();
            Add(VString);

            PyLong = new PyScalarDTypeMeta("_PyLongDType", PyScalarKind.Int, NPTypeCode.Int64);
            PyFloat = new PyScalarDTypeMeta("_PyFloatDType", PyScalarKind.Float, NPTypeCode.Double);
            PyComplex = new PyScalarDTypeMeta("_PyComplexDType", PyScalarKind.Complex, NPTypeCode.Complex);
            _byName[PyLong.Name] = PyLong;
            _byName[PyFloat.Name] = PyFloat;
            _byName[PyComplex.Name] = PyComplex;
            AppendToAll(PyLong);
            AppendToAll(PyFloat);
            AppendToAll(PyComplex);

            InitializeNumericCasts();
            InitializeDatetimeCasts();
            InitializeStringCasts();
            InitializeVStringCasts();
        }

        /// <summary>Builds, installs the singleton of, and registers one builtin class (see <see cref="LegacyBuiltinDTypeMeta"/>'s constructor for the parameters).</summary>
        /// <param name="name">NumPy's class name.</param>
        /// <param name="module">The class's module.</param>
        /// <param name="typeNum">NumPy's type number.</param>
        /// <param name="code">The storage lane.</param>
        /// <param name="scalar">The C# element type.</param>
        /// <param name="scalarName">The scalar class's name.</param>
        /// <param name="kind">NumPy's kind character.</param>
        /// <param name="typeChar">NumPy's type character.</param>
        /// <param name="itemSize">Element size in bytes.</param>
        /// <param name="alignment">Alignment in bytes.</param>
        /// <param name="flags">The <c>NPY_DT_*</c> flags.</param>
        /// <param name="aliases">The alias class names, or null.</param>
        /// <param name="ownsLane">False for a lane-sharing C-type class.</param>
        /// <returns>The registered class.</returns>
        /// <exception cref="InvalidOperationException">The type number, name, an alias or (for a lane owner) the storage code is already registered.</exception>
        private static LegacyBuiltinDTypeMeta Legacy(string name, string module, int typeNum, NPTypeCode code, Type scalar, string scalarName,
            char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags, string[] aliases, bool ownsLane = true)
        {
            var meta = new LegacyBuiltinDTypeMeta(name, module, typeNum, code, scalar, scalarName, kind, typeChar, itemSize, alignment, flags, aliases, ownsLane);
            meta.singleton = new DType(meta, DType.NPY_NATIVE, null);
            Add(meta);
            return meta;
        }

        /// <summary>
        ///     Adds <paramref name="meta"/> to every table it belongs in: its type number, its name and aliases, its storage
        ///     code when it has storage, its C# scalar type unless another class claimed that type first, and the list behind
        ///     <see cref="All"/>.
        /// </summary>
        /// <param name="meta">The class to add.</param>
        /// <exception cref="InvalidOperationException">
        ///     The type number, the name or an alias is already registered; or the class has storage and its
        ///     <see cref="NPTypeCode"/> is outside the code table or already registered. Nothing is written then.
        /// </exception>
        /// <remarks>
        ///     Callers serialize: the static constructor runs on one thread, and <see cref="Register"/> holds
        ///     <see cref="_lock"/>. Lookups run alongside the writes without waiting and see each table before or after
        ///     its write.
        /// </remarks>
        private static void Add(DTypeMeta meta)
        {
            if (_byTypeNum.TryGetValue(meta.type_num, out var holder))
                throw new InvalidOperationException($"A DType with type number {meta.type_num} is already registered ({holder}).");
            if (_byName.ContainsKey(meta.Name))
                throw new InvalidOperationException($"A DType named '{meta.Name}' is already registered.");
            foreach (var alias in meta.Aliases)
                if (_byName.ContainsKey(alias))
                    throw new InvalidOperationException($"A DType named '{alias}' is already registered.");

            // The storage code is checked with the rest, before the first write: a refused class must not stay findable
            // by FromTypeNum or FromName, and a corrected retry of it must not fail on "type number already registered".
            // A storage lane maps back to ONE class. Two classes may share a lane (S and U share NPTypeCode.String), in
            // which case only the lane's owner is indexed — the other is reached by name and through its descriptors.
            int code = -1;
            if (meta.HasStorage && meta.OwnsTypeCode)
            {
                code = (int)meta.TypeCode;
                if (code < 0 || code >= _byCode.Length)
                    throw new InvalidOperationException($"NPTypeCode {meta.TypeCode} is outside the registry's code range.");
                if (Volatile.Read(ref _byCode[code]) is { } taken)
                    throw new InvalidOperationException($"NPTypeCode {meta.TypeCode} is already registered ({taken}).");
            }

            _byTypeNum[meta.type_num] = meta;
            _byName[meta.Name] = meta;
            foreach (var alias in meta.Aliases)
                _byName[alias] = meta;
            if (code >= 0)
                Volatile.Write(ref _byCode[code], meta);
            // Likewise a C# type maps back to one class: the first class registered for it keeps it (StrDType owns
            // typeof(string)), and BytesDType opts out so byte[] keeps meaning "a uint8 array".
            if (meta.type != null && meta.OwnsScalarType)
                _byScalarType.TryAdd(meta.type, meta);
            AppendToAll(meta);
        }

        /// <summary>
        ///     Publishes a copy of the class list with <paramref name="meta"/> appended. A plain copy is enough because
        ///     writers are serialized (the static constructor, or <see cref="Register"/> under <see cref="_lock"/>); a reader
        ///     keeps whichever array it read.
        /// </summary>
        /// <param name="meta">The class to append.</param>
        private static void AppendToAll(DTypeMeta meta)
        {
            var current = _all;
            var next = new DTypeMeta[current.Length + 1];
            Array.Copy(current, next, current.Length);
            next[current.Length] = meta;
            Volatile.Write(ref _all, next);
        }

        /// <summary>NumPy's <c>PyArray_InitializeNumericCasts</c>: one <see cref="BuiltinCastingImpl"/> per ordered pair of storage classes.</summary>
        private static void InitializeNumericCasts()
        {
            var storage = NumericStorage;
            foreach (var from in storage)
                foreach (var to in storage)
                    from.AddCastingImpl(new BuiltinCastingImpl(from, to));
        }

        /// <summary>
        ///     NumPy's <c>PyArray_InitializeDatetimeCasts</c>: the unit conversions within each datetime class, the
        ///     datetime ↔ timedelta pair, and the numeric ↔ datetime/timedelta casts (<c>unsafe</c> to and from
        ///     <c>datetime64</c>; to <c>timedelta64</c> <c>safe</c> from bool and the integers, <c>same_kind</c> from a 64-bit
        ///     unsigned integer, <c>unsafe</c> from floats and complex; <c>unsafe</c> from <c>timedelta64</c>).
        /// </summary>
        private static void InitializeDatetimeCasts()
        {
            DateTime64.AddCastingImpl(new TimeToTimeCastingImpl(DateTime64));
            TimeDelta64.AddCastingImpl(new TimeToTimeCastingImpl(TimeDelta64));

            /*
             * Casting between timedelta and datetime uses legacy casting loops, but
             * custom dtype resolution (to handle copying of the time unit).
             */
            TimeDelta64.AddCastingImpl(new DatetimeTimedeltaCastingImpl(TimeDelta64, DateTime64));
            DateTime64.AddCastingImpl(new DatetimeTimedeltaCastingImpl(DateTime64, TimeDelta64));

            /*
             * Cast from numeric types to times.  These use the cast functions
             * as stored on the datatype, which should be replaced at some point.
             * Some of these casts can fail (casting to unitless datetime), but these
             * are rather special.
             */
            var storage = NumericStorage;
            foreach (var num in storage)
            {
                num.AddCastingImpl(new LegacyWrappingCastingImpl(num, DateTime64, NPY_CASTING.NPY_UNSAFE_CASTING));
                DateTime64.AddCastingImpl(new LegacyWrappingCastingImpl(DateTime64, num, NPY_CASTING.NPY_UNSAFE_CASTING));

                var toTimedeltaCasting = NPY_CASTING.NPY_UNSAFE_CASTING;
                if (num.Kind == 'b' || num.Kind == 'i' || num.Kind == 'u')
                {
                    /* timedelta casts like int64 right now... */
                    toTimedeltaCasting = num.Kind == 'u' && num.ItemSize == 8
                        ? NPY_CASTING.NPY_SAME_KIND_CASTING
                        : NPY_CASTING.NPY_SAFE_CASTING;
                }
                num.AddCastingImpl(new LegacyWrappingCastingImpl(num, TimeDelta64, toTimedeltaCasting));
                TimeDelta64.AddCastingImpl(new LegacyWrappingCastingImpl(TimeDelta64, num, NPY_CASTING.NPY_UNSAFE_CASTING));
            }
        }

        /// <summary>
        ///     NumPy's <c>PyArray_InitializeStringCasts</c> plus the string half of <c>PyArray_InitializeDatetimeCasts</c>:
        ///     <list type="bullet">
        ///       <item><description>every numeric storage class → <c>S</c>/<c>U</c>: <see cref="ToStringCastingImpl"/>
        ///       (<c>same_kind</c>, <c>safe</c> when the destination is long enough);</description></item>
        ///       <item><description><c>S</c>/<c>U</c> → every numeric storage class: <c>unsafe</c> legacy casts;</description></item>
        ///       <item><description><c>S → U</c>: <see cref="ToStringCastingImpl"/> (<c>same_kind</c>); <c>U → S</c>: the same
        ///       resolver with an <c>unsafe</c> class level;</description></item>
        ///       <item><description><c>S → S</c>, <c>U → U</c>: <see cref="StringToStringCastingImpl"/>;</description></item>
        ///       <item><description><c>datetime64</c>/<c>timedelta64</c> → <c>S</c>/<c>U</c>: <see cref="TimeToStringCastingImpl"/>;
        ///       <c>S</c>/<c>U</c> → <c>datetime64</c>: <see cref="StringToDatetimeCastingImpl"/>; <c>S</c>/<c>U</c> →
        ///       <c>timedelta64</c>: an <c>unsafe</c> legacy cast.</description></item>
        ///     </list>
        ///     NumPy's registration loop runs over the legacy type numbers and skips the datetime, void and object classes;
        ///     NumSharp's Decimal and Char join it as the numeric storage classes they are.
        /// </summary>
        private static void InitializeStringCasts()
        {
            var numeric = NumericStorage;
            foreach (var str in new[] { Bytes, Str })
            {
                foreach (var num in numeric)
                {
                    /* Casting from string, is always a simple legacy-style cast */
                    str.AddCastingImpl(new LegacyWrappingCastingImpl(str, num, NPY_CASTING.NPY_UNSAFE_CASTING));
                    /* Casting to strings ... requires a custom resolver to define the correct string length. */
                    num.AddCastingImpl(new ToStringCastingImpl(num, str));
                }

                /* string<->string and unicode<->unicode have their own specialized casts */
                str.AddCastingImpl(new StringToStringCastingImpl(str));

                /* Cast times to string and unicode */
                DateTime64.AddCastingImpl(new TimeToStringCastingImpl(DateTime64, str));
                TimeDelta64.AddCastingImpl(new TimeToStringCastingImpl(TimeDelta64, str));
                /* Cast strings to timedelta are currently only legacy casts */
                str.AddCastingImpl(new LegacyWrappingCastingImpl(str, TimeDelta64, NPY_CASTING.NPY_UNSAFE_CASTING));
                /* Cast strings to datetime */
                str.AddCastingImpl(new StringToDatetimeCastingImpl(str, DateTime64));
            }

            /* S -> U is same-kind (safe when long enough); U -> S is unsafe (a code point above 127 cannot be stored). */
            Bytes.AddCastingImpl(new ToStringCastingImpl(Bytes, Str));
            Str.AddCastingImpl(new ToStringCastingImpl(Str, Bytes));
        }

        /// <summary>
        ///     NumPy's <c>stringdtype/casts.cpp</c> <c>get_casts()</c>, class by class: within <c>StringDType</c> (minimal
        ///     <c>unsafe</c>, resolved to <c>no</c> casting unless an NA-holding descriptor loses its NA); <c>U</c> ↔ <c>T</c>
        ///     and <c>S</c> ↔ <c>T</c> <c>same_kind</c> (and only to a sized <c>U</c> / <c>S</c>); <c>T</c> ↔ bool (minimal
        ///     <c>same_kind</c> both ways, resolved <c>unsafe</c> out of <c>T</c> and <c>safe</c> into it); numbers and the
        ///     datetime pair into <c>T</c> <c>safe</c>, <c>T</c> into them <c>unsafe</c>. NumSharp's Decimal takes float64's
        ///     casts and its Char uint16's (the classes they convert as); NumPy's void casts have no NumSharp class.
        /// </summary>
        private static void InitializeVStringCasts()
        {
            VString.AddCastingImpl(new VStringToVStringCastingImpl(VString));

            Str.AddCastingImpl(new AnyToVStringCastingImpl(Str, VString, NPY_CASTING.NPY_SAME_KIND_CASTING, NPY_CASTING.NPY_SAME_KIND_CASTING));
            VString.AddCastingImpl(new VStringToFixedWidthCastingImpl(VString, Str));
            Bytes.AddCastingImpl(new AnyToVStringCastingImpl(Bytes, VString, NPY_CASTING.NPY_SAME_KIND_CASTING, NPY_CASTING.NPY_SAME_KIND_CASTING));
            VString.AddCastingImpl(new VStringToFixedWidthCastingImpl(VString, Bytes));

            Bool.AddCastingImpl(new AnyToVStringCastingImpl(Bool, VString, NPY_CASTING.NPY_SAME_KIND_CASTING, NPY_CASTING.NPY_SAFE_CASTING));
            VString.AddCastingImpl(new VStringToNumberCastingImpl(VString, Bool, NPY_CASTING.NPY_SAME_KIND_CASTING));

            // Every numeric class but bool, which has its own pair of casts above.
            var numbers = NumericStorage.Where(m => !ReferenceEquals(m, Bool));
            foreach (var num in numbers)
            {
                num.AddCastingImpl(new AnyToVStringCastingImpl(num, VString, NPY_CASTING.NPY_SAFE_CASTING, NPY_CASTING.NPY_SAFE_CASTING));
                VString.AddCastingImpl(new VStringToNumberCastingImpl(VString, num, NPY_CASTING.NPY_UNSAFE_CASTING));
            }

            foreach (var time in new[] { DateTime64, TimeDelta64 })
            {
                time.AddCastingImpl(new AnyToVStringCastingImpl(time, VString, NPY_CASTING.NPY_SAFE_CASTING, NPY_CASTING.NPY_SAFE_CASTING));
                VString.AddCastingImpl(new VStringToNumberCastingImpl(VString, time, NPY_CASTING.NPY_UNSAFE_CASTING));
            }
        }

        // ---- lookups ----

        /// <summary>
        ///     The class whose storage discriminator is <paramref name="typeCode"/> — behind every
        ///     <see cref="DType.From(NPTypeCode)"/>. Takes no lock.
        /// </summary>
        /// <param name="typeCode">The storage code.</param>
        /// <returns>The class, or null for <see cref="NPTypeCode.Empty"/>, a code outside the table, or an unregistered code.</returns>
        public static DTypeMeta FromTypeCode(NPTypeCode typeCode)
        {
            int code = (int)typeCode;
            if (code <= 0 || code >= _byCode.Length)
                return null;
            return Volatile.Read(ref _byCode[code]);
        }

        /// <summary>
        ///     The class of a C# scalar <see cref="Type"/> (NEP 41's <c>np.dtype[np.float64]</c> / NumPy's
        ///     <c>PyArray_DiscoverDTypeFromScalarType</c>): <c>typeof(double)</c> → <see cref="Double"/>. Array types map to
        ///     their element type (<c>np.array(int[])</c> is int32). Null when the type is not a NumSharp scalar.
        /// </summary>
        public static DTypeMeta FromScalarType(Type type)
        {
            if (type == null)
                return null;
            if (_byScalarType.TryGetValue(type, out var meta))
                return meta;
            var code = type.GetTypeCode();
            return code == NPTypeCode.Empty ? null : FromTypeCode(code);
        }

        /// <summary>The class with NumPy type number <paramref name="typeNum"/>, or null.</summary>
        public static DTypeMeta FromTypeNum(int typeNum) => _byTypeNum.TryGetValue(typeNum, out var meta) ? meta : null;

        /// <summary>The class named <paramref name="name"/> (<c>"Int32DType"</c>) or one of its aliases (<c>"IntDType"</c>), or null.</summary>
        public static DTypeMeta FromName(string name) => name != null && _byName.TryGetValue(name, out var meta) ? meta : null;

        /// <summary>Every registered class (builtins, datetime pair, the abstract scalar classes, user classes), in registration order.</summary>
        /// <value>
        ///     A new array on every read, taken without a lock: the caller may keep or change it, and a registration that
        ///     runs meanwhile is either wholly in it or not in it.
        /// </value>
        public static IReadOnlyList<DTypeMeta> All => (DTypeMeta[])Volatile.Read(ref _all).Clone();

        /// <summary>
        ///     The published registration array itself, without <see cref="All"/>'s copy — for internal readers that cache
        ///     something derived from the registry and must notice a registration: every registration publishes a NEW
        ///     array, so a reference compare with the array a cache was built from tells whether it is still current.
        /// </summary>
        /// <value>The live array. Callers must never write into it: it is shared with every other reader.</value>
        internal static DTypeMeta[] AllSnapshot => Volatile.Read(ref _all);

        /// <summary>
        ///     Registers a new DType class (a user dtype). Its type number, name and aliases must be unused; a class with
        ///     storage must carry an unused <see cref="NPTypeCode"/>. Casting implementations are added separately with
        ///     <see cref="AddCastingImpl"/>.
        /// </summary>
        /// <param name="meta">The class to register.</param>
        /// <exception cref="ArgumentNullException"><paramref name="meta"/> is null.</exception>
        /// <exception cref="InvalidOperationException">
        ///     The type number, the name or an alias is already registered, or the storage code is outside the code table or
        ///     already registered. Nothing is registered then, so the class can be registered once its conflict is fixed.
        /// </exception>
        /// <remarks>
        ///     The only member that takes a lock: <see cref="_lock"/> keeps two registrations from interleaving between their
        ///     checks and their writes. Lookups never wait for it.
        /// </remarks>
        public static void Register(DTypeMeta meta)
        {
            if (meta == null)
                throw new ArgumentNullException(nameof(meta));
            lock (_lock)
                Add(meta);
        }

        /// <summary>Registers a casting implementation on its source class — NumPy's <c>PyArray_AddCastingImplementation</c>.</summary>
        public static void AddCastingImpl(CastingImpl impl)
        {
            if (impl == null)
                throw new ArgumentNullException(nameof(impl));
            impl.From.AddCastingImpl(impl);
        }
    }
}
