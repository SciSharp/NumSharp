using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
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
    ///     Static initialisation builds the 16 legacy classes (the 15 storage types plus the vestigial
    ///     <see cref="NPTypeCode.String"/>), the two datetime classes, the three NEP 50 scalar classes, their singleton
    ///     descriptors, and the casting implementations NumPy registers in <c>PyArray_InitializeNumericCasts</c> /
    ///     <c>PyArray_InitializeDatetimeCasts</c>. It touches no <c>np</c> static state, so it cannot cycle with the
    ///     <c>np</c> static constructor; the promotion TABLE (<see cref="np._nptypemap_arr_arr"/>) is consulted lazily at
    ///     call time.
    /// </remarks>
    public static class DTypeRegistry
    {
        private static readonly object _lock = new object();
        private static readonly DTypeMeta[] _byCode = new DTypeMeta[129];
        private static readonly Dictionary<int, DTypeMeta> _byTypeNum = new Dictionary<int, DTypeMeta>();
        private static readonly Dictionary<string, DTypeMeta> _byName = new Dictionary<string, DTypeMeta>(StringComparer.Ordinal);
        private static readonly Dictionary<Type, DTypeMeta> _byScalarType = new Dictionary<Type, DTypeMeta>();
        private static readonly List<DTypeMeta> _all = new List<DTypeMeta>();

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
        /// <summary>NumSharp-only: <see cref="System.Decimal"/> (type number 256, the user-defined range).</summary>
        public static LegacyBuiltinDTypeMeta Decimal { get; }
        /// <summary>NumSharp-only: <see cref="System.Char"/> (type number 257, the user-defined range; promotes as uint16).</summary>
        public static LegacyBuiltinDTypeMeta Char { get; }
        /// <summary>The vestigial <see cref="NPTypeCode.String"/> class (NumPy's <c>StrDType</c> slot, type number 19): reachable only through the <c>"string"</c>/<c>"String"</c> aliases, no storage kernels.</summary>
        public static LegacyBuiltinDTypeMeta Str { get; }
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
            const DTypeFlags builtin = DTypeFlags.Legacy | DTypeFlags.Numeric;

            string[] longAliases32 = CLongIs32Bit ? new[] { "IntDType", "LongDType" } : new[] { "IntDType" };
            string[] ulongAliases32 = CLongIs32Bit ? new[] { "UIntDType", "ULongDType" } : new[] { "UIntDType" };
            string[] longAliases64 = CLongIs32Bit ? new[] { "LongLongDType" } : new[] { "LongLongDType", "LongDType" };
            string[] ulongAliases64 = CLongIs32Bit ? new[] { "ULongLongDType" } : new[] { "ULongLongDType", "ULongDType" };

            // NumPy type numbers (NPY_TYPES, LP64 convention), kinds and chars per docs/plans/dtype-system.md §2.2.
            Bool = Legacy("BoolDType", numpy, 0, NPTypeCode.Boolean, typeof(bool), "bool", 'b', '?', 1, 1, builtin, null);
            Int8 = Legacy("Int8DType", numpy, 1, NPTypeCode.SByte, typeof(sbyte), "int8", 'i', 'b', 1, 1, builtin, new[] { "ByteDType" });
            UInt8 = Legacy("UInt8DType", numpy, 2, NPTypeCode.Byte, typeof(byte), "uint8", 'u', 'B', 1, 1, builtin, new[] { "UByteDType" });
            Int16 = Legacy("Int16DType", numpy, 3, NPTypeCode.Int16, typeof(short), "int16", 'i', 'h', 2, 2, builtin, new[] { "ShortDType" });
            UInt16 = Legacy("UInt16DType", numpy, 4, NPTypeCode.UInt16, typeof(ushort), "uint16", 'u', 'H', 2, 2, builtin, new[] { "UShortDType" });
            Int32 = Legacy("Int32DType", numpy, 5, NPTypeCode.Int32, typeof(int), "int32", 'i', 'i', 4, 4, builtin, longAliases32);
            UInt32 = Legacy("UInt32DType", numpy, 6, NPTypeCode.UInt32, typeof(uint), "uint32", 'u', 'I', 4, 4, builtin, ulongAliases32);
            Int64 = Legacy("Int64DType", numpy, 7, NPTypeCode.Int64, typeof(long), "int64", 'i', 'l', 8, 8, builtin, longAliases64);
            UInt64 = Legacy("UInt64DType", numpy, 8, NPTypeCode.UInt64, typeof(ulong), "uint64", 'u', 'L', 8, 8, builtin, ulongAliases64);
            Single = Legacy("Float32DType", numpy, 11, NPTypeCode.Single, typeof(float), "float32", 'f', 'f', 4, 4, builtin, null);
            Double = Legacy("Float64DType", numpy, 12, NPTypeCode.Double, typeof(double), "float64", 'f', 'd', 8, 8, builtin, new[] { "LongDoubleDType" });
            Complex128 = Legacy("Complex128DType", numpy, 15, NPTypeCode.Complex, typeof(Complex), "complex128", 'c', 'D', 16, 8, builtin, new[] { "CLongDoubleDType" });
            Half = Legacy("Float16DType", numpy, 23, NPTypeCode.Half, typeof(System.Half), "float16", 'f', 'e', 2, 2, builtin, null);
            Decimal = Legacy("DecimalDType", numsharp, 256, NPTypeCode.Decimal, typeof(decimal), "decimal", 'f', (char)NPY_TYPECHAR.NPY_LONGLONGLTR, 16, 8, builtin, null);
            Char = Legacy("CharDType", numsharp, 257, NPTypeCode.Char, typeof(char), "char", 'u', (char)NPY_TYPECHAR.NPY_CHARLTR, 2, 2, builtin, null);
            // The vestigial string slot is NumPy's UNSIZED `U0`: itemsize 0, so it renders as `<U0` / dtype('<U') and never
            // masquerades as a 1-byte type (NPTypeCode.String.SizeOf() still reports 1 for the old char-based code paths).
            Str = Legacy("StrDType", numpy, 19, NPTypeCode.String, typeof(string), "str", 'U', 'U', 0, 1, DTypeFlags.Legacy | DTypeFlags.Parametric, null);

            DateTime64 = new DatetimeDTypeMeta(isTimedelta: false);
            TimeDelta64 = new DatetimeDTypeMeta(isTimedelta: true);
            Add(DateTime64);
            Add(TimeDelta64);
            DateTime64.Singleton = DateTime64.Descr(DatetimeMetaData.Generic);
            TimeDelta64.Singleton = TimeDelta64.Descr(DatetimeMetaData.Generic);

            PyLong = new PyScalarDTypeMeta("_PyLongDType", PyScalarKind.Int, NPTypeCode.Int64);
            PyFloat = new PyScalarDTypeMeta("_PyFloatDType", PyScalarKind.Float, NPTypeCode.Double);
            PyComplex = new PyScalarDTypeMeta("_PyComplexDType", PyScalarKind.Complex, NPTypeCode.Complex);
            _byName[PyLong.Name] = PyLong;
            _byName[PyFloat.Name] = PyFloat;
            _byName[PyComplex.Name] = PyComplex;
            _all.Add(PyLong);
            _all.Add(PyFloat);
            _all.Add(PyComplex);

            InitializeNumericCasts();
            InitializeDatetimeCasts();
        }

        private static LegacyBuiltinDTypeMeta Legacy(string name, string module, int typeNum, NPTypeCode code, Type scalar, string scalarName,
            char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags, string[] aliases)
        {
            var meta = new LegacyBuiltinDTypeMeta(name, module, typeNum, code, scalar, scalarName, kind, typeChar, itemSize, alignment, flags, aliases);
            meta.Singleton = new DType(meta, DType.NativeByteOrder, null);
            Add(meta);
            return meta;
        }

        private static void Add(DTypeMeta meta)
        {
            if (_byTypeNum.ContainsKey(meta.TypeNum))
                throw new InvalidOperationException($"A DType with type number {meta.TypeNum} is already registered ({_byTypeNum[meta.TypeNum]}).");
            if (_byName.ContainsKey(meta.Name))
                throw new InvalidOperationException($"A DType named '{meta.Name}' is already registered.");
            foreach (var alias in meta.Aliases)
                if (_byName.ContainsKey(alias))
                    throw new InvalidOperationException($"A DType named '{alias}' is already registered.");

            _byTypeNum[meta.TypeNum] = meta;
            _byName[meta.Name] = meta;
            foreach (var alias in meta.Aliases)
                _byName[alias] = meta;
            if (meta.HasStorage)
            {
                int code = (int)meta.TypeCode;
                if (code < 0 || code >= _byCode.Length)
                    throw new InvalidOperationException($"NPTypeCode {meta.TypeCode} is outside the registry's code range.");
                if (_byCode[code] != null)
                    throw new InvalidOperationException($"NPTypeCode {meta.TypeCode} is already registered ({_byCode[code]}).");
                _byCode[code] = meta;
            }
            if (meta.ScalarType != null && !_byScalarType.ContainsKey(meta.ScalarType))
                _byScalarType[meta.ScalarType] = meta;
            _all.Add(meta);
        }

        /// <summary>NumPy's <c>PyArray_InitializeNumericCasts</c>: one <see cref="BuiltinCastingImpl"/> per ordered pair of storage classes.</summary>
        private static void InitializeNumericCasts()
        {
            var storage = new[] { Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Half, Single, Double, Complex128, Decimal, Char };
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
            var storage = new[] { Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Half, Single, Double, Complex128, Decimal, Char };
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

        // ---- lookups ----

        /// <summary>The class whose storage discriminator is <paramref name="typeCode"/>; null for <see cref="NPTypeCode.Empty"/> / an unregistered code.</summary>
        public static DTypeMeta FromTypeCode(NPTypeCode typeCode)
        {
            int code = (int)typeCode;
            if (code <= 0 || code >= _byCode.Length)
                return null;
            return _byCode[code];
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

        /// <summary>Every registered class (builtins, datetime pair, the abstract scalar classes, user classes).</summary>
        public static IReadOnlyList<DTypeMeta> All
        {
            get
            {
                lock (_lock)
                    return _all.ToArray();
            }
        }

        /// <summary>
        ///     Registers a new DType class (a user dtype). Its type number, name and aliases must be unused; a class with
        ///     storage must carry an unused <see cref="NPTypeCode"/>. Casting implementations are added separately with
        ///     <see cref="AddCastingImpl"/>.
        /// </summary>
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
