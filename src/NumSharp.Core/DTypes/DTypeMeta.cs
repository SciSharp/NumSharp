using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     The DType CLASS — NumPy's <c>PyArray_DTypeMeta</c> (<c>type(np.dtype('f8'))</c>, i.e.
    ///     <c>np.dtypes.Float64DType</c>). One live object per dtype class; a <see cref="DType"/> is an INSTANCE of it
    ///     (the descriptor). This is NEP 41/42's two-level model: behaviour lives on the class as virtual "slots"
    ///     (<see cref="CommonDType"/>, <see cref="CommonInstance"/>, <see cref="DefaultDescr"/>,
    ///     <see cref="EnsureCanonical"/>, <see cref="DiscoverDescrFromObject"/>, <see cref="IsKnownScalarType"/>, the
    ///     casting implementations), parameters live on the instance (a datetime unit, a byte order).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Compatibility contract.</b> <see cref="NPTypeCode"/> stays the storage / kernel discriminator: every meta
    ///     that has storage exposes it as <see cref="TypeCode"/>, and a future storage-backed dtype adds one enum
    ///     member and registers it here — the enum is an optimisation the registry hands out, not the identity. A meta
    ///     without storage (<see cref="DatetimeDTypeMeta"/> in Stage A) has <see cref="TypeCode"/> ==
    ///     <see cref="NPTypeCode.Empty"/>; allocating storage for its descriptors raises
    ///     <see cref="NotSupportedException"/> naming the missing piece.
    ///     </para>
    ///     <para>
    ///     <b>Type numbers</b> follow NumPy's <c>NPY_TYPES</c> (LP64 convention: <c>Int32 = NPY_INT (5)</c>,
    ///     <c>Int64 = NPY_LONG (7)</c>, <c>Half = 23</c>, <c>DateTime = 21</c>, <c>TimeDelta = 22</c>); the two NumSharp-only
    ///     dtypes take the user-defined range (<c>Decimal = 256</c>, <c>Char = 257</c>), which is also what makes their
    ///     descriptors report <c>isbuiltin == 2</c>. The number ORDER is load-bearing: the builtin
    ///     <c>common_dtype</c> defers to the operand with the larger type number ("the more generic one handles it"),
    ///     exactly as NumPy's <c>default_builtin_common_dtype</c> does.
    ///     </para>
    ///     <para>
    ///     <b>NotImplemented</b> is spelled <see langword="null"/>: <see cref="CommonDType"/> returns null when this class
    ///     does not know the other one, and <see cref="DTypePromotion.CommonDType"/> then asks the other class
    ///     (NEP 42's two-sided <c>__common_dtype__</c> protocol) before raising <see cref="DTypePromotionError"/>.
    ///     </para>
    ///     <para>
    ///     <b>NEP 55 reservations.</b> <see cref="HasReferences"/>, <see cref="GetClearLoop"/> and
    ///     <see cref="GetFillZeroLoop"/> are the slots a reference-holding dtype (<c>StringDType</c>) will override; the
    ///     builtins report false / null.
    ///     </para>
    /// </remarks>
    public abstract class DTypeMeta
    {
        private readonly Dictionary<DTypeMeta, CastingImpl> _castingImpls = new Dictionary<DTypeMeta, CastingImpl>();
        private CastingImpl _withinDTypeCastingImpl;
        private DType _singleton;

        /// <summary>
        ///     Builds a DType class. Called by the registry / subclasses only — every class is a singleton reachable via
        ///     <see cref="DTypeRegistry"/> and <c>np.dtypes</c>.
        /// </summary>
        /// <param name="name">NumPy's class name (<c>"Int32DType"</c>, <c>"DateTime64DType"</c>, <c>"_PyLongDType"</c>).</param>
        /// <param name="modulePath">The module the class lives in — <c>"numpy.dtypes"</c> for the NumPy-mirrored classes,
        /// <c>"numsharp.dtypes"</c> for the NumSharp-only ones — so <see cref="ToString"/> renders NumPy's
        /// <c>&lt;class 'numpy.dtypes.Int32DType'&gt;</c>.</param>
        /// <param name="typeNum">NumPy's <c>type_num</c> (see the class remarks); -1 for abstract classes.</param>
        /// <param name="scalarType">The C# scalar type of one element (<c>typeof(int)</c>); null when there is none yet.</param>
        /// <param name="scalarName">NumPy's <c>dtype.type.__name__</c> (<c>"int32"</c>, <c>"datetime64"</c>, <c>"decimal"</c>).</param>
        /// <param name="typeCode">The storage discriminator, <see cref="NPTypeCode.Empty"/> when the class has no storage.</param>
        /// <param name="kind">NumPy's <c>dtype.kind</c> character (<c>'b' 'i' 'u' 'f' 'c' 'M' 'm' 'U'</c>).</param>
        /// <param name="typeChar">NumPy's <c>dtype.char</c> (<c>'?' 'b' 'B' 'h' 'H' 'i' 'I' 'l' 'L' 'e' 'f' 'd' 'D' 'M' 'm'</c>).</param>
        /// <param name="itemSize">Element size in bytes (-1 for abstract classes).</param>
        /// <param name="alignment">Required alignment in bytes (NumPy's <c>dtype.alignment</c>).</param>
        /// <param name="flags">The <see cref="DTypeFlags"/> (legacy / abstract / parametric / numeric).</param>
        /// <param name="aliases">Alias class names NumPy also exports (<c>"IntDType"</c> for <c>Int32DType</c>).</param>
        protected DTypeMeta(string name, string modulePath, int typeNum, Type scalarType, string scalarName,
            NPTypeCode typeCode, char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags,
            IReadOnlyList<string> aliases = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            ModulePath = modulePath ?? throw new ArgumentNullException(nameof(modulePath));
            TypeNum = typeNum;
            ScalarType = scalarType;
            ScalarName = scalarName ?? name;
            TypeCode = typeCode;
            Kind = kind;
            TypeChar = typeChar;
            ItemSize = itemSize;
            Alignment = alignment;
            Flags = flags;
            Aliases = aliases ?? Array.Empty<string>();
        }

        /// <summary>NumPy's class name: <c>"Int32DType"</c>, <c>"DateTime64DType"</c>, <c>"_PyLongDType"</c>.</summary>
        public string Name { get; }

        /// <summary>The module path used in <see cref="ToString"/>: <c>"numpy.dtypes"</c> or <c>"numsharp.dtypes"</c>.</summary>
        public string ModulePath { get; }

        /// <summary>The fully qualified class name: <c>"numpy.dtypes.Int32DType"</c>.</summary>
        public string FullName => ModulePath + "." + Name;

        /// <summary>NumPy's <c>type_num</c> (<c>NPY_TYPES</c>, LP64 convention; 256+ for the NumSharp-only dtypes; -1 for abstract).</summary>
        public int TypeNum { get; }

        /// <summary>The C# scalar type of one element (NumPy's <c>DType.type</c>); null when the class has no C# scalar yet.</summary>
        public Type ScalarType { get; }

        /// <summary>NumPy's <c>dtype.type.__name__</c>: <c>"int32"</c>, <c>"bool"</c>, <c>"datetime64"</c>, <c>"decimal"</c>.</summary>
        public string ScalarName { get; }

        /// <summary>The storage / kernel discriminator; <see cref="NPTypeCode.Empty"/> when the class has no storage.</summary>
        public NPTypeCode TypeCode { get; }

        /// <summary>NumPy's <c>dtype.kind</c> character (<c>'b'</c> bool, <c>'i'</c>/<c>'u'</c> integers, <c>'f'</c>, <c>'c'</c>, <c>'M'</c>, <c>'m'</c>, <c>'U'</c>).</summary>
        public char Kind { get; }

        /// <summary>NumPy's <c>dtype.char</c> — the unique one-character type code.</summary>
        public char TypeChar { get; }

        /// <summary>Element size in bytes (-1 for abstract classes).</summary>
        public int ItemSize { get; }

        /// <summary>Required alignment in bytes.</summary>
        public int Alignment { get; }

        /// <summary>The class flags.</summary>
        public DTypeFlags Flags { get; }

        /// <summary>Alias class names NumPy also exports for this class (e.g. <c>ByteDType</c> for <c>Int8DType</c>).</summary>
        public IReadOnlyList<string> Aliases { get; }

        /// <summary>A legacy (pre-NEP 41 layout) dtype — every builtin and the datetime pair.</summary>
        public bool IsLegacy => (Flags & DTypeFlags.Legacy) != 0;

        /// <summary>An abstract dtype (no instances; promotion only).</summary>
        public bool IsAbstract => (Flags & DTypeFlags.Abstract) != 0;

        /// <summary>A parametric dtype (instances carry parameters — a unit, a length).</summary>
        public bool IsParametric => (Flags & DTypeFlags.Parametric) != 0;

        /// <summary>A numeric dtype (bool, integer, float, complex).</summary>
        public bool IsNumeric => (Flags & DTypeFlags.Numeric) != 0;

        /// <summary>True when descriptors of this class can back an <see cref="NDArray"/> today (a storage <see cref="NPTypeCode"/> exists).</summary>
        public bool HasStorage => TypeCode != NPTypeCode.Empty;

        /// <summary>
        ///     The canonical native descriptor of a non-parametric class — NumPy's <c>DType.singleton</c>, the object
        ///     <c>np.dtype('i8')</c> returns every time (<c>np.dtype('i8') is np.dtype('i8')</c>). Null for abstract
        ///     classes; for the parametric datetime pair it is the generic-unit descriptor and <see cref="DefaultDescr"/>
        ///     hands out COPIES of it (a copy reports <c>isbuiltin == 0</c>, as in NumPy).
        /// </summary>
        public DType Singleton
        {
            get => _singleton;
            internal set => _singleton = value;
        }

        // ---------------------------------------------------------------------------------------------
        // Slots (NEP 42)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>default_descr</c>: the descriptor a bare class stands for — the singleton for the builtins, a fresh
        ///     generic-unit descriptor for datetime64 / timedelta64, the default (<c>int64</c>/<c>float64</c>/<c>complex128</c>)
        ///     for the abstract NEP 50 scalar classes.
        /// </summary>
        public abstract DType DefaultDescr();

        /// <summary>
        ///     <c>ensure_canonical</c>: the native-byte-order form of <paramref name="descr"/> (the descriptor itself when
        ///     already native).
        /// </summary>
        public virtual DType EnsureCanonical(DType descr)
        {
            if (descr == null)
                throw new ArgumentNullException(nameof(descr));
            if (descr.isnative)
                return descr;
            // A non-parametric class has exactly one canonical native instance: hand back the singleton itself
            // (NumPy's ensure_native_byteorder returns the builtin descriptor), not a fresh copy of it.
            if (!IsParametric && Singleton != null && ReferenceEquals(descr.Meta, this))
                return Singleton;
            return descr.WithByteOrder(DType.NativeByteOrder);
        }

        /// <summary>
        ///     <c>__common_dtype__</c>: the class that can hold values of both this class and <paramref name="other"/>, or
        ///     <see langword="null"/> (NumPy's <c>NotImplemented</c>) when this class does not know the other — the caller
        ///     then asks <paramref name="other"/> the mirrored question.
        /// </summary>
        public abstract DTypeMeta CommonDType(DTypeMeta other);

        /// <summary>
        ///     <c>__common_instance__</c>: for a PARAMETRIC class, the descriptor that can hold values of two descriptors
        ///     of this class (the unit GCD for datetimes; the longer length for strings). Non-parametric classes return
        ///     their default descriptor.
        /// </summary>
        public virtual DType CommonInstance(DType descr1, DType descr2) => DefaultDescr();

        /// <summary>
        ///     <c>discover_descr_from_pyobject</c>: the descriptor a scalar of this class should get (the default for
        ///     every non-parametric class).
        /// </summary>
        public virtual DType DiscoverDescrFromObject(object obj) => DefaultDescr();

        /// <summary><c>is_known_scalar_type</c>: whether a C# scalar of <paramref name="type"/> belongs to this class.</summary>
        public virtual bool IsKnownScalarType(Type type) => ScalarType != null && type == ScalarType;

        /// <summary>
        ///     The NumPy class call <c>Int8DType()</c>: the no-argument instantiation returns the singleton for a
        ///     non-parametric legacy class, and raises NumPy's <c>TypeError</c> for a parametric one
        ///     (<c>DateTime64DType()</c> → "Preliminary-API: Flexible/Parametric legacy DType … can only be instantiated
        ///     using `np.dtype(...)`"), verbatim.
        /// </summary>
        public virtual DType Instantiate()
        {
            if (IsAbstract)
                throw new TypeError($"Cannot instantiate abstract DType {this}");
            if (IsParametric)
                throw new TypeError($"Preliminary-API: Flexible/Parametric legacy DType '{this}' can only be instantiated using `np.dtype(...)`");
            return Singleton;
        }

        // ---------------------------------------------------------------------------------------------
        // NEP 55 reservations (reference-holding dtypes)
        // ---------------------------------------------------------------------------------------------

        /// <summary>Whether items of this dtype hold references that must be cleared (NumPy's <c>NPY_ITEM_REFCOUNT</c>); false for every builtin.</summary>
        public virtual bool HasReferences => false;

        /// <summary><c>get_clear_loop</c>: a loop that releases the references an item holds; null when <see cref="HasReferences"/> is false.</summary>
        public virtual NDInnerLoopFunc GetClearLoop(DType descr) => null;

        /// <summary><c>get_fill_zero_loop</c>: a loop that writes this dtype's zero; null when a zeroed buffer already is the zero.</summary>
        public virtual NDInnerLoopFunc GetFillZeroLoop(DType descr) => null;

        // ---------------------------------------------------------------------------------------------
        // Casting implementations (NEP 43)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The <see cref="CastingImpl"/> from this class to <paramref name="to"/> (the within-dtype implementation when
        ///     <paramref name="to"/> is this class), or null when no cast exists — NumPy's <c>PyArray_GetCastingImpl</c>.
        /// </summary>
        public CastingImpl GetCastingImpl(DTypeMeta to)
        {
            if (to == null)
                throw new ArgumentNullException(nameof(to));
            if (ReferenceEquals(to, this))
                return _withinDTypeCastingImpl;
            lock (_castingImpls)
                return _castingImpls.TryGetValue(to, out var impl) ? impl : null;
        }

        /// <summary>The casting implementation between two instances of this class (<c>within_dtype_castingimpl</c>).</summary>
        public CastingImpl WithinDTypeCastingImpl => _withinDTypeCastingImpl;

        /// <summary>Registers <paramref name="impl"/> (whose <see cref="CastingImpl.From"/> must be this class) — <c>PyArray_AddCastingImplementation</c>.</summary>
        internal void AddCastingImpl(CastingImpl impl)
        {
            if (impl == null)
                throw new ArgumentNullException(nameof(impl));
            if (!ReferenceEquals(impl.From, this))
                throw new ArgumentException($"casting implementation '{impl.Name}' casts from {impl.From}, not from {this}", nameof(impl));

            if (ReferenceEquals(impl.To, this))
            {
                if (_withinDTypeCastingImpl != null)
                    throw new InvalidOperationException($"A cast within {this} is already registered.");
                _withinDTypeCastingImpl = impl;
                return;
            }

            lock (_castingImpls)
            {
                if (_castingImpls.ContainsKey(impl.To))
                    throw new InvalidOperationException($"A cast from {this} to {impl.To} is already registered.");
                _castingImpls.Add(impl.To, impl);
            }
        }

        /// <summary>NumPy's rendering of the class object: <c>&lt;class 'numpy.dtypes.Int32DType'&gt;</c>.</summary>
        public override string ToString() => $"<class '{FullName}'>";
    }
}
