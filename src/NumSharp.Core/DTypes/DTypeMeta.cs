using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     The DType CLASS — NumPy's <c>PyArray_DTypeMeta</c> (<c>type(np.dtype('f8'))</c>, i.e.
    ///     <c>np.dtypes.Float64DType</c>). One live object per dtype class; a <see cref="DType"/> is an INSTANCE of it
    ///     (the descriptor). This is NEP 41/42's two-level model: behaviour lives on the class as virtual "slots"
    ///     (<see cref="common_dtype"/>, <see cref="common_instance"/>, <see cref="default_descr"/>,
    ///     <see cref="ensure_canonical"/>, <see cref="discover_descr_from_pyobject"/>, <see cref="is_known_scalar_type"/>,
    ///     the casting implementations), parameters live on the instance (a datetime unit, a byte order).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Names are NumPy's, case for case.</b> Every member NumPy names keeps NumPy's spelling: the Python class
    ///     attributes (<see cref="type"/>, <see cref="_abstract"/>, <see cref="_parametric"/>, <see cref="_legacy"/>,
    ///     <see cref="_is_numeric"/>), the C struct fields of <c>PyArray_DTypeMeta</c> / <c>NPY_DType_Slots</c>
    ///     (<see cref="singleton"/>, <see cref="type_num"/>, <see cref="flags"/>, <see cref="castingimpls"/>,
    ///     <see cref="within_dtype_castingimpl"/>) and the NEP 42 slots of <c>dtype_api.h</c> (<see cref="default_descr"/>,
    ///     <see cref="common_dtype"/>, …). C function names keep NumPy's PascalCase after the <c>PyArray_</c> prefix
    ///     (<see cref="GetCastingImpl"/> is <c>PyArray_GetCastingImpl</c>, <see cref="AddCastingImpl"/> is
    ///     <c>PyArray_AddCastingImplementation</c>). PascalCase members (<see cref="Name"/>, <see cref="TypeCode"/>,
    ///     <see cref="Kind"/>, …) are NumSharp's own and have no NumPy name. NumPy's slots <c>setitem</c>,
    ///     <c>getitem</c>, <c>finalize_descr</c> and <c>get_constant</c>, the legacy <c>PyArray_ArrFuncs</c> table and
    ///     the 2.4 <c>sort_meth</c> / <c>argsort_meth</c> are not ported: NumSharp's kernels select on
    ///     <see cref="NPTypeCode"/> instead.
    ///     </para>
    ///     <para>
    ///     <b>Compatibility contract.</b> <see cref="NPTypeCode"/> stays the storage / kernel discriminator: every meta
    ///     that has storage exposes it as <see cref="TypeCode"/>, and a future storage-backed dtype adds one enum
    ///     member and registers it here — the enum is an optimisation the registry hands out, not the identity. A meta
    ///     without storage (<see cref="DatetimeDTypeMeta"/> in Stage A) has <see cref="TypeCode"/> ==
    ///     <see cref="NPTypeCode.Empty"/>; allocating storage for its descriptors raises
    ///     <see cref="NotSupportedException"/> naming the missing piece.
    ///     </para>
    ///     <para>
    ///     <b>Type numbers</b> follow NumPy's <c>NPY_TYPES</c> on the host's C <c>long</c> model (<c>Half = 23</c>,
    ///     <c>DateTime = 21</c>, <c>TimeDelta = 22</c>; the 4- and 8-byte integers take the numbers NumPy gives them on
    ///     this platform, see <see cref="DTypeRegistry"/>); the two NumSharp-only dtypes take the user-defined range
    ///     (<c>Decimal = 256</c>, <c>Char = 257</c>), which is also what makes their descriptors report
    ///     <c>isbuiltin == 2</c>. The number ORDER is load-bearing: the builtin <c>common_dtype</c> defers to the operand
    ///     with the larger type number ("the more generic one handles it"), exactly as NumPy's
    ///     <c>default_builtin_common_dtype</c> does.
    ///     </para>
    ///     <para>
    ///     <b>NotImplemented</b> is spelled <see langword="null"/>: <see cref="common_dtype"/> returns null when this
    ///     class does not know the other one, and <see cref="DTypePromotion.CommonDType"/> then asks the other class
    ///     (NEP 42's two-sided <c>__common_dtype__</c> protocol) before raising <see cref="DTypePromotionError"/>.
    ///     </para>
    ///     <para>
    ///     <b>NEP 55 reservations.</b> <see cref="HasReferences"/>, <see cref="get_clear_loop"/> and
    ///     <see cref="get_fill_zero_loop"/> are the slots a reference-holding dtype (<c>StringDType</c>) will override;
    ///     the builtins report false / null.
    ///     </para>
    /// </remarks>
    public abstract class DTypeMeta
    {
        /// <summary>
        ///     The casts from this class to every OTHER class, keyed by the destination class — NumPy's per-DType
        ///     <c>castingimpls</c> dict. A <see cref="ConcurrentDictionary{TKey,TValue}"/> so <see cref="GetCastingImpl"/>
        ///     reads it without taking a lock (every <c>np.can_cast</c> over descriptors lands there, and so does promotion
        ///     whenever the common class is parametric), while a registration can still run at any time: the public
        ///     <see cref="DTypeRegistry.AddCastingImpl"/> admits a user cast after the registry is built.
        /// </summary>
        /// <remarks>
        ///     Keys compare by reference, because no <see cref="DTypeMeta"/> overrides <see cref="object.Equals(object)"/>;
        ///     that is NumPy's rule too, whose dict is keyed by the class object itself. The constructor arguments are
        ///     deliberate: <c>concurrencyLevel: 1</c> gives the dictionary ONE write lock, which is enough because writes
        ///     are rare (the registry's static constructor, the occasional user registration) and reads never take it;
        ///     <c>capacity: 31</c> holds every class's registrations (16 at most today) without a resize. The default
        ///     constructor would allocate one lock object per CPU for each class, and on a many-core host (one lock per
        ///     bucket or so) it would also grow its table and its lock array while the registry's static constructor
        ///     registers the casts.
        /// </remarks>
        private readonly ConcurrentDictionary<DTypeMeta, CastingImpl> _castingImpls =
            new ConcurrentDictionary<DTypeMeta, CastingImpl>(concurrencyLevel: 1, capacity: 31);

        /// <summary>
        ///     The read-only face of <see cref="_castingImpls"/> that <see cref="castingimpls"/> hands out — built once per
        ///     class so a reader allocates nothing, and a wrapper rather than the dictionary itself so a caller cannot
        ///     downcast it and register a cast past <see cref="AddCastingImpl"/>'s one-registration-per-pair rule.
        /// </summary>
        private readonly ReadOnlyDictionary<DTypeMeta, CastingImpl> _castingImplsView;

        /// <summary>
        ///     The cast between two instances of this class, held in its own slot as NumPy holds
        ///     <c>NPY_DT_SLOTS(DType)-&gt;within_dtype_castingimpl</c>. Written at most once, by the compare-exchange in
        ///     <see cref="AddCastingImpl"/>, and read through <see cref="Volatile.Read{T}"/>, so a reader on another
        ///     thread sees either null or the implementation, never a stale value cached across a registration.
        /// </summary>
        private CastingImpl _withinDTypeCastingImpl;

        /// <summary>Backing field of <see cref="singleton"/>; set once by the registry while it builds the class.</summary>
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
        /// <param name="flags">The <see cref="DTypeFlags"/> (<c>NPY_DT_LEGACY</c> / <c>_ABSTRACT</c> / <c>_PARAMETRIC</c> / <c>_NUMERIC</c>).</param>
        /// <param name="aliases">Alias class names NumPy also exports (<c>"IntDType"</c> for <c>Int32DType</c>).</param>
        /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="modulePath"/> is null.</exception>
        protected DTypeMeta(string name, string modulePath, int typeNum, Type scalarType, string scalarName,
            NPTypeCode typeCode, char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags,
            IReadOnlyList<string> aliases = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            ModulePath = modulePath ?? throw new ArgumentNullException(nameof(modulePath));
            type_num = typeNum;
            type = scalarType;
            ScalarName = scalarName ?? name;
            TypeCode = typeCode;
            Kind = kind;
            TypeChar = typeChar;
            ItemSize = itemSize;
            Alignment = alignment;
            this.flags = flags;
            Aliases = aliases ?? Array.Empty<string>();
            _castingImplsView = new ReadOnlyDictionary<DTypeMeta, CastingImpl>(_castingImpls);
        }

        /// <summary>NumPy's class name (<c>type.__name__</c>): <c>"Int32DType"</c>, <c>"DateTime64DType"</c>, <c>"_PyLongDType"</c>.</summary>
        public string Name { get; }

        /// <summary>The module path (<c>type.__module__</c>) used in <see cref="ToString"/>: <c>"numpy.dtypes"</c> or <c>"numsharp.dtypes"</c>.</summary>
        public string ModulePath { get; }

        /// <summary>The fully qualified class name: <c>"numpy.dtypes.Int32DType"</c>.</summary>
        public string FullName => ModulePath + "." + Name;

        /// <summary>
        ///     NumPy's <c>PyArray_DTypeMeta.type_num</c>: the <c>NPY_TYPES</c> number of the class (the host's C
        ///     <c>long</c> model for the 4- and 8-byte integers; 256+ for the NumSharp-only dtypes; -1 for an abstract
        ///     class). An <see cref="int"/> as in NumPy's struct; a descriptor's <see cref="DType.type_num"/> reports the
        ///     same number typed as <see cref="NPY_TYPES"/>.
        /// </summary>
        public int type_num { get; }

        /// <summary>
        ///     NumPy's class attribute <c>DType.type</c> (the C field <c>scalar_type</c>): the scalar type of one element,
        ///     which in NumSharp is the C# type (<c>typeof(double)</c> where NumPy has <c>numpy.float64</c>). Null when the
        ///     class has no C# scalar yet — the datetime pair before Stage C, the abstract classes.
        /// </summary>
        public Type type { get; }

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

        /// <summary>
        ///     NumPy's <c>PyArray_DTypeMeta.flags</c>: the <c>NPY_DT_*</c> word with NumPy's bit values. Python reads its
        ///     bits as <see cref="_legacy"/>, <see cref="_abstract"/>, <see cref="_parametric"/> and
        ///     <see cref="_is_numeric"/>.
        /// </summary>
        public DTypeFlags flags { get; }

        /// <summary>Alias class names NumPy also exports for this class (e.g. <c>ByteDType</c> for <c>Int8DType</c>).</summary>
        public IReadOnlyList<string> Aliases { get; }

        /// <summary>
        ///     NumPy's class attribute <c>DType._legacy</c> (<c>NPY_DT_LEGACY</c>): a dtype whose descriptor layout predates
        ///     NEP 41/42 — every builtin and the datetime pair. True means promotion and casting fall back to the legacy
        ///     tables for this class.
        /// </summary>
        public bool _legacy => (flags & DTypeFlags.NPY_DT_LEGACY) != 0;

        /// <summary>
        ///     NumPy's class attribute <c>DType._abstract</c> (<c>NPY_DT_ABSTRACT</c>): the class has no instances and only
        ///     takes part in promotion (the NEP 50 Python-scalar classes). Building a descriptor of it raises
        ///     <see cref="TypeError"/>.
        /// </summary>
        public bool _abstract => (flags & DTypeFlags.NPY_DT_ABSTRACT) != 0;

        /// <summary>
        ///     NumPy's class attribute <c>DType._parametric</c> (<c>NPY_DT_PARAMETRIC</c>): the instances carry parameters
        ///     (a unit, a length), so promotion must run <see cref="common_instance"/> and the bare class call raises.
        /// </summary>
        public bool _parametric => (flags & DTypeFlags.NPY_DT_PARAMETRIC) != 0;

        /// <summary>NumPy's class attribute <c>DType._is_numeric</c> (<c>NPY_DT_NUMERIC</c>): bool, integer, float or complex.</summary>
        public bool _is_numeric => (flags & DTypeFlags.NPY_DT_NUMERIC) != 0;

        /// <summary>True when descriptors of this class can back an <see cref="NDArray"/> today (a storage <see cref="NPTypeCode"/> exists).</summary>
        public bool HasStorage => TypeCode != NPTypeCode.Empty;

        /// <summary>
        ///     NumPy's <c>PyArray_DTypeMeta.singleton</c>: the canonical native descriptor of a non-parametric class, the
        ///     object <c>np.dtype('i8')</c> returns every time (<c>np.dtype('i8') is np.dtype('i8')</c>). Null for abstract
        ///     classes; for the parametric datetime pair it is the generic-unit descriptor and <see cref="default_descr"/>
        ///     hands out COPIES of it (a copy reports <c>isbuiltin == 0</c>, as in NumPy).
        /// </summary>
        public DType singleton
        {
            get => _singleton;
            internal set => _singleton = value;
        }

        // ---------------------------------------------------------------------------------------------
        // Slots (NEP 42) — NumPy's dtype_api.h slot names
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's slot <c>default_descr</c> (<c>NPY_DT_default_descr</c>): the descriptor a bare class stands for — the
        ///     singleton for the builtins, a fresh generic-unit descriptor for datetime64 / timedelta64, the default
        ///     (<c>int64</c>/<c>float64</c>/<c>complex128</c>) for the abstract NEP 50 scalar classes.
        /// </summary>
        /// <returns>The default descriptor; whether it is a shared singleton or a fresh copy is the class's choice, as in NumPy.</returns>
        public abstract DType default_descr();

        /// <summary>
        ///     NumPy's slot <c>ensure_canonical</c> (<c>ensure_native_byteorder</c> for every legacy class): the descriptor
        ///     itself when its byte order is native, otherwise a FRESH copy of it with byte order <c>'='</c>
        ///     (<c>PyArray_DescrNewByteorder(descr, NPY_NATIVE)</c>) — never the class singleton, so the result of
        ///     canonicalising a non-native descriptor reports <c>isbuiltin == 0</c> and keeps its metadata, as NumPy's does
        ///     (<c>np.result_type(np.dtype('&gt;f8')).isbuiltin</c> is 0).
        /// </summary>
        /// <param name="descr">A descriptor of THIS class.</param>
        /// <returns><paramref name="descr"/> when native (an explicit host-order byte order counts as native), else a native copy.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="descr"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="descr"/> is an instance of another class (NumPy only ever calls a class's slot on its own instances).</exception>
        public virtual DType ensure_canonical(DType descr)
        {
            if (descr == null)
                throw new ArgumentNullException(nameof(descr));
            if (!ReferenceEquals(descr.Meta, this))
                throw new ArgumentException($"{descr.ToString(true)} is not an instance of {this}", nameof(descr));
            if (descr.isnative)
                return descr;
            return descr.WithByteOrder(DType.NPY_NATIVE);
        }

        /// <summary>
        ///     NumPy's slot <c>common_dtype</c> (<c>__common_dtype__</c>): the class that can hold values of both this class
        ///     and <paramref name="other"/>, or <see langword="null"/> (NumPy's <c>NotImplemented</c>) when this class does
        ///     not know the other — the caller then asks <paramref name="other"/> the mirrored question.
        /// </summary>
        /// <param name="other">The other operand's class.</param>
        /// <returns>The common class, or null for NotImplemented.</returns>
        public abstract DTypeMeta common_dtype(DTypeMeta other);

        /// <summary>
        ///     NumPy's slot <c>common_instance</c> (<c>__common_instance__</c>): for a PARAMETRIC class, the descriptor that
        ///     can hold values of two descriptors of this class (the unit GCD for datetimes; the longer length for
        ///     strings). Non-parametric classes return their default descriptor (NumPy never calls the slot for them).
        /// </summary>
        /// <param name="descr1">The first descriptor, an instance of this class.</param>
        /// <param name="descr2">The second descriptor, an instance of this class.</param>
        /// <returns>The common descriptor.</returns>
        public virtual DType common_instance(DType descr1, DType descr2) => default_descr();

        /// <summary>
        ///     NumPy's slot <c>discover_descr_from_pyobject</c>: the descriptor a scalar of this class should get (the
        ///     default for every non-parametric class).
        /// </summary>
        /// <param name="obj">The scalar being discovered.</param>
        /// <returns>The descriptor an array built from <paramref name="obj"/> takes.</returns>
        public virtual DType discover_descr_from_pyobject(object obj) => default_descr();

        /// <summary>NumPy's slot <c>is_known_scalar_type</c>: whether a C# scalar of <paramref name="type"/> belongs to this class.</summary>
        /// <param name="type">The C# type of the scalar.</param>
        /// <returns>True when <paramref name="type"/> is this class's scalar type.</returns>
        public virtual bool is_known_scalar_type(Type type) => this.type != null && type == this.type;

        /// <summary>
        ///     The NumPy class call <c>Int8DType()</c>: the no-argument instantiation returns the singleton for a
        ///     non-parametric legacy class, and raises NumPy's <c>TypeError</c> for a parametric one
        ///     (<c>DateTime64DType()</c> → "Preliminary-API: Flexible/Parametric legacy DType … can only be instantiated
        ///     using `np.dtype(...)`"), verbatim. C# cannot call a class object, so the call is this method.
        /// </summary>
        /// <returns>The class's singleton descriptor.</returns>
        /// <exception cref="TypeError">The class is abstract, or parametric (NumPy's texts).</exception>
        public virtual DType Instantiate()
        {
            if (_abstract)
                throw new TypeError($"Cannot instantiate abstract DType {this}");
            if (_parametric)
                throw new TypeError($"Preliminary-API: Flexible/Parametric legacy DType '{this}' can only be instantiated using `np.dtype(...)`");
            return singleton;
        }

        // ---------------------------------------------------------------------------------------------
        // NEP 55 reservations (reference-holding dtypes)
        // ---------------------------------------------------------------------------------------------

        /// <summary>Whether items of this dtype hold references that must be cleared (NumPy's <c>NPY_ITEM_REFCOUNT</c>); false for every builtin.</summary>
        public virtual bool HasReferences => false;

        /// <summary>NumPy's slot <c>get_clear_loop</c>: a loop that releases the references an item holds.</summary>
        /// <param name="descr">The descriptor whose items are cleared.</param>
        /// <returns>The loop, or null when <see cref="HasReferences"/> is false (nothing to clear).</returns>
        public virtual NDInnerLoopFunc get_clear_loop(DType descr) => null;

        /// <summary>NumPy's slot <c>get_fill_zero_loop</c>: a loop that writes this dtype's zero.</summary>
        /// <param name="descr">The descriptor whose items are zero-filled.</param>
        /// <returns>The loop, or null when a zeroed buffer already is this dtype's zero (every builtin).</returns>
        public virtual NDInnerLoopFunc get_fill_zero_loop(DType descr) => null;

        // ---------------------------------------------------------------------------------------------
        // Casting implementations (NEP 43)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The <see cref="CastingImpl"/> from this class to <paramref name="to"/> (the within-dtype implementation when
        ///     <paramref name="to"/> is this class), or null when no cast exists — NumPy's <c>PyArray_GetCastingImpl</c>.
        ///     Every cast question of <see cref="DTypeCasting"/> starts here (each <c>np.can_cast</c> over descriptors and
        ///     each <see cref="DType"/> <c>&lt;</c> / <c>&lt;=</c> / <c>&gt;</c> / <c>&gt;=</c> comparison), and so does
        ///     <see cref="DTypePromotion.CastDescrToDType"/> whenever the common class is parametric (datetime64,
        ///     timedelta64).
        /// </summary>
        /// <param name="to">The destination class, compared by reference.</param>
        /// <returns>
        ///     The registered implementation, or null when none is registered for the pair — an impossible cast, which
        ///     <see cref="DTypeCasting.GetCastInfo"/> reports as null and <see cref="DTypeCasting.CheckCastSafety"/> as false.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="to"/> is null.</exception>
        /// <remarks>
        ///     Takes no lock: <see cref="ConcurrentDictionary{TKey,TValue}.TryGetValue"/> reads the dictionary without its
        ///     write lock and the within-dtype slot is a volatile read, so concurrent cast and promotion queries never
        ///     contend with each other or with a registration. A query racing the registration of the same pair returns
        ///     either null or the registered implementation.
        /// </remarks>
        public CastingImpl GetCastingImpl(DTypeMeta to)
        {
            if (to == null)
                throw new ArgumentNullException(nameof(to));
            if (ReferenceEquals(to, this))
                return Volatile.Read(ref _withinDTypeCastingImpl);
            return _castingImpls.TryGetValue(to, out var impl) ? impl : null;
        }

        /// <summary>
        ///     NumPy's slot field <c>within_dtype_castingimpl</c>: the casting implementation between two instances of this
        ///     class — the one <see cref="DTypeCasting"/> consults when both descriptors belong to this class (a byte-order
        ///     change, a datetime unit change, a plain copy).
        /// </summary>
        /// <value>The registered within-dtype cast, or null when the class registered none (the abstract NEP 50 scalar
        ///     classes and the vestigial string slot). Read without a lock, like <see cref="GetCastingImpl"/>.</value>
        public CastingImpl within_dtype_castingimpl => Volatile.Read(ref _withinDTypeCastingImpl);

        /// <summary>
        ///     NumPy's slot field <c>castingimpls</c>: the casts from this class to every OTHER class, keyed by the
        ///     destination class (the within-dtype cast is <see cref="within_dtype_castingimpl"/>, not an entry here — the
        ///     same split as NumPy's).
        /// </summary>
        /// <value>
        ///     A read-only live view: a cast registered later shows up in it, enumerating it never blocks a registration,
        ///     and it cannot be cast back to a writable dictionary. Use <see cref="GetCastingImpl"/> for a single lookup.
        /// </value>
        public IReadOnlyDictionary<DTypeMeta, CastingImpl> castingimpls => _castingImplsView;

        /// <summary>
        ///     Registers <paramref name="impl"/> (whose <see cref="CastingImpl.From"/> must be this class) — NumPy's
        ///     <c>PyArray_AddCastingImplementation</c>. Each class pair is registered once: a second registration for the
        ///     same destination raises instead of replacing the first, so a user cast can never silently displace a
        ///     builtin one.
        /// </summary>
        /// <param name="impl">The cast to register; its <see cref="CastingImpl.To"/> is the key it is stored under.</param>
        /// <exception cref="ArgumentNullException"><paramref name="impl"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="impl"/> casts from another class.</exception>
        /// <exception cref="InvalidOperationException">
        ///     A cast from this class to <see cref="CastingImpl.To"/> is already registered (for a within-dtype
        ///     <paramref name="impl"/>: a cast within this class).
        /// </exception>
        /// <remarks>
        ///     Takes no lock of its own. The within-dtype slot is claimed by one
        ///     <see cref="Interlocked.CompareExchange{T}(ref T, T, T)"/> and every other pair by
        ///     <see cref="ConcurrentDictionary{TKey,TValue}.TryAdd"/>, which serialises writers on the dictionary's
        ///     single write lock and never blocks a reader. Each check-and-store is one atomic step, so when several
        ///     threads register the same pair, exactly one succeeds and every other one raises.
        /// </remarks>
        internal void AddCastingImpl(CastingImpl impl)
        {
            if (impl == null)
                throw new ArgumentNullException(nameof(impl));
            if (!ReferenceEquals(impl.From, this))
                throw new ArgumentException($"casting implementation '{impl.name}' casts from {impl.From}, not from {this}", nameof(impl));

            if (ReferenceEquals(impl.To, this))
            {
                // A null check followed by a plain store would let two racing registrations both pass the check, and the
                // later one would silently replace the earlier instead of raising. The compare-exchange checks and stores
                // in one step, and it publishes the implementation fully built to the volatile readers.
                if (Interlocked.CompareExchange(ref _withinDTypeCastingImpl, impl, null) != null)
                    throw new InvalidOperationException($"A cast within {this} is already registered.");
                return;
            }

            // TryAdd is ContainsKey + Add as one atomic step: false means another registration of this pair got there first.
            if (!_castingImpls.TryAdd(impl.To, impl))
                throw new InvalidOperationException($"A cast from {this} to {impl.To} is already registered.");
        }

        /// <summary>NumPy's rendering of the class object (<c>repr(np.dtypes.Int32DType)</c>): <c>&lt;class 'numpy.dtypes.Int32DType'&gt;</c>.</summary>
        /// <returns>The class repr.</returns>
        public override string ToString() => $"<class '{FullName}'>";
    }
}
