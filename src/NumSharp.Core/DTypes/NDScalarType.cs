using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading;

namespace NumSharp
{
    /// <summary>
    ///     A NumPy scalar TYPE: one class of NumPy's <c>numpy.generic</c> hierarchy as a C# object. The ten abstract
    ///     classes are <c>np.generic</c>, <c>np.number</c>, <c>np.integer</c>, <c>np.signedinteger</c>,
    ///     <c>np.unsignedinteger</c>, <c>np.inexact</c>, <c>np.floating</c>, <c>np.complexfloating</c>,
    ///     <c>np.flexible</c> and <c>np.character</c>; every registered dtype class has one concrete class under them
    ///     (<c>numpy.float64</c>, <c>numpy.int8</c>, <c>numpy.datetime64</c>, ...). They are what
    ///     <c>np.issubdtype(a.dtype, np.integer)</c>, <c>isinstance(x, np.floating)</c> and
    ///     <c>issubclass(t, np.number)</c> test against.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why a type of its own.</b> NumSharp spells a CONCRETE scalar class by its descriptor: <c>np.float64</c> is a
    ///     <see cref="DType"/>, because a C# <see cref="double"/> already is the scalar and every <c>dtype=</c> parameter
    ///     takes a <see cref="DType"/>. An ABSTRACT class has no descriptor (<c>np.dtype(np.integer)</c> raises in NumPy
    ///     2.x), so it cannot be a <see cref="DType"/>; it is an <see cref="NDScalarType"/>. The concrete classes exist as
    ///     <see cref="NDScalarType"/> objects too, so the hierarchy is complete: a descriptor converts IMPLICITLY to its
    ///     scalar class (NumPy's <c>dtype.type</c>: <c>NDScalarType t = np.float64;</c>), and a scalar class converts
    ///     EXPLICITLY back to its descriptor (NumPy's <c>np.dtype(t)</c>). The way back is explicit because it raises for
    ///     an abstract class, with NumPy's own text (<c>Converting `np.inexact` or `np.floating` to a dtype not
    ///     allowed</c>), so an abstract class passed where a dtype is expected is a compile error rather than a runtime
    ///     one — every such call raises in NumPy too (<c>np.zeros(3, np.integer)</c>, <c>np.can_cast(x, np.floating)</c>).
    ///     </para>
    ///     <para>
    ///     <b>Identity.</b> Each class is ONE object, compared by reference, as Python compares class objects: the
    ///     abstract classes are static singletons, and the concrete class of a dtype class is built once and cached on its
    ///     <see cref="DTypeMeta"/>, so <c>(NDScalarType)np.dtype("f8")</c> and <c>(NDScalarType)np.float64</c> are the same
    ///     object. There are no <c>==</c> operators: reference equality is Python's <c>is</c> and <c>==</c> for classes.
    ///     </para>
    ///     <para>
    ///     <b>Instances.</b> NumSharp has no scalar objects: a C# value of a dtype's element type (a boxed
    ///     <see cref="double"/>, <see cref="int"/>, <see cref="System.Half"/>, <see cref="System.Numerics.Complex"/>,
    ///     <see cref="decimal"/>, <see cref="char"/> or <see cref="bool"/>) IS a scalar of that dtype, and so is a 0-d
    ///     <see cref="NDArray"/> that stands for a scalar (every computed 0-d result, <c>a[0]</c> of a 1-D array,
    ///     <see cref="NDArray.Scalar{T}(T)"/>). An explicit <c>np.array(5)</c> is a 0-d ndarray, not a scalar, exactly as
    ///     in NumPy (<c>isinstance(np.array(5), np.integer)</c> is False). See <see cref="__instancecheck__"/>.
    ///     </para>
    ///     <para>
    ///     <b>NumSharp's classes against NumPy's.</b> NumSharp-only dtypes get NumSharp-only classes under the family their
    ///     kind names (<c>numsharp.decimal</c> under <c>floating</c>, <c>numsharp.char</c> under <c>unsignedinteger</c>),
    ///     the same placement <c>np.isdtype</c> has always used for them. NumPy classes NumSharp collapses onto one dtype
    ///     collapse here too: Windows NumPy's distinct <c>intc</c> / <c>long</c> classes are NumSharp's one <c>int32</c>
    ///     class, <c>longdouble</c> / <c>clongdouble</c> are <c>float64</c> / <c>complex128</c>, and <c>complex64</c>,
    ///     <c>object_</c> and <c>void</c> are absent until NumSharp has those dtypes. The fixed-width string classes are
    ///     <c>numpy.bytes_</c> (<c>S</c>) and <c>numpy.str_</c> (<c>U</c>); a <c>StringDType</c> descriptor's class is
    ///     Python's own <c>str</c>, outside <c>numpy.generic</c>.
    ///     </para>
    ///     <para>
    ///     <b>Not modelled.</b> Python's built-in bases are not NumSharp objects, so <see cref="__mro__"/> and
    ///     <see cref="__bases__"/> end at <c>np.generic</c>: NumPy's <c>numpy.float64.__mro__</c> also lists Python's
    ///     <c>float</c> and <c>object</c>, and <c>numpy.generic.__base__</c> is <c>object</c> where this class reports
    ///     null. The class call (<c>np.float64(1.5)</c>, which makes a NumPy scalar) and the unbound scalar methods that
    ///     NumPy's classes carry (<c>np.integer.is_integer</c>) have no counterpart.
    ///     </para>
    ///     <para><b>Thread safety.</b> Immutable after construction; the lazily built concrete classes are published with
    ///     a compare-exchange, so concurrent first uses agree on one object.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// np.issubdtype(a.dtype, np.integer)           // NumPy's most common type test
    /// np.issubdtype(np.float32, np.floating)       // True
    /// np.floating.__instancecheck__(a.sum())        // isinstance(a.sum(), np.floating)
    /// np.number.__subclasscheck__(a.dtype.type)     // issubclass(a.dtype.type, np.number)
    /// NDScalarType t = a.dtype;                     // a.dtype.type: the scalar class, &lt;class 'numpy.float64'&gt;
    /// DType d = (DType)t;                           // np.dtype(t): the descriptor again
    /// </code>
    /// </example>
    public sealed class NDScalarType
    {
        // ---------------------------------------------------------------------------------------------------------
        // The abstract classes, created in NumPy's registration order (multiarraymodule.c, setup_scalartypes):
        //   SINGLE_INHERIT(Number, Generic); SINGLE_INHERIT(Integer, Number); SINGLE_INHERIT(Inexact, Number);
        //   SINGLE_INHERIT(SignedInteger, Integer); SINGLE_INHERIT(UnsignedInteger, Integer);
        //   SINGLE_INHERIT(Floating, Inexact); SINGLE_INHERIT(ComplexFloating, Inexact);
        //   SINGLE_INHERIT(Flexible, Generic); SINGLE_INHERIT(Character, Flexible);
        // The rank is that order: __subclasses__() lists siblings by it. Each carries the text NumPy's
        // PyArray_DescrFromTypeObject raises when the class is converted to a dtype (scalarapi.c): several classes share
        // a text, and np.number and np.flexible report another class's name, both exactly as NumPy 2.4.2 does.
        // ---------------------------------------------------------------------------------------------------------

        /// <summary><c>numpy.generic</c>: the base of every NumPy scalar class.</summary>
        internal static readonly NDScalarType Generic =
            new NDScalarType("generic", "numpy", null, 0, "Converting `np.generic` to a dtype is not allowed.", null);

        /// <summary><c>numpy.number</c>: integers, floats and complex numbers (not bool).</summary>
        internal static readonly NDScalarType Number =
            new NDScalarType("number", "numpy", Generic, 1, "Converting `np.inexact` or `np.floating` to a dtype not allowed", null);

        /// <summary><c>numpy.integer</c>: the signed and unsigned integers (and timedelta64, a signed integer with a unit).</summary>
        internal static readonly NDScalarType Integer =
            new NDScalarType("integer", "numpy", Number, 2, "Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed", null);

        /// <summary><c>numpy.inexact</c>: the floating-point and complex classes.</summary>
        internal static readonly NDScalarType Inexact =
            new NDScalarType("inexact", "numpy", Number, 3, "Converting `np.inexact` or `np.floating` to a dtype not allowed", null);

        /// <summary><c>numpy.signedinteger</c>: int8 … int64 and timedelta64.</summary>
        internal static readonly NDScalarType SignedInteger =
            new NDScalarType("signedinteger", "numpy", Integer, 4, "Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed", null);

        /// <summary><c>numpy.unsignedinteger</c>: uint8 … uint64 (and NumSharp's char).</summary>
        internal static readonly NDScalarType UnsignedInteger =
            new NDScalarType("unsignedinteger", "numpy", Integer, 5, "Converting `np.unsignedinteger` to a dtype is not allowed", null);

        /// <summary><c>numpy.floating</c>: float16, float32, float64 (and NumSharp's decimal).</summary>
        internal static readonly NDScalarType Floating =
            new NDScalarType("floating", "numpy", Inexact, 6, "Converting `np.inexact` or `np.floating` to a dtype not allowed", null);

        /// <summary><c>numpy.complexfloating</c>: complex128.</summary>
        internal static readonly NDScalarType ComplexFloating =
            new NDScalarType("complexfloating", "numpy", Inexact, 7, "Converting `np.complex` to a dtype is not allowed.", null);

        /// <summary><c>numpy.flexible</c>: the classes whose instances carry a length (the string and void classes).</summary>
        internal static readonly NDScalarType Flexible =
            new NDScalarType("flexible", "numpy", Generic, 8, "Converting `np.generic` to a dtype is not allowed.", null);

        /// <summary><c>numpy.character</c>: the fixed-width string classes (<c>bytes_</c>, <c>str_</c>).</summary>
        internal static readonly NDScalarType Character =
            new NDScalarType("character", "numpy", Flexible, 9, "Converting `np.character` to a dtype is not allowed", null);

        /// <summary>
        ///     The abstract classes in NumPy's registration order — the order <see cref="__subclasses__"/> lists abstract
        ///     siblings in. Declared after every class it holds: static fields initialize in declaration order.
        /// </summary>
        private static readonly NDScalarType[] AbstractClasses =
            { Generic, Number, Integer, Inexact, SignedInteger, UnsignedInteger, Floating, ComplexFloating, Flexible, Character };

        /// <summary>The direct base, or null for <c>np.generic</c> (whose Python base, <c>object</c>, is not a NumPy class).</summary>
        private readonly NDScalarType _base;

        /// <summary>The dtype class of a concrete class; null for an abstract class, which has no descriptor.</summary>
        private readonly DTypeMeta _meta;

        /// <summary>
        ///     NumPy's <c>TypeError</c> text for converting this ABSTRACT class to a dtype; null for a concrete class, which
        ///     converts to its descriptor.
        /// </summary>
        private readonly string _conversionError;

        /// <summary>
        ///     The position of this class in NumPy's registration order: the order <see cref="__subclasses__"/> lists
        ///     siblings in, since Python lists a class's subclasses in the order they were created.
        /// </summary>
        private readonly int _rank;

        /// <summary>The method resolution order, this class first and <c>np.generic</c> last, handed out read-only.</summary>
        private readonly ReadOnlyCollection<NDScalarType> _mro;

        /// <summary>The direct bases (none for <c>np.generic</c>, one otherwise), handed out read-only.</summary>
        private readonly ReadOnlyCollection<NDScalarType> _bases;

        /// <summary>Builds one class. Called by the static initializers (abstract classes) and by <see cref="Of(DTypeMeta)"/> (concrete ones).</summary>
        /// <param name="name">Python's <c>__name__</c> (<c>"integer"</c>, <c>"float64"</c>).</param>
        /// <param name="module">Python's <c>__module__</c>: <c>"numpy"</c>, <c>"numsharp"</c> for a NumSharp-only class, <c>"builtins"</c> for Python's <c>str</c>.</param>
        /// <param name="base">The direct base; null for <c>np.generic</c> and for a class outside NumPy's hierarchy.</param>
        /// <param name="rank">The registration rank that orders siblings (see <see cref="_rank"/>).</param>
        /// <param name="conversionError">NumPy's conversion text for an abstract class; null for a concrete one.</param>
        /// <param name="meta">The dtype class of a concrete class; null for an abstract one.</param>
        private NDScalarType(string name, string module, NDScalarType @base, int rank, string conversionError, DTypeMeta meta)
        {
            __name__ = name;
            __module__ = module;
            _base = @base;
            _rank = rank;
            _conversionError = conversionError;
            _meta = meta;

            // The bases are built before their subclasses (static initializers run in declaration order, and a concrete
            // class is built after the abstract statics), so the base's MRO is complete here and this one only prepends.
            var mro = new List<NDScalarType>(@base is null ? 1 : @base._mro.Count + 1) { this };
            if (@base != null)
                mro.AddRange(@base._mro);
            _mro = mro.AsReadOnly();
            _bases = (@base is null ? new List<NDScalarType>() : new List<NDScalarType> { @base }).AsReadOnly();
        }

        /// <summary>Python's <c>__name__</c>: <c>"integer"</c>, <c>"float64"</c>, <c>"bool"</c>, <c>"datetime64"</c>.</summary>
        public string __name__ { get; }

        /// <summary>Python's <c>__qualname__</c>: the same as <see cref="__name__"/>, since NumPy's scalar classes are not nested.</summary>
        public string __qualname__ => __name__;

        /// <summary>
        ///     Python's <c>__module__</c>: <c>"numpy"</c>, <c>"numsharp"</c> for NumSharp's own classes (<c>decimal</c>,
        ///     <c>char</c>), or <c>"builtins"</c> for Python's <c>str</c>, the class of a <c>StringDType</c> descriptor.
        /// </summary>
        public string __module__ { get; }

        /// <summary>
        ///     Python's <c>__base__</c>: the direct base class (<c>np.number</c> for <c>np.integer</c>). Null for
        ///     <c>np.generic</c>, whose Python base is the built-in <c>object</c>, which NumSharp does not model.
        /// </summary>
        public NDScalarType __base__ => _base;

        /// <summary>
        ///     Python's <c>__bases__</c>: the direct bases, read-only. Empty for <c>np.generic</c>; one entry otherwise —
        ///     NumPy's second, built-in base of <c>float64</c> (<c>float</c>) and <c>complex128</c> (<c>complex</c>) is not
        ///     modelled.
        /// </summary>
        public IReadOnlyList<NDScalarType> __bases__ => _bases;

        /// <summary>
        ///     Python's <c>__mro__</c>: this class, then each base up to <c>np.generic</c>, read-only
        ///     (<c>[integer, number, generic]</c> for <c>np.integer</c>). NumPy's tuple also ends in Python's
        ///     <c>object</c> (and lists <c>float</c> / <c>complex</c> for float64 / complex128), which are not NumSharp objects.
        /// </summary>
        public IReadOnlyList<NDScalarType> __mro__ => _mro;

        /// <summary>
        ///     True for the ten abstract classes, which have no descriptor (converting one to a dtype raises NumPy's
        ///     <see cref="TypeError"/>) and no instances; false for the class of a dtype.
        /// </summary>
        public bool IsAbstract => _meta is null;

        /// <summary>Python's <c>type.mro()</c>: the method resolution order as a NEW, mutable list (Python returns a fresh list each call).</summary>
        /// <returns>A new list holding <see cref="__mro__"/>'s classes.</returns>
        public List<NDScalarType> mro() => new List<NDScalarType>(_mro);

        /// <summary>
        ///     Python's <c>type.__subclasses__()</c>: the DIRECT subclasses, as a new list, in NumPy's registration order —
        ///     abstract subclasses first, then the concrete ones (<c>np.signedinteger.__subclasses__()</c> is
        ///     <c>[int8, int16, int32, int64, timedelta64]</c>; NumPy's list also holds the classes NumSharp collapses or
        ///     lacks, such as <c>intc</c>). A concrete class has none.
        /// </summary>
        /// <returns>A new list of the direct subclasses.</returns>
        /// <remarks>
        ///     The list is computed once per registry state and copied out on each call: a dtype registered after this
        ///     class was first asked still shows up as its subclass (as a Python class created later does), because every
        ///     registration publishes a new registry array and the cached list remembers the array it was built from.
        /// </remarks>
        public List<NDScalarType> __subclasses__()
        {
            var registry = DTypeRegistry.AllSnapshot;
            var cached = Volatile.Read(ref _subclasses);
            if (cached is null || !ReferenceEquals(cached.Registry, registry))
            {
                cached = new SubclassList(registry, ComputeSubclasses(registry));
                // Two racing first calls compute equal lists from the same array; whichever write lands last is as good
                // as the other, so no compare-exchange is needed.
                Volatile.Write(ref _subclasses, cached);
            }

            return new List<NDScalarType>(cached.Subclasses);
        }

        /// <summary>
        ///     The last computed result of <see cref="__subclasses__"/>, with the registry array it was computed from; null
        ///     until the first call. Recomputing it on every call (a registry scan plus a sort) made the call cost about what
        ///     Python's own <c>type.__subclasses__()</c> does, where a copy of a cached array costs a fraction of it.
        /// </summary>
        private SubclassList _subclasses;

        /// <summary>Orders concrete siblings by their NumPy registration rank (every rank is distinct).</summary>
        private static readonly Comparer<NDScalarType> ByRank = Comparer<NDScalarType>.Create((a, b) => a._rank.CompareTo(b._rank));

        /// <summary>
        ///     The direct subclasses for one registry state (see <see cref="__subclasses__"/>): abstract children in their
        ///     registration order, then the concrete classes of the registered dtype classes whose base is this class,
        ///     ordered by NumPy's registration rank.
        /// </summary>
        /// <param name="registry">The registry's published array (read, never written).</param>
        /// <returns>The subclasses, in NumPy's order.</returns>
        private NDScalarType[] ComputeSubclasses(DTypeMeta[] registry)
        {
            var result = new List<NDScalarType>();

            // NumPy registers every abstract class before any concrete one, so a parent lists its abstract children
            // first, in their own registration order — which is AbstractClasses' order.
            foreach (var cls in AbstractClasses)
                if (ReferenceEquals(cls._base, this))
                    result.Add(cls);

            // Only an abstract class has concrete subclasses: no NumSharp dtype class derives from another one (NumPy's
            // only such pair, record under void, has no NumSharp counterpart).
            if (IsAbstract)
            {
                int firstConcrete = result.Count;
                foreach (var meta in registry)
                {
                    // An abstract dtype class (NEP 50's weak Python-scalar classes) has no instances and therefore no
                    // scalar class; Of() would refuse it.
                    if (meta._abstract)
                        continue;
                    var cls = Of(meta);
                    if (ReferenceEquals(cls._base, this))
                        result.Add(cls);
                }

                // Every rank is distinct, so the unstable sort is deterministic.
                result.Sort(firstConcrete, result.Count - firstConcrete, ByRank);
            }

            return result.ToArray();
        }

        /// <summary>
        ///     One computed <see cref="__subclasses__"/> result: the registry array it was computed from and the subclasses.
        ///     Immutable, so publishing it is a single reference write and a reader always sees a matching pair.
        /// </summary>
        private sealed class SubclassList
        {
            /// <summary>Pairs a registry state with the subclasses computed from it.</summary>
            /// <param name="registry">The registry array the list was computed from.</param>
            /// <param name="subclasses">The subclasses, in NumPy's order.</param>
            internal SubclassList(DTypeMeta[] registry, NDScalarType[] subclasses)
            {
                Registry = registry;
                Subclasses = subclasses;
            }

            /// <summary>The registry array the list was computed from (compared by reference).</summary>
            internal DTypeMeta[] Registry { get; }

            /// <summary>The subclasses, in NumPy's order; never handed out (callers get a copy).</summary>
            internal NDScalarType[] Subclasses { get; }
        }

        /// <summary>
        ///     Python's <c>isinstance(instance, cls)</c>: true when <paramref name="instance"/> is a NumPy scalar whose class
        ///     is this class or one of its subclasses (<c>np.integer.__instancecheck__(a[0])</c> for an int32 array is true).
        /// </summary>
        /// <param name="instance">The value to test; null is not an instance of anything (Python's <c>None</c>).</param>
        /// <returns>
        ///     For a boxed C# value: whether its type is the element type of a dtype whose class is under this one (a
        ///     <see cref="double"/> is a <c>numpy.float64</c>, a <see cref="decimal"/> a <c>numsharp.decimal</c>). For an
        ///     <see cref="NDArray"/>: true only when it is a 0-d array that stands for a scalar — a computed 0-d result,
        ///     <c>a[0]</c> of a 1-D array or <see cref="NDArray.Scalar{T}(T)"/> — and its dtype's class is under this one.
        ///     False for anything else: an array with dimensions, an explicit <c>np.array(x)</c> 0-d array, a descriptor or
        ///     a class (a class is not an instance of itself), a value of a type no dtype stores.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     NumSharp has no scalar objects, so a C# value of a dtype's element type IS the scalar. That is where this
        ///     differs from Python, whose built-in <c>1.0</c> is a <c>float</c> but not a <c>numpy.floating</c>: in C# a
        ///     <see cref="double"/> stands for both. A ported <c>isinstance(x, np.floating)</c> therefore answers for the
        ///     values NumSharp's arrays hand out, which is what such a test is written for (a JSON encoder turning NumPy
        ///     scalars into numbers, an index check accepting <c>(int, np.integer)</c>). The same rule makes a C#
        ///     <see cref="string"/> a <c>numpy.str_</c> scalar, because the string dtype stores <see cref="string"/>
        ///     (<c>np.character.__instancecheck__("abc")</c> is true, where Python's <c>str</c> is not an
        ///     <c>np.str_</c>).
        ///     </para>
        ///     <para>
        ///     The element type must match EXACTLY: a boxed enum is not an integer scalar even though its underlying type
        ///     is, because no dtype stores enums.
        ///     </para>
        /// </remarks>
        public bool __instancecheck__(object instance)
        {
            NDScalarType cls;
            switch (instance)
            {
                case null:
                    return false;
                case NDArray nd:
                    // NumPy's scalars come back from NumSharp as 0-d arrays; the explicit np.array(x) form is the one 0-d
                    // array that NumPy keeps as an ndarray (NDArray.IsExplicitZeroDimArray marks exactly that spelling).
                    if (nd.ndim != 0 || nd.IsExplicitZeroDimArray)
                        return false;
                    cls = Of(nd.dtype.Meta);
                    break;
                default:
                    cls = OfElementType(instance.GetType());
                    if (cls is null)
                        return false;
                    break;
            }

            return cls.IsSubclassOf(this);
        }

        /// <summary>
        ///     Python's <c>issubclass(subclass, cls)</c>: true when <paramref name="subclass"/> is this class or derives from
        ///     it. A <see cref="DType"/> converts implicitly to its scalar class, so a descriptor is accepted where NumPy
        ///     passes a class (<c>np.number.__subclasscheck__(np.float32)</c> is true).
        /// </summary>
        /// <param name="subclass">The class to test.</param>
        /// <returns>True when this class is in <paramref name="subclass"/>'s <see cref="__mro__"/>.</returns>
        /// <exception cref="TypeError"><paramref name="subclass"/> is null (Python's <c>issubclass() arg 1 must be a class</c>).</exception>
        public bool __subclasscheck__(NDScalarType subclass)
        {
            if (subclass is null)
                throw new TypeError("issubclass() arg 1 must be a class");
            return subclass.IsSubclassOf(this);
        }

        /// <summary>
        ///     Python's <c>issubclass(subclass, cls)</c> for a C# type — NumSharp's <see cref="DType.type"/> is the C#
        ///     element type, so a port of <c>issubclass(a.dtype.type, np.integer)</c> lands here:
        ///     <c>np.integer.__subclasscheck__(typeof(int))</c> is true.
        /// </summary>
        /// <param name="subclass">The C# type to test.</param>
        /// <returns>
        ///     True when <paramref name="subclass"/> is the element type of a dtype whose class derives from this one;
        ///     false for every other type (a C# type no dtype stores is a class, but not a NumPy scalar class).
        /// </returns>
        /// <exception cref="TypeError"><paramref name="subclass"/> is null (Python's <c>issubclass() arg 1 must be a class</c>).</exception>
        public bool __subclasscheck__(Type subclass)
        {
            if (subclass is null)
                throw new TypeError("issubclass() arg 1 must be a class");
            var cls = OfElementType(subclass);
            return cls != null && cls.IsSubclassOf(this);
        }

        /// <summary>
        ///     NumPy's rendering of a class, <c>repr(np.integer)</c> and <c>str(np.integer)</c> alike:
        ///     <c>&lt;class 'numpy.integer'&gt;</c>. Python leaves out the module of a built-in class, so the class of a
        ///     <c>StringDType</c> descriptor (Python's own <c>str</c>) renders as <c>&lt;class 'str'&gt;</c>.
        /// </summary>
        /// <returns>The class repr.</returns>
        public override string ToString()
            => __module__ == "builtins" ? $"<class '{__name__}'>" : $"<class '{__module__}.{__name__}'>";

        /// <summary>
        ///     The scalar class of a descriptor — NumPy's <c>dtype.type</c> (<c>np.dtype('f8').type is np.float64</c>). Every
        ///     descriptor has one, so the conversion never fails; descriptors of one dtype class (<c>'&lt;f8'</c>,
        ///     <c>'&gt;f8'</c>; <c>'M8[s]'</c>, <c>'M8[ns]'</c>) share it.
        /// </summary>
        /// <param name="dtype">The descriptor, or null.</param>
        /// <returns>Its scalar class, or null for a null descriptor (C#'s null maps through, as every DType conversion does).</returns>
        public static implicit operator NDScalarType(DType dtype) => dtype is null ? null : Of(dtype.Meta);

        /// <summary>
        ///     The descriptor of a scalar class — NumPy's <c>np.dtype(cls)</c>: a concrete class gives its dtype class's
        ///     builtin descriptor, the same object <c>np.dtype("f8")</c> returns.
        /// </summary>
        /// <param name="sctype">The scalar class, or null.</param>
        /// <returns>The descriptor, or null for a null class.</returns>
        /// <exception cref="TypeError">
        ///     <paramref name="sctype"/> is abstract: NumPy's texts, e.g. <c>Converting 'np.integer' or 'np.signedinteger' to
        ///     a dtype is not allowed</c> (the conversion is explicit for exactly this reason).
        /// </exception>
        public static explicit operator DType(NDScalarType sctype) => sctype?.ToDescriptor();

        /// <summary>
        ///     NumPy's <c>PyArray_DescrFromTypeObject</c>: the builtin descriptor of a concrete class (the shared singleton:
        ///     <c>np.dtype(np.float64) is np.dtype('f8')</c>, and <c>np.dtype(np.datetime64)</c> is the generic
        ///     <c>dtype('&lt;M8')</c> singleton), or the class's default descriptor when it has no shared one.
        /// </summary>
        /// <returns>The descriptor.</returns>
        /// <exception cref="TypeError">This class is abstract (NumPy's per-class text).</exception>
        internal DType ToDescriptor()
        {
            if (_meta is null)
                throw new TypeError(_conversionError);
            return _meta.singleton ?? _meta.default_descr();
        }

        /// <summary>
        ///     Python's subclass relation: true when <paramref name="other"/> is this class or one of its bases. A walk of
        ///     at most five links (<c>float64 → floating → inexact → number → generic</c>) comparing references.
        /// </summary>
        /// <param name="other">The candidate base; null is no class's base.</param>
        /// <returns>True when <paramref name="other"/> appears in this class's MRO.</returns>
        internal bool IsSubclassOf(NDScalarType other)
        {
            for (var cls = this; cls != null; cls = cls._base)
                if (ReferenceEquals(cls, other))
                    return true;
            // A StringDType descriptor's class is Python's own str, which NumPy's np.str_ (the U class) derives from as its
            // FIRST base (np.str_.__mro__ is (str_, str, character, flexible, generic, object)). The single-base chain above
            // cannot express that second line of descent, so it is checked here: issubdtype('U5', StringDType()) is True.
            return other?._meta is VStringDTypeMeta && _meta is StringDTypeMeta { IsUnicode: true };
        }

        /// <summary>
        ///     The scalar class of a dtype class, built on first use and cached on <paramref name="meta"/>: a concrete class
        ///     under the abstract class its kind names (NumPy's <c>SINGLE_INHERIT</c> table), named and ordered as NumPy names
        ///     and registers it.
        /// </summary>
        /// <param name="meta">A non-abstract dtype class.</param>
        /// <returns>The class — the same object on every call.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="meta"/> is null.</exception>
        /// <exception cref="InvalidOperationException">
        ///     <paramref name="meta"/> is abstract (the NEP 50 Python-scalar classes): it has no instances, so no descriptor
        ///     can reach it; asking for its class is a bug in the caller.
        /// </exception>
        internal static NDScalarType Of(DTypeMeta meta)
        {
            if (meta is null)
                throw new ArgumentNullException(nameof(meta));
            var known = Volatile.Read(ref meta._scalarType);
            if (known != null)
                return known;
            if (meta._abstract)
                throw new InvalidOperationException($"{meta} is an abstract DType class: it has no instances and no scalar class.");

            var built = Build(meta);
            // Two first uses may build the class at once; the compare-exchange keeps the first one published and every
            // caller returns that one, so the class stays a singleton (issubclass and isinstance compare references).
            return Interlocked.CompareExchange(ref meta._scalarType, built, null) ?? built;
        }

        /// <summary>
        ///     The scalar class of a C# element type: the class of the dtype class that stores values of exactly
        ///     <paramref name="type"/> (<see cref="double"/> → <c>numpy.float64</c>), or null when no dtype does.
        /// </summary>
        /// <param name="type">The C# type of a value or a <see cref="DType.type"/>.</param>
        /// <returns>The scalar class, or null for a type no dtype stores.</returns>
        /// <remarks>
        ///     <see cref="DTypeRegistry.FromScalarType"/> falls back on the type's <see cref="TypeCode"/>, which maps an
        ///     enum onto its underlying integer type; the exact-type check (<see cref="DTypeMeta.is_known_scalar_type"/>)
        ///     keeps an enum, which no dtype stores, out of the integer classes.
        /// </remarks>
        internal static NDScalarType OfElementType(Type type)
        {
            var meta = DTypeRegistry.FromScalarType(type);
            if (meta is null || meta._abstract || !meta.is_known_scalar_type(type))
                return null;
            return Of(meta);
        }

        /// <summary>Builds the concrete class of <paramref name="meta"/> (see <see cref="Of(DTypeMeta)"/>).</summary>
        /// <param name="meta">A non-abstract dtype class.</param>
        /// <returns>The new class.</returns>
        private static NDScalarType Build(DTypeMeta meta)
        {
            // NEP 55's StringDType stores Python's own str (np.dtypes.StringDType().type is str): its class is that built-in
            // class, outside NumPy's scalar hierarchy — no base, module "builtins".
            if (meta is VStringDTypeMeta)
                return new NDScalarType("str", "builtins", null, ConcreteRank(meta), conversionError: null, meta);

            // NumPy names the flexible and object classes by kind (one class per kind: numpy.bytes_, numpy.str_,
            // numpy.void, numpy.object_), whatever a NumSharp dtype class calls its scalar; the others by the scalar name
            // the dtype class records (numpy.float64, numsharp.decimal).
            string name = meta.Kind switch
            {
                'S' => "bytes_",
                'U' => "str_",
                'V' => "void",
                'O' => "object_",
                _ => meta.ScalarName,
            };
            string module = meta.ModulePath.StartsWith("numpy", StringComparison.Ordinal) ? "numpy" : "numsharp";
            return new NDScalarType(name, module, ParentOfKind(meta.Kind), ConcreteRank(meta), conversionError: null, meta);
        }

        /// <summary>
        ///     The abstract class a concrete class of <paramref name="kind"/> derives from — NumPy's <c>SINGLE_INHERIT</c> /
        ///     <c>DUAL_INHERIT</c> table: bool, datetime64 and object_ directly under <c>generic</c> ("datetime doesn't fit
        ///     in any category"), timedelta64 under <c>signedinteger</c> ("timedelta is an integer with an associated
        ///     unit"), the string classes under <c>character</c>, void under <c>flexible</c>.
        /// </summary>
        /// <param name="kind">NumPy's <c>dtype.kind</c> character of the dtype class.</param>
        /// <returns>
        ///     The base class, or null for a kind with no NumPy scalar class: NEP 55's <c>StringDType</c> (kind <c>'T'</c>)
        ///     stores Python's own <c>str</c>, so its class stands outside <c>numpy.generic</c>
        ///     (<c>np.issubdtype(StringDType(), np.generic)</c> is False).
        /// </returns>
        private static NDScalarType ParentOfKind(char kind) => kind switch
        {
            'b' or 'M' or 'O' => Generic,
            'i' or 'm' => SignedInteger,
            'u' => UnsignedInteger,
            'f' => Floating,
            'c' => ComplexFloating,
            'S' or 'U' => Character,
            'V' => Flexible,
            _ => null,
        };

        /// <summary>
        ///     The registration rank of a concrete class. NumPy creates its concrete classes in this order
        ///     (<c>setup_scalartypes</c>): Bool, Byte, Short, Int, Long, LongLong, Datetime, Timedelta, UByte, UShort, UInt,
        ///     ULong, ULongLong, Half, Float, Double, LongDouble, CFloat, CDouble, CLongDouble, String, Unicode, Void,
        ///     Object — so <c>np.floating.__subclasses__()</c> starts with float16 although its type number is the highest.
        /// </summary>
        /// <param name="meta">The dtype class.</param>
        /// <returns>
        ///     100 and up for a NumPy class, by its place in that list (identified by kind and item size, which is what
        ///     tells NumPy's C integer classes apart); 1000 plus the type number for a class NumPy does not have
        ///     (NumSharp's decimal 256 and char 257, a user dtype), so those follow NumPy's classes in registration order.
        /// </returns>
        private static int ConcreteRank(DTypeMeta meta)
        {
            if (meta.ModulePath == "numpy.dtypes")
            {
                int rank = (meta.Kind, meta.ItemSize) switch
                {
                    ('b', _) => 100,
                    ('i', 1) => 101,
                    ('i', 2) => 102,
                    ('i', 4) => 103,
                    ('i', 8) => 105,
                    ('M', _) => 106,
                    ('m', _) => 107,
                    ('u', 1) => 108,
                    ('u', 2) => 109,
                    ('u', 4) => 110,
                    ('u', 8) => 112,
                    ('f', 2) => 113,
                    ('f', 4) => 114,
                    ('f', 8) => 115,
                    ('f', _) => 116,
                    ('c', 8) => 117,
                    ('c', 16) => 118,
                    ('c', _) => 119,
                    ('S', _) => 120,
                    ('U', _) => 121,
                    ('V', _) => 122,
                    ('O', _) => 123,
                    _ => -1,
                };
                if (rank >= 0)
                    return rank;
            }

            return 1000 + meta.type_num;
        }
    }
}
