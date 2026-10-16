using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Numerics;
using System.Threading;

namespace NumSharp
{
    /// <summary>
    ///     A NumPy scalar TYPE: one class of NumPy's <c>numpy.generic</c> hierarchy, or one of the Python built-in classes
    ///     that hierarchy derives from, as a C# object. The ten abstract classes are <c>np.generic</c>, <c>np.number</c>,
    ///     <c>np.integer</c>, <c>np.signedinteger</c>, <c>np.unsignedinteger</c>, <c>np.inexact</c>, <c>np.floating</c>,
    ///     <c>np.complexfloating</c>, <c>np.flexible</c> and <c>np.character</c>; every registered dtype class has one
    ///     concrete class under them (<c>numpy.float64</c>, <c>numpy.int8</c>, <c>numpy.datetime64</c>, ...), and so do
    ///     the NumPy classes NumSharp has no dtype for (<c>numpy.complex64</c>, <c>numpy.object_</c>, <c>numpy.void</c>).
    ///     They are what <c>np.issubdtype(a.dtype, np.integer)</c>, <c>isinstance(x, np.floating)</c> and
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
    ///     <b>Identity.</b> Each class is ONE object, compared by reference, as Python compares class objects: the abstract,
    ///     dtype-less and built-in classes are static singletons, and the concrete class of a dtype class is built once and
    ///     cached on its <see cref="DTypeMeta"/>, so <c>(NDScalarType)np.dtype("f8")</c> and <c>(NDScalarType)np.float64</c>
    ///     are the same object. There are no <c>==</c> operators: reference equality is Python's <c>is</c> and <c>==</c>
    ///     for classes.
    ///     </para>
    ///     <para>
    ///     <b>Python's built-in classes.</b> NumPy's classes derive from Python's: <c>numpy.generic</c> from <c>object</c>,
    ///     <c>numpy.float64</c> also from <c>float</c>, <c>numpy.complex128</c> from <c>complex</c>, <c>numpy.bytes_</c> from
    ///     <c>bytes</c> and <c>numpy.str_</c> from <c>str</c>; <c>np.ScalarType</c> lists <c>int</c>, <c>float</c>,
    ///     <c>complex</c>, <c>bool</c>, <c>bytes</c>, <c>str</c> and <c>memoryview</c>; and a <c>StringDType</c> descriptor's
    ///     class IS Python's <c>str</c>. Those eight classes are modelled (<see cref="PyObject"/>, <see cref="PyInt"/>,
    ///     <see cref="PyBool"/>, <see cref="PyFloat"/>, <see cref="PyComplex"/>, <see cref="PyStr"/>, <see cref="PyBytes"/>,
    ///     <see cref="PyMemoryView"/>), with Python's own bases (<c>bool</c> derives from <c>int</c>), so every
    ///     <see cref="__mro__"/> is NumPy's whole tuple, computed by Python's C3 linearisation.
    ///     </para>
    ///     <para>
    ///     <b>Instances.</b> NumSharp has no scalar objects: a C# value of a dtype's element type (a boxed
    ///     <see cref="double"/>, <see cref="int"/>, <see cref="System.Half"/>, <see cref="System.Numerics.Complex"/>,
    ///     <see cref="decimal"/>, <see cref="char"/>, <see cref="bool"/> or <see cref="string"/>) IS a scalar of that dtype,
    ///     and so is a 0-d <see cref="NDArray"/> that stands for a scalar (every computed 0-d result, <c>a[0]</c> of a 1-D
    ///     array, <see cref="NDArray.Scalar{T}(T)"/>). A C# value is also the Python object it stands for in NumSharp's
    ///     NEP 50 rules (a C# <see cref="int"/> is a Python <c>int</c>, a <see cref="double"/> a Python <c>float</c>). An
    ///     explicit <c>np.array(5)</c> is a 0-d ndarray, not a scalar, exactly as in NumPy
    ///     (<c>isinstance(np.array(5), np.integer)</c> is False). See <see cref="__instancecheck__"/>.
    ///     </para>
    ///     <para>
    ///     <b>NumSharp's classes against NumPy's.</b> NumSharp-only dtypes get NumSharp-only classes under the family their
    ///     kind names (<c>numsharp.decimal</c> under <c>floating</c>, <c>numsharp.char</c> under <c>unsignedinteger</c>),
    ///     the same placement <c>np.isdtype</c> has always used for them. The NumPy classes NumSharp has no dtype for
    ///     (<c>complex64</c>, <c>object_</c>, <c>void</c>) exist as classes and refuse the conversion to a dtype with a
    ///     <see cref="NotSupportedException"/>; a dtype class registered later under their NumPy type number adopts them.
    ///     </para>
    ///     <para>
    ///     <b>Not modelled.</b> The class call (<c>np.float64(1.5)</c>, which makes a NumPy scalar), the unbound scalar
    ///     methods NumPy's classes carry (<c>np.integer.is_integer</c>), NumPy's <c>numpy.record</c> (the class of a
    ///     structured record, NumPy's one subclass of <c>void</c>), and Python's other classes: a built-in class's
    ///     <see cref="__subclasses__"/> lists only the modelled classes (Python's <c>str.__subclasses__()</c> also holds
    ///     <c>enum.StrEnum</c>, and <c>object.__subclasses__()</c> some 360 classes).
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
    /// t.__mro__                                     // [float64, floating, inexact, number, generic, float, object]
    /// </code>
    /// </example>
    public sealed class NDScalarType
    {
        /// <summary>What a class is, which decides its conversion to a dtype and its membership in NumPy's lists.</summary>
        private enum ClassKind
        {
            /// <summary>One of Python's built-in classes (<c>object</c>, <c>int</c>, <c>float</c>, ...).</summary>
            Builtin,

            /// <summary>One of NumPy's ten abstract classes: no descriptor, no instances.</summary>
            Abstract,

            /// <summary>A concrete NumPy (or NumSharp) class: the class of a dtype class, or a NumPy class NumSharp has no dtype for.</summary>
            Concrete,
        }

        // ---------------------------------------------------------------------------------------------------------
        // Python's built-in classes, created first (everything derives from object). Their ranks come before NumPy's so a
        // built-in class's subclasses list them first, as Python created them first.
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>Python's <c>object</c>: the base of every class, NumPy's <c>numpy.generic</c> included.</summary>
        internal static readonly NDScalarType PyObjectClass = Builtin("object", null, 0);

        /// <summary>Python's <c>int</c>: the class of a C# integer in NumSharp's NEP 50 rules; no NumPy integer derives from it.</summary>
        internal static readonly NDScalarType PyIntClass = Builtin("int", PyObjectClass, 1);

        /// <summary>Python's <c>bool</c>, which derives from <c>int</c> (<c>numpy.bool</c> derives from neither).</summary>
        internal static readonly NDScalarType PyBoolClass = Builtin("bool", PyIntClass, 2);

        /// <summary>Python's <c>float</c>, the second base of <c>numpy.float64</c>.</summary>
        internal static readonly NDScalarType PyFloatClass = Builtin("float", PyObjectClass, 3);

        /// <summary>Python's <c>complex</c>, the second base of <c>numpy.complex128</c>.</summary>
        internal static readonly NDScalarType PyComplexClass = Builtin("complex", PyObjectClass, 4);

        /// <summary>Python's <c>str</c>: the first base of <c>numpy.str_</c>, and the class of a <c>StringDType</c> descriptor.</summary>
        internal static readonly NDScalarType PyStrClass = Builtin("str", PyObjectClass, 5);

        /// <summary>Python's <c>bytes</c>, the first base of <c>numpy.bytes_</c>.</summary>
        internal static readonly NDScalarType PyBytesClass = Builtin("bytes", PyObjectClass, 6);

        /// <summary>Python's <c>memoryview</c>, which <c>np.ScalarType</c> lists (<c>np.dtype(memoryview)</c> is <c>dtype('V')</c>).</summary>
        internal static readonly NDScalarType PyMemoryViewClass = Builtin("memoryview", PyObjectClass, 7);

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

        /// <summary><c>numpy.generic</c>: the base of every NumPy scalar class; its own base is Python's <c>object</c>.</summary>
        internal static readonly NDScalarType Generic =
            Abstract("generic", PyObjectClass, 10, "Converting `np.generic` to a dtype is not allowed.");

        /// <summary><c>numpy.number</c>: integers, floats and complex numbers (not bool).</summary>
        internal static readonly NDScalarType Number =
            Abstract("number", Generic, 11, "Converting `np.inexact` or `np.floating` to a dtype not allowed");

        /// <summary><c>numpy.integer</c>: the signed and unsigned integers (and timedelta64, a signed integer with a unit).</summary>
        internal static readonly NDScalarType Integer =
            Abstract("integer", Number, 12, "Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed");

        /// <summary><c>numpy.inexact</c>: the floating-point and complex classes.</summary>
        internal static readonly NDScalarType Inexact =
            Abstract("inexact", Number, 13, "Converting `np.inexact` or `np.floating` to a dtype not allowed");

        /// <summary><c>numpy.signedinteger</c>: int8 … int64 and timedelta64.</summary>
        internal static readonly NDScalarType SignedInteger =
            Abstract("signedinteger", Integer, 14, "Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed");

        /// <summary><c>numpy.unsignedinteger</c>: uint8 … uint64 (and NumSharp's char).</summary>
        internal static readonly NDScalarType UnsignedInteger =
            Abstract("unsignedinteger", Integer, 15, "Converting `np.unsignedinteger` to a dtype is not allowed");

        /// <summary><c>numpy.floating</c>: float16, float32, float64 (and NumSharp's decimal).</summary>
        internal static readonly NDScalarType Floating =
            Abstract("floating", Inexact, 16, "Converting `np.inexact` or `np.floating` to a dtype not allowed");

        /// <summary><c>numpy.complexfloating</c>: complex64 and complex128.</summary>
        internal static readonly NDScalarType ComplexFloating =
            Abstract("complexfloating", Inexact, 17, "Converting `np.complex` to a dtype is not allowed.");

        /// <summary><c>numpy.flexible</c>: the classes whose instances carry a length (the string and void classes).</summary>
        internal static readonly NDScalarType Flexible =
            Abstract("flexible", Generic, 18, "Converting `np.generic` to a dtype is not allowed.");

        /// <summary><c>numpy.character</c>: the fixed-width string classes (<c>bytes_</c>, <c>str_</c>).</summary>
        internal static readonly NDScalarType Character =
            Abstract("character", Flexible, 19, "Converting `np.character` to a dtype is not allowed");

        // ---------------------------------------------------------------------------------------------------------
        // The NumPy classes NumSharp has no dtype for. NumPy has them, so the hierarchy, np.ScalarType and every type test
        // see them; converting one to a dtype refuses (NumSharp cannot store such an array). A dtype class registered later
        // under the NumPy type number adopts the class (Of), so the object a program captured stays the class.
        // ---------------------------------------------------------------------------------------------------------

        /// <summary><c>numpy.complex64</c> (two float32 parts, type number 14): NumSharp's one complex width is complex128.</summary>
        internal static readonly NDScalarType Complex64 = Dtypeless("complex64", ComplexFloating, typeNum: 14, kind: 'c', itemSize: 8,
            "NumSharp has no complex64 dtype (two float32 parts): its one complex width is complex128 (np.complex128).");

        /// <summary><c>numpy.object_</c> (type number 17): NumSharp has no object arrays.</summary>
        internal static readonly NDScalarType Object_ = Dtypeless("object_", Generic, typeNum: 17, kind: 'O', itemSize: 8,
            "NumSharp has no object dtype (np.dtype(object) is dtype('O') in NumPy): arrays of Python objects cannot be stored.");

        /// <summary><c>numpy.void</c> (type number 20): NumSharp has no void or structured arrays.</summary>
        internal static readonly NDScalarType Void = Dtypeless("void", Flexible, typeNum: 20, kind: 'V', itemSize: 0,
            "NumSharp has no void dtype (np.dtype(np.void) is dtype('V') in NumPy): raw-byte and structured arrays cannot be stored.");

        /// <summary>
        ///     Every statically created class (built-in, abstract, dtype-less), in creation order — the candidates
        ///     <see cref="__subclasses__"/> scans besides the classes of the registered dtype classes. Declared after every
        ///     class it holds: static fields initialize in declaration order.
        /// </summary>
        private static readonly NDScalarType[] StaticClasses =
        {
            PyObjectClass, PyIntClass, PyBoolClass, PyFloatClass, PyComplexClass, PyStrClass, PyBytesClass, PyMemoryViewClass,
            Generic, Number, Integer, Inexact, SignedInteger, UnsignedInteger, Floating, ComplexFloating, Flexible, Character,
            Complex64, Object_, Void,
        };

        /// <summary>The NumPy classes NumSharp has no dtype for, by their NumPy type number (the adoption table of <see cref="Of"/>).</summary>
        private static readonly NDScalarType[] DtypelessClasses = { Complex64, Object_, Void };

        /// <summary>What this class is (see <see cref="ClassKind"/>).</summary>
        private readonly ClassKind _kind;

        /// <summary>The direct bases in Python's order (none for <c>object</c>); read-only.</summary>
        private readonly ReadOnlyCollection<NDScalarType> _bases;

        /// <summary>The method resolution order (Python's C3 linearisation), this class first and <c>object</c> last; read-only.</summary>
        private readonly ReadOnlyCollection<NDScalarType> _mro;

        /// <summary>
        ///     The array <see cref="_mro"/> wraps, scanned by <see cref="IsSubclassOf"/> with reference comparisons: every
        ///     type test ends in that scan, and the collection's <c>Contains</c> goes through the default equality comparer
        ///     for each entry. Never written after construction and never handed out.
        /// </summary>
        private readonly NDScalarType[] _mroItems;

        /// <summary>
        ///     The dtype class of a concrete class. Null for an abstract or built-in class, and for a dtype-less NumPy class
        ///     until a dtype class registered under its type number adopts it (<see cref="Of"/> sets it once, with a
        ///     compare-exchange).
        /// </summary>
        private DTypeMeta _meta;

        /// <summary>
        ///     The text of the conversion refusal: NumPy's <c>TypeError</c> text for an abstract class, the
        ///     <see cref="NotSupportedException"/> text for a dtype-less class; null for every other class.
        /// </summary>
        private readonly string _conversionError;

        /// <summary>
        ///     The position of this class in creation order, which orders <see cref="__subclasses__"/> (Python lists a
        ///     class's subclasses in the order they were created): the built-in classes 0-7, the abstract classes 10-19,
        ///     NumPy's concrete classes 100-123 by NumPy's C-type registration order, NumSharp-only classes from 1000.
        /// </summary>
        private readonly int _rank;

        /// <summary>The NumPy type number of a dtype-less class (14, 17, 20); -1 for every other class.</summary>
        private readonly int _dtypelessTypeNum;

        /// <summary>The NumPy <c>dtype.kind</c> of a concrete class (<c>'f'</c> for float64), used by <c>np.ScalarType</c>'s sort and <c>np.iinfo</c>'s text; <c>'\0'</c> otherwise.</summary>
        private readonly char _numpyKind;

        /// <summary>The item size NumPy's descriptor of a concrete class has (8 for float64), used by <c>np.ScalarType</c>'s sort; 0 otherwise.</summary>
        private readonly int _numpyItemSize;

        /// <summary>Builds one class. Called by the static initializers and by <see cref="Of(DTypeMeta)"/> (concrete classes).</summary>
        /// <param name="name">Python's <c>__name__</c> (<c>"integer"</c>, <c>"float64"</c>, <c>"float"</c>).</param>
        /// <param name="module">Python's <c>__module__</c>: <c>"numpy"</c>, <c>"numsharp"</c> for a NumSharp-only class, <c>"builtins"</c> for Python's own.</param>
        /// <param name="kind">What the class is.</param>
        /// <param name="bases">The direct bases in Python's order (each already built, so its MRO is complete).</param>
        /// <param name="rank">The creation rank that orders siblings (see <see cref="_rank"/>).</param>
        /// <param name="conversionError">The conversion refusal text (abstract and dtype-less classes); null otherwise.</param>
        /// <param name="meta">The dtype class of a concrete class; null otherwise.</param>
        /// <param name="dtypelessTypeNum">The NumPy type number of a dtype-less class; -1 otherwise.</param>
        /// <param name="numpyKind">NumPy's <c>dtype.kind</c> of a concrete class; <c>'\0'</c> otherwise.</param>
        /// <param name="numpyItemSize">NumPy's item size of a concrete class; 0 otherwise.</param>
        /// <exception cref="InvalidOperationException">The bases admit no consistent method resolution order (a bug in the tables above).</exception>
        private NDScalarType(string name, string module, ClassKind kind, NDScalarType[] bases, int rank, string conversionError,
            DTypeMeta meta, int dtypelessTypeNum, char numpyKind, int numpyItemSize)
        {
            __name__ = name;
            __module__ = module;
            _kind = kind;
            _rank = rank;
            _conversionError = conversionError;
            _meta = meta;
            _dtypelessTypeNum = dtypelessTypeNum;
            _numpyKind = numpyKind;
            _numpyItemSize = numpyItemSize;
            _bases = Array.AsReadOnly(bases);
            _mroItems = Linearize(this, bases);
            _mro = Array.AsReadOnly(_mroItems);
        }

        /// <summary>Builds one of Python's built-in classes.</summary>
        /// <param name="name">Python's name (<c>"float"</c>).</param>
        /// <param name="base">The single base; null for <c>object</c>.</param>
        /// <param name="rank">The creation rank.</param>
        /// <returns>The class.</returns>
        private static NDScalarType Builtin(string name, NDScalarType @base, int rank)
            => new NDScalarType(name, "builtins", ClassKind.Builtin, @base is null ? Array.Empty<NDScalarType>() : new[] { @base },
                rank, conversionError: null, meta: null, dtypelessTypeNum: -1, numpyKind: '\0', numpyItemSize: 0);

        /// <summary>Builds one of NumPy's abstract classes.</summary>
        /// <param name="name">NumPy's name (<c>"integer"</c>).</param>
        /// <param name="base">The single base.</param>
        /// <param name="rank">The creation rank.</param>
        /// <param name="conversionError">NumPy's <c>TypeError</c> text for <c>np.dtype(cls)</c>.</param>
        /// <returns>The class.</returns>
        private static NDScalarType Abstract(string name, NDScalarType @base, int rank, string conversionError)
            => new NDScalarType(name, "numpy", ClassKind.Abstract, new[] { @base }, rank, conversionError, meta: null,
                dtypelessTypeNum: -1, numpyKind: '\0', numpyItemSize: 0);

        /// <summary>Builds a NumPy concrete class NumSharp has no dtype for.</summary>
        /// <param name="name">NumPy's name (<c>"complex64"</c>).</param>
        /// <param name="base">The single (abstract) base.</param>
        /// <param name="typeNum">NumPy's type number, which ranks the class and lets a later dtype class adopt it.</param>
        /// <param name="kind">NumPy's <c>dtype.kind</c>.</param>
        /// <param name="itemSize">NumPy's item size of the class's descriptor.</param>
        /// <param name="conversionError">The <see cref="NotSupportedException"/> text for <c>np.dtype(cls)</c>.</param>
        /// <returns>The class.</returns>
        private static NDScalarType Dtypeless(string name, NDScalarType @base, int typeNum, char kind, int itemSize, string conversionError)
            => new NDScalarType(name, "numpy", ClassKind.Concrete, new[] { @base }, RankOfTypeNum(typeNum), conversionError,
                meta: null, dtypelessTypeNum: typeNum, numpyKind: kind, numpyItemSize: itemSize);

        // ---------------------------------------------------------------------------------------------------------
        // Python's built-in classes, as NumSharp hands them out
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>Python's <c>object</c>: <c>numpy.generic</c>'s base, the end of every <see cref="__mro__"/>; every non-null value is an instance.</summary>
        public static NDScalarType PyObject => PyObjectClass;

        /// <summary>Python's <c>int</c>: a C# integer value is an instance (<c>isinstance(5, int)</c>); no NumPy integer class derives from it.</summary>
        public static NDScalarType PyInt => PyIntClass;

        /// <summary>Python's <c>bool</c> (a subclass of <c>int</c>): a C# <see cref="bool"/> is an instance; <c>numpy.bool</c> is not a subclass.</summary>
        public static NDScalarType PyBool => PyBoolClass;

        /// <summary>Python's <c>float</c>: a C# <see cref="double"/> / <see cref="float"/> is an instance, and <c>numpy.float64</c> a subclass.</summary>
        public static NDScalarType PyFloat => PyFloatClass;

        /// <summary>Python's <c>complex</c>: a C# <see cref="Complex"/> is an instance, and <c>numpy.complex128</c> a subclass.</summary>
        public static NDScalarType PyComplex => PyComplexClass;

        /// <summary>Python's <c>str</c>: a C# <see cref="string"/> is an instance, <c>numpy.str_</c> a subclass, and a <c>StringDType</c> descriptor's class this class itself.</summary>
        public static NDScalarType PyStr => PyStrClass;

        /// <summary>Python's <c>bytes</c>: a C# <c>byte[]</c> is an instance (NumSharp's text APIs read it as <c>bytes</c>), and <c>numpy.bytes_</c> a subclass.</summary>
        public static NDScalarType PyBytes => PyBytesClass;

        /// <summary>Python's <c>memoryview</c>: an <see cref="np.MemoryView"/> (<c>ndarray.data</c>) is an instance.</summary>
        public static NDScalarType PyMemoryView => PyMemoryViewClass;

        // ---------------------------------------------------------------------------------------------------------
        // The class surface
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>Python's <c>__name__</c>: <c>"integer"</c>, <c>"float64"</c>, <c>"bool"</c>, <c>"datetime64"</c>, <c>"float"</c>.</summary>
        public string __name__ { get; }

        /// <summary>Python's <c>__qualname__</c>: the same as <see cref="__name__"/>, since none of these classes is nested.</summary>
        public string __qualname__ => __name__;

        /// <summary>
        ///     Python's <c>__module__</c>: <c>"numpy"</c>, <c>"numsharp"</c> for NumSharp's own classes (<c>decimal</c>,
        ///     <c>char</c>), or <c>"builtins"</c> for Python's classes (the class of a <c>StringDType</c> descriptor included).
        /// </summary>
        public string __module__ { get; }

        /// <summary>
        ///     Python's <c>__base__</c>: the first direct base (<c>np.number</c> for <c>np.integer</c>, <c>numpy.floating</c>
        ///     for <c>numpy.float64</c>, Python's <c>bytes</c> for <c>numpy.bytes_</c>, <c>object</c> for
        ///     <c>numpy.generic</c>). Null only for <c>object</c>.
        /// </summary>
        public NDScalarType __base__ => _bases.Count == 0 ? null : _bases[0];

        /// <summary>
        ///     Python's <c>__bases__</c>: the direct bases in Python's order, read-only — one for most classes, two for
        ///     NumPy's classes that also derive from a Python one (<c>numpy.float64</c>: <c>[floating, float]</c>,
        ///     <c>numpy.str_</c>: <c>[str, character]</c>), none for <c>object</c>.
        /// </summary>
        public IReadOnlyList<NDScalarType> __bases__ => _bases;

        /// <summary>
        ///     Python's <c>__mro__</c>: the method resolution order, read-only — this class, then its bases in Python's C3
        ///     order, ending in <c>object</c> (<c>[float64, floating, inexact, number, generic, float, object]</c> for
        ///     <c>numpy.float64</c>).
        /// </summary>
        public IReadOnlyList<NDScalarType> __mro__ => _mro;

        /// <summary>
        ///     True for NumPy's ten abstract classes, which have no descriptor (converting one to a dtype raises NumPy's
        ///     <see cref="TypeError"/>) and no instances; false for every concrete and built-in class.
        /// </summary>
        public bool IsAbstract => _kind == ClassKind.Abstract;

        /// <summary>
        ///     True for Python's built-in classes (<see cref="PyObject"/>, <see cref="PyFloat"/>, ...), which are not NumPy
        ///     dtypes: <c>np.isdtype</c> refuses them, and <c>np.issubdtype</c> reads one as NumPy's <c>dtype(cls).type</c>
        ///     (<c>float</c> as <c>numpy.float64</c>).
        /// </summary>
        public bool IsBuiltin => _kind == ClassKind.Builtin;

        /// <summary>Python's <c>type.mro()</c>: the method resolution order as a NEW, mutable list (Python returns a fresh list each call).</summary>
        /// <returns>A new list holding <see cref="__mro__"/>'s classes.</returns>
        public List<NDScalarType> mro() => new List<NDScalarType>(_mro);

        /// <summary>
        ///     Python's <c>type.__subclasses__()</c>: the DIRECT subclasses, as a new list, in creation order — NumPy's
        ///     registration order for NumPy's classes, abstract subclasses first (<c>np.signedinteger.__subclasses__()</c>
        ///     is <c>[int8, int16, int32, int64, timedelta64]</c> on a host whose C <c>int</c> is NumSharp's int32 class).
        ///     A built-in class lists the modelled classes only (Python's own list also holds classes outside NumPy).
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

        /// <summary>Orders siblings by their creation rank (every rank is distinct).</summary>
        private static readonly Comparer<NDScalarType> ByRank = Comparer<NDScalarType>.Create((a, b) => a._rank.CompareTo(b._rank));

        /// <summary>
        ///     The direct subclasses for one registry state (see <see cref="__subclasses__"/>): every statically created class
        ///     and every registered dtype class's class whose bases include this class, ordered by creation rank.
        /// </summary>
        /// <param name="registry">The registry's published array (read, never written).</param>
        /// <returns>The subclasses, in creation order.</returns>
        private NDScalarType[] ComputeSubclasses(DTypeMeta[] registry)
        {
            var result = new List<NDScalarType>();
            foreach (var cls in StaticClasses)
                if (cls._bases.Contains(this))
                    result.Add(cls);

            foreach (var meta in registry)
            {
                // An abstract dtype class (NEP 50's weak Python-scalar classes) has no instances and therefore no scalar
                // class; Of() would refuse it.
                if (meta._abstract)
                    continue;
                var cls = Of(meta);
                // A dtype class can map onto a statically created class (StringDType onto Python's str, a dtype class that
                // adopted a dtype-less class): that class was counted above.
                if (cls._kind != ClassKind.Concrete || cls._dtypelessTypeNum >= 0 || result.Contains(cls))
                    continue;
                if (cls._bases.Contains(this))
                    result.Add(cls);
            }

            // Every rank is distinct, so the unstable sort is deterministic.
            result.Sort(ByRank);
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
            /// <param name="subclasses">The subclasses, in creation order.</param>
            internal SubclassList(DTypeMeta[] registry, NDScalarType[] subclasses)
            {
                Registry = registry;
                Subclasses = subclasses;
            }

            /// <summary>The registry array the list was computed from (compared by reference).</summary>
            internal DTypeMeta[] Registry { get; }

            /// <summary>The subclasses, in creation order; never handed out (callers get a copy).</summary>
            internal NDScalarType[] Subclasses { get; }
        }

        /// <summary>
        ///     Python's <c>isinstance(instance, cls)</c>: true when <paramref name="instance"/> is an object of this class or
        ///     of one of its subclasses (<c>np.integer.__instancecheck__(a[0])</c> for an int32 array is true).
        /// </summary>
        /// <param name="instance">The value to test; null is not an instance of anything (Python's <c>None</c> is, but of no class NumSharp models).</param>
        /// <returns>
        ///     <para>For <see cref="PyObject"/>: true for every non-null value (every Python object is an <c>object</c>).</para>
        ///     <para>
        ///     For an <see cref="NDArray"/>: true only when it is a 0-d array that stands for a scalar — a computed 0-d
        ///     result, <c>a[0]</c> of a 1-D array or <see cref="NDArray.Scalar{T}(T)"/> — and its dtype's class is under
        ///     this one (a 0-d float64 array is a <c>numpy.float64</c>, so also a Python <c>float</c>; a 0-d float32 is
        ///     not). An array with dimensions and an explicit <c>np.array(x)</c> 0-d array are ndarrays, not scalars.
        ///     </para>
        ///     <para>
        ///     For a boxed C# value: true when its type is the element type of a dtype whose class is under this one (a
        ///     <see cref="double"/> is a <c>numpy.float64</c>, a <see cref="decimal"/> a <c>numsharp.decimal</c>), OR when the
        ///     Python object it stands for is (a C# integer is a Python <c>int</c>, a <see cref="bool"/> a <c>bool</c>, a
        ///     <see cref="float"/> / <see cref="double"/> a <c>float</c>, a <see cref="Complex"/> a <c>complex</c>, a
        ///     <see cref="string"/> a <c>str</c>, a <c>byte[]</c> a <c>bytes</c>, an <see cref="np.MemoryView"/> a
        ///     <c>memoryview</c>). False for a descriptor or a class (a class is not an instance of itself), and for a value
        ///     of a type neither mapping knows.
        ///     </para>
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     NumSharp has no scalar objects, so a C# value stands for both of the objects Python keeps apart: a
        ///     <see cref="double"/> is the <c>numpy.float64</c> an array's element is and the Python <c>float</c> a literal
        ///     is. So <c>np.float64.__instancecheck__(1.5)</c> and <c>PyFloat.__instancecheck__(1.5)</c> are both true,
        ///     where NumPy's <c>1.0</c> is a <c>float</c> but not a <c>numpy.float64</c>, and <c>PyInt.__instancecheck__(5)</c>
        ///     is true although a NumPy int32 scalar is not an <c>int</c>. A ported <c>isinstance(x, (int, np.integer))</c>
        ///     therefore answers for the values NumSharp hands out, which is what such a test is written for. The same rule
        ///     makes a C# <see cref="string"/> a <c>numpy.str_</c> scalar (the U dtype stores <see cref="string"/>), where
        ///     NumPy's <c>"abc"</c> is a <c>str</c> but not an <c>np.str_</c>.
        ///     </para>
        ///     <para>
        ///     The element type must match EXACTLY: a boxed enum is not an integer even though its underlying type is,
        ///     because no dtype stores enums and an enum is not a Python <c>int</c>.
        ///     </para>
        /// </remarks>
        public bool __instancecheck__(object instance)
        {
            if (instance is null)
                return false;
            if (ReferenceEquals(this, PyObjectClass))
                return true;

            switch (instance)
            {
                case NDArray nd:
                    // NumPy's scalars come back from NumSharp as 0-d arrays; the explicit np.array(x) form is the one 0-d
                    // array that NumPy keeps as an ndarray (NDArray.IsExplicitZeroDimArray marks exactly that spelling).
                    if (nd.ndim != 0 || nd.IsExplicitZeroDimArray)
                        return false;
                    return Of(nd.dtype.Meta).IsSubclassOf(this);
                case NDScalarType _:
                case DType _:
                case Type _:
                    // Classes and descriptors are instances of Python's type / numpy.dtype, not of a scalar class.
                    return false;
            }

            var type = instance.GetType();
            var numpyClass = OfElementType(type);
            if (numpyClass != null && numpyClass.IsSubclassOf(this))
                return true;
            var pythonClass = PythonClassOf(type);
            return pythonClass != null && pythonClass.IsSubclassOf(this);
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
        ///     <c>np.integer.__subclasscheck__(typeof(int))</c> is true. As for instances, a C# type stands for both the
        ///     NumPy class of the dtype that stores it and the Python class its values are (<c>typeof(int)</c> is
        ///     <c>numpy.int32</c> and <c>int</c>).
        /// </summary>
        /// <param name="subclass">The C# type to test.</param>
        /// <returns>
        ///     True when either class <paramref name="subclass"/> stands for derives from this one, and true for
        ///     <see cref="PyObject"/> whatever the type (every class derives from <c>object</c>); false otherwise.
        /// </returns>
        /// <exception cref="TypeError"><paramref name="subclass"/> is null (Python's <c>issubclass() arg 1 must be a class</c>).</exception>
        public bool __subclasscheck__(Type subclass)
        {
            if (subclass is null)
                throw new TypeError("issubclass() arg 1 must be a class");
            if (ReferenceEquals(this, PyObjectClass))
                return true;
            var numpyClass = OfElementType(subclass);
            if (numpyClass != null && numpyClass.IsSubclassOf(this))
                return true;
            var pythonClass = PythonClassOf(subclass);
            return pythonClass != null && pythonClass.IsSubclassOf(this);
        }

        /// <summary>
        ///     NumPy's rendering of a class, <c>repr(np.integer)</c> and <c>str(np.integer)</c> alike:
        ///     <c>&lt;class 'numpy.integer'&gt;</c>. Python leaves out the module of a built-in class
        ///     (<c>&lt;class 'float'&gt;</c>), the class of a <c>StringDType</c> descriptor included (<c>&lt;class 'str'&gt;</c>).
        /// </summary>
        /// <returns>The class repr.</returns>
        public override string ToString()
            => _kind == ClassKind.Builtin ? $"<class '{__name__}'>" : $"<class '{__module__}.{__name__}'>";

        // ---------------------------------------------------------------------------------------------------------
        // Conversions
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The scalar class of a descriptor — NumPy's <c>dtype.type</c> (<c>np.dtype('f8').type is np.float64</c>). Every
        ///     descriptor has one, so the conversion never fails; descriptors of one dtype class (<c>'&lt;f8'</c>,
        ///     <c>'&gt;f8'</c>; <c>'M8[s]'</c>, <c>'M8[ns]'</c>) share it, and a <c>StringDType</c> descriptor's is Python's
        ///     <c>str</c>.
        /// </summary>
        /// <param name="dtype">The descriptor, or null.</param>
        /// <returns>Its scalar class, or null for a null descriptor (C#'s null maps through, as every DType conversion does).</returns>
        public static implicit operator NDScalarType(DType dtype) => dtype is null ? null : Of(dtype.Meta);

        /// <summary>
        ///     The descriptor of a scalar class — NumPy's <c>np.dtype(cls)</c>: a concrete class gives its dtype class's
        ///     builtin descriptor, the same object <c>np.dtype("f8")</c> returns; a Python built-in class gives NumPy's
        ///     descriptor for it (<c>np.dtype(float)</c> is float64, <c>np.dtype(int)</c> the pointer-sized integer,
        ///     <c>np.dtype(str)</c> the unsized <c>&lt;U</c>).
        /// </summary>
        /// <param name="sctype">The scalar class, or null.</param>
        /// <returns>The descriptor, or null for a null class.</returns>
        /// <exception cref="TypeError">
        ///     <paramref name="sctype"/> is abstract: NumPy's texts, e.g. <c>Converting 'np.integer' or 'np.signedinteger' to
        ///     a dtype is not allowed</c> (the conversion is explicit for exactly this reason).
        /// </exception>
        /// <exception cref="NotSupportedException">
        ///     <paramref name="sctype"/> is a NumPy class NumSharp has no dtype for (<c>complex64</c>, <c>object_</c>,
        ///     <c>void</c>, and the built-in <c>object</c> / <c>memoryview</c>, whose NumPy dtypes are those).
        /// </exception>
        public static explicit operator DType(NDScalarType sctype) => sctype?.ToDescriptor();

        /// <summary>
        ///     NumPy's <c>PyArray_DescrFromTypeObject</c>: the builtin descriptor of a concrete class (the shared singleton:
        ///     <c>np.dtype(np.float64) is np.dtype('f8')</c>, and <c>np.dtype(np.datetime64)</c> is the generic
        ///     <c>dtype('&lt;M8')</c> singleton), or the class's default descriptor when it has no shared one; for a Python
        ///     built-in class the descriptor of its NumPy class (<see cref="NumPyClass"/>).
        /// </summary>
        /// <returns>The descriptor.</returns>
        /// <exception cref="TypeError">This class is abstract (NumPy's per-class text).</exception>
        /// <exception cref="NotSupportedException">This class (or the NumPy class of this built-in class) has no NumSharp dtype.</exception>
        internal DType ToDescriptor()
        {
            switch (_kind)
            {
                case ClassKind.Abstract:
                    throw new TypeError(_conversionError);
                case ClassKind.Builtin:
                    return NumPyClass.ToDescriptor();
            }

            var meta = Volatile.Read(ref _meta);
            if (meta is null)
                throw new NotSupportedException(_conversionError);
            return meta.singleton ?? meta.default_descr();
        }

        /// <summary>
        ///     The NumPy class NumPy reads a class as wherever it wants a dtype (<c>dtype(cls).type</c>): the class itself
        ///     for a NumPy class; for a Python built-in class the class of <c>np.dtype(cls)</c> — <c>int</c> is the
        ///     pointer-sized integer (<c>numpy.int64</c> on a 64-bit host), <c>float</c> <c>numpy.float64</c>,
        ///     <c>complex</c> <c>numpy.complex128</c>, <c>bool</c> <c>numpy.bool</c>, <c>str</c> <c>numpy.str_</c>,
        ///     <c>bytes</c> <c>numpy.bytes_</c>, <c>object</c> <c>numpy.object_</c> and <c>memoryview</c> <c>numpy.void</c>.
        ///     This is how <c>np.issubdtype(float, np.floating)</c> is True and <c>np.issubdtype(bool, int)</c> False
        ///     (<c>numpy.bool</c> is no <c>numpy.int64</c>).
        /// </summary>
        internal NDScalarType NumPyClass
        {
            get
            {
                if (_kind != ClassKind.Builtin)
                    return this;
                if (ReferenceEquals(this, PyIntClass))
                    return Of(IntPtr.Size == 8 ? DTypeRegistry.Int64 : DTypeRegistry.Int32);
                if (ReferenceEquals(this, PyBoolClass))
                    return Of(DTypeRegistry.Bool);
                if (ReferenceEquals(this, PyFloatClass))
                    return Of(DTypeRegistry.Double);
                if (ReferenceEquals(this, PyComplexClass))
                    return Of(DTypeRegistry.Complex128);
                if (ReferenceEquals(this, PyStrClass))
                    return Of(DTypeRegistry.Str);
                if (ReferenceEquals(this, PyBytesClass))
                    return Of(DTypeRegistry.Bytes);
                if (ReferenceEquals(this, PyMemoryViewClass))
                    return Void;
                return Object_;
            }
        }

        /// <summary>
        ///     Whether this class is in one of <c>np.isdtype</c>'s named kinds — NumPy's sets of CONCRETE classes built from
        ///     <c>sctypes</c>: <c>"bool"</c> is <c>numpy.bool</c>; <c>"signed integer"</c> / <c>"unsigned integer"</c> /
        ///     <c>"real floating"</c> / <c>"complex floating"</c> the concrete subclasses of <c>signedinteger</c> /
        ///     <c>unsignedinteger</c> / <c>floating</c> / <c>complexfloating</c>; <c>"integral"</c> both integer sets;
        ///     <c>"numeric"</c> all four. <c>datetime64</c> and <c>timedelta64</c> are in none (NumPy leaves them out of
        ///     <c>sctypes</c>), nor is an abstract or built-in class.
        /// </summary>
        /// <param name="kind">A kind name the caller validated.</param>
        /// <returns>True when the class belongs to the kind.</returns>
        internal bool IsInKind(string kind)
        {
            // The datetime classes are out of every set first, so below "integral" is every integer class and "numeric"
            // every number class: one MRO scan per test.
            if (_kind != ClassKind.Concrete || Volatile.Read(ref _meta) is DatetimeDTypeMeta)
                return false;
            return kind switch
            {
                "bool" => ReferenceEquals(this, Of(DTypeRegistry.Bool)),
                "signed integer" => IsSubclassOf(SignedInteger),
                "unsigned integer" => IsSubclassOf(UnsignedInteger),
                "integral" => IsSubclassOf(Integer),
                "real floating" => IsSubclassOf(Floating),
                "complex floating" => IsSubclassOf(ComplexFloating),
                "numeric" => IsSubclassOf(Number),
                _ => false,
            };
        }

        /// <summary>
        ///     NumPy's <c>dtype.kind</c> character of the descriptor <c>np.dtype(cls)</c> (the text of <c>np.iinfo</c>'s
        ///     refusal, <c>Invalid integer data type 'c'.</c>): the dtype class's kind, the kind recorded for a dtype-less
        ///     class (<c>'c'</c>, <c>'O'</c>, <c>'V'</c>), or the NumPy class's for a built-in class.
        /// </summary>
        /// <exception cref="InvalidOperationException">An abstract class, which has no descriptor and so no kind.</exception>
        internal char NumPyKind
        {
            get
            {
                if (_kind == ClassKind.Abstract)
                    throw new InvalidOperationException($"{this} is abstract: it has no descriptor and no kind.");
                if (_kind == ClassKind.Builtin)
                    return NumPyClass.NumPyKind;
                return Volatile.Read(ref _meta)?.Kind ?? _numpyKind;
            }
        }

        /// <summary>True for a NumPy class NumSharp has no dtype for yet (<c>complex64</c>, <c>object_</c>, <c>void</c>).</summary>
        internal bool IsDtypeless => _dtypelessTypeNum >= 0 && Volatile.Read(ref _meta) is null;

        /// <summary>
        ///     Python's subclass relation: true when <paramref name="other"/> is this class or one of its bases — a scan of
        ///     the MRO (at most seven entries) comparing references.
        /// </summary>
        /// <param name="other">The candidate base; null is no class's base.</param>
        /// <returns>True when <paramref name="other"/> appears in this class's MRO.</returns>
        internal bool IsSubclassOf(NDScalarType other)
        {
            if (other is null)
                return false;
            foreach (var cls in _mroItems)
                if (ReferenceEquals(cls, other))
                    return true;
            return false;
        }

        /// <summary>
        ///     The scalar class of a dtype class, built on first use and cached on <paramref name="meta"/>: a concrete class
        ///     under the abstract class its kind names (NumPy's <c>SINGLE_INHERIT</c> / <c>DUAL_INHERIT</c> table), named and
        ///     ordered as NumPy names and registers it. A <c>StringDType</c> class maps onto Python's <c>str</c>, and a dtype
        ///     class registered under a dtype-less class's NumPy type number adopts that class.
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

        /// <summary>
        ///     The Python built-in class a C# value of <paramref name="type"/> stands for in NumSharp's NEP 50 rules: a C#
        ///     integer is a Python <c>int</c>, a <see cref="bool"/> a <c>bool</c>, a <see cref="float"/> or <see cref="double"/>
        ///     a <c>float</c>, a <see cref="Complex"/> a <c>complex</c>, a <see cref="string"/> a <c>str</c>, a <c>byte[]</c> a
        ///     <c>bytes</c>, an <see cref="np.MemoryView"/> a <c>memoryview</c>; null for every other type (a
        ///     <see cref="System.Half"/>, <see cref="char"/> or <see cref="decimal"/> is a NumPy scalar, not a Python one).
        /// </summary>
        /// <param name="type">The C# type.</param>
        /// <returns>The Python class, or null.</returns>
        internal static NDScalarType PythonClassOf(Type type)
        {
            if (type == typeof(bool))
                return PyBoolClass;
            if (type == typeof(int) || type == typeof(long) || type == typeof(short) || type == typeof(sbyte) || type == typeof(byte)
                || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong) || type == typeof(BigInteger))
                return PyIntClass;
            if (type == typeof(double) || type == typeof(float))
                return PyFloatClass;
            if (type == typeof(Complex))
                return PyComplexClass;
            if (type == typeof(string))
                return PyStrClass;
            if (type == typeof(byte[]))
                return PyBytesClass;
            if (type == typeof(np.MemoryView))
                return PyMemoryViewClass;
            return null;
        }

        /// <summary>Builds the concrete class of <paramref name="meta"/> (see <see cref="Of(DTypeMeta)"/>).</summary>
        /// <param name="meta">A non-abstract dtype class.</param>
        /// <returns>The class — new, or the statically created class the dtype class maps onto.</returns>
        private static NDScalarType Build(DTypeMeta meta)
        {
            // NEP 55's StringDType stores Python's own str (np.dtypes.StringDType().type is str): its class IS that built-in
            // class, outside NumPy's scalar hierarchy.
            if (meta is VStringDTypeMeta)
                return PyStrClass;

            // A dtype class registered under a dtype-less class's NumPy type number adopts that class, so the object
            // np.complex64 handed out before the registration stays the class (and converts to the new descriptor).
            foreach (var cls in DtypelessClasses)
            {
                if (cls._dtypelessTypeNum == meta.type_num)
                {
                    Interlocked.CompareExchange(ref cls._meta, meta, null);
                    return cls;
                }
            }

            // NumPy names the flexible and object classes by kind (one class per kind: numpy.bytes_, numpy.str_), whatever
            // a NumSharp dtype class calls its scalar; the others by the scalar name the dtype class records
            // (numpy.float64, numsharp.decimal).
            string name = meta.Kind switch
            {
                'S' => "bytes_",
                'U' => "str_",
                _ => meta.ScalarName,
            };
            string module = meta.ModulePath.StartsWith("numpy", StringComparison.Ordinal) ? "numpy" : "numsharp";
            return new NDScalarType(name, module, ClassKind.Concrete, BasesOf(meta), RankOf(meta), conversionError: null, meta,
                dtypelessTypeNum: -1, numpyKind: meta.Kind, numpyItemSize: meta.ItemSize);
        }

        /// <summary>
        ///     The direct bases of a dtype class's concrete class, in Python's order — NumPy's <c>SINGLE_INHERIT</c> /
        ///     <c>DUAL_INHERIT</c> / <c>DUAL_INHERIT2</c> table: the abstract class its kind names (bool, datetime64 and
        ///     object_ directly under <c>generic</c> — "datetime doesn't fit in any category" —, timedelta64 under
        ///     <c>signedinteger</c> — "timedelta is an integer with an associated unit" —, the string classes under
        ///     <c>character</c>, void under <c>flexible</c>), plus a Python built-in base for exactly four NumPy C types:
        ///     <c>double</c> (<c>[floating, float]</c>), <c>cdouble</c> (<c>[complexfloating, complex]</c>), and the strings,
        ///     whose built-in base comes FIRST (<c>DUAL_INHERIT2</c>: <c>[bytes, character]</c>, <c>[str, character]</c>).
        /// </summary>
        /// <param name="meta">The dtype class.</param>
        /// <returns>The bases.</returns>
        /// <exception cref="InvalidOperationException">A kind with no NumPy scalar class (a bug: StringDType is handled before).</exception>
        private static NDScalarType[] BasesOf(DTypeMeta meta)
        {
            NDScalarType parent = meta.Kind switch
            {
                'b' or 'M' or 'O' => Generic,
                'i' or 'm' => SignedInteger,
                'u' => UnsignedInteger,
                'f' => Floating,
                'c' => ComplexFloating,
                'S' or 'U' => Character,
                'V' => Flexible,
                _ => throw new InvalidOperationException($"{meta} has kind '{meta.Kind}', which no NumPy scalar class takes."),
            };

            // The Python base belongs to NumPy's C type (its type number), not to the kind: float64 derives from float,
            // longdouble does not; complex128 from complex, complex64 does not. NumSharp-only classes have none.
            if (meta.ModulePath.StartsWith("numpy", StringComparison.Ordinal))
            {
                switch (meta.type_num)
                {
                    case 12: return new[] { parent, PyFloatClass };   // NPY_DOUBLE: DUAL_INHERIT(Double, Float, Floating)
                    case 15: return new[] { parent, PyComplexClass }; // NPY_CDOUBLE: DUAL_INHERIT(CDouble, Complex, ComplexFloating)
                    case 18: return new[] { PyBytesClass, parent };   // NPY_STRING: DUAL_INHERIT2(String, Bytes, Character)
                    case 19: return new[] { PyStrClass, parent };     // NPY_UNICODE: DUAL_INHERIT2(Unicode, Unicode, Character)
                }
            }

            return new[] { parent };
        }

        /// <summary>The creation rank of a dtype class's class: NumPy's C-type registration rank, or 1000 plus the type number for a class NumPy does not have.</summary>
        /// <param name="meta">The dtype class.</param>
        /// <returns>The rank.</returns>
        private static int RankOf(DTypeMeta meta)
        {
            if (meta.ModulePath.StartsWith("numpy", StringComparison.Ordinal))
            {
                int rank = RankOfTypeNum(meta.type_num);
                if (rank >= 0)
                    return rank;
            }

            return 1000 + meta.type_num;
        }

        /// <summary>
        ///     NumPy's creation order of its concrete classes (<c>setup_scalartypes</c>): Bool, Byte, Short, Int, Long,
        ///     LongLong, Datetime, Timedelta, UByte, UShort, UInt, ULong, ULongLong, Half, Float, Double, LongDouble, CFloat,
        ///     CDouble, CLongDouble, String, Unicode, Void, Object — so <c>np.floating.__subclasses__()</c> starts with
        ///     float16 although its type number is the highest, and Windows' <c>intc</c> comes before its <c>int32</c>
        ///     (<c>long</c>). Keyed by NumPy's type number, because NumSharp registers each class under the type number of
        ///     the C type it is on this host.
        /// </summary>
        /// <param name="typeNum">A NumPy type number.</param>
        /// <returns>The rank (100 and up), or -1 for a type number outside NumPy's builtin set.</returns>
        private static int RankOfTypeNum(int typeNum) => typeNum switch
        {
            0 => 100,   // NPY_BOOL
            1 => 101,   // NPY_BYTE
            3 => 102,   // NPY_SHORT
            5 => 103,   // NPY_INT
            7 => 104,   // NPY_LONG
            9 => 105,   // NPY_LONGLONG
            21 => 106,  // NPY_DATETIME
            22 => 107,  // NPY_TIMEDELTA
            2 => 108,   // NPY_UBYTE
            4 => 109,   // NPY_USHORT
            6 => 110,   // NPY_UINT
            8 => 111,   // NPY_ULONG
            10 => 112,  // NPY_ULONGLONG
            23 => 113,  // NPY_HALF
            11 => 114,  // NPY_FLOAT
            12 => 115,  // NPY_DOUBLE
            13 => 116,  // NPY_LONGDOUBLE
            14 => 117,  // NPY_CFLOAT
            15 => 118,  // NPY_CDOUBLE
            16 => 119,  // NPY_CLONGDOUBLE
            18 => 120,  // NPY_STRING
            19 => 121,  // NPY_UNICODE
            20 => 122,  // NPY_VOID
            17 => 123,  // NPY_OBJECT
            _ => -1,
        };

        /// <summary>
        ///     Python's C3 method resolution order of a class with the given bases: the class, then the merge of its bases'
        ///     orders and the base list, taking at each step the first head that appears in no other list's tail
        ///     (<c>numpy.float64</c>: <c>[float64, floating, inexact, number, generic, float, object]</c>).
        /// </summary>
        /// <param name="cls">The class being built.</param>
        /// <param name="bases">Its direct bases, each with a complete MRO.</param>
        /// <returns>The MRO, <paramref name="cls"/> first.</returns>
        /// <exception cref="InvalidOperationException">No consistent order exists (Python's <c>TypeError: Cannot create a consistent method resolution order</c>).</exception>
        private static NDScalarType[] Linearize(NDScalarType cls, NDScalarType[] bases)
        {
            var result = new List<NDScalarType> { cls };
            var sequences = new List<List<NDScalarType>>(bases.Length + 1);
            foreach (var b in bases)
                sequences.Add(new List<NDScalarType>(b._mro));
            sequences.Add(new List<NDScalarType>(bases));

            while (true)
            {
                sequences.RemoveAll(s => s.Count == 0);
                if (sequences.Count == 0)
                    return result.ToArray();

                NDScalarType head = null;
                foreach (var sequence in sequences)
                {
                    var candidate = sequence[0];
                    bool inATail = false;
                    foreach (var other in sequences)
                    {
                        if (other.IndexOf(candidate) > 0)
                        {
                            inATail = true;
                            break;
                        }
                    }

                    if (!inATail)
                    {
                        head = candidate;
                        break;
                    }
                }

                if (head is null)
                    throw new InvalidOperationException($"Cannot create a consistent method resolution order (MRO) for {cls.__name__}.");
                result.Add(head);
                foreach (var sequence in sequences)
                    if (ReferenceEquals(sequence[0], head))
                        sequence.RemoveAt(0);
            }
        }
    }
}
