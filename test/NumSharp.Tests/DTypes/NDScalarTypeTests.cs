using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using NumSharp.Generic;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     NumPy's scalar-type hierarchy as <see cref="NDScalarType"/> objects (<c>np.generic</c>, <c>np.number</c>,
    ///     <c>np.integer</c>, ...), the Python built-in classes it derives from (<c>object</c>, <c>float</c>, <c>str</c>, ...),
    ///     the NumPy classes NumSharp has no dtype for (<c>np.complex64</c>, <c>np.object_</c>, <c>np.void</c>), their
    ///     integration in <c>np.issubdtype</c> / <c>np.isdtype</c> / <c>np.dtype</c> / <c>np.finfo</c> / <c>np.iinfo</c>, and
    ///     the relatives <c>np.typecodes</c>, <c>np.sctypeDict</c>, <c>np.True_</c> / <c>np.False_</c>, <c>np.longdouble</c>
    ///     and <c>np.typing</c>. Every NumPy value was probed against <c>numpy==2.4.2</c> (win-amd64); NumSharp's own
    ///     modelling choices (C# values are scalars, the collapsed and NumSharp-only classes) are named where they apply.
    /// </summary>
    [TestClass]
    public class NDScalarTypeTests
    {
        /// <summary>The ten abstract classes, in NumPy's registration order.</summary>
        private static NDScalarType[] Abstract => new[]
        {
            np.generic, np.number, np.integer, np.inexact, np.signedinteger, np.unsignedinteger, np.floating,
            np.complexfloating, np.flexible, np.character,
        };

        /// <summary>The eight modelled Python built-in classes, in the order Python created them.</summary>
        private static NDScalarType[] Builtins => new[]
        {
            NDScalarType.PyObject, NDScalarType.PyInt, NDScalarType.PyBool, NDScalarType.PyFloat, NDScalarType.PyComplex,
            NDScalarType.PyStr, NDScalarType.PyBytes, NDScalarType.PyMemoryView,
        };

        /// <summary>
        ///     The names of a class list as NumPy's probes print them: <c>module.name</c> for a NumPy or NumSharp class
        ///     (<c>numpy.float64</c>, <c>numsharp.char</c>), the bare name for a Python built-in (<c>float</c>, <c>object</c>)
        ///     — the only way to tell Python's <c>bool</c> from <c>numpy.bool</c> in one list.
        /// </summary>
        /// <param name="classes">The classes.</param>
        /// <returns>Their qualified names, in order.</returns>
        private static string[] Names(IEnumerable<NDScalarType> classes)
            => classes.Select(c => c.IsBuiltin ? c.__name__ : c.__module__ + "." + c.__name__).ToArray();

        // ---- the abstract classes' own surface ---------------------------------------------------------------------

        /// <summary><c>__name__</c>, <c>__qualname__</c>, <c>__module__</c> and the repr are NumPy's.</summary>
        [TestMethod]
        public void AbstractClasses_NamesAndRepr_AreNumPys()
        {
            foreach (var (cls, name) in Abstract.Zip(new[] { "generic", "number", "integer", "inexact", "signedinteger",
                                                             "unsignedinteger", "floating", "complexfloating", "flexible",
                                                             "character" }))
            {
                cls.__name__.Should().Be(name);
                cls.__qualname__.Should().Be(name);
                cls.__module__.Should().Be("numpy");
                cls.ToString().Should().Be($"<class 'numpy.{name}'>");
                cls.IsAbstract.Should().BeTrue();
            }
        }

        /// <summary>Each property returns the one class object (identity is what issubclass compares).</summary>
        [TestMethod]
        public void AbstractClasses_AreSingletons()
        {
            ReferenceEquals(np.integer, np.integer).Should().BeTrue();
            ReferenceEquals(np.floating.__base__, np.inexact).Should().BeTrue();
            ReferenceEquals(np.number.__base__, np.generic).Should().BeTrue();
        }

        /// <summary>
        ///     The MRO and the bases are NumPy's whole tuples: every one ends in Python's <c>object</c>, which is
        ///     <c>np.generic</c>'s base (<c>np.generic.__base__</c> is <c>object</c>, whose own base is None).
        /// </summary>
        [TestMethod]
        public void AbstractClasses_MroAndBases()
        {
            Names(np.generic.__mro__).Should().Equal("numpy.generic", "object");
            Names(np.integer.__mro__).Should().Equal("numpy.integer", "numpy.number", "numpy.generic", "object");
            Names(np.signedinteger.__mro__).Should().Equal("numpy.signedinteger", "numpy.integer", "numpy.number", "numpy.generic", "object");
            Names(np.complexfloating.__mro__).Should().Equal("numpy.complexfloating", "numpy.inexact", "numpy.number", "numpy.generic", "object");
            Names(np.character.__mro__).Should().Equal("numpy.character", "numpy.flexible", "numpy.generic", "object");
            Names(np.generic.__bases__).Should().Equal("object");
            ReferenceEquals(np.generic.__base__, NDScalarType.PyObject).Should().BeTrue();
            NDScalarType.PyObject.__base__.Should().BeNull();
            NDScalarType.PyObject.__bases__.Should().BeEmpty();
            Names(np.floating.__bases__).Should().Equal("numpy.inexact");
        }

        /// <summary><c>mro()</c> is a NEW mutable list each call, as Python's is; <c>__mro__</c> is read-only.</summary>
        [TestMethod]
        public void Mro_ReturnsAFreshList()
        {
            var first = np.integer.mro();
            first.Add(np.generic);
            Names(np.integer.mro()).Should().Equal("numpy.integer", "numpy.number", "numpy.generic", "object");
            new Action(() => ((IList<NDScalarType>)np.integer.__mro__).Add(np.generic)).Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     <c>__subclasses__()</c> lists the direct subclasses in NumPy's registration order — abstract first, then
        ///     concrete, the dtype-less classes in their NumPy places (<c>object_</c> last under generic, <c>void</c> after
        ///     <c>character</c>, <c>complex64</c> before <c>complex128</c>). NumSharp's lists differ from NumPy's only by the
        ///     classes it collapses (intc, longdouble, clongdouble) or adds (decimal, char).
        /// </summary>
        [TestMethod]
        public void Subclasses_FollowNumPysRegistrationOrder()
        {
            Names(np.generic.__subclasses__()).Should().Equal("numpy.number", "numpy.flexible", "numpy.bool", "numpy.datetime64", "numpy.object_");
            Names(np.number.__subclasses__()).Should().Equal("numpy.integer", "numpy.inexact");
            Names(np.integer.__subclasses__()).Should().Equal("numpy.signedinteger", "numpy.unsignedinteger");
            Names(np.signedinteger.__subclasses__()).Should().Equal("numpy.int8", "numpy.int16", "numpy.int32", "numpy.int64", "numpy.timedelta64");
            Names(np.unsignedinteger.__subclasses__()).Should().Equal("numpy.uint8", "numpy.uint16", "numpy.uint32", "numpy.uint64", "numsharp.char");
            Names(np.inexact.__subclasses__()).Should().Equal("numpy.floating", "numpy.complexfloating");
            Names(np.floating.__subclasses__()).Should().Equal("numpy.float16", "numpy.float32", "numpy.float64", "numsharp.decimal");
            Names(np.complexfloating.__subclasses__()).Should().Equal("numpy.complex64", "numpy.complex128");
            Names(np.flexible.__subclasses__()).Should().Equal("numpy.character", "numpy.void");
            Names(np.character.__subclasses__()).Should().Equal("numpy.bytes_", "numpy.str_");
            ((NDScalarType)np.float64).__subclasses__().Should().BeEmpty();
            np.complex64.__subclasses__().Should().BeEmpty();
            // NumPy's void lists numpy.record (the class of a structured record), which NumSharp does not model.
            np.@void.__subclasses__().Should().BeEmpty();
        }

        // ---- Python's built-in classes ------------------------------------------------------------------------------

        /// <summary>
        ///     The built-in classes print, name and derive as Python's do: module <c>builtins</c>, a repr without the module
        ///     (<c>&lt;class 'float'&gt;</c>), every one directly under <c>object</c> but <c>bool</c>, which derives from
        ///     <c>int</c>. None is abstract.
        /// </summary>
        [TestMethod]
        public void BuiltinClasses_NamesBasesAndRepr()
        {
            var names = new[] { "object", "int", "bool", "float", "complex", "str", "bytes", "memoryview" };
            foreach (var (cls, name) in Builtins.Zip(names))
            {
                cls.__name__.Should().Be(name);
                cls.__qualname__.Should().Be(name);
                cls.__module__.Should().Be("builtins");
                cls.ToString().Should().Be($"<class '{name}'>");
                cls.IsBuiltin.Should().BeTrue();
                cls.IsAbstract.Should().BeFalse();
            }

            Names(NDScalarType.PyBool.__mro__).Should().Equal("bool", "int", "object");
            Names(NDScalarType.PyFloat.__mro__).Should().Equal("float", "object");
            Names(NDScalarType.PyObject.__mro__).Should().Equal("object");
            ReferenceEquals(NDScalarType.PyBool.__base__, NDScalarType.PyInt).Should().BeTrue();
            ReferenceEquals(NDScalarType.PyStr.__base__, NDScalarType.PyObject).Should().BeTrue();
            foreach (var cls in Abstract)
                cls.IsBuiltin.Should().BeFalse();
            ((NDScalarType)np.float64).IsBuiltin.Should().BeFalse();
        }

        /// <summary>
        ///     NumPy's four classes that also derive from a Python class (its <c>DUAL_INHERIT</c> / <c>DUAL_INHERIT2</c>
        ///     table): float64 from <c>float</c> and complex128 from <c>complex</c> after their NumPy base, bytes_ and str_
        ///     from <c>bytes</c> / <c>str</c> BEFORE <c>character</c> (so their <c>__base__</c> is the Python class); no
        ///     other class has a Python base (float32, complex64 and the integers do not). The MROs are Python's C3 order.
        /// </summary>
        [TestMethod]
        public void DualInheritance_FollowsNumPysTable()
        {
            Names(((NDScalarType)np.float64).__bases__).Should().Equal("numpy.floating", "float");
            Names(((NDScalarType)np.float64).__mro__).Should().Equal(
                "numpy.float64", "numpy.floating", "numpy.inexact", "numpy.number", "numpy.generic", "float", "object");
            Names(((NDScalarType)np.complex128).__bases__).Should().Equal("numpy.complexfloating", "complex");
            Names(((NDScalarType)np.complex128).__mro__).Should().Equal(
                "numpy.complex128", "numpy.complexfloating", "numpy.inexact", "numpy.number", "numpy.generic", "complex", "object");
            Names(((NDScalarType)np.bytes_).__bases__).Should().Equal("bytes", "numpy.character");
            Names(((NDScalarType)np.bytes_).__mro__).Should().Equal(
                "numpy.bytes_", "bytes", "numpy.character", "numpy.flexible", "numpy.generic", "object");
            Names(((NDScalarType)np.str_).__bases__).Should().Equal("str", "numpy.character");
            Names(((NDScalarType)np.str_).__mro__).Should().Equal(
                "numpy.str_", "str", "numpy.character", "numpy.flexible", "numpy.generic", "object");
            ReferenceEquals(((NDScalarType)np.str_).__base__, NDScalarType.PyStr).Should().BeTrue();
            ReferenceEquals(((NDScalarType)np.float64).__base__, np.floating).Should().BeTrue();
            Names(((NDScalarType)np.float32).__bases__).Should().Equal("numpy.floating");
            Names(np.complex64.__bases__).Should().Equal("numpy.complexfloating");
            Names(((NDScalarType)np.int64).__mro__).Should().Equal(
                "numpy.int64", "numpy.signedinteger", "numpy.integer", "numpy.number", "numpy.generic", "object");
        }

        /// <summary>
        ///     A built-in class's <c>__subclasses__()</c> lists the modelled classes deriving from it: <c>float</c> →
        ///     <c>[numpy.float64]</c>, <c>int</c> → <c>[bool]</c>, <c>object</c> → the other built-ins and
        ///     <c>numpy.generic</c> (Python's own lists also hold classes outside NumPy: <c>enum.IntEnum</c> under
        ///     <c>int</c>, <c>enum.StrEnum</c> under <c>str</c>, some 360 classes under <c>object</c>).
        /// </summary>
        [TestMethod]
        public void BuiltinClasses_Subclasses()
        {
            Names(NDScalarType.PyFloat.__subclasses__()).Should().Equal("numpy.float64");
            Names(NDScalarType.PyComplex.__subclasses__()).Should().Equal("numpy.complex128");
            Names(NDScalarType.PyBytes.__subclasses__()).Should().Equal("numpy.bytes_");
            Names(NDScalarType.PyStr.__subclasses__()).Should().Equal("numpy.str_");
            Names(NDScalarType.PyInt.__subclasses__()).Should().Equal("bool");
            NDScalarType.PyMemoryView.__subclasses__().Should().BeEmpty();
            NDScalarType.PyBool.__subclasses__().Should().BeEmpty();
            Names(NDScalarType.PyObject.__subclasses__()).Should().Equal(
                "int", "float", "complex", "str", "bytes", "memoryview", "numpy.generic");
        }

        /// <summary>
        ///     A StringDType descriptor's class IS Python's <c>str</c> (<c>np.dtypes.StringDType().type is str</c>), outside
        ///     NumPy's scalar hierarchy.
        /// </summary>
        [TestMethod]
        public void StringDType_ClassIsPythonsStr()
        {
            ReferenceEquals((NDScalarType)np.dtype("T"), NDScalarType.PyStr).Should().BeTrue();
            np.generic.__subclasscheck__(np.dtype("T")).Should().BeFalse();
            np.issubdtype(np.dtype("U5"), np.dtype("T")).Should().BeTrue();
            np.issubdtype(np.dtype("T"), np.character).Should().BeFalse();
        }

        // ---- the NumPy classes NumSharp has no dtype for --------------------------------------------------------------

        /// <summary>
        ///     <c>np.complex64</c> (= <c>np.csingle</c>), <c>np.object_</c> and <c>np.void</c> are NumPy's concrete classes
        ///     in their NumPy places of the hierarchy; converting one to a dtype refuses with a
        ///     <see cref="NotSupportedException"/> that names what NumSharp stores instead.
        /// </summary>
        [TestMethod]
        public void DtypelessClasses_ExistAndRefuseTheConversion()
        {
            ReferenceEquals(np.csingle, np.complex64).Should().BeTrue();
            np.complex64.ToString().Should().Be("<class 'numpy.complex64'>");
            np.object_.ToString().Should().Be("<class 'numpy.object_'>");
            np.@void.ToString().Should().Be("<class 'numpy.void'>");
            Names(np.complex64.__mro__).Should().Equal(
                "numpy.complex64", "numpy.complexfloating", "numpy.inexact", "numpy.number", "numpy.generic", "object");
            Names(np.object_.__mro__).Should().Equal("numpy.object_", "numpy.generic", "object");
            Names(np.@void.__mro__).Should().Equal("numpy.void", "numpy.flexible", "numpy.generic", "object");
            foreach (var cls in new[] { np.complex64, np.object_, np.@void })
            {
                cls.IsAbstract.Should().BeFalse();
                cls.IsBuiltin.Should().BeFalse();
                new Action(() => np.dtype(cls)).Should().Throw<NotSupportedException>();
                new Action(() => _ = (DType)cls).Should().Throw<NotSupportedException>();
            }

            new Action(() => np.dtype(np.complex64)).Should().Throw<NotSupportedException>()
                .WithMessage("*complex64*").WithMessage("*complex128*");
            new Action(() => np.dtype(np.object_)).Should().Throw<NotSupportedException>().WithMessage("*object dtype*");
            new Action(() => np.dtype(np.@void)).Should().Throw<NotSupportedException>().WithMessage("*void dtype*");
        }

        /// <summary>
        ///     <c>np.dtype</c> of a Python built-in class is NumPy's descriptor for it — the shared singletons for
        ///     <c>float</c> / <c>complex</c> / <c>bool</c> / <c>int</c> (pointer-sized), the unsized string descriptors for
        ///     <c>str</c> / <c>bytes</c> — and refuses <c>object</c> and <c>memoryview</c>, whose NumPy dtypes (<c>O</c>,
        ///     <c>V</c>) NumSharp lacks.
        /// </summary>
        [TestMethod]
        public void BuiltinClass_ToDtype_IsNumPysDescriptor()
        {
            ReferenceEquals(np.dtype(NDScalarType.PyFloat), np.float64).Should().BeTrue();
            ReferenceEquals(np.dtype(NDScalarType.PyComplex), np.complex128).Should().BeTrue();
            ReferenceEquals(np.dtype(NDScalarType.PyBool), np.bool_).Should().BeTrue();
            ReferenceEquals(np.dtype(NDScalarType.PyInt), np.intp).Should().BeTrue();
            np.dtype(NDScalarType.PyStr).ToString(true).Should().Be("dtype('<U')");
            np.dtype(NDScalarType.PyBytes).ToString(true).Should().Be("dtype('S')");
            new Action(() => np.dtype(NDScalarType.PyObject)).Should().Throw<NotSupportedException>();
            new Action(() => np.dtype(NDScalarType.PyMemoryView)).Should().Throw<NotSupportedException>();
        }

        // ---- concrete classes and the conversions ------------------------------------------------------------------

        /// <summary>
        ///     A descriptor converts to its scalar class (NumPy's <c>dtype.type</c>): one object per dtype class, shared by
        ///     every descriptor of the class (byte orders, datetime units).
        /// </summary>
        [TestMethod]
        public void Descriptor_ConvertsToItsClass_OneObjectPerDTypeClass()
        {
            NDScalarType f8 = np.float64;
            ReferenceEquals(f8, (NDScalarType)np.dtype("<f8")).Should().BeTrue();
            ReferenceEquals(f8, (NDScalarType)np.dtype(">f8")).Should().BeTrue();
            ReferenceEquals((NDScalarType)np.dtype("M8[s]"), (NDScalarType)np.dtype("M8[ns]")).Should().BeTrue();
            f8.ToString().Should().Be("<class 'numpy.float64'>");
            f8.IsAbstract.Should().BeFalse();
            Names(((NDScalarType)np.dtype("m8[s]")).__mro__).Should().Equal(
                "numpy.timedelta64", "numpy.signedinteger", "numpy.integer", "numpy.number", "numpy.generic", "object");
            Names(((NDScalarType)np.dtype("M8[s]")).__mro__).Should().Equal("numpy.datetime64", "numpy.generic", "object");
            Names(((NDScalarType)np.bool_).__mro__).Should().Equal("numpy.bool", "numpy.generic", "object");
            ((NDScalarType)np.@decimal).ToString().Should().Be("<class 'numsharp.decimal'>");
            Names(((NDScalarType)np.@decimal).__mro__).Should().Equal(
                "numsharp.decimal", "numpy.floating", "numpy.inexact", "numpy.number", "numpy.generic", "object");
            ((NDScalarType)(DType)null).Should().BeNull();
        }

        /// <summary>
        ///     The way back gives the builtin descriptor (<c>np.dtype(np.float64) is np.dtype('f8')</c>; the generic
        ///     <c>dtype('&lt;M8')</c> singleton for datetime64), and NumPy's TypeError for an abstract class.
        /// </summary>
        [TestMethod]
        public void Class_ConvertsBackToTheBuiltinDescriptor()
        {
            ReferenceEquals((DType)(NDScalarType)np.float64, np.float64).Should().BeTrue();
            ReferenceEquals(np.dtype((NDScalarType)np.int8), np.int8).Should().BeTrue();
            var m8 = np.dtype((NDScalarType)np.dtype("M8[s]"));
            m8.ToString(true).Should().Be("dtype('<M8')");
            ReferenceEquals(m8, np.dtype("M")).Should().BeTrue();
            ReferenceEquals(np.dtype((NDScalarType)np.float32, copy: true), np.float32).Should().BeFalse();
            ((DType)(NDScalarType)null).Should().BeNull();
        }

        /// <summary>Converting an abstract class to a dtype raises NumPy's per-class TypeError, verbatim (two texts name other classes, as in NumPy).</summary>
        [TestMethod]
        public void AbstractClass_ToDtype_RaisesNumPysTypeError()
        {
            var expected = new Dictionary<NDScalarType, string>
            {
                [np.generic] = "Converting `np.generic` to a dtype is not allowed.",
                [np.number] = "Converting `np.inexact` or `np.floating` to a dtype not allowed",
                [np.integer] = "Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed",
                [np.signedinteger] = "Converting 'np.integer' or 'np.signedinteger' to a dtype is not allowed",
                [np.unsignedinteger] = "Converting `np.unsignedinteger` to a dtype is not allowed",
                [np.inexact] = "Converting `np.inexact` or `np.floating` to a dtype not allowed",
                [np.floating] = "Converting `np.inexact` or `np.floating` to a dtype not allowed",
                [np.complexfloating] = "Converting `np.complex` to a dtype is not allowed.",
                [np.flexible] = "Converting `np.generic` to a dtype is not allowed.",
                [np.character] = "Converting `np.character` to a dtype is not allowed",
            };
            foreach (var (cls, text) in expected)
            {
                new Action(() => np.dtype(cls)).Should().Throw<TypeError>().WithMessage(text);
                new Action(() => _ = (DType)cls).Should().Throw<TypeError>().WithMessage(text);
            }
            new Action(() => np.dtype((NDScalarType)null)).Should().Throw<ArgumentNullException>();
        }

        /// <summary>
        ///     <c>np.finfo</c> / <c>np.iinfo</c> of an abstract class raise NumPy's ValueErrors (both name the object
        ///     dtype: NumPy retries with the argument's type, Python's <c>type</c>); a concrete class gives its limits.
        /// </summary>
        [TestMethod]
        public void FinfoIinfo_OfAClass()
        {
            new Action(() => np.finfo(np.floating)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.object_'> not inexact");
            new Action(() => np.iinfo(np.integer)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'O'.");
            np.finfo((NDScalarType)np.float32).bits.Should().Be(32);
            np.iinfo((NDScalarType)np.int16).max.Should().Be(short.MaxValue);
            new Action(() => np.finfo((NDScalarType)np.int32)).Should().Throw<ValueError>()
                .WithMessage("data type <class 'numpy.int32'> not inexact");
        }

        /// <summary>
        ///     <c>np.finfo</c> / <c>np.iinfo</c> of the dtype-less and the built-in classes, NumPy 2.4.2's answers: a complex
        ///     class reports its real part's limits (<c>np.finfo(np.complex64)</c> is float32's, which NumSharp has); a
        ///     built-in class is read as NumPy's class for it (<c>float</c> → float64, <c>int</c> → int64); every refusal
        ///     names that class, or its dtype's kind character.
        /// </summary>
        [TestMethod]
        public void FinfoIinfo_DtypelessAndBuiltinClasses()
        {
            np.finfo(np.complex64).dtype.Should().Be(np.float32);
            new Action(() => np.iinfo(np.complex64)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'c'.");
            new Action(() => np.finfo(np.object_)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.object_'> not inexact");
            new Action(() => np.iinfo(np.object_)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'O'.");
            new Action(() => np.finfo(np.@void)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.void'> not inexact");
            new Action(() => np.iinfo(np.@void)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'V'.");

            np.finfo(NDScalarType.PyFloat).dtype.Should().Be(np.float64);
            np.finfo(NDScalarType.PyComplex).dtype.Should().Be(np.float64);
            np.iinfo(NDScalarType.PyInt).dtype.Should().Be(np.int64);
            new Action(() => np.iinfo(NDScalarType.PyFloat)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'f'.");
            new Action(() => np.finfo(NDScalarType.PyInt)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.int64'> not inexact");
            new Action(() => np.finfo(NDScalarType.PyBool)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.bool'> not inexact");
            new Action(() => np.iinfo(NDScalarType.PyBool)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'b'.");
            new Action(() => np.finfo(NDScalarType.PyStr)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.str_'> not inexact");
            new Action(() => np.iinfo(NDScalarType.PyStr)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'U'.");
            new Action(() => np.finfo(NDScalarType.PyBytes)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.bytes_'> not inexact");
            new Action(() => np.iinfo(NDScalarType.PyBytes)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'S'.");
            new Action(() => np.finfo(NDScalarType.PyObject)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.object_'> not inexact");
            new Action(() => np.iinfo(NDScalarType.PyObject)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'O'.");
            new Action(() => np.finfo(NDScalarType.PyMemoryView)).Should().Throw<ValueError>().WithMessage("data type <class 'numpy.void'> not inexact");
            new Action(() => np.iinfo(NDScalarType.PyMemoryView)).Should().Throw<ValueError>().WithMessage("Invalid integer data type 'V'.");
        }

        // ---- np.issubdtype --------------------------------------------------------------------------------------------

        /// <summary>The abstract hierarchy, every pair (NumPy 2.4.2's answers).</summary>
        [TestMethod]
        public void IsSubdtype_AbstractPairs()
        {
            np.issubdtype(np.integer, np.number).Should().BeTrue();
            np.issubdtype(np.number, np.integer).Should().BeFalse();
            np.issubdtype(np.signedinteger, np.integer).Should().BeTrue();
            np.issubdtype(np.floating, np.inexact).Should().BeTrue();
            np.issubdtype(np.complexfloating, np.floating).Should().BeFalse();
            np.issubdtype(np.character, np.flexible).Should().BeTrue();
            np.issubdtype(np.flexible, np.number).Should().BeFalse();
            foreach (var cls in Abstract)
            {
                np.issubdtype(cls, np.generic).Should().BeTrue();
                np.issubdtype(cls, cls).Should().BeTrue();
            }
        }

        /// <summary>Every dtype against every abstract class, from NumPy's hierarchy (bool and datetime64 directly under generic, timedelta64 a signed integer).</summary>
        [TestMethod]
        public void IsSubdtype_EveryDtype_AgainstTheFamilies()
        {
            // dtype -> the abstract classes it is under (besides generic)
            var under = new Dictionary<string, string[]>
            {
                ["bool"] = new string[0],
                ["int8"] = new[] { "number", "integer", "signedinteger" },
                ["int64"] = new[] { "number", "integer", "signedinteger" },
                ["uint8"] = new[] { "number", "integer", "unsignedinteger" },
                ["uint64"] = new[] { "number", "integer", "unsignedinteger" },
                ["float16"] = new[] { "number", "inexact", "floating" },
                ["float64"] = new[] { "number", "inexact", "floating" },
                ["complex128"] = new[] { "number", "inexact", "complexfloating" },
                ["M8[s]"] = new string[0],
                ["m8[s]"] = new[] { "number", "integer", "signedinteger" },
            };
            foreach (var (dtype, families) in under)
                foreach (var cls in Abstract)
                    np.issubdtype(dtype, cls).Should().Be(cls == np.generic || families.Contains(cls.__name__), $"issubdtype({dtype}, np.{cls.__name__})");
        }

        /// <summary>
        ///     NumSharp's own dtypes sit in the family their kind names (decimal floating, char an unsigned integer),
        ///     as np.isdtype has always placed them.
        /// </summary>
        [TestMethod]
        public void IsSubdtype_NumSharpOnlyDtypes()
        {
            np.issubdtype(np.@decimal, np.floating).Should().BeTrue();
            np.issubdtype(np.@decimal, np.integer).Should().BeFalse();
            np.issubdtype(np.@char, np.unsignedinteger).Should().BeTrue();
            np.issubdtype(np.@char, np.number).Should().BeTrue();
        }

        /// <summary>Every dtype spelling binds: a C# type, a storage code, a dtype string; a class against a dtype; null as NumPy's None (float64).</summary>
        [TestMethod]
        public void IsSubdtype_EverySpelling()
        {
            np.issubdtype(typeof(short), np.signedinteger).Should().BeTrue();
            np.issubdtype(NPTypeCode.Half, np.floating).Should().BeTrue();
            np.issubdtype("u2", np.unsignedinteger).Should().BeTrue();
            np.issubdtype(np.integer, "i4").Should().BeFalse();
            np.issubdtype(np.integer, np.int32).Should().BeFalse();
            np.issubdtype((DType)null, np.floating).Should().BeTrue();
            np.issubdtype((NDScalarType)null, np.inexact).Should().BeTrue();
            np.issubdtype(np.integer, (NDScalarType)null).Should().BeFalse();
        }

        /// <summary>
        ///     A Python built-in class is read as NumPy's <c>dtype(cls).type</c> (<c>float</c> as numpy.float64, <c>int</c>
        ///     as numpy.int64, <c>object</c> as numpy.object_, <c>memoryview</c> as numpy.void), so the pairs answer NumPy
        ///     2.4.2's way: <c>np.issubdtype(float, np.floating)</c> is True and <c>np.issubdtype(bool, int)</c> False
        ///     (numpy.bool is no numpy.int64), although Python's <c>bool</c> derives from <c>int</c>.
        /// </summary>
        [TestMethod]
        public void IsSubdtype_PythonBuiltins()
        {
            var py = new Dictionary<string, NDScalarType>
            {
                ["int"] = NDScalarType.PyInt, ["float"] = NDScalarType.PyFloat, ["complex"] = NDScalarType.PyComplex,
                ["bool"] = NDScalarType.PyBool, ["str"] = NDScalarType.PyStr, ["bytes"] = NDScalarType.PyBytes,
                ["object"] = NDScalarType.PyObject, ["memoryview"] = NDScalarType.PyMemoryView,
            };
            var cases = new (NDScalarType arg1, NDScalarType arg2, bool expected)[]
            {
                (np.float64, py["float"], true), (py["float"], np.floating, true), (py["float"], np.float64, true),
                (np.str_, py["str"], true), (py["str"], np.str_, true), (py["int"], np.integer, true),
                (np.int64, py["int"], true), (py["int"], np.int64, true), (py["object"], np.generic, true),
                (np.object_, py["object"], true), (py["memoryview"], np.@void, true), (py["bool"], np.bool_, true),
                (np.bool_, py["bool"], true), (py["complex"], np.complexfloating, true), (np.complex128, py["complex"], true),
                (py["bytes"], np.bytes_, true), (np.bytes_, py["bytes"], true), (py["str"], np.character, true),
                (py["str"], py["str"], true), (py["float"], py["float"], true), (np.float32, py["float"], false),
                (py["bool"], py["int"], false), (np.bool_, py["int"], false), (py["memoryview"], py["memoryview"], true),
                (py["object"], py["object"], true), (py["float"], py["int"], false),
            };
            foreach (var (a, b, expected) in cases)
                np.issubdtype(a, b).Should().Be(expected, $"np.issubdtype({a}, {b})");
        }

        /// <summary>The dtype-less classes take part in issubdtype like any NumPy class (NumPy 2.4.2's answers).</summary>
        [TestMethod]
        public void IsSubdtype_DtypelessClasses()
        {
            np.issubdtype(np.complex64, np.complexfloating).Should().BeTrue();
            np.issubdtype(np.complex64, np.inexact).Should().BeTrue();
            np.issubdtype(np.complex64, np.complex128).Should().BeFalse();
            np.issubdtype(np.complex128, np.complex64).Should().BeFalse();
            np.issubdtype(np.object_, np.generic).Should().BeTrue();
            np.issubdtype(np.object_, np.number).Should().BeFalse();
            np.issubdtype(np.@void, np.flexible).Should().BeTrue();
            np.issubdtype(np.@void, np.character).Should().BeFalse();
            np.issubdtype(np.@void, np.@void).Should().BeTrue();
        }

        // ---- np.isdtype ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     np.isdtype compares classes by IDENTITY, not by subclass (NumPy's <c>dtype in processed_kinds</c>), and an
        ///     abstract class is in no named kind.
        /// </summary>
        [TestMethod]
        public void IsDtype_WithClasses()
        {
            np.isdtype(np.integer, np.integer).Should().BeTrue();
            np.isdtype(np.integer, np.number).Should().BeFalse();
            np.isdtype(np.int64, np.integer).Should().BeFalse();
            np.isdtype(np.integer, "integral").Should().BeFalse();
            np.isdtype((NDScalarType)np.int8, "integral").Should().BeTrue();
            np.isdtype((NDScalarType)np.int64, np.int64).Should().BeTrue();
            np.isdtype(np.floating, "integral", "real floating").Should().BeFalse();
        }

        /// <summary>The mixed tuple form: names, dtypes and classes together; every kind validated, NumPy's texts for a refused one.</summary>
        [TestMethod]
        public void IsDtype_MixedTuple()
        {
            np.isdtype(np.int64, np.integer, "integral").Should().BeTrue();
            np.isdtype(np.float32, np.integer, np.float32).Should().BeTrue();
            np.isdtype(np.floating, np.number, np.floating).Should().BeTrue();
            new Action(() => np.isdtype(np.int64, np.integer, 3)).Should().Throw<TypeError>()
                .WithMessage("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'int'>.");
            new Action(() => np.isdtype(np.int64, "integral", "bad")).Should().Throw<ValueError>()
                .WithMessage("kind argument is a string, but 'bad' is not a known kind name.");
            new Action(() => np.isdtype((NDScalarType)null, "integral")).Should().Throw<TypeError>()
                .WithMessage("dtype argument must be a NumPy dtype, but it is a <class 'NoneType'>.");
        }

        /// <summary>
        ///     np.isdtype accepts NumPy's classes only (<c>allTypes</c>): the dtype-less ones are in their named kinds
        ///     (complex64 is "complex floating" and "numeric", object_ and void in none) and identical to themselves; a
        ///     Python built-in class is refused as an argument and as a kind with NumPy's texts (it is a Python
        ///     <c>type</c>).
        /// </summary>
        [TestMethod]
        public void IsDtype_DtypelessAndBuiltinClasses()
        {
            np.isdtype(np.complex64, "complex floating").Should().BeTrue();
            np.isdtype(np.complex64, "numeric").Should().BeTrue();
            np.isdtype(np.complex64, "real floating").Should().BeFalse();
            np.isdtype(np.complex64, np.complex64).Should().BeTrue();
            np.isdtype(np.complex64, np.complex128).Should().BeFalse();
            foreach (var cls in new[] { np.object_, np.@void })
                foreach (var kind in new[] { "bool", "signed integer", "unsigned integer", "integral", "real floating", "complex floating", "numeric" })
                    np.isdtype(cls, kind).Should().BeFalse($"np.isdtype({cls}, '{kind}')");
            np.isdtype(np.float64, np.object_).Should().BeFalse();
            np.isdtype(np.float64, np.object_, "numeric").Should().BeTrue();

            foreach (var cls in Builtins)
            {
                new Action(() => np.isdtype(cls, "numeric")).Should().Throw<TypeError>()
                    .WithMessage("dtype argument must be a NumPy dtype, but it is a <class 'type'>.");
                new Action(() => np.isdtype(np.float64, cls)).Should().Throw<TypeError>()
                    .WithMessage("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'type'>.");
            }
        }

        // ---- isinstance / issubclass ---------------------------------------------------------------------------------

        /// <summary>
        ///     A C# value of a dtype's element type IS a scalar of that dtype (NumSharp has no scalar objects), so
        ///     isinstance answers for the values NumSharp hands out.
        /// </summary>
        [TestMethod]
        public void InstanceCheck_BoxedValues()
        {
            np.integer.__instancecheck__(5).Should().BeTrue();
            np.signedinteger.__instancecheck__((sbyte)5).Should().BeTrue();
            np.unsignedinteger.__instancecheck__((byte)5).Should().BeTrue();
            np.floating.__instancecheck__(1.5).Should().BeTrue();
            np.floating.__instancecheck__((Half)1.5).Should().BeTrue();
            np.complexfloating.__instancecheck__(new Complex(1, 2)).Should().BeTrue();
            np.number.__instancecheck__(true).Should().BeFalse();
            np.generic.__instancecheck__(true).Should().BeTrue();
            np.floating.__instancecheck__(1.5m).Should().BeTrue();   // NumSharp's decimal class is floating
            np.integer.__instancecheck__(DayOfWeek.Monday).Should().BeFalse();   // no dtype stores an enum
            np.integer.__instancecheck__(null).Should().BeFalse();
            np.integer.__instancecheck__(new object()).Should().BeFalse();
            // The string dtype stores System.String, so by the same rule a C# string is a numpy.str_ scalar (NumPy's
            // Python str is not an np.str_: the documented divergence, as a C# double against Python's float). A
            // change to what the string dtype stores (the strings branch) must decide this again.
            np.character.__instancecheck__("abc").Should().BeTrue();
            np.number.__instancecheck__("abc").Should().BeFalse();
        }

        /// <summary>
        ///     A 0-d array that stands for a NumPy scalar (a computed result, <c>a[0]</c>) is an instance; an explicit
        ///     <c>np.array(5)</c> is a 0-d ndarray and is not (NumPy: <c>isinstance(np.array(5), np.integer)</c> is False),
        ///     nor is an array with dimensions.
        /// </summary>
        [TestMethod]
        public void InstanceCheck_Arrays()
        {
            using var a = np.arange(3);
            using var sum = a.sum();
            using var first = a[0];
            using var explicitArray = np.array(5);
            using var seven = NDArray.Scalar(7);
            np.integer.__instancecheck__(sum).Should().BeTrue();
            np.integer.__instancecheck__(first).Should().BeTrue();
            np.floating.__instancecheck__(first).Should().BeFalse();
            np.integer.__instancecheck__(seven).Should().BeTrue();
            np.integer.__instancecheck__(explicitArray).Should().BeFalse();
            np.integer.__instancecheck__(a).Should().BeFalse();
        }

        /// <summary>
        ///     A class or a descriptor is not an INSTANCE of a scalar class (NumPy: <c>isinstance(np.int64, np.integer)</c> is
        ///     False) — but it is an <c>object</c>, as every Python value is.
        /// </summary>
        [TestMethod]
        public void InstanceCheck_ClassesAreNotInstances()
        {
            np.floating.__instancecheck__(np.float64).Should().BeFalse();
            np.generic.__instancecheck__(np.integer).Should().BeFalse();
            np.generic.__instancecheck__(typeof(double)).Should().BeFalse();
            NDScalarType.PyFloat.__instancecheck__(np.float64).Should().BeFalse();
            NDScalarType.PyObject.__instancecheck__(np.float64).Should().BeTrue();
            NDScalarType.PyObject.__instancecheck__(np.integer).Should().BeTrue();
        }

        /// <summary>
        ///     The Python classes' isinstance: a C# value is the Python object it stands for in NumSharp's NEP 50 rules (an
        ///     integer an <c>int</c>, a <see cref="bool"/> a <c>bool</c> and so an <c>int</c>, a <see cref="double"/> /
        ///     <see cref="float"/> a <c>float</c>, a <see cref="string"/> a <c>str</c>, a <c>byte[]</c> a <c>bytes</c>, an
        ///     <see cref="np.MemoryView"/> a <c>memoryview</c>); a <see cref="Half"/> or <see cref="decimal"/> is a NumPy
        ///     scalar only; every non-null value is an <c>object</c>.
        /// </summary>
        [TestMethod]
        public void InstanceCheck_PythonClasses_BoxedValues()
        {
            NDScalarType.PyFloat.__instancecheck__(1.5).Should().BeTrue();
            NDScalarType.PyFloat.__instancecheck__(1.5f).Should().BeTrue();
            NDScalarType.PyFloat.__instancecheck__(5).Should().BeFalse();
            NDScalarType.PyFloat.__instancecheck__((Half)1.5).Should().BeFalse();
            NDScalarType.PyFloat.__instancecheck__(1.5m).Should().BeFalse();
            NDScalarType.PyInt.__instancecheck__(5).Should().BeTrue();
            NDScalarType.PyInt.__instancecheck__(5UL).Should().BeTrue();
            NDScalarType.PyInt.__instancecheck__(BigInteger.Pow(2, 70)).Should().BeTrue();
            NDScalarType.PyInt.__instancecheck__(true).Should().BeTrue();
            NDScalarType.PyInt.__instancecheck__(DayOfWeek.Monday).Should().BeFalse();
            NDScalarType.PyBool.__instancecheck__(true).Should().BeTrue();
            NDScalarType.PyBool.__instancecheck__(1).Should().BeFalse();
            NDScalarType.PyComplex.__instancecheck__(new Complex(1, 2)).Should().BeTrue();
            NDScalarType.PyStr.__instancecheck__("a").Should().BeTrue();
            NDScalarType.PyBytes.__instancecheck__(new byte[] { 1 }).Should().BeTrue();
            NDScalarType.PyBytes.__instancecheck__("a").Should().BeFalse();
            using (var a = np.arange(3))
                NDScalarType.PyMemoryView.__instancecheck__(a.data).Should().BeTrue();
            NDScalarType.PyObject.__instancecheck__(new object()).Should().BeTrue();
            NDScalarType.PyObject.__instancecheck__(DayOfWeek.Monday).Should().BeTrue();
            NDScalarType.PyObject.__instancecheck__(null).Should().BeFalse();
        }

        /// <summary>
        ///     The Python classes' isinstance over NumPy scalars (0-d arrays that stand for one) follows NumPy's MROs: a
        ///     float64 scalar is a <c>float</c> (<c>isinstance(np.float64(1.5), float)</c>) and a complex128 one a
        ///     <c>complex</c>; float32, int64 and bool scalars are not Python's (<c>isinstance(np.int64(1), int)</c> is
        ///     False); an explicit <c>np.array(1.5)</c> is an ndarray, so only an <c>object</c>.
        /// </summary>
        [TestMethod]
        public void InstanceCheck_PythonClasses_NumPyScalars()
        {
            using var f64 = np.arange(3.0);
            using var f32 = f64.astype(np.float32);
            using var i64 = np.arange(3L);
            using var c128 = f64.astype(np.complex128);
            using var s64 = f64.sum();
            using var s32 = f32.sum();
            using var si = i64.sum();
            using var sc = c128.sum();
            using var positive = i64 > 0;
            // a[i] of a 1-D bool array through the untyped indexer: the 0-d array that stands for a numpy.bool scalar (the
            // typed NDArray<bool> indexer returns the C# bool itself)
            using var b = ((NDArray)positive)[1];
            using var explicitArray = np.array(1.5);
            NDScalarType.PyFloat.__instancecheck__(s64).Should().BeTrue();
            NDScalarType.PyFloat.__instancecheck__(s32).Should().BeFalse();
            NDScalarType.PyInt.__instancecheck__(si).Should().BeFalse();
            NDScalarType.PyComplex.__instancecheck__(sc).Should().BeTrue();
            NDScalarType.PyBool.__instancecheck__(b).Should().BeFalse();
            NDScalarType.PyInt.__instancecheck__(b).Should().BeFalse();
            NDScalarType.PyFloat.__instancecheck__(explicitArray).Should().BeFalse();
            NDScalarType.PyObject.__instancecheck__(explicitArray).Should().BeTrue();
            NDScalarType.PyObject.__instancecheck__(f64).Should().BeTrue();
        }

        /// <summary>
        ///     issubclass over a class, a descriptor (its class) or a C# element type (NumSharp's <c>dtype.type</c>):
        ///     <c>np.number.__subclasscheck__(a.dtype.type)</c> ports <c>issubclass(a.dtype.type, np.number)</c>.
        /// </summary>
        [TestMethod]
        public void SubclassCheck()
        {
            np.number.__subclasscheck__(np.integer).Should().BeTrue();
            np.integer.__subclasscheck__(np.number).Should().BeFalse();
            np.number.__subclasscheck__(np.float32).Should().BeTrue();
            np.signedinteger.__subclasscheck__(np.dtype("m8[s]")).Should().BeTrue();
            np.integer.__subclasscheck__(typeof(int)).Should().BeTrue();
            np.integer.__subclasscheck__(typeof(DayOfWeek)).Should().BeFalse();
            np.generic.__subclasscheck__(typeof(object)).Should().BeFalse();
            new Action(() => np.integer.__subclasscheck__((NDScalarType)null)).Should().Throw<TypeError>()
                .WithMessage("issubclass() arg 1 must be a class");
            new Action(() => np.integer.__subclasscheck__((Type)null)).Should().Throw<TypeError>();
        }

        /// <summary>
        ///     issubclass with the Python classes follows the MROs (NumPy 2.4.2 / CPython's answers): float64 is a
        ///     <c>float</c>, float32 is not; no NumPy integer is an <c>int</c>; Python's <c>bool</c> is an <c>int</c> while
        ///     numpy.bool is not; every class is an <c>object</c>, a C# type included. A C# type also stands for the
        ///     Python class of its values (<c>typeof(long)</c> is an <c>int</c>).
        /// </summary>
        [TestMethod]
        public void SubclassCheck_PythonClasses()
        {
            NDScalarType.PyFloat.__subclasscheck__(np.float64).Should().BeTrue();
            NDScalarType.PyFloat.__subclasscheck__(np.float32).Should().BeFalse();
            NDScalarType.PyComplex.__subclasscheck__(np.complex128).Should().BeTrue();
            NDScalarType.PyComplex.__subclasscheck__(np.complex64).Should().BeFalse();
            NDScalarType.PyInt.__subclasscheck__(np.int64).Should().BeFalse();
            NDScalarType.PyInt.__subclasscheck__(NDScalarType.PyBool).Should().BeTrue();
            NDScalarType.PyInt.__subclasscheck__(np.bool_).Should().BeFalse();
            NDScalarType.PyStr.__subclasscheck__(np.str_).Should().BeTrue();
            NDScalarType.PyBytes.__subclasscheck__(np.bytes_).Should().BeTrue();
            NDScalarType.PyObject.__subclasscheck__(np.object_).Should().BeTrue();
            NDScalarType.PyObject.__subclasscheck__(np.generic).Should().BeTrue();
            np.generic.__subclasscheck__(NDScalarType.PyFloat).Should().BeFalse();
            np.generic.__subclasscheck__(NDScalarType.PyObject).Should().BeFalse();
            np.floating.__subclasscheck__(NDScalarType.PyFloat).Should().BeFalse();

            NDScalarType.PyInt.__subclasscheck__(typeof(long)).Should().BeTrue();
            NDScalarType.PyInt.__subclasscheck__(typeof(bool)).Should().BeTrue();
            NDScalarType.PyFloat.__subclasscheck__(typeof(double)).Should().BeTrue();
            NDScalarType.PyFloat.__subclasscheck__(typeof(Half)).Should().BeFalse();
            NDScalarType.PyBool.__subclasscheck__(typeof(int)).Should().BeFalse();
            NDScalarType.PyObject.__subclasscheck__(typeof(System.IO.Stream)).Should().BeTrue();
        }

        // ---- the relatives ---------------------------------------------------------------------------------------------

        /// <summary>np.typecodes holds NumPy's strings in NumPy's order, read-only.</summary>
        [TestMethod]
        public void TypeCodes_AreNumPys()
        {
            np.typecodes.Keys.Should().Equal("Character", "Integer", "UnsignedInteger", "Float", "Complex", "AllInteger",
                                             "AllFloat", "Datetime", "All");
            np.typecodes["AllFloat"].Should().Be("efdgFDG");
            np.typecodes["All"].Should().Be("?bhilqnpBHILQNPefdgFDGSUVOMm");
            np.typecodes["AllFloat"].Should().Contain(np.float32.@char.ToString());
            new Action(() => ((IDictionary<string, string>)np.typecodes).Add("x", "y")).Should().Throw<NotSupportedException>();
            ReferenceEquals(np.typecodes, np.typecodes).Should().BeTrue();
        }

        /// <summary>
        ///     np.sctypeDict maps NumPy's names to NumSharp's spelling of the class (the descriptor itself), in NumPy's key
        ///     order; collapsed classes map to the class they collapse onto; absent classes have no key.
        /// </summary>
        [TestMethod]
        public void SctypeDict_MapsNamesToTheDescriptors()
        {
            ReferenceEquals(np.sctypeDict["float64"], np.float64).Should().BeTrue();
            ReferenceEquals(np.sctypeDict["float"], np.float64).Should().BeTrue();
            ReferenceEquals(np.sctypeDict["longdouble"], np.float64).Should().BeTrue();
            ReferenceEquals(np.sctypeDict["intc"], np.int32).Should().BeTrue();
            ReferenceEquals(np.sctypeDict["datetime64"], np.dtype("M")).Should().BeTrue();
            ReferenceEquals(np.sctypeDict["long"], np.@long).Should().BeTrue();
            np.sctypeDict["int"].Should().Be(np.intp);
            np.sctypeDict.ContainsKey("complex64").Should().BeFalse();
            np.sctypeDict.ContainsKey("object").Should().BeFalse();
            np.sctypeDict.Keys.Take(5).Should().Equal("bool", "float16", "float32", "float64", "longdouble");
            np.sctypeDict.Keys.Skip(7).Take(3).Should().Equal("bytes_", "str_", "datetime64");
            np.sctypeDict.Keys.TakeLast(5).Should().Equal("bytes", "a", "int", "str", "unicode");
            foreach (var name in new[] { "bytes_", "bytes", "a" })
                ReferenceEquals(np.sctypeDict[name], np.bytes_).Should().BeTrue(name);
            foreach (var name in new[] { "str_", "str", "unicode" })
                ReferenceEquals(np.sctypeDict[name], np.str_).Should().BeTrue(name);
            foreach (var value in np.sctypeDict.Values)
                np.issubdtype(value, np.generic).Should().BeTrue();
        }

        /// <summary>np.True_ / np.False_ are NumSharp's bool scalars (C# bools); np.longdouble is the float64 descriptor.</summary>
        [TestMethod]
        public void BoolScalars_AndLongDouble()
        {
            np.True_.Should().BeTrue();
            np.False_.Should().BeFalse();
            ReferenceEquals(np.longdouble, np.float64).Should().BeTrue();
            np.issubdtype(np.longdouble, np.floating).Should().BeTrue();
        }

        /// <summary>
        ///     np.typing names the C# types that play NumPy's typing aliases: <c>NDArray[T]</c> is the generic array,
        ///     ArrayLike the array_like parameter type, DTypeLike the dtype parameter type; NBitBase is the deprecated
        ///     marker class.
        /// </summary>
        [TestMethod]
        public void Typing_NamesTheCSharpTypes()
        {
            np.typing.NDArray.Should().Be(typeof(NDArray<>));
            np.typing.NDArray.MakeGenericType(typeof(double)).Should().Be(typeof(NDArray<double>));
            np.typing.ArrayLike.Should().Be(typeof(NDArray));
            np.typing.DTypeLike.Should().Be(typeof(DType));
#pragma warning disable CS0618 // NBitBase is deprecated by NumPy 2.3; the test reads the deprecated class on purpose
            var nbit = typeof(np.typing.NBitBase);
#pragma warning restore CS0618
            nbit.IsAbstract.Should().BeTrue();
            nbit.GetCustomAttribute<ObsoleteAttribute>().Message.Should().Contain("deprecated in NumPy 2.3");
            nbit.GetConstructors(BindingFlags.Public | BindingFlags.Instance).Should().BeEmpty();
        }
    }
}
