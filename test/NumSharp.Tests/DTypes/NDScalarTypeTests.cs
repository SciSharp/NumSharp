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
    ///     <c>np.integer</c>, ...), its integration in <c>np.issubdtype</c> / <c>np.isdtype</c> / <c>np.dtype</c> /
    ///     <c>np.finfo</c> / <c>np.iinfo</c>, and the relatives <c>np.typecodes</c>, <c>np.sctypeDict</c>,
    ///     <c>np.True_</c> / <c>np.False_</c>, <c>np.longdouble</c> and <c>np.typing</c>. Every NumPy value was probed against
    ///     <c>numpy==2.4.2</c> (win-amd64); NumSharp's own modelling choices (C# values are scalars, the collapsed and
    ///     NumSharp-only classes) are named where they apply.
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

        /// <summary>The names of a class list, for compact assertions.</summary>
        /// <param name="classes">The classes.</param>
        /// <returns>Their <c>__name__</c>s, in order.</returns>
        private static string[] Names(IEnumerable<NDScalarType> classes) => classes.Select(c => c.__name__).ToArray();

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
        ///     The MRO and the bases end at <c>np.generic</c> (NumPy's also list Python's <c>object</c>, which is not a
        ///     NumSharp object; <c>np.generic.__base__</c> is therefore null where NumPy's is <c>object</c>).
        /// </summary>
        [TestMethod]
        public void AbstractClasses_MroAndBases()
        {
            Names(np.generic.__mro__).Should().Equal("generic");
            Names(np.integer.__mro__).Should().Equal("integer", "number", "generic");
            Names(np.signedinteger.__mro__).Should().Equal("signedinteger", "integer", "number", "generic");
            Names(np.complexfloating.__mro__).Should().Equal("complexfloating", "inexact", "number", "generic");
            Names(np.character.__mro__).Should().Equal("character", "flexible", "generic");
            np.generic.__bases__.Should().BeEmpty();
            np.generic.__base__.Should().BeNull();
            Names(np.floating.__bases__).Should().Equal("inexact");
        }

        /// <summary><c>mro()</c> is a NEW mutable list each call, as Python's is; <c>__mro__</c> is read-only.</summary>
        [TestMethod]
        public void Mro_ReturnsAFreshList()
        {
            var first = np.integer.mro();
            first.Add(np.generic);
            Names(np.integer.mro()).Should().Equal("integer", "number", "generic");
            new Action(() => ((IList<NDScalarType>)np.integer.__mro__).Add(np.generic)).Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     <c>__subclasses__()</c> lists the direct subclasses in NumPy's registration order — abstract first, then
        ///     concrete. NumSharp's lists differ from NumPy's only by the classes it lacks (object_, void, bytes_, complex64),
        ///     collapses (intc, longdouble, clongdouble) or adds (decimal, char).
        /// </summary>
        [TestMethod]
        public void Subclasses_FollowNumPysRegistrationOrder()
        {
            Names(np.generic.__subclasses__()).Should().Equal("number", "flexible", "bool", "datetime64");
            Names(np.number.__subclasses__()).Should().Equal("integer", "inexact");
            Names(np.integer.__subclasses__()).Should().Equal("signedinteger", "unsignedinteger");
            Names(np.signedinteger.__subclasses__()).Should().Equal("int8", "int16", "int32", "int64", "timedelta64");
            Names(np.unsignedinteger.__subclasses__()).Should().Equal("uint8", "uint16", "uint32", "uint64", "char");
            Names(np.inexact.__subclasses__()).Should().Equal("floating", "complexfloating");
            Names(np.floating.__subclasses__()).Should().Equal("float16", "float32", "float64", "decimal");
            Names(np.complexfloating.__subclasses__()).Should().Equal("complex128");
            Names(np.flexible.__subclasses__()).Should().Equal("character");
            ((NDScalarType)np.float64).__subclasses__().Should().BeEmpty();
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
            Names(f8.__mro__).Should().Equal("float64", "floating", "inexact", "number", "generic");
            Names(((NDScalarType)np.dtype("m8[s]")).__mro__).Should().Equal("timedelta64", "signedinteger", "integer", "number", "generic");
            Names(((NDScalarType)np.bool_).__mro__).Should().Equal("bool", "generic");
            ((NDScalarType)np.@decimal).ToString().Should().Be("<class 'numsharp.decimal'>");
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
            np.dtype((NDScalarType)null).Should().BeSameAs(DType.Double, "a null class is NumPy's None, float64");
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
            new Action(() => np.finfo((NDScalarType)np.int32)).Should().Throw<ArgumentException>();
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

        /// <summary>A class or a descriptor is not an INSTANCE of a scalar class (NumPy: <c>isinstance(np.int64, np.integer)</c> is False).</summary>
        [TestMethod]
        public void InstanceCheck_ClassesAreNotInstances()
        {
            np.floating.__instancecheck__(np.float64).Should().BeFalse();
            np.generic.__instancecheck__(np.integer).Should().BeFalse();
            np.generic.__instancecheck__(typeof(double)).Should().BeFalse();
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
            np.sctypeDict.Keys.Take(5).Should().Equal("bool", "float16", "float32", "float64", "longdouble");
            np.sctypeDict.Keys.Last().Should().Be("int");
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
