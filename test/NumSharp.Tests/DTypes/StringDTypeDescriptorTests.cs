using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The NEP 55 <c>StringDType</c> descriptor (strings phase 1, P1.2): the class (<see cref="VStringDTypeMeta"/>), its
    ///     parameters (<c>na_object</c>, <c>coerce</c>), the descriptor surface, equality and hash, the parser, promotion,
    ///     the cast-safety table, <c>np.isdtype</c> / <c>np.issubdtype</c>, and the CPython protocols the descriptor runs on
    ///     its missing-data object. Every expected value was probed against <c>numpy==2.4.2</c> (CPython 3.12, win-amd64).
    /// </summary>
    [TestClass]
    public class StringDTypeDescriptorTests
    {
        // ------------------------------------------------------------------------------------------------------------
        // the descriptor surface
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.dtype('T')</c>'s attributes: char / kind <c>'T'</c>, num 2056, a 16-byte cell aligned to 8, the name
        ///     <c>StringDType128</c>, str / descr / repr <c>StringDType()</c>, not builtin, native, holding references, flags
        ///     107, byte order <c>'|'</c>, scalar type <c>str</c>, storage lane <see cref="NPTypeCode.String"/>.
        /// </summary>
        [TestMethod]
        public void Descriptor_Surface_IsNumPys()
        {
            var t = np.dtype("T");
            t.@char.Should().Be('T');
            t.kind.Should().Be('T');
            t.num.Should().Be(2056);
            t.type_num.Should().Be(NPY_TYPES.NPY_VSTRING);
            t.itemsize.Should().Be(16);
            t.alignment.Should().Be(8);
            t.name.Should().Be("StringDType128");
            t.str.Should().Be("StringDType()");
            t.descr.Single().Should().Be(("", "StringDType()"));
            t.isbuiltin.Should().Be(0);
            t.isnative.Should().BeTrue();
            t.hasobject.Should().BeTrue();
            t.flags.Should().Be(107);
            t.byteorder.Should().Be('|');
            t.type.Should().Be(typeof(string));
            t.typecode.Should().Be(NPTypeCode.String);
            t.ToString().Should().Be("StringDType()");
            t.ToString(true).Should().Be("StringDType()", "the class has its own repr: no dtype(...) wrapper");
            t.Meta.Should().BeSameAs(DTypeRegistry.VString);
            t.Meta.ToString().Should().Be("<class 'numpy.dtypes.StringDType'>");
            t.metadata.Should().BeNull();
            t.fields.Should().BeNull();
            t.names.Should().BeNull();
            t.subdtype.Should().BeNull();
            t.@base.Should().BeSameAs(t);
            t.ndim.Should().Be(0);
        }

        /// <summary>
        ///     The parameters: <c>coerce</c> reads back; <c>na_object</c> does not exist without a sentinel (NumPy's
        ///     AttributeError text, naming the class), exists for <c>None</c>; a non-<c>StringDType</c> descriptor has
        ///     neither attribute.
        /// </summary>
        [TestMethod]
        public void Parameters_ReadBack_AndAbsentNAObjectIsAnAttributeError()
        {
            var t = np.dtypes.StringDType();
            t.coerce.Should().BeTrue();
            t.HasNAObject.Should().BeFalse();
            Action na = () => _ = t.na_object;
            na.Should().Throw<AttributeError>().WithMessage("'numpy.dtypes.StringDType' object has no attribute 'na_object'");

            var none = np.dtypes.StringDType(na_object: null, coerce: false);
            none.HasNAObject.Should().BeTrue();
            none.na_object.Should().BeNull("None is a sentinel of its own, unlike 'not given'");
            none.coerce.Should().BeFalse();

            np.dtypes.StringDType(double.NaN).na_object.Should().Be(double.NaN);
            np.dtypes.StringDType("NA").na_object.Should().Be("NA");
            np.dtypes.StringDType('?').na_object.Should().Be("?", "a char sentinel is Python's one-character str");

            var f8 = np.dtype("f8");
            f8.HasNAObject.Should().BeFalse();
            Action c = () => _ = f8.coerce;
            c.Should().Throw<AttributeError>().WithMessage("'numpy.dtypes.Float64DType' object has no attribute 'coerce'");
            Action n = () => _ = f8.na_object;
            n.Should().Throw<AttributeError>().WithMessage("'numpy.dtypes.Float64DType' object has no attribute 'na_object'");
        }

        /// <summary>
        ///     <c>repr()</c> / <c>str()</c> of descriptors with every kind of sentinel: Python's repr of the object (CPython's
        ///     quote choice and escapes for text, <c>b'\x80'</c>, <c>np.float16(1.5)</c>, tuples and lists) and the four
        ///     <c>coerce</c> / <c>na_object</c> combinations.
        /// </summary>
        [TestMethod]
        public void Repr_PrintsTheParametersAsNumPyDoes()
        {
            np.dtypes.StringDType().ToString().Should().Be("StringDType()");
            np.dtypes.StringDType(false).ToString().Should().Be("StringDType(coerce=False)");
            np.dtypes.StringDType(null).ToString().Should().Be("StringDType(na_object=None)");
            np.dtypes.StringDType(null, false).ToString().Should().Be("StringDType(na_object=None, coerce=False)");
            np.dtypes.StringDType("NA", false).ToString(true).Should().Be("StringDType(na_object='NA', coerce=False)");
            np.dtypes.StringDType(double.NaN).str.Should().Be("StringDType(na_object=nan)");
            np.dtypes.StringDType(-double.NaN).str.Should().Be("StringDType(na_object=nan)");
            np.dtypes.StringDType(float.NaN).str.Should().Be("StringDType(na_object=nan)", "a C# float is a Python float");
            np.dtypes.StringDType("").str.Should().Be("StringDType(na_object='')");
            np.dtypes.StringDType("None").str.Should().Be("StringDType(na_object='None')");
            np.dtypes.StringDType("it's").str.Should().Be("StringDType(na_object=\"it's\")");
            np.dtypes.StringDType("q\"z").str.Should().Be("StringDType(na_object='q\"z')");
            np.dtypes.StringDType("a\nb").str.Should().Be("StringDType(na_object='a\\nb')");
            np.dtypes.StringDType(5).str.Should().Be("StringDType(na_object=5)");
            np.dtypes.StringDType(1.5).str.Should().Be("StringDType(na_object=1.5)");
            np.dtypes.StringDType(-0.0).str.Should().Be("StringDType(na_object=-0.0)");
            np.dtypes.StringDType(double.PositiveInfinity).str.Should().Be("StringDType(na_object=inf)");
            np.dtypes.StringDType(na_object: true).str.Should().Be("StringDType(na_object=True)");
            np.dtypes.StringDType(new byte[] { 0x80 }).str.Should().Be("StringDType(na_object=b'\\x80')");
            np.dtypes.StringDType((Half)1.5).str.Should().Be("StringDType(na_object=np.float16(1.5))");
            np.dtypes.StringDType(Half.NaN).str.Should().Be("StringDType(na_object=np.float16(nan))");
            np.dtypes.StringDType(new Complex(1, 2)).str.Should().Be("StringDType(na_object=(1+2j))");
            np.dtypes.StringDType(new Complex(double.NaN, 0)).str.Should().Be("StringDType(na_object=(nan+0j))");
            np.dtypes.StringDType(BigInteger.Pow(2, 70)).str.Should().Be("StringDType(na_object=1180591620717411303424)");
            np.dtypes.StringDType(ValueTuple.Create(1)).str.Should().Be("StringDType(na_object=(1,))");
            np.dtypes.StringDType((1, 2)).str.Should().Be("StringDType(na_object=(1, 2))");
            np.dtypes.StringDType(new object[0]).str.Should().Be("StringDType(na_object=[])");
            np.dtypes.StringDType(new object[] { 1, "a" }).str.Should().Be("StringDType(na_object=[1, 'a'])");
            np.dtypes.StringDType(new System.Collections.Hashtable()).str.Should().Be("StringDType(na_object={})");
            np.dtypes.StringDType(new System.Collections.Hashtable { { "k", 2 } }).str.Should().Be("StringDType(na_object={'k': 2})");
            np.dtypes.StringDType("NA").descr.Single().typestr.Should().Be("StringDType(na_object='NA')");
            np.dtypes.StringDType("NA").name.Should().Be("StringDType128");
        }

        /// <summary>
        ///     NumPy's derived fields (<c>new_stringdtype_instance</c>): a string sentinel is a string NA whose default string
        ///     is itself; a NaN-like sentinel (<c>na != na</c> truthy: a float / float16 NaN, a complex with a NaN part, a
        ///     multi-element array) is a NaN NA; <c>None</c> and other objects are neither; <c>na_name</c> is <c>str()</c>.
        /// </summary>
        [TestMethod]
        public void Parameters_DerivedFields_FollowNewStringDTypeInstance()
        {
            var s = np.dtypes.StringDType("NA").VString;
            s.HasStringNA.Should().BeTrue();
            s.HasNanNA.Should().BeFalse();
            s.DefaultString.Should().Be("NA");
            s.NAName.Should().Be("NA");

            var nan = np.dtypes.StringDType(double.NaN).VString;
            nan.HasNanNA.Should().BeTrue();
            nan.HasStringNA.Should().BeFalse();
            nan.DefaultString.Should().BeEmpty();
            nan.NAName.Should().Be("nan");

            var none = np.dtypes.StringDType(null).VString;
            none.HasNanNA.Should().BeFalse();
            none.HasStringNA.Should().BeFalse();
            none.NAName.Should().Be("None");

            np.dtypes.StringDType(Half.NaN).VString.HasNanNA.Should().BeTrue();
            np.dtypes.StringDType(new Complex(1, double.NaN)).VString.HasNanNA.Should().BeTrue();
            np.dtypes.StringDType(5).VString.HasNanNA.Should().BeFalse();
            np.dtypes.StringDType(5).VString.NAName.Should().Be("5");
            np.dtypes.StringDType((Half)1.5).VString.NAName.Should().Be("1.5");
            np.dtypes.StringDType(new byte[] { (byte)'x' }).VString.NAName.Should().Be("b'x'");
            np.dtypes.StringDType(na_object: true).VString.NAName.Should().Be("True");
            np.dtypes.StringDType('?').VString.HasStringNA.Should().BeTrue();

            var t = np.dtypes.StringDType().VString;
            t.HasNAObject.Should().BeFalse();
            t.NAName.Should().BeEmpty();
            t.DefaultString.Should().BeEmpty();
            np.dtype("f8").VString.Should().BeNull();
        }

        /// <summary>
        ///     NumPy keeps a string sentinel (<c>default_string</c>) and every sentinel's <c>str()</c> (<c>na_name</c>) as
        ///     UTF-8, so a lone surrogate in either is CPython's encode error when the descriptor is built; a surrogate that
        ///     only appears ESCAPED in the <c>str()</c> (a list's repr) is fine, and a valid pair is one character.
        /// </summary>
        [TestMethod]
        public void NAObject_LoneSurrogate_IsAnEncodeError()
        {
            Action a = () => np.dtypes.StringDType("\ud800");
            a.Should().Throw<UnicodeEncodeError>().WithMessage("'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed");
            Action b = () => np.dtypes.StringDType("a\udc00b", false);
            b.Should().Throw<UnicodeEncodeError>().WithMessage("'utf-8' codec can't encode character '\\udc00' in position 1: surrogates not allowed");
            Action c = () => np.dtypes.StringDType('\udfff');
            c.Should().Throw<UnicodeEncodeError>().WithMessage("'utf-8' codec can't encode character '\\udfff' in position 0: surrogates not allowed");
            Action d = () => np.dtypes.StringDType(new SurrogateText());
            d.Should().Throw<UnicodeEncodeError>().WithMessage("'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed");
            np.dtypes.StringDType(new object[] { "\ud800" }).ToString().Should().Be("StringDType(na_object=['\\ud800'])");
            np.dtypes.StringDType("😀").ToString().Should().Be("StringDType(na_object='😀')");
            np.dtypes.StringDType("😀").VString.DefaultString.Should().Be("😀");
        }

        /// <summary>An "other" sentinel whose <c>str()</c> holds a lone surrogate (Python: <c>__str__</c> returning <c>'x\ud800'</c>).</summary>
        private sealed class SurrogateText
        {
            /// <inheritdoc/>
            public override string ToString() => "x\ud800";
        }

        // ------------------------------------------------------------------------------------------------------------
        // equality and hash
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>stringdtype_richcompare</c>: equal when <c>coerce</c> matches and the sentinels are the same missing value —
        ///     NaN equals NaN (Python floats), <c>5 == 5.0</c>, <c>True == 1</c>, <c>0.0 == -0.0</c>; not equal to
        ///     <c>None</c> vs "not given", different coerce, different strings, bytes vs str, two float16 NaN objects.
        /// </summary>
        [TestMethod]
        public void Equality_FollowsNaEqCmp()
        {
            np.dtypes.StringDType().Should().Be(np.dtype("T"));
            (np.dtypes.StringDType() == np.dtypes.StringDType()).Should().BeTrue();
            ReferenceEquals(np.dtype("T"), np.dtype("T")).Should().BeFalse("every call builds a fresh descriptor");
            (np.dtypes.StringDType(double.NaN) == np.dtypes.StringDType(double.NaN)).Should().BeTrue();
            (np.dtypes.StringDType(double.NaN) == np.dtypes.StringDType(float.NaN)).Should().BeTrue();
            (np.dtypes.StringDType(null) == np.dtypes.StringDType(null)).Should().BeTrue();
            (np.dtypes.StringDType(null) == np.dtypes.StringDType()).Should().BeFalse();
            (np.dtypes.StringDType(false) == np.dtypes.StringDType()).Should().BeFalse();
            (np.dtypes.StringDType(false) != np.dtypes.StringDType()).Should().BeTrue();
            (np.dtypes.StringDType("NA") == np.dtypes.StringDType("NA")).Should().BeTrue();
            (np.dtypes.StringDType("NA") == np.dtypes.StringDType("NB")).Should().BeFalse();
            (np.dtypes.StringDType("x") == np.dtypes.StringDType('x')).Should().BeTrue();
            (np.dtypes.StringDType(5) == np.dtypes.StringDType(5.0)).Should().BeTrue();
            (np.dtypes.StringDType(na_object: true) == np.dtypes.StringDType(1)).Should().BeTrue();
            (np.dtypes.StringDType(new Complex(1, 0)) == np.dtypes.StringDType(1)).Should().BeTrue();
            (np.dtypes.StringDType(0.0) == np.dtypes.StringDType(-0.0)).Should().BeTrue();
            (np.dtypes.StringDType(double.PositiveInfinity) == np.dtypes.StringDType(double.PositiveInfinity)).Should().BeTrue();
            (np.dtypes.StringDType(BigInteger.Pow(2, 70)) == np.dtypes.StringDType(BigInteger.Pow(2, 70))).Should().BeTrue();
            (np.dtypes.StringDType((Half)1.5) == np.dtypes.StringDType(1.5)).Should().BeTrue();
            (np.dtypes.StringDType(new byte[] { (byte)'x' }) == np.dtypes.StringDType("x")).Should().BeFalse();
            (np.dtypes.StringDType(new byte[] { (byte)'x' }) == np.dtypes.StringDType(new byte[] { (byte)'x' })).Should().BeTrue();
            (np.dtypes.StringDType(Half.NaN) == np.dtypes.StringDType(Half.NaN)).Should().BeFalse("np.float16 is not a Python float");
            var same = np.dtypes.StringDType(Half.NaN);
            (same == same).Should().BeTrue("a descriptor is equal to itself");
            (np.dtypes.StringDType(1L << 53) == np.dtypes.StringDType((double)(1L << 53) + 1)).Should().BeTrue("2**53 + 1 rounds to 2**53 as a double");
            (np.dtypes.StringDType((1L << 53) + 1) == np.dtypes.StringDType((double)(1L << 53))).Should().BeFalse("Python compares an int and a float exactly");
        }

        /// <summary>
        ///     A <c>StringDType</c> descriptor does not coerce a dtype spelling in <c>Equals</c> (its <c>__eq__</c> returns
        ///     NotImplemented): <c>T == 'T'</c>, <c>T == str</c>, <c>T == 'U'</c> are False, and a <c>U</c> descriptor is not
        ///     equal to it either.
        /// </summary>
        [TestMethod]
        public void Equality_DoesNotCoerceSpellings()
        {
            var t = np.dtype("T");
            t.Equals("T").Should().BeFalse();
            t.Equals((object)"T").Should().BeFalse();
            t.Equals(typeof(string)).Should().BeFalse();
            t.Equals(NPTypeCode.String).Should().BeFalse();
            t.Equals("U").Should().BeFalse();
            np.dtype("U5").Equals(t).Should().BeFalse();
            np.dtype("U5").Equals((object)"T").Should().BeFalse();
            t.Equals((object)np.dtypes.StringDType()).Should().BeTrue();
            t.Equals((DType)null).Should().BeFalse();
            t.Equals((object)null).Should().BeFalse();
        }

        /// <summary>The hash agrees with equality: equal descriptors (NaN sentinels, 5 and 5.0, char and string) hash alike.</summary>
        [TestMethod]
        public void Hash_IsConsistentWithEquality()
        {
            np.dtypes.StringDType().GetHashCode().Should().Be(np.dtype("T").GetHashCode());
            np.dtypes.StringDType(double.NaN).GetHashCode().Should().Be(np.dtypes.StringDType(float.NaN).GetHashCode());
            np.dtypes.StringDType(5).GetHashCode().Should().Be(np.dtypes.StringDType(5.0).GetHashCode());
            np.dtypes.StringDType(na_object: true).GetHashCode().Should().Be(np.dtypes.StringDType(1).GetHashCode());
            np.dtypes.StringDType('x').GetHashCode().Should().Be(np.dtypes.StringDType("x").GetHashCode());
            np.dtypes.StringDType(null).GetHashCode().Should().Be(np.dtypes.StringDType(null).GetHashCode());
            np.dtypes.StringDType(new byte[] { 1, 2 }).GetHashCode().Should().Be(np.dtypes.StringDType(new byte[] { 1, 2 }).GetHashCode());
            np.dtypes.StringDType(new Complex(3, 0)).GetHashCode().Should().Be(np.dtypes.StringDType(3).GetHashCode());
            var set = new System.Collections.Generic.HashSet<DType> { np.dtypes.StringDType(), np.dtypes.StringDType(), np.dtypes.StringDType(false) };
            set.Count.Should().Be(2);
        }

        // ------------------------------------------------------------------------------------------------------------
        // the parser, the factories, the byte order
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.dtype</c>: <c>'T'</c> with any byte-order prefix is <c>StringDType()</c>; <c>'T8'</c>, <c>'T16'</c>,
        ///     <c>'StringDType'</c>, <c>'t'</c>, <c>'TT'</c> are "not understood".
        /// </summary>
        [TestMethod]
        public void Parser_ReadsT()
        {
            foreach (var spec in new[] { "T", "<T", ">T", "=T", "|T" })
            {
                var d = np.dtype(spec);
                d.Should().Be(np.dtypes.StringDType(), spec);
                d.byteorder.Should().Be('|');
            }
            foreach (var spec in new[] { "T8", "T16", "StringDType", "StringDType()", "t", "TT", "T()", "T(2)", "(2)T", "|T(2)" })
            {
                Action a = () => np.dtype(spec);
                a.Should().Throw<NotSupportedException>().WithMessage($"data type '{spec}' not understood", spec);
            }
            ((DType)"T").Should().Be(np.dtypes.StringDType(), "the implicit string conversion parses too");
            np.dtype(np.dtype("T")).Should().Be(np.dtypes.StringDType());
            var own = np.dtypes.StringDType(null);
            np.dtype(own).Should().BeSameAs(own, "np.dtype(descr) is descr");
        }

        /// <summary>
        ///     The factories (decision D5): every call is a fresh descriptor; the bool overload is <c>coerce</c>, the object
        ///     overload is <c>na_object</c> (<see langword="null"/> is <c>None</c>, a bool sentinel goes by name); the class
        ///     call <see cref="DTypeMeta.Instantiate()"/> and the default descriptor are <c>StringDType()</c>.
        /// </summary>
        [TestMethod]
        public void Factories_BuildFreshDescriptors()
        {
            ReferenceEquals(np.dtypes.StringDType(), np.dtypes.StringDType()).Should().BeFalse();
            np.dtypes.StringDType(true).Should().Be(np.dtypes.StringDType());
            np.dtypes.StringDType(true).HasNAObject.Should().BeFalse("a bool positional argument is coerce");
            np.dtypes.StringDType(na_object: false).na_object.Should().Be(false);
            np.dtypes.StringDType(coerce: false).coerce.Should().BeFalse();
            DTypeRegistry.VString.Instantiate().Should().Be(np.dtypes.StringDType());
            DTypeRegistry.VString.DefaultDescr().Should().Be(np.dtypes.StringDType());
            ReferenceEquals(DTypeRegistry.VString.DefaultDescr(), DTypeRegistry.VString.DefaultDescr()).Should().BeFalse();
            DTypeRegistry.VString.Singleton.Should().BeNull("the class has no builtin instance");
            DTypeRegistry.FromName("StringDType").Should().BeSameAs(DTypeRegistry.VString);
            DTypeRegistry.FromTypeNum(2056).Should().BeSameAs(DTypeRegistry.VString);
            DTypeRegistry.FromTypeCode(NPTypeCode.String).Should().BeSameAs(DTypeRegistry.Str, "the lane belongs to StrDType");
            DTypeRegistry.FromScalarType(typeof(string)).Should().BeSameAs(DTypeRegistry.Str, "the C# string stays <U");
            DTypeRegistry.All.Should().Contain(DTypeRegistry.VString);
        }

        /// <summary>
        ///     <c>newbyteorder</c> refuses every new-style descriptor (after parsing the order, so a bad code is still the
        ///     ValueError); the canonical form of a <c>StringDType</c> descriptor is the descriptor itself.
        /// </summary>
        [TestMethod]
        public void ByteOrder_NewByteOrderRefuses_EnsureCanonicalIsIdentity()
        {
            var t = np.dtype("T");
            Action swap = () => t.newbyteorder();
            swap.Should().Throw<TypeError>().WithMessage("Cannot use DescrNewByteOrder for this new style DTypes.");
            Action native = () => t.newbyteorder("=");
            native.Should().Throw<TypeError>().WithMessage("Cannot use DescrNewByteOrder for this new style DTypes.");
            Action bad = () => t.newbyteorder("x");
            bad.Should().Throw<ValueError>().WithMessage("byteorder not recognized (got 'x')");
            DTypeRegistry.VString.EnsureCanonical(t).Should().BeSameAs(t);
            Action nul = () => DTypeRegistry.VString.EnsureCanonical(null);
            nul.Should().Throw<ArgumentNullException>();
            t.WithByteOrder('>').Should().Be(t, "an internal byte-order copy keeps the parameters and the '|' order");
            t.WithByteOrder('>').byteorder.Should().Be('|');
        }

        /// <summary>
        ///     A <c>StringDType</c> descriptor cannot be built without its parameters, and parameters cannot ride on another
        ///     class (the constructor guards both, so no descriptor can disagree with its class).
        /// </summary>
        [TestMethod]
        public void Constructor_ParametersBelongToStringDTypeOnly()
        {
            Action noParams = () => new DType(DTypeRegistry.VString, '|', null);
            noParams.Should().Throw<ArgumentException>();
            Action foreign = () => new DType(DTypeRegistry.Double, '=', null, 8, 'd', VStringParameters.Default);
            foreign.Should().Throw<ArgumentException>();
            Action nullParams = () => new DType(DTypeRegistry.VString, (VStringParameters)null);
            nullParams.Should().Throw<ArgumentNullException>();
        }

        // ------------------------------------------------------------------------------------------------------------
        // isdtype / issubdtype
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.isdtype</c>: a <c>StringDType</c> dtype argument is refused first (before any kind, even an unknown one or
        ///     an empty tuple); a <c>StringDType</c> kind is refused with the kind text.
        /// </summary>
        [TestMethod]
        public void IsDtype_RefusesStringDType()
        {
            var t = np.dtype("T");
            const string dtypeText = "dtype argument must be a NumPy dtype, but it is a <class 'numpy.dtypes.StringDType'>.";
            const string kindText = "kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'numpy.dtypes.StringDType'>.";
            Action a1 = () => np.isdtype(t, "numeric");
            a1.Should().Throw<TypeError>().WithMessage(dtypeText);
            Action a2 = () => np.isdtype(t, "foo");
            a2.Should().Throw<TypeError>().WithMessage(dtypeText);
            Action a3 = () => np.isdtype(t, np.float64);
            a3.Should().Throw<TypeError>().WithMessage(dtypeText);
            Action a4 = () => np.isdtype(t, new string[0]);
            a4.Should().Throw<TypeError>().WithMessage(dtypeText);
            Action a5 = () => np.isdtype(t, new DType[0]);
            a5.Should().Throw<TypeError>().WithMessage(dtypeText);
            Action a6 = () => np.isdtype(np.float64, t);
            a6.Should().Throw<TypeError>().WithMessage(kindText);
            Action a7 = () => np.isdtype(np.float64, new[] { np.float64, t });
            a7.Should().Throw<TypeError>().WithMessage(kindText);
            np.isdtype(np.dtype("U5"), "numeric").Should().BeFalse();
        }

        /// <summary>
        ///     <c>np.issubdtype</c>: <c>StringDType</c>'s scalar type is Python's <c>str</c>, the base class of
        ///     <c>np.str_</c> — so <c>U</c> is a subtype of <c>T</c> but not the reverse; <c>T</c> is under no category.
        /// </summary>
        [TestMethod]
        public void IsSubDtype_FollowsPythonsStrHierarchy()
        {
            var t = np.dtype("T");
            np.issubdtype(np.dtype("U5"), t).Should().BeTrue();
            np.issubdtype(t, np.dtype("U5")).Should().BeFalse();
            np.issubdtype(np.dtype("S5"), t).Should().BeFalse();
            np.issubdtype(t, np.dtype("S5")).Should().BeFalse();
            np.issubdtype(t, t).Should().BeTrue();
            np.issubdtype(np.dtypes.StringDType(null), t).Should().BeTrue();
            np.issubdtype(t, np.dtypes.StringDType(false)).Should().BeTrue();
            np.issubdtype(t, "T").Should().BeTrue();
            np.issubdtype("T", "T").Should().BeTrue();
            np.issubdtype("U5", "T").Should().BeTrue();
            np.issubdtype(t, np.str_).Should().BeFalse();
            np.issubdtype(np.dtype("i8"), t).Should().BeFalse();
            np.issubdtype(t, np.dtype("i8")).Should().BeFalse();
            foreach (var category in new[] { "character", "generic", "flexible", "number", "integer", "inexact", "bool" })
                np.issubdtype(t, category).Should().BeFalse(category);
        }

        // ------------------------------------------------------------------------------------------------------------
        // promotion
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The common DType: <c>T</c> with <c>U</c> (either order) is <c>T</c>; with <c>S</c>, a number, bool, complex
        ///     or a datetime there is none (NumPy's DTypePromotionError text, both operand orders).
        /// </summary>
        [TestMethod]
        public void PromoteTypes_TAbsorbsUOnly()
        {
            var t = np.dtype("T");
            np.promote_types(t, np.dtype("U5")).Should().Be(np.dtypes.StringDType());
            np.promote_types(np.dtype("U5"), t).Should().Be(np.dtypes.StringDType());
            np.promote_types(t, t).Should().Be(np.dtypes.StringDType());
            var classes = new (string spec, string cls)[]
            {
                ("S5", "BytesDType"), ("i8", "Int64DType"), ("f8", "Float64DType"), ("?", "BoolDType"),
                ("c16", "Complex128DType"), ("M8[s]", "DateTime64DType"), ("m8[s]", "TimeDelta64DType"),
            };
            foreach (var (spec, cls) in classes)
            {
                Action a = () => np.promote_types(t, np.dtype(spec));
                a.Should().Throw<DTypePromotionError>().WithMessage(
                    $"The DTypes <class 'numpy.dtypes.StringDType'> and <class 'numpy.dtypes.{cls}'> do not have a common DType. " +
                    "For example they cannot be stored in a single array unless the dtype is `object`.", spec);
                Action b = () => np.promote_types(np.dtype(spec), t);
                b.Should().Throw<DTypePromotionError>().WithMessage(
                    $"The DTypes <class 'numpy.dtypes.{cls}'> and <class 'numpy.dtypes.StringDType'> do not have a common DType. " +
                    "For example they cannot be stored in a single array unless the dtype is `object`.", spec);
            }
        }

        /// <summary>
        ///     The common instance: compatible sentinels (at most one present, or equal) give the present one and the FIRST
        ///     descriptor's <c>coerce</c>; two different sentinels are NumPy's TypeError naming both descriptors.
        /// </summary>
        [TestMethod]
        public void PromoteTypes_CommonInstance()
        {
            np.promote_types(np.dtypes.StringDType(null), np.dtypes.StringDType()).ToString().Should().Be("StringDType(na_object=None)");
            np.promote_types(np.dtypes.StringDType(), np.dtypes.StringDType(null)).ToString().Should().Be("StringDType(na_object=None)");
            np.promote_types(np.dtypes.StringDType(false), np.dtypes.StringDType()).ToString().Should().Be("StringDType(coerce=False)");
            np.promote_types(np.dtypes.StringDType(), np.dtypes.StringDType(false)).ToString().Should().Be("StringDType()");
            np.promote_types(np.dtypes.StringDType("NA"), np.dtypes.StringDType("NA")).ToString().Should().Be("StringDType(na_object='NA')");
            np.promote_types(np.dtypes.StringDType(null, false), np.dtype("T")).ToString().Should().Be("StringDType(na_object=None, coerce=False)");
            np.promote_types(np.dtypes.StringDType(false), np.dtype("U5")).ToString().Should().Be("StringDType(coerce=False)");
            np.promote_types(np.dtype("U5"), np.dtypes.StringDType(false)).ToString().Should().Be("StringDType()",
                "U5 casts to a fresh StringDType(), the first operand of the common instance");
            np.promote_types(np.dtype("U5"), np.dtypes.StringDType(null)).ToString().Should().Be("StringDType(na_object=None)");
            Action a = () => np.promote_types(np.dtypes.StringDType(null), np.dtypes.StringDType(double.NaN));
            a.Should().Throw<TypeError>().WithMessage(
                "Cannot find common instance for incompatible dtypes 'StringDType(na_object=None)' and 'StringDType(na_object=nan)'");
            Action b = () => np.promote_types(np.dtypes.StringDType("NA"), np.dtypes.StringDType("NB"));
            b.Should().Throw<TypeError>().WithMessage(
                "Cannot find common instance for incompatible dtypes 'StringDType(na_object='NA')' and 'StringDType(na_object='NB')'");
            Action c = () => DTypeRegistry.VString.CommonInstance(np.dtype("U5"), np.dtype("T"));
            c.Should().Throw<ArgumentException>();
        }

        /// <summary>
        ///     <c>np.result_type</c>: the sequence promotion (NumPy's most-knowledgeable reduction) gives <c>T</c> for
        ///     <c>(U5, T, S2)</c> but refuses <c>(T, U5, i8)</c> and a weak Python literal, with NumPy's texts naming the
        ///     DTypes in NumPy's order.
        /// </summary>
        [TestMethod]
        public void ResultType_FollowsNumPysSequencePromotion()
        {
            var t = np.dtype("T");
            np.result_type(t).Should().Be(np.dtypes.StringDType());
            np.result_type(np.dtypes.StringDType(null)).ToString().Should().Be("StringDType(na_object=None)");
            np.result_type(t, np.dtype("U5")).Should().Be(np.dtypes.StringDType());
            np.result_type(np.dtype("U5"), t, np.dtype("S2")).Should().Be(np.dtypes.StringDType());
            np.result_type(t, np.dtypes.StringDType(null)).ToString().Should().Be("StringDType(na_object=None)");

            const string tail = ". This means that no common DType exists for the given inputs. For example they cannot be stored in a single " +
                                "array unless the dtype is `object`. The full list of DTypes is: ";
            Action a = () => np.result_type(t, np.dtype("U5"), np.dtype("i8"));
            a.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.StrDType'>" + tail +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.Int64DType'>)");
            Action b = () => np.result_type((object)t, 1);
            b.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes._PyLongDType'>" + tail +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes._PyLongDType'>)");
            Action c = () => np.result_type((object)t, 1.5);
            c.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes._PyFloatDType'>" + tail +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes._PyFloatDType'>)");
            Action d = () => np.result_type(np.dtype("i8"), t);
            d.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.Int64DType'> could not be promoted by <class 'numpy.dtypes.StringDType'>" + tail +
                "(<class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.StringDType'>)");
            Action e = () => np.result_type((object)t, true);
            e.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.BoolDType'>" + tail +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.BoolDType'>)");
            Action f = () => np.result_type(np.dtypes.StringDType(null), np.dtypes.StringDType(double.NaN));
            f.Should().Throw<TypeError>().WithMessage(
                "Cannot find common instance for incompatible dtypes 'StringDType(na_object=None)' and 'StringDType(na_object=nan)'");
        }

        /// <summary>
        ///     The common-DType slot itself: only <c>StrDType</c> (and the class itself) promote to <c>StringDType</c>; the
        ///     other classes defer, and the abstract Python-literal classes do not know it either.
        /// </summary>
        [TestMethod]
        public void CommonDType_Slot()
        {
            var v = DTypeRegistry.VString;
            v.CommonDType(v).Should().BeSameAs(v);
            v.CommonDType(DTypeRegistry.Str).Should().BeSameAs(v);
            v.CommonDType(DTypeRegistry.Bytes).Should().BeNull();
            v.CommonDType(DTypeRegistry.Double).Should().BeNull();
            v.CommonDType(DTypeRegistry.PyLong).Should().BeNull();
            DTypeRegistry.Str.CommonDType(v).Should().BeNull("StrDType defers to a non-legacy class");
            DTypeRegistry.PyFloat.CommonDType(v).Should().BeNull();
            DTypePromotion.CommonDType(DTypeRegistry.Str, v).Should().BeSameAs(v);
            Action a = () => v.CommonDType(null);
            a.Should().Throw<ArgumentNullException>();
        }

        // ------------------------------------------------------------------------------------------------------------
        // casting
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.can_cast</c> to and from <c>T</c>, every casting rule, against NumPy's table: numbers / bool / datetime
        ///     into <c>T</c> safe; <c>T</c> into numbers and datetimes unsafe only; <c>T</c> into bool same_kind;
        ///     <c>T</c> ↔ <c>U</c> / <c>S</c> (sized or not) same_kind.
        /// </summary>
        [TestMethod]
        public void CanCast_TableMatchesNumPy()
        {
            var t = np.dtype("T");
            string[] rules = { "no", "equiv", "safe", "same_kind", "unsafe" };
            // (other, the weakest rule allowing T -> other, the weakest rule allowing other -> T)
            var table = new (string other, string toOther, string fromOther)[]
            {
                ("U5", "same_kind", "same_kind"), ("U", "same_kind", "same_kind"), ("S5", "same_kind", "same_kind"),
                ("S", "same_kind", "same_kind"), ("i8", "unsafe", "safe"), ("u1", "unsafe", "safe"), ("f8", "unsafe", "safe"),
                ("f2", "unsafe", "safe"), ("?", "same_kind", "safe"), ("c16", "unsafe", "safe"), ("M8[s]", "unsafe", "safe"),
                ("m8[s]", "unsafe", "safe"), ("i1", "unsafe", "safe"), ("u8", "unsafe", "safe"), ("f4", "unsafe", "safe"),
            };
            foreach (var (other, toOther, fromOther) in table)
            {
                var o = np.dtype(other);
                for (int i = 0; i < rules.Length; i++)
                {
                    np.can_cast(t, o, rules[i]).Should().Be(i >= Array.IndexOf(rules, toOther), $"T -> {other} under {rules[i]}");
                    np.can_cast(o, t, rules[i]).Should().Be(i >= Array.IndexOf(rules, fromOther), $"{other} -> T under {rules[i]}");
                }
            }
            // NumSharp's own classes cast as the NumPy classes they convert as: decimal like float64, char like uint16.
            np.can_cast(np.dtype("decimal"), t, "safe").Should().BeTrue();
            np.can_cast(t, np.dtype("decimal"), "same_kind").Should().BeFalse();
            np.can_cast(np.dtype("char"), t, "safe").Should().BeTrue();
            np.can_cast(t, np.dtype("char"), "unsafe").Should().BeTrue();
        }

        /// <summary>
        ///     <c>T</c> → <c>T</c>: no casting between any two descriptors, except from one WITH a sentinel to one WITHOUT,
        ///     which is unsafe (the resolver runs because the method's minimal safety is unsafe).
        /// </summary>
        [TestMethod]
        public void CanCast_WithinStringDType()
        {
            var t = np.dtype("T");
            var none = np.dtypes.StringDType(null);
            var nan = np.dtypes.StringDType(double.NaN);
            foreach (var rule in new[] { "no", "equiv", "safe", "same_kind", "unsafe" })
            {
                np.can_cast(t, t, rule).Should().BeTrue(rule);
                np.can_cast(t, none, rule).Should().BeTrue(rule);
                np.can_cast(np.dtypes.StringDType(false), t, rule).Should().BeTrue(rule);
                np.can_cast(nan, none, rule).Should().BeTrue("both have a sentinel: " + rule);
                np.can_cast(none, t, rule).Should().Be(rule == "unsafe", rule);
            }
            DTypeCasting.EquivTypes(np.dtypes.StringDType(), np.dtypes.StringDType()).Should().BeTrue();
            DTypeCasting.EquivTypes(none, t).Should().BeFalse();
        }

        /// <summary>
        ///     The resolved descriptors and view offsets: <c>T</c> → <c>T</c> without a destination keeps the source's own
        ///     descriptor (a view, offset 0); into <c>T</c> without a destination is a fresh <c>StringDType()</c>; out of
        ///     <c>T</c> into a number defaults that class's descriptor; the operator forms read the safe-cast order.
        /// </summary>
        [TestMethod]
        public void ResolveDescriptors_Defaults()
        {
            var t = np.dtypes.StringDType(null);
            DTypeCasting.GetCastInfo(t, null, DTypeRegistry.VString, out long view).Should().Be(NPY_CASTING.NPY_NO_CASTING);
            view.Should().Be(0, "the same descriptor shares the allocator, so the cast can be a view");
            DTypeCasting.GetCastInfo(t, np.dtypes.StringDType(null), null, out view).Should().Be(NPY_CASTING.NPY_NO_CASTING);
            view.Should().Be(ArrayMethod.NoView, "another descriptor object owns another allocator");

            var given = new[] { np.dtype("U5"), null };
            var loop = new DType[2];
            DTypeRegistry.Str.GetCastingImpl(DTypeRegistry.VString).ResolveDescriptors(given, loop, out _).Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            loop[1].Should().Be(np.dtypes.StringDType());
            DTypeRegistry.Bool.GetCastingImpl(DTypeRegistry.VString).Casting.Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            DTypeRegistry.Bool.GetCastingImpl(DTypeRegistry.VString).ResolveDescriptors(new[] { np.dtype("?"), null }, loop, out _)
                .Should().Be(NPY_CASTING.NPY_SAFE_CASTING);

            var impl = DTypeRegistry.VString.GetCastingImpl(DTypeRegistry.Int16);
            impl.ResolveDescriptors(new[] { np.dtype("T"), null }, loop, out view).Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            loop[1].Should().Be(np.int16);
            view.Should().Be(ArrayMethod.NoView);
            DTypeRegistry.VString.GetCastingImpl(DTypeRegistry.Bool).ResolveDescriptors(new[] { np.dtype("T"), null }, loop, out _)
                .Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            loop[1].Should().Be(np.bool_);

            (np.dtype("i8") < np.dtype("T")).Should().BeTrue();
            (np.dtype("T") < np.dtype("U5")).Should().BeFalse();
            (np.dtype("U5") < np.dtype("T")).Should().BeFalse();
            (np.dtype("T") <= np.dtype("T")).Should().BeTrue();
        }

        /// <summary>
        ///     Casting to an UNSIZED destination class: <c>T</c> → <c>U</c> / <c>S</c> / <c>M8</c> cannot pick a width or
        ///     unit, so <see cref="DTypePromotion.CastDescrToDType"/> raises NumPy's <c>cannot cast dtype …</c> wrapper
        ///     around the resolver's own text.
        /// </summary>
        [TestMethod]
        public void CastDescrToDType_UnsizedDestinationsRefuse()
        {
            var t = np.dtype("T");
            foreach (var (meta, cls) in new (DTypeMeta, string)[] { (DTypeRegistry.Str, "StrDType"), (DTypeRegistry.Bytes, "BytesDType"), (DTypeRegistry.DateTime64, "DateTime64DType") })
            {
                Action a = () => DTypePromotion.CastDescrToDType(t, meta);
                a.Should().Throw<TypeError>().WithMessage($"cannot cast dtype StringDType() to <class 'numpy.dtypes.{cls}'>.")
                    .WithInnerException<TypeError>();
            }
            DTypePromotion.CastDescrToDType(np.dtype("U3"), DTypeRegistry.VString).Should().Be(np.dtypes.StringDType());
            DTypePromotion.CastDescrToDType(t, DTypeRegistry.Double).Should().Be(np.float64);
            DTypePromotion.CastDescrToDType(t, DTypeRegistry.VString).Should().BeSameAs(t);
            var s5 = np.dtype("S5");
            var loop = new DType[2];
            DTypeRegistry.VString.GetCastingImpl(DTypeRegistry.Bytes).ResolveDescriptors(new[] { t, s5 }, loop, out _)
                .Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            loop[1].Should().BeSameAs(s5);
            Action m8 = () => DTypeRegistry.VString.GetCastingImpl(DTypeRegistry.TimeDelta64).ResolveDescriptors(new[] { t, null }, loop, out _);
            m8.Should().Throw<TypeError>().WithMessage("Casting from StringDType to datetimes without a unit is not currently supported");
        }

        // ------------------------------------------------------------------------------------------------------------
        // the class slots and the guards
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The class's discovery and known-scalar slots: every Python builtin and NumPy scalar is a known scalar, and
        ///     discovery yields a fresh <c>StringDType()</c> — refusing bytes that are not UTF-8 with CPython's decoder text.
        /// </summary>
        [TestMethod]
        public void DiscoveryAndKnownScalars()
        {
            var v = DTypeRegistry.VString;
            foreach (var type in new[] { typeof(string), typeof(char), typeof(byte[]), typeof(bool), typeof(int), typeof(ulong), typeof(BigInteger),
                                         typeof(float), typeof(double), typeof(Half), typeof(decimal), typeof(Complex), typeof(sbyte), typeof(short) })
                v.IsKnownScalarType(type).Should().BeTrue(type.Name);
            v.IsKnownScalarType(typeof(object)).Should().BeFalse();
            v.IsKnownScalarType(typeof(NDArray)).Should().BeFalse();
            v.DiscoverDescrFromObject("abc").Should().Be(np.dtypes.StringDType());
            v.DiscoverDescrFromObject(3.5).Should().Be(np.dtypes.StringDType());
            v.DiscoverDescrFromObject(new byte[] { 0xC3, 0xA9 }).Should().Be(np.dtypes.StringDType());
            Action bad = () => v.DiscoverDescrFromObject(new byte[] { 0x80 });
            bad.Should().Throw<UnicodeDecodeError>().WithMessage("'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte");
            v.HasReferences.Should().BeTrue();
            v.IsLegacy.Should().BeFalse();
            v.IsParametric.Should().BeTrue();
            v.GetClearLoop(np.dtype("T")).Should().BeNull("the clear loop arrives with the storage (P1.3)");
        }

        /// <summary>
        ///     A <c>StringDType</c> descriptor as a ufunc <c>dtype=</c> is NumPy's new-style-instance refusal (NumSharp cannot
        ///     pass the class instead).
        /// </summary>
        [TestMethod]
        public void UfuncDtype_RefusesStringDTypeInstances()
        {
            Action a = () => np.add(np.arange(3), np.arange(3), dtype: np.dtype("T"));
            a.Should().Throw<TypeError>().WithMessage(
                "Cannot pass a new user DType instance to the `dtype` or `signature` arguments of ufuncs. Pass the DType class instead.");
            Action b = () => DefaultEngine.ThrowIfNotGeneralDType(np.dtypes.StringDType(null));
            b.Should().Throw<TypeError>().WithMessage("Cannot pass a new user DType instance*");
        }

        /// <summary>
        ///     Since P1.3 a <c>StringDType</c> descriptor allocates NumPy's 16-byte cells: every element of a fresh array is
        ///     <c>''</c> (an all-zero cell), the array keeps the descriptor INSTANCE it was given (its <c>na_object</c> /
        ///     <c>coerce</c>), and itemsize / nbytes / strides are NumPy's.
        /// </summary>
        [TestMethod]
        public void Allocation_GivesEmptyCellsWithTheDescriptor()
        {
            var t = np.dtype("T");
            using var a = np.zeros(3, t);
            ReferenceEquals(a.dtype, t).Should().BeTrue();
            a.itemsize.Should().Be(16);
            a.nbytes.Should().Be(48);
            a.strides.Should().Equal(16L);
            a.GetAtIndex(0).Should().Be(string.Empty);
            var tn = np.dtypes.StringDType(null);
            using var b = new NDArray(tn, new Shape(2, 2));
            b.dtype.Should().Be(tn);
            b.strides.Should().Equal(32L, 16L);
            b.GetAtIndex(3).Should().Be(string.Empty);
        }
    }
}
