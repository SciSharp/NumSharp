using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The descriptor surface of <see cref="DType"/> (NumPy's <c>numpy.dtype</c> instance) after the NEP 41/42
    ///     re-foundation: class + parameters, singletons, structural equality, coercing equality, byte order, rendering.
    ///     Every expectation was probed live against <c>numpy==2.4.2</c> (win-amd64); the two documented divergences
    ///     are marked <see cref="MisalignedAttribute"/>.
    /// </summary>
    [TestClass]
    public class DTypeDescriptorTests
    {
        // ---- the 15 storage-backed builtins: NumPy's surface, one row each ---------------------------------------

        [DataTestMethod]
        [DataRow("?", 0, 'b', '?', '|', 1, 1, "bool", "|b1", "dtype('bool')", "bool", "BoolDType")]
        [DataRow("i1", 1, 'i', 'b', '|', 1, 1, "int8", "|i1", "dtype('int8')", "int8", "Int8DType")]
        [DataRow("u1", 2, 'u', 'B', '|', 1, 1, "uint8", "|u1", "dtype('uint8')", "uint8", "UInt8DType")]
        [DataRow("i2", 3, 'i', 'h', '=', 2, 2, "int16", "<i2", "dtype('int16')", "int16", "Int16DType")]
        [DataRow("u2", 4, 'u', 'H', '=', 2, 2, "uint16", "<u2", "dtype('uint16')", "uint16", "UInt16DType")]
        [DataRow("i4", 5, 'i', 'i', '=', 4, 4, "int32", "<i4", "dtype('int32')", "int32", "Int32DType")]
        [DataRow("u4", 6, 'u', 'I', '=', 4, 4, "uint32", "<u4", "dtype('uint32')", "uint32", "UInt32DType")]
        [DataRow("i8", 7, 'i', 'l', '=', 8, 8, "int64", "<i8", "dtype('int64')", "int64", "Int64DType")]
        [DataRow("u8", 8, 'u', 'L', '=', 8, 8, "uint64", "<u8", "dtype('uint64')", "uint64", "UInt64DType")]
        [DataRow("e", 23, 'f', 'e', '=', 2, 2, "float16", "<f2", "dtype('float16')", "float16", "Float16DType")]
        [DataRow("f", 11, 'f', 'f', '=', 4, 4, "float32", "<f4", "dtype('float32')", "float32", "Float32DType")]
        [DataRow("d", 12, 'f', 'd', '=', 8, 8, "float64", "<f8", "dtype('float64')", "float64", "Float64DType")]
        [DataRow("D", 15, 'c', 'D', '=', 16, 8, "complex128", "<c16", "dtype('complex128')", "complex128", "Complex128DType")]
        [DataRow("g", 12, 'f', 'd', '=', 8, 8, "float64", "<f8", "dtype('float64')", "float64", "Float64DType")]
        [DataRow("G", 15, 'c', 'D', '=', 16, 8, "complex128", "<c16", "dtype('complex128')", "complex128", "Complex128DType")]
        public void Builtin_Surface(string spelling, int num, char kind, char ch, char byteorder, int itemsize, int alignment,
            string name, string str, string repr, string tostr, string cls)
        {
            var d = np.dtype(spelling);
            d.num.Should().Be(num);
            d.kind.Should().Be(kind);
            d.@char.Should().Be(ch);
            d.byteorder.Should().Be(byteorder);
            d.itemsize.Should().Be(itemsize);
            d.alignment.Should().Be(alignment);
            d.name.Should().Be(name);
            d.str.Should().Be(str);
            d.ToString(true).Should().Be(repr);
            d.ToString().Should().Be(tostr);
            d.Meta.Name.Should().Be(cls);
            d.isbuiltin.Should().Be(1);
            d.isnative.Should().BeTrue();
            d.hasobject.Should().BeFalse();
            d.flags.Should().Be(0);
            d.metadata.Should().BeNull();
            d.shape.Should().BeEmpty();
            d.ndim.Should().Be(0);
            d.@base.Should().BeSameAs(d);
            d.subdtype.Should().BeNull();
            d.fields.Should().BeNull();
            d.names.Should().BeNull();
            d.descr.Should().ContainSingle().Which.Should().Be(("", str));
            d.DatetimeMetadata.Should().BeNull();
        }

        [TestMethod]
        public void NumSharpOnly_Decimal_And_Char_AreUserRangeClasses()
        {
            var dec = np.dtype("decimal");
            dec.num.Should().Be(256);
            dec.kind.Should().Be('f');
            dec.itemsize.Should().Be(16);
            dec.name.Should().Be("decimal");
            dec.str.Should().Be("<f16");
            dec.ToString(true).Should().Be("dtype(decimal)", "NumPy renders a user-defined dtype's repr without quotes");
            dec.ToString().Should().Be("decimal");
            dec.isbuiltin.Should().Be(2, "type numbers >= 256 are the user-defined range");
            dec.Meta.Should().BeSameAs(np.dtypes.DecimalDType);

            var ch = np.dtype("char");
            ch.num.Should().Be(257);
            ch.kind.Should().Be('u', "Char promotes and prints as uint16 everywhere");
            ch.@char.Should().Be('c');
            ch.itemsize.Should().Be(2);
            ch.name.Should().Be("char");
            ch.str.Should().Be("<u2");
            ch.ToString(true).Should().Be("dtype(char)");
            ch.isbuiltin.Should().Be(2);
            ch.Meta.Should().BeSameAs(np.dtypes.CharDType);
        }

        // ---- datetime64 / timedelta64: parametric descriptors -----------------------------------------------------

        [DataTestMethod]
        [DataRow("M8", 21, 'M', "datetime64", "<M8", "dtype('<M8')", "datetime64", "generic", 1)]
        [DataRow("m8", 22, 'm', "timedelta64", "<m8", "dtype('<m8')", "timedelta64", "generic", 1)]
        [DataRow("M8[ns]", 21, 'M', "datetime64[ns]", "<M8[ns]", "dtype('<M8[ns]')", "datetime64[ns]", "ns", 1)]
        [DataRow("M8[10ns]", 21, 'M', "datetime64[10ns]", "<M8[10ns]", "dtype('<M8[10ns]')", "datetime64[10ns]", "ns", 10)]
        [DataRow("m8[2W]", 22, 'm', "timedelta64[2W]", "<m8[2W]", "dtype('<m8[2W]')", "timedelta64[2W]", "W", 2)]
        [DataRow("datetime64[D]", 21, 'M', "datetime64[D]", "<M8[D]", "dtype('<M8[D]')", "datetime64[D]", "D", 1)]
        [DataRow("timedelta64[us]", 22, 'm', "timedelta64[us]", "<m8[us]", "dtype('<m8[us]')", "timedelta64[us]", "us", 1)]
        [DataRow("M", 21, 'M', "datetime64", "<M8", "dtype('<M8')", "datetime64", "generic", 1)]
        [DataRow("m", 22, 'm', "timedelta64", "<m8", "dtype('<m8')", "timedelta64", "generic", 1)]
        public void Datetime_Surface(string spelling, int num, char kind, string name, string str, string repr, string tostr, string unit, int count)
        {
            var d = np.dtype(spelling);
            d.num.Should().Be(num);
            d.kind.Should().Be(kind);
            d.@char.Should().Be(kind);
            d.byteorder.Should().Be('=');
            d.itemsize.Should().Be(8);
            d.alignment.Should().Be(8);
            d.name.Should().Be(name);
            d.str.Should().Be(str);
            d.ToString(true).Should().Be(repr);
            d.ToString().Should().Be(tostr);
            d.isbuiltin.Should().Be(0, "NumPy's datetime default_descr hands out a COPY of the singleton, never the singleton");
            d.isnative.Should().BeTrue();
            d.type.Should().BeNull("no C# scalar type until the storage lane lands (Stage C)");
            d.typecode.Should().Be(NPTypeCode.Empty);
            d.Meta.IsParametric.Should().BeTrue();
            d.Meta.IsLegacy.Should().BeTrue();
            d.Meta.IsNumeric.Should().BeFalse();
            np.datetime_data(d).Should().Be((unit, count));
            d.DatetimeMetadata.Should().NotBeNull();
        }

        [TestMethod]
        public void Datetime_HasNoStorage_ConversionsRaiseNotSupported()
        {
            var d = np.dtype("M8[ns]");
            Action toType = () => { Type _ = d; };
            Action toCode = () => { NPTypeCode _ = d; };
            Action getCode = () => d.GetTypeCode();
            toType.Should().Throw<NotSupportedException>().WithMessage("*has no NumSharp storage yet*Stage C*");
            toCode.Should().Throw<NotSupportedException>();
            getCode.Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Datetime_Data_OfNonDatetime_Raises_NumPyText()
        {
            Action act = () => np.datetime_data(np.dtype("i4"));
            act.Should().Throw<TypeError>().WithMessage("cannot get datetime metadata from non-datetime type");
            np.datetime_data("M8[s]").Should().Be(("s", 1));
        }

        // ---- singletons ----------------------------------------------------------------------------------------

        [TestMethod]
        public void Builtins_AreSingletons_DatetimesAreNot()
        {
            np.dtype("i8").Should().BeSameAs(np.dtype("i8"));
            np.dtype("i8").Should().BeSameAs(DType.Int64);
            np.dtype("int64").Should().BeSameAs(np.dtype("q"));
            ((DType)typeof(int)).Should().BeSameAs(np.dtype("int32"));
            ((DType)NPTypeCode.Double).Should().BeSameAs(DType.Double);
            DType.From(typeof(int[])).Should().BeSameAs(DType.Int32, "array types map to their element class");
            np.dtypes.Int8DType.Instantiate().Should().BeSameAs(np.dtype("i1"));

            np.dtype("M8[ns]").Should().NotBeSameAs(np.dtype("M8[ns]"));
            np.dtype("M8[ns]").Should().Be(np.dtype("M8[ns]"));
            np.dtype(">i4").Should().NotBeSameAs(np.dtype(">i4"));
        }

        [TestMethod]
        public void PublicConstructors_ProduceEqualCopies_NotTheSingleton()
        {
            var copy = new DType(typeof(int));
            copy.Should().Be(DType.Int32);
            copy.Should().NotBeSameAs(DType.Int32);
            copy.isbuiltin.Should().Be(0, "isbuiltin == 1 is reserved for the canonical instance, as in NumPy");
            new DType(NPTypeCode.Half).Should().Be(np.dtype("f2"));
        }

        // ---- equality: structural + coercing -----------------------------------------------------------------------

        [TestMethod]
        public void Equality_IsStructural()
        {
            (np.dtype("M8[ns]") == np.dtype("M8[s]")).Should().BeFalse();
            (np.dtype("M8[1s]") == np.dtype("M8[s]")).Should().BeTrue();
            (np.dtype("M8") == np.dtype("m8")).Should().BeFalse();
            (np.dtype("M8[s]") == np.dtype("m8[s]")).Should().BeFalse();
            (np.dtype(">i4") == np.dtype("<i4")).Should().BeFalse();
            (np.dtype(">i1") == np.dtype("i1")).Should().BeTrue("single-byte types have no byte order");
            (np.dtype("f8") == np.dtype("d")).Should().BeTrue();
            (np.dtype("M8[us]") == np.dtype("M8[μs]")).Should().BeTrue();
            ((DType)null == (DType)null).Should().BeTrue();
            (np.dtype("i4") == (DType)null).Should().BeFalse();
        }

        [TestMethod]
        public void Equality_Coerces_Type_NPTypeCode_And_String_LikeNumPy()
        {
            var d = np.dtype("i4");
            d.Equals("i4").Should().BeTrue();
            d.Equals("int32").Should().BeTrue();
            d.Equals(typeof(int)).Should().BeTrue();
            d.Equals(NPTypeCode.Int32).Should().BeTrue();
            d.Equals(typeof(long)).Should().BeFalse();
            d.Equals("garbage").Should().BeFalse("np.dtype('i4') == 'garbage' is False, never an error");
            d.Equals(typeof(DTypeDescriptorTests)).Should().BeFalse();
            d.Equals(NPTypeCode.Empty).Should().BeFalse();
            d.Equals((object)null).Should().BeFalse();
            d.Equals(np.dtypes.Int32DType).Should().BeFalse("a class object is not a descriptor (NumPy: np.dtype(cls) is the object dtype)");
            (d == "i4").Should().BeTrue("a valid dtype string converts implicitly");
            (d != "i4").Should().BeFalse();
            ("f8" == np.dtype("double")).Should().BeTrue();
            (np.dtype(">i4") == "i4").Should().BeFalse("NumPy: np.dtype('>i4') == 'i4' is False");
            // `d == "garbage"` goes through the implicit string->DType conversion and raises like np.dtype("garbage");
            // the NumPy-shaped "unequal, never an error" answer is Equals(string) (asserted above). There are deliberately
            // no ==(DType, string) operators: a null literal would bind them and break every `descr == null` check.
            Action viaOperator = () => { bool _ = d == "garbage"; };
            viaOperator.Should().Throw<NotSupportedException>();
            DType none = null;
            (none == null).Should().BeTrue();
            (none != null).Should().BeFalse();
        }

        [TestMethod]
        public void Hash_FollowsNumPy_ClassAndByteOrder()
        {
            np.dtype("M8[ns]").GetHashCode().Should().Be(np.dtype("M8[s]").GetHashCode(), "NumPy: hash(M8[ns]) == hash(M8[s])");
            np.dtype("M8[s]").GetHashCode().Should().Be(np.dtype("M8[1000ms]").GetHashCode());
            np.dtype("i4").GetHashCode().Should().NotBe(np.dtype(">i4").GetHashCode());
            np.dtype("M8").GetHashCode().Should().NotBe(np.dtype("m8").GetHashCode());
            np.dtype("i4").GetHashCode().Should().Be(new DType(typeof(int)).GetHashCode());
        }

        [TestMethod]
        [Misaligned]
        public void Equality_Structural_WhereNumPyEquivTypesIsAsymmetric()
        {
            // NumPy's dtype.__eq__ is PyArray_EquivTypes = "casting is a no-op", and time_to_time's exact 1000-fold rule
            // makes that ASYMMETRIC: np.dtype('M8[1000ms]') == np.dtype('M8[s]') is True while the reverse is False.
            // NumSharp keeps the (symmetric) structural equality and exposes NumPy's relation as DTypeCasting.EquivTypes.
            (np.dtype("M8[1000ms]") == np.dtype("M8[s]")).Should().BeFalse();
            DTypeCasting.EquivTypes(np.dtype("M8[1000ms]"), np.dtype("M8[s]")).Should().BeTrue();
            DTypeCasting.EquivTypes(np.dtype("M8[s]"), np.dtype("M8[1000ms]")).Should().BeFalse();
        }

        // ---- ordering operators (NumPy: <, <= via safe casting) ---------------------------------------------------

        [TestMethod]
        public void OrderingOperators_FollowSafeCasting()
        {
            (np.dtype("i4") < np.dtype("i8")).Should().BeTrue();
            (np.dtype("i8") <= np.dtype("i8")).Should().BeTrue();
            (np.dtype("i8") < np.dtype("i8")).Should().BeFalse();
            (np.dtype("f8") > np.dtype("i8")).Should().BeTrue();
            (np.dtype("i8") > np.dtype("f8")).Should().BeFalse();
            (np.dtype("M8[s]") < np.dtype("M8[ms]")).Should().BeTrue();
            (np.dtype("M8[ms]") < np.dtype("M8[s]")).Should().BeFalse();
            (np.dtype("i4") >= np.dtype("i2")).Should().BeTrue();
        }

        // ---- byte order --------------------------------------------------------------------------------------------

        [TestMethod]
        public void NonNative_ByteOrder_IsKeptOnTheDescriptor()
        {
            var d = np.dtype(">i4");
            d.byteorder.Should().Be('>');
            d.isnative.Should().BeFalse();
            d.isbuiltin.Should().Be(0);
            d.name.Should().Be("int32", "name ignores byte order");
            d.str.Should().Be(">i4");
            d.ToString().Should().Be(">i4", "str() of a non-native descriptor is its typestr");
            d.ToString(true).Should().Be("dtype('>i4')");
            d.Meta.Should().BeSameAs(np.dtypes.Int32DType);

            np.dtype(">f8").str.Should().Be(">f8");
            np.dtype(">e").ToString(true).Should().Be("dtype('>f2')");
            np.dtype(">D").str.Should().Be(">c16");
            np.dtype(">M8[ns]").ToString(true).Should().Be("dtype('>M8[ns]')");
            np.dtype(">M8[ns]").ToString().Should().Be(">M8[ns]");
            np.dtype(">M8[ns]").name.Should().Be("datetime64[ns]");
            np.dtype(">m8").str.Should().Be(">m8");

            np.dtype("<i4").byteorder.Should().Be('=');
            np.dtype("=i4").byteorder.Should().Be('=');
            np.dtype("|i4").byteorder.Should().Be('=');
            np.dtype("<i4").Should().BeSameAs(np.dtype("i4"), "the host order normalises to native, which is the singleton");
            np.dtype(">b1").byteorder.Should().Be('|');
            np.dtype(">?").ToString(true).Should().Be("dtype('bool')");
            np.dtype(">i1").byteorder.Should().Be('|');
            np.dtype(">u1").byteorder.Should().Be('|');
        }

        [TestMethod]
        public void NewByteOrder_FollowsNumPy()
        {
            np.dtype("i4").newbyteorder(">").str.Should().Be(">i4");
            np.dtype("i4").newbyteorder("S").str.Should().Be(">i4");
            np.dtype(">i4").newbyteorder("S").str.Should().Be("<i4");
            np.dtype("i4").newbyteorder("=").str.Should().Be("<i4");
            np.dtype(">i4").newbyteorder("=").str.Should().Be("<i4");
            np.dtype("i4").newbyteorder("|").str.Should().Be("<i4");
            np.dtype(">i4").newbyteorder("|").str.Should().Be(">i4", "'|' means ignore: no change");
            np.dtype("i1").newbyteorder(">").str.Should().Be("|i1");
            np.dtype("i1").newbyteorder(">").byteorder.Should().Be('|');
            np.dtype("i4").newbyteorder("B").str.Should().Be(">i4");
            np.dtype("i4").newbyteorder("big").str.Should().Be(">i4");
            np.dtype("i4").newbyteorder("little").str.Should().Be("<i4");
            np.dtype("i4").newbyteorder("L").str.Should().Be("<i4");
            np.dtype("i4").newbyteorder("N").str.Should().Be("<i4");
            np.dtype("i4").newbyteorder("I").str.Should().Be("<i4");
            np.dtype(">i4").newbyteorder(">").str.Should().Be(">i4");
            np.dtype(">M8[ns]").newbyteorder("S").str.Should().Be("<M8[ns]");
            np.dtype("i4").newbyteorder().str.Should().Be(">i4", "the default is 'S' (swap)");
            np.dtype("i4").newbyteorder('>').Should().Be(np.dtype(">i4"));
            np.dtype("i4").newbyteorder("=").isbuiltin.Should().Be(0, "newbyteorder always returns a new descriptor");
        }

        [TestMethod]
        public void NewByteOrder_Rejects_UnknownCode_NumPyText()
        {
            Action bad = () => np.dtype("i4").newbyteorder("x");
            bad.Should().Throw<ValueError>().WithMessage("byteorder not recognized (got 'x')");
            Action empty = () => np.dtype("i4").newbyteorder("");
            empty.Should().Throw<ValueError>().WithMessage("byteorder not recognized (got '')");
        }

        [TestMethod]
        public void EnsureCanonical_ReturnsNativeForm()
        {
            var big = np.dtype(">i4");
            big.Meta.EnsureCanonical(big).Should().BeSameAs(DType.Int32);
            DType.Int32.Meta.EnsureCanonical(DType.Int32).Should().BeSameAs(DType.Int32);
            var bigDt = np.dtype(">M8[ns]");
            var canon = bigDt.Meta.EnsureCanonical(bigDt);
            canon.isnative.Should().BeTrue();
            canon.Should().Be(np.dtype("M8[ns]"));
        }

        // ---- implicit conversions (the four spellings) -------------------------------------------------------------

        [TestMethod]
        public void ImplicitConversions_RoundTrip()
        {
            Type t = DType.Double;
            t.Should().Be(typeof(double));
            NPTypeCode c = (DType)"int32";
            c.Should().Be(NPTypeCode.Int32);
            DType fromNullable = (NPTypeCode?)null;
            fromNullable.Should().BeNull();
            DType fromEmpty = NPTypeCode.Empty;
            fromEmpty.Should().BeNull();
            DType fromNullString = (string)null;
            fromNullString.Should().BeNull();
            Type nullType = (DType)null;
            nullType.Should().BeNull();
            NPTypeCode empty = (DType)null;
            empty.Should().Be(NPTypeCode.Empty);
            ((DType)"M8[ns]").Meta.Should().BeSameAs(np.dtypes.DateTime64DType);
        }

        [TestMethod]
        public void Dtype_Factories_AcceptEverySpelling()
        {
            np.dtype(typeof(int)).Should().BeSameAs(DType.Int32);
            np.dtype(NPTypeCode.Single).Should().BeSameAs(DType.Single);
            np.dtype(DType.Half).Should().BeSameAs(DType.Half);
            np.dtype(typeof(string)).ToString(true).Should().Be("dtype('<U')", "the vestigial string slot is NumPy's unsized U");
            np.dtype("String").str.Should().Be("<U0");
            np.dtype("String").name.Should().Be("str");
            Action nullType = () => np.dtype((Type)null);
            nullType.Should().Throw<ArgumentNullException>();
            Action foreign = () => np.dtype(typeof(DTypeDescriptorTests));
            foreign.Should().Throw<NotSupportedException>();
        }

        // ---- issubdtype / isdtype over descriptors ----------------------------------------------------------------

        [TestMethod]
        public void Issubdtype_Descriptors_CompareScalarTypes()
        {
            np.issubdtype(np.dtype("M8[s]"), np.dtype("M8[ns]")).Should().BeTrue();
            np.issubdtype(np.dtype(">i4"), np.dtype("i4")).Should().BeTrue();
            np.issubdtype(np.dtype("i4"), np.dtype("i8")).Should().BeFalse();
            np.issubdtype(np.dtype("m8"), np.dtype("M8")).Should().BeFalse();
            np.issubdtype(np.dtype("M8"), "generic").Should().BeTrue();
            np.issubdtype(np.dtype("M8"), "number").Should().BeFalse("Datetime doesn't fit in any category");
            np.issubdtype(np.dtype("m8"), "signedinteger").Should().BeTrue("Timedelta is an integer with an associated unit");
            np.issubdtype(np.dtype("m8"), "integer").Should().BeTrue();
            np.issubdtype(np.dtype("m8"), "number").Should().BeTrue();
            np.issubdtype(np.dtype("m8"), "floating").Should().BeFalse();
            np.issubdtype("i4", "integer").Should().BeTrue("a dtype string binds the dtype grammar, not the char-array conversion");
            np.issubdtype("M8[ns]", "generic").Should().BeTrue();
            np.issubdtype(np.dtype("?"), "integer").Should().BeFalse("NumPy 2.x: bool is not an integer");
        }

        [TestMethod]
        public void Isdtype_Descriptors_UseNumPy2Categories()
        {
            np.isdtype(np.dtype("M8"), "numeric").Should().BeFalse();
            np.isdtype(np.dtype("m8"), "integral").Should().BeFalse("sctypes['int'] excludes timedelta64");
            np.isdtype(np.dtype("m8"), "signed integer").Should().BeFalse();
            np.isdtype(np.dtype("M8[s]"), np.dtype("M8[ns]")).Should().BeTrue();
            np.isdtype(np.dtype("f4"), np.dtype("f8")).Should().BeFalse();
            np.isdtype(np.dtype("i4"), "signed integer").Should().BeTrue();
            np.isdtype(np.dtype("u4"), "unsigned integer").Should().BeTrue();
            np.isdtype(np.dtype("u4"), "signed integer").Should().BeFalse();
            np.isdtype(np.dtype("?"), "numeric").Should().BeFalse();
            np.isdtype(np.dtype("?"), "bool").Should().BeTrue();
            np.isdtype(np.dtype("i4"), "bool", "integral").Should().BeTrue();
            np.isdtype(np.dtype("c16"), "real floating", "complex floating").Should().BeTrue();
            np.isdtype("i4", "integral").Should().BeTrue();
            Action bad = () => np.isdtype(np.dtype("i4"), "bogus");
            bad.Should().Throw<ValueError>().WithMessage("kind argument is a string, but 'bogus' is not a known kind name.");
        }

        // ---- the legacy kind map is untouched (the old promotion engine keys on it) ---------------------------------

        [TestMethod]
        public void LegacyKindMap_StillCarriesTheOldCodes()
        {
            DType._kind_list_map[NPTypeCode.Boolean].Should().Be('?', "np._FindCommonType orders kinds through this map; it must not move");
            DType._kind_list_map[NPTypeCode.Char].Should().Be('S');
            np.dtype("bool").kind.Should().Be('b', "while the descriptor reports NumPy's kind");
            np.dtype("char").kind.Should().Be('u');
        }

        [TestMethod]
        public void MetaToString_IsNumPysClassRepr()
        {
            np.dtypes.Float64DType.ToString().Should().Be("<class 'numpy.dtypes.Float64DType'>");
            np.dtypes.DateTime64DType.ToString().Should().Be("<class 'numpy.dtypes.DateTime64DType'>");
            np.dtypes.DecimalDType.ToString().Should().Be("<class 'numsharp.dtypes.DecimalDType'>");
            DTypeRegistry.PyLong.ToString().Should().Be("<class 'numpy.dtypes._PyLongDType'>");
        }
    }
}
