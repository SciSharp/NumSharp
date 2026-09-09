using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The NEP 42 / NEP 50 promotion engine (<see cref="DTypePromotion"/>) behind <c>np.promote_types</c> and
    ///     <c>np.result_type</c>: the builtin table, the datetime unit GCD, weak C# literals, order-independent
    ///     sequences and NumPy's verbatim error texts — all probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class DTypePromotionTests
    {
        private static readonly string[] Names =
            { "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex128" };

        // np.promote_types(a, b).name for a in Names, b in Names — NumPy 2.4.2, verbatim.
        private static readonly string[][] NumPyMatrix =
        {
            new[] { "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex128" },
            new[] { "int8", "int8", "int16", "int16", "int32", "int32", "int64", "int64", "float64", "float16", "float32", "float64", "complex128" },
            new[] { "uint8", "int16", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex128" },
            new[] { "int16", "int16", "int16", "int16", "int32", "int32", "int64", "int64", "float64", "float32", "float32", "float64", "complex128" },
            new[] { "uint16", "int32", "uint16", "int32", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float32", "float64", "complex128" },
            new[] { "int32", "int32", "int32", "int32", "int32", "int32", "int64", "int64", "float64", "float64", "float64", "float64", "complex128" },
            new[] { "uint32", "int64", "uint32", "int64", "uint32", "int64", "uint32", "int64", "uint64", "float64", "float64", "float64", "complex128" },
            new[] { "int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64", "float64", "float64", "float64", "float64", "complex128" },
            new[] { "uint64", "float64", "uint64", "float64", "uint64", "float64", "uint64", "float64", "uint64", "float64", "float64", "float64", "complex128" },
            new[] { "float16", "float16", "float16", "float32", "float32", "float64", "float64", "float64", "float64", "float16", "float32", "float64", "complex128" },
            new[] { "float32", "float32", "float32", "float32", "float32", "float64", "float64", "float64", "float64", "float32", "float32", "float64", "complex128" },
            new[] { "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "complex128" },
            new[] { "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128" },
        };

        [TestMethod]
        public void PromoteTypes_BuiltinMatrix_MatchesNumPy()
        {
            for (int i = 0; i < Names.Length; i++)
                for (int j = 0; j < Names.Length; j++)
                {
                    np.promote_types(np.dtype(Names[i]), np.dtype(Names[j])).name.Should().Be(NumPyMatrix[i][j], $"promote_types({Names[i]}, {Names[j]})");
                    np.result_type(np.dtype(Names[i]), np.dtype(Names[j])).name.Should().Be(NumPyMatrix[i][j], $"result_type({Names[i]}, {Names[j]})");
                    // the NPTypeCode spelling is the same engine
                    np.promote_types(np.dtype(Names[i]).typecode, np.dtype(Names[j]).typecode).Should().Be(np.dtype(NumPyMatrix[i][j]).typecode);
                }
        }

        [TestMethod]
        public void PromoteTypes_IdenticalNative_ReturnsTheInput()
        {
            np.promote_types(DType.Int32, DType.Int32).Should().BeSameAs(DType.Int32);
            var d = np.dtype("M8[3s]");
            np.promote_types(d, np.dtype("M8[3s]")).Should().Be(d);
            np.promote_types(np.dtype(">i4"), np.dtype(">i4")).Should().BeSameAs(DType.Int32, "a non-native pair skips the fast path and lands on the native default");
            np.promote_types(np.dtype(">i4"), np.dtype("i8")).Should().BeSameAs(DType.Int64);
            np.promote_types(np.dtype(">f4"), np.dtype(">f4")).str.Should().Be("<f4");
        }

        [DataTestMethod]
        [DataRow("m8[s]", "i4", "<m8[s]")]
        [DataRow("i4", "m8[s]", "<m8[s]")]
        [DataRow("m8[s]", "u4", "<m8[s]")]
        [DataRow("m8[s]", "i8", "<m8[s]")]
        [DataRow("m8[s]", "?", "<m8[s]")]
        [DataRow("m8", "i4", "<m8")]
        [DataRow("m8", "i8", "<m8")]
        [DataRow("M8[s]", "m8[ms]", "<M8[ms]")]
        [DataRow("m8[ms]", "M8[s]", "<M8[ms]")]
        [DataRow("M8[s]", "M8[ms]", "<M8[ms]")]
        [DataRow("M8[ms]", "M8[s]", "<M8[ms]")]
        [DataRow("m8[s]", "m8[ms]", "<m8[ms]")]
        [DataRow("M8[Y]", "M8[D]", "<M8[D]")]
        [DataRow("M8[D]", "M8[Y]", "<M8[D]")]
        [DataRow("m8[Y]", "m8[M]", "<m8[M]")]
        [DataRow("m8[M]", "m8[Y]", "<m8[M]")]
        [DataRow("M8[Y]", "m8[D]", "<M8[D]")]
        [DataRow("m8[D]", "M8[Y]", "<M8[D]")]
        [DataRow("m8[Y]", "M8[D]", "<M8[D]")]
        [DataRow("M8[D]", "m8[Y]", "<M8[D]")]
        [DataRow("M8[M]", "m8[h]", "<M8[h]")]
        [DataRow("m8[h]", "M8[M]", "<M8[h]")]
        [DataRow("m8[M]", "M8[h]", "<M8[h]")]
        [DataRow("m8[5s]", "m8[7s]", "<m8[s]")]
        [DataRow("m8[10s]", "m8[15s]", "<m8[5s]")]
        [DataRow("m8[h]", "m8[30m]", "<m8[30m]")]
        [DataRow("M8[D]", "m8[h]", "<M8[h]")]
        [DataRow("m8[W]", "m8[D]", "<m8[D]")]
        [DataRow("m8[2W]", "m8[3D]", "<m8[D]")]
        [DataRow("m8[2W]", "m8[7D]", "<m8[7D]")]
        [DataRow("M8", "M8[s]", "<M8[s]")]
        [DataRow("M8[s]", "M8", "<M8[s]")]
        [DataRow("m8", "m8", "<m8")]
        [DataRow("M8", "M8", "<M8")]
        [DataRow("m8", "M8", "<M8")]
        [DataRow("M8", "m8", "<M8")]
        [DataRow("m8", "M8[s]", "<M8[s]")]
        [DataRow("M8[Y]", "M8[M]", "<M8[M]")]
        [DataRow("M8[M]", "M8[W]", "<M8[W]")]
        [DataRow("M8[W]", "M8[Y]", "<M8[W]")]
        [DataRow(">M8[s]", "M8[ms]", "<M8[ms]")]
        [DataRow(">M8[s]", ">M8[s]", "<M8[s]")]
        public void PromoteTypes_Datetime_IsTheUnitGcd(string a, string b, string expected)
        {
            // datetime_type_promotion: strict about the non-linear Y/M units for a timedelta operand, relaxed for a
            // datetime one — and a timedelta operand is first CAST to the datetime class (keeping its unit), which is
            // why m8[Y] × M8[D] is M8[D] while m8[Y] × m8[D] raises (next test).
            np.promote_types(np.dtype(a), np.dtype(b)).str.Should().Be(expected);
            np.result_type(np.dtype(a), np.dtype(b)).str.Should().Be(expected);
        }

        [DataTestMethod]
        [DataRow("m8[Y]", "m8[D]", "Cannot get a common metadata divisor for Numpy datetime metadata [Y] and [D] because they have incompatible nonlinear base time units.")]
        [DataRow("m8[D]", "m8[Y]", "Cannot get a common metadata divisor for Numpy datetime metadata [D] and [Y] because they have incompatible nonlinear base time units.")]
        [DataRow("m8[M]", "m8[D]", "Cannot get a common metadata divisor for Numpy datetime metadata [M] and [D] because they have incompatible nonlinear base time units.")]
        [DataRow("m8[as]", "m8[Y]", "Cannot get a common metadata divisor for Numpy datetime metadata [as] and [Y] because they have incompatible nonlinear base time units.")]
        public void PromoteTypes_TimedeltaNonlinearBarrier_RaisesTypeError(string a, string b, string message)
        {
            Action act = () => np.promote_types(np.dtype(a), np.dtype(b));
            act.Should().Throw<TypeError>().WithMessage(message);
        }

        [TestMethod]
        public void PromoteTypes_UnitOverflow_RaisesOverflow()
        {
            Action act = () => np.promote_types(np.dtype("M8[as]"), np.dtype("M8[Y]"));
            act.Should().Throw<OverflowException>().WithMessage("Integer overflow getting a common metadata divisor for NumPy datetime metadata [as] and [Y].");
        }

        [DataTestMethod]
        [DataRow("m8[s]", "u8", "TimeDelta64DType", "UInt64DType")]
        [DataRow("m8[s]", "f8", "TimeDelta64DType", "Float64DType")]
        [DataRow("m8[s]", "f2", "TimeDelta64DType", "Float16DType")]
        [DataRow("m8[s]", "c16", "TimeDelta64DType", "Complex128DType")]
        [DataRow("M8[s]", "i4", "DateTime64DType", "Int32DType")]
        [DataRow("M8[s]", "i8", "DateTime64DType", "Int64DType")]
        [DataRow("M8[s]", "?", "DateTime64DType", "BoolDType")]
        [DataRow("M8[s]", "f8", "DateTime64DType", "Float64DType")]
        [DataRow("M8", "i8", "DateTime64DType", "Int64DType")]
        public void PromoteTypes_NoCommonDType_RaisesDTypePromotionError_Verbatim(string a, string b, string clsA, string clsB)
        {
            Action act = () => np.promote_types(np.dtype(a), np.dtype(b));
            act.Should().Throw<DTypePromotionError>().WithMessage(
                $"The DTypes <class 'numpy.dtypes.{clsA}'> and <class 'numpy.dtypes.{clsB}'> do not have a common DType. " +
                "For example they cannot be stored in a single array unless the dtype is `object`.");
            typeof(DTypePromotionError).Should().BeDerivedFrom<TypeError>("numpy.exceptions.DTypePromotionError is a TypeError");
        }

        [TestMethod]
        public void PromoteTypes_NumSharpOnlyClasses()
        {
            np.promote_types(np.dtype("decimal"), np.dtype("char")).Should().BeSameAs(DType.Decimal);
            np.promote_types(np.dtype("char"), np.dtype("f2")).Should().BeSameAs(DType.Single, "Char ranks as uint16");
            Action td = () => np.promote_types(np.dtype("m8[s]"), np.dtype("decimal"));
            td.Should().Throw<DTypePromotionError>().WithMessage("The DTypes <class 'numpy.dtypes.TimeDelta64DType'> and <class 'numsharp.dtypes.DecimalDType'> do not have a common DType.*");
            Action str = () => np.promote_types(np.dtype("i4"), np.dtype("String"));
            str.Should().Throw<DTypePromotionError>();
        }

        // ---- result_type: NEP 50 weak literals ------------------------------------------------------------------------

        [TestMethod]
        public void ResultType_WeakLiterals_FollowNep50()
        {
            np.result_type(np.dtype("i1"), 300).name.Should().Be("int8", "a weak int adopts the array dtype with no value check");
            np.result_type(np.dtype("i1"), -300).name.Should().Be("int8");
            np.result_type(np.dtype("u1"), -1).name.Should().Be("uint8");
            np.result_type(np.dtype("i1"), 1.0).name.Should().Be("float64");
            np.result_type(np.dtype("f4"), 1e300).name.Should().Be("float32");
            np.result_type(np.dtype("f2"), 1).name.Should().Be("float16");
            np.result_type(np.dtype("f2"), 1.5).name.Should().Be("float16");
            np.result_type(np.dtype("?"), 1).name.Should().Be("int64", "bool + weak int is the default integer");
            np.result_type(np.dtype("?"), 1.5).name.Should().Be("float64");
            np.result_type(np.dtype("?"), Complex.ImaginaryOne).name.Should().Be("complex128");
            np.result_type(np.dtype("i8"), Complex.ImaginaryOne).name.Should().Be("complex128");
            np.result_type(np.dtype("f8"), Complex.ImaginaryOne).name.Should().Be("complex128");
            np.result_type(np.dtype("m8[s]"), 5).str.Should().Be("<m8[s]", "timedelta absorbs a weak int");
            np.result_type(np.dtype("m8[s]"), true).str.Should().Be("<m8[s]");
            np.result_type(np.dtype("m8[s]"), 5L).str.Should().Be("<m8[s]");
            np.result_type(np.dtype("m8[s]"), (byte)5).str.Should().Be("<m8[s]");
        }

        [TestMethod]
        public void ResultType_LoneLiterals_FallBackToDefaults()
        {
            np.result_type(new object[] { 1 }).name.Should().Be("int64");
            np.result_type(new object[] { 300 }).name.Should().Be("int64");
            np.result_type(new object[] { 1.0 }).name.Should().Be("float64");
            np.result_type(new object[] { 1.5f }).name.Should().Be("float64");
            np.result_type(new object[] { Complex.ImaginaryOne }).name.Should().Be("complex128");
            np.result_type(new object[] { true }).name.Should().Be("bool", "bool is strong");
            np.result_type(new object[] { 'x' }).name.Should().Be("char");
            np.result_type(new object[] { 1.5m }).name.Should().Be("decimal");
            np.result_type(new object[] { (Half)1 }).name.Should().Be("float16");
            np.result_type(new object[] { 1, 2 }).name.Should().Be("int64");
            np.result_type(new object[] { 1, 1.0 }).name.Should().Be("float64");
            np.result_type(new object[] { 1, 1.0, Complex.ImaginaryOne }).name.Should().Be("complex128");
        }

        [TestMethod]
        public void ResultType_Arrays_AreStrong_ZeroDIncluded()
        {
            var arr = np.zeros(new Shape(3), typeof(sbyte));
            np.result_type(new object[] { arr, 300 }).name.Should().Be("int8");
            np.result_type(arr.dtype, 300).name.Should().Be("int8");
            np.result_type(new object[] { arr, 1.5 }).name.Should().Be("float64");
            np.result_type(new object[] { NDArray.Scalar((sbyte)1), NDArray.Scalar(300L) }).name.Should().Be("int64", "a 0-d array is a full participant (NumPy 2.x)");
            np.result_type(new NDArray[] { np.zeros(new Shape(4, 5), typeof(int)), NDArray.Scalar(5L) }).Should().Be(NPTypeCode.Int64);
            np.result_type(np.zeros(new Shape(4, 5), typeof(int)), NDArray.Scalar(5L)).Should().Be(NPTypeCode.Int64);
            np.result_type(NDArray.Scalar(5L), np.zeros(new Shape(4, 5), typeof(int))).Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void ResultType_Sequences_AreOrderIndependent()
        {
            np.result_type("i1", "i1", "f8", "i1").name.Should().Be("float64", "the old pairwise fold dropped every other pair and answered int8");
            np.result_type(NPTypeCode.SByte, NPTypeCode.SByte, NPTypeCode.Double, NPTypeCode.SByte).Should().Be(NPTypeCode.Double);
            np.result_type("u8", "i8").name.Should().Be("float64");
            np.result_type("u4", "i4").name.Should().Be("int64");
            np.result_type("i1", "u1").name.Should().Be("int16");
            np.result_type("i1", "u2", "f2").name.Should().Be("float32");
            np.result_type("i8", "u8", "f2").name.Should().Be("float64");
            np.result_type("u1", "i1", "u1", "i1", "u1", "i1").name.Should().Be("int16");
            np.result_type("M8[s]", "m8[ms]", "m8[us]").str.Should().Be("<M8[us]");
            np.result_type("M8[s]", "M8[ms]").str.Should().Be("<M8[ms]");
            np.result_type("i8", "m8[s]").str.Should().Be("<m8[s]");

            var codes = new[] { NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.Half, NPTypeCode.UInt32, NPTypeCode.Double };
            var expected = np.result_type(codes);
            foreach (var perm in new[] { Enumerable.Reverse(codes).ToArray(), new[] { codes[3], codes[0], codes[5], codes[1], codes[4], codes[2] } })
                np.result_type(perm).Should().Be(expected);
        }

        [TestMethod]
        public void ResultType_SingleOperand_IsCanonical()
        {
            np.result_type(np.dtype(">i4")).str.Should().Be("<i4");
            np.result_type(np.dtype(">i4"), np.dtype(">i4")).str.Should().Be("<i4");
            np.result_type(np.dtypes.Float64DType).Should().BeSameAs(DType.Double, "a bare class resolves to its default descriptor");
            np.result_type(np.dtypes.Float64DType, np.dtype("i4")).Should().BeSameAs(DType.Double);
            np.result_type(np.dtypes.DateTime64DType, np.dtype("m8[ms]")).str.Should().Be("<M8[ms]");
        }

        [TestMethod]
        public void ResultType_Errors_Verbatim()
        {
            Action none = () => np.result_type(Array.Empty<DType>());
            none.Should().Throw<ValueError>().WithMessage("at least one array or dtype is required");
            Action noneObj = () => np.result_type(Array.Empty<object>());
            noneObj.Should().Throw<ValueError>().WithMessage("at least one array or dtype is required");

            Action a = () => np.result_type("i4", "M8[s]", "f8");
            a.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.Float64DType'> could not be promoted by <class 'numpy.dtypes.DateTime64DType'>. " +
                "This means that no common DType exists for the given inputs. For example they cannot be stored in a single array " +
                "unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.Int32DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Float64DType'>)");

            Action b = () => np.result_type("M8[s]", "f8", "i4");
            b.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.DateTime64DType'> could not be promoted by <class 'numpy.dtypes.Float64DType'>. " +
                "This means that no common DType exists for the given inputs. For example they cannot be stored in a single array " +
                "unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.Int32DType'>)");

            Action c = () => np.result_type(np.dtype("M8[s]"), 5);
            c.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.DateTime64DType'> could not be promoted by <class 'numpy.dtypes._PyLongDType'>. " +
                "This means that no common DType exists for the given inputs. For example they cannot be stored in a single array " +
                "unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes._PyLongDType'>)");

            Action d = () => np.result_type(np.dtype("m8[s]"), 5.0);
            d.Should().Throw<DTypePromotionError>().WithMessage("The DType <class 'numpy.dtypes.TimeDelta64DType'> could not be promoted by <class 'numpy.dtypes._PyFloatDType'>.*");

            Action e = () => np.result_type("m8[Y]", "m8[D]");
            e.Should().Throw<TypeError>().WithMessage("Cannot get a common metadata divisor for Numpy datetime metadata [Y] and [D] because they have incompatible nonlinear base time units.");

            Action f = () => np.result_type(new object[] { new object() });
            f.Should().Throw<TypeError>().WithMessage("Cannot interpret '*' as a data type");
        }

        [TestMethod]
        public void ResultType_NumSharpOnlyClasses_BehaveAsTheirMasquerade()
        {
            np.result_type(np.dtype("char"), 1.5).name.Should().Be("float64", "char is an unsigned integer: uint16 + weak float is float64");
            np.result_type(np.dtype("char"), 5).name.Should().Be("char");
            np.result_type(np.dtype("char"), Complex.ImaginaryOne).name.Should().Be("complex128");
            np.result_type(np.dtype("decimal"), 1.5).name.Should().Be("decimal");
            np.result_type(np.dtype("decimal"), 5).name.Should().Be("decimal");
            np.result_type(np.dtype("decimal"), Complex.ImaginaryOne).name.Should().Be("complex128");
        }

        [TestMethod]
        [Misaligned]
        public void ResultType_WeakComplexWithFloat32_IsComplex128()
        {
            // NumPy: result_type(float32, 1j) is complex64 and result_type(float16, 1j) is complex64. NumSharp has a single
            // complex width (System.Numerics.Complex), so both collapse onto complex128 — the same collapse np.dtype('c8') documents.
            np.result_type(np.dtype("f4"), Complex.ImaginaryOne).name.Should().Be("complex128");
            np.result_type(np.dtype("f2"), Complex.ImaginaryOne).name.Should().Be("complex128");
        }

        // ---- the class-level engine ------------------------------------------------------------------------------------

        [TestMethod]
        public void CommonDType_ClassLevel()
        {
            DTypePromotion.CommonDType(np.dtypes.Int8DType, np.dtypes.UInt8DType).Should().BeSameAs(np.dtypes.Int16DType);
            DTypePromotion.CommonDType(np.dtypes.UInt64DType, np.dtypes.Int64DType).Should().BeSameAs(np.dtypes.Float64DType);
            DTypePromotion.CommonDType(np.dtypes.Float16DType, np.dtypes.Float16DType).Should().BeSameAs(np.dtypes.Float16DType);
            DTypePromotion.CommonDType(np.dtypes.DateTime64DType, np.dtypes.TimeDelta64DType).Should().BeSameAs(np.dtypes.DateTime64DType);
            DTypePromotion.CommonDType(np.dtypes.TimeDelta64DType, np.dtypes.Int32DType).Should().BeSameAs(np.dtypes.TimeDelta64DType);
            DTypePromotion.CommonDType(DTypeRegistry.PyLong, np.dtypes.BoolDType).Should().BeSameAs(np.dtypes.Int64DType);
            DTypePromotion.CommonDType(DTypeRegistry.PyFloat, np.dtypes.Int8DType).Should().BeSameAs(np.dtypes.Float64DType);
            DTypePromotion.CommonDType(DTypeRegistry.PyFloat, DTypeRegistry.PyLong).Should().BeSameAs(DTypeRegistry.PyFloat);
            DTypePromotion.CommonDType(DTypeRegistry.PyComplex, DTypeRegistry.PyFloat).Should().BeSameAs(DTypeRegistry.PyComplex);
            DTypePromotion.CommonDType(np.dtypes.Float32DType, DTypeRegistry.PyComplex).Should().BeSameAs(np.dtypes.Complex128DType);
            Action act = () => DTypePromotion.CommonDType(np.dtypes.DateTime64DType, np.dtypes.Float64DType);
            act.Should().Throw<DTypePromotionError>();
            // NotImplemented in both directions is null at the slot level
            np.dtypes.Int32DType.CommonDType(np.dtypes.DateTime64DType).Should().BeNull("the class with the larger type number answers");
            np.dtypes.DateTime64DType.CommonDType(np.dtypes.Int32DType).Should().BeNull();
        }

        [TestMethod]
        public void PromoteDTypeSequence_IsOrderIndependent()
        {
            var metas = new DTypeMeta[] { np.dtypes.Int8DType, np.dtypes.UInt32DType, np.dtypes.Float16DType, np.dtypes.Int16DType };
            var expected = DTypePromotion.PromoteDTypeSequence(metas);
            expected.Should().BeSameAs(np.dtypes.Float64DType);
            DTypePromotion.PromoteDTypeSequence(Enumerable.Reverse(metas).ToArray()).Should().BeSameAs(expected);
            DTypePromotion.PromoteDTypeSequence(new DTypeMeta[] { np.dtypes.Int8DType }).Should().BeSameAs(np.dtypes.Int8DType);
            DTypePromotion.PromoteDTypeSequence(new DTypeMeta[] { np.dtypes.Int8DType, np.dtypes.Int8DType, np.dtypes.Int8DType }).Should().BeSameAs(np.dtypes.Int8DType);
            Action empty = () => DTypePromotion.PromoteDTypeSequence(Array.Empty<DTypeMeta>());
            empty.Should().Throw<ValueError>();
        }

        [TestMethod]
        public void CastDescrToDType_AdaptsInstances()
        {
            DTypePromotion.CastDescrToDType(np.dtype("m8[Y]"), np.dtypes.DateTime64DType).str.Should().Be("<M8[Y]", "the timedelta→datetime cast keeps the unit");
            DTypePromotion.CastDescrToDType(np.dtype("i4"), np.dtypes.TimeDelta64DType).str.Should().Be("<m8", "an integer lands on the generic unit");
            DTypePromotion.CastDescrToDType(np.dtype("M8[ns]"), np.dtypes.DateTime64DType).str.Should().Be("<M8[ns]");
            DTypePromotion.CastDescrToDType(np.dtype("i4"), np.dtypes.Float64DType).Should().BeSameAs(DType.Double, "a non-parametric target is always its default");
            Action act = () => DTypePromotion.CastDescrToDType(np.dtype("i4"), DTypeRegistry.Str);
            act.Should().Throw<TypeError>().WithMessage("cannot cast dtype int32 to <class 'numpy.dtypes.StrDType'>.");
            DTypePromotion.CastToDTypeAndPromoteDescriptors(new[] { np.dtype("m8[s]"), np.dtype("m8[ms]"), np.dtype("M8[us]") }, np.dtypes.DateTime64DType).str.Should().Be("<M8[us]");
        }
    }
}
