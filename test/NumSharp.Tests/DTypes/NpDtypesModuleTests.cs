using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The <c>np.dtypes</c> facade (NEP 56 / NumPy 1.25+), the <see cref="DTypeMeta"/> classes and the
    ///     <see cref="DTypeRegistry"/> — names, aliases, flags, instantiation, lookups. Probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class NpDtypesModuleTests
    {
        [DataTestMethod]
        [DataRow("BoolDType", "?", 0)]
        [DataRow("Int8DType", "i1", 1)]
        [DataRow("UInt8DType", "u1", 2)]
        [DataRow("Int16DType", "i2", 3)]
        [DataRow("UInt16DType", "u2", 4)]
        [DataRow("Int32DType", "i4", 5)]
        [DataRow("UInt32DType", "u4", 6)]
        [DataRow("Int64DType", "i8", 7)]
        [DataRow("UInt64DType", "u8", 8)]
        [DataRow("Float16DType", "e", 23)]
        [DataRow("Float32DType", "f", 11)]
        [DataRow("Float64DType", "d", 12)]
        [DataRow("Complex128DType", "D", 15)]
        [DataRow("DateTime64DType", "M8", 21)]
        [DataRow("TimeDelta64DType", "m8", 22)]
        public void Classes_AreTheClassesOfTheirDescriptors(string className, string spelling, int typeNum)
        {
            // The four C-integer classes carry a platform-dependent type_num (LLP64 Windows: 7/8/9/10;
            // LP64: 5/6/7/8). The DataRow carries the LP64 values; remap to the local platform, matching
            // np.dtype('int32'/'int64').num on the host (see DTypeRegistry's CLongIs32Bit).
            bool cLongIs32 = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows) || IntPtr.Size != 8;
            if (cLongIs32)
                typeNum = className switch
                {
                    "Int32DType" => 7,
                    "UInt32DType" => 8,
                    "Int64DType" => 9,
                    "UInt64DType" => 10,
                    _ => typeNum
                };

            var meta = (DTypeMeta)typeof(np.dtypes).GetProperty(className).GetValue(null);
            meta.Name.Should().Be(className);
            meta.FullName.Should().Be("numpy.dtypes." + className);
            meta.type_num.Should().Be(typeNum);
            np.dtype(spelling).Meta.Should().BeSameAs(meta, "type(np.dtype(spelling)) is the class");
            DTypeRegistry.FromName(className).Should().BeSameAs(meta);
            DTypeRegistry.FromTypeNum(typeNum).Should().BeSameAs(meta);
            DTypeRegistry.All.Should().Contain(meta);
        }

        [TestMethod]
        public void Aliases_MatchNumPy()
        {
            np.dtypes.ByteDType.Should().BeSameAs(np.dtypes.Int8DType);
            np.dtypes.UByteDType.Should().BeSameAs(np.dtypes.UInt8DType);
            np.dtypes.ShortDType.Should().BeSameAs(np.dtypes.Int16DType);
            np.dtypes.UShortDType.Should().BeSameAs(np.dtypes.UInt16DType);
            np.dtypes.IntDType.Should().BeSameAs(np.dtypes.Int32DType);
            np.dtypes.UIntDType.Should().BeSameAs(np.dtypes.UInt32DType);
            np.dtypes.LongLongDType.Should().BeSameAs(np.dtypes.Int64DType);
            np.dtypes.ULongLongDType.Should().BeSameAs(np.dtypes.UInt64DType);
            np.dtypes.LongDoubleDType.Should().BeSameAs(np.dtypes.Float64DType, "no extended precision, as on NumPy's win-amd64 build");
            np.dtypes.CLongDoubleDType.Should().BeSameAs(np.dtypes.Complex128DType);

            // C long: 32-bit on Windows and 32-bit hosts, 64-bit on LP64 Unix — the same rule np.dtype('l') follows.
            np.dtypes.LongDType.Should().BeSameAs(np.dtype("l").Meta);
            np.dtypes.ULongDType.Should().BeSameAs(np.dtype("L").Meta);
            DTypeRegistry.FromName("LongDType").Should().BeSameAs(np.dtypes.LongDType);
            DTypeRegistry.FromName("ByteDType").Should().BeSameAs(np.dtypes.Int8DType);
            np.dtypes.Int8DType.Aliases.Should().Equal("ByteDType");
            np.dtypes.Int32DType.Aliases.Should().Contain("IntDType");
        }

        [TestMethod]
        public void NumSharpOnlyClasses()
        {
            np.dtypes.DecimalDType.FullName.Should().Be("numsharp.dtypes.DecimalDType");
            np.dtypes.DecimalDType.type_num.Should().Be(256);
            np.dtypes.DecimalDType.Kind.Should().Be('f');
            np.dtypes.DecimalDType.type.Should().Be(typeof(decimal));
            np.dtypes.CharDType.FullName.Should().Be("numsharp.dtypes.CharDType");
            np.dtypes.CharDType.type_num.Should().Be(257);
            np.dtypes.CharDType.Kind.Should().Be('u');
            np.dtypes.CharDType.TypeChar.Should().Be('c');
            DTypeRegistry.Str.Name.Should().Be("StrDType");
            DTypeRegistry.Str.type_num.Should().Be(19);
            DTypeRegistry.Str.HasStorage.Should().BeTrue("the vestigial NPTypeCode.String is its code");
            typeof(np.dtypes).GetProperty("StrDType").Should().BeNull("the vestigial string slot is not exported on np.dtypes");
        }

        [TestMethod]
        public void Flags_MatchNumPy()
        {
            (np.dtypes.Int8DType._abstract, np.dtypes.Int8DType._legacy, np.dtypes.Int8DType._parametric, np.dtypes.Int8DType._is_numeric)
                .Should().Be((false, true, false, true));
            (np.dtypes.BoolDType._abstract, np.dtypes.BoolDType._legacy, np.dtypes.BoolDType._parametric, np.dtypes.BoolDType._is_numeric)
                .Should().Be((false, true, false, true));
            (np.dtypes.DateTime64DType._abstract, np.dtypes.DateTime64DType._legacy, np.dtypes.DateTime64DType._parametric, np.dtypes.DateTime64DType._is_numeric)
                .Should().Be((false, true, true, false));
            (np.dtypes.TimeDelta64DType._abstract, np.dtypes.TimeDelta64DType._legacy, np.dtypes.TimeDelta64DType._parametric, np.dtypes.TimeDelta64DType._is_numeric)
                .Should().Be((false, true, true, false));
            (DTypeRegistry.PyLong._abstract, DTypeRegistry.PyLong._legacy, DTypeRegistry.PyLong._parametric, DTypeRegistry.PyLong._is_numeric)
                .Should().Be((true, false, false, false));
            np.dtypes.Int8DType.flags.Should().Be(DTypeFlags.NPY_DT_LEGACY | DTypeFlags.NPY_DT_NUMERIC);
            ((int)DTypeFlags.NPY_DT_LEGACY, (int)DTypeFlags.NPY_DT_ABSTRACT, (int)DTypeFlags.NPY_DT_PARAMETRIC, (int)DTypeFlags.NPY_DT_NUMERIC).Should().Be((1, 2, 4, 8), "NumPy's NPY_DT_* bits");
            np.dtypes.DateTime64DType.HasStorage.Should().BeFalse("Stage A: descriptor-level only");
            np.dtypes.Int8DType.HasStorage.Should().BeTrue();
            np.dtypes.Int8DType.TypeCode.Should().Be(NPTypeCode.SByte);
            np.dtypes.DateTime64DType.TypeCode.Should().Be(NPTypeCode.Empty);
            np.dtypes.DateTime64DType.IsTimedelta.Should().BeFalse();
            np.dtypes.TimeDelta64DType.IsTimedelta.Should().BeTrue();
            np.dtypes.Int8DType.HasReferences.Should().BeFalse();
            np.dtypes.Int8DType.get_clear_loop(DType.SByte).Should().BeNull();
            np.dtypes.Int8DType.get_fill_zero_loop(DType.SByte).Should().BeNull();
        }

        [TestMethod]
        public void ClassSurface()
        {
            np.dtypes.Float64DType.type.Should().Be(typeof(double));
            np.dtypes.Float64DType.ScalarName.Should().Be("float64");
            np.dtypes.Float64DType.Kind.Should().Be('f');
            np.dtypes.Float64DType.TypeChar.Should().Be('d');
            np.dtypes.Float64DType.ItemSize.Should().Be(8);
            np.dtypes.Float64DType.Alignment.Should().Be(8);
            np.dtypes.Complex128DType.Alignment.Should().Be(8);
            np.dtypes.Float64DType.singleton.Should().BeSameAs(DType.Double);
            np.dtypes.Float64DType.default_descr().Should().BeSameAs(DType.Double);
            np.dtypes.DateTime64DType.type.Should().BeNull();
            np.dtypes.DateTime64DType.ScalarName.Should().Be("datetime64");
            np.dtypes.DateTime64DType.default_descr().Should().Be(np.dtype("M8"));
            np.dtypes.DateTime64DType.default_descr().Should().NotBeSameAs(np.dtypes.DateTime64DType.singleton, "NumPy copies the singleton");
            np.dtypes.DateTime64DType.Descr(NPY_DATETIMEUNIT.NPY_FR_ns).Should().Be(np.dtype("M8[ns]"));
            np.dtypes.TimeDelta64DType.Descr(new DatetimeMetaData(NPY_DATETIMEUNIT.NPY_FR_W, 2)).Should().Be(np.dtype("m8[2W]"));
            np.dtypes.Float64DType.is_known_scalar_type(typeof(double)).Should().BeTrue();
            np.dtypes.Float64DType.is_known_scalar_type(typeof(double[])).Should().BeTrue();
            np.dtypes.Float64DType.is_known_scalar_type(typeof(float)).Should().BeFalse();
            DTypeRegistry.PyLong.is_known_scalar_type(typeof(int)).Should().BeTrue();
            DTypeRegistry.PyLong.is_known_scalar_type(typeof(bool)).Should().BeFalse("bool is strong");
            DTypeRegistry.PyFloat.is_known_scalar_type(typeof(float)).Should().BeTrue();
            DTypeRegistry.PyComplex.is_known_scalar_type(typeof(Complex)).Should().BeTrue();
            DTypeRegistry.PyLong.default_descr().Should().BeSameAs(DType.Int64);
            DTypeRegistry.PyFloat.default_descr().Should().BeSameAs(DType.Double);
            DTypeRegistry.PyComplex.default_descr().Should().BeSameAs(DType.Complex);
            DTypeRegistry.PyLong.PyKind.Should().Be(PyScalarKind.Int);
        }

        [TestMethod]
        public void Instantiate_IsNumPysClassCall()
        {
            np.dtypes.Int8DType.Instantiate().Should().BeSameAs(np.dtype("i1"));
            np.dtypes.Int8DType.Instantiate().ToString(true).Should().Be("dtype('int8')");
            Action dt = () => np.dtypes.DateTime64DType.Instantiate();
            dt.Should().Throw<TypeError>().WithMessage(
                "Preliminary-API: Flexible/Parametric legacy DType '<class 'numpy.dtypes.DateTime64DType'>' can only be instantiated using `np.dtype(...)`");
            Action td = () => np.dtypes.TimeDelta64DType.Instantiate();
            td.Should().Throw<TypeError>();
            Action abs = () => DTypeRegistry.PyLong.Instantiate();
            abs.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void Registry_Lookups()
        {
            DTypeRegistry.FromTypeCode(NPTypeCode.Int32).Should().BeSameAs(np.dtypes.Int32DType);
            DTypeRegistry.FromTypeCode(NPTypeCode.Empty).Should().BeNull();
            DTypeRegistry.FromScalarType(typeof(int)).Should().BeSameAs(np.dtypes.Int32DType);
            DTypeRegistry.FromScalarType(typeof(int[])).Should().BeSameAs(np.dtypes.Int32DType);
            DTypeRegistry.FromScalarType(typeof(Complex)).Should().BeSameAs(np.dtypes.Complex128DType);
            DTypeRegistry.FromScalarType(typeof(Half)).Should().BeSameAs(np.dtypes.Float16DType);
            DTypeRegistry.FromScalarType(typeof(string)).Should().BeSameAs(DTypeRegistry.Str);
            DTypeRegistry.FromScalarType(typeof(DateTime)).Should().BeNull("DateTime is not a NumSharp scalar (the dtype has no storage yet)");
            DTypeRegistry.FromScalarType(typeof(NpDtypesModuleTests)).Should().BeNull();
            DTypeRegistry.FromScalarType(null).Should().BeNull();
            DTypeRegistry.FromTypeNum(21).Should().BeSameAs(np.dtypes.DateTime64DType);
            DTypeRegistry.FromTypeNum(999).Should().BeNull();
            DTypeRegistry.FromName("_PyLongDType").Should().BeSameAs(DTypeRegistry.PyLong);
            DTypeRegistry.FromName("nope").Should().BeNull();
            DTypeRegistry.FromName(null).Should().BeNull();
            DTypeRegistry.All.Count.Should().BeGreaterThanOrEqualTo(21, "16 legacy + 2 datetime + 3 abstract");
            DTypeRegistry.All.Select(m => m.type_num).Where(n => n >= 0).Should().OnlyHaveUniqueItems();
        }

        [TestMethod]
        public void Registry_Register_RejectsDuplicates()
        {
            Action dup = () => DTypeRegistry.Register(new DuplicateMeta());
            dup.Should().Throw<InvalidOperationException>().WithMessage("*type number 3*");
            Action nullMeta = () => DTypeRegistry.Register(null);
            nullMeta.Should().Throw<ArgumentNullException>();
        }

        // Uses NPY_SHORT (3) — Int16's number, which is registered on EVERY platform (unlike NPY_INT=5, which is
        // unused on LLP64/Windows where Int32 is NPY_LONG=7), so the duplicate-rejection assertion is platform-stable.
        private sealed class DuplicateMeta : DTypeMeta
        {
            public DuplicateMeta() : base("DuplicateDType", "test", 3, typeof(int), "dup", NPTypeCode.Empty, 'i', 'x', 4, 4, DTypeFlags.NPY_DT_LEGACY) { }
            public override DType default_descr() => throw new NotSupportedException();
            public override DTypeMeta common_dtype(DTypeMeta other) => null;
        }
    }
}
