using System;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #549, "[NEP42 NEP43 NEP55] DType System and String Support"
    ///     (https://github.com/SciSharp/NumSharp/issues/549): NumPy's two-level dtype model (dtype classes in
    ///     <c>np.dtypes</c> and their instances), <c>common_dtype</c> / <c>ensure_canonical</c> / <c>castingimpls</c>, and
    ///     NEP 55's <c>StringDType</c> with its NA object and coercion rule. Expected values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue549
    {
        /// <summary>
        ///     A dtype instance knows its class, and the classes carry NumPy's names: <c>float64</c>'s class is
        ///     <c>np.dtypes.Float64DType</c>, <c>U3</c>'s is <c>StrDType</c>, <c>T</c>'s is <c>StringDType</c>.
        /// </summary>
        [TestMethod]
        public void DtypeInstances_BelongToNumPysDtypeClasses()
        {
            ReferenceEquals(np.dtype("float64").Meta, np.dtypes.Float64DType).Should().BeTrue();
            ReferenceEquals(np.dtype("U3").Meta, np.dtypes.StrDType).Should().BeTrue();
            np.dtypes.Float64DType.Name.Should().Be("Float64DType");
            np.dtypes.StrDType.Name.Should().Be("StrDType");
            np.dtype("T").Meta.Name.Should().Be("StringDType");
        }

        /// <summary>
        ///     NEP 42's class-level promotion follows NumPy's <c>common_dtype</c> slot contract: the class with the larger
        ///     type number answers (<c>Float64DType.common_dtype(Int64DType)</c> is <c>Float64DType</c>,
        ///     <c>UInt8DType.common_dtype(Int8DType)</c> is <c>Int16DType</c>, NumPy's <c>result_type</c> answers), and the
        ///     smaller one defers with NumPy's <c>NotImplemented</c> (null here), so the caller asks the mirrored question, as
        ///     NumPy's <c>default_builtin_common_dtype</c> does.
        /// </summary>
        [TestMethod]
        public void CommonDtype_IsNumPysPromotionSlot()
        {
            ReferenceEquals(np.dtypes.Float64DType.common_dtype(np.dtypes.Int64DType), np.dtypes.Float64DType).Should().BeTrue();
            ReferenceEquals(np.dtypes.UInt8DType.common_dtype(np.dtypes.Int8DType), np.dtypes.Int16DType).Should().BeTrue();
            np.dtypes.Int8DType.common_dtype(np.dtypes.UInt8DType).Should().BeNull(
                "int8's type number is below uint8's, so NumPy's slot returns NotImplemented and lets uint8 answer");
        }

        /// <summary>
        ///     <c>ensure_canonical</c> turns a big-endian <c>&gt;i4</c> into the native <c>&lt;i4</c>, as NumPy's
        ///     <c>newbyteorder('=')</c> does on a little-endian host.
        /// </summary>
        [TestMethod]
        public void EnsureCanonical_GivesTheNativeByteOrder()
        {
            Assert.IsTrue(BitConverter.IsLittleEndian, "NumSharp's supported hosts are little-endian");
            np.dtype(">i4").byteorder.Should().Be('>');
            np.dtypes.Int32DType.ensure_canonical(np.dtype(">i4")).str.Should().Be("<i4");
        }

        /// <summary>
        ///     NEP 43's casting table: <c>Float64DType</c> has a casting implementation to <c>Int64DType</c>, and once
        ///     resolved it is listed in <c>castingimpls</c>.
        /// </summary>
        [TestMethod]
        public void CastingImpls_ListTheResolvedCast()
        {
            var impl = np.dtypes.Float64DType.GetCastingImpl(np.dtypes.Int64DType);

            impl.Should().NotBeNull();
            np.dtypes.Float64DType.castingimpls.ContainsKey(np.dtypes.Int64DType).Should().BeTrue();
        }

        /// <summary>
        ///     NEP 55: a <c>StringDType(na_object: NaN)</c> array holds a missing value that <c>np.isnan</c> finds
        ///     (<c>[False, True, False]</c>), and <c>coerce: false</c> refuses a non-string element with NumPy's text while the
        ///     default coerces it (<c>['a', '1']</c>).
        /// </summary>
        [TestMethod]
        public void StringDType_NaObjectAndCoercion()
        {
            var withNa = np.array(new object[] { "hello", double.NaN, "world" }, dtype: np.dtypes.StringDType(na_object: double.NaN));
            np.array_repr(np.isnan(withNa)).Should().Be("array([False,  True, False])");

            new Action(() => np.array(new object[] { "a", 1 }, dtype: np.dtypes.StringDType(coerce: false))).Should().Throw<ValueError>()
                .WithMessage("StringDType only allows string data when string coercion is disabled.");
            np.array_repr(np.array(new object[] { "a", 1 }, dtype: np.dtypes.StringDType()))
                .Should().Be("array(['a', '1'], dtype=StringDType())");
        }
    }
}
