using System;
using System.Linq;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The NEP 43 casting engine (<see cref="DTypeCasting"/>, the <see cref="CastingImpl"/> registry) behind
    ///     <c>np.can_cast</c>: the builtin cast levels, byte order, the datetime unit rules, and NumPy's verbatim errors.
    ///     All expectations probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class DTypeCastingTests
    {
        private static readonly string[] Units = { "generic", "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as" };
        private static readonly string[] Rules = { "no", "equiv", "safe", "same_kind", "unsafe" };

        private static string Spell(string type, string unit) => unit == "generic" ? type : $"{type}[{unit}]";

        private static string Row(string type, string src, string rule)
            => string.Concat(Units.Select(dst => np.can_cast(np.dtype(Spell(type, src)), np.dtype(Spell(type, dst)), rule) ? '1' : '0'));

        private static readonly string[] Identity = Enumerable.Range(0, 14).Select(i => new string('0', i) + "1" + new string('0', 13 - i)).ToArray();

        // np.can_cast(M8[src], M8[dst], rule) over the 14 units, NumPy 2.4.2 verbatim (rows = src, columns = dst).
        private static readonly string[] DatetimeSafe =
        {
            "11111111111111", "01111111111111", "00111111111111", "00011111111000", "00001111111000", "00000111111100", "00000011111110",
            "00000001111110", "00000000111111", "00000000011111", "00000000001111", "00000000000111", "00000000000011", "00000000000001",
        };

        private static readonly string[] TimedeltaSafe =
        {
            "11111111111111", "01100000000000", "00100000000000", "00011111111000", "00001111111000", "00000111111100", "00000011111110",
            "00000001111110", "00000000111111", "00000000011111", "00000000001111", "00000000000111", "00000000000011", "00000000000001",
        };

        private static readonly string[] DatetimeSameKind = new[] { "11111111111111" }.Concat(Enumerable.Repeat("01111111111111", 13)).ToArray();

        private static readonly string[] TimedeltaSameKind =
            new[] { "11111111111111", "01100000000000", "01100000000000" }.Concat(Enumerable.Repeat("00011111111111", 11)).ToArray();

        [TestMethod]
        public void CanCast_DatetimeUnits_MatchNumPyMatrices()
        {
            for (int i = 0; i < Units.Length; i++)
            {
                Row("M8", Units[i], "no").Should().Be(Identity[i], $"M8 no {Units[i]}");
                Row("m8", Units[i], "no").Should().Be(Identity[i], $"m8 no {Units[i]}");
                Row("M8", Units[i], "equiv").Should().Be(Identity[i], $"M8 equiv {Units[i]}");
                Row("m8", Units[i], "equiv").Should().Be(Identity[i], $"m8 equiv {Units[i]}");
                Row("M8", Units[i], "safe").Should().Be(DatetimeSafe[i], $"M8 safe {Units[i]}");
                Row("m8", Units[i], "safe").Should().Be(TimedeltaSafe[i], $"m8 safe {Units[i]}");
                Row("M8", Units[i], "same_kind").Should().Be(DatetimeSameKind[i], $"M8 same_kind {Units[i]}");
                Row("m8", Units[i], "same_kind").Should().Be(TimedeltaSameKind[i], $"m8 same_kind {Units[i]}");
                Row("M8", Units[i], "unsafe").Should().Be(new string('1', 14), $"M8 unsafe {Units[i]}");
                Row("m8", Units[i], "unsafe").Should().Be(new string('1', 14), $"m8 unsafe {Units[i]}");
            }
        }

        [TestMethod]
        public void CanCast_DatetimeMultipliers()
        {
            np.can_cast(np.dtype("M8[s]"), np.dtype("M8[1000ms]"), "no").Should().BeFalse();
            np.can_cast(np.dtype("M8[1000ms]"), np.dtype("M8[s]"), "no").Should().BeTrue("the exact 1000-fold metric prefix fold is a no-op cast");
            np.can_cast(np.dtype("M8[2s]"), np.dtype("M8[s]"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("M8[s]"), np.dtype("M8[2s]"), "safe").Should().BeFalse();
            np.can_cast(np.dtype("M8[2s]"), np.dtype("M8[3s]"), "safe").Should().BeFalse();
            np.can_cast(np.dtype("M8[2s]"), np.dtype("M8[3s]"), "same_kind").Should().BeTrue();
            np.can_cast(np.dtype("M8[6s]"), np.dtype("M8[3s]"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("M8[D]"), np.dtype("M8[36h]"), "safe").Should().BeFalse();
            np.can_cast(np.dtype("M8[D]"), np.dtype("M8[12h]"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("M8[Y]"), np.dtype("M8[D]"), "safe").Should().BeTrue("datetime is relaxed about the non-linear units");
            np.can_cast(np.dtype("m8[Y]"), np.dtype("m8[D]"), "safe").Should().BeFalse("timedelta enforces the years/months barrier");
            np.can_cast(np.dtype("m8[Y]"), np.dtype("m8[D]"), "same_kind").Should().BeFalse();
            np.can_cast(np.dtype("m8[Y]"), np.dtype("m8[M]"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("m8[M]"), np.dtype("m8[Y]"), "safe").Should().BeFalse();
            np.can_cast(np.dtype("m8[2Y]"), np.dtype("m8[M]"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("m8[Y]"), np.dtype("m8[6M]"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("m8[Y]"), np.dtype("m8[7M]"), "safe").Should().BeFalse();
        }

        [DataTestMethod]
        [DataRow("i1", "M8[s]", false, false, true, false)]
        [DataRow("i1", "m8[s]", true, true, true, false)]
        [DataRow("i1", "m8", true, true, true, false)]
        [DataRow("i8", "M8[s]", false, false, true, false)]
        [DataRow("i8", "m8[s]", true, true, true, false)]
        [DataRow("u4", "m8[s]", true, true, true, false)]
        [DataRow("u8", "M8[s]", false, false, true, false)]
        [DataRow("u8", "m8[s]", false, true, true, false)]
        [DataRow("u8", "m8", false, true, true, false)]
        [DataRow("?", "M8[s]", false, false, true, false)]
        [DataRow("?", "m8[s]", true, true, true, false)]
        [DataRow("f8", "M8[s]", false, false, true, false)]
        [DataRow("f8", "m8[s]", false, false, true, false)]
        [DataRow("e", "m8", false, false, true, false)]
        [DataRow("c16", "m8[s]", false, false, true, false)]
        [DataRow("decimal", "m8[s]", false, false, true, false)]
        [DataRow("char", "m8[s]", true, true, true, false)]
        [DataRow("char", "M8[s]", false, false, true, false)]
        public void CanCast_NumericToTime(string from, string to, bool safe, bool sameKind, bool unsafeCast, bool equiv)
        {
            np.can_cast(np.dtype(from), np.dtype(to), "safe").Should().Be(safe, $"{from}->{to} safe");
            np.can_cast(np.dtype(from), np.dtype(to), "same_kind").Should().Be(sameKind, $"{from}->{to} same_kind");
            np.can_cast(np.dtype(from), np.dtype(to), "unsafe").Should().Be(unsafeCast, $"{from}->{to} unsafe");
            np.can_cast(np.dtype(from), np.dtype(to), "equiv").Should().Be(equiv, $"{from}->{to} equiv");
            np.can_cast(np.dtype(from), np.dtype(to), "no").Should().BeFalse();
        }

        [TestMethod]
        public void CanCast_TimeToNumeric_And_AcrossTheDatetimePair_IsUnsafeOnly()
        {
            foreach (var from in new[] { "M8[s]", "m8[s]", "M8", "m8" })
                foreach (var to in new[] { "i8", "f8", "?", "u8", "i4" })
                    Rules.Select(r => np.can_cast(np.dtype(from), np.dtype(to), r)).Should().Equal(new[] { false, false, false, false, true }, $"{from}->{to}");
            Rules.Select(r => np.can_cast(np.dtype("M8[s]"), np.dtype("m8[s]"), r)).Should().Equal(false, false, false, false, true);
            Rules.Select(r => np.can_cast(np.dtype("m8[s]"), np.dtype("M8[s]"), r)).Should().Equal(false, false, false, false, true);
            Rules.Select(r => np.can_cast(np.dtype("m8[s]"), np.dtype("M8"), r)).Should().Equal(false, false, false, false, true);
        }

        [TestMethod]
        public void CanCast_ByteOrder()
        {
            Rules.Select(r => np.can_cast(np.dtype(">i4"), np.dtype("i4"), r)).Should().Equal(false, true, true, true, true);
            Rules.Select(r => np.can_cast(np.dtype(">i4"), np.dtype(">i4"), r)).Should().Equal(true, true, true, true, true);
            Rules.Select(r => np.can_cast(np.dtype(">M8[s]"), np.dtype("M8[s]"), r)).Should().Equal(false, true, true, true, true);
            Rules.Select(r => np.can_cast(np.dtype(">M8[s]"), np.dtype("M8[ms]"), r)).Should().Equal(false, false, true, true, true);
            np.can_cast(np.dtype(">i4"), np.dtype("i8"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("i4"), np.dtype(">i8"), "safe").Should().BeTrue();
            np.can_cast(np.dtype(">i1"), np.dtype("i1"), "no").Should().BeTrue("single-byte types have no byte order");
        }

        [TestMethod]
        public void CanCast_Builtins_KeepTheOldAnswers()
        {
            np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64).Should().BeTrue();
            np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32).Should().BeFalse();
            np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Single, NPTypeCode.Int32, "same_kind").Should().BeFalse();
            np.can_cast(NPTypeCode.Int16, NPTypeCode.Byte, "same_kind").Should().BeFalse("signed -> unsigned moves down the kind order");
            np.can_cast(NPTypeCode.UInt64, NPTypeCode.SByte, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Complex, NPTypeCode.Double, "same_kind").Should().BeFalse();
            np.can_cast(NPTypeCode.Double, NPTypeCode.Half, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Int32, NPTypeCode.Int32, "no").Should().BeTrue();
            np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "no").Should().BeFalse();
            np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, "equiv").Should().BeFalse();
            np.can_cast(NPTypeCode.Char, NPTypeCode.UInt16, "safe").Should().BeTrue();
            np.can_cast(NPTypeCode.Char, NPTypeCode.UInt16, "no").Should().BeFalse("Char and UInt16 are distinct storage classes, not an equiv pair");
            np.can_cast(NPTypeCode.UInt16, NPTypeCode.Char, "safe").Should().BeFalse();
            np.can_cast(NPTypeCode.UInt16, NPTypeCode.Char, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Decimal, NPTypeCode.Double, "same_kind").Should().BeTrue();
            np.can_cast(NPTypeCode.Double, NPTypeCode.Decimal, "safe").Should().BeTrue();
            np.can_cast(typeof(int), typeof(long)).Should().BeTrue();
            np.can_cast<int, double>("safe").Should().BeTrue();
            np.can_cast<int, float>("safe").Should().BeFalse("NumPy: int32 -> float32 loses precision, only same_kind");
            np.can_cast<int, float>("same_kind").Should().BeTrue();
            np.can_cast("i4", "i8").Should().BeTrue("a dtype string binds the grammar, not the char-array conversion");
            np.can_cast("i8", DType.Int32).Should().BeFalse();
            np.can_cast(np.zeros(new Shape(2), typeof(int)), DType.Int64).Should().BeTrue();
            np.can_cast(np.zeros(new Shape(2), typeof(int)), NPTypeCode.Int16).Should().BeFalse();
        }

        [TestMethod]
        public void CanCast_CastingString_IsCaseSensitive_NumPyText()
        {
            foreach (var bad in new[] { "bogus", "Safe", "SAFE", "same_value", "" })
            {
                Action act = () => np.can_cast(np.dtype("i4"), np.dtype("i8"), bad);
                act.Should().Throw<ValueError>().WithMessage($"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{bad}')");
                Action act2 = () => np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64, bad);
                act2.Should().Throw<ValueError>();
            }
            Action none = () => np.can_cast(np.dtype("i4"), (DType)null);
            none.Should().Throw<TypeError>().WithMessage("did not understand one of the types; 'None' not accepted");
            np.can_cast(np.dtype("i4"), np.dtype("i8"), NPY_CASTING.NPY_SAFE_CASTING).Should().BeTrue();
        }

        [TestMethod]
        public void MinCastSafety_TakesTheLessSafe()
        {
            DTypeCasting.MinCastSafety(NPY_CASTING.NPY_SAFE_CASTING, NPY_CASTING.NPY_EQUIV_CASTING).Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.MinCastSafety(NPY_CASTING.NPY_NO_CASTING, NPY_CASTING.NPY_UNSAFE_CASTING).Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            DTypeCasting.MinCastSafety(NPY_CASTING.NPY_SAME_KIND_CASTING, NPY_CASTING.NPY_SAME_KIND_CASTING).Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            DTypeCasting.ParseCasting("same_kind").Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            DTypeCasting.CastingToString(NPY_CASTING.NPY_EQUIV_CASTING).Should().Be("equiv");
            DTypeCasting.KindToOrdering('b').Should().Be(0);
            DTypeCasting.KindToOrdering('u').Should().Be(1);
            DTypeCasting.KindToOrdering('i').Should().Be(2);
            DTypeCasting.KindToOrdering('f').Should().Be(4);
            DTypeCasting.KindToOrdering('c').Should().Be(5);
            DTypeCasting.KindToOrdering('M').Should().Be(-1, "datetime does not fit the kind hierarchy");
        }

        [TestMethod]
        public void CastingImpls_CarryNumPysMinimalSafety()
        {
            DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.Int32DType).Casting.Should().Be(NPY_CASTING.NPY_EQUIV_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.Int64DType).Casting.Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.Int64DType, np.dtypes.Int32DType).Casting.Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.Float64DType, np.dtypes.Int32DType).Casting.Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.Float64DType).Casting.Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.CharDType, np.dtypes.UInt16DType).Casting.Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.UInt16DType, np.dtypes.CharDType).Casting.Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.TimeDelta64DType).Casting.Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.UInt64DType, np.dtypes.TimeDelta64DType).Casting.Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.Float64DType, np.dtypes.TimeDelta64DType).Casting.Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.DateTime64DType).Casting.Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            DTypeCasting.GetCastingImpl(np.dtypes.DateTime64DType, np.dtypes.DateTime64DType).Should().BeOfType<TimeToTimeCastingImpl>();
            DTypeCasting.GetCastingImpl(np.dtypes.TimeDelta64DType, np.dtypes.DateTime64DType).Should().BeOfType<DatetimeTimedeltaCastingImpl>();
            DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, DTypeRegistry.Str).Should().BeNull("no cast is registered for the vestigial string slot");
            np.can_cast(np.dtype("i4"), np.dtype("String")).Should().BeFalse("an unregistered cast is simply impossible");
            var impl = DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.Int64DType);
            impl.Name.Should().Be("numeric_cast");
            impl.NIn.Should().Be(1);
            impl.NOut.Should().Be(1);
            impl.DTypes.Should().Equal(np.dtypes.Int32DType, np.dtypes.Int64DType);
            impl.From.Should().BeSameAs(np.dtypes.Int32DType);
            impl.To.Should().BeSameAs(np.dtypes.Int64DType);
            impl.IsWithinDType.Should().BeFalse();
        }

        [TestMethod]
        public void ResolveDescriptors_And_GetCastInfo()
        {
            var impl = DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.Int32DType);
            var loop = new DType[2];
            impl.ResolveDescriptors(new[] { DType.Int32, (DType)null }, loop, out long view).Should().Be(NPY_CASTING.NPY_NO_CASTING);
            view.Should().Be(0);
            loop[1].Should().BeSameAs(DType.Int32);
            impl.ResolveDescriptors(new[] { np.dtype(">i4"), DType.Int32 }, loop, out view).Should().Be(NPY_CASTING.NPY_EQUIV_CASTING);
            view.Should().Be(ArrayMethod.NoView);

            DTypeCasting.GetCastInfo(DType.Int32, DType.Int64, null, out view).Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.GetCastInfo(np.dtype("M8[s]"), np.dtype("M8[ms]"), null, out view).Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.GetCastInfo(np.dtype("M8[ms]"), np.dtype("M8[s]"), null, out view).Should().Be(NPY_CASTING.NPY_SAME_KIND_CASTING);
            DTypeCasting.GetCastInfo(np.dtype("M8"), np.dtype("M8[s]"), null, out view).Should().Be(NPY_CASTING.NPY_SAFE_CASTING);
            DTypeCasting.GetCastInfo(np.dtype("M8[s]"), np.dtype("M8"), null, out view).Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            DTypeCasting.GetCastInfo(np.dtype("i4"), null, np.dtypes.TimeDelta64DType, out view).Should().Be(NPY_CASTING.NPY_SAFE_CASTING, "a null 'to' asks about the class");
            DTypeCasting.GetCastInfo(np.dtype("i4"), np.dtype("String"), null, out view).Should().BeNull();
            DTypeCasting.EquivTypes(DType.Int32, new DType(typeof(int))).Should().BeTrue();
            DTypeCasting.EquivTypes(DType.Int32, np.dtype(">i4")).Should().BeFalse();
            DTypeCasting.EquivTypes(np.dtype("M8[1000ms]"), np.dtype("M8[s]")).Should().BeTrue();
        }

        [TestMethod]
        public void GetStridedLoop_IsNotProvidedInStageA()
        {
            var impl = DTypeCasting.GetCastingImpl(np.dtypes.Int32DType, np.dtypes.Int64DType);
            var ctx = new ArrayMethodContext(impl, new[] { DType.Int32, DType.Int64 });
            Action act = () => impl.GetStridedLoop(ctx, aligned: true, new long[] { 4, 8 }, out _);
            act.Should().Throw<NotSupportedException>().WithMessage("*does not provide a strided loop yet*Stage C*");
        }
    }
}
