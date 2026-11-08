using System;
using System.Globalization;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     The text side of <see cref="Converts"/> and the converter classes built on it: number → text is NumPy's
    ///     <c>str()</c> of the scalar and text → number is NumPy's store of a Python str (<c>a[0] = text</c>), both
    ///     culture-invariant. Every expected value below was probed against NumPy 2.4.2 (<c>str(np.float16(1))</c>,
    ///     <c>a = np.zeros(1, 'i8'); a[0] = '1.5'</c>, ...).
    /// </summary>
    /// <remarks>
    ///     These routes used .NET's text before: <c>double.ToString</c> / <c>int.Parse</c> in the CURRENT culture (under
    ///     de-DE <c>"1.5"</c> read as 15 and 1.5 wrote as <c>"1,5"</c>), <c>Complex.ToString()</c>'s <c>&lt;3; 4&gt;</c>, and a
    ///     <c>"1.5"</c> into an integer through <c>IConvertible.ToDouble</c> truncated to 1 where NumPy raises.
    /// </remarks>
    [TestClass]
    public class ConvertsNumPyTextTests
    {
        /// <summary>Runs <paramref name="body"/> with <paramref name="culture"/> as the thread's culture.</summary>
        /// <param name="culture">The culture name (de-DE has a decimal comma).</param>
        /// <param name="body">The assertions.</param>
        private static void UnderCulture(string culture, Action body)
        {
            var previous = CultureInfo.CurrentCulture;
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            try
            {
                body();
            }
            finally
            {
                CultureInfo.CurrentCulture = previous;
            }
        }

        // ---------------------------------------------------------------------------------------------
        // Number -> text
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void ToString_Scalars_AreNumPyStr()
        {
            // str(np.float64(...)), str(np.float32(...)), str(np.float16(...)), str(np.complex128(...)), str(np.bool_(...)).
            Converts.ToString(1.0).Should().Be("1.0");
            Converts.ToString(1e20).Should().Be("1e+20");
            Converts.ToString(1e16).Should().Be("1e+16");
            Converts.ToString(1e15).Should().Be("1000000000000000.0");
            Converts.ToString(0.1f).Should().Be("0.1");
            Converts.ToString(-0.0).Should().Be("-0.0");
            Converts.ToString(double.NaN).Should().Be("nan");
            Converts.ToString(double.NegativeInfinity).Should().Be("-inf");
            Converts.ToString((Half)0.1).Should().Be("0.1");
            Converts.ToString((Half)65504).Should().Be("6.55e+04");
            Converts.ToString(new Complex(1, 2)).Should().Be("(1+2j)");
            Converts.ToString(new Complex(-0.0, -1)).Should().Be("(-0-1j)");
            Converts.ToString(new Complex(0, 2)).Should().Be("2j");
            Converts.ToString(true).Should().Be("True");
            Converts.ToString((sbyte)-128).Should().Be("-128");
            Converts.ToString(ulong.MaxValue).Should().Be("18446744073709551615");
        }

        [TestMethod]
        public void ToString_NumSharpOnlyDtypes_FollowTheirTextRules()
        {
            // char crosses to text as its uint16 number (np.array(new[] {'a'}).astype("U") is '97'); decimal as its float64.
            Converts.ToString('a').Should().Be("97");
            Converts.ToString(1.5m).Should().Be("1.5");
            Converts.ToString((object)'a').Should().Be("97");
        }

        [TestMethod]
        public void StringTarget_EveryRoute_IsNumPyStr()
        {
            Converts.ChangeType<string>((object)0.1f).Should().Be("0.1");
            ((string)Converts.ChangeType((object)1e20, NPTypeCode.String)).Should().Be("1e+20");
            ((string)Converts.ChangeType((object)1e20, NPTypeCode.String, CultureInfo.GetCultureInfo("de-DE"))).Should().Be("1e+20");
            ((string)Converts.ChangeType((object)2.5, TypeCode.String)).Should().Be("2.5");
            ((string)Converts.ChangeType((object)2.5, TypeCode.String, CultureInfo.GetCultureInfo("de-DE"))).Should().Be("2.5");
            Converts.FindConverter<float, string>()(0.1f).Should().Be("0.1");
            Converts<double>.ToString(1e20).Should().Be("1e+20");
            NonGenericConvert.FromDoubleToString(2.5).Should().Be("2.5");
            ((string)TypelessConvert.FromSingleToString(0.1f)).Should().Be("0.1");
        }

        [TestMethod]
        public void StringTarget_NonDtypeObject_KeepsInvariantDotNetText()
        {
            // A DateTime is not a NumSharp dtype value: the invariant .NET text (null stays null for a string target).
            Converts.ChangeType<string>((object)new DateTime(2020, 1, 2)).Should().Be(new DateTime(2020, 1, 2).ToString(CultureInfo.InvariantCulture));
            Converts.ChangeType((object)null, NPTypeCode.String).Should().BeNull();
        }

        [TestMethod]
        public void ArrayConvert_ToString_SByteHalfComplex_AreNumPyStr()
        {
            // These three used ToString(InvariantCulture) — .NET text — while every other element type used Converts.
            ArrayConvert.ToString(new sbyte[] { -5, 127 }).Should().Equal("-5", "127");
            ArrayConvert.ToString(new Half[] { (Half)0.1, (Half)65504, Half.NaN }).Should().Equal("0.1", "6.55e+04", "nan");
            ArrayConvert.ToString(new[] { new Complex(1, 2), new Complex(-0.0, -1) }).Should().Equal("(1+2j)", "(-0-1j)");
        }

        // ---------------------------------------------------------------------------------------------
        // Text -> number
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void Parse_PythonGrammar_IsNumPysSetitem()
        {
            // a = np.zeros(1, dt); a[0] = text.
            Converts.ToInt64(" 1_2 ").Should().Be(12);
            Converts.ToInt64(" -7 ").Should().Be(-7);
            Converts.ToInt32("٣٢").Should().Be(32);
            Converts.ToDouble("1e3").Should().Be(1000.0);
            Converts.ToDouble("-Infinity").Should().Be(double.NegativeInfinity);
            Converts.ToSingle("0.1").Should().Be(0.1f);
            Converts.ToHalf("65520").Should().Be(Half.PositiveInfinity);
            Converts.ToComplex("1+2j").Should().Be(new Complex(1, 2));
            Converts.ToComplex("(1+2j)").Should().Be(new Complex(1, 2));
        }

        [TestMethod]
        public void Parse_NumPysErrors_QuoteThePythonStr()
        {
            Action intOfFloat = () => Converts.ToInt64("1.5");
            intOfFloat.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '1.5'");
            Action hex = () => Converts.ToInt32("0x1F");
            hex.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '0x1F'");
            Action notFloat = () => Converts.ToDouble("x");
            notFloat.Should().Throw<ValueError>().WithMessage("could not convert string to float: 'x'");
            Action badComplex = () => Converts.ToComplex("1+");
            badComplex.Should().Throw<ValueError>().WithMessage("complex() arg is a malformed string");
            Action outOfRange = () => Converts.ToByte("300");
            outOfRange.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for uint8");
            Action negativeUnsigned = () => Converts.ToUInt16("-1");
            negativeUnsigned.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint16");
        }

        [TestMethod]
        public void Parse_BoolTarget_IsTheStrsTruth()
        {
            // NumPy 2.x: a non-empty str is True, 'False' included.
            Converts.ToBoolean("False").Should().BeTrue();
            Converts.ToBoolean("0").Should().BeTrue();
            Converts.ToBoolean("").Should().BeFalse();
            Converts.ChangeType<bool>((object)"False").Should().BeTrue();
        }

        [TestMethod]
        public void Parse_EveryRoute_AgreesWithNumPy()
        {
            // The routes that reach a text source: object overloads, ChangeType (generic / type code / .NET TypeCode), the
            // cached converters and the generated converter classes.
            Converts.ToInt32((object)" 1_2 ").Should().Be(12);
            Converts.ChangeType<int>((object)"12").Should().Be(12);
            Converts.ChangeType<string, int>("12").Should().Be(12);
            Converts.ChangeType((object)"12", NPTypeCode.Int16).Should().Be((short)12);
            Converts.ChangeType("12", TypeCode.Int64).Should().Be(12L);
            Converts.FindConverter<string, int>()(" 1_2 ").Should().Be(12);
            Converts<double>.From("1e3").Should().Be(1000.0);
            NonGenericConvert.FromStringToInt64(" 1_2 ").Should().Be(12L);
            ((long)TypelessConvert.FromStringToInt64("12")).Should().Be(12L);
            ArrayConvert.ToInt32(new[] { "1", " 2_0 " }).Should().Equal(1, 20);

            // FindConverter<string, int> used to parse through IConvertible.ToDouble (current culture) and TRUNCATE: "1.5" -> 1.
            Action truncating = () => Converts.FindConverter<string, int>()("1.5");
            truncating.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '1.5'");
        }

        [TestMethod]
        public void CharTarget_TextIsAUInt16_NumbersFollowAstype()
        {
            // NumSharp's char is NumPy's uint16 in every type rule.
            Converts.ChangeType((object)"97", NPTypeCode.Char).Should().Be('a');
            Action tooBig = () => Converts.ChangeType((object)"70000", NPTypeCode.Char);
            tooBig.Should().Throw<OverflowException>().WithMessage("Python integer 70000 out of bounds for uint16");

            // A float beyond int32 reaches char through astype's int32 intermediate (NumPy's MSVC cast), 0 — the old int64
            // route gave 0xFFFF.
            ((int)(char)Converts.ChangeType((object)4294967295.0, NPTypeCode.Char)).Should().Be(0);
            ((int)Converts.ChangeType<char>((object)4294967295.0)).Should().Be(0);
            ((int)Converts.ChangeType<char>((object)-2147483649.0m)).Should().Be(0);
        }

        // ---------------------------------------------------------------------------------------------
        // Culture invariance
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void Text_IsCultureInvariant()
        {
            UnderCulture("de-DE", () =>
            {
                Converts.ToDouble("1.5").Should().Be(1.5);
                Converts.ToDouble("1.5", CultureInfo.GetCultureInfo("de-DE")).Should().Be(1.5);
                Converts.ToString(1.5).Should().Be("1.5");
                Converts.ToString(1.5, CultureInfo.GetCultureInfo("de-DE")).Should().Be("1.5");
                Converts.ChangeType<string>((object)1.5f).Should().Be("1.5");
                Converts.ChangeType<double>((object)"1.5").Should().Be(1.5);
                Converts.FindConverter<string, double>()("1.5").Should().Be(1.5);
                py.complex("1.5").Should().Be(new Complex(1.5, 0));
                Converts.ToDateTime64("01/02/2020").ToString().Should().Be("2020-01-02T00:00:00.0000000");
            });
        }

        // ---------------------------------------------------------------------------------------------
        // py.complex and ArrayConvert.ToComplex(string[])
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void PyComplex_IsCPythonsComplexOfStr()
        {
            // complex(s) in CPython 3.12.
            py.complex("10+5j").Should().Be(new Complex(10, 5));
            py.complex("j").Should().Be(new Complex(0, 1));
            py.complex("-j").Should().Be(new Complex(0, -1));
            py.complex("1e3").Should().Be(new Complex(1000, 0));
            py.complex("(1+2j)").Should().Be(new Complex(1, 2));
            py.complex(" 1_0 ").Should().Be(new Complex(10, 0));
            py.complex("1.5e-3+2J").Should().Be(new Complex(0.0015, 2));
            Complex.IsNaN(py.complex("nan")).Should().BeTrue();
            Action empty = () => py.complex("");
            empty.Should().Throw<ValueError>().WithMessage("complex() arg is a malformed string");
            Action dangling = () => py.complex("1+");
            dangling.Should().Throw<ValueError>().WithMessage("complex() arg is a malformed string");
            Action none = () => py.complex(null);
            none.Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void ArrayConvert_ToComplex_NullIsZero_EmptyRaises()
        {
            ArrayConvert.ToComplex(new[] { null, "2j", "(1+2j)" }).Should().Equal(Complex.Zero, new Complex(0, 2), new Complex(1, 2));
            Action empty = () => ArrayConvert.ToComplex(new[] { "" });
            empty.Should().Throw<ValueError>().WithMessage("complex() arg is a malformed string");
        }

        // ---------------------------------------------------------------------------------------------
        // np.loadtxt converters and np.asanyarray items
        // ---------------------------------------------------------------------------------------------

        [TestMethod]
        public void Loadtxt_ConverterResult_IsStoredByPyArrayPack()
        {
            // np.loadtxt(['7'], converters=lambda s: v, dtype=dt): the converter's value goes through the dtype's setitem.
            NDArray Load(object value, DType dtype) => np.loadtxt(new[] { "7" }, dtype: dtype, converters: (Func<string, object>)(_ => value));

            Load(1.7, np.int64).GetAtIndex<long>(0).Should().Be(1);    // truncated (Convert.ToInt64 ROUNDED to 2)
            Load(-1.7, np.int64).GetAtIndex<long>(0).Should().Be(-1);
            Load(" 1_0 ", np.int64).GetAtIndex<long>(0).Should().Be(10);
            Load("False", np.@bool).GetAtIndex<bool>(0).Should().BeTrue();
            Load("", np.@bool).GetAtIndex<bool>(0).Should().BeFalse();
            Load((Half)300, np.uint8).GetAtIndex<byte>(0).Should().Be(44);   // a NumPy scalar is cast (wraps)
            Load(2.5, np.uint8).GetAtIndex<byte>(0).Should().Be(2);
            Load(70000, np.float16).GetAtIndex<Half>(0).Should().Be(Half.PositiveInfinity);

            // None: NumPy's positive quiet NaN for a float dtype, nan+nanj for complex, False for bool.
            BitConverter.DoubleToInt64Bits(Load(null, np.float64).GetAtIndex<double>(0)).Should().Be(0x7FF8000000000000);
            BitConverter.HalfToInt16Bits(Load(null, np.float16).GetAtIndex<Half>(0)).Should().Be((short)0x7E00);
            var c = Load(null, np.complex128).GetAtIndex<Complex>(0);
            BitConverter.DoubleToInt64Bits(c.Real).Should().Be(0x7FF8000000000000);
            BitConverter.DoubleToInt64Bits(c.Imaginary).Should().Be(0x7FF8000000000000);
            Load(null, np.@bool).GetAtIndex<bool>(0).Should().BeFalse();

            // NumPy's error for every refused value: the field's text, row and column.
            foreach (var (value, dtype, name) in new (object, DType, string)[]
                     {
                         ("1.5", np.int64, "int64"), (300, np.uint8, "uint8"), (new Complex(1, 2), np.float64, "float64"),
                         (double.NaN, np.int32, "int32"), (null, np.int64, "int64"),
                     })
            {
                Action act = () => Load(value, dtype);
                act.Should().Throw<ValueError>().WithMessage($"could not convert string '7' to {name} at row 0, column 1.");
            }
        }

        [TestMethod]
        public void Asanyarray_MixedObjectItems_ConvertThroughConverts()
        {
            // Half and Complex are not IConvertible: System.Convert threw for a float16 item beside a float64 one.
            var a = np.asanyarray(new object[] { (Half)1, 2.5 });
            a.dtype.Should().Be(np.float64);
            a.ToArray<double>().Should().Equal(1.0, 2.5);

            var c = np.asanyarray(new object[] { (Half)1, new Complex(0, 1) });
            c.dtype.Should().Be(np.complex128);
            c.ToArray<Complex>().Should().Equal(new Complex(1, 0), new Complex(0, 1));

            var t = np.asanyarray((object)((Half)1, 2.5));
            t.ToArray<double>().Should().Equal(1.0, 2.5);
        }
    }
}
