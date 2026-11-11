using System;
using System.Numerics;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     NumPy's scalar class calls (<c>np.float32(1.5)</c>, <c>np.int8("12")</c>, <c>np.complex128(1, 2)</c>,
    ///     <c>np.str_(5)</c>): the C# 14 static extension members of <see cref="NDScalarClassCalls"/> beside the same-named
    ///     <see cref="DType"/> fields, ported from NumPy's <c>*_arrtype_new</c> constructors. Every expected value and message
    ///     is NumPy 2.4.2's, probed live on win-amd64; the 3,210-case class-call replay (the scratch parity run of the scalar
    ///     work) matched NumPy everywhere except the documented differences these tests name.
    /// </summary>
    [TestClass]
    public class NDScalarClassCallsTests
    {
        /// <summary>Without parentheses the name is still the descriptor field: the same object, every member intact.</summary>
        [TestMethod]
        public void ValueForm_IsTheSameDTypeField()
        {
            DType d = np.float32;
            ReferenceEquals(d, np.dtype("float32")).Should().BeTrue();
            np.float32.itemsize.Should().Be(4);
            np.zeros(3, np.float32).dtype.Should().Be(np.float32);
        }

        /// <summary>
        ///     A call returns NumPy's scalar: a read-only 0-d array (<c>np.float64(1.5).flags.num</c> is 263) that
        ///     <c>np.isscalar</c> accepts.
        /// </summary>
        [TestMethod]
        public void Call_ReturnsTheReadOnlyScalar()
        {
            var s = np.float64(1.5);
            s.ndim.Should().Be(0);
            s.dtype.Should().Be(np.float64);
            s.GetDouble().Should().Be(1.5);
            s.flags.num.Should().Be(263);
            np.isscalar(s).Should().BeTrue();
        }

        /// <summary>
        ///     NumPy's no-argument call is the zero; an explicit null is Python's <c>None</c>, a different call: nan for a
        ///     float class, False for bool, CPython's <c>int()</c> TypeError for an integer class.
        /// </summary>
        [TestMethod]
        public void NoArgument_IsTheZero_NullIsNone()
        {
            np.float32().GetSingle().Should().Be(0f);
            np.int64().GetInt64().Should().Be(0L);
            np.@bool().GetBoolean().Should().BeFalse();
            np.complex128().GetComplex().Should().Be(Complex.Zero);

            float.IsNaN(np.float32(null).GetSingle()).Should().BeTrue();
            np.bool_(null).GetBoolean().Should().BeFalse();
            new Action(() => np.int32(null)).Should().Throw<TypeError>()
                .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
        }

        /// <summary>
        ///     An integer class packs a C# number with NumPy's range check (NEP 50): out of range raises OverflowError, a
        ///     float is truncated toward zero first, NaN and infinity raise CPython's <c>int()</c> errors.
        /// </summary>
        [TestMethod]
        public void IntegerClasses_PackWithTheRangeCheck()
        {
            np.int8(127).GetAtIndex<sbyte>(0).Should().Be(127);
            np.int8(3.9).GetAtIndex<sbyte>(0).Should().Be(3);
            np.int8(-3.9).GetAtIndex<sbyte>(0).Should().Be(-3);
            new Action(() => np.int8(300)).Should().Throw<OverflowError>().WithMessage("Python integer 300 out of bounds for int8");
            new Action(() => np.uint8(-1)).Should().Throw<OverflowError>().WithMessage("Python integer -1 out of bounds for uint8");
            new Action(() => np.int8(300.0)).Should().Throw<OverflowError>().WithMessage("Python integer 300 out of bounds for int8");
            new Action(() => np.int8(double.NaN)).Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            new Action(() => np.int8(double.PositiveInfinity)).Should().Throw<OverflowError>()
                .WithMessage("cannot convert float infinity to integer");
            new Action(() => np.int8(new Complex(1, 2))).Should().Throw<TypeError>()
                .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'complex'");
        }

        /// <summary>
        ///     A NumPy scalar or an array argument is CAST unsafely, not range-checked: <c>np.int8(np.int64(300))</c> is 44
        ///     and <c>np.uint8(np.int64(-1))</c> 255.
        /// </summary>
        [TestMethod]
        public void NumPyScalarArgument_IsCast()
        {
            np.int8(NDArray.Scalar(300L)).GetAtIndex<sbyte>(0).Should().Be(44);
            np.uint8(NDArray.Scalar(-1L)).GetAtIndex<byte>(0).Should().Be(255);
            np.float32(NDArray.Scalar(3.9)).GetSingle().Should().Be(3.9f);
            np.int8(np.array(300L)).GetAtIndex<sbyte>(0).Should().Be(44);
            // A C# Half is NumPy's np.float16 scalar: cast too (np.int8(np.float16(300)) is 44).
            np.int8((Half)300).GetAtIndex<sbyte>(0).Should().Be(44);
        }

        /// <summary>
        ///     A <see cref="string"/> is parsed by CPython's <c>int()</c> / <c>float()</c> / <c>complex()</c> grammar, with
        ///     its error texts.
        /// </summary>
        [TestMethod]
        public void TextArgument_IsParsedByPythonsGrammar()
        {
            np.int8("12").GetAtIndex<sbyte>(0).Should().Be(12);
            np.int8(" 7 ").GetAtIndex<sbyte>(0).Should().Be(7);
            np.float64("1e3").GetDouble().Should().Be(1000.0);
            double.IsNaN(np.float32("nan").GetSingle()).Should().BeTrue();
            np.complex128("1+2j").GetComplex().Should().Be(new Complex(1, 2));
            new Action(() => np.int8("1.5")).Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '1.5'");
            new Action(() => np.float64("abc")).Should().Throw<ValueError>().WithMessage("could not convert string to float: 'abc'");
            new Action(() => np.int8("300")).Should().Throw<OverflowError>().WithMessage("Python integer 300 out of bounds for int8");
        }

        /// <summary>
        ///     A float class rounds to its precision (<c>np.float16(65520.0)</c> is inf, <c>np.float16(65504.0)</c> the
        ///     largest finite value) and refuses a complex with CPython's <c>float()</c> TypeError.
        /// </summary>
        [TestMethod]
        public void FloatClasses_RoundToTheirPrecision()
        {
            Half.IsPositiveInfinity(np.float16(65520.0).GetAtIndex<Half>(0)).Should().BeTrue();
            np.float16(65504.0).GetAtIndex<Half>(0).Should().Be((Half)65504.0);
            np.float32(0.1).GetSingle().Should().Be(0.1f);
            new Action(() => np.float64(new Complex(1, 2))).Should().Throw<TypeError>()
                .WithMessage("float() argument must be a string or a real number, not 'complex'");
        }

        /// <summary>
        ///     A sequence or an array with dimensions comes back as the ndarray NumPy's constructor returns (an array that
        ///     already satisfies the request is returned as itself, <c>np.float64(a) is a</c>), each Python int of a list
        ///     range-checked.
        /// </summary>
        [TestMethod]
        public void SequenceArgument_IsTheArray()
        {
            var r = np.float64(new object[] { 1, 2 });
            r.ndim.Should().Be(1);
            r.dtype.Should().Be(np.float64);
            np.isscalar(r).Should().BeFalse();
            r.ToArray<double>().Should().Equal(1.0, 2.0);

            var a = np.array(new[] { 1.5, 2.5 });
            ReferenceEquals(np.float64(a), a).Should().BeTrue();

            new Action(() => np.int8(new object[] { 1, 300 })).Should().Throw<OverflowError>()
                .WithMessage("Python integer 300 out of bounds for int8");
        }

        /// <summary>
        ///     <c>bool_</c>'s own constructor: a C# bool is itself, anything else its truth (any non-empty text is True,
        ///     NaN is True), a sequence the bool array.
        /// </summary>
        [TestMethod]
        public void Bool_TakesTheTruth()
        {
            np.bool_(true).GetBoolean().Should().BeTrue();
            np.bool_("").GetBoolean().Should().BeFalse();
            np.bool_("False").GetBoolean().Should().BeTrue();
            np.bool_(double.NaN).GetBoolean().Should().BeTrue();
            var empty = np.@bool(new object[0]);
            empty.ndim.Should().Be(1);
            empty.dtype.Should().Be(np.@bool);
        }

        /// <summary>
        ///     <c>complex128</c>'s two-argument form builds <c>complex(real, imag)</c> from two real numbers (bool and NumPy
        ///     scalars included) and refuses anything else with NumPy's parse error quoting the argument tuple.
        /// </summary>
        [TestMethod]
        public void Complex_TwoArguments()
        {
            np.complex128(1, 2).GetComplex().Should().Be(new Complex(1, 2));
            np.cdouble(1.5, -2.0).GetComplex().Should().Be(new Complex(1.5, -2.0));
            np.complex128(true, 2).GetComplex().Should().Be(new Complex(1, 2));
            np.complex128(NDArray.Scalar(1.5), NDArray.Scalar(2L)).GetComplex().Should().Be(new Complex(1.5, 2));
            new Action(() => np.complex128(new Complex(1, 1), 2)).Should().Throw<TypeError>()
                .WithMessage("Could not convert arguments into a complex scalar. '((1+1j), 2)' given.");
            new Action(() => np.complex128("1", 2)).Should().Throw<TypeError>()
                .WithMessage("Could not convert arguments into a complex scalar. '('1', 2)' given.");
            new Action(() => np.complex128(null, 2)).Should().Throw<TypeError>()
                .WithMessage("Could not convert arguments into a complex scalar. '(None, 2)' given.");
            new Action(() => np.complex128(imag: 2)).Should().Throw<TypeError>()
                .WithMessage("'imag' is an invalid keyword argument for this function");
        }

        /// <summary>
        ///     <c>np.str_</c> is CPython's <c>str()</c> of the argument as a <c>&lt;U{len}&gt;</c> scalar - a list's repr, a
        ///     bytes object's repr, an array's <c>str()</c> - or, with an encoding, the decoded bytes.
        /// </summary>
        [TestMethod]
        public void Str_IsPythonsStr()
        {
            Text(np.str_(5)).Should().Be("5");
            np.str_(5).dtype.str.Should().Be("<U1");
            Text(np.str_(null)).Should().Be("None");
            Text(np.str_(true)).Should().Be("True");
            Text(np.str_(1e300)).Should().Be("1e+300");
            Text(np.str_(new Complex(1, 2))).Should().Be("(1+2j)");
            Text(np.str_(new object[] { 1, 2 })).Should().Be("[1, 2]");
            Text(np.str_(new byte[] { 97, 98 })).Should().Be("b'ab'");
            Text(np.str_(NDArray.Scalar(300L))).Should().Be("300");
            Text(np.str_(np.array(new[] { 1.5, 2.5 }))).Should().Be("[1.5 2.5]");
            Text(np.str_(new byte[] { 0xc3, 0xa9 }, "utf-8")).Should().Be("é");
            new Action(() => np.str_("abc", "utf-8")).Should().Throw<TypeError>().WithMessage("decoding str is not supported");
        }

        /// <summary>
        ///     The empty text: NumPy's <c>np.str_('')</c> scalar is <c>&lt;U0</c>, NumSharp's 0-d array is at least one
        ///     character wide (<c>&lt;U1</c>, as <c>np.array('')</c> is); the value is the empty text in both.
        /// </summary>
        [TestMethod, Misaligned]
        public void Str_Empty_IsOneCharacterWide()
        {
            var s = np.str_();
            Text(s).Should().Be("");
            s.dtype.str.Should().Be("<U1", "NumPy's scalar is <U0; a NumSharp 0-d text array is at least one character wide");
        }

        /// <summary>
        ///     <c>np.bytes_</c> is CPython's <c>bytes()</c> first - a count of NULs, the bytes of a list of 0..255 integers, an
        ///     array's raw buffer - and NumPy's conversion where <c>bytes()</c> raises (the text of a float, a negative int, a
        ///     str, None).
        /// </summary>
        [TestMethod]
        public void Bytes_IsPythonsBytesThenNumPysConversion()
        {
            var five = np.bytes_(5);
            five.dtype.str.Should().Be("|S5");
            Bytes(five).Should().BeEmpty("five NUL bytes read back without their trailing NULs");
            Bytes(np.bytes_(-1)).Should().Equal((byte)'-', (byte)'1');
            Bytes(np.bytes_("a")).Should().Equal((byte)'a');
            Bytes(np.bytes_(1.5)).Should().Equal((byte)'1', (byte)'.', (byte)'5');
            np.bytes_(1.5).dtype.str.Should().Be("|S3");
            Bytes(np.bytes_(null)).Should().Equal("None"u8.ToArray());
            Bytes(np.bytes_(new object[] { 1, 2 })).Should().Equal((byte)1, (byte)2);
            np.bytes_(NDArray.Scalar(2.5)).dtype.str.Should().Be("|S8");
            np.bytes_("é", "utf-8").dtype.str.Should().Be("|S2");
            Bytes(np.bytes_("é", "utf-8")).Should().Equal((byte)0xc3, (byte)0xa9);

            var list = np.bytes_(new object[] { 1, 300 });
            list.ndim.Should().Be(1);
            list.dtype.str.Should().Be("|S3");

            new Action(() => np.bytes_("é")).Should().Throw<UnicodeEncodeError>()
                .WithMessage("'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");
        }

        /// <summary>
        ///     The C-type aliases construct their own class: <c>np.byte</c> is int8, Windows' <c>np.long</c> is int32,
        ///     <c>np.longdouble</c> keeps its <c>'g'</c> descriptor and <c>np.intc</c> its <c>'i'</c>.
        /// </summary>
        [TestMethod]
        public void Aliases_ConstructTheirOwnClass()
        {
            np.@byte(5).dtype.Should().Be(np.int8);
            np.@long(5).dtype.Should().Be(np.@long);
            np.longdouble(1.5).dtype.@char.Should().Be('g');
            np.intc(5).dtype.@char.Should().Be('i');
            np.clongdouble(new Complex(1, 2)).GetComplex().Should().Be(new Complex(1, 2));
        }

        /// <summary>
        ///     The forms a call composes with: the class's own spelling for a C# 12 caller (<c>NDScalarClassCalls.float32</c>),
        ///     a lambda, and an expression tree.
        /// </summary>
        [TestMethod]
        public void CallForms_Compose()
        {
            NDScalarClassCalls.float32(1.5).GetSingle().Should().Be(1.5f);
            Func<object, NDArray> f = x => np.float32(x);
            f(2.5).GetSingle().Should().Be(2.5f);
            System.Linq.Expressions.Expression<Func<object, NDArray>> e = x => np.int16(x);
            e.Compile()(7).GetAtIndex<short>(0).Should().Be(7);
        }

        /// <summary>The text a 0-d <c>U</c> scalar holds.</summary>
        /// <param name="s">The scalar.</param>
        /// <returns>Its text.</returns>
        private static string Text(NDArray s) => (string)s.Storage.GetAtIndex(0);

        /// <summary>The bytes a 0-d <c>S</c> scalar holds (without its trailing NULs).</summary>
        /// <param name="s">The scalar.</param>
        /// <returns>Its bytes.</returns>
        private static byte[] Bytes(NDArray s) => (byte[])s.Storage.GetAtIndex(0);
    }
}
