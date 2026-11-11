using System;
using System.Numerics;
using NumSharp;
using NumSharp.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the code examples of <c>docs/website-src/docs/dtypes.md</c> outside its "Scalar Types and
    ///     Type Checks" section (which <see cref="DtypesScalarTypesDocTests"/> covers): the three ways to name a dtype, the
    ///     class-level aliases, complex128-only storage, the Char dtype, creation and conversion, scalar casts, special
    ///     and boundary values, promotion of numbers and text, the refused dtype strings and the common patterns. Each test
    ///     mirrors one documented snippet and asserts what the page claims (NumPy 2.4.2's values where NumPy has the
    ///     call), so the page cannot drift from the library.
    /// </summary>
    [TestClass]
    public class DtypesDocTests
    {
        /// <summary>The "Referring to Dtypes in Code" snippets: an <see cref="NPTypeCode"/>, an <c>np.*</c> alias and a
        /// dtype string name the same descriptors.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ReferringToDtypes_ThreeFormsAgree()
        {
            np.zeros(new Shape(10), NPTypeCode.Int32).dtype.Should().Be(np.int32);
            np.zeros(new Shape(2, 3), NPTypeCode.Complex).dtype.Should().Be(np.complex128);
            np.ones(new Shape(5), np.float16).dtype.Should().Be(np.float16);

            np.dtype("int32").Should().Be(np.int32);
            np.dtype("float16").Should().Be(np.float16);
            np.dtype("complex128").Should().Be(np.complex128);
            np.dtype("i4").Should().Be(np.int32, "NumPy's shorthand");
            np.dtype("<f8").Should().Be(np.float64, "an explicit host order normalizes to native");
        }

        /// <summary>The "Class-level Aliases" snippets: the aliases are descriptors (the unsized text ones included), a
        /// builtin alias and the parser give the same object, and a text width is part of the descriptor's identity.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ClassLevelAliases_AreDescriptors()
        {
            DType unicode = np.str_;
            unicode.ToString(true).Should().Be("dtype('<U')", "unsized: creation infers a width");
            DType bytes = np.bytes_;
            bytes.ToString(true).Should().Be("dtype('S')");
            DType text = np.dtypes.StringDType(na_object: null);
            text.ToString(true).Should().Be("StringDType(na_object=None)");
            np.int32.type.Should().Be(typeof(int));

            (np.int32 == np.dtype("int32")).Should().BeTrue();
            (np.str_ == np.dtype("U")).Should().BeTrue();
            (np.dtype("U3") == np.dtype("U5")).Should().BeFalse("width matters");
            ReferenceEquals(np.int32, np.dtype("int32")).Should().BeTrue();
            ReferenceEquals(np.@long, np.dtype("long")).Should().BeTrue();
            ReferenceEquals(np.float16, np.dtype("float16")).Should().BeTrue();
        }

        /// <summary>The "Complex: Only 128-bit Is Supported" snippets: every complex128 spelling resolves to the one
        /// complex dtype, the complex64 requests throw, and a complex scalar converts to a real only through
        /// <c>np.real</c> (the direct cast is Python's <c>TypeError</c>).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Complex_Only128BitIsSupported()
        {
            foreach (var spelling in new[] { np.complex_, np.cdouble, np.dtype("complex"), np.dtype("D"), np.dtype("c16"), np.dtype("G") })
                spelling.Should().Be(np.complex128);
            np.clongdouble.Should().Be(np.complex128, "its own class, stored as complex128 and equal to it");

            foreach (var name in new[] { "complex64", "F", "c8" })
            {
                Action request = () => np.dtype(name);
                request.Should().Throw<NotSupportedException>(name);
            }
            // NumPy 2.0 removed the name 'complex_' (np.complex_ is kept as a C# member): NumPy's own TypeError.
            new Action(() => np.dtype("complex_")).Should().Throw<TypeError>().WithMessage("data type 'complex_' not understood");

            var c = NDArray.Scalar<Complex>(new Complex(3, 4));
            Action toInt = () => _ = (int)c;
            toInt.Should().Throw<TypeError>()
                 .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'complex'");
            ((int)np.real(c)).Should().Be(3);
        }

        /// <summary>The "<c>Char</c> - 16-bit UTF-16 code unit" snippet: a <c>char</c> array is the 2-byte unsigned Char
        /// dtype, and a cast to text writes the numeric code unit, never the character.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Char_IsANumericCodeUnit()
        {
            var letters = np.array(new[] { 'a', 'b', 'c' });
            letters.typecode.Should().Be(NPTypeCode.Char);
            InfoOf<char>.Size.Should().Be(2);
            np.dtype(typeof(char)).itemsize.Should().Be(2);
            np.dtype(typeof(char)).kind.Should().Be('u');
            NDArray.AsStringArray(letters.astype("U")).Should().Equal("97", "98", "99");
        }

        /// <summary>The "Creating Arrays with a Specific Dtype" snippets: an explicit dtype, the dtype a C# array infers
        /// (a collection expression of narrow integers is int32), and <c>astype</c> conversions.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Creating_ExplicitInferredAndConverted()
        {
            np.zeros(new Shape(3, 4), NPTypeCode.Single).dtype.Should().Be(np.float32);
            np.full(new Shape(2), (Half)3.14).dtype.Should().Be(np.float16);
            np.arange(0, 10, dtype: np.int8).dtype.Should().Be(np.int8);
            np.empty(new Shape(100), np.complex128).dtype.Should().Be(np.complex128);

            np.array([1, 2, 3]).dtype.Should().Be(np.int32);
            np.array([1.0, 2.0]).dtype.Should().Be(np.float64);
            np.array([(Half)1, (Half)2]).dtype.Should().Be(np.float16);
            np.array([new Complex(1, 2), new Complex(3, 4)]).dtype.Should().Be(np.complex128);
            np.array(new sbyte[] { -1, 0, 1 }).dtype.Should().Be(np.int8);
            np.array([1L, 2L, 3L]).dtype.Should().Be(np.int64);

            var doubles = np.array([1.5, 2.7, 3.9]);
            doubles.astype(NPTypeCode.Int32).ToArray<int>().Should().Equal(1, 2, 3);
            doubles.astype(NPTypeCode.Half).dtype.Should().Be(np.float16);
            doubles.astype(NPTypeCode.Complex).ToArray<Complex>().Should().Equal(new Complex(1.5, 0), new Complex(2.7, 0), new Complex(3.9, 0));
        }

        /// <summary>The "Scalar ↔ NDArray casts" snippets: a C# number is a 0-d array of its dtype, and only a 0-d array
        /// casts back to a C# number.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ScalarCasts_RequireZeroDimensions()
        {
            NDArray s1 = (sbyte)42;
            s1.dtype.Should().Be(np.int8);
            s1.ndim.Should().Be(0);
            NDArray s2 = (Half)3.14;
            s2.dtype.Should().Be(np.float16);
            NDArray s3 = new Complex(1, 2);
            s3.dtype.Should().Be(np.complex128);

            var scalar = np.array([42])[0];
            ((int)scalar).Should().Be(42);
            var oneD = np.array([42]);
            Action toInt = () => _ = (int)oneD;
            toInt.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
            ((bool)oneD).Should().BeTrue();   // bool() reads the only element of a one-element array of any rank
        }

        /// <summary>The "NumPy Scalars" element snippet: an element is a read-only copy a later write does not reach, and
        /// <c>s[...]</c> of a scalar is a writeable 0-d array over a copy of its value (NumPy 2.4.2's answers).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void NumPyScalars_ElementAndScalarIndex_AsDocumented()
        {
            var a = np.arange(5.0);
            var x = a[2];
            x.flags.writeable.Should().BeFalse();
            a[2] = 99.0;
            ((double)x).Should().Be(2.0);

            var s = a.sum();
            var v = s[Slice.Ellipsis];
            v.ndim.Should().Be(0);
            v.flags.writeable.Should().BeTrue();
            ((double)v).Should().Be((double)s);
            Action refused = () => _ = s[0];
            refused.Should().Throw<IndexError>().WithMessage("invalid index to scalar variable.");
        }

        /// <summary>The "Special Values" snippets: NaN and infinity survive a scalar round trip, and a complex value is NaN
        /// or infinite through either part, as in NumPy.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void SpecialValues_RoundTrip()
        {
            Half.IsNaN((Half)NDArray.Scalar<Half>(Half.NaN)).Should().BeTrue();
            double.IsPositiveInfinity((double)NDArray.Scalar<double>(double.PositiveInfinity)).Should().BeTrue();
            np.isnan(np.array(new[] { Complex.NaN, new Complex(1, double.NaN), new Complex(1, 2) })).ToArray<bool>()
              .Should().Equal(true, true, false);
            np.isinf(np.array(new[] { Complex.Infinity, new Complex(0, double.PositiveInfinity), new Complex(1, 2) })).ToArray<bool>()
              .Should().Equal(true, true, false);
        }

        /// <summary>The "Boundary values" snippet: <c>iinfo</c> clamps uint64's maximum to <c>long</c> beside the true
        /// <c>maxUnsigned</c>, <c>finfo</c> gives float16's epsilon and smallest normal, and complex128 reports float64's
        /// precision.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void BoundaryValues_IinfoAndFinfo()
        {
            np.iinfo(np.int8).min.Should().Be(-128);
            np.iinfo(np.int8).max.Should().Be(127);
            np.iinfo(np.uint64).max.Should().Be(long.MaxValue);
            np.iinfo(np.uint64).maxUnsigned.Should().Be(ulong.MaxValue);
            np.finfo(np.float16).eps.Should().Be(0.0009765625);
            np.finfo(np.float16).smallest_normal.Should().Be(System.Math.Pow(2, -14));
            np.finfo(np.float64).max.Should().Be(double.MaxValue);
            np.finfo(np.complex128).dtype.Should().Be(np.float64);
            np.finfo(np.complex128).bits.Should().Be(64);
            np.finfo(np.complex128).precision.Should().Be(15);
        }

        /// <summary>The "Type Promotion" snippets and quick-reference table (NumPy 2.4.2's results): NEP 50 promotion of
        /// numbers, string promotion by width and class, <c>StringDType</c>'s refusals, and <c>np.find_common_type</c>'s
        /// older rule beside <c>np.result_type</c>.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void TypePromotion_NumbersAndText()
        {
            (np.array([1, 2, 3]) + np.array([1.5, 2.5, 3.5])).dtype.Should().Be(np.float64);
            np.promote_types(np.int8, np.uint8).Should().Be(np.int16);
            np.promote_types(np.int32, np.uint32).Should().Be(np.int64);
            np.promote_types(np.int32, np.uint64).Should().Be(np.float64);
            np.promote_types(np.float16, np.int16).Should().Be(np.float32);
            np.promote_types(np.float16, np.float32).Should().Be(np.float32);

            np.promote_types(np.dtype("S3"), np.dtype("U2")).ToString(true).Should().Be("dtype('<U3')");
            np.promote_types(np.dtype("T"), np.dtype("U2")).ToString(true).Should().Be("StringDType()");
            np.promote_types(np.dtype("U2"), np.int32).ToString(true).Should().Be("dtype('<U11')", "a strong number promotes to a wide enough string");
            Action number = () => np.promote_types(np.dtype("T"), np.int32);
            number.Should().Throw<DTypePromotionError>();
            Action bytes = () => np.promote_types(np.dtype("T"), np.dtype("S3"));
            bytes.Should().Throw<DTypePromotionError>();

            np.find_common_type(new[] { np.int8 }, new[] { np.int64 }).Should().Be(np.int8);
            np.result_type(np.int8, np.int64).Should().Be(np.int64);
        }

        /// <summary>The "Dtype String Parsing" paragraph on comma strings: a field whose NumPy result is a plain dtype parses
        /// (<c>"()f8"</c>, <c>"5U"</c>, <c>"2&gt;U"</c>, <c>"3S"</c>), a sub-array or structured result throws
        /// <see cref="NotSupportedException"/> with the quoted text, and NumPy's own errors keep NumPy's type and text -
        /// the <c>TypeError</c> of <c>"F,xyz"</c> included, although NumSharp has no <c>F</c>.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void DtypeStringParsing_CommaStrings()
        {
            np.dtype("()f8").Should().BeSameAs(np.float64);
            np.dtype("5U").ToString(true).Should().Be("dtype('<U5')");
            np.dtype("2>U").ToString(true).Should().Be("dtype('>U2')");
            np.dtype("3S").ToString(true).Should().Be("dtype('S3')");

            Action subArray = () => np.dtype("3f8");
            subArray.Should().Throw<NotSupportedException>().WithMessage("NumPy dtype '3f8' (a sub-array dtype) is not supported by NumSharp");
            foreach (var s in new[] { "(2,3)i4", "i4,f8" })
            {
                Action refused = () => np.dtype(s);
                refused.Should().Throw<NotSupportedException>(s);
            }

            Action wideDim = () => np.dtype("3000000000f8");
            wideDim.Should().Throw<ValueError>().WithMessage("invalid shape in fixed-type tuple: dimension does not fit into a C int.");
            Action orders = () => np.dtype("<2>i4");
            orders.Should().Throw<ValueError>().WithMessage("inconsistent byte-order specification < and >");
            Action unsized = () => np.dtype("()U");
            unsized.Should().Throw<ValueError>().WithMessage("invalid itemsize in generic type tuple");
            Action laterField = () => np.dtype("F,xyz");
            laterField.Should().Throw<TypeError>().WithMessage("data type 'xyz' not understood");
            Action leadingZeros = () => np.dtype("01f8");
            leadingZeros.Should().Throw<SyntaxError>()
                .Which.Message.Should().Be("leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers (<unknown>, line 1)");
        }

        /// <summary>The "Invalid Dtype Strings" snippets: NumPy's invalid strings throw NumPy's <see cref="TypeError"/> with
        /// NumPy's text, the dtypes NumSharp has no storage for throw <see cref="NotSupportedException"/> (the sub-array and
        /// structured comma strings included), a comma string NumPy refuses throws NumPy's <see cref="ValueError"/> /
        /// <see cref="SyntaxError"/>, and the fixed-width string dtypes parse in both spellings.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void InvalidDtypeStrings_Throw()
        {
            foreach (var name in new[] { "xyz", "f32", "i3", "?1", "   i4", "<float64", "<str" })
            {
                Action parse = () => np.dtype(name);
                parse.Should().Throw<TypeError>().WithMessage($"data type '{name}' not understood");
            }
            foreach (var name in new[] { "V7", "object" })
            {
                Action parse = () => np.dtype(name);
                parse.Should().Throw<NotSupportedException>().WithMessage($"NumPy dtype '{name}' is not supported by NumSharp");
            }
            Action subArray = () => np.dtype("3f8");
            subArray.Should().Throw<NotSupportedException>().WithMessage("NumPy dtype '3f8' (a sub-array dtype) is not supported by NumSharp");
            Action structured = () => np.dtype("i4,f8");
            structured.Should().Throw<NotSupportedException>().WithMessage("NumPy dtype 'i4,f8' (a structured dtype) is not supported by NumSharp");
            Action wideDim = () => np.dtype("3000000000f8");
            wideDim.Should().Throw<ValueError>().WithMessage("invalid shape in fixed-type tuple: dimension does not fit into a C int.");
            Action notLiteral = () => np.dtype("1 2f8");
            notLiteral.Should().Throw<SyntaxError>().WithMessage("invalid syntax (<unknown>, line 1)");
            np.dtype("S10").itemsize.Should().Be(10);
            np.dtype("U32").itemsize.Should().Be(128);
            np.dtype("10S").itemsize.Should().Be(10);
            np.dtype("32U").itemsize.Should().Be(128);
            Action datetime = () => np.zeros(new Shape(3), np.dtype("M8[s]"));
            datetime.Should().Throw<NotSupportedException>();
        }

        /// <summary>The "Common Patterns" snippets: a text template's full descriptor keeps its width through
        /// <c>np.ones</c> and <c>np.ones_like</c>; the default unsafe cast wraps 300.0 to 44, and the <c>'safe'</c> rule
        /// refuses the float64 to uint8 pair itself (NumPy's TypeError text).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void CommonPatterns_TemplatesAndCasts()
        {
            var template = np.array(new[] { "cat", "bird" }, "U8");
            np.ones(template.shape, template.dtype).ToString(true).Should().Be("array(['1', '1'], dtype='<U8')");
            np.ones_like(template).ToString(true).Should().Be("array(['1', '1'], dtype='<U8')");

            np.array([300.0]).astype(NPTypeCode.Byte).ToArray<byte>().Should().Equal((byte)44);
            Action safe = () => np.array([300.0]).astype(np.uint8, 'K', "safe");
            safe.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast array data from dtype('float64') to dtype('uint8') according to the rule 'safe'");
        }
    }
}
