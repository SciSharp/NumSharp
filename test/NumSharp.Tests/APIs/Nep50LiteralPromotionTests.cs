using System;
using System.Numerics;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     NEP 50 in the functions that promote their operands by NumPy's <c>result_type</c> — <c>np.where</c>,
    ///     <c>np.choose</c>, <c>np.select</c>, <c>np.r_</c> / <c>np.c_</c> — with text and with C# number literals. A C# integer
    ///     / <c>float</c> / <c>double</c> / <see cref="Complex"/> literal is NumPy's WEAK Python scalar, whether it arrives
    ///     boxed or as the 0-d array its implicit conversion marks (<c>NDArray.WasPythonLiteral</c>); every other operand —
    ///     a 0-d array included — is strong. Expected values and texts were probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A Python number and text have no common DType, so NumPy raises <c>DTypePromotionError</c> (a TypeError) where these
    ///     functions used to read the literal's 0-d int32 array as a strong number and promote to <c>&lt;U11</c>. A strong
    ///     number promotes to a string wide enough for its text (int64 → <c>&lt;U21</c>, bool → <c>&lt;U5</c>), and
    ///     <c>StringDType</c> promotes only with <c>U</c>.
    ///     </para>
    ///     <para>
    ///     <c>np.choose</c> also converts every choice to the common dtype under 'safe' (<c>PyArray_ConvertToCommonType</c>),
    ///     which a U choice beside a <c>StringDType</c> one fails. The numeric routes of <c>np.where</c> keep the house rule
    ///     that reads every 0-d operand as weak (a <c>[Misaligned]</c> difference recorded for the next phase); the literal is
    ///     weak under both rules.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class Nep50LiteralPromotionTests
    {
        private const string PyLongVsStr =
            "The DType <class 'numpy.dtypes.StrDType'> could not be promoted by <class 'numpy.dtypes._PyLongDType'>. This means " +
            "that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the " +
            "dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyLongDType'>)";

        private static NDArray C => np.array(new[] { true, false });
        private static NDArray U => np.array(new[] { "ab", "cd" });
        private static NDArray T => np.array(new[] { "ab", "cd" }, np.dtypes.StringDType());
        private static NDArray Idx => np.array(new long[] { 0, 1 });

        [TestMethod]
        public void Where_TextAndPythonNumber_HaveNoCommonDType()
        {
            Action literal = () => np.where(C, U, 1);
            literal.Should().Throw<DTypePromotionError>().WithMessage(PyLongVsStr);
            Action boxed = () => np.where(C, U, (object)1);
            boxed.Should().Throw<DTypePromotionError>().WithMessage(PyLongVsStr);
            Action reversed = () => np.where(C, 1.5, U);
            reversed.Should().Throw<DTypePromotionError>().WithMessage("The DType <class 'numpy.dtypes._PyFloatDType'> could not be promoted by <class 'numpy.dtypes.StrDType'>*");
            Action vstring = () => np.where(C, 1, T);
            vstring.Should().Throw<DTypePromotionError>().WithMessage("The DType <class 'numpy.dtypes._PyLongDType'> could not be promoted by <class 'numpy.dtypes.StringDType'>*");
        }

        [TestMethod]
        public void Where_TextAndStrongNumber_PromotesToItsTextWidth()
        {
            var r = np.where(C, U, NDArray.Scalar(1L));
            r.dtype.Should().Be(np.dtype("<U21"));
            NDArray.AsStringArray(r).Should().Equal("ab", "1");
            var b = np.where(C, true, U);
            b.dtype.Should().Be(np.dtype("<U5"));
            NDArray.AsStringArray(b).Should().Equal("True", "cd");
        }

        [TestMethod]
        public void Choose_WeakLiterals_AdoptTheChoicesDtype()
        {
            // np.choose(idx, [int8_arr, 1000]) is int8 [1, -24]: the literal is weak in an NDArray[] list too.
            var i8 = np.array(new sbyte[] { 1, 2 });
            var r = np.choose(Idx, new NDArray[] { i8, 1000 });
            r.dtype.Should().Be(np.int8);
            r.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)-24);
            var f4 = np.choose(Idx, new NDArray[] { np.array(new float[] { 1, 2 }), 1e300 });
            f4.dtype.Should().Be(np.float32);
            float.IsPositiveInfinity(f4.GetAtIndex<float>(1)).Should().BeTrue();
        }

        [TestMethod]
        public void Choose_TextAndPythonNumber_HaveNoCommonDType()
        {
            Action act = () => np.choose(Idx, new NDArray[] { U, 1 });
            act.Should().Throw<DTypePromotionError>().WithMessage(PyLongVsStr);
        }

        [TestMethod]
        public void Choose_EachChoice_MustCastSafelyToTheCommonDtype()
        {
            Action array = () => np.choose(Idx, new object[] { T, np.array(new[] { "zz", "yy" }) });
            array.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('<U2') to StringDType() according to the rule 'safe'");
            Action scalar = () => np.choose(Idx, new object[] { T, "zz" });
            scalar.Should().Throw<TypeError>().WithMessage("Cannot cast scalar from dtype('<U2') to StringDType() according to the rule 'safe'");

            // Safe: S -> U, a number -> its text width, U2 -> U5.
            NDArray.AsStringArray(np.choose(Idx, new object[] { U, np.array(new long[] { 1, 2 }) })).Should().Equal("ab", "2");
            np.choose(Idx, new object[] { U, np.array(new[] { "zzzzz", "y" }) }).dtype.Should().Be(np.dtype("<U5"));
            NDArray.AsStringArray(np.choose(Idx, new object[] { U, true })).Should().Equal("ab", "True");
        }

        [TestMethod]
        public void Select_OneResultTypeCall_OverChoicesAndDefault()
        {
            // NumPy 2.4.2 promotes the choices and the default together: one error text, the default's DType in the list.
            Action act = () => np.select(new[] { C, C }, new object[] { U, 1 });
            act.Should().Throw<TypeError>().WithMessage(
                "Choicelist and default value do not have a common dtype: The DType <class 'numpy.dtypes.StrDType'> could not be " +
                "promoted by <class 'numpy.dtypes._PyLongDType'>. This means that no common DType exists for the given inputs. For " +
                "example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: " +
                "(<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyLongDType'>, <class 'numpy.dtypes._PyLongDType'>)");
            Action strDefault = () => np.select(new[] { C }, new object[] { 1 }, U["0"]);
            strDefault.Should().Throw<TypeError>().WithMessage("Choicelist and default value do not have a common dtype: The DType <class 'numpy.dtypes._PyLongDType'> could not be promoted by <class 'numpy.dtypes.StrDType'>*");

            // A strong number default widens the text.
            NDArray.AsStringArray(np.select(new[] { C }, new object[] { np.array(new long[] { 1, 2 }) }, "x")).Should().Equal("1", "x");
        }

        [TestMethod]
        public void Select_LiteralDefault_IsWeakAsAnNDArray()
        {
            var i8 = np.array(new sbyte[] { 1, 2 });
            var r = np.select(new[] { C }, new object[] { i8 }, (NDArray)1000);
            r.dtype.Should().Be(np.int8);
            r.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)-24);
        }

        [TestMethod]
        public void RConcat_Text_PromotesByDescriptor()
        {
            var i = np.array(new long[] { 1, 2 });
            NDArray.AsStringArray(np.r_[U, true]).Should().Equal("ab", "cd", "True");
            np.r_[U, true].dtype.Should().Be(np.dtype("<U5"));
            np.r_[i, U].dtype.Should().Be(np.dtype("<U21"));
            NDArray.AsStringArray(np.r_[i, U]).Should().Equal("1", "2", "ab", "cd");
            NDArray.AsStringArray(np.r_[U, NDArray.Scalar(1f)]).Should().Equal("ab", "cd", "1.0");
            var c = np.c_[i, U];
            c.shape.Should().Equal(2L, 2L);
            c.dtype.Should().Be(np.dtype("<U21"));
            np.r_[T, np.array(new[] { "zz" })].dtype.Should().Be(np.dtypes.StringDType());

            Action weak = () => { _ = np.r_[U, 1]; };
            weak.Should().Throw<DTypePromotionError>().WithMessage(PyLongVsStr);
            Action vstringNumber = () => { _ = np.r_[i, T]; };
            vstringNumber.Should().Throw<DTypePromotionError>().WithMessage("The DType <class 'numpy.dtypes.Int64DType'> could not be promoted by <class 'numpy.dtypes.StringDType'>*");
        }

        [TestMethod]
        public void RConcat_LiteralNDArrayEntry_IsWeak()
        {
            var i8 = np.array(new sbyte[] { 1, 2 });
            var r = np.r_[i8, (NDArray)1];
            r.dtype.Should().Be(np.int8);
            r.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)2, (sbyte)1);
        }

        [TestMethod]
        public void PutAndInsert_StrIntoNumbers_QuoteThePythonStr()
        {
            // np.put(a, 0, '1.5') / np.insert(a, 0, '1.5'): PyArray_FromAny(str, int64) runs int() on the str itself.
            Action put = () => np.put(np.zeros(new Shape(2), NPTypeCode.Int64), 0L, (object)"1.5");
            put.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '1.5'");
            Action insert = () => np.insert(np.zeros(new Shape(1), NPTypeCode.Int64), 0, (object)"1.5");
            insert.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '1.5'");

            var a = np.zeros(new Shape(2), NPTypeCode.Int64);
            np.put(a, 0L, (object)" 1_2 ");
            a.ToArray<long>().Should().Equal(12L, 0L);
            np.insert(np.zeros(new Shape(1), NPTypeCode.Double), 0, (object)"1e3").ToArray<double>().Should().Equal(1000.0, 0.0);
        }
    }
}
