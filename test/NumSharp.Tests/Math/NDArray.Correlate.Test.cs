using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.correlate — cross-correlation of two 1-D sequences. All expected values are from
    /// NumPy 2.4.2 (probed directly). Covers the docstring examples, the swap+reverse of the
    /// non-commutative engine, complex conjugation, every dtype, and the error taxonomy.
    /// </summary>
    [TestClass]
    public class NdArrayCorrelateTest
    {
        // ---------------------------- docstring examples ----------------------------

        [TestMethod]
        public void Correlate_Docstring_Valid_IsDefault()
        {
            // np.correlate([1,2,3],[0,1,0.5]) -> [3.5]   (default mode is 'valid')
            var r = np.correlate(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 0, 1, 0.5 }));
            r.Data<double>().Should().Equal(new double[] { 3.5 });
        }

        [TestMethod]
        public void Correlate_Docstring_Same()
        {
            var r = np.correlate(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 0, 1, 0.5 }), "same");
            r.Data<double>().Should().Equal(new double[] { 2.0, 3.5, 3.0 });
        }

        [TestMethod]
        public void Correlate_Docstring_Full()
        {
            var r = np.correlate(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 0, 1, 0.5 }), "full");
            r.Data<double>().Should().Equal(new double[] { 0.5, 2.0, 3.5, 3.0, 0.0 });
        }

        // -------------------- non-commutative swap + output reversal --------------------
        // When len(a) < len(v) the engine correlates(v, a) then reverses the output, so
        // correlate(a, v) is the time-reversal of correlate(v, a).

        [TestMethod]
        public void Correlate_ShorterA_SwapsAndReverses_Valid()
        {
            var r = np.correlate(np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 2, 3, 4 }), "valid");
            r.Data<long>().Should().Equal(new long[] { 11, 8, 5 });
        }

        [TestMethod]
        public void Correlate_ShorterA_SwapsAndReverses_Full()
        {
            var r = np.correlate(np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 2, 3, 4 }), "full");
            r.Data<long>().Should().Equal(new long[] { 4, 11, 8, 5, 2 });
        }

        [TestMethod]
        public void Correlate_IsReverseOf_Swapped()
        {
            // correlate([1,2],[1,2,3,4]) == reverse of correlate([1,2,3,4],[1,2])
            var ab = np.correlate(np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 2, 3, 4 }), "valid");
            var ba = np.correlate(np.array(new long[] { 1, 2, 3, 4 }), np.array(new long[] { 1, 2 }), "valid");
            ab.Data<long>().Should().Equal(new long[] { 11, 8, 5 });
            ba.Data<long>().Should().Equal(new long[] { 5, 8, 11 });
        }

        // ----------------------------- complex conjugation -----------------------------

        [TestMethod]
        public void Correlate_Complex_ConjugatesV()
        {
            // np.correlate([1+1j,2,3-1j],[0,1,0.5j],'full') -> [0.5-0.5j, 1+0j, 1.5-1.5j, 3-1j, 0+0j]
            var a = np.array(new Complex[] { new(1, 1), 2, new(3, -1) });
            var v = np.array(new Complex[] { 0, 1, new(0, 0.5) });
            var r = np.correlate(a, v, "full").ToArray<Complex>();
            r.Should().Equal(new Complex[] { new(0.5, -0.5), new(1, 0), new(1.5, -1.5), new(3, -1), new(0, 0) });
        }

        [TestMethod]
        public void Correlate_Complex_SwappedIsTimeReversedConjugate()
        {
            // np.correlate([0,1,0.5j],[1+1j,2,3-1j],'full') -> [0+0j, 3+1j, 1.5+1.5j, 1+0j, 0.5+0.5j]
            var a = np.array(new Complex[] { 0, 1, new(0, 0.5) });
            var v = np.array(new Complex[] { new(1, 1), 2, new(3, -1) });
            var r = np.correlate(a, v, "full").ToArray<Complex>();
            r.Should().Equal(new Complex[] { new(0, 0), new(3, 1), new(1.5, 1.5), new(1, 0), new(0.5, 0.5) });
        }

        [TestMethod]
        public void Correlate_Complex_Valid()
        {
            var a = np.array(new Complex[] { new(1, 2), new(3, -1), new(0, 2) });
            var v = np.array(new Complex[] { new(0, 1), 2 });
            var r = np.correlate(a, v, "valid").ToArray<Complex>();
            r.Should().Equal(new Complex[] { new(8, -3), new(-1, 1) });
        }

        // -------------------------------- dtype coverage --------------------------------

        [TestMethod]
        public void Correlate_Int32_ValidAndFull()
        {
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var v = np.array(new int[] { 1, 0, -1 });
            np.correlate(a, v, "valid").Data<int>().Should().Equal(new[] { -2, -2, -2 });
            np.correlate(a, v, "full").Data<int>().Should().Equal(new[] { -1, -2, -2, -2, -2, 4, 5 });
            np.correlate(a, v).GetTypeCode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Correlate_UInt8_Wraps()
        {
            var a = np.array(new byte[] { 10, 20, 30 });
            var v = np.array(new byte[] { 2, 1 });
            np.correlate(a, v, "valid").Data<byte>().Should().Equal(new byte[] { 40, 70 });
        }

        [TestMethod]
        public void Correlate_Float32_Same()
        {
            var a = np.array(new float[] { 1, 2, 3, 4 });
            var v = np.array(new float[] { 0.5f, 1f, 0.5f });
            np.correlate(a, v, "same").Data<float>().Should().Equal(new float[] { 2f, 4f, 6f, 5.5f });
        }

        [TestMethod]
        public void Correlate_Float16_Valid()
        {
            var a = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 });
            var v = np.array(new Half[] { (Half)1, (Half)1 });
            np.correlate(a, v, "valid").Data<Half>().Should().Equal(new Half[] { (Half)3, (Half)5, (Half)7 });
        }

        [TestMethod]
        public void Correlate_Bool_OrOfAnds()
        {
            var a = np.array(new bool[] { true, false, true });
            var v = np.array(new bool[] { true, true });
            np.correlate(a, v, "full").Data<bool>().Should().Equal(new bool[] { true, true, true, true });
        }

        [TestMethod]
        public void Correlate_Char_RoutesThroughUShortKernel()
        {
            // 'A'=65 etc.  valid corr with [1,1] -> [A+B, B+C, C+D] = [131,133,135]
            var a = np.array(new char[] { 'A', 'B', 'C', 'D' });
            var v = np.array(new char[] { (char)1, (char)1 });
            var r = np.correlate(a, v, "valid");
            r.GetTypeCode.Should().Be(NPTypeCode.Char);
            r.ToArray<char>().Select(c => (int)c).Should().Equal(new[] { 131, 133, 135 });
        }

        [TestMethod]
        public void Correlate_DtypePromotion()
        {
            // NEP50: int32 + float32 promotes to float64 (int32 exceeds float32's exact range).
            np.correlate(np.array(new int[] { 1 }), np.array(new float[] { 1 })).GetTypeCode.Should().Be(NPTypeCode.Double);
            np.correlate(np.array(new byte[] { 1 }), np.array(new sbyte[] { 1 })).GetTypeCode.Should().Be(NPTypeCode.Int16);
            np.correlate(np.array(new float[] { 1 }), np.array(new Complex[] { 1 })).GetTypeCode.Should().Be(NPTypeCode.Complex);
        }

        // ------------------------------ non-contiguous inputs ------------------------------

        [TestMethod]
        public void Correlate_StridedInput_MatchesContiguous()
        {
            // a[::2] strided view over arange(20) correlated with [1,2,0.5], full.
            var baseA = np.arange(20).astype(NPTypeCode.Double);
            var v = np.array(new double[] { 1, 2, 0.5 });
            var strided = np.correlate(baseA["::2"], v, "full");
            var copied = np.correlate(baseA["::2"].copy(), v, "full");
            strided.Data<double>().Should().Equal(copied.Data<double>().ToArray());
            strided.GetDouble(2).Should().Be(6.0); // matches numpy
        }

        [TestMethod]
        public void Correlate_ReversedView_MatchesCopy()
        {
            var baseA = np.arange(20).astype(NPTypeCode.Double);
            var v = np.array(new double[] { 1, 2, 0.5 });
            var rev = baseA["10:16"]["::-1"];
            np.correlate(rev, v, "same").Data<double>().Should().Equal(np.correlate(rev.copy(), v, "same").Data<double>().ToArray());
        }

        // ---------------------------------- error taxonomy ----------------------------------

        [TestMethod]
        public void Correlate_EmptyA_Throws()
        {
            Action act = () => np.correlate(np.array(new double[0]), np.array(new double[] { 1 }));
            act.Should().Throw<ArgumentException>().WithMessage("first array argument cannot be empty*");
        }

        [TestMethod]
        public void Correlate_EmptyV_Throws()
        {
            Action act = () => np.correlate(np.array(new double[] { 1 }), np.array(new double[0]));
            act.Should().Throw<ArgumentException>().WithMessage("second array argument cannot be empty*");
        }

        [TestMethod]
        public void Correlate_BadMode_Throws()
        {
            // NumPy 2.4.2: ValueError, the exact text (no .NET " (Parameter 'mode')" suffix).
            Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), "bogus");
            act.Should().ThrowExactly<ValueError>().Which.Message.Should()
                .Be("mode must be one of 'valid', 'same', or 'full' (got 'bogus')");
        }

        [TestMethod]
        public void Correlate_ModeIsCaseSensitive()
        {
            // NumPy's parser is case-sensitive: 'VALID' is a near-miss, not an alias.
            Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), "VALID");
            act.Should().ThrowExactly<ValueError>().Which.Message.Should()
                .Be("Use one of 'valid', 'same', or 'full' for convolve/correlate mode");
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>correlatemode_parser</c> decides by the FIRST character: one that names a mode, followed by
        ///     anything but that mode's exact name (another case, a suffix, a prefix, a trailing newline or NUL), is its
        ///     own near-miss text, which does not quote the argument.
        /// </summary>
        [TestMethod]
        public void Correlate_ModeNearMisses_AreTheParsersOwnText()
        {
            foreach (string mode in new[] { "Valid", "valid ", "v", "s", "f", "fu", "Same", "FULL", "same\n", "valid\0" })
            {
                Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), mode);
                act.Should().ThrowExactly<ValueError>().Which.Message.Should()
                    .Be("Use one of 'valid', 'same', or 'full' for convolve/correlate mode", "mode {0} is a near miss",
                        mode.Replace("\0", "\\0").Replace("\n", "\\n"));
            }
        }

        /// <summary>
        ///     NumPy 2.4.2: a mode whose first character names no mode (the empty text included) is
        ///     <c>string_converter_helper</c>'s text with the argument in Python's <c>repr</c> — double quotes around a text
        ///     holding a single quote, escapes for controls, printable non-ASCII characters (an astral one included) raw.
        /// </summary>
        [TestMethod]
        public void Correlate_BadMode_QuotesTheArgumentAsPythonRepr()
        {
            var cases = new (string Mode, string Got)[]
            {
                ("it's", "(got \"it's\")"),
                ("a\"b'c", "(got 'a\"b\\'c')"),
                ("", "(got '')"),
                ("x", "(got 'x')"),
                ("\n", "(got '\\n')"),
                ("\t", "(got '\\t')"),
                ("\0", "(got '\\x00')"),
                ("\u007f", "(got '\\x7f')"),
                ("\u00e9", "(got '\u00e9')"),
                ("\U0001F600", "(got '\U0001F600')"),
                ("\U00010000", "(got '\U00010000')"),
            };
            foreach (var (mode, got) in cases)
            {
                Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), mode);
                act.Should().ThrowExactly<ValueError>().Which.Message.Should()
                    .Be("mode must be one of 'valid', 'same', or 'full' " + got, "NumPy quotes the argument as {0}", got);
            }
        }

        /// <summary>
        ///     NumPy 2.4.2: <c>mode=None</c> reaches the converter's integer branch, which cannot read it — TypeError
        ///     <c>convolve/correlate mode not understood</c>, raised while the arguments are parsed, so before the depth
        ///     check of a 0-d operand. NumSharp's null mode is that None (it was a NullReferenceException).
        /// </summary>
        [TestMethod]
        public void Correlate_NullMode_IsNumPysNoneTypeError()
        {
            Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), null);
            act.Should().ThrowExactly<TypeError>().Which.Message.Should().Be("convolve/correlate mode not understood");

            Action zeroD = () => np.correlate(NDArray.Scalar(1.0), np.array(new double[] { 1 }), null);
            zeroD.Should().ThrowExactly<TypeError>().Which.Message.Should().Be("convolve/correlate mode not understood");

            Action instance = () => np.array(new double[] { 1, 2 }).correlate(np.array(new double[] { 1 }), null);
            instance.Should().ThrowExactly<TypeError>().Which.Message.Should().Be("convolve/correlate mode not understood");
        }

        /// <summary>
        ///     NumPy 2.4.2 encodes the mode to UTF-8 before its parser reads it, so a lone surrogate fails with CPython's
        ///     <c>UnicodeEncodeError</c>, even after a letter that names a mode.
        /// </summary>
        [TestMethod]
        public void Correlate_LoneSurrogateMode_IsUnicodeEncodeError()
        {
            var cases = new (string Mode, string Text)[]
            {
                ("\ud800", "'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed"),
                ("v\ud800", "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed"),
                ("\udc00\ud800", "'utf-8' codec can't encode characters in position 0-1: surrogates not allowed"),
            };
            foreach (var (mode, text) in cases)
            {
                Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), mode);
                act.Should().ThrowExactly<UnicodeEncodeError>().Which.Message.Should().Be(text);
            }
        }

        /// <summary>
        ///     NumPy 2.4.2 parses np.correlate's mode with its arguments, before the operands are read: an empty, a 0-d and a
        ///     two-dimensional operand all report a bad mode first.
        /// </summary>
        [TestMethod]
        public void Correlate_ModeIsParsedBeforeTheOperands()
        {
            var operands = new[] { np.array(new double[0]), NDArray.Scalar(1.0), np.zeros(new Shape(2L, 2L)) };
            foreach (var a in operands)
            {
                Action act = () => np.correlate(a, np.array(new double[] { 1, 0.5 }), "bogus");
                act.Should().ThrowExactly<ValueError>().Which.Message.Should()
                    .Be("mode must be one of 'valid', 'same', or 'full' (got 'bogus')", "an operand of shape {0}", a.Shape);
            }
        }

        [TestMethod]
        public void Correlate_TwoDimensional_Throws()
        {
            // NumPy 2.4.2: ValueError "object too deep for desired array" (PyArray_FromAny's depth check, 1..1).
            Action act = () => np.correlate(np.arange(6).reshape(2, 3).astype(NPTypeCode.Double), np.array(new double[] { 1 }));
            act.Should().Throw<ValueError>().WithMessage("object too deep for desired array");
        }
    }
}
