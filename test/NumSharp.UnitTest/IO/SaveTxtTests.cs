using System;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    ///     <c>np.savetxt</c> — text serialization of a 1-D / 2-D array.
    /// </summary>
    /// <remarks>
    ///     Every expected string was probed against NumPy 2.4.2 (via <c>savetxt</c> into a
    ///     <c>io.StringIO</c>, i.e. the file-handle path with verbatim <c>\n</c>). The content cases use the
    ///     stream overload, which — like NumPy's file-handle path — does not translate newlines, so the
    ///     assertions are platform-independent. The filename overload's Python text-mode newline
    ///     translation (CRLF on Windows) is exercised separately in <see cref="FilePath_TranslatesNewline_ToOsLineSeparator"/>.
    /// </remarks>
    [TestClass]
    public class SaveTxtTests
    {
        private static string Cap(NDArray X, string fmt = "%.18e", string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ")
        {
            using var ms = new MemoryStream();
            np.savetxt(ms, X, fmt, delimiter, newline, header, footer, comments);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        private static string Cap(NDArray X, string[] fmt, string delimiter = " ")
        {
            using var ms = new MemoryStream();
            np.savetxt(ms, X, fmt, delimiter);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        // ---- default %.18e across dtypes -------------------------------------------------

        [TestMethod]
        public void Default_Float64_ColumnAndRows()
        {
            Assert.AreEqual("1.000000000000000000e+00\n2.000000000000000000e+00\n3.000000000000000000e+00\n",
                Cap(np.array(new double[] { 1, 2, 3 })));
            Assert.AreEqual("1.000000000000000000e+00 2.000000000000000000e+00\n3.000000000000000000e+00 4.000000000000000000e+00\n",
                Cap(np.array(new double[,] { { 1, 2 }, { 3, 4 } })));
        }

        [TestMethod]
        public void Default_WidensToDouble_LikeNumpyPercentOperator()
        {
            Assert.AreEqual("1.000000014901161194e-01\n", Cap(np.array(new float[] { 0.1f })));
            Assert.AreEqual("9.997558593750000000e-02\n", Cap(np.array(new float[] { 0.1f }).astype(NPTypeCode.Half)));
            Assert.AreEqual("1.000000000000000000e+00\n2.000000000000000000e+00\n3.000000000000000000e+00\n",
                Cap(np.array(new int[] { 1, 2, 3 })));
            // int64 past 2^53 loses precision in the double widening, exactly as NumPy's `%e` does.
            Assert.AreEqual("9.007199254740992000e+15\n", Cap(np.array(new long[] { 9007199254740993L })));
            Assert.AreEqual("1.844674407370955162e+19\n", Cap(np.array(new ulong[] { 18446744073709551615UL })));
            Assert.AreEqual("1.000000000000000000e+00\n0.000000000000000000e+00\n", Cap(np.array(new bool[] { true, false })));
        }

        [TestMethod]
        public void Default_ExtraDtypes()
        {
            Assert.AreEqual("-5.000000000000000000e+00\n5.000000000000000000e+00\n", Cap(np.array(new sbyte[] { -5, 5 })));
            Assert.AreEqual("2.000000000000000000e+02\n5.000000000000000000e+00\n", Cap(np.array(new byte[] { 200, 5 })));
            Assert.AreEqual("-3.000000000000000000e+02\n3.000000000000000000e+02\n", Cap(np.array(new short[] { -300, 300 })));
            Assert.AreEqual("4.000000000000000000e+04\n", Cap(np.array(new ushort[] { 40000 })));
            Assert.AreEqual("4.000000000000000000e+09\n", Cap(np.array(new uint[] { 4000000000 })));
        }

        // ---- rank handling ---------------------------------------------------------------

        [TestMethod]
        public void Rank_ZeroAndAboveTwo_Throw()
        {
            var e0 = Assert.ThrowsException<ValueError>(() => Cap(np.array(5.0)));
            Assert.AreEqual("Expected 1D or 2D array, got 0D array instead", e0.Message);
            var e3 = Assert.ThrowsException<ValueError>(() => Cap(np.ones(new Shape(2, 2, 2))));
            Assert.AreEqual("Expected 1D or 2D array, got 3D array instead", e3.Message);
        }

        [TestMethod]
        public void Rank_Empties()
        {
            Assert.AreEqual("", Cap(np.array(new double[0])));
            Assert.AreEqual("", Cap(np.zeros(new Shape(0, 3))));
            Assert.AreEqual("\n\n", Cap(np.zeros(new Shape(2, 0))));         // 2 rows, 0 columns
            Assert.AreEqual("7.000000000000000000e+00\n", Cap(np.array(new double[] { 7 })));
        }

        // ---- fmt string ------------------------------------------------------------------

        [TestMethod]
        public void Fmt_SingleSpec_Variants()
        {
            Assert.AreEqual("1\n2\n3\n", Cap(np.array(new int[] { 1, 2, 3 }), "%d"));
            Assert.AreEqual("1\n-1\n2\n", Cap(np.array(new double[] { 1.9, -1.9, 2.5 }), "%d"));       // truncate toward zero
            Assert.AreEqual("1.235\n2.000\n", Cap(np.array(new double[] { 1.23456, 2.0 }), "%.3f"));
            Assert.AreEqual("   1.50000\n   2.50000\n", Cap(np.array(new double[] { 1.5, 2.5 }), "%10.5f"));
            Assert.AreEqual("+1.50e+00\n-2.50e+00\n", Cap(np.array(new double[] { 1.5, -2.5 }), "%+.2e"));
            Assert.AreEqual("00001\n00022\n", Cap(np.array(new int[] { 1, 22 }), "%05d"));
            Assert.AreEqual("ff\n10\n", Cap(np.array(new int[] { 255, 16 }), "%x"));
            Assert.AreEqual("1         |2         \n", Cap(np.array(new int[,] { { 1, 2 } }), "%-10d", "|"));
        }

        [TestMethod]
        public void Fmt_MultiSpecStringAndList()
        {
            Assert.AreEqual("1-2\n3-4\n", Cap(np.array(new int[,] { { 1, 2 }, { 3, 4 } }), "%d-%d"));
            Assert.AreEqual("1 2\n", Cap(np.array(new int[,] { { 1, 2 } }), "%d"));                  // single spec replicated
            Assert.AreEqual("1 2.50\n3 4.50\n", Cap(np.array(new double[,] { { 1, 2.5 }, { 3, 4.5 } }), new[] { "%d", "%.2f" }));
        }

        [TestMethod]
        public void Fmt_IntegerPrecision_MinDigits()
        {
            // Python's `.precision` on d/i/u/x/X/o is the MINIMUM number of digits (left-zero-filled).
            Assert.AreEqual("00042\n-00001\n", Cap(np.array(new int[] { 42, -1 }), "%.5d"));
            Assert.AreEqual("0\n5\n", Cap(np.array(new int[] { 0, 5 }), "%.0d"));            // "0" is already one digit
            Assert.AreEqual("  007\n", Cap(np.array(new int[] { 7 }), "%5.3d"));
            // the '0' flag is honored TOGETHER with a precision (Python differs from C here).
            Assert.AreEqual("00000007\n", Cap(np.array(new int[] { 7 }), "%08.3d"));
            Assert.AreEqual("-0000007\n", Cap(np.array(new int[] { -7 }), "%08.3d"));
            Assert.AreEqual("+0000007\n", Cap(np.array(new int[] { 7 }), "%+08.3d"));
            Assert.AreEqual("007     \n", Cap(np.array(new int[] { 7 }), "%-8.3d"));         // left-justify -> spaces
            Assert.AreEqual("00FF\n", Cap(np.array(new int[] { 255 }), "%.4X"));
            Assert.AreEqual("052\n", Cap(np.array(new int[] { 42 }), "%.3o"));
            // alt-prefix + precision + zero-fill: the zeros go BETWEEN the "0x" and the digits.
            Assert.AreEqual("0x000ff\n", Cap(np.array(new int[] { 255 }), "%#.5x"));
            Assert.AreEqual("0x0000ff\n", Cap(np.array(new int[] { 255 }), "%#08x"));
            Assert.AreEqual("0x000000ff\n", Cap(np.array(new int[] { 255 }), "%#010.3x"));
            Assert.AreEqual("0o00052\n", Cap(np.array(new int[] { 42 }), "%#.5o"));
        }

        [TestMethod]
        public void Fmt_CharAndBool_MatchNumpyRaises()
        {
            // %c on an integer out of [0, 0x10FFFF] raises OverflowError (NumPy lets it propagate uncaught).
            var neg = Assert.ThrowsException<OverflowException>(() => Cap(np.array(new int[] { -1 }), "%c"));
            Assert.AreEqual("%c arg not in range(0x110000)", neg.Message);
            var over = Assert.ThrowsException<OverflowException>(() => Cap(np.array(new int[] { 0x110000 }), "%c"));
            Assert.AreEqual("%c arg not in range(0x110000)", over.Message);
            Assert.AreEqual("A\n", Cap(np.array(new int[] { 65 }), "%c"));                   // in-range -> the code point
            // bool is NOT an int for %c/%x/%X/%o — a TypeError reported as the dtype/format mismatch.
            var bc = Assert.ThrowsException<TypeError>(() => Cap(np.array(new bool[] { true }), "%c"));
            Assert.AreEqual("Mismatch between array dtype ('bool') and format specifier ('%c')", bc.Message);
            var bx = Assert.ThrowsException<TypeError>(() => Cap(np.array(new bool[] { true }), "%x"));
            Assert.AreEqual("Mismatch between array dtype ('bool') and format specifier ('%x')", bx.Message);
            var bo = Assert.ThrowsException<TypeError>(() => Cap(np.array(new bool[] { true }), "%o"));
            Assert.AreEqual("Mismatch between array dtype ('bool') and format specifier ('%o')", bo.Message);
            // bool DOES take %d/%i/%u (renders 1/0).
            Assert.AreEqual("1\n0\n", Cap(np.array(new bool[] { true, false }), "%d"));
        }

        [TestMethod]
        public void Fmt_Errors()
        {
            var a = Assert.ThrowsException<AttributeError>(() => Cap(np.array(new double[,] { { 1, 2.5 } }), new[] { "%d" }));
            Assert.AreEqual("fmt has wrong shape.  ['%d']", a.Message);
            var v = Assert.ThrowsException<ValueError>(() => Cap(np.array(new int[,] { { 1, 2 } }), "%d-%d-%d"));
            Assert.AreEqual("fmt has wrong number of % formats:  %d-%d-%d", v.Message);
        }

        // ---- delimiter / newline / header / footer / comments ----------------------------

        [TestMethod]
        public void DelimiterNewlineHeaderFooter()
        {
            Assert.AreEqual("1,2\n3,4\n", Cap(np.array(new int[,] { { 1, 2 }, { 3, 4 } }), "%d", ","));
            Assert.AreEqual("1 2;3 4;", Cap(np.array(new int[,] { { 1, 2 }, { 3, 4 } }), "%d", " ", ";"));
            Assert.AreEqual("# col\n1\n2\n", Cap(np.array(new int[] { 1, 2 }), "%d", " ", "\n", "col"));
            Assert.AreEqual("1\n2\n# end\n", Cap(np.array(new int[] { 1, 2 }), "%d", " ", "\n", "", "end"));
            Assert.AreEqual("# a\n# b\n1\n2\n", Cap(np.array(new int[] { 1, 2 }), "%d", " ", "\n", "a\nb"));
            Assert.AreEqual("// x\n1\n2\n", Cap(np.array(new int[] { 1, 2 }), "%d", " ", "\n", "x", "", "// "));
            Assert.AreEqual("# H\n1\n2\n# F\n", Cap(np.array(new int[] { 1, 2 }), "%d", " ", "\n", "H", "F"));
        }

        // ---- complex ---------------------------------------------------------------------

        [TestMethod]
        public void Complex_DefaultAndFmt()
        {
            Assert.AreEqual(
                " (1.000000000000000000e+00+2.000000000000000000e+00j)\n (3.000000000000000000e+00-4.000000000000000000e+00j)\n",
                Cap(np.array(new Complex[] { new Complex(1, 2), new Complex(3, -4) })));
            Assert.AreEqual(" (1.50e+00+2.50e+00j)\n", Cap(np.array(new Complex[] { new Complex(1.5, 2.5) }), "%.2e"));
            // the '+-' -> '-' fix-up on a negative imaginary part
            Assert.AreEqual(" (1.0-2.0j)\n", Cap(np.array(new Complex[] { new Complex(1, -2) }), "%.1f"));
            Assert.AreEqual(" 1.0 +2.0j 3.0 +4.0j\n",
                Cap(np.array(new Complex[,] { { new Complex(1, 2), new Complex(3, 4) } }), " %.1f %+.1fj %.1f %+.1fj"));
            Assert.AreEqual("1.0+2.0j (3.0+4.0j)\n",
                Cap(np.array(new Complex[,] { { new Complex(1, 2), new Complex(3, 4) } }), new[] { "%.1f%+.1fj", "(%.1f%+.1fj)" }));
            var v = Assert.ThrowsException<ValueError>(() => Cap(np.array(new Complex[,] { { new Complex(1, 2) } }), "%.1f %.1f %.1f"));
            Assert.AreEqual("fmt has wrong number of % formats:  %.1f %.1f %.1f", v.Message);
        }

        // ---- specials / %g / %s ----------------------------------------------------------

        [TestMethod]
        public void SpecialValues()
        {
            Assert.AreEqual("nan\ninf\n-inf\n",
                Cap(np.array(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })));
            Assert.AreEqual("+nan\n+inf\n-inf\n",
                Cap(np.array(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity }), "%+f"));
            Assert.AreEqual("  nan\n  inf\n", Cap(np.array(new double[] { double.NaN, double.PositiveInfinity }), "%5.1f"));
            Assert.AreEqual("NAN\nINF\n", Cap(np.array(new double[] { double.NaN, double.PositiveInfinity }), "%E"));
            Assert.AreEqual("-0.000000\n0.000000\n", Cap(np.array(new double[] { -0.0, 0.0 }), "%f"));
            Assert.AreEqual("-0.000000000000000000e+00\n", Cap(np.array(new double[] { -0.0 })));
        }

        [TestMethod]
        public void GeneralAndString()
        {
            Assert.AreEqual("1e-05\n0.0001\n1.23457e-10\n", Cap(np.array(new double[] { 1e-5, 1e-4, 1.234567e-10 }), "%g"));
            Assert.AreEqual("1e+06\n123456\n1.23457e+06\n", Cap(np.array(new double[] { 1e6, 123456.0, 1234567.0 }), "%g"));
            Assert.AreEqual("100000.\n1.50000\n", Cap(np.array(new double[] { 100000.0, 1.5 }), "%#g"));
            Assert.AreEqual("1.5\n2.0\n0.1\n", Cap(np.array(new double[] { 1.5, 2.0, 0.1 }), "%s"));
            Assert.AreEqual("0.1\n1.5\n", Cap(np.array(new float[] { 0.1f, 1.5f }), "%s"));  // float32 shortest, not the widened double
            Assert.AreEqual("True\nFalse\n", Cap(np.array(new bool[] { true, false }), "%s"));
            Assert.AreEqual("A\nB\n", Cap(np.array(new int[] { 65, 66 }), "%c"));
        }

        [TestMethod]
        public void HugeMagnitudes()
        {
            Assert.AreEqual("1.000000000000000053e+300\n1.000000000000000025e-300\n",
                Cap(np.array(new double[] { 1e300, 1e-300 })));
            Assert.AreEqual("1.23e+123\n", Cap(np.array(new double[] { 1.23456789e123 }), "%.2e"));
            // %d on a large finite float is an exact big-integer truncation, exactly as int(1e20).
            Assert.AreEqual("100000000000000000000\n", Cap(np.array(new double[] { 1e20 }), "%.0f"));
        }

        // ---- integer-format-on-float rules (CPython %-operator) --------------------------

        [TestMethod]
        public void IntegerFormat_OnFloat_MatchesNumpyRaises()
        {
            // %x/%X/%o reject a float outright -> TypeError, reported as the dtype/format mismatch.
            var x = Assert.ThrowsException<TypeError>(() => Cap(np.array(new double[] { 1.5, 255.0 }), "%x"));
            Assert.AreEqual("Mismatch between array dtype ('float64') and format specifier ('%x')", x.Message);
            // %d/%i/%u accept a FINITE float (truncating) but raise on nan/inf.
            Assert.AreEqual("3\n-3\n", Cap(np.array(new double[] { 3.9, -3.9 }), "%d"));
            var nan = Assert.ThrowsException<ValueError>(() => Cap(np.array(new double[] { 1.5, double.NaN }), "%d"));
            Assert.AreEqual("cannot convert float NaN to integer", nan.Message);
            var inf = Assert.ThrowsException<OverflowException>(() => Cap(np.array(new double[] { double.PositiveInfinity }), "%d"));
            Assert.AreEqual("cannot convert float infinity to integer", inf.Message);
            // integral dtypes still take %x/%o.
            Assert.AreEqual("ff\n8\n", Cap(np.array(new int[] { 255, 8 }), "%x"));
        }

        // ---- memory layouts (read in logical C-order) ------------------------------------

        [TestMethod]
        public void Layouts_ReadInLogicalCOrder()
        {
            var a6 = np.arange(6).astype(NPTypeCode.Int32);
            Assert.AreEqual("0 3\n1 4\n2 5\n", Cap(a6.reshape(2, 3).T, "%d"));
            Assert.AreEqual("2\n4\n6\n", Cap(np.arange(10).astype(NPTypeCode.Int32)["2:8:2"], "%d"));
            Assert.AreEqual("4\n3\n2\n1\n0\n", Cap(np.arange(5).astype(NPTypeCode.Int32)["::-1"], "%d"));
            Assert.AreEqual("2 1 0\n5 4 3\n", Cap(a6.reshape(2, 3)[":, ::-1"], "%d"));
            Assert.AreEqual("0 1 2\n3 4 5\n", Cap(np.asfortranarray(a6.reshape(2, 3)), "%d"));
            Assert.AreEqual("0 1 2\n0 1 2\n",
                Cap(np.broadcast_to(np.arange(3).astype(NPTypeCode.Int32), new Shape(2, 3)), "%d"));
        }

        // ---- filename target: text-mode newline translation + gzip -----------------------

        [TestMethod]
        public void FilePath_TranslatesNewline_ToOsLineSeparator()
        {
            string dir = Path.Combine(Path.GetTempPath(), "ns_savetxt_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string f = Path.Combine(dir, "a.txt");
                np.savetxt(f, np.array(new int[,] { { 1, 2 }, { 3, 4 } }), fmt: "%d");
                string nl = Environment.NewLine;
                Assert.AreEqual($"1 2{nl}3 4{nl}", File.ReadAllText(f));

                // header newline is translated too
                string f2 = Path.Combine(dir, "b.txt");
                np.savetxt(f2, np.array(new int[] { 1 }), fmt: "%d", header: "h1\nh2");
                Assert.AreEqual($"# h1{nl}# h2{nl}1{nl}", File.ReadAllText(f2));
            }
            finally { Directory.Delete(dir, true); }
        }

        [TestMethod]
        public void FilePath_Gzip_RoundTrips()
        {
            string dir = Path.Combine(Path.GetTempPath(), "ns_savetxt_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string f = Path.Combine(dir, "c.txt.gz");
                np.savetxt(f, np.array(new int[] { 1, 2, 3 }), fmt: "%d");

                byte[] raw = File.ReadAllBytes(f);
                Assert.AreEqual(0x1f, raw[0]);   // gzip magic
                Assert.AreEqual(0x8b, raw[1]);

                using var gz = new GZipStream(File.OpenRead(f), CompressionMode.Decompress);
                using var sr = new StreamReader(gz);
                string nl = Environment.NewLine;
                Assert.AreEqual($"1{nl}2{nl}3{nl}", sr.ReadToEnd());
            }
            finally { Directory.Delete(dir, true); }
        }

        [TestMethod]
        public void FilePath_LeavesEmptyFile_OnValidationError()
        {
            // NumPy creates/truncates the file, THEN validates, so a rank error leaves an empty file.
            string dir = Path.Combine(Path.GetTempPath(), "ns_savetxt_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string f = Path.Combine(dir, "err.txt");
                Assert.ThrowsException<ValueError>(() => np.savetxt(f, np.array(5.0)));
                Assert.IsTrue(File.Exists(f));
                Assert.AreEqual(0, new FileInfo(f).Length);
            }
            finally { Directory.Delete(dir, true); }
        }

        // ---- TextWriter target: verbatim newline -----------------------------------------

        [TestMethod]
        public void TextWriter_WritesVerbatimNewline()
        {
            var sw = new StringWriter();
            np.savetxt(sw, np.array(new int[,] { { 1, 2 }, { 3, 4 } }), fmt: "%d");
            Assert.AreEqual("1 2\n3 4\n", sw.ToString());
        }

        // ---- argument guards -------------------------------------------------------------

        [TestMethod]
        public void NullGuards()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.savetxt((string)null, np.array(new int[] { 1 })));
            Assert.ThrowsException<ArgumentNullException>(() => np.savetxt((Stream)null, np.array(new int[] { 1 })));
            Assert.ThrowsException<ArgumentNullException>(() => np.savetxt(new MemoryStream(), (NDArray)null));
        }
    }
}
