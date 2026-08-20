using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    ///     <c>np.loadtxt</c> — text parsing into a 1-D / 2-D array. Every expected result was probed against
    ///     NumPy 2.4.2. bool parses through int (so "0"/"1"), ints range-check and unsigned reject negatives,
    ///     float uses <c>PyOS_string_to_double</c> semantics (case-insensitive inf/nan; rejects hex/"_"/junk),
    ///     complex is <c>to_complex_int</c>; <c>ndmin</c> squeezes/expands the C parser's 2-D result.
    /// </summary>
    [TestClass]
    public class LoadTxtTests
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        private static NDArray L(string text, NPTypeCode dt = NPTypeCode.Double, string comments = "#",
            string delimiter = null, int skiprows = 0, int[] usecols = null, bool unpack = false, int ndmin = 0,
            int? max_rows = null, string quotechar = null)
            => np.loadtxt(new StringReader(text), dt, comments, delimiter, null, skiprows, usecols, unpack, ndmin, max_rows, quotechar);

        private static double[] Vals(NDArray a)
        {
            var r = a.ravel();
            var v = new double[r.size];
            for (long i = 0; i < r.size; i++) v[i] = Convert.ToDouble(r.GetAtIndex(i), CI);
            return v;
        }

        private static void AssertArr(NDArray a, int[] shape, NPTypeCode tc, double[] vals)
        {
            Assert.AreEqual(string.Join(",", shape), string.Join(",", a.shape), "shape");
            Assert.AreEqual(tc, a.typecode);
            double[] got = Vals(a);
            Assert.AreEqual(vals.Length, got.Length, "value count");
            for (int i = 0; i < vals.Length; i++)
                Assert.IsTrue(got[i] == vals[i] || (double.IsNaN(got[i]) && double.IsNaN(vals[i])), $"val[{i}] {got[i]} != {vals[i]}");
        }

        // ---- shapes / ndmin --------------------------------------------------------------

        [TestMethod]
        public void Shapes_AndNdminSqueeze()
        {
            AssertArr(L("0 1\n2 3"), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 0, 1, 2, 3 });
            AssertArr(L("0 1 2"), new[] { 3 }, NPTypeCode.Double, new double[] { 0, 1, 2 });        // one row -> squeezed
            AssertArr(L("0\n1\n2"), new[] { 3 }, NPTypeCode.Double, new double[] { 0, 1, 2 });       // one col -> squeezed
            AssertArr(L("5"), Array.Empty<int>(), NPTypeCode.Double, new double[] { 5 });            // single -> 0-d
            AssertArr(L("5", ndmin: 2), new[] { 1, 1 }, NPTypeCode.Double, new double[] { 5 });
            AssertArr(L("5", ndmin: 1), new[] { 1 }, NPTypeCode.Double, new double[] { 5 });
            AssertArr(L("0 1 2", ndmin: 2), new[] { 1, 3 }, NPTypeCode.Double, new double[] { 0, 1, 2 });
            AssertArr(L(""), new[] { 0 }, NPTypeCode.Double, Array.Empty<double>());
            AssertArr(L("", ndmin: 2), new[] { 0, 1 }, NPTypeCode.Double, Array.Empty<double>());
        }

        [TestMethod]
        public void Whitespace_CollapsesAndSkipsBlankLines()
        {
            AssertArr(L("  1   2  \n 3  4 "), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 2\n\n3 4"), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 2\n3 4\n"), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1\t2\n3\t4"), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("\n\n1 2\n\n"), new[] { 2 }, NPTypeCode.Double, new double[] { 1, 2 });
        }

        // ---- dtypes ----------------------------------------------------------------------

        [TestMethod]
        public void Dtypes()
        {
            AssertArr(L("1 2\n3 4", NPTypeCode.Int64), new[] { 2, 2 }, NPTypeCode.Int64, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 2", NPTypeCode.Int32), new[] { 2 }, NPTypeCode.Int32, new double[] { 1, 2 });
            AssertArr(L("0.1 0.2", NPTypeCode.Single), new[] { 2 }, NPTypeCode.Single, new double[] { (float)0.1, (float)0.2 });
            AssertArr(L("0 1\n2 0", NPTypeCode.Boolean), new[] { 2, 2 }, NPTypeCode.Boolean, new double[] { 0, 1, 1, 0 });
            AssertArr(L("-1 0 1", NPTypeCode.Boolean), new[] { 3 }, NPTypeCode.Boolean, new double[] { 1, 0, 1 }); // bool via int
            AssertArr(L("-300 300", NPTypeCode.Int16), new[] { 2 }, NPTypeCode.Int16, new double[] { -300, 300 });
            AssertArr(L("18446744073709551615", NPTypeCode.UInt64), Array.Empty<int>(), NPTypeCode.UInt64, new double[] { 1.8446744073709552e19 });
        }

        [TestMethod]
        public void Complex_ParsesLikeToComplexInt()
        {
            var a = L("1+2j 3-4j", NPTypeCode.Complex);
            Assert.AreEqual("2", string.Join(",", a.shape));
            Assert.AreEqual(new Complex(1, 2), (Complex)a.ravel().GetAtIndex(0));
            Assert.AreEqual(new Complex(3, -4), (Complex)a.ravel().GetAtIndex(1));
            Assert.AreEqual(new Complex(1, 2), (Complex)L("(1+2j)", NPTypeCode.Complex).GetAtIndex(0));
            Assert.AreEqual(new Complex(0, 2), (Complex)L("2j", NPTypeCode.Complex).GetAtIndex(0));
            Assert.AreEqual(new Complex(1, -2), (Complex)L("1+-2j", NPTypeCode.Complex).GetAtIndex(0)); // '+-' -> '-'
        }

        [TestMethod]
        public void FloatParsing_InfNanSci()
        {
            AssertArr(L("inf -inf nan"), new[] { 3 }, NPTypeCode.Double,
                new double[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            AssertArr(L("Inf NAN Infinity"), new[] { 3 }, NPTypeCode.Double,
                new double[] { double.PositiveInfinity, double.NaN, double.PositiveInfinity });
            AssertArr(L("1e5 1.5E-3 +2.5"), new[] { 3 }, NPTypeCode.Double, new double[] { 1e5, 1.5e-3, 2.5 });
            AssertArr(L(".5 5."), new[] { 2 }, NPTypeCode.Double, new double[] { 0.5, 5.0 });
        }

        // ---- delimiter / comments / skiprows / max_rows / usecols ------------------------

        [TestMethod]
        public void DelimiterAndComments()
        {
            AssertArr(L("1,2\n3,4", delimiter: ","), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 , 2", delimiter: ","), new[] { 2 }, NPTypeCode.Double, new double[] { 1, 2 });
            AssertArr(L("# header\n1 2\n3 4"), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 2 # note\n3 4"), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 2\n3 4", comments: null), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 2\n//c\n3 4", comments: "//"), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
        }

        [TestMethod]
        public void SkiprowsMaxRowsUsecolsUnpack()
        {
            AssertArr(L("skip\n1 2\n3 4", skiprows: 1), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(L("1 2\n3 4\n5 6", max_rows: 1), new[] { 2 }, NPTypeCode.Double, new double[] { 1, 2 });
            AssertArr(L("1 2\n\n3 4\n5 6", max_rows: 2), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 }); // blanks not counted
            AssertArr(L("1 2\n3 4", max_rows: 0), new[] { 0 }, NPTypeCode.Double, Array.Empty<double>());
            AssertArr(L("1 2 3\n4 5 6", usecols: new[] { 1 }), new[] { 2 }, NPTypeCode.Double, new double[] { 2, 5 });
            AssertArr(L("1 2 3\n4 5 6", usecols: new[] { 0, 2 }), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 3, 4, 6 });
            AssertArr(L("1 2 3\n4 5 6", usecols: new[] { -1 }), new[] { 2 }, NPTypeCode.Double, new double[] { 3, 6 });
            AssertArr(L("1 2\n3 4", unpack: true), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 3, 2, 4 });
        }

        // ---- errors ----------------------------------------------------------------------

        [TestMethod]
        public void Errors_MatchNumpyVerbatim()
        {
            AssertMsg<ValueError>(() => L("1 2\n3 4 5"),
                "the number of columns changed from 2 to 3 at row 2; use `usecols` to select a subset and avoid this error");
            AssertMsg<ValueError>(() => L("True False", NPTypeCode.Boolean), "could not convert string 'True' to bool at row 0, column 1.");
            AssertMsg<ValueError>(() => L("300", NPTypeCode.Byte), "could not convert string '300' to uint8 at row 0, column 1.");
            AssertMsg<ValueError>(() => L("1.5", NPTypeCode.Int64), "could not convert string '1.5' to int64 at row 0, column 1.");
            AssertMsg<ValueError>(() => L("1,,3", delimiter: ","), "could not convert string '' to float64 at row 0, column 2.");
            AssertMsg<ValueError>(() => L("0x1.4p+2"), "could not convert string '0x1.4p+2' to float64 at row 0, column 1.");
            AssertMsg<ValueError>(() => L("1 2\n3 x"), "could not convert string 'x' to float64 at row 1, column 2.");
            AssertMsg<ValueError>(() => L("-1", NPTypeCode.Byte), "could not convert string '-1' to uint8 at row 0, column 1.");
            AssertMsg<ValueError>(() => L("200", NPTypeCode.SByte), "could not convert string '200' to int8 at row 0, column 1.");
            AssertMsg<ValueError>(() => L("1 2 3", usecols: new[] { 5 }), "invalid column index 5 at row 1 with 3 columns");
            AssertMsg<TypeError>(() => L("1::2", delimiter: "::"), "Text reading control character must be a single unicode character or None; but got: '::'");
            AssertMsg<ValueError>(() => L("1 2", ndmin: 3), "Illegal value of ndmin keyword: 3");
        }

        private static void AssertMsg<T>(Func<NDArray> f, string msg) where T : Exception
        {
            var e = Assert.ThrowsException<T>(f);
            Assert.AreEqual(msg, e.Message);
        }

        // ---- converters / quotechar ------------------------------------------------------

        [TestMethod]
        public void Converters()
        {
            var dict = new Dictionary<int, Func<string, object>>
            {
                { 0, x => Math.Floor(double.Parse(x, NumberStyles.Float, CI)) },
                { 1, x => Math.Ceiling(double.Parse(x, NumberStyles.Float, CI)) },
            };
            var a = np.loadtxt(new StringReader("1.618, 2.296"), NPTypeCode.Double, "#", ",", dict);
            AssertArr(a, new[] { 2 }, NPTypeCode.Double, new double[] { 1, 3 });

            Func<string, object> hex = x => (double)long.Parse(x.Trim().Substring(2), NumberStyles.HexNumber, CI);
            var b = np.loadtxt(new StringReader("0xDE 0xAD"), NPTypeCode.Double, "#", null, hex);
            AssertArr(b, new[] { 2 }, NPTypeCode.Double, new double[] { 222, 173 });
        }

        [TestMethod]
        public void QuoteChar()
        {
            var a = np.loadtxt(new StringReader("\"1\" \"2\""), NPTypeCode.Double, "#", null, null, 0, null, false, 0, null, "\"");
            AssertArr(a, new[] { 2 }, NPTypeCode.Double, new double[] { 1, 2 });
            var b = np.loadtxt(new StringReader("1,\"2\",3"), NPTypeCode.Double, "#", ",", null, 0, null, false, 0, null, "\"");
            AssertArr(b, new[] { 3 }, NPTypeCode.Double, new double[] { 1, 2, 3 });
        }

        // ---- input sources + round-trip --------------------------------------------------

        [TestMethod]
        public void InputSources()
        {
            using (var ms = new MemoryStream(Encoding.ASCII.GetBytes("1 2\n3 4")))
                AssertArr(np.loadtxt(ms), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(np.loadtxt(new[] { "1 2", "3 4" }), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });
            AssertArr(np.loadtxt(new[] { "1 2\n3 4" }), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 }); // embedded newline
        }

        [TestMethod]
        public void FilePath_PlainAndGzip_AndCrlf()
        {
            string dir = Path.Combine(Path.GetTempPath(), "ns_loadtxt_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string fp = Path.Combine(dir, "a.txt");
                File.WriteAllText(fp, "1 2\n3 4\n");
                AssertArr(np.loadtxt(fp), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });

                string fc = Path.Combine(dir, "c.txt");
                File.WriteAllBytes(fc, Encoding.ASCII.GetBytes("1 2\r\n3 4\r\n"));
                AssertArr(np.loadtxt(fc), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 1, 2, 3, 4 });

                string fg = Path.Combine(dir, "a.txt.gz");
                using (var gz = new GZipStream(File.Create(fg), CompressionMode.Compress))
                    gz.Write(Encoding.ASCII.GetBytes("5 6\n7 8\n"));
                AssertArr(np.loadtxt(fg), new[] { 2, 2 }, NPTypeCode.Double, new double[] { 5, 6, 7, 8 });
            }
            finally { Directory.Delete(dir, true); }
        }

        [TestMethod]
        public void RoundTrip_SaveThenLoad()
        {
            string dir = Path.Combine(Path.GetTempPath(), "ns_loadtxt_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                // int 2-D with a comma delimiter
                var orig = np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4);
                string f = Path.Combine(dir, "rt.txt");
                np.savetxt(f, orig, fmt: "%d", delimiter: ",");
                AssertArr(np.loadtxt(f, NPTypeCode.Int32, "#", ","), new[] { 3, 4 }, NPTypeCode.Int32,
                    Enumerable.Range(0, 12).Select(x => (double)x).ToArray());

                // floats incl. nan/inf/-0.0 — text canonicalizes NaN to +qNaN (0x7FF8…), as NumPy does.
                var fa = np.array(new double[] { 1.5, double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.0 });
                string f2 = Path.Combine(dir, "rt2.txt");
                np.savetxt(f2, fa);
                var rt = np.loadtxt(f2).ravel();
                long[] expBits =
                {
                    BitConverter.DoubleToInt64Bits(1.5), 0x7FF8000000000000L,
                    BitConverter.DoubleToInt64Bits(double.PositiveInfinity),
                    BitConverter.DoubleToInt64Bits(double.NegativeInfinity),
                    BitConverter.DoubleToInt64Bits(-0.0),
                };
                for (int i = 0; i < 5; i++)
                    Assert.AreEqual(expBits[i], BitConverter.DoubleToInt64Bits(Convert.ToDouble(rt.GetAtIndex(i), CI)), $"bits[{i}]");
            }
            finally { Directory.Delete(dir, true); }
        }

        [TestMethod]
        public void NullGuards()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.loadtxt((string)null));
            Assert.ThrowsException<ArgumentNullException>(() => np.loadtxt((Stream)null));
            Assert.ThrowsException<ArgumentNullException>(() => np.loadtxt((TextReader)null));
        }
    }
}
