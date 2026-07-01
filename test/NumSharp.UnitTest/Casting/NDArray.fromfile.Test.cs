using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    ///     NumPy-parity tests for <see cref="np.fromfile(string,NPTypeCode,int,string,long)"/> and its
    ///     Stream / default-dtype overloads — <c>np.fromfile(file, dtype=float, count=-1, sep='', offset=0)</c>.
    ///
    ///     Verified against NumPy 2.4.2. Binary reads reinterpret raw bytes (count/offset honored, reads what
    ///     is present past EOF); text reads parse <c>sep</c>-separated items in NumPy's whitespace-tolerant way,
    ///     wrapping integers into the dtype and rejecting malformed data. Round-trips with
    ///     <see cref="NDArray.tofile(string,string,string)"/>.
    /// </summary>
    [TestClass]
    public class FromfileTests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
        };

        private static string TempWithText(string content)
        {
            string p = Path.Combine(Path.GetTempPath(), "ns_ff_" + Guid.NewGuid().ToString("N") + ".txt");
            File.WriteAllText(p, content);
            return p;
        }

        // ---- binary: count / offset / EOF / default dtype / Stream ------------------

        [TestMethod]
        public void Binary_Count_Offset_EofClamp()
        {
            string p = Path.Combine(Path.GetTempPath(), "ns_ff_bin.bin");
            np.arange(12).astype(NPTypeCode.Int32).tofile(p);
            try
            {
                CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, np.fromfile(p, NPTypeCode.Int32, count: 4).ToArray<int>());
                CollectionAssert.AreEqual(Enumerable.Range(2, 10).ToArray(), np.fromfile(p, NPTypeCode.Int32, offset: 8).ToArray<int>());
                CollectionAssert.AreEqual(new[] { 2, 3 }, np.fromfile(p, NPTypeCode.Int32, count: 2, offset: 8).ToArray<int>());
                // count past EOF reads what is present (no error).
                Assert.AreEqual(12, np.fromfile(p, NPTypeCode.Int32, count: 100).size);
            }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Binary_DefaultDtype_IsFloat64()
        {
            string p = Path.Combine(Path.GetTempPath(), "ns_ff_def.bin");
            np.arange(4).astype(NPTypeCode.Double).tofile(p);
            try
            {
                var r = np.fromfile(p); // no dtype -> float64
                Assert.AreEqual(NPTypeCode.Double, r.typecode);
                CollectionAssert.AreEqual(new double[] { 0, 1, 2, 3 }, r.ToArray<double>());
            }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Binary_StreamOverload_ReadsFromCurrentPosition()
        {
            string p = Path.Combine(Path.GetTempPath(), "ns_ff_stream.bin");
            np.arange(6).astype(NPTypeCode.Int32).tofile(p);
            try
            {
                using var fs = File.OpenRead(p);
                fs.Seek(4, SeekOrigin.Begin);
                var r = np.fromfile(fs, NPTypeCode.Int32, count: 2);
                CollectionAssert.AreEqual(new[] { 1, 2 }, r.ToArray<int>());
                Assert.IsTrue(fs.CanRead, "stream must be left open");
            }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Binary_Empty_ReturnsEmpty()
        {
            string p = Path.Combine(Path.GetTempPath(), "ns_ff_empty.bin");
            File.WriteAllBytes(p, Array.Empty<byte>());
            try { Assert.AreEqual(0, np.fromfile(p, NPTypeCode.Int32).size); }
            finally { File.Delete(p); }
        }

        // ---- text: separators & whitespace ------------------------------------------

        [TestMethod]
        public void Text_Sep_Whitespace_TrailingSep()
        {
            string p = TempWithText("1,2,3,4,5");
            try { CollectionAssert.AreEqual(new[] { 1, 2, 3, 4, 5 }, np.fromfile(p, NPTypeCode.Int32, sep: ",").ToArray<int>()); }
            finally { File.Delete(p); }

            p = TempWithText("1, 2 , 3,4 , 5"); // whitespace around items and separators
            try { CollectionAssert.AreEqual(new[] { 1, 2, 3, 4, 5 }, np.fromfile(p, NPTypeCode.Int32, sep: ",").ToArray<int>()); }
            finally { File.Delete(p); }

            p = TempWithText("1,2,3,"); // trailing separator ignored
            try { CollectionAssert.AreEqual(new[] { 1, 2, 3 }, np.fromfile(p, NPTypeCode.Int32, sep: ",").ToArray<int>()); }
            finally { File.Delete(p); }

            p = TempWithText("1  2\t3\n4 5"); // whitespace-only separator splits on any whitespace run
            try { CollectionAssert.AreEqual(new[] { 1, 2, 3, 4, 5 }, np.fromfile(p, NPTypeCode.Int32, sep: " ").ToArray<int>()); }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_Count_LimitsItems()
        {
            string p = TempWithText("1,2,3,4,5");
            try { CollectionAssert.AreEqual(new[] { 1, 2, 3 }, np.fromfile(p, NPTypeCode.Int32, count: 3, sep: ",").ToArray<int>()); }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_Float_Scientific_NanInf()
        {
            string p = TempWithText("1.5,2.25,3.0,0.1");
            try { CollectionAssert.AreEqual(new[] { 1.5, 2.25, 3.0, 0.1 }, np.fromfile(p, NPTypeCode.Double, sep: ",").ToArray<double>()); }
            finally { File.Delete(p); }

            p = TempWithText("1.5e3,2.5e-2,1e10");
            try { CollectionAssert.AreEqual(new[] { 1500.0, 0.025, 1e10 }, np.fromfile(p, NPTypeCode.Double, sep: ",").ToArray<double>()); }
            finally { File.Delete(p); }

            p = TempWithText("nan,inf,-inf,1.5");
            try
            {
                var r = np.fromfile(p, NPTypeCode.Double, sep: ",").ToArray<double>();
                Assert.IsTrue(double.IsNaN(r[0]) && double.IsPositiveInfinity(r[1]) && double.IsNegativeInfinity(r[2]) && r[3] == 1.5);
                // NumPy's canonical text NaN is the POSITIVE quiet NaN (0x7FF8…), not .NET's default.
                Assert.AreEqual(0x7FF8000000000000L, BitConverter.DoubleToInt64Bits(r[0]));
            }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_Integer_WrapsIntoDtype()
        {
            // "%d"-style parse wraps modulo the dtype width, like NumPy (int8 300 -> 44, uint8 -1 -> 255).
            string p = TempWithText("300,44,-1");
            try { CollectionAssert.AreEqual(new sbyte[] { 44, 44, -1 }, np.fromfile(p, NPTypeCode.SByte, sep: ",").ToArray<sbyte>()); }
            finally { File.Delete(p); }

            p = TempWithText("-1,256,255");
            try { CollectionAssert.AreEqual(new byte[] { 255, 0, 255 }, np.fromfile(p, NPTypeCode.Byte, sep: ",").ToArray<byte>()); }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_Bool_ParsesAsInt()
        {
            // NumPy reads bool text as an int (nonzero => True); "True"/"False" would raise.
            string p = TempWithText("1,0,2,0");
            try { CollectionAssert.AreEqual(new[] { true, false, true, false }, np.fromfile(p, NPTypeCode.Boolean, sep: ",").ToArray<bool>()); }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_Complex_BareForm_And_ParenthesizedSuperset()
        {
            // NumPy reads the bare "a+bj" form (whitespace-separated).
            string p = TempWithText("1+2j 3-4j 5j");
            try
            {
                CollectionAssert.AreEqual(
                    new[] { new Complex(1, 2), new Complex(3, -4), new Complex(0, 5) },
                    np.fromfile(p, NPTypeCode.Complex, sep: " ").ToArray<Complex>());
            }
            finally { File.Delete(p); }

            // Superset: NumSharp also reads the parenthesized "(1+2j)" form its own tofile writes
            // (NumPy's text reader errors on this, so tofile/fromfile round-trips complex here, unlike NumPy).
            p = TempWithText("(1+2j),(3+4j)");
            try
            {
                CollectionAssert.AreEqual(
                    new[] { new Complex(1, 2), new Complex(3, 4) },
                    np.fromfile(p, NPTypeCode.Complex, sep: ",").ToArray<Complex>());
            }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_Empty_ReturnsEmpty()
        {
            string p = TempWithText("");
            try { Assert.AreEqual(0, np.fromfile(p, NPTypeCode.Int32, sep: ",").size); }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_Malformed_Throws()
        {
            string p = TempWithText("1,2,x,4");
            try { Assert.ThrowsException<ValueError>(() => np.fromfile(p, NPTypeCode.Int32, sep: ",")); }
            finally { File.Delete(p); }
        }

        [TestMethod]
        public void Text_OffsetNotPermitted_Throws()
        {
            string p = TempWithText("1,2,3");
            try { Assert.ThrowsException<ArgumentException>(() => np.fromfile(p, NPTypeCode.Int32, sep: ",", offset: 4)); }
            finally { File.Delete(p); }
        }

        // ---- round-trips ------------------------------------------------------------

        [TestMethod]
        public void RoundTrip_Binary_AllDtypes_IncludingViews()
        {
            foreach (var tc in AllDtypes)
            {
                var a = np.arange(10).astype(tc)["1:9"]; // offset view -> tofile writes logical C-order
                string p = Path.Combine(Path.GetTempPath(), $"ns_ffrt_{tc}.bin");
                try
                {
                    a.tofile(p);
                    var b = np.fromfile(p, tc);
                    Assert.AreEqual(a.size, b.size, $"{tc} size");
                    CollectionAssert.AreEqual(a.tobytes('C'), b.tobytes('C'), $"{tc} bytes");
                }
                finally { File.Delete(p); }
            }
        }

        [TestMethod]
        public void RoundTrip_Text_IntAndFloatDtypes()
        {
            // int + float dtypes round-trip through text; bool ("True"/"False") and complex parens are the
            // documented asymmetries covered elsewhere.
            foreach (var tc in new[] { NPTypeCode.Int16, NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.UInt32,
                                       NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double })
            {
                var a = np.arange(6).astype(tc);
                string p = Path.Combine(Path.GetTempPath(), $"ns_ffrtt_{tc}.txt");
                try
                {
                    a.tofile(p, ",");
                    var b = np.fromfile(p, tc, sep: ",");
                    CollectionAssert.AreEqual(a.tobytes('C'), b.tobytes('C'), $"{tc} text round-trip");
                }
                finally { File.Delete(p); }
            }
        }
    }
}
