using System;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    ///     NumPy-parity tests for <see cref="NDArray.tofile(System.IO.Stream,string,string)"/> and the
    ///     filename overload — <c>ndarray.tofile(fid, sep='', format='%s')</c>.
    ///
    ///     Oracle: NumPy 2.4.2. Binary mode (sep=="") writes raw item bytes in C (row-major) order —
    ///     a sliced/strided/transposed/broadcast view writes its LOGICAL elements, not the raw buffer
    ///     (the bug this method previously had). Text mode (sep!="") writes <c>format % item</c> per
    ///     element in C-order, joined by <c>sep</c> with no trailing separator. Exact byte/text vectors
    ///     below were captured from actual <c>np.ndarray.tofile</c> output.
    /// </summary>
    [TestClass]
    public class TofileTests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
        };

        /// <summary>
        ///     A temp path unique to THIS test process. The suite multi-targets net8.0 and net10.0 and
        ///     <c>dotnet test</c> runs both TFMs CONCURRENTLY, so a FIXED name under %TEMP% is shared by
        ///     two live processes — whichever writes second dies with "The process cannot access the
        ///     file ... because it is being used by another process". Keying on the PID isolates them.
        ///     (Flaked roughly 1 local combined run in 3; invisible to CI, which pins one --framework.)
        /// </summary>
        private static string TempPath(string name) =>
            Path.Combine(Path.GetTempPath(), $"ns_p{Environment.ProcessId}_{name}");

        private static byte[] Bin(NDArray nd)
        {
            using var ms = new MemoryStream();
            nd.tofile(ms);
            return ms.ToArray();
        }

        private static string Text(NDArray nd, string sep, string format = "%s")
        {
            using var ms = new MemoryStream();
            nd.tofile(ms, sep, format);
            return Encoding.ASCII.GetString(ms.ToArray());
        }

        // ---- Binary mode: the C-order bug fix ---------------------------------------

        [TestMethod]
        public void Binary_NonContiguous_WritesLogicalCOrder_AllDtypes()
        {
            // Before the fix, a non-contiguous view leaked its raw parent buffer. tofile must write the
            // logical C-order bytes — identical to tobytes('C') — for every layout and dtype.
            foreach (var tc in AllDtypes)
            {
                foreach (var (name, nd) in new (string, NDArray)[]
                {
                    ("strided",   np.arange(12).astype(tc)["::2"]),
                    ("reversed",  np.arange(6).astype(tc)["::-1"]),
                    ("transpose", np.arange(6).astype(tc).reshape(3, 2).T),
                    ("broadcast", np.broadcast_to(np.arange(3).astype(tc).reshape(1, 3), new Shape(2, 3))),
                    ("offset",    np.arange(10).astype(tc)["3:8"]),
                })
                {
                    CollectionAssert.AreEqual(nd.tobytes('C'), Bin(nd), $"{tc}/{name}");
                }
            }
        }

        [TestMethod]
        public void Binary_Strided_Int32_ExactBytes()
        {
            // np.arange(12, dtype=int32)[::2].tofile() -> [0,2,4,6,8,10] in C-order, NOT the 48-byte parent.
            var v = np.arange(12).astype(NPTypeCode.Int32)["::2"];
            CollectionAssert.AreEqual(
                Convert.FromHexString("00000000020000000400000006000000080000000a000000"),
                Bin(v));
        }

        [TestMethod]
        public void Binary_RoundTrip_FromFile_AllDtypes_FilenameOverload()
        {
            // tofile(string) creates/truncates; fromfile reads it back. A strided+offset view survives.
            foreach (var tc in AllDtypes)
            {
                var a = np.arange(20).astype(tc)["2:18"]["::2"];
                string path = TempPath($"tofile_rt_{tc}.bin");
                try
                {
                    a.tofile(path);
                    var b = np.fromfile(path, tc);
                    Assert.AreEqual(a.size, b.size, $"{tc} size");
                    CollectionAssert.AreEqual(a.tobytes('C'), b.tobytes('C'), $"{tc} bytes");
                }
                finally { if (File.Exists(path)) File.Delete(path); }
            }
        }

        [TestMethod]
        public void Binary_FilenameOverload_TruncatesExistingFile()
        {
            string path = TempPath("tofile_trunc.bin");
            try
            {
                File.WriteAllBytes(path, new byte[1000]); // pre-existing longer content
                np.arange(3).astype(NPTypeCode.Int32).tofile(path);
                Assert.AreEqual(12, new FileInfo(path).Length, "file must be truncated to 3*4 bytes");
            }
            finally { if (File.Exists(path)) File.Delete(path); }
        }

        // ---- Text mode: default (%s) ------------------------------------------------

        [TestMethod]
        public void Text_Default_COrder_ExactBytes()
        {
            // Transpose flattens in C-order: [[0,2,4],[1,3,5]] -> 0,2,4,1,3,5
            Assert.AreEqual("0,2,4,1,3,5", Text(np.arange(6).astype(NPTypeCode.Int32).reshape(3, 2).T, ","));
            // reversed uint8
            Assert.AreEqual("5,4,3,2,1,0", Text(np.arange(6).astype(NPTypeCode.Byte)["::-1"], ","));
            // Single widens to a Python double (NumPy tofile semantics); integers render as "N.0"
            Assert.AreEqual("0.0,1.0,2.0,3.0,4.0,5.0", Text(np.arange(6).astype(NPTypeCode.Single), ","));
            Assert.AreEqual("0.0,1.0,2.0,3.0,4.0,5.0", Text(np.arange(6).astype(NPTypeCode.Double), ","));
        }

        [TestMethod]
        public void Text_Complex_Default_ZeroImaginary_NoLeadingPlus()
        {
            // Regression: str(0j) is '0j' NOT '+0j' (the pure-imaginary +0.0-real form drops the sign).
            Assert.AreEqual("0j,(1+0j),(2+0j),(3+0j),(4+0j),(5+0j)",
                Text(np.arange(6).astype(NPTypeCode.Complex), ","));
        }

        [TestMethod]
        public void Text_NewlineSeparator_NoTrailing()
        {
            Assert.AreEqual("0\n1\n2", Text(np.arange(3).astype(NPTypeCode.Int32), "\n"));
        }

        // ---- Text mode: format strings (Python %-formatting) ------------------------

        [TestMethod]
        public void Text_Format_Fixed_Scientific_General_Pad_ExactBytes()
        {
            var d = np.arange(6).astype(NPTypeCode.Double);
            Assert.AreEqual("0.000,1.000,2.000,3.000,4.000,5.000", Text(d, ",", "%.3f"));
            Assert.AreEqual(
                "0.000000e+00|1.000000e+00|2.000000e+00|3.000000e+00|4.000000e+00|5.000000e+00",
                Text(d, "|", "%e"));
            Assert.AreEqual("0;1;2;3;4;5", Text(d, ";", "%g"));
            Assert.AreEqual("00000.00,00001.00,00002.00,00003.00,00004.00,00005.00", Text(d, ",", "%08.2f"));
            Assert.AreEqual("+0.0 +1.0 +2.0 +3.0 +4.0 +5.0", Text(d, " ", "%+.1f"));
            Assert.AreEqual("0 1 2 3 4 5", Text(np.arange(6).astype(NPTypeCode.Int32), " ", "%d"));
        }

        [TestMethod]
        public void Text_Format_IntegerRadixAndFlags_ExactBytes()
        {
            var a = np.array(new long[] { 0, 7, -7, 255, -255 });
            Assert.AreEqual("0,7,-7,ff,-ff", Text(a, ",", "%x"));
            Assert.AreEqual("0x0,0x7,-0x7,0xff,-0xff", Text(a, ",", "%#x"));
            Assert.AreEqual("0,7,-7,377,-377", Text(a, ",", "%o"));
            Assert.AreEqual("    0,    7,   -7,  255, -255", Text(a, ",", "%5d"));
            Assert.AreEqual("00000,00007,-0007,00255,-0255", Text(a, ",", "%05d"));
        }

        [TestMethod]
        public void Text_Format_NonFinite_Floats()
        {
            var a = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            Assert.AreEqual("inf,-inf,nan", Text(a, ",", "%f"));
            Assert.AreEqual("     inf,    -inf,     nan", Text(a, ",", "%8.2f"));
            Assert.AreEqual("00000inf,-0000inf,00000nan", Text(a, ",", "%08.2f")); // Python zero-pads inf/nan
            Assert.AreEqual("+inf,-inf,+nan", Text(a, ",", "%+.1f"));
        }

        [TestMethod]
        public void Text_Format_FloatTruncatesUnderIntegerConversion()
        {
            // "%d" % 3.9 == 3 (truncate toward zero), "%d" % -3.9 == -3
            var a = np.array(new double[] { 3.9, -3.9, 0.4 });
            Assert.AreEqual("3,-3,0", Text(a, ",", "%d"));
        }

        [TestMethod]
        public void Text_Format_AltFlag_ForcesDecimalPoint()
        {
            // Fuzz regression: '#' must keep a decimal point on f/e/g even with no fractional digits.
            var d = np.array(new double[] { 3.0, 100000.0 });
            Assert.AreEqual("3.,100000.", Text(d, ",", "%#.0f"));
            Assert.AreEqual("3.e+00,1.e+05", Text(d, ",", "%#.0e"));
            Assert.AreEqual("3.00000,100000.", Text(d, ",", "%#g"));
            Assert.AreEqual("3.00,1.00e+05", Text(d, ",", "%#.3g"));
        }

        [TestMethod]
        public void Text_Float32_Float16_WidenToDouble_MatchesNumPy()
        {
            // NumPy's tofile pulls each element through getitem, which returns a Python float (double)
            // for float16/float32 — so the emitted text is the WIDENED double's shortest repr, NOT the
            // value rendered at the dtype's own (narrower) precision. Verified against NumPy 2.4.2:
            //   np.array([1e15,1e-4,1e5,1e6], np.float32).tofile(sep=',')
            //     -> '999999986991104.0,9.999999747378752e-05,100000.0,1000000.0'
            var f32 = np.array(new double[] { 1e15, 1e-4, 1e5, 1e6 }).astype(NPTypeCode.Single);
            Assert.AreEqual("999999986991104.0,9.999999747378752e-05,100000.0,1000000.0", Text(f32, ","));

            var f16 = np.array(new double[] { 65500, 1000, 100, 0.0001 }).astype(NPTypeCode.Half);
            Assert.AreEqual("65504.0,1000.0,100.0,0.00010001659393310547", Text(f16, ","));

            // Contrast: the ARRAY-PRINT surface (ToString / 0-d str) DOES use the native scalar str,
            // which switches to scientific per the dtype's max_positional (float32 -> 1e6, float16 ->
            // 1e3). tofile and str deliberately differ here, which is exactly what this test pins.
            Assert.AreEqual("1e+15", np.array(new double[] { 1e15 }).astype(NPTypeCode.Single).reshape(new int[0]).ToString());
            Assert.AreEqual("6.55e+04", np.array(new double[] { 65500 }).astype(NPTypeCode.Half).reshape(new int[0]).ToString());
        }

        // ---- API surface + edges ----------------------------------------------------

        [TestMethod]
        public void Text_AllDtypes_DoesNotThrow_AndMatchesElementCount()
        {
            // Every dtype must render text without crashing; 6 elements => 5 separators.
            foreach (var tc in AllDtypes)
            {
                string s = Text(np.arange(6).astype(tc).reshape(3, 2).T, ",");
                Assert.AreEqual(6, s.Split(',').Length, $"{tc}: {s}");
            }
        }

        [TestMethod]
        public void Empty_AllModes_WritesNothing()
        {
            foreach (var tc in AllDtypes)
            {
                var e = np.zeros(new Shape(0, 3)).astype(tc);
                Assert.AreEqual(0, Bin(e).Length, $"empty bin {tc}");
                Assert.AreEqual(0, Text(e, ",").Length, $"empty text {tc}");
            }
        }

        [TestMethod]
        public void Scalar0d_AllModes()
        {
            var z = np.arange(1).astype(NPTypeCode.Double).reshape(new int[0]);
            Assert.AreEqual(8, Bin(z).Length, "0-d binary is one item");
            Assert.AreEqual("0.0", Text(z, ","), "0-d text is the single scalar, no separator");
        }

        [TestMethod]
        public void Stream_IsLeftOpen_And_NoTrailingSeparator()
        {
            using var ms = new MemoryStream();
            np.arange(3).astype(NPTypeCode.Int32).tofile(ms, ",");
            Assert.IsTrue(ms.CanWrite, "the caller's stream must be left open");
            Assert.AreEqual("0,1,2", Encoding.ASCII.GetString(ms.ToArray()), "no trailing separator");
        }

        [TestMethod]
        public void Binary_Equals_Tobytes_ForContiguous()
        {
            // sep="" default is the binary path; equals tobytes('C').
            var a = np.arange(24).astype(NPTypeCode.Double).reshape(2, 3, 4);
            CollectionAssert.AreEqual(a.tobytes('C'), Bin(a));
        }
    }
}
