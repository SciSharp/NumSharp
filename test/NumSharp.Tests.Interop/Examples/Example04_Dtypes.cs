using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/04-dtypes.cs</c>, section by section.</summary>
    [TestClass]
    public class Example04_Dtypes : ExampleTestBase
    {
        [TestMethod]
        public void TheTable_EveryDtypeBothDirections()
        {
            var table = new (NPTypeCode Tc, string DtypeStr, string Format, string NumpyName, NPTypeCode Back)[]
            {
                (NPTypeCode.Boolean, "|b1", "?", "bool", NPTypeCode.Boolean),
                (NPTypeCode.Byte, "|u1", "B", "uint8", NPTypeCode.Byte),
                (NPTypeCode.SByte, "|i1", "b", "int8", NPTypeCode.SByte),
                (NPTypeCode.Int16, "<i2", "h", "int16", NPTypeCode.Int16),
                (NPTypeCode.UInt16, "<u2", "H", "uint16", NPTypeCode.UInt16),
                (NPTypeCode.Int32, "<i4", "i", "int32", NPTypeCode.Int32),
                (NPTypeCode.UInt32, "<u4", "I", "uint32", NPTypeCode.UInt32),
                (NPTypeCode.Int64, "<i8", "q", "int64", NPTypeCode.Int64),
                (NPTypeCode.UInt64, "<u8", "Q", "uint64", NPTypeCode.UInt64),
                (NPTypeCode.Half, "<f2", "e", "float16", NPTypeCode.Half),
                (NPTypeCode.Single, "<f4", "f", "float32", NPTypeCode.Single),
                (NPTypeCode.Double, "<f8", "d", "float64", NPTypeCode.Double),
                (NPTypeCode.Complex, "<c16", "Zd", "complex128", NPTypeCode.Complex),
                (NPTypeCode.Char, "<u2", "H", "uint16", NPTypeCode.UInt16),
            };
            foreach (var row in table)
            using (NDScope.Open())
            using (Gil())
            {
                NDArray src = row.Tc == NPTypeCode.Char ? np.array(new[] { 'a', 'b', 'c' }) : np.arange(3).astype(row.Tc);
                NDArrayPythonInterop.ToNumpyDtypeStr(row.Tc).Should().Be(row.DtypeStr, row.Tc.ToString());
                NDArrayPythonInterop.ToBufferFormat(row.Tc).Should().Be(row.Format, row.Tc.ToString());
                py.x = src;
                string numpyName = py.x.dtype.name;
                NDArray back = py.x;
                bool bytesAreTheValues = py.np.array_equal(py.x, py.np.frombuffer(py.x.tobytes(), py.x.dtype));
                numpyName.Should().Be(row.NumpyName, row.Tc.ToString());
                back.typecode.Should().Be(row.Back, row.Tc.ToString());
                bytesAreTheValues.Should().BeTrue(row.Tc.ToString());
                py.Exec("del x");
            }
        }

        [TestMethod]
        public void TheReverseMaps()
        {
            NDArrayPythonInterop.FromNumpyDtypeStr("<f8").Should().Be(NPTypeCode.Double);
            NDArrayPythonInterop.FromNumpyDtypeStr("=i4").Should().Be(NPTypeCode.Int32);
            NDArrayPythonInterop.FromNumpyDtypeStr("|b1").Should().Be(NPTypeCode.Boolean);
            NDArrayPythonInterop.FromNumpyDtypeStr("u2").Should().Be(NPTypeCode.UInt16);
            NDArrayPythonInterop.FromBufferFormat("i", 4).Should().Be(NPTypeCode.Int32);
            NDArrayPythonInterop.FromBufferFormat("l", 8).Should().Be(NPTypeCode.Int64);
            NDArrayPythonInterop.FromBufferFormat("l", 4).Should().Be(NPTypeCode.Int32);
            NDArrayPythonInterop.FromBufferFormat("", 1).Should().Be(NPTypeCode.Byte, "an empty format is raw bytes");
            NDArrayPythonInterop.FromBufferFormat("u", 2).Should().Be(NPTypeCode.Char, "a 2-byte wchar_t IS a UTF-16 code unit");
            NDArrayPythonInterop.FromBufferFormat("g", 8).Should().Be(NPTypeCode.Double, "MSVC's 8-byte long double is IEEE double");
            Throws(() => NDArrayPythonInterop.FromBufferFormat("g", 16)).Should().BeOfType<NotSupportedException>()
                .Which.Message.Should().Contain("astype(np.float64)");
        }

        [TestMethod]
        public void SpecialValues_CrossBitForBit()
        {
            using var scope = NDScope.Open();
            NDArray specials = np.array(new[]
            {
                -0.0, double.NegativeInfinity, double.NaN,
                BitConverter.Int64BitsToDouble(unchecked((long)0xfff8123456789abc)),
                double.Epsilon, double.MaxValue,
            });
            using (Gil())
            {
                py.sp = specials;
                for (int i = 0; i < specials.size; i++)
                {
                    long numpyBits = py.Eval($"int(sp.view('i8')[{i}])");
                    numpyBits.Should().Be(BitConverter.DoubleToInt64Bits(specials.GetDouble(i)), $"sp[{i}]");
                }

                py.bg = np.array(new[] { ulong.MaxValue, 0UL, 1UL << 63 });
                string maxima = py.bg.tolist().ToString();
                maxima.Should().Be("[18446744073709551615, 0, 9223372036854775808]");

                py.hf = np.array(new[] { Half.NaN, Half.PositiveInfinity, (Half)(-0.0), Half.Epsilon });
                string halfName = py.hf.dtype.name;
                halfName.Should().Be("float16");
                PyBool("np.isnan(hf[0]) and np.isposinf(hf[1]) and np.signbit(hf[2])").Should().BeTrue();

                py.cx = np.array(new[] { new Complex(1, -2), new Complex(double.NaN, double.PositiveInfinity) });
                string complexName = py.cx.dtype.name;
                complexName.Should().Be("complex128");
                PyStr("cx[0]").Should().Be("(1-2j)");
                PyBool("np.isnan(cx[1].real) and np.isposinf(cx[1].imag)").Should().BeTrue();
                py.Exec("del sp, bg, hf, cx");
            }
        }

        [TestMethod]
        public void Decimal_ConvertsToFloat64OnExport_TheLowLevelMapsRefuse()
        {
            using var scope = NDScope.Open();
            NDArray dec = np.array(new[] { 1.5m, -2.25m, 1234567890.123456789m });
            Throws(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal)).Should().BeOfType<NotSupportedException>()
                .Which.Message.Should().Contain("no numpy dtype");
            Throws(() => NDArrayPythonInterop.ToBufferFormat(NPTypeCode.Decimal)).Should().BeOfType<NotSupportedException>()
                .Which.Message.Should().Contain("no PEP 3118 format");
            using (Gil())
            {
                Throws(() => dec.ToMemoryView()).Should().BeOfType<NotSupportedException>();
                py.dc = dec;
                string decName = py.dc.dtype.name;
                decName.Should().Be("float64");
                PyStr("dc.tolist()").Should().Be("[1.5, -2.25, 1234567890.1234567]", "lossy past ~16 significant digits, by design");
                py.Exec("dc[0] = 99.0");
                ((decimal)dec.GetValue(0)).Should().Be(1.5m, "writes never reach the Decimal source");
                bool ownsData = py.dc.flags.owndata;
                ownsData.Should().BeFalse("the float64 temporary is kept alive by the export pin");
                py.Exec("del dc");
            }
        }

        [TestMethod]
        public void Complex64_WidenedOnCopy_NeverViewed()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject c8 = py.Eval("np.array([1+2j, 3-4j], dtype='c8')");
                Throws(() => c8.AsNDArray()).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("complex64");
                NDArray widened = c8.ToNDArray();
                widened.typecode.Should().Be(NPTypeCode.Complex);
                ((Complex)widened.GetValue(1)).Should().Be(new Complex(3, -4));
                Throws(() => NDArrayPythonInterop.FromNumpyDtypeStr("<c8")).Should().BeOfType<NotSupportedException>();
            }
        }

        [TestMethod]
        public void Ucs4Text_NarrowedToCharOnCopy_TwoByteWcharViews()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject u1 = py.Eval("np.array(['a', 'Z', '\\u00e9'], dtype='U1')");
                Throws(() => u1.AsNDArray()).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("UCS-4");
                NDArray narrowed = u1.ToNDArray();
                narrowed.typecode.Should().Be(NPTypeCode.Char);
                ((char)narrowed.GetValue(0)).Should().Be('a');
                ((char)narrowed.GetValue(1)).Should().Be('Z');
                ((char)narrowed.GetValue(2)).Should().Be('é');
                using PyObject astral = py.Eval("np.array(['\\U0001F600'], dtype='U1')");
                Throws(() => astral.ToNDArray()).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("non-BMP");

                long wcharSize = py.Eval("__import__('array').array('u').itemsize");
                if (wcharSize != 2)
                    Assert.Inconclusive("array.array('u') is 4-byte on this platform — it takes the UCS-4 copy route");
                py.Exec("import array\nwide = array.array('u', 'hi!')");
                NDArray chars = py.wide;
                chars.typecode.Should().Be(NPTypeCode.Char);
                ((char)chars.GetValue(1)).Should().Be('i');
                chars[0] = 'H';
                PyStr("wide.tounicode()").Should().Be("Hi!");
            }
        }

        [TestMethod]
        public void BigEndian_ByteReversedOnCopy_NeverViewed_SingleByteViewsRegardless()
        {
            using var scope = NDScope.Open();
            Throws(() => NDArrayPythonInterop.FromNumpyDtypeStr(">f8")).Should().BeOfType<NotSupportedException>()
                .Which.Message.Should().Contain("newbyteorder('<')");
            using (Gil())
            {
                using PyObject be = py.Eval("np.array([1, -2, 300000], dtype='>i4')");
                Throws(() => be.AsNDArray()).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("big-endian");
                NDArray swapped = be.ToNDArray();
                swapped.typecode.Should().Be(NPTypeCode.Int32);
                swapped.GetInt32(0).Should().Be(1);
                swapped.GetInt32(1).Should().Be(-2);
                swapped.GetInt32(2).Should().Be(300000);

                using PyObject bec = py.Eval("np.array([1+2j], dtype='>c16')");
                NDArray swappedComplex = bec.ToNDArray();
                ((Complex)swappedComplex.GetValue(0)).Should().Be(new Complex(1, 2), "each 8-byte half is reversed independently");

                using PyObject be1 = py.Eval("np.array([1, 2, 3], dtype='>i1')");
                NDArray oneByte = be1.AsNDArray();
                oneByte.typecode.Should().Be(NPTypeCode.SByte);
                ((sbyte)oneByte.GetValue(2)).Should().Be(3, "byte order is meaningless at one byte");
            }
        }

        [TestMethod]
        public void NoNumSharpDtype_RefusedOnBothPaths()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject dt = py.Eval("np.array(['2021-01-01'], dtype='M8[D]')");
                Throws(() => dt.AsNDArray(allowReadonly: true)).Should().BeOfType<NotSupportedException>();
                Throws(() => dt.ToNDArray()).Should().BeOfType<NotSupportedException>();
                using PyObject rec = py.Eval("np.zeros(2, dtype=[('x', 'i4'), ('y', 'f8')])");
                Throws(() => rec.ToNDArray()).Should().BeOfType<NotSupportedException>();
            }
        }
    }
}
