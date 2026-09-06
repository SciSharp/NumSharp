using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/07-codec.cs</c>, section by section.</summary>
    [TestClass]
    public class Example07_Codec : ExampleTestBase
    {
        [TestInitialize]
        public void ImportCtypes() => PyExec("import ctypes");

        [TestMethod]
        public void TheOrderingTrap_RegisterAtStartup_ThenEveryFreshPairDecodes()
        {
            // The script registers the codec itself to show the trap: a decode attempted BEFORE registration poisons its
            // (Python type, NDArray) pair for the session. This suite registers before any test, so the trap is pinned by
            // DocExamples_PythonnetNumpyPage.Codec_RegisterBeforeFirstConversion_OrThePairIsPoisoned; what holds regardless:
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject fresh = py.Eval("(ctypes.c_short * 8)(1, 2, 3, 4, 5, 6, 7, 8)");
                NDArray ok = fresh.As<NDArray>();
                ok.typecode.Should().Be(NPTypeCode.Int16, "a pair first touched after registration decodes");
                ((short)ok.GetValue(7)).Should().Be(8);
            }

            NDArrayPythonInterop.RegisterCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy })
                .Should().BeFalse("a second registration, even with other options, is a no-op — the first policy sticks");
        }

        [TestMethod]
        public void Encode_EveryPythonnetBoundary_IncludingTuples()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            using (Gil())
            {
                py.c = nd;
                PyStr("type(c).__name__").Should().Be("ndarray");
                PyBool("c.flags['OWNDATA']").Should().BeFalse("a real ndarray borrowing NumSharp's memory");
                py.Exec("c[1] = 88.5");
                nd.GetDouble(1).Should().Be(88.5);

                py.Exec("def fsum(a):\n    return float(a.sum())\n\ndef process(a):\n    return a * 2.0 + 1.0");
                double s = py.fsum(nd);
                s.Should().Be(93.5, "a Python function receives the NDArray as numpy");
                NDArray processed = py.process(nd);
                processed.GetDouble(1).Should().Be(178.0);
                processed.GetDouble(3).Should().Be(7.0);

                NDArray a = np.arange(4).astype(np.float64).reshape(2, 2);
                NDArray b = (np.arange(4).astype(np.float64) + 1).reshape(2, 2);
                NDArray product = py.np.matmul(a, b);
                NDArray expected = np.matmul(a, b);
                product.GetDouble(1, 1).Should().Be(expected.GetDouble(1, 1)).And.Be(16.0);

                NDArray zeros = py.np.zeros((2, 3));
                zeros.shape.Should().Equal(2, 3);
                zeros.typecode.Should().Be(NPTypeCode.Double);
                (long, long) shape = py.np.zeros((2, 3)).shape;
                shape.Should().Be((2L, 3L), "a Python tuple came back as a C# one");
                py.Exec("del c");
            }
        }

        [TestMethod]
        public void Decode_AsNDArray_AsManagedObject_AndTheDynamicSpelling_AreAllViews()
        {
            int imports0 = NDArrayPythonInterop.LiveImports;
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("mm = np.arange(6, dtype='f8').reshape(2, 3)");
                using PyObject mmPy = py.mm;
                NDArray dec = mmPy.As<NDArray>();
                NDArray dec2 = (NDArray)mmPy.AsManagedObject(typeof(NDArray));
                NDArray dec3 = py.mm;
                dec[0, 0] = -1.0;
                PyFloat("mm[0, 0]").Should().Be(-1.0);
                dec2.GetDouble(0, 0).Should().Be(-1.0);
                dec3.GetDouble(0, 0).Should().Be(-1.0);
                NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 3, "each view holds its own lease");
            }
        }

        [TestMethod]
        public void Decode_WhatCanDecodeAccepts()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                NDArray matrix = py.Eval("np.matrix('1 2; 3 4').astype('i8')");
                matrix.shape.Should().Equal(2, 2);
                matrix.GetInt64(1, 1).Should().Be(4, "ndarray subclasses decode through an __mro__ walk");

                NDArray mv = py.Eval("memoryview(b'ab')");
                mv.typecode.Should().Be(NPTypeCode.Byte);
                mv.Shape.IsWriteable.Should().BeFalse("a read-only source is a non-writeable view");

                NDArray ct = py.Eval("(ctypes.c_double * 3)(0.5, 1.5, 2.5)");
                ct.typecode.Should().Be(NPTypeCode.Double, "any exporter via the PEP 688 __buffer__ capability check");
                ct.GetDouble(2).Should().Be(2.5);

                NDArray list = py.Eval("[[1, 2], [3, 4]]");
                list.typecode.Should().Be(NPTypeCode.Int64);
                list.GetInt64(1, 0).Should().Be(3);
                list.Shape.IsWriteable.Should().BeTrue("array-likes are always an owning copy");

                NDArray scalar = py.Eval("3.5");
                scalar.ndim.Should().Be(0);
                scalar.GetDouble().Should().Be(3.5);

                using PyObject dict = py.Eval("{'a': 1}");
                Throws(() => dict.As<NDArray>()).Should().BeOfType<InvalidCastException>("neither a buffer nor array-like");
            }
        }

        [TestMethod]
        public void TheModes_AutoFallsBack_ViewDeclines_CopyDetaches()
        {
            var auto = new NumpyCodec(NumpyCodecOptions.Default);
            var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View, EncodeMode = NumpyCodecMode.View });
            var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy, EncodeMode = NumpyCodecMode.Copy });
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject c8 = py.Eval("np.array([1+2j, 3+4j], dtype='c8')");
                auto.TryDecode(c8, out NDArray widened).Should().BeTrue("Auto falls back to a widening copy");
                widened.typecode.Should().Be(NPTypeCode.Complex);
                viewOnly.TryDecode(c8, out NDArray declined).Should().BeFalse("View declines rather than copy silently");
                declined.Should().BeNull();

                py.Exec("f8 = np.arange(4, dtype='f8')");
                using PyObject f8 = py.f8;
                viewOnly.TryDecode(f8, out NDArray strict).Should().BeTrue();
                strict[0] = 5.5;
                PyFloat("f8[0]").Should().Be(5.5, "a View-mode decode shares");

                py.Exec("ba = bytearray(b'abcd')");
                using PyObject ba = py.ba;
                copyOnly.TryDecode(ba, out NDArray snapshot).Should().BeTrue();
                snapshot.Shape.IsWriteable.Should().BeTrue();
                snapshot[0] = (byte)90;
                PyLong("ba[0]").Should().Be(97, "a snapshot never shares memory");
                py.Exec("ba.append(1)");
                PyLong("len(ba)").Should().Be(5, "...and never locks the source");

                NDArray src = np.arange(3).astype(np.float64);
                py.encV = auto.TryEncode(src);
                py.encC = copyOnly.TryEncode(src);
                py.Exec("encV[0] = 10.0; encC[1] = 20.0");
                src.GetDouble(0).Should().Be(10.0, "Auto/View encode is a view");
                src.GetDouble(1).Should().Be(1.0, "Copy encode is a copy");

                NDArray dec = np.arange(3).astype(NPTypeCode.Decimal);
                py.encD = viewOnly.TryEncode(dec);
                PyStr("encD.dtype").Should().Be("float64", "Decimal encodes as float64 in every mode");
                py.Exec("del f8, ba, encV, encC, encD");
            }
        }

        [TestMethod]
        public void TheOptions_WhichPythonTypesDecode()
        {
            var auto = new NumpyCodec(NumpyCodecOptions.Default);
            var noBuffers = new NumpyCodec(new NumpyCodecOptions { DecodeAnyBuffer = false });
            var noArrayLike = new NumpyCodec(new NumpyCodecOptions { DecodeArrayLike = false });
            using (Gil())
            {
                PyType TypeOf(string expr)
                {
                    using PyObject o = py.Eval(expr);
                    using PyObject t = o.GetPythonType();
                    return new PyType(t);
                }

                auto.CanDecode(TypeOf("memoryview(b'ab')"), typeof(NDArray)).Should().BeTrue();
                noBuffers.CanDecode(TypeOf("memoryview(b'ab')"), typeof(NDArray)).Should().BeFalse("DecodeAnyBuffer = false");
                auto.CanDecode(TypeOf("[1, 2]"), typeof(NDArray)).Should().BeTrue();
                noArrayLike.CanDecode(TypeOf("[1, 2]"), typeof(NDArray)).Should().BeFalse("DecodeArrayLike = false");
                auto.CanDecode(TypeOf("'text'"), typeof(NDArray)).Should().BeFalse("str is never array-like");
                auto.CanDecode(TypeOf("range(3)"), typeof(NDArray)).Should().BeFalse("range is never array-like");
                auto.CanDecode(TypeOf("np.zeros(1)"), typeof(string)).Should().BeFalse("CanDecode is keyed on the CLR target");
                auto.CanDecode(TypeOf("np.zeros(1)"), typeof(NDArray)).Should().BeTrue();
                NumpyCodecOptions.Default.ConvertTuples.Should().BeTrue();
            }
        }
    }
}
