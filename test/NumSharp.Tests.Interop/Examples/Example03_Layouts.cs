using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/03-layouts.cs</c>, section by section.</summary>
    [TestClass]
    public class Example03_Layouts : ExampleTestBase
    {
        private NDArray b, scalar;

        /// <summary>The script's export block: nine layouts bound by name through the codec.</summary>
        private void ExportEveryLayout()
        {
            b = np.arange(24).reshape(4, 6).astype(np.float64);
            scalar = np.array(2.5);
            using (Gil())
            {
                py.l_c = b;
                py.l_rows = b["1:3"];
                py.l_cols = b["1:3, 0:6:2"];
                py.l_t = b.T;
                py.l_r = b["::-1"];
                py.l_f = np.asfortranarray(b);
                py.l_b = np.broadcast_to(np.arange(3).astype(np.float64), new Shape(2, 3));
                py.l_0 = scalar;
                py.l_e = np.zeros(new Shape(0, 3), np.float64);
            }
        }

        [TestMethod]
        public void Export_WhatNumpySeesPerLayout_TheTable()
        {
            using var scope = NDScope.Open();
            ExportEveryLayout();

            var table = new (string Name, string Shape, string Strides, bool C, bool F, bool Writeable, bool OwnData)[]
            {
                ("l_c", "(4, 6)", "(48, 8)", true, false, true, false),
                ("l_rows", "(2, 6)", "(48, 8)", true, false, true, false),
                ("l_cols", "(2, 3)", "(48, 16)", false, false, true, false),
                ("l_t", "(6, 4)", "(8, 48)", false, true, true, false),
                ("l_r", "(4, 6)", "(-48, 8)", false, false, true, false),
                ("l_f", "(4, 6)", "(8, 32)", false, true, true, false),
                ("l_b", "(2, 3)", "(0, 8)", false, false, false, false),
                ("l_0", "()", "()", true, true, true, false),
                ("l_e", "(0, 3)", "(0, 0)", true, true, true, true),
            };
            foreach (var row in table)
            {
                PyStr($"{row.Name}.shape").Should().Be(row.Shape, row.Name);
                PyStr($"{row.Name}.strides").Should().Be(row.Strides, row.Name);
                PyBool($"{row.Name}.flags.c_contiguous").Should().Be(row.C, row.Name);
                PyBool($"{row.Name}.flags.f_contiguous").Should().Be(row.F, row.Name);
                PyBool($"{row.Name}.flags.writeable").Should().Be(row.Writeable, row.Name);
                PyBool($"{row.Name}.flags.owndata").Should().Be(row.OwnData, row.Name);
            }

            using (Gil())
            {
                (long, long) cStrides = py.l_c.strides;
                (long, long) tStrides = py.l_t.strides;
                (long, long) rStrides = py.l_r.strides;
                (long, long) bStrides = py.l_b.strides;
                cStrides.Should().Be((48L, 8L), "the tuple codec reads a Python tuple straight into a C# one");
                tStrides.Should().Be((8L, 48L));
                rStrides.Should().Be((-48L, 8L), "a negative stride survives");
                bStrides.Should().Be((0L, 8L), "a stride-0 axis survives");

                bool rowsShare = py.np.shares_memory(py.l_rows, py.l_c);
                string cols = py.l_cols.tolist().ToString();
                string lastRowFirst = py.l_r[0].tolist().ToString();
                rowsShare.Should().BeTrue("a slice is a window over the same export");
                cols.Should().Be("[[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]");
                lastRowFirst.Should().Be("[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]");

                Exception readOnly = Throws(() => py.Exec("l_b[0, 0] = 1.0"));
                readOnly.Should().BeOfType<PythonException>().Which.Message.Should().Contain("read-only");
                py.Exec("l_0[()] = 7.5");
                scalar.GetDouble().Should().Be(7.5, "a 0-d view writes through");
                string emptyDtype = py.l_e.dtype.name;
                emptyDtype.Should().Be("float64");
            }
        }

        [TestMethod]
        public void Export_WhatTheNumpyArrayIsMadeOf()
        {
            using var scope = NDScope.Open();
            ExportEveryLayout();
            using (Gil())
            {
                string baseType = py.Eval("type(l_c.base).__name__");
                string windowType = py.Eval("type(l_c.base.base).__name__");
                string stridedType = py.Eval("type(l_cols.base).__name__");
                baseType.Should().Be("ndarray", "a reshape over...");
                windowType.Should().Be("c_char_Array_192", "...np.frombuffer over a ctypes window of 24 x 8 bytes");
                stridedType.Should().Be("DummyArray", "as_strided over the same window");
            }
        }

        [TestMethod]
        public void ImportRoute1_AContiguousPep3118Exporter()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("r1 = bytearray(b'abcd')");
                NDArray v1 = py.r1;
                v1[0] = (byte)90;
                PyLong("r1[0]").Should().Be(90);
                Exception locked = Throws(() => py.Exec("r1.append(1)"));
                locked.Should().BeOfType<PythonException>().Which.Message.Should().Contain("Existing exports of data");
            }
        }

        [TestMethod]
        public void ImportRoute2_NonContiguousNumpyArrays_ViaArrayInterface()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("r2 = np.arange(20, dtype='i8')");
                NDArray v2 = py.Eval("r2[::2]");
                v2.size.Should().Be(10);
                v2.Shape.Strides[0].Should().Be(2, "the layout is reproduced, not copied");
                v2[1] = -3L;
                PyLong("int(r2[2])").Should().Be(-3, "logical index 1 of the strided view (int(): a numpy scalar is not a Python int)");

                NDArray vt = py.Eval("np.arange(12, dtype='f8').reshape(3, 4).T");
                NDArray vf = py.Eval("np.asfortranarray(np.arange(12, dtype='f8').reshape(3, 4))");
                NDArray vr = py.Eval("np.arange(6, dtype='f8')[::-1]");
                NDArray vb = py.Eval("np.broadcast_to(np.arange(3, dtype='f8'), (2, 3))");
                vt.shape.Should().Equal(4, 3);
                vt.Shape.Strides.Should().Equal(1, 4);
                vf.Shape.IsFContiguous.Should().BeTrue();
                vf.Shape.IsContiguous.Should().BeFalse();
                vr.Shape.Strides[0].Should().Be(-1);
                vr.GetDouble(0).Should().Be(5.0, "the window is normalized below numpy's data pointer");
                vb.Shape.IsBroadcasted.Should().BeTrue();
                vb.Shape.IsWriteable.Should().BeFalse("Auto mode takes the read-only view");

                py.Exec("z0 = np.array(2.5)");
                NDArray v0 = py.z0;
                v0.ndim.Should().Be(0);
                v0.SetValue(9.5);
                PyFloat("float(z0)").Should().Be(9.5, "float(): a 0-d ndarray is not a Python float");
            }
        }

        [TestMethod]
        public void ImportRoute3_NonContiguousNonNumpyExporters_ViaPyBufStrided()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("r3 = bytearray(range(16))");
                NDArray v3 = py.Eval("memoryview(r3)[::2]");
                v3.size.Should().Be(8);
                v3.Shape.Strides[0].Should().Be(2);
                v3[3] = (byte)77;
                PyLong("r3[6]").Should().Be(77);
                NDArray v3r = py.Eval("memoryview(r3)[::-1]");
                v3r.Shape.Strides[0].Should().Be(-1);
                v3r.GetByte(0).Should().Be(15);
            }
        }

        [TestMethod]
        public void TheOnlyLayoutsThatCannotBeViewed_DeclineWithGuidance_AutoCopies()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject sub = py.Eval("np.lib.stride_tricks.as_strided(np.arange(4, dtype='i4'), shape=(2,), strides=(2,))");
                Exception declined = Throws(() => sub.AsNDArray());
                declined.Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("not a multiple of itemsize");
                NDArray linearized = sub.ToNDArray();
                linearized.size.Should().Be(2);
                linearized.typecode.Should().Be(NPTypeCode.Int32);
                NDArray auto = sub.As<NDArray>();
                auto.Shape.IsWriteable.Should().BeTrue("the codec's Auto mode falls back to the same owning copy");
            }
        }

        [TestMethod]
        public void WhoFreesWhat()
        {
            int exports0 = NDArrayPythonInterop.LiveExports, imports0 = NDArrayPythonInterop.LiveImports;
            using (NDScope.Open())
            {
                ExportEveryLayout();
                using (Gil())
                {
                    py.Exec("r1 = bytearray(b'abcd')");
                    NDArray v1 = py.r1;
                    NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1);
                }
                NDArrayPythonInterop.LiveExports.Should().Be(exports0 + 8, "eight non-empty layouts are pinned; the empty one shares no bytes");
            }

            PyExec("del l_c, l_rows, l_cols, l_t, l_r, l_f, l_b, l_0, l_e, r1");
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports0 && NDArrayPythonInterop.LiveImports == imports0)
                .Should().BeTrue();
        }
    }
}
