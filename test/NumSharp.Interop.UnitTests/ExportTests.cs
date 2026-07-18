using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>NumSharp → numpy: values, dtypes, every memory layout, copy semantics, error paths.</summary>
    [TestClass]
    public class ExportTests : InteropTestBase
    {
        [TestMethod]
        public void Contiguous2D_SharesMemoryBothWays()
        {
            var ns = np.arange(6).reshape(2, 3);
            ExportTo("x", ns);

            PyStr("type(x).__name__").Should().Be("ndarray");
            PyStr("x.dtype.str").Should().Be(NDArrayInterop.ToNumpyDtypeStr(ns.typecode));
            PyStr("x.shape").Should().Be("(2, 3)");
            PyStr("x.tolist()").Should().Be("[[0, 1, 2], [3, 4, 5]]");

            PyExec("x[1, 2] = 99");
            ReadAt<long>(ns, 1, 2).Should().Be(99, "a python write must land in NumSharp's buffer");   // arange is int64 (NumPy 2.x parity)

            WriteAt(ns, 42L, 0, 0);
            PyLong("int(x[0, 0])").Should().Be(42, "a NumSharp write must be visible in python");

            ExportTo("y", ns);
            PyBool("np.shares_memory(x, y)").Should().BeTrue("two exports of one NDArray alias one buffer");
        }

        [TestMethod]
        public void EveryMappableDtype_ExportsBitExact()
        {
            (NPTypeCode tc, string npName, double sum)[] cases =
            {
                (NPTypeCode.Boolean, "bool", 2),          // arange(3) -> [False, True, True]
                (NPTypeCode.Byte, "uint8", 3),
                (NPTypeCode.SByte, "int8", 3),
                (NPTypeCode.Int16, "int16", 3),
                (NPTypeCode.UInt16, "uint16", 3),
                (NPTypeCode.Int32, "int32", 3),
                (NPTypeCode.UInt32, "uint32", 3),
                (NPTypeCode.Int64, "int64", 3),
                (NPTypeCode.UInt64, "uint64", 3),
                (NPTypeCode.Half, "float16", 3),
                (NPTypeCode.Single, "float32", 3),
                (NPTypeCode.Double, "float64", 3),
                (NPTypeCode.Complex, "complex128", 3),
                (NPTypeCode.Char, "uint16", 294),         // 'a'+'b'+'c' code units
            };

            foreach (var (tc, npName, sum) in cases)
            {
                var src = tc == NPTypeCode.Char ? np.arange(97, 100).astype(tc) : np.arange(3).astype(tc);
                ExportTo("d", src);
                PyStr("d.dtype").Should().Be(npName, tc.ToString());
                double actual = tc == NPTypeCode.Complex ? PyFloat("float(d.real.sum())") : PyFloat("float(d.sum())");
                actual.Should().BeApproximately(sum, 1e-9, tc.ToString());
            }

            PyStr("chr(int(d[0]))").Should().Be("a", "char exports as UTF-16 code units");
        }

        [TestMethod]
        public void Decimal_HasNoNumpyDtype_Throws()
        {
            var dec = np.arange(3).astype(NPTypeCode.Decimal);
            ((Action)(() => NDArrayInterop.ToNumpy(dec)))
                .Should().Throw<NotSupportedException>().WithMessage("*decimal*astype*");
        }

        [TestMethod]
        public void SlicedView_ExportsAsStridedAlias_NotACopy()
        {
            var b = np.arange(24).reshape(4, 6);
            ExportTo("xb", b);
            ExportTo("sv", b["1:3, ::2"]);

            PyStr("sv.tolist()").Should().Be("[[6, 8, 10], [12, 14, 16]]");
            PyBool("np.shares_memory(sv, xb)").Should().BeTrue("the slice must alias the base buffer");

            PyExec("sv[0, 0] = -1");
            ReadAt<long>(b, 1, 0).Should().Be(-1, "writing the exported slice writes the base");
        }

        [TestMethod]
        public void ReversedAndTransposedViews_KeepExactLayout()
        {
            var b = np.arange(24).reshape(4, 6);

            ExportTo("rv", b["::-1"]);
            PyStr("rv[0].tolist()").Should().Be("[18, 19, 20, 21, 22, 23]", "negative strides export as negative strides");

            ExportTo("tv", b.T);
            PyStr("tv.shape").Should().Be("(6, 4)");
            PyStr("tv[0].tolist()").Should().Be("[0, 6, 12, 18]");
            PyBool("np.shares_memory(tv, rv)").Should().BeTrue("all views of one array alias one buffer");
        }

        [TestMethod]
        public void ContiguousPrefixWindow_ExportsOnlyTheWindow()
        {
            // Regression: a contiguous offset-0 slice keeps the FULL base block as its InternalArray
            // (NumSharp only re-slices when offset > 0). The export must still be trimmed to the
            // view's extent — this once exported the whole 32-element buffer for an 8-element window.
            var full = np.arange(32).astype(NPTypeCode.Double);
            var window = full["0:8"];
            ExportTo("win", window);
            ExportTo("whole", full);

            PyLong("win.shape[0]").Should().Be(8, "the export must cover the window, not the backing buffer");
            PyStr("win.tolist()").Should().Be("[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]");
            PyBool("np.shares_memory(win, whole)").Should().BeTrue("it is still a zero-copy view of the base");

            PyExec("win[2] = -3.0");
            ReadAt<double>(full, 2).Should().BeApproximately(-3.0, 1e-12);
        }

        [TestMethod]
        public void BroadcastView_ExportsReadOnly()
        {
            var bc = np.broadcast_to(np.arange(3), new Shape(2, 3));
            ExportTo("bv", bc);

            PyStr("bv.tolist()").Should().Be("[[0, 1, 2], [0, 1, 2]]");
            PyBool("bv.flags.writeable").Should().BeFalse("stride-0 views are read-only in NumSharp and must stay so in numpy");

            PyExec("try:\n    bv[0, 0] = 5\n    wfail = False\nexcept ValueError:\n    wfail = True");
            PyBool("wfail").Should().BeTrue();
        }

        [TestMethod]
        public void ScalarAndEmpty_Export()
        {
            ExportTo("s0", np.mean(np.arange(5)));   // 0-d scalar, 2.0
            PyLong("s0.ndim").Should().Be(0);
            PyFloat("float(s0)").Should().BeApproximately(2.0, 1e-12);

            ExportTo("e0", new NDArray(NPTypeCode.Single, new Shape(0, 3)));
            PyStr("e0.shape").Should().Be("(0, 3)");
            PyStr("e0.dtype").Should().Be("float32");
        }

        [TestMethod]
        public void Copy_IsIndependentOfTheSource()
        {
            var ns = np.arange(5);
            ExportCopyTo("cp", ns);
            ExportTo("cv", ns);

            PyBool("np.shares_memory(cp, cv)").Should().BeFalse("ToNumpyCopy must not alias");
            PyExec("cp[0] = 777");
            PyLong("int(cv[0])").Should().Be(0, "mutating the copy must not touch the source");
            ReadAt<long>(ns, 0).Should().Be(0);

            ExportCopyTo("cp2", np.arange(24).reshape(4, 6)["::2, ::3"]);
            PyStr("cp2.tolist()").Should().Be("[[0, 3], [12, 15]]", "strided sources copy element-exact");
        }

        [TestMethod]
        public void MemoryView_ExposesWritableRawBytes()
        {
            var nm = np.arange(3).astype(NPTypeCode.Double);
            ExportMemoryViewTo("m", nm);

            PyBool("bytes(m) == np.arange(3.0).tobytes()").Should().BeTrue("byte-exact view of the buffer");
            PyExec("m[0:8] = np.float64(9.5).tobytes()");
            ReadAt<double>(nm, 0).Should().BeApproximately(9.5, 1e-12, "memoryview writes land in NumSharp's buffer");
        }

        [TestMethod]
        public void MemoryView_NonContiguous_ThrowsWithGuidance()
        {
            var strided = np.arange(24).reshape(4, 6)["::2"];
            ((Action)(() => NDArrayInterop.ToMemoryView(strided)))
                .Should().Throw<InvalidOperationException>().WithMessage("*contiguous*");
        }

        [TestMethod]
        public void DisposedSource_IsRefusedNotDereferenced()
        {
            var nd = np.arange(3);
            nd.Dispose();
            ((Action)(() => NDArrayInterop.ToNumpy(nd)))
                .Should().Throw<ObjectDisposedException>("exporting freed memory would be a use-after-free");
        }

        [TestMethod]
        public void ExtensionMethods_CoverTheSameSurface()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                using (var p = nd.ToNumpy()) Scope.Set("e1", p);
                using (var p = nd.ToPython()) Scope.Set("e2", p);           // NDArray-specific overload wins over pythonnet's object.ToPython
                using (var p = nd.ToNumpyCopy()) Scope.Set("e3", p);
                using (var p = nd.ToNumpy(copy: true)) Scope.Set("e4", p);
                using (var p = nd.ToMemoryView()) Scope.Set("e5", p);
            }

            PyStr("type(e1).__name__").Should().Be("ndarray");
            PyStr("type(e2).__name__").Should().Be("ndarray");
            PyBool("np.shares_memory(e1, e2)").Should().BeTrue("both are views of the same NDArray");
            PyBool("np.shares_memory(e1, e3)").Should().BeFalse("e3 is a copy");
            PyBool("np.shares_memory(e1, e4)").Should().BeFalse("copy:true is a copy");
            PyStr("type(e5).__name__").Should().Be("memoryview");
        }
    }
}
