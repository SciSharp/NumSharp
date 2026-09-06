using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/05-buffers-and-libraries.cs</c>, section by section.</summary>
    [TestClass]
    public class Example05_BuffersAndLibraries : ExampleTestBase
    {
        [TestInitialize]
        public void ImportTheStdlib() => PyExec("import array, ctypes, io, mmap, struct, hashlib, zlib, socket");

        [TestMethod]
        public void EveryBuiltinExporter_IsAWritableView()
        {
            var exporters = new (string Expr, NPTypeCode Tc)[]
            {
                ("bytearray(b'abcd')", NPTypeCode.Byte), ("memoryview(bytearray(b'abcd'))", NPTypeCode.Byte),
                ("io.BytesIO(b'abcd').getbuffer()", NPTypeCode.Byte), ("array.array('d', [1.5, 2.5, 3.5])", NPTypeCode.Double),
                ("array.array('q', [1, 2, 3])", NPTypeCode.Int64), ("(ctypes.c_int32 * 4)(1, 2, 3, 4)", NPTypeCode.Int32),
                ("(ctypes.c_double * 2)(0.5, 1.5)", NPTypeCode.Double), ("mmap.mmap(-1, 8)", NPTypeCode.Byte),
                ("np.arange(4, dtype='u2')", NPTypeCode.UInt16),
            };
            foreach (var (expr, tc) in exporters)
            using (NDScope.Open())
            using (Gil())
            {
                py.Exec($"e = {expr}");
                NDArray v = py.e;
                v.typecode.Should().Be(tc, expr);
                v.Shape.IsWriteable.Should().BeTrue(expr);
                v.SetValue(Convert.ChangeType(7, v.dtype), 0);
                long e0 = py.Eval("int(e[0])");
                e0.Should().Be(7, "the NumSharp write landed in {0}", expr);
            }

            PyExec("del e");
        }

        [TestMethod]
        public void ArrayArray_AllTwelveTypecodes_MapByCTypeAndItemsize()
        {
            foreach (char code in "bBhHiIlLqQfd")
            using (NDScope.Open())
            using (Gil())
            {
                py.Exec($"aa = array.array('{code}', [1, 2, 3])");
                NDArray v = py.aa;
                long itemsize = py.aa.itemsize;
                v.size.Should().Be(3, code.ToString());
                v.dtypesize.Should().Be((int)itemsize, code.ToString());
                Convert.ToDouble(v.GetValue(2)).Should().Be(3, code.ToString());
            }

            PyExec("del aa");
        }

        [TestMethod]
        public void ReadOnlyExporters_Bytes()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject bs = py.Eval("b'hello'");
                Throws(() => bs.AsNDArray()).Should().BeOfType<InvalidOperationException>().Which.Message.Should().Contain("read-only");
                NDArray ro = bs.AsNDArray(allowReadonly: true);
                ro.Shape.IsWriteable.Should().BeFalse();
                ro.GetByte(1).Should().Be((byte)'e');
                NDArray owned = bs.ToNDArray();
                owned.Shape.IsWriteable.Should().BeTrue();
                NDArray implicitly = py.Eval("b'hello'");
                implicitly.Shape.IsWriteable.Should().BeFalse("the codec's Auto mode takes the read-only view by itself");
            }
        }

        [TestMethod]
        public void SharedMemoryAcrossProcesses_ViewedInPlace_LeaseGoneBeforeClose()
        {
            using (Gil())
            {
                py.Exec("from multiprocessing import shared_memory\nshm = shared_memory.SharedMemory(create=True, size=32)");
                ViewTheSharedBlock();
                Drain();
                py.Exec("shm.close(); shm.unlink()");                   // succeeds only because the lease is gone
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void ViewTheSharedBlock()
        {
            using (NDScope.Open())
            {
                NDArray view = py.shm.buf;
                view[0] = (byte)42;
                PyLong("shm.buf[0]").Should().Be(42, "the write went straight into the shared block");
            }
        }

        [TestMethod]
        public void ToMemoryView_RawBytesForConsumersThatNeverTouchNumpy()
        {
            using var scope = NDScope.Open();
            NDArray pcm = np.arange(8).astype(NPTypeCode.Int16);
            using (Gil())
            {
                py.smv = pcm.ToMemoryView();
                PyStr("smv.format").Should().Be("B");
                PyLong("smv.nbytes").Should().Be(16);
                py.Exec("first4 = struct.unpack_from('<4h', smv)");
                (long, long, long, long) first4 = py.first4;
                first4.Should().Be((0L, 1L, 2L, 3L), "struct reads the raw bytes; the tuple codec reads Python's tuple");
                py.Exec("digest = hashlib.sha256(smv).hexdigest()\npacked = zlib.compress(smv)");
                PyLong("len(digest)").Should().Be(64);
                PyLong("len(packed)").Should().BeGreaterThan(0);
                py.Exec("s1, s2 = socket.socketpair(); s1.sendall(smv); got = s2.recv(64); s1.close(); s2.close()");
                PyBool("got == bytes(smv)").Should().BeTrue("a socket sends the bytes directly");
                py.Exec("smv.cast('h')[0] = 77");
                ((short)pcm.GetValue(0)).Should().Be(77, "the memoryview writes back into NumSharp");
                py.Exec("a = np.frombuffer(smv, '<i2').reshape(2, 4)");
                PyStr("a[0].tolist()").Should().Be("[77, 1, 2, 3]");
                PyBool("a.flags['OWNDATA']").Should().BeFalse();

                NDArray matrix = np.arange(24).reshape(4, 6).astype(np.float64);
                Throws(() => matrix.T.ToMemoryView()).Should().BeOfType<InvalidOperationException>().Which.Message.Should().Contain("not C-contiguous");
                py.rmv = matrix["1:3"].ToMemoryView();
                long windowBytes = py.Eval("len(rmv)");
                double windowStart = py.Eval("np.frombuffer(rmv, '<f8')[0]");
                windowBytes.Should().Be(96, "a contiguous row window exports exactly [offset, offset + size)");
                windowStart.Should().Be(6.0);
                py.Exec("del smv, rmv, a, packed");
            }
        }

        [TestMethod]
        public void Pillow_FrombufferReadsInPlace_AnImageCannotBeViewed()
        {
            SkipUnless("PIL");
            using var scope = NDScope.Open();
            using (Gil())
            {
                NDArray img = np.zeros(new Shape(4, 6, 3), NPTypeCode.Byte);
                img[2, 3] = (NDArray)new byte[] { 10, 20, 30 };
                py.pmv = img.ToMemoryView();
                py.Exec("from PIL import Image\nim = Image.frombuffer('RGB', (6, 4), pmv, 'raw', 'RGB', 0, 1)");
                (long, long, long) pixel = py.im.getpixel((3, 2));
                pixel.Should().Be((10L, 20L, 30L), "Image.frombuffer reads NumSharp's bytes in place; the (x, y) tuple crossed both ways");
                using PyObject im = py.im;
                Throws(() => im.AsNDArray(allowReadonly: true)).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("(pointer, readonly) tuple");
                NDArray pixels = py.np.asarray(im);
                pixels.shape.Should().Equal(4, 6, 3);
                pixels.GetByte(2, 3, 2).Should().Be(30);
                pixels.Shape.IsWriteable.Should().BeFalse();
                py.Exec("del im, pmv");
            }
        }

        [TestMethod]
        public void PyArrow_PyBufferWrapsZeroCopy_AnArrowBufferImportsAsRawBytes()
        {
            SkipUnless("pyarrow");
            using var scope = NDScope.Open();
            using (Gil())
            {
                NDArray col = np.arange(5).astype(np.float64);
                py.amv = col.ToMemoryView();
                py.Exec("import pyarrow as pa\nabuf = pa.py_buffer(amv)\naarr = pa.Array.from_buffers(pa.float64(), 5, [None, abuf])");
                bool onNumSharpBytes = py.Eval("aarr.to_numpy(zero_copy_only=True).ctypes.data == abuf.address");
                onNumSharpBytes.Should().BeTrue();
                py.Exec("arrow_native = pa.array([1.5, 2.5, 3.5], type=pa.float64())");
                NDArray raw = py.Eval("arrow_native.buffers()[1]");
                raw.dtypesize.Should().Be(1, "pyarrow exports format 'b'");
                (raw.size >= 24 && raw.size % 8 == 0).Should().BeTrue("possibly padded to a 64-byte multiple");
                NDArray asDoubles = raw.view(typeof(double));
                asDoubles.GetDouble(1).Should().Be(2.5);
                asDoubles.GetDouble(2).Should().Be(3.5);
                py.Exec("del aarr, abuf, amv, arrow_native");
            }
        }

        [TestMethod]
        public void Torch_FrombufferShares_TheSameAddressAsNumpy()
        {
            SkipUnless("torch");
            using var scope = NDScope.Open();
            using (Gil())
            {
                NDArray f32 = np.arange(6).astype(np.float32);
                py.tmv = f32.ToMemoryView();
                py.Exec("import torch\nt = torch.frombuffer(tmv, dtype=torch.float32)\nt[0] = 42.0");
                f32.GetSingle(0).Should().Be(42f);
                bool sameAddress = py.Eval("t.data_ptr() == torch.from_numpy(np.frombuffer(tmv, '<f4')).data_ptr()");
                sameAddress.Should().BeTrue();
                py.Exec("del t, tmv");
            }
        }
    }
}
