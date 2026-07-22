using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     The complete import decision map. Every exporter scenario the bridge can meet is pinned to
    ///     its outcome — <b>view</b> (zero-copy, shared memory), <b>copy</b> (<c>ToNDArray</c>
    ///     materializes an owning array, widening complex64 / narrowing UCS-4 text where the dtype
    ///     demands it), or <b>rejected</b> (both paths refuse, with guidance).
    ///
    ///     <para>The docs census (<see cref="DocExamples_ZeroCopyModelPage"/>) pins the HEADLINE
    ///     numbers the documentation quotes; this matrix pins the OUTCOME PER SCENARIO, including the
    ///     corners the census does not name: mmap, cross-process shared memory, numpy scalars,
    ///     np.memmap, ctypes scalars / multi-dimensional arrays / wide chars, text buffers, and the
    ///     <c>__array_interface__</c> shapes third-party producers actually emit.</para>
    ///
    ///     <para>Platform-dependent scenarios compute their expectation at runtime: a 2-byte
    ///     <c>wchar_t</c> (windows) makes <c>array.array('u')</c> / <c>ctypes.c_wchar</c> a zero-copy
    ///     <see cref="NPTypeCode.Char"/> view, a 4-byte one (linux/macOS) narrows as a copy; a 16-byte
    ///     <c>np.longdouble</c> (linux) is rejected while the 8-byte alias (windows) views as Double.</para>
    /// </summary>
    [TestClass]
    public class ViewabilityMatrixTests : InteropTestBase
    {
        // =====================  the matrix  ====================================================

        /// <summary>
        ///     Runs one exporter expression through the documented decision order — view first
        ///     (<c>allowReadonly:true</c>), copy on decline, rejected when both refuse — and reports
        ///     which path took it (plus the landed dtype, for the readable report).
        /// </summary>
        private (string Outcome, string Dtype) Classify(string expr)
        {
            NDArray nd = null;
            try
            {
                try
                {
                    nd = ViewOf(expr, allowReadonly: true);
                    return ("view", nd.typecode.ToString());
                }
                catch
                {
                    try
                    {
                        nd = ImportOf(expr);
                        return ("copy", nd.typecode.ToString());
                    }
                    catch
                    {
                        return ("rejected", "-");
                    }
                }
            }
            finally
            {
                nd?.Dispose();
            }
        }

        [TestMethod]
        public void ImportMatrix_EveryScenario_LandsOnItsPinnedOutcome()
        {
            PyExec("import io, ctypes, mmap, sys");
            PyExec("ro_src = np.arange(4, dtype='f8')\nro_src.flags.writeable = False");
            PyExec("ai_src = np.arange(6, dtype='f8')[::2]\n" +                          // non-contiguous, so the interface path decides
                   "class IntFlagInterface:\n" +                                          // readonly emitted as 0/1 ints (real-world producers)
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        ai = dict(ai_src.__array_interface__); ai['data'] = (ai['data'][0], 0); return ai\n" +
                   "class ReadonlyIntFlagInterface:\n" +
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        ai = dict(ai_src.__array_interface__); ai['data'] = (ai['data'][0], 1); return ai\n" +
                   "class BufferObjectData:\n" +                                          // the PIL.Image shape: 'data' is bytes
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        return {'version': 3, 'shape': (4,), 'typestr': '|u1', 'data': b'\\x01\\x02\\x03\\x04'}\n" +
                   "class MissingData:\n" +
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        return {'version': 3, 'shape': (4,), 'typestr': '|u1'}\n" +
                   "class ShortDataTuple:\n" +
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        return {'version': 3, 'shape': (4,), 'typestr': '|u1', 'data': (0,)}\n" +
                   "if_int = IntFlagInterface(); if_ro = ReadonlyIntFlagInterface()\n" +
                   "if_bytes = BufferObjectData(); if_nodata = MissingData(); if_short = ShortDataTuple()");

            long wcharSize = PyLong("ctypes.sizeof(ctypes.c_wchar)");                     // 2 windows, 4 linux/macOS
            string wcharOutcome = wcharSize == 2 ? "view" : "copy";
            long longdoubleSize = PyLong("np.dtype(np.longdouble).itemsize");             // 8 windows (== f8), 16 linux
            string longdoubleOutcome = longdoubleSize == 8 ? "view" : "rejected";
            bool hasUcs4ArrayCode = PyBool("sys.version_info >= (3, 13)");                // array.array('w') is 3.13+

            var rows = new List<(string Category, string Name, string Expr, string Expected)>();
            void Add(string category, string name, string expr, string expected) => rows.Add((category, name, expr, expected));

            // --- builtins -----------------------------------------------------------------------
            Add("builtin", "bytes (read-only)", "b'abcd'", "view");
            Add("builtin", "bytearray", "bytearray(b'abcd')", "view");
            Add("builtin", "memoryview", "memoryview(bytearray(b'abcd'))", "view");
            Add("builtin", "BytesIO.getbuffer", "io.BytesIO(b'abcd').getbuffer()", "view");
            Add("builtin", "mmap (anonymous)", "mmap.mmap(-1, 64)", "view");

            // --- memoryview forms ---------------------------------------------------------------
            Add("memoryview form", "cast('B') of f8", "memoryview(np.arange(4, dtype='f8')).cast('B')", "view");
            Add("memoryview form", "typed cast('i')", "memoryview(bytearray(16)).cast('i')", "view");
            Add("memoryview form", "2-D cast('B',(4,4))", "memoryview(bytearray(16)).cast('B', (4,4))", "view");
            Add("memoryview form", "3-D cast('h',(2,2,2))", "memoryview(bytearray(16)).cast('h', (2,2,2))", "view");
            Add("memoryview form", "toreadonly()", "memoryview(bytearray(b'abcd')).toreadonly()", "view");
            Add("memoryview form", "strided [::2]", "memoryview(bytearray(range(16)))[::2]", "view");
            Add("memoryview form", "offset [1::2]", "memoryview(bytearray(range(16)))[1::2]", "view");
            Add("memoryview form", "reversed [::-1]", "memoryview(bytearray(range(16)))[::-1]", "view");

            // --- array.array, every typecode ------------------------------------------------------
            foreach (char tc in "bBhHiIlLqQfd")
                Add("array.array", $"array('{tc}')", $"array.array('{tc}', [1,2,3,4])", "view");
            Add("array.array", $"array('u') wchar={wcharSize}", "array.array('u', 'hi')", wcharOutcome);
            if (hasUcs4ArrayCode)
                Add("array.array", "array('w') UCS-4", "array.array('w', 'hi')", "copy");

            // --- ctypes ---------------------------------------------------------------------------
            foreach (string ct in new[] { "c_int", "c_double", "c_ubyte", "c_int16", "c_uint32", "c_int64", "c_bool" })
                Add("ctypes", ct, $"(ctypes.{ct} * 4)()", "view");
            Add("ctypes", "multi-dim (c_int*3*4)", "(ctypes.c_int * 3 * 4)()", "view");
            Add("ctypes", "scalar c_double (0-d)", "ctypes.c_double(5)", "view");
            Add("ctypes", "c_char (bytes chars)", "(ctypes.c_char * 4)()", "view");
            Add("ctypes", $"c_wchar wchar={wcharSize}", "(ctypes.c_wchar * 4)()", wcharOutcome);

            // --- numpy dtypes ---------------------------------------------------------------------
            foreach (string dt in new[] { "?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16" })
                Add("numpy dtype", $"'{dt}'", $"np.zeros(4, dtype='{dt}')", "view");
            Add("numpy dtype", "'>i1' (single byte)", "np.zeros(4, dtype='>i1')", "view");
            Add("numpy dtype", $"longdouble ({longdoubleSize}B)", "np.zeros(4, dtype=np.longdouble)", longdoubleOutcome);

            // --- numpy layouts (all zero-copy) ----------------------------------------------------
            Add("numpy layout", "contiguous", "np.arange(12, dtype='f8')", "view");
            Add("numpy layout", "contiguous offset [2:]", "np.arange(12, dtype='f8')[2:]", "view");
            Add("numpy layout", "strided [::2]", "np.arange(12, dtype='f8')[::2]", "view");
            Add("numpy layout", "reversed [::-1]", "np.arange(12, dtype='f8')[::-1]", "view");
            Add("numpy layout", "transposed", "np.arange(12, dtype='f8').reshape(3,4).T", "view");
            Add("numpy layout", "F-order", "np.asfortranarray(np.arange(12, dtype='f8').reshape(3,4))", "view");
            Add("numpy layout", "broadcast", "np.broadcast_to(np.arange(3, dtype='f8'), (2,3))", "view");
            Add("numpy layout", "row window [1:3]", "np.arange(24, dtype='f8').reshape(4,6)[1:3]", "view");
            Add("numpy layout", "column [:, 1]", "np.arange(24, dtype='f8').reshape(4,6)[:, 1]", "view");
            Add("numpy layout", "diagonal() (read-only)", "np.arange(16, dtype='f8').reshape(4,4).diagonal()", "view");
            Add("numpy layout", "read-only flags", "ro_src", "view");
            Add("numpy layout", "0-d", "np.float64(2.5).reshape(())", "view");

            // --- numpy scalars (np.generic exports a 0-d buffer) ----------------------------------
            Add("numpy scalar", "float64", "np.float64(2.5)", "view");
            Add("numpy scalar", "int32", "np.int32(7)", "view");
            Add("numpy scalar", "bool_", "np.bool_(True)", "view");
            Add("numpy scalar", "float16", "np.float16(1.5)", "view");
            Add("numpy scalar", "complex64 (widens)", "np.complex64(1+2j)", "copy");

            // --- text buffers ----------------------------------------------------------------------
            Add("text", "numpy U1 (UCS-4 narrow)", "np.array(['a','b'], dtype='U1')", "copy");
            Add("text", "numpy U1 strided [::2]", "np.array(['a','b','c','d'], dtype='U1')[::2]", "copy");
            Add("text", "numpy U3 (multi-char)", "np.array(['abc'], dtype='U3')", "rejected");
            Add("text", "numpy U1 non-BMP", "np.array(['\\U0001F600'], dtype='U1')", "rejected");

            // --- the genuinely uncopyable / unviewable ---------------------------------------------
            Add("unviewable", "complex64 array", "np.array([1+2j, 3+4j], dtype='c8')", "copy");
            Add("unviewable", "complex64 strided", "np.zeros(8, dtype='c8')[::2]", "copy");
            Add("unviewable", "sub-item stride", "np.lib.stride_tricks.as_strided(np.arange(4, dtype='i4'), shape=(2,), strides=(2,))", "copy");
            Add("rejected", "big-endian '>i4'", "np.arange(4, dtype='>i4')", "rejected");
            Add("rejected", "big-endian '>f8'", "np.arange(4, dtype='>f8')", "rejected");
            Add("rejected", "datetime64", "np.array(['2021-01-01'], dtype='M8[D]')", "rejected");
            Add("rejected", "timedelta64", "np.array([1,2], dtype='m8[s]')", "rejected");
            Add("rejected", "object dtype", "np.array([{}, {}], dtype=object)", "rejected");
            Add("rejected", "structured dtype", "np.zeros(3, dtype=[('a','i4'),('b','f8')])", "rejected");
            Add("rejected", "void 'V8'", "np.zeros(3, dtype='V8')", "rejected");

            // --- __array_interface__ shapes third parties emit -------------------------------------
            Add("array interface", "data=(ptr, 0) int flag", "if_int", "view");
            Add("array interface", "data=(ptr, 1) int flag", "if_ro", "view");
            Add("array interface", "data=bytes (PIL shape)", "if_bytes", "rejected");
            Add("array interface", "no 'data' entry", "if_nodata", "rejected");
            Add("array interface", "data 1-tuple", "if_short", "rejected");

            // ---- run + report ----------------------------------------------------------------------
            var failures = new List<string>();
            var report = new StringBuilder();
            foreach (var (category, name, expr, expected) in rows)
            {
                var (outcome, dtype) = Classify(expr);
                report.AppendLine($"  {outcome,-9} {dtype,-8} {category,-16} {name}");
                if (outcome != expected)
                    failures.Add($"{category} / {name}: expected {expected}, got {outcome}   [{expr}]");
            }

            Console.WriteLine($"import viewability matrix ({rows.Count} scenarios):\n{report}");
            failures.Should().BeEmpty("every import scenario must land on its pinned outcome");
        }

        // =====================  text units: the Char routes  ===================================

        [TestMethod]
        public void WcharArray_TwoByteUnits_ViewAsChar_FourByteUnits_CopyNarrowToChar()
        {
            PyExec("import ctypes");
            long unit = PyLong("array.array('u').itemsize");
            PyExec("ua = array.array('u', 'hi!')");

            if (unit == 2)
            {
                // windows wchar_t: the buffer IS UTF-16 code units — System.Char, bit-exact, zero-copy.
                NDArray v = ViewOf("ua");
                v.typecode.Should().Be(NPTypeCode.Char, "a 2-byte wchar buffer is exactly a Char buffer");
                ReadAt<char>(v, 0).Should().Be('h');
                ReadAt<char>(v, 2).Should().Be('!');
                WriteAt(v, 'H', 0);
                PyStr("ua[0]").Should().Be("H", "the view shares the array's memory");
                v.Dispose();
            }
            else
            {
                // linux/macOS wchar_t: UCS-4 — no zero-copy Char view exists; ToNDArray narrows a copy.
                ((Action)(() => { using var _ = ViewOf("ua"); }))
                    .Should().Throw<NotSupportedException>().WithMessage("*UCS-4*", "a 4-byte text unit cannot alias a 2-byte Char");
                NDArray c = ImportOf("ua");
                c.typecode.Should().Be(NPTypeCode.Char);
                ReadAt<char>(c, 0).Should().Be('h');
                ReadAt<char>(c, 2).Should().Be('!');
                WriteAt(c, 'H', 0);
                PyStr("ua[0]").Should().Be("h", "the narrow import is an independent copy");
                c.Dispose();
            }
        }

        [TestMethod]
        public void CtypesWchar_FollowsTheSameWidthRule()
        {
            PyExec("import ctypes");
            PyExec("cw = (ctypes.c_wchar * 3)(); cw[0] = 'a'; cw[1] = 'b'; cw[2] = 'c'");
            long unit = PyLong("ctypes.sizeof(ctypes.c_wchar)");

            NDArray nd = unit == 2 ? ViewOf("cw") : ImportOf("cw");
            nd.typecode.Should().Be(NPTypeCode.Char);
            ReadAt<char>(nd, 0).Should().Be('a');
            ReadAt<char>(nd, 2).Should().Be('c');
            nd.Dispose();
        }

        [TestMethod]
        public void NumpyU1_CopyNarrowsToChar_BmpExact()
        {
            PyExec("us = np.array(['a', 'Z', '\\u05d0'], dtype='U1')");   // hebrew aleph — BMP, non-latin
            NDArray c = ImportOf("us");
            c.typecode.Should().Be(NPTypeCode.Char, "numpy '<U1' holds one UCS-4 code point per element");
            ReadAt<char>(c, 0).Should().Be('a');
            ReadAt<char>(c, 1).Should().Be('Z');
            ReadAt<char>(c, 2).Should().Be('\u05D0');
            c.Dispose();

            ((Action)(() => { using var _ = ViewOf("us"); }))
                .Should().Throw<NotSupportedException>().WithMessage("*UCS-4*ToNDArray*",
                    "numpy U1 exports the count-prefixed '1w' — viewable never, narrowable as a copy");
        }

        [TestMethod]
        public void NumpyU1_NonBmpCodePoint_RefusesTheNarrow()
        {
            PyExec("astral = np.array(['\\U0001F600'], dtype='U1')");
            ((Action)(() => { using var _ = ImportOf("astral"); }))
                .Should().Throw<NotSupportedException>().WithMessage("*non-BMP*surrogate*",
                    "a single Char cannot hold an astral code point — silently emitting half a pair would corrupt text");
        }

        // =====================  __array_interface__ robustness  ================================

        [TestMethod]
        public void ArrayInterface_IntegerReadonlyFlags_AreReadByTruthiness()
        {
            // The spec shows data=(pointer, bool) but real-world producers emit 0/1 ints; pythonnet's
            // As<bool> rejects ints, so the flag must be read by truthiness.
            PyExec("ai_src = np.arange(6, dtype='f8')[::2]\n" +
                   "class IntFlag:\n" +
                   "    def __init__(self, flag): self.flag = flag\n" +
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        ai = dict(ai_src.__array_interface__); ai['data'] = (ai['data'][0], self.flag); return ai\n" +
                   "writable = IntFlag(0); readonly = IntFlag(1)");

            NDArray w = ViewOf("writable");
            w.Shape.IsWriteable.Should().BeTrue("0 is falsy — the source declares itself writable");
            w.typecode.Should().Be(NPTypeCode.Double);
            WriteAt(w, -9.0, 0);
            PyFloat("float(ai_src[0])").Should().BeApproximately(-9.0, 1e-12, "the interface view shares the numpy buffer");
            w.Dispose();

            NDArray r = ViewOf("readonly", allowReadonly: true);
            r.Shape.IsWriteable.Should().BeFalse("1 is truthy — the view carries the read-only mark");
            r.Dispose();

            ((Action)(() => { using var _ = ViewOf("readonly"); }))
                .Should().Throw<InvalidOperationException>().WithMessage("*read-only*",
                    "a truthy int flag refuses a writable view exactly like a bool True");
        }

        [TestMethod]
        public void ArrayInterface_BufferObjectData_IsRefusedWithGuidance_NeverDereferenced()
        {
            // PIL.Image emits 'data' as BYTES. PySequence_Tuple would happily turn that into a tuple
            // of byte VALUES — the first pixel byte must never be promoted to a pointer.
            PyExec("class PilShaped:\n" +
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        return {'version': 3, 'shape': (4,), 'typestr': '|u1', 'data': b'\\x89PNG'}\n" +
                   "pil_shaped = PilShaped()");

            ((Action)(() => { using var _ = ViewOf("pil_shaped", allowReadonly: true); }))
                .Should().Throw<NotSupportedException>().WithMessage("*(pointer, readonly)*",
                    "buffer-object data names no address a zero-copy view could share");
            ((Action)(() => { using var _ = ImportOf("pil_shaped"); }))
                .Should().Throw<NotSupportedException>("without a buffer protocol the copy path refuses too");
        }

        [TestMethod]
        public void ArrayInterface_MissingOrMalformedData_IsRefusedWithGuidance()
        {
            PyExec("class NoData:\n" +
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        return {'version': 3, 'shape': (4,), 'typestr': '|u1'}\n" +
                   "class ShortData:\n" +
                   "    @property\n" +
                   "    def __array_interface__(self):\n" +
                   "        return {'version': 3, 'shape': (4,), 'typestr': '|u1', 'data': (0,)}\n" +
                   "no_data = NoData(); short_data = ShortData()");

            ((Action)(() => { using var _ = ViewOf("no_data", allowReadonly: true); }))
                .Should().Throw<NotSupportedException>().WithMessage("*'data'*",
                    "an absent data entry defers to the buffer protocol, which this object lacks");
            ((Action)(() => { using var _ = ViewOf("short_data", allowReadonly: true); }))
                .Should().Throw<NotSupportedException>().WithMessage("*(pointer, readonly)*");
        }

        // =====================  exotic real-world exporters  ===================================

        [TestMethod]
        public void Mmap_IsViewable_SharedMutation()
        {
            PyExec("import mmap\nmm = mmap.mmap(-1, 32)\nmm[0:4] = b'\\x2a\\x00\\x00\\x00'");
            NDArray v = ViewOf("mm");
            v.typecode.Should().Be(NPTypeCode.Byte, "an mmap exports raw bytes");
            v.size.Should().Be(32);
            ReadAt<byte>(v, 0).Should().Be(42);
            WriteAt(v, (byte)7, 1);
            PyLong("mm[1]").Should().Be(7, "mmap memory is shared, not copied");
            v.Dispose();
            Pump();                     // flush the lease release so close() sees no exports
            PyExec("mm.close()");
        }

        [TestMethod]
        public void SharedMemory_Buf_IsViewable()
        {
            PyExec("from multiprocessing import shared_memory\n" +
                   "shm = shared_memory.SharedMemory(create=True, size=64)");
            try
            {
                NDArray v = ViewOf("shm.buf");
                v.size.Should().Be(64);
                WriteAt(v, (byte)5, 0);
                PyLong("shm.buf[0]").Should().Be(5, "cross-process shared memory is viewable in place");
                v.Dispose();
            }
            finally
            {
                Pump();                 // the lease must be gone before close() (BufferError otherwise)
                PyExec("shm.close()\nshm.unlink()");
            }
        }

        [TestMethod]
        public void NumpyMemmap_IsViewable_FileBacked()
        {
            string path = Path.Combine(Path.GetTempPath(), $"ns_interop_memmap_{Guid.NewGuid():N}.dat");
            PyExec($"memmap_path = r'{path}'");
            PyExec("nm = np.memmap(memmap_path, dtype='f8', mode='w+', shape=(4,))\n" +
                   "nm[:] = [1.5, 2.5, 3.5, 4.5]\nnm.flush()");
            try
            {
                NDArray v = ViewOf("nm");
                v.typecode.Should().Be(NPTypeCode.Double, "np.memmap is an ndarray subclass over file-backed memory");
                ReadAt<double>(v, 2).Should().BeApproximately(3.5, 1e-12);
                WriteAt(v, -1.0, 0);
                PyFloat("float(nm[0])").Should().BeApproximately(-1.0, 1e-12, "writes land in the mapping");
                v.Dispose();
            }
            finally
            {
                Pump();
                PyExec("del nm\nimport gc; gc.collect()");   // drop the mapping so the file unlocks
                try { File.Delete(path); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }

        [TestMethod]
        public void NumpyScalars_ViewAsZeroD()
        {
            foreach (var (expr, tc) in new[]
                     {
                         ("np.float64(2.5)", NPTypeCode.Double),
                         ("np.int32(7)", NPTypeCode.Int32),
                         ("np.bool_(True)", NPTypeCode.Boolean),
                         ("np.float16(1.5)", NPTypeCode.Half),
                     })
            {
                NDArray v = ViewOf(expr, allowReadonly: true);
                v.typecode.Should().Be(tc, expr);
                v.ndim.Should().Be(0, "numpy scalars export a 0-d buffer");
                v.size.Should().Be(1);
                v.Dispose();
            }

            // complex64 scalars follow the array rule: no view, ToNDArray widens a copy.
            NDArray c = ImportOf("np.complex64(1+2j)");
            c.typecode.Should().Be(NPTypeCode.Complex);
            c.ndim.Should().Be(0);
            var value = ReadAt<System.Numerics.Complex>(c);
            value.Real.Should().BeApproximately(1.0, 1e-6);
            value.Imaginary.Should().BeApproximately(2.0, 1e-6);
            c.Dispose();
        }

        [TestMethod]
        public void NumpyDiagonal_IsAReadonlyStridedView()
        {
            PyExec("dg = np.arange(16, dtype='f8').reshape(4,4).diagonal()");   // numpy returns a READ-ONLY strided view
            ((Action)(() => { using var _ = ViewOf("dg"); }))
                .Should().Throw<InvalidOperationException>().WithMessage("*read-only*");

            NDArray v = ViewOf("dg", allowReadonly: true);
            v.Shape.IsWriteable.Should().BeFalse("numpy marks diagonal() writeable=False and the view carries it");
            ReadAt<double>(v, 0).Should().Be(0.0);
            ReadAt<double>(v, 2).Should().Be(10.0, "stride 5 elements — the exact diagonal layout is preserved");
            v.Dispose();
        }
    }
}
