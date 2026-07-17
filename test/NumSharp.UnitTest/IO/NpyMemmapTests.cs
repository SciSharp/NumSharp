using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.IO;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    ///     <c>np.load(path, mmap_mode=…)</c> — memory-mapped <c>.npy</c> loading.
    /// </summary>
    /// <remarks>
    ///     Behavior probed against NumPy 2.4.2. Through <c>np.load</c> only <c>r</c>/<c>readonly</c>,
    ///     <c>r+</c> and <c>c</c> actually work; the other four documented spellings validate but fail
    ///     downstream in NumPy too, and are reproduced verbatim. A memmap holds the file open until the
    ///     array (and every view of it) is released — the tests GC before reopening / deleting, exactly
    ///     as a NumPy user does with <c>del a; gc.collect()</c>.
    /// </remarks>
    [TestClass]
    public class NpyMemmapTests
    {
        private string _dir;

        [TestInitialize]
        public void Init()
        {
            _dir = Path.Combine(Path.GetTempPath(), "ns_mmap_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        [TestCleanup]
        public void Cleanup()
        {
            // A mapped file stays open until its array + views are collected; release before deleting.
            FullGc();
            try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
        }

        private static void FullGc()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private string Write(string name, NDArray arr)
        {
            string p = Path.Combine(_dir, name);
            np.save(p, arr);
            return p;
        }

        private static NDArray Arange(int n) => np.arange(n).astype(NPTypeCode.Int32);

        // ---- working modes ----------------------------------------------------------

        [TestMethod]
        public void Mmap_Read_IsReadOnly_AndMatchesNormalLoad()
        {
            string p = Write("a.npy", Arange(6).reshape(2, 3));
            var expected = np.load_npy(p);

            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.IsFalse(m.Shape.IsWriteable, "'r' must produce a non-writeable array");
            Assert.IsTrue(np.array_equal(m, expected));
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_Readonly_Alias_EqualsR()
        {
            string p = Write("a.npy", Arange(6));
            var m = (NDArray)np.load(p, mmap_mode: "readonly");
            Assert.IsFalse(m.Shape.IsWriteable);
            Assert.IsTrue(np.array_equal(m, np.load_npy(p)));
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_ReadWrite_Persists()
        {
            string p = Write("a.npy", Arange(6));

            { var m = (NDArray)np.load(p, mmap_mode: "r+"); Assert.IsTrue(m.Shape.IsWriteable); m[0] = 999; m.Dispose(); }
            FullGc(); // release the mapping before reopening

            var disk = np.load_npy(p);
            Assert.AreEqual(999, disk.GetInt32(0), "r+ write must flush to disk");
            Assert.AreEqual(1, disk.GetInt32(1));
        }

        [TestMethod]
        public void Mmap_CopyOnWrite_DoesNotPersist()
        {
            string p = Write("a.npy", Arange(6));

            { var m = (NDArray)np.load(p, mmap_mode: "c"); Assert.IsTrue(m.Shape.IsWriteable); m[0] = 777; Assert.AreEqual(777, m.GetInt32(0)); m.Dispose(); }
            FullGc();

            Assert.AreEqual(0, np.load_npy(p).GetInt32(0), "copy-on-write must NOT reach disk");
        }

        [TestMethod]
        public void Mmap_ReadOnly_Write_Throws()
        {
            string p = Write("a.npy", Arange(6));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.ThrowsException<NumSharpException>(() => m[0] = 5);
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_FortranOrder_MatchesNormalLoad()
        {
            var f = np.asfortranarray(Arange(6).reshape(2, 3));
            string p = Write("f.npy", f);
            var expected = np.load_npy(p);

            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.IsTrue(m.Shape.IsFContiguous, "a fortran-order file maps to an F-contiguous view");
            Assert.IsTrue(np.array_equal(m, expected), "F-order logical layout must match the normal read");
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_SliceView_OfMappedArray_Works()
        {
            string p = Write("a.npy", (np.arange(10).astype(NPTypeCode.Double)) / 2.0);
            var expected = np.load_npy(p);
            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.IsTrue(np.array_equal(m["2:5"], expected["2:5"]));
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_Lifetime_ReleasesFileHandle_OnDisposeThenGc()
        {
            string p = Write("a.npy", Arange(6));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.IsTrue(np.array_equal(m, np.load_npy(p)));
            m.Dispose();
            FullGc();

            // If the mapping were leaked this exclusive open would throw IOException.
            using var ex = new FileStream(p, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            Assert.IsTrue(ex.CanWrite);
        }

        // ---- view writeability propagation (memory safety) --------------------------

        [TestMethod]
        public void Mmap_ReadOnly_Views_StayReadOnly()
        {
            // Every VIEW of a read-only memmap is read-only in NumPy. If a view leaked WRITEABLE, a write
            // through it would hit read-only mapped pages — a hard AccessViolation, not a catchable error.
            string p = Write("a.npy", Arange(12).reshape(3, 4));
            var m = (NDArray)np.load(p, mmap_mode: "r");

            Assert.IsFalse(m["1:3"].Shape.IsWriteable, "slice");
            Assert.IsFalse(m[":, 1"].Shape.IsWriteable, "column");
            Assert.IsFalse(m["::-1"].Shape.IsWriteable, "reversed");
            Assert.IsFalse(m.reshape(4, 3).Shape.IsWriteable, "reshape");
            Assert.IsFalse(m.ravel().Shape.IsWriteable, "ravel");
            Assert.IsFalse(m.T.Shape.IsWriteable, "transpose");
            Assert.IsFalse(np.swapaxes(m, 0, 1).Shape.IsWriteable, "swapaxes");
            Assert.IsFalse(np.expand_dims(m, 0).Shape.IsWriteable, "expand_dims");
            Assert.IsFalse(np.squeeze(np.expand_dims(m, 0)).Shape.IsWriteable, "squeeze");
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_ReadOnly_WriteThroughView_ThrowsNotSegfaults()
        {
            string p = Write("a.npy", Arange(6));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            var view = m["1:4"];
            // Guard: assert read-only FIRST so the write below can't reach the read-only pages.
            Assert.IsFalse(view.Shape.IsWriteable, "a slice of an 'r' memmap must be read-only");
            Assert.ThrowsException<NumSharpException>(() => view[0] = 99);
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_ReadOnly_OperationResults_AreWriteable()
        {
            // A computed result owns fresh memory, so it is writeable even though the operand is read-only
            // (NumPy: only .copy()/views inherit; arithmetic makes a new writeable array).
            string p = Write("a.npy", Arange(6));
            var m = (NDArray)np.load(p, mmap_mode: "r");

            Assert.IsTrue((m + 1).Shape.IsWriteable, "m + 1");
            Assert.IsTrue((-m).Shape.IsWriteable, "-m");
            Assert.IsTrue(np.sqrt(m).Shape.IsWriteable, "sqrt");
            Assert.IsTrue(m.copy().Shape.IsWriteable, "copy");
            Assert.IsTrue(m.astype(NPTypeCode.Double).Shape.IsWriteable, "astype");

            var r = m + 1;
            r[0] = 123; // writing a computed result is allowed
            Assert.AreEqual(123, r.GetInt32(0));
            m.Dispose();
        }

        // ---- 0-d / empty shapes -----------------------------------------------------

        [TestMethod]
        public void Mmap_ZeroD_Scalar()
        {
            string p = Write("z.npy", NDArray.Scalar(7).astype(NPTypeCode.Int32));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.AreEqual(0, m.ndim);
            Assert.AreEqual(1, m.size);
            Assert.IsFalse(m.Shape.IsWriteable);
            Assert.AreEqual(7, m.GetInt32(0));
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_Empty_ShapesPreserved()
        {
            foreach (var (name, shape) in new[] { ("e1", new Shape(0)), ("e03", new Shape(0, 3)), ("e30", new Shape(3, 0)) })
            {
                string p = Path.Combine(_dir, name + ".npy");
                np.save(p, np.zeros(shape, NPTypeCode.Int32));
                var m = (NDArray)np.load(p, mmap_mode: "r");
                CollectionAssert.AreEqual(shape.Dimensions, m.shape, name);
                Assert.AreEqual(0, m.size, name);
                Assert.IsFalse(m.Shape.IsWriteable, name);
                m.Dispose();
                FullGc();
            }
        }

        [TestMethod]
        public void Mmap_FancyAndMask_ReturnWriteableCopies()
        {
            // Advanced indexing always copies, so the result is writeable even off a read-only memmap.
            string p = Write("a.npy", Arange(6));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.IsTrue(m[m > 2].Shape.IsWriteable, "boolean mask copies");
            Assert.IsTrue(m[np.array(new[] { 0, 2, 4 })].Shape.IsWriteable, "fancy index copies");
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_CopyToReadOnly_Throws()
        {
            string p = Write("a.npy", Arange(6));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            Assert.ThrowsException<ArgumentException>(() => np.copyto(m, np.zeros(new Shape(6), NPTypeCode.Int32)));
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_MaxHeaderSize_EnforcedOnNpyBranch()
        {
            string p = Write("a.npy", Arange(6));

            // The header guard applies to the mmap path, matching NumPy.
            Assert.ThrowsException<FormatException>(() => np.load(p, mmap_mode: "r", max_header_size: 10));
            FullGc();

            // …and allow_pickle lifts it (declares the file trusted), also matching NumPy.
            var m = (NDArray)np.load(p, mmap_mode: "r", max_header_size: 10, allow_pickle: true);
            Assert.AreEqual(0, m.GetInt32(0));
            m.Dispose();
        }

        [TestMethod]
        public void Mmap_Save_RoundTrips()
        {
            string p = Write("src.npy", Arange(6).reshape(2, 3));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            string dst = Path.Combine(_dir, "dst.npy");
            np.save(dst, m); // saving a mapped (read-only) array is a plain read
            m.Dispose();
            FullGc();
            Assert.IsTrue(np.array_equal(np.load_npy(dst), Arange(6).reshape(2, 3)));
        }

        // ---- npz ignores mmap_mode (NumPy parity) -----------------------------------

        [TestMethod]
        public void Mmap_Npz_IgnoresValidMode()
        {
            string p = Path.Combine(_dir, "a.npz");
            np.savez(p, Arange(3));
            object r = np.load(p, mmap_mode: "r");
            Assert.IsInstanceOfType(r, typeof(NpzFile));
            ((NpzFile)r).Dispose();
        }

        [TestMethod]
        public void Mmap_Npz_IgnoresInvalidMode()
        {
            string p = Path.Combine(_dir, "a.npz");
            np.savez(p, Arange(3));
            // An invalid mode is NEVER validated for an npz — NumPy ignores mmap_mode entirely there.
            object r = np.load(p, mmap_mode: "garbage");
            Assert.IsInstanceOfType(r, typeof(NpzFile));
            ((NpzFile)r).Dispose();
        }

        // ---- mmap needs a path: stream/byte[] are rejected --------------------------

        [TestMethod]
        public void Mmap_Stream_Rejected()
        {
            string p = Write("a.npy", Arange(6));
            using var fs = new FileStream(p, FileMode.Open, FileAccess.Read);
            var ex = Assert.ThrowsException<ValueError>(() => np.load(fs, mmap_mode: "r"));
            StringAssert.Contains(ex.Message, "Memmap cannot use existing file handles");
        }

        [TestMethod]
        public void Mmap_Bytes_Rejected()
        {
            string p = Write("a.npy", Arange(6));
            byte[] bytes = File.ReadAllBytes(p);
            Assert.ThrowsException<ValueError>(() => np.load(bytes, mmap_mode: "r"));
        }

        // ---- the 4 spellings that validate but fail downstream in NumPy too ---------

        [TestMethod]
        public void Mmap_BrokenModes_ReproduceNumPyErrors()
        {
            string p = Write("a.npy", Arange(6));

            var wp = Assert.ThrowsException<TypeError>(() => np.load(p, mmap_mode: "w+"));
            StringAssert.Contains(wp.Message, "object of type 'NoneType' has no len()");

            foreach (string mode in new[] { "write", "copyonwrite", "readwrite" })
            {
                var e = Assert.ThrowsException<ValueError>(() => np.load(p, mmap_mode: mode));
                StringAssert.Contains(e.Message, $"invalid mode: '{mode}b'");
            }
        }

        [TestMethod]
        public void Mmap_InvalidMode_ListsAllEight()
        {
            string p = Write("a.npy", Arange(6));
            var e = Assert.ThrowsException<ValueError>(() => np.load(p, mmap_mode: "zzz"));
            StringAssert.Contains(e.Message,
                "mode must be one of ['r', 'c', 'r+', 'w+', 'readonly', 'copyonwrite', 'readwrite', 'write'] (got 'zzz')");
        }

        // ---- dtypes that cannot be zero-copy mapped ---------------------------------

        [TestMethod]
        public void Mmap_Char_Rejected()
        {
            string p = Write("c.npy", np.array(new[] { 'A', 'B', 'C' }));
            var e = Assert.ThrowsException<NotSupportedException>(() => np.load(p, mmap_mode: "r"));
            StringAssert.Contains(e.Message, "cannot be memory-mapped");
        }

        [TestMethod]
        public void Mmap_BigEndian_Rejected_ButNormalLoadWorks()
        {
            string p = MakeBigEndianInt32("be.npy", new[] { 1, 2, 3 });

            // NumSharp byte-swaps big-endian on a normal read, so the crafting is valid…
            var normal = np.load_npy(p);
            Assert.IsTrue(np.array_equal(normal, np.array(new[] { 1, 2, 3 }).astype(NPTypeCode.Int32)));

            // …but a zero-copy view can't swap, so mmap must reject it.
            var e = Assert.ThrowsException<NotSupportedException>(() => np.load(p, mmap_mode: "r"));
            StringAssert.Contains(e.Message, "Big-endian");
        }

        // NumSharp only writes native little-endian, so craft a '>i4' file by flipping the descr's endian
        // byte and byte-swapping the data of a NumSharp-written '<i4' file (header length is unchanged, so
        // the 64-byte alignment is preserved).
        private string MakeBigEndianInt32(string name, int[] values)
        {
            string p = Write(name, np.array(values).astype(NPTypeCode.Int32));
            byte[] b = File.ReadAllBytes(p);

            int headerLen = b[8] | (b[9] << 8);           // v1.0: uint16 LE
            int dataOffset = 10 + headerLen;

            for (int i = 10; i + 2 < dataOffset; i++)     // '<i4' -> '>i4'
                if (b[i] == (byte)'<' && b[i + 1] == (byte)'i' && b[i + 2] == (byte)'4') { b[i] = (byte)'>'; break; }

            for (int i = dataOffset; i + 4 <= b.Length; i += 4)
            {
                (b[i], b[i + 3]) = (b[i + 3], b[i]);
                (b[i + 1], b[i + 2]) = (b[i + 2], b[i + 1]);
            }

            File.WriteAllBytes(p, b);
            return p;
        }
    }
}
