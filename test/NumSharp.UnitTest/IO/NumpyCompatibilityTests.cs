using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.IO;
using TUnit.Core;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    /// Tests loading files created by actual NumPy to verify cross-compatibility.
    /// Test files are in test_compat/ directory, created by Python/NumPy.
    /// </summary>
    public class NumpyCompatibilityTests
    {
        private const string TestDir = "../../../../test_compat";

        private static bool TestFilesExist()
        {
            return Directory.Exists(TestDir) && File.Exists(Path.Combine(TestDir, "int32_1d.npy"));
        }

        [Test]
        public void Load_NumPy_Int32_1D()
        {
            if (!TestFilesExist()) return;

            var arr = np.load_npy(Path.Combine(TestDir, "int32_1d.npy"));

            Assert.AreEqual(NPTypeCode.Int32, arr.typecode);
            Assert.AreEqual(1, arr.ndim);
            Assert.AreEqual(5, arr.size);
            Assert.IsTrue(new int[] { 1, 2, 3, 4, 5 }.SequenceEqual(arr.Data<int>()));
        }

        [Test]
        public void Load_NumPy_Float64_2D()
        {
            if (!TestFilesExist()) return;

            var arr = np.load_npy(Path.Combine(TestDir, "float64_2d.npy"));

            Assert.AreEqual(NPTypeCode.Double, arr.typecode);
            Assert.AreEqual(2, arr.ndim);
            Assert.IsTrue(new int[] { 3, 4 }.SequenceEqual(arr.shape));
            Assert.AreEqual(0.0, arr.GetDouble(0));
            Assert.AreEqual(11.0, arr.GetDouble(11));
        }

        [Test]
        public void Load_NumPy_Boolean()
        {
            if (!TestFilesExist()) return;

            var arr = np.load_npy(Path.Combine(TestDir, "bool.npy"));

            Assert.AreEqual(NPTypeCode.Boolean, arr.typecode);
            Assert.AreEqual(3, arr.size);
            Assert.IsTrue(arr.GetBoolean(0));
            Assert.IsFalse(arr.GetBoolean(1));
            Assert.IsTrue(arr.GetBoolean(2));
        }

        [Test]
        public void Load_NumPy_Scalar()
        {
            if (!TestFilesExist()) return;

            var arr = np.load_npy(Path.Combine(TestDir, "scalar.npy"));

            // NumPy scalars have shape () and ndim 0
            Console.WriteLine($"Scalar loaded: ndim={arr.ndim}, shape=[{string.Join(",", arr.shape)}], size={arr.size}");

            Assert.AreEqual(1, arr.size);
            Assert.AreEqual(42, arr.GetInt64(0)); // NumPy default int is int64
        }

        [Test]
        public void Load_NumPy_Empty()
        {
            if (!TestFilesExist()) return;

            var arr = np.load_npy(Path.Combine(TestDir, "empty.npy"));

            Assert.AreEqual(NPTypeCode.Double, arr.typecode);
            Assert.AreEqual(0, arr.size);
        }

        [Test]
        public void Load_NumPy_FortranOrder()
        {
            if (!TestFilesExist()) return;

            var arr = np.load_npy(Path.Combine(TestDir, "fortran.npy"));

            Console.WriteLine($"Fortran array loaded: shape=[{string.Join(",", arr.shape)}]");
            Console.WriteLine($"Data: {arr}");

            // Should be 2x3 with values 0-5 in C-order view
            Assert.AreEqual(2, arr.ndim);
            Assert.IsTrue(new int[] { 2, 3 }.SequenceEqual(arr.shape));

            // Verify data is correct (F-order [0,3,1,4,2,5] should appear as C-order [[0,1,2],[3,4,5]])
            Assert.AreEqual(0, arr.GetInt32(0, 0));
            Assert.AreEqual(1, arr.GetInt32(0, 1));
            Assert.AreEqual(2, arr.GetInt32(0, 2));
            Assert.AreEqual(3, arr.GetInt32(1, 0));
            Assert.AreEqual(4, arr.GetInt32(1, 1));
            Assert.AreEqual(5, arr.GetInt32(1, 2));
        }

        [Test]
        public void Load_NumPy_Npz()
        {
            if (!TestFilesExist()) return;

            using var npz = np.load_npz(Path.Combine(TestDir, "multi.npz"));

            Assert.AreEqual(2, npz.Count);
            Assert.IsTrue(npz.ContainsKey("a"));
            Assert.IsTrue(npz.ContainsKey("b"));

            var a = npz["a"];
            var b = npz["b"];

            Assert.IsTrue(new long[] { 1, 2, 3 }.SequenceEqual(a.Data<long>()));
            Assert.IsTrue(np.allclose(np.array(new double[] { 4.0, 5.0 }), b));
        }

        [Test]
        public void Load_NumPy_NpzCompressed()
        {
            if (!TestFilesExist()) return;

            using var npz = np.load_npz(Path.Combine(TestDir, "compressed.npz"));

            Assert.IsTrue(npz.ContainsKey("data"));
            var data = npz["data"];

            Assert.AreEqual(1000, data.size);
            Assert.AreEqual(0, data.GetInt64(0));
            Assert.AreEqual(999, data.GetInt64(999));
        }

        [Test]
        public void RoundTrip_NumSharpToNumPy()
        {
            if (!TestFilesExist()) return;

            // Save from NumSharp
            var original = np.arange(12).reshape(3, 4);
            var path = Path.Combine(TestDir, "numsharp_created.npy");
            np.save(path, original);

            // Verify file exists and has correct header
            var bytes = File.ReadAllBytes(path);
            Assert.AreEqual(0x93, bytes[0], "Magic byte 0");
            Assert.AreEqual((byte)'N', bytes[1], "Magic byte 1");
            Assert.AreEqual((byte)'U', bytes[2], "Magic byte 2");
            Assert.AreEqual((byte)'M', bytes[3], "Magic byte 3");
            Assert.AreEqual((byte)'P', bytes[4], "Magic byte 4");
            Assert.AreEqual((byte)'Y', bytes[5], "Magic byte 5");
            Assert.AreEqual(1, bytes[6], "Version major");
            Assert.AreEqual(0, bytes[7], "Version minor");

            // Header should contain shape info
            var headerEnd = Array.IndexOf(bytes, (byte)'\n');
            var header = System.Text.Encoding.ASCII.GetString(bytes, 10, headerEnd - 10);
            Console.WriteLine($"Header: {header}");
            Assert.IsTrue(header.Contains("'shape': (3, 4)"), $"Header should contain shape: {header}");
            Assert.IsTrue(header.Contains("'fortran_order': False"), $"Header should contain fortran_order: {header}");
        }
    }
}
