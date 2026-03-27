using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.IO;

namespace NumSharp.UnitTest
{
    /// <summary>
    /// Tests for np.save, np.load, np.savez, np.savez_compressed
    /// </summary>
    [TestClass]
    public class NpySaveLoadTests
    {
        private string _tempDir = null!;

        [TestInitialize]
        public void Setup()
        {
            _tempDir = Path.Combine(Path.GetTempPath(), "numsharp_tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempDir);
        }

        [TestCleanup]
        public void Cleanup()
        {
            if (Directory.Exists(_tempDir))
            {
                try { Directory.Delete(_tempDir, recursive: true); }
                catch { /* ignore cleanup errors */ }
            }
        }

        private string TempFile(string name) => Path.Combine(_tempDir, name);

        #region np.save / np.load - Basic Types

        [TestMethod]
        public void Save_Load_Int32_1D()
        {
            var original = np.array(new int[] { 1, 2, 3, 4, 5 });
            var path = TempFile("int32_1d.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(original.shape.SequenceEqual(loaded!.shape));
            Assert.IsTrue(np.array_equal(original, loaded));
        }

        [TestMethod]
        public void Save_Load_Float32_1D()
        {
            var original = np.array(new float[] { 1.0f, 1.5f, 2.0f, 2.5f, 3.0f });
            var path = TempFile("float32_1d.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.allclose(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_Float64_1D()
        {
            var original = np.array(new double[] { 1.0, 1.5, 2.0, 2.5, 3.0 });
            var path = TempFile("float64_1d.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.allclose(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_Int32_2D()
        {
            var original = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            var path = TempFile("int32_2d.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(new int[] { 3, 2 }.SequenceEqual(loaded!.shape));
            Assert.IsTrue(np.array_equal(original, loaded));
        }

        [TestMethod]
        public void Save_Load_Float64_3D()
        {
            var original = np.arange(24.0).reshape(2, 3, 4);
            var path = TempFile("float64_3d.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(new int[] { 2, 3, 4 }.SequenceEqual(loaded!.shape));
            Assert.IsTrue(np.allclose(original, loaded));
        }

        #endregion

        #region np.save / np.load - All Supported Types

        [TestMethod]
        public void Save_Load_Boolean()
        {
            var original = np.array(new bool[] { true, false, true, false });
            var path = TempFile("bool.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_Byte()
        {
            var original = np.array(new byte[] { 0, 127, 255 });
            var path = TempFile("byte.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_Int16()
        {
            var original = np.array(new short[] { -32768, 0, 32767 });
            var path = TempFile("int16.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_UInt16()
        {
            var original = np.array(new ushort[] { 0, 32768, 65535 });
            var path = TempFile("uint16.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_UInt32()
        {
            var original = np.array(new uint[] { 0, 2147483648, 4294967295 });
            var path = TempFile("uint32.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_Int64()
        {
            var original = np.array(new long[] { long.MinValue, 0, long.MaxValue });
            var path = TempFile("int64.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_UInt64()
        {
            var original = np.array(new ulong[] { 0, 9223372036854775808, ulong.MaxValue });
            var path = TempFile("uint64.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_Single_SpecialValues()
        {
            var original = np.array(new float[] { float.MinValue, 0, float.MaxValue, float.NaN, float.PositiveInfinity });
            var path = TempFile("single.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.AreEqual(float.MinValue, loaded!.GetSingle(0));
            Assert.AreEqual(0f, loaded.GetSingle(1));
            Assert.AreEqual(float.MaxValue, loaded.GetSingle(2));
            Assert.IsTrue(float.IsNaN(loaded.GetSingle(3)));
            Assert.IsTrue(float.IsPositiveInfinity(loaded.GetSingle(4)));
        }

        #endregion

        #region np.save / np.load - Edge Cases

        [TestMethod]
        public void Save_Load_SingleElement()
        {
            // np.array(42) creates a 1D array with shape [1]
            var original = np.array(42);
            var path = TempFile("single_element.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.AreEqual(1, loaded!.size);
            Assert.AreEqual(42, loaded.GetInt32(0));
        }

        [TestMethod]
        public void Save_Load_Empty()
        {
            var original = np.array(Array.Empty<double>());
            var path = TempFile("empty.npy");

            np.save(path, original);
            var loaded = np.load(path) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.AreEqual(0, loaded!.size);
        }

        [TestMethod]
        public void Save_AddsNpyExtension()
        {
            var original = np.arange(5);
            var pathWithoutExt = TempFile("no_extension");
            var pathWithExt = pathWithoutExt + ".npy";

            np.save(pathWithoutExt, original);

            Assert.IsTrue(File.Exists(pathWithExt));
            Assert.IsFalse(File.Exists(pathWithoutExt));
        }

        [TestMethod]
        public void Save_Load_ToStream()
        {
            var original = np.arange(10).reshape(2, 5);

            using var ms = new MemoryStream();
            np.save(ms, original);

            ms.Position = 0;
            var loaded = np.load(ms) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        [TestMethod]
        public void Save_Load_ToBytes()
        {
            var original = np.arange(10);

            byte[] bytes = np.save(original);
            var loaded = np.load(bytes) as NDArray;

            Assert.IsNotNull(loaded);
            Assert.IsTrue(np.array_equal(original, loaded!));
        }

        #endregion

        #region np.savez / np.load - NPZ Format

        [TestMethod]
        public void Savez_Load_SingleArray()
        {
            var arr = np.arange(10);
            var path = TempFile("single.npz");

            np.savez(path, arr);

            using var npz = np.load(path) as NpzFile;
            Assert.IsNotNull(npz);
            Assert.IsTrue(npz!.Files.Contains("arr_0"));

            var loaded = npz["arr_0"];
            Assert.IsTrue(np.array_equal(arr, loaded));
        }

        [TestMethod]
        public void Savez_Load_MultipleArrays()
        {
            var arr0 = np.arange(10);
            var arr1 = np.arange(20).reshape(4, 5);
            var arr2 = np.array(new double[] { 1.1, 2.2, 3.3 });
            var path = TempFile("multiple.npz");

            np.savez(path, arr0, arr1, arr2);

            using var npz = np.load(path) as NpzFile;
            Assert.IsNotNull(npz);
            Assert.AreEqual(3, npz!.Count);

            Assert.IsTrue(np.array_equal(arr0, npz["arr_0"]));
            Assert.IsTrue(np.array_equal(arr1, npz["arr_1"]));
            Assert.IsTrue(np.allclose(arr2, npz["arr_2"]));
        }

        [TestMethod]
        public void Savez_Load_NamedArrays()
        {
            var weights = np.random.randn(10, 5);
            var biases = np.zeros(5);
            var path = TempFile("named.npz");

            np.savez(path, new Dictionary<string, NDArray>
            {
                ["weights"] = weights,
                ["biases"] = biases
            });

            using var npz = np.load(path) as NpzFile;
            Assert.IsNotNull(npz);
            Assert.IsTrue(npz!.Files.Contains("weights"));
            Assert.IsTrue(npz.Files.Contains("biases"));

            Assert.IsTrue(np.allclose(weights, npz["weights"]));
            Assert.IsTrue(np.array_equal(biases, npz["biases"]));
        }

        [TestMethod]
        public void Savez_Compressed_IsSmallerThanUncompressed()
        {
            var largeArray = np.arange(10000).reshape(100, 100);
            var pathUncompressed = TempFile("uncompressed.npz");
            var pathCompressed = TempFile("compressed.npz");

            np.savez(pathUncompressed, largeArray);
            np.savez_compressed(pathCompressed, largeArray);

            var uncompressedSize = new FileInfo(pathUncompressed).Length;
            var compressedSize = new FileInfo(pathCompressed).Length;
            Assert.IsTrue(compressedSize < uncompressedSize,
                $"Compressed ({compressedSize}) should be smaller than uncompressed ({uncompressedSize})");

            // Data should be identical
            using var npzCompressed = np.load(pathCompressed) as NpzFile;
            Assert.IsTrue(np.array_equal(largeArray, npzCompressed!["arr_0"]));
        }

        [TestMethod]
        public void Savez_AddsNpzExtension()
        {
            var arr = np.arange(5);
            var pathWithoutExt = TempFile("no_ext");
            var pathWithExt = pathWithoutExt + ".npz";

            np.savez(pathWithoutExt, arr);

            Assert.IsTrue(File.Exists(pathWithExt));
        }

        [TestMethod]
        public void NpzFile_BothKeyFormatsWork()
        {
            var arr = np.arange(10);
            var path = TempFile("keys.npz");

            np.savez(path, arr);

            using var npz = np.load_npz(path);

            // Both "arr_0" and "arr_0.npy" should work
            var loaded1 = npz["arr_0"];
            var loaded2 = npz["arr_0.npy"];

            Assert.IsTrue(np.array_equal(loaded1, loaded2));
        }

        #endregion

        #region Typed Load Methods

        [TestMethod]
        public void Load_Npy_ReturnsNDArray()
        {
            var original = np.arange(10);
            var path = TempFile("typed.npy");
            np.save(path, original);

            NDArray loaded = np.load_npy(path);
            Assert.IsTrue(np.array_equal(original, loaded));
        }

        [TestMethod]
        public void Load_Npz_ReturnsNpzFile()
        {
            var original = np.arange(10);
            var path = TempFile("typed.npz");
            np.savez(path, original);

            using NpzFile npz = np.load_npz(path);
            Assert.AreEqual(1, npz.Count);
        }

        #endregion

        #region File Type Detection

        [TestMethod]
        public void Load_DetectsNpyFile()
        {
            var arr = np.arange(10);
            var path = TempFile("detect.npy");
            np.save(path, arr);

            var result = np.load(path);
            Assert.IsInstanceOfType(result, typeof(NDArray));
        }

        [TestMethod]
        public void Load_DetectsNpzFile()
        {
            var arr = np.arange(10);
            var path = TempFile("detect.npz");
            np.savez(path, arr);

            var result = np.load(path);
            Assert.IsInstanceOfType(result, typeof(NpzFile));

            // Clean up
            (result as NpzFile)?.Dispose();
        }

        [TestMethod]
        public void Load_ThrowsOnUnknownFormat()
        {
            var path = TempFile("unknown.dat");
            File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });

            Assert.ThrowsException<FormatException>(() => np.load(path));
        }

        #endregion
    }
}
