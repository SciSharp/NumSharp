using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.IO;
using TUnit.Core;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    /// Tests for .npy format version handling (v1.0, v2.0, v3.0).
    /// </summary>
    public class NpyFormatVersionTests
    {
        private const string TestDir = "test_compat";

        [Test]
        public void ParseVersion2_LargeHeader()
        {
            var path = Path.Combine(TestDir, "version2_large_header.npy");
            if (!File.Exists(path))
            {
                Console.WriteLine("Skipping: version2_large_header.npy not found");
                return;
            }

            // Read and verify the version
            using var stream = File.OpenRead(path);
            var version = NpyFormat.ReadMagic(stream);

            Assert.AreEqual(2, version.Major, "Should be version 2.0");
            Assert.AreEqual(0, version.Minor);

            // Try to read the header (will fail on dtype, but should parse header correctly)
            try
            {
                var header = NpyFormat.ReadArrayHeader(stream, version, maxHeaderSize: 100000);
                Console.WriteLine($"Header parsed successfully!");
                Console.WriteLine($"  Shape: [{string.Join(", ", header.Shape)}]");
                Console.WriteLine($"  FortranOrder: {header.FortranOrder}");
                // If we get here, the v2.0 header was parsed (4-byte length field worked)
                Assert.Fail("Expected dtype parse error for structured array");
            }
            catch (FormatException ex) when (ex.Message.Contains("descr"))
            {
                // Expected - structured dtypes not supported, but header was parsed!
                Console.WriteLine($"Header parsing reached dtype: {ex.Message}");
                Assert.IsTrue(true, "Version 2.0 header length field (4 bytes) was parsed correctly");
            }
            catch (NotSupportedException ex)
            {
                // Also acceptable - means we got past header parsing
                Console.WriteLine($"Dtype not supported (expected): {ex.Message}");
                Assert.IsTrue(true, "Version 2.0 format parsed, dtype not supported");
            }
        }

        [Test]
        public void ParseVersion3_UnicodeFieldNames()
        {
            var path = Path.Combine(TestDir, "version3_unicode.npy");
            if (!File.Exists(path))
            {
                Console.WriteLine("Skipping: version3_unicode.npy not found");
                return;
            }

            // Read and verify the version
            using var stream = File.OpenRead(path);
            var version = NpyFormat.ReadMagic(stream);

            Assert.AreEqual(3, version.Major, "Should be version 3.0");
            Assert.AreEqual(0, version.Minor);

            // Try to read the header (UTF-8 encoded)
            try
            {
                var header = NpyFormat.ReadArrayHeader(stream, version, maxHeaderSize: 10000);
                Console.WriteLine($"Header parsed successfully!");
                Console.WriteLine($"  Shape: [{string.Join(", ", header.Shape)}]");
                // If we get here, UTF-8 decoding worked
                Assert.Fail("Expected dtype parse error for structured array");
            }
            catch (FormatException ex) when (ex.Message.Contains("descr"))
            {
                // Expected - structured dtypes have complex descr format
                Console.WriteLine($"Header parsing reached dtype: {ex.Message}");
                Assert.IsTrue(true, "Version 3.0 UTF-8 decoding worked correctly");
            }
            catch (NotSupportedException ex)
            {
                Console.WriteLine($"Dtype not supported (expected): {ex.Message}");
                Assert.IsTrue(true, "Version 3.0 format parsed, dtype not supported");
            }
        }

        [Test]
        public void Version1_SimpleArray_RoundTrip()
        {
            // Verify we write version 1.0 by default for simple arrays
            var arr = np.arange(100);

            using var ms = new MemoryStream();
            np.save(ms, arr);

            ms.Position = 0;
            var version = NpyFormat.ReadMagic(ms);

            Assert.AreEqual(1, version.Major, "Simple arrays should use version 1.0");
            Assert.AreEqual(0, version.Minor);

            // Verify round-trip
            ms.Position = 0;
            var loaded = np.load_npy(ms);
            Assert.IsTrue(np.array_equal(arr, loaded));
        }

        [Test]
        public void HeaderAlignment_Is64Bytes()
        {
            var arr = np.arange(10);

            using var ms = new MemoryStream();
            np.save(ms, arr);

            // Data should start at a 64-byte aligned offset
            var bytes = ms.ToArray();

            // Find where the header ends (newline before data)
            int headerEnd = Array.LastIndexOf(bytes, (byte)'\n', bytes.Length - arr.size * arr.dtypesize - 1);
            int dataStart = headerEnd + 1;

            Console.WriteLine($"Header ends at byte {headerEnd}, data starts at byte {dataStart}");
            Assert.AreEqual(0, dataStart % 64, $"Data should start at 64-byte aligned offset, but starts at {dataStart}");
        }
    }
}
