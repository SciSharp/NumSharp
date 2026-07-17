using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.IO;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Behaviour of the public save/load surface — signatures, path handling, overloads and
    ///     round-trips. Byte-level conformance to NumPy is proven separately by <c>NpyOracleTests</c>,
    ///     which replays real NumPy output.
    /// </summary>
    [TestClass]
    public class NumpySaveLoad
    {
        private string _dir;

        [TestInitialize]
        public void Setup()
        {
            _dir = Path.Combine(Path.GetTempPath(), "numsharp_npy_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        [TestCleanup]
        public void Cleanup()
        {
            try { Directory.Delete(_dir, recursive: true); }
            catch (IOException) { /* a leaked handle fails the test that leaked it, not this one */ }
        }

        private string At(string name) => Path.Combine(_dir, name);

        #region round-trips

        [TestMethod]
        public void Save_Load_Int1D()
        {
            var x = np.array(new[] { 1, 2, 3, 4, 5 });
            string file = At("ints.npy");

            np.save(file, x);
            var loaded = np.load_npy(file);

            loaded.typecode.Should().Be(NPTypeCode.Int32);
            loaded.shape.Should().Equal(new long[] { 5 });
            loaded.ToArray<int>().Should().Equal(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Save_Load_Float1D()
        {
            var x = np.array(new[] { 1.0f, 1.5f, 2.0f, 2.5f, 3.0f });
            string file = At("floats.npy");

            np.save(file, x);
            np.load_npy(file).ToArray<float>().Should().Equal(1.0f, 1.5f, 2.0f, 2.5f, 3.0f);
        }

        [TestMethod]
        public void Save_Load_Double1D()
        {
            var x = np.array(new[] { 1.0, 1.5, 2.0, 2.5, 3.0 });
            string file = At("doubles.npy");

            np.save(file, x);
            np.load_npy(file).ToArray<double>().Should().Equal(1.0, 1.5, 2.0, 2.5, 3.0);
        }

        [TestMethod]
        public void Save_Load_MultiDim()
        {
            var x = np.arange(24).reshape(2, 3, 4);
            string file = At("md.npy");

            np.save(file, x);
            var loaded = np.load_npy(file);

            loaded.shape.Should().Equal(new long[] { 2, 3, 4 });
            np.array_equal(x, loaded).Should().BeTrue();
        }

        [TestMethod]
        public void Save_Load_SystemArrayOverload()
        {
            int[,] x = { { 1, 2 }, { 3, 4 } };
            string file = At("sysarray.npy");

            np.save(file, x);
            var loaded = np.load_npy(file);

            loaded.shape.Should().Equal(new long[] { 2, 2 });
            loaded.ToArray<int>().Should().Equal(1, 2, 3, 4);
        }

        [TestMethod]
        public void Save_Load_Bytes_NoFile()
        {
            var x = np.arange(6).reshape(2, 3);

            byte[] encoded = np.save(x);

            np.array_equal(x, np.load_npy(encoded)).Should().BeTrue();
        }

        #endregion

        #region path handling

        [TestMethod]
        public void Save_AppendsNpyExtension_WhenMissing()
        {
            string stem = At("noext");
            np.save(stem, np.arange(9.0).reshape(3, 3));

            File.Exists(stem + ".npy").Should().BeTrue("np.save appends '.npy' when the path lacks it");
            File.Exists(stem).Should().BeFalse();
        }

        [TestMethod]
        public void Save_DoesNotDoubleAppendExtension()
        {
            string file = At("withext.npy");
            np.save(file, np.arange(4.0));

            File.Exists(file).Should().BeTrue();
            File.Exists(file + ".npy").Should().BeFalse("'.npy' is already present, so nothing is appended");
        }

        [TestMethod]
        public void Savez_AppendsNpzExtension_WhenMissing()
        {
            string stem = At("archive");
            np.savez(stem, np.arange(3));

            File.Exists(stem + ".npz").Should().BeTrue();
        }

        [TestMethod]
        public void SaveAndLoadWithNpyFileExt()
        {
            string stem = At("ext_roundtrip");

            var f1 = np.arange(9.0f).reshape(3, 3);
            np.save(stem, f1);
            np.all(f1 == np.load_npy(stem + ".npy")).Should().BeTrue();

            var d1 = np.arange(9.0d).reshape(3, 3);
            np.save(stem, d1);
            np.all(d1 == np.load_npy(stem + ".npy")).Should().BeTrue();
        }

        #endregion

        #region npz

        [TestMethod]
        public void Savez_PositionalArrays_GetArrNNames()
        {
            string file = At("positional.npz");
            np.savez(file, np.arange(3), np.arange(4.0));

            using var npz = np.load_npz(file);

            npz.Files.Should().Equal("arr_0", "arr_1");
            npz["arr_0"].ToArray<long>().Should().Equal(0L, 1L, 2L);
            npz["arr_1"].ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void Savez_NamedArrays_KeepTheirNames()
        {
            string file = At("named.npz");
            np.savez(file, new Dictionary<string, NDArray>
            {
                ["weights"] = np.arange(6.0).reshape(2, 3),
                ["biases"] = np.arange(3.0),
            });

            using var npz = np.load_npz(file);

            npz.Files.Should().BeEquivalentTo(new[] { "weights", "biases" });
            npz["weights"].shape.Should().Equal(new long[] { 2, 3 });
            npz["biases"].ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
        }

        [TestMethod]
        public void Savez_Compressed_IsSmallerAndReadsBack()
        {
            var big = np.zeros(new Shape(20_000), NPTypeCode.Double); // compresses well

            byte[] stored = np.savez(big);
            byte[] deflated = np.savez_compressed(big);

            deflated.Length.Should().BeLessThan(stored.Length / 4,
                "20k zeroed doubles should deflate to a small fraction of their stored size");

            using var npz = np.load_npz(deflated);
            npz["arr_0"].shape.Should().Equal(new long[] { 20_000 });
            np.array_equal(big, npz["arr_0"]).Should().BeTrue();
        }

        [TestMethod]
        public void Savez_PositionalCollidingWithKeyword_Throws()
        {
            Action act = () => np.savez(At("collide.npz"),
                new[] { np.arange(3) },
                new Dictionary<string, NDArray> { ["arr_0"] = np.arange(3) });

            act.Should().Throw<ArgumentException>()
               .WithMessage("*Cannot use un-named variables and keyword arr_0*",
                   "NumPy rejects a positional array whose generated name is already taken");
        }

        [TestMethod]
        public void Npz_KeysWorkWithAndWithoutNpySuffix()
        {
            using var npz = np.load_npz(np.savez(np.arange(3)));

            npz.ContainsKey("arr_0").Should().BeTrue();
            npz.ContainsKey("arr_0.npy").Should().BeTrue();
            npz["arr_0.npy"].Should().BeSameAs(npz["arr_0"], "both spellings name the same member");
            npz.Files.Should().Equal(new[] { "arr_0" }, "'.files' reports names with '.npy' stripped");
        }

        [TestMethod]
        public void Npz_IsLazyAndCaches()
        {
            using var npz = np.load_npz(np.savez(np.arange(3), np.arange(4)));

            var first = npz["arr_0"];

            npz["arr_0"].Should().BeSameAs(first, "a loaded member is cached, not re-read");
        }

        [TestMethod]
        public void Npz_MissingKey_Throws()
        {
            using var npz = np.load_npz(np.savez(np.arange(3)));

            Action act = () => { var _ = npz["nope"]; };
            act.Should().Throw<KeyNotFoundException>().WithMessage("*nope is not a file in the archive*");
        }

        [TestMethod]
        public void Npz_DotAccessViaF()
        {
            using var npz = np.load_npz(np.savez(new Dictionary<string, NDArray>
            {
                ["weights"] = np.arange(3.0),
            }));

            NDArray w = npz.f.weights;
            w.ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
        }

        [TestMethod]
        public void Npz_EnumerationYieldsEveryMember()
        {
            using var npz = np.load_npz(np.savez(np.arange(3), np.arange(4), np.arange(5)));

            npz.Count.Should().Be(3);
            npz.Select(kv => kv.Key).Should().Equal("arr_0", "arr_1", "arr_2");
            npz.Sum(kv => kv.Value.size).Should().Be(3 + 4 + 5);
        }

        [TestMethod]
        public void Npz_UseAfterDispose_Throws()
        {
            var npz = np.load_npz(np.savez(np.arange(3)));
            npz.Dispose();

            Action act = () => { var _ = npz["arr_0"]; };
            act.Should().Throw<ObjectDisposedException>();
        }

        [TestMethod]
        public void Npz_DisposeReleasesTheFileHandle()
        {
            string file = At("handle.npz");
            np.savez(file, np.arange(3));

            using (var npz = np.load_npz(file))
                npz["arr_0"].size.Should().Be(3);

            // A leaked handle would make this throw IOException.
            Action act = () => File.Delete(file);
            act.Should().NotThrow("np.load_npz owns the FileStream it opened and must close it on dispose");
        }

        [TestMethod]
        public void Npz_NestedNamesRoundTrip()
        {
            // Zip entry names may contain '/', which NumPy neither forbids nor treats as a directory.
            using var npz = np.load_npz(np.savez(new Dictionary<string, NDArray>
            {
                ["A/A"] = np.arange(4),
                ["B/A"] = np.arange(4),
            }));

            npz.Count.Should().Be(2);
            npz.Files.Should().BeEquivalentTo(new[] { "A/A", "B/A" });
            npz["A/A"].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L);
        }

        #endregion

        #region streams

        [TestMethod]
        public void Save_ToStream_AppendsSoOneFileHoldsManyArrays()
        {
            using var ms = new MemoryStream();
            np.save(ms, np.arange(3));
            np.save(ms, np.arange(4.0).reshape(2, 2));

            ms.Position = 0;
            np.load_npy(ms).ToArray<long>().Should().Equal(0L, 1L, 2L);
            np.load_npy(ms).shape.Should().Equal(new long[] { 2, 2 });

            Action act = () => np.load_npy(ms);
            act.Should().Throw<EndOfStreamException>("there is no third array");
        }

        [TestMethod]
        public void Load_LeavesCallerOwnedStreamOpen()
        {
            using var ms = new MemoryStream(np.save(np.arange(3)));

            np.load(ms);

            ms.CanRead.Should().BeTrue("np.load(Stream) does not own the caller's stream");
        }

        #endregion

        #region load dispatch

        [TestMethod]
        public void Load_ReturnsNDArrayForNpy_AndNpzFileForNpz()
        {
            np.load(np.save(np.arange(3))).Should().BeOfType<NDArray>();

            object archive = np.load(np.savez(np.arange(3)));
            archive.Should().BeOfType<NpzFile>();
            ((NpzFile)archive).Dispose();
        }

        [TestMethod]
        public void Load_DetectsTypeByMagic_NotByExtension()
        {
            // A .npz archive under a .npy name is still an archive: NumPy dispatches on magic bytes.
            string misnamed = At("actually_an_archive.npy");
            File.WriteAllBytes(misnamed, np.savez(np.arange(3)));

            object loaded = np.load(misnamed);
            loaded.Should().BeOfType<NpzFile>();
            ((NpzFile)loaded).Dispose();
        }

        [TestMethod]
        public void LoadNpy_OnNpzArchive_SaysUseLoadNpz()
        {
            Action act = () => np.load_npy(np.savez(np.arange(3)));

            act.Should().Throw<FormatException>().WithMessage("*is a .npz archive*np.load_npz*",
                "the error should name the function that would work");
        }

        [TestMethod]
        public void LoadNpz_OnNpyFile_SaysUseLoadNpy()
        {
            Action act = () => np.load_npz(np.save(np.arange(3)));

            act.Should().Throw<FormatException>().WithMessage("*is a .npy file*np.load_npy*");
        }

        [TestMethod]
        public void Load_EmptyFile_ThrowsEof()
        {
            Action act = () => np.load(Array.Empty<byte>());
            act.Should().Throw<EndOfStreamException>().WithMessage("*No data left in file*");
        }

        #endregion

        #region parameter validation

        [TestMethod]
        public void Load_RejectsEncodingsThatCorruptData()
        {
            // NumPy allows only these three; anything else can silently corrupt numeric data.
            foreach (string ok in new[] { "ASCII", "latin1", "bytes" })
            {
                Action valid = () => np.load(np.save(np.arange(3)), encoding: ok);
                valid.Should().NotThrow($"'{ok}' is one of NumPy's three allowed encodings");
            }

            Action act = () => np.load(np.save(np.arange(3)), encoding: "utf-8");
            act.Should().Throw<ArgumentException>().WithMessage("*encoding must be 'ASCII', 'latin1', or 'bytes'*");
        }

        [TestMethod]
        public void Load_EncodingIsValidatedBeforeTheFileIsEvenRead()
        {
            // NumPy checks encoding first, so an empty file with a bad encoding reports the encoding.
            Action act = () => np.load(Array.Empty<byte>(), encoding: "utf-8");
            act.Should().Throw<ArgumentException>().WithMessage("*encoding must be*");
        }

        [TestMethod]
        public void Load_MmapMode_OnInMemoryImage_RejectedNoPath()
        {
            // mmap needs a real file to map. An in-memory image (byte[] / stream) has none, so NumSharp
            // raises the same error NumPy raises for a file object — validated on the .npy branch, so an
            // .npz still ignores mmap_mode (covered by NpyMemmapTests). Full mode/behaviour parity lives
            // in NpyMemmapTests; here we only pin the byte[] overload.
            byte[] data = np.save(np.arange(3));

            foreach (string mode in new[] { "r", "r+", "c", "readonly" })
            {
                Action act = () => np.load(data, mmap_mode: mode);
                act.Should().Throw<ValueError>()
                   .WithMessage("*Memmap cannot use existing file handles*", $"'{mode}' needs a path to map");
            }
        }

        [TestMethod]
        public void Load_MaxHeaderSize_GuardsAgainstOversizedHeaders()
        {
            byte[] data = np.save(np.arange(3)); // a normal 118-byte header

            Action tiny = () => np.load(data, max_header_size: 10);
            tiny.Should().Throw<FormatException>().WithMessage("*is large and may not be safe to load securely*");

            Action trusted = () => np.load(data, max_header_size: 10, allow_pickle: true);
            trusted.Should().NotThrow("allow_pickle declares the file trusted, which lifts the header guard");
        }

        [TestMethod]
        public void Save_Decimal_ThrowsWithTheFix()
        {
            var dec = np.array(new[] { 1.5m, 2.5m });

            Action act = () => np.save(At("dec.npy"), dec);
            act.Should().Throw<NotSupportedException>()
               .WithMessage("*Decimal has no NumPy dtype*astype*",
                   "the error should name the workaround, since no NumPy dtype can hold a Decimal");
        }

        #endregion

        #region legacy fixture

        [TestMethod]
        public void Load_LegacyNumPyFixture()
        {
            var arr = np.load_npy(@"data/1-dim-int32_4_comma_empty.npy");

            arr.typecode.Should().Be(NPTypeCode.Int32);
            arr.shape.Should().Equal(new long[] { 4 });
            arr.ToArray<int>().Should().Equal(0, 1, 2, 3);
        }

        #endregion
    }
}
