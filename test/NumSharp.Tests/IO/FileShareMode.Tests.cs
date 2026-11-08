using System;
using System.IO;
using NumSharp.IO;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     NumPy's file functions open their files through Python's <c>open()</c>, whose Windows share mode admits other
    ///     readers and writers (<c>FILE_SHARE_READ | FILE_SHARE_WRITE</c>); NumSharp opens every file it is given by path
    ///     the same way (<see cref="NDFileShare"/>, parity QA-39). Every expectation is NumPy 2.4.2's behaviour probed on
    ///     win-amd64 (<c>p103</c> for the memmap pairs, <c>rp3</c> for a file another handle holds open).
    /// </summary>
    /// <remarks>
    ///     Before the fix a live <c>r+</c> memmap locked out every other load of its file and the readers / writers refused
    ///     a file another handle held open, each with <see cref="IOException"/> "The process cannot access the file …
    ///     because it is being used by another process"; only <c>np.fromfile</c> shared. The tests that need a mapping gone
    ///     dispose it; the cleanup collects whatever a failing assertion left mapped before deleting the directory, which
    ///     is why the class is <c>[DoNotParallelize]</c> (a forced collection suspends every thread).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class FileShareModeTests
    {
        private string _dir;

        /// <summary>Creates the test's private temporary directory.</summary>
        [TestInitialize]
        public void Init()
        {
            _dir = Path.Combine(Path.GetTempPath(), "ns_share_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        /// <summary>Releases any mapping a failing test left open and removes the directory (best effort).</summary>
        [TestCleanup]
        public void Cleanup() => MappedFileCleanup.DeleteDirectory(_dir);

        /// <summary>
        ///     While <c>np.load(p, mmap_mode='r+')</c> lives (and has written <c>a[1] = 99</c>), NumPy's plain
        ///     <c>np.load(p)</c>, an <c>'r'</c> map and a second <c>'r+'</c> map all open the file and read
        ///     <c>[0, 99, 2, 3]</c>; while an <c>'r'</c> or <c>'c'</c> map lives an <c>'r+'</c> map opens too.
        /// </summary>
        [TestMethod]
        public void LiveMemmap_AdmitsEveryOtherLoadOfItsFile()
        {
            string p = Path.Combine(_dir, "a.npy");
            using (var seed = np.arange(4.0))
                np.save(p, seed);

            using (var rw = (NDArray)np.load(p, "r+"))
            {
                rw[1] = (NDArray)99.0;
                using var plain = (NDArray)np.load(p);
                using var readOnly = (NDArray)np.load(p, "r");
                using var secondReadWrite = (NDArray)np.load(p, "r+");

                plain.ToArray<double>().Should().Equal(0.0, 99.0, 2.0, 3.0);
                readOnly.GetDouble(1).Should().Be(99.0, "an 'r' map of the file sees the 'r+' map's write");
                secondReadWrite.GetDouble(1).Should().Be(99.0);
            }

            using (var ro = (NDArray)np.load(p, "r"))
            using (var rw = (NDArray)np.load(p, "r+"))
                rw.GetDouble(3).Should().Be(3.0);

            using (var cow = (NDArray)np.load(p, "c"))
            using (var rw = (NDArray)np.load(p, "r+"))
                cow.GetDouble(1).Should().Be(99.0);
        }

        /// <summary>
        ///     A file another handle holds open for reading AND writing (another program's writer, here a .NET
        ///     <see cref="FileStream"/> with <see cref="FileShare.ReadWrite"/>) is read by <c>np.load</c> (.npy and .npz),
        ///     <c>np.loadtxt</c> and <c>np.fromfile</c> and overwritten by <c>np.save</c>, <c>np.savez</c>,
        ///     <c>np.savetxt</c> and <c>ndarray.tofile</c>, as NumPy's twin does with <c>open(p, 'r+b')</c> held open.
        /// </summary>
        [TestMethod]
        public void FileHeldOpenByAnotherHandle_IsReadAndWritten()
        {
            string npy = Path.Combine(_dir, "a.npy");
            string npz = Path.Combine(_dir, "a.npz");
            string txt = Path.Combine(_dir, "a.txt");
            string raw = Path.Combine(_dir, "a.bin");
            using var values = np.array(new[] { 1.5, 2.5, 3.5 });
            np.save(npy, values);
            np.savez(npz, values);
            np.savetxt(txt, values);
            values.tofile(raw);

            using (HoldOpen(npy))
            {
                using var loaded = (NDArray)np.load(npy);
                loaded.ToArray<double>().Should().Equal(1.5, 2.5, 3.5);
                using var doubled = values * 2.0;
                np.save(npy, doubled);
            }
            using (var reread = (NDArray)np.load(npy))
                reread.ToArray<double>().Should().Equal(new[] { 3.0, 5.0, 7.0 }, "the overwrite went through");

            using (HoldOpen(npz))
            {
                using (var archive = np.load_npz(npz))
                using (var member = archive["arr_0"])
                    member.ToArray<double>().Should().Equal(1.5, 2.5, 3.5);
                np.savez(npz, values, values);
            }

            using (HoldOpen(txt))
            {
                using var text = np.loadtxt(txt);
                text.ToArray<double>().Should().Equal(1.5, 2.5, 3.5);
                np.savetxt(txt, values);
            }

            using (HoldOpen(raw))
            {
                using var read = np.fromfile(raw);
                read.ToArray<double>().Should().Equal(1.5, 2.5, 3.5);
                values.tofile(raw);
            }
        }

        /// <summary>
        ///     The other direction: while NumSharp holds a file (a live <c>'r+'</c> memmap), another handle may open it for
        ///     reading and writing, as one may while NumPy holds it through <c>open()</c>.
        /// </summary>
        [TestMethod]
        public void LiveMemmap_LetsAnotherHandleOpenTheFileReadWrite()
        {
            string p = Path.Combine(_dir, "b.npy");
            using (var seed = np.arange(3.0))
                np.save(p, seed);

            using var rw = (NDArray)np.load(p, "r+");
            Action open = () => HoldOpen(p).Dispose();

            open.Should().NotThrow();
        }

        /// <summary>Opens <paramref name="path"/> the way another program's writer would: read-write, sharing both.</summary>
        /// <param name="path">The file to hold.</param>
        /// <returns>The open handle; dispose it to let go.</returns>
        private static FileStream HoldOpen(string path)
            => new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite);
    }
}
