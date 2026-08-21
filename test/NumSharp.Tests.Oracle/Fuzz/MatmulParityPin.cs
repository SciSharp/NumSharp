using System;
using System.IO;
using System.Text.Json;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The host pin shipped alongside the <c>matmul_parity</c> corpus.
    /// </summary>
    /// <remarks>
    ///     Every other tier records values that any correct implementation must reproduce on any
    ///     machine. This one cannot: NumPy computes float matrix products with cblas, and
    ///     scipy-openblas' <c>sgemm</c>/<c>dgemm</c> accumulate in a multi-accumulator register
    ///     scheme whose bits depend on the library build, on the DYNAMIC_ARCH kernel it selects
    ///     for the running CPU, and on the worker-thread count (measured: 1, 2, 4 and 24 threads
    ///     all give different bytes). So the corpus records the identity of the BLAS that produced
    ///     it, and the gate compares before it judges — a mismatch is Inconclusive, not red.
    /// </remarks>
    public sealed class MatmulParityPin
    {
        public string Numpy { get; set; }
        public string Platform { get; set; }
        public string Machine { get; set; }
        public string Blas_Library { get; set; }

        /// <summary>
        ///     SHA-256 of the BLAS binary NumPy loaded when the corpus was generated.
        /// </summary>
        /// <remarks>
        ///     This, not <see cref="Blas_Library"/>, is the real identity. A file name is a bad
        ///     proxy for a binary in both directions: pip's delvewheel mangles it per build (numpy
        ///     ships <c>libscipy_openblas64_-74a4….dll</c>, the hash baked into the name), while the
        ///     copy this repo's package bundles is the SAME BYTES under the plain name
        ///     <c>libscipy_openblas64_.dll</c>. Matching on the name alone therefore declared a
        ///     genuinely bit-identical host "different" and skipped the tier — and would equally
        ///     have accepted a differently-built library that happened to share a name.
        /// </remarks>
        public string Blas_Library_Sha256 { get; set; }

        public string Blas_Config { get; set; }
        public string Blas_Corename { get; set; }
        public int Blas_Threads { get; set; }

        private static readonly JsonSerializerOptions J = new() { PropertyNameCaseInsensitive = true };

        public static MatmulParityPin Load()
        {
            var path = FuzzCorpus.CorpusPath("matmul_parity.host.jsonl");
            foreach (var line in File.ReadLines(path))
            {
                if (!string.IsNullOrWhiteSpace(line))
                    return JsonSerializer.Deserialize<MatmulParityPin>(line, J);
            }

            throw new InvalidDataException($"empty host pin: {path}");
        }

        /// <summary>
        ///     Turns the parity backend on and checks the loaded library against the pin.
        /// </summary>
        /// <returns>
        ///     Null when the host matches and the corpus may be replayed; otherwise the reason the
        ///     tier is inconclusive on this machine.
        /// </returns>
        public string TryEnableParityBackend()
        {
            try
            {
                OpenBlasEngine.Enable(threads: Blas_Threads > 0 ? Blas_Threads : 0);
            }
            catch (Exception e)
            {
                return "matmul_parity is host-pinned and no CBLAS library could be loaded here " +
                       $"({e.GetType().Name}: {e.Message.Split('\n')[0]}). The corpus was generated " +
                       $"against '{Blas_Library}' shipped with numpy {Numpy}; install that wheel or " +
                       "point NUMSHARP_OPENBLAS_LIBRARY at its BLAS to run this gate.";
            }

            var info = OpenBlasEngine.Info ?? string.Empty;

            // Identity by CONTENT when the corpus records it: the same bytes under a different file
            // name are the same library, and that is not a corner case — it is the normal outcome
            // now that the package bundles NumPy's own pinned OpenBLAS build under its unmangled
            // name. Fall back to the file name only for corpora generated before the hash existed.
            var loadedHash = TryHashLoadedLibrary();
            if (!string.IsNullOrEmpty(Blas_Library_Sha256) && loadedHash != null)
            {
                if (!string.Equals(loadedHash, Blas_Library_Sha256, StringComparison.OrdinalIgnoreCase))
                {
                    OpenBlasEngine.Disable();
                    return $"matmul_parity is host-pinned to the BLAS binary sha256 " +
                           $"{Blas_Library_Sha256} ('{Blas_Library}', numpy {Numpy}); this host " +
                           $"loaded {loadedHash} — {info}. Different builds round differently — " +
                           "regenerate the corpus here (python test/oracle/gen_oracle.py matmul_parity) " +
                           "to gate against your own host.";
                }
            }
            else if (!string.IsNullOrEmpty(Blas_Library) &&
                     !info.Contains(Blas_Library, StringComparison.OrdinalIgnoreCase))
            {
                OpenBlasEngine.Disable();
                return $"matmul_parity is host-pinned to '{Blas_Library}' (numpy {Numpy}); this host " +
                       $"loaded a different BLAS: {info}. Different binaries round differently — " +
                       "regenerate the corpus here (python test/oracle/gen_oracle.py matmul_parity) " +
                       "to gate against your own host.";
            }

            if (!string.IsNullOrEmpty(Blas_Corename) && !info.Contains(Blas_Corename, StringComparison.OrdinalIgnoreCase))
            {
                OpenBlasEngine.Disable();
                return $"matmul_parity is host-pinned to the '{Blas_Corename}' OpenBLAS kernel " +
                       $"({Blas_Config}); this CPU dispatched to a different one: {info}. " +
                       "DYNAMIC_ARCH picks a different accumulator layout per micro-architecture.";
            }

            return null;
        }

        /// <summary>SHA-256 of the BLAS binary actually loaded, or null if it cannot be read.</summary>
        private static string TryHashLoadedLibrary()
        {
            try
            {
                var path = OpenBlasEngine.LibraryPath;
                if (string.IsNullOrEmpty(path) || !File.Exists(path))
                    return null;   // a bare loader name resolved by the OS — no path to hash

                using var stream = File.OpenRead(path);
                return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream))
                              .ToLowerInvariant();
            }
            catch (IOException)
            {
                return null;
            }
            catch (UnauthorizedAccessException)
            {
                return null;
            }
        }
    }
}
