using System;
using System.IO;
using System.Text.Json;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
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
                np.parity_matmul(true, threads: Blas_Threads > 0 ? Blas_Threads : 0);
            }
            catch (Exception e)
            {
                return "matmul_parity is host-pinned and no CBLAS library could be loaded here " +
                       $"({e.GetType().Name}: {e.Message.Split('\n')[0]}). The corpus was generated " +
                       $"against '{Blas_Library}' shipped with numpy {Numpy}; install that wheel or " +
                       "point NUMSHARP_PARITY_BLAS at its BLAS to run this gate.";
            }

            var info = np.parity_matmul_info() ?? string.Empty;
            if (!string.IsNullOrEmpty(Blas_Library) && !info.Contains(Blas_Library, StringComparison.OrdinalIgnoreCase))
            {
                np.parity_matmul(false);
                return $"matmul_parity is host-pinned to '{Blas_Library}' (numpy {Numpy}); this host " +
                       $"loaded a different BLAS: {info}. Different binaries round differently — " +
                       "regenerate the corpus here (python test/oracle/gen_oracle.py matmul_parity) " +
                       "to gate against your own host.";
            }

            if (!string.IsNullOrEmpty(Blas_Corename) && !info.Contains(Blas_Corename, StringComparison.OrdinalIgnoreCase))
            {
                np.parity_matmul(false);
                return $"matmul_parity is host-pinned to the '{Blas_Corename}' OpenBLAS kernel " +
                       $"({Blas_Config}); this CPU dispatched to a different one: {info}. " +
                       "DYNAMIC_ARCH picks a different accumulator layout per micro-architecture.";
            }

            return null;
        }
    }
}
