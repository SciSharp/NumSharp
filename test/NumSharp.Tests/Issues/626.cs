using System;
using System.Linq;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the progress comment on GitHub #626, "[NumPy Parity] Neural Networks Project to fully train byte-identical
    ///     weights" (https://github.com/SciSharp/NumSharp/issues/626), which stays open: its four pillars are bit-exact with
    ///     NumPy 2.4.2, sampled here on the MLP's own operations - the seeded weight init (P1), the GEMM through the OpenBLAS
    ///     backend (P2), float32 <c>exp</c> (P3) and the axis sums (P4). The <c>mnist_mlp.py</c> twin and the S0-S7 ladder are
    ///     the open part. Every expected value is a NumPy 2.4.2 bit pattern.
    /// </summary>
    /// <remarks>
    ///     <c>[DoNotParallelize]</c>: P1 seeds the GLOBAL <c>np.random</c> and P2 binds the process-wide OpenBLAS backend
    ///     (TestParallelization's rules). The full gates are the random-API oracle, the <c>exp_f32</c> and
    ///     <c>matmul_parity</c> corpus tiers and the add.reduce schedule tiers; these tests pin one MLP-shaped sample each.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class Issue626
    {
        /// <summary>Puts the process back on NumSharp's own kernels after the P2 test bound OpenBLAS.</summary>
        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        /// <summary>The bit patterns of a float32 array, in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>Each element's 32 bits.</returns>
        private static uint[] Bits(NDArray a) => a.ToArray<float>().Select(IssueAnswerSupport.Bits).ToArray();

        /// <summary>
        ///     P1: <c>seed(1337)</c>, <c>normal(0, 0.05, 6)</c> and <c>astype(float32)</c>, the MLP's weight init, gives NumPy's
        ///     exact float32 bits.
        /// </summary>
        [TestMethod]
        public void P1_SeededWeightInit_IsNumPysBits()
        {
            np.random.seed(1337);
            var w = np.random.normal(0.0, 0.05, new Shape(6)).astype(np.float32);

            Bits(w).Should().Equal(0xBD100344u, 0xBCC8D1D5u, 0xBC83D0AEu, 0xBDB3B856u, 0x3C294CAEu, 0xBDCDF418u);
        }

        /// <summary>
        ///     P3: float32 <c>np.exp</c> is NumPy's own kernel, bit for bit, at ordinary inputs, the subnormal band, overflow
        ///     and <c>-0.0</c>.
        /// </summary>
        [TestMethod]
        public void P3_Float32Exp_IsNumPysBits()
        {
            var x = np.array(new[] { 0.5f, -87.5f, 88.0f, 1e-3f, -1.25f, 10.0f, -0.0f, 3.0f });

            Bits(np.exp(x)).Should().Equal(0x3FD3094Cu, 0x006CB2BCu, 0x7EF882B7u, 0x3F8020C9u, 0x3E92B0C1u, 0x46AC14EFu, 0x3F800000u, 0x41A0AF2Eu);
        }

        /// <summary>
        ///     P4: the MLP's sums - <c>sum(axis=1)</c> over contiguous 128-element rows and <c>sum(axis=0)</c> down the
        ///     columns of a (2, 128) float32 array - give NumPy's add.reduce bits.
        /// </summary>
        [TestMethod]
        public void P4_AxisSums_AreNumPysBits()
        {
            var s = (np.arange(2 * 128).astype(np.float32).reshape(2, 128) * NDArray.Scalar(0.1f)).astype(np.float32);

            Bits(s.sum(axis: 1)).Should().Equal(0x444B3334u, 0x45193333u);
            Bits(s.sum(axis: 0)["0:4"]).Should().Equal(0x414CCCCDu, 0x41500001u, 0x41533333u, 0x41566667u);
        }

        /// <summary>
        ///     P2: with the OpenBLAS backend bound, a float32 (3, 300) @ (300, 2) product gives NumPy's bits, which a sequential
        ///     sum misses in 4 of the 6 elements (NumPy's bits are OpenBLAS's multi-accumulator sgemm). Inconclusive on a host
        ///     without the native library.
        /// </summary>
        [TestMethod]
        public void P2_OpenBlasGemm_IsNumPysBits()
        {
            try
            {
                OpenBlasEngine.Enable();
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no OpenBLAS library on this host: " + e.Message.Split('\n')[0]);
            }

            // IEEE division only, so both runtimes build the same input bits.
            var a = (np.arange(900).astype(np.float32) % NDArray.Scalar(97f) / NDArray.Scalar(97f)).reshape(3, 300);
            var b = (np.arange(600).astype(np.float32) % NDArray.Scalar(89f) / NDArray.Scalar(89f) - NDArray.Scalar(0.5f)).reshape(300, 2);

            Bits(np.matmul(a, b)).Should().Equal(0xC06FCF1Bu, 0xC04F3FDAu, 0x3FC13B13u, 0x3FDFEC0Eu, 0x400E85DDu, 0x400CA79Du);
        }
    }
}
