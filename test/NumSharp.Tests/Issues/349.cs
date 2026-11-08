using System;
using System.Linq;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #349, "Scipy.Signal" (https://github.com/SciSharp/NumSharp/issues/349) as not
    ///     planned: <c>scipy.signal</c> is out of a NumPy port's scope, but a spectrogram can be built from the window functions
    ///     and <c>np.fft.rfft</c>. The SciPy-through-pythonnet half of the answer is backed in NumSharp.Tests.Interop.
    /// </summary>
    [TestClass]
    public class Issue349
    {
        /// <summary>
        ///     A 32-sample two-tone signal, cut into 8-sample frames with a hop of 4, Hann-windowed and transformed with
        ///     <c>np.fft.rfft</c>, gives NumPy 2.4.2's (7, 5) power spectrogram. The comparison is relative to 1e-12 because the
        ///     signal goes through <c>sin</c> / <c>cos</c>, whose last bit is the platform C library's, not NumPy's.
        /// </summary>
        [TestMethod]
        public void Spectrogram_FromHanningAndRfft_MatchesNumPy()
        {
            var t = np.arange(32) / 32.0;
            var sig = np.sin(2 * System.Math.PI * 5 * t) + 0.5 * np.cos(2 * System.Math.PI * 12 * t);

            const int nperseg = 8, hop = 4;
            var frames = np.stack(Enumerable.Range(0, (32 - nperseg) / hop + 1).Select(k => sig[$"{k * hop}:{k * hop + nperseg}"]).ToArray());
            var spec = np.power(np.abs(np.fft.rfft(frames * np.hanning(nperseg), axis: 1)), 2);

            spec.shape.Should().Equal(7L, 5L);
            var expected = new[]
            {
                0.18198914884614042, 3.035721021542835, 3.353759441267158, 1.4000394754273904, 0.7791463938944285,
                1.7349749066357054, 3.0396832620430163, 3.0335661240682925, 1.268563599116912, 0.8507223611631584,
                2.045213643537573, 2.926796914412602, 2.122376755198002, 0.926299409904112, 0.969341048096889,
                0.4889189713666119, 2.750940964999565, 1.1634526142038824, 0.5746694138586734, 1.066818426354479,
                0.20009641794416386, 2.676705831822308, 0.6707642365829036, 0.4149843714735589, 1.079512047342497,
                1.789987061627365, 2.7598257541786317, 0.9234228867682872, 0.539856828926391, 0.9986849703280104,
                2.104905310124105, 2.890033820979374, 1.8211787751427333, 0.8808075164614151, 0.8782261232538682,
            };
            var actual = spec.ToArray<double>();
            for (int i = 0; i < expected.Length; i++)
                actual[i].Should().BeApproximately(expected[i], 1e-12 * System.Math.Max(1.0, System.Math.Abs(expected[i])), $"cell {i}");
        }
    }
}
