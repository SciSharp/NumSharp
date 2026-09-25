using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact parity of the modern <see cref="Generator"/>'s distribution surface — the 29 samplers beyond its core
    ///     draws (<c>beta</c> … <c>multivariate_normal</c>) — against NumPy 2.4.2's <c>np.random.default_rng</c>: every sampler
    ///     at parameters that reach each internal branch, every parameter constraint (type, message, the <c>-0.0</c>/NaN/inf
    ///     corners), the multivariate conversions and broadcasting, long-stream digests, and the two documented divergences.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every expected value, digest and message is NumPy's own output on win-amd64 (<c>default_rng(42)</c> for the tables,
    ///     <c>default_rng(2024)</c> for the digests). Floats compare BIT-for-bit (a NumPy <c>nan</c> matches any NaN — its repr
    ///     hides the sign). Integer samplers return int64 on both sides.
    ///     </para>
    ///     <para>
    ///     Documented divergences, each pinned to its measured bound rather than skipped: <c>pareto</c>/<c>power</c> call the
    ///     C runtime's <c>expm1</c>, whose approximation inside <c>[-ln 2, ln 1.5]</c> is closed (<see cref="ParetoPower_InBandExpm1_WithinDocumentedUlp"/>);
    ///     <c>multivariate_normal</c> is byte-identical only with a LAPACK backend (<see cref="MultivariateNormal_WithLapackBackend_ByteIdenticalToNumPy"/>),
    ///     its managed fallback exact for diagonal covariances and otherwise free to pick another eigenvector sign.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class GeneratorDistributionsTests
    {
        // ---------------------------------------------------------------- harness

        /// <summary>Calls the scalar-parameter sampler a probe line names, with NumPy's positional arguments and the given size.</summary>
        /// <param name="r">The generator (a fresh <c>default_rng(seed)</c>).</param>
        /// <param name="method">The sampler name as NumPy spells it.</param>
        /// <param name="a">The positional arguments as NumPy printed them (<c>nan</c>, <c>inf</c>, <c>10**9</c> accepted).</param>
        /// <param name="size">The output shape.</param>
        /// <returns>The draws.</returns>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not a scalar-parameter sampler.</exception>
        private static NDArray Draw(Generator r, string method, string[] a, Shape size)
        {
            double D(int i) => ParseDouble(a[i]);
            long L(int i) => (long)ParseDouble(a[i]);
            switch (method)
            {
                case "beta": return r.beta(D(0), D(1), size);
                case "chisquare": return r.chisquare(D(0), size);
                case "f": return r.f(D(0), D(1), size);
                case "noncentral_chisquare": return r.noncentral_chisquare(D(0), D(1), size);
                case "noncentral_f": return r.noncentral_f(D(0), D(1), D(2), size);
                case "standard_cauchy": return r.standard_cauchy(size);
                case "standard_t": return r.standard_t(D(0), size);
                case "vonmises": return r.vonmises(D(0), D(1), size);
                case "pareto": return r.pareto(D(0), size);
                case "weibull": return r.weibull(D(0), size);
                case "power": return r.power(D(0), size);
                case "laplace": return r.laplace(D(0), D(1), size);
                case "gumbel": return r.gumbel(D(0), D(1), size);
                case "logistic": return r.logistic(D(0), D(1), size);
                case "lognormal": return r.lognormal(D(0), D(1), size);
                case "rayleigh": return r.rayleigh(D(0), size);
                case "wald": return r.wald(D(0), D(1), size);
                case "triangular": return r.triangular(D(0), D(1), D(2), size);
                case "binomial": return r.binomial(L(0), D(1), size);
                case "negative_binomial": return r.negative_binomial(D(0), D(1), size);
                case "poisson": return r.poisson(D(0), size);
                case "zipf": return r.zipf(D(0), size);
                case "geometric": return r.geometric(D(0), size);
                case "hypergeometric": return r.hypergeometric(L(0), L(1), L(2), size);
                case "logseries": return r.logseries(D(0), size);
                default: throw new ArgumentException("unknown sampler " + method);
            }
        }

        /// <summary>
        ///     The multivariate probe lines, keyed by the exact name NumPy's probe printed: each is the same call spelled in
        ///     NumSharp (list arguments as C# arrays or NDArrays, NumPy's <c>size=k</c> as <c>new Shape(k)</c>).
        /// </summary>
        private static readonly Dictionary<string, Func<Generator, NDArray>> MultivariateCalls = new()
        {
            // ---- branch table (seed 42)
            ["multinomial 20,[.2,.3,.5]x3"] = r => r.multinomial(20, new[] { 0.2, 0.3, 0.5 }, new Shape(3)),
            ["multinomial 20,[.2,.3,.5]"] = r => r.multinomial(20, new[] { 0.2, 0.3, 0.5 }),
            ["multinomial [10,20],[.5,.5]"] = r => r.multinomial(np.array(new long[] { 10, 20 }), np.array(new[] { 0.5, 0.5 })),
            ["multinomial 10,[[.5,.5],[.1,.9]]"] = r => r.multinomial(10, np.array(new[,] { { 0.5, 0.5 }, { 0.1, 0.9 } })),
            ["multinomial [[5],[10]],[[.5,.5],[.1,.9]]"] = r => r.multinomial(np.array(new long[,] { { 5 }, { 10 } }), np.array(new[,] { { 0.5, 0.5 }, { 0.1, 0.9 } })),
            ["multinomial 1000,[.25,.25,.5]x2"] = r => r.multinomial(1000, new[] { 0.25, 0.25, 0.5 }, new Shape(2)),
            ["dirichlet 2,3,5"] = r => r.dirichlet(new[] { 2.0, 3.0, 5.0 }, new Shape(3)),
            ["dirichlet 0.05,0.05,0.05"] = r => r.dirichlet(new[] { 0.05, 0.05, 0.05 }, new Shape(3)),
            ["dirichlet 0.05,0"] = r => r.dirichlet(new[] { 0.05, 0.0 }, new Shape(3)),
            ["dirichlet 0,0"] = r => r.dirichlet(new[] { 0.0, 0.0 }, new Shape(2)),
            ["dirichlet 1,0.5"] = r => r.dirichlet(new[] { 1.0, 0.5 }, new Shape(3)),
            ["multivariate_hypergeometric count [3,2,5],4"] = r => r.multivariate_hypergeometric(new long[] { 3, 2, 5 }, 4, new Shape(3), method: "count"),
            ["multivariate_hypergeometric marginals [3,2,5],4"] = r => r.multivariate_hypergeometric(new long[] { 3, 2, 5 }, 4, new Shape(3)),
            ["multivariate_hypergeometric marginals [100,200,300],450"] = r => r.multivariate_hypergeometric(new long[] { 100, 200, 300 }, 450, new Shape(2)),
            ["multivariate_hypergeometric count [100,200,300],450"] = r => r.multivariate_hypergeometric(new long[] { 100, 200, 300 }, 450, new Shape(2), method: "count"),
            ["multivariate_hypergeometric marginals [3,2,5],0"] = r => r.multivariate_hypergeometric(new long[] { 3, 2, 5 }, 0, new Shape(2)),
            // ---- validation table (seed 42, size 2 unless NumPy's call had none)
            ["multinomial(-1,[.5,.5])"] = r => r.multinomial(-1, new[] { 0.5, 0.5 }, new Shape(2)),
            ["multinomial(10,[.6,.6])"] = r => r.multinomial(10, new[] { 0.6, 0.6 }, new Shape(2)),
            ["multinomial(10,[-0.1,1.1])"] = r => r.multinomial(10, new[] { -0.1, 1.1 }, new Shape(2)),
            ["multinomial(10,[nan,1])"] = r => r.multinomial(10, new[] { double.NaN, 1.0 }, new Shape(2)),
            ["multinomial(10,[])"] = r => r.multinomial(10, new double[0], new Shape(2)),
            ["multinomial(10,[[.6,.6],[.1,.9]])"] = r => r.multinomial(10, np.array(new[,] { { 0.6, 0.6 }, { 0.1, 0.9 } })),
            ["multinomial([10,20],[.5,.5],size=3)"] = r => r.multinomial(np.array(new long[] { 10, 20 }), np.array(new[] { 0.5, 0.5 }), new Shape(3)),
            ["multinomial([10,-1],[.5,.5])"] = r => r.multinomial(np.array(new long[] { 10, -1 }), np.array(new[] { 0.5, 0.5 })),
            ["multinomial(10,f32 sum>1)"] = r => r.multinomial(10, np.array(new[] { 0.5f, 0.50000006f, 0.0f }), new Shape(2)),
            ["dirichlet([-1,1])"] = r => r.dirichlet(new[] { -1.0, 1.0 }, new Shape(2)),
            ["dirichlet([nan,1])"] = r => r.dirichlet(new[] { double.NaN, 1.0 }, new Shape(2)),
            ["dirichlet([])"] = r => r.dirichlet(new double[0], new Shape(2)),
            ["multivariate_hypergeometric([3,2],6)"] = r => r.multivariate_hypergeometric(new long[] { 3, 2 }, 6, new Shape(2)),
            ["multivariate_hypergeometric([3,-2],1)"] = r => r.multivariate_hypergeometric(new long[] { 3, -2 }, 1, new Shape(2)),
            ["multivariate_hypergeometric([3,2],-1)"] = r => r.multivariate_hypergeometric(new long[] { 3, 2 }, -1, new Shape(2)),
            ["multivariate_hypergeometric([3,2],1,method=x)"] = r => r.multivariate_hypergeometric(new long[] { 3, 2 }, 1, new Shape(2), method: "x"),
            ["multivariate_hypergeometric([10**9,1],1)"] = r => r.multivariate_hypergeometric(new long[] { 1000000000, 1 }, 1, new Shape(2)),
            ["multivariate_hypergeometric([10**9,1],1,count)"] = r => r.multivariate_hypergeometric(new long[] { 1000000000, 1 }, 1, new Shape(2), method: "count"),
            ["multivariate_hypergeometric([],0)"] = r => r.multivariate_hypergeometric(new long[0], 0, new Shape(2)),
            ["multivariate_hypergeometric([[1,2]],1)"] = r => r.multivariate_hypergeometric(np.array(new long[,] { { 1, 2 } }), 1, new Shape(2)),
            ["multivariate_hypergeometric([1.5,2],1)"] = r => r.multivariate_hypergeometric(np.array(new[] { 1.5, 2.0 }), 1, new Shape(2)),
        };

        /// <summary>Parses one value as NumPy printed it (<c>nan</c>, <c>inf</c>, <c>-inf</c>, <c>10**9</c>, <c>10**9-1</c> or a literal).</summary>
        /// <param name="s">The printed value.</param>
        /// <returns>The double.</returns>
        private static double ParseDouble(string s)
        {
            switch (s.Trim())
            {
                case "nan": return double.NaN;
                case "inf": return double.PositiveInfinity;
                case "-inf": return double.NegativeInfinity;
                case "10**9": return 1e9;
                case "10**9-1": return 1e9 - 1;
                default: return double.Parse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture);
            }
        }

        /// <summary>Asserts the first values of <paramref name="actual"/> (C order) equal NumPy's printed ones bit-for-bit.</summary>
        /// <param name="actual">NumSharp's draws.</param>
        /// <param name="expected">NumPy's values, comma-separated (empty for an empty result).</param>
        /// <param name="context">The case name for failure messages.</param>
        private static void AssertValues(NDArray actual, string expected, string context)
        {
            var exp = expected.Length == 0 ? Array.Empty<string>() : expected.Split(',').Select(t => t.Trim()).ToArray();
            Assert.IsTrue(actual.size >= exp.Length, $"{context}: {actual.size} values drawn, {exp.Length} expected");
            for (int i = 0; i < exp.Length; i++)
            {
                object got = actual.GetAtIndex(i);
                if (got is double d)
                {
                    double e = ParseDouble(exp[i]);
                    if (double.IsNaN(e))
                        Assert.IsTrue(double.IsNaN(d), $"{context}[{i}]: expected nan, got {d:R}");
                    else
                        Assert.AreEqual(BitConverter.DoubleToInt64Bits(e), BitConverter.DoubleToInt64Bits(d),
                            $"{context}[{i}]: expected {exp[i]}, got {d:R}");
                }
                else
                {
                    Assert.AreEqual(long.Parse(exp[i], CultureInfo.InvariantCulture), Convert.ToInt64(got), $"{context}[{i}]");
                }
            }
        }

        /// <summary>Runs a NumPy probe line: either values to match, or <c>ERR Type: message</c> to raise verbatim.</summary>
        /// <param name="draw">The NumSharp call.</param>
        /// <param name="expected">NumPy's values or <c>ERR Type: message</c>.</param>
        /// <param name="context">The case name for failure messages.</param>
        private static void AssertProbe(Func<NDArray> draw, string expected, string context)
        {
            if (expected.StartsWith("ERR ", StringComparison.Ordinal))
            {
                int colon = expected.IndexOf(": ", StringComparison.Ordinal);
                string type = expected.Substring(4, colon - 4);
                string message = expected.Substring(colon + 2);
                try
                {
                    draw();
                }
                catch (Exception e)
                {
                    Assert.AreEqual(type, e.GetType().Name, $"{context}: exception type ({e.Message})");
                    Assert.AreEqual(message, e.Message, $"{context}: message");
                    return;
                }
                Assert.Fail($"{context}: expected {type}: {message}, but nothing was thrown");
            }
            AssertValues(draw(), expected, context);
        }

        /// <summary>SHA-256 of an array's elements in C order, as NumPy's <c>hashlib.sha256(a.tobytes())</c> computes it.</summary>
        /// <param name="a">A float64 or int64 array.</param>
        /// <returns>The lowercase hex digest.</returns>
        private static string Sha256(NDArray a)
        {
            byte[] raw = a.typecode == NPTypeCode.Double
                ? MemoryMarshal.AsBytes<double>(a.ToArray<double>()).ToArray()
                : MemoryMarshal.AsBytes<long>(a.ToArray<long>()).ToArray();
            return Convert.ToHexString(SHA256.HashData(raw)).ToLowerInvariant();
        }

        // ---------------------------------------------------------------- every branch, seed 42

        /// <summary>
        ///     Each scalar-parameter sampler at parameters reaching each internal branch — beta's Johnk / tiny-threshold /
        ///     log-space / gamma paths, noncentral chi-square's Poisson mixture (<c>df &lt;= 1</c>) and NaN short-circuit,
        ///     vonmises' uniform / small-kappa / Best-Fisher / wrapped-normal paths, binomial inversion vs BTPE on both sides of
        ///     <c>p = 0.5</c>, Poisson product vs PTRS, geometric search vs inversion, the hypergeometric urn vs HRUA, zipf's
        ///     <c>a &gt;= 1025</c> shortcut — drawing 8 values from <c>np.random.default_rng(42)</c>.
        /// </summary>
        /// <param name="probe">The sampler and its arguments, as NumPy's probe named them.</param>
        /// <param name="expected">NumPy 2.4.2's 8 values, or <c>ERR Type: message</c>.</param>
        [TestMethod]
        [DataRow("beta 2,3", "0.33841183847153383, 0.41040972646061824, 0.5259001067191458, 0.41439318981265894, 0.5643058890979785, 0.6519194715482499, 0.4191360919177246, 0.22254322828336792")]
        [DataRow("beta 0.5,0.5", "0.7566840941053092, 0.009232118044248207, 0.07485660069290462, 0.1379900752243652, 0.791997219411125, 0.9869314519282654, 0.8205436447442123, 0.9411520958778484")]
        [DataRow("beta 1e-3,1e-3", "1.0, 1.0, 0.0, 1.0146124969103826e-14, 0.0, 0.0, 3.318010441431549e-107, 1.0")]
        [DataRow("beta 1e-105,1e-105", "0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0")]
        [DataRow("beta 0.5,2", "0.22102589079334944, 0.004804422916701851, 0.0053019576338894725, 0.16047255417051204, 0.09182129084901962, 0.16403852998750526, 0.25109679613290103, 0.021392637497741177")]
        [DataRow("beta 1,1", "0.22145847498048798, 0.6611679233762598, 0.8968029669142108, 0.7999647830577266, 0.9141985494084729, 0.18426180349383842, 0.46794822250730794, 0.7124984103754657")]
        [DataRow("beta 0,1", "ERR ValueError: a <= 0")]
        [DataRow("beta nan,1", "nan, nan, nan, nan, nan, nan, nan, nan")]
        [DataRow("chisquare 3", "3.0554394899785056, 4.358928659107372, 2.6205388975367514, 2.2972265456113243, 4.795249613001758, 2.4789023888119672, 3.495986440338192, 3.224016177954705")]
        [DataRow("chisquare 0.5", "0.7221819906451932, 1.2687953645726084, 0.00015733154968739177, 0.6721578190438395, 0.0005387798921232884, 0.03780764628190145, 0.343723789316858, 0.07731582223568848")]
        [DataRow("chisquare 2", "4.808417207931989, 4.672379311648907, 4.76952199974851, 0.5595885797031664, 0.17287479939675404, 2.9053210314123015, 2.819921388515147, 6.248591913098995")]
        [DataRow("chisquare 0", "ERR ValueError: df <= 0")]
        [DataRow("f 5,7", "0.788657119902264, 1.0534347033181144, 1.591901384026539, 1.060248160151596, 1.8349102866119666, 2.5627443976173163, 1.0779050290654908, 0.4768718811079487")]
        [DataRow("f 0.5,0.5", "0.5691871288389038, 0.0002340693587574413, 0.014250553660654692, 4.44571084388206, 0.18734926073371558, 7.383029641352632, 83.73055807058113, 27.29800296392803")]
        [DataRow("f 0,1", "ERR ValueError: dfnum <= 0")]
        [DataRow("noncentral_chisquare 3,1.5", "4.842553748271206, 9.45808761069264, 0.17887092216723874, 3.645297779550569, 0.2967499860022168, 4.1510262762666095, 8.99449321856509, 2.5967267882077")]
        [DataRow("noncentral_chisquare 0.5,1.5", "3.6784878039531472, 6.294769205308427, 0.6369753907484437, 3.3630266683762926, 1.0098336085843456, 5.3120946712425825, 3.0531993711571346, 0.435348764376272")]
        [DataRow("noncentral_chisquare 3,0", "3.0554394899785056, 4.358928659107372, 2.6205388975367514, 2.2972265456113243, 4.795249613001758, 2.4789023888119672, 3.495986440338192, 3.224016177954705")]
        [DataRow("noncentral_chisquare 3,nan", "nan, nan, nan, nan, nan, nan, nan, nan")]
        [DataRow("noncentral_chisquare 0.5,30", "42.56921300289814, 35.69128927616798, 30.67623358910165, 36.247953823948386, 40.53120047524009, 36.61057765603526, 30.51873531040179, 15.080624886091925")]
        [DataRow("noncentral_f 5,7,1.5", "1.0983780270802377, 0.4017311744952452, 0.9735605464264456, 0.22970982106529406, 2.1407904558259134, 0.8898100843093328, 0.692007816798907, 0.7651132505359128")]
        [DataRow("noncentral_f 0.5,7,1.5", "7.574133504591063, 0.12096488326871997, 2.5265473938607874, 23.61552419600117, 1.2896023658426021e-05, 0.8390154012403686, 5.591234652700629, 4.465636070894449")]
        [DataRow("standard_cauchy", "-0.2930016698581793, 0.7978730040882357, 1.4982843597001279, -0.40424789785555826, 0.01969553613971374, 1.1306339585249685, 0.05857725671581893, -0.544063124542466")]
        [DataRow("standard_t 3.5", "0.5676045589620277, 0.9916500852779986, -0.02812301505087557, 0.841030994729864, 0.7852582458203111, -0.7662413172794084, -0.2822072081830003, -0.20803662334780634")]
        [DataRow("standard_t 0.5", "0.7910033254946393, 53.02318054229338, 0.31511457509906, 0.9380881339565066, 4.526844982293566, -0.6747230241939323, -0.7353979288518486, -0.12695490488525973")]
        [DataRow("standard_t 2", "0.19936216868499293, 1.4187407090417825, -1.6187621316562906, 0.07232560050875605, -0.016423168988434003, 0.8426880006692907, 0.10615719268063999, 1.1922021255000101")]
        [DataRow("vonmises 0.5,2", "2.0156046999931245, 1.6965784456425403, 0.3570899324887584, 0.23978983189661562, 1.9415760234132486, 0.27940937241159247, 0.6730458135093618, 0.04627267376802546")]
        [DataRow("vonmises 0,1e-9", "1.7213166190998062, -0.3840380893017967, 2.2531371815723604, 1.2400999002927888, -2.5498589250729733, 2.9884233715702675, 1.6407891386670415, 1.797395039824692")]
        [DataRow("vonmises 1,1e-6", "-2.8517313227734657, -3.092339844087965, -1.3911902141955013, 2.4149281658956543, -1.0227610315575362, 0.28610915580955476, -2.6831058127988108, 2.113775279612385")]
        [DataRow("vonmises -2,1e7", "-1.999903639998602, -2.0003288718506096, -1.9997626864948455, -1.9997025673209414, -2.000616971499128, -2.000411785316408, -1.9999595733148998, -2.000100004688496")]
        [DataRow("vonmises 3,1e7", "3.000096360001398, 2.9996711281493904, 3.0002373135051545, 3.0002974326790586, 2.999383028500872, 2.999588214683592, 3.0000404266851004, 2.999899995311504")]
        [DataRow("vonmises 0,nan", "nan, nan, nan, nan, nan, nan, nan, nan")]
        [DataRow("vonmises 0,inf", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0")]
        [DataRow("pareto 3", "1.228665259618174, 1.1787033042726671, 1.2142645866056139, 0.09775234131023085, 0.029231561046690786, 0.6229089252642819, 0.5999732303681398, 1.8332713142957373")]
        [DataRow("pareto 0.5", "121.53751254099629, 105.951911868175, 116.86289000255597, 0.7499523862412976, 0.18871726764795982, 17.271108310494277, 15.775531870834524, 516.2839308886976")]
        [DataRow("weibull 1.79", "1.632426370003547, 1.6064619557611481, 1.6250362400587255, 0.4908755748909164, 0.2546704758160878, 1.2319539867202327, 1.2115905274114604, 1.889721736319996")]
        [DataRow("weibull 0", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0")]
        [DataRow("power 2.5", "0.9628357780513209, 0.960138075671411, 0.9620842462664194, 0.5688518711444587, 0.36917042103416314, 0.8988837267384739, 0.8940742460671552, 0.9821752674699098")]
        [DataRow("power 0.3", "0.7293486200057221, 0.7124932786776623, 0.724618127262586, 0.00908503471992583, 0.00024748989638220695, 0.4113353025528656, 0.3933506536824284, 0.8608119387154509")]
        [DataRow("laplace 0,1", "0.7938786426417068, -0.1303856263065431, 1.263000634398772, 0.5020906485159728, -1.6694284140409832, 3.0209414366405944, 0.7387292445495692, 0.8489326205054745")]
        [DataRow("gumbel 0.5,2", "-0.2935560666001952, 1.5969936744344135, -0.8419542775542994, 0.14330962854239226, 5.127054676785422, -2.1242666459166766, -0.21797153900042843, -0.3662640509315259")]
        [DataRow("logistic 0,1", "1.2307856313855068, -0.24571509486821116, 1.8036932691125598, 0.8347958402379975, -2.2636638542802237, 3.6894089149545044, 1.158938064000753, 1.3013631244954191")]
        [DataRow("lognormal 0,1", "1.3562412406168636, 0.35346029972713455, 2.1179554136618033, 2.5614274900691822, 0.1421268672871279, 0.2719384549479855, 1.136371626911503, 0.7288826073782175")]
        [DataRow("lognormal 1,0.5", "3.16565042457333, 1.6160872449928683, 3.9559690805896013, 4.350463356639649, 1.0247845605540022, 1.4175219571728652, 2.8977083532478187, 2.3207228346901947")]
        [DataRow("rayleigh 1.5", "3.2892155170871633, 3.2423530731877492, 3.275885300103492, 1.122084802647342, 0.6236732306606534, 2.5567503438305583, 2.5188932339738184, 3.7495775501345134")]
        [DataRow("wald 3,2", "2.069990769639344, 1.2325882864415432, 22.733531404311037, 3.5079249356569857, 2.9389004658607654, 8.409861373805754, 3.252620650814415, 1.7050238285801314")]
        [DataRow("wald 0.5,10", "0.4670724859018932, 0.5912375153928726, 0.7708433461471245, 0.5144987410764484, 0.49812509865815696, 0.6084605159839084, 0.5074371582518286, 0.4503915762808832")]
        [DataRow("triangular 0,3,10", "6.022176901736005, 3.7327430866960354, 6.85386815180876, 5.397366192510583, 1.680868952842394, 8.69369399242479, 5.910963333415157, 6.130181059711807")]
        [DataRow("triangular 0,0,1", "0.5245592030083697, 0.25091952351703384, 0.6239653206303739, 0.44988003949989586, 0.04825284234080818, 0.8438665687200717, 0.5112666391480452, 0.5374681689623446")]
        [DataRow("triangular 0,1,1", "0.8797477186989253, 0.6624790108011365, 0.9266055902655577, 0.8350856417514098, 0.30688328056062214, 0.9877359726347704, 0.8724332077530939, 0.8866026761052291")]
        [DataRow("binomial 10,0.35", "5, 3, 5, 4, 2, 7, 5, 5")]
        [DataRow("binomial 100,0.4", "46, 31, 34, 39, 40, 34, 38, 31")]
        [DataRow("binomial 1000,0.7", "705, 724, 707, 711, 696, 688, 678, 694")]
        [DataRow("binomial 20,0.9", "17, 18, 17, 17, 20, 15, 17, 17")]
        [DataRow("binomial 0,0.5", "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow("binomial 10,0", "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow("binomial 10,1", "10, 10, 10, 10, 10, 10, 10, 10")]
        [DataRow("negative_binomial 5,0.4", "12, 4, 15, 8, 8, 7, 9, 4")]
        [DataRow("negative_binomial 0.5,0.5", "2, 3, 2, 1, 3, 0, 0, 0")]
        [DataRow("negative_binomial 1,1", "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow("poisson 3.5", "4, 5, 5, 3, 5, 1, 5, 1")]
        [DataRow("poisson 100", "109, 112, 108, 87, 96, 104, 98, 102")]
        [DataRow("poisson 0", "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow("poisson 1e6", "1000845, 1001226, 998432, 1000798, 998694, 999629, 1000415, 999841")]
        [DataRow("zipf 3", "2, 1, 1, 1, 1, 1, 1, 2")]
        [DataRow("zipf 1.5", "19, 1, 1, 3, 5, 17, 20, 3")]
        [DataRow("zipf 1.05", "12094692525, 11, 22020, 766585, 327506162750, 5019555127, 144927078024861824, 15567694326")]
        [DataRow("zipf 2000", "1, 1, 1, 1, 1, 1, 1, 1")]
        [DataRow("geometric 0.35", "4, 2, 5, 3, 1, 9, 4, 4")]
        [DataRow("geometric 0.1", "23, 23, 23, 3, 1, 14, 14, 30")]
        [DataRow("geometric 1e-5", "240420, 233618, 238475, 27980, 8644, 145266, 140996, 312429")]
        [DataRow("geometric 1e-300", "9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807")]
        [DataRow("geometric 1", "1, 1, 1, 1, 1, 1, 1, 1")]
        [DataRow("hypergeometric 10,7,8", "6, 4, 5, 6, 5, 6, 5, 4")]
        [DataRow("hypergeometric 100,200,50", "16, 18, 19, 14, 20, 13, 12, 18")]
        [DataRow("hypergeometric 300,100,250", "187, 189, 191, 184, 192, 182, 189, 186")]
        [DataRow("hypergeometric 15,15,20", "10, 9, 9, 11, 12, 9, 11, 11")]
        [DataRow("hypergeometric 5,5,0", "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow("hypergeometric 0,5,3", "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow("hypergeometric 1000,1000,1995", "999, 996, 998, 998, 997, 997, 997, 995")]
        [DataRow("logseries 0.6", "1, 2, 1, 5, 1, 1, 2, 2")]
        [DataRow("logseries 0.99", "2, 4, 210, 11, 16, 71, 20, 2")]
        [DataRow("logseries 0.1", "1, 1, 1, 1, 2, 1, 1, 1")]
        [DataRow("logseries 0", "1, 1, 1, 1, 1, 1, 1, 1")]
        public void Branch_Seed42_MatchesNumPy(string probe, string expected)
        {
            int sp = probe.IndexOf(' ');
            string method = sp < 0 ? probe : probe.Substring(0, sp);
            string[] args = sp < 0 ? Array.Empty<string>() : probe.Substring(sp + 1).Split(',');
            AssertProbe(() => Draw(np.random.default_rng(42), method, args, new Shape(8)), expected, probe);
        }

        /// <summary>
        ///     The multivariate samplers' branches: multinomial's plain path (with and without size) and its vector path
        ///     (broadcast counts, stacked probability rows), dirichlet's gamma normalization vs stick-breaking (incl. a zero
        ///     alpha, all-zero alphas), and multivariate_hypergeometric's count vs marginals methods incl. <c>nsample &gt;
        ///     total / 2</c> and <c>nsample = 0</c>.
        /// </summary>
        /// <param name="probe">The call, as NumPy's probe named it (see <see cref="MultivariateCalls"/>).</param>
        /// <param name="expected">NumPy 2.4.2's values in C order.</param>
        [TestMethod]
        [DataRow("multinomial 20,[.2,.3,.5]x3", "5, 5, 10, 6, 6, 8, 2, 11, 7")]
        [DataRow("multinomial 20,[.2,.3,.5]", "5, 5, 10")]
        [DataRow("multinomial [10,20],[.5,.5]", "6, 4, 10, 10")]
        [DataRow("multinomial 10,[[.5,.5],[.1,.9]]", "6, 4, 1, 9")]
        [DataRow("multinomial [[5],[10]],[[.5,.5],[.1,.9]]", "3, 2, 0, 5, 7, 3, 1, 9")]
        [DataRow("multinomial 1000,[.25,.25,.5]x2", "228, 251, 521, 241, 257, 502")]
        [DataRow("dirichlet 2,3,5", "0.18795090692457098, 0.3674401449176678, 0.44460894815776125, 0.1518747085022099, 0.40398994546391204, 0.44413534603387805, 0.1877632401978632, 0.26534082814541293, 0.5468959316567238")]
        [DataRow("dirichlet 0.05,0.05,0.05", "0.957324777930032, 0.04201923963082357, 0.0006559824391444192, 3.855775227079375e-21, 0.34424498637085515, 0.6557550136291448, 4.131033273246499e-15, 1.1049836399546985e-08, 0.9999999889501595")]
        [DataRow("dirichlet 0.05,0", "1.0, 0.0, 1.0, 0.0, 1.0, 0.0")]
        [DataRow("dirichlet 0,0", "0.0, 0.0, 0.0, 0.0")]
        [DataRow("dirichlet 1,0.5", "0.9258269458011242, 0.07417305419887592, 0.969274370137255, 0.030725629862744897, 0.6226095261333335, 0.37739047386666646")]
        [DataRow("multivariate_hypergeometric count [3,2,5],4", "1, 1, 2, 2, 1, 1, 0, 1, 3")]
        [DataRow("multivariate_hypergeometric marginals [3,2,5],4", "2, 0, 2, 2, 0, 2, 1, 0, 3")]
        [DataRow("multivariate_hypergeometric marginals [100,200,300],450", "76, 147, 227, 72, 155, 223")]
        [DataRow("multivariate_hypergeometric count [100,200,300],450", "76, 155, 219, 68, 153, 229")]
        [DataRow("multivariate_hypergeometric marginals [3,2,5],0", "0, 0, 0, 0, 0, 0")]
        public void MultivariateBranch_Seed42_MatchesNumPy(string probe, string expected)
            => AssertProbe(() => MultivariateCalls[probe](np.random.default_rng(42)), expected, probe);

        // ---------------------------------------------------------------- every constraint, seed 42

        /// <summary>
        ///     Every scalar-parameter sampler at the boundary values of its constraint — negative, <c>-0.0</c>, <c>0</c>, NaN,
        ///     inf, and the <c>&gt; 1</c> / <c>== 1</c> probability edges — plus the discrete bounds (<c>lam value too large</c>,
        ///     negative_binomial's <c>n too large or p too small</c>, hypergeometric's <c>10**9</c> caps and order of checks).
        ///     Each either draws NumPy's first two values or raises NumPy's exception type and message verbatim.
        /// </summary>
        /// <param name="probe">The call as NumPy's probe named it (<c>method(arg,...)</c>).</param>
        /// <param name="expected">NumPy 2.4.2's first two values, or <c>ERR Type: message</c>.</param>
        [TestMethod]
        [DataRow("f(-1.0,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("f(1,-1.0)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(-1.0,1,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("noncentral_f(1,-1.0,1)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(1,1,-1.0)", "ERR ValueError: nonc < 0")]
        [DataRow("chisquare(-1.0)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(-1.0,1)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(1,-1.0)", "ERR ValueError: nonc < 0")]
        [DataRow("standard_t(-1.0)", "ERR ValueError: df <= 0")]
        [DataRow("vonmises(0,-1.0)", "ERR ValueError: kappa < 0")]
        [DataRow("vonmises(-1.0,1)", "0.8378819513682036, 0.513096044773119")]
        [DataRow("pareto(-1.0)", "ERR ValueError: a <= 0")]
        [DataRow("weibull(-1.0)", "ERR ValueError: a < 0")]
        [DataRow("power(-1.0)", "ERR ValueError: a <= 0")]
        [DataRow("laplace(0,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("laplace(-1.0,1)", "-0.20612135735829318, -1.130385626306543")]
        [DataRow("gumbel(0,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("logistic(0,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("lognormal(0,-1.0)", "ERR ValueError: sigma < 0")]
        [DataRow("lognormal(-1.0,1)", "0.4989332696917954, 0.13003077753990877")]
        [DataRow("rayleigh(-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("wald(-1.0,1)", "ERR ValueError: mean <= 0")]
        [DataRow("wald(1,-1.0)", "ERR ValueError: scale <= 0")]
        [DataRow("beta(-1.0,1)", "ERR ValueError: a <= 0")]
        [DataRow("beta(1,-1.0)", "ERR ValueError: b <= 0")]
        [DataRow("negative_binomial(-1.0,0.5)", "ERR ValueError: n <= 0")]
        [DataRow("poisson(-1.0)", "ERR ValueError: lam < 0 or lam is NaN")]
        [DataRow("triangular(-1.0,1,2)", "1.176512383619456, 0.6227355417665301")]
        [DataRow("triangular(0,-1.0,2)", "ERR ValueError: left > mode")]
        [DataRow("triangular(0,1,-1.0)", "ERR ValueError: mode > right")]
        [DataRow("f(-0.0,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("f(1,-0.0)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(-0.0,1,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("noncentral_f(1,-0.0,1)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(1,1,-0.0)", "ERR ValueError: nonc < 0")]
        [DataRow("chisquare(-0.0)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(-0.0,1)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(1,-0.0)", "ERR ValueError: nonc < 0")]
        [DataRow("standard_t(-0.0)", "ERR ValueError: df <= 0")]
        [DataRow("vonmises(0,-0.0)", "ERR ValueError: kappa < 0")]
        [DataRow("vonmises(-0.0,1)", "1.8378819513682032, 1.5130960447731194")]
        [DataRow("pareto(-0.0)", "ERR ValueError: a <= 0")]
        [DataRow("weibull(-0.0)", "ERR ValueError: a < 0")]
        [DataRow("power(-0.0)", "ERR ValueError: a <= 0")]
        [DataRow("laplace(0,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("laplace(-0.0,1)", "0.7938786426417068, -0.1303856263065431")]
        [DataRow("gumbel(0,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("logistic(0,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("lognormal(0,-0.0)", "ERR ValueError: sigma < 0")]
        [DataRow("lognormal(-0.0,1)", "1.3562412406168636, 0.35346029972713455")]
        [DataRow("rayleigh(-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("wald(-0.0,1)", "ERR ValueError: mean <= 0")]
        [DataRow("wald(1,-0.0)", "ERR ValueError: scale <= 0")]
        [DataRow("beta(-0.0,1)", "ERR ValueError: a <= 0")]
        [DataRow("beta(1,-0.0)", "ERR ValueError: b <= 0")]
        [DataRow("negative_binomial(-0.0,0.5)", "ERR ValueError: n <= 0")]
        [DataRow("poisson(-0.0)", "0, 0")]
        [DataRow("triangular(-0.0,1,2)", "1.3276251767889629, 0.9368868018624793")]
        [DataRow("triangular(0,-0.0,2)", "1.0491184060167393, 0.5018390470340677")]
        [DataRow("triangular(0,1,-0.0)", "ERR ValueError: mode > right")]
        [DataRow("f(0.0,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("f(1,0.0)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(0.0,1,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("noncentral_f(1,0.0,1)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(1,1,0.0)", "0.628371916702715, 0.011735159212395891")]
        [DataRow("chisquare(0.0)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(0.0,1)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(1,0.0)", "1.6090002922630262, 2.5605859356446223")]
        [DataRow("standard_t(0.0)", "ERR ValueError: df <= 0")]
        [DataRow("vonmises(0,0.0)", "1.7213166190998062, -0.3840380893017967")]
        [DataRow("vonmises(0.0,1)", "1.8378819513682032, 1.5130960447731194")]
        [DataRow("pareto(0.0)", "ERR ValueError: a <= 0")]
        [DataRow("weibull(0.0)", "0.0, 0.0")]
        [DataRow("power(0.0)", "ERR ValueError: a <= 0")]
        [DataRow("laplace(0,0.0)", "0.0, 0.0")]
        [DataRow("laplace(0.0,1)", "0.7938786426417068, -0.1303856263065431")]
        [DataRow("gumbel(0,0.0)", "0.0, 0.0")]
        [DataRow("logistic(0,0.0)", "0.0, 0.0")]
        [DataRow("lognormal(0,0.0)", "1.0, 1.0")]
        [DataRow("lognormal(0.0,1)", "1.3562412406168636, 0.35346029972713455")]
        [DataRow("rayleigh(0.0)", "0.0, 0.0")]
        [DataRow("wald(0.0,1)", "ERR ValueError: mean <= 0")]
        [DataRow("wald(1,0.0)", "ERR ValueError: scale <= 0")]
        [DataRow("beta(0.0,1)", "ERR ValueError: a <= 0")]
        [DataRow("beta(1,0.0)", "ERR ValueError: b <= 0")]
        [DataRow("negative_binomial(0.0,0.5)", "ERR ValueError: n <= 0")]
        [DataRow("poisson(0.0)", "0, 0")]
        [DataRow("triangular(0.0,1,2)", "1.3276251767889629, 0.9368868018624793")]
        [DataRow("triangular(0,0.0,2)", "1.0491184060167393, 0.5018390470340677")]
        [DataRow("triangular(0,1,0.0)", "ERR ValueError: mode > right")]
        [DataRow("f(nan,1)", "nan, nan")]
        [DataRow("f(1,nan)", "nan, nan")]
        [DataRow("noncentral_f(nan,1,1)", "nan, nan")]
        [DataRow("noncentral_f(1,nan,1)", "nan, nan")]
        [DataRow("noncentral_f(1,1,nan)", "nan, nan")]
        [DataRow("chisquare(nan)", "nan, nan")]
        [DataRow("noncentral_chisquare(nan,1)", "nan, nan")]
        [DataRow("noncentral_chisquare(1,nan)", "nan, nan")]
        [DataRow("standard_t(nan)", "nan, nan")]
        [DataRow("vonmises(0,nan)", "nan, nan")]
        [DataRow("vonmises(nan,1)", "nan, nan")]
        [DataRow("pareto(nan)", "nan, nan")]
        [DataRow("weibull(nan)", "nan, nan")]
        [DataRow("power(nan)", "nan, nan")]
        [DataRow("laplace(0,nan)", "nan, nan")]
        [DataRow("laplace(nan,1)", "nan, nan")]
        [DataRow("gumbel(0,nan)", "nan, nan")]
        [DataRow("logistic(0,nan)", "nan, nan")]
        [DataRow("lognormal(0,nan)", "nan, nan")]
        [DataRow("lognormal(nan,1)", "nan, nan")]
        [DataRow("rayleigh(nan)", "nan, nan")]
        [DataRow("wald(nan,1)", "nan, nan")]
        [DataRow("wald(1,nan)", "nan, nan")]
        [DataRow("beta(nan,1)", "nan, nan")]
        [DataRow("beta(1,nan)", "nan, nan")]
        [DataRow("negative_binomial(nan,0.5)", "ERR ValueError: n must not be NaN")]
        [DataRow("poisson(nan)", "ERR ValueError: lam < 0 or lam is NaN")]
        [DataRow("triangular(nan,1,2)", "nan, nan")]
        [DataRow("triangular(0,nan,2)", "nan, nan")]
        [DataRow("triangular(0,1,nan)", "nan, nan")]
        [DataRow("f(inf,1)", "nan, nan")]
        [DataRow("f(1,inf)", "nan, nan")]
        [DataRow("noncentral_f(inf,1,1)", "nan, nan")]
        [DataRow("noncentral_f(1,inf,1)", "nan, nan")]
        [DataRow("noncentral_f(1,1,inf)", "144.34988682189768, 0.11937568286989897")]
        [DataRow("chisquare(inf)", "inf, inf")]
        [DataRow("noncentral_chisquare(inf,1)", "inf, inf")]
        [DataRow("noncentral_chisquare(1,inf)", "2.5605859356446223, 0.03282620575464939")]
        [DataRow("standard_t(inf)", "nan, nan")]
        [DataRow("vonmises(0,inf)", "0.0, 0.0")]
        [DataRow("vonmises(inf,1)", "nan, nan")]
        [DataRow("pareto(inf)", "0.0, 0.0")]
        [DataRow("weibull(inf)", "1.0, 1.0")]
        [DataRow("power(inf)", "1.0, 1.0")]
        [DataRow("laplace(0,inf)", "inf, -inf")]
        [DataRow("laplace(inf,1)", "inf, inf")]
        [DataRow("gumbel(0,inf)", "-inf, inf")]
        [DataRow("logistic(0,inf)", "inf, -inf")]
        [DataRow("lognormal(0,inf)", "inf, 0.0")]
        [DataRow("lognormal(inf,1)", "inf, inf")]
        [DataRow("rayleigh(inf)", "inf, inf")]
        [DataRow("wald(inf,1)", "nan, nan")]
        [DataRow("wald(1,inf)", "1.0, 1.0")]
        [DataRow("beta(inf,1)", "nan, nan")]
        [DataRow("beta(1,inf)", "0.0, 0.0")]
        [DataRow("negative_binomial(inf,0.5)", "ERR ValueError: n too large or p too small, see Generator.negative_binomial Notes")]
        [DataRow("poisson(inf)", "ERR ValueError: lam value too large")]
        [DataRow("triangular(inf,1,2)", "ERR ValueError: left > mode")]
        [DataRow("triangular(0,inf,2)", "ERR ValueError: mode > right")]
        [DataRow("triangular(0,1,inf)", "nan, nan")]
        [DataRow("binomial(10,-0.5)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("negative_binomial(1,-0.5)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("geometric(-0.5)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(-0.5)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("binomial(10,-0.0)", "0, 0")]
        [DataRow("negative_binomial(1,-0.0)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("geometric(-0.0)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(-0.0)", "1, 1")]
        [DataRow("binomial(10,0.0)", "0, 0")]
        [DataRow("negative_binomial(1,0.0)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("geometric(0.0)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(0.0)", "1, 1")]
        [DataRow("binomial(10,0.5)", "6, 5")]
        [DataRow("negative_binomial(1,0.5)", "3, 2")]
        [DataRow("geometric(0.5)", "3, 1")]
        [DataRow("logseries(0.5)", "1, 2")]
        [DataRow("binomial(10,1.0)", "10, 10")]
        [DataRow("negative_binomial(1,1.0)", "0, 0")]
        [DataRow("geometric(1.0)", "1, 1")]
        [DataRow("logseries(1.0)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("binomial(10,1.5)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("negative_binomial(1,1.5)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("geometric(1.5)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(1.5)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("binomial(10,nan)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("negative_binomial(1,nan)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("geometric(nan)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(nan)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("zipf(0.5)", "ERR ValueError: a <= 1 or a is NaN")]
        [DataRow("zipf(1.0)", "ERR ValueError: a <= 1 or a is NaN")]
        [DataRow("zipf(1.0000001)", "476413292096858, 19195960335915580")]
        [DataRow("zipf(nan)", "ERR ValueError: a <= 1 or a is NaN")]
        [DataRow("zipf(inf)", "1, 1")]
        [DataRow("zipf(1025.0)", "1, 1")]
        [DataRow("triangular(1,0,2)", "ERR ValueError: left > mode")]
        [DataRow("triangular(0,3,2)", "ERR ValueError: mode > right")]
        [DataRow("triangular(1,1,1)", "ERR ValueError: left == right")]
        [DataRow("binomial(-1,0.5)", "ERR ValueError: n < 0")]
        [DataRow("poisson(1e19)", "ERR ValueError: lam value too large")]
        [DataRow("poisson(9.2e18)", "9200000002564161536, 9200000003718269952")]
        [DataRow("negative_binomial(1e20,0.5)", "ERR ValueError: n too large or p too small, see Generator.negative_binomial Notes")]
        [DataRow("negative_binomial(10,1e-18)", "ERR ValueError: n too large or p too small, see Generator.negative_binomial Notes")]
        [DataRow("negative_binomial(10,1e-17)", "1064535845977195520, 478102843959582144")]
        [DataRow("hypergeometric(-1,5,3)", "ERR ValueError: ngood < 0")]
        [DataRow("hypergeometric(5,-1,3)", "ERR ValueError: nbad < 0")]
        [DataRow("hypergeometric(5,5,-1)", "ERR ValueError: nsample < 0")]
        [DataRow("hypergeometric(5,5,11)", "ERR ValueError: ngood + nbad < nsample")]
        [DataRow("hypergeometric(10**9,5,3)", "ERR ValueError: both ngood and nbad must be less than 1000000000")]
        [DataRow("hypergeometric(5,10**9,3)", "ERR ValueError: both ngood and nbad must be less than 1000000000")]
        [DataRow("hypergeometric(10**9-1,5,3)", "3, 3")]
        [DataRow("hypergeometric(-1,5,11)", "ERR ValueError: ngood + nbad < nsample")]
        public void Validation_Seed42_MatchesNumPy(string probe, string expected)
        {
            int paren = probe.IndexOf('(');
            string method = probe.Substring(0, paren);
            string[] args = probe.Substring(paren + 1, probe.Length - paren - 2).Split(',');
            AssertProbe(() => Draw(np.random.default_rng(42), method, args, new Shape(2)), expected, probe);
        }

        /// <summary>
        ///     The multivariate samplers' validations in NumPy's order: multinomial's empty / out-of-range / over-summed
        ///     probabilities (incl. the float32 cast text), negative counts on both paths, the broadcast-vs-size mismatch;
        ///     dirichlet's <c>alpha &lt; 0</c> and NaN/empty alphas; multivariate_hypergeometric's method, nsample, colors
        ///     (shape, dtype, sign) and <c>10**9</c> marginals cap.
        /// </summary>
        /// <param name="probe">The call, as NumPy's probe named it (see <see cref="MultivariateCalls"/>).</param>
        /// <param name="expected">NumPy 2.4.2's first two values, or <c>ERR Type: message</c>.</param>
        [TestMethod]
        [DataRow("multinomial(-1,[.5,.5])", "ERR ValueError: n < 0")]
        [DataRow("multinomial(10,[.6,.6])", "5, 5")]
        [DataRow("multinomial(10,[-0.1,1.1])", "ERR ValueError: pvals < 0, pvals > 1 or pvals contains NaNs")]
        [DataRow("multinomial(10,[nan,1])", "ERR ValueError: pvals < 0, pvals > 1 or pvals contains NaNs")]
        [DataRow("multinomial(10,[])", "ERR ValueError: pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0.")]
        [DataRow("multinomial(10,[[.6,.6],[.1,.9]])", "5, 5")]
        [DataRow("multinomial([10,20],[.5,.5],size=3)", "ERR ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 2 with shape (3,).")]
        [DataRow("multinomial([10,-1],[.5,.5])", "ERR ValueError: n < 0")]
        [DataRow("multinomial(10,f32 sum>1)", "ERR ValueError: sum(pvals[:-1].astype(np.float64)) > 1.0. The pvals array is cast to 64-bit floating point prior to checking the sum. Precision changes when casting may cause problems even if the sum of the original pvals is valid.")]
        [DataRow("dirichlet([-1,1])", "ERR ValueError: alpha < 0")]
        [DataRow("dirichlet([nan,1])", "nan, nan")]
        [DataRow("dirichlet([])", "")]
        [DataRow("multivariate_hypergeometric([3,2],6)", "ERR ValueError: nsample > sum(colors)")]
        [DataRow("multivariate_hypergeometric([3,-2],1)", "ERR ValueError: colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807.")]
        [DataRow("multivariate_hypergeometric([3,2],-1)", "ERR ValueError: nsample must be nonnegative.")]
        [DataRow("multivariate_hypergeometric([3,2],1,method=x)", "ERR ValueError: method must be \"count\" or \"marginals\".")]
        [DataRow("multivariate_hypergeometric([10**9,1],1)", "ERR ValueError: When method is \"marginals\", sum(colors) must be less than 1000000000.")]
        [DataRow("multivariate_hypergeometric([10**9,1],1,count)", "1, 0")]
        [DataRow("multivariate_hypergeometric([],0)", "")]
        [DataRow("multivariate_hypergeometric([[1,2]],1)", "ERR ValueError: colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807.")]
        [DataRow("multivariate_hypergeometric([1.5,2],1)", "ERR ValueError: colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807.")]
        public void MultivariateValidation_Seed42_MatchesNumPy(string probe, string expected)
            => AssertProbe(() => MultivariateCalls[probe](np.random.default_rng(42)), expected, probe);

        // ---------------------------------------------------------------- long streams, seed 2024

        /// <summary>The 20000-draw calls behind <see cref="LongStream_Seed2024_MatchesNumPyDigest"/>, keyed as NumPy's generator named them.</summary>
        private static readonly Dictionary<string, Func<Generator, NDArray>> LongStreamCalls = new()
        {
            ["beta 2,3"] = r => r.beta(2.0, 3.0, 20000),
            ["beta 0.5,0.5"] = r => r.beta(0.5, 0.5, 20000),
            ["beta 1e-3,1e-3"] = r => r.beta(1e-3, 1e-3, 20000),
            ["beta 1e-105,1e-105"] = r => r.beta(1e-105, 1e-105, 20000),
            ["beta 0.5,2"] = r => r.beta(0.5, 2.0, 20000),
            ["chisquare 3"] = r => r.chisquare(3.0, 20000),
            ["chisquare 0.5"] = r => r.chisquare(0.5, 20000),
            ["f 5,7"] = r => r.f(5.0, 7.0, 20000),
            ["f 0.5,0.5"] = r => r.f(0.5, 0.5, 20000),
            ["noncentral_chisquare 3,1.5"] = r => r.noncentral_chisquare(3.0, 1.5, 20000),
            ["noncentral_chisquare 0.5,1.5"] = r => r.noncentral_chisquare(0.5, 1.5, 20000),
            ["noncentral_chisquare 0.5,30"] = r => r.noncentral_chisquare(0.5, 30.0, 20000),
            ["noncentral_f 5,7,1.5"] = r => r.noncentral_f(5.0, 7.0, 1.5, 20000),
            ["noncentral_f 0.5,7,1.5"] = r => r.noncentral_f(0.5, 7.0, 1.5, 20000),
            ["standard_cauchy"] = r => r.standard_cauchy(20000),
            ["standard_t 3.5"] = r => r.standard_t(3.5, 20000),
            ["standard_t 0.5"] = r => r.standard_t(0.5, 20000),
            ["vonmises 0.5,2"] = r => r.vonmises(0.5, 2.0, 20000),
            ["vonmises 0,1e-9"] = r => r.vonmises(0.0, 1e-9, 20000),
            ["vonmises 1,1e-6"] = r => r.vonmises(1.0, 1e-6, 20000),
            ["vonmises -2,1e7"] = r => r.vonmises(-2.0, 1e7, 20000),
            ["vonmises 3,50"] = r => r.vonmises(3.0, 50.0, 20000),
            ["weibull 1.79"] = r => r.weibull(1.79, 20000),
            ["weibull 0.3"] = r => r.weibull(0.3, 20000),
            ["laplace 0,1"] = r => r.laplace(0.0, 1.0, 20000),
            ["gumbel 0.5,2"] = r => r.gumbel(0.5, 2.0, 20000),
            ["logistic 0,1"] = r => r.logistic(0.0, 1.0, 20000),
            ["lognormal 1,0.5"] = r => r.lognormal(1.0, 0.5, 20000),
            ["rayleigh 1.5"] = r => r.rayleigh(1.5, 20000),
            ["wald 3,2"] = r => r.wald(3.0, 2.0, 20000),
            ["wald 0.5,10"] = r => r.wald(0.5, 10.0, 20000),
            ["triangular 0,3,10"] = r => r.triangular(0.0, 3.0, 10.0, 20000),
            ["triangular 0,0,1"] = r => r.triangular(0.0, 0.0, 1.0, 20000),
            ["binomial 10,0.35"] = r => r.binomial(10, 0.35, 20000),
            ["binomial 100,0.4"] = r => r.binomial(100, 0.4, 20000),
            ["binomial 1000,0.7"] = r => r.binomial(1000, 0.7, 20000),
            ["binomial 20,0.9"] = r => r.binomial(20, 0.9, 20000),
            ["negative_binomial 5,0.4"] = r => r.negative_binomial(5.0, 0.4, 20000),
            ["negative_binomial 0.5,0.5"] = r => r.negative_binomial(0.5, 0.5, 20000),
            ["negative_binomial 1000,0.01"] = r => r.negative_binomial(1000.0, 0.01, 20000),
            ["poisson 3.5"] = r => r.poisson(3.5, 20000),
            ["poisson 100"] = r => r.poisson(100.0, 20000),
            ["poisson 1e6"] = r => r.poisson(1e6, 20000),
            ["zipf 3"] = r => r.zipf(3.0, 20000),
            ["zipf 1.5"] = r => r.zipf(1.5, 20000),
            ["zipf 1.05"] = r => r.zipf(1.05, 20000),
            ["geometric 0.35"] = r => r.geometric(0.35, 20000),
            ["geometric 0.1"] = r => r.geometric(0.1, 20000),
            ["geometric 1e-5"] = r => r.geometric(1e-5, 20000),
            ["hypergeometric 10,7,8"] = r => r.hypergeometric(10, 7, 8, 20000),
            ["hypergeometric 100,200,50"] = r => r.hypergeometric(100, 200, 50, 20000),
            ["hypergeometric 300,100,250"] = r => r.hypergeometric(300, 100, 250, 20000),
            ["hypergeometric 15,15,20"] = r => r.hypergeometric(15, 15, 20, 20000),
            ["hypergeometric 1000,1000,1995"] = r => r.hypergeometric(1000, 1000, 1995, 20000),
            ["logseries 0.6"] = r => r.logseries(0.6, 20000),
            ["logseries 0.99"] = r => r.logseries(0.99, 20000),
            ["logseries 0.1"] = r => r.logseries(0.1, 20000),
            ["multinomial 20,[.2,.3,.5]x5000"] = r => r.multinomial(20, new[] { 0.2, 0.3, 0.5 }, new Shape(5000)),
            ["multinomial 1000,[.25,.25,.5]x2000"] = r => r.multinomial(1000, new[] { 0.25, 0.25, 0.5 }, new Shape(2000)),
            ["multinomial vector"] = r => r.multinomial(np.arange(1L, 1001L).reshape(1000, 1), np.array(new[,] { { 0.3, 0.7 }, { 0.9, 0.1 } })),
            ["dirichlet 2,3,5 x5000"] = r => r.dirichlet(new[] { 2.0, 3.0, 5.0 }, new Shape(5000)),
            ["dirichlet 0.05x3 x5000"] = r => r.dirichlet(new[] { 0.05, 0.05, 0.05 }, new Shape(5000)),
            ["dirichlet 0.5,0.05 x5000"] = r => r.dirichlet(new[] { 0.5, 0.05 }, new Shape(5000)),
            ["mvhg count x3000"] = r => r.multivariate_hypergeometric(new long[] { 30, 20, 50 }, 40, new Shape(3000), method: "count"),
            ["mvhg marginals x3000"] = r => r.multivariate_hypergeometric(new long[] { 30, 20, 50 }, 40, new Shape(3000)),
            ["mvhg marginals big x2000"] = r => r.multivariate_hypergeometric(new long[] { 300, 2000, 5000, 40 }, 3000, new Shape(2000)),
        };

        /// <summary>
        ///     Long streams reach what 8 draws cannot — rejection loops' rare retries, BTPE's squeeze/acceptance tails, HRUA's
        ///     second test, zipf's retry on out-of-range candidates, the read-ahead buffers' refill boundaries and the
        ///     binomial setup memo's hits — so each sampler's 20000-draw stream from <c>np.random.default_rng(2024)</c> must
        ///     hash to NumPy's SHA-256 exactly (the whole stream, bit for bit). pareto/power are absent: their in-band
        ///     <c>expm1</c> divergence is pinned by <see cref="ParetoPower_InBandExpm1_WithinDocumentedUlp"/>.
        /// </summary>
        /// <param name="probe">The stream, as NumPy's generator named it (see <see cref="LongStreamCalls"/>).</param>
        /// <param name="sha256">NumPy 2.4.2's <c>hashlib.sha256(a.tobytes()).hexdigest()</c>.</param>
        [TestMethod]
        [DataRow("beta 2,3", "e3de8320dc86c09bd2a0ffa75629b65552b84466a1ebfa95c7f9c7745228fdbb")]
        [DataRow("beta 0.5,0.5", "dceaf37535c72162292f4699f3d849a8fc9308d94b7b7df585c459d807cdbcc4")]
        [DataRow("beta 1e-3,1e-3", "b0e81821ecdc15b3318120269f2f076739ec13a999b572a75694d941079895dc")]
        [DataRow("beta 1e-105,1e-105", "95e2f698db32f17f5cf55ab4ecd831bd81ebb631e6592a6a3de21f6fbbb5061d")]
        [DataRow("beta 0.5,2", "e5147772af8dfac02605d27d6eab1d8c560c36d7a9012113255cf6b59c267b84")]
        [DataRow("chisquare 3", "173c143fa3bce27789a7f135c0c5f98f59cb2db0229fd305ec40faa6a1e0400a")]
        [DataRow("chisquare 0.5", "4b7b9e9a866e763fc2fc5bcd124c6c8a047d4ccd91438adcdf57ee81dc4c9d85")]
        [DataRow("f 5,7", "21f563aeb55294b431283e43225f0e221907735446102941fc70f25ffc8450fb")]
        [DataRow("f 0.5,0.5", "80ccc5017b237022fdb6a538812b4b276295972bbceecc4bc89312bc1a951554")]
        [DataRow("noncentral_chisquare 3,1.5", "703278cb16077daa8caef14fedfc3a70d4418872287d8f89961dcb0ea0e8bc82")]
        [DataRow("noncentral_chisquare 0.5,1.5", "11867dc6edefc0ed63529c5d0d912b00818bc098a0b3b13bd7689c5c330b0b90")]
        [DataRow("noncentral_chisquare 0.5,30", "eafe477100aecbbd0897d2ee72a510b0c85c17ff9f6b5d8b33657a0e2655b86e")]
        [DataRow("noncentral_f 5,7,1.5", "668e7aa449d5843b7bd4267f4899f71161050ad899fd9e67e2706a0ec44fa819")]
        [DataRow("noncentral_f 0.5,7,1.5", "a363565419b396cf5a7fd04fc7cd4959b6f3e24d75169bac0f58cb2136743fe2")]
        [DataRow("standard_cauchy", "7bfc73a710943c52e1e53995fa36a1a1351b623b11ce1c785bce75d92ddd89fc")]
        [DataRow("standard_t 3.5", "af59aee405dfe9ef115960cc0dbca7dda53a3bd8456f93462b58c336297912fd")]
        [DataRow("standard_t 0.5", "bc41678459abd980a21e9e78506953302078dc4215d93de66cddfefda1236f65")]
        [DataRow("vonmises 0.5,2", "ad3da9feb1af5a007dae117cf72dd88db7293306953800b4dacb862bfaec51c4")]
        [DataRow("vonmises 0,1e-9", "044a19b856933fe9c906b5d75cb99cb679989e4cb654fafacbdcf71a8d16c329")]
        [DataRow("vonmises 1,1e-6", "8e9e98611bffdde970ae7db072067f3440e9145e179786db794e7445ec2e2ae6")]
        [DataRow("vonmises -2,1e7", "afcfb2a8cd5152497c1b43eb06ff5da37b47ca105f55bdb8b74a16ad5efc6899")]
        [DataRow("vonmises 3,50", "f1c36330cda7259a54ede81c54454c6fdf1aed5308c77c34ee5d9870e5cef4d4")]
        [DataRow("weibull 1.79", "5822a8e695e545b138d42019e215534c98ba9142becb68166ca4867e2edcb257")]
        [DataRow("weibull 0.3", "9258f43cc373c4d9bc074cb1ad517f66863ebd9ab3ccd2eeefb4cd719a997d29")]
        [DataRow("laplace 0,1", "6c29af3ccbd2c65345a42634737c3131374de9b9b293585ccb988226c79fc734")]
        [DataRow("gumbel 0.5,2", "c01061d9e3adf6f98ac94540c36ba9d1f3528d8466f4a271fe073b2393947371")]
        [DataRow("logistic 0,1", "12c9d8127b68a7bb8ab3189f972dc4a253c0a45050f55a254426aebc5b8601e3")]
        [DataRow("lognormal 1,0.5", "2da0d64f73699a94225bdd3e82a0b39a06651437791b373f7a4bb8db2247a0fc")]
        [DataRow("rayleigh 1.5", "f43ad38921472d4bc494f192db7941c26159730048a48c13257e6fd72bd51181")]
        [DataRow("wald 3,2", "df6db4c45cc06289546fafdfe0987c9d5caebb7fc62c1e9191097050c5192151")]
        [DataRow("wald 0.5,10", "5f78ee6494612ea83aa7e5dc19a6c7d78c6c344fa16378fdf3d88f0b9d41c376")]
        [DataRow("triangular 0,3,10", "7394e0e2cf0a40308af5a7ac31de979d61d6248f6c43c898977856a27ba70842")]
        [DataRow("triangular 0,0,1", "45e288f9e55533faa05dbce254128cca2ad7f826e183188d49c6d8e4fe7da2b2")]
        [DataRow("binomial 10,0.35", "bd97bd99a6fb4c00285f7b05433d92124003eace49242d2e4dfaff221775279a")]
        [DataRow("binomial 100,0.4", "9ad44487142f8bc09cf13a0e416279a2cd0633adfd2cb147411aaf96d9f8e9c0")]
        [DataRow("binomial 1000,0.7", "1637b6102ef861ca1f1800708b21bf465cdd39b452b69d27a1940619198360cd")]
        [DataRow("binomial 20,0.9", "407b3a6d012e71bc487cf74801a67a15ba6faed99be9ae83af9762c17e9695e6")]
        [DataRow("negative_binomial 5,0.4", "2b08d8f7e570c03639421f60ad908b65b9df4c5428ba7a94be4d0d4d73cfeb61")]
        [DataRow("negative_binomial 0.5,0.5", "38cc6ab37d40b0ad8450de00f03de76c1fbe614df0243e395f0cbdf83650c811")]
        [DataRow("negative_binomial 1000,0.01", "08088bfe412e024ebb59e212de091207a41c114932e053a12a7b9bfa5c905b95")]
        [DataRow("poisson 3.5", "182ba99942dbff364d7ba6fef89532f96f7c746117a85603b5fdeb4a47e45eb4")]
        [DataRow("poisson 100", "9c6aa2dcae657befca648800f287483d2d73d38a0f4623d12bdc5bda08677793")]
        [DataRow("poisson 1e6", "5e9ebdbb2b1c2c8e70af1ed8eb96dca2167461361a17a651737e5527cd6fe5e0")]
        [DataRow("zipf 3", "e30e893b8d8cb55d3d486ca2b2e92f14bd5176bcec8066d8859142c32e5831a7")]
        [DataRow("zipf 1.5", "a88d2c0463a27db7567d0a20e2662bdfc981e6d06664bc919d71a195d57db145")]
        [DataRow("zipf 1.05", "4f2d7fbe27c1e0539ca1e4e3fe64b56540dbe4e5cc713148631eb0cf16c33f79")]
        [DataRow("geometric 0.35", "01cc2414c764962b00e2f1a0ef889cc19da2f1ef53b6efde8b6b57142515018b")]
        [DataRow("geometric 0.1", "e2da8b3da8e47d5b81f562f992827fa171ea6f747ae1b62920e666a528eb8e6f")]
        [DataRow("geometric 1e-5", "1b1f8d8100bcc4be2aaa22314b9fa627e3576392665dbf573a6c9781f21d17e0")]
        [DataRow("hypergeometric 10,7,8", "8ec00471993776b42b2a8774d857995a1766e8e88e0e5967d6a02695a29fc85d")]
        [DataRow("hypergeometric 100,200,50", "49bbaa81e04a071a176e75dcb310c9af9e5d6df6d54c6c20690eb4ce14c2980b")]
        [DataRow("hypergeometric 300,100,250", "833cec5751f4f1312e4285b34b5e7da2bff426bff0ac17b5719fe02981b8a4c2")]
        [DataRow("hypergeometric 15,15,20", "ef815bf2e44841123895b3ce0e4a983ae0665a8ef25dea89ca70a9360085bf67")]
        [DataRow("hypergeometric 1000,1000,1995", "2a8089a62e2edb48e732973406194821d9dd8a756d988aa2128235cdab395b9a")]
        [DataRow("logseries 0.6", "11027f8af68283e9af544b34cf534896ed920ccb3d3bbdeafbbe588a11db338f")]
        [DataRow("logseries 0.99", "e44439a4bcd025550aa7070d4956f510a600e22e4469737e7aafff72b432331f")]
        [DataRow("logseries 0.1", "0e5f95b707b3045af13879ed70fa633785119e7a27cb13328ad31fcaa5ea8102")]
        [DataRow("multinomial 20,[.2,.3,.5]x5000", "bdbd340dbc53632808162c63d379eadf246263deed30266295daf5425816b653")]
        [DataRow("multinomial 1000,[.25,.25,.5]x2000", "ec88c0320417e261e5137c67fd279b5bb6a47835d2f90db48c2c66ff0a978a7b")]
        [DataRow("multinomial vector", "172d3d6bb2fa398abe50706e971b7187183ee3d13554485f7f2a131fdf78bd9e")]
        [DataRow("dirichlet 2,3,5 x5000", "9da6211b6d56819fe12c4352e862038ee93aa4f41974ea2c8077edce815755cc")]
        [DataRow("dirichlet 0.05x3 x5000", "4a558cff3b9bcb19498576d729ab322e106bb4d7bbe377651e6c435ce1d7d497")]
        [DataRow("dirichlet 0.5,0.05 x5000", "95bb9a0a70abf0c305606d8ac4f901126dbc7b02e27d38a5e4e24bfba539dcaf")]
        [DataRow("mvhg count x3000", "d0d868a42bc2d58278ed0487a2569868ffa5e5eadaf7c38df09669c6be42e9d1")]
        [DataRow("mvhg marginals x3000", "3d586a13300031e93fc2928829d8430375a234fdd1df58f5c77be03708d32d01")]
        [DataRow("mvhg marginals big x2000", "23c10042df03bed8b18e02a60abce2fa7f0b25f46d383165c796fc0aa6438ec4")]
        public void LongStream_Seed2024_MatchesNumPyDigest(string probe, string sha256)
            => Assert.AreEqual(sha256, Sha256(LongStreamCalls[probe](np.random.default_rng(2024))), probe);

        // ---------------------------------------------------------------- documented divergence: pareto / power

        /// <summary>
        ///     pareto (<c>expm1(E / a)</c>) and power (<c>pow(-expm1(-E), 1/a)</c>) are NumPy's modern samplers over the
        ///     C runtime's <c>expm1</c>. Outside <c>[-ln 2, ln 1.5]</c> ucrtbase computes <c>exp(x) - 1</c>, which
        ///     <c>Generator.Expm1</c> reproduces — every such value must be BIT-identical. Inside, ucrtbase's approximation is
        ///     closed and Sun's <c>s_expm1</c> stands in: pareto may differ by at most 2 ulp, power by
        ///     <c>ceil(2/a) + 3</c> ulp (its exponent <c>1/a</c> scales the relative difference). The draws themselves
        ///     (ziggurat exponentials) are NumPy's, which is what decides in/out of the band here.
        /// </summary>
        /// <param name="probe"><c>pareto a</c> or <c>power a</c>.</param>
        /// <param name="expected">NumPy 2.4.2's first 32 values from <c>default_rng(2024)</c>.</param>
        [TestMethod]
        [DataRow("pareto 3", "0.328954761496851, 0.03891758104574866, 0.05802595586223852, 0.540205668063106, 1.1751200163825422, 0.08109359988239144, 0.18894972143905178, 0.15578323664999272, 0.029209177118014937, 0.4053064020292717, 0.1384413279959382, 0.15663447014356963, 0.271427684393542, 0.0005009434507712861, 0.04011067417482427, 3.664292024521825, 1.8838831648672647, 0.17229675883409526, 0.16642179850492086, 0.0237102710613797, 1.785212995325753, 0.7248841342604762, 0.44976347662898264, 0.0999018848220433, 0.0949873082559362, 2.209863095013332, 0.1448783914654981, 0.14110865263517286, 0.041185494848943995, 0.39277178269914037, 0.04933603574648409, 0.309706542036959")]
        [DataRow("pareto 0.5", "4.5088530184549285, 0.25743798064817613, 0.402742426388861, 12.349724516866063, 104.90082657161403, 0.5965399043625321, 1.8247559806288427, 1.3837370533716271, 0.18856216135185533, 6.7024014162318135, 1.1770275218892245, 1.3942901910207888, 3.224253589889984, 0.003009427384864043, 0.266127145802603, 10296.087397627922, 574.2623759271731, 1.5955260109661569, 1.5184524980350993, 0.15096565381688656, 465.8210994356355, 25.336591344699148, 8.285021787837534, 0.7706131185376681, 0.7236715457855996, 1092.75249654331, 1.251936557446732, 1.2078114988298418, 0.27399773225694296, 6.299275518315215, 0.3350192396539223, 4.047124072318939")]
        [DataRow("power 2.5", "0.8008422919524482, 0.4108907438221425, 0.47520887721630584, 0.8799298854636198, 0.9599345265543155, 0.5342006049648794, 0.6966046640272612, 0.6588208202780795, 0.36906367840175586, 0.8363449407799015, 0.6357383647673828, 0.6598884148818502, 0.7659496087740593, 0.07423180069030175, 0.41550704423826657, 0.9960464100860292, 0.9831092828187972, 0.6785619393866835, 0.6717653898362655, 0.3409677136473687, 0.9812230604262141, 0.916956643651714, 0.8529073752166026, 0.5729536053502114, 0.5634611159552557, 0.9877936317832612, 0.6446156646488358, 0.6394636425376109, 0.41958315248580585, 0.8311874900458582, 0.4482463882595543, 0.7900950472673163")]
        [DataRow("power 0.3", "0.1571176587695895, 0.0006040223144981062, 0.0020294165978010695, 0.34440324448642395, 0.7112355195300052, 0.005381084302049851, 0.049153353058029814, 0.030883579358340234, 0.00024689419742007405, 0.2255339628501521, 0.02294317428237834, 0.03130311324138301, 0.1083931046256393, 3.8748060018432177e-10, 0.000662958878396991, 0.9675270619837967, 0.86765746409907, 0.03949836660368194, 0.036320086364846224, 0.00012762736768940936, 0.8538820537944984, 0.48555676824208965, 0.2655719782309033, 0.009645590162993036, 0.008392015143894345, 0.902717475210727, 0.025753748974294312, 0.024087892809751238, 0.0007191459082591526, 0.21420272166329624, 0.00124730689869246, 0.14038730224289295")]
        public void ParetoPower_InBandExpm1_WithinDocumentedUlp(string probe, string expected)
        {
            const double bandLow = -0.6931471805599453, bandHigh = 0.4054651081081644;
            string method = probe.Split(' ')[0];
            double a = ParseDouble(probe.Split(' ')[1]);
            var exp = expected.Split(',').Select(ParseDouble).ToArray();
            var ours = (method == "pareto" ? np.random.default_rng(2024).pareto(a, exp.Length)
                                           : np.random.default_rng(2024).power(a, exp.Length)).ToArray<double>();
            // The exponentials each value consumed (bit-exact with NumPy's; pinned by the Generator's core tests).
            var e = np.random.default_rng(2024).standard_exponential(exp.Length).ToArray<double>();
            long bound = method == "pareto" ? 2 : (long)System.Math.Ceiling(2.0 / a) + 3;
            int exactOutside = 0, inBand = 0;
            for (int i = 0; i < exp.Length; i++)
            {
                double x = method == "pareto" ? e[i] / a : -e[i];
                long ulps = System.Math.Abs(BitConverter.DoubleToInt64Bits(ours[i]) - BitConverter.DoubleToInt64Bits(exp[i]));
                if (x < bandLow || x > bandHigh)
                {
                    Assert.AreEqual(0, ulps, $"{probe}[{i}]: expm1({x:R}) is out of band, so it must be bit-exact");
                    exactOutside++;
                }
                else
                {
                    Assert.IsTrue(ulps <= bound, $"{probe}[{i}]: {ulps} ulp from NumPy (bound {bound}) at expm1({x:R})");
                    inBand++;
                }
            }
            // Non-vacuous: the literals must exercise both sides of the band.
            Assert.IsTrue(exactOutside > 0 && inBand > 0, $"{probe}: {exactOutside} out-of-band, {inBand} in-band values");
        }

        // ---------------------------------------------------------------- multivariate_normal

        /// <summary>The <c>multivariate_normal</c> probe calls, keyed as NumPy's generator named them (seed 42).</summary>
        private static readonly Dictionary<string, Func<Generator, NDArray>> MultivariateNormalCalls = BuildMultivariateNormalCalls();

        /// <summary>Builds <see cref="MultivariateNormalCalls"/>: each covariance × each method, plus the validation cases.</summary>
        /// <returns>The call table.</returns>
        private static Dictionary<string, Func<Generator, NDArray>> BuildMultivariateNormalCalls()
        {
            var i2 = new double[,] { { 1, 0 }, { 0, 1 } };
            var c2 = new double[,] { { 1, 0.5 }, { 0.5, 2 } };
            var c3 = new double[,] { { 2.0, 0.3, 0.1 }, { 0.3, 1.0, 0.2 }, { 0.1, 0.2, 0.5 } };
            var sing = new double[,] { { 1, 1 }, { 1, 1 } };
            var indef = new double[,] { { 1, 2 }, { 2, 1 } };
            var d = new Dictionary<string, Func<Generator, NDArray>>();
            foreach (var m in new[] { "svd", "eigh", "cholesky" })
            {
                d[$"identity {m}"] = r => r.multivariate_normal(new[] { 0.0, 0.0 }, i2, new Shape(3), method: m);
                d[$"c2 {m}"] = r => r.multivariate_normal(new[] { 0.0, 1.0 }, c2, new Shape(2), method: m);
                d[$"c3 {m} size(2,)"] = r => r.multivariate_normal(new[] { 1.0, 2.0, 3.0 }, c3, new Shape(2), method: m);
                d[$"c3 {m} size(2,2)"] = r => r.multivariate_normal(new[] { 1.0, 2.0, 3.0 }, c3, new Shape(2, 2), method: m);
                d[$"c3 {m} size None"] = r => r.multivariate_normal(new[] { 1.0, 2.0, 3.0 }, c3, method: m);
                d[$"singular {m}"] = r => r.multivariate_normal(new[] { 0.0, 0.0 }, sing, new Shape(2), method: m);
                d[$"indefinite {m} warn"] = r => r.multivariate_normal(new[] { 0.0, 0.0 }, indef, new Shape(2), method: m);
                d[$"indefinite {m} ignore"] = r => r.multivariate_normal(new[] { 0.0, 0.0 }, indef, new Shape(2), check_valid: "ignore", method: m);
                d[$"indefinite {m} raise"] = r => r.multivariate_normal(new[] { 0.0, 0.0 }, indef, new Shape(2), check_valid: "raise", method: m);
                d[$"bad check_valid {m}"] = r => r.multivariate_normal(new[] { 0.0, 0.0 }, i2, new Shape(2), check_valid: "loud", method: m);
                d[$"int mean {m}"] = r => r.multivariate_normal(np.array(new long[] { 1, 2 }), np.array(new long[,] { { 2, 1 }, { 1, 2 } }), new Shape(2), method: m);
                d[$"f32 {m}"] = r => r.multivariate_normal(np.array(new float[] { 1, 2 }), np.array(new float[,] { { 2, 1 }, { 1, 2 } }), new Shape(2), method: m);
            }
            d["bad method"] = r => r.multivariate_normal(new[] { 0.0, 0.0 }, i2, new Shape(2), method: "qr");
            d["mean 2d"] = r => r.multivariate_normal(np.zeros(new Shape(2, 1)), np.array(i2));
            d["cov not square"] = r => r.multivariate_normal(np.zeros(2), np.zeros(new Shape(2, 3)));
            d["length mismatch"] = r => r.multivariate_normal(np.zeros(3), np.array(i2));
            d["complex mean"] = r => r.multivariate_normal(np.zeros(new Shape(2), NPTypeCode.Complex), np.array(i2));
            d["empty"] = r => r.multivariate_normal(np.zeros(0), np.zeros(new Shape(0, 0)), new Shape(2));
            // The stream advances past the normals before check_valid / the factorization can fail.
            d["after raise"] = r =>
            {
                Assert.ThrowsException<ValueError>(() => r.multivariate_normal(new[] { 0.0, 0.0 }, indef, new Shape(2), check_valid: "raise"));
                return r.random(new Shape(1));
            };
            d["after cholesky failure"] = r =>
            {
                Assert.ThrowsException<LinAlgError>(() => r.multivariate_normal(new[] { 0.0, 0.0 }, indef, new Shape(2), method: "cholesky"));
                return r.random(new Shape(1));
            };
            return d;
        }

        /// <summary>
        ///     With the LAPACK backend installed, <c>svd</c>/<c>eigh</c>/<c>cholesky</c> are NumPy's own
        ///     <c>gesdd</c>/<c>syevd</c>/<c>potrf</c> and the transform NumPy's <c>gemm</c>, so every draw is byte-identical to
        ///     <c>default_rng(42).multivariate_normal</c> — PSD, singular and indefinite covariances under every
        ///     <c>check_valid</c>, integer/float32 inputs, and the validation order (the normals are drawn first, so a later
        ///     failure has already advanced the stream).
        /// </summary>
        /// <param name="probe">The call, as NumPy's generator named it (see <see cref="MultivariateNormalCalls"/>).</param>
        /// <param name="expected">NumPy 2.4.2's values in C order, or <c>ERR Type: message</c>.</param>
        [TestMethod]
        [DataRow("identity svd", "0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139, -1.9510351886538364, -1.302179506862318")]
        [DataRow("c2 svd", "-0.6823179059679519, 1.7726219956159874, 1.2004210980718155, 1.7095233938671255")]
        [DataRow("c3 svd size(2,)", "0.2781853118087412, 2.637609032475059, 3.749286633008141, -0.8709623927900345, 3.6383514183201138, 2.6795286088545063")]
        [DataRow("c3 svd size(2,2)", "0.2781853118087412, 2.637609032475059, 3.749286633008141, -0.8709623927900345, 3.6383514183201138, 2.6795286088545063, 0.7312520150650741, 2.2331240573456426, 3.0715425492640787, 2.4376702327446305, 1.3930169789850748, 3.3213038502097505")]
        [DataRow("c3 svd size None", "0.2781853118087412, 2.637609032475059, 3.749286633008141")]
        [DataRow("singular svd", "-0.30471707975443124, -0.30471707975443135, -0.750451195806457, -0.7504511958064572")]
        [DataRow("indefinite svd warn", "0.36217913319422373, -1.1085804945035467, -1.5841909423986724, -0.2540315641886145")]
        [DataRow("indefinite svd ignore", "0.36217913319422373, -1.1085804945035467, -1.5841909423986724, -0.2540315641886145")]
        [DataRow("indefinite svd raise", "ERR ValueError: covariance is not symmetric positive-semidefinite.")]
        [DataRow("bad check_valid svd", "ERR ValueError: check_valid must equal 'warn', 'raise', or 'ignore'")]
        [DataRow("int mean svd", "1.3621791331942235, 0.8914195054964531, -0.5841909423986722, 1.7459684358113856")]
        [DataRow("f32 svd", "1.3621791331942235, 0.8914195054964531, -0.5841909423986722, 1.7459684358113856")]
        [DataRow("identity eigh", "0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139, -1.9510351886538364, -1.302179506862318")]
        [DataRow("c2 eigh", "-0.8419392585706049, -0.3235913764240801, -0.08263274357954108, 2.5466914752367806")]
        [DataRow("c3 eigh size(2,)", "0.26374742291630016, 0.7066270882001855, 2.7583697519875456, 3.370653959374436, 0.5863842217790805, 3.146493399631343")]
        [DataRow("c3 eigh size(2,2)", "0.26374742291630016, 0.7066270882001855, 2.7583697519875456, 3.370653959374436, 0.5863842217790805, 3.146493399631343, 1.115310194004396, 1.6978674933105262, 2.981880002713198, -0.33368448096064074, 2.653750479285454, 2.6424913517682835")]
        [DataRow("c3 eigh size None", "0.26374742291630016, 0.7066270882001855, 2.7583697519875456")]
        [DataRow("singular eigh", "-1.0399841062404955, -1.0399841062404955, 0.9405647163912139, 0.9405647163912139")]
        [DataRow("indefinite eigh warn", "-1.4891827138845326, -1.0582476870090918, 0.6213026831077241, 1.682600942116323")]
        [DataRow("indefinite eigh ignore", "-1.4891827138845326, -1.0582476870090918, 0.6213026831077241, 1.682600942116323")]
        [DataRow("indefinite eigh raise", "ERR ValueError: covariance is not symmetric positive-semidefinite.")]
        [DataRow("bad check_valid eigh", "ERR ValueError: check_valid must equal 'warn', 'raise', or 'ignore'")]
        [DataRow("int mean eigh", "-0.4891827138845326, 0.9417523129909082, 1.6213026831077242, 3.682600942116323")]
        [DataRow("f32 eigh", "-0.4891827138845326, 0.9417523129909082, 1.6213026831077242, 3.682600942116323")]
        [DataRow("identity cholesky", "0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139, -1.9510351886538364, -1.302179506862318")]
        [DataRow("c2 cholesky", "0.30471707975443135, -0.22341111640884814, 0.7504511958064572, 2.6194757636698034")]
        [DataRow("c3 cholesky size(2,)", "1.4309350268754408, 1.0483251312089878, 3.333186059026911, 2.3301593782100585, 0.2928923000664816, 1.81478391134785")]
        [DataRow("c3 cholesky size(2,2)", "1.4309350268754408, 1.0483251312089878, 3.333186059026911, 2.3301593782100585, 0.2928923000664816, 1.81478391134785, 1.1807936319784194, 1.7180738130949282, 2.937787561680194, -0.20638629167456912, 2.678425825163591, 3.63320177541757")]
        [DataRow("c3 cholesky size None", "1.4309350268754408, 1.0483251312089878, 3.333186059026911")]
        [DataRow("singular cholesky", "ERR LinAlgError: Matrix is not positive definite")]
        [DataRow("indefinite cholesky warn", "ERR LinAlgError: Matrix is not positive definite")]
        [DataRow("indefinite cholesky ignore", "ERR LinAlgError: Matrix is not positive definite")]
        [DataRow("indefinite cholesky raise", "ERR LinAlgError: Matrix is not positive definite")]
        [DataRow("bad check_valid cholesky", "0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139")]
        [DataRow("int mean cholesky", "1.4309350268754408, 0.9417523129909082, 2.0612982590085993, 3.682600942116323")]
        [DataRow("f32 cholesky", "1.4309350268754408, 0.9417523129909082, 2.0612982590085993, 3.682600942116323")]
        [DataRow("bad method", "ERR ValueError: method must be one of {'eigh', 'svd', 'cholesky'}")]
        [DataRow("mean 2d", "ERR ValueError: mean must be 1 dimensional")]
        [DataRow("cov not square", "ERR ValueError: cov must be 2 dimensional and square")]
        [DataRow("length mismatch", "ERR ValueError: mean and cov must have same length")]
        [DataRow("complex mean", "ERR TypeError: mean and cov must not be complex")]
        [DataRow("empty", "ERR ValueError: cannot reshape array of size 0 into shape (0)")]
        [DataRow("after raise", "0.09417734788764953")]
        [DataRow("after cholesky failure", "0.09417734788764953")]
        public void MultivariateNormal_WithLapackBackend_ByteIdenticalToNumPy(string probe, string expected)
        {
            try
            {
                OpenBlasEngine.Enable(threads: 1);
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }
            try
            {
                if (!OpenBlasEngine.LapackAvailable)
                    Assert.Inconclusive("the loaded BLAS exports no LAPACK routines.");
                AssertProbe(() => MultivariateNormalCalls[probe](np.random.default_rng(42)), expected, probe);
            }
            finally
            {
                OpenBlasEngine.Disable();
            }
        }

        /// <summary>
        ///     Without a backend the factorizations are managed (Jacobi eigen for svd/eigh, Cholesky-Banachiewicz): every
        ///     validation and error matches NumPy verbatim, a DIAGONAL covariance is byte-identical under all three methods
        ///     (no rotation, eigenvalue ties keep their order), and a general covariance's cholesky — a unique factor — agrees
        ///     to rounding. svd/eigh of a general covariance may pick the other sign of an eigenvector (a valid sample, not
        ///     NumPy's), which is why only their errors and the diagonal cases are pinned here.
        /// </summary>
        [TestMethod]
        public void MultivariateNormal_ManagedFallback_ExactWhereTheFactorIsUnique()
        {
            Assert.IsNull(np.zeros(1).TensorEngine.Blas, "this assembly runs without a BLAS backend by default");
            var d2 = new double[,] { { 2, 0 }, { 0, 1 } };
            var d3 = new double[,] { { 3, 0, 0 }, { 0, 1, 0 }, { 0, 0, 2 } };
            // NumPy 2.4.2: default_rng(42).multivariate_normal(mean, cov, 2, method=m)
            var expectedD2 = new Dictionary<string, string>
            {
                ["svd"] = "1.4309350268754408, -2.0399841062404955, 2.0612982590085993, -0.05943528360878614",
                ["eigh"] = "-0.4707596276977706, -0.6952829202455686, 2.3301593782100585, -0.24954880419354275",
                ["cholesky"] = "1.4309350268754408, -2.0399841062404955, 2.0612982590085993, -0.05943528360878614",
            };
            var expectedD3 = new Dictionary<string, string>
            {
                ["svd"] = "0.5277854640686928, 1.7504511958064572, 0.5292403723022294, 1.6291058765961939, -0.3021795068623181, -0.7591804244614058",
                ["eigh"] = "1.2998195997376039, 1.3047170797544314, 0.5292403723022294, -2.2554410664605204, 1.9405647163912139, -0.7591804244614058",
                ["cholesky"] = "0.5277854640686928, -0.03998410624049553, 3.0612982590085993, 1.6291058765961939, -0.9510351886538364, 0.1584400807510009",
            };
            foreach (var m in new[] { "svd", "eigh", "cholesky" })
            {
                AssertValues(np.random.default_rng(42).multivariate_normal(new[] { 1.0, -1.0 }, d2, new Shape(2), method: m), expectedD2[m], "d2 " + m);
                AssertValues(np.random.default_rng(42).multivariate_normal(new[] { 0.0, 1.0, 2.0 }, d3, new Shape(2), method: m), expectedD3[m], "d3 " + m);
                // identity: the draws are the normals themselves, under every method
                AssertValues(MultivariateNormalCalls[$"identity {m}"](np.random.default_rng(42)),
                    "0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139, -1.9510351886538364, -1.302179506862318", "identity " + m);
            }

            // cholesky of a general PD covariance: the factor is unique, so managed == LAPACK to rounding.
            var managed = MultivariateNormalCalls["c3 cholesky size(2,)"](np.random.default_rng(42)).ToArray<double>();
            var numpy = new[] { 1.4309350268754408, 1.0483251312089878, 3.333186059026911, 2.3301593782100585, 0.2928923000664816, 1.81478391134785 };
            for (int i = 0; i < numpy.Length; i++)
                Assert.AreEqual(numpy[i], managed[i], 1e-14 * System.Math.Abs(numpy[i]) + 1e-15, $"c3 cholesky[{i}]");

            // Every NumPy error, verbatim, on the managed path too.
            AssertProbe(() => MultivariateNormalCalls["indefinite svd raise"](np.random.default_rng(42)), "ERR ValueError: covariance is not symmetric positive-semidefinite.", "indefinite svd raise");
            AssertProbe(() => MultivariateNormalCalls["indefinite eigh raise"](np.random.default_rng(42)), "ERR ValueError: covariance is not symmetric positive-semidefinite.", "indefinite eigh raise");
            AssertProbe(() => MultivariateNormalCalls["singular cholesky"](np.random.default_rng(42)), "ERR LinAlgError: Matrix is not positive definite", "singular cholesky");
            AssertProbe(() => MultivariateNormalCalls["indefinite cholesky ignore"](np.random.default_rng(42)), "ERR LinAlgError: Matrix is not positive definite", "indefinite cholesky ignore");
            AssertProbe(() => MultivariateNormalCalls["bad check_valid svd"](np.random.default_rng(42)), "ERR ValueError: check_valid must equal 'warn', 'raise', or 'ignore'", "bad check_valid svd");
            AssertProbe(() => MultivariateNormalCalls["bad method"](np.random.default_rng(42)), "ERR ValueError: method must be one of {'eigh', 'svd', 'cholesky'}", "bad method");
            AssertProbe(() => MultivariateNormalCalls["complex mean"](np.random.default_rng(42)), "ERR TypeError: mean and cov must not be complex", "complex mean");
            AssertProbe(() => MultivariateNormalCalls["empty"](np.random.default_rng(42)), "ERR ValueError: cannot reshape array of size 0 into shape (0)", "empty");
            AssertProbe(() => MultivariateNormalCalls["after raise"](np.random.default_rng(42)), "0.09417734788764953", "after raise");
            AssertProbe(() => MultivariateNormalCalls["after cholesky failure"](np.random.default_rng(42)), "0.09417734788764953", "after cholesky failure");
            // check_valid is not validated for cholesky (NumPy skips it: the factorization already proves validity).
            AssertValues(MultivariateNormalCalls["bad check_valid cholesky"](np.random.default_rng(42)),
                "0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139", "bad check_valid cholesky");
        }

        // ---------------------------------------------------------------- conversions and size=()

        /// <summary>
        ///     The array-parameter conversions NumPy performs before sampling — <c>PyArray_FROM_OTF(n, NPY_INT64)</c>,
        ///     <c>PyArray_FROM_OTF(pvals, NPY_DOUBLE)</c>, <c>PyArray_FROMANY(alpha, NPY_DOUBLE, 1, 1)</c> — are gated by the
        ///     <c>'safe'</c> casting rule (float/uint64 counts and complex probabilities are refused, with <c>scalar</c> for a
        ///     0-d array), while bool/int probabilities pass. None-like arguments reproduce NumPy's texts. All from NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void Multivariate_ArrayConversions_MatchNumPy()
        {
            var half = np.array(new[] { 0.5, 0.5 });
            AssertProbe(() => np.random.default_rng(42).multinomial(np.array(new[] { 10.0, 20.0 }), half),
                "ERR TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'", "float counts");
            AssertProbe(() => np.random.default_rng(42).multinomial(NDArray.Scalar(10.5), half),
                "ERR TypeError: Cannot cast scalar from dtype('float64') to dtype('int64') according to the rule 'safe'", "0-d float count");
            AssertProbe(() => np.random.default_rng(42).multinomial(np.array(new ulong[] { 10, 20 }), half),
                "ERR TypeError: Cannot cast array data from dtype('uint64') to dtype('int64') according to the rule 'safe'", "uint64 counts");
            AssertProbe(() => np.random.default_rng(42).multinomial(10, np.array(new[] { new System.Numerics.Complex(0.5, 0), new System.Numerics.Complex(0.5, 0) })),
                "ERR TypeError: Cannot cast array data from dtype('complex128') to dtype('float64') according to the rule 'safe'", "complex pvals");
            AssertProbe(() => np.random.default_rng(42).dirichlet(np.array(new[] { new System.Numerics.Complex(1, 0), new System.Numerics.Complex(2, 0) })),
                "ERR TypeError: Cannot cast array data from dtype('complex128') to dtype('float64') according to the rule 'safe'", "complex alpha");
            AssertProbe(() => np.random.default_rng(42).multinomial((NDArray)null, half),
                "ERR TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'", "n=None");
            AssertProbe(() => np.random.default_rng(42).multinomial(NDArray.Scalar(10L), (NDArray)null),
                "ERR ValueError: pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0.", "pvals=None");
            AssertProbe(() => np.random.default_rng(42).multivariate_normal((NDArray)null, np.eye(2)), "ERR ValueError: mean must be 1 dimensional", "mean=None");
            AssertProbe(() => np.random.default_rng(42).multivariate_normal(np.zeros(2), (NDArray)null), "ERR ValueError: cov must be 2 dimensional and square", "cov=None");
            AssertProbe(() => np.random.default_rng(42).multivariate_normal((NDArray)null, np.eye(2).astype(np.complex128)),
                "ERR TypeError: mean and cov must not be complex", "mean=None, complex cov (the complex check comes first)");
            AssertProbe(() => np.random.default_rng(42).multivariate_hypergeometric(np.array(new[] { true, true }), 1),
                "ERR ValueError: colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807.", "bool colors");
            AssertProbe(() => np.random.default_rng(42).multivariate_hypergeometric(np.array(new ulong[] { 1UL << 63, 1 }), 1),
                "ERR ValueError: colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807.", "uint64 colors past int64");
            // int / bool probabilities are safe casts: pvals [1, 0] puts every trial in the first outcome.
            AssertValues(np.random.default_rng(42).multinomial(10, np.array(new long[] { 1, 0 })), "10, 0", "int pvals");
            AssertValues(np.random.default_rng(42).multinomial(10, np.array(new[] { true, false })), "10, 0", "bool pvals");
            // An empty colors array of ANY dtype is valid (NumPy only checks the dtype of a non-empty one).
            var none = np.random.default_rng(42).multivariate_hypergeometric(np.array(new double[0]), 0);
            CollectionAssert.AreEqual(new long[] { 0 }, none.shape);
            Assert.AreEqual(NPTypeCode.Int64, none.typecode);
        }

        /// <summary>
        ///     NumPy distinguishes <c>size=None</c> from <c>size=()</c>: the scalar samplers return a 0-d array for
        ///     <c>()</c>, the vector samplers prepend nothing (<c>(d,)</c>), and multinomial's broadcast path compares the
        ///     broadcast shape against <c>()</c> and refuses (<c>Output size () is not compatible ...</c>). C#'s default
        ///     <see cref="Shape"/> is <c>None</c>; <c>new Shape()</c> is <c>()</c>. Values from NumPy 2.4.2, seed 42.
        /// </summary>
        [TestMethod]
        public void SizeEmptyTuple_MatchesNumPy()
        {
            var beta = np.random.default_rng(42).beta(2.0, 3.0, new Shape());
            Assert.AreEqual(0, beta.ndim);
            AssertValues(beta, "0.33841183847153383", "beta size=()");
            var binom = np.random.default_rng(42).binomial(10, 0.5, new Shape());
            Assert.AreEqual(0, binom.ndim);
            AssertValues(binom, "6", "binomial size=()");

            var mn = np.random.default_rng(42).multinomial(10, new[] { 0.5, 0.5 }, new Shape());
            CollectionAssert.AreEqual(new long[] { 2 }, mn.shape);
            AssertValues(mn, "6, 4", "multinomial size=()");
            var dir = np.random.default_rng(42).dirichlet(new[] { 1.0, 2.0 }, new Shape());
            CollectionAssert.AreEqual(new long[] { 2 }, dir.shape);
            AssertValues(dir, "0.7865850744181714, 0.2134149255818285", "dirichlet size=()");
            var mvn = np.random.default_rng(42).multivariate_normal(new[] { 0.0, 0.0 }, new double[,] { { 1, 0 }, { 0, 1 } }, new Shape());
            CollectionAssert.AreEqual(new long[] { 2 }, mvn.shape);
            AssertValues(mvn, "0.30471707975443135, -1.0399841062404955", "multivariate_normal size=()");
            var mvhg = np.random.default_rng(42).multivariate_hypergeometric(new long[] { 3, 2 }, 2, new Shape());
            CollectionAssert.AreEqual(new long[] { 2 }, mvhg.shape);
            AssertValues(mvhg, "2, 0", "multivariate_hypergeometric size=()");

            // multinomial's vector path: the broadcast shape must equal size exactly.
            var ten = np.array(new long[] { 10 });
            AssertProbe(() => np.random.default_rng(42).multinomial(ten, np.array(new[] { 0.5, 0.5 }), new Shape()),
                "ERR ValueError: Output size () is not compatible with broadcast dimensions of inputs (1,).", "vector size=()");
            AssertProbe(() => np.random.default_rng(42).multinomial(np.array(new long[] { 10, 20 }), np.array(new[] { 0.5, 0.5 }), new Shape(1)),
                "ERR ValueError: Output size (1,) is not compatible with broadcast dimensions of inputs (2,).", "vector size=(1,)");
            var wide = np.random.default_rng(42).multinomial(np.array(new long[] { 10, 20 }), np.array(new[] { 0.5, 0.5 }), new Shape(3, 2));
            CollectionAssert.AreEqual(new long[] { 3, 2, 2 }, wide.shape);
            AssertValues(wide, "6, 4, 10, 10, 7, 3", "vector size=(3,2)");
            var rows = np.random.default_rng(42).multinomial(10, np.array(new[,] { { 0.5, 0.5 }, { 0.2, 0.8 } }), new Shape(2));
            CollectionAssert.AreEqual(new long[] { 2, 2 }, rows.shape);
            AssertValues(rows, "6, 4, 2, 8", "2-D pvals size=(2,)");
            // A 0-d count with 1-D probabilities is the plain path, like a scalar.
            AssertValues(np.random.default_rng(42).multinomial(NDArray.Scalar(10L), np.array(new[] { 0.5, 0.5 })), "6, 4", "0-d count");
        }

        // ---------------------------------------------------------------- read-ahead fills == per-draw calls

        /// <summary>
        ///     The sized fills read their 64-bit words / doubles through read-ahead buffers bounded by the draws still owed,
        ///     and multinomial through the binomial setup memo; neither may change the stream. Each fill must equal the same
        ///     number of single-value calls on an identically seeded generator, value for value, and leave the generator at
        ///     the same place (the next draw after the fill equals the next draw after the calls).
        /// </summary>
        [TestMethod]
        public void SizedFills_EqualPerValueCalls()
        {
            const int n = 300;
            var cases = new (string Name, Func<Generator, NDArray> Fill, Func<Generator, NDArray> One)[]
            {
                ("beta", g => g.beta(0.5, 0.5, n), g => g.beta(0.5, 0.5)),
                ("noncentral_chisquare", g => g.noncentral_chisquare(0.5, 1.5, n), g => g.noncentral_chisquare(0.5, 1.5)),
                ("noncentral_f", g => g.noncentral_f(5.0, 7.0, 1.5, n), g => g.noncentral_f(5.0, 7.0, 1.5)),
                ("standard_t", g => g.standard_t(3.5, n), g => g.standard_t(3.5)),
                ("vonmises", g => g.vonmises(0.5, 2.0, n), g => g.vonmises(0.5, 2.0)),
                ("wald", g => g.wald(3.0, 2.0, n), g => g.wald(3.0, 2.0)),
                ("standard_cauchy", g => g.standard_cauchy(n), g => g.standard_cauchy()),
                ("lognormal", g => g.lognormal(1.0, 0.5, n), g => g.lognormal(1.0, 0.5)),
                ("binomial", g => g.binomial(100, 0.4, n), g => g.binomial(100, 0.4)),
                ("negative_binomial", g => g.negative_binomial(5.0, 0.4, n), g => g.negative_binomial(5.0, 0.4)),
                ("poisson", g => g.poisson(100.0, n), g => g.poisson(100.0)),
                ("zipf", g => g.zipf(1.5, n), g => g.zipf(1.5)),
                ("geometric", g => g.geometric(0.1, n), g => g.geometric(0.1)),
                ("hypergeometric", g => g.hypergeometric(100, 200, 50, n), g => g.hypergeometric(100, 200, 50)),
                ("logseries", g => g.logseries(0.99, n), g => g.logseries(0.99)),
                ("multinomial", g => g.multinomial(20, new[] { 0.2, 0.3, 0.5 }, new Shape(n)), g => g.multinomial(20, new[] { 0.2, 0.3, 0.5 })),
                ("dirichlet", g => g.dirichlet(new[] { 2.0, 0.3, 5.0 }, new Shape(n)), g => g.dirichlet(new[] { 2.0, 0.3, 5.0 })),
            };
            foreach (var (name, fill, one) in cases)
            {
                var bulk = np.random.default_rng(7);
                var single = np.random.default_rng(7);
                var filled = fill(bulk);
                var parts = new List<NDArray>();
                for (int i = 0; i < n; i++)
                    parts.Add(one(single));
                var stitched = np.concatenate(parts.Select(p => p.reshape(1, -1)).ToArray()).reshape(filled.shape);
                Assert.IsTrue(np.array_equal(filled, stitched), $"{name}: the fill differs from {n} per-value calls");
                Assert.AreEqual(single.random().GetAtIndex<double>(0), bulk.random().GetAtIndex<double>(0),
                    $"{name}: the fill left the stream elsewhere");
            }
        }

        /// <summary>
        ///     Rows that draw NOTHING — a zero count, all-zero probabilities before the last outcome, all-zero Dirichlet
        ///     alphas — must not let a read-ahead run past the draws the drawing rows consume: the value drawn right after
        ///     the call is NumPy's (a read-ahead sized by "rows still owed" would have consumed doubles for the empty rows).
        ///     NumPy 2.4.2, <c>default_rng(42)</c>; <c>0.7739560485559633</c> is the untouched stream's first double.
        /// </summary>
        [TestMethod]
        public void NonDrawingRows_LeaveTheStreamWhereNumPyDoes()
        {
            var g = np.random.default_rng(42);
            AssertValues(g.multinomial(np.array(new long[] { 5, 0, 0, 0, 7 }), np.array(new[] { 0.5, 0.5 })),
                "3, 2, 0, 0, 0, 0, 0, 0, 3, 4", "counts with zeros");
            AssertValues(g.random(new Shape(1)), "0.8585979199113825", "next draw after counts with zeros");

            g = np.random.default_rng(42);
            AssertValues(g.multinomial(np.array(new long[,] { { 3 }, { 0 } }), np.array(new[,] { { 0.0, 1.0 }, { 0.4, 0.6 } })),
                "0, 3, 2, 1, 0, 0, 0, 0", "zero first probability + zero count");
            AssertValues(g.random(new Shape(1)), "0.4388784397520523", "next draw after the broadcast rows");

            g = np.random.default_rng(42);
            AssertValues(g.dirichlet(new[] { 0.0, 0.0 }, new Shape(3)), "0.0, 0.0, 0.0, 0.0, 0.0, 0.0", "all-zero alpha");
            AssertValues(g.random(new Shape(1)), "0.7739560485559633", "all-zero alpha draws nothing");

            g = np.random.default_rng(42);
            AssertValues(g.multinomial(0, new[] { 0.5, 0.5 }, new Shape(4)), "0, 0, 0, 0, 0, 0, 0, 0", "zero trials");
            AssertValues(g.random(new Shape(1)), "0.7739560485559633", "zero trials draw nothing");

            // Stick-breaking stops at the first zero tail sum: beta(0.05, 0) is 1, and NumPy's `break` skips the
            // beta(0, 0) draws of the zero alphas behind it — their values would be 0 anyway, so only the NEXT draw can
            // tell a loop that kept drawing from one that stopped.
            g = np.random.default_rng(42);
            AssertValues(g.dirichlet(new[] { 0.05, 0.0, 0.0 }, new Shape(3)), "1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0", "zero tail");
            AssertValues(g.random(new Shape(1)), "0.761139701990353", "the zero tail draws nothing");
        }

        /// <summary>
        ///     The univariate hypergeometric switches from the urn walk to HRUA at <c>10 &lt;= sample &lt;= good + bad - 10</c>,
        ///     BOTH ends inclusive — and it is dispatched twice: by <c>hypergeometric</c> itself and by the marginals method
        ///     of <c>multivariate_hypergeometric</c>, one univariate draw per color. <c>(8, 12, 10)</c> sits exactly on the
        ///     upper edge (<c>10 == 20 - 10</c>), where the two algorithms consume different numbers. NumPy 2.4.2, seed 42.
        /// </summary>
        [TestMethod]
        public void Hypergeometric_HruaWindowEdge_MatchesNumPy()
        {
            AssertValues(np.random.default_rng(42).hypergeometric(8, 12, 10, new Shape(6)), "4, 5, 3, 2, 5, 3", "hypergeometric(8,12,10)");
            var g = np.random.default_rng(42);
            AssertValues(g.multivariate_hypergeometric(new long[] { 8, 12 }, 10, new Shape(6)), "4, 6, 5, 5, 3, 7, 2, 8, 5, 5, 3, 7",
                "multivariate_hypergeometric([8,12],10) marginals");
            AssertValues(g.random(new Shape(1)), "0.9706980243949033", "next draw after the marginals");
        }
    }
}
