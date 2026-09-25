using System;
using System.Globalization;
using System.Linq;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact parity of the legacy <c>RandomState</c> (<see cref="NumPyRandom"/>) against NumPy 2.4.2: every sampler
    ///     at parameters that reach each internal branch, every parameter constraint (type, message, the <c>-0.0</c>/NaN/inf
    ///     corners), the <c>RandomState(bit_generator)</c> / seed / state surface, and the documented divergences.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The <c>[DataRow]</c> tables are NumPy's own output (<c>np.random.RandomState(42).&lt;method&gt;(..., size)</c> on
    ///     win-amd64): <see cref="Branch_Seed42_MatchesNumPy"/> draws 8 values (3 Dirichlet rows), and
    ///     <see cref="Validation_Seed42_MatchesNumPy"/> draws 2 values or expects NumPy's exception type and message verbatim.
    ///     Float values compare BIT-for-bit (a NumPy <c>nan</c> matches any NaN — NumPy's repr hides the sign).
    ///     </para>
    ///     <para>
    ///     NumPy's legacy integer samplers compute in C <c>long</c>; NumSharp's return int64 and model a 64-bit <c>long</c> (the
    ///     LP64 build of Linux/macOS). The cells where win-amd64's 32-bit <c>long</c> overflows are pinned against Linux NumPy
    ///     2.4.2 in <see cref="CLong_Is64Bit_MatchesLinuxNumPy"/> rather than in the Windows-authored tables.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class LegacyRandomStateTests
    {
        // ---------------------------------------------------------------- harness

        /// <summary>Calls the sampler a probe line names, with NumPy's positional arguments and the given size.</summary>
        private static NDArray Draw(NumPyRandom r, string method, string[] a, Shape size)
        {
            double D(int i) => ParseDouble(a[i]);
            long L(int i) => (long)ParseDouble(a[i]);
            switch (method)
            {
                case "poisson": return r.poisson(D(0), size);
                case "geometric": return r.geometric(D(0), size);
                case "zipf": return r.zipf(D(0), size);
                case "logseries": return r.logseries(D(0), size);
                case "hypergeometric": return r.hypergeometric(L(0), L(1), L(2), size);
                case "vonmises": return r.vonmises(D(0), D(1), size);
                case "beta": return r.beta(D(0), D(1), size);
                case "weibull": return r.weibull(D(0), size);
                case "noncentral_chisquare": return r.noncentral_chisquare(D(0), D(1), size);
                case "noncentral_f": return r.noncentral_f(D(0), D(1), D(2), size);
                case "standard_t": return r.standard_t(D(0), size);
                case "wald": return r.wald(D(0), D(1), size);
                case "laplace": return r.laplace(D(0), D(1), size);
                case "gumbel": return r.gumbel(D(0), D(1), size);
                case "logistic": return r.logistic(D(0), D(1), size);
                case "power": return r.power(D(0), size);
                case "rayleigh": return r.rayleigh(D(0), size);
                case "triangular": return r.triangular(D(0), D(1), D(2), size);
                case "chisquare": return r.chisquare(D(0), size);
                case "lognormal": return r.lognormal(D(0), D(1), size);
                case "exponential": return r.exponential(D(0), size);
                case "standard_exponential": return r.standard_exponential(size);
                case "normal": return r.normal(D(0), D(1), size);
                case "uniform": return r.uniform(D(0), D(1), size);
                case "standard_gamma": return r.standard_gamma(D(0), size);
                case "gamma": return r.gamma(D(0), D(1), size);
                case "binomial": return r.binomial(L(0), D(1), size);
                case "negative_binomial": return r.negative_binomial(D(0), D(1), size);
                case "f": return r.f(D(0), D(1), size);
                case "pareto": return r.pareto(D(0), size);
                case "standard_cauchy": return r.standard_cauchy(size);
                case "dirichlet": return r.dirichlet(a.Select(ParseDouble).ToArray(), size);
                default: throw new ArgumentException("unknown sampler " + method);
            }
        }

        private static double ParseDouble(string s)
        {
            switch (s.Trim())
            {
                case "nan": return double.NaN;
                case "inf": return double.PositiveInfinity;
                case "-inf": return double.NegativeInfinity;
                default: return double.Parse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture);
            }
        }

        /// <summary>Asserts the first values of <paramref name="actual"/> equal NumPy's printed ones bit-for-bit.</summary>
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
                    Assert.AreEqual(long.Parse(exp[i], CultureInfo.InvariantCulture), Convert.ToInt64(got),
                        $"{context}[{i}]");
                }
            }
        }

        /// <summary>Runs a NumPy probe line: either values to match, or <c>ERR Type: message</c> to raise.</summary>
        private static void AssertProbe(Func<NDArray> draw, string expected, string context)
        {
            if (expected.StartsWith("ERR ", StringComparison.Ordinal))
            {
                int colon = expected.IndexOf(": ", StringComparison.Ordinal);
                string type = expected.Substring(4, colon - 4);
                string message = expected.Substring(colon + 2);
                // NumPy's OverflowError is .NET's OverflowException (NumSharp has no OverflowError type).
                string expectedType = type == "OverflowError" ? nameof(OverflowException) : type;
                try
                {
                    draw();
                }
                catch (Exception e)
                {
                    Assert.AreEqual(expectedType, e.GetType().Name, $"{context}: exception type ({e.Message})");
                    Assert.AreEqual(message, e.Message, $"{context}: message");
                    return;
                }
                Assert.Fail($"{context}: expected {type}: {message}, but nothing was thrown");
            }
            AssertValues(draw(), expected, context);
        }

        // ---------------------------------------------------------------- every branch, seed 42

        /// <summary>
        ///     Each legacy sampler at parameters reaching each internal branch — Poisson product/PTRS, geometric search
        ///     vs inversion, hypergeometric urn vs HRUA, vonmises' small-kappa/Taylor/large-kappa paths, Johnk vs gamma
        ///     beta (incl. the log-space underflow path), binomial inversion vs BTPE on both sides of <c>p = 0.5</c>, … —
        ///     8 draws each (3 rows for Dirichlet), byte-identical to NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        [DataRow("poisson 3.5", "4, 1, 3, 4, 3, 2, 3, 2")]
        [DataRow("poisson 10", "12, 6, 11, 14, 7, 8, 9, 11")]
        [DataRow("poisson 15", "18, 10, 16, 19, 11, 13, 14, 16")]
        [DataRow("poisson 100", "96, 107, 88, 103, 111, 90, 94, 98")]
        [DataRow("poisson 1e6", "999640, 1000696, 998852, 997926, 1000288, 1001090, 998975, 999424")]
        [DataRow("geometric 0.35", "2, 7, 4, 3, 1, 1, 1, 5")]
        [DataRow("geometric 0.1", "5, 29, 13, 9, 2, 2, 1, 20")]
        [DataRow("geometric 1e-5", "46927, 301011, 131674, 91294, 16963, 16960, 5984, 201123")]
        [DataRow("geometric 1.0", "1, 1, 1, 1, 1, 1, 1, 1")]
        [DataRow("zipf 3", "1, 1, 1, 1, 1, 1, 2, 1")]
        [DataRow("zipf 1.5", "13, 1, 1, 1, 35, 1, 2, 3")]
        [DataRow("logseries 0.6", "2, 1, 1, 1, 1, 1, 1, 8")]
        [DataRow("logseries 0.99", "78, 5, 3, 153, 14, 337, 1, 4")]
        [DataRow("logseries 0.1", "1, 1, 1, 1, 1, 1, 2, 1")]
        [DataRow("hypergeometric 10,7,8", "5, 4, 7, 5, 4, 5, 3, 7")]
        [DataRow("hypergeometric 100,200,50", "18, 19, 15, 17, 14, 13, 14, 21")]
        [DataRow("hypergeometric 300,100,250", "189, 191, 184, 188, 183, 183, 194, 188")]
        [DataRow("hypergeometric 5,5,0", "ERR ValueError: nsample < 1 or nsample is NaN")]
        [DataRow("vonmises 0.5,2", "0.9598674699258964, -0.3963223738522026, 0.5641801515977654, 1.735806043476293, -1.331800190116148, 0.7072022028581144, 1.0513166676503367, 0.34404768089766513")]
        [DataRow("vonmises 0,1e-9", "-0.7882876818987491, 2.8319215077704234, 1.4576609265440963, 0.6198895383354298, -2.1612986243157413, -2.1614501754128375, -2.776642555026645, 2.3007525789727232")]
        [DataRow("vonmises 1,1e-6", "2.176651562520016, -0.8807401436120497, 1.1824748678175796, -3.0587104924191477, -1.615195182451929, 1.5761817159644584, 2.3569943200893917, 0.5617675364944783")]
        [DataRow("vonmises -2,1e7", "-1.9997890336278452, -2.000433326987646, -2.0004368571533027, -2.0001095687104957, -2.00016378949842, -2.0001556761817127, -2.000156241553867, -2.0003306949807182")]
        [DataRow("beta 2,3", "0.4944725879278468, 0.3751527747114991, 0.5298115439843974, 0.2363558476234208, 0.7680577944720322, 0.17279156167892437, 0.6534810368956206, 0.2705978714913083")]
        [DataRow("beta 0.5,0.5", "0.5992069666276891, 0.5000773047714704, 0.0044765792479401325, 0.41884401362412454, 0.0004502172091998639, 0.9389093143969878, 0.4956752858064321, 0.25157686103189675")]
        [DataRow("beta 1e-3,1e-3", "0.0, 1.0, 0.5385755737398579, 0.0, 7.582255876375736e-72, 0.0, 1.0, 0.00017515850237553296")]
        [DataRow("beta 0.5,2", "0.05026638760101149, 0.005369693564920609, 0.24935449882866298, 0.013436334481829567, 0.21750106559154805, 0.05990974859908524, 0.03631453329682964, 0.015777058949869673")]
        [DataRow("weibull 1.79", "0.6552942028273683, 1.8508250632419403, 1.1661672525249214, 0.9503889301414045, 0.37114808400031857, 0.37111314898273473, 0.2073719839266719, 1.4775146704123043")]
        [DataRow("weibull 0", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0")]
        [DataRow("noncentral_chisquare 3,1.5", "0.9512746325012711, 2.7220388175766628, 2.381050363027285, 7.458888089296003, 0.4561629914872856, 0.8946281161337545, 2.65150779353597, 5.419569307210394")]
        [DataRow("noncentral_chisquare 0.5,1.5", "3.571053553082895, 2.4212470829570494, 5.357238989365533, 0.0021859693450730507, 0.15165651657684298, 0.28029675625183453, 0.03603055380094746, 4.528917782694487")]
        [DataRow("noncentral_chisquare 3,0", "3.579246352985752, 2.0472117146208575, 1.8631756716660912, 1.863206197770533, 7.677525590890684, 4.4148160261578955, 1.458993968698631, 3.7125976658364617")]
        [DataRow("noncentral_f 5,7,1.5", "2.6581869236416944, 0.8525721526249321, 1.6719063086836394, 3.415908185848533, 0.4930085512830893, 2.080795607463497, 1.2044454990117768, 0.7021864070178195")]
        [DataRow("noncentral_f 0.5,7,1.5", "15.906246443475235, 2.505500260014797, 16.73106248518274, 0.4573792939112855, 3.258288778533602, 0.28885423944519684, 10.371396281496216, 0.0007609097060359516")]
        [DataRow("standard_t 3.5", "0.5857577548238303, -1.0869651773057356, 1.3106954651808416, -0.8196066046141552, 0.850376255013144, 2.09083386813635, -2.1474057785036553, -0.5546726907661551")]
        [DataRow("standard_t 0.5", "0.4635126225574857, -2.8400572132968636, 1863.5037023497734, 0.5310544686362968, -2.503246321878733, -1.2480932908018798, -5.933503059485928, 0.7554028573827192")]
        [DataRow("wald 3,2", "5.462736285308775, 3.552846468927796, 2.2542287006403976, 3.9924169464369457, 0.5394667163724391, 7.440746175777657, 1.7010840587499085, 1.56169231531472")]
        [DataRow("laplace 0,1", "-0.28890917477433936, 2.316974250357576, 0.6235985129855038, 0.21979537321600792, -1.1646326082953993, -1.164787217839572, -2.152724536860913, 1.3180836839199936")]
        [DataRow("gumbel 0.5,2", "2.0131621068576706, -1.7039608408891258, -0.05032661771606628, 0.6821646413118817, 4.048331848996323, 4.048668839148807, 6.1322030471822755, -0.8974938100735572")]
        [DataRow("logistic 0,1", "-0.5127882653574256, 2.9595797554777894, 1.0047626522764483, 0.39987856707212394, -1.6881549183929985, -1.688338106484912, -2.7860329488121782, 1.8675638749646242")]
        [DataRow("power 2.5", "0.6751485473767549, 0.9799863164943249, 0.8826794072449688, 0.814463557037856, 0.47563148510739983, 0.4756020711501022, 0.32034757942172426, 0.9441532397131729")]
        [DataRow("power 0.3", "0.03787283477484421, 0.844954709348313, 0.3534746629997689, 0.18082722004447113, 0.0020445055650856556, 0.002043452169639033, 7.588898129582281e-05, 0.6194706443634856")]
        [DataRow("rayleigh 1.5", "1.453171154715048, 3.6804274804876735, 2.434205336645724, 2.026879742853973, 0.8736772385043337, 0.8736036364483171, 0.5189166202185694, 3.0084113565401474")]
        [DataRow("triangular 0,3,10", "3.383188707490244, 8.14258282787472, 5.668669480032533, 4.699631512224133, 2.1634600096311223, 2.1632927703124416, 1.3200410467277084, 6.93933507293684")]
        [DataRow("triangular 0,0,1", "0.2091397840625453, 0.7779961856406881, 0.48230698460516686, 0.3664847943395807, 0.08131541889636384, 0.0813022914669933, 0.029476230156210703, 0.6341805715587746")]
        [DataRow("triangular 0,1,1", "0.6119968291154477, 0.9750457970833556, 0.8555664450008573, 0.7737302399396295, 0.3949919498451032, 0.3949614162626555, 0.24100541937516562, 0.9306858469832531")]
        [DataRow("chisquare 3", "3.579246352985752, 2.0472117146208575, 1.8631756716660912, 1.863206197770533, 7.677525590890684, 4.4148160261578955, 1.458993968698631, 3.7125976658364617")]
        [DataRow("chisquare 0.5", "0.03935712548755209, 0.5741957088926019, 0.0011850480284706065, 2.2763784382687446e-05, 0.26113211779443657, 3.5907965881190097e-07, 1.0441785474824883, 0.0021859693450730507")]
        [DataRow("lognormal 0,1", "1.6433127155860012, 0.8708684897640193, 1.9111182426600926, 4.586099388741997, 0.7912404503034572, 0.7912534408171102, 4.851135569125472, 2.154232968599504")]
        [DataRow("exponential 2", "0.9385361799537182, 6.020242861835042, 2.6334913870908987, 1.8258851075519065, 0.33924974092469257, 0.33919258382921036, 0.11967753721736135, 4.022461728959879")]
        [DataRow("standard_exponential", "0.4692680899768591, 3.010121430917521, 1.3167456935454493, 0.9129425537759532, 0.16962487046234628, 0.16959629191460518, 0.059838768608680676, 2.0112308644799395")]
        [DataRow("normal 1,2", "1.9934283060224653, 0.7234713976576307, 2.295377076201385, 4.046059712816051, 0.5316932505533281, 0.5317260861016389, 4.158425631014783, 2.534869458305818")]
        [DataRow("uniform -1,3", "0.49816047538944996, 2.8028572256396647, 1.9279757672456204, 1.3946339367881464, -0.3759254382302539, -0.3760219186551894, -0.7676655513272022, 2.4647045830997407")]
        [DataRow("standard_gamma 0.5", "0.14028030062619642, 0.6590180328421851, 0.024341816165506288, 0.0033737060025058087, 0.3757291395473922, 0.00042372140541392347, 1.0954577625968716, 0.033060318699863214")]
        [DataRow("standard_gamma 3", "3.5628186625547027, 2.4471943983022544, 2.302280566853262, 2.302304875447745, 6.166139937186394, 4.126452255001466, 1.9711400756530786, 3.6544096987496513")]
        [DataRow("gamma 0.5,2", "0.28056060125239285, 1.3180360656843702, 0.048683632331012576, 0.006747412005011617, 0.7514582790947844, 0.0008474428108278469, 2.190915525193743, 0.06612063739972643")]
        [DataRow("gamma 3,2", "7.125637325109405, 4.894388796604509, 4.604561133706524, 4.60460975089549, 12.332279874372787, 8.252904510002931, 3.942280151306157, 7.3088193974993025")]
        [DataRow("binomial 10,0.35", "3, 6, 4, 4, 2, 2, 1, 5")]
        [DataRow("binomial 100,0.4", "40, 42, 35, 33, 27, 42, 42, 33")]
        [DataRow("binomial 1000,0.7", "711, 714, 698, 722, 694, 727, 693, 697")]
        [DataRow("binomial 20,0.9", "19, 16, 17, 18, 19, 19, 20, 17")]
        [DataRow("negative_binomial 5,0.4", "7, 5, 1, 5, 5, 6, 29, 18")]
        [DataRow("negative_binomial 0.5,0.5", "0, 0, 0, 2, 0, 0, 0, 1")]
        [DataRow("f 5,7", "1.426878677609774, 0.9306053722130452, 1.6019765944592734, 0.51140262823974, 4.25605676526683, 0.36927520510613915, 2.576540613011974, 0.6068891344418865")]
        [DataRow("pareto 3", "0.16932036645405568, 1.7274682836709054, 0.5510238033601347, 0.35569492764272526, 0.058170658636719, 0.05816057835783717, 0.02014651200627582, 0.9550392855774972")]
        [DataRow("standard_cauchy", "-3.5924974762375737, 0.425263191903688, 1.000070120387526, 2.0577812750936095, -0.8652948028240994, 0.9950356172435398, -0.12646462651858437, 3.0676793327454464")]
        [DataRow("dirichlet 2,3,5", "0.2653815050911402, 0.2713145859970568, 0.463303908911803, 0.09819562487471749, 0.4380286183997543, 0.4637757567255281, 0.11316878986639724, 0.3656380219993532, 0.5211931881342495")]
        [DataRow("dirichlet 0.1,0.1", "0.001228498736354108, 0.9987715012636459, 0.9999488613724509, 5.1138627549225066e-05, 0.9999999999999978, 2.2173093195020756e-15")]
        public void Branch_Seed42_MatchesNumPy(string name, string expected)
        {
            int sp = name.IndexOf(' ');
            string method = sp < 0 ? name : name.Substring(0, sp);
            string[] args = sp < 0 ? Array.Empty<string>() : name.Substring(sp + 1).Split(',');
            Shape size = method == "dirichlet" ? new Shape(3) : new Shape(8);
            AssertProbe(() => Draw(np.random.RandomState(42), method, args, size), expected, name);
        }

        // ---------------------------------------------------------------- every constraint, seed 42

        /// <summary>
        ///     Every parameter constraint at its boundary (negative, <c>-0.0</c>, 0, NaN, inf, and the p/a/n/sample edges):
        ///     NumPy's exception type and message verbatim, or — where NumPy accepts the value — its first two draws. Pins the
        ///     sign-bit test that rejects <c>-0.0</c> for <c>CONS_NON_NEGATIVE</c>, the NaN pass-through of
        ///     <c>CONS_POSITIVE</c>, the zero-draw paths (<c>weibull(0)</c>, <c>poisson(0)</c>), and the draw-consuming
        ///     <c>scale == 0</c> paths (<c>gumbel</c>/<c>logistic</c>/<c>laplace</c>/<c>rayleigh</c>).
        /// </summary>
        [TestMethod]
        [DataRow("exponential(-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("normal(0,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("standard_gamma(-1.0)", "ERR ValueError: shape < 0")]
        [DataRow("gamma(-1.0,1)", "ERR ValueError: shape < 0")]
        [DataRow("gamma(1,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("f(-1.0,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("f(1,-1.0)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(-1.0,1,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("noncentral_f(1,1,-1.0)", "ERR ValueError: nonc < 0")]
        [DataRow("chisquare(-1.0)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(-1.0,1)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(1,-1.0)", "ERR ValueError: nonc < 0")]
        [DataRow("standard_t(-1.0)", "ERR ValueError: df <= 0")]
        [DataRow("vonmises(0,-1.0)", "ERR ValueError: kappa < 0")]
        [DataRow("pareto(-1.0)", "ERR ValueError: a <= 0")]
        [DataRow("weibull(-1.0)", "ERR ValueError: a < 0")]
        [DataRow("power(-1.0)", "ERR ValueError: a <= 0")]
        [DataRow("laplace(0,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("gumbel(0,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("logistic(0,-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("lognormal(0,-1.0)", "ERR ValueError: sigma < 0")]
        [DataRow("rayleigh(-1.0)", "ERR ValueError: scale < 0")]
        [DataRow("wald(-1.0,1)", "ERR ValueError: mean <= 0")]
        [DataRow("wald(1,-1.0)", "ERR ValueError: scale <= 0")]
        [DataRow("beta(-1.0,1)", "ERR ValueError: a <= 0")]
        [DataRow("beta(1,-1.0)", "ERR ValueError: b <= 0")]
        [DataRow("negative_binomial(-1.0,0.5)", "ERR ValueError: n <= 0")]
        [DataRow("poisson(-1.0)", "ERR ValueError: lam < 0 or lam is NaN")]
        [DataRow("exponential(-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("normal(0,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("standard_gamma(-0.0)", "ERR ValueError: shape < 0")]
        [DataRow("gamma(-0.0,1)", "ERR ValueError: shape < 0")]
        [DataRow("gamma(1,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("f(-0.0,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("f(1,-0.0)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(-0.0,1,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("noncentral_f(1,1,-0.0)", "ERR ValueError: nonc < 0")]
        [DataRow("chisquare(-0.0)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(-0.0,1)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(1,-0.0)", "ERR ValueError: nonc < 0")]
        [DataRow("standard_t(-0.0)", "ERR ValueError: df <= 0")]
        [DataRow("vonmises(0,-0.0)", "ERR ValueError: kappa < 0")]
        [DataRow("pareto(-0.0)", "ERR ValueError: a <= 0")]
        [DataRow("weibull(-0.0)", "ERR ValueError: a < 0")]
        [DataRow("power(-0.0)", "ERR ValueError: a <= 0")]
        [DataRow("laplace(0,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("gumbel(0,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("logistic(0,-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("lognormal(0,-0.0)", "ERR ValueError: sigma < 0")]
        [DataRow("rayleigh(-0.0)", "ERR ValueError: scale < 0")]
        [DataRow("wald(-0.0,1)", "ERR ValueError: mean <= 0")]
        [DataRow("wald(1,-0.0)", "ERR ValueError: scale <= 0")]
        [DataRow("beta(-0.0,1)", "ERR ValueError: a <= 0")]
        [DataRow("beta(1,-0.0)", "ERR ValueError: b <= 0")]
        [DataRow("negative_binomial(-0.0,0.5)", "ERR ValueError: n <= 0")]
        [DataRow("poisson(-0.0)", "0, 0")]
        [DataRow("exponential(0.0)", "0.0, 0.0")]
        [DataRow("normal(0,0.0)", "0.0, 0.0")]
        [DataRow("standard_gamma(0.0)", "0.0, 0.0")]
        [DataRow("gamma(0.0,1)", "0.0, 0.0")]
        [DataRow("gamma(1,0.0)", "0.0, 0.0")]
        [DataRow("f(0.0,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("f(1,0.0)", "ERR ValueError: dfden <= 0")]
        [DataRow("noncentral_f(0.0,1,1)", "ERR ValueError: dfnum <= 0")]
        [DataRow("noncentral_f(1,1,0.0)", "0.21286261321439337, 7.215156313984232")]
        [DataRow("chisquare(0.0)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(0.0,1)", "ERR ValueError: df <= 0")]
        [DataRow("noncentral_chisquare(1,0.0)", "0.28056060125239285, 1.3180360656843702")]
        [DataRow("standard_t(0.0)", "ERR ValueError: df <= 0")]
        [DataRow("vonmises(0,0.0)", "-0.7882876818987491, 2.8319215077704234")]
        [DataRow("pareto(0.0)", "ERR ValueError: a <= 0")]
        [DataRow("weibull(0.0)", "0.0, 0.0")]
        [DataRow("power(0.0)", "ERR ValueError: a <= 0")]
        [DataRow("laplace(0,0.0)", "0.0, 0.0")]
        [DataRow("gumbel(0,0.0)", "0.0, 0.0")]
        [DataRow("logistic(0,0.0)", "0.0, 0.0")]
        [DataRow("lognormal(0,0.0)", "1.0, 1.0")]
        [DataRow("rayleigh(0.0)", "0.0, 0.0")]
        [DataRow("wald(0.0,1)", "ERR ValueError: mean <= 0")]
        [DataRow("wald(1,0.0)", "ERR ValueError: scale <= 0")]
        [DataRow("beta(0.0,1)", "ERR ValueError: a <= 0")]
        [DataRow("beta(1,0.0)", "ERR ValueError: b <= 0")]
        [DataRow("negative_binomial(0.0,0.5)", "ERR ValueError: n <= 0")]
        [DataRow("poisson(0.0)", "0, 0")]
        [DataRow("exponential(nan)", "nan, nan")]
        [DataRow("normal(0,nan)", "nan, nan")]
        [DataRow("standard_gamma(nan)", "nan, nan")]
        [DataRow("gamma(nan,1)", "nan, nan")]
        [DataRow("gamma(1,nan)", "nan, nan")]
        [DataRow("f(nan,1)", "nan, nan")]
        [DataRow("f(1,nan)", "nan, nan")]
        [DataRow("noncentral_f(nan,1,1)", "nan, nan")]
        [DataRow("noncentral_f(1,1,nan)", "nan, nan")]
        [DataRow("chisquare(nan)", "nan, nan")]
        [DataRow("noncentral_chisquare(nan,1)", "nan, nan")]
        [DataRow("noncentral_chisquare(1,nan)", "nan, nan")]
        [DataRow("standard_t(nan)", "nan, nan")]
        [DataRow("vonmises(0,nan)", "nan, nan")]
        [DataRow("pareto(nan)", "nan, nan")]
        [DataRow("weibull(nan)", "nan, nan")]
        [DataRow("power(nan)", "nan, nan")]
        [DataRow("laplace(0,nan)", "nan, nan")]
        [DataRow("gumbel(0,nan)", "nan, nan")]
        [DataRow("logistic(0,nan)", "nan, nan")]
        [DataRow("lognormal(0,nan)", "nan, nan")]
        [DataRow("rayleigh(nan)", "nan, nan")]
        [DataRow("wald(nan,1)", "nan, nan")]
        [DataRow("wald(1,nan)", "nan, nan")]
        [DataRow("beta(nan,1)", "nan, nan")]
        [DataRow("beta(1,nan)", "nan, nan")]
        [DataRow("negative_binomial(nan,0.5)", "0, 0")]
        [DataRow("poisson(nan)", "ERR ValueError: lam < 0 or lam is NaN")]
        [DataRow("exponential(inf)", "inf, inf")]
        [DataRow("normal(0,inf)", "inf, -inf")]
        [DataRow("standard_gamma(inf)", "inf, inf")]
        [DataRow("gamma(inf,1)", "inf, inf")]
        [DataRow("gamma(1,inf)", "inf, inf")]
        [DataRow("f(inf,1)", "nan, nan")]
        [DataRow("f(1,inf)", "nan, nan")]
        [DataRow("noncentral_f(inf,1,1)", "nan, nan")]
        [DataRow("noncentral_f(1,1,inf)", "7.215156313984232, 0.0003867984872456036")]
        [DataRow("chisquare(inf)", "inf, inf")]
        [DataRow("noncentral_chisquare(inf,1)", "inf, inf")]
        [DataRow("noncentral_chisquare(1,inf)", "0.048683632331012576, 0.0008474428108278469")]
        [DataRow("standard_t(inf)", "nan, nan")]
        [DataRow("pareto(inf)", "0.0, 0.0")]
        [DataRow("weibull(inf)", "1.0, 1.0")]
        [DataRow("power(inf)", "1.0, 1.0")]
        [DataRow("laplace(0,inf)", "-inf, inf")]
        [DataRow("gumbel(0,inf)", "inf, -inf")]
        [DataRow("logistic(0,inf)", "-inf, inf")]
        [DataRow("lognormal(0,inf)", "inf, 0.0")]
        [DataRow("rayleigh(inf)", "inf, inf")]
        [DataRow("wald(inf,1)", "nan, nan")]
        [DataRow("wald(1,inf)", "nan, nan")]
        [DataRow("beta(inf,1)", "nan, nan")]
        [DataRow("beta(1,inf)", "0.0, 0.0")]
        [DataRow("poisson(inf)", "ERR ValueError: lam value too large")]
        [DataRow("binomial(10,-0.5)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("negative_binomial(1,-0.5)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("geometric(-0.5)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(-0.5)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("binomial(10,-0.0)", "0, 0")]
        [DataRow("negative_binomial(1,-0.0)", "0, 0")]
        [DataRow("geometric(-0.0)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(-0.0)", "1, 1")]
        [DataRow("binomial(10,0.0)", "0, 0")]
        [DataRow("geometric(0.0)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(0.0)", "1, 1")]
        [DataRow("binomial(10,0.5)", "4, 8")]
        [DataRow("negative_binomial(1,0.5)", "2, 0")]
        [DataRow("geometric(0.5)", "1, 5")]
        [DataRow("logseries(0.5)", "2, 1")]
        [DataRow("binomial(10,1.0)", "10, 10")]
        [DataRow("negative_binomial(1,1.0)", "0, 0")]
        [DataRow("geometric(1.0)", "1, 1")]
        [DataRow("logseries(1.0)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("binomial(10,1.5)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("negative_binomial(1,1.5)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("geometric(1.5)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(1.5)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("binomial(10,nan)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("negative_binomial(1,nan)", "ERR ValueError: p < 0, p > 1 or p is NaN")]
        [DataRow("geometric(nan)", "ERR ValueError: p <= 0, p > 1 or p contains NaNs")]
        [DataRow("logseries(nan)", "ERR ValueError: p < 0, p >= 1 or p is NaN")]
        [DataRow("zipf(0.5)", "ERR ValueError: a <= 1 or a is NaN")]
        [DataRow("zipf(1.0)", "ERR ValueError: a <= 1 or a is NaN")]
        [DataRow("zipf(nan)", "ERR ValueError: a <= 1 or a is NaN")]
        [DataRow("binomial(-1,0.5)", "ERR ValueError: n < 0")]
        [DataRow("poisson(1e19)", "ERR ValueError: lam value too large")]
        [DataRow("hypergeometric(-1,5,1)", "ERR ValueError: ngood < 0")]
        [DataRow("hypergeometric(5,-1,1)", "ERR ValueError: nbad < 0")]
        [DataRow("hypergeometric(5,5,0)", "ERR ValueError: nsample < 1 or nsample is NaN")]
        [DataRow("hypergeometric(5,5,11)", "ERR ValueError: ngood + nbad < nsample")]
        [DataRow("hypergeometric(5,5,10)", "5, 5")]
        [DataRow("hypergeometric(0,5,3)", "0, 0")]
        [DataRow("triangular(1,0,2)", "ERR ValueError: left > mode")]
        [DataRow("triangular(0,3,2)", "ERR ValueError: mode > right")]
        [DataRow("triangular(1,1,1)", "ERR ValueError: left == right")]
        [DataRow("uniform(0,inf)", "ERR OverflowError: Range exceeds valid bounds")]
        [DataRow("uniform(1,0)", "0.6254598811526375, 0.049285693590083834")]
        [DataRow("uniform(nan,1)", "ERR OverflowError: Range exceeds valid bounds")]
        [DataRow("dirichlet([0,1])", "ERR ValueError: alpha <= 0")]
        [DataRow("dirichlet([-1,1])", "ERR ValueError: alpha <= 0")]
        [DataRow("dirichlet([nan,1])", "nan, nan")]
        [DataRow("dirichlet([])", "")]
        public void Validation_Seed42_MatchesNumPy(string name, string expected)
        {
            int open = name.IndexOf('(');
            string method = name.Substring(0, open);
            string inner = name.Substring(open + 1, name.Length - open - 2);
            string[] args;
            if (method == "dirichlet")
            {
                string list = inner.Trim('[', ']');
                args = list.Length == 0 ? Array.Empty<string>() : list.Split(',');
            }
            else
            {
                args = inner.Length == 0 ? Array.Empty<string>() : inner.Split(',');
            }
            AssertProbe(() => Draw(np.random.RandomState(42), method, args, new Shape(2)), expected, name);
        }

        // ---------------------------------------------------------------- C long width (LP64 model)

        /// <summary>
        ///     The legacy integer samplers compute in C <c>long</c>. NumSharp returns int64 and models the 64-bit <c>long</c> of
        ///     NumPy's LP64 builds: every value here equals NumPy 2.4.2 on Linux (probed); NumPy's win-amd64 build, whose
        ///     <c>long</c> is 32-bit, differs exactly where that width overflows — <c>-2147483648</c> for the two infinite
        ///     negative-binomial means, candidates above <c>2**31 - 1</c> rejected by <c>zipf</c>, <c>poisson(9.2e18)</c>
        ///     raising <c>lam value too large</c>, and <c>tomaxint</c> drawing <c>next_uint32 &gt;&gt; 1</c>.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void CLong_Is64Bit_MatchesLinuxNumPy()
        {
            AssertValues(np.random.RandomState(42).negative_binomial(double.PositiveInfinity, 0.5, new Shape(2)),
                "-9223372036854775808, -9223372036854775808", "negative_binomial(inf, 0.5)");
            AssertValues(np.random.RandomState(42).negative_binomial(1.0, 0.0, new Shape(2)),
                "-9223372036854775808, -9223372036854775808", "negative_binomial(1, 0)");
            AssertValues(np.random.RandomState(42).zipf(1.0000001, new Shape(2)), "9569555325, 4710132439", "zipf(1.0000001)");
            AssertValues(np.random.RandomState(42).poisson(9.2e18, new Shape(2)),
                "9200000002110169088, 9199999996520137728", "poisson(9.2e18)");
            AssertValues(np.random.RandomState(42).tomaxint(new Shape(3)),
                "3454522818714476249, 8768791796696926855, 6751452423619668515", "tomaxint");
            AssertValues(np.random.RandomState(42).binomial(1L << 40, 0.5, new Shape(2)), "549755290311, 549756239949",
                "binomial(2**40, 0.5)");
            // The bounds follow: poisson accepts up to the int64 lambda cap and rejects just past it.
            Action tooLarge = () => np.random.RandomState(42).poisson(1e19);
            tooLarge.Should().Throw<ValueError>().WithMessage("lam value too large");
            NumPyRandom._poisson_lam_max.Should().Be(9.223372006484771e+18);
        }

        [TestMethod]
        public void Tomaxint_IsInt64_AndScalarIsZeroD()
        {
            var a = np.random.RandomState(42).tomaxint(new Shape(2, 3));
            a.dtype.Should().Be(np.int64);
            a.shape.Should().Equal(2L, 3L);
            var s = np.random.RandomState(42).tomaxint();
            s.ndim.Should().Be(0);
            Convert.ToInt64(s.GetAtIndex(0)).Should().Be(3454522818714476249L);
        }

        [TestMethod]
        public void Ranf_And_Sample_Alias_RandomSample()
        {
            AssertValues(np.random.RandomState(42).ranf(4), "0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366", "ranf");
            AssertValues(np.random.RandomState(42).sample(4), "0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366", "sample");
        }

        // ---------------------------------------------------------------- NumPy never returns here

        /// <summary>
        ///     NumPy's legacy <c>zipf</c> loops forever for <c>a &gt;= 1025</c> (an infinite <c>2**(a-1)</c> makes every
        ///     acceptance test NaN); NumSharp returns 1 — the distribution's limit and the modern sampler's answer — without
        ///     drawing, so the stream after it is untouched.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Zipf_AtLeast1025_ReturnsOne_WhereNumPyHangs()
        {
            var rs = np.random.RandomState(42);
            AssertValues(rs.zipf(double.PositiveInfinity, new Shape(3)), "1, 1, 1", "zipf(inf)");
            AssertValues(rs.zipf(1025.0, new Shape(2)), "1, 1", "zipf(1025)");
            // No draw was consumed: the next uniform is seed 42's first.
            AssertValues(rs.random_sample(1), "0.3745401188473625", "stream untouched");
        }

        /// <summary>
        ///     NumPy's legacy <c>vonmises</c> loops forever once <c>4*kappa^2</c> overflows (the envelope parameter is NaN);
        ///     NumSharp returns <c>mu</c> wrapped to <c>[-pi, pi]</c> — the limit of the distribution — without drawing.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Vonmises_OverflowingKappa_ReturnsWrappedMu_WhereNumPyHangs()
        {
            var rs = np.random.RandomState(42);
            AssertValues(rs.vonmises(0.5, double.PositiveInfinity, new Shape(2)), "0.5, 0.5", "vonmises(0.5, inf)");
            var wrapped = rs.vonmises(4.0, 1e200, new Shape(1));
            ((double)wrapped.GetAtIndex(0)).Should().BeApproximately(4.0 - 2 * System.Math.PI, 1e-15);
            AssertValues(rs.random_sample(1), "0.3745401188473625", "stream untouched");
        }

        // ---------------------------------------------------------------- RandomState(bit_generator)

        [TestMethod]
        public void RandomState_OverPCG64_DrawsNumPysLegacyStream()
        {
            AssertValues(np.random.RandomState(new PCG64(42)).random_sample(3),
                "0.7739560485559633, 0.4388784397520523, 0.8585979199113825", "random_sample");
            AssertValues(np.random.RandomState(new PCG64(42)).randn(3),
                "-0.33091407603531797, 1.4832067819346502, 0.43138215439433264", "randn (polar method on PCG64 doubles)");
            var ri = np.random.RandomState(new PCG64(42)).randint(0, 10, new Shape(5));
            ri.dtype.Should().Be(np.int32);
            AssertValues(ri, "8, 1, 1, 4, 2", "randint");
            AssertValues(np.random.RandomState(new PCG64(42)).standard_exponential(new Shape(3)),
                "1.4870258232016522, 0.5778177119982773, 1.9561478149587175", "standard_exponential");
            AssertValues(np.random.RandomState(new PCG64(42)).gamma(2.0, 1.0, new Shape(3)),
                "1.2749202275294806, 4.40838803717237, 2.8278461324046007", "gamma");
        }

        [TestMethod]
        public void RandomState_OverPhiloxAndSeedSequenceMT19937_DrawNumPysStreams()
        {
            AssertValues(np.random.RandomState(new Philox(42)).random_sample(3),
                "0.08607763073528474, 0.14155732377913233, 0.27009303504774695", "Philox");
            // MT19937(42) is SeedSequence-seeded — a different stream from RandomState(42)'s legacy seeding.
            AssertValues(np.random.RandomState(new MT19937(42)).random_sample(3),
                "0.5419938930062744, 0.6196672126927824, 0.05736978170666862", "MT19937(42)");
        }

        [TestMethod]
        public void RandomState_SharesTheBitGenerator()
        {
            var bg = new PCG64(42);
            var rs = np.random.RandomState(bg);
            ReferenceEquals(rs._bit_generator, bg).Should().BeTrue();
            // default_rng(RandomState) wraps the SAME engine (NumPy 2.x).
            AssertValues(np.random.default_rng(np.random.RandomState(new PCG64(3))).random(new Shape(2)),
                "0.08564916714362436, 0.2368105065960997", "default_rng(RandomState(PCG64(3)))");
            np.random.RandomState((BitGenerator)null)._bit_generator.Should().BeOfType<MT19937>();
        }

        [TestMethod]
        public void RandomState_Str_NamesTheBitGenerator()
        {
            np.random.RandomState(new PCG64(1)).ToString().Should().Be("RandomState(PCG64)");
            np.random.RandomState(42).ToString().Should().Be("RandomState(MT19937)");
        }

        [TestMethod]
        public void Seed_OnNonMT19937_RaisesTypeError_BeforeValidatingTheSeed()
        {
            var rs = np.random.RandomState(new PCG64(1));
            Action a = () => rs.seed(5);
            a.Should().Throw<TypeError>().WithMessage("can only re-seed a MT19937 BitGenerator");
            // NumPy checks the bit generator first, so an out-of-range seed still reports the TypeError.
            Action b = () => rs.seed(-1);
            b.Should().Throw<TypeError>();
            Action c = () => rs.seed();
            c.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void RandomState_Seed_FullUInt32Range_AndArrays()
        {
            // RandomState(seed) over NumPy's whole [0, 2**32 - 1] range and the init_by_array forms.
            AssertValues(np.random.RandomState(4294967295L).random_sample(1), "0.0976320289940138", "seed 2**32 - 1");
            AssertValues(np.random.RandomState(new[] { 1, 2, 3 }).random_sample(1), "0.6098612722867289", "seed [1, 2, 3]");
            AssertValues(np.random.RandomState(new uint[] { 1, 2, 3 }).random_sample(1), "0.6098612722867289", "seed uint[]");
            AssertValues(np.random.RandomState(new long[] { 1, 2, 3 }).random_sample(1), "0.6098612722867289", "seed long[]");
            Action neg = () => np.random.RandomState(-1L);
            neg.Should().Throw<ValueError>().WithMessage("Seed must be between 0 and 2**32 - 1");
            Action big = () => np.random.RandomState(4294967296L);
            big.Should().Throw<ValueError>().WithMessage("Seed must be between 0 and 2**32 - 1");
            Action empty = () => np.random.RandomState(new int[0]);
            empty.Should().Throw<ValueError>().WithMessage("Seed must be non-empty");
        }

        // ---------------------------------------------------------------- get_state / set_state

        [TestMethod]
        public void GetState_NonLegacy_IsTheBitGeneratorDictPlusGauss()
        {
            var st = (NumPyRandom.State)np.random.RandomState(new PCG64(1)).get_state(legacy: false);
            st.bit_generator.Should().Be("PCG64");
            var pcg = (PCG64.State)st.state;
            pcg.state.Should().Be(UInt128.Parse("207833532711051698738587646355624148094"));
            pcg.inc.Should().Be(UInt128.Parse("194290289479364712180083596243593368443"));
            pcg.has_uint32.Should().Be(0);
            pcg.uinteger.Should().Be(0u);
            st.has_gauss.Should().Be(0);
            st.gauss.Should().Be(0.0);

            // legacy=True on a non-MT19937 generator: NumPy warns and returns the dict — so does get_state(true).
            np.random.RandomState(new PCG64(1)).get_state(legacy: true).Should().BeOfType<NumPyRandom.State>();
            // ... and the typed legacy-tuple overload, which cannot return a dict, refuses.
            Action tuple = () => np.random.RandomState(new PCG64(1)).get_state();
            tuple.Should().Throw<ValueError>().WithMessage("get_state and legacy can only be used with the MT19937 BitGenerator.*");

            // An MT19937 RandomState gives the tuple for legacy=True and the dict for legacy=False.
            np.random.RandomState(42).get_state(legacy: true).Should().BeOfType<NativeRandomState>();
            var mt = (NumPyRandom.State)np.random.RandomState(42).get_state(legacy: false);
            mt.bit_generator.Should().Be("MT19937");
            ((MT19937.State)mt.state).pos.Should().Be(624);
        }

        [TestMethod]
        public void SetState_DictRoundTrip_RestoresStreamAndGaussCache()
        {
            var rs = np.random.RandomState(new PCG64(7));
            rs.randn(); // leaves a cached Gaussian
            var st = rs.get_state(legacy: false);
            var a = rs.randn(3);
            rs.set_state(st);
            var b = rs.randn(3);
            const string expected = "0.2568975630239209, -0.7065128420760581, 0.7088085038621023";
            AssertValues(a, expected, "before");
            AssertValues(b, expected, "after restore");
        }

        [TestMethod]
        public void SetState_Errors_MatchNumPy()
        {
            Action wrongGenerator = () => np.random.RandomState(new PCG64(7)).set_state(np.random.RandomState(1).get_state());
            wrongGenerator.Should().Throw<ValueError>().WithMessage("state must be for a PCG64 RNG");

            Action wrongAlgorithm = () => np.random.RandomState(1).set_state(new NativeRandomState(new uint[624], 0) { Algorithm = "XYZ" });
            wrongAlgorithm.Should().Throw<ValueError>().WithMessage("set_state can only be used with legacy MT19937 state instances.");

            Action invalidDict = () => np.random.RandomState(1).set_state(new NumPyRandom.State());
            invalidDict.Should().Throw<ValueError>().WithMessage("state dictionary is not valid.");

            Action notAState = () => np.random.RandomState(1).set_state((object)"MT19937");
            notAState.Should().Throw<TypeError>().WithMessage("state must be a dict or a tuple.");
        }

        /// <summary>
        ///     NumPy writes the cached Gaussian BEFORE handing the bit generator its state, so a state the bit generator
        ///     rejects still leaves the tuple's Gaussian installed — the next normal is that value.
        /// </summary>
        [TestMethod]
        public void SetState_Rejected_StillInstallsTheGauss_AsNumPyDoes()
        {
            var rs = np.random.RandomState(new PCG64(7));
            var key = np.random.RandomState(1).get_state().Key;
            Action act = () => rs.set_state(new NativeRandomState(key, 0, hasGauss: 1, cachedGaussian: 5.0));
            act.Should().Throw<ValueError>().WithMessage("state must be for a PCG64 RNG");
            AssertValues(rs.randn(1), "5.0", "the rejected tuple's Gaussian");
        }

        [TestMethod]
        public void SetState_ThreeFieldTuple_ClearsTheGaussCache()
        {
            var rs = np.random.RandomState(42);
            rs.randn();
            var st = rs.get_state();
            rs.set_state(new NativeRandomState(st.Key, st.Pos)); // NumPy's ('MT19937', key, pos) form
            var after = rs.get_state();
            after.HasGauss.Should().Be(0);
            after.CachedGaussian.Should().Be(0.0);
        }

        [TestMethod]
        public void SetState_BareBitGeneratorDict_ClearsTheGaussCache()
        {
            var rs = np.random.RandomState(new PCG64(7));
            var bare = rs._bit_generator.state; // no has_gauss/gauss keys -> NumPy defaults them to 0
            rs.randn();                          // caches a Gaussian
            rs.set_state((object)bare);
            var st = (NumPyRandom.State)rs.get_state(legacy: false);
            st.has_gauss.Should().Be(0);
            st.gauss.Should().Be(0.0);
            // With the cache cleared the stream restarts exactly like a fresh RandomState(PCG64(7)).
            var fresh = np.random.RandomState(new PCG64(7)).randn(3);
            np.array_equal(rs.randn(3), fresh).Should().BeTrue();
        }

        [TestMethod]
        public void ConsumingTheGaussCache_ZeroesIt()
        {
            var rs = np.random.RandomState(42);
            rs.randn();
            var s1 = rs.get_state();
            rs.randn(); // consumes the cached value
            var s2 = rs.get_state();
            s1.HasGauss.Should().Be(1);
            s1.CachedGaussian.Should().Be(-0.13826430117118466);
            s2.HasGauss.Should().Be(0);
            s2.CachedGaussian.Should().Be(0.0); // NumPy zeroes it; NumSharp used to keep the stale value
        }

        // ---------------------------------------------------------------- the carved samplers, now exact

        [TestMethod]
        [DataRow(0.5, 1.0, "0.14028030062619642, 0.6590180328421851, 0.024341816165506288, 0.0033737060025058087, 0.3757291395473922, 0.00042372140541392347")]
        [DataRow(0.3, 2.0, "0.07574566954968842, 0.712883483408045, 0.004089011130171311, 0.00015177796259164563, 0.3666248584964874, 4.7807643338440905e-06")]
        [DataRow(0.9, 1.0, "0.3889985685673327, 1.2135036290073526, 0.12856528567576725, 0.04233760054385244, 0.8155503686006947, 0.013370750351154027")]
        [DataRow(1.0, 3.0, "1.4078042699305773, 9.030364292752562, 3.950237080636348, 2.7388276613278597, 0.5088746113870388, 0.5087888757438155")]
        [DataRow(0.0, 1.0, "0.0, 0.0, 0.0, 0.0, 0.0, 0.0")]
        [DataRow(2.5, 1.0, "2.983135365424813, 1.9694529574820125, 1.8399551367455784, 1.8399768082363774, 5.4216067408766335, 3.5039914140566815")]
        public void Gamma_IncludingShapeBelowOne_MatchesNumPy(double shape, double scale, string expected)
            => AssertValues(np.random.RandomState(42).gamma(shape, scale, new Shape(6)), expected, $"gamma({shape}, {scale})");

        [TestMethod]
        [DataRow(10L, 0.35, "3, 6, 4, 4, 2, 2, 1, 5")]
        [DataRow(100L, 0.4, "40, 42, 35, 33, 27, 42, 42, 33")]
        [DataRow(1000L, 0.7, "711, 714, 698, 722, 694, 727, 693, 697")]
        [DataRow(20L, 0.9, "19, 16, 17, 18, 19, 19, 20, 17")]
        [DataRow(10L, 0.0, "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow(10L, 1.0, "10, 10, 10, 10, 10, 10, 10, 10")]
        [DataRow(0L, 0.5, "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow(5L, 0.5, "2, 4, 3, 3, 1, 1, 1, 4")]
        [DataRow(200L, 0.1, "19, 27, 23, 21, 16, 16, 14, 25")]
        [DataRow(60L, 0.5, "29, 36, 32, 31, 26, 26, 24, 34")]
        public void Binomial_InversionAndBtpe_MatchNumPy(long n, double p, string expected)
            => AssertValues(np.random.RandomState(42).binomial(n, p, new Shape(8)), expected, $"binomial({n}, {p})");

        [TestMethod]
        public void Binomial_CacheAcrossCalls_MatchesNumPy()
        {
            var rs = np.random.RandomState(42);
            var a = rs.binomial(100, 0.4, new Shape(3));
            var b = rs.binomial(10, 0.35, new Shape(3));
            var c = rs.binomial(100, 0.4, new Shape(3));
            AssertValues(np.concatenate(new[] { a, b, c }), "40, 42, 35, 4, 4, 1, 43, 41, 37", "sequence");
            var s = np.random.RandomState(42).binomial(10, 0.35);
            s.ndim.Should().Be(0);
            AssertValues(s, "3", "scalar");
        }

        [TestMethod]
        [DataRow(3.5, "4, 1, 3, 4, 3, 2, 3, 2")]
        [DataRow(10.0, "12, 6, 11, 14, 7, 8, 9, 11")]
        [DataRow(15.0, "18, 10, 16, 19, 11, 13, 14, 16")]
        [DataRow(100.0, "96, 107, 88, 103, 111, 90, 94, 98")]
        [DataRow(0.0, "0, 0, 0, 0, 0, 0, 0, 0")]
        [DataRow(1e6, "999640, 1000696, 998852, 997926, 1000288, 1001090, 998975, 999424")]
        public void Poisson_ProductAndPtrs_MatchNumPy(double lam, string expected)
            => AssertValues(np.random.RandomState(42).poisson(lam, new Shape(8)), expected, $"poisson({lam})");

        [TestMethod]
        [DataRow(5.0, 0.4, "7, 5, 1, 5, 5, 6, 29, 18")]
        [DataRow(1.5, 0.2, "4, 1, 2, 2, 1, 0, 3, 4")]
        [DataRow(100.0, 0.9, "13, 12, 7, 11, 8, 15, 7, 16")]
        [DataRow(0.5, 0.5, "0, 0, 0, 2, 0, 0, 0, 1")]
        public void NegativeBinomial_MatchesNumPy(double n, double p, string expected)
            => AssertValues(np.random.RandomState(42).negative_binomial(n, p, new Shape(8)), expected, $"negative_binomial({n}, {p})");

        [TestMethod]
        public void F_Pareto_StandardCauchy_MatchNumPy()
        {
            AssertValues(np.random.RandomState(42).f(5.0, 7.0, new Shape(5)),
                "1.426878677609774, 0.9306053722130452, 1.6019765944592734, 0.51140262823974, 4.25605676526683", "f(5,7)");
            AssertValues(np.random.RandomState(42).f(0.5, 0.5, new Shape(5)),
                "0.06854305052794027, 52.05848063522653, 727226.1499257665, 477.67300572442105, 0.24612978762284055", "f(0.5,0.5)");
            AssertValues(np.random.RandomState(42).pareto(0.5, new Shape(5)),
                "1.5562368075977924, 410.6785645985084, 12.922293260891735, 5.208287590218708, 0.40389391119912044", "pareto(0.5)");
            AssertValues(np.random.RandomState(42).pareto(100.0, new Shape(5)),
                "0.004703708770126491, 0.030558835975647014, 0.01325452965221996, 0.009171225850531206, 0.0016976881482271278", "pareto(100)");
            AssertValues(np.random.RandomState(42).standard_cauchy(new Shape(6)),
                "-3.5924974762375737, 0.425263191903688, 1.000070120387526, 2.0577812750936095, -0.8652948028240994, 0.9950356172435398",
                "standard_cauchy");
            AssertValues(np.random.RandomState(42).standard_cauchy(), "-3.5924974762375737", "standard_cauchy scalar");
        }

        [TestMethod]
        public void Multinomial_MatchesNumPy()
        {
            var one = np.random.RandomState(42).multinomial(20, new[] { 0.2, 0.3, 0.5 });
            one.dtype.Should().Be(np.int32);
            one.shape.Should().Equal(3L);
            AssertValues(one, "3, 10, 7", "single");
            var four = np.random.RandomState(42).multinomial(20, new[] { 0.2, 0.3, 0.5 }, 4);
            four.shape.Should().Equal(4L, 3L);
            AssertValues(four, "3, 10, 7, 5, 6, 9, 2, 5, 13, 1, 9, 10", "size=4");
            var grid = np.random.RandomState(42).multinomial(100, Enumerable.Repeat(1 / 6.0, 6).ToArray(), new[] { 2, 3 });
            grid.shape.Should().Equal(2L, 3L, 6L);
            AssertValues(grid, "15, 23, 18, 15, 12, 17, 13, 12, 23, 18, 19, 15, 9, 26, 20, 12", "size=(2,3)");
            AssertValues(np.random.RandomState(42).multinomial(10, new[] { 0.0, 1.0 }), "0, 10", "[0,1]");
            AssertValues(np.random.RandomState(42).multinomial(10, new[] { 1.0 }), "10", "[1]");
            AssertValues(np.random.RandomState(42).multinomial(0, new[] { 0.5, 0.5 }), "0, 0", "n=0");
            AssertValues(np.random.RandomState(42).multinomial(1000, new[] { 0.1, 0.2, 0.3, 0.4 }), "96, 194, 306, 404", "n=1000");
            // Only pvals[:-1] must sum to <= 1: the last category takes the remainder.
            AssertValues(np.random.RandomState(42).multinomial(10, new[] { 0.6, 0.6 }), "7, 3", "[.6,.6]");
            // Empty pvals is legal (an empty row).
            np.random.RandomState(42).multinomial(10, new double[0], new[] { 3 }).shape.Should().Equal(3L, 0L);
        }

        [TestMethod]
        public void Multinomial_Errors_MatchNumPyOrder()
        {
            Action negN = () => np.random.RandomState(42).multinomial(-1, new[] { 0.5, 0.5 });
            negN.Should().Throw<ValueError>().WithMessage("n < 0");
            Action badP = () => np.random.RandomState(42).multinomial(-1, new[] { 1.5, 0.5 }); // pvals checked before n
            badP.Should().Throw<ValueError>().WithMessage("pvals < 0, pvals > 1 or pvals contains NaNs");
            Action overSum = () => np.random.RandomState(42).multinomial(10, new[] { 0.5, 0.5 + 2e-12, 0.0 });
            overSum.Should().Throw<ValueError>().WithMessage("sum(pvals[:-1]) > 1.0");
            Action withinTol = () => np.random.RandomState(42).multinomial(10, new[] { 0.5, 0.5 + 1e-13, 0.0 });
            withinTol.Should().NotThrow();
            Action scalar = () => np.random.RandomState(42).multinomial(10, np.array(0.5));
            scalar.Should().Throw<TypeError>().WithMessage("pvals must be a 1-d sequence");
            Action nullP = () => np.random.RandomState(42).multinomial(10, (double[])null);
            nullP.Should().Throw<TypeError>().WithMessage("pvals must be a 1-d sequence");
            Action twoD = () => np.random.RandomState(42).multinomial(10, np.array(new double[,] { { 0.5, 0.5 } }));
            twoD.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.*");
            // A float32 pvals that sums below 1.0001 in its own dtype but fails after the float64 cast gets NumPy's longer text.
            Action f32 = () => np.random.RandomState(42).multinomial(10, np.array(new[] { 0.9999999f, 0.0000002f, 0.0f }));
            f32.Should().Throw<ValueError>().WithMessage("sum(pvals[:-1].astype(np.float64)) > 1.0. The pvals array is cast to 64-bit "
                + "floating point prior to checking the sum. Precision changes when casting may cause problems even if the sum of "
                + "the original pvals is valid.");
            // float16's 1.0001 rounds to 1.0 (NEP 50 compares in the array's dtype), so a float16 sum of exactly 1.0 keeps
            // the plain message — as in NumPy.
            Action f16 = () => np.random.RandomState(42).multinomial(10, np.array(new[] { (Half)0.9999, (Half)0.0002, (Half)0.0 }));
            f16.Should().Throw<ValueError>().WithMessage("sum(pvals[:-1]) > 1.0");
        }

        [TestMethod]
        public void Multinomial_NDArrayPvals_AnyDtypeAndLayout_MatchesDoubleArray()
        {
            var expected = np.random.RandomState(42).multinomial(20, new[] { 0.2, 0.3, 0.5 }, 4);
            var reversed = np.array(new[] { 0.5, 0.3, 0.2 })["::-1"];
            var viaView = np.random.RandomState(42).multinomial(20, reversed, new Shape(4));
            np.array_equal(expected, viaView).Should().BeTrue();
        }

        [TestMethod]
        public void MultivariateNormal_DrawsNormalsFirst_AndValidatesLikeNumPy()
        {
            // Identity covariance needs no sign convention: the draws are the legacy normals themselves.
            AssertValues(np.random.RandomState(42).multivariate_normal(new[] { 0.0, 0.0 }, new double[,] { { 1, 0 }, { 0, 1 } }),
                "0.4967141530112327, -0.13826430117118466", "identity");

            Action notSquare = () => np.random.RandomState(42).multivariate_normal(np.zeros(2), np.zeros(new Shape(2, 3)));
            notSquare.Should().Throw<ValueError>().WithMessage("cov must be 2 dimensional and square");
            Action mean2d = () => np.random.RandomState(42).multivariate_normal(np.zeros(new Shape(2, 1)), np.zeros(new Shape(2, 2)));
            mean2d.Should().Throw<ValueError>().WithMessage("mean must be 1 dimensional");
            Action length = () => np.random.RandomState(42).multivariate_normal(np.zeros(3), np.zeros(new Shape(2, 2)));
            length.Should().Throw<ValueError>().WithMessage("mean and cov must have same length");

            // check_valid is validated AFTER the normals are drawn (the stream has advanced), as in NumPy.
            var rs = np.random.RandomState(42);
            Action badMode = () => rs.multivariate_normal(new[] { 0.0, 0.0 }, new double[,] { { 1, 0 }, { 0, 1 } }, check_valid: "loud");
            badMode.Should().Throw<ValueError>().WithMessage("check_valid must equal 'warn', 'raise', or 'ignore'");
            AssertValues(rs.random_sample(1), "0.7319939418114051", "the polar pair consumed seed 42's first two uniforms");

            Action notPsd = () => np.random.RandomState(42).multivariate_normal(new[] { 0.0, 0.0 },
                new double[,] { { 1, 2 }, { 2, 1 } }, check_valid: "raise");
            notPsd.Should().Throw<ValueError>().WithMessage("covariance is not symmetric positive-semidefinite.");
        }

        /// <summary>
        ///     With the LAPACK backend installed the SVD and both products are NumPy's own <c>gesdd</c>/<c>gemm</c>, so the
        ///     samples are byte-identical to <c>np.random.RandomState(42).multivariate_normal</c> (the managed fallback's
        ///     Jacobi SVD cannot reproduce LAPACK's singular-vector signs in general).
        /// </summary>
        [TestMethod]
        public void MultivariateNormal_WithLapackBackend_ByteIdenticalToNumPy()
        {
            try
            {
                OpenBlasEngine.Enable();
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }
            try
            {
                if (!OpenBlasEngine.LapackAvailable)
                    Assert.Inconclusive("the loaded BLAS exports no LAPACK routines.");
                AssertValues(np.random.RandomState(42).multivariate_normal(new[] { 0.0, 1.0 },
                        new double[,] { { 1.0, 0.5 }, { 0.5, 2.0 } }, 2),
                    "0.16865044563071147, 1.728877966541595, 1.6211710484410384, 1.3699967921934395", "2x2");
                AssertValues(np.random.RandomState(42).multivariate_normal(new[] { 1.0, 2.0, 3.0 },
                        new double[,] { { 2.0, 0.3, 0.1 }, { 0.3, 1.0, 0.2 }, { 0.1, 0.2, 0.5 } }, new Shape(2)),
                    "0.2738498251073418, 1.7812484330188778, 3.3758068661054406, -1.1765413470471926, 1.6447506819726518, 2.719985664206707",
                    "3x3");
            }
            finally
            {
                OpenBlasEngine.Disable();
            }
        }

        // ---------------------------------------------------------------- locking

        /// <summary>
        ///     Every legacy sampler draws under the bit generator's lock, so concurrent callers interleave whole calls: the
        ///     values two threads receive are exactly a split of what one thread would have drawn call by call.
        /// </summary>
        [TestMethod]
        public void ConcurrentCalls_InterleaveWholeCalls()
        {
            const int calls = 400;
            var shared = np.random.RandomState(7);
            var bag = new System.Collections.Concurrent.ConcurrentBag<double>();
            System.Threading.Tasks.Parallel.For(0, calls, _ =>
            {
                var v = shared.normal(0.0, 1.0, new Shape(4));
                bag.Add((double)v.GetAtIndex(0) + (double)v.GetAtIndex(1) + (double)v.GetAtIndex(2) + (double)v.GetAtIndex(3));
            });
            var serial = np.random.RandomState(7);
            var expected = Enumerable.Range(0, calls).Select(_ =>
            {
                var v = serial.normal(0.0, 1.0, new Shape(4));
                return (double)v.GetAtIndex(0) + (double)v.GetAtIndex(1) + (double)v.GetAtIndex(2) + (double)v.GetAtIndex(3);
            }).OrderBy(x => x).ToArray();
            bag.OrderBy(x => x).ToArray().Should().Equal(expected);
        }
    }
}
