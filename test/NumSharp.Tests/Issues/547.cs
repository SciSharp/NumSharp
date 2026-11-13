using System;
using System.Linq;
using System.Reflection;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the progress comment on GitHub #547, "[NEP50 NEP52 NEP56] NumPy 2.0 Breaking Changes"
    ///     (https://github.com/SciSharp/NumSharp/issues/547), which stays open: what 0.80.0 delivers passes here, and what the
    ///     comment lists as still missing is pinned <c>[OpenBugs]</c>, so each pin turns green when its gap closes. Expected
    ///     values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue547
    {
        /// <summary>The NumPy 2.4.2 names the comment lists as missing from NumSharp (Wave 0 of the port map).</summary>
        internal static readonly string[] MissingNumPy2Names =
        {
            "acos", "asin", "atan", "atan2", "pow", "bitwise_invert", "bitwise_left_shift", "bitwise_right_shift",
            "cumulative_sum", "cumulative_prod", "astype",
        };

        /// <summary>Whether <c>np</c> has a public static member of the given name.</summary>
        /// <param name="name">The NumPy name.</param>
        /// <returns>True when <c>np.&lt;name&gt;</c> exists.</returns>
        internal static bool NpHas(string name) => typeof(np).GetMember(name, BindingFlags.Public | BindingFlags.Static).Length > 0;

        /// <summary>
        ///     NEP 50 is done: a C# literal is a weak Python scalar (a float32 array plus 0.1 stays float32), an out-of-range
        ///     literal raises (OverflowException, NumPy's OverflowError), and promotion is order-independent
        ///     (<c>result_type(int8, uint8)</c> and <c>result_type(uint8, int8)</c> are both int16).
        /// </summary>
        [TestMethod]
        public void Nep50_IsDone()
        {
            (np.ones(new Shape(2), np.float32) + 0.1).typecode.Should().Be(NPTypeCode.Single);
            new Action(() => { var _ = np.array(new byte[] { 1 }) + 300; }).Should().Throw<OverflowError>();
            np.result_type(np.int8, np.uint8).Should().Be((DType)np.int16);
            np.result_type(np.uint8, np.int8).Should().Be((DType)np.int16);
        }

        /// <summary>
        ///     The NEP 56 names the comment lists as present exist, and a sample computes: <c>.mT</c> swaps the last two axes,
        ///     <c>concat</c> joins, and <c>ceil</c> / <c>floor</c> / <c>trunc</c> keep integer dtypes (int64, int8, uint16).
        /// </summary>
        [TestMethod]
        public void Nep56_PresentNames_AndIntegerRounding()
        {
            foreach (var name in new[] { "isdtype", "unique_values", "unique_counts", "unique_inverse", "unique_all", "matrix_transpose", "vecdot", "concat", "permute_dims" })
                NpHas(name).Should().BeTrue($"np.{name} is present");
            foreach (var name in new[] { "mT", "device", "to_device" })
                typeof(NDArray).GetMember(name, BindingFlags.Public | BindingFlags.Instance).Should().NotBeEmpty($"NDArray.{name} is present");
            typeof(np.linalg).GetMember("matrix_norm").Should().NotBeEmpty();
            typeof(np.linalg).GetMember("vector_norm").Should().NotBeEmpty();
            typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == "asarray")
                .Should().Contain(m => m.GetParameters().Any(p => p.Name == "copy"), "asarray takes copy=");

            np.ones(new Shape(2, 3, 4)).mT.Should().BeShaped(2, 4, 3);
            np.concat(new[] { np.arange(2), np.arange(3) }).Should().BeShaped(5);
            np.ceil(np.array(new long[] { 1, 2 })).typecode.Should().Be(NPTypeCode.Int64);
            np.floor(np.array(new sbyte[] { 1, 2 })).typecode.Should().Be(NPTypeCode.SByte);
            np.trunc(np.array(new ushort[] { 1, 2 })).typecode.Should().Be(NPTypeCode.UInt16);
        }

        /// <summary>
        ///     The comment's two corrections of the issue text: <c>.T</c> on a 3-D array reverses the axes (it is not an
        ///     error), and <c>np.cross</c> of 2-vectors — the z-component with a DeprecationWarning when the comment was
        ///     posted (NumPy 2.4) — raises since NumPy 2.5 expired that deprecation, so NumSharp raises NumPy's text too.
        /// </summary>
        [TestMethod]
        public void IssueTextCorrections_TransposeAndCross()
        {
            np.ones(new Shape(2, 3, 4)).T.Should().BeShaped(4, 3, 2);
            new Action(() => np.cross(np.array(new[] { 1.0, 2.0 }), np.array(new[] { 3.0, 4.0 })))
                .Should().Throw<ValueError>().WithMessage(
                    "Both input arrays must be (arrays of) 3-dimensional vectors, but they are 2 and 2 dimensional instead.");
        }

        /// <summary>
        ///     <c>np.find_common_type</c> is kept on purpose, as NumPy 1.26.4's (the last NumPy that has it).
        /// </summary>
        [TestMethod]
        public void FindCommonType_IsKeptOnPurpose()
        {
            NpHas("find_common_type").Should().BeTrue();
        }

        /// <summary>
        ///     Regression pin for the comment's "still missing" list: the 11 NumPy 2.4.2 names it named all exist now (the
        ///     last one, <c>np.astype</c>, arrived with API-parity batch 7 on 2026-11-12). A failure lists the ones missing
        ///     again.
        /// </summary>
        [TestMethod]
        public void StillMissing_NumPy2Names()
        {
            var missing = MissingNumPy2Names.Where(n => !NpHas(n)).ToArray();
            missing.Should().BeEmpty($"NumPy 2.4.2 has them; missing: {string.Join(", ", missing)}");
        }

        /// <summary>
        ///     NEP 52: <c>np.round_</c> and <c>np.asscalar</c> still exist though NumPy removed them. Fails until both are
        ///     removed.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "https://github.com/SciSharp/NumSharp/issues/547")]
        public void Nep52_RemovedAliases_StillExist()
        {
            new[] { "round_", "asscalar" }.Where(NpHas).Should().BeEmpty("NumPy 2.x removed np.round_ and np.asscalar");
        }
    }
}
