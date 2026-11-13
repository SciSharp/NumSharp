using System;
using System.Linq;
using System.Reflection;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the progress comment on GitHub #560, "[Array API Standard] Cross-Platform Array Library Compatibility"
    ///     (https://github.com/SciSharp/NumSharp/issues/560), which stays open: the present part passes, and the missing part
    ///     (complex64 and 11 spec names) is pinned <c>[OpenBugs]</c>. Expected values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue560
    {
        /// <summary>
        ///     The array object's attributes (<c>dtype</c>, <c>device</c>, <c>ndim</c>, <c>shape</c>, <c>size</c>, <c>T</c>,
        ///     <c>mT</c>), <c>to_device</c>, and the constants <c>e</c>, <c>inf</c>, <c>nan</c>, <c>newaxis</c>, <c>pi</c> exist.
        /// </summary>
        [TestMethod]
        public void ArrayObject_AndConstants_ArePresent()
        {
            foreach (var name in new[] { "dtype", "device", "ndim", "shape", "size", "T", "mT", "to_device" })
                typeof(NDArray).GetMember(name, BindingFlags.Public | BindingFlags.Instance).Should().NotBeEmpty($"NDArray.{name}");
            foreach (var name in new[] { "e", "inf", "nan", "newaxis", "pi" })
                Issue547.NpHas(name).Should().BeTrue($"np.{name}");
            np.pi.Should().Be(System.Math.PI);
            double.IsPositiveInfinity(np.inf).Should().BeTrue();
        }

        /// <summary>
        ///     The functions the comment lists as present exist, and <c>std</c> with <c>correction: 1</c> is NumPy's sample
        ///     standard deviation (1.2909944487358056 for [1, 2, 3, 4]).
        /// </summary>
        [TestMethod]
        public void PresentFunctions_AndStdCorrection()
        {
            foreach (var name in new[]
                     {
                         "isdtype", "unique_all", "unique_counts", "unique_inverse", "unique_values", "take", "take_along_axis",
                         "unstack", "copysign", "nextafter", "logaddexp", "hypot", "signbit", "std", "var",
                     })
                Issue547.NpHas(name).Should().BeTrue($"np.{name}");

            np.std(np.array(new[] { 1.0, 2.0, 3.0, 4.0 }), correction: 1).GetDouble().Should().Be(1.2909944487358056);
        }

        /// <summary>
        ///     The <c>fft</c> and <c>linalg</c> extensions are there (a sample of each computes without a backend), and int8 and
        ///     complex128 are dtypes.
        /// </summary>
        [TestMethod]
        public void Extensions_AndDtypes()
        {
            np.fft.rfft(np.array(new[] { 1.0, 0.0, 0.0, 0.0 })).Should().BeShaped(3);
            np.linalg.det(np.array(new[,] { { 4.0, 2.0 }, { 2.0, 3.0 } })).GetDouble().Should().BeApproximately(8.0, 1e-12);
            np.zeros(new Shape(2), np.int8).typecode.Should().Be(NPTypeCode.SByte);
            np.zeros(new Shape(2), np.complex128).typecode.Should().Be(NPTypeCode.Complex);
        }

        /// <summary>The spec's complex64 dtype is still missing (#569). Fails until <c>np.dtype("complex64")</c> resolves.</summary>
        [TestMethod, OpenBugs(IssueUrl = "https://github.com/SciSharp/NumSharp/issues/560")]
        public void StillMissing_Complex64()
        {
            new Action(() => np.zeros(new Shape(2), np.dtype("complex64"))).Should().NotThrow();
        }

        /// <summary>Regression pin for the spec names the comment listed as missing (shared with #547): all exist since
        /// API-parity batch 7 added <c>np.astype</c> (2026-11-12). A failure lists the ones missing again.</summary>
        [TestMethod]
        public void StillMissing_SpecNames()
        {
            var missing = Issue547.MissingNumPy2Names.Where(n => !Issue547.NpHas(n)).ToArray();
            missing.Should().BeEmpty($"the Array API spec requires them; missing: {string.Join(", ", missing)}");
        }
    }
}
