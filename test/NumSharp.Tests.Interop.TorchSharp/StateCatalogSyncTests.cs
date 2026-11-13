using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TorchSharp;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     The C# catalogue and the fixture name the same states: a recipe without an oracle answer would test nothing against
    ///     NumPy / PyTorch, and an answer without a recipe would be silently skipped.
    /// </summary>
    [TestClass]
    public class StateCatalogSyncTests : TorchTestBase
    {
        /// <summary>Every array state of the fixture has a recipe, and every recipe an array state.</summary>
        [TestMethod]
        public void ArrayStates_MatchTheFixture()
        {
            Console.WriteLine($"fixture: numpy {StateOracle.NumPyVersion}, torch {StateOracle.TorchVersion}");
            StateCatalog.ArrayStates.Keys.OrderBy(k => k, StringComparer.Ordinal)
                .Should().Equal(StateOracle.ArrayStates.Keys.OrderBy(k => k, StringComparer.Ordinal));
            StateOracle.ArrayStates.Values.Sum(rows => rows.Count).Should().BeGreaterThan(300, "every layout runs over the ten shared dtypes");
        }

        /// <summary>Every tensor state of the fixture has a recipe, and every recipe a tensor state; every dtype name is known.</summary>
        [TestMethod]
        public void TensorStates_MatchTheFixture()
        {
            StateCatalog.TensorStates.Keys.OrderBy(k => k, StringComparer.Ordinal)
                .Should().Equal(StateOracle.TensorStates.Keys.OrderBy(k => k, StringComparer.Ordinal));
            foreach (string dtype in StateOracle.TensorStates.Values.SelectMany(rows => rows.Keys).Distinct())
                ((Action)(() => StateCatalog.ScalarTypeOf(dtype))).Should().NotThrow($"the fixture's torch dtype '{dtype}' must map to a ScalarType");
        }

        /// <summary>
        ///     The fixture was recorded with the NumPy release the repository pins (<c>numpy-version.txt</c>, compiled in as
        ///     <see cref="NumPyPin.Version"/>), so the state matrix holds the bridge to that release's flags and strides.
        /// </summary>
        /// <remarks>
        ///     Compared against the pin rather than a literal: after a NumPy bump this fails until
        ///     <c>test/oracle/gen_torch_interop_states.py</c> regenerates the fixture under the new release, where a literal
        ///     would keep agreeing with the old fixture.
        /// </remarks>
        [TestMethod]
        public void Fixture_IsThePinnedNumPy()
        {
            StateOracle.NumPyVersion.Should().Be(NumPyPin.Version, "the fixture must hold the pinned NumPy's answers (numpy-version.txt)");
        }
    }
}
