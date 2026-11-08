using System;
using System.Collections.Generic;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     The indexer never writes into the caller's key array, as NumPy never modifies its index object: after
    ///     <c>key = ([True, False, True], slice(None)); a[key]</c> the list is still the list. Expected values probed against
    ///     NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Until 2026-10-29 the general path converted a <c>bool[]</c>, a <c>List&lt;long&gt;</c> or a <c>List&lt;bool&gt;</c>
    ///     entry IN the caller's array (<c>NormalizeIndexInputs</c>), and the converted arrays belonged to the
    ///     <c>[NDScoped]</c> dispatch: the key came back holding disposed arrays whose blocks had been released, which the
    ///     next call with the same key read (found while auditing the indexer for P3 batch 8 of the np-layer dtype plan).
    /// </remarks>
    [TestClass]
    public class index_key_unchanged_numpy_port_tests
    {
        /// <summary>The three entry kinds the general path converts, each beside a slice.</summary>
        /// <returns>Fresh keys, one per kind.</returns>
        private static IEnumerable<object[]> Keys()
        {
            yield return new object[] { new bool[] { true, false, true }, Slice.All };
            yield return new object[] { new List<long> { 0, 2 }, Slice.All };
            yield return new object[] { new List<bool> { true, false, true }, Slice.All };
        }

        /// <summary>A get leaves every entry of the key the object the caller put there, and the key selects again.</summary>
        [TestMethod]
        public void Get_LeavesTheKeyUnchanged()
        {
            foreach (var key in Keys())
            {
                var a = np.arange(12.0).reshape(3, 4);
                object entry = key[0];
                a[key].flatten().ToArray<double>().Should().Equal(0, 1, 2, 3, 8, 9, 10, 11);
                key[0].Should().BeSameAs(entry);
                a[key].flatten().ToArray<double>().Should().Equal(0, 1, 2, 3, 8, 9, 10, 11);
                key[0].Should().BeSameAs(entry);
            }
        }

        /// <summary>A store through the indexer and through <c>__setitem__</c> leaves the key unchanged too.</summary>
        [TestMethod]
        public void Store_LeavesTheKeyUnchanged()
        {
            foreach (var key in Keys())
            {
                var a = np.arange(12.0).reshape(3, 4);
                object entry = key[0];
                a[key] = np.array(-1.0);
                key[0].Should().BeSameAs(entry);
                a.__setitem__(key, -2.0);
                key[0].Should().BeSameAs(entry);
                a.__setitem__(key, new object[] { 7.0, 8.0, 9.0, 10.0 });
                key[0].Should().BeSameAs(entry);
                a.flatten().ToArray<double>().Should().Equal(7, 8, 9, 10, 4, 5, 6, 7, 7, 8, 9, 10);
            }
        }
    }
}
