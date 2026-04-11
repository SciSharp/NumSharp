using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities
{

/// <summary>
/// Tests for the long-indexed Hashset implementation.
/// Verifies that the Hashset supports indexing beyond int.MaxValue
/// and uses 33% growth for large collections.
/// </summary>
public class HashsetLongIndexingTests
{
    #region Basic Functionality Tests

    [Test]
    public void Hashset_BasicAddContains()
    {
        var set = new Hashset<int>();
        Assert.IsTrue(set.Add(1));
        Assert.IsTrue(set.Add(2));
        Assert.IsTrue(set.Add(3));
        Assert.IsFalse(set.Add(1)); // Duplicate

        Assert.IsTrue(set.Contains(1));
        Assert.IsTrue(set.Contains(2));
        Assert.IsTrue(set.Contains(3));
        Assert.IsFalse(set.Contains(4));

        Assert.AreEqual(3, set.Count);
        Assert.AreEqual(3L, set.LongCount);
    }

    [Test]
    public void Hashset_Remove()
    {
        var set = new Hashset<int>(new[] { 1, 2, 3, 4, 5 });

        Assert.IsTrue(set.Remove(3));
        Assert.IsFalse(set.Contains(3));
        Assert.AreEqual(4, set.Count);

        Assert.IsFalse(set.Remove(3)); // Already removed
        Assert.AreEqual(4, set.Count);
    }

    [Test]
    public void Hashset_Clear()
    {
        var set = new Hashset<int>(new[] { 1, 2, 3, 4, 5 });
        Assert.AreEqual(5, set.Count);

        set.Clear();
        Assert.AreEqual(0, set.Count);
        Assert.AreEqual(0L, set.LongCount);
        Assert.IsFalse(set.Contains(1));
    }

    [Test]
    public void Hashset_Enumeration()
    {
        var set = new Hashset<int>(new[] { 1, 2, 3 });
        var items = set.ToArray();

        Assert.AreEqual(3, items.Length);
        CollectionAssert.AreEquivalent(new[] { 1, 2, 3 }, items);
    }

    #endregion

    #region Long Index Support Tests

    [Test]
    public void Hashset_LongCount_ReturnsSameAsCount_WhenSmall()
    {
        var set = new Hashset<int>();
        for (int i = 0; i < 1000; i++)
        {
            set.Add(i);
        }

        Assert.AreEqual(1000, set.Count);
        Assert.AreEqual(1000L, set.LongCount);
    }

    [Test]
    public void Hashset_LongCapacityConstructor()
    {
        // Create with long capacity
        var set = new Hashset<int>((long)1_000_000);
        Assert.AreEqual(0, set.Count);

        for (int i = 0; i < 100; i++)
        {
            set.Add(i);
        }
        Assert.AreEqual(100, set.Count);
    }

    [Test]
    public void Hashset_Slot_UsesLongNext()
    {
        // This test verifies that the internal Slot structure uses long for next pointer
        // We can't directly test the internal structure, but we can verify behavior
        // by adding many elements and ensuring proper chaining

        var set = new Hashset<int>();
        const int count = 100_000;

        for (int i = 0; i < count; i++)
        {
            set.Add(i);
        }

        Assert.AreEqual(count, set.Count);
        Assert.AreEqual((long)count, set.LongCount);

        // Verify all elements are present
        for (int i = 0; i < count; i++)
        {
            Assert.IsTrue(set.Contains(i), $"Element {i} should be in the set");
        }
    }

    #endregion

    #region HashHelpersLong Tests

    [Test]
    public void HashHelpersLong_IsPrime()
    {
        // Known primes
        Assert.IsTrue(HashHelpersLong.IsPrime(2));
        Assert.IsTrue(HashHelpersLong.IsPrime(3));
        Assert.IsTrue(HashHelpersLong.IsPrime(5));
        Assert.IsTrue(HashHelpersLong.IsPrime(7));
        Assert.IsTrue(HashHelpersLong.IsPrime(11));
        Assert.IsTrue(HashHelpersLong.IsPrime(13));
        Assert.IsTrue(HashHelpersLong.IsPrime(7199369)); // From primes array

        // Non-primes
        Assert.IsFalse(HashHelpersLong.IsPrime(1));
        Assert.IsFalse(HashHelpersLong.IsPrime(4));
        Assert.IsFalse(HashHelpersLong.IsPrime(6));
        Assert.IsFalse(HashHelpersLong.IsPrime(9));
        Assert.IsFalse(HashHelpersLong.IsPrime(15));

        // Large primes beyond int.MaxValue
        Assert.IsTrue(HashHelpersLong.IsPrime(4252464407L)); // Valid large prime from extended array
    }

    [Test]
    [OpenBugs] // HashHelpersLong.primes array contains non-prime values
    public void HashHelpersLong_GetPrime()
    {
        // Should return the prime from the table for small values
        Assert.AreEqual(3L, HashHelpersLong.GetPrime(0));
        Assert.AreEqual(3L, HashHelpersLong.GetPrime(3));
        Assert.AreEqual(7L, HashHelpersLong.GetPrime(4));
        Assert.AreEqual(7L, HashHelpersLong.GetPrime(7));
        Assert.AreEqual(11L, HashHelpersLong.GetPrime(8));

        // Large value
        var result = HashHelpersLong.GetPrime(10_000_000_000L);
        Assert.IsTrue(result >= 10_000_000_000L);
        Assert.IsTrue(HashHelpersLong.IsPrime(result));
    }

    [Test]
    public void HashHelpersLong_ExpandPrime_DoublesForSmallSizes()
    {
        // For sizes below 1 billion, should approximately double
        long oldSize = 1000;
        long newSize = HashHelpersLong.ExpandPrime(oldSize);

        // Should be at least 2x old size (then rounded to prime)
        Assert.IsTrue(newSize >= 2 * oldSize, $"Expected newSize >= {2 * oldSize}, got {newSize}");
    }

    [Test]
    public void HashHelpersLong_ExpandPrime_Uses33PercentForLargeSizes()
    {
        // For sizes at or above 1 billion, should use 33% growth
        long oldSize = HashHelpersLong.LargeGrowthThreshold; // 1 billion
        long newSize = HashHelpersLong.ExpandPrime(oldSize);

        // Should be approximately 1.33x old size (then rounded to prime)
        // 33% of 1 billion is ~333 million
        long expectedMin = oldSize + (oldSize / 3);
        Assert.IsTrue(newSize >= expectedMin,
            $"Expected newSize >= {expectedMin} (33% growth), got {newSize}");

        // Should NOT be close to 2x (would indicate doubling instead of 33%)
        Assert.IsTrue(newSize < 2 * oldSize,
            $"Expected newSize < {2 * oldSize} (should use 33% growth, not doubling), got {newSize}");
    }

    [Test]
    [OpenBugs] // HashHelpersLong.primes array contains non-prime values
    public void HashHelpersLong_ExpandPrime_ProgressiveGrowthTest()
    {
        // Verify growth pattern changes at threshold
        long[] sizes = { 100_000, 1_000_000, 100_000_000, 500_000_000,
                         1_000_000_000, 2_000_000_000, 5_000_000_000 };

        for (int i = 0; i < sizes.Length; i++)
        {
            long oldSize = sizes[i];
            long newSize = HashHelpersLong.ExpandPrime(oldSize);

            if (oldSize < HashHelpersLong.LargeGrowthThreshold)
            {
                // Should double
                Assert.IsTrue(newSize >= 2 * oldSize,
                    $"Size {oldSize}: Expected doubling, newSize={newSize}");
            }
            else
            {
                // Should use 33% growth
                long expectedMin = oldSize + (oldSize / 3);
                Assert.IsTrue(newSize >= expectedMin,
                    $"Size {oldSize}: Expected 33% growth (>={expectedMin}), got {newSize}");
                Assert.IsTrue(newSize < 2 * oldSize,
                    $"Size {oldSize}: Should not double, got {newSize}");
            }
        }
    }

    #endregion

    #region BitHelperLong Tests

    [Test]
    public void BitHelperLong_MarkAndCheck()
    {
        long[] bitArray = new long[10];
        var helper = new BitHelperLong(bitArray, 10);

        // Mark some bits
        helper.MarkBit(0);
        helper.MarkBit(63);  // Last bit of first long
        helper.MarkBit(64);  // First bit of second long
        helper.MarkBit(500); // Later bit

        // Verify marked bits
        Assert.IsTrue(helper.IsMarked(0));
        Assert.IsTrue(helper.IsMarked(63));
        Assert.IsTrue(helper.IsMarked(64));
        Assert.IsTrue(helper.IsMarked(500));

        // Verify unmarked bits
        Assert.IsFalse(helper.IsMarked(1));
        Assert.IsFalse(helper.IsMarked(62));
        Assert.IsFalse(helper.IsMarked(65));
        Assert.IsFalse(helper.IsMarked(100));
    }

    [Test]
    public void BitHelperLong_ToLongArrayLength()
    {
        // 64 bits per long
        Assert.AreEqual(0L, BitHelperLong.ToLongArrayLength(0));
        Assert.AreEqual(1L, BitHelperLong.ToLongArrayLength(1));
        Assert.AreEqual(1L, BitHelperLong.ToLongArrayLength(64));
        Assert.AreEqual(2L, BitHelperLong.ToLongArrayLength(65));
        Assert.AreEqual(2L, BitHelperLong.ToLongArrayLength(128));
        Assert.AreEqual(3L, BitHelperLong.ToLongArrayLength(129));

        // Large values
        Assert.AreEqual(156_250_000L, BitHelperLong.ToLongArrayLength(10_000_000_000L));
    }

    #endregion

    #region Set Operations Tests

    [Test]
    public void Hashset_UnionWith()
    {
        var set1 = new Hashset<int>(new[] { 1, 2, 3 });
        var set2 = new[] { 3, 4, 5 };

        set1.UnionWith(set2);

        Assert.AreEqual(5, set1.Count);
        CollectionAssert.AreEquivalent(new[] { 1, 2, 3, 4, 5 }, set1.ToArray());
    }

    [Test]
    public void Hashset_IntersectWith()
    {
        var set1 = new Hashset<int>(new[] { 1, 2, 3, 4 });
        var set2 = new[] { 2, 3, 5 };

        set1.IntersectWith(set2);

        Assert.AreEqual(2, set1.Count);
        CollectionAssert.AreEquivalent(new[] { 2, 3 }, set1.ToArray());
    }

    [Test]
    public void Hashset_ExceptWith()
    {
        var set1 = new Hashset<int>(new[] { 1, 2, 3, 4 });
        var set2 = new[] { 2, 3, 5 };

        set1.ExceptWith(set2);

        Assert.AreEqual(2, set1.Count);
        CollectionAssert.AreEquivalent(new[] { 1, 4 }, set1.ToArray());
    }

    [Test]
    public void Hashset_SymmetricExceptWith()
    {
        var set1 = new Hashset<int>(new[] { 1, 2, 3 });
        var set2 = new[] { 2, 3, 4 };

        set1.SymmetricExceptWith(set2);

        Assert.AreEqual(2, set1.Count);
        CollectionAssert.AreEquivalent(new[] { 1, 4 }, set1.ToArray());
    }

    #endregion

    #region Stress Tests

    [Test]
    public void Hashset_MediumScaleTest()
    {
        // Test with 1 million elements to verify correct operation
        const int count = 1_000_000;
        var set = new Hashset<int>(count);

        for (int i = 0; i < count; i++)
        {
            Assert.IsTrue(set.Add(i));
        }

        Assert.AreEqual(count, set.Count);
        Assert.AreEqual((long)count, set.LongCount);

        // Verify random samples
        var random = new Random(42);
        for (int i = 0; i < 1000; i++)
        {
            int value = random.Next(count);
            Assert.IsTrue(set.Contains(value), $"Should contain {value}");
        }

        // Verify non-existing elements
        for (int i = 0; i < 100; i++)
        {
            Assert.IsFalse(set.Contains(count + i), $"Should not contain {count + i}");
        }
    }

    [Test]
    public void Hashset_RemoveWhereWithLongReturn()
    {
        var set = new Hashset<int>(Enumerable.Range(0, 1000));

        // Remove all even numbers
        long removed = set.RemoveWhere(x => x % 2 == 0);

        Assert.AreEqual(500L, removed);
        Assert.AreEqual(500, set.Count);

        // Verify only odd numbers remain
        foreach (var item in set)
        {
            Assert.IsTrue(item % 2 == 1, $"Even number {item} should have been removed");
        }
    }

    #endregion

    #region CopyTo Tests

    [Test]
    public void Hashset_CopyTo_Array()
    {
        var set = new Hashset<int>(new[] { 1, 2, 3, 4, 5 });
        var array = new int[5];

        set.CopyTo(array);

        CollectionAssert.AreEquivalent(new[] { 1, 2, 3, 4, 5 }, array);
    }

    [Test]
    public void Hashset_CopyTo_LongOffsetAndCount()
    {
        var set = new Hashset<int>(new[] { 1, 2, 3, 4, 5 });
        var array = new int[10];

        // Copy 3 elements starting at offset 2 in the array
        set.CopyTo(array, 2L, 3L);

        Assert.AreEqual(0, array[0]); // Not filled
        Assert.AreEqual(0, array[1]); // Not filled
        // Elements 2-4 should have values
        Assert.AreNotEqual(0, array[2]);
        Assert.AreNotEqual(0, array[3]);
        Assert.AreNotEqual(0, array[4]);
        Assert.AreEqual(0, array[5]); // Not filled
    }

    #endregion

    #region Edge Cases

    [Test]
    public void Hashset_TrimExcess()
    {
        var set = new Hashset<int>(1000);
        for (int i = 0; i < 10; i++)
        {
            set.Add(i);
        }

        set.TrimExcess();

        Assert.AreEqual(10, set.Count);
        // All elements should still be present
        for (int i = 0; i < 10; i++)
        {
            Assert.IsTrue(set.Contains(i));
        }
    }

    [Test]
    public void Hashset_TryGetValue()
    {
        var set = new Hashset<string>(new[] { "hello", "world" });

        Assert.IsTrue(set.TryGetValue("hello", out var value1));
        Assert.AreEqual("hello", value1);

        Assert.IsFalse(set.TryGetValue("foo", out var value2));
        Assert.IsNull(value2);
    }

    [Test]
    public void Hashset_SetEquals()
    {
        var set1 = new Hashset<int>(new[] { 1, 2, 3 });
        var set2 = new[] { 3, 2, 1 };
        var set3 = new[] { 1, 2, 3, 4 };

        Assert.IsTrue(set1.SetEquals(set2));
        Assert.IsFalse(set1.SetEquals(set3));
    }

    [Test]
    public void Hashset_Overlaps()
    {
        var set1 = new Hashset<int>(new[] { 1, 2, 3 });

        Assert.IsTrue(set1.Overlaps(new[] { 3, 4, 5 }));
        Assert.IsFalse(set1.Overlaps(new[] { 4, 5, 6 }));
    }

    [Test]
    public void Hashset_IsSubsetOf()
    {
        var set1 = new Hashset<int>(new[] { 1, 2 });
        var set2 = new[] { 1, 2, 3 };
        var set3 = new[] { 1, 4 };

        Assert.IsTrue(set1.IsSubsetOf(set2));
        Assert.IsFalse(set1.IsSubsetOf(set3));
    }

    [Test]
    public void Hashset_IsSupersetOf()
    {
        var set1 = new Hashset<int>(new[] { 1, 2, 3 });
        var set2 = new[] { 1, 2 };
        var set3 = new[] { 1, 4 };

        Assert.IsTrue(set1.IsSupersetOf(set2));
        Assert.IsFalse(set1.IsSupersetOf(set3));
    }

    #endregion
}
}
