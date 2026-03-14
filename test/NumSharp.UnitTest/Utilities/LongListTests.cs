using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Utilities
{
    /// <summary>
    /// Tests for LongList&lt;T&gt; long-indexed list implementation.
    /// </summary>
    public class LongListTests
    {
        #region ListHelpersLong Tests

        [Test]
        public void ListHelpersLong_GetNewCapacity_DoublesForSmallSizes()
        {
            // For sizes below the threshold, capacity should double
            long oldCapacity = 100;
            long minCapacity = 101;

            long newCapacity = ListHelpersLong.GetNewCapacity(oldCapacity, minCapacity, 4);

            Assert.AreEqual(200L, newCapacity); // Doubled
        }

        [Test]
        public void ListHelpersLong_GetNewCapacity_UsesDefaultForEmptyCollection()
        {
            // When old capacity is 0, should use default
            long oldCapacity = 0;
            long minCapacity = 1;
            long defaultCapacity = 4;

            long newCapacity = ListHelpersLong.GetNewCapacity(oldCapacity, minCapacity, defaultCapacity);

            Assert.AreEqual(4L, newCapacity);
        }

        [Test]
        public void ListHelpersLong_GetNewCapacity_Uses33PercentForLargeSizes()
        {
            // For sizes at or above the threshold, should grow by 33%
            long oldCapacity = ListHelpersLong.LargeGrowthThreshold; // 1 billion
            long minCapacity = oldCapacity + 1;

            long newCapacity = ListHelpersLong.GetNewCapacity(oldCapacity, minCapacity, 4);

            // Should grow by ~33%, not double
            long expected33Percent = oldCapacity + (oldCapacity / 3);
            Assert.IsTrue(newCapacity >= expected33Percent);
            Assert.IsTrue(newCapacity < 2 * oldCapacity); // Should NOT double
        }

        [Test]
        public void ListHelpersLong_GetNewCapacity_RespectsMinCapacity()
        {
            // If min capacity is larger than computed capacity, use min
            long oldCapacity = 4;
            long minCapacity = 1000;

            long newCapacity = ListHelpersLong.GetNewCapacity(oldCapacity, minCapacity, 4);

            Assert.IsTrue(newCapacity >= 1000);
        }

        [Test]
        public void ListHelpersLong_GetNewCapacity_CapsAtMaxArrayLength()
        {
            // Should not exceed MaxArrayLength
            long oldCapacity = ListHelpersLong.MaxArrayLength - 100;
            long minCapacity = oldCapacity + 1;

            long newCapacity = ListHelpersLong.GetNewCapacity(oldCapacity, minCapacity, 4);

            Assert.IsTrue(newCapacity <= ListHelpersLong.MaxArrayLength);
        }

        [Test]
        public void ListHelpersLong_LargeGrowthThreshold_IsOneBillion()
        {
            Assert.AreEqual(1_000_000_000L, ListHelpersLong.LargeGrowthThreshold);
        }

        #endregion

        #region Constructor Tests

        [Test]
        public void LongList_DefaultConstructor_CreatesEmptyList()
        {
            var list = new LongList<int>();

            Assert.AreEqual(0L, list.LongCount);
            Assert.AreEqual(0L, list.Capacity);
        }

        [Test]
        public void LongList_CapacityConstructor_SetsCapacity()
        {
            var list = new LongList<int>(100);

            Assert.AreEqual(0L, list.LongCount);
            Assert.AreEqual(100L, list.Capacity);
        }

        [Test]
        public void LongList_CapacityConstructor_ThrowsForNegative()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
            {
                var list = new LongList<int>(-1);
            });
        }

        [Test]
        public void LongList_CollectionConstructor_CopiesElements()
        {
            var source = new[] { 1, 2, 3, 4, 5 };
            var list = new LongList<int>(source);

            Assert.AreEqual(5L, list.LongCount);
            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(source[i], list[i]);
            }
        }

        [Test]
        public void LongList_CollectionConstructor_HandlesEnumerable()
        {
            IEnumerable<int> source = Enumerable.Range(1, 10);
            var list = new LongList<int>(source);

            Assert.AreEqual(10L, list.LongCount);
        }

        #endregion

        #region Add/Insert Tests

        [Test]
        public void LongList_Add_IncreasesCount()
        {
            var list = new LongList<int>();

            list.Add(42);

            Assert.AreEqual(1L, list.LongCount);
            Assert.AreEqual(42, list[0]);
        }

        [Test]
        public void LongList_Add_GrowsCapacity()
        {
            var list = new LongList<int>(2);
            list.Add(1);
            list.Add(2);
            list.Add(3); // Should trigger growth

            Assert.AreEqual(3L, list.LongCount);
            Assert.IsTrue(list.Capacity >= 3);
        }

        [Test]
        public void LongList_AddRange_AddsAllElements()
        {
            var list = new LongList<int>();
            list.AddRange(new[] { 1, 2, 3, 4, 5 });

            Assert.AreEqual(5L, list.LongCount);
        }

        [Test]
        public void LongList_Insert_AtBeginning()
        {
            var list = new LongList<int>(new[] { 2, 3, 4 });
            list.Insert(0, 1);

            Assert.AreEqual(1, list[0]);
            Assert.AreEqual(2, list[1]);
            Assert.AreEqual(4L, list.LongCount);
        }

        [Test]
        public void LongList_Insert_AtEnd()
        {
            var list = new LongList<int>(new[] { 1, 2, 3 });
            list.Insert(3, 4);

            Assert.AreEqual(4, list[3]);
            Assert.AreEqual(4L, list.LongCount);
        }

        [Test]
        public void LongList_Insert_InMiddle()
        {
            var list = new LongList<int>(new[] { 1, 3, 4 });
            list.Insert(1, 2);

            Assert.AreEqual(1, list[0]);
            Assert.AreEqual(2, list[1]);
            Assert.AreEqual(3, list[2]);
            Assert.AreEqual(4L, list.LongCount);
        }

        [Test]
        public void LongList_InsertRange_InsertsAllElements()
        {
            var list = new LongList<int>(new[] { 1, 5 });
            list.InsertRange(1, new[] { 2, 3, 4 });

            Assert.AreEqual(5L, list.LongCount);
            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(i + 1, list[i]);
            }
        }

        #endregion

        #region Remove Tests

        [Test]
        public void LongList_Remove_RemovesFirstOccurrence()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 2, 4 });

            bool removed = list.Remove(2);

            Assert.IsTrue(removed);
            Assert.AreEqual(4L, list.LongCount);
            Assert.AreEqual(3, list[1]); // Second 2 shifted
        }

        [Test]
        public void LongList_Remove_ReturnsFalseIfNotFound()
        {
            var list = new LongList<int>(new[] { 1, 2, 3 });

            bool removed = list.Remove(99);

            Assert.IsFalse(removed);
            Assert.AreEqual(3L, list.LongCount);
        }

        [Test]
        public void LongList_RemoveAt_RemovesCorrectElement()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            list.RemoveAt(2);

            Assert.AreEqual(4L, list.LongCount);
            Assert.AreEqual(4, list[2]);
        }

        [Test]
        public void LongList_RemoveRange_RemovesCorrectRange()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            list.RemoveRange(1, 3);

            Assert.AreEqual(2L, list.LongCount);
            Assert.AreEqual(1, list[0]);
            Assert.AreEqual(5, list[1]);
        }

        [Test]
        public void LongList_RemoveAll_RemovesMatchingElements()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5, 6 });

            long removed = list.RemoveAll(x => x % 2 == 0);

            Assert.AreEqual(3L, removed);
            Assert.AreEqual(3L, list.LongCount);
            Assert.AreEqual(1, list[0]);
            Assert.AreEqual(3, list[1]);
            Assert.AreEqual(5, list[2]);
        }

        [Test]
        public void LongList_Clear_RemovesAllElements()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            list.Clear();

            Assert.AreEqual(0L, list.LongCount);
        }

        #endregion

        #region Search Tests

        [Test]
        public void LongList_Contains_FindsElement()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            Assert.IsTrue(list.Contains(3));
            Assert.IsFalse(list.Contains(99));
        }

        [Test]
        public void LongList_IndexOf_ReturnsCorrectIndex()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 2, 4 });

            Assert.AreEqual(1L, list.IndexOf(2));
            Assert.AreEqual(-1L, list.IndexOf(99));
        }

        [Test]
        public void LongList_LastIndexOf_ReturnsLastOccurrence()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 2, 4 });

            Assert.AreEqual(3L, list.LastIndexOf(2));
        }

        [Test]
        public void LongList_BinarySearch_FindsElement()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            long index = list.BinarySearch(3);

            Assert.AreEqual(2L, index);
        }

        [Test]
        public void LongList_Find_ReturnsFirstMatch()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            var found = list.Find(x => x > 2);

            Assert.AreEqual(3, found);
        }

        [Test]
        public void LongList_FindAll_ReturnsAllMatches()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5, 6 });

            var evens = list.FindAll(x => x % 2 == 0);

            Assert.AreEqual(3L, evens.LongCount);
        }

        [Test]
        public void LongList_FindIndex_ReturnsCorrectIndex()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            long index = list.FindIndex(x => x > 2);

            Assert.AreEqual(2L, index);
        }

        [Test]
        public void LongList_FindLast_ReturnsLastMatch()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            var found = list.FindLast(x => x < 4);

            Assert.AreEqual(3, found);
        }

        [Test]
        public void LongList_Exists_ReturnsCorrectResult()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            Assert.IsTrue(list.Exists(x => x == 3));
            Assert.IsFalse(list.Exists(x => x == 99));
        }

        #endregion

        #region Indexer Tests

        [Test]
        public void LongList_Indexer_GetAndSet()
        {
            var list = new LongList<int>(new[] { 1, 2, 3 });

            Assert.AreEqual(2, list[1]);

            list[1] = 99;
            Assert.AreEqual(99, list[1]);
        }

        [Test]
        public void LongList_Indexer_ThrowsForOutOfRange()
        {
            var list = new LongList<int>(new[] { 1, 2, 3 });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
            {
                var _ = list[5];
            });
        }

        [Test]
        public void LongList_Indexer_AcceptsLongIndex()
        {
            var list = new LongList<int>();
            for (int i = 0; i < 100; i++)
                list.Add(i);

            long index = 50L;
            Assert.AreEqual(50, list[index]);
        }

        #endregion

        #region Copy and Transform Tests

        [Test]
        public void LongList_ToArray_ReturnsCorrectArray()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            var array = list.ToArray();

            Assert.AreEqual(5, array.Length);
            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(i + 1, array[i]);
            }
        }

        [Test]
        public void LongList_CopyTo_CopiesElements()
        {
            var list = new LongList<int>(new[] { 1, 2, 3 });
            var array = new int[5];

            list.CopyTo(array, 1);

            Assert.AreEqual(0, array[0]);
            Assert.AreEqual(1, array[1]);
            Assert.AreEqual(2, array[2]);
            Assert.AreEqual(3, array[3]);
        }

        [Test]
        public void LongList_GetRange_ReturnsCorrectSublist()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            var range = list.GetRange(1, 3);

            Assert.AreEqual(3L, range.LongCount);
            Assert.AreEqual(2, range[0]);
            Assert.AreEqual(3, range[1]);
            Assert.AreEqual(4, range[2]);
        }

        [Test]
        public void LongList_ConvertAll_TransformsElements()
        {
            var list = new LongList<int>(new[] { 1, 2, 3 });

            var strings = list.ConvertAll(x => x.ToString());

            Assert.AreEqual(3L, strings.LongCount);
            Assert.AreEqual("1", strings[0]);
        }

        #endregion

        #region Sort and Reverse Tests

        [Test]
        public void LongList_Sort_SortsElements()
        {
            var list = new LongList<int>(new[] { 5, 2, 4, 1, 3 });

            list.Sort();

            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(i + 1, list[i]);
            }
        }

        [Test]
        public void LongList_Reverse_ReversesElements()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            list.Reverse();

            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(5 - i, list[i]);
            }
        }

        [Test]
        public void LongList_Sort_WithComparer()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            // Sort descending
            list.Sort(Comparer<int>.Create((a, b) => b.CompareTo(a)));

            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(5 - i, list[i]);
            }
        }

        #endregion

        #region Iteration Tests

        [Test]
        public void LongList_ForEach_IteratesAllElements()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });
            var sum = 0;

            list.ForEach(x => sum += x);

            Assert.AreEqual(15, sum);
        }

        [Test]
        public void LongList_TrueForAll_ReturnsCorrectResult()
        {
            var list = new LongList<int>(new[] { 2, 4, 6, 8 });

            Assert.IsTrue(list.TrueForAll(x => x % 2 == 0));
            Assert.IsFalse(list.TrueForAll(x => x > 5));
        }

        [Test]
        public void LongList_Enumerator_IteratesAllElements()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });
            var enumerated = new List<int>();

            foreach (var item in list)
            {
                enumerated.Add(item);
            }

            Assert.AreEqual(5, enumerated.Count);
            for (int i = 0; i < 5; i++)
            {
                Assert.AreEqual(i + 1, enumerated[i]);
            }
        }

        [Test]
        public void LongList_Enumerator_ThrowsOnModification()
        {
            var list = new LongList<int>(new[] { 1, 2, 3 });

            Assert.ThrowsException<InvalidOperationException>(() =>
            {
                foreach (var item in list)
                {
                    list.Add(99);
                }
            });
        }

        #endregion

        #region Capacity Tests

        [Test]
        public void LongList_EnsureCapacity_IncreasesCapacity()
        {
            var list = new LongList<int>();

            list.EnsureCapacity(100);

            Assert.IsTrue(list.Capacity >= 100);
        }

        [Test]
        public void LongList_TrimExcess_ReducesCapacity()
        {
            var list = new LongList<int>(100);
            for (int i = 0; i < 10; i++)
                list.Add(i);

            list.TrimExcess();

            Assert.AreEqual(10L, list.Capacity);
        }

        [Test]
        public void LongList_Capacity_CanBeSetDirectly()
        {
            var list = new LongList<int>();
            list.Add(1);
            list.Add(2);

            list.Capacity = 100;

            Assert.AreEqual(100L, list.Capacity);
            Assert.AreEqual(2L, list.LongCount);
        }

        [Test]
        public void LongList_Capacity_ThrowsIfLessThanCount()
        {
            var list = new LongList<int>(new[] { 1, 2, 3, 4, 5 });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
            {
                list.Capacity = 3;
            });
        }

        #endregion

        #region LongCount vs Count Tests

        [Test]
        public void LongList_Count_ReturnsIntCount()
        {
            var list = new LongList<int>();
            for (int i = 0; i < 100; i++)
                list.Add(i);

            Assert.AreEqual(100, list.Count);
            Assert.AreEqual(100L, list.LongCount);
        }

        #endregion

        #region Growth Strategy Verification

        [Test]
        public void LongList_GrowthStrategy_DoublesForSmallLists()
        {
            var list = new LongList<int>(10);

            // Fill the list
            for (int i = 0; i < 10; i++)
                list.Add(i);

            // Trigger growth
            list.Add(10);

            // Should have doubled to 20
            Assert.AreEqual(20L, list.Capacity);
        }

        [Test]
        public void LongList_GrowthStrategy_IncreasesCapacityOnDemand()
        {
            var list = new LongList<int>();

            // Add enough elements to trigger multiple growths
            for (int i = 0; i < 1000; i++)
            {
                list.Add(i);
            }

            Assert.AreEqual(1000L, list.LongCount);
            Assert.IsTrue(list.Capacity >= 1000);
        }

        #endregion

        #region Edge Cases

        [Test]
        public void LongList_EmptyList_Operations()
        {
            var list = new LongList<int>();

            Assert.IsFalse(list.Contains(1));
            Assert.AreEqual(-1L, list.IndexOf(1));
            Assert.AreEqual(-1L, list.LastIndexOf(1));
            Assert.AreEqual(0, list.ToArray().Length);
            Assert.AreEqual(default(int), list.Find(x => true));
        }

        [Test]
        public void LongList_SingleElement_Operations()
        {
            var list = new LongList<int>(new[] { 42 });

            Assert.IsTrue(list.Contains(42));
            Assert.AreEqual(0L, list.IndexOf(42));
            Assert.AreEqual(0L, list.LastIndexOf(42));
            Assert.AreEqual(42, list[0]);
        }

        [Test]
        public void LongList_LargeCapacity_Works()
        {
            // Test with a reasonably large capacity
            long capacity = 10_000_000; // 10 million
            var list = new LongList<int>(capacity);

            Assert.AreEqual(capacity, list.Capacity);
            Assert.AreEqual(0L, list.LongCount);

            // Add some elements
            for (int i = 0; i < 1000; i++)
            {
                list.Add(i);
            }

            Assert.AreEqual(1000L, list.LongCount);
        }

        #endregion
    }
}
