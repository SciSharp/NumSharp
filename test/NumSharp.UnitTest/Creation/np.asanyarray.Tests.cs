using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Tests for np.asanyarray covering all built-in C# collection types.
    /// </summary>
    public class np_asanyarray_tests
    {
        #region NDArray passthrough

        [Test]
        public void NDArray_ReturnsAsIs()
        {
            var original = np.array(1, 2, 3, 4, 5);
            var result = np.asanyarray(original);

            // Should return the same instance (no copy)
            ReferenceEquals(original, result).Should().BeTrue();
        }

        [Test]
        public void NDArray_WithDtype_ReturnsConverted()
        {
            var original = np.array(1, 2, 3, 4, 5);
            var result = np.asanyarray(original, typeof(double));

            result.dtype.Should().Be(typeof(double));
            result.Should().BeShaped(5);
        }

        [Test]
        public void NDArray_WithSameDtype_ReturnsAsIs()
        {
            var original = np.array(1, 2, 3, 4, 5);
            var result = np.asanyarray(original, typeof(int));

            // Same dtype, should return same instance
            ReferenceEquals(original, result).Should().BeTrue();
        }

        #endregion

        #region Array types

        [Test]
        public void Array_1D()
        {
            var arr = new int[] { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(arr);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void Array_2D()
        {
            var arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var result = np.asanyarray(arr);

            result.Should().BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [Test]
        public void Array_WithDtype()
        {
            var arr = new int[] { 1, 2, 3 };
            var result = np.asanyarray(arr, typeof(double));

            result.dtype.Should().Be(typeof(double));
            result.Should().BeShaped(3);
        }

        #endregion

        #region Scalars

        [Test]
        public void Scalar_Int()
        {
            var result = np.asanyarray(42);

            result.Should().BeScalar().And.BeOfValues(42);
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void Scalar_Double()
        {
            var result = np.asanyarray(3.14);

            result.Should().BeScalar();
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void Scalar_Decimal()
        {
            var result = np.asanyarray(123.456m);

            result.Should().BeScalar();
            result.dtype.Should().Be(typeof(decimal));
        }

        [Test]
        public void Scalar_Bool()
        {
            var result = np.asanyarray(true);

            result.Should().BeScalar().And.BeOfValues(true);
            result.dtype.Should().Be(typeof(bool));
        }

        #endregion

        #region List<T>

        [Test]
        public void List_Int()
        {
            var list = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void List_Double()
        {
            var list = new List<double> { 1.1, 2.2, 3.3 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(3);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void List_Bool()
        {
            var list = new List<bool> { true, false, true };
            var result = np.asanyarray(list);

            result.Should().BeShaped(3).And.BeOfValues(true, false, true);
            result.dtype.Should().Be(typeof(bool));
        }

        [Test]
        public void List_Empty()
        {
            var list = new List<int>();
            var result = np.asanyarray(list);

            result.Should().BeShaped(0);
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void List_WithDtype()
        {
            var list = new List<int> { 1, 2, 3 };
            var result = np.asanyarray(list, typeof(float));

            result.dtype.Should().Be(typeof(float));
            result.Should().BeShaped(3);
        }

        #endregion

        #region IList<T> / ICollection<T> / IEnumerable<T>

        [Test]
        public void IList_Int()
        {
            IList<int> list = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void ICollection_Int()
        {
            ICollection<int> collection = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(collection);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void IEnumerable_Int()
        {
            IEnumerable<int> enumerable = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(enumerable);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void IEnumerable_FromLinq()
        {
            var enumerable = Enumerable.Range(1, 5);
            var result = np.asanyarray(enumerable);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void IEnumerable_FromLinqSelect()
        {
            var enumerable = new[] { 1, 2, 3 }.Select(x => x * 2);
            var result = np.asanyarray(enumerable);

            result.Should().BeShaped(3).And.BeOfValues(2, 4, 6);
        }

        #endregion

        #region IReadOnlyList<T> / IReadOnlyCollection<T>

        [Test]
        public void IReadOnlyList_Int()
        {
            IReadOnlyList<int> list = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void IReadOnlyCollection_Int()
        {
            IReadOnlyCollection<int> collection = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(collection);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        #endregion

        #region ReadOnlyCollection<T>

        [Test]
        public void ReadOnlyCollection_Int()
        {
            var collection = new ReadOnlyCollection<int>(new List<int> { 1, 2, 3, 4, 5 });
            var result = np.asanyarray(collection);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        #endregion

        #region LinkedList<T>

        [Test]
        public void LinkedList_Int()
        {
            var linkedList = new LinkedList<int>();
            linkedList.AddLast(1);
            linkedList.AddLast(2);
            linkedList.AddLast(3);
            var result = np.asanyarray(linkedList);

            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
        }

        #endregion

        #region HashSet<T> / SortedSet<T>

        [Test]
        public void HashSet_Int()
        {
            var set = new HashSet<int> { 3, 1, 4, 1, 5, 9 }; // Duplicates removed
            var result = np.asanyarray(set);

            result.size.Should().Be(5); // 1, 3, 4, 5, 9 (no duplicates)
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void SortedSet_Int()
        {
            var set = new SortedSet<int> { 3, 1, 4, 1, 5, 9 };
            var result = np.asanyarray(set);

            result.Should().BeShaped(5).And.BeOfValues(1, 3, 4, 5, 9); // Sorted, no duplicates
        }

        #endregion

        #region Queue<T> / Stack<T>

        [Test]
        public void Queue_Int()
        {
            var queue = new Queue<int>();
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            var result = np.asanyarray(queue);

            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
        }

        [Test]
        public void Stack_Int()
        {
            var stack = new Stack<int>();
            stack.Push(1);
            stack.Push(2);
            stack.Push(3);
            var result = np.asanyarray(stack);

            result.Should().BeShaped(3).And.BeOfValues(3, 2, 1); // LIFO order
        }

        #endregion

        #region ArraySegment<T>

        [Test]
        public void ArraySegment_Int()
        {
            var array = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var segment = new ArraySegment<int>(array, 2, 5); // Elements 2,3,4,5,6
            var result = np.asanyarray(segment);

            result.Should().BeShaped(5).And.BeOfValues(2, 3, 4, 5, 6);
        }

        [Test]
        public void ArraySegment_Empty()
        {
            var array = new int[] { 1, 2, 3 };
            var segment = new ArraySegment<int>(array, 0, 0);
            var result = np.asanyarray(segment);

            result.Should().BeShaped(0);
        }

        [Test]
        public void ArraySegment_Full()
        {
            var array = new int[] { 1, 2, 3, 4, 5 };
            var segment = new ArraySegment<int>(array);
            var result = np.asanyarray(segment);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        #endregion

        #region Memory<T> / ReadOnlyMemory<T>

        [Test]
        public void Memory_Int()
        {
            var array = new int[] { 1, 2, 3, 4, 5 };
            var memory = new Memory<int>(array, 1, 3); // Elements 2,3,4
            var result = np.asanyarray(memory);

            result.Should().BeShaped(3).And.BeOfValues(2, 3, 4);
        }

        [Test]
        public void ReadOnlyMemory_Int()
        {
            var array = new int[] { 1, 2, 3, 4, 5 };
            var memory = new ReadOnlyMemory<int>(array, 1, 3); // Elements 2,3,4
            var result = np.asanyarray(memory);

            result.Should().BeShaped(3).And.BeOfValues(2, 3, 4);
        }

        #endregion

        #region ImmutableArray<T> / ImmutableList<T>

        [Test]
        public void ImmutableArray_Int()
        {
            var immutableArray = ImmutableArray.Create(1, 2, 3, 4, 5);
            var result = np.asanyarray(immutableArray);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void ImmutableList_Int()
        {
            var immutableList = ImmutableList.Create(1, 2, 3, 4, 5);
            var result = np.asanyarray(immutableList);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [Test]
        public void ImmutableHashSet_Int()
        {
            var immutableSet = ImmutableHashSet.Create(3, 1, 4, 1, 5);
            var result = np.asanyarray(immutableSet);

            result.size.Should().Be(4); // 1, 3, 4, 5 (no duplicates)
        }

        #endregion

        #region All supported dtypes via List<T>

        [Test]
        public void List_Byte()
        {
            var list = new List<byte> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(byte));
            result.Should().BeShaped(3);
        }

        // Note: sbyte is NOT supported by NumSharp (12 supported types: bool, byte, short, ushort, int, uint, long, ulong, char, float, double, decimal)

        [Test]
        public void List_Short()
        {
            var list = new List<short> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(short));
            result.Should().BeShaped(3);
        }

        [Test]
        public void List_UShort()
        {
            var list = new List<ushort> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(ushort));
            result.Should().BeShaped(3);
        }

        [Test]
        public void List_UInt()
        {
            var list = new List<uint> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(uint));
            result.Should().BeShaped(3);
        }

        [Test]
        public void List_Long()
        {
            var list = new List<long> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(long));
            result.Should().BeShaped(3);
        }

        [Test]
        public void List_ULong()
        {
            var list = new List<ulong> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(ulong));
            result.Should().BeShaped(3);
        }

        [Test]
        public void List_Float()
        {
            var list = new List<float> { 1.1f, 2.2f, 3.3f };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(float));
            result.Should().BeShaped(3);
        }

        [Test]
        public void List_Char()
        {
            var list = new List<char> { 'a', 'b', 'c' };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(char));
            result.Should().BeShaped(3);
        }

        #endregion

        #region Error cases

        [Test]
        public void Null_ThrowsArgumentNullException()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.asanyarray(null));
        }

        [Test]
        public void UnsupportedType_ThrowsNotSupportedException()
        {
            // String collections are not supported (string is not primitive/decimal)
            var stringList = new List<string> { "a", "b", "c" };
            Assert.ThrowsException<NotSupportedException>(() => np.asanyarray(stringList));
        }

        [Test]
        public void CustomClass_ThrowsNotSupportedException()
        {
            var customObject = new object();
            Assert.ThrowsException<NotSupportedException>(() => np.asanyarray(customObject));
        }

        #endregion

        #region String special case

        [Test]
        public void String_CreatesCharArray()
        {
            var result = np.asanyarray("hello");

            result.Should().BeShaped(5);
            result.dtype.Should().Be(typeof(char));
        }

        #endregion
    }
}
