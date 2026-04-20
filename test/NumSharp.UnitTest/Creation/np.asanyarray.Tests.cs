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
    [TestClass]
    public class np_asanyarray_tests
    {
        #region NDArray passthrough

        [TestMethod]
        public void NDArray_ReturnsAsIs()
        {
            var original = np.array(1, 2, 3, 4, 5);
            var result = np.asanyarray(original);

            // Should return the same instance (no copy)
            ReferenceEquals(original, result).Should().BeTrue();
        }

        [TestMethod]
        public void NDArray_WithDtype_ReturnsConverted()
        {
            var original = np.array(1, 2, 3, 4, 5);
            var result = np.asanyarray(original, typeof(double));

            result.dtype.Should().Be(typeof(double));
            result.Should().BeShaped(5);
        }

        [TestMethod]
        public void NDArray_WithSameDtype_ReturnsAsIs()
        {
            var original = np.array(1, 2, 3, 4, 5);
            var result = np.asanyarray(original, typeof(int));

            // Same dtype, should return same instance
            ReferenceEquals(original, result).Should().BeTrue();
        }

        #endregion

        #region Array types

        [TestMethod]
        public void Array_1D()
        {
            var arr = new int[] { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(arr);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Array_2D()
        {
            var arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var result = np.asanyarray(arr);

            result.Should().BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void Array_WithDtype()
        {
            var arr = new int[] { 1, 2, 3 };
            var result = np.asanyarray(arr, typeof(double));

            result.dtype.Should().Be(typeof(double));
            result.Should().BeShaped(3);
        }

        #endregion

        #region Scalars

        [TestMethod]
        public void Scalar_Int()
        {
            var result = np.asanyarray(42);

            result.Should().BeScalar().And.BeOfValues(42);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void Scalar_Double()
        {
            var result = np.asanyarray(3.14);

            result.Should().BeScalar();
            result.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Scalar_Decimal()
        {
            var result = np.asanyarray(123.456m);

            result.Should().BeScalar();
            result.dtype.Should().Be(typeof(decimal));
        }

        [TestMethod]
        public void Scalar_Bool()
        {
            var result = np.asanyarray(true);

            result.Should().BeScalar().And.BeOfValues(true);
            result.dtype.Should().Be(typeof(bool));
        }

        #endregion

        #region List<T>

        [TestMethod]
        public void List_Int()
        {
            var list = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void List_Double()
        {
            var list = new List<double> { 1.1, 2.2, 3.3 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(3);
            result.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void List_Bool()
        {
            var list = new List<bool> { true, false, true };
            var result = np.asanyarray(list);

            result.Should().BeShaped(3).And.BeOfValues(true, false, true);
            result.dtype.Should().Be(typeof(bool));
        }

        [TestMethod]
        public void List_Empty()
        {
            var list = new List<int>();
            var result = np.asanyarray(list);

            result.Should().BeShaped(0);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void List_WithDtype()
        {
            var list = new List<int> { 1, 2, 3 };
            var result = np.asanyarray(list, typeof(float));

            result.dtype.Should().Be(typeof(float));
            result.Should().BeShaped(3);
        }

        #endregion

        #region IList<T> / ICollection<T> / IEnumerable<T>

        [TestMethod]
        public void IList_Int()
        {
            IList<int> list = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void ICollection_Int()
        {
            ICollection<int> collection = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(collection);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void IEnumerable_Int()
        {
            IEnumerable<int> enumerable = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(enumerable);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void IEnumerable_FromLinq()
        {
            var enumerable = Enumerable.Range(1, 5);
            var result = np.asanyarray(enumerable);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void IEnumerable_FromLinqSelect()
        {
            var enumerable = new[] { 1, 2, 3 }.Select(x => x * 2);
            var result = np.asanyarray(enumerable);

            result.Should().BeShaped(3).And.BeOfValues(2, 4, 6);
        }

        #endregion

        #region IReadOnlyList<T> / IReadOnlyCollection<T>

        [TestMethod]
        public void IReadOnlyList_Int()
        {
            IReadOnlyList<int> list = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(list);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void IReadOnlyCollection_Int()
        {
            IReadOnlyCollection<int> collection = new List<int> { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(collection);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        #endregion

        #region ReadOnlyCollection<T>

        [TestMethod]
        public void ReadOnlyCollection_Int()
        {
            var collection = new ReadOnlyCollection<int>(new List<int> { 1, 2, 3, 4, 5 });
            var result = np.asanyarray(collection);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        #endregion

        #region LinkedList<T>

        [TestMethod]
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

        [TestMethod]
        public void HashSet_Int()
        {
            var set = new HashSet<int> { 3, 1, 4, 1, 5, 9 }; // Duplicates removed
            var result = np.asanyarray(set);

            result.size.Should().Be(5); // 1, 3, 4, 5, 9 (no duplicates)
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void SortedSet_Int()
        {
            var set = new SortedSet<int> { 3, 1, 4, 1, 5, 9 };
            var result = np.asanyarray(set);

            result.Should().BeShaped(5).And.BeOfValues(1, 3, 4, 5, 9); // Sorted, no duplicates
        }

        #endregion

        #region Queue<T> / Stack<T>

        [TestMethod]
        public void Queue_Int()
        {
            var queue = new Queue<int>();
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);
            var result = np.asanyarray(queue);

            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
        }

        [TestMethod]
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

        [TestMethod]
        public void ArraySegment_Int()
        {
            var array = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var segment = new ArraySegment<int>(array, 2, 5); // Elements 2,3,4,5,6
            var result = np.asanyarray(segment);

            result.Should().BeShaped(5).And.BeOfValues(2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void ArraySegment_Empty()
        {
            var array = new int[] { 1, 2, 3 };
            var segment = new ArraySegment<int>(array, 0, 0);
            var result = np.asanyarray(segment);

            result.Should().BeShaped(0);
        }

        [TestMethod]
        public void ArraySegment_Full()
        {
            var array = new int[] { 1, 2, 3, 4, 5 };
            var segment = new ArraySegment<int>(array);
            var result = np.asanyarray(segment);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        #endregion

        #region Memory<T> / ReadOnlyMemory<T>

        [TestMethod]
        public void Memory_Int()
        {
            var array = new int[] { 1, 2, 3, 4, 5 };
            var memory = new Memory<int>(array, 1, 3); // Elements 2,3,4
            var result = np.asanyarray(memory);

            result.Should().BeShaped(3).And.BeOfValues(2, 3, 4);
        }

        [TestMethod]
        public void ReadOnlyMemory_Int()
        {
            var array = new int[] { 1, 2, 3, 4, 5 };
            var memory = new ReadOnlyMemory<int>(array, 1, 3); // Elements 2,3,4
            var result = np.asanyarray(memory);

            result.Should().BeShaped(3).And.BeOfValues(2, 3, 4);
        }

        #endregion

        #region ImmutableArray<T> / ImmutableList<T>

        [TestMethod]
        public void ImmutableArray_Int()
        {
            var immutableArray = ImmutableArray.Create(1, 2, 3, 4, 5);
            var result = np.asanyarray(immutableArray);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void ImmutableList_Int()
        {
            var immutableList = ImmutableList.Create(1, 2, 3, 4, 5);
            var result = np.asanyarray(immutableList);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void ImmutableHashSet_Int()
        {
            var immutableSet = ImmutableHashSet.Create(3, 1, 4, 1, 5);
            var result = np.asanyarray(immutableSet);

            result.size.Should().Be(4); // 1, 3, 4, 5 (no duplicates)
        }

        #endregion

        #region All supported dtypes via List<T>

        [TestMethod]
        public void List_Byte()
        {
            var list = new List<byte> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(byte));
            result.Should().BeShaped(3);
        }

        // Note: sbyte is NOT supported by NumSharp (12 supported types: bool, byte, short, ushort, int, uint, long, ulong, char, float, double, decimal)

        [TestMethod]
        public void List_Short()
        {
            var list = new List<short> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(short));
            result.Should().BeShaped(3);
        }

        [TestMethod]
        public void List_UShort()
        {
            var list = new List<ushort> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(ushort));
            result.Should().BeShaped(3);
        }

        [TestMethod]
        public void List_UInt()
        {
            var list = new List<uint> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(uint));
            result.Should().BeShaped(3);
        }

        [TestMethod]
        public void List_Long()
        {
            var list = new List<long> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(long));
            result.Should().BeShaped(3);
        }

        [TestMethod]
        public void List_ULong()
        {
            var list = new List<ulong> { 1, 2, 3 };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(ulong));
            result.Should().BeShaped(3);
        }

        [TestMethod]
        public void List_Float()
        {
            var list = new List<float> { 1.1f, 2.2f, 3.3f };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(float));
            result.Should().BeShaped(3);
        }

        [TestMethod]
        public void List_Char()
        {
            var list = new List<char> { 'a', 'b', 'c' };
            var result = np.asanyarray(list);
            result.dtype.Should().Be(typeof(char));
            result.Should().BeShaped(3);
        }

        #endregion

        #region Error cases

        [TestMethod]
        public void Null_ThrowsArgumentNullException()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.asanyarray(null));
        }

        [TestMethod]
        public void UnsupportedType_ThrowsNotSupportedException()
        {
            // String collections are not supported (string is not primitive/decimal)
            var stringList = new List<string> { "a", "b", "c" };
            Assert.ThrowsException<NotSupportedException>(() => np.asanyarray(stringList));
        }

        [TestMethod]
        public void CustomClass_ThrowsNotSupportedException()
        {
            var customObject = new object();
            Assert.ThrowsException<NotSupportedException>(() => np.asanyarray(customObject));
        }

        #endregion

        #region String special case

        [TestMethod]
        public void String_CreatesCharArray()
        {
            var result = np.asanyarray("hello");

            result.Should().BeShaped(5);
            result.dtype.Should().Be(typeof(char));
        }

        #endregion

        #region Non-generic IEnumerable fallback

        [TestMethod]
        public void ArrayList_Int()
        {
            var arrayList = new System.Collections.ArrayList { 1, 2, 3, 4, 5 };
            var result = np.asanyarray(arrayList);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void ArrayList_Double()
        {
            var arrayList = new System.Collections.ArrayList { 1.1, 2.2, 3.3 };
            var result = np.asanyarray(arrayList);

            result.Should().BeShaped(3);
            result.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Hashtable_Keys()
        {
            var hashtable = new System.Collections.Hashtable { { 1, "a" }, { 2, "b" }, { 3, "c" } };
            var result = np.asanyarray(hashtable.Keys);

            result.size.Should().Be(3);
            result.dtype.Should().Be(typeof(int));
        }

        #endregion

        #region IEnumerator fallback

        [TestMethod]
        public void IEnumerator_Int()
        {
            static System.Collections.IEnumerator GetEnumerator()
            {
                yield return 10;
                yield return 20;
                yield return 30;
            }

            var result = np.asanyarray(GetEnumerator());

            result.Should().BeShaped(3).And.BeOfValues(10, 20, 30);
            result.dtype.Should().Be(typeof(int));
        }

        #endregion

        #region NumPy Parity - Misaligned Behaviors

        /// <summary>
        /// NumPy treats strings as scalar Unicode values, NumSharp treats as char arrays.
        /// NumPy: np.asanyarray("hello") -> dtype=&lt;U5, shape=(), ndim=0 (SCALAR)
        /// NumSharp: dtype=Char, shape=(5), ndim=1 (ARRAY)
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void String_IsCharArray_NotScalar()
        {
            var result = np.asanyarray("hello");

            // NumSharp behavior: char array
            result.ndim.Should().Be(1);
            result.shape.Should().BeEquivalentTo(new[] { 5 });
            result.dtype.Should().Be(typeof(char));

            // NumPy would be: ndim=0, shape=(), dtype=<U5 (scalar)
        }

        /// <summary>
        /// NumPy stores sets as object scalars (not iterated).
        /// NumSharp iterates sets and converts to array.
        /// NumPy: np.asanyarray({1,2,3}) -> dtype=object, shape=() (SCALAR)
        /// NumSharp: dtype=Int32, shape=(3) (ARRAY)
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void HashSet_IsIterated_NotObjectScalar()
        {
            var set = new HashSet<int> { 1, 2, 3 };
            var result = np.asanyarray(set);

            // NumSharp behavior: iterates and creates array
            result.ndim.Should().Be(1);
            result.size.Should().Be(3);
            result.dtype.Should().Be(typeof(int));

            // NumPy would be: dtype=object, shape=() (object scalar containing set)
        }

        /// <summary>
        /// NumPy stores generators as object scalars (NOT consumed).
        /// NumSharp consumes IEnumerable and converts to array.
        /// This is arguably more useful behavior for C#.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void LinqEnumerable_IsConsumed_NotObjectScalar()
        {
            var enumerable = new[] { 1, 2, 3 }.Select(x => x * 2);
            var result = np.asanyarray(enumerable);

            // NumSharp behavior: consumes and creates array
            result.ndim.Should().Be(1);
            result.Should().BeShaped(3).And.BeOfValues(2, 4, 6);

            // NumPy generator would be: dtype=object, shape=() (NOT consumed)
        }

        /// <summary>
        /// For typed empty collections (List&lt;T&gt;), NumSharp preserves the generic type parameter.
        /// NumPy defaults to float64 for untyped empty lists.
        /// This is a design choice: C# generics provide type information that NumPy doesn't have.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void EmptyTypedList_PreservesTypeParameter()
        {
            var result = np.asanyarray(new List<int>());

            // NumSharp behavior: preserves int dtype from generic type parameter
            result.dtype.Should().Be(typeof(int));
            result.shape.Should().BeEquivalentTo(new[] { 0 });

            // NumPy would be: dtype=float64, shape=(0,)
            // NumSharp can do better because C# generics provide the type at compile time
        }

        #endregion

        #region Tuple support

        /// <summary>
        /// C# ValueTuples are iterable like Python tuples.
        /// NumPy: np.asanyarray((1,2,3)) -> dtype=int64, shape=(3,)
        /// </summary>
        [TestMethod]
        public void ValueTuple_IsIterable()
        {
            var tuple = (1, 2, 3);
            var result = np.asanyarray(tuple);

            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
            result.dtype.Should().Be(typeof(int));
        }

        /// <summary>
        /// C# Tuple class is iterable like Python tuples.
        /// </summary>
        [TestMethod]
        public void Tuple_IsIterable()
        {
            var tuple = Tuple.Create(1, 2, 3);
            var result = np.asanyarray(tuple);

            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void ValueTuple_MixedTypes_PromotesToCommonType()
        {
            // Mixed int + double promotes to double (NumPy behavior)
            var tuple = (1, 2.5, 3);
            var result = np.asanyarray(tuple);

            result.Should().BeShaped(3);
            result.dtype.Should().Be(typeof(double)); // Promoted from int to double
        }

        [TestMethod]
        public void ValueTuple_IntAndBool_PromotesToInt()
        {
            // Mixed int + bool promotes to int (NumPy behavior)
            var tuple = (1, true, 3);
            var result = np.asanyarray(tuple);

            result.Should().BeShaped(3);
            result.dtype.Should().Be(typeof(int));
        }

        [TestMethod]
        public void EmptyTuple_ReturnsEmptyDoubleArray()
        {
            var tuple = ValueTuple.Create();
            var result = np.asanyarray(tuple);

            result.Should().BeShaped(0);
            result.dtype.Should().Be(typeof(double));
        }

        #endregion

        #region Empty non-generic collections

        /// <summary>
        /// Empty non-generic collections return empty double[] (NumPy defaults to float64).
        /// </summary>
        [TestMethod]
        public void EmptyArrayList_ReturnsEmptyDoubleArray()
        {
            var arrayList = new System.Collections.ArrayList();
            var result = np.asanyarray(arrayList);

            result.size.Should().Be(0);
            result.ndim.Should().Be(1);
            result.dtype.Should().Be(typeof(double)); // NumPy: float64
        }

        #endregion

        #region object[] regression

        [TestMethod]
        public void ObjectArray_Homogeneous_Int()
        {
            var arr = new object[] { 1, 2, 3 };
            var result = np.asanyarray(arr);

            result.dtype.Should().Be(typeof(int));
            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
        }

        [TestMethod]
        public void ObjectArray_MixedIntFloat_PromotesToDouble()
        {
            var arr = new object[] { 1, 2.5, 3 };
            var result = np.asanyarray(arr);

            result.dtype.Should().Be(typeof(double));
            result.Should().BeShaped(3).And.BeOfValues(1.0, 2.5, 3.0);
        }

        [TestMethod]
        public void ObjectArray_MixedBoolInt_PromotesToInt()
        {
            var arr = new object[] { true, 2, false };
            var result = np.asanyarray(arr);

            result.dtype.Should().Be(typeof(int));
            result.Should().BeShaped(3).And.BeOfValues(1, 2, 0);
        }

        [TestMethod]
        public void ObjectArray_Empty_ReturnsFloat64()
        {
            var arr = new object[0];
            var result = np.asanyarray(arr);

            result.size.Should().Be(0);
            result.ndim.Should().Be(1);
            result.dtype.Should().Be(typeof(double));
        }

        #endregion
    }
}
