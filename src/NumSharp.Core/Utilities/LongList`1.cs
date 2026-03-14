// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// Modified for NumSharp to support long-based indexing.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Helper class for LongList capacity calculations.
    /// Uses 33% growth for very large collections (above 1 billion elements).
    /// </summary>
    internal static class ListHelpersLong
    {
        /// <summary>
        /// Threshold above which we use 33% growth instead of doubling (1 billion elements).
        /// </summary>
        public const long LargeGrowthThreshold = 1_000_000_000L;

        /// <summary>
        /// Maximum array length supported by .NET runtime.
        /// </summary>
        public const long MaxArrayLength = 0x7FFFFFC7L; // Array.MaxLength

        /// <summary>
        /// Calculates the new capacity for growth.
        /// For collections below LargeGrowthThreshold, doubles the size.
        /// For larger collections, grows by 33% to avoid excessive memory allocation.
        /// </summary>
        /// <param name="oldCapacity">Current capacity.</param>
        /// <param name="minCapacity">Minimum required capacity.</param>
        /// <param name="defaultCapacity">Default capacity for empty collections.</param>
        /// <returns>New capacity.</returns>
        public static long GetNewCapacity(long oldCapacity, long minCapacity, long defaultCapacity)
        {
            Debug.Assert(oldCapacity < minCapacity);

            long newCapacity;

            if (oldCapacity == 0)
            {
                newCapacity = defaultCapacity;
            }
            else if (oldCapacity < LargeGrowthThreshold)
            {
                // Standard doubling for smaller collections
                newCapacity = 2 * oldCapacity;
            }
            else
            {
                // 33% growth for very large collections to avoid excessive memory allocation
                newCapacity = oldCapacity + (oldCapacity / 3);
            }

            // Handle overflow - if newCapacity overflowed or exceeds max, cap it
            if ((ulong)newCapacity > (ulong)MaxArrayLength)
            {
                newCapacity = MaxArrayLength;
            }

            // If the computed capacity is still less than specified, use the minimum
            if (newCapacity < minCapacity)
            {
                newCapacity = minCapacity;
            }

            return newCapacity;
        }
    }

    /// <summary>
    /// Implements a variable-size List that uses long-based indexing.
    /// Supports collections with element counts tracked as long values.
    /// Uses 33% growth for very large collections (above 1 billion elements).
    /// </summary>
    /// <typeparam name="T">The type of elements in the list.</typeparam>
    [DebuggerDisplay("Count = {LongCount}")]
    public class LongList<T> : IList<T>, IReadOnlyList<T>
    {
        private const long DefaultCapacity = 4;

        internal T[] _items;
        internal long _size;
        internal int _version;

        private static readonly T[] s_emptyArray = Array.Empty<T>();

        #region Constructors

        /// <summary>
        /// Constructs a LongList. The list is initially empty and has a capacity of zero.
        /// Upon adding the first element, the capacity is increased to DefaultCapacity,
        /// and then increased based on growth strategy.
        /// </summary>
        public LongList()
        {
            _items = s_emptyArray;
        }

        /// <summary>
        /// Constructs a LongList with a given initial capacity.
        /// </summary>
        /// <param name="capacity">Initial capacity.</param>
        public LongList(long capacity)
        {
            if (capacity < 0)
                throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be non-negative.");

            if (capacity == 0)
                _items = s_emptyArray;
            else
                _items = new T[capacity];
        }

        /// <summary>
        /// Constructs a LongList, copying the contents of the given collection.
        /// </summary>
        /// <param name="collection">The collection to copy from.</param>
        public LongList(IEnumerable<T> collection)
        {
            if (collection == null)
                throw new ArgumentNullException(nameof(collection));

            if (collection is ICollection<T> c)
            {
                int count = c.Count;
                if (count == 0)
                {
                    _items = s_emptyArray;
                }
                else
                {
                    _items = new T[count];
                    c.CopyTo(_items, 0);
                    _size = count;
                }
            }
            else
            {
                _items = s_emptyArray;
                foreach (var item in collection)
                {
                    Add(item);
                }
            }
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets the capacity of this list.
        /// </summary>
        public long Capacity
        {
            get => _items.LongLength;
            set
            {
                if (value < _size)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Capacity cannot be less than Count.");
                }

                if (value != _items.LongLength)
                {
                    if (value > 0)
                    {
                        T[] newItems = new T[value];
                        if (_size > 0)
                        {
                            Array.Copy(_items, newItems, _size);
                        }
                        _items = newItems;
                    }
                    else
                    {
                        _items = s_emptyArray;
                    }
                }
            }
        }

        /// <summary>
        /// Gets the number of elements in the list as a long.
        /// </summary>
        public long LongCount => _size;

        /// <summary>
        /// Gets the number of elements in the list.
        /// Returns int.MaxValue if the count exceeds int.MaxValue.
        /// </summary>
        public int Count => _size > int.MaxValue ? int.MaxValue : (int)_size;

        bool ICollection<T>.IsReadOnly => false;

        /// <summary>
        /// Gets or sets the element at the given index (long-based).
        /// </summary>
        public T this[long index]
        {
            get
            {
                if ((ulong)index >= (ulong)_size)
                {
                    throw new ArgumentOutOfRangeException(nameof(index), "Index was out of range.");
                }
                return _items[index];
            }
            set
            {
                if ((ulong)index >= (ulong)_size)
                {
                    throw new ArgumentOutOfRangeException(nameof(index), "Index was out of range.");
                }
                _items[index] = value;
                _version++;
            }
        }

        /// <summary>
        /// Gets or sets the element at the given index (int-based for IList compatibility).
        /// </summary>
        T IList<T>.this[int index]
        {
            get => this[(long)index];
            set => this[(long)index] = value;
        }

        /// <summary>
        /// Gets the element at the given index (int-based for IReadOnlyList compatibility).
        /// </summary>
        T IReadOnlyList<T>.this[int index] => this[(long)index];

        #endregion

        #region Add/Insert Methods

        /// <summary>
        /// Adds the given object to the end of this list.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(T item)
        {
            _version++;
            T[] array = _items;
            long size = _size;
            if ((ulong)size < (ulong)array.LongLength)
            {
                _size = size + 1;
                array[size] = item;
            }
            else
            {
                AddWithResize(item);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void AddWithResize(T item)
        {
            Debug.Assert(_size == _items.LongLength);
            long size = _size;
            Grow(size + 1);
            _size = size + 1;
            _items[size] = item;
        }

        /// <summary>
        /// Adds the elements of the given collection to the end of this list.
        /// </summary>
        public void AddRange(IEnumerable<T> collection)
        {
            if (collection == null)
                throw new ArgumentNullException(nameof(collection));

            if (collection is ICollection<T> c)
            {
                int count = c.Count;
                if (count > 0)
                {
                    if (_items.LongLength - _size < count)
                    {
                        Grow(checked(_size + count));
                    }

                    c.CopyTo(_items, (int)_size);
                    _size += count;
                    _version++;
                }
            }
            else
            {
                foreach (var item in collection)
                {
                    Add(item);
                }
            }
        }

        /// <summary>
        /// Inserts an element into this list at a given index.
        /// </summary>
        public void Insert(long index, T item)
        {
            if ((ulong)index > (ulong)_size)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the bounds of the List.");
            }

            if (_size == _items.LongLength)
            {
                GrowForInsertion(index, 1);
            }
            else if (index < _size)
            {
                Array.Copy(_items, index, _items, index + 1, _size - index);
            }
            _items[index] = item;
            _size++;
            _version++;
        }

        void IList<T>.Insert(int index, T item) => Insert((long)index, item);

        /// <summary>
        /// Inserts the elements of the given collection at a given index.
        /// </summary>
        public void InsertRange(long index, IEnumerable<T> collection)
        {
            if (collection == null)
                throw new ArgumentNullException(nameof(collection));

            if ((ulong)index > (ulong)_size)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the bounds of the List.");
            }

            if (collection is ICollection<T> c)
            {
                int count = c.Count;
                if (count > 0)
                {
                    if (_items.LongLength - _size < count)
                    {
                        GrowForInsertion(index, count);
                    }
                    else if (index < _size)
                    {
                        Array.Copy(_items, index, _items, index + count, _size - index);
                    }

                    // Handle self-insertion
                    if (ReferenceEquals(this, c))
                    {
                        Array.Copy(_items, 0, _items, index, index);
                        Array.Copy(_items, index + count, _items, index * 2, _size - index);
                    }
                    else
                    {
                        c.CopyTo(_items, (int)index);
                    }
                    _size += count;
                    _version++;
                }
            }
            else
            {
                foreach (var item in collection)
                {
                    Insert(index++, item);
                }
            }
        }

        #endregion

        #region Remove Methods

        /// <summary>
        /// Removes the first occurrence of the given element.
        /// </summary>
        public bool Remove(T item)
        {
            long index = IndexOf(item);
            if (index >= 0)
            {
                RemoveAt(index);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Removes the element at the given index.
        /// </summary>
        public void RemoveAt(long index)
        {
            if ((ulong)index >= (ulong)_size)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Index was out of range.");
            }

            _size--;
            if (index < _size)
            {
                Array.Copy(_items, index + 1, _items, index, _size - index);
            }

            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            {
                _items[_size] = default!;
            }
            _version++;
        }

        void IList<T>.RemoveAt(int index) => RemoveAt((long)index);

        /// <summary>
        /// Removes a range of elements from this list.
        /// </summary>
        public void RemoveRange(long index, long count)
        {
            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative.");

            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative.");

            if (_size - index < count)
                throw new ArgumentException("Invalid offset/length.");

            if (count > 0)
            {
                _size -= count;
                if (index < _size)
                {
                    Array.Copy(_items, index + count, _items, index, _size - index);
                }

                _version++;
                if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                {
                    Array.Clear(_items, (int)_size, (int)count);
                }
            }
        }

        /// <summary>
        /// Removes all items which match the predicate.
        /// </summary>
        public long RemoveAll(Predicate<T> match)
        {
            if (match == null)
                throw new ArgumentNullException(nameof(match));

            long freeIndex = 0;

            // Find the first item to remove
            while (freeIndex < _size && !match(_items[freeIndex])) freeIndex++;
            if (freeIndex >= _size) return 0;

            long current = freeIndex + 1;
            while (current < _size)
            {
                while (current < _size && match(_items[current])) current++;

                if (current < _size)
                {
                    _items[freeIndex++] = _items[current++];
                }
            }

            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            {
                Array.Clear(_items, (int)freeIndex, (int)(_size - freeIndex));
            }

            long result = _size - freeIndex;
            _size = freeIndex;
            _version++;
            return result;
        }

        /// <summary>
        /// Clears the contents of the list.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Clear()
        {
            _version++;
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            {
                long size = _size;
                _size = 0;
                if (size > 0)
                {
                    Array.Clear(_items, 0, (int)size);
                }
            }
            else
            {
                _size = 0;
            }
        }

        #endregion

        #region Search Methods

        /// <summary>
        /// Returns true if the specified element is in the List.
        /// </summary>
        public bool Contains(T item)
        {
            return _size != 0 && IndexOf(item) >= 0;
        }

        /// <summary>
        /// Returns the index of the first occurrence of a given value.
        /// </summary>
        public long IndexOf(T item)
        {
            return Array.IndexOf(_items, item, 0, (int)Math.Min(_size, int.MaxValue));
        }

        int IList<T>.IndexOf(T item)
        {
            long index = IndexOf(item);
            return index > int.MaxValue ? -1 : (int)index;
        }

        /// <summary>
        /// Returns the index of the first occurrence of a given value in a range.
        /// </summary>
        public long IndexOf(T item, long index)
        {
            if ((ulong)index > (ulong)_size)
                throw new ArgumentOutOfRangeException(nameof(index), "Index was out of range.");

            return Array.IndexOf(_items, item, (int)index, (int)(_size - index));
        }

        /// <summary>
        /// Returns the index of the first occurrence of a given value in a range.
        /// </summary>
        public long IndexOf(T item, long index, long count)
        {
            if ((ulong)index > (ulong)_size)
                throw new ArgumentOutOfRangeException(nameof(index), "Index was out of range.");

            if (count < 0 || index > _size - count)
                throw new ArgumentOutOfRangeException(nameof(count), "Count was out of range.");

            return Array.IndexOf(_items, item, (int)index, (int)count);
        }

        /// <summary>
        /// Returns the index of the last occurrence of a given value.
        /// </summary>
        public long LastIndexOf(T item)
        {
            if (_size == 0)
                return -1;

            return LastIndexOf(item, _size - 1, _size);
        }

        /// <summary>
        /// Returns the index of the last occurrence of a given value in a range.
        /// </summary>
        public long LastIndexOf(T item, long index)
        {
            if ((ulong)index >= (ulong)_size)
                throw new ArgumentOutOfRangeException(nameof(index), "Index was out of range.");

            return LastIndexOf(item, index, index + 1);
        }

        /// <summary>
        /// Returns the index of the last occurrence of a given value in a range.
        /// </summary>
        public long LastIndexOf(T item, long index, long count)
        {
            if (_size == 0)
                return -1;

            if ((ulong)index >= (ulong)_size)
                throw new ArgumentOutOfRangeException(nameof(index), "Index was out of range.");

            if (count < 0 || index - count + 1 < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count was out of range.");

            return Array.LastIndexOf(_items, item, (int)index, (int)count);
        }

        /// <summary>
        /// Searches for an element using binary search.
        /// </summary>
        public long BinarySearch(long index, long count, T item, IComparer<T>? comparer)
        {
            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative.");

            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative.");

            if (_size - index < count)
                throw new ArgumentException("Invalid offset/length.");

            return Array.BinarySearch(_items, (int)index, (int)count, item, comparer);
        }

        public long BinarySearch(T item) => BinarySearch(0, _size, item, null);

        public long BinarySearch(T item, IComparer<T>? comparer) => BinarySearch(0, _size, item, comparer);

        /// <summary>
        /// Returns true if an element matching the predicate exists.
        /// </summary>
        public bool Exists(Predicate<T> match) => FindIndex(match) >= 0;

        /// <summary>
        /// Finds the first element matching the predicate.
        /// </summary>
        public T? Find(Predicate<T> match)
        {
            if (match == null)
                throw new ArgumentNullException(nameof(match));

            for (long i = 0; i < _size; i++)
            {
                if (match(_items[i]))
                    return _items[i];
            }
            return default;
        }

        /// <summary>
        /// Finds all elements matching the predicate.
        /// </summary>
        public LongList<T> FindAll(Predicate<T> match)
        {
            if (match == null)
                throw new ArgumentNullException(nameof(match));

            var list = new LongList<T>();
            for (long i = 0; i < _size; i++)
            {
                if (match(_items[i]))
                    list.Add(_items[i]);
            }
            return list;
        }

        /// <summary>
        /// Finds the index of the first element matching the predicate.
        /// </summary>
        public long FindIndex(Predicate<T> match) => FindIndex(0, _size, match);

        public long FindIndex(long startIndex, Predicate<T> match) => FindIndex(startIndex, _size - startIndex, match);

        public long FindIndex(long startIndex, long count, Predicate<T> match)
        {
            if ((ulong)startIndex > (ulong)_size)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "Index was out of range.");

            if (count < 0 || startIndex > _size - count)
                throw new ArgumentOutOfRangeException(nameof(count), "Count was out of range.");

            if (match == null)
                throw new ArgumentNullException(nameof(match));

            long endIndex = startIndex + count;
            for (long i = startIndex; i < endIndex; i++)
            {
                if (match(_items[i]))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Finds the last element matching the predicate.
        /// </summary>
        public T? FindLast(Predicate<T> match)
        {
            if (match == null)
                throw new ArgumentNullException(nameof(match));

            for (long i = _size - 1; i >= 0; i--)
            {
                if (match(_items[i]))
                    return _items[i];
            }
            return default;
        }

        /// <summary>
        /// Finds the index of the last element matching the predicate.
        /// </summary>
        public long FindLastIndex(Predicate<T> match) => FindLastIndex(_size - 1, _size, match);

        public long FindLastIndex(long startIndex, Predicate<T> match) => FindLastIndex(startIndex, startIndex + 1, match);

        public long FindLastIndex(long startIndex, long count, Predicate<T> match)
        {
            if (match == null)
                throw new ArgumentNullException(nameof(match));

            if (_size == 0)
            {
                if (startIndex != -1)
                    throw new ArgumentOutOfRangeException(nameof(startIndex), "Index was out of range.");
            }
            else
            {
                if ((ulong)startIndex >= (ulong)_size)
                    throw new ArgumentOutOfRangeException(nameof(startIndex), "Index was out of range.");
            }

            if (count < 0 || startIndex - count + 1 < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count was out of range.");

            long endIndex = startIndex - count;
            for (long i = startIndex; i > endIndex; i--)
            {
                if (match(_items[i]))
                    return i;
            }
            return -1;
        }

        #endregion

        #region Capacity Management

        /// <summary>
        /// Ensures that the capacity is at least the specified value.
        /// </summary>
        public long EnsureCapacity(long capacity)
        {
            if (capacity < 0)
                throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be non-negative.");

            if (_items.LongLength < capacity)
            {
                Grow(capacity);
            }

            return _items.LongLength;
        }

        internal void Grow(long capacity)
        {
            Capacity = ListHelpersLong.GetNewCapacity(_items.LongLength, capacity, DefaultCapacity);
        }

        internal void GrowForInsertion(long indexToInsert, long insertionCount = 1)
        {
            Debug.Assert(insertionCount > 0);

            long requiredCapacity = checked(_size + insertionCount);
            long newCapacity = ListHelpersLong.GetNewCapacity(_items.LongLength, requiredCapacity, DefaultCapacity);

            T[] newItems = new T[newCapacity];
            if (indexToInsert != 0)
            {
                Array.Copy(_items, newItems, indexToInsert);
            }

            if (_size != indexToInsert)
            {
                Array.Copy(_items, indexToInsert, newItems, indexToInsert + insertionCount, _size - indexToInsert);
            }

            _items = newItems;
        }

        /// <summary>
        /// Sets the capacity to the actual number of elements.
        /// </summary>
        public void TrimExcess()
        {
            long threshold = (long)(_items.LongLength * 0.9);
            if (_size < threshold)
            {
                Capacity = _size;
            }
        }

        #endregion

        #region Copy and Transform Methods

        /// <summary>
        /// Copies to an array.
        /// </summary>
        public void CopyTo(T[] array) => CopyTo(array, 0);

        public void CopyTo(T[] array, int arrayIndex)
        {
            Array.Copy(_items, 0, array, arrayIndex, _size);
        }

        public void CopyTo(long index, T[] array, int arrayIndex, long count)
        {
            if (_size - index < count)
                throw new ArgumentException("Invalid offset/length.");

            Array.Copy(_items, index, array, arrayIndex, count);
        }

        /// <summary>
        /// Copies from a LongList to an ArraySlice (NumSharp unmanaged memory).
        /// Static method to handle the unmanaged constraint.
        /// </summary>
        public static void CopyTo<TItem>(LongList<TItem> src, ArraySlice<TItem> array) where TItem : unmanaged
        {
            CopyTo(src, array, 0, src._size);
        }

        /// <summary>
        /// Copies from a LongList to an ArraySlice with offset and count.
        /// Static method to handle the unmanaged constraint.
        /// </summary>
        public static unsafe void CopyTo<TItem>(LongList<TItem> src, ArraySlice<TItem> array, long arrayIndex, long count) where TItem : unmanaged
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));

            // ArraySlice<T> is a struct, no null check needed

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "Index must be non-negative.");

            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative.");

            if (arrayIndex > array.Count || count > array.Count - arrayIndex)
                throw new ArgumentException("Invalid offset/length.");

            if (count > src._size)
                throw new ArgumentException("Count exceeds list size.");

            TItem* dst = array.Address;
            for (long i = 0; i < count; i++)
            {
                dst[arrayIndex + i] = src._items[i];
            }
        }

        /// <summary>
        /// Returns an array containing all elements.
        /// </summary>
        public T[] ToArray()
        {
            if (_size == 0)
                return s_emptyArray;

            T[] array = new T[_size];
            Array.Copy(_items, array, _size);
            return array;
        }

        /// <summary>
        /// Gets a range of elements as a new LongList.
        /// </summary>
        public LongList<T> GetRange(long index, long count)
        {
            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative.");

            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative.");

            if (_size - index < count)
                throw new ArgumentException("Invalid offset/length.");

            var list = new LongList<T>(count);
            Array.Copy(_items, index, list._items, 0, count);
            list._size = count;
            return list;
        }

        /// <summary>
        /// Creates a shallow copy of a range of elements.
        /// </summary>
        public LongList<T> Slice(long start, long length) => GetRange(start, length);

        /// <summary>
        /// Converts all elements using a converter.
        /// </summary>
        public LongList<TOutput> ConvertAll<TOutput>(Converter<T, TOutput> converter)
        {
            if (converter == null)
                throw new ArgumentNullException(nameof(converter));

            var list = new LongList<TOutput>(_size);
            for (long i = 0; i < _size; i++)
            {
                list._items[i] = converter(_items[i]);
            }
            list._size = _size;
            return list;
        }

        #endregion

        #region Sort and Reverse

        /// <summary>
        /// Reverses the elements in this list.
        /// </summary>
        public void Reverse() => Reverse(0, _size);

        public void Reverse(long index, long count)
        {
            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative.");

            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative.");

            if (_size - index < count)
                throw new ArgumentException("Invalid offset/length.");

            if (count > 1)
            {
                Array.Reverse(_items, (int)index, (int)count);
            }
            _version++;
        }

        /// <summary>
        /// Sorts the elements in this list.
        /// </summary>
        public void Sort() => Sort(0, _size, null);

        public void Sort(IComparer<T>? comparer) => Sort(0, _size, comparer);

        public void Sort(long index, long count, IComparer<T>? comparer)
        {
            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative.");

            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count must be non-negative.");

            if (_size - index < count)
                throw new ArgumentException("Invalid offset/length.");

            if (count > 1)
            {
                Array.Sort(_items, (int)index, (int)count, comparer);
            }
            _version++;
        }

        public void Sort(Comparison<T> comparison)
        {
            if (comparison == null)
                throw new ArgumentNullException(nameof(comparison));

            if (_size > 1)
            {
                new Span<T>(_items, 0, (int)_size).Sort(comparison);
            }
            _version++;
        }

        #endregion

        #region Iteration

        /// <summary>
        /// Performs the specified action on each element.
        /// </summary>
        public void ForEach(Action<T> action)
        {
            if (action == null)
                throw new ArgumentNullException(nameof(action));

            int version = _version;

            for (long i = 0; i < _size; i++)
            {
                if (version != _version)
                    break;

                action(_items[i]);
            }

            if (version != _version)
                throw new InvalidOperationException("Collection was modified during enumeration.");
        }

        /// <summary>
        /// Returns true if all elements match the predicate.
        /// </summary>
        public bool TrueForAll(Predicate<T> match)
        {
            if (match == null)
                throw new ArgumentNullException(nameof(match));

            for (long i = 0; i < _size; i++)
            {
                if (!match(_items[i]))
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Returns an enumerator for this list.
        /// </summary>
        public Enumerator GetEnumerator() => new Enumerator(this);

        IEnumerator<T> IEnumerable<T>.GetEnumerator() =>
            _size == 0 ? ((IEnumerable<T>)Array.Empty<T>()).GetEnumerator() : GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable<T>)this).GetEnumerator();

        #endregion

        #region Enumerator

        /// <summary>
        /// Enumerator for LongList with long-based indexing.
        /// </summary>
        public struct Enumerator : IEnumerator<T>, IEnumerator
        {
            private readonly LongList<T> _list;
            private readonly int _version;
            private long _index;
            private T? _current;

            internal Enumerator(LongList<T> list)
            {
                _list = list;
                _version = list._version;
                _index = 0;
                _current = default;
            }

            public void Dispose()
            {
            }

            public bool MoveNext()
            {
                LongList<T> localList = _list;

                if (_version != _list._version)
                {
                    throw new InvalidOperationException("Collection was modified during enumeration.");
                }

                if ((ulong)_index < (ulong)localList._size)
                {
                    _current = localList._items[_index];
                    _index++;
                    return true;
                }

                _current = default;
                _index = -1;
                return false;
            }

            public T Current => _current!;

            object? IEnumerator.Current
            {
                get
                {
                    if (_index <= 0)
                    {
                        throw new InvalidOperationException("Enumeration has not started or has already finished.");
                    }
                    return _current;
                }
            }

            void IEnumerator.Reset()
            {
                if (_version != _list._version)
                {
                    throw new InvalidOperationException("Collection was modified during enumeration.");
                }

                _index = 0;
                _current = default;
            }
        }

        #endregion
    }
}
