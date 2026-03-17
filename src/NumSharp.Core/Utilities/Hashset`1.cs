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
    /// Implementation notes:
    /// This uses an array-based implementation similar to Dictionary&lt;T&gt;, using a buckets array
    /// to map hash values to the Slots array. Items in the Slots array that hash to the same value
    /// are chained together through the "next" indices.
    ///
    /// This implementation supports long indexing for collections exceeding int.MaxValue elements.
    ///
    /// The capacity is always prime; so during resizing, the capacity is chosen as the next prime
    /// greater than double the last capacity (or 33% growth for very large sets above 1 billion elements).
    ///
    /// The underlying data structures are lazily initialized. Because of the observation that,
    /// in practice, hashtables tend to contain only a few elements, the initial capacity is
    /// set very small (3 elements) unless the ctor with a collection is used.
    ///
    /// The +/- 1 modifications in methods that add, check for containment, etc allow us to
    /// distinguish a hash code of 0 from an uninitialized bucket. This saves us from having to
    /// reset each bucket to -1 when resizing. See Contains, for example.
    ///
    /// Set methods such as UnionWith, IntersectWith, ExceptWith, and SymmetricExceptWith modify
    /// this set.
    ///
    /// Some operations can perform faster if we can assume "other" contains unique elements
    /// according to this equality comparer. The only times this is efficient to check is if
    /// other is a hashset. Note that checking that it's a hashset alone doesn't suffice; we
    /// also have to check that the hashset is using the same equality comparer. If other
    /// has a different equality comparer, it will have unique elements according to its own
    /// equality comparer, but not necessarily according to ours. Therefore, to go these
    /// optimized routes we check that other is a hashset using the same equality comparer.
    ///
    /// A HashSet with no elements has the properties of the empty set. (See IsSubset, etc. for
    /// special empty set checks.)
    ///
    /// A couple of methods have a special case if other is this (e.g. SymmetricExceptWith).
    /// If we didn't have these checks, we could be iterating over the set and modifying at
    /// the same time.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [DebuggerDisplay("Count = {" + nameof(Count) + "}")]
    [SuppressMessage("Microsoft.Naming", "CA1710:IdentifiersShouldHaveCorrectSuffix", Justification = "By design")]
    public class Hashset<T> : ICollection<T>, ISet<T>, IReadOnlyCollection<T>
    {
        // store lower 31 bits of hash code
        private const int Lower31BitMask = 0x7FFFFFFF;

        // cutoff point, above which we won't do stackallocs. This corresponds to 100 integers.
        private const int StackAllocThreshold = 100;

        // when constructing a hashset from an existing collection, it may contain duplicates,
        // so this is used as the max acceptable excess ratio of capacity to count. Note that
        // this is only used on the ctor and not to automatically shrink if the hashset has, e.g,
        // a lot of adds followed by removes. Users must explicitly shrink by calling TrimExcess.
        // This is set to 3 because capacity is acceptable as 2x rounded up to nearest prime.
        private const int ShrinkThreshold = 3;

        private long[] m_buckets;
        private Slot[] m_slots;
        private long m_count;
        private long m_lastIndex;
        private long m_freeList;
        private IEqualityComparer<T> m_comparer;
        private int m_version;

        #region Constructors

        public Hashset()
            : this(EqualityComparer<T>.Default)
        { }

        public Hashset(long capacity)
            : this(capacity, EqualityComparer<T>.Default)
        { }

        public Hashset(int capacity)
            : this((long)capacity, EqualityComparer<T>.Default)
        { }

        public Hashset(IEqualityComparer<T> comparer)
        {
            if (comparer == null)
            {
                comparer = EqualityComparer<T>.Default;
            }

            this.m_comparer = comparer;
            m_lastIndex = 0;
            m_count = 0;
            m_freeList = -1;
            m_version = 0;
        }

        public Hashset(IEnumerable<T> collection)
            : this(collection, EqualityComparer<T>.Default)
        { }

        /// <summary>
        /// Implementation Notes:
        /// Since resizes are relatively expensive (require rehashing), this attempts to minimize
        /// the need to resize by setting the initial capacity based on size of collection.
        /// </summary>
        /// <param name="collection"></param>
        /// <param name="comparer"></param>
        public Hashset(IEnumerable<T> collection, IEqualityComparer<T> comparer)
            : this(comparer)
        {
            if (collection == null)
            {
                throw new ArgumentNullException(nameof(collection));
            }

            Contract.EndContractBlock();

            if (collection is Hashset<T> otherAsHashSet && AreEqualityComparersEqual(this, otherAsHashSet))
            {
                CopyFrom(otherAsHashSet);
            }
            else
            {
                // to avoid excess resizes, first set size based on collection's count. Collection
                // may contain duplicates, so call TrimExcess if resulting hashset is larger than
                // threshold
                long suggestedCapacity = !(collection is ICollection<T> coll) ? 0 : coll.Count;
                Initialize(suggestedCapacity);

                this.UnionWith(collection);

                if (m_count > 0 && m_slots.LongLength / m_count > ShrinkThreshold)
                {
                    TrimExcess();
                }
            }
        }

        // Initializes the HashSet from another HashSet with the same element type and
        // equality comparer.
        private void CopyFrom(Hashset<T> source)
        {
            long count = source.m_count;
            if (count == 0)
            {
                // As well as short-circuiting on the rest of the work done,
                // this avoids errors from trying to access otherAsHashSet.m_buckets
                // or otherAsHashSet.m_slots when they aren't initialized.
                return;
            }

            long capacity = source.m_buckets.LongLength;
            long threshold = HashHelpersLong.ExpandPrime(count + 1);

            if (threshold >= capacity)
            {
                m_buckets = (long[])source.m_buckets.Clone();
                m_slots = (Slot[])source.m_slots.Clone();

                m_lastIndex = source.m_lastIndex;
                m_freeList = source.m_freeList;
            }
            else
            {
                long lastIndex = source.m_lastIndex;
                Slot[] slots = source.m_slots;
                Initialize(count);
                long index = 0;
                for (long i = 0; i < lastIndex; ++i)
                {
                    int hashCode = slots[i].hashCode;
                    if (hashCode >= 0)
                    {
                        AddValue(index, hashCode, slots[i].value);
                        ++index;
                    }
                }

                Debug.Assert(index == count);
                m_lastIndex = index;
            }

            m_count = count;
        }

        public Hashset(long capacity, IEqualityComparer<T> comparer)
            : this(comparer)
        {
            if (capacity < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity));
            }

            Contract.EndContractBlock();

            if (capacity > 0)
            {
                Initialize(capacity);
            }
        }

        public Hashset(int capacity, IEqualityComparer<T> comparer)
            : this((long)capacity, comparer)
        { }

        #endregion

        #region ICollection<T> methods

        /// <summary>
        /// Add item to this hashset. This is the explicit implementation of the ICollection&lt;T&gt;
        /// interface. The other Add method returns bool indicating whether item was added.
        /// </summary>
        /// <param name="item">item to add</param>
        void ICollection<T>.Add(T item)
        {
            AddIfNotPresent(item);
        }

        /// <summary>
        /// Remove all items from this set. This clears the elements but not the underlying
        /// buckets and slots array. Follow this call by TrimExcess to release these.
        /// </summary>
        public void Clear()
        {
            if (m_lastIndex > 0)
            {
                Debug.Assert(m_buckets != null, "m_buckets was null but m_lastIndex > 0");

                // clear the elements so that the gc can reclaim the references.
                // clear only up to m_lastIndex for m_slots
                Array.Clear(m_slots, 0, (int)Math.Min(m_lastIndex, int.MaxValue));
                if (m_lastIndex > int.MaxValue)
                {
                    // For very large arrays, clear in chunks
                    for (long i = int.MaxValue; i < m_lastIndex; i += int.MaxValue)
                    {
                        Array.Clear(m_slots, (int)(i % int.MaxValue), (int)Math.Min(m_lastIndex - i, int.MaxValue));
                    }
                }
                Array.Clear(m_buckets, 0, (int)Math.Min(m_buckets.LongLength, int.MaxValue));
                if (m_buckets.LongLength > int.MaxValue)
                {
                    for (long i = int.MaxValue; i < m_buckets.LongLength; i += int.MaxValue)
                    {
                        Array.Clear(m_buckets, (int)(i % int.MaxValue), (int)Math.Min(m_buckets.LongLength - i, int.MaxValue));
                    }
                }
                m_lastIndex = 0;
                m_count = 0;
                m_freeList = -1;
            }

            m_version++;
        }

        /// <summary>
        /// Checks if this hashset contains the item
        /// </summary>
        /// <param name="item">item to check for containment</param>
        /// <returns>true if item contained; false if not</returns>
        public bool Contains(T item)
        {
            if (m_buckets != null)
            {
                int hashCode = InternalGetHashCode(item);
                long bucketIndex = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);
                // see note at "HashSet" level describing why "- 1" appears in for loop
                for (long i = m_buckets[bucketIndex] - 1; i >= 0; i = m_slots[i].next)
                {
                    if (m_slots[i].hashCode == hashCode && m_comparer.Equals(m_slots[i].value, item))
                    {
                        return true;
                    }
                }
            }

            // either m_buckets is null or wasn't found
            return false;
        }

        /// <summary>
        /// Copy items in this hashset to array, starting at arrayIndex
        /// </summary>
        /// <param name="array">array to add items to</param>
        /// <param name="arrayIndex">index to start at</param>
        public void CopyTo(T[] array, int arrayIndex)
        {
            CopyTo(array, arrayIndex, m_count);
        }

        /// <summary>
        /// Remove item from this hashset
        /// </summary>
        /// <param name="item">item to remove</param>
        /// <returns>true if removed; false if not (i.e. if the item wasn't in the HashSet)</returns>
        public bool Remove(T item)
        {
            if (m_buckets != null)
            {
                int hashCode = InternalGetHashCode(item);
                long bucketIndex = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);
                long last = -1;
                for (long i = m_buckets[bucketIndex] - 1; i >= 0; last = i, i = m_slots[i].next)
                {
                    if (m_slots[i].hashCode == hashCode && m_comparer.Equals(m_slots[i].value, item))
                    {
                        if (last < 0)
                        {
                            // first iteration; update buckets
                            m_buckets[bucketIndex] = m_slots[i].next + 1;
                        }
                        else
                        {
                            // subsequent iterations; update 'next' pointers
                            m_slots[last].next = m_slots[i].next;
                        }

                        m_slots[i].hashCode = -1;
                        m_slots[i].value = default(T);
                        m_slots[i].next = m_freeList;

                        m_count--;
                        m_version++;
                        if (m_count == 0)
                        {
                            m_lastIndex = 0;
                            m_freeList = -1;
                        }
                        else
                        {
                            m_freeList = i;
                        }

                        return true;
                    }
                }
            }

            // either m_buckets is null or wasn't found
            return false;
        }

        /// <summary>
        /// Number of elements in this hashset (long version for large sets)
        /// </summary>
        public long LongCount
        {
            get { return m_count; }
        }

        /// <summary>
        /// Number of elements in this hashset
        /// </summary>
        public int Count
        {
            get
            {
                if (m_count > int.MaxValue)
                    throw new OverflowException($"Count ({m_count}) exceeds int.MaxValue. Use LongCount instead.");
                return (int)m_count;
            }
        }

        /// <summary>
        /// Whether this is readonly
        /// </summary>
        bool ICollection<T>.IsReadOnly
        {
            get { return false; }
        }

        #endregion

        #region IEnumerable methods

        public Enumerator GetEnumerator()
        {
            return new Enumerator(this);
        }

        IEnumerator<T> IEnumerable<T>.GetEnumerator()
        {
            return new Enumerator(this);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return new Enumerator(this);
        }

        #endregion

        #region HashSet methods

        /// <summary>
        /// Add item to this HashSet. Returns bool indicating whether item was added (won't be
        /// added if already present)
        /// </summary>
        /// <param name="item"></param>
        /// <returns>true if added, false if already present</returns>
        public bool Add(T item)
        {
            return AddIfNotPresent(item);
        }

        /// <summary>
        /// Searches the set for a given value and returns the equal value it finds, if any.
        /// </summary>
        /// <param name="equalValue">The value to search for.</param>
        /// <param name="actualValue">The value from the set that the search found, or the default value of <typeparamref name="T"/> when the search yielded no match.</param>
        /// <returns>A value indicating whether the search was successful.</returns>
        /// <remarks>
        /// This can be useful when you want to reuse a previously stored reference instead of
        /// a newly constructed one (so that more sharing of references can occur) or to look up
        /// a value that has more complete data than the value you currently have, although their
        /// comparer functions indicate they are equal.
        /// </remarks>
        public bool TryGetValue(T equalValue, out T actualValue)
        {
            if (m_buckets != null)
            {
                long i = InternalIndexOf(equalValue);
                if (i >= 0)
                {
                    actualValue = m_slots[i].value;
                    return true;
                }
            }

            actualValue = default(T);
            return false;
        }

        /// <summary>
        /// Take the union of this HashSet with other. Modifies this set.
        ///
        /// Implementation note: GetSuggestedCapacity (to increase capacity in advance avoiding
        /// multiple resizes ended up not being useful in practice; quickly gets to the
        /// point where it's a wasteful check.
        /// </summary>
        /// <param name="other">enumerable with items to add</param>
        public void UnionWith(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            foreach (T item in other)
            {
                AddIfNotPresent(item);
            }
        }

        /// <summary>
        /// Takes the intersection of this set with other. Modifies this set.
        ///
        /// Implementation Notes:
        /// We get better perf if other is a hashset using same equality comparer, because we
        /// get constant contains check in other. Resulting cost is O(n1) to iterate over this.
        ///
        /// If we can't go above route, iterate over the other and mark intersection by checking
        /// contains in this. Then loop over and delete any unmarked elements. Total cost is n2+n1.
        ///
        /// Attempts to return early based on counts alone, using the property that the
        /// intersection of anything with the empty set is the empty set.
        /// </summary>
        /// <param name="other">enumerable with items to add </param>
        public void IntersectWith(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            // intersection of anything with empty set is empty set, so return if count is 0
            if (m_count == 0)
            {
                return;
            }

            // if other is empty, intersection is empty set; remove all elements and we're done
            // can only figure this out if implements ICollection<T>. (IEnumerable<T> has no count)
            if (other is ICollection<T> otherAsCollection)
            {
                if (otherAsCollection.Count == 0)
                {
                    Clear();
                    return;
                }

                // faster if other is a hashset using same equality comparer; so check
                // that other is a hashset using the same equality comparer.
                if (other is Hashset<T> otherAsSet && AreEqualityComparersEqual(this, otherAsSet))
                {
                    IntersectWithHashSetWithSameEC(otherAsSet);
                    return;
                }
            }

            IntersectWithEnumerable(other);
        }

        /// <summary>
        /// Remove items in other from this set. Modifies this set.
        /// </summary>
        /// <param name="other">enumerable with items to remove</param>
        public void ExceptWith(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            // this is already the enpty set; return
            if (m_count == 0)
            {
                return;
            }

            // special case if other is this; a set minus itself is the empty set
            if (other == this)
            {
                Clear();
                return;
            }

            // remove every element in other from this
            foreach (T element in other)
            {
                Remove(element);
            }
        }

        /// <summary>
        /// Takes symmetric difference (XOR) with other and this set. Modifies this set.
        /// </summary>
        /// <param name="other">enumerable with items to XOR</param>
        public void SymmetricExceptWith(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            // if set is empty, then symmetric difference is other
            if (m_count == 0)
            {
                UnionWith(other);
                return;
            }

            // special case this; the symmetric difference of a set with itself is the empty set
            if (other == this)
            {
                Clear();
                return;
            }

            // If other is a HashSet, it has unique elements according to its equality comparer,
            // but if they're using different equality comparers, then assumption of uniqueness
            // will fail. So first check if other is a hashset using the same equality comparer;
            // symmetric except is a lot faster and avoids bit array allocations if we can assume
            // uniqueness
            if (other is Hashset<T> otherAsSet && AreEqualityComparersEqual(this, otherAsSet))
            {
                SymmetricExceptWithUniqueHashSet(otherAsSet);
            }
            else
            {
                SymmetricExceptWithEnumerable(other);
            }
        }

        /// <summary>
        /// Checks if this is a subset of other.
        ///
        /// Implementation Notes:
        /// The following properties are used up-front to avoid element-wise checks:
        /// 1. If this is the empty set, then it's a subset of anything, including the empty set
        /// 2. If other has unique elements according to this equality comparer, and this has more
        /// elements than other, then it can't be a subset.
        ///
        /// Furthermore, if other is a hashset using the same equality comparer, we can use a
        /// faster element-wise check.
        /// </summary>
        /// <param name="other"></param>
        /// <returns>true if this is a subset of other; false if not</returns>
        public bool IsSubsetOf(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            // The empty set is a subset of any set
            if (m_count == 0)
            {
                return true;
            }

            // faster if other has unique elements according to this equality comparer; so check
            // that other is a hashset using the same equality comparer.
            if (other is Hashset<T> otherAsSet && AreEqualityComparersEqual(this, otherAsSet))
            {
                // if this has more elements then it can't be a subset
                if (m_count > otherAsSet.m_count)
                {
                    return false;
                }

                // already checked that we're using same equality comparer. simply check that
                // each element in this is contained in other.
                return IsSubsetOfHashSetWithSameEC(otherAsSet);
            }
            else
            {
                ElementCount result = CheckUniqueAndUnfoundElements(other, false);
                return (result.uniqueCount == m_count && result.unfoundCount >= 0);
            }
        }

        /// <summary>
        /// Checks if this is a proper subset of other (i.e. strictly contained in)
        ///
        /// Implementation Notes:
        /// The following properties are used up-front to avoid element-wise checks:
        /// 1. If this is the empty set, then it's a proper subset of a set that contains at least
        /// one element, but it's not a proper subset of the empty set.
        /// 2. If other has unique elements according to this equality comparer, and this has >=
        /// the number of elements in other, then this can't be a proper subset.
        ///
        /// Furthermore, if other is a hashset using the same equality comparer, we can use a
        /// faster element-wise check.
        /// </summary>
        /// <param name="other"></param>
        /// <returns>true if this is a proper subset of other; false if not</returns>
        public bool IsProperSubsetOf(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            if (other is ICollection<T> otherAsCollection)
            {
                // the empty set is a proper subset of anything but the empty set
                if (m_count == 0)
                {
                    return otherAsCollection.Count > 0;
                }

                // faster if other is a hashset (and we're using same equality comparer)
                if (other is Hashset<T> otherAsSet && AreEqualityComparersEqual(this, otherAsSet))
                {
                    if (m_count >= otherAsSet.m_count)
                    {
                        return false;
                    }

                    // this has strictly less than number of items in other, so the following
                    // check suffices for proper subset.
                    return IsSubsetOfHashSetWithSameEC(otherAsSet);
                }
            }

            ElementCount result = CheckUniqueAndUnfoundElements(other, false);
            return (result.uniqueCount == m_count && result.unfoundCount > 0);
        }

        /// <summary>
        /// Checks if this is a superset of other
        ///
        /// Implementation Notes:
        /// The following properties are used up-front to avoid element-wise checks:
        /// 1. If other has no elements (it's the empty set), then this is a superset, even if this
        /// is also the empty set.
        /// 2. If other has unique elements according to this equality comparer, and this has less
        /// than the number of elements in other, then this can't be a superset
        ///
        /// </summary>
        /// <param name="other"></param>
        /// <returns>true if this is a superset of other; false if not</returns>
        public bool IsSupersetOf(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            // try to fall out early based on counts
            if (other is ICollection<T> otherAsCollection)
            {
                // if other is the empty set then this is a superset
                if (otherAsCollection.Count == 0)
                {
                    return true;
                }

                // try to compare based on counts alone if other is a hashset with
                // same equality comparer
                if (other is Hashset<T> otherAsSet && AreEqualityComparersEqual(this, otherAsSet))
                {
                    if (otherAsSet.m_count > m_count)
                    {
                        return false;
                    }
                }
            }

            return ContainsAllElements(other);
        }

        /// <summary>
        /// Checks if this is a proper superset of other (i.e. other strictly contained in this)
        ///
        /// Implementation Notes:
        /// This is slightly more complicated than above because we have to keep track if there
        /// was at least one element not contained in other.
        ///
        /// The following properties are used up-front to avoid element-wise checks:
        /// 1. If this is the empty set, then it can't be a proper superset of any set, even if
        /// other is the empty set.
        /// 2. If other is an empty set and this contains at least 1 element, then this is a proper
        /// superset.
        /// 3. If other has unique elements according to this equality comparer, and other's count
        /// is greater than or equal to this count, then this can't be a proper superset
        ///
        /// Furthermore, if other has unique elements according to this equality comparer, we can
        /// use a faster element-wise check.
        /// </summary>
        /// <param name="other"></param>
        /// <returns>true if this is a proper superset of other; false if not</returns>
        public bool IsProperSupersetOf(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            // the empty set isn't a proper superset of any set.
            if (m_count == 0)
            {
                return false;
            }

            if (other is ICollection<T> otherAsCollection)
            {
                // if other is the empty set then this is a superset
                if (otherAsCollection.Count == 0)
                {
                    // note that this has at least one element, based on above check
                    return true;
                }

                // faster if other is a hashset with the same equality comparer
                if (other is Hashset<T> otherAsSet && AreEqualityComparersEqual(this, otherAsSet))
                {
                    if (otherAsSet.m_count >= m_count)
                    {
                        return false;
                    }

                    // now perform element check
                    return ContainsAllElements(otherAsSet);
                }
            }

            // couldn't fall out in the above cases; do it the long way
            ElementCount result = CheckUniqueAndUnfoundElements(other, true);
            return (result.uniqueCount < m_count && result.unfoundCount == 0);
        }

        /// <summary>
        /// Checks if this set overlaps other (i.e. they share at least one item)
        /// </summary>
        /// <param name="other"></param>
        /// <returns>true if these have at least one common element; false if disjoint</returns>
        public bool Overlaps(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            if (m_count == 0)
            {
                return false;
            }

            foreach (T element in other)
            {
                if (Contains(element))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Checks if this and other contain the same elements. This is set equality:
        /// duplicates and order are ignored
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool SetEquals(IEnumerable<T> other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            Contract.EndContractBlock();

            // faster if other is a hashset and we're using same equality comparer
            if (other is Hashset<T> otherAsSet && AreEqualityComparersEqual(this, otherAsSet))
            {
                // attempt to return early: since both contain unique elements, if they have
                // different counts, then they can't be equal
                if (m_count != otherAsSet.m_count)
                {
                    return false;
                }

                // already confirmed that the sets have the same number of distinct elements, so if
                // one is a superset of the other then they must be equal
                return ContainsAllElements(otherAsSet);
            }
            else
            {
                if (other is ICollection<T> otherAsCollection)
                {
                    // if this count is 0 but other contains at least one element, they can't be equal
                    if (m_count == 0 && otherAsCollection.Count > 0)
                    {
                        return false;
                    }
                }

                ElementCount result = CheckUniqueAndUnfoundElements(other, true);
                return (result.uniqueCount == m_count && result.unfoundCount == 0);
            }
        }

        public void CopyTo(T[] array) { CopyTo(array, 0, m_count); }

        public void CopyTo(T[] array, long arrayIndex, long count)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            Contract.EndContractBlock();

            // check array index valid index into array
            if (arrayIndex < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "SR.GetString(SR.ArgumentOutOfRange_NeedNonNegNum)");
            }

            // also throw if count less than 0
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(count), "SR.GetString(SR.ArgumentOutOfRange_NeedNonNegNum)");
            }

            // will array, starting at arrayIndex, be able to hold elements? Note: not
            // checking arrayIndex >= array.Length (consistency with list of allowing
            // count of 0; subsequent check takes care of the rest)
            if (arrayIndex > array.LongLength || count > array.LongLength - arrayIndex)
            {
                throw new ArgumentException("SR.GetString(SR.Arg_ArrayPlusOffTooSmall)");
            }

            long numCopied = 0;
            for (long i = 0; i < m_lastIndex && numCopied < count; i++)
            {
                if (m_slots[i].hashCode >= 0)
                {
                    array[arrayIndex + numCopied] = m_slots[i].value;
                    numCopied++;
                }
            }
        }

        public static void CopyTo<T>(Hashset<T> src, ArraySlice<T> array) where T : unmanaged
        {
            CopyTo<T>(src, array, 0, src.m_count);
        }

        public static void CopyTo<T>(Hashset<T> src, ArraySlice<T> array, long arrayIndex, long count) where T : unmanaged
        {
            unsafe
            {
                if (src == null)
                    throw new ArgumentNullException(nameof(src));

                Contract.EndContractBlock();

                // check array index valid index into array
                if (arrayIndex < 0)
                    throw new ArgumentOutOfRangeException(nameof(arrayIndex), "SR.GetString(SR.ArgumentOutOfRange_NeedNonNegNum)");

                // also throw if count less than 0
                if (count < 0)
                    throw new ArgumentOutOfRangeException(nameof(count), "SR.GetString(SR.ArgumentOutOfRange_NeedNonNegNum)");

                // will array, starting at arrayIndex, be able to hold elements? Note: not
                // checking arrayIndex >= array.Length (consistency with list of allowing
                // count of 0; subsequent check takes care of the rest)
                if (arrayIndex > array.Count || count > array.Count - arrayIndex)
                {
                    throw new ArgumentException("SR.GetString(SR.Arg_ArrayPlusOffTooSmall)");
                }

                var m_slots = src.m_slots;
                var m_lastIndex = src.m_lastIndex;
                long numCopied = 0;
                T* dst = array.Address;
                for (long i = 0; i < m_lastIndex && numCopied < count; i++)
                {
                    if (m_slots[i].hashCode >= 0)
                    {
                        dst[arrayIndex + numCopied] = m_slots[i].value;
                        numCopied++;
                    }
                }
            }
        }

        /// <summary>
        /// Remove elements that match specified predicate. Returns the number of elements removed
        /// </summary>
        /// <param name="match"></param>
        /// <returns></returns>
        public long RemoveWhere(Predicate<T> match)
        {
            if (match == null)
            {
                throw new ArgumentNullException(nameof(match));
            }

            Contract.EndContractBlock();

            long numRemoved = 0;
            for (long i = 0; i < m_lastIndex; i++)
            {
                if (m_slots[i].hashCode >= 0)
                {
                    // cache value in case delegate removes it
                    T value = m_slots[i].value;
                    if (match(value))
                    {
                        // check again that remove actually removed it
                        if (Remove(value))
                        {
                            numRemoved++;
                        }
                    }
                }
            }

            return numRemoved;
        }

        /// <summary>
        /// Gets the IEqualityComparer that is used to determine equality of keys for
        /// the HashSet.
        /// </summary>
        public IEqualityComparer<T> Comparer
        {
            get
            {
                return m_comparer;
            }
        }

        /// <summary>
        /// Sets the capacity of this list to the size of the list (rounded up to nearest prime),
        /// unless count is 0, in which case we release references.
        ///
        /// This method can be used to minimize a list's memory overhead once it is known that no
        /// new elements will be added to the list. To completely clear a list and release all
        /// memory referenced by the list, execute the following statements:
        ///
        /// list.Clear();
        /// list.TrimExcess();
        /// </summary>
        public void TrimExcess()
        {
            Debug.Assert(m_count >= 0, "m_count is negative");

            if (m_count == 0)
            {
                // if count is zero, clear references
                m_buckets = null;
                m_slots = null;
                m_version++;
            }
            else
            {
                Debug.Assert(m_buckets != null, "m_buckets was null but m_count > 0");

                // similar to IncreaseCapacity but moves down elements in case add/remove/etc
                // caused fragmentation
                long newSize = HashHelpersLong.GetPrime(m_count);
                Slot[] newSlots = new Slot[newSize];
                long[] newBuckets = new long[newSize];

                // move down slots and rehash at the same time. newIndex keeps track of current
                // position in newSlots array
                long newIndex = 0;
                for (long i = 0; i < m_lastIndex; i++)
                {
                    if (m_slots[i].hashCode >= 0)
                    {
                        newSlots[newIndex] = m_slots[i];

                        // rehash
                        long bucket = (long)(((uint)newSlots[newIndex].hashCode) % (ulong)newSize);
                        newSlots[newIndex].next = newBuckets[bucket] - 1;
                        newBuckets[bucket] = newIndex + 1;

                        newIndex++;
                    }
                }

                Debug.Assert(newSlots.LongLength <= m_slots.LongLength, "capacity increased after TrimExcess");

                m_lastIndex = newIndex;
                m_slots = newSlots;
                m_buckets = newBuckets;
                m_freeList = -1;
            }
        }

        /// <summary>
        /// Used for deep equality of HashSet testing
        /// </summary>
        /// <returns></returns>
        public static IEqualityComparer<Hashset<T>> CreateSetComparer()
        {
            return new HashSetEqualityComparer<T>();
        }

        /// <summary>
        /// Equality comparer for hashsets of hashsets
        /// </summary>
        /// <typeparam name="T"></typeparam>
        internal class HashSetEqualityComparer<T> : IEqualityComparer<Hashset<T>>
        {
            private IEqualityComparer<T> m_comparer;

            public HashSetEqualityComparer()
            {
                m_comparer = EqualityComparer<T>.Default;
            }

            public HashSetEqualityComparer(IEqualityComparer<T> comparer)
            {
                if (comparer == null)
                {
                    m_comparer = EqualityComparer<T>.Default;
                }
                else
                {
                    m_comparer = comparer;
                }
            }

            // using m_comparer to keep equals properties in tact; don't want to choose one of the comparers
            public bool Equals(Hashset<T> x, Hashset<T> y)
            {
                return Hashset<T>.HashSetEquals(x, y, m_comparer);
            }

            public int GetHashCode(Hashset<T> obj)
            {
                int hashCode = 0;
                if (obj != null)
                {
                    foreach (T t in obj)
                    {
                        hashCode = hashCode ^ (m_comparer.GetHashCode(t) & 0x7FFFFFFF);
                    }
                } // else returns hashcode of 0 for null hashsets

                return hashCode;
            }

            // Equals method for the comparer itself.
            public override bool Equals(Object obj)
            {
                if (!(obj is HashSetEqualityComparer<T> comparer))
                {
                    return false;
                }

                return (this.m_comparer == comparer.m_comparer);
            }

            public override int GetHashCode()
            {
                return m_comparer.GetHashCode();
            }
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Initializes buckets and slots arrays. Uses suggested capacity by finding next prime
        /// greater than or equal to capacity.
        /// </summary>
        /// <param name="capacity"></param>
        private void Initialize(long capacity)
        {
            Debug.Assert(m_buckets == null, "Initialize was called but m_buckets was non-null");

            long size = HashHelpersLong.GetPrime(capacity);

            m_buckets = new long[size];
            m_slots = new Slot[size];
        }

        /// <summary>
        /// Expand to new capacity. New capacity is next prime greater than or equal to suggested
        /// size. This is called when the underlying array is filled. This performs no
        /// defragmentation, allowing faster execution; note that this is reasonable since
        /// AddIfNotPresent attempts to insert new elements in re-opened spots.
        /// </summary>
        /// <param name="sizeSuggestion"></param>
        private void IncreaseCapacity()
        {
            Debug.Assert(m_buckets != null, "IncreaseCapacity called on a set with no elements");

            long newSize = HashHelpersLong.ExpandPrime(m_count);
            if (newSize <= m_count)
            {
                throw new ArgumentException("SR.GetString(SR.Arg_HSCapacityOverflow)");
            }

            // Able to increase capacity; copy elements to larger array and rehash
            SetCapacity(newSize, false);
        }

        /// <summary>
        /// Set the underlying buckets array to size newSize and rehash.  Note that newSize
        /// *must* be a prime.  It is very likely that you want to call IncreaseCapacity()
        /// instead of this method.
        /// </summary>
        private void SetCapacity(long newSize, bool forceNewHashCodes)
        {
            Contract.Assert(HashHelpersLong.IsPrime(newSize), "New size is not prime!");

            Contract.Assert(m_buckets != null, "SetCapacity called on a set with no elements");

            Slot[] newSlots = new Slot[newSize];
            if (m_slots != null)
            {
                Array.Copy(m_slots, 0, newSlots, 0, m_lastIndex);
            }

            if (forceNewHashCodes)
            {
                for (long i = 0; i < m_lastIndex; i++)
                {
                    if (newSlots[i].hashCode != -1)
                    {
                        newSlots[i].hashCode = InternalGetHashCode(newSlots[i].value);
                    }
                }
            }

            long[] newBuckets = new long[newSize];
            for (long i = 0; i < m_lastIndex; i++)
            {
                long bucket = (long)(((uint)newSlots[i].hashCode) % (ulong)newSize);
                newSlots[i].next = newBuckets[bucket] - 1;
                newBuckets[bucket] = i + 1;
            }

            m_slots = newSlots;
            m_buckets = newBuckets;
        }

        /// <summary>
        /// Adds value to HashSet if not contained already
        /// Returns true if added and false if already present
        /// </summary>
        /// <param name="value">value to find</param>
        /// <returns></returns>
        private bool AddIfNotPresent(T value)
        {
            if (m_buckets == null)
            {
                Initialize(0);
            }

            int hashCode = InternalGetHashCode(value);
            long bucketIndex = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);
            for (long i = m_buckets[bucketIndex] - 1; i >= 0; i = m_slots[i].next)
            {
                if (m_slots[i].hashCode == hashCode && m_comparer.Equals(m_slots[i].value, value))
                {
                    return false;
                }
            }

            long index;
            if (m_freeList >= 0)
            {
                index = m_freeList;
                m_freeList = m_slots[index].next;
            }
            else
            {
                if (m_lastIndex == m_slots.LongLength)
                {
                    IncreaseCapacity();
                    // this will change during resize
                    bucketIndex = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);
                }

                index = m_lastIndex;
                m_lastIndex++;
            }

            m_slots[index].hashCode = hashCode;
            m_slots[index].value = value;
            m_slots[index].next = m_buckets[bucketIndex] - 1;
            m_buckets[bucketIndex] = index + 1;
            m_count++;
            m_version++;

            return true;
        }

        // Add value at known index with known hash code. Used only
        // when constructing from another HashSet.
        private void AddValue(long index, int hashCode, T value)
        {
            long bucket = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);

#if DEBUG
            Debug.Assert(InternalGetHashCode(value) == hashCode);
            for (long i = m_buckets[bucket] - 1; i >= 0; i = m_slots[i].next)
            {
                Debug.Assert(!m_comparer.Equals(m_slots[i].value, value));
            }
#endif

            Debug.Assert(m_freeList == -1);
            m_slots[index].hashCode = hashCode;
            m_slots[index].value = value;
            m_slots[index].next = m_buckets[bucket] - 1;
            m_buckets[bucket] = index + 1;
        }

        /// <summary>
        /// Checks if this contains of other's elements. Iterates over other's elements and
        /// returns false as soon as it finds an element in other that's not in this.
        /// Used by SupersetOf, ProperSupersetOf, and SetEquals.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        private bool ContainsAllElements(IEnumerable<T> other)
        {
            foreach (T element in other)
            {
                if (!Contains(element))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Implementation Notes:
        /// If other is a hashset and is using same equality comparer, then checking subset is
        /// faster. Simply check that each element in this is in other.
        ///
        /// Note: if other doesn't use same equality comparer, then Contains check is invalid,
        /// which is why callers must take are of this.
        ///
        /// If callers are concerned about whether this is a proper subset, they take care of that.
        ///
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        private bool IsSubsetOfHashSetWithSameEC(Hashset<T> other)
        {
            foreach (T item in this)
            {
                if (!other.Contains(item))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// If other is a hashset that uses same equality comparer, intersect is much faster
        /// because we can use other's Contains
        /// </summary>
        /// <param name="other"></param>
        private void IntersectWithHashSetWithSameEC(Hashset<T> other)
        {
            for (long i = 0; i < m_lastIndex; i++)
            {
                if (m_slots[i].hashCode >= 0)
                {
                    T item = m_slots[i].value;
                    if (!other.Contains(item))
                    {
                        Remove(item);
                    }
                }
            }
        }

        /// <summary>
        /// Iterate over other. If contained in this, mark an element in bit array corresponding to
        /// its position in m_slots. If anything is unmarked (in bit array), remove it.
        ///
        /// This attempts to allocate on the stack, if below StackAllocThreshold.
        /// </summary>
        /// <param name="other"></param>
        private unsafe void IntersectWithEnumerable(IEnumerable<T> other)
        {
            Debug.Assert(m_buckets != null, "m_buckets shouldn't be null; callers should check first");

            // keep track of current last index; don't want to move past the end of our bit array
            // (could happen if another thread is modifying the collection)
            long originalLastIndex = m_lastIndex;
            long intArrayLength = BitHelperLong.ToLongArrayLength(originalLastIndex);

            BitHelperLong bitHelper;
            if (intArrayLength <= StackAllocThreshold)
            {
                long* bitArrayPtr = stackalloc long[(int)intArrayLength];
                bitHelper = new BitHelperLong(bitArrayPtr, intArrayLength);
            }
            else
            {
                long[] bitArray = new long[intArrayLength];
                bitHelper = new BitHelperLong(bitArray, intArrayLength);
            }

            // mark if contains: find index of in slots array and mark corresponding element in bit array
            foreach (T item in other)
            {
                long index = InternalIndexOf(item);
                if (index >= 0)
                {
                    bitHelper.MarkBit(index);
                }
            }

            // if anything unmarked, remove it. Perf can be optimized here if BitHelper had a
            // FindFirstUnmarked method.
            for (long i = 0; i < originalLastIndex; i++)
            {
                if (m_slots[i].hashCode >= 0 && !bitHelper.IsMarked(i))
                {
                    Remove(m_slots[i].value);
                }
            }
        }

        /// <summary>
        /// Used internally by set operations which have to rely on bit array marking. This is like
        /// Contains but returns index in slots array.
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        private long InternalIndexOf(T item)
        {
            Debug.Assert(m_buckets != null, "m_buckets was null; callers should check first");

            int hashCode = InternalGetHashCode(item);
            long bucketIndex = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);
            for (long i = m_buckets[bucketIndex] - 1; i >= 0; i = m_slots[i].next)
            {
                if ((m_slots[i].hashCode) == hashCode && m_comparer.Equals(m_slots[i].value, item))
                {
                    return i;
                }
            }

            // wasn't found
            return -1;
        }

        /// <summary>
        /// if other is a set, we can assume it doesn't have duplicate elements, so use this
        /// technique: if can't remove, then it wasn't present in this set, so add.
        ///
        /// As with other methods, callers take care of ensuring that other is a hashset using the
        /// same equality comparer.
        /// </summary>
        /// <param name="other"></param>
        private void SymmetricExceptWithUniqueHashSet(Hashset<T> other)
        {
            foreach (T item in other)
            {
                if (!Remove(item))
                {
                    AddIfNotPresent(item);
                }
            }
        }

        /// <summary>
        /// Implementation notes:
        ///
        /// Used for symmetric except when other isn't a HashSet. This is more tedious because
        /// other may contain duplicates. HashSet technique could fail in these situations:
        /// 1. Other has a duplicate that's not in this: HashSet technique would add then
        /// remove it.
        /// 2. Other has a duplicate that's in this: HashSet technique would remove then add it
        /// back.
        /// In general, its presence would be toggled each time it appears in other.
        ///
        /// This technique uses bit marking to indicate whether to add/remove the item. If already
        /// present in collection, it will get marked for deletion. If added from other, it will
        /// get marked as something not to remove.
        ///
        /// </summary>
        /// <param name="other"></param>
        private unsafe void SymmetricExceptWithEnumerable(IEnumerable<T> other)
        {
            long originalLastIndex = m_lastIndex;
            long intArrayLength = BitHelperLong.ToLongArrayLength(originalLastIndex);

            BitHelperLong itemsToRemove;
            BitHelperLong itemsAddedFromOther;
            if (intArrayLength <= StackAllocThreshold / 2)
            {
                long* itemsToRemovePtr = stackalloc long[(int)intArrayLength];
                itemsToRemove = new BitHelperLong(itemsToRemovePtr, intArrayLength);

                long* itemsAddedFromOtherPtr = stackalloc long[(int)intArrayLength];
                itemsAddedFromOther = new BitHelperLong(itemsAddedFromOtherPtr, intArrayLength);
            }
            else
            {
                long[] itemsToRemoveArray = new long[intArrayLength];
                itemsToRemove = new BitHelperLong(itemsToRemoveArray, intArrayLength);

                long[] itemsAddedFromOtherArray = new long[intArrayLength];
                itemsAddedFromOther = new BitHelperLong(itemsAddedFromOtherArray, intArrayLength);
            }

            foreach (T item in other)
            {
                long location = 0;
                bool added = AddOrGetLocation(item, out location);
                if (added)
                {
                    // wasn't already present in collection; flag it as something not to remove
                    // *NOTE* if location is out of range, we should ignore. BitHelper will
                    // detect that it's out of bounds and not try to mark it. But it's
                    // expected that location could be out of bounds because adding the item
                    // will increase m_lastIndex as soon as all the free spots are filled.
                    itemsAddedFromOther.MarkBit(location);
                }
                else
                {
                    // already there...if not added from other, mark for remove.
                    // *NOTE* Even though BitHelper will check that location is in range, we want
                    // to check here. There's no point in checking items beyond originalLastIndex
                    // because they could not have been in the original collection
                    if (location < originalLastIndex && !itemsAddedFromOther.IsMarked(location))
                    {
                        itemsToRemove.MarkBit(location);
                    }
                }
            }

            // if anything marked, remove it
            for (long i = 0; i < originalLastIndex; i++)
            {
                if (itemsToRemove.IsMarked(i))
                {
                    Remove(m_slots[i].value);
                }
            }
        }

        /// <summary>
        /// Add if not already in hashset. Returns an out param indicating index where added. This
        /// is used by SymmetricExcept because it needs to know the following things:
        /// - whether the item was already present in the collection or added from other
        /// - where it's located (if already present, it will get marked for removal, otherwise
        /// marked for keeping)
        /// </summary>
        /// <param name="value"></param>
        /// <param name="location"></param>
        /// <returns></returns>
        private bool AddOrGetLocation(T value, out long location)
        {
            Debug.Assert(m_buckets != null, "m_buckets is null, callers should have checked");

            int hashCode = InternalGetHashCode(value);
            long bucketIndex = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);
            for (long i = m_buckets[bucketIndex] - 1; i >= 0; i = m_slots[i].next)
            {
                if (m_slots[i].hashCode == hashCode && m_comparer.Equals(m_slots[i].value, value))
                {
                    location = i;
                    return false; //already present
                }
            }

            long index;
            if (m_freeList >= 0)
            {
                index = m_freeList;
                m_freeList = m_slots[index].next;
            }
            else
            {
                if (m_lastIndex == m_slots.LongLength)
                {
                    IncreaseCapacity();
                    // this will change during resize
                    bucketIndex = (long)(((uint)hashCode) % (ulong)m_buckets.LongLength);
                }

                index = m_lastIndex;
                m_lastIndex++;
            }

            m_slots[index].hashCode = hashCode;
            m_slots[index].value = value;
            m_slots[index].next = m_buckets[bucketIndex] - 1;
            m_buckets[bucketIndex] = index + 1;
            m_count++;
            m_version++;
            location = index;
            return true;
        }

        /// <summary>
        /// Determines counts that can be used to determine equality, subset, and superset. This
        /// is only used when other is an IEnumerable and not a HashSet. If other is a HashSet
        /// these properties can be checked faster without use of marking because we can assume
        /// other has no duplicates.
        ///
        /// The following count checks are performed by callers:
        /// 1. Equals: checks if unfoundCount = 0 and uniqueFoundCount = m_count; i.e. everything
        /// in other is in this and everything in this is in other
        /// 2. Subset: checks if unfoundCount >= 0 and uniqueFoundCount = m_count; i.e. other may
        /// have elements not in this and everything in this is in other
        /// 3. Proper subset: checks if unfoundCount > 0 and uniqueFoundCount = m_count; i.e
        /// other must have at least one element not in this and everything in this is in other
        /// 4. Proper superset: checks if unfound count = 0 and uniqueFoundCount strictly less
        /// than m_count; i.e. everything in other was in this and this had at least one element
        /// not contained in other.
        ///
        /// An earlier implementation used delegates to perform these checks rather than returning
        /// an ElementCount struct; however this was changed due to the perf overhead of delegates.
        /// </summary>
        /// <param name="other"></param>
        /// <param name="returnIfUnfound">Allows us to finish faster for equals and proper superset
        /// because unfoundCount must be 0.</param>
        /// <returns></returns>
        private unsafe ElementCount CheckUniqueAndUnfoundElements(IEnumerable<T> other, bool returnIfUnfound)
        {
            ElementCount result;

            // need special case in case this has no elements.
            if (m_count == 0)
            {
                long numElementsInOther = 0;
                foreach (T item in other)
                {
                    numElementsInOther++;
                    // break right away, all we want to know is whether other has 0 or 1 elements
                    break;
                }

                result.uniqueCount = 0;
                result.unfoundCount = numElementsInOther;
                return result;
            }


            Debug.Assert((m_buckets != null) && (m_count > 0), "m_buckets was null but count greater than 0");

            long originalLastIndex = m_lastIndex;
            long intArrayLength = BitHelperLong.ToLongArrayLength(originalLastIndex);

            BitHelperLong bitHelper;
            if (intArrayLength <= StackAllocThreshold)
            {
                long* bitArrayPtr = stackalloc long[(int)intArrayLength];
                bitHelper = new BitHelperLong(bitArrayPtr, intArrayLength);
            }
            else
            {
                long[] bitArray = new long[intArrayLength];
                bitHelper = new BitHelperLong(bitArray, intArrayLength);
            }

            // count of items in other not found in this
            long unfoundCount = 0;
            // count of unique items in other found in this
            long uniqueFoundCount = 0;

            foreach (T item in other)
            {
                long index = InternalIndexOf(item);
                if (index >= 0)
                {
                    if (!bitHelper.IsMarked(index))
                    {
                        // item hasn't been seen yet
                        bitHelper.MarkBit(index);
                        uniqueFoundCount++;
                    }
                }
                else
                {
                    unfoundCount++;
                    if (returnIfUnfound)
                    {
                        break;
                    }
                }
            }

            result.uniqueCount = uniqueFoundCount;
            result.unfoundCount = unfoundCount;
            return result;
        }

        /// <summary>
        /// Copies this to an array. Used for DebugView
        /// </summary>
        /// <returns></returns>
        internal T[] ToArray()
        {
            T[] newArray = new T[m_count];
            CopyTo(newArray);
            return newArray;
        }

        /// <summary>
        /// Internal method used for HashSetEqualityComparer. Compares set1 and set2 according
        /// to specified comparer.
        ///
        /// Because items are hashed according to a specific equality comparer, we have to resort
        /// to n^2 search if they're using different equality comparers.
        /// </summary>
        /// <param name="set1"></param>
        /// <param name="set2"></param>
        /// <param name="comparer"></param>
        /// <returns></returns>
        internal static bool HashSetEquals(Hashset<T> set1, Hashset<T> set2, IEqualityComparer<T> comparer)
        {
            // handle null cases first
            if (set1 == null)
            {
                return (set2 == null);
            }
            else if (set2 == null)
            {
                // set1 != null
                return false;
            }

            // all comparers are the same; this is faster
            if (AreEqualityComparersEqual(set1, set2))
            {
                if (set1.m_count != set2.m_count)
                {
                    return false;
                }

                // suffices to check subset
                foreach (T item in set2)
                {
                    if (!set1.Contains(item))
                    {
                        return false;
                    }
                }

                return true;
            }
            else
            {
                // n^2 search because items are hashed according to their respective ECs
                foreach (T set2Item in set2)
                {
                    bool found = false;
                    foreach (T set1Item in set1)
                    {
                        if (comparer.Equals(set2Item, set1Item))
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        /// <summary>
        /// Checks if equality comparers are equal. This is used for algorithms that can
        /// speed up if it knows the other item has unique elements. I.e. if they're using
        /// different equality comparers, then uniqueness assumption between sets break.
        /// </summary>
        /// <param name="set1"></param>
        /// <param name="set2"></param>
        /// <returns></returns>
        private static bool AreEqualityComparersEqual(Hashset<T> set1, Hashset<T> set2)
        {
            return set1.Comparer.Equals(set2.Comparer);
        }

        /// <summary>
        /// Workaround Comparers that throw ArgumentNullException for GetHashCode(null).
        /// </summary>
        /// <param name="item"></param>
        /// <returns>hash code</returns>
        private int InternalGetHashCode(T item)
        {
            if (item == null)
            {
                return 0;
            }

            return m_comparer.GetHashCode(item) & Lower31BitMask;
        }

        #endregion

        // used for set checking operations (using enumerables) that rely on counting
        internal struct ElementCount
        {
            internal long uniqueCount;
            internal long unfoundCount;
        }

        internal struct Slot
        {
            internal int hashCode;      // Lower 31 bits of hash code, -1 if unused
            internal long next;         // Index of next entry, -1 if last
            internal T value;
        }

        public struct Enumerator : IEnumerator<T>, IEnumerator
        {
            private Hashset<T> set;
            private long index;
            private int version;
            private T current;

            internal Enumerator(Hashset<T> set)
            {
                this.set = set;
                index = 0;
                version = set.m_version;
                current = default(T);
            }

            public void Dispose() { }

            public bool MoveNext()
            {
                if (version != set.m_version)
                {
                    throw new InvalidOperationException("SR.GetString(SR.InvalidOperation_EnumFailedVersion)");
                }

                while (index < set.m_lastIndex)
                {
                    if (set.m_slots[index].hashCode >= 0)
                    {
                        current = set.m_slots[index].value;
                        index++;
                        return true;
                    }

                    index++;
                }

                index = set.m_lastIndex + 1;
                current = default(T);
                return false;
            }

            public T Current
            {
                get
                {
                    return current;
                }
            }

            Object IEnumerator.Current
            {
                get
                {
                    if (index == 0 || index == set.m_lastIndex + 1)
                    {
                        throw new InvalidOperationException("SR.GetString(SR.InvalidOperation_EnumOpCantHappen)");
                    }

                    return Current;
                }
            }

            void IEnumerator.Reset()
            {
                if (version != set.m_version)
                {
                    throw new InvalidOperationException("SR.GetString(SR.InvalidOperation_EnumFailedVersion)");
                }

                index = 0;
                current = default(T);
            }
        }
    }


    /// <summary>
    /// BitHelper for long-indexed collections supporting more than int.MaxValue elements.
    /// Uses long[] internally for bit storage.
    /// </summary>
    unsafe internal class BitHelperLong
    {
        private const byte MarkedBitFlag = 1;
        private const int LongSize = 64;

        // length of underlying long array (not logical bit array)
        private long m_length;

        // ptr to stack alloc'd array of longs
        private long* m_arrayPtr;

        // array of longs
        private long[] m_array;

        // whether to operate on stack alloc'd or heap alloc'd array
        private bool useStackAlloc;

        /// <summary>
        /// Instantiates a BitHelperLong with a stack alloc'd array of longs
        /// </summary>
        internal BitHelperLong(long* bitArrayPtr, long length)
        {
            this.m_arrayPtr = bitArrayPtr;
            this.m_length = length;
            useStackAlloc = true;
        }

        /// <summary>
        /// Instantiates a BitHelperLong with a heap alloc'd array of longs
        /// </summary>
        internal BitHelperLong(long[] bitArray, long length)
        {
            this.m_array = bitArray;
            this.m_length = length;
        }

        /// <summary>
        /// Mark bit at specified position
        /// </summary>
        internal unsafe void MarkBit(long bitPosition)
        {
            long bitArrayIndex = bitPosition / LongSize;
            if (bitArrayIndex < m_length && bitArrayIndex >= 0)
            {
                long mask = 1L << (int)(bitPosition % LongSize);
                if (useStackAlloc)
                {
                    m_arrayPtr[bitArrayIndex] |= mask;
                }
                else
                {
                    m_array[bitArrayIndex] |= mask;
                }
            }
        }

        /// <summary>
        /// Is bit at specified position marked?
        /// </summary>
        internal unsafe bool IsMarked(long bitPosition)
        {
            long bitArrayIndex = bitPosition / LongSize;
            if (bitArrayIndex < m_length && bitArrayIndex >= 0)
            {
                long mask = 1L << (int)(bitPosition % LongSize);
                if (useStackAlloc)
                {
                    return (m_arrayPtr[bitArrayIndex] & mask) != 0;
                }
                else
                {
                    return (m_array[bitArrayIndex] & mask) != 0;
                }
            }

            return false;
        }

        /// <summary>
        /// How many longs must be allocated to represent n bits. Returns (n+63)/64, but
        /// avoids overflow
        /// </summary>
        internal static long ToLongArrayLength(long n)
        {
            return n > 0 ? ((n - 1) / LongSize + 1) : 0;
        }
    }

    /// <summary>
    /// Hash helpers for long-indexed collections.
    /// Supports capacity beyond int.MaxValue and uses 33% growth for very large collections.
    /// </summary>
    internal static class HashHelpersLong
    {
        // Threshold above which we use 33% growth instead of doubling (1 billion elements)
        public const long LargeGrowthThreshold = 1_000_000_000L;

        // Maximum supported array size (limited by .NET array indexing with long)
        // In practice, limited by available memory
        public const long MaxPrimeArrayLength = 0x7FFFFFC7L; // Same as Array.MaxLength on 64-bit

        // Precomputed primes for small sizes
        public static readonly long[] primes = new long[] {
            3, 7, 11, 17, 23, 29, 37, 47, 59, 71, 89, 107, 131, 163, 197, 239, 293, 353, 431, 521,
            631, 761, 919, 1103, 1327, 1597, 1931, 2333, 2801, 3371, 4049, 4861, 5839, 7013, 8419,
            10103, 12143, 14591, 17519, 21023, 25229, 30293, 36353, 43627, 52361, 62851, 75431,
            90523, 108631, 130363, 156437, 187751, 225307, 270371, 324449, 389357, 467237, 560689,
            672827, 807403, 968897, 1162687, 1395263, 1674319, 2009191, 2411033, 2893249, 3471899,
            4166287, 4999559, 5999471, 7199369,
            // Extended primes for large collections
            8639243, 10367101, 12440537, 14928671, 17914409, 21497293, 25796759, 30956117,
            37147349, 44576827, 53492203, 64190647, 77028793, 92434559, 110921473, 133105769,
            159726947, 191672339, 230006821, 276008189, 331209833, 397451801, 476942167,
            572330603, 686796727, 824156077, 988987301, 1186784761, 1424141717, 1708970063,
            2050764077, 2460916897, 2953100281, 3543720337, 4252464407, 5102957291, 6123548753,
            7348258507, 8817910211, 10581492263, 12697790717, 15237348869, 18284818643,
            21941782381, 26330138861, 31596166633, 37915399963
        };

        public static bool IsPrime(long candidate)
        {
            if (candidate < 2)
                return false;
            if ((candidate & 1) == 0)
                return candidate == 2;

            long limit = (long)Math.Sqrt((double)candidate);
            for (long divisor = 3; divisor <= limit; divisor += 2)
            {
                if (candidate % divisor == 0)
                    return false;
            }

            return true;
        }

        public static long GetPrime(long min)
        {
            if (min < 0)
                throw new ArgumentException("Capacity overflow");

            // First check the precomputed primes table
            for (int i = 0; i < primes.Length; i++)
            {
                long prime = primes[i];
                if (prime >= min)
                    return prime;
            }

            // Outside of our predefined table; compute the prime the hard way
            for (long candidate = min | 1; candidate < long.MaxValue; candidate += 2)
            {
                if (IsPrime(candidate) && (candidate - 1) % 101 != 0)
                    return candidate;
            }

            return min;
        }

        /// <summary>
        /// Expands to a new capacity.
        /// For collections below LargeGrowthThreshold (1 billion), doubles the size.
        /// For larger collections, grows by 33% to avoid excessive memory allocation.
        /// </summary>
        public static long ExpandPrime(long oldSize)
        {
            long newSize;

            if (oldSize < LargeGrowthThreshold)
            {
                // Standard doubling for smaller collections
                newSize = 2 * oldSize;
            }
            else
            {
                // 33% growth for very large collections to avoid OOM
                newSize = oldSize + (oldSize / 3);
            }

            // Handle overflow
            if (newSize < oldSize)
            {
                // Overflow occurred
                if (MaxPrimeArrayLength > oldSize)
                    return GetPrime(MaxPrimeArrayLength);
                return oldSize; // Can't grow further
            }

            // Cap at max array length
            if (newSize > MaxPrimeArrayLength && MaxPrimeArrayLength > oldSize)
            {
                return GetPrime(MaxPrimeArrayLength);
            }

            return GetPrime(newSize);
        }
    }
}
