using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.Serialization;
using System.Threading;

namespace NumSharp.Utilities
{
    [DebuggerDisplay("Count = {Count}")]
    [Serializable]
    public class ConcurrentHashSet<T> : ICollection<T>, ISet<T>, ISerializable, IDeserializationCallback
    {
        private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim(LockRecursionPolicy.SupportsRecursion);

        private readonly Hashset<T> hashset = new Hashset<T>();

        public ConcurrentHashSet()
        { }

        public ConcurrentHashSet(IEqualityComparer<T> comparer)
        {
            hashset = new Hashset<T>(comparer);
        }

        public ConcurrentHashSet(IEnumerable<T> collection)
        {
            hashset = new Hashset<T>(collection);
        }

        public ConcurrentHashSet(IEnumerable<T> collection, IEqualityComparer<T> comparer)
        {
            hashset = new Hashset<T>(collection, comparer);
        }

        protected ConcurrentHashSet(SerializationInfo info, StreamingContext context)
        {
            hashset = new Hashset<T>();

            // not sure about this one really...
            var iSerializable = hashset as ISerializable;
            iSerializable.GetObjectData(info, context);
        }

        #region Dispose

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
                if (_lock != null)
                    _lock.Dispose();
        }

        public IEnumerator<T> GetEnumerator()
        {
            return hashset.GetEnumerator();
        }

        ~ConcurrentHashSet()
        {
            Dispose(false);
        }

        public void OnDeserialization(object sender)
        {
            hashset.OnDeserialization(sender);
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            hashset.GetObjectData(info, context);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        public bool Add(T item)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.Add(item);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        void ICollection<T>.Add(T item)
        {
            _lock.EnterWriteLock();
            try
            {
                hashset.Add(item);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public void UnionWith(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            _lock.EnterReadLock();
            try
            {
                hashset.UnionWith(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
                if (_lock.IsReadLockHeld) _lock.ExitReadLock();
            }
        }

        public void IntersectWith(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            _lock.EnterReadLock();
            try
            {
                hashset.IntersectWith(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
                if (_lock.IsReadLockHeld) _lock.ExitReadLock();
            }
        }

        public void ExceptWith(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            _lock.EnterReadLock();
            try
            {
                hashset.ExceptWith(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
                if (_lock.IsReadLockHeld) _lock.ExitReadLock();
            }
        }

        public void SymmetricExceptWith(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            try
            {
                hashset.SymmetricExceptWith(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool IsSubsetOf(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.IsSubsetOf(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool IsSupersetOf(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.IsSupersetOf(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool IsProperSupersetOf(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.IsProperSupersetOf(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool IsProperSubsetOf(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.IsProperSubsetOf(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool Overlaps(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.Overlaps(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool SetEquals(IEnumerable<T> other)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.SetEquals(other);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        bool ISet<T>.Add(T item)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.Add(item);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public void Clear()
        {
            _lock.EnterWriteLock();
            try
            {
                hashset.Clear();
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool Contains(T item)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.Contains(item);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            _lock.EnterWriteLock();
            try
            {
                hashset.CopyTo(array, arrayIndex);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public bool Remove(T item)
        {
            _lock.EnterWriteLock();
            try
            {
                return hashset.Remove(item);
            }
            finally
            {
                if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
            }
        }

        public int Count
        {
            get
            {
                _lock.EnterWriteLock();
                try
                {
                    return hashset.Count;
                }
                finally
                {
                    if (_lock.IsWriteLockHeld) _lock.ExitWriteLock();
                }
            }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }
    }
}
