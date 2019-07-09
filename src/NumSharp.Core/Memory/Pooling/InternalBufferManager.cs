using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Security;
using System.Security.Permissions;
using System.Threading;
using OOMath.MemoryPooling.System.Runtime;

namespace OOMath.MemoryPooling {
    public abstract class InternalBufferManager {
        protected InternalBufferManager() { }

        public abstract byte[] TakeBuffer(int bufferSize);
        public abstract void ReturnBuffer(byte[] buffer);
        public abstract void Clear();

        public static InternalBufferManager Create(long maxBufferPoolSize, int maxBufferSize) {
            if (maxBufferPoolSize == 0) {
                return GCBufferManager.Value;
            } else {
                //Fx.Assert(maxBufferPoolSize > 0 && maxBufferSize >= 0, "bad params, caller should verify");
                return new PooledBufferManager(maxBufferPoolSize, maxBufferSize);
            }
        }

        public class PooledBufferManager : InternalBufferManager {
            const int minBufferSize = 128;
            const int maxMissesBeforeTuning = 8;
            const int initialBufferCount = 1;
            readonly object tuningLock;

            readonly int[] bufferSizes;
            readonly BufferPool[] bufferPools;
            long memoryLimit;
            long remainingMemory;
            bool areQuotasBeingTuned;
            int totalMisses;
#if DEBUG
            ConcurrentDictionary<int, string> buffersPooled = new ConcurrentDictionary<int, string>();
#endif //DEBUG

            [MethodImpl((MethodImplOptions) 512)]
            public PooledBufferManager(long maxMemoryToPool, int maxBufferSize) {
                this.tuningLock = new object();
                this.memoryLimit = maxMemoryToPool;
                this.remainingMemory = maxMemoryToPool;
                List<BufferPool> bufferPoolList = new List<BufferPool>();

                for (int bufferSize = minBufferSize;;) {
                    long bufferCountLong = this.remainingMemory / bufferSize;

                    int bufferCount = bufferCountLong > int.MaxValue ? int.MaxValue : (int) bufferCountLong;

                    if (bufferCount > initialBufferCount) {
                        bufferCount = initialBufferCount;
                    }

                    bufferPoolList.Add(BufferPool.CreatePool(bufferSize, bufferCount));

                    this.remainingMemory -= (long) bufferCount * bufferSize;

                    if (bufferSize >= maxBufferSize) {
                        break;
                    }

                    long newBufferSizeLong = (long) bufferSize * 2;

                    if (newBufferSizeLong > (long) maxBufferSize) {
                        bufferSize = maxBufferSize;
                    } else {
                        bufferSize = (int) newBufferSizeLong;
                    }
                }

                this.bufferPools = bufferPoolList.ToArray();
                this.bufferSizes = new int[bufferPools.Length];
                for (int i = 0; i < bufferPools.Length; i++) {
                    this.bufferSizes[i] = bufferPools[i].BufferSize;
                }
            }

            public override void Clear() {
#if DEBUG
                this.buffersPooled.Clear();
#endif //DEBUG

                for (int i = 0; i < this.bufferPools.Length; i++) {
                    BufferPool bufferPool = this.bufferPools[i];
                    bufferPool.Clear();
                }
            }

            void ChangeQuota(ref BufferPool bufferPool, int delta) {
                //if (TraceCore.BufferPoolChangeQuotaIsEnabled(Fx.Trace))
                //{
                //    TraceCore.BufferPoolChangeQuota(Fx.Trace, bufferPool.BufferSize, delta);
                //}

                BufferPool oldBufferPool = bufferPool;
                int newLimit = oldBufferPool.Limit + delta;
                BufferPool newBufferPool = BufferPool.CreatePool(oldBufferPool.BufferSize, newLimit);
                for (int i = 0; i < newLimit; i++) {
                    byte[] buffer = oldBufferPool.Take();
                    if (buffer == null) {
                        break;
                    }

                    newBufferPool.Return(buffer);
                    newBufferPool.IncrementCount();
                }

                this.remainingMemory -= oldBufferPool.BufferSize * delta;
                bufferPool = newBufferPool;
            }

            void DecreaseQuota(ref BufferPool bufferPool) {
                ChangeQuota(ref bufferPool, -1);
            }

            int FindMostExcessivePool() {
                long maxBytesInExcess = 0;
                int index = -1;

                for (int i = 0; i < this.bufferPools.Length; i++) {
                    BufferPool bufferPool = this.bufferPools[i];

                    if (bufferPool.Peak < bufferPool.Limit) {
                        long bytesInExcess = (bufferPool.Limit - bufferPool.Peak) * (long) bufferPool.BufferSize;

                        if (bytesInExcess > maxBytesInExcess) {
                            index = i;
                            maxBytesInExcess = bytesInExcess;
                        }
                    }
                }

                return index;
            }

            int FindMostStarvedPool() {
                long maxBytesMissed = 0;
                int index = -1;

                for (int i = 0; i < this.bufferPools.Length; i++) {
                    BufferPool bufferPool = this.bufferPools[i];

                    if (bufferPool.Peak == bufferPool.Limit) {
                        long bytesMissed = bufferPool.Misses * (long) bufferPool.BufferSize;

                        if (bytesMissed > maxBytesMissed) {
                            index = i;
                            maxBytesMissed = bytesMissed;
                        }
                    }
                }

                return index;
            }

            [MethodImpl((MethodImplOptions) 768)]
            BufferPool FindPool(int desiredBufferSize) {
                for (int i = 0; i < this.bufferSizes.Length; i++) {
                    if (desiredBufferSize <= this.bufferSizes[i]) {
                        return this.bufferPools[i];
                    }
                }

                return null;
            }

            void IncreaseQuota(ref BufferPool bufferPool) {
                ChangeQuota(ref bufferPool, 1);
            }

            [MethodImpl((MethodImplOptions) 768)]
            public override void ReturnBuffer(byte[] buffer) {
                Fx.Assert(buffer != null, "caller must verify");

#if DEBUG
                int hash = buffer.GetHashCode();
                if (!this.buffersPooled.TryAdd(hash, CaptureStackTrace())) {
                    string originalStack;
                    if (!this.buffersPooled.TryGetValue(hash, out originalStack)) {
                        originalStack = "NULL";
                    }

                    Fx.Assert(
                        string.Format(
                            CultureInfo.InvariantCulture,
                            "Buffer '{0}' has already been returned to the bufferManager before. Previous CallStack: {1} Current CallStack: {2}",
                            hash,
                            originalStack,
                            CaptureStackTrace()));
                }
#endif //DEBUG

                BufferPool bufferPool = FindPool(buffer.Length);
                if (bufferPool != null) {
                    if (buffer.Length != bufferPool.BufferSize) {
                        throw new ArgumentException("BufferIsNotRightSizeForBufferManager", nameof(buffer));
                    }

                    if (bufferPool.Return(buffer)) {
                        bufferPool.IncrementCount();
                    }
                }
            }

            [MethodImpl((MethodImplOptions) 768)]
            public override byte[] TakeBuffer(int bufferSize) {
                Fx.Assert(bufferSize >= 0, "caller must ensure a non-negative argument");

                BufferPool bufferPool = FindPool(bufferSize);
                byte[] returnValue;
                if (bufferPool != null) {
                    byte[] buffer = bufferPool.Take();
                    if (buffer != null) {
                        bufferPool.DecrementCount();
                        returnValue = buffer;
                    } else {
                        if (bufferPool.Peak == bufferPool.Limit) {
                            bufferPool.Misses++;
                            if (++totalMisses >= maxMissesBeforeTuning) {
                                TuneQuotas();
                            }
                        }

                        //if (TraceCore.BufferPoolAllocationIsEnabled(Fx.Trace))
                        //{
                        //    TraceCore.BufferPoolAllocation(Fx.Trace, bufferPool.BufferSize);
                        //}

                        returnValue = Fx.AllocateByteArray(bufferPool.BufferSize);
                    }
                } else {
                    //if (TraceCore.BufferPoolAllocationIsEnabled(Fx.Trace))
                    //{
                    //    TraceCore.BufferPoolAllocation(Fx.Trace, bufferSize);
                    //}

                    returnValue = Fx.AllocateByteArray(bufferSize);
                }

#if DEBUG
                string dummy;
                this.buffersPooled.TryRemove(returnValue.GetHashCode(), out dummy);
#endif //DEBUG

                return returnValue;
            }

#if DEBUG
            [SecuritySafeCritical]
            private static string CaptureStackTrace() {
                return new StackTrace(true).ToString();
            }
#endif //DEBUG

            void TuneQuotas() {
                if (this.areQuotasBeingTuned) {
                    return;
                }

                bool lockHeld = false;
                try {
                    Monitor.TryEnter(this.tuningLock, ref lockHeld);

                    // Don't bother if another thread already has the lock
                    if (!lockHeld || this.areQuotasBeingTuned) {
                        return;
                    }

                    this.areQuotasBeingTuned = true;
                } finally {
                    if (lockHeld) {
                        Monitor.Exit(this.tuningLock);
                    }
                }

                // find the "poorest" pool
                int starvedIndex = FindMostStarvedPool();
                if (starvedIndex >= 0) {
                    BufferPool starvedBufferPool = this.bufferPools[starvedIndex];

                    if (this.remainingMemory < starvedBufferPool.BufferSize) {
                        // find the "richest" pool
                        int excessiveIndex = FindMostExcessivePool();
                        if (excessiveIndex >= 0) {
                            // steal from the richest
                            DecreaseQuota(ref this.bufferPools[excessiveIndex]);
                        }
                    }

                    if (this.remainingMemory >= starvedBufferPool.BufferSize) {
                        // give to the poorest
                        IncreaseQuota(ref this.bufferPools[starvedIndex]);
                    }
                }

                // reset statistics
                for (int i = 0; i < this.bufferPools.Length; i++) {
                    BufferPool bufferPool = this.bufferPools[i];
                    bufferPool.Misses = 0;
                }

                this.totalMisses = 0;
                this.areQuotasBeingTuned = false;
            }

            abstract class BufferPool {
                readonly int bufferSize;
                int count;
                readonly int limit;
                int misses;
                int peak;

                public BufferPool(int bufferSize, int limit) {
                    this.bufferSize = bufferSize;
                    this.limit = limit;
                }

                public int BufferSize {
                    [MethodImpl((MethodImplOptions) 768)] get { return this.bufferSize; }
                }

                public int Limit {
                    [MethodImpl((MethodImplOptions) 768)] get { return this.limit; }
                }

                public int Misses {
                    [MethodImpl((MethodImplOptions) 768)] get { return this.misses; }
                    [MethodImpl((MethodImplOptions) 768)] set { this.misses = value; }
                }

                public int Peak {
                    [MethodImpl((MethodImplOptions) 768)] get { return this.peak; }
                }

                [MethodImpl((MethodImplOptions) 768)]
                public void Clear() {
                    this.OnClear();
                    this.count = 0;
                }

                [MethodImpl((MethodImplOptions) 768)]
                public void DecrementCount() {
                    int newValue = this.count - 1;
                    if (newValue >= 0) {
                        this.count = newValue;
                    }
                }

                [MethodImpl((MethodImplOptions) 768)]
                public void IncrementCount() {
                    int newValue = this.count + 1;
                    if (newValue <= this.limit) {
                        this.count = newValue;
                        if (newValue > this.peak) {
                            this.peak = newValue;
                        }
                    }
                }

                internal abstract byte[] Take();
                internal abstract bool Return(byte[] buffer);
                internal abstract void OnClear();

                internal static BufferPool CreatePool(int bufferSize, int limit) {
                    // To avoid many buffer drops during training of large objects which
                    // get allocated on the LOH, we use the LargeBufferPool and for 
                    // bufferSize < 85000, the SynchronizedPool. However if bufferSize < 85000
                    // and (bufferSize + array-overhead) > 85000, this would still use 
                    // the SynchronizedPool even though it is allocated on the LOH.
                    if (bufferSize < 85000) {
                        return new SynchronizedBufferPool(bufferSize, limit);
                    } else {
                        return new LargeBufferPool(bufferSize, limit);
                    }
                }

                class SynchronizedBufferPool : BufferPool {
                    readonly SynchronizedPool<byte[]> innerPool;

                    internal SynchronizedBufferPool(int bufferSize, int limit)
                        : base(bufferSize, limit) {
                        this.innerPool = new SynchronizedPool<byte[]>(limit);
                    }

                    internal override void OnClear() {
                        this.innerPool.Clear();
                    }

                    internal override byte[] Take() {
                        return this.innerPool.Take();
                    }

                    internal override bool Return(byte[] buffer) {
                        return this.innerPool.Return(buffer);
                    }
                }

                class LargeBufferPool : BufferPool {
                    readonly Stack<byte[]> items;

                    internal LargeBufferPool(int bufferSize, int limit)
                        : base(bufferSize, limit) {
                        this.items = new Stack<byte[]>(limit);
                    }

                    object ThisLock {
                        [MethodImpl((MethodImplOptions) 768)] get => this.items;
                    }

                    internal override void OnClear() {
                        lock (ThisLock) {
                            this.items.Clear();
                        }
                    }

                    internal override byte[] Take() {
                        lock (ThisLock) {
                            if (this.items.Count > 0) {
                                return this.items.Pop();
                            }
                        }

                        return null;
                    }

                    internal override bool Return(byte[] buffer) {
                        lock (ThisLock) {
                            if (this.items.Count < this.Limit) {
                                this.items.Push(buffer);
                                return true;
                            }
                        }

                        return false;
                    }
                }
            }
        }

        class GCBufferManager : InternalBufferManager {
            static readonly GCBufferManager value = new GCBufferManager();

            GCBufferManager() { }

            public static GCBufferManager Value {
                [MethodImpl((MethodImplOptions) 768)] get => value;
            }

            [MethodImpl((MethodImplOptions) 768)]
            public override void Clear() { }

            [MethodImpl((MethodImplOptions) 768)]
            public override byte[] TakeBuffer(int bufferSize) {
                return Fx.AllocateByteArray(bufferSize);
            }

            [MethodImpl((MethodImplOptions) 768)]
            public override void ReturnBuffer(byte[] buffer) {
                // do nothing, GC will reclaim this buffer
            }
        }
    }


    namespace System.Runtime {
        // A simple synchronized pool would simply lock a stack and push/pop on return/take.
        //
        // This implementation tries to reduce locking by exploiting the case where an item
        // is taken and returned by the same thread, which turns out to be common in our 
        // scenarios.  
        //
        // Initially, all the quota is allocated to a global (non-thread-specific) pool, 
        // which takes locks.  As different threads take and return values, we record their IDs, 
        // and if we detect that a thread is taking and returning "enough" on the same thread, 
        // then we decide to "promote" the thread.  When a thread is promoted, we decrease the 
        // quota of the global pool by one, and allocate a thread-specific entry for the thread 
        // to store it's value.  Once this entry is allocated, the thread can take and return 
        // it's value from that entry without taking any locks.  Not only does this avoid 
        // locks, but it affinitizes pooled items to a particular thread.
        //
        // There are a couple of additional things worth noting:
        // 
        // It is possible for a thread that we have reserved an entry for to exit.  This means
        // we will still have a entry allocated for it, but the pooled item stored there 
        // will never be used.  After a while, we could end up with a number of these, and 
        // as a result we would begin to exhaust the quota of the overall pool.  To mitigate this
        // case, we throw away the entire per-thread pool, and return all the quota back to 
        // the global pool if we are unable to promote a thread (due to lack of space).  Then 
        // the set of active threads will be re-promoted as they take and return items.
        // 
        // You may notice that the code does not immediately promote a thread, and does not
        // immediately throw away the entire per-thread pool when it is unable to promote a 
        // thread.  Instead, it uses counters (based on the number of calls to the pool) 
        // and a threshold to figure out when to do these operations.  In the case where the
        // pool to misconfigured to have too few items for the workload, this avoids constant 
        // promoting and rebuilding of the per thread entries.
        //
        // You may also notice that we do not use interlocked methods when adjusting statistics.
        // Since the statistics are a heuristic as to how often something is happening, they 
        // do not need to be perfect.
        // 
        [Fx.Tag.SynchronizationObject(Blocking = false)]
        class SynchronizedPool<T> where T : class {
            const int maxPendingEntries = 128;
            const int maxPromotionFailures = 64;
            const int maxReturnsBeforePromotion = 64;
            const int maxThreadItemsPerProcessor = 16;
            Entry[] entries;
            readonly GlobalPool globalPool;
            readonly int maxCount;
            PendingEntry[] pending;
            int promotionFailures;

            public SynchronizedPool(int maxCount) {
                int threadCount = maxCount;
                int maxThreadCount = maxThreadItemsPerProcessor + SynchronizedPoolHelper.ProcessorCount;
                if (threadCount > maxThreadCount) {
                    threadCount = maxThreadCount;
                }

                this.maxCount = maxCount;
                this.entries = new Entry[threadCount];
                this.pending = new PendingEntry[4];
                this.globalPool = new GlobalPool(maxCount);
            }

            object ThisLock {
                get { return this; }
            }

            [MethodImpl((MethodImplOptions) 768)]
            public void Clear() {
                Entry[] entries = this.entries;

                for (int i = 0; i < entries.Length; i++) {
                    entries[i].value = null;
                }

                globalPool.Clear();
            }

            [MethodImpl((MethodImplOptions) 768)]
            void HandlePromotionFailure(int thisThreadID) {
                int newPromotionFailures = this.promotionFailures + 1;

                if (newPromotionFailures >= maxPromotionFailures) {
                    lock (ThisLock) {
                        this.entries = new Entry[this.entries.Length];

                        globalPool.MaxCount = maxCount;
                    }

                    PromoteThread(thisThreadID);
                } else {
                    this.promotionFailures = newPromotionFailures;
                }
            }

            [MethodImpl((MethodImplOptions) 768)]
            bool PromoteThread(int thisThreadID) {
                lock (ThisLock) {
                    for (int i = 0; i < this.entries.Length; i++) {
                        int threadID = this.entries[i].threadID;

                        if (threadID == thisThreadID) {
                            return true;
                        } else if (threadID == 0) {
                            globalPool.DecrementMaxCount();
                            this.entries[i].threadID = thisThreadID;
                            return true;
                        }
                    }
                }

                return false;
            }

            [MethodImpl((MethodImplOptions) 768)]
            void RecordReturnToGlobalPool(int thisThreadID) {
                PendingEntry[] localPending = this.pending;

                for (int i = 0; i < localPending.Length; i++) {
                    int threadID = localPending[i].threadID;

                    if (threadID == thisThreadID) {
                        int newReturnCount = localPending[i].returnCount + 1;

                        if (newReturnCount >= maxReturnsBeforePromotion) {
                            localPending[i].returnCount = 0;

                            if (!PromoteThread(thisThreadID)) {
                                HandlePromotionFailure(thisThreadID);
                            }
                        } else {
                            localPending[i].returnCount = newReturnCount;
                        }

                        break;
                    } else if (threadID == 0) {
                        break;
                    }
                }
            }

            [MethodImpl((MethodImplOptions) 768)]
            void RecordTakeFromGlobalPool(int thisThreadID) {
                PendingEntry[] localPending = this.pending;

                for (int i = 0; i < localPending.Length; i++) {
                    int threadID = localPending[i].threadID;

                    if (threadID == thisThreadID) {
                        return;
                    } else if (threadID == 0) {
                        lock (localPending) {
                            if (localPending[i].threadID == 0) {
                                localPending[i].threadID = thisThreadID;
                                return;
                            }
                        }
                    }
                }

                if (localPending.Length >= maxPendingEntries) {
                    this.pending = new PendingEntry[localPending.Length];
                } else {
                    PendingEntry[] newPending = new PendingEntry[localPending.Length * 2];
                    Array.Copy(localPending, newPending, localPending.Length);
                    this.pending = newPending;
                }
            }

            [MethodImpl((MethodImplOptions) 768)]
            public bool Return(T value) {
                int thisThreadID = Thread.CurrentThread.ManagedThreadId;

                if (thisThreadID == 0) {
                    return false;
                }

                if (ReturnToPerThreadPool(thisThreadID, value)) {
                    return true;
                }

                return ReturnToGlobalPool(thisThreadID, value);
            }

            [MethodImpl((MethodImplOptions) 768)]
            bool ReturnToPerThreadPool(int thisThreadID, T value) {
                Entry[] entries = this.entries;

                for (int i = 0; i < entries.Length; i++) {
                    int threadID = entries[i].threadID;

                    if (threadID == thisThreadID) {
                        if (entries[i].value == null) {
                            entries[i].value = value;
                            return true;
                        } else {
                            return false;
                        }
                    } else if (threadID == 0) {
                        break;
                    }
                }

                return false;
            }

            [MethodImpl((MethodImplOptions) 768)]
            bool ReturnToGlobalPool(int thisThreadID, T value) {
                RecordReturnToGlobalPool(thisThreadID);

                return globalPool.Return(value);
            }

            [MethodImpl((MethodImplOptions) 768)]
            public T Take() {
                int thisThreadID = Thread.CurrentThread.ManagedThreadId;

                if (thisThreadID == 0) {
                    return null;
                }

                T value = TakeFromPerThreadPool(thisThreadID);

                if (value != null) {
                    return value;
                }

                return TakeFromGlobalPool(thisThreadID);
            }

            [MethodImpl((MethodImplOptions) 768)]
            T TakeFromPerThreadPool(int thisThreadID) {
                Entry[] entries = this.entries;

                for (int i = 0; i < entries.Length; i++) {
                    int threadID = entries[i].threadID;

                    if (threadID == thisThreadID) {
                        T value = entries[i].value;

                        if (value != null) {
                            entries[i].value = null;
                            return value;
                        } else {
                            return null;
                        }
                    } else if (threadID == 0) {
                        break;
                    }
                }

                return null;
            }

            [MethodImpl((MethodImplOptions) 768)]
            T TakeFromGlobalPool(int thisThreadID) {
                RecordTakeFromGlobalPool(thisThreadID);

                return globalPool.Take();
            }

            struct Entry {
                public int threadID;
                public T value;
            }

            struct PendingEntry {
                public int returnCount;
                public int threadID;
            }

            static class SynchronizedPoolHelper {
                public static readonly int ProcessorCount = GetProcessorCount();

                [Fx.Tag.SecurityNote(Critical = "Asserts in order to get the processor count from the environment", Safe = "This data isn't actually protected so it's ok to leak")]
                [SecuritySafeCritical]
                static int GetProcessorCount() {
                    return Environment.ProcessorCount;
                }
            }

            [Fx.Tag.SynchronizationObject(Blocking = false)]
            class GlobalPool {
                readonly Stack<T> items;

                int maxCount;

                public GlobalPool(int maxCount) {
                    this.items = new Stack<T>();
                    this.maxCount = maxCount;
                }

                public int MaxCount {
                    [MethodImpl((MethodImplOptions) 768)] get { return maxCount; }
                    [MethodImpl((MethodImplOptions) 768)]
                    set {
                        lock (ThisLock) {
                            while (items.Count > value) {
                                items.Pop();
                            }

                            maxCount = value;
                        }
                    }
                }

                object ThisLock {
                    [MethodImpl((MethodImplOptions) 768)] get { return this; }
                }

                [MethodImpl((MethodImplOptions) 768)]
                public void DecrementMaxCount() {
                    lock (ThisLock) {
                        if (items.Count == maxCount) {
                            items.Pop();
                        }

                        maxCount--;
                    }
                }

                [MethodImpl((MethodImplOptions) 768)]
                public T Take() {
                    if (this.items.Count > 0) {
                        lock (ThisLock) {
                            if (this.items.Count > 0) {
                                return this.items.Pop();
                            }
                        }
                    }

                    return null;
                }

                [MethodImpl((MethodImplOptions) 768)]
                public bool Return(T value) {
                    if (this.items.Count < this.MaxCount) {
                        lock (ThisLock) {
                            if (this.items.Count < this.MaxCount) {
                                this.items.Push(value);
                                return true;
                            }
                        }
                    }

                    return false;
                }

                [MethodImpl((MethodImplOptions) 768)]
                public void Clear() {
                    lock (ThisLock) {
                        this.items.Clear();
                    }
                }
            }
        }
    }
}
