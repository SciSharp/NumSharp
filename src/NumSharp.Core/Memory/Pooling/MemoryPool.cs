using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace OOMath.MemoryPooling {
    /// <summary>
    /// Types of exceptions that can happen while dealing with unmanaged memory
    /// </summary>
    /// <author>Jackson Dunstan, http://JacksonDunstan.com</author>
    /// <license>MIT</license>
    public enum UnmanagedMemoryExceptionType {
        PointerCanNotBeNull,
        CountMustBeNonNegative,
        AllocationSizeMustBePositive,
        AllocationTableIsFull,
        NullPool,
        UninitializedOrDestroyedPool,
        PoolIsDry,
        SentinelOverwritten,
        BlockSizeBelowMinimum,
        NumberOfBlocksMustBePositive,
        PointerDoesNotPointToBlockInPool
    }

    /// <summary>
    /// An exception indicating an error related to unmanaged memory
    /// </summary>
    /// <author>Jackson Dunstan, http://JacksonDunstan.com</author>
    /// <license>MIT</license>
    public unsafe class UnmanagedMemoryException : Exception {
        /// <summary>
        /// Create the exception
        /// </summary>
        /// <param name="type">Type of the exception</param>
        /// <param name="pointer">Pointer related to the exception</param>
        public UnmanagedMemoryException(UnmanagedMemoryExceptionType type, void* pointer = null) {
            Type = type;
            Pointer = pointer;
        }

        /// <summary>
        /// Get the type of the exception
        /// </summary>
        /// <value>The type of the exception</value>
        public UnmanagedMemoryExceptionType Type { get; private set; }

        /// <summary>
        /// Pointer related to the exception
        /// </summary>
        /// <value>The pointer related to the exception</value>
        public void* Pointer { get; private set; }

        /// <summary>
        /// Get a string version of this exception
        /// </summary>
        /// <returns>A string version of this exception</returns>
        public override string ToString() {
            return string.Format(
                "[UnmanagedMemoryException: Type={0}, Pointer={1}]",
                Type,
                (IntPtr) Pointer
            );
        }
    }

    /// <summary>
    /// A pool of unmanaged memory. Consists of a fixed number of equal-sized blocks that can be
    /// allocated and freed.
    /// </summary>
    /// <author>Jackson Dunstan, http://JacksonDunstan.com</author>
    /// <license>MIT</license>
    public unsafe struct UnmanagedMemoryPool {
        /// <summary>
        /// Unmanaged memory containing all the blocks
        /// </summary>
        public byte* Alloc;

        /// <summary>
        /// Pointer to the next free block
        /// </summary>
        public void* Free;

        /// <summary>
        /// Size of a single block. May include extra bytes for internal usage, such as a sentinel.
        /// </summary>
        public int BlockSize;

        /// <summary>
        /// Number of blocks
        /// </summary>
        public int NumBlocks;
    }

    /// <summary>
    /// Tools for dealing with unmanaged memory
    /// </summary>
    /// <author>Jackson Dunstan, http://JacksonDunstan.com</author>
    /// <license>MIT</license>
    public static unsafe class UnmanagedMemory {
#if UNITY_EDITOR
		/// <summary>
		/// Hash table that keeps track of all the allocations that haven't been freed
		/// </summary>
		private static void** allocations;
 
		/// <summary>
		/// Size/length of the <see cref="allocations"/> hash table
		/// </summary>
		private static int maxAllocations;
#endif

        /// <summary>
        /// The size of a pointer, in bytes
        /// </summary>
        public static readonly int SizeOfPointer = sizeof(void*);

        /// <summary>
        /// The minimum size of an <see cref="UnmanagedMemoryPool"/> block, in bytes
        /// </summary>
        public static readonly int MinimumPoolBlockSize = SizeOfPointer;

#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
		/// <summary>
		/// Value added to the end of an <see cref="UnmanagedMemoryPool"/> block. Used to detect
		/// out-of-bound memory writes.
		/// </summary>
		private const ulong SentinelValue = 0x8899AABBCCDDEEFF;
#endif

        /// <summary>
        /// Prepare this class for use
        /// </summary>
        /// <param name="maxAllocations">Maximum number of allocations expected</param>
        [Conditional("UNITY_EDITOR")]
        public static void SetUp(int maxAllocations) {
#if UNITY_EDITOR
			// Create the allocations hash table
			if (allocations == null)
			{
				int size = maxAllocations * SizeOfPointer;
				allocations = (void**)Marshal.AllocHGlobal(size);
				Memset(allocations, 0, size);
				UnmanagedMemory.maxAllocations = maxAllocations;
			}
#endif
        }

        /// <summary>
        /// Stop using this class. This frees all unmanaged memory that was allocated since
        /// <see cref="SetUp"/> was called.
        /// </summary>
        [Conditional("UNITY_EDITOR")]
        public static void TearDown() {
#if UNITY_EDITOR
			if (allocations != null)
			{
				// Free all the allocations
				for (int i = 0; i < maxAllocations; ++i)
				{
					void* ptr = allocations[i];
					if (ptr != null)
					{
						Marshal.FreeHGlobal((IntPtr)ptr);
					}
				}
 
				// Free the allocations table itself
				Marshal.FreeHGlobal((IntPtr)allocations);
				allocations = null;
				maxAllocations = 0;
			}
#endif
        }

        [Conditional("UNITY_ASSERTIONS"), Conditional("UNMANAGED_MEMORY_DEBUG")]
        public static void Assert(bool condition, UnmanagedMemoryExceptionType type, void* data = null) {
#if UNITY_ASSERTIONS
			if (!condition)
			{
				throw new UnmanagedMemoryException(type, data);
			}
#endif
        }

        /// <summary>
        /// Set a series of bytes to the same value
        /// </summary>
        /// <param name="ptr">Pointer to the first byte to set</param>
        /// <param name="value">Value to set to all the bytes</param>
        /// <param name="count">Number of bytes to set</param>
        public static void Memset(void* ptr, byte value, int count) {
            Assert(ptr != null, UnmanagedMemoryExceptionType.PointerCanNotBeNull);
            Assert(count >= 0, UnmanagedMemoryExceptionType.CountMustBeNonNegative);

            new Span<byte>(ptr, count).Fill(value);

            //byte* pCur = (byte*) ptr;
            //for (int i = 0; i < count; ++i) {
            //    *pCur++ = value;
            //}
        }

        /// <summary>
        /// Allocate unmanaged heap memory and track it
        /// </summary>
        /// <param name="size">Number of bytes of unmanaged heap memory to allocate</param>
        public static IntPtr Alloc(int size) {
            Assert(size > 0, UnmanagedMemoryExceptionType.AllocationSizeMustBePositive);

            IntPtr intPtr = Marshal.AllocHGlobal(size);
#if UNITY_EDITOR
			void* ptr = (void*)intPtr;
			int index = (int)(((long)ptr) % maxAllocations);
			for (int i = index; i < maxAllocations; ++i)
			{
				if (allocations[i] == null)
				{
					allocations[i] = ptr;
					return intPtr;
				}
			}
			for (int i = 0; i < index; ++i)
			{
				if (allocations[i] == null)
				{
					allocations[i] = ptr;
					return intPtr;
				}
			}
			Assert(false, UnmanagedMemoryExceptionType.AllocationTableIsFull);
#endif
            return intPtr;
        }

        /// <summary>
        /// Allocate unmanaged heap memory filled with zeroes and track it
        /// </summary>
        /// <param name="size">Number of bytes of unmanaged heap memory to allocate</param>
        public static IntPtr Calloc(int size) {
            IntPtr intPtr = Alloc(size);
            Memset((void*) intPtr, 0, size);
            return intPtr;
        }

        /// <summary>
        /// Allocate a block of memory from a pool
        /// </summary>
        /// <param name="pool">Pool to allocate from</param>
        public static void* Alloc(UnmanagedMemoryPool* pool) {
            Assert(pool != null, UnmanagedMemoryExceptionType.NullPool);
            Assert(pool->Alloc != null, UnmanagedMemoryExceptionType.UninitializedOrDestroyedPool);
            Assert(pool->Free != null, UnmanagedMemoryExceptionType.PoolIsDry);

            void* pRet = pool->Free;

            // Make sure the sentinel is still intact
#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
			if (*((ulong*)(((byte*)pRet)+pool->BlockSize-sizeof(ulong))) != SentinelValue)
			{
				Assert(false, UnmanagedMemoryExceptionType.SentinelOverwritten, pRet);
			}
#endif

            // Return the head of the free list and advance the free list pointer
            pool->Free = *((byte**) pool->Free);
#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
			*((ulong*)(((byte*)pRet)+pool->BlockSize-sizeof(ulong))) = SentinelValue;
#endif
            return pRet;
        }

        /// <summary>
        /// Allocate a zero-filled block of memory from a pool
        /// </summary>
        /// <param name="pool">Pool to allocate from</param>
        public static void* Calloc(UnmanagedMemoryPool* pool) {
            void* ptr = Alloc(pool);
            Memset(ptr, 0, pool->BlockSize);
            return ptr;
        }

        /// <summary>
        /// Allocate a pool of memory. The pool is made up of a fixed number of equal-sized blocks.
        /// Allocations from the pool return one of these blocks.
        /// </summary>
        /// <returns>The allocated pool</returns>
        /// <param name="blockSize">Size of each block, in bytes</param>
        /// <param name="numBlocks">The number of blocks in the pool</param>
        public static UnmanagedMemoryPool AllocPool(int blockSize, int numBlocks) {
            Assert(
                blockSize >= MinimumPoolBlockSize,
                UnmanagedMemoryExceptionType.BlockSizeBelowMinimum
            );
            Assert(numBlocks > 0, UnmanagedMemoryExceptionType.NumberOfBlocksMustBePositive);

#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
			// Add room for the sentinel
			blockSize += sizeof(ulong);
#endif

            UnmanagedMemoryPool pool = new UnmanagedMemoryPool();

            pool.Free = null;
            pool.NumBlocks = numBlocks;
            pool.BlockSize = blockSize;

            // Allocate unmanaged memory large enough to fit all the blocks
            pool.Alloc = (byte*) Alloc(blockSize * numBlocks);

#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
		{
			// Set the sentinel value at the end of each block
			byte* pCur = pool.Alloc + blockSize - sizeof(ulong);
			for (int i = 0; i < numBlocks; ++i)
			{
				*((ulong*)pCur) = SentinelValue;
				pCur += blockSize;
			}
		}
#endif

            // Reset the free list
            FreeAll(&pool);

            return pool;
        }

        /// <summary>
        /// Free unmanaged heap memory and stop tracking it
        /// </summary>
        /// <param name="ptr">Pointer to the unmanaged heap memory to free. If null, this is a no-op.
        /// </param>
        public static void Free(IntPtr ptr) {
            if (ptr != IntPtr.Zero) {
                Marshal.FreeHGlobal(ptr);
#if UNITY_EDITOR
				void* voidPtr = (void*)ptr;
				int index = (int)(((long)voidPtr) % maxAllocations);
				for (int i = index; i < maxAllocations; ++i)
				{
					if (allocations[i] == voidPtr)
					{
						allocations[i] = null;
						return;
					}
				}
				for (int i = 0; i < index; ++i)
				{
					if (allocations[i] == voidPtr)
					{
						allocations[i] = null;
						return;
					}
				}
#endif
            }
        }

        /// <summary>
        /// Free a block from a pool
        /// </summary>
        /// <param name="pool">Pool the block is from</param>
        /// <param name="ptr">Pointer to the block to free. If null, this is a no-op.</param>
        public static void Free(UnmanagedMemoryPool* pool, void* ptr) {
            Assert(pool != null, UnmanagedMemoryExceptionType.NullPool);
            Assert(pool->Alloc != null, UnmanagedMemoryExceptionType.UninitializedOrDestroyedPool);

            // Freeing a null pointer is a no-op, not an error
            if (ptr != null) {
                // Pointer must be in the pool and on a block boundary
                Assert(
                    ptr >= pool->Alloc
                    && ptr < pool->Alloc + pool->BlockSize * pool->NumBlocks
                    && (((uint) ((byte*) ptr - pool->Alloc)) % pool->BlockSize) == 0,
                    UnmanagedMemoryExceptionType.PointerDoesNotPointToBlockInPool
                );

                // Make sure the sentinel is still intact for this block and the one before it
#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
				if (*((ulong*)(((byte*)ptr)+pool->BlockSize-sizeof(ulong))) != SentinelValue)
				{
					Assert(
						false,
						UnmanagedMemoryExceptionType.SentinelOverwritten,
						ptr
					);
				}
				if (ptr != pool->Alloc && *((ulong*)(((byte*)ptr)-sizeof(ulong))) != SentinelValue)
				{
					Assert(
						false,
						UnmanagedMemoryExceptionType.SentinelOverwritten,
						(((byte*)ptr)-sizeof(ulong))
					);
				}
#endif

                // Insert the block to free at the start of the free list
                void** pHead = (void**) ptr;
                *pHead = pool->Free;
                pool->Free = pHead;
            }
        }

        /// <summary>
        /// Free all the blocks of a pool. This does not free the pool itself, but rather makes all of
        /// its blocks available for allocation again.
        /// </summary>
        /// <param name="pool">Pool whose blocks should be freed</param>
        public static void FreeAll(UnmanagedMemoryPool* pool) {
            Assert(pool != null, UnmanagedMemoryExceptionType.NullPool);
            Assert(pool->Alloc != null, UnmanagedMemoryExceptionType.UninitializedOrDestroyedPool);

            // Point each block except the last one to the next block. Check their sentinels while we're
            // at it.
            void** pCur = (void**) pool->Alloc;
            byte* pNext = pool->Alloc + pool->BlockSize;
#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
			byte* pSentinel = pool->Alloc + pool->BlockSize - sizeof(ulong);
#endif
            for (int i = 0, count = pool->NumBlocks - 1; i < count; ++i) {
#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
				if (*((ulong*)pSentinel) != SentinelValue)
				{
					Assert(false, UnmanagedMemoryExceptionType.SentinelOverwritten, pCur);
				}
				pSentinel += pool->BlockSize;
#endif
                *pCur = pNext;
                pCur = (void**) pNext;
                pNext += pool->BlockSize;
            }

            // Check the last block's sentinel.
#if UNITY_ASSERTIONS || UNMANAGED_MEMORY_DEBUG
			if (*((ulong*)pSentinel) != SentinelValue)
			{
				Assert(false, UnmanagedMemoryExceptionType.SentinelOverwritten, pCur);
			}
#endif

            // Point the last block to null
            *pCur = default(void*);

            // The first block is now the head of the free list
            pool->Free = pool->Alloc;
        }

        /// <summary>
        /// Free a pool and all of its blocks. Double-freeing a pool is a no-op.
        /// </summary>
        /// <param name="pool">Pool to free</param>
        public static void FreePool(UnmanagedMemoryPool* pool) {
            Assert(pool != null, UnmanagedMemoryExceptionType.NullPool);

            // Free the unmanaged memory for all the blocks and set to null to allow double-Destroy()
            Free((IntPtr) pool->Alloc);
            pool->Alloc = null;
            pool->Free = null;
        }
    }
}