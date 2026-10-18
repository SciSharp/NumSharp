using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The string allocator of ONE NEP 55 <c>StringDType</c> buffer — a statement-for-statement port of NumPy 2.4.2's
    ///     <c>numpy/_core/src/multiarray/stringdtype/static_string.c</c>: the 16-byte packed cell of every element, the
    ///     per-buffer ARENA that holds the text of elements longer than 15 bytes, the HEAP strings that a later, longer
    ///     write moves an element to, and the mutex that guards them.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The cell</b> (<c>npy_packed_static_string</c>, little-endian 64-bit layout). Bytes 0–7 hold an arena OFFSET or
    ///     a heap POINTER; bytes 8–15 hold the size, whose highest byte (byte 15) is the FLAGS byte:
    ///     <see cref="NPY_STRING_MISSING"/> (an NA value), <see cref="NPY_STRING_INITIALIZED"/> (written at least once),
    ///     <see cref="NPY_STRING_OUTSIDE_ARENA"/> (short or heap), <see cref="NPY_STRING_LONG"/> (an arena string of more
    ///     than 255 bytes, or a heap string). A string of up to <see cref="NPY_SHORT_STRING_MAX_SIZE"/> bytes lives IN the
    ///     cell: bytes 0–14 are the text and the low nibble of byte 15 is its length, with the flags
    ///     <c>NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA</c>. An all-zero cell is an uninitialized empty string, so a
    ///     zero-filled buffer is a buffer of <c>''</c> for free (NumPy's <c>NPY_NEEDS_INIT</c>).
    ///     </para>
    ///     <para>
    ///     <b>The arena.</b> A medium string (16–255 bytes) is stored after a one-byte size prefix, a long one after an
    ///     eight-byte prefix; the cell holds the offset of the TEXT (after the prefix). The arena grows by reallocation to
    ///     125 % of what it needs, and its new tail is zeroed. A write that fits the old allocation reuses it (the prefix
    ///     records the allocation's capacity); a longer one goes to the heap. A string set to NA keeps its allocation's
    ///     metadata (only the <see cref="NPY_STRING_MISSING"/> bit is added), so a later write can reuse its arena space.
    ///     After a short write the arena information is gone, and a later long write always goes to the heap.
    ///     </para>
    ///     <para>
    ///     <b>NumPy's names.</b> The constants carry the names of <c>static_string.c</c>'s macros (<c>NPY_STRING_*</c>,
    ///     <c>NPY_SHORT_STRING_*</c>, <c>NPY_MEDIUM_STRING_MAX_SIZE</c>, <c>NPY_MAX_STRING_SIZE</c>; the cell size is
    ///     <c>stringdtype/dtype.h</c>'s <c>SIZEOF_NPY_PACKED_STATIC_STRING</c>), and the members that port the
    ///     <c>NpyString_*</c> functions carry the function's name after the prefix (<see cref="load"/>, <see cref="pack"/>,
    ///     <see cref="pack_null"/>, <see cref="free"/>, <see cref="dup"/>, …). The two NumPy functions that take a list of
    ///     allocators, <c>NpyString_acquire_allocators</c> and <c>NpyString_release_allocators</c>, have a two-allocator
    ///     overload each. The ports of <c>static_string.c</c>'s unprefixed helpers (<c>is_short_string</c>,
    ///     <c>arena_malloc</c>, <c>heap_or_arena_allocate</c>, <c>vstring_buffer</c>, …) keep NumSharp names, as do the
    ///     members NumPy has no function for. The test <c>DTypeOracleTests.Members_ExistUnderNumPysNames</c> holds the class
    ///     to the names and values it reads from NumPy's source.
    ///     </para>
    ///     <para>
    ///     <b>Ownership.</b> One allocator per BUFFER (NumPy gives every array that owns data its own descriptor instance
    ///     and allocator, <c>stringdtype_finalize_descr</c>): views and slices of the buffer share it, a copy gets a new one.
    ///     The cells' memory block owns the allocator — its release runs <see cref="ClearAndFree"/> (NumPy's clear loop,
    ///     which frees the heap strings, then the arena), so the allocator lives exactly as long as the last view of its
    ///     cells. Cells are never valid with another buffer's allocator: an offset reads someone else's arena, and a heap
    ///     pointer copied into another buffer would be freed twice. Every movement of <c>StringDType</c> elements therefore
    ///     goes through <see cref="dup"/>, never through a raw copy of the cells.
    ///     </para>
    ///     <para>
    ///     <b>Locking.</b> Like NumPy's allocator mutex: a caller takes the lock ONCE per operation
    ///     (<see cref="acquire_allocator"/> / <see cref="release_allocator"/>, or
    ///     <see cref="acquire_allocators(VStringAllocator, VStringAllocator)"/> for two allocators), then calls the per-cell
    ///     members, which assume it is held. The allocator does not lock per element.
    ///     </para>
    ///     <para>
    ///     <b>Errors.</b> NumPy's members return -1 and the caller raises <c>MemoryError</c> with its own text; here the
    ///     members return the same codes (0 success, -1 failure, 1 = "is null" for <see cref="load"/>) and the callers
    ///     throw <see cref="OutOfMemoryException"/> with NumPy's text. The arena is <see cref="NativeMemory"/> memory
    ///     (NumPy's <c>PyMem_RawMalloc</c> / <c>PyMem_RawRealloc</c>).
    ///     </para>
    ///     <para>
    ///     <b>Heap strings come from pages, not from one <c>malloc</c> each</b> (the one place this port leaves NumPy's
    ///     memory calls; the cells are NumPy's byte for byte, but the heap pointer they hold is an address either way). NumPy
    ///     <c>PyMem_RawMalloc</c>s every text that outgrows its arena slot, and that call was half of an assignment of
    ///     longer texts over a held array: 44 ns a text of 20–60 bytes, 149 ns of 300–500 (measured, pinned). A text of at
    ///     most <see cref="HeapPoolMaxBytes"/> bytes is carved instead from a page of its SIZE CLASS (block strides of 32
    ///     to 1,040 bytes in steps of 16): pages of 1 to 64 KiB taken from <see cref="Unmanaged.Pooling.SizeBucketedBufferPool"/>
    ///     (warm memory, recycled across buffers, accounted there), a free list and a bump pointer per page, an 8-byte header
    ///     in front of every text that names its page. Freeing a text pushes it on its page's free list; a page whose last
    ///     text is freed goes back to the buffer pool (the allocator keeps ONE emptied page as a spare for its next page), so
    ///     the memory a buffer holds follows its live texts (as pymalloc's pools do), not its peak. A longer text is a block
    ///     of its own from the C allocator (a null page in its header). The clear loop returns the pages whole and visits the
    ///     cells only while such a block exists. Measured (pinned) on <c>np.copyto(dst, b)</c> of 100,000 texts over a held
    ///     array: 62.2 → 39.6 ns a text at 20–60 bytes and 189 → 89 at 300–500 (NumPy 64.1 and 154).
    ///     </para>
    /// </remarks>
    internal sealed unsafe class VStringAllocator
    {
        // ------------------------------------------------------------------------------------------------------------
        // the packed cell (static_string.c, NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN, sizeof(size_t) == 8)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>SIZEOF_NPY_PACKED_STATIC_STRING</c> (<c>stringdtype/dtype.h</c>): the size of one packed cell
        ///     (<c>sizeof(npy_packed_static_string)</c>), so the stride between cells and the descriptor's itemsize. It
        ///     is the same constant as <see cref="VStringDTypeMeta.SIZEOF_NPY_PACKED_STATIC_STRING"/>; the dtype oracle
        ///     test holds both to NumPy's value, so the two cannot drift apart.
        /// </summary>
        internal const int SIZEOF_NPY_PACKED_STATIC_STRING = 16;

        /// <summary><c>NPY_STRING_MISSING</c>: the cell is an NA value (its other flags keep the old allocation's metadata).</summary>
        internal const byte NPY_STRING_MISSING = 0x80;

        /// <summary><c>NPY_STRING_INITIALIZED</c>: the element has been written at least once.</summary>
        internal const byte NPY_STRING_INITIALIZED = 0x40;

        /// <summary><c>NPY_STRING_OUTSIDE_ARENA</c>: the text is in the cell (short) or on the heap (with <see cref="NPY_STRING_LONG"/>).</summary>
        internal const byte NPY_STRING_OUTSIDE_ARENA = 0x20;

        /// <summary><c>NPY_STRING_LONG</c>: an arena string longer than 255 bytes (eight-byte prefix), or a heap string.</summary>
        internal const byte NPY_STRING_LONG = 0x10;

        /// <summary><c>NPY_STRING_FLAG_MASK</c>: the high nibble of byte 15.</summary>
        internal const byte NPY_STRING_FLAG_MASK = 0xF0;

        /// <summary><c>NPY_SHORT_STRING_SIZE_MASK</c>: the low nibble of byte 15, a short string's length.</summary>
        internal const byte NPY_SHORT_STRING_SIZE_MASK = 0x0F;

        /// <summary><c>NPY_SHORT_STRING_MAX_SIZE</c>: the longest text a cell holds itself (15 bytes).</summary>
        internal const int NPY_SHORT_STRING_MAX_SIZE = SIZEOF_NPY_PACKED_STATIC_STRING - 1;

        /// <summary><c>NPY_MEDIUM_STRING_MAX_SIZE</c>: the longest arena string with a one-byte size prefix.</summary>
        internal const int NPY_MEDIUM_STRING_MAX_SIZE = 0xFF;

        /// <summary><c>NPY_MAX_STRING_SIZE</c>: one byte of the size field is the flags, so a string has at most 2^56 − 1 bytes.</summary>
        internal const long NPY_MAX_STRING_SIZE = (1L << 56) - 1;

        /// <summary>The mask that keeps the 56 size bits of the size field (<c>~HIGH_BYTE_MASK</c>).</summary>
        private const ulong SizeMask = 0x00FF_FFFF_FFFF_FFFFUL;

        /// <summary>Heap bytes that are reported to the GC in batches of at least this many (see <see cref="AccountHeap"/>).</summary>
        private const long PressureBatch = 64 * 1024;

        // ------------------------------------------------------------------------------------------------------------
        // the heap pages (NumSharp's: NumPy mallocs each heap string — see the class remarks)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The longest heap text a page block holds (the payload of the largest size class, a block stride of 1,040 bytes):
        ///     a longer text is a block of its own from the C allocator, whose cost is small next to copying the text.
        /// </summary>
        internal const int HeapPoolMaxBytes = 1032;

        /// <summary>The number of heap size classes: block strides of 32, 48, …, 1,040 bytes (class k strides 16·(k + 2)).</summary>
        private const int HeapClassCount = 64;

        /// <summary>The bytes in front of every heap text: the address of its <see cref="HeapPage"/>, or null for a text that is a
        ///     C-allocator block of its own.</summary>
        private const int HeapBlockHeaderBytes = 8;

        /// <summary>The bytes of a page's <see cref="HeapPage"/> header; the first block starts right after it.</summary>
        private const int HeapPageHeaderBytes = 64;

        /// <summary>log2 of the smallest page (1 KiB): a buffer with a few heap texts holds little memory for them.</summary>
        private const int HeapMinPageShift = 10;

        /// <summary>log2 of the largest page (64 KiB): pages double per class up to here, so a bulk assignment takes few pages.</summary>
        private const int HeapMaxPageShift = 16;

        /// <summary>
        ///     The header of a heap page: one size class's blocks follow it. A page sits in two doubly linked lists — every page
        ///     of the allocator (<see cref="ClearAndFree"/> returns them) and, while it has room, its class's OPEN list
        ///     (allocations take the first page there).
        /// </summary>
        /// <remarks>Exactly <see cref="HeapPageHeaderBytes"/> bytes, so the blocks after it keep the page's alignment (the buffer
        ///     pool aligns pages of 4 KiB and more to 64 bytes).</remarks>
        [StructLayout(LayoutKind.Sequential, Size = HeapPageHeaderBytes)]
        private struct HeapPage
        {
            /// <summary>The next page of the class's open list (null at its end).</summary>
            public HeapPage* NextOpen;

            /// <summary>The previous page of the class's open list (null for its first).</summary>
            public HeapPage* PrevOpen;

            /// <summary>The next page of the allocator (null at the end).</summary>
            public HeapPage* NextAll;

            /// <summary>The previous page of the allocator (null for the first).</summary>
            public HeapPage* PrevAll;

            /// <summary>The first freed block (its header still names this page; the 8 bytes after the header link the next
            ///     freed block), or null.</summary>
            public byte* Free;

            /// <summary>The first block never handed out; equals <see cref="End"/> once every block was used once.</summary>
            public byte* Bump;

            /// <summary>One past the page's last whole block.</summary>
            public byte* End;

            /// <summary>The blocks handed out and not freed: the page is released (<see cref="ReleaseHeapPage"/>) when this
            ///     reaches 0.</summary>
            public int Live;

            /// <summary>The size class (block stride 16·(Class + 2)).</summary>
            public byte Class;

            /// <summary>log2 of the page's bytes (the size it was taken from the buffer pool with, which Return needs).</summary>
            public byte Shift;

            /// <summary>1 while the page is in its class's open list.</summary>
            public byte Open;
        }

        // ------------------------------------------------------------------------------------------------------------
        // fault injection (tests only)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Whether an allocation failure is scheduled on some thread (<see cref="FailAllocationAfter"/>). Production code
        ///     never sets it, so an allocation pays one static read and no thread-static access.
        /// </summary>
        private static volatile bool s_faultInjectionArmed;

        /// <summary>
        ///     The number of arena / heap allocations this thread may still make before the scheduled one fails (1 = the next
        ///     one fails); 0 when nothing is scheduled. Thread-static, so a scheduled failure never reaches another thread.
        /// </summary>
        [ThreadStatic]
        private static int t_failCountdown;

        /// <summary>
        ///     Test hook: makes the arena or heap allocation after <paramref name="successes"/> more successful ones on THIS
        ///     thread fail, as an exhausted allocator fails (the request returns null, the member returns -1, and the caller
        ///     raises NumPy's <c>MemoryError</c> text and runs its clean-up). Only medium and long strings allocate (a short
        ///     string lives in its cell), so only they count. The failure fires once; <see cref="DisarmFaultInjection"/> cancels
        ///     a failure that has not fired.
        /// </summary>
        /// <param name="successes">The number of allocations that still succeed first (0: the next one fails).</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="successes"/> is negative.</exception>
        /// <remarks>Out-of-memory paths are otherwise unreachable on a test host: a real failure needs the native heap
        ///     exhausted. NumPy's own tests cannot reach them either; this hook is how NumSharp proves each one raises the right
        ///     text and frees what the operation duplicated before the failure.</remarks>
        internal static void FailAllocationAfter(int successes)
        {
            if (successes < 0)
                throw new ArgumentOutOfRangeException(nameof(successes), successes, "The number of successful allocations is not negative.");
            t_failCountdown = successes + 1;
            s_faultInjectionArmed = true;
        }

        /// <summary>Test hook: cancels this thread's scheduled allocation failure (<see cref="FailAllocationAfter"/>).</summary>
        internal static void DisarmFaultInjection()
        {
            t_failCountdown = 0;
            s_faultInjectionArmed = false;
        }

        /// <summary>
        ///     Whether the allocation being made is the one <see cref="FailAllocationAfter"/> scheduled to fail: counts this
        ///     thread's allocations down while a failure is scheduled.
        /// </summary>
        /// <returns>True for the scheduled failure (once).</returns>
        private static bool InjectedFailure()
        {
            // The common case: nothing scheduled anywhere — one volatile read, no thread-static access.
            if (!s_faultInjectionArmed || t_failCountdown <= 0)
                return false;
            if (--t_failCountdown != 0)
                return false;
            s_faultInjectionArmed = false;
            return true;
        }

        /// <summary>Refuses a big-endian host, whose cell would hold the flags in byte 0 (NumPy's other layout).</summary>
        /// <exception cref="PlatformNotSupportedException">The host is big-endian.</exception>
        static VStringAllocator()
        {
            // The cell layout above is NumPy's little-endian one; a big-endian host would put the flags in byte 0.
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("StringDType storage follows NumPy's little-endian cell layout; this host is big-endian.");
        }

        // ------------------------------------------------------------------------------------------------------------
        // state (npy_string_allocator: the arena and the mutex)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The arena buffer (<c>arena.buffer</c>); null until the first arena string.</summary>
        private byte* _arena;

        /// <summary>The arena's capacity in bytes (<c>arena.size</c>).</summary>
        private ulong _arenaSize;

        /// <summary>The first free arena byte (<c>arena.cursor</c>).</summary>
        private ulong _arenaCursor;

        /// <summary>The allocator mutex (<c>allocator_lock</c>).</summary>
        private readonly object _lock = new object();

        /// <summary>Arena and heap bytes reported through the pool's coarse accounting
        ///     (<see cref="Unmanaged.Pooling.SizeBucketedBufferPool.AccountForeignAllocation"/>); released at <see cref="ClearAndFree"/>.</summary>
        private long _reportedPressure;

        /// <summary>Heap bytes allocated since the last report (reported in batches of <see cref="PressureBatch"/>).</summary>
        private long _unreportedHeap;

        /// <summary>Set once <see cref="ClearAndFree"/> ran: the arena is gone and no cell of this buffer may be read.</summary>
        private bool _freed;

        /// <summary>The number of live heap strings (diagnostic: tests prove the clear loop frees every one).</summary>
        internal long HeapStrings { get; private set; }

        /// <summary>Every heap page of this allocator, linked through <see cref="HeapPage.NextAll"/>; null when there is none.</summary>
        private HeapPage* _heapPages;

        /// <summary>Per size class, the first page of its open list (pages with a free or never-used block), as an address;
        ///     created with the first heap page, so a buffer without heap texts pays nothing for it.</summary>
        private nint[] _heapOpen;

        /// <summary>Per size class, log2 of the bytes of the next page it takes (0 before its first page); created with
        ///     <see cref="_heapOpen"/>.</summary>
        private byte[] _heapShift;

        /// <summary>The heap texts longer than <see cref="HeapPoolMaxBytes"/>, each a C-allocator block of its own: the clear
        ///     loop visits the cells only while there is one.</summary>
        private long _soloHeapStrings;

        /// <summary>An emptied page kept for the next page any class needs (in neither list; see
        ///     <see cref="ReleaseHeapPage"/>), or null.</summary>
        private HeapPage* _heapSpare;

        /// <summary>The number of heap pages this allocator holds, the spare not counted (diagnostic: tests prove an emptied
        ///     page is released).</summary>
        internal long HeapPages { get; private set; }

        /// <summary>The heap texts that are C-allocator blocks of their own (diagnostic: tests prove the clear loop frees every
        ///     one).</summary>
        internal long HeapSoloStrings => _soloHeapStrings;

        /// <summary>Whether the allocator holds an emptied spare page (diagnostic; not counted in <see cref="HeapPages"/>).</summary>
        internal bool HasHeapSpare => _heapSpare != null;

        /// <summary>The arena's capacity in bytes (diagnostic).</summary>
        internal long ArenaCapacity => (long)_arenaSize;

        /// <summary>The arena's used bytes (diagnostic).</summary>
        internal long ArenaUsed => (long)_arenaCursor;

        /// <summary>
        ///     The arena's first byte (null before the first arena string): the base an arena cell's OFFSET is added to. The
        ///     <c>StringDType</c> kernels read it once per call and decode cells with it (<see cref="NDVStringCells.Load"/>)
        ///     instead of calling <see cref="load"/> per element.
        /// </summary>
        /// <remarks>
        ///     Valid only while the lock is held AND nothing allocates in this allocator: an arena string's allocation can
        ///     reallocate the arena (<see cref="ArenaMalloc"/>), which moves every arena text. A loop whose output allocator is
        ///     also an input's must therefore re-read the base after each allocation, or copy the input text first (NumPy's
        ///     in-place temporaries).
        /// </remarks>
        internal byte* ArenaBase => _arena;

        /// <summary>True once the owning buffer was released and <see cref="ClearAndFree"/> ran (diagnostic).</summary>
        internal bool IsFreed => _freed;

        /// <summary>The number of allocators whose <see cref="ClearAndFree"/> has not run (diagnostic, process-wide).</summary>
        internal static long LiveAllocators => Interlocked.Read(ref _liveAllocators);

        /// <summary>Backing field of <see cref="LiveAllocators"/>.</summary>
        private static long _liveAllocators;

        /// <summary>
        ///     A new allocator with an empty arena (<c>NpyString_new_allocator</c>; the arena buffer is allocated by the first
        ///     arena string).
        /// </summary>
        internal VStringAllocator()
        {
            Interlocked.Increment(ref _liveAllocators);
        }

        // ------------------------------------------------------------------------------------------------------------
        // locking (NpyString_acquire_allocator / NpyString_release_allocator / ..._allocators)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Acquires the allocator mutex (<c>NpyString_acquire_allocator</c>); pair it with exactly one <see cref="release_allocator"/>.</summary>
        internal void acquire_allocator() => Monitor.Enter(_lock);

        /// <summary>Releases the allocator mutex (<c>NpyString_release_allocator</c>).</summary>
        internal void release_allocator() => Monitor.Exit(_lock);

        /// <summary>
        ///     Acquires two allocators' mutexes at once (<c>NpyString_acquire_allocators</c>): the same allocator twice is locked
        ///     once. Distinct allocators are locked in the order of their creation (<see cref="_id"/>, unique per allocator),
        ///     so two threads copying in opposite directions cannot deadlock (NumPy relies on the GIL for that).
        /// </summary>
        /// <param name="a">The first allocator.</param>
        /// <param name="b">The second allocator (may be <paramref name="a"/>).</param>
        internal static void acquire_allocators(VStringAllocator a, VStringAllocator b)
        {
            if (ReferenceEquals(a, b))
            {
                a.acquire_allocator();
                return;
            }
            var (first, second) = a._id < b._id ? (a, b) : (b, a);
            first.acquire_allocator();
            second.acquire_allocator();
        }

        /// <summary>
        ///     A unique, increasing number per allocator: the lock order of
        ///     <see cref="acquire_allocators(VStringAllocator, VStringAllocator)"/> and of its list overload.
        /// </summary>
        private readonly long _id = Interlocked.Increment(ref _nextId);

        /// <summary>The last <see cref="_id"/> handed out.</summary>
        private static long _nextId;

        /// <summary>
        ///     Releases what <see cref="acquire_allocators(VStringAllocator, VStringAllocator)"/> acquired
        ///     (<c>NpyString_release_allocators</c> for two allocators).
        /// </summary>
        /// <param name="a">The first allocator.</param>
        /// <param name="b">The second allocator (may be <paramref name="a"/>).</param>
        internal static void release_allocators(VStringAllocator a, VStringAllocator b)
        {
            a.release_allocator();
            if (!ReferenceEquals(a, b))
                b.release_allocator();
        }

        /// <summary>
        ///     Acquires the mutexes of several allocators (<c>NpyString_acquire_allocators</c> for any count): the list is sorted
        ///     into lock order (<see cref="_id"/>) and each distinct allocator is locked once. Pair it with
        ///     <see cref="release_allocators(System.Collections.Generic.List{VStringAllocator})"/> on the SAME list, which this
        ///     method leaves sorted.
        /// </summary>
        /// <param name="allocators">The allocators (duplicates allowed); sorted in place.</param>
        /// <exception cref="ArgumentNullException"><paramref name="allocators"/> or one of its entries is null.</exception>
        internal static void acquire_allocators(System.Collections.Generic.List<VStringAllocator> allocators)
        {
            if (allocators is null)
                throw new ArgumentNullException(nameof(allocators));
            foreach (var a in allocators)
                if (a is null)
                    throw new ArgumentNullException(nameof(allocators), "An allocator to lock is null.");
            allocators.Sort((x, y) => x._id.CompareTo(y._id));
            VStringAllocator previous = null;
            foreach (var a in allocators)
            {
                // The same allocator twice (two operands over one buffer) is locked once; the order is global, so two threads
                // locking overlapping sets cannot deadlock.
                if (!ReferenceEquals(a, previous))
                    a.acquire_allocator();
                previous = a;
            }
        }

        /// <summary>
        ///     Releases what <see cref="acquire_allocators(System.Collections.Generic.List{VStringAllocator})"/> acquired,
        ///     given the same (now sorted) list (<c>NpyString_release_allocators</c>).
        /// </summary>
        /// <param name="allocators">The list passed to
        ///     <see cref="acquire_allocators(System.Collections.Generic.List{VStringAllocator})"/>.</param>
        internal static void release_allocators(System.Collections.Generic.List<VStringAllocator> allocators)
        {
            VStringAllocator previous = null;
            foreach (var a in allocators)
            {
                if (!ReferenceEquals(a, previous))
                    a.release_allocator();
                previous = a;
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // cell inspection (no allocator state)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The flags of a cell (<c>STRING_FLAGS</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>The high nibble of byte 15.</returns>
        internal static byte Flags(byte* s) => (byte)(s[15] & NPY_STRING_FLAG_MASK);

        /// <summary>Whether a cell holds its text itself (<c>is_short_string</c>: initialized and outside the arena, not missing, not long).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>True for a short string.</returns>
        internal static bool IsShort(byte* s) => Flags(s) == (NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA);

        /// <summary>Whether a cell is an NA value (<c>NpyString_isnull</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>True when the <see cref="NPY_STRING_MISSING"/> flag is set.</returns>
        internal static bool isnull(byte* s) => (s[15] & NPY_STRING_MISSING) == NPY_STRING_MISSING;

        /// <summary>The size field of a vstring cell without its flags byte (<c>VSTRING_SIZE</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>The size in bytes.</returns>
        private static ulong VStringSize(byte* s) => *(ulong*)(s + 8) & SizeMask;

        /// <summary>Sets a vstring cell's size and keeps its flags byte (<c>set_vstring_size</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <param name="size">The new size (at most <see cref="NPY_MAX_STRING_SIZE"/>).</param>
        private static void SetVStringSize(byte* s, ulong size)
        {
            byte flags = s[15];
            *(ulong*)(s + 8) = size;
            s[15] = flags;
        }

        /// <summary>The text length of a cell in bytes (<c>NpyString_size</c>): 0 for an NA value.</summary>
        /// <param name="s">The cell.</param>
        /// <returns>The UTF-8 byte count.</returns>
        internal static long size(byte* s)
        {
            if (isnull(s))
                return 0;
            if (IsShort(s))
                return s[15] & NPY_SHORT_STRING_SIZE_MASK;
            return (long)VStringSize(s);
        }

        /// <summary>
        ///     Whether a cell is a heap string that owns memory: <see cref="NPY_STRING_OUTSIDE_ARENA"/> set, not a
        ///     short string, and a nonzero size.
        /// </summary>
        /// <param name="s">The cell.</param>
        /// <returns>True when freeing the cell frees heap memory.</returns>
        internal static bool OwnsHeapString(byte* s) => (s[15] & NPY_STRING_OUTSIDE_ARENA) != 0 && !IsShort(s) && VStringSize(s) > 0;

        /// <summary>
        ///     The arena bytes a duplicate of this cell appends to a destination arena whose cells hold no arena strings (so no
        ///     slot can be reused): the text and its size prefix for a medium or long string (an arena or heap string of the
        ///     source — <c>NpyString_newsize</c> puts both in the arena of a fresh cell), nothing for a short, empty or NA cell.
        /// </summary>
        /// <param name="s">The source cell.</param>
        /// <returns>The bytes <see cref="ArenaMalloc"/> would append for it.</returns>
        internal static long ArenaBytesForDuplicate(byte* s)
        {
            if (isnull(s) || IsShort(s))
                return 0;
            ulong size = VStringSize(s);
            return size == 0 ? 0 : (long)(size <= NPY_MEDIUM_STRING_MAX_SIZE ? size + 1 : size + 8);
        }

        /// <summary>
        ///     The arena bytes packing a text of <paramref name="size"/> bytes into a fresh cell appends (nothing for a short
        ///     text).
        /// </summary>
        /// <param name="size">The UTF-8 size.</param>
        /// <returns>The bytes, prefix included.</returns>
        /// <remarks>Inlined and branch-free (sign masks: <c>(limit - size) &gt;&gt; 63</c> is all ones past the limit): the size
        ///     passes call it once per element over text whose sizes straddle the 15- and 255-byte lines at random, where the
        ///     JIT's compare-and-branch form mispredicted half the time. Valid for every size below 2^62.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long ArenaBytesForText(long size)
        {
            long overShort = (NPY_SHORT_STRING_MAX_SIZE - size) >> 63;
            long overMedium = (NPY_MEDIUM_STRING_MAX_SIZE - size) >> 63;
            // the size prefix: one byte for a medium text, eight for a long one; nothing at all for a short text
            return (size + 1 + (7 & overMedium)) & overShort;
        }

        /// <summary>Whether this allocator holds any medium, long or heap string (a copy of its cells may need arena space).</summary>
        internal bool HoldsLongStrings => _arenaCursor > 0 || HeapStrings > 0;

        /// <summary>
        ///     Grows the arena so that <paramref name="bytes"/> more can be appended without another reallocation — the bulk
        ///     copies call it with the exact total a fresh destination will take, instead of letting <see cref="ArenaMalloc"/>
        ///     reallocate (and copy and zero) the arena some 50 times on the way up by 25 % steps (measured: a copy of a
        ///     million 20–40-byte strings 93 ms → ~32 ms).
        /// </summary>
        /// <param name="bytes">The bytes about to be appended.</param>
        /// <remarks>
        ///     Best effort: a reallocation that fails leaves the arena as it was (the appends then grow it themselves and
        ///     report NumPy's MemoryError if that fails too). Only the arena's CAPACITY differs from NumPy's growth sequence —
        ///     never an offset (the cursor decides those), so no cell changes. The lock must be held.
        /// </remarks>
        internal void TryReserve(long bytes)
        {
            if (bytes <= 0 || _arenaSize - _arenaCursor >= (ulong)bytes)
                return;
            ulong newsize = _arenaCursor + (ulong)bytes;
            if (newsize < _arenaCursor)
                return; // overflow: leave it to the appends
            // Best effort: a failed growth leaves the arena as it was, and the appends grow it themselves.
            TryGrowArena(newsize);
        }

        /// <summary>
        ///     Reallocates the arena to exactly <paramref name="newsize"/> bytes and zeroes the space past the cursor: the
        ///     growth step of <c>arena_malloc</c>, which <see cref="ArenaMalloc"/> (125 % of what it needs) and
        ///     <see cref="TryReserve"/> (the exact total of a bulk copy) share, so the arena has ONE reallocation site.
        /// </summary>
        /// <param name="newsize">The new capacity in bytes; larger than the current capacity.</param>
        /// <returns>True when the arena now holds <paramref name="newsize"/> bytes; false when the reallocation failed, in
        ///     which case the arena, its capacity and the accounted bytes are unchanged.</returns>
        /// <remarks>
        ///     The space past the cursor must read as zero, as NumPy's <c>memset</c> leaves it. The first growth creates the
        ///     arena with <c>calloc</c> (<c>NativeMemory.AllocZeroed</c>): a large block's pages come zeroed from the operating
        ///     system as they are first written, so no clearing pass touches them all up front (a reserved arena of a few
        ///     megabytes is written once, by the strings, instead of twice). A later growth reallocates, and a realloc returns
        ///     the new tail uninitialized, so that tail is cleared. Both blocks are freed by <c>NativeMemory.Free</c> (the same
        ///     C allocator). The lock must be held.
        /// </remarks>
        private bool TryGrowArena(ulong newsize)
        {
            byte* newbuf;
            bool fresh = _arena == null;
            try
            {
                newbuf = fresh ? (byte*)NativeMemory.AllocZeroed((nuint)newsize) : (byte*)NativeMemory.Realloc(_arena, (nuint)newsize);
            }
            catch (OutOfMemoryException)
            {
                return false;
            }
            if (!fresh)
                NativeMemory.Clear(newbuf + _arenaCursor, (nuint)(newsize - _arenaCursor));
            AccountArena((long)(newsize - _arenaSize));
            _arena = newbuf;
            _arenaSize = newsize;
            return true;
        }

        // ------------------------------------------------------------------------------------------------------------
        // the arena (static_string.c: vstring_buffer, arena_malloc, arena_free)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The text of a vstring cell (<c>vstring_buffer</c>): the heap pointer of a heap string, else the arena address of
        ///     its offset — null when the arena does not exist yet.
        /// </summary>
        /// <param name="s">A vstring cell (not short).</param>
        /// <returns>The first text byte, or null. An arena address is valid only until this allocator next allocates (the
        ///     arena can move), so a caller writes the text before its next allocation — as the <c>numpy.strings</c> loops do
        ///     after <see cref="newemptysize"/> (NumPy's <c>load_new_string</c>).</returns>
        internal byte* VStringBuffer(byte* s)
        {
            if ((s[15] & NPY_STRING_OUTSIDE_ARENA) != 0)
                return (byte*)*(ulong*)s;
            if (_arena == null)
                return null;
            return _arena + *(ulong*)s;
        }

        /// <summary>
        ///     Appends an allocation of <paramref name="size"/> bytes to the arena (<c>arena_malloc</c>): a one-byte size prefix
        ///     for a medium string, an eight-byte one for a long string; when it does not fit, the arena is reallocated to 125 %
        ///     of what it needs and the new tail is zeroed.
        /// </summary>
        /// <param name="size">The text size.</param>
        /// <returns>The first text byte (after the prefix), or null when the size overflows or the reallocation fails.</returns>
        private byte* ArenaMalloc(ulong size)
        {
            // A failure scheduled by a test (FailAllocationAfter): the same null an exhausted arena returns.
            if (InjectedFailure())
                return null;
            // one extra size_t to store the size of the allocation
            ulong storage = size <= NPY_MEDIUM_STRING_MAX_SIZE ? size + 1 : size + 8;
            if (_arenaSize - _arenaCursor < storage)
            {
                ulong minsize = _arenaCursor + storage;
                if (minsize < _arenaCursor)
                    return null; // overflow means out of memory
                // Allocate 25% more than needed for this string.
                ulong newsize = minsize + minsize / 4;
                if (newsize < minsize)
                    return null; // overflow means out of memory
                if (!TryGrowArena(newsize))
                    return null;
            }
            byte* ret;
            if (size <= NPY_MEDIUM_STRING_MAX_SIZE)
            {
                _arena[_arenaCursor] = (byte)size;
                ret = _arena + _arenaCursor + 1;
            }
            else
            {
                // Long string size not necessarily memory-aligned, so an unaligned store (NumPy: memcpy).
                Unsafe.WriteUnaligned(_arena + _arenaCursor, size);
                ret = _arena + _arenaCursor + 8;
            }
            _arenaCursor += storage;
            return ret;
        }

        /// <summary>
        ///     Zeroes an arena string's text (<c>arena_free</c>), after checking that it lies inside the arena.
        /// </summary>
        /// <param name="s">An arena cell with a non-zero size.</param>
        /// <returns>0, or -1 when the text is not inside the arena (a corrupt cell).</returns>
        private int ArenaFree(byte* s)
        {
            ulong size = VStringSize(s);
            byte* ptr = VStringBuffer(s);
            if (ptr == null)
                return -1;
            ulong bufStart = (ulong)_arena, ptrLoc = (ulong)ptr, endLoc = ptrLoc + size, bufEnd = bufStart + _arenaSize;
            if (ptrLoc < bufStart || ptrLoc > bufEnd || endLoc > bufEnd)
                return -1;
            NativeMemory.Clear(ptr, (nuint)size);
            return 0;
        }

        // ------------------------------------------------------------------------------------------------------------
        // the NpyString_* API
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Unpacks a cell (<c>NpyString_load</c>): its text pointer and size, read-only, valid while the lock is held and the
        ///     cell is not written.
        /// </summary>
        /// <param name="packed">The cell.</param>
        /// <param name="buf">Receives the first text byte (null for an NA value; a non-null pointer to nothing for <c>''</c>).</param>
        /// <param name="size">Receives the text size in bytes.</param>
        /// <returns>1 for an NA value, 0 for a string, -1 when an arena string's arena is missing (a corrupt cell).</returns>
        internal int load(byte* packed, out byte* buf, out long size)
        {
            if (isnull(packed))
            {
                size = 0;
                buf = null;
                return 1;
            }
            if (IsShort(packed))
            {
                size = packed[15] & NPY_SHORT_STRING_SIZE_MASK;
                buf = packed;
                return 0;
            }
            ulong vsize = VStringSize(packed);
            byte* b = EmptyString;
            if (vsize > 0)
            {
                b = VStringBuffer(packed);
                if (b == null)
                {
                    buf = null;
                    size = 0;
                    return -1;
                }
            }
            size = (long)vsize;
            buf = b;
            return 0;
        }

        /// <summary>
        ///     The process's one static empty text (NumPy's <c>EMPTY_STRING</c>): the non-null pointer to nothing that a read of
        ///     an empty vstring cell hands out, here and in the readers outside the allocator (<c>NDVStringCells</c>'s cell
        ///     loads, <c>VStringSort</c>'s NA-as-default-string reads).
        /// </summary>
        /// <remarks>
        ///     One 1-byte block allocated once and never freed (the native-allocation chokepoint gate pins this site). Shared
        ///     rather than allocated per reader, as NumPy shares its one constant. Never write through it and never free it:
        ///     every empty text in the process is this pointer, and a caller that treats it as an owned buffer corrupts them
        ///     all. Its single byte is zero, so a reader that dereferences it as a C string sees an empty one.
        /// </remarks>
        internal static readonly byte* EmptyString = (byte*)NativeMemory.AllocZeroed(1);

        /// <summary>
        ///     The allocation for a new string of <paramref name="size"/> bytes in a cell (<c>heap_or_arena_allocate</c>): a cell
        ///     never allocated appends to the arena; an arena cell reuses its old allocation when the text fits its capacity
        ///     (read from the size prefix); anything else allocates on the heap.
        /// </summary>
        /// <param name="toInit">The cell (its flags byte is updated).</param>
        /// <param name="size">The text size (more than <see cref="NPY_SHORT_STRING_MAX_SIZE"/>).</param>
        /// <param name="onHeap">Receives true for a heap allocation (the cell then stores the pointer, not an offset).</param>
        /// <returns>The first text byte, or null on failure.</returns>
        private byte* HeapOrArenaAllocate(byte* toInit, ulong size, out bool onHeap)
        {
            onHeap = false;
            byte flags = toInit[15];
            if ((flags & NPY_STRING_OUTSIDE_ARENA) == 0)
            {
                // Arena allocation or re-allocation.
                if (flags == 0)
                {
                    // string isn't previously allocated, so add to existing arena allocation
                    byte* ret = ArenaMalloc(size);
                    toInit[15] = size <= NPY_MEDIUM_STRING_MAX_SIZE ? NPY_STRING_INITIALIZED : (byte)(NPY_STRING_INITIALIZED | NPY_STRING_LONG);
                    return ret;
                }
                // String was in arena. See if there is still space.
                // The size is stored "behind" the beginning of the allocation.
                byte* buf = VStringBuffer(toInit);
                if (buf == null)
                    return null;
                ulong allocSize = (flags & NPY_STRING_LONG) != 0
                    ? Unsafe.ReadUnaligned<ulong>(buf - 8) // long string size not necessarily memory-aligned
                    : *(buf - 1);                          // medium string size is stored in a char
                if (size <= allocSize)
                    return buf; // we have room!
                // No room, fall through to heap allocation.
            }
            // Heap allocate
            onHeap = true;
            toInit[15] = NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA | NPY_STRING_LONG;
            // A failure scheduled by a test (FailAllocationAfter): the same null an exhausted heap gives.
            if (InjectedFailure())
                return null;
            byte* heap = HeapMalloc(size);
            if (heap == null)
                return null;
            HeapStrings++;
            return heap;
        }

        /// <summary>
        ///     The memory of a new heap text (NumPy's <c>allocator->malloc</c> in <c>heap_or_arena_allocate</c>): a block of the
        ///     text's size class from one of this allocator's heap pages, or, for a text longer than
        ///     <see cref="HeapPoolMaxBytes"/>, a C-allocator block of its own (see the class remarks).
        /// </summary>
        /// <param name="size">The text size in bytes (more than <see cref="NPY_SHORT_STRING_MAX_SIZE"/>).</param>
        /// <returns>The first text byte (the block's contents are undefined; the caller writes the text), or null when the
        ///     memory is exhausted.</returns>
        /// <remarks>The lock must be held. Every byte up to the size class's capacity belongs to the block, so a reader of the
        ///     text's <paramref name="size"/> bytes never leaves it.</remarks>
        private byte* HeapMalloc(ulong size)
        {
            if (size > HeapPoolMaxBytes)
            {
                // A long text: the C allocator's block (the copy of the text costs more than the call), a null page in its
                // header so the free hands it back to the C allocator.
                byte* solo;
                try
                {
                    solo = (byte*)NativeMemory.Alloc((nuint)(size + HeapBlockHeaderBytes));
                }
                catch (OutOfMemoryException)
                {
                    return null;
                }
                *(HeapPage**)solo = null;
                _soloHeapStrings++;
                AccountHeap((long)size);
                return solo + HeapBlockHeaderBytes;
            }
            int k = HeapClass(size);
            HeapPage* page = _heapOpen == null ? null : (HeapPage*)_heapOpen[k];
            if (page == null && (page = NewHeapPage(k)) == null)
                return null;
            byte* block = page->Free;
            if (block != null)
                page->Free = *(byte**)(block + HeapBlockHeaderBytes);
            else
            {
                block = page->Bump;
                page->Bump = block + HeapStride(k);
            }
            page->Live++;
            // A full page leaves its class's open list, so the next allocation of the class finds room at the list's head.
            if (page->Free == null && page->Bump == page->End)
                CloseHeapPage(page);
            *(HeapPage**)block = page;
            return block + HeapBlockHeaderBytes;
        }

        /// <summary>
        ///     Releases the memory of a heap text (NumPy's <c>allocator->free</c>): a page block goes on its page's free list,
        ///     and the page itself is released (<see cref="ReleaseHeapPage"/>: the spare, or back to the buffer pool) when that
        ///     was its last live block; a block of its own goes back to the C allocator.
        /// </summary>
        /// <param name="text">The first text byte, as <see cref="HeapMalloc"/> returned it.</param>
        /// <remarks>The lock must be held.</remarks>
        private void HeapFree(byte* text)
        {
            byte* block = text - HeapBlockHeaderBytes;
            HeapPage* page = *(HeapPage**)block;
            if (page == null)
            {
                NativeMemory.Free(block);
                _soloHeapStrings--;
                return;
            }
            System.Diagnostics.Debug.Assert(block >= (byte*)page + HeapPageHeaderBytes && block < page->End
                         && (block - ((byte*)page + HeapPageHeaderBytes)) % HeapStride(page->Class) == 0,
                "A heap text's header names a page the text is not a block of.");
            *(byte**)(block + HeapBlockHeaderBytes) = page->Free;
            page->Free = block;
            if (--page->Live == 0)
            {
                ReleaseHeapPage(page);
                return;
            }
            // A page that was full has room again: back into its class's open list.
            if (page->Open == 0)
                OpenHeapPage(page);
        }

        /// <summary>The size class of a heap text: the smallest class whose blocks hold it (16 → 0, 25 → 1, …, 1,032 → 63).</summary>
        /// <param name="size">The text size, 16 to <see cref="HeapPoolMaxBytes"/>.</param>
        /// <returns>The class index.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HeapClass(ulong size) => (int)((size + HeapBlockHeaderBytes + 15) >> 4) - 2;

        /// <summary>The block stride of a size class: its header and its text capacity (16·(k + 2) bytes).</summary>
        /// <param name="k">The class index.</param>
        /// <returns>The stride in bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HeapStride(int k) => (k + 2) << 4;

        /// <summary>
        ///     Takes a new page for size class <paramref name="k"/> from the buffer pool and puts it at the head of the class's
        ///     open list: the class's first page is the smallest power of two of at least 1 KiB that holds four blocks, each
        ///     later one twice the last, up to 64 KiB.
        /// </summary>
        /// <param name="k">The class index.</param>
        /// <returns>The page, or null when the memory is exhausted.</returns>
        /// <remarks>The lock must be held.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private HeapPage* NewHeapPage(int k)
        {
            _heapOpen ??= new nint[HeapClassCount];
            _heapShift ??= new byte[HeapClassCount];
            int stride = HeapStride(k);
            int shift = _heapShift[k];
            if (shift == 0)
            {
                shift = HeapMinPageShift;
                while ((1 << shift) - HeapPageHeaderBytes < 4 * stride)
                    shift++;
            }
            HeapPage* page;
            if (_heapSpare != null && (1 << _heapSpare->Shift) - HeapPageHeaderBytes >= 4 * stride)
            {
                // The spare (the last emptied page) holds four blocks of this class: re-use it at its own size, so one
                // element rewritten back and forth (its page emptied, then needed again) costs no buffer-pool round trip.
                page = _heapSpare;
                _heapSpare = null;
                shift = page->Shift;
            }
            else
            {
                try
                {
                    page = (HeapPage*)Unmanaged.Pooling.SizeBucketedBufferPool.Take(1 << shift);
                }
                catch (OutOfMemoryException)
                {
                    return null;
                }
            }
            int bytes = 1 << shift;
            byte* first = (byte*)page + HeapPageHeaderBytes;
            page->Free = null;
            page->Bump = first;
            page->End = first + (bytes - HeapPageHeaderBytes) / stride * stride;
            page->Live = 0;
            page->Class = (byte)k;
            page->Shift = (byte)shift;
            page->Open = 0;
            page->PrevAll = null;
            page->NextAll = _heapPages;
            if (_heapPages != null)
                _heapPages->PrevAll = page;
            _heapPages = page;
            OpenHeapPage(page);
            HeapPages++;
            _heapShift[k] = (byte)Math.Min(shift + 1, HeapMaxPageShift);
            return page;
        }

        /// <summary>Puts a page at the head of its class's open list (it has a free or never-used block).</summary>
        /// <param name="page">The page (not in the list).</param>
        private void OpenHeapPage(HeapPage* page)
        {
            ref nint head = ref _heapOpen[page->Class];
            page->PrevOpen = null;
            page->NextOpen = (HeapPage*)head;
            if (head != 0)
                ((HeapPage*)head)->PrevOpen = page;
            head = (nint)page;
            page->Open = 1;
        }

        /// <summary>Takes a page out of its class's open list (it is full, or about to go back to the buffer pool).</summary>
        /// <param name="page">The page (in the list).</param>
        private void CloseHeapPage(HeapPage* page)
        {
            if (page->PrevOpen != null)
                page->PrevOpen->NextOpen = page->NextOpen;
            else
                _heapOpen[page->Class] = (nint)page->NextOpen;
            if (page->NextOpen != null)
                page->NextOpen->PrevOpen = page->PrevOpen;
            page->NextOpen = page->PrevOpen = null;
            page->Open = 0;
        }

        /// <summary>
        ///     Checks every invariant of the heap pages (diagnostic, for tests): both lists are consistent, each page's bump
        ///     pointer, free list and live count agree, every live block's header names its page, and a page is in its class's
        ///     open list exactly when it has room. A text written past its block's capacity overwrites the next block's header,
        ///     which this finds while that block is live.
        /// </summary>
        /// <returns>Null when every invariant holds, else a description of the first violation.</returns>
        /// <remarks>Takes the lock. Walks every block of every page: for tests only.</remarks>
        internal string ValidateHeapPages()
        {
            acquire_allocator();
            try
            {
                long pages = 0, openFlagged = 0, listedTotal = 0;
                HeapPage* previous = null;
                for (HeapPage* page = _heapPages; page != null; previous = page, page = page->NextAll)
                {
                    if (++pages > HeapPages)
                        return $"the page list holds more than the {HeapPages} pages counted";
                    if (page == _heapSpare)
                        return $"page {pages}: the spare page is still in the page list";
                    if (page->PrevAll != previous)
                        return $"page {pages}: its PrevAll does not name the page before it";
                    if (page->Class >= HeapClassCount)
                        return $"page {pages}: class {page->Class} out of range";
                    int stride = HeapStride(page->Class);
                    long bytes = 1L << page->Shift;
                    byte* first = (byte*)page + HeapPageHeaderBytes;
                    if (page->End != first + (bytes - HeapPageHeaderBytes) / stride * stride)
                        return $"page {pages}: End is not the end of its last whole block";
                    if (page->Bump < first || page->Bump > page->End || (page->Bump - first) % stride != 0)
                        return $"page {pages}: Bump is not a block boundary of the page";
                    long used = (page->Bump - first) / stride;
                    var free = new System.Collections.Generic.HashSet<long>();
                    for (byte* block = page->Free; block != null; block = *(byte**)(block + HeapBlockHeaderBytes))
                    {
                        if (block < first || block >= page->Bump || (block - first) % stride != 0)
                            return $"page {pages}: a freed block is not a used block of the page";
                        if (!free.Add(block - first) || free.Count > used)
                            return $"page {pages}: the free list repeats a block";
                    }
                    if (page->Live != used - free.Count)
                        return $"page {pages}: Live {page->Live}, but {used} blocks used and {free.Count} freed";
                    if (page->Live < 1)
                        return $"page {pages}: a page without live blocks was not released";
                    for (long b = 0; b < used; b++)
                        if (!free.Contains(b * stride) && *(HeapPage**)(first + b * stride) != page)
                            return $"page {pages}: the header of live block {b} does not name its page";
                    bool room = page->Free != null || page->Bump < page->End;
                    if ((page->Open != 0) != room)
                        return $"page {pages}: Open is {page->Open} but the page {(room ? "has" : "has no")} room";
                    if (page->Open != 0)
                        openFlagged++;
                }
                if (pages != HeapPages)
                    return $"the page list holds {pages} pages, {HeapPages} counted";
                for (int k = 0; _heapOpen != null && k < HeapClassCount; k++)
                {
                    HeapPage* before = null;
                    long listed = 0;
                    for (HeapPage* page = (HeapPage*)_heapOpen[k]; page != null; before = page, page = page->NextOpen)
                    {
                        // a cycle in the list would walk forever: no list holds more pages than the allocator has
                        if (++listed > HeapPages)
                            return $"class {k}: the open list holds more pages than the allocator has";
                        if (page->PrevOpen != before)
                            return $"class {k}: the open list's PrevOpen links are broken";
                        if (page->Class != k || page->Open == 0)
                            return $"class {k}: the open list holds a page of class {page->Class} (Open {page->Open})";
                    }
                    listedTotal += listed;
                }
                // every page flagged open is reachable from its class's list (the walk above saw each listed page once)
                if (listedTotal != openFlagged)
                    return $"{openFlagged} pages are flagged open, {listedTotal} are in the open lists";
                return null;
            }
            finally
            {
                release_allocator();
            }
        }

        /// <summary>
        ///     Unlinks a page without live blocks and keeps it as the allocator's one SPARE page when there is none (the next
        ///     page any class needs re-uses it if it holds four of that class's blocks), or returns it to the buffer pool. The
        ///     class's next page is taken at this page's size again, so a class that empties and refills keeps pages near its
        ///     working size instead of doubling.
        /// </summary>
        /// <param name="page">The page (no live block).</param>
        /// <remarks>Why a spare: one element rewritten back and forth between two lengths frees its page's only text and then
        ///     needs a page again, each time; releasing the page made each such write a buffer-pool round trip, 21 % slower
        ///     than the <c>malloc</c> / <c>free</c> pair it replaced (131 → 159 ns a write, measured). A spare bounds what an
        ///     allocator holds beyond its live texts to one page (at most 64 KiB).</remarks>
        private void ReleaseHeapPage(HeapPage* page)
        {
            if (page->Open != 0)
                CloseHeapPage(page);
            if (page->PrevAll != null)
                page->PrevAll->NextAll = page->NextAll;
            else
                _heapPages = page->NextAll;
            if (page->NextAll != null)
                page->NextAll->PrevAll = page->PrevAll;
            _heapShift[page->Class] = page->Shift;
            HeapPages--;
            if (_heapSpare == null)
            {
                page->NextAll = page->PrevAll = null;
                _heapSpare = page;
                return;
            }
            Unmanaged.Pooling.SizeBucketedBufferPool.Return((IntPtr)page, 1L << page->Shift);
        }

        /// <summary>
        ///     Releases a vstring cell's text (<c>heap_or_arena_deallocate</c>): frees a heap string (and clears its pointer),
        ///     zeroes an arena string's text.
        /// </summary>
        /// <param name="s">A vstring cell with a non-zero size.</param>
        /// <returns>0, or -1 for an arena string outside the arena.</returns>
        private int HeapOrArenaDeallocate(byte* s)
        {
            if ((s[15] & NPY_STRING_OUTSIDE_ARENA) != 0)
            {
                // It's a heap string (not in the arena buffer) so it needs to be deallocated with free(). For heap strings
                // the offset is a raw address so this cast is safe. (NumSharp's heap is paged: HeapFree.)
                HeapFree((byte*)*(ulong*)s);
                HeapStrings--;
                *(ulong*)s = 0;
                return 0;
            }
            // In the arena buffer.
            return ArenaFree(s);
        }

        /// <summary>
        ///     Packs the empty string into a cell (<c>NpyString_pack_empty</c>): a heap / short cell becomes the short empty
        ///     string; an arena cell keeps its allocation and gets size 0, so a later write can reuse the arena space.
        /// </summary>
        /// <param name="s">The cell (its old text must already be freed when it was a heap string).</param>
        internal static void pack_empty(byte* s)
        {
            if ((s[15] & NPY_STRING_OUTSIDE_ARENA) != 0)
                s[15] = NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA; // This also sets short string size to 0.
            else
                SetVStringSize(s, 0);
        }

        /// <summary>
        ///     Prepares a cell for a new text of <paramref name="size"/> bytes without writing it (<c>NpyString_newemptysize</c>):
        ///     the empty string, a short string's length, or an arena / heap allocation whose offset or pointer the cell stores.
        /// </summary>
        /// <param name="size">The text size.</param>
        /// <param name="s">The cell (already freed with <see cref="free"/>, or never written).</param>
        /// <returns>0, or -1 when the size exceeds <see cref="NPY_MAX_STRING_SIZE"/> or the memory is exhausted.</returns>
        internal int newemptysize(long size, byte* s)
        {
            if (size == 0)
            {
                pack_empty(s);
                return 0;
            }
            if ((ulong)size > NPY_MAX_STRING_SIZE)
                return -1;
            if (size > NPY_SHORT_STRING_MAX_SIZE)
            {
                byte* buf = HeapOrArenaAllocate(s, (ulong)size, out bool onHeap);
                if (buf == null)
                    return -1;
                *(ulong*)s = onHeap ? (ulong)buf : (ulong)(buf - _arena);
                SetVStringSize(s, (ulong)size);
            }
            else
            {
                // Size can be no larger than 15, so the size data is in the 4 least significant bits of the byte and it is
                // safe to | with the flags. All other metadata of a previous allocation is wiped.
                s[15] = (byte)(NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA | size);
            }
            return 0;
        }

        /// <summary>
        ///     The bump-pointer form of <see cref="newemptysize"/> for a FRESH cell — all zero, never written, so nothing to free
        ///     and no arena slot to reuse — and a medium or long text that fits the arena's free capacity (a fresh output's
        ///     arena is reserved from its size pass): it leaves exactly the bytes <c>NpyString_newemptysize</c> →
        ///     <c>heap_or_arena_allocate</c> → <c>arena_malloc</c> leave (the size prefix, the offset, and the size field with
        ///     <c>INITIALIZED</c> — plus <c>LONG</c> past 255 bytes — in byte 15), without the general path's per-call checks.
        /// </summary>
        /// <param name="cell">The fresh cell (all zero).</param>
        /// <param name="size">The text size: more than <see cref="NPY_SHORT_STRING_MAX_SIZE"/> (a short text belongs in its
        ///     cell).</param>
        /// <returns>The first text byte (the caller writes <paramref name="size"/> bytes there; valid until the arena next
        ///     grows), or null when the text does not fit the free capacity or a test scheduled an allocation failure — the
        ///     caller then takes <see cref="newemptysize"/>, which grows the arena (or fails) exactly as NumPy does.</returns>
        /// <remarks>
        ///     The fast path skips <see cref="InjectedFailure"/>'s countdown entirely while any failure is scheduled (one
        ///     volatile read), so a fault-injection test still sees its failure at the scheduled allocation of the general path.
        ///     A text of size 0 must not come here (NumPy packs the empty string without an allocation).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal byte* AppendFresh(byte* cell, long size)
        {
            bool medium = (ulong)size <= NPY_MEDIUM_STRING_MAX_SIZE;
            // one byte (medium) or one size_t (long) in front of the text holds its allocation size (arena_malloc)
            ulong storage = medium ? (ulong)size + 1 : (ulong)size + 8;
            if (_arenaSize - _arenaCursor < storage || s_faultInjectionArmed)
                return null;
            byte* p = _arena + _arenaCursor;
            ulong flags;
            if (medium)
            {
                *p = (byte)size;
                p += 1;
                flags = NPY_STRING_INITIALIZED;
            }
            else
            {
                Unsafe.WriteUnaligned(p, (ulong)size);
                p += 8;
                flags = NPY_STRING_INITIALIZED | NPY_STRING_LONG;
            }
            // The cell: the offset into the arena (bytes 0-7), then the size with the flags in byte 15.
            Unsafe.WriteUnaligned(cell, (ulong)(p - _arena));
            Unsafe.WriteUnaligned(cell + 8, (ulong)size | (flags << 56));
            _arenaCursor += storage;
            return p;
        }

        /// <summary>
        ///     Allocates a cell's new text and copies <paramref name="size"/> bytes of <paramref name="init"/> into it
        ///     (<c>NpyString_newsize</c>).
        /// </summary>
        /// <param name="init">The UTF-8 text (not inside this allocator's arena: see <see cref="dup"/>).</param>
        /// <param name="size">The text size.</param>
        /// <param name="toInit">The cell (already freed, or never written).</param>
        /// <returns>0, or -1 on failure.</returns>
        internal int newsize(byte* init, long size, byte* toInit)
        {
            if (newemptysize(size, toInit) < 0)
                return -1;
            if (size == 0)
                return 0;
            byte* buf = size > NPY_SHORT_STRING_MAX_SIZE ? VStringBuffer(toInit) : toInit;
            Buffer.MemoryCopy(init, buf, size, size);
            return 0;
        }

        /// <summary>
        ///     Frees a cell's text and leaves an empty string (<c>NpyString_free</c>): the missing flag is removed, a short
        ///     string becomes the short empty string, a vstring is deallocated and gets size 0 (its arena allocation stays
        ///     recorded for reuse).
        /// </summary>
        /// <param name="s">The cell.</param>
        /// <returns>0, or -1 for a corrupt arena cell.</returns>
        internal int free(byte* s)
        {
            // Unconditionally remove flag indicating something was missing. For that case, the string should have been
            // deallocated already, but test anyway in case we later implement the option to mask the string.
            s[15] = (byte)(s[15] & ~NPY_STRING_MISSING);
            if (IsShort(s))
            {
                if ((s[15] & NPY_SHORT_STRING_SIZE_MASK) > 0)
                {
                    // zero buffer and set flags for initialized out-of-arena empty string.
                    new Span<byte>(s, SIZEOF_NPY_PACKED_STATIC_STRING).Clear();
                    s[15] = NPY_STRING_OUTSIDE_ARENA | NPY_STRING_INITIALIZED;
                }
            }
            else if (VStringSize(s) > 0)
            {
                // Deallocate string and set size to 0 to indicate that happened.
                if (HeapOrArenaDeallocate(s) < 0)
                    return -1;
                SetVStringSize(s, 0);
            }
            return 0;
        }

        /// <summary>
        ///     Packs an NA value into a cell (<c>NpyString_pack_null</c>): the old text is freed, and the flags keep the old
        ///     allocation's metadata so a later write can reuse its arena space.
        /// </summary>
        /// <param name="s">The cell.</param>
        /// <returns>0, or -1 on failure.</returns>
        internal int pack_null(byte* s)
        {
            if (free(s) < 0)
                return -1;
            // preserve the flags because we allow mutation, so we need metadata about the original allocation associated
            // with this string in order to determine if there is space in the arena allocation for a new string after a
            // user mutates this one to a non-NULL value.
            s[15] |= NPY_STRING_MISSING;
            return 0;
        }

        /// <summary>
        ///     Packs a UTF-8 text into a cell (<c>NpyString_pack</c>): frees the old text, then allocates and copies.
        /// </summary>
        /// <param name="s">The cell.</param>
        /// <param name="buf">The UTF-8 text (not inside this allocator's arena).</param>
        /// <param name="size">Its size in bytes.</param>
        /// <returns>0, or -1 on failure.</returns>
        internal int pack(byte* s, byte* buf, long size)
        {
            if (free(s) < 0)
                return -1;
            return newsize(buf, size, s);
        }

        /// <summary>
        ///     Copies the string of one cell into another (<c>NpyString_dup</c>), across allocators or within one: an NA stays
        ///     NA, a short string is copied as the cell, an arena / heap string is packed into <paramref name="outAllocator"/>
        ///     (through a temporary copy when both cells share an allocator, because the new allocation can move the arena).
        /// </summary>
        /// <param name="input">The source cell.</param>
        /// <param name="output">The destination cell (freed with <see cref="free"/>, or never written).</param>
        /// <param name="inAllocator">The source's allocator (locked).</param>
        /// <param name="outAllocator">The destination's allocator (locked; may be <paramref name="inAllocator"/>).</param>
        /// <returns>0, or -1 on failure.</returns>
        internal static int dup(byte* input, byte* output, VStringAllocator inAllocator, VStringAllocator outAllocator)
        {
            if (isnull(input))
                return outAllocator.pack_null(output);
            long size = VStringAllocator.size(input); // qualified: the local of the same name hides the method
            if (size == 0)
            {
                pack_empty(output);
                return 0;
            }
            if (IsShort(input))
            {
                Buffer.MemoryCopy(input, output, SIZEOF_NPY_PACKED_STATIC_STRING, SIZEOF_NPY_PACKED_STATIC_STRING);
                return 0;
            }
            byte* inBuf;
            bool usedMalloc = false;
            if (ReferenceEquals(inAllocator, outAllocator))
            {
                // The arena may move when the copy allocates: copy the text out first.
                inBuf = (byte*)NativeMemory.Alloc((nuint)size);
                Buffer.MemoryCopy(inAllocator.VStringBuffer(input), inBuf, size, size);
                usedMalloc = true;
            }
            else
            {
                inBuf = inAllocator.VStringBuffer(input);
            }
            int ret = outAllocator.newsize(inBuf, size, output);
            if (usedMalloc)
                NativeMemory.Free(inBuf);
            return ret;
        }

        /// <summary>
        ///     Frees a destination cell and copies a source cell into it (<c>free_and_copy</c> of NumPy's <c>dtype.c</c>), raising
        ///     NumPy's <c>MemoryError</c> texts on failure.
        /// </summary>
        /// <param name="inAllocator">The source's allocator (locked).</param>
        /// <param name="outAllocator">The destination's allocator (locked; may be <paramref name="inAllocator"/>).</param>
        /// <param name="input">The source cell.</param>
        /// <param name="output">The destination cell (its old text is freed first).</param>
        /// <param name="location">The operation named in the error (<c>"string to string cast"</c>).</param>
        /// <exception cref="OutOfMemoryException"><c>Failed to deallocate string in {location}</c> (a corrupt destination
        ///     cell) or <c>Failed to allocate string in {location}</c> (the memory is exhausted).</exception>
        internal static void FreeAndCopy(VStringAllocator inAllocator, VStringAllocator outAllocator, byte* input, byte* output, string location)
        {
            if (outAllocator.free(output) < 0)
                throw new OutOfMemoryException($"Failed to deallocate string in {location}");
            if (dup(input, output, inAllocator, outAllocator) < 0)
                throw new OutOfMemoryException($"Failed to allocate string in {location}");
        }

        /// <summary>
        ///     Whether two cells refer to the same text bytes (<c>NpyString_share_memory</c>): only possible for the same
        ///     allocator and two vstring (not short, not NA) cells with one buffer.
        /// </summary>
        /// <param name="s1">The first cell.</param>
        /// <param name="a1">Its allocator.</param>
        /// <param name="s2">The second cell.</param>
        /// <param name="a2">Its allocator.</param>
        /// <returns>True when both cells point at the same text.</returns>
        internal static bool share_memory(byte* s1, VStringAllocator a1, byte* s2, VStringAllocator a2)
        {
            if (!ReferenceEquals(a1, a2) || IsShort(s1) || IsShort(s2) || isnull(s1) || isnull(s2))
                return false;
            return a1.VStringBuffer(s1) == a2.VStringBuffer(s2);
        }

        // ------------------------------------------------------------------------------------------------------------
        // release (the clear loop + NpyString_free_allocator)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The release of the buffer that owns this allocator: NumPy's clear loop over every cell of the buffer (which frees
        ///     the heap strings) followed by <c>NpyString_free_allocator</c> (the arena). Runs from the cells' memory block when
        ///     its last reference is released — possibly on the finalizer thread, which is why the allocator holds no managed
        ///     resource but its lock.
        /// </summary>
        /// <param name="cells">The first cell of the WHOLE buffer (not of a slice).</param>
        /// <param name="count">The number of cells in the buffer.</param>
        /// <remarks>
        ///     The arena strings are not zeroed one by one (NumPy's clear loop does, through <c>NpyString_free</c>): the arena is
        ///     freed whole right after, and the cells go back to the pool, which zeroes a buffer before a <c>StringDType</c>
        ///     allocation takes it again. The heap pages go back to the buffer pool whole too, so the loop over the cells runs
        ///     only while a heap text longer than <see cref="HeapPoolMaxBytes"/> (a C-allocator block of its own) is left — a
        ///     buffer that holds 100,000 heap texts of page size is released without reading its cells. Idempotent: a second call
        ///     does nothing.
        /// </remarks>
        internal void ClearAndFree(byte* cells, long count)
        {
            acquire_allocator();
            try
            {
                if (_freed)
                    return;
                _freed = true;
                // NumPy's clear loop frees every heap string. A page block goes back with its page (below), so the cells are
                // visited only while some text is a C-allocator block of its own, and only until the last one is found.
                for (long i = 0; i < count && _soloHeapStrings > 0; i++)
                {
                    byte* s = cells + i * SIZEOF_NPY_PACKED_STATIC_STRING;
                    if (OwnsHeapString(s))
                    {
                        byte* block = (byte*)*(ulong*)s - HeapBlockHeaderBytes;
                        if (*(HeapPage**)block == null)
                        {
                            NativeMemory.Free(block);
                            _soloHeapStrings--;
                        }
                    }
                }
                System.Diagnostics.Debug.Assert(_soloHeapStrings == 0,
                    "A heap text of its own was not in any cell of the buffer.");
                HeapStrings = 0;
                while (_heapPages != null)
                {
                    HeapPage* next = _heapPages->NextAll;
                    Unmanaged.Pooling.SizeBucketedBufferPool.Return((IntPtr)_heapPages, 1L << _heapPages->Shift);
                    _heapPages = next;
                }
                if (_heapSpare != null)
                {
                    Unmanaged.Pooling.SizeBucketedBufferPool.Return((IntPtr)_heapSpare, 1L << _heapSpare->Shift);
                    _heapSpare = null;
                }
                HeapPages = 0;
                _heapOpen = null;
                _heapShift = null;
                if (_arena != null)
                {
                    NativeMemory.Free(_arena);
                    _arena = null;
                }
                _arenaSize = _arenaCursor = 0;
                if (_reportedPressure > 0)
                {
                    Unmanaged.Pooling.SizeBucketedBufferPool.AccountForeignRelease(_reportedPressure);
                    _reportedPressure = 0;
                }
                _unreportedHeap = 0;
                Interlocked.Decrement(ref _liveAllocators);
            }
            finally
            {
                release_allocator();
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // GC memory pressure (the arena and heap strings are native memory the GC does not see)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Reports an arena growth through the buffer pool's coarse accounting
        ///     (<see cref="Unmanaged.Pooling.SizeBucketedBufferPool.AccountForeignAllocation"/>), so a process holding large
        ///     arenas still pushes the GC to collect dropped buffers, without a <see cref="GC.AddMemoryPressure"/> call per
        ///     growth (that drove constant gen-2 collections).
        /// </summary>
        /// <param name="bytes">The growth in bytes.</param>
        private void AccountArena(long bytes)
        {
            if (bytes <= 0)
                return;
            Unmanaged.Pooling.SizeBucketedBufferPool.AccountForeignAllocation(bytes);
            _reportedPressure += bytes;
        }

        /// <summary>Accumulates a heap allocation and reports it in batches through the pool's accounting (two interlocked
        ///     adds per report).</summary>
        /// <param name="bytes">The heap allocation size.</param>
        private void AccountHeap(long bytes)
        {
            _unreportedHeap += bytes;
            if (_unreportedHeap < PressureBatch)
                return;
            Unmanaged.Pooling.SizeBucketedBufferPool.AccountForeignAllocation(_unreportedHeap);
            _reportedPressure += _unreportedHeap;
            _unreportedHeap = 0;
        }
    }
}
