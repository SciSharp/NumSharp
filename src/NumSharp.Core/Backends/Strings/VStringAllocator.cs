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
    ///     <see cref="Missing"/> (an NA value), <see cref="Initialized"/> (written at least once),
    ///     <see cref="OutsideArena"/> (short or heap), <see cref="Long"/> (an arena string of more than 255 bytes, or a heap
    ///     string). A string of up to <see cref="ShortMaxSize"/> bytes lives IN the cell: bytes 0–14 are the text and the low
    ///     nibble of byte 15 is its length, with the flags <c>Initialized | OutsideArena</c>. An all-zero cell is an
    ///     uninitialized empty string, so a zero-filled buffer is a buffer of <c>''</c> for free (NumPy's
    ///     <c>NPY_NEEDS_INIT</c>).
    ///     </para>
    ///     <para>
    ///     <b>The arena.</b> A medium string (16–255 bytes) is stored after a one-byte size prefix, a long one after an
    ///     eight-byte prefix; the cell holds the offset of the TEXT (after the prefix). The arena grows by reallocation to
    ///     125 % of what it needs, and its new tail is zeroed. A write that fits the old allocation reuses it (the prefix
    ///     records the allocation's capacity); a longer one goes to the heap. A string set to NA keeps its allocation's
    ///     metadata (only the <see cref="Missing"/> bit is added), so a later write can reuse its arena space. After a short
    ///     write the arena information is gone, and a later long write always goes to the heap.
    ///     </para>
    ///     <para>
    ///     <b>Ownership.</b> One allocator per BUFFER (NumPy gives every array that owns data its own descriptor instance
    ///     and allocator, <c>stringdtype_finalize_descr</c>): views and slices of the buffer share it, a copy gets a new one.
    ///     The cells' memory block owns the allocator — its release runs <see cref="ClearAndFree"/> (NumPy's clear loop,
    ///     which frees the heap strings, then the arena), so the allocator lives exactly as long as the last view of its
    ///     cells. Cells are never valid with another buffer's allocator: an offset reads someone else's arena, and a heap
    ///     pointer copied into another buffer would be freed twice. Every movement of <c>StringDType</c> elements therefore
    ///     goes through <see cref="Dup"/>, never through a raw copy of the cells.
    ///     </para>
    ///     <para>
    ///     <b>Locking.</b> Like NumPy's allocator mutex: a caller takes the lock ONCE per operation
    ///     (<see cref="Acquire"/> / <see cref="Release"/>, or <see cref="AcquirePair"/> for two allocators), then calls the
    ///     per-cell members, which assume it is held. The allocator does not lock per element.
    ///     </para>
    ///     <para>
    ///     <b>Errors.</b> NumPy's members return -1 and the caller raises <c>MemoryError</c> with its own text; here the
    ///     members return the same codes (0 success, -1 failure, 1 = "is null" for <see cref="Load"/>) and the callers
    ///     throw <see cref="OutOfMemoryException"/> with NumPy's text. Every pointer the allocator hands out is
    ///     <see cref="NativeMemory"/> memory (NumPy's <c>PyMem_RawMalloc</c>).
    ///     </para>
    /// </remarks>
    internal sealed unsafe class VStringAllocator
    {
        // ------------------------------------------------------------------------------------------------------------
        // the packed cell (static_string.c, NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN, sizeof(size_t) == 8)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The size of one packed cell (<c>sizeof(npy_packed_static_string)</c>): the descriptor's itemsize.</summary>
        internal const int CellSize = 16;

        /// <summary><c>NPY_STRING_MISSING</c>: the cell is an NA value (its other flags keep the old allocation's metadata).</summary>
        internal const byte Missing = 0x80;

        /// <summary><c>NPY_STRING_INITIALIZED</c>: the element has been written at least once.</summary>
        internal const byte Initialized = 0x40;

        /// <summary><c>NPY_STRING_OUTSIDE_ARENA</c>: the text is in the cell (short) or on the heap (with <see cref="Long"/>).</summary>
        internal const byte OutsideArena = 0x20;

        /// <summary><c>NPY_STRING_LONG</c>: an arena string longer than 255 bytes (eight-byte prefix), or a heap string.</summary>
        internal const byte Long = 0x10;

        /// <summary><c>NPY_STRING_FLAG_MASK</c>: the high nibble of byte 15.</summary>
        internal const byte FlagMask = 0xF0;

        /// <summary><c>NPY_SHORT_STRING_SIZE_MASK</c>: the low nibble of byte 15, a short string's length.</summary>
        internal const byte ShortSizeMask = 0x0F;

        /// <summary><c>NPY_SHORT_STRING_MAX_SIZE</c>: the longest text a cell holds itself (15 bytes).</summary>
        internal const int ShortMaxSize = CellSize - 1;

        /// <summary><c>NPY_MEDIUM_STRING_MAX_SIZE</c>: the longest arena string with a one-byte size prefix.</summary>
        internal const int MediumMaxSize = 0xFF;

        /// <summary><c>NPY_MAX_STRING_SIZE</c>: one byte of the size field is the flags, so a string has at most 2^56 − 1 bytes.</summary>
        internal const long MaxStringSize = (1L << 56) - 1;

        /// <summary>The mask that keeps the 56 size bits of the size field (<c>~HIGH_BYTE_MASK</c>).</summary>
        private const ulong SizeMask = 0x00FF_FFFF_FFFF_FFFFUL;

        /// <summary>Heap bytes that are reported to the GC in batches of at least this many (see <see cref="AccountHeap"/>).</summary>
        private const long PressureBatch = 64 * 1024;

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

        /// <summary>The arena's capacity in bytes (diagnostic).</summary>
        internal long ArenaCapacity => (long)_arenaSize;

        /// <summary>The arena's used bytes (diagnostic).</summary>
        internal long ArenaUsed => (long)_arenaCursor;

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

        /// <summary>Acquires the allocator mutex (<c>NpyString_acquire_allocator</c>); pair it with exactly one <see cref="Release"/>.</summary>
        internal void Acquire() => Monitor.Enter(_lock);

        /// <summary>Releases the allocator mutex (<c>NpyString_release_allocator</c>).</summary>
        internal void Release() => Monitor.Exit(_lock);

        /// <summary>
        ///     Acquires two allocators' mutexes at once (<c>NpyString_acquire_allocators</c>): the same allocator twice is locked
        ///     once. Distinct allocators are locked in the order of their creation (<see cref="_id"/>, unique per allocator),
        ///     so two threads copying in opposite directions cannot deadlock (NumPy relies on the GIL for that).
        /// </summary>
        /// <param name="a">The first allocator.</param>
        /// <param name="b">The second allocator (may be <paramref name="a"/>).</param>
        internal static void AcquirePair(VStringAllocator a, VStringAllocator b)
        {
            if (ReferenceEquals(a, b))
            {
                a.Acquire();
                return;
            }
            var (first, second) = a._id < b._id ? (a, b) : (b, a);
            first.Acquire();
            second.Acquire();
        }

        /// <summary>A unique, increasing number per allocator: the lock order of <see cref="AcquirePair"/>.</summary>
        private readonly long _id = Interlocked.Increment(ref _nextId);

        /// <summary>The last <see cref="_id"/> handed out.</summary>
        private static long _nextId;

        /// <summary>Releases what <see cref="AcquirePair"/> acquired (<c>NpyString_release_allocators</c>).</summary>
        /// <param name="a">The first allocator.</param>
        /// <param name="b">The second allocator (may be <paramref name="a"/>).</param>
        internal static void ReleasePair(VStringAllocator a, VStringAllocator b)
        {
            a.Release();
            if (!ReferenceEquals(a, b))
                b.Release();
        }

        /// <summary>
        ///     Acquires the mutexes of several allocators (<c>NpyString_acquire_allocators</c> for any count): the list is sorted
        ///     into lock order (<see cref="_id"/>) and each distinct allocator is locked once. Pair it with
        ///     <see cref="ReleaseAll"/> on the SAME list, which this method leaves sorted.
        /// </summary>
        /// <param name="allocators">The allocators (duplicates allowed); sorted in place.</param>
        /// <exception cref="ArgumentNullException"><paramref name="allocators"/> or one of its entries is null.</exception>
        internal static void AcquireAll(System.Collections.Generic.List<VStringAllocator> allocators)
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
                    a.Acquire();
                previous = a;
            }
        }

        /// <summary>Releases what <see cref="AcquireAll"/> acquired, given the same (now sorted) list.</summary>
        /// <param name="allocators">The list passed to <see cref="AcquireAll"/>.</param>
        internal static void ReleaseAll(System.Collections.Generic.List<VStringAllocator> allocators)
        {
            VStringAllocator previous = null;
            foreach (var a in allocators)
            {
                if (!ReferenceEquals(a, previous))
                    a.Release();
                previous = a;
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // cell inspection (no allocator state)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The flags of a cell (<c>STRING_FLAGS</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>The high nibble of byte 15.</returns>
        internal static byte Flags(byte* s) => (byte)(s[15] & FlagMask);

        /// <summary>Whether a cell holds its text itself (<c>is_short_string</c>: initialized and outside the arena, not missing, not long).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>True for a short string.</returns>
        internal static bool IsShort(byte* s) => Flags(s) == (Initialized | OutsideArena);

        /// <summary>Whether a cell is an NA value (<c>NpyString_isnull</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>True when the <see cref="Missing"/> flag is set.</returns>
        internal static bool IsNull(byte* s) => (s[15] & Missing) == Missing;

        /// <summary>The size field of a vstring cell without its flags byte (<c>VSTRING_SIZE</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>The size in bytes.</returns>
        private static ulong VStringSize(byte* s) => *(ulong*)(s + 8) & SizeMask;

        /// <summary>Sets a vstring cell's size and keeps its flags byte (<c>set_vstring_size</c>).</summary>
        /// <param name="s">The cell.</param>
        /// <param name="size">The new size (at most <see cref="MaxStringSize"/>).</param>
        private static void SetVStringSize(byte* s, ulong size)
        {
            byte flags = s[15];
            *(ulong*)(s + 8) = size;
            s[15] = flags;
        }

        /// <summary>The text length of a cell in bytes (<c>NpyString_size</c>): 0 for an NA value.</summary>
        /// <param name="s">The cell.</param>
        /// <returns>The UTF-8 byte count.</returns>
        internal static long Size(byte* s)
        {
            if (IsNull(s))
                return 0;
            if (IsShort(s))
                return s[15] & ShortSizeMask;
            return (long)VStringSize(s);
        }

        /// <summary>Whether a cell is a heap string that owns memory (<c>OutsideArena</c> and not short, with a size).</summary>
        /// <param name="s">The cell.</param>
        /// <returns>True when freeing the cell frees heap memory.</returns>
        internal static bool OwnsHeapString(byte* s) => (s[15] & OutsideArena) != 0 && !IsShort(s) && VStringSize(s) > 0;

        /// <summary>
        ///     The arena bytes a duplicate of this cell appends to a destination arena whose cells hold no arena strings (so no
        ///     slot can be reused): the text and its size prefix for a medium or long string (an arena or heap string of the
        ///     source — <c>NpyString_newsize</c> puts both in the arena of a fresh cell), nothing for a short, empty or NA cell.
        /// </summary>
        /// <param name="s">The source cell.</param>
        /// <returns>The bytes <see cref="ArenaMalloc"/> would append for it.</returns>
        internal static long ArenaBytesForDuplicate(byte* s)
        {
            if (IsNull(s) || IsShort(s))
                return 0;
            ulong size = VStringSize(s);
            return size == 0 ? 0 : (long)(size <= MediumMaxSize ? size + 1 : size + 8);
        }

        /// <summary>
        ///     The arena bytes packing a text of <paramref name="size"/> bytes into a fresh cell appends (nothing for a short
        ///     text).
        /// </summary>
        /// <param name="size">The UTF-8 size.</param>
        /// <returns>The bytes, prefix included.</returns>
        internal static long ArenaBytesForText(long size)
            => size <= ShortMaxSize ? 0 : size <= MediumMaxSize ? size + 1 : size + 8;

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
        ///     <c>NativeMemory.Realloc</c> with a null pointer is an allocation, as <c>realloc(NULL, n)</c> is, so the first
        ///     growth creates the arena. The space past the cursor must read as zero, as NumPy's <c>memset</c> leaves it: a
        ///     realloc returns the new tail uninitialized. The lock must be held.
        /// </remarks>
        private bool TryGrowArena(ulong newsize)
        {
            byte* newbuf;
            try
            {
                newbuf = (byte*)NativeMemory.Realloc(_arena, (nuint)newsize);
            }
            catch (OutOfMemoryException)
            {
                return false;
            }
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
        /// <returns>The first text byte, or null.</returns>
        private byte* VStringBuffer(byte* s)
        {
            if ((s[15] & OutsideArena) != 0)
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
            ulong storage = size <= MediumMaxSize ? size + 1 : size + 8;
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
            if (size <= MediumMaxSize)
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
        internal int Load(byte* packed, out byte* buf, out long size)
        {
            if (IsNull(packed))
            {
                size = 0;
                buf = null;
                return 1;
            }
            if (IsShort(packed))
            {
                size = packed[15] & ShortSizeMask;
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

        /// <summary>A static empty text (NumPy's <c>EMPTY_STRING</c>) for empty vstring cells.</summary>
        private static readonly byte* EmptyString = (byte*)NativeMemory.AllocZeroed(1);

        /// <summary>
        ///     The allocation for a new string of <paramref name="size"/> bytes in a cell (<c>heap_or_arena_allocate</c>): a cell
        ///     never allocated appends to the arena; an arena cell reuses its old allocation when the text fits its capacity
        ///     (read from the size prefix); anything else allocates on the heap.
        /// </summary>
        /// <param name="toInit">The cell (its flags byte is updated).</param>
        /// <param name="size">The text size (more than <see cref="ShortMaxSize"/>).</param>
        /// <param name="onHeap">Receives true for a heap allocation (the cell then stores the pointer, not an offset).</param>
        /// <returns>The first text byte, or null on failure.</returns>
        private byte* HeapOrArenaAllocate(byte* toInit, ulong size, out bool onHeap)
        {
            onHeap = false;
            byte flags = toInit[15];
            if ((flags & OutsideArena) == 0)
            {
                // Arena allocation or re-allocation.
                if (flags == 0)
                {
                    // string isn't previously allocated, so add to existing arena allocation
                    byte* ret = ArenaMalloc(size);
                    toInit[15] = size <= MediumMaxSize ? Initialized : (byte)(Initialized | Long);
                    return ret;
                }
                // String was in arena. See if there is still space.
                // The size is stored "behind" the beginning of the allocation.
                byte* buf = VStringBuffer(toInit);
                if (buf == null)
                    return null;
                ulong allocSize = (flags & Long) != 0
                    ? Unsafe.ReadUnaligned<ulong>(buf - 8) // long string size not necessarily memory-aligned
                    : *(buf - 1);                          // medium string size is stored in a char
                if (size <= allocSize)
                    return buf; // we have room!
                // No room, fall through to heap allocation.
            }
            // Heap allocate
            onHeap = true;
            toInit[15] = Initialized | OutsideArena | Long;
            // A failure scheduled by a test (FailAllocationAfter): the same null an exhausted heap gives.
            if (InjectedFailure())
                return null;
            byte* heap;
            try
            {
                heap = (byte*)NativeMemory.Alloc((nuint)size);
            }
            catch (OutOfMemoryException)
            {
                return null;
            }
            HeapStrings++;
            AccountHeap((long)size);
            return heap;
        }

        /// <summary>
        ///     Releases a vstring cell's text (<c>heap_or_arena_deallocate</c>): frees a heap string (and clears its pointer),
        ///     zeroes an arena string's text.
        /// </summary>
        /// <param name="s">A vstring cell with a non-zero size.</param>
        /// <returns>0, or -1 for an arena string outside the arena.</returns>
        private int HeapOrArenaDeallocate(byte* s)
        {
            if ((s[15] & OutsideArena) != 0)
            {
                // It's a heap string (not in the arena buffer) so it needs to be deallocated with free(). For heap strings
                // the offset is a raw address so this cast is safe.
                NativeMemory.Free((void*)*(ulong*)s);
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
        internal static void PackEmpty(byte* s)
        {
            if ((s[15] & OutsideArena) != 0)
                s[15] = Initialized | OutsideArena; // This also sets short string size to 0.
            else
                SetVStringSize(s, 0);
        }

        /// <summary>
        ///     Prepares a cell for a new text of <paramref name="size"/> bytes without writing it (<c>NpyString_newemptysize</c>):
        ///     the empty string, a short string's length, or an arena / heap allocation whose offset or pointer the cell stores.
        /// </summary>
        /// <param name="size">The text size.</param>
        /// <param name="s">The cell (already freed with <see cref="Free"/>, or never written).</param>
        /// <returns>0, or -1 when the size exceeds <see cref="MaxStringSize"/> or the memory is exhausted.</returns>
        internal int NewEmptySize(long size, byte* s)
        {
            if (size == 0)
            {
                PackEmpty(s);
                return 0;
            }
            if ((ulong)size > MaxStringSize)
                return -1;
            if (size > ShortMaxSize)
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
                s[15] = (byte)(Initialized | OutsideArena | size);
            }
            return 0;
        }

        /// <summary>
        ///     Allocates a cell's new text and copies <paramref name="size"/> bytes of <paramref name="init"/> into it
        ///     (<c>NpyString_newsize</c>).
        /// </summary>
        /// <param name="init">The UTF-8 text (not inside this allocator's arena: see <see cref="Dup"/>).</param>
        /// <param name="size">The text size.</param>
        /// <param name="toInit">The cell (already freed, or never written).</param>
        /// <returns>0, or -1 on failure.</returns>
        internal int NewSize(byte* init, long size, byte* toInit)
        {
            if (NewEmptySize(size, toInit) < 0)
                return -1;
            if (size == 0)
                return 0;
            byte* buf = size > ShortMaxSize ? VStringBuffer(toInit) : toInit;
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
        internal int Free(byte* s)
        {
            // Unconditionally remove flag indicating something was missing. For that case, the string should have been
            // deallocated already, but test anyway in case we later implement the option to mask the string.
            s[15] = (byte)(s[15] & ~Missing);
            if (IsShort(s))
            {
                if ((s[15] & ShortSizeMask) > 0)
                {
                    // zero buffer and set flags for initialized out-of-arena empty string.
                    new Span<byte>(s, CellSize).Clear();
                    s[15] = OutsideArena | Initialized;
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
        internal int PackNull(byte* s)
        {
            if (Free(s) < 0)
                return -1;
            // preserve the flags because we allow mutation, so we need metadata about the original allocation associated
            // with this string in order to determine if there is space in the arena allocation for a new string after a
            // user mutates this one to a non-NULL value.
            s[15] |= Missing;
            return 0;
        }

        /// <summary>
        ///     Packs a UTF-8 text into a cell (<c>NpyString_pack</c>): frees the old text, then allocates and copies.
        /// </summary>
        /// <param name="s">The cell.</param>
        /// <param name="buf">The UTF-8 text (not inside this allocator's arena).</param>
        /// <param name="size">Its size in bytes.</param>
        /// <returns>0, or -1 on failure.</returns>
        internal int Pack(byte* s, byte* buf, long size)
        {
            if (Free(s) < 0)
                return -1;
            return NewSize(buf, size, s);
        }

        /// <summary>
        ///     Copies the string of one cell into another (<c>NpyString_dup</c>), across allocators or within one: an NA stays
        ///     NA, a short string is copied as the cell, an arena / heap string is packed into <paramref name="outAllocator"/>
        ///     (through a temporary copy when both cells share an allocator, because the new allocation can move the arena).
        /// </summary>
        /// <param name="input">The source cell.</param>
        /// <param name="output">The destination cell (freed with <see cref="Free"/>, or never written).</param>
        /// <param name="inAllocator">The source's allocator (locked).</param>
        /// <param name="outAllocator">The destination's allocator (locked; may be <paramref name="inAllocator"/>).</param>
        /// <returns>0, or -1 on failure.</returns>
        internal static int Dup(byte* input, byte* output, VStringAllocator inAllocator, VStringAllocator outAllocator)
        {
            if (IsNull(input))
                return outAllocator.PackNull(output);
            long size = Size(input);
            if (size == 0)
            {
                PackEmpty(output);
                return 0;
            }
            if (IsShort(input))
            {
                Buffer.MemoryCopy(input, output, CellSize, CellSize);
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
            int ret = outAllocator.NewSize(inBuf, size, output);
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
            if (outAllocator.Free(output) < 0)
                throw new OutOfMemoryException($"Failed to deallocate string in {location}");
            if (Dup(input, output, inAllocator, outAllocator) < 0)
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
        internal static bool ShareMemory(byte* s1, VStringAllocator a1, byte* s2, VStringAllocator a2)
        {
            if (!ReferenceEquals(a1, a2) || IsShort(s1) || IsShort(s2) || IsNull(s1) || IsNull(s2))
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
        ///     allocation takes it again. Idempotent: a second call does nothing.
        /// </remarks>
        internal void ClearAndFree(byte* cells, long count)
        {
            Acquire();
            try
            {
                if (_freed)
                    return;
                _freed = true;
                for (long i = 0; i < count; i++)
                {
                    byte* s = cells + i * CellSize;
                    if (OwnsHeapString(s))
                    {
                        NativeMemory.Free((void*)*(ulong*)s);
                        HeapStrings--;
                    }
                }
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
                Release();
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
