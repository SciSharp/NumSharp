using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The storage of NEP 55 <c>StringDType</c> arrays (strings phase 1, P1.3): NumPy's 16-byte packed cell, the per-buffer
    ///     allocator (<see cref="VStringAllocator"/>: arena, heap strings, NA metadata), the slice that carries both
    ///     (<see cref="VStringArraySlice"/>), and the lifetime rule — the cells' memory block owns the allocator, so views keep it
    ///     alive and the last release frees every heap string and the arena. Every cell layout asserted here was read from
    ///     <c>numpy==2.4.2</c>'s <c>tobytes()</c> for the same writes.
    /// </summary>
    [TestClass]
    public class StringDTypeStorageTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor (no NA object, coercing).</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor (C# null is the NA object).</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>A fresh <c>StringDType(coerce=False)</c> descriptor (only text may be stored).</summary>
        private static DType TC => np.dtypes.StringDType(coerce: false);

        /// <summary>The cell of element <paramref name="i"/> as 16 bytes (a copy, from <c>tobytes()</c>).</summary>
        /// <param name="a">A C-contiguous <c>StringDType</c> array.</param>
        /// <param name="i">The flat element index.</param>
        /// <returns>The 16 cell bytes.</returns>
        private static byte[] Cell(NDArray a, int i) => a.tobytes().Skip(16 * i).Take(16).ToArray();

        /// <summary>The 64-bit little-endian word at <paramref name="at"/> of a cell.</summary>
        /// <param name="cell">The 16 cell bytes.</param>
        /// <param name="at">The byte position (0 for the offset / pointer word, 8 for the size-and-flags word).</param>
        /// <returns>The word.</returns>
        private static ulong Word(byte[] cell, int at) => BitConverter.ToUInt64(cell, at);

        // ------------------------------------------------------------------------------------------------------------
        // the packed cell (static_string.c), as numpy==2.4.2 writes it
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A short string lives in the cell (text, then the flags byte <c>INITIALIZED | OUTSIDE_ARENA</c> with the length in
        ///     its low nibble); a medium one (16-255 bytes) in the arena after a one-byte prefix (offset 1, flags
        ///     <c>INITIALIZED</c>); a long one after an eight-byte prefix (offset 1 + 20 + 8 = 29, flags
        ///     <c>INITIALIZED | LONG</c>); the empty string written into a fresh cell leaves it all-zero.
        /// </summary>
        [TestMethod]
        public void Cell_ShortMediumLongEmpty_AreNumPysBytes()
        {
            using var a = np.array(new[] { "ab", new string('x', 20), new string('y', 300), "" }, T);
            Convert.ToHexString(Cell(a, 0)).Should().Be("61620000000000000000000000000062");
            Convert.ToHexString(Cell(a, 1)).Should().Be("01000000000000001400000000000040");
            Convert.ToHexString(Cell(a, 2)).Should().Be("1D000000000000002C01000000000050");
            Convert.ToHexString(Cell(a, 3)).Should().Be("00000000000000000000000000000000");
            a.itemsize.Should().Be(16);
            a.nbytes.Should().Be(64);
            a.Storage.ItemBytes.Should().Be(16);
        }

        /// <summary>
        ///     The write state machine: an NA write keeps the arena slot's offset (flags <c>MISSING | INITIALIZED</c>, size 0)
        ///     and a later write of no more bytes reuses it; a longer rewrite moves to the HEAP (flags
        ///     <c>INITIALIZED | OUTSIDE_ARENA | LONG</c>) and a shorter heap rewrite stays there; a short write wipes the arena
        ///     metadata, so a later medium write goes to the heap; the empty string keeps the slot (size 0) for reuse.
        /// </summary>
        [TestMethod]
        public void Cell_WriteStateMachine_IsNumPys()
        {
            using (var a = np.array(new object[] { new string('q', 20), "a" }, TN))
            {
                a.SetValue(null, 0);
                Convert.ToHexString(Cell(a, 0)).Should().Be("010000000000000000000000000000C0");
                a.SetValue(new string('z', 19), 0);
                Convert.ToHexString(Cell(a, 0)).Should().Be("01000000000000001300000000000040");
            }
            using (var a = np.array(new[] { new string('q', 20), "a" }, T))
            {
                a.SetValue(new string('s', 40), 0);
                var heap = Cell(a, 0);
                heap[15].Should().Be(0x70);
                (Word(heap, 8) & 0x00FF_FFFF_FFFF_FFFFUL).Should().Be(40UL);
                a.Storage.VStringSlice.Allocator.HeapStrings.Should().Be(1);
                a.SetValue(new string('t', 30), 0);
                var shorter = Cell(a, 0);
                shorter[15].Should().Be(0x70, "a shorter rewrite of a heap string stays on the heap");
                (Word(shorter, 8) & 0x00FF_FFFF_FFFF_FFFFUL).Should().Be(30UL);
                a.SetValue("u", 0);
                Convert.ToHexString(Cell(a, 0)).Should().Be("75000000000000000000000000000061");
                a.Storage.VStringSlice.Allocator.HeapStrings.Should().Be(0, "the short write freed the heap string");
            }
            using (var a = np.array(new[] { new string('q', 20), "a" }, T))
            {
                a.SetValue("x", 0);
                a.SetValue(new string('y', 18), 0);
                Cell(a, 0)[15].Should().Be(0x70, "a short write forgets the arena slot: the medium write goes to the heap");
            }
            using (var a = np.array(new[] { new string('q', 20), "a" }, T))
            {
                a.SetValue("", 0);
                Convert.ToHexString(Cell(a, 0)).Should().Be("01000000000000000000000000000040");
                a.SetValue(new string('w', 16), 0);
                Convert.ToHexString(Cell(a, 0)).Should().Be("01000000000000001000000000000040");
            }
        }

        /// <summary>
        ///     <c>DTypeSize</c> — the element width every raw byte route reads — refuses a <c>StringDType</c> storage loudly, so
        ///     a route without its <c>StringDType</c> form can never copy cells; the byte arithmetic reads <c>ItemBytes</c>.
        /// </summary>
        [TestMethod]
        public void DTypeSize_RefusesStringDType_ItemBytesIsTheCell()
        {
            using var a = np.array(new[] { "a", "b" }, T);
            Action raw = () => { _ = a.Storage.DTypeSize; };
            raw.Should().Throw<NotSupportedException>().WithMessage("*StringDType()*");
            Action wrapper = () => { _ = a.dtypesize; };
            wrapper.Should().Throw<NotSupportedException>();
            a.Storage.ItemBytes.Should().Be(16);
            a.strides.Should().Equal(16L);
            using var m = np.array(new[] { "a", "b", "c", "d" }, T).reshape(2, 2);
            m.strides.Should().Equal(32L, 16L);
        }

        // ------------------------------------------------------------------------------------------------------------
        // VStringArraySlice: the IArraySlice surface
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The slice's constructor and allocation refuse what is not a whole buffer of <c>StringDType</c> cells.</summary>
        [TestMethod]
        public void Slice_ConstructionChecks()
        {
            var allocator = new VStringAllocator();
            using var bytes = new NDArray(NPTypeCode.Byte, new Shape(32L), true);
            var cells = bytes.Storage.GetData<byte>();
            Action noAllocator = () => _ = new VStringArraySlice(cells, null, T);
            noAllocator.Should().Throw<ArgumentNullException>();
            Action noDescr = () => _ = new VStringArraySlice(cells, allocator, null);
            noDescr.Should().Throw<ArgumentNullException>();
            Action notT = () => _ = new VStringArraySlice(cells, allocator, np.dtype("U3"));
            notT.Should().Throw<ArgumentException>().WithMessage("*is not a StringDType descriptor*");
            Action partial = () => _ = new VStringArraySlice(cells.Slice(0, 20), allocator, T);
            partial.Should().Throw<ArgumentException>().WithMessage("A buffer of 20 bytes is not a whole number of 16-byte cells.*");
            Action allocNull = () => VStringArraySlice.Allocate(null, 1);
            allocNull.Should().Throw<ArgumentNullException>();
            Action allocNotT = () => VStringArraySlice.Allocate(np.dtype("f8"), 1);
            allocNotT.Should().Throw<ArgumentException>();
            new VStringArraySlice(cells, allocator, T).Count.Should().Be(2);
        }

        /// <summary>
        ///     The slice surface: sizes, the type code, element access (the NA object back for an NA cell), the typed accessors'
        ///     refusals (a cell is text, not a number), <c>Fill</c> (the value converted even with no elements), sub-slices that
        ///     share the allocator, <c>ToArray</c> (strings, or objects when there is an NA object), enumeration and the raw-byte
        ///     views, which are the only typed views a cell buffer offers.
        /// </summary>
        [TestMethod]
        public unsafe void Slice_Surface()
        {
            var s = VStringArraySlice.Allocate(TN, 3);
            try
            {
                s.ItemLength.Should().Be(16);
                s.BytesLength.Should().Be(48);
                s.TypeCode.Should().Be(NPTypeCode.String);
                s.HasNAObject.Should().BeTrue();
                ((IMemoryBlock)s).ItemLength.Should().Be(16);
                ((IntPtr)((IMemoryBlock)s).Address).Should().Be((IntPtr)s.Address);
                s.MemoryBlock.Should().NotBeNull();
                s.ToString().Should().Be("VStringArraySlice(StringDType(na_object=None), Count=3)");

                s[0] = "ab";
                s[1] = null;
                s.SetIndex(2, (object)'c'); // a C# char is a Python str of one character
                s[0].Should().Be("ab");
                s.GetIndex(1).Should().BeNull("an NA cell reads back as the NA object");
                s[2].Should().Be("c");
                s.Cast<object>().Should().Equal(new object[] { "ab", null, "c" });
                s.ToArray().Should().BeOfType<object[]>().Which.Should().Equal(new object[] { "ab", null, "c" });

                Action typedGet = () => s.GetIndex<int>(0);
                typedGet.Should().Throw<NotSupportedException>().WithMessage("*element is text*");
                Action typedSet = () => s.SetIndex<int>(0, 5);
                typedSet.Should().Throw<NotSupportedException>();
                Action typedClone = () => s.Clone<int>();
                typedClone.Should().Throw<NotSupportedException>().WithMessage("*refer to its own allocator*");
                Action intSpan = () => s.AsSpan<int>();
                intSpan.Should().Throw<NotSupportedException>().WithMessage("*only as raw bytes*");

                // the raw byte views: the cells themselves
                var span = s.AsSpan<byte>();
                span.Length.Should().Be(48);
                span[0].Should().Be((byte)'a');
                ref byte first = ref s.GetPinnableReference<byte>();
                first.Should().Be((byte)'a');
                var copy = new byte[48];
                s.CopyTo<byte>(copy.AsSpan());
                copy[0].Should().Be((byte)'a');
                Action shortCopy = () => s.CopyTo<byte>(new byte[47].AsSpan());
                shortCopy.Should().Throw<ArgumentException>().WithMessage("Destination was too short.");
                using (var target = new NDArray(NPTypeCode.Byte, new Shape(48L), true))
                {
                    s.CopyTo(((IArraySlice)target.Storage.GetData<byte>()).AsSpan<byte>());
                    target.GetByte(1).Should().Be((byte)'b');
                }

                // sub-slices share the cells and the allocator
                var tail = (VStringArraySlice)s.Slice(1);
                tail.Count.Should().Be(2);
                tail.Allocator.Should().BeSameAs(s.Allocator);
                tail[1].Should().Be("c");
                var mid = (VStringArraySlice)s.Slice(1, 1);
                mid.Count.Should().Be(1);
                mid.GetIndex(0).Should().BeNull();

                s.IsReleased.Should().BeFalse();
                s.IsUniquelyReferenced.Should().BeTrue();
            }
            finally
            {
                s.DangerousFree();
            }
        }

        /// <summary>
        ///     <c>ToArray</c> of a descriptor without an NA object is a <see cref="string"/> array; <c>Fill</c> converts the value
        ///     once (refused under <c>coerce=False</c> even for an empty buffer, as NumPy packs before it looks at the size) and
        ///     packs a long text into every element separately.
        /// </summary>
        [TestMethod]
        public void Slice_ToArray_And_Fill()
        {
            var s = VStringArraySlice.Allocate(T, 3);
            try
            {
                s.Fill(new string('f', 30));
                s.ToArray().Should().BeOfType<string[]>().Which.Should().AllBe(new string('f', 30));
                s.Allocator.ArenaUsed.Should().Be(3 * 31, "each element gets its own arena allocation (1-byte prefix + 30)");
                s.Fill(5);
                s.ToArray().Should().BeOfType<string[]>().Which.Should().AllBe("5");
            }
            finally
            {
                s.DangerousFree();
            }

            var empty = VStringArraySlice.Allocate(TC, 0);
            try
            {
                Action fill = () => empty.Fill(5);
                fill.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
                empty.Fill("ok"); // a text needs no conversion: nothing to write
            }
            finally
            {
                empty.DangerousFree();
            }
        }

        /// <summary>
        ///     <c>Clone</c> duplicates every string into a new buffer with its own allocator (arena strings copied, heap strings
        ///     re-allocated, NA cells kept), so writes to one never reach the other.
        /// </summary>
        [TestMethod]
        public void Slice_Clone_IsIndependent()
        {
            var s = VStringArraySlice.Allocate(TN, 3);
            var c = (VStringArraySlice)null;
            try
            {
                s[0] = new string('a', 20);
                s[1] = null;
                s[2] = new string('b', 40);
                s[2] = new string('c', 50); // a heap string
                c = (VStringArraySlice)s.Clone();
                c.Allocator.Should().NotBeSameAs(s.Allocator);
                c.ToArray().Should().BeOfType<object[]>().Which.Should().Equal(new object[] { new string('a', 20), null, new string('c', 50) });
                c[0] = "changed";
                s[0].Should().Be(new string('a', 20));
                ((ICloneable)s).Clone().Should().BeOfType<VStringArraySlice>().Which.DangerousFree();
            }
            finally
            {
                c?.DangerousFree();
                s.DangerousFree();
            }
        }

        /// <summary>
        ///     A cell that claims an arena string while the buffer has no arena (a corrupt cell) is NumPy's
        ///     <c>MemoryError: Failed to load string in StringDType getitem</c>, never a read through a null pointer.
        /// </summary>
        [TestMethod]
        public unsafe void Slice_CorruptArenaCell_IsTheLoadError()
        {
            var s = VStringArraySlice.Allocate(T, 1);
            try
            {
                // offset 1, size 20, flags INITIALIZED: an arena string, in a buffer whose arena was never allocated
                *(ulong*)s.Address = 1;
                *(ulong*)(s.Address + 8) = 20UL | (0x40UL << 56);
                Action read = () => s.GetIndex(0);
                read.Should().Throw<OutOfMemoryException>().WithMessage("Failed to load string in StringDType getitem");
            }
            finally
            {
                new Span<byte>(s.Address, 16).Clear(); // a valid cell again, so the clear loop has nothing to free
                s.DangerousFree();
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // the allocator API (static_string.c)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>Load</c> / <c>Size</c> of every cell kind, <c>Pack</c> of sizes around both boundaries, the arena's growth to
        ///     125 % of its need, NA (size 0, load result 1), and <c>NewEmptySize</c>'s refusal of a size above
        ///     <c>MaxStringSize</c> (NumPy's -1, which callers turn into a MemoryError).
        /// </summary>
        [TestMethod]
        public unsafe void Allocator_PackLoadSize()
        {
            var s = VStringArraySlice.Allocate(T, 6);
            try
            {
                var al = s.Allocator;
                byte* c0 = s.Address;
                al.Acquire();
                try
                {
                    int[] sizes = { 0, 15, 16, 255, 256, 1000 };
                    for (int i = 0; i < sizes.Length; i++)
                    {
                        var text = new byte[sizes[i]];
                        Array.Fill(text, (byte)('a' + i));
                        fixed (byte* t = text)
                            al.Pack(c0 + 16 * i, t, text.Length).Should().Be(0);
                    }
                    for (int i = 0; i < sizes.Length; i++)
                    {
                        al.Load(c0 + 16 * i, out byte* buf, out long size).Should().Be(0);
                        size.Should().Be(sizes[i]);
                        VStringAllocator.Size(c0 + 16 * i).Should().Be(sizes[i]);
                        if (size > 0)
                            buf[size - 1].Should().Be((byte)('a' + i));
                    }
                    VStringAllocator.IsShort(c0 + 16).Should().BeTrue();
                    VStringAllocator.IsShort(c0 + 32).Should().BeFalse();
                    VStringAllocator.Flags(c0 + 64).Should().Be((byte)(VStringAllocator.Initialized | VStringAllocator.Long));
                    // 16+1, 255+1, 256+8, 1000+8 bytes appended; the arena grew to 125 % of each need
                    al.ArenaUsed.Should().Be(17 + 256 + 264 + 1008);
                    // arena_malloc's growth: each time the arena is too small it becomes minsize + minsize / 4, with minsize the
                    // cursor plus the new allocation: 17 -> 21, 273 -> 341, 537 -> 671, 1545 -> 1931.
                    al.ArenaCapacity.Should().Be(1931);

                    al.PackNull(c0).Should().Be(0);
                    VStringAllocator.IsNull(c0).Should().BeTrue();
                    VStringAllocator.Size(c0).Should().Be(0);
                    al.Load(c0, out byte* nb, out long ns).Should().Be(1);
                    ((IntPtr)nb).Should().Be(IntPtr.Zero);
                    ns.Should().Be(0);

                    al.NewEmptySize(VStringAllocator.MaxStringSize + 1, c0 + 16).Should().Be(-1, "past MaxStringSize");
                }
                finally
                {
                    al.Release();
                }
            }
            finally
            {
                s.DangerousFree();
            }
        }

        /// <summary>
        ///     A size the arena cannot grow to (just under <c>MaxStringSize</c>) is NumPy's allocation failure (-1, so the caller
        ///     raises the MemoryError), not an exception from the reallocation.
        /// </summary>
        [TestMethod]
        public unsafe void Allocator_ArenaGrowthFailure_IsMinusOne()
        {
            var s = VStringArraySlice.Allocate(T, 1);
            try
            {
                var al = s.Allocator;
                al.Acquire();
                try
                {
                    al.NewEmptySize(VStringAllocator.MaxStringSize, s.Address).Should().Be(-1);
                    VStringAllocator.Size(s.Address).Should().Be(0, "the failed allocation leaves the cell empty");
                }
                finally
                {
                    al.Release();
                }
            }
            finally
            {
                s.DangerousFree();
            }
        }

        /// <summary>
        ///     <c>Dup</c> between two allocators and within one (through a temporary, because the new allocation can move the
        ///     arena), <c>ShareMemory</c> (only two vstring cells of one allocator with one buffer), and <c>FreeAndCopy</c>'s
        ///     MemoryError texts: a corrupt destination cell (an arena string outside the arena) is "Failed to deallocate".
        /// </summary>
        [TestMethod]
        public unsafe void Allocator_DupShareFree()
        {
            var a = VStringArraySlice.Allocate(T, 4);
            var b = VStringArraySlice.Allocate(T, 2);
            try
            {
                a[0] = new string('m', 30);
                a[1] = "short";
                VStringAllocator.AcquirePair(a.Allocator, b.Allocator);
                try
                {
                    VStringAllocator.Dup(a.Address, b.Address, a.Allocator, b.Allocator).Should().Be(0);
                    VStringAllocator.Dup(a.Address + 16, b.Address + 16, a.Allocator, b.Allocator).Should().Be(0);
                    VStringAllocator.Dup(a.Address, a.Address + 32, a.Allocator, a.Allocator).Should().Be(0, "a duplicate within one arena");
                    VStringAllocator.ShareMemory(a.Address, a.Allocator, a.Address, a.Allocator).Should().BeTrue();
                    VStringAllocator.ShareMemory(a.Address, a.Allocator, a.Address + 32, a.Allocator).Should().BeFalse();
                    VStringAllocator.ShareMemory(a.Address, a.Allocator, b.Address, b.Allocator).Should().BeFalse();
                    VStringAllocator.ShareMemory(a.Address + 16, a.Allocator, a.Address + 16, a.Allocator).Should().BeFalse("a short cell owns no buffer");
                }
                finally
                {
                    VStringAllocator.ReleasePair(a.Allocator, b.Allocator);
                }
                b[0].Should().Be(new string('m', 30));
                b[1].Should().Be("short");
                a[2].Should().Be(new string('m', 30));

                // a corrupt destination: an arena string whose text lies far past the arena
                *(ulong*)(a.Address + 48) = 1UL << 40;
                *(ulong*)(a.Address + 56) = 20UL | (0x40UL << 56);
                Action copy = () =>
                {
                    VStringAllocator.AcquirePair(a.Allocator, b.Allocator);
                    try
                    {
                        VStringAllocator.FreeAndCopy(b.Allocator, a.Allocator, b.Address, a.Address + 48, "string to string cast");
                    }
                    finally
                    {
                        VStringAllocator.ReleasePair(a.Allocator, b.Allocator);
                    }
                };
                copy.Should().Throw<OutOfMemoryException>().WithMessage("Failed to deallocate string in string to string cast");
                new Span<byte>(a.Address + 48, 16).Clear();
            }
            finally
            {
                a.DangerousFree();
                b.DangerousFree();
            }
        }

        /// <summary>
        ///     <c>AcquireAll</c> locks a set of allocators in one global order (each once, duplicates skipped) so two operations
        ///     over the same allocators never deadlock; a null list or a null entry is refused.
        /// </summary>
        [TestMethod]
        public void Allocator_AcquireAll()
        {
            var x = new VStringAllocator();
            var y = new VStringAllocator();
            var list = new List<VStringAllocator> { y, x, y };
            VStringAllocator.AcquireAll(list);
            try
            {
                System.Threading.Monitor.IsEntered(GetLock(x)).Should().BeTrue();
                System.Threading.Monitor.IsEntered(GetLock(y)).Should().BeTrue();
            }
            finally
            {
                VStringAllocator.ReleaseAll(list);
            }
            System.Threading.Monitor.IsEntered(GetLock(x)).Should().BeFalse();
            Action nullList = () => VStringAllocator.AcquireAll(null);
            nullList.Should().Throw<ArgumentNullException>();
            Action nullEntry = () => VStringAllocator.AcquireAll(new List<VStringAllocator> { x, null });
            nullEntry.Should().Throw<ArgumentNullException>().WithMessage("An allocator to lock is null.*");
        }

        /// <summary>The allocator's private lock object (the tests read whether it is held).</summary>
        /// <param name="a">The allocator.</param>
        /// <returns>Its lock object.</returns>
        private static object GetLock(VStringAllocator a)
            => typeof(VStringAllocator).GetField("_lock", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(a);

        /// <summary>
        ///     Heap strings are reported to the GC in batches; many of them cross the batch threshold, and the release frees
        ///     every one (the clear loop) and withdraws the reported pressure.
        /// </summary>
        [TestMethod]
        public void Allocator_HeapStrings_AreFreedByTheClearLoop()
        {
            const int n = 2000;
            var a = np.array(Enumerable.Repeat(new string('q', 20), n).ToArray(), T);
            var allocator = a.Storage.VStringSlice.Allocator;
            for (int i = 0; i < n; i++)
                a.SetValue(new string('h', 600), i); // each moves to the heap (600 > 20)
            allocator.HeapStrings.Should().Be(n);
            a.Dispose();
            allocator.IsFreed.Should().BeTrue();
            allocator.HeapStrings.Should().Be(0);
            allocator.ArenaCapacity.Should().Be(0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // the storage around the slice
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>CopyTo</c> of the raw cells checks the destination's size first ("Destination was too short.", for a
        ///     <see cref="Span{T}"/> and an <see cref="UnmanagedSpan{T}"/> alike) and copies the cells exactly.
        /// </summary>
        [TestMethod]
        public unsafe void Slice_CopyTo_ChecksTheDestination()
        {
            var s = VStringArraySlice.Allocate(T, 2);
            try
            {
                s[0] = "ab";
                Action shortSpan = () => s.CopyTo(new byte[31].AsSpan());
                shortSpan.Should().Throw<ArgumentException>().WithMessage("Destination was too short.");
                byte* raw = stackalloc byte[32];
                Action shortUnmanaged = () => s.CopyTo(new UnmanagedSpan<byte>(raw, 31));
                shortUnmanaged.Should().Throw<ArgumentException>().WithMessage("Destination was too short.");
                s.CopyTo(new UnmanagedSpan<byte>(raw, 32));
                new ReadOnlySpan<byte>(raw, 16).ToArray().Should().Equal(new ReadOnlySpan<byte>(s.Address, 16).ToArray());
            }
            finally
            {
                s.DangerousFree();
            }
        }

        /// <summary>
        ///     The storage's raw <c>CopyTo(address)</c> of a <c>StringDType</c> array writes its cells in C order — the bytes
        ///     <c>tobytes()</c> returns — gathering a non-contiguous view first and writing nothing for an empty array.
        /// </summary>
        [TestMethod]
        public unsafe void Storage_CopyTo_IsTheCellBytes()
        {
            using var a = np.array(new[] { new string('a', 20), "b", "c", new string('d', 30) }, T);
            var all = new byte[64];
            fixed (byte* p = all)
                a.CopyTo((void*)p); // the untyped overload: a T array has no element type
            all.Should().Equal(a.tobytes());
            using var stepped = a["::2"];
            var two = new byte[32];
            fixed (byte* p = two)
                stepped.CopyTo((void*)p);
            two.Should().Equal(a.tobytes().Take(16).Concat(a.tobytes().Skip(32).Take(16)));
            using var empty = np.empty(new Shape(0), T);
            var none = new byte[1];
            fixed (byte* p = none)
                empty.CopyTo((void*)p); // nothing to write
        }

        /// <summary>
        ///     A storage built straight over a <c>StringDType</c> slice reads its descriptor from the slice (the instance with
        ///     its <c>na_object</c> and <c>coerce</c>); a string storage without data still knows its element size from its
        ///     descriptor; <c>na_eq_cmp</c> of a parameter set against none is False.
        /// </summary>
        [TestMethod]
        public void Storage_DescriptorAndParameters()
        {
            var descr = TN;
            var s = VStringArraySlice.Allocate(descr, 2);
            try
            {
                var storage = new UnmanagedStorage(s, new Shape(2));
                storage.Descr.Should().BeSameAs(s.Descr);
            }
            finally
            {
                s.DangerousFree();
            }
            new UnmanagedStorage(np.dtype("U5")).DTypeSize.Should().Be(20);
            VStringParameters.AreEqual(descr.VString, null).Should().BeFalse();
            VStringParameters.AreEqual(null, descr.VString).Should().BeFalse();
        }

        // ------------------------------------------------------------------------------------------------------------
        // lifetime: the cells' block owns the allocator
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A view keeps the buffer — and its allocator — alive after its base is disposed (the view's texts still read); the
        ///     last release runs the clear loop once (a second call of it does nothing).
        /// </summary>
        [TestMethod]
        public void Lifetime_ViewsKeepTheAllocator()
        {
            var a = np.array(new[] { new string('a', 20), "b", new string('c', 300) }, T);
            var allocator = a.Storage.VStringSlice.Allocator;
            var view = a["::2"];
            var viewOfView = view["::-1"];
            a.Dispose();
            allocator.IsFreed.Should().BeFalse();
            view.GetValue(1).Should().Be(new string('c', 300));
            view.Dispose();
            allocator.IsFreed.Should().BeFalse();
            viewOfView.GetValue(1).Should().Be(new string('a', 20));
            viewOfView.Dispose();
            allocator.IsFreed.Should().BeTrue();
            unsafe
            {
                allocator.ClearAndFree(null, 0); // idempotent: the freed allocator returns at once
            }
            allocator.IsFreed.Should().BeTrue();
        }

        /// <summary>A copy gets its own buffer and allocator; disposing the source leaves the copy readable.</summary>
        [TestMethod]
        public void Lifetime_CopiesOwnTheirAllocator()
        {
            var a = np.array(new[] { new string('a', 20), "b" }, T);
            var copy = a.copy();
            copy.Storage.VStringSlice.Allocator.Should().NotBeSameAs(a.Storage.VStringSlice.Allocator);
            a.Dispose();
            copy.GetValue(0).Should().Be(new string('a', 20));
            copy.Dispose();
        }

        /// <summary>
        ///     The live-allocator count returns to its baseline once the arrays are released, through <c>Dispose</c> and through
        ///     the garbage collector: no allocator (no arena, no heap string) outlives its last array.
        /// </summary>
        [TestMethod]
        public void Lifetime_LiveAllocatorsReturnToBaseline()
        {
            GcQuiescence.CollectFull();
            long baseline = VStringAllocator.LiveAllocators;
            var arrays = Enumerable.Range(0, 10).Select(i => np.array(new[] { new string('x', 20 + i), "y" }, T)).ToList();
            VStringAllocator.LiveAllocators.Should().Be(baseline + 10);
            foreach (var a in arrays)
                a.Dispose();
            VStringAllocator.LiveAllocators.Should().Be(baseline);

            var probe = MakeGarbage();
            GcQuiescence.WaitCollected(probe).Should().BeTrue();
            GcQuiescence.CollectFull();
            VStringAllocator.LiveAllocators.Should().Be(baseline, "a collected array's block runs the clear loop");
        }

        /// <summary>A StringDType array that becomes garbage (never disposed), and a weak reference to it.</summary>
        /// <returns>A weak reference to the garbage array (NoInlining: no stack slot keeps it alive).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference MakeGarbage()
        {
            var a = np.array(new[] { new string('g', 40), "h" }, T);
            a.SetValue(new string('z', 500), 0);
            return new WeakReference(a);
        }
    }
}
