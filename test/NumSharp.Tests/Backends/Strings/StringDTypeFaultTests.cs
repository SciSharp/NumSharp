using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The failure paths of NEP 55 <c>StringDType</c> storage (strings phase 1, P1.3–P1.6): an exhausted allocator
    ///     (scheduled by <see cref="VStringAllocator.FailAllocationAfter"/>) and a corrupt cell. Each operation must raise
    ///     NumPy's <c>MemoryError</c> text for its loop and release what it duplicated before the failure: the half-built
    ///     result is disposed at once (no allocator outlives the call), and the operands stay intact.
    /// </summary>
    [TestClass]
    public class StringDTypeFaultTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A text of 30 characters: a MEDIUM string (an arena allocation), unlike a short one stored in its cell.</summary>
        /// <param name="c">The repeated character.</param>
        /// <returns>The text.</returns>
        private static string Medium(char c) => new string(c, 30);

        /// <summary>Cancels a scheduled allocation failure that did not fire (a test that failed before the operation).</summary>
        [TestCleanup]
        public void Disarm() => VStringAllocator.DisarmFaultInjection();

        /// <summary>
        ///     Runs <paramref name="operation"/> with the allocation after <paramref name="successes"/> successful ones failing,
        ///     and asserts NumPy's MemoryError text and that no allocator the operation created is still alive.
        /// </summary>
        /// <param name="operation">The operation (its operands are created by the caller, before the baseline is read).</param>
        /// <param name="text">NumPy's MemoryError text (a wildcard pattern).</param>
        /// <param name="successes">The allocations that still succeed first.</param>
        /// <remarks>
        ///     The allocator count is process-global, and a collection inside the window would finalize OTHER tests' garbage
        ///     arrays and move it: the window is opened and checked with <see cref="GcQuiescence"/>'s protocol, and a disturbed
        ///     window is measured again (the operation fails the same way each time).
        /// </remarks>
        private static void AssertFailsCleanly(Action operation, string text, int successes = 0)
        {
            for (int attempt = 0; ; attempt++)
            {
                GcQuiescence.CollectYoung();
                var window = GcQuiescence.OpenWindow();
                long baseline = VStringAllocator.LiveAllocators;
                VStringAllocator.FailAllocationAfter(successes);
                try
                {
                    operation.Should().Throw<OutOfMemoryException>().WithMessage(text);
                }
                finally
                {
                    VStringAllocator.DisarmFaultInjection();
                }
                long after = VStringAllocator.LiveAllocators;
                if (GcQuiescence.Undisturbed(window))
                {
                    after.Should().Be(baseline, "the half-built result is released at once");
                    return;
                }
                if (attempt == 4)
                    Assert.Inconclusive("Every measurement window was disturbed by a collection.");
            }
        }

        /// <summary>
        ///     The fault-injection hook itself: it fires exactly once after the given number of allocations, only on the
        ///     scheduling thread, refuses a negative count, and short strings (stored in their cells) never count.
        /// </summary>
        [TestMethod]
        public void Hook_FiresOnceAfterTheGivenAllocations()
        {
            Action negative = () => VStringAllocator.FailAllocationAfter(-1);
            negative.Should().Throw<ArgumentOutOfRangeException>();
            using var a = np.empty(new Shape(4), T);
            VStringAllocator.FailAllocationAfter(1);
            a[0] = "short"; // a short string allocates nothing
            a[1] = Medium('a');
            Action third = () => a[2] = Medium('b');
            third.Should().Throw<OutOfMemoryException>().WithMessage("Failed to pack string during StringDType setitem");
            a[3] = Medium('c'); // fired once: later allocations succeed
            a.GetValue(3).Should().Be(Medium('c'));
            VStringAllocator.FailAllocationAfter(0);
            var other = new System.Threading.Thread(() => { using var b = np.array(new[] { Medium('d') }, T); });
            other.Start();
            other.Join(); // another thread's allocations are not counted
            Action mine = () => a[0] = Medium('e');
            mine.Should().Throw<OutOfMemoryException>();
        }

        /// <summary>
        ///     Every loop that writes a new medium string raises NumPy's text for that loop on an exhausted allocator: the
        ///     unicode, bytes and number casts into <c>StringDType</c>, setitem (the indexer and a fill), and a copy of NA cells
        ///     whose NA object's <c>str()</c> is itself a medium string.
        /// </summary>
        [TestMethod]
        public void Casts_And_Stores_RaiseEachLoopsText()
        {
            using var u = np.array(new[] { Medium('u') });
            AssertFailsCleanly(() => u.astype(T), "Failed to allocate string in unicode to string cast");
            using var s = np.array(new[] { System.Text.Encoding.ASCII.GetBytes(Medium('s')) }, np.dtype("S30"));
            AssertFailsCleanly(() => s.astype(T), "Failed to allocate string in bytes to string cast");
            using var f = np.array(new[] { 1.2345678901234567e-300 });
            AssertFailsCleanly(() => f.astype(T), "Failed to pack string while converting from python string");

            using var t = np.array(new[] { "a", "b" }, T);
            AssertFailsCleanly(() => t[0] = Medium('x'), "Failed to pack string during StringDType setitem");
            // A slice assignment and a fill write every element through the copy loop (NumPy's PyArray_AssignRawScalar →
            // string_to_string → free_and_copy), whose text names the cast.
            AssertFailsCleanly(() => t[":"] = Medium('y'), "Failed to allocate string in string to string cast", successes: 1);
            AssertFailsCleanly(() => t.fill(Medium('y')), "Failed to allocate string in string to string cast");

            // An NA cell copied into a descriptor without an NA object becomes the NA object's str(): a medium string here.
            var naName = np.dtypes.StringDType(na_object: Medium('n'));
            using var withNA = np.array(new object[] { "a", null }, np.dtypes.StringDType(na_object: null));
            using var na = withNA.astype(naName);
            na.GetValue(1).Should().Be(Medium('n'), "the NA element reads as the NA object");
            AssertFailsCleanly(() => na.astype(T), "Failed to pack string in string to string cast.");
        }

        /// <summary>
        ///     The gathers duplicate every picked string into a fresh buffer; a failure part-way through releases that buffer
        ///     at once (its clear loop frees the strings already duplicated): a fancy index, <c>np.take</c>, <c>np.resize</c>
        ///     (a view of a fresh buffer), <c>np.where</c> (several sources), <c>tobytes</c> of a non-contiguous array, a
        ///     copy, and the in-place <c>ndarray.resize</c>.
        /// </summary>
        [TestMethod]
        public void Gathers_ReleaseTheHalfBuiltResult()
        {
            using var t = np.array(new[] { Medium('a'), Medium('b'), Medium('c'), Medium('d') }, T);
            const string Dup = "Failed to allocate string in string to string cast";
            AssertFailsCleanly(() => t[np.array(new[] { 3, 2, 1 })].Dispose(), Dup, successes: 1);
            AssertFailsCleanly(() => np.take(t, np.array(new[] { 0, 1 })).Dispose(), Dup, successes: 1);
            AssertFailsCleanly(() => np.resize(t, new Shape(6)).Dispose(), Dup, successes: 2);
            using var other = np.array(new[] { Medium('w'), Medium('x'), Medium('y'), Medium('z') }, T);
            AssertFailsCleanly(() => np.where(np.array(new[] { true, false, true, false }), t, other).Dispose(), Dup, successes: 1);
            AssertFailsCleanly(() =>
            {
                using var stepped = t["::2"];
                stepped.tobytes();
            }, Dup, successes: 1);
            AssertFailsCleanly(() => t.copy().Dispose(), Dup, successes: 2);
            t.GetValue(3).Should().Be(Medium('d'), "a failed gather leaves its source intact");

            using var owner = np.array(new[] { Medium('p'), Medium('q') }, T);
            AssertFailsCleanly(() => owner.resize(new Shape(4), refcheck: false), Dup, successes: 1);
            owner.shape.Should().Equal(new[] { 2L }, "a failed in-place resize leaves the array as it was");
            owner.GetValue(1).Should().Be(Medium('q'));
        }

        /// <summary>
        ///     A slice clone and the allocator's own members on an exhausted allocator: <c>Clone</c> frees the half-built copy
        ///     and raises the copy loop's text; <c>NewSize</c> returns -1; <c>FreeAndCopy</c> raises "Failed to allocate"; a heap
        ///     allocation the native heap refuses (a longer rewrite of an arena string) returns -1.
        /// </summary>
        [TestMethod]
        public unsafe void Slice_And_Allocator_Failures()
        {
            var s = VStringArraySlice.Allocate(T, 3);
            var d = VStringArraySlice.Allocate(T, 1);
            try
            {
                s[0] = Medium('a');
                s[1] = Medium('b');
                VStringAllocator.FailAllocationAfter(1);
                Action clone = () => s.Clone();
                clone.Should().Throw<OutOfMemoryException>().WithMessage("Failed to allocate string in string to string cast");
                VStringAllocator.DisarmFaultInjection();

                var al = s.Allocator;
                al.Acquire();
                try
                {
                    byte* text = stackalloc byte[30];
                    VStringAllocator.FailAllocationAfter(0);
                    al.NewSize(text, 30, s.Address + 32).Should().Be(-1);
                    VStringAllocator.DisarmFaultInjection();
                    // A longer rewrite of the arena string in cell 0 moves to the heap; a size the native heap cannot give
                    // fails there (NativeMemory's OutOfMemoryException becomes NumPy's -1).
                    al.Free(s.Address).Should().Be(0);
                    al.NewEmptySize(1L << 55, s.Address).Should().Be(-1);
                    new Span<byte>(s.Address, 16).Clear();
                }
                finally
                {
                    al.Release();
                }

                VStringAllocator.AcquirePair(s.Allocator, d.Allocator);
                try
                {
                    VStringAllocator.FailAllocationAfter(0);
                    Action copy = () => VStringAllocator.FreeAndCopy(s.Allocator, d.Allocator, s.Address + 16, d.Address, "string to string cast");
                    copy.Should().Throw<OutOfMemoryException>().WithMessage("Failed to allocate string in string to string cast");
                }
                finally
                {
                    VStringAllocator.DisarmFaultInjection();
                    VStringAllocator.ReleasePair(s.Allocator, d.Allocator);
                }
            }
            finally
            {
                s.DangerousFree();
                d.DangerousFree();
            }
        }

        /// <summary>
        ///     A CORRUPT cell (an arena string in a buffer whose arena was never allocated, or one whose flags claim an arena
        ///     allocation that does not exist) cannot be freed or reused. NumPy's texts, probed with the same bytes laid over a
        ///     buffer (<c>np.ndarray((2,), StringDType(na_object=None), buffer=…)</c>): an element store is setitem's
        ///     "Failed to pack string during StringDType setitem" (and "Failed to pack null string …" for NA); a slice
        ///     assignment and <c>fill</c> run the copy loop, "Failed to deallocate string in string to string cast".
        /// </summary>
        [TestMethod]
        public unsafe void CorruptCells_AreThePackTexts()
        {
            using var t = np.empty(new Shape(2), np.dtypes.StringDType(na_object: null));
            byte* cell = t.Storage.VStringSlice.Address;
            void Corrupt(ulong flags, ulong size)
            {
                *(ulong*)cell = 1;
                *(ulong*)(cell + 8) = size | (flags << 56);
            }
            try
            {
                Corrupt(0x40, 20); // an arena string of 20 bytes, no arena
                Action text = () => t[0] = "x";
                text.Should().Throw<OutOfMemoryException>().WithMessage("Failed to pack string during StringDType setitem");
                Action na = () => t[0] = null;
                na.Should().Throw<OutOfMemoryException>().WithMessage("Failed to pack null string during StringDType setitem");
                const string Free = "Failed to deallocate string in string to string cast";
                Action sliceNA = () => t[":"] = null;
                sliceNA.Should().Throw<OutOfMemoryException>().WithMessage(Free);
                Action sliceText = () => t[":"] = "x";
                sliceText.Should().Throw<OutOfMemoryException>().WithMessage(Free);
                Action fill = () => t.fill("x");
                fill.Should().Throw<OutOfMemoryException>().WithMessage(Free);
                Action fillNA = () => t.fill(null);
                fillNA.Should().Throw<OutOfMemoryException>().WithMessage(Free);

                Corrupt(0x40, 0); // an emptied arena string (its allocation would be reused), no arena
                Action reuse = () => t[0] = Medium('r');
                reuse.Should().Throw<OutOfMemoryException>().WithMessage("Failed to pack string during StringDType setitem");
            }
            finally
            {
                new Span<byte>(cell, 16).Clear();
            }
        }
    }
}
