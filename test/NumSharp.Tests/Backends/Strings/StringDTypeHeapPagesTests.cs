using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The heap pages of the NEP 55 <c>StringDType</c> allocator (<see cref="VStringAllocator"/>): a text that outgrows its
    ///     arena slot is carved from a page of its size class (64 classes, block strides of 32 to 1,040 bytes) instead of one
    ///     C <c>malloc</c> each, and a page goes back to the buffer pool when its last text is freed. The cells stay NumPy's;
    ///     these tests hold what the pages add: every text keeps its bytes at every class edge, freed blocks are reused, an
    ///     emptied page is released, the memory follows the live texts under random churn, and
    ///     <see cref="VStringAllocator.ValidateHeapPages"/> finds no broken invariant on the way.
    /// </summary>
    [TestClass]
    public class StringDTypeHeapPagesTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor (no NA object, coercing).</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor (C# null is the NA object).</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>
        ///     A text of <paramref name="length"/> characters that differs for every (length, tag) pair, so two blocks that
        ///     overlapped, or a text written into another element's block, would change a value the test reads back.
        /// </summary>
        /// <param name="length">The length in characters (ASCII, so also in UTF-8 bytes).</param>
        /// <param name="tag">The element or generation it belongs to.</param>
        /// <returns>The text.</returns>
        private static string Text(int length, int tag)
        {
            var chars = new char[length];
            for (int i = 0; i < length; i++)
                chars[i] = (char)('A' + (i * 7 + tag * 13 + length) % 58);
            return new string(chars);
        }

        /// <summary>An array of <paramref name="n"/> one-character texts: each element is a SHORT cell, so any longer text
        ///     written into it goes to the heap (<c>heap_or_arena_allocate</c>).</summary>
        /// <param name="n">The element count.</param>
        /// <param name="descr">The descriptor (<see cref="T"/> by default).</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray ShortCells(int n, DType descr = null)
            => np.array(Enumerable.Range(0, n).Select(i => ((char)('a' + i % 26)).ToString()).ToArray(), descr ?? T);

        /// <summary>The allocator of a <c>StringDType</c> array's buffer.</summary>
        /// <param name="a">The array.</param>
        /// <returns>Its allocator.</returns>
        private static VStringAllocator Allocator(NDArray a) => a.Storage.VStringSlice.Allocator;

        /// <summary>Asserts that the heap pages of <paramref name="allocator"/> keep every invariant.</summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="because">What the test just did.</param>
        private static void AssertPagesValid(VStringAllocator allocator, string because)
            => allocator.ValidateHeapPages().Should().BeNull(because);

        /// <summary>
        ///     Every text size from 16 to 1,040 bytes (each class's first and last size, and past the largest class, where a
        ///     text is a block of its own) plus long texts: written over short cells, rewritten with other text of the SAME size
        ///     (the freed block is popped again, so a write past a block's capacity would now overwrite a LIVE neighbour's
        ///     header, which the page check finds), read back, then replaced by short texts — after which no heap text and no
        ///     page is left.
        /// </summary>
        [TestMethod]
        public void EverySize_RoundTripsAndReleasesItsPages()
        {
            var sizes = Enumerable.Range(16, 1040 - 16 + 1).Concat(new[] { 1100, 4096, 70_000 }).ToArray();
            const int n = 9;
            foreach (int size in sizes)
            {
                using var a = ShortCells(n);
                var allocator = Allocator(a);
                for (int i = 0; i < n; i++)
                    a.SetValue(Text(size, i), i);
                allocator.HeapStrings.Should().Be(n, $"size {size}: every text outgrew its short cell");
                if (size <= VStringAllocator.HeapPoolMaxBytes)
                    allocator.HeapPages.Should().BeGreaterThan(0, $"size {size}: a page-sized text is carved from a page");
                else
                    allocator.HeapPages.Should().Be(0, $"size {size}: a longer text is a block of its own");
                // the same size again with other bytes: free + allocate pops the block just freed
                for (int i = 0; i < n; i++)
                    a.SetValue(Text(size, 100 + i), i);
                AssertPagesValid(allocator, $"size {size}: after the rewrite");
                for (int i = 0; i < n; i++)
                    a.GetValue(i).Should().Be(Text(size, 100 + i), $"size {size}, element {i}");
                for (int i = 0; i < n; i++)
                    a.SetValue("z", i);
                allocator.HeapStrings.Should().Be(0, $"size {size}: the short writes freed every heap text");
                allocator.HeapPages.Should().Be(0, $"size {size}: an emptied page goes back to the buffer pool");
            }
        }

        /// <summary>
        ///     Freed blocks are reused before the class takes another page, and a page with a live block is kept: a thousand
        ///     texts of one class, half of them freed, refilled with texts of the same class — the page count does not move —
        ///     then all freed, which releases every page.
        /// </summary>
        [TestMethod]
        public void FreedBlocks_AreReusedBeforeANewPage()
        {
            const int n = 1000;
            using var a = ShortCells(n);
            var allocator = Allocator(a);
            for (int i = 0; i < n; i++)
                a.SetValue(Text(40, i), i);
            long pages = allocator.HeapPages;
            pages.Should().BeGreaterThan(1, "a thousand 40-byte texts fill more than one page");
            for (int i = 0; i < n; i += 2)
                a.SetValue("s", i);
            allocator.HeapStrings.Should().Be(n / 2);
            allocator.HeapPages.Should().Be(pages, "every page still holds the odd texts");
            AssertPagesValid(allocator, "after freeing every second text");
            for (int i = 0; i < n; i += 2)
                a.SetValue(Text(33, i + n), i);
            allocator.HeapPages.Should().Be(pages, "texts of the same class take the freed blocks");
            AssertPagesValid(allocator, "after the refill");
            for (int i = 0; i < n; i++)
                a.GetValue(i).Should().Be(i % 2 == 0 ? Text(33, i + n) : Text(40, i), $"element {i}");
            for (int i = 0; i < n; i++)
                a.SetValue("", i);
            allocator.HeapStrings.Should().Be(0);
            allocator.HeapPages.Should().Be(0);
        }

        /// <summary>
        ///     One element rewritten back and forth (its page's only text freed, then a page needed again): the emptied page is
        ///     kept as the allocator's one spare and taken back, so the count never passes one page; a class whose four blocks
        ///     the spare cannot hold takes a page of its own and leaves the spare for a class it fits.
        /// </summary>
        [TestMethod]
        public void OneElementPingPong_ReusesTheSparePage()
        {
            using var a = ShortCells(4);
            var allocator = Allocator(a);
            for (int i = 0; i < 1000; i++)
            {
                a.SetValue(Text(i % 2 == 0 ? 40 : 41, i), 2);
                allocator.HeapPages.Should().Be(1, $"write {i}: the spare page came back");
                a.GetValue(2).Should().Be(Text(i % 2 == 0 ? 40 : 41, i), $"write {i}");
            }
            a.SetValue("q", 2);
            allocator.HeapPages.Should().Be(0);
            allocator.HasHeapSpare.Should().BeTrue("the emptied page is kept as the spare");
            a.SetValue(Text(400, 7), 1);
            allocator.HeapPages.Should().Be(1);
            allocator.HasHeapSpare.Should().BeTrue("a 1 KiB spare cannot hold four 416-byte blocks: the class took its own page");
            a.SetValue(Text(30, 8), 0);
            allocator.HeapPages.Should().Be(2);
            allocator.HasHeapSpare.Should().BeFalse("a class whose blocks fit re-used the spare");
            AssertPagesValid(allocator, "after the last writes");
            a.GetValue(0).Should().Be(Text(30, 8));
            a.GetValue(1).Should().Be(Text(400, 7));
        }

        /// <summary>
        ///     Random writes of every kind — empty, short (in the cell), medium and long texts that may reuse their arena slot
        ///     or go to the heap, texts past the largest class, NA — over three thousand elements, checked against a managed
        ///     mirror: every value is right at each checkpoint, the pages keep their invariants, and no page is held without a
        ///     live text (the memory follows the live texts, not the peak).
        /// </summary>
        [TestMethod]
        public void RandomChurn_MatchesAMirror_AndHoldsNoEmptyPage()
        {
            const int n = 3000;
            var rng = new System.Random(20261018);
            using var a = ShortCells(n, TN);
            var allocator = Allocator(a);
            var mirror = Enumerable.Range(0, n).Select(i => (string)((char)('a' + i % 26)).ToString()).ToArray();
            for (int step = 1; step <= 60_000; step++)
            {
                int i = rng.Next(n);
                int kind = rng.Next(100);
                string text = kind switch
                {
                    < 5 => null,                                   // NA
                    < 10 => "",
                    < 25 => Text(rng.Next(1, 16), step),          // short
                    < 60 => Text(rng.Next(16, 61), step),         // medium
                    < 85 => Text(rng.Next(61, 1033), step),       // up to the largest class
                    < 95 => Text(rng.Next(1033, 1300), step),     // a block of its own
                    _ => Text(rng.Next(16, 33), step),            // small classes again (reuse)
                };
                a.SetValue(text, i);
                mirror[i] = text;
                if (step % 10_000 == 0)
                {
                    AssertPagesValid(allocator, $"step {step}");
                    allocator.HeapPages.Should().BeLessThanOrEqualTo(allocator.HeapStrings,
                        $"step {step}: every page holds at least one live heap text");
                    for (int j = 0; j < n; j++)
                        a.GetValue(j).Should().Be(mirror[j], $"step {step}, element {j}");
                }
            }
            allocator.HeapSoloStrings.Should().BeGreaterThan(0, "the churn leaves texts longer than the largest class");
            a.Dispose();
            allocator.IsFreed.Should().BeTrue();
            allocator.HeapStrings.Should().Be(0);
            allocator.HeapPages.Should().Be(0);
            allocator.HeapSoloStrings.Should().Be(0, "the clear loop freed every block of its own");
        }

        /// <summary>
        ///     The bulk case the pages are for: an array holding texts (each in its arena slot) assigned texts that are longer
        ///     for about half the elements (<c>np.copyto</c>), which sends those to the heap; the values are the source's and the
        ///     pages are valid; a view keeps the buffer, and the last release returns everything.
        /// </summary>
        [TestMethod]
        public void Copyto_LongerTexts_GoToPages()
        {
            var rng = new System.Random(7);
            const int n = 5000;
            var olds = Enumerable.Range(0, n).Select(i => Text(rng.Next(20, 61), i)).ToArray();
            var news = Enumerable.Range(0, n).Select(i => Text(rng.Next(20, 61), i + n)).ToArray();
            var dst = np.array(olds, T);
            using var src = np.array(news, T);
            var allocator = Allocator(dst);
            np.copyto(dst, src);
            int longer = Enumerable.Range(0, n).Count(i => news[i].Length > olds[i].Length);
            allocator.HeapStrings.Should().Be(longer, "a text longer than the slot it replaces goes to the heap");
            allocator.HeapPages.Should().BeGreaterThan(0);
            AssertPagesValid(allocator, "after the assignment");
            NDArray.AsStringArray(dst).Should().Equal(news);
            var view = dst["::2"];
            dst.Dispose();
            allocator.IsFreed.Should().BeFalse("the view keeps the buffer and its allocator");
            NDArray.AsStringArray(view).Should().Equal(news.Where((_, i) => i % 2 == 0));
            view.Dispose();
            allocator.IsFreed.Should().BeTrue();
            allocator.HeapPages.Should().Be(0);
        }

        /// <summary>
        ///     Eight threads assigning texts of random classes into disjoint quarters of ONE array (the allocator's lock guards
        ///     its pages, as it guards the arena): every value is the last written, and the pages are valid afterwards.
        /// </summary>
        [TestMethod]
        public void ConcurrentWriters_OneArray_KeepEveryText()
        {
            const int n = 4096, threads = 8, writes = 20_000;
            using var a = ShortCells(n);
            var allocator = Allocator(a);
            var finals = new string[n];
            Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, t =>
            {
                var rng = new System.Random(1000 + t);
                int lo = t * (n / threads), hi = lo + n / threads;
                for (int w = 0; w < writes; w++)
                {
                    int i = rng.Next(lo, hi);
                    string text = Text(rng.Next(1, 600), w * threads + t);
                    a.SetValue(text, i);
                    finals[i] = text;
                }
            });
            AssertPagesValid(allocator, "after the concurrent writes");
            for (int i = 0; i < n; i++)
                a.GetValue(i).Should().Be(finals[i] ?? ((char)('a' + i % 26)).ToString(), $"element {i}");
        }

        /// <summary>
        ///     Twenty arrays that each send two thousand texts of every class to the heap pages, free a third of them, and are
        ///     released: the workload of <see cref="Pages_ReturnToTheBufferPool"/>.
        /// </summary>
        private static void PageWorkload()
        {
            for (int r = 0; r < 20; r++)
            {
                using var a = ShortCells(2000);
                for (int i = 0; i < 2000; i++)
                    a.SetValue(Text(16 + i % 1000, i), i);
                for (int i = 0; i < 2000; i += 3)
                    a.SetValue("x", i);
            }
        }

        /// <summary>
        ///     The pages come from the buffer pool and go back to it: across many arrays that each take and release pages, the
        ///     pool's hand-outs and returns balance (a page leaked by the allocator would leave a hand-out unmatched).
        /// </summary>
        /// <remarks>Not parallel: it reads the buffer pool's process-wide counters, which every other test moves.</remarks>
        [TestMethod]
        [DoNotParallelize]
        public void Pages_ReturnToTheBufferPool()
        {
            PageWorkload(); // warm the buckets, so the measured run is not dominated by first allocations
            GC.Collect();
            GC.WaitForPendingFinalizers();
            long taken = SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
            long returned = SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;
            PageWorkload();
            long takenAfter = SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses + SizeBucketedBufferPool.ZeroedAllocs;
            long returnedAfter = SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;
            (takenAfter - taken).Should().BeGreaterThan(20, "the workload takes pages (and the arrays' buffers)");
            (returnedAfter - returned).Should().Be(takenAfter - taken, "every page and buffer the workload took went back");
        }
    }
}
