using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.array(IEnumerable&lt;T&gt;, size)</c> is NumPy's <c>np.fromiter(data, dtype=T, count=size)</c>:
    ///     exactly <c>size</c> items are read and an enumeration that ends earlier raises NumPy's <c>ValueError</c>.
    ///     Every expectation is NumPy 2.4.2's answer (probed on win-amd64): <c>np.fromiter(iter([1, 2]), np.int64,
    ///     5)</c> raises <c>ValueError: iterator too short: Expected 5 but iterator had only 2 items.</c>;
    ///     <c>it = iter([1, 2, 3, 4]); np.fromiter(it, np.int32, 2)</c> is <c>[1, 2]</c> and <c>next(it)</c> is 3;
    ///     <c>np.fromiter(it, np.int32, 0)</c> is an empty int32 array and reads nothing; an iterator that raises
    ///     propagates its own exception.
    /// </summary>
    /// <remarks>
    ///     Before 2026-10-30 a short enumeration returned an array of <c>size</c> elements whose tail was never
    ///     written: <c>np.array(new List&lt;int&gt; { 1, 2 }, 5)</c> gave <c>[1, 2, 1066396624, 32762, 920]</c> (heap
    ///     contents, a different tail on every call). The two tests that expect the <c>ValueError</c>
    ///     (<see cref="ShortEnumeration_RaisesNumPysIteratorTooShort"/>, <see cref="FailedCall_ReturnsItsBuffer"/>)
    ///     fail on that code.
    /// </remarks>
    [TestClass]
    public class np_array_SizedEnumerable_Tests
    {
        /// <summary>
        ///     A shorter enumeration is NumPy's <c>iterator too short</c> error, with both counts, every time (the old
        ///     array's tail was heap contents, so a single call could happen to look fine).
        /// </summary>
        [TestMethod]
        public void ShortEnumeration_RaisesNumPysIteratorTooShort()
        {
            var two = new List<int> { 1, 2 };
            for (int attempt = 0; attempt < 50; attempt++)
            {
                var e = Assert.ThrowsExactly<ValueError>(() => np.array(two, 5));
                Assert.AreEqual("iterator too short: Expected 5 but iterator had only 2 items.", e.Message);
            }

            // The long overload, an enumeration with no items at all, and a lazy (LINQ) enumeration.
            var empty = Assert.ThrowsExactly<ValueError>(() => np.array(Enumerable.Empty<double>(), 3L));
            Assert.AreEqual("iterator too short: Expected 3 but iterator had only 0 items.", empty.Message);
            var lazy = Assert.ThrowsExactly<ValueError>(
                () => np.array(Enumerable.Range(0, 7).Select(x => (long)x), 8));
            Assert.AreEqual("iterator too short: Expected 8 but iterator had only 7 items.", lazy.Message);
        }

        /// <summary>
        ///     An enumeration of exactly <c>size</c> items, or a longer one, gives the first <c>size</c> items, and a
        ///     longer one is advanced exactly <c>size</c> times (NumPy's loop calls <c>PyIter_Next</c> once per element
        ///     and never past <c>count</c>). <c>size</c> 0 is an empty array of the element type and reads nothing.
        /// </summary>
        [TestMethod]
        public void LongEnoughEnumeration_ReadsExactlySizeItems()
        {
            var exact = np.array(new List<int> { 1, 2, 3 }, 3);
            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, exact.ToArray<int>());
            Assert.AreEqual(NPTypeCode.Int32, exact.typecode);
            CollectionAssert.AreEqual(new long[] { 3 }, exact.shape);

            var counting = new CountingEnumerable<int>(new[] { 1, 2, 3, 4 });
            var first = np.array(counting, 2);
            CollectionAssert.AreEqual(new[] { 1, 2 }, first.ToArray<int>());
            Assert.AreEqual(2, counting.MoveNextCalls, "a longer enumeration keeps its remaining items");
            Assert.AreEqual(1, counting.Disposals, "the enumerator is disposed");

            var none = new CountingEnumerable<int>(new[] { 1, 2 });
            var empty = np.array(none, 0L);
            Assert.AreEqual(0, empty.size);
            Assert.AreEqual(NPTypeCode.Int32, empty.typecode);
            CollectionAssert.AreEqual(new long[] { 0 }, empty.shape);
            Assert.AreEqual(0, none.MoveNextCalls, "size 0 reads nothing");
        }

        /// <summary>
        ///     An exception the enumeration throws propagates unchanged, as an iterator's error does in NumPy, and the
        ///     enumerator is still disposed. A negative size keeps its ValueError (NumSharp's documented divergence:
        ///     NumPy's <c>count=-1</c> reads every item, which is <c>np.array(IEnumerable&lt;T&gt;)</c> here). A null
        ///     enumeration is a <see cref="NullReferenceException"/>, as before.
        /// </summary>
        [TestMethod]
        public void EnumerationError_PropagatesUnchanged_NegativeSizeRefused()
        {
            var throwing = new CountingEnumerable<int>(new[] { 1, 2 }, throwAfter: 2);
            var e = Assert.ThrowsExactly<InvalidOperationException>(() => np.array(throwing, 5));
            Assert.AreEqual("boom", e.Message);
            Assert.AreEqual(1, throwing.Disposals, "the enumerator is disposed when it throws");

            var negative = Assert.ThrowsExactly<ValueError>(() => np.array(new List<int> { 1 }, -1));
            Assert.AreEqual("negative dimensions are not allowed", negative.Message);
            Assert.ThrowsExactly<NullReferenceException>(() => np.array((IEnumerable<int>)null, 2));
        }

        /// <summary>
        ///     A failed call gives its buffer back before the exception leaves: the short enumeration and the throwing
        ///     one each take one pooled buffer and return it (measured in an exact pool window, re-run when a
        ///     collection lands in it). The buffer is 64 KB, so it comes from the bucketed pool.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // reads the process-wide pool counters
        public void FailedCall_ReturnsItsBuffer()
        {
            const int n = 8192;
            var shortData = Enumerable.Range(0, n - 1).Select(x => (double)x).ToArray();
            // Warm-up: the bucket exists and the throw paths are compiled before the window.
            Assert.ThrowsExactly<ValueError>(() => np.array(shortData, n));
            Assert.ThrowsExactly<InvalidOperationException>(
                () => np.array(new CountingEnumerable<double>(new[] { 1.0, 2.0 }, throwAfter: 1), n));

            foreach (var call in new Action[]
                     {
                         () => np.array(shortData, n),
                         () => np.array(new CountingEnumerable<double>(new[] { 1.0, 2.0 }, throwAfter: 1), n),
                     })
            {
                long acquired = 0, released = 0;
                bool exact = false;
                for (int attempt = 0; attempt < 5 && !exact; attempt++)
                {
                    GcQuiescence.CollectYoung();
                    var window = GcQuiescence.OpenWindow();
                    long a0 = Acquisitions, r0 = Releases;
                    try
                    {
                        call();
                        Assert.Fail("the call must raise");
                    }
                    catch (Exception ex) when (ex is ValueError or InvalidOperationException)
                    {
                        // The expected failure; the buffer accounting below is what this test checks.
                    }
                    acquired = Acquisitions - a0;
                    released = Releases - r0;
                    exact = GcQuiescence.Undisturbed(window);
                }

                Assert.IsTrue(acquired >= 1,
                    "the result buffer must come from the pool, or the window measures nothing");
                if (exact)
                    Assert.AreEqual(acquired, released, "a failed call returns every buffer it took");
                else
                    Assert.IsTrue(released >= acquired, "a failed call returns every buffer it took");
            }
        }

        /// <summary>Pooled buffers handed out so far (process-wide).</summary>
        private static long Acquisitions => SizeBucketedBufferPool.Hits + SizeBucketedBufferPool.Misses
                                            + SizeBucketedBufferPool.ZeroedAllocs;

        /// <summary>Pooled buffers given back so far (process-wide).</summary>
        private static long Releases => SizeBucketedBufferPool.Returns + SizeBucketedBufferPool.ReturnsFreed;

        /// <summary>
        ///     An enumeration over fixed items that counts its enumerators' <c>MoveNext</c> calls and disposals, and
        ///     can throw after a number of items (Python's generator that raises).
        /// </summary>
        /// <typeparam name="T">The item type.</typeparam>
        private sealed class CountingEnumerable<T> : IEnumerable<T>
        {
            private readonly T[] _items;
            private readonly int _throwAfter;

            /// <summary>Creates the enumeration.</summary>
            /// <param name="items">The items it yields.</param>
            /// <param name="throwAfter">After this many items <c>MoveNext</c> throws
            ///     <see cref="InvalidOperationException"/> ("boom"); -1 never throws.</param>
            public CountingEnumerable(T[] items, int throwAfter = -1)
            {
                _items = items;
                _throwAfter = throwAfter;
            }

            /// <summary>The <c>MoveNext</c> calls of every enumerator so far.</summary>
            public int MoveNextCalls { get; private set; }

            /// <summary>The enumerator disposals so far.</summary>
            public int Disposals { get; private set; }

            /// <summary>Starts an enumeration.</summary>
            /// <returns>An enumerator over the items.</returns>
            public IEnumerator<T> GetEnumerator() => new Enumerator(this);

            /// <inheritdoc cref="GetEnumerator"/>
            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

            /// <summary>The counting enumerator.</summary>
            private sealed class Enumerator : IEnumerator<T>
            {
                private readonly CountingEnumerable<T> _owner;
                private int _index = -1;

                /// <summary>Creates the enumerator.</summary>
                /// <param name="owner">The enumeration it counts for.</param>
                public Enumerator(CountingEnumerable<T> owner) => _owner = owner;

                /// <summary>The current item.</summary>
                public T Current => _owner._items[_index];

                /// <inheritdoc cref="Current"/>
                object IEnumerator.Current => Current;

                /// <summary>Advances to the next item.</summary>
                /// <returns>Whether there is one.</returns>
                /// <exception cref="InvalidOperationException">The enumeration's <c>throwAfter</c> items were
                ///     read.</exception>
                public bool MoveNext()
                {
                    _owner.MoveNextCalls++;
                    if (_owner._throwAfter >= 0 && _index + 1 == _owner._throwAfter)
                        throw new InvalidOperationException("boom");
                    return ++_index < _owner._items.Length;
                }

                /// <summary>Not supported.</summary>
                /// <exception cref="NotSupportedException">Always.</exception>
                public void Reset() => throw new NotSupportedException();

                /// <summary>Counts the disposal.</summary>
                public void Dispose() => _owner.Disposals++;
            }
        }
    }
}
