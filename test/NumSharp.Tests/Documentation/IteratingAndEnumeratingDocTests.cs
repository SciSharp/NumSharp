using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the code examples of <c>docs/website-src/docs/iterating-and-enumerating.md</c>, except
    ///     its "Iterating text" snippet, which <see cref="TextInGuidesDocTests"/> covers. Each test mirrors one section and
    ///     asserts what the page claims: the typed walks require the dtype's own element type (a <c>ref T</c> cannot
    ///     convert), C-order versus memory order on a transposed or reversed view, the write-through forms, and the NumPy
    ///     iterator objects (<c>nditer</c>, <c>nested_iters</c>, <c>broadcast</c>) with NumPy 2.4.2's orders.
    /// </summary>
    [TestClass]
    public class IteratingAndEnumeratingDocTests
    {
        /// <summary>"Views and copies of the data": the BCL views alias the array (a Span write is an array write), and
        /// the LINQ / copy routes read in C order with the dtype's own type, refusing another.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void BclViewsAndCopies()
        {
            var a = np.arange(6).astype(np.float64);
            Span<double> s = a.Unsafe.Span<double>();
            s[2] = 99;
            ((double)a[2]).Should().Be(99.0, "the Span is the array's memory");
            a.Unsafe.Bytes().Length.Should().Be(6 * sizeof(double));

            a[2] = 2.0;
            a.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            a.AsEnumerable<double>().Average().Should().Be(2.5);
            a.AsEnumerable<double>().Where(x => x % 2 == 0).ToList().Should().Equal(0.0, 2.0, 4.0);
            Action otherType = () => a.AsEnumerable<int>().ToList();
            otherType.Should().Throw<ArgumentException>("the element type must be the dtype's own");
            np.flat<double>(a).ToArray().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            var dest = new double[6];
            np.nditer<double>(a).CopyTo(dest).Should().Be(6);
            dest.Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        }

        /// <summary>"Fast typed iteration": a by-ref walk reads and writes the array, and the chunk walk hands out one
        /// Span per inner loop.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void TypedWalks_ByRefAndChunks()
        {
            var a = np.arange(6).astype(np.float64);
            double total = 0;
            foreach (ref double x in np.nditer<double>(a))
                total += x;
            total.Should().Be(15.0);
            foreach (ref double x in np.nditer<double>(a, writeable: true))
                x *= 2;
            a.ToArray<double>().Should().Equal(0.0, 2.0, 4.0, 6.0, 8.0, 10.0);
            // The page feeds each chunk to TensorPrimitives.Multiply (the System.Numerics.Tensors package, which this
            // test project does not reference); the same in-place doubling over the Span is a plain loop here.
            foreach (Span<double> chunk in np.nditer_chunks<double>(a, writeable: true))
                for (int i = 0; i < chunk.Length; i++)
                    chunk[i] *= 2.0;
            a.ToArray<double>().Should().Equal(0.0, 4.0, 8.0, 12.0, 16.0, 20.0);
        }

        /// <summary>"C-order typed walks": over a transposed view, <c>np.flat&lt;T&gt;</c> reads in C order (0 3 1 4 2 5)
        /// and its writeable form and <c>flatiter.AsTyped</c> write through; the flat iterator's stores, cursor and copy
        /// follow C order too.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void COrderWalks_OnATransposedView()
        {
            var a = np.arange(6).reshape(2, 3).T;
            np.flat<long>(a).ToArray().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
            foreach (ref long x in np.flat<long>(a, writeable: true))
                x *= 2;
            a.ToArray<long>().Should().Equal(0L, 6L, 2L, 8L, 4L, 10L);
            foreach (ref long x in a.flatiter.AsTyped<long>(writeable: true))
                x *= 2;
            a.ToArray<long>().Should().Equal(0L, 12L, 4L, 16L, 8L, 20L);

            var t = np.arange(6).reshape(2, 3).T;
            t.flatiter[5] = 99;
            t.flatiter["::2"] = 0;
            t.flatiter[new[] { 0, 2 }] = np.array([7, 8]);
            t.flatiter.copy().ToArray<long>().Should().Equal(7L, 3L, 8L, 4L, 0L, 99L);
            t.flatiter.index.Should().Be(0L);
            t.flatiter.coords.Should().Equal(0L, 0L);
            ReferenceEquals(t.flatiter.Base, t).Should().BeTrue("Base is the array being iterated");
        }

        /// <summary>"Indices with values": <c>ndenumerate&lt;T&gt;.AsRef</c> writes through by reference, and
        /// <c>ndindex</c> walks the index space in C order.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void NdenumerateAndNdindex()
        {
            var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            foreach (var e in np.ndenumerate<int>(a).AsRef(writeable: true))
                e.Value += (int)e.Index[0];
            a.ToArray<int>().Should().Equal(1, 2, 4, 5);

            var seen = new List<string>();
            foreach (var idx in np.ndindex(3, 2).AsSpans())
                seen.Add($"{idx[0]},{idx[1]}");
            seen.Should().Equal("0,0", "0,1", "1,0", "1,1", "2,0", "2,1");
        }

        /// <summary>"Iterating the first axis": each row is a view, so a store into it mutates the array.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Rows_AreViews()
        {
            var m = np.arange(6).reshape(2, 3);
            foreach (NDArray row in m)
                row[":"] = row * 2;
            m.ToArray<long>().Should().Equal(0L, 2L, 4L, 6L, 8L, 10L);
        }

        /// <summary>The <c>np.nditer</c> object: <c>multi_index</c> tracking, one <c>external_loop</c> chunk over a
        /// contiguous array, a <c>readwrite</c> operand, two operands broadcast together, and an allocated output
        /// (NumPy's <c>None</c>).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void NditerObject_FlagsOperandsAndAllocation()
        {
            var visited = new List<string>();
            using (var it = np.nditer(np.arange(6).reshape(2, 3), flags: new[] { "multi_index" }))
                while (!it.finished)
                {
                    visited.Add($"{string.Join(",", it.multi_index)}={(long)it[0]}");
                    it.iternext();
                }
            visited.Should().Equal("0,0=0", "0,1=1", "0,2=2", "1,0=3", "1,1=4", "1,2=5");

            var chunks = new List<long[]>();
            using (var it = np.nditer(np.arange(6).reshape(2, 3), flags: new[] { "external_loop" }))
                foreach (var vals in it)
                    chunks.Add(vals[0].ToArray<long>());
            chunks.Should().HaveCount(1);
            chunks[0].Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);

            var c = np.arange(3);
            using (var it = np.nditer(c, op_flags: new[] { "readwrite" }))
                foreach (var vals in it)
                    vals[0].SetAtIndex(Convert.ToInt64(vals[0].GetAtIndex(0)) * 2, 0);
            c.ToArray<long>().Should().Equal(0L, 2L, 4L);

            var pairs = new List<string>();
            using (var it = np.nditer(new[] { np.array([1, 2, 3]), np.array([[10], [20]]) }))
                foreach (var vals in it)
                    pairs.Add($"{(long)vals[0]}/{(long)vals[1]}");
            pairs.Should().Equal("1/10", "2/10", "3/10", "1/20", "2/20", "3/20");

            using (var it = np.nditer(new[] { np.arange(6), null },
                       op_flags: new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } },
                       op_dtypes: new DType[] { np.int64, np.int64 }))
            {
                foreach (var vals in it)
                    vals[1].SetAtIndex(Convert.ToInt64(vals[0].GetAtIndex(0)) * 2, 0);
                it.operands[1].ToArray<long>().Should().Equal(0L, 2L, 4L, 6L, 8L, 10L);
            }
        }

        /// <summary><c>np.nested_iters</c>: the outer level walks axis 1, the inner level axes 0 and 2 re-based to the
        /// outer position, as NumPy 2.4.2 orders them.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void NestedIters_OuterAndInnerAxes()
        {
            var levels = np.nested_iters(np.arange(12).reshape(2, 3, 2), new[] { new[] { 1 }, new[] { 0, 2 } }, flags: new[] { "multi_index" });
            var values = new List<long>();
            using (var outer = levels[0])
            using (var inner = levels[1])
                foreach (var _ in outer)
                    foreach (var __ in inner)
                        values.Add((long)inner[0]);
            values.Should().Equal(0L, 1L, 6L, 7L, 2L, 3L, 8L, 9L, 4L, 5L, 10L, 11L);
        }

        /// <summary>The broadcast object's per-operand streams, its enumeration and its cursor.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void BroadcastObject_StreamsAndCursor()
        {
            var bc = np.broadcast(np.array([1L, 2L, 3L]), np.array([[10L], [20L]]));
            bc.shape.ToString().Should().Be("(2, 3)");
            bc.numiter.Should().Be(2);
            bc.size.Should().Be(6);
            bc.iters[0].Cast<object>().Select(Convert.ToInt64).Should().Equal(1L, 2L, 3L, 1L, 2L, 3L);
            bc.iters[1].Cast<object>().Select(Convert.ToInt64).Should().Equal(10L, 10L, 10L, 20L, 20L, 20L);
            var sums = new List<string>();
            foreach (object[] vals in bc)
                sums.Add($"{vals[0]}+{vals[1]}");
            sums.Should().Equal("1+10", "2+10", "3+10", "1+20", "2+20", "3+20");
            bc.index.Should().Be(6);
            bc.reset();
            bc.index.Should().Be(0);
        }

        /// <summary>"Iteration order": over a reversed view the default order (<c>'K'</c>) walks memory, 0 1 2 3 4 5, while
        /// <c>order: 'C'</c> walks the logical C order, 2 1 0 5 4 3.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void IterationOrder_MemoryVersusLogical()
        {
            var rev = np.arange(6).reshape(2, 3)[":, ::-1"];
            np.nditer<long>(rev).ToArray().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.nditer<long>(rev, order: 'C').ToArray().Should().Equal(2L, 1L, 0L, 5L, 4L, 3L);
        }
    }
}
