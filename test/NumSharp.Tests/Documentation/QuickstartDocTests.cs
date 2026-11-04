using System;
using System.Numerics;
using NumSharp;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the code examples of <c>docs/website-src/docs/quickstart.md</c> (NumPy's quickstart
    ///     converted to C#), except its strings snippet, which <see cref="TextInGuidesDocTests"/> covers. Each test mirrors
    ///     one section and asserts the values the page prints; every value was re-run in NumPy 2.4.2, so the page cannot
    ///     drift from either library. The examples that draw from <c>np.random</c> assert shapes only, and never seed the
    ///     shared generator (the suite runs in parallel).
    /// </summary>
    [TestClass]
    public class QuickstartDocTests
    {
        /// <summary>"An example" and "Array creation": shape facts of an arange reshape, the dtype a C# array infers
        /// (int32 for C# int, unlike NumPy's int64), 2-D and complex creation, the constructors and the ranges.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Basics_AndArrayCreation()
        {
            var a = np.arange(15).reshape(3, 5);
            a.shape.Should().Equal(new long[] { 3, 5 });
            a.ndim.Should().Be(2);
            a.dtype.name.Should().Be("int64", "np.arange gives int64");
            a.itemsize.Should().Be(8);
            a.size.Should().Be(15);
            a.GetType().Should().Be(typeof(NDArray));
            np.array([6, 7, 8]).ToArray<int>().Should().Equal(6, 7, 8);

            np.array([2, 3, 4]).dtype.Should().Be(np.int32, "a C# int literal is int32");
            np.array([1.2, 3.5, 5.1]).dtype.Should().Be(np.float64);
            np.array(new[,] { { 1.5, 2, 3 }, { 4, 5, 6 } }).ToString().Should().Be("[[1.5 2.  3. ]\n [4.  5.  6. ]]");
            np.array(new[,] { { 1, 2 }, { 3, 4 } }, np.complex128).ToString().Should().Be("[[1.+0.j 2.+0.j]\n [3.+0.j 4.+0.j]]");

            np.zeros((3, 4)).dtype.Should().Be(np.float64);
            np.ones((2, 3, 4), np.int16).dtype.Should().Be(np.int16);
            np.empty((2, 3)).shape.Should().Equal(new long[] { 2, 3 });
            np.arange(10, 30, 5).ToArray<long>().Should().Equal(10L, 15L, 20L, 25L);
            np.arange(0, 2, 0.3).ToString().Should().Be("[0.  0.3 0.6 0.9 1.2 1.5 1.8]");
            np.linspace(0, 2, 9).ToString().Should().Be("[0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]");
            np.sin(np.linspace(0, 2 * np.pi, 100)).size.Should().Be(100);
        }

        /// <summary>"Printing arrays": 1-D prints as a row, 2-D as a matrix, and a large array elides its middle.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void PrintingArrays()
        {
            np.arange(6).ToString().Should().Be("[0 1 2 3 4 5]");
            np.arange(12).reshape(4, 3).ToString().Should().Be("[[ 0  1  2]\n [ 3  4  5]\n [ 6  7  8]\n [ 9 10 11]]");
            np.arange(24).reshape(2, 3, 4).shape.Should().Equal(new long[] { 2, 3, 4 });
            np.arange(10000).ToString().Should().Be("[   0    1    2 ... 9997 9998 9999]");
        }

        /// <summary>"Basic operations": elementwise arithmetic and comparison, the elementwise <c>*</c> beside the matrix
        /// product, C# compound assignment rebinding the variable (the documented difference from NumPy), the in-place
        /// forms, the same_kind refusal of a float result into an int out, and upcasting to complex.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void BasicOperations()
        {
            var a = np.array([20, 30, 40, 50]);
            var b = np.arange(4);
            (a - b).ToArray<long>().Should().Equal(20L, 29L, 38L, 47L);
            np.power(b, 2).ToArray<long>().Should().Equal(0L, 1L, 4L, 9L);
            var tenSin = (10 * np.sin(a)).ToArray<double>();
            tenSin[0].Should().BeApproximately(9.12945251, 1e-8);
            tenSin[1].Should().BeApproximately(-9.88031624, 1e-8);
            tenSin[2].Should().BeApproximately(7.4511316, 1e-7);
            tenSin[3].Should().BeApproximately(-2.62374854, 1e-8);
            (a < 35).ToArray<bool>().Should().Equal(true, true, false, false);

            var A = np.array(new[,] { { 1, 1 }, { 0, 1 } });
            var B = np.array(new[,] { { 2, 0 }, { 3, 4 } });
            (A * B).ToArray<int>().Should().Equal(2, 0, 0, 4);
            np.matmul(A, B).ToArray<int>().Should().Equal(5, 4, 3, 4);
            A.dot(B).ToArray<int>().Should().Equal(5, 4, 3, 4);

            var ones = np.ones((2, 3), np.int32);
            var alias = ones;
            ones *= 3;
            ones.ToArray<int>().Should().Equal(3, 3, 3, 3, 3, 3);
            alias.ToArray<int>().Should().Equal(new[] { 1, 1, 1, 1, 1, 1 }, "C# compound assignment rebinds the variable");
            alias[":"] = alias * 3;
            alias.ToArray<int>().Should().Equal(new[] { 3, 3, 3, 3, 3, 3 }, "a slice assignment writes in place");
            np.multiply(alias, 3, @out: alias);
            alias.ToArray<int>().Should().Equal(9, 9, 9, 9, 9, 9);

            var ai = np.ones(3, np.int32);
            Action intoInt = () => np.add(ai, np.linspace(0, 3, 3), @out: ai);
            intoInt.Should().Throw<ArgumentException>()
                   .WithMessage("Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");

            var c = np.ones(3, np.int32) + np.linspace(0, np.pi, 3);
            c.dtype.name.Should().Be("float64");
            np.exp(c * new Complex(0, 1)).dtype.name.Should().Be("complex128");
        }

        /// <summary>"Unary operations" and the per-axis forms, then "Universal functions": NumPy's values for sum / min /
        /// max, axis reductions, a cumulative sum along rows, exp / sqrt and a mixed-dtype add.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ReductionsAndUfuncs()
        {
            var a = np.array([0.5, 0.4, 0.55, 0.03, 0.75, 0.53]);
            ((double)a.sum()).Should().BeApproximately(2.76, 1e-12);
            ((double)a.min()).Should().Be(0.03);
            ((double)a.max()).Should().Be(0.75);

            var b = np.arange(12).reshape(3, 4);
            b.sum(axis: 0).ToArray<long>().Should().Equal(12L, 15L, 18L, 21L);
            b.min(axis: 1).ToArray<long>().Should().Equal(0L, 4L, 8L);
            np.cumsum(b, axis: 1).ToArray<long>().Should().Equal(0L, 1L, 3L, 6L, 4L, 9L, 15L, 22L, 8L, 17L, 27L, 38L);

            var B = np.arange(3);
            np.exp(B).ToString().Should().Be("[1.         2.71828183 7.3890561 ]");
            np.sqrt(B).ToString().Should().Be("[0.         1.         1.41421356]");
            np.add(B, np.array([2.0, -1.0, 4.0])).ToArray<double>().Should().Equal(2.0, 0.0, 6.0);
        }

        /// <summary>"Indexing, slicing and iterating": 1-D slicing and a stepped store, the newaxis outer table read by
        /// element, column, rows and the last row, the ellipsis on a 3-D array, and iteration by rows and by elements.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void IndexingSlicingAndIterating()
        {
            var a = np.power(np.arange(10), 3);
            ((long)a[2]).Should().Be(8L);
            a["2:5"].ToArray<long>().Should().Equal(8L, 27L, 64L);
            a[":6:2"] = 1000;
            a.ToArray<long>().Should().Equal(1000L, 1L, 1000L, 27L, 1000L, 125L, 216L, 343L, 512L, 729L);
            ((long)a["::-1"][0]).Should().Be(729L);

            var b = np.arange(5)[Slice.All, np.newaxis] * 10 + np.arange(4);
            b.shape.Should().Equal(new long[] { 5, 4 });
            ((long)b[2, 3]).Should().Be(23L);
            b["0:5, 1"].ToArray<long>().Should().Equal(1L, 11L, 21L, 31L, 41L);
            b[":, 1"].ToArray<long>().Should().Equal(1L, 11L, 21L, 31L, 41L);
            b["1:3, :"].shape.Should().Equal(new long[] { 2, 4 });
            b[-1].ToArray<long>().Should().Equal(40L, 41L, 42L, 43L);

            var c = np.array(new[,,] { { { 0, 1, 2 }, { 10, 12, 13 } }, { { 100, 101, 102 }, { 110, 112, 113 } } });
            c.shape.Should().Equal(new long[] { 2, 2, 3 });
            c["1, ..."].ToArray<int>().Should().Equal(100, 101, 102, 110, 112, 113);
            c["..., 2"].ToArray<int>().Should().Equal(2, 13, 102, 113);

            int rows = 0;
            foreach (var row in b)
            {
                ((NDArray)row).shape.Should().Equal(new long[] { 4 }, "each row is an NDArray");
                rows++;
            }
            rows.Should().Be(5);
            int elements = 0;
            foreach (var _ in b.flat)
                elements++;
            elements.Should().Be(20);
        }

        /// <summary>"Shape manipulation": ravel / reshape / transpose of a random 3x4, an in-place resize, a -1 dimension,
        /// stacking (column_stack vs hstack of 1-D arrays, a newaxis column), the <c>r_</c> grammar and the splits.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ShapeManipulation()
        {
            var a = np.floor(10 * np.random.rand(3, 4));
            a.shape.Should().Equal(new long[] { 3, 4 });
            a.ravel().shape.Should().Equal(new long[] { 12 });
            a.reshape(6, 2).shape.Should().Equal(new long[] { 6, 2 });
            a.shape.Should().Equal(new long[] { 3, 4 }, "reshape returns a new array");
            a.T.shape.Should().Equal(new long[] { 4, 3 });
            a.resize(2, 6);
            a.shape.Should().Equal(new long[] { 2, 6 }, "resize changes the array itself");
            np.arange(12).reshape(3, -1).shape.Should().Equal(new long[] { 3, 4 });

            var p = np.floor(10 * np.random.rand(2, 2));
            var q = np.floor(10 * np.random.rand(2, 2));
            np.vstack(p, q).shape.Should().Equal(new long[] { 4, 2 });
            np.hstack(p, q).shape.Should().Equal(new long[] { 2, 4 });

            var x = np.array([4.0, 2.0]);
            var y = np.array([3.0, 8.0]);
            np.column_stack(x, y).ToString().Should().Be("[[4. 3.]\n [2. 8.]]");
            np.hstack(x, y).ToArray<double>().Should().Equal(4.0, 2.0, 3.0, 8.0);
            x[Slice.All, np.newaxis].ToString().Should().Be("[[4.]\n [2.]]");
            np.r_["1:4", 0, 4].ToArray<long>().Should().Equal(1L, 2L, 3L, 0L, 4L);

            var wide = np.floor(10 * np.random.rand(2, 12));
            np.hsplit(wide, 3).Length.Should().Be(3);
            var cuts = np.hsplit(wide, [3, 4]);
            cuts.Length.Should().Be(3);
            cuts[0].shape.Should().Equal(new long[] { 2, 3 });
            cuts[1].shape.Should().Equal(new long[] { 2, 1 });
            cuts[2].shape.Should().Equal(new long[] { 2, 8 });
            np.vsplit(np.arange(16).reshape(4, 4), 2).Length.Should().Be(2);
        }

        /// <summary>"Copies and views": assignment aliases the object, <c>view()</c> shares the buffer without owning it,
        /// a slice store writes through, and <c>copy()</c> detaches.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void CopiesAndViews()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = a;
            ReferenceEquals(a, b).Should().BeTrue();
            var c = a.view();
            ReferenceEquals(c, a).Should().BeFalse();
            c.flags.owndata.Should().BeFalse("c is a view");
            np.shares_memory(a, c).Should().BeTrue();
            var s = a[":, 1:3"];
            s[":"] = 10;
            a.ToArray<long>().Should().Equal(0L, 10L, 10L, 3L, 4L, 10L, 10L, 7L, 8L, 10L, 10L, 11L);
            np.shares_memory(a, a.copy()).Should().BeFalse();
            np.shares_memory(np.arange(1000), np.arange(1000)["0:100"].copy()).Should().BeFalse("the copy detaches");
        }

        /// <summary>"Advanced indexing and index tricks": gathers by 1-D and 2-D index arrays, the palette lookup, two
        /// index arrays (and one with a scalar), the per-column argmax pick, fancy stores (the last repeated index wins)
        /// and boolean masks.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void AdvancedIndexing()
        {
            var squares = np.power(np.arange(12), 2);
            squares[np.array([1, 1, 3, 8, 5])].ToArray<long>().Should().Equal(1L, 1L, 9L, 64L, 25L);
            var j2 = squares[np.array(new[,] { { 3, 4 }, { 9, 7 } })];
            j2.shape.Should().Equal(new long[] { 2, 2 });
            j2.ToArray<long>().Should().Equal(9L, 16L, 81L, 49L);

            var palette = np.array(new[,] { { 0, 0, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 255 } });
            var image = np.array(new[,] { { 0, 1, 2, 0 }, { 0, 3, 4, 0 } });
            palette[image].shape.Should().Equal(new long[] { 2, 4, 3 });

            var a = np.arange(12).reshape(3, 4);
            var i = np.array(new[,] { { 0, 1 }, { 1, 2 } });
            var j = np.array(new[,] { { 2, 1 }, { 3, 3 } });
            a[i, j].ToArray<long>().Should().Equal(2L, 5L, 7L, 11L);
            a[i, 2].ToArray<long>().Should().Equal(2L, 6L, 6L, 10L);
            a[Slice.All, j].shape.Should().Equal(new long[] { 3, 2, 2 });

            var data = np.sin(np.arange(20).astype(np.float64)).reshape(5, 4);
            var ind = data.argmax(axis: 0);
            ind.ToArray<long>().Should().Equal(2L, 0L, 3L, 1L);
            var time = np.linspace(20, 145, 5);
            time[ind].ToArray<double>().Should().Equal(82.5, 20.0, 113.75, 51.25);
            var dataMax = data[ind, np.arange(data.shape[1])].ToArray<double>();
            dataMax[0].Should().BeApproximately(0.98935825, 1e-8);
            dataMax[3].Should().BeApproximately(0.6569866, 1e-7);

            var z = np.arange(5);
            z[new[] { 1, 3, 4 }] = 0;
            z.ToArray<long>().Should().Equal(0L, 0L, 2L, 0L, 0L);
            var w = np.arange(5);
            w[new[] { 0, 0, 2 }] = np.array([1, 2, 3]);
            w.ToArray<long>().Should().Equal(new[] { 2L, 1L, 3L, 3L, 4L }, "index 0 is written twice and the last write wins");

            var m = np.arange(12).reshape(3, 4);
            var mask = m > 4;
            m[mask].ToArray<long>().Should().Equal(5L, 6L, 7L, 8L, 9L, 10L, 11L);
            m[mask] = 0;
            m.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
        }
    }
}
