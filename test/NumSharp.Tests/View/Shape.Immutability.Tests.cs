using System;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.View
{
    /// <summary>
    ///     NumPy's contract that an array OWNS its dimensions: <c>PyArray_NewFromDescr_int</c> copies the caller's shape into the
    ///     array, and <c>a.shape</c> / <c>a.strides</c> are immutable tuples, so nothing a caller holds can change an array's
    ///     dimensions afterwards. NumSharp's shapes and arrays SHARE dimension and stride arrays (an allocation keeps the
    ///     caller's shape, an elementwise result keeps its input's), which is safe only while no public route writes into them.
    ///     Plan <c>docs/plans/view-shape-open-bugs.md</c>, bug 2.
    /// </summary>
    /// <remarks>
    ///     A changed dimension does not change an array's cached size or its buffer, so the rows it adds read past the buffer
    ///     (probed on 2026-10-05: <c>a.GetDouble(50, 0)</c> on a <c>(100, 4)</c> "view" of 12 elements returned heap memory).
    ///     Every test here therefore checks shapes only.
    /// </remarks>
    [TestClass]
    public class ShapeImmutabilityTests
    {
        /// <summary>
        ///     Bug 2a: <see cref="Shape"/> has no writable indexer (or any other writable indexer). It had a setter that wrote
        ///     into the shared dimension array: <c>var s = new Shape(3, 4); var a = np.zeros(s); s[0] = 100;</c> made <c>a</c>
        ///     <c>(100, 4)</c> over 12 elements, and <c>s = x.Shape; s[0] = 100</c> changed <c>x</c> itself.
        /// </summary>
        [TestMethod]
        public void Indexer_IsReadOnly()
        {
            var indexers = typeof(Shape).GetProperties(BindingFlags.Public | BindingFlags.Instance)
                                        .Where(p => p.GetIndexParameters().Length > 0)
                                        .ToArray();
            indexers.Should().NotBeEmpty("Shape keeps its read accessor shape[i]");
            indexers.Where(p => p.CanWrite).Select(p => p.Name).Should().BeEmpty("a Shape is immutable, like NumPy's shape tuple");
        }

        /// <summary>
        ///     The read accessor keeps NumPy's indexing of a shape tuple: every axis, and negative axes from the end.
        /// </summary>
        [TestMethod]
        public void Indexer_ReadsEveryAxis_NegativeFromTheEnd()
        {
            var s = new Shape(2, 3, 4);
            s[0].Should().Be(2);
            s[1].Should().Be(3);
            s[2].Should().Be(4);
            s[-1].Should().Be(4);
            s[-3].Should().Be(2);

            Action past = () => { var unused = s[3]; };
            past.Should().Throw<IndexOutOfRangeException>();
        }

        /// <summary>
        ///     Bug 2b: <c>a.shape</c> is NumPy's tuple — a value the caller owns. Writing into it does not change the array (it
        ///     made <c>x</c> <c>(100, 4)</c> over 12 elements), and every read is a new array.
        /// </summary>
        [TestMethod]
        public void NDArrayShape_ReturnsACopy()
        {
            using var x = np.zeros(new Shape(3, 4));
            var d = x.shape;
            d[0] = 100;

            x.shape.Should().Equal(new long[] { 3, 4 }, "the caller's copy is not the array's dimensions");
            x.size.Should().Be(12);
            ReferenceEquals(x.shape, x.shape).Should().BeFalse("each read is a fresh array, like a fresh tuple");
        }

        /// <summary>
        ///     An elementwise result shares its input's dimension array, so a write through the input's old <c>shape</c>
        ///     changed BOTH arrays (probe P16: <c>y = x + 1; d = x.shape; d[1] = 9</c> made both <c>(3, 9)</c>). NumPy:
        ///     <c>y.shape == (3, 4)</c>.
        /// </summary>
        [TestMethod]
        public void NDArrayShape_WriteDoesNotReachArraysSharingTheDimensions()
        {
            using var x = np.zeros(new Shape(3, 4));
            using var y = x + 1;
            var d = x.shape;
            d[1] = 9;

            x.shape.Should().Equal(new long[] { 3, 4 });
            y.shape.Should().Equal(new long[] { 3, 4 });
        }

        /// <summary>
        ///     A view's <c>shape</c> is a copy too: <c>v = x[0:1]; d = v.shape; d[1] = 50</c> left <c>v</c> as <c>(1, 50)</c>
        ///     over a <c>(2, 3)</c> buffer (probe P13 through the old getter).
        /// </summary>
        [TestMethod]
        public void NDArrayShape_OfAView_ReturnsACopy()
        {
            using var x = np.zeros(new Shape(2, 3));
            using var v = x["0:1"];
            var d = v.shape;
            d[1] = 50;

            v.shape.Should().Equal(new long[] { 1, 3 });
            x.shape.Should().Equal(new long[] { 2, 3 });
        }

        /// <summary>
        ///     A 0-d array's <c>shape</c> is empty (NumPy's <c>()</c>); there is no element to write into.
        /// </summary>
        [TestMethod]
        public void NDArrayShape_ZeroDimensional_IsEmpty()
        {
            using var a = np.array(5.0);
            a.shape.Should().BeEmpty();
        }

        /// <summary>
        ///     The public wrappers that hand out NumPy's <c>shape</c> (<c>MaskedArray.shape</c>, <c>np.unique</c>'s result)
        ///     return copies too: writing into what they return does not change the wrapped array.
        /// </summary>
        [TestMethod]
        public void Wrappers_ReturnCopies()
        {
            using var data = np.zeros(new Shape(2, 3));
            var masked = np.ma.array(data);
            var md = masked.shape;
            md[0] = 7;
            masked.shape.Should().Equal(new long[] { 2, 3 });
            data.shape.Should().Equal(new long[] { 2, 3 });

            var u = np.unique(np.array(new[] { 3, 1, 3, 2 }));
            var ud = u.shape;
            ud[0] = 99;
            u.shape.Should().Equal(new long[] { 3 });
        }

        /// <summary>
        ///     The setter reads the caller's dimensions once (NumPy's <c>a.shape = s; s[0] = 100</c> leaves <c>a</c> alone).
        ///     This already held; it guards the setter while the getter changes.
        /// </summary>
        [TestMethod]
        public void NDArrayShape_Setter_ReadsTheCallersArrayOnce()
        {
            using var a = np.arange(6);
            long[] dims = { 3, 2 };
            a.shape = dims;
            dims[0] = 100;

            a.shape.Should().Equal(new long[] { 3, 2 });
        }
    }
}
