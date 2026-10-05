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
    }
}
