using System;
using System.Collections.Generic;
using NumSharp.Generic;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     NumPy's element access over NumSharp's scalar model (<c>NDArray.ScalarApi.cs</c>, the indexers): a full integer
    ///     index is an ITEM (<c>array_item</c> / <c>PyArray_Scalar</c>), NumPy's scalar, a read-only 0-d COPY of the element;
    ///     any other basic index is a view (a 0-d one is an explicit 0-d array); and every index of a NumPy scalar indexes the
    ///     0-d array of its value (<c>gentype_subscript</c>). Every expectation is NumPy 2.4.2's, probed live.
    /// </summary>
    [TestClass]
    public class NDArrayItemScalarTests
    {
        /// <summary>
        ///     <c>x = a[2]; a[2] = 99</c> leaves <c>x</c> at 2.0: the item is a read-only scalar holding a copy
        ///     (<c>type(a[2])</c> is <c>np.float64</c>, <c>flags.writeable</c> False).
        /// </summary>
        [TestMethod]
        public void Item_IsAReadOnlyCopy_ThatTheArraysWritesDoNotReach()
        {
            var a = np.arange(5.0);
            var item = a[2];

            item.IsNumPyScalar.Should().BeTrue();
            item.flags.writeable.Should().BeFalse();
            item.flags.owndata.Should().BeTrue("a scalar owns its value");
            np.shares_memory(item, a).Should().BeFalse();

            a[2] = 99.0;
            ((double)item).Should().Be(2.0);
            ((double)a[2]).Should().Be(99.0);
        }

        /// <summary>
        ///     A typed array's item is a typed scalar copy (<c>(a &gt; 1)[3]</c> is <c>np.True_</c>), and through an
        ///     <see cref="NDArray"/> reference the typed getter still answers.
        /// </summary>
        [TestMethod]
        public void TypedItem_IsATypedReadOnlyCopy()
        {
            var a = np.arange(5.0);
            NDArray<bool> mask = a > 1.0;
            NDArray<bool> item = mask[3];

            item.GetType().Should().Be(typeof(NDArray<bool>));
            item.IsNumPyScalar.Should().BeTrue();
            item.flags.writeable.Should().BeFalse();
            ((bool)item).Should().BeTrue();

            NDArray plain = np.arange(6).reshape(2, 3).MakeGeneric<long>();
            var element = plain[1, 2];
            element.GetType().Should().Be(typeof(NDArray<long>));
            element.flags.writeable.Should().BeFalse();
            ((long)element).Should().Be(5L);
        }

        /// <summary>
        ///     The item's dtype is its class's own descriptor (<c>PyArray_DescrFromScalar</c>): an <c>'i'</c> array's item
        ///     stays <c>np.intc</c> (char <c>'i'</c>), typed or not, and a descriptor's metadata stays with the array
        ///     (<c>a[0].dtype.metadata</c> is None while <c>a.dtype.metadata</c> keeps the dict).
        /// </summary>
        [TestMethod]
        public void ItemDtype_IsTheClassDescriptor_WithoutTheArraysMetadata()
        {
            var intc = np.zeros(new Shape(3), np.dtype("i"));
            intc[0].dtype.@char.Should().Be('i');
            intc.MakeGeneric<int>()[1].dtype.@char.Should().Be('i');

            var tagged = np.dtype("f8", metadata: new Dictionary<string, object> { ["k"] = 1 });
            var a = np.zeros(new Shape(2), tagged);
            a.dtype.metadata.Should().NotBeNull();
            a[0].dtype.metadata.Should().BeNull();
            a[0].dtype.Should().Be(np.dtype("f8"));
        }

        /// <summary>
        ///     A bool item is NumPy's canonical True / False whatever nonzero byte the element holds
        ///     (<c>np.frombuffer(bytes([0, 2, 0]), bool)[1]</c> is True, its byte 1).
        /// </summary>
        [TestMethod]
        public void BoolItem_IsCanonical()
        {
            var raw = np.array(new byte[] { 0, 2, 0 }).view(np.@bool);
            var item = raw[1];
            ((bool)item).Should().BeTrue();
            ((byte)np.array(item).view(np.uint8)).Should().Be(1);
        }

        /// <summary>
        ///     An out-of-bounds item raises NumPy's text, the index as given and the axis in the array
        ///     (<c>a[1, -5]</c> of a (3, 4) array: <c>index -5 is out of bounds for axis 1 with size 4</c>).
        /// </summary>
        [TestMethod]
        public void Item_OutOfBounds_IsNumPysIndexError()
        {
            var a = np.arange(12).reshape(3, 4);
            new Action(() => _ = a[3, 0]).Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            new Action(() => _ = a[1, -5]).Should().Throw<IndexError>().WithMessage("index -5 is out of bounds for axis 1 with size 4");
            new Action(() => _ = np.arange(3)[5]).Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 3");
            var typed = a.MakeGeneric<long>();
            new Action(() => _ = typed[1, 4]).Should().Throw<IndexError>().WithMessage("index 4 is out of bounds for axis 1 with size 4");
        }

        /// <summary>
        ///     An index that is not an item stays a view, and a 0-d one is an explicit 0-d ARRAY that writes through
        ///     (<c>a[1, 1, ...]</c> is <c>array(5)</c>, writeable, its base <c>a</c>; <c>z[...]</c> of <c>np.array(5)</c>).
        /// </summary>
        [TestMethod]
        public void ZeroDimView_IsAnExplicitArray_ThatWritesThrough()
        {
            var a = np.arange(12).reshape(3, 4);
            var view = a[1, 1, Slice.Ellipsis];
            view.ndim.Should().Be(0);
            view.IsNumPyScalar.Should().BeFalse();
            view.flags.writeable.Should().BeTrue();
            view.fill(77);
            ((long)a[1, 1]).Should().Be(77L);

            var z = np.array(5L);
            var zview = z[Slice.Ellipsis];
            zview.IsNumPyScalar.Should().BeFalse();
            zview.fill(7L);
            ((long)z).Should().Be(7L);
            z[new object[0]].IsNumPyScalar.Should().BeTrue("z[()] of a 0-d array is its item");
        }

        /// <summary>
        ///     Every key of a NumPy scalar indexes the 0-d array of its value (<c>gentype_subscript</c>): <c>s[...]</c> is a
        ///     writeable <c>array(3.)</c> over a copy, <c>s[None]</c> and <c>s[np.array(True)]</c> are <c>array([3.])</c>,
        ///     <c>s[()]</c> is the scalar again, and a refused key is <c>IndexError: invalid index to scalar variable.</c>
        /// </summary>
        [TestMethod]
        public void ScalarSubscript_IndexesACopyOfTheValue()
        {
            var s = np.sum(np.arange(3.0));
            s.IsNumPyScalar.Should().BeTrue();

            var all = s[Slice.Ellipsis];
            all.ndim.Should().Be(0);
            all.IsNumPyScalar.Should().BeFalse();
            all.flags.writeable.Should().BeTrue();
            all.flags.owndata.Should().BeFalse();
            all.fill(5.0);
            ((double)s).Should().Be(3.0, "the scalar is immutable: s[...] views a copy");

            s[Slice.NewAxis].Should().BeShaped(1).And.BeOfValues(3.0);
            s[np.array(true)].Should().BeShaped(1).And.BeOfValues(3.0);
            s[np.array(false)].Should().BeShaped(0);
            s[new object[0]].IsNumPyScalar.Should().BeTrue();

            foreach (var key in new object[] { 0, -1, ":", 1.5, new[] { 0 }, np.array(0L) })
                new Action(() => _ = s[key]).Should().Throw<IndexError>().WithMessage("invalid index to scalar variable.");
        }

        /// <summary>
        ///     A typed scalar answers the same way, typed: <c>t[...]</c> is a writeable explicit 0-d typed array, <c>t[()]</c>
        ///     stays a scalar, and a refused key raises the scalar text.
        /// </summary>
        [TestMethod]
        public void TypedScalarSubscript_IsTyped()
        {
            var t = np.arange(4).reshape(2, 2).MakeGeneric<long>()[1, 1];
            t.IsNumPyScalar.Should().BeTrue();

            var all = t[Slice.Ellipsis];
            all.GetType().Should().Be(typeof(NDArray<long>));
            all.IsNumPyScalar.Should().BeFalse();
            all.flags.writeable.Should().BeTrue();

            var item = t[new object[0]];
            item.GetType().Should().Be(typeof(NDArray<long>));
            item.IsNumPyScalar.Should().BeTrue();
            ((long)item).Should().Be(3L);

            new Action(() => _ = t[0]).Should().Throw<IndexError>().WithMessage("invalid index to scalar variable.");
        }

        /// <summary>
        ///     A matrix element is NumPy's scalar too (<c>matrix.__getitem__</c> returns ndarray's item unwrapped):
        ///     <c>m[0, 1]</c> is a read-only <c>np.float64(1.0)</c> that a later <c>m[0, 1] = 50</c> does not reach, and the
        ///     general indexer's full integer index is the same copy as a 0-d matrix.
        /// </summary>
        [TestMethod]
        public void MatrixElement_IsAReadOnlyCopy()
        {
            var m = np.asmatrix(np.arange(12.0).reshape(3, 4));
            NDArray e = m[0, 1];
            e.flags.writeable.Should().BeFalse();
            ((double)e).Should().Be(1.0);

            var general = m["0, 1"];
            general.ndim.Should().Be(0);
            general.flags.writeable.Should().BeFalse();

            m[0, 1] = 50.0;
            ((double)e).Should().Be(1.0);
            ((double)general).Should().Be(1.0);
            ((double)m[0, 1]).Should().Be(50.0);
        }

        /// <summary>
        ///     The reductions' broadcast collapse views the unique data with the library's own view
        ///     (<see cref="NDArray.BasicIndexView"/>), not the indexer: a broadcast of a 0-d value indexes every axis, and
        ///     the sums keep NumPy's values (<c>np.sum(np.broadcast_to(True, (3, 4)))</c> is 12).
        /// </summary>
        [TestMethod]
        public void BroadcastOfAZeroDimValue_ReducesThroughAView()
        {
            var b = np.broadcast_to(np.array(true), new Shape(3, 4));
            ((long)np.sum(b)).Should().Be(12L);
            ((double)np.mean(b)).Should().Be(1.0);
            ((long)np.nansum(np.broadcast_to(NDArray.Scalar(true), new Shape(2, 5)))).Should().Be(10L);

            var view = np.arange(6).reshape(2, 3).BasicIndexView(Slice.Index(1), Slice.Index(2));
            view.ndim.Should().Be(0);
            view.flags.writeable.Should().BeTrue();
            view.flags.owndata.Should().BeFalse();
        }
    }
}
