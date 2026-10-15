using System;
using System.Collections;
using System.Linq;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     A scalar <c>bool</c> index is NumPy's <c>HAS_0D_BOOL</c>: an ADVANCED index (prepare_index turns it into a zeros
    ///     index array of length 1 / 0 on a new axis), so <c>a[True]</c> is a fresh array of shape <c>(1, ...)</c> and
    ///     <c>a[False]</c> a fresh empty one of shape <c>(0, ...)</c> — never a view of <c>a</c>. Every expectation was probed
    ///     against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by the numpy.char phase-2 oracle (2026-10-15): <c>a[true]</c> returned an <c>expand_dims</c> VIEW sharing
    ///     <c>a</c>'s memory, <c>a[false]</c> returned the dims-less sentinel array (a NullReferenceException at its first use),
    ///     and a string array's index counted the bool as an axis (a 0-d string array refused <c>s[true]</c> as "too many
    ///     indices").
    /// </remarks>
    [TestClass]
    public class ScalarBoolIndexTests
    {
        /// <summary><c>a[True]</c> is a copy with a new leading axis of length 1; <c>a[False]</c> a fresh empty array of
        ///     shape <c>(0, ...)</c>.</summary>
        [TestMethod]
        public void ScalarBool_IsACopy_WithANewAxis()
        {
            var a = np.arange(12).reshape(3, 4);

            var yes = a[new object[] { true }];
            yes.shape.Should().Equal(1L, 3L, 4L);
            yes.flags.owndata.Should().BeTrue();
            np.shares_memory(yes, a).Should().BeFalse();
            yes.ToArray<long>().Should().Equal(Enumerable.Range(0, 12).Select(i => (long)i));

            var no = a[new object[] { false }];
            no.shape.Should().Equal(0L, 3L, 4L);
            no.dtype.Should().Be(np.int64);
            no.flags.owndata.Should().BeTrue();
        }

        /// <summary>
        ///     The copy keeps the SOURCE's memory order (MapIter's subspace): <c>a.T[True]</c> of a C (3, 4) block is (1, 4, 3)
        ///     F-ordered on its last two axes, as NumPy's (12, 1, 4) element strides.
        /// </summary>
        [TestMethod]
        public void ScalarBool_KeepsTheSourcesMemoryOrder()
        {
            var t = np.arange(12).reshape(3, 4).T;
            var r = t[new object[] { true }];
            r.shape.Should().Equal(1L, 4L, 3L);
            r.Shape.strides[1].Should().Be(1);
            r.Shape.strides[2].Should().Be(4);
            r.flags.f_contiguous.Should().BeTrue();
        }

        /// <summary>A 0-d array and a 0-d string array accept the scalar bool (it consumes no axis): <c>(1,)</c> / <c>(0,)</c>.</summary>
        [TestMethod]
        public void ScalarBool_OfAZeroDArray_AddsTheAxis()
        {
            var z = NDArray.Scalar(5L);
            z[new object[] { true }].shape.Should().Equal(1L);
            z[new object[] { false }].shape.Should().Equal(0L);

            var s = np.array("ab ");
            var r = s[new object[] { true }];
            r.shape.Should().Equal(1L);
            ((IList)r.tolist()).Cast<object>().Should().Equal("ab ");
        }

        /// <summary>A scalar bool beside basic indices is still advanced: <c>a[True, :]</c> and <c>a[True, 0]</c> are copies,
        ///     <c>a[False, :]</c> a fresh empty array.</summary>
        [TestMethod]
        public void ScalarBool_BesideBasicIndices_IsACopy()
        {
            var a = np.arange(12).reshape(3, 4);

            var full = a[new object[] { np.array(true), Slice.All }];
            full.shape.Should().Equal(1L, 3L, 4L);
            np.shares_memory(full, a).Should().BeFalse();

            var row = a[new object[] { np.array(true), 0 }];
            row.shape.Should().Equal(1L, 4L);
            np.shares_memory(row, a).Should().BeFalse();

            var none = a[new object[] { np.array(false), Slice.All }];
            none.shape.Should().Equal(0L, 3L, 4L);
            none.flags.owndata.Should().BeTrue();
        }
    }
}
