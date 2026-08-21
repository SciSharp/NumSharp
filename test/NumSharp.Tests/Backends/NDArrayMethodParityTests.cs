using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     ndarray instance-method parity with NumPy 2.4.2. Each of these methods mirrors an
    ///     <c>np.*</c> free function (all/any/clip/compress/cumprod/diagonal/nonzero/put/repeat/
    ///     round/searchsorted/squeeze/take/trace), so the contract is twofold: the instance method
    ///     must (a) return exactly what the corresponding <c>np.*</c> function returns — those are
    ///     already differentially fuzzed against NumPy — and (b) reproduce NumPy's probed values,
    ///     shapes, dtypes and overload behavior. conj/conjugate/cumsum already existed and are
    ///     covered elsewhere; the sibling check here only guards against accidental removal.
    /// </summary>
    [TestClass]
    public class NDArrayMethodParityTests
    {
        /// <summary>Assert an instance-method result is byte-for-byte the free-function result (shape + dtype + values).</summary>
        private static void SameAsFree(NDArray inst, NDArray free)
        {
            inst.Shape.dimensions.Should().BeEquivalentTo(free.Shape.dimensions, "instance method must reproduce the np.* shape");
            inst.typecode.Should().Be(free.typecode, "instance method must reproduce the np.* dtype");
            np.array_equal(inst, free).Should().BeTrue("instance method must reproduce the np.* values");
        }

        // ---------------- all / any ----------------

        [TestMethod]
        public void All_MatchesFree_And_Numpy()
        {
            var b = np.array(new[] { 1, 0, 3, 0, 5, 6 }).reshape(2, 3);
            SameAsFree(b.all(1), np.all(b, 1));
            SameAsFree(b.all(0, null, true, null), np.all(b, 0, null, true, null));
            // NumPy: b.all() -> False (0-d), b.all(axis=1) -> [False, False]
            b.all().Should().BeOfValues(false);
            b.all().Should().BeOfType(NPTypeCode.Boolean);
            b.all(1).Should().BeShaped(2);
            b.all(1).Should().BeOfValues(false, false);
        }

        [TestMethod]
        public void Any_MatchesFree_And_Numpy()
        {
            var b = np.array(new[] { 1, 0, 3, 0, 5, 6 }).reshape(2, 3);
            SameAsFree(b.any(0), np.any(b, 0));
            // NumPy: b.any(axis=0) -> [True, True, True]
            b.any(0).Should().BeOfValues(true, true, true);
            b.any().Should().BeOfValues(true);
        }

        // ---------------- clip ----------------

        [TestMethod]
        public void Clip_MatchesFree_And_Numpy()
        {
            var c = np.arange(10);
            SameAsFree(c.clip(2, 7), np.clip(c, 2, 7));
            SameAsFree(c.clip(min: 3), np.clip(c, a_min: 3));
            SameAsFree(c.clip(max: 4), np.clip(c, a_max: 4));
            // NumPy: np.arange(10).clip(2,7) -> [2,2,2,3,4,5,6,7,7,7]
            c.clip(2, 7).Should().BeOfValues(2, 2, 2, 3, 4, 5, 6, 7, 7, 7);
        }

        // ---------------- compress ----------------

        [TestMethod]
        public void Compress_MatchesFree_And_Numpy()
        {
            var m = np.arange(6).reshape(3, 2);
            var cond = np.array(new[] { true, false, true });
            SameAsFree(m.compress(cond, 0), np.compress(cond, m, 0));
            // NumPy: rows 0 and 2 -> [[0,1],[4,5]]
            m.compress(cond, 0).Should().BeShaped(2, 2);
            m.compress(cond, 0).Should().BeOfValues(0, 1, 4, 5);
        }

        // ---------------- cumprod (the ⚠ sibling gap) ----------------

        [TestMethod]
        public void Cumprod_MatchesFree_And_Numpy()
        {
            var d = np.arange(1, 7).reshape(2, 3);   // int64 (np.arange default, matching NumPy)
            SameAsFree(d.cumprod(), np.cumprod(d));
            SameAsFree(d.cumprod(1), np.cumprod(d, 1));
            SameAsFree(d.cumprod(0, typeof(double)), np.cumprod(d, 0, NPTypeCode.Double));
            // NumPy: flat cumprod -> [1,2,6,24,120,720] (int64 preserved)
            d.cumprod().Should().BeOfValues(1, 2, 6, 24, 120, 720);
            d.cumprod().Should().BeOfType(NPTypeCode.Int64);
            // axis=1 -> [[1,2,6],[4,20,120]]
            d.cumprod(1).Should().BeOfValues(1, 2, 6, 4, 20, 120);
        }

        [TestMethod]
        public void Cumprod_Int32_PromotesToInt64_NEP50()
        {
            // NumPy: cumprod of a 32-bit integer array accumulates in int64 (NEP50), exactly like
            // the cumsum sibling. This is the behavior the instance method must inherit.
            var i32 = np.array(new[] { 1, 2, 3, 4 });   // int32
            i32.cumprod().Should().BeOfType(NPTypeCode.Int64);
            i32.cumprod().Should().BeOfValues(1, 2, 6, 24);
            SameAsFree(i32.cumprod(), np.cumprod(i32));
        }

        [TestMethod]
        public void Cumsum_Sibling_StillPresent()
        {
            var d = np.arange(1, 7).reshape(2, 3);
            SameAsFree(d.cumsum(1), np.cumsum(d, 1));
        }

        // ---------------- diagonal ----------------

        [TestMethod]
        public void Diagonal_MatchesFree_And_Numpy()
        {
            var e = np.arange(9).reshape(3, 3);
            SameAsFree(e.diagonal(), np.diagonal(e));
            SameAsFree(e.diagonal(1), np.diagonal(e, 1));
            // NumPy: main diagonal -> [0,4,8]; offset 1 -> [1,5]
            e.diagonal().Should().BeOfValues(0, 4, 8);
            e.diagonal(1).Should().BeOfValues(1, 5);
        }

        [TestMethod]
        public void Diagonal_ReturnsReadOnlyView()
        {
            // np.diagonal of a 2-D array returns a READ-ONLY view (probed NumPy 2.4.2: writeable
            // False, shares_memory True). A fresh copy would be writeable, so this proves the
            // instance method does not materialize a new buffer.
            var e = np.arange(9).reshape(3, 3);
            e.diagonal().Shape.IsWriteable.Should().BeFalse("ndarray.diagonal returns a read-only view sharing storage");
        }

        // ---------------- nonzero ----------------

        [TestMethod]
        public void Nonzero_MatchesFree_And_Numpy()
        {
            var a = np.array(new[] { 0, 3, 0, 5 });
            var inst = a.nonzero();
            var free = np.nonzero(a);
            inst.Length.Should().Be(free.Length);
            SameAsFree(inst[0], free[0]);
            // NumPy: indices of non-zero -> [1,3]
            inst[0].Should().BeOfValues(1L, 3L);
        }

        // ---------------- put (in-place, void) ----------------

        [TestMethod]
        public void Put_MatchesFree_And_Numpy()
        {
            var p1 = np.arange(6);
            p1.put(np.array(new[] { 0, 2 }), np.array(new[] { 10, 20 }));
            var p2 = np.arange(6);
            np.put(p2, np.array(new[] { 0, 2 }), np.array(new[] { 10, 20 }));
            SameAsFree(p1, p2);
            // NumPy: a.put([0,2],[10,20]) -> [10,1,20,3,4,5]
            p1.Should().BeOfValues(10, 1, 20, 3, 4, 5);
        }

        [TestMethod]
        public void Put_ScalarOverloadViaImplicitConversion()
        {
            var p = np.arange(6);
            p.put(0, 99); // scalars flow through NDArray implicit conversion
            Convert.ToInt32(p.GetAtIndex(0)).Should().Be(99);
        }

        [TestMethod]
        public void Put_WrapMode_MatchesFree()
        {
            var p1 = np.arange(5);
            p1.put(np.array(new[] { 7 }), np.array(new[] { 99 }), mode: "wrap");
            var p2 = np.arange(5);
            np.put(p2, np.array(new[] { 7 }), np.array(new[] { 99 }), "wrap");
            SameAsFree(p1, p2); // index 7 wraps to 2
        }

        // ---------------- repeat ----------------

        [TestMethod]
        public void Repeat_MatchesFree_And_Numpy()
        {
            var r = np.array(new[] { 1, 2, 3 });
            SameAsFree(r.repeat(2), np.repeat(r, 2));
            SameAsFree(r.repeat(2L), np.repeat(r, 2L));
            SameAsFree(r.repeat(np.array(new[] { 1, 2, 3 })), np.repeat(r, np.array(new[] { 1, 2, 3 })));
            // NumPy: [1,2,3].repeat(2) -> [1,1,2,2,3,3]; per-element [1,2,3] -> [1,2,2,3,3,3]
            r.repeat(2).Should().BeOfValues(1, 1, 2, 2, 3, 3);
            r.repeat(np.array(new[] { 1, 2, 3 })).Should().BeOfValues(1, 2, 2, 3, 3, 3);
            // repeat preserves dtype (int32 in, int32 out) like NumPy preserves int64->int64
            r.repeat(2).Should().BeOfType(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Repeat_Axis_MatchesFree()
        {
            var r2 = np.arange(6).reshape(2, 3);
            SameAsFree(r2.repeat(2, 1), np.repeat(r2, 2, 1));
            r2.repeat(2, 1).Should().BeShaped(2, 6);
        }

        // ---------------- round ----------------

        [TestMethod]
        public void Round_MatchesFree_And_Numpy()
        {
            var f = np.array(new[] { 1.234, 5.678, -2.5, 3.5 });
            SameAsFree(f.round(), np.around(f));
            SameAsFree(f.round(2), np.around(f, 2));
            // NumPy round-half-to-even: -2.5 -> -2, 3.5 -> 4
            f.round().Should().BeOfValues(1.0, 6.0, -2.0, 4.0);
            f.round(2).Should().BeOfValues(1.23, 5.68, -2.5, 3.5);
        }

        // ---------------- searchsorted ----------------

        [TestMethod]
        public void Searchsorted_MatchesFree_And_Numpy()
        {
            var s = np.array(new[] { 1, 3, 5, 7, 9 });
            s.searchsorted(4).Should().Be(np.searchsorted(s, 4));
            s.searchsorted(4.5).Should().Be(np.searchsorted(s, 4.5));
            SameAsFree(s.searchsorted(np.array(new[] { 2, 6 }), "right"),
                       np.searchsorted(s, np.array(new[] { 2, 6 }), "right"));
            // NumPy: searchsorted(4)=2 (left); [2,6] right -> [1,3]
            s.searchsorted(4).Should().Be(2L);
            s.searchsorted(np.array(new[] { 2, 6 }), "right").Should().BeOfValues(1L, 3L);
        }

        // ---------------- squeeze ----------------

        [TestMethod]
        public void Squeeze_MatchesFree_And_Numpy()
        {
            var sq = np.arange(3).reshape(1, 3, 1);
            SameAsFree(sq.squeeze(), np.squeeze(sq));
            SameAsFree(sq.squeeze(0), np.squeeze(sq, 0));
            // NumPy: squeeze() -> shape (3,); squeeze(axis=0) -> shape (3,1)
            sq.squeeze().Should().BeShaped(3);
            sq.squeeze(0).Should().BeShaped(3, 1);
        }

        [TestMethod]
        public void Squeeze_ReturnsView_WritesThrough()
        {
            // ndarray.squeeze returns a view; a write to it must land in the source (shared memory).
            var sq = np.arange(3).reshape(1, 3, 1);
            var v = sq.squeeze();
            v.Shape.IsWriteable.Should().BeTrue();
            v[1] = (NDArray)99L;
            Convert.ToInt64(sq.GetAtIndex(1)).Should().Be(99, "squeeze shares storage with the source");
        }

        // ---------------- take ----------------

        [TestMethod]
        public void Take_MatchesFree_And_Numpy()
        {
            var t = np.array(new[] { 10, 20, 30, 40, 50 });
            SameAsFree(t.take(np.array(new[] { 0, 2, 4 })), np.take(t, np.array(new[] { 0, 2, 4 })));
            SameAsFree(t.take(2L), np.take(t, 2L));
            // NumPy: take([0,2,4]) -> [10,30,50]; take(2) -> 30
            t.take(np.array(new[] { 0, 2, 4 })).Should().BeOfValues(10, 30, 50);
            Convert.ToInt32(t.take(2L).GetAtIndex(0)).Should().Be(30);
        }

        [TestMethod]
        public void Take_Axis_MatchesFree()
        {
            var t2 = np.arange(6).reshape(2, 3);
            SameAsFree(t2.take(np.array(new[] { 0, 2 }), 1), np.take(t2, np.array(new[] { 0, 2 }), 1));
            t2.take(np.array(new[] { 0, 2 }), 1).Should().BeShaped(2, 2);
        }

        // ---------------- trace ----------------

        [TestMethod]
        public void Trace_MatchesFree_And_Numpy()
        {
            var g = np.arange(9).reshape(3, 3);
            SameAsFree(g.trace(), np.trace(g));
            SameAsFree(g.trace(1), np.trace(g, 1));
            // NumPy: trace()=12 (0+4+8), trace(offset=1)=6 (1+5)
            Convert.ToInt64(g.trace().GetAtIndex(0)).Should().Be(12);
            Convert.ToInt64(g.trace(1).GetAtIndex(0)).Should().Be(6);
        }

        [TestMethod]
        public void Trace_3D_MatchesFree_And_Numpy()
        {
            var g3 = np.arange(24).reshape(2, 3, 4);
            SameAsFree(g3.trace(0, 1, 2), np.trace(g3, 0, 1, 2));
            // NumPy: g3.trace(0,1,2) reduces axes 1&2 -> [15, 51]
            g3.trace(0, 1, 2).Should().BeShaped(2);
            g3.trace(0, 1, 2).Should().BeOfValues(15, 51);
        }
    }
}
