using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     NumPy's index tricks (<c>lib/_index_tricks_impl.py</c>) as NumPy 2.4.2 runs them, found when the np-layer dtype
    ///     plan's P3 batch 7 moved their statements into <c>NDArray.IndexTricks.cs</c> and replayed the snapshot (11,514 cases)
    ///     against NumPy on Windows and on Linux: <c>np.ix_</c>'s result object for one sequence and its empty-sequence rule,
    ///     the C-order copy of <c>np.r_</c> / <c>np.c_</c>'s matrix directive and the setitem conversion of a Python int
    ///     entry. Every expectation is NumPy 2.4.2's output; where the platform's C <c>long</c> changes NumPy's text, Linux
    ///     NumPy's is the one NumSharp pins.
    /// </summary>
    [TestClass]
    public class index_tricks_numpy_port_tests
    {
        /// <summary>A single sequence gets no axis, and NumPy's reshape to the same shape is still a new view.</summary>
        [TestMethod]
        public void Ix_SingleSequence_IsANewView()
        {
            // a = np.arange(3); r = np.ix_(a)[0] -> r is a: False, r.flags.owndata: False, np.shares_memory(r, a): True
            var a = np.arange(3L);
            var r = np.ix_(a)[0];
            ReferenceEquals(r, a).Should().BeFalse();
            r.flags.owndata.Should().BeFalse();
            np.shares_memory(r, a).Should().BeTrue();
            // r[0] = 9 writes through: a is [9, 1, 2]
            r[0] = 9L;
            a.ToArray<long>().Should().Equal(9L, 1L, 2L);

            // np.ix_([1, 2])[0].flags.owndata -> False: the array a list became is not handed out either
            var fromList = np.ix_(new object[] { new object[] { 1, 2 } })[0];
            fromList.flags.owndata.Should().BeFalse();
            fromList.ToArray<long>().Should().Equal(1L, 2L);

            // np.ix_(np.array([1, 2], np.int32))[0] -> array([1, 2], dtype=int32), owndata False
            var typed = np.ix_(new[] { 1, 2 })[0];
            typed.dtype.Should().Be(np.int32);
            typed.flags.owndata.Should().BeFalse();
        }

        /// <summary>Only a sequence that is not an ndarray is cast to intp when it comes out empty.</summary>
        [TestMethod]
        public void Ix_EmptySequence_OnlyAListBecomesIntp()
        {
            // np.ix_(np.array([], np.int32))[0].dtype -> int32; np.ix_(np.array([], np.float64))[0].dtype -> float64
            np.ix_(new int[0])[0].dtype.Should().Be(np.int32);
            np.ix_(new double[0])[0].dtype.Should().Be(np.float64);
            // np.ix_([])[0].dtype -> int64 (intp)
            np.ix_(new object[] { new object[0] })[0].dtype.Should().Be(np.int64);
        }

        /// <summary>The matrix directives make NumPy's <c>matrix(res)</c>, whose copy is C-ordered.</summary>
        [TestMethod]
        public void MatrixDirective_CopiesTheConcatenationInCOrder()
        {
            // np.r_['r', np.asfortranarray(np.arange(6).reshape(2, 3))] -> matrix([[0, 1, 2], [3, 4, 5]]),
            // flags C True, F False, OWNDATA True; 'c', and np.c_ with either directive, the same
            var f = np.asfortranarray(np.arange(6L).reshape(2, 3));
            foreach (var m in new[] { np.r_["r", f], np.r_["c", f], np.c_["r", f], np.c_["c", f] })
            {
                m.GetType().Should().Be(typeof(NDMatrix));
                m.flags.c_contiguous.Should().BeTrue();
                m.flags.f_contiguous.Should().BeFalse();
                m.flags.owndata.Should().BeTrue();
                m.Should().BeShaped(2, 3).And.BeOfValues(0L, 1L, 2L, 3L, 4L, 5L);
            }

            // np.r_['c', [1, 2, 3]] -> a (3, 1) matrix, the transpose of the row the copy made: C and F, OWNDATA False
            var col = np.r_["c", np.array(new long[] { 1, 2, 3 })];
            col.GetType().Should().Be(typeof(NDMatrix));
            col.Should().BeShaped(3, 1);
            col.flags.c_contiguous.Should().BeTrue();
            col.flags.f_contiguous.Should().BeTrue();
            col.flags.owndata.Should().BeFalse();
        }

        /// <summary>A Python int entry is stored by the result dtype's setitem: range-checked, never wrapped.</summary>
        [TestMethod]
        public void PythonIntEntry_IsStoredBySetitem()
        {
            // np.r_[np.array([1], np.int8), 1000] -> OverflowError: Python integer 1000 out of bounds for int8
            Action int8 = () => { _ = np.r_[np.array(new sbyte[] { 1 }), 1000]; };
            int8.Should().ThrowExactly<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
            // np.r_[np.array([1], np.uint8), -1] -> OverflowError: Python integer -1 out of bounds for uint8
            Action uint8 = () => { _ = np.r_[np.array(new byte[] { 1 }), -1]; };
            uint8.Should().ThrowExactly<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
            // np.r_[np.array([1], np.int8), 2**64 - 1] -> OverflowError: Python int too large to convert to C long
            Action past = () => { _ = np.r_[np.array(new sbyte[] { 1 }), ulong.MaxValue]; };
            past.Should().ThrowExactly<OverflowException>().WithMessage("Python int too large to convert to C long");
            // np.r_[np.array([True]), 2**64 - 1] (int64, the default int a bool adopts) -> the same text
            Action fromBool = () => { _ = np.r_[np.array(new[] { true }), ulong.MaxValue]; };
            fromBool.Should().ThrowExactly<OverflowException>().WithMessage("Python int too large to convert to C long");
            // Linux: np.r_[np.array([1], np.uint32), 2**64 - 1] -> Python integer 18446744073709551615 out of bounds for uint32
            // (uint32 converts through C unsigned long, which holds 2**64 - 1)
            Action uint32 = () => { _ = np.r_[np.array(new uint[] { 1 }), ulong.MaxValue]; };
            uint32.Should().ThrowExactly<OverflowException>()
                .WithMessage("Python integer 18446744073709551615 out of bounds for uint32");

            // np.r_[np.array([1], np.uint64), 2**64 - 1] -> [1, 18446744073709551615]
            np.r_[np.array(new ulong[] { 1 }), ulong.MaxValue].ToArray<ulong>().Should().Equal(1UL, ulong.MaxValue);
            // np.r_[np.array([1], np.int8), 100] -> int8 [1, 100]; np.r_[np.array([True]), 2] -> int64 [1, 2]
            var small = np.r_[np.array(new sbyte[] { 1 }), 100];
            small.dtype.Should().Be(np.int8);
            small.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)100);
            var lifted = np.r_[np.array(new[] { true }), 2];
            lifted.dtype.Should().Be(np.int64);
            lifted.ToArray<long>().Should().Equal(1L, 2L);
            // np.r_[np.array([1], np.float16), 70000] -> float16 [1., inf]
            var half = np.r_[np.array(new[] { (Half)1 }), 70000];
            half.dtype.Should().Be(np.float16);
            half.ToArray<Half>().Should().Equal((Half)1, Half.PositiveInfinity);
        }
    }
}
