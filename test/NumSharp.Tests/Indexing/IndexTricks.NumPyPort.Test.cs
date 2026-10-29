using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     NumPy's index tricks (<c>lib/_index_tricks_impl.py</c>) as NumPy 2.4.2 runs them, found when the np-layer dtype
    ///     plan's P3 batch 7 moved their statements into <c>NDArray.IndexTricks.cs</c> and replayed the snapshot (11,514 cases)
    ///     against NumPy on Windows and on Linux: <c>np.ix_</c>'s result object for one sequence and its empty-sequence rule,
    ///     the C-order copy of <c>np.r_</c> / <c>np.c_</c>'s matrix directive, the setitem conversion of a Python int entry,
    ///     and <c>np.fill_diagonal</c>'s value conversion (NumPy's <c>PyArray_FromAny(val, a.dtype, FORCECAST)</c>, made before
    ///     the slice is walked). Every expectation is NumPy 2.4.2's output; where the platform's C <c>long</c> changes NumPy's
    ///     text, Linux NumPy's is the one NumSharp pins.
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

        /// <summary>A str value is parsed by the dtype's setitem: Python's int() / float() / complex(), quoting the str.</summary>
        [TestMethod]
        public void FillDiagonal_PythonStr_IsParsedBySetitem()
        {
            // np.fill_diagonal(np.zeros((3, 3), np.int8), 'x') -> ValueError: invalid literal for int() with base 10: 'x'
            Action int8 = () => np.fill_diagonal(np.zeros(new Shape(3, 3), np.int8), "x");
            int8.Should().ThrowExactly<ValueError>().WithMessage("invalid literal for int() with base 10: 'x'");
            // float64: could not convert string to float: 'x'; complex128: complex() arg is a malformed string
            Action float64 = () => np.fill_diagonal(np.zeros(new Shape(3, 3), np.float64), "x");
            float64.Should().ThrowExactly<ValueError>().WithMessage("could not convert string to float: 'x'");
            Action complex = () => np.fill_diagonal(np.zeros(new Shape(3, 3), np.complex128), "x");
            complex.Should().ThrowExactly<ValueError>().WithMessage("complex() arg is a malformed string");

            // np.fill_diagonal(np.zeros((3, 3), np.int64), '7') -> diagonal 7; float64 '2.5' -> diagonal 2.5
            var i64 = np.zeros(new Shape(3, 3), np.int64);
            np.fill_diagonal(i64, "7");
            np.diagonal(i64).ToArray<long>().Should().Equal(7L, 7L, 7L);
            i64.GetInt64(0, 1).Should().Be(0L);
            var f64 = np.zeros(new Shape(3, 3), np.float64);
            np.fill_diagonal(f64, "2.5");
            np.diagonal(f64).ToArray<double>().Should().Equal(2.5, 2.5, 2.5);

            // bool('x') is True, bool('') False
            var truthy = np.zeros(new Shape(3, 3), np.@bool);
            np.fill_diagonal(truthy, "x");
            np.diagonal(truthy).ToArray<bool>().Should().Equal(true, true, true);
            var falsy = np.ones(new Shape(3, 3), np.@bool);
            np.fill_diagonal(falsy, "");
            np.diagonal(falsy).ToArray<bool>().Should().Equal(false, false, false);
        }

        /// <summary>The value is converted before the slice is walked, so an empty diagonal still raises.</summary>
        [TestMethod]
        public void FillDiagonal_EmptyArray_StillConvertsTheValue()
        {
            // np.fill_diagonal(np.zeros((0, 3), np.int8), 'x') -> ValueError: invalid literal for int() with base 10: 'x'
            Action text = () => np.fill_diagonal(np.zeros(new Shape(0, 3), np.int8), "x");
            text.Should().ThrowExactly<ValueError>().WithMessage("invalid literal for int() with base 10: 'x'");
            // np.fill_diagonal(np.zeros((0, 3), np.int8), [300]) -> OverflowError: Python integer 300 out of bounds for int8
            Action list = () => np.fill_diagonal(np.zeros(new Shape(0, 3), np.int8), new object[] { 300 });
            list.Should().ThrowExactly<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
            // np.fill_diagonal(np.zeros((0, 3), np.int32), []) -> no error (an empty value assigns nothing)
            np.fill_diagonal(np.zeros(new Shape(0, 3), np.int32), new object[0]);
        }

        /// <summary>A Python list stores every item by the setitem; an ndarray value is cast.</summary>
        [TestMethod]
        public void FillDiagonal_PythonList_StoresEveryItemBySetitem()
        {
            // np.fill_diagonal(np.zeros((3, 3), np.int8), [300, 2]) -> OverflowError: Python integer 300 out of bounds for int8
            Action overflow = () => np.fill_diagonal(np.zeros(new Shape(3, 3), np.int8), new object[] { 300, 2 });
            overflow.Should().ThrowExactly<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
            // an ndarray is cast: np.fill_diagonal(np.zeros((3, 3), np.int8), np.array([300, 2])) -> diagonal [44, 2, 44]
            var cast = np.zeros(new Shape(3, 3), np.int8);
            np.fill_diagonal(cast, np.array(new long[] { 300, 2 }));
            np.diagonal(cast).ToArray<sbyte>().Should().Equal((sbyte)44, (sbyte)2, (sbyte)44);

            // np.fill_diagonal(np.zeros((3, 3)), [1.5, None]) -> diagonal [1.5, nan, 1.5]
            var withNone = np.zeros(new Shape(3, 3), np.float64);
            np.fill_diagonal(withNone, new object[] { 1.5, null });
            withNone.GetDouble(0, 0).Should().Be(1.5);
            double.IsNaN(withNone.GetDouble(1, 1)).Should().BeTrue();
            withNone.GetDouble(2, 2).Should().Be(1.5);
            // np.fill_diagonal(np.zeros((3, 3), np.int64), [1.5, 2.5]) -> diagonal [1, 2, 1] (int() truncates a float)
            var truncated = np.zeros(new Shape(3, 3), np.int64);
            np.fill_diagonal(truncated, new object[] { 1.5, 2.5 });
            np.diagonal(truncated).ToArray<long>().Should().Equal(1L, 2L, 1L);
            // np.fill_diagonal(np.zeros((3, 3), np.int64), [1.5, np.nan]) -> ValueError: cannot convert float NaN to integer
            Action nan = () => np.fill_diagonal(np.zeros(new Shape(3, 3), np.int64), new object[] { 1.5, double.NaN });
            nan.Should().ThrowExactly<ValueError>().WithMessage("cannot convert float NaN to integer");
            // np.fill_diagonal(np.zeros((3, 3), np.int32), [None]) -> TypeError: int() argument must be ... not 'NoneType'
            Action none = () => np.fill_diagonal(np.zeros(new Shape(3, 3), np.int32), new object[] { null });
            none.Should().ThrowExactly<TypeError>()
                .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
        }

        /// <summary>Into a text array a Python scalar stores its str(), a Python int past int64 included.</summary>
        [TestMethod]
        public void FillDiagonal_PythonScalar_IntoText_StoresItsStr()
        {
            // a = np.zeros((2, 2), 'S5'); np.fill_diagonal(a, 2**70) -> [[b'11805', b''], [b'', b'11805']]
            var bytes = np.zeros(new Shape(2, 2), np.dtype("S5"));
            np.fill_diagonal(bytes, BigInteger.Pow(2, 70));
            np.array_equal(bytes, np.array(new[] { "11805", "", "", "11805" }, np.dtype("S5")).reshape(2, 2))
                .Should().BeTrue();
            // 'U5' -> [['11805', ''], ['', '11805']]
            var text = np.zeros(new Shape(2, 2), np.dtype("U5"));
            np.fill_diagonal(text, BigInteger.Pow(2, 70));
            np.array_equal(text, np.array(new[] { "11805", "", "", "11805" }, np.dtype("U5")).reshape(2, 2))
                .Should().BeTrue();
            // np.fill_diagonal(np.zeros((2, 2), 'S5'), 1.5) -> [[b'1.5', b''], [b'', b'1.5']]
            var floats = np.zeros(new Shape(2, 2), np.dtype("S5"));
            np.fill_diagonal(floats, 1.5);
            np.array_equal(floats, np.array(new[] { "1.5", "", "", "1.5" }, np.dtype("S5")).reshape(2, 2))
                .Should().BeTrue();
            // np.fill_diagonal(np.zeros((3, 3), 'U5'), [300, 'ab']) -> diagonal ['300', 'ab', '300']
            var mixed = np.zeros(new Shape(3, 3), np.dtype("U5"));
            np.fill_diagonal(mixed, new object[] { 300, "ab" });
            np.array_equal(np.diagonal(mixed), np.array(new[] { "300", "ab", "300" }, np.dtype("U5"))).Should().BeTrue();
        }
    }
}
