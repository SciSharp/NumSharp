using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// np.isin parity with NumPy 2.4.2 (element-wise membership, broadcasting over `element` only).
    /// Result is bool with element's shape. Comparison happens in result_type(element, test), matching
    /// NumPy's brute/sort/table methods (kind never changes the RESULT).
    /// </summary>
    [TestClass]
    public class isinTests
    {
        private static void Isin<T>(T[] element, T[] test, bool[] expected, bool invert = false)
            where T : unmanaged
        {
            var r = np.isin(np.array(element), np.array(test), invert: invert);
            r.typecode.Should().Be(NPTypeCode.Boolean);
            r.array_equal(np.array(expected)).Should().BeTrue(
                $"isin<{typeof(T).Name}> got [{string.Join(",", r.ToArray<bool>())}]");
        }

        [TestMethod]
        public void Basic_And_Invert()
        {
            var element = (np.arange(4) * 2).reshape(2, 2);              // [[0,2],[4,6]]
            var test = np.array(new long[] { 1, 2, 4, 8 });
            np.isin(element, test).array_equal(
                np.array(new[] { false, true, true, false }).reshape(2, 2)).Should().BeTrue();
            np.isin(element, test, invert: true).array_equal(
                np.array(new[] { true, false, false, true }).reshape(2, 2)).Should().BeTrue();
        }

        [TestMethod]
        public void ShapePreserved_AndDtypeBool()
        {
            var element = (np.arange(4) * 2).reshape(2, 2);
            var r = np.isin(element, np.array(new long[] { 1, 2, 4, 8 }));
            r.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            r.typecode.Should().Be(NPTypeCode.Boolean);
        }

        [TestMethod]
        public void ScalarElement_Returns0d()
        {
            var r = np.isin((NDArray)5L, np.array(new long[] { 1, 2, 5 }));
            r.ndim.Should().Be(0);
            Convert.ToBoolean(r.GetAtIndex(0)).Should().BeTrue();

            var r2 = np.isin((NDArray)9L, np.array(new long[] { 1, 2, 5 }));
            r2.ndim.Should().Be(0);
            Convert.ToBoolean(r2.GetAtIndex(0)).Should().BeFalse();
        }

        [TestMethod]
        public void EmptyElement_EmptyBoolResult()
        {
            var r = np.isin(np.zeros(new Shape(0), NPTypeCode.Int64), np.array(new long[] { 1, 2 }));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Boolean);
        }

        [TestMethod]
        public void EmptyTest_AllFalse_InvertAllTrue()
        {
            var empty = np.zeros(new Shape(0), NPTypeCode.Int64);
            np.isin(np.array(new long[] { 1, 2, 3 }), empty)
                .array_equal(np.array(new[] { false, false, false })).Should().BeTrue();
            np.isin(np.array(new long[] { 1, 2, 3 }), empty, invert: true)
                .array_equal(np.array(new[] { true, true, true })).Should().BeTrue();
        }

        // ---- all 15 dtypes -----------------------------------------------------------------
        [TestMethod] public void Dtype_Boolean() => Isin(new[] { true, false, true }, new[] { false }, new[] { false, true, false });
        [TestMethod] public void Dtype_Byte() => Isin(new byte[] { 1, 2, 3 }, new byte[] { 2, 3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_SByte() => Isin(new sbyte[] { -1, 2, 3 }, new sbyte[] { 2, -1 }, new[] { true, true, false });
        [TestMethod] public void Dtype_Int16() => Isin(new short[] { 1, 2, 3 }, new short[] { 2, 3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_UInt16() => Isin(new ushort[] { 1, 2, 3 }, new ushort[] { 2, 3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_Int32() => Isin(new[] { 1, 2, 3 }, new[] { 2, 3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_UInt32() => Isin(new uint[] { 1, 2, 3 }, new uint[] { 2, 3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_Int64() => Isin(new long[] { 1, 2, 3 }, new long[] { 2, 3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_UInt64() => Isin(new ulong[] { 1, 2, 3 }, new ulong[] { 2, 3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_Char() => Isin(new[] { 'a', 'b', 'c' }, new[] { 'b', 'c' }, new[] { false, true, true });
        [TestMethod] public void Dtype_Half() => Isin(new[] { (Half)1, (Half)2, (Half)3 }, new[] { (Half)2, (Half)3 }, new[] { false, true, true });
        [TestMethod] public void Dtype_Single() => Isin(new[] { 1f, 2f, 3f }, new[] { 2f, 3f }, new[] { false, true, true });
        [TestMethod] public void Dtype_Double() => Isin(new[] { 1.0, 2.0, 3.0 }, new[] { 2.0, 3.0 }, new[] { false, true, true });
        [TestMethod] public void Dtype_Decimal() => Isin(new[] { 1m, 2m, 3m }, new[] { 2m, 3m }, new[] { false, true, true });
        [TestMethod] public void Dtype_Complex() => Isin(new[] { new Complex(1, 2), new Complex(3, 0), new Complex(0, 1) }, new[] { new Complex(3, 0), new Complex(0, 1) }, new[] { false, true, true });

        // ---- special float values ----------------------------------------------------------
        [TestMethod]
        public void Nan_IsNeverMember()
        {
            // NaN != NaN, so NaN in element is never found even when NaN is in the test set.
            np.isin(np.array(new[] { double.NaN }), np.array(new[] { double.NaN }))
                .array_equal(np.array(new[] { false })).Should().BeTrue();
            np.isin(np.array(new[] { double.NaN, 1.0 }), np.array(new[] { 1.0, double.NaN }))
                .array_equal(np.array(new[] { false, true })).Should().BeTrue();
        }

        [TestMethod]
        public void Inf_And_NegativeZero()
        {
            np.isin(np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, 1.0 }),
                    np.array(new[] { double.PositiveInfinity, 2.0 }))
                .array_equal(np.array(new[] { true, false, false })).Should().BeTrue();
            // -0.0 == 0.0
            np.isin(np.array(new[] { -0.0 }), np.array(new[] { 0.0 }))
                .array_equal(np.array(new[] { true })).Should().BeTrue();
        }

        // ---- mixed dtypes (promotion matches NumPy) ----------------------------------------
        [TestMethod]
        public void Mixed_Int8_In_Int64()
        {
            np.isin(np.array(new sbyte[] { 1, 2, 127 }), np.array(new long[] { 2, 200 }))
                .array_equal(np.array(new[] { false, true, false })).Should().BeTrue();
        }

        [TestMethod]
        public void Mixed_Float_In_Int()
        {
            np.isin(np.array(new[] { 1.5, 2.0 }), np.array(new long[] { 2 }))
                .array_equal(np.array(new[] { false, true })).Should().BeTrue();
        }

        /// <summary>
        /// uint64 paired with a signed integer promotes to float64 (NEP50), which loses precision past
        /// 2^53. Membership between integers must nonetheless be EXACT — two distinct integers that
        /// collide as float64 must not compare equal. Regression pin: uint64(2^63+1) and int64(2^63-1)
        /// are distinct integers that both round to 2^63 as float64.
        /// </summary>
        [TestMethod]
        public void Mixed_UInt64_Signed_ExactInteger()
        {
            // uint64(2^63+1) is NOT in int64([2^63-1]) even though both round to 2^63 as float64.
            np.isin(np.array(new ulong[] { 9223372036854775809UL }), np.array(new long[] { 9223372036854775807L }))
                .array_equal(np.array(new[] { false })).Should().BeTrue();
            // int64(-1) reinterpreted as uint64 is 2^64-1; it must NOT match a genuine uint64 2^64-1.
            np.isin(np.array(new long[] { -1L }), np.array(new ulong[] { 18446744073709551615UL }))
                .array_equal(np.array(new[] { false })).Should().BeTrue();
            // element uint64, test int64: exact hits/misses across the sign boundary.
            np.isin(np.array(new ulong[] { 9223372036854775808UL, 5UL, 7UL }),
                    np.array(new long[] { 5L, 9223372036854775807L }))
                .array_equal(np.array(new[] { false, true, false })).Should().BeTrue();
            // element int64 (with a negative), test uint64.
            np.isin(np.array(new long[] { 5L, -3L, 4611686018427387904L }),
                    np.array(new ulong[] { 5UL, 4611686018427387904UL }))
                .array_equal(np.array(new[] { true, false, true })).Should().BeTrue();
        }

        /// <summary>
        /// Large-range integers decline the counting-table (range > 6·(|element|+|test|)) and fall to the
        /// sort/search path — which must give the identical result. Gates the sort side of the branch for
        /// integer inputs (the small-range side is covered by the dtype tests and the corpus).
        /// </summary>
        [TestMethod]
        public void Integer_LargeRange_UsesSortPath()
        {
            // range = 1_000_000 - 5 >> 6·(2+3), so the table is declined.
            np.isin(np.array(new long[] { 1, 1_000_000, 7, 42 }), np.array(new long[] { 1_000_000, 5, 42 }))
                .array_equal(np.array(new[] { false, true, false, true })).Should().BeTrue();
        }

        // ---- kind parameter (result is identical; only speed/memory differ) -----------------
        [TestMethod]
        public void Kind_Sort_And_Table_SameResult()
        {
            var el = np.array(new long[] { 1, 2, 3 });
            var t = np.array(new long[] { 2, 3 });
            var expected = np.array(new[] { false, true, true });
            np.isin(el, t, kind: "sort").array_equal(expected).Should().BeTrue();
            np.isin(el, t, kind: "table").array_equal(expected).Should().BeTrue();
        }

        [TestMethod]
        public void Kind_Invalid_Raises()
        {
            Action act = () => np.isin(np.array(new long[] { 1 }), np.array(new long[] { 1 }), kind: "bogus");
            act.Should().Throw<ValueError>().WithMessage("Invalid kind: 'bogus'. Please use None, 'sort' or 'table'.");
        }

        [TestMethod]
        public void Kind_Table_NonInteger_Raises()
        {
            Action act = () => np.isin(np.array(new[] { 1.0, 2.0 }), np.array(new[] { 2.0 }), kind: "table");
            act.Should().Throw<ValueError>().WithMessage(
                "The 'table' method is only supported for boolean or integer arrays. Please select 'sort' or None for kind.");
        }

        [TestMethod]
        public void Kind_Table_RangeOverflow_Raises()
        {
            // int64 [min, max] range = 2^64-1 > iinfo(int64).max -> NumPy raises RuntimeError.
            Action act = () => np.isin(np.array(new long[] { 1 }),
                np.array(new[] { long.MinValue, long.MaxValue }), kind: "table");
            act.Should().Throw<InvalidOperationException>().WithMessage(
                "You have specified kind='table', but the range of values in `ar2` or `ar1` exceed the "
                + "maximum integer of the datatype. Please set `kind` to None or 'sort'.");
        }

        [TestMethod]
        public void Kind_Table_EmptyTest_AllFalse()
        {
            np.isin(np.array(new long[] { 1, 2 }), np.zeros(new Shape(0), NPTypeCode.Int64), kind: "table")
                .array_equal(np.array(new[] { false, false })).Should().BeTrue();
        }

        // ---- assume_unique (valid contract: inputs actually unique) ------------------------
        [TestMethod]
        public void AssumeUnique_Valid()
        {
            np.isin(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 2, 3 }), assume_unique: true)
                .array_equal(np.array(new[] { false, true, true })).Should().BeTrue();
        }

        // ---- non-contiguous element layouts (reshape-to-C-order must be correct) -----------
        [TestMethod]
        public void Layout_Transposed()
        {
            var m = np.arange(6).reshape(2, 3);                          // [[0,1,2],[3,4,5]]
            np.isin(m.T, np.array(new long[] { 1, 4, 5 }))              // T = [[0,3],[1,4],[2,5]]
                .array_equal(np.array(new[] { false, false, true, true, false, true }).reshape(3, 2))
                .Should().BeTrue();
        }

        [TestMethod]
        public void Layout_NegativeStride()
        {
            np.isin(np.arange(5)["::-1"], np.array(new long[] { 0, 2, 4 }))   // [4,3,2,1,0]
                .array_equal(np.array(new[] { true, false, true, false, true })).Should().BeTrue();
        }

        [TestMethod]
        public void Layout_Strided()
        {
            var m = np.arange(6).reshape(2, 3);
            np.isin(m[":, ::2"], np.array(new long[] { 2, 4 }))         // [[0,2],[3,5]]
                .array_equal(np.array(new[] { false, true, false, false }).reshape(2, 2)).Should().BeTrue();
        }

        [TestMethod]
        public void Layout_SliceOffset()
        {
            np.isin(np.arange(10)["3:8"], np.array(new long[] { 4, 6 }))   // [3,4,5,6,7]
                .array_equal(np.array(new[] { false, true, false, true, false })).Should().BeTrue();
        }

        /// <summary>
        /// DELIBERATE DIVERGENCE (documented in MisalignedRegistry): when assume_unique=True but the
        /// inputs are NOT actually unique, NumPy documents that "incorrect results ... could result",
        /// and its actual output depends on the internal method chosen (sort-path returns garbage).
        /// NumSharp always returns the CORRECT membership. Here NumPy returns [True, False]; NumSharp
        /// returns [False, False] (5.0 is genuinely not in the test set).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void AssumeUnique_NonUnique_ReturnsCorrectMembership_NotNumPyGarbage()
        {
            var element = np.array(new[] { 5.0, 5.0 });
            var test = np.arange(20).astype(NPTypeCode.Double) + 100.0;   // large -> NumPy's sort path
            np.isin(element, test, assume_unique: true)
                .array_equal(np.array(new[] { false, false })).Should().BeTrue();
        }
    }
}
