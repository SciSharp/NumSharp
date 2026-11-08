using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Manipulation
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
        /// 2^53. NumPy's table method and its loop path (whose int64 / uint64 comparisons are exact) still
        /// answer the EXACT membership — two distinct integers that collide as float64 do not compare
        /// equal — and these sets take one of them. Regression pin: uint64(2^63+1) and int64(2^63-1) are
        /// distinct integers that both round to 2^63 as float64. (NumPy's sort path concatenates in
        /// float64 and does merge them: np.isin.NumPyPort.Test.cs.)
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
        /// When assume_unique=True but the inputs are NOT actually unique, NumPy documents that "incorrect
        /// results ... could result": its sort path skips np.unique, so the two 5.0 are neighbours after the
        /// stable sort and the first is flagged a member. NumSharp gives NumPy's answer, [True, False] (it
        /// returned the membership, [False, False], as a documented divergence until the np-layer dtype
        /// plan's np.isin family). The loop path ignores assume_unique: a short set answers [False, False].
        /// </summary>
        [TestMethod]
        public void AssumeUnique_NonUnique_IsNumPysSortPathAnswer()
        {
            var element = np.array(new[] { 5.0, 5.0 });
            var test = np.arange(20).astype(NPTypeCode.Double) + 100.0;   // large -> NumPy's sort path
            // >>> np.isin(np.array([5.0, 5.0]), np.arange(20.) + 100, assume_unique=True)  array([ True, False])
            np.isin(element, test, assume_unique: true)
                .array_equal(np.array(new[] { true, false })).Should().BeTrue();
            // >>> np.isin(np.array([5.0, 5.0]), np.array([100.0]), assume_unique=True)  array([False, False])
            np.isin(element, np.array(new[] { 100.0 }), assume_unique: true)
                .array_equal(np.array(new[] { false, false })).Should().BeTrue();
        }

        /// <summary>
        ///     Every NumPy path ends in <c>_isin(...).reshape(element.shape)</c>, so the answer is always a VIEW of a fresh
        ///     1-D array (<c>flags.owndata</c> False) — the empty-operand answers included.
        /// </summary>
        [TestMethod]
        public void EmptyOperand_AnswerIsAViewLikeEveryOther()
        {
            using var emptyElement = np.isin(np.zeros(new Shape(0, 3), np.int64), np.array(new long[] { 1, 2 }));
            emptyElement.shape.Should().Equal(0L, 3L);
            emptyElement.flags.owndata.Should().BeFalse();
            using var emptyTest = np.isin(np.array(new long[] { 1, 2 }), np.zeros(new Shape(0), np.int64));
            emptyTest.flags.owndata.Should().BeFalse();
            emptyTest.ToArray<bool>().Should().Equal(false, false);
            using var inverted = np.isin(np.array(new long[] { 1, 2 }), np.zeros(new Shape(0), np.int64), invert: true);
            inverted.flags.owndata.Should().BeFalse();
            inverted.ToArray<bool>().Should().Equal(true, true);
            using var floats = np.isin(np.zeros(new Shape(2, 0)), np.array(new[] { 1.5 }));
            floats.shape.Should().Equal(2L, 0L);
            floats.flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     NumPy's table method and loop path run as one-pass kernels (a bool table over the set's range; every element
        ///     compared with each set value in vector registers). Their answer is the membership itself, so a seeded sweep holds
        ///     them to its definition: an element is a member when some set value equals it (NaN never, -0.0 equal to 0.0).
        ///     Sizes straddle every vector width (4, 8, 16 and 32 lanes), the loop kernel's four-vector step and the tails;
        ///     the larger sets take NumPy's sort path at these sizes (24 values or more), which the membership routes answer;
        ///     the operands include views at an offset and reversed ones; <c>kind='sort'</c> sends integer pairs to the loop
        ///     path. The loop kernel's value limit is <see cref="LoopKernel_AtItsValueLimit_IsTheMembership"/>'s.
        /// </summary>
        [TestMethod]
        public void OnePassKernels_AgreeWithTheMembershipDefinition()
        {
            var rng = new System.Random(20261027);
            DType[] dtypes = { np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.@bool,
                np.@char, np.float32, np.float64 };
            int[] sizes = { 1, 3, 7, 31, 32, 33, 63, 100, 257 };
            int[] setSizes = { 1, 2, 5, 16, 24, 25, 40 };
            int checkedCalls = 0;
            foreach (var ed in dtypes)
            foreach (var sd in dtypes)
            {
                int n = sizes[rng.Next(sizes.Length)], m = setSizes[rng.Next(setSizes.Length)];
                bool floatPair = ed == np.float32 || ed == np.float64 || sd == np.float32 || sd == np.float64;
                // Small values, so every dtype holds them exactly and the definition can be checked in double; an
                // unsigned or char dtype gets non-negative ones, a float pair a NaN and a negative zero.
                double[] Values(int count, DType dt) => Enumerable.Range(0, count).Select(_ =>
                {
                    if (dt == np.@bool) return (double)rng.Next(2);
                    bool unsigned = dt == np.uint8 || dt == np.uint16 || dt == np.uint32 || dt == np.uint64 || dt == np.@char;
                    double v = unsigned ? rng.Next(0, 40) : rng.Next(-20, 20);
                    if (floatPair && (dt == np.float32 || dt == np.float64))
                    {
                        int pick = rng.Next(12);
                        if (pick == 0) v = double.NaN;
                        else if (pick == 1) v = -0.0;
                    }
                    return v;
                }).ToArray();
                NDArray Build(double[] v, DType dt) => dt == np.@char
                    ? np.array(v.Select(x => (char)(int)x).ToArray())
                    : np.array(v).astype(dt);
                double[] ev = Values(n + 4, ed), sv = Values(m, sd);
                // The element as a view at an offset of a longer array, and reversed: both reach the kernels' pointers.
                NDArray full = Build(ev, ed);
                var forms = new[] { (Label: "offset", Array: full[$"4:"], Values: ev.Skip(4).ToArray()),
                    (Label: "reversed", Array: full["::-1"], Values: Enumerable.Reverse(ev).ToArray()) };
                NDArray set = Build(sv, sd);
                double[] setValues = sv.Select(x => sd == np.@bool ? (x != 0 ? 1 : 0) : x).ToArray();
                foreach (var (label, element, values) in forms)
                foreach (bool invert in new[] { false, true })
                foreach (string kind in new[] { null, "sort" })
                {
                    // kind='sort' takes an integer pair off the table method, onto the loop path (the one-byte lanes too).
                    var expected = values.Select(x => setValues.Any(s => s == (ed == np.@bool ? (x != 0 ? 1 : 0) : x)) != invert)
                        .ToArray();
                    var got = np.isin(element, set, invert: invert, kind: kind).ToArray<bool>();
                    got.Should().Equal(expected,
                        $"isin({ed.name} {label} n={values.Length}, {sd.name} m={m}, invert={invert}, kind={kind ?? "None"})");
                    checkedCalls++;
                }
            }
            checkedCalls.Should().Be(dtypes.Length * dtypes.Length * 8);
        }

        /// <summary>
        ///     The table kernel reads a uint64 operand as its 64 bits and computes the offsets modulo 2**64, which is exact for
        ///     values past 2**63 beside another unsigned operand; a uint64 operand beside a signed one is declined to the
        ///     membership routes (its bits name another number there). NumPy 2.4.2's answers, probed.
        /// </summary>
        [TestMethod]
        public void TableKernel_UInt64PastTwoToThe63_IsNumPysAnswer()
        {
            var e = np.array(new ulong[] { ulong.MaxValue, ulong.MaxValue - 1, 1UL << 63, 0, ulong.MaxValue - 2 });
            var s = np.array(new ulong[] { ulong.MaxValue - 2, ulong.MaxValue });
            np.isin(e, s).ToArray<bool>().Should().Equal(true, false, false, false, true);
            np.isin(e, s, invert: true).ToArray<bool>().Should().Equal(false, true, true, true, false);
            np.isin(e, s, kind: "table").ToArray<bool>().Should().Equal(true, false, false, false, true);
            // Beside int64, -1 is not 2**64 - 1 (whose bits it shares), in either operand.
            np.isin(np.array(new long[] { -1, 3 }), np.array(new ulong[] { ulong.MaxValue, 3 })).ToArray<bool>()
                .Should().Equal(false, true);
            np.isin(np.array(new ulong[] { ulong.MaxValue, 3 }), np.array(new long[] { -1, 3 })).ToArray<bool>()
                .Should().Equal(false, true);
            // Beside uint8, every value is non-negative: the kernel runs.
            np.isin(np.array(new ulong[] { ulong.MaxValue, 255, 7 }), np.array(new byte[] { 255, 7 })).ToArray<bool>()
                .Should().Equal(false, true, true);
        }

        /// <summary>
        ///     Elements of two bytes from 8,192 on read a table over their dtype's whole domain (65,536 flags, indexed by the
        ///     element's bits): a negative int16 sits at its two's complement, and a set value outside the element dtype's
        ///     domain names no element. Fewer elements read NumPy's table over the set's range. Every set keeps NumPy's table
        ///     constraints (its range within its dtype's maximum and at most <c>6 * (len(ar1) + len(ar2))</c>), so NumPy takes
        ///     its table method, whose answer is the membership; held to the definition, with invert, at 10,000 and 100
        ///     elements. Each element array starts with its domain's extremes and every set value of the domain with both
        ///     neighbours, so a member the table misses shows; the rest is spread over the whole domain.
        /// </summary>
        [TestMethod]
        public void TableKernel_TwoByteElementsOverTheirWholeDomain_IsTheMembership()
        {
            var cases = new (DType Dtype, long Lo, long Hi, NDArray Set, long[] Members, int N)[]
            {
                (np.int16, short.MinValue, short.MaxValue, np.array(new[] { -32770, -30000, -1, 0, 5, 20000 }),
                    new long[] { -30000, -1, 0, 5, 20000 }, 10_000),
                (np.uint16, 0, ushort.MaxValue, np.array(new[] { -5, 3, 50000, 55000 }), new long[] { 3, 50000, 55000 }, 10_000),
                (np.@char, 0, char.MaxValue, np.array(new ulong[] { 3, 100, 40000 }), new long[] { 3, 100, 40000 }, 10_000),
                (np.int16, short.MinValue, short.MaxValue, np.array(new sbyte[] { -100, -1, 0, 27 }),
                    new long[] { -100, -1, 0, 27 }, 10_000),
                (np.uint16, 0, ushort.MaxValue, np.array(new long[] { 65535, 70000 }), new long[] { 65535 }, 10_000),
                (np.int16, short.MinValue, short.MaxValue, np.array(new[] { true }), new long[] { 1 }, 10_000),
                // Below DomainTableMinElements: NumPy's table over the set's range.
                (np.int16, short.MinValue, short.MaxValue, np.array(new sbyte[] { -100, -1, 0, 27 }),
                    new long[] { -100, -1, 0, 27 }, 100),
                (np.uint16, 0, ushort.MaxValue, np.array(new short[] { 0, 3, 200 }), new long[] { 0, 3, 200 }, 100),
                (np.@char, 0, char.MaxValue, np.array(new ulong[] { 3, 100, 300 }), new long[] { 3, 100, 300 }, 100),
            };
            foreach (var (dtype, lo, hi, set, members, n) in cases)
            {
                var head = new[] { lo, hi }.Concat(members.SelectMany(v => new[] { v - 1, v, v + 1 }))
                    .Where(v => v >= lo && v <= hi).ToArray();
                long[] values = head.Concat(Enumerable.Range(0, n - head.Length).Select(i => lo + (long)i * 7919 % 65536))
                    .ToArray();
                NDArray element = dtype == np.@char
                    ? np.array(values.Select(v => (char)v).ToArray())
                    : np.array(values).astype(dtype);
                foreach (bool invert in new[] { false, true })
                {
                    var got = np.isin(element, set, invert: invert).ToArray<bool>();
                    for (int i = 0; i < n; i++)
                        if (got[i] != (members.Contains(values[i]) != invert))
                            Assert.Fail($"isin({dtype.name} n={n} [{i}] = {values[i]}, {set.dtype.name} set, invert={invert})");
                }
            }
        }

        /// <summary>
        ///     The loop kernel's value limit grows with the common dtype's vector lanes (9 values per lane: 36 for float64 and
        ///     int64, 72 for float32); both sides of it answer the membership. The element counts make each set NumPy's loop
        ///     path (<c>len(ar2) &lt; 10 * len(ar1) ** 0.145</c>: 38.0 at 10,000 elements, 74.1 at 1,000,000);
        ///     <c>kind='sort'</c> keeps the integer pair off the table method.
        /// </summary>
        [TestMethod]
        public void LoopKernel_AtItsValueLimit_IsTheMembership()
        {
            foreach (var (dt, m, n) in new[] { (np.float64, 36, 10_000), (np.float64, 37, 10_000), (np.int64, 36, 10_000),
                (np.int64, 37, 10_000), (np.float32, 72, 1_000_000), (np.float32, 73, 1_000_000) })
            {
                var element = (np.arange(n) % 997).astype(dt);
                var set = (np.arange(m) * 7).astype(dt);
                var members = new System.Collections.Generic.HashSet<long>(Enumerable.Range(0, m).Select(j => (long)j * 7));
                foreach (bool invert in new[] { false, true })
                {
                    var got = np.isin(element, set, invert: invert, kind: dt == np.int64 ? "sort" : null).ToArray<bool>();
                    for (int i = 0; i < n; i++)
                        if (got[i] != (members.Contains(i % 997) != invert))
                            Assert.Fail($"isin({dt.name} n={n}, m={m}, invert={invert}) differs at {i}");
                }
            }
        }
    }
}
