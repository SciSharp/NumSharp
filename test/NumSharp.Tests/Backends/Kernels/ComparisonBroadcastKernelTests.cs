using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     The comparison kernels of an array against a one-element operand compare a whole vector at a time (the scalar
    ///     broadcast into a vector once), store four masks at a time as NumPy's packed bytes, and never touch an element
    ///     past the array: the old 4×-unrolled and one-vector loops ran one body too many when the length was one less
    ///     than a multiple of their step, reading one element past both inputs and writing one bool past the result.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The overrun was in the same-dtype array-against-array kernel (<c>SimdFull</c>) and the float / signbit
    ///     predicate kernel (isnan, isinf, isfinite, isposinf, isneginf, signbit) since they were written; the
    ///     scalar-broadcast kernels ran one element at a time until the vector loops were added. It was silent: an
    ///     <c>out=</c> view's next slot in its parent buffer changed, a fresh result wrote one byte past its allocation
    ///     (the pool's buckets have no slack). A NUMSHARP_DEBUG_GUARD_PAGES sweep of 58,320 elementwise / comparison /
    ///     cast / reduction cases faults at <c>equal Boolean n=31</c> on the old build and passes on this one.
    ///     </para>
    ///     <para>Expected values come from an independent per-element reference, and from NumPy 2.4.2 where noted.</para>
    /// </remarks>
    [TestClass]
    public unsafe class ComparisonBroadcastKernelTests : TestClass
    {
        /// <summary>The dtypes the vector comparison kernels serve.</summary>
        private static readonly NPTypeCode[] SimdTypes =
        {
            NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
            NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Single, NPTypeCode.Double,
        };

        /// <summary>Every dtype, for the value test (the non-vector ones keep their scalar loops).</summary>
        private static readonly NPTypeCode[] AllTypes = SimdTypes
            .Concat(new[] { NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Decimal, NPTypeCode.Complex }).ToArray();

        private static readonly ComparisonOp[] Ops =
        {
            ComparisonOp.Equal, ComparisonOp.NotEqual, ComparisonOp.Less, ComparisonOp.LessEqual, ComparisonOp.Greater,
            ComparisonOp.GreaterEqual,
        };

        /// <summary>Every tail length of every vector and unroll step, plus a few long ones.</summary>
        private static readonly int[] Lengths = Enumerable.Range(0, 70).Concat(new[] { 127, 128, 129, 255, 256, 257, 1023, 1025 }).ToArray();

        /// <summary>
        ///     The three contiguous comparison kernels write exactly <c>totalSize</c> bools: called directly with a result
        ///     buffer whose slots past the length hold a sentinel, at every length, the sentinel survives.
        /// </summary>
        [TestMethod]
        public void ComparisonKernels_NeverWritePastTheResult()
        {
            var paths = new[] { ExecutionPath.SimdFull, ExecutionPath.SimdScalarRight, ExecutionPath.SimdScalarLeft };
            const byte Sentinel = 0x5A;
            long shape = 0, stride = 1;
            foreach (var t in SimdTypes)
            {
                int size = DirectILKernelGenerator.GetTypeSize(t);
                foreach (int n in Lengths)
                {
                    // The inputs carry 64 extra bytes so an over-read stays inside the allocation (the guard-page sweep
                    // is the check for reads; here the written bytes are).
                    byte* lhs = (byte*)NativeMemory.AllocZeroed((nuint)(n * size + 64));
                    byte* rhs = (byte*)NativeMemory.AllocZeroed((nuint)(n * size + 64));
                    byte* res = (byte*)NativeMemory.Alloc((nuint)(n + 64));
                    try
                    {
                        foreach (var path in paths)
                        foreach (var op in Ops)
                        {
                            new Span<byte>(res, n + 64).Fill(Sentinel);
                            shape = n;
                            var kernel = DirectILKernelGenerator.GetComparisonKernel(new ComparisonKernelKey(t, t, op, path));
                            kernel(lhs, rhs, (bool*)res, &stride, &stride, &shape, 1, n);
                            for (int k = n; k < n + 64; k++)
                                res[k].Should().Be(Sentinel, $"{t} {op} {path} n={n} wrote result slot {k}");
                        }
                    }
                    finally
                    {
                        NativeMemory.Free(lhs);
                        NativeMemory.Free(rhs);
                        NativeMemory.Free(res);
                    }
                }
            }
        }

        /// <summary>The float / signbit predicate kernel writes exactly <c>totalSize</c> bools (the same off-by-one).</summary>
        [TestMethod]
        public void PredicateKernel_NeverWritesPastTheResult()
        {
            var cases = new (UnaryOp Op, NPTypeCode Type)[]
            {
                (UnaryOp.IsNan, NPTypeCode.Single), (UnaryOp.IsNan, NPTypeCode.Double), (UnaryOp.IsInf, NPTypeCode.Single),
                (UnaryOp.IsInf, NPTypeCode.Double), (UnaryOp.IsFinite, NPTypeCode.Single), (UnaryOp.IsFinite, NPTypeCode.Double),
                (UnaryOp.IsPosInf, NPTypeCode.Double), (UnaryOp.IsNegInf, NPTypeCode.Single), (UnaryOp.SignBit, NPTypeCode.Single),
                (UnaryOp.SignBit, NPTypeCode.Double), (UnaryOp.SignBit, NPTypeCode.Int32), (UnaryOp.SignBit, NPTypeCode.Int64),
            };
            const byte Sentinel = 0x5A;
            long shape = 0, stride = 1;
            foreach (var (op, t) in cases)
            {
                var kernel = DirectILKernelGenerator.GetPredicateContiguousKernel(op, t);
                if (kernel is null)
                    continue;   // no SIMD on this host: the scalar route writes element by element
                int size = DirectILKernelGenerator.GetTypeSize(t);
                foreach (int n in Lengths)
                {
                    byte* input = (byte*)NativeMemory.AllocZeroed((nuint)(n * size + 64));
                    byte* res = (byte*)NativeMemory.Alloc((nuint)(n + 64));
                    try
                    {
                        new Span<byte>(res, n + 64).Fill(Sentinel);
                        shape = n;
                        kernel(input, res, &stride, &shape, 1, n);
                        for (int k = n; k < n + 64; k++)
                            res[k].Should().Be(Sentinel, $"{op} {t} n={n} wrote result slot {k}");
                    }
                    finally
                    {
                        NativeMemory.Free(input);
                        NativeMemory.Free(res);
                    }
                }
            }
        }

        /// <summary>
        ///     Through the ufuncs: an <c>out=</c> view of the first n slots of a longer bool buffer keeps the rest of the
        ///     buffer, for an array, a scalar on the right and a scalar on the left.
        /// </summary>
        [TestMethod]
        public void Comparisons_KeepTheParentBufferPastAnOutView()
        {
            foreach (var t in SimdTypes)
            foreach (int n in new[] { 3, 7, 15, 31, 39, 63, 127, 255 })
            {
                using var abase = np.ones(new Shape(n + 8), t);
                using var bbase = np.ones(new Shape(n + 8), t);
                using var s = NDArray.Scalar(1, t);
                var a = abase[$":{n}"];
                var b = bbase[$":{n}"];
                foreach (var form in new[] { "array", "scalar-right", "scalar-left" })
                {
                    using var obase = np.zeros(new Shape(n + 8), NPTypeCode.Boolean);
                    for (int k = n; k < n + 8; k++)
                        obase.SetAtIndex(true, k);
                    var o = obase[$":{n}"];
                    // 1 > 1 is False everywhere, so a True past the view turned False is a write past it.
                    _ = form == "array" ? np.greater(a, b, o) : form == "scalar-right" ? np.greater(a, s, o) : np.greater(s, a, o);
                    for (int k = n; k < n + 8; k++)
                        ((bool)obase.GetAtIndex(k)).Should().BeTrue($"{t} n={n} {form}: slot {k} past the out view changed");
                }
            }
        }

        /// <summary>
        ///     Every comparison, in every operand form, equals an independent per-element reference over random values
        ///     (NaN, signed zeros, infinities, integer extremes), at every tail length of the vector loops.
        /// </summary>
        [TestMethod]
        public void Comparisons_MatchAnElementReference_InEveryForm()
        {
            var rng = new System.Random(20261007);
            var lengths = new[] { 0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 127, 128, 129, 257 };
            foreach (var t in AllTypes)
            foreach (int n in lengths)
            {
                var av = new object[n];
                var ov = new object[n];
                using var a = np.empty(new Shape(n), t);
                using var other = np.empty(new Shape(n), t);
                for (int i = 0; i < n; i++)
                {
                    av[i] = Pick(rng, t);
                    a.SetAtIndex(av[i], i);
                    ov[i] = Pick(rng, t);
                    other.SetAtIndex(ov[i], i);
                }
                object sv = Pick(rng, t);
                using var s = NDArray.Scalar(sv, t);
                foreach (var op in Ops)
                {
                    using var right = Compare(op, a, s);
                    using var left = Compare(op, s, a);
                    using var both = Compare(op, a, other);
                    var r = right.ToArray<bool>();
                    var l = left.ToArray<bool>();
                    var bth = both.ToArray<bool>();
                    for (int i = 0; i < n; i++)
                    {
                        r[i].Should().Be(Reference(op, t, av[i], sv), $"{t} {op} array op scalar n={n} i={i} a={av[i]} s={sv}");
                        l[i].Should().Be(Reference(op, t, sv, av[i]), $"{t} {op} scalar op array n={n} i={i} a={av[i]} s={sv}");
                        bth[i].Should().Be(Reference(op, t, av[i], ov[i]), $"{t} {op} array op array n={n} i={i} a={av[i]} b={ov[i]}");
                    }
                }
            }
        }

        /// <summary>
        ///     A bool array holding bytes other than 0 and 1 (a <c>frombuffer</c> view) compares by truth value in the
        ///     vector body too. NumPy 2.4.2: <c>np.frombuffer(bytes([0, 1, 0x80, 0xff, 2]), bool) == True</c> is
        ///     <c>[False, True, True, True, True]</c>.
        /// </summary>
        [TestMethod]
        public void BoolArray_NonCanonicalBytes_CompareByTruthValue_AgainstAScalar()
        {
            var cycle = new byte[] { 0, 1, 0x80, 0xff, 2 };
            var raw = Enumerable.Range(0, 70).Select(i => cycle[i % cycle.Length]).ToArray();
            using var b = np.frombuffer(raw, np.bool_);
            using var t = NDArray.Scalar(true);
            using var f = NDArray.Scalar(false);
            bool[] truth = raw.Select(x => x != 0).ToArray();
            np.equal(b, t).ToArray<bool>().Should().Equal(truth);
            np.not_equal(b, t).ToArray<bool>().Should().Equal(truth.Select(x => !x));
            np.greater(b, f).ToArray<bool>().Should().Equal(truth);
            np.less(t, b).ToArray<bool>().Should().Equal(new bool[70]);
            np.less_equal(b, f).ToArray<bool>().Should().Equal(truth.Select(x => !x));
            np.greater_equal(t, b).ToArray<bool>().Should().Equal(Enumerable.Repeat(true, 70));
        }

        /// <summary>
        ///     An integer array against a Python int of another width compares exactly, in range or not (NumPy's
        ///     <c>special_integer_comparisons</c>). An in-range int is re-packed into the array's dtype, so the comparison
        ///     runs the same-dtype vector kernel; an out-of-range one keeps the exact promoted compare.
        /// </summary>
        [TestMethod]
        public void IntegerArray_AgainstAnotherWidthPythonInt_ComparesExactly()
        {
            // NumPy 2.4.2: (np.arange(256) % 256).astype(np.uint8) > 128 -> True from 129 on; int8 > 1000 all False;
            // int8 < 1000 all True; uint64 > -1 all True; uint64 == -1 all False; int16 == 300 only at 300.
            using var u8 = (np.arange(300) % 256).astype(np.uint8);
            var gt = (u8 > 128).ToArray<bool>();
            for (int i = 0; i < 300; i++)
                gt[i].Should().Be(i % 256 > 128, $"uint8 {i % 256} > 128");

            using var i8 = (np.arange(100) - 50).astype(np.int8);
            (i8 > 1000).ToArray<bool>().Should().AllBeEquivalentTo(false);
            (i8 < 1000).ToArray<bool>().Should().AllBeEquivalentTo(true);
            (i8 > -1000).ToArray<bool>().Should().AllBeEquivalentTo(true);

            using var u64 = np.array(new[] { 0UL, 1UL, ulong.MaxValue, 1UL << 63 });
            (u64 > -1).ToArray<bool>().Should().AllBeEquivalentTo(true);
            (u64 == -1).ToArray<bool>().Should().AllBeEquivalentTo(false);
            (u64 == ulong.MaxValue).ToArray<bool>().Should().Equal(false, false, true, false);

            using var i16 = np.arange(1000).astype(np.int16);
            var eq = (i16 == 300L).ToArray<bool>();
            for (int i = 0; i < 1000; i++)
                eq[i].Should().Be(i == 300);
            var le = (300L >= i16).ToArray<bool>();
            for (int i = 0; i < 1000; i++)
                le[i].Should().Be(i <= 300);
        }

        /// <summary>Calls the comparison ufunc for <paramref name="op"/>.</summary>
        private static NDArray Compare(ComparisonOp op, NDArray a, NDArray b) => op switch
        {
            ComparisonOp.Equal => np.equal(a, b),
            ComparisonOp.NotEqual => np.not_equal(a, b),
            ComparisonOp.Less => np.less(a, b),
            ComparisonOp.LessEqual => np.less_equal(a, b),
            ComparisonOp.Greater => np.greater(a, b),
            _ => np.greater_equal(a, b),
        };

        /// <summary>A random value of <paramref name="t"/>, special values included.</summary>
        private static object Pick(System.Random rng, NPTypeCode t)
        {
            double r = rng.NextDouble();
            return t switch
            {
                NPTypeCode.Boolean => rng.Next(2) == 1,
                NPTypeCode.SByte => r < 0.05 ? sbyte.MinValue : r < 0.1 ? sbyte.MaxValue : (sbyte)rng.Next(-3, 4),
                NPTypeCode.Byte => r < 0.05 ? byte.MaxValue : (byte)rng.Next(0, 6),
                NPTypeCode.Int16 => r < 0.05 ? short.MinValue : (short)rng.Next(-3, 4),
                NPTypeCode.UInt16 => r < 0.05 ? ushort.MaxValue : (ushort)rng.Next(0, 6),
                NPTypeCode.Int32 => r < 0.05 ? int.MinValue : r < 0.1 ? int.MaxValue : rng.Next(-3, 4),
                NPTypeCode.UInt32 => r < 0.05 ? uint.MaxValue : (uint)rng.Next(0, 6),
                NPTypeCode.Int64 => r < 0.05 ? long.MinValue : r < 0.1 ? long.MaxValue : (long)rng.Next(-3, 4),
                NPTypeCode.UInt64 => r < 0.05 ? ulong.MaxValue : r < 0.1 ? 1UL << 63 : (ulong)rng.Next(0, 6),
                NPTypeCode.Single => r < 0.06 ? float.NaN : r < 0.1 ? -0.0f : r < 0.13 ? float.PositiveInfinity
                    : r < 0.16 ? float.NegativeInfinity : rng.Next(-3, 4) / 2f,
                NPTypeCode.Double => r < 0.06 ? double.NaN : r < 0.1 ? -0.0 : r < 0.13 ? double.PositiveInfinity
                    : r < 0.16 ? double.NegativeInfinity : rng.Next(-3, 4) / 2.0,
                NPTypeCode.Half => r < 0.08 ? Half.NaN : (Half)(rng.Next(-3, 4) / 2.0),
                NPTypeCode.Char => (char)rng.Next(60, 66),
                NPTypeCode.Decimal => rng.Next(-3, 4) / 2m,
                _ => new Complex(rng.Next(-1, 2), rng.Next(-1, 2)),
            };
        }

        /// <summary>
        ///     NumPy's comparison of two elements of <paramref name="t"/>, computed without NumSharp: floats unordered
        ///     with a NaN (only <c>!=</c> is True), integers exactly, bools by truth value, complex lexicographically.
        /// </summary>
        private static bool Reference(ComparisonOp op, NPTypeCode t, object a, object b)
        {
            int c;
            switch (t)
            {
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Half:
                {
                    double x = a is Half ha ? (double)ha : Convert.ToDouble(a);
                    double y = b is Half hb ? (double)hb : Convert.ToDouble(b);
                    if (double.IsNaN(x) || double.IsNaN(y))
                        return op == ComparisonOp.NotEqual;
                    c = x < y ? -1 : x > y ? 1 : 0;   // -0.0 == +0.0
                    break;
                }
                case NPTypeCode.Decimal:
                    c = ((decimal)a).CompareTo((decimal)b);
                    break;
                case NPTypeCode.Complex:
                {
                    var x = (Complex)a;
                    var y = (Complex)b;
                    c = x.Real != y.Real ? x.Real.CompareTo(y.Real) : x.Imaginary.CompareTo(y.Imaginary);
                    break;
                }
                case NPTypeCode.Boolean:
                    c = ((bool)a ? 1 : 0).CompareTo((bool)b ? 1 : 0);
                    break;
                case NPTypeCode.Char:
                    c = ((char)a).CompareTo((char)b);
                    break;
                default:
                {
                    Int128 x = a is ulong ua ? ua : Convert.ToInt64(a);
                    Int128 y = b is ulong ub ? ub : Convert.ToInt64(b);
                    c = x.CompareTo(y);
                    break;
                }
            }
            return op switch
            {
                ComparisonOp.Equal => c == 0,
                ComparisonOp.NotEqual => c != 0,
                ComparisonOp.Less => c < 0,
                ComparisonOp.LessEqual => c <= 0,
                ComparisonOp.Greater => c > 0,
                _ => c >= 0,
            };
        }
    }
}
