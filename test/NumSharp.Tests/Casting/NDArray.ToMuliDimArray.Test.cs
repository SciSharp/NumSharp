using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Pins <see cref="NDArray.ToMuliDimArray{T}"/> — the copy of an NDArray into a .NET <c>T[]</c> /
    ///     <c>T[,…]</c> — across all 15 dtypes, every memory layout its fill engine tells apart, conversions,
    ///     degenerate shapes, the IL-emitted rank ≥ 4 allocator, and its error contract.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The method fills the pinned result in ONE pass: a contiguous source is one memcpy; a strided one is
    ///         copied plane by plane (a memcpy per row for a unit column stride, 64-column transpose bands built from
    ///         8×4 / 8×8 AVX register blocks for transposed / F-ordered planes, AVX2 stride-2 deinterleave or hardware
    ///         gathers for plain strided rows of 4/8-byte elements, a scalar loop otherwise); an element type other
    ///         than the dtype is converted by astype's own cast core (plus a runtime-verified AVX2 replica of the
    ///         decimal → double conversion); and rank ≥ 4 is allocated through a cached <c>DynamicMethod</c>
    ///         <c>newobj</c> rather than <see cref="Array.CreateInstance(Type, int[])"/>. Each of those is its own code
    ///         path PER ELEMENT WIDTH, so the tests walk every dtype through every layout that selects one — including
    ///         the ragged edges (rows past the last 8-row strip, columns past the last whole register block, a second
    ///         64-column band, a negative column stride inside a transpose block).
    ///     </para>
    ///     <para>
    ///         Behaviour that changed with the rewrite, each pinned below: float16 and complex128 used to throw
    ///         <see cref="ArgumentException"/> (<c>Buffer.BlockCopy</c> accepts CLR primitives only), so
    ///         <c>(Array)nd</c> failed for both; an element type other than the dtype threw
    ///         <see cref="ArrayTypeMismatchException"/> and now converts with astype semantics; a 0-d array threw
    ///         ("Must provide at least one rank.") and now yields a one-element <c>T[]</c>; an EMPTY decimal array threw
    ///         (the per-element decimal path read one element unconditionally); a non-dtype element type (Guid, nint,
    ///         DateTime) now fails up front with <see cref="NotSupportedException"/>.
    ///     </para>
    ///     <para>
    ///         The oracle is NumSharp's own logical C-order walk — <see cref="NDArray.ToArray{T}"/> of the view, or of
    ///         <c>view.astype(T)</c> for a conversion. That is the contract (the NumPy analog is
    ///         <c>np.asarray(view, dtype)</c> read in C order): the copy holds the view's elements in row-major order
    ///         whatever its strides, converted exactly as astype converts them. Elements are compared as RAW BYTES — NaN
    ///         payloads, signalling NaNs and signed zeros must survive the SIMD lane moves — except decimal, compared by
    ///         VALUE per the project's oracle convention (1.0m and 1.00m are the same number).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class NDArrayToMuliDimArrayTests
    {
        /// <summary>
        ///     Views of a 2-D base that each select a different plane-copy path (and, through a strided base, the
        ///     ragged-edge handling of those paths). Written against the base's own dimensions, so any 2-D base works.
        /// </summary>
        /// <remarks>
        ///     Paths, for a C-contiguous (R, C) base: transposed / F-order → transpose bands with element-contiguous
        ///     columns (the AVX 8×4 / 8×8 blocks for 8/4-byte elements); <c>[:, ::2].T</c> and a stepped window's
        ///     <c>.T</c> → transpose bands with a non-unit row stride; <c>[::-1, :].T</c> → blocks with a NEGATIVE
        ///     column stride; <c>[:, ::2]</c> → AVX2 deinterleave; <c>[:, ::3]</c> / <c>[:, ::-1]</c> → gathers (a
        ///     negative index vector for the reversed one); <c>[::-1, :]</c> and offset windows → a memcpy per row;
        ///     broadcasts → a memcpy per row with row stride 0, or the scalar loop for column stride 0; 1-, 2- and
        ///     16-byte elements take the scalar variants of the same planes.
        /// </remarks>
        private static readonly (string Name, Func<NDArray, NDArray> Make)[] PlaneLayouts =
        {
            ("C-contiguous [:, :]", a => a[":, :"]),
            ("transposed .T", a => a.T),
            ("F-contiguous copy", a => np.asfortranarray(a)),
            ("column-stepped [:, ::2]", a => a[":, ::2"]),
            ("column-strided [:, ::3]", a => a[":, ::3"]),
            ("column-reversed [:, ::-1]", a => a[":, ::-1"]),
            ("row-reversed [::-1, :]", a => a["::-1, :"]),
            ("offset window [10:60:3, 5:100]", a => a["10:60:3, 5:100"]),
            ("both stepped [::2, ::2]", a => a["::2, ::2"]),
            ("transposed column-stepped [:, ::2].T", a => a[":, ::2"].T),
            ("transposed row-reversed [::-1, :].T", a => a["::-1, :"].T),
            ("transposed row-stepped [::2, :].T", a => a["::2, :"].T),
            ("transposed stepped window [3:67, 1:141:7].T", a => a["3:67, 1:141:7"].T),
            ("row broadcast", a => np.broadcast_to(a["0"], new Shape(a.shape))),
            ("column broadcast", a => np.broadcast_to(a[":, 0:1"], new Shape(a.shape))),
            ("scalar broadcast", a => np.broadcast_to(a["3, 4"], new Shape(5, 7))),
            ("single column [:, 7:8]", a => a[":, 7:8"]),
            ("single stepped row [5:6, ::2]", a => a["5:6, ::2"]),
        };

        /// <summary>
        ///     Views of a 3-D base: the outer-axis odometer of the strided copy around every plane kind, plus a
        ///     middle-axis broadcast.
        /// </summary>
        private static readonly (string Name, Func<NDArray, NDArray> Make)[] CubeLayouts =
        {
            ("C-contiguous [:, :, :]", a => a[":, :, :"]),
            ("transpose(2, 0, 1)", a => a.transpose(new[] { 2, 0, 1 })),
            ("transpose(0, 2, 1)", a => a.transpose(new[] { 0, 2, 1 })),
            ("transpose(1, 2, 0)", a => a.transpose(new[] { 1, 2, 0 })),
            ("[:, ::2, ::-1]", a => a[":, ::2, ::-1"]),
            ("[::-1, :, ::3]", a => a["::-1, :, ::3"]),
            ("window [1:5, 2:7, 3:20]", a => a["1:5, 2:7, 3:20"]),
            ("middle-axis broadcast", a => np.broadcast_to(a[":, 0:1, :"], new Shape(a.shape))),
        };

        /// <summary>Views of a 1-D base: the one-axis strided loop, offsets, and a broadcast element.</summary>
        private static readonly (string Name, Func<NDArray, NDArray> Make)[] LineLayouts =
        {
            ("contiguous [:]", a => a[":"]),
            ("stepped [::3]", a => a["::3"]),
            ("reversed [::-1]", a => a["::-1"]),
            ("offset window [100:2100]", a => a["100:2100"]),
            ("short window [5:9]", a => a["5:9"]),
            ("broadcast element", a => np.broadcast_to(a["7:8"], new Shape(2000))),
        };

        /// <summary>
        ///     The layouts a conversion is checked on: every fill route a converting call can take — the multi-element
        ///     cast core over contiguous / transposed / strided / broadcast sources, the inline one-element conversion
        ///     (a one-element view and a 0-d view), and an empty result.
        /// </summary>
        private static readonly (string Name, Func<NDArray, NDArray> Make)[] ConversionLayouts =
        {
            ("C-contiguous [:, :]", a => a[":, :"]),
            ("transposed .T", a => a.T),
            ("column-stepped [:, ::2]", a => a[":, ::2"]),
            ("reversed stepped [::-1, ::3]", a => a["::-1, ::3"]),
            ("row broadcast", a => np.broadcast_to(a["0"], new Shape(a.shape))),
            ("one element [2:3, 5:6]", a => a["2:3, 5:6"]),
            ("0-d element [2, 5]", a => a["2, 5"]),
            ("empty [0:0, :]", a => a["0:0, :"]),
        };

        /// <summary>
        ///     Every dtype, through every layout that selects a distinct copy path, copies exactly the view's logical
        ///     C-order elements into a <c>T[]</c> / <c>T[,…]</c> of the view's rank and lengths — and the
        ///     <c>(Array)nd</c> cast, which dispatches here on the dtype, returns the same thing.
        /// </summary>
        /// <param name="dtype">The dtype under test (also the requested element type: no conversion).</param>
        /// <exception cref="AssertFailedException">A layout returned the wrong CLR type, lengths or elements.</exception>
        /// <remarks>
        ///     The 2-D base is (70, 150): its transpose has 150 rows (18 whole 8-row strips plus 6 ragged rows) and 70
        ///     columns (one full 64-column band plus a 6-column second band), so the register blocks, their ragged
        ///     columns and the ragged rows all run for every element width.
        /// </remarks>
        [TestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void EveryDtype_EveryLayout_CopiesTheLogicalElements(NPTypeCode dtype)
        {
            using (var plane = Sample(dtype, 70, 150))
                foreach (var (name, make) in PlaneLayouts)
                    CheckView(plane, make, dtype, $"{dtype} 2-D {name}");
            using (var cube = Sample(dtype, 6, 7, 25))
                foreach (var (name, make) in CubeLayouts)
                    CheckView(cube, make, dtype, $"{dtype} 3-D {name}");
            using (var line = Sample(dtype, 3000))
                foreach (var (name, make) in LineLayouts)
                    CheckView(line, make, dtype, $"{dtype} 1-D {name}");
        }

        /// <summary>
        ///     A contiguous view at a non-zero <see cref="Shape"/> offset (an <c>np.split</c> child) copies its own
        ///     window, not the start of the shared buffer.
        /// </summary>
        /// <exception cref="AssertFailedException">The copy read the wrong window.</exception>
        [TestMethod]
        public void SplitChild_ContiguousAtAnOffset_CopiesItsOwnWindow()
        {
            using var a = np.arange(12.0);
            var parts = np.split(a, 3);
            try
            {
                ((double[])parts[1].ToMuliDimArray<double>()).Should().Equal(4.0, 5.0, 6.0, 7.0);
                ((double[])parts[2].ToMuliDimArray<double>()).Should().Equal(8.0, 9.0, 10.0, 11.0);
                Check(NPTypeCode.Single, parts[2], "split child converted to float32");
            }
            finally
            {
                foreach (var p in parts)
                    p.Dispose();
            }
        }

        /// <summary>
        ///     An element type other than the dtype converts with astype semantics, on every route a converting call
        ///     can take — including the inline one-element and 0-d conversion and an empty result.
        /// </summary>
        /// <param name="from">The source dtype.</param>
        /// <param name="to">The requested element type.</param>
        /// <exception cref="AssertFailedException">A conversion differed from astype's.</exception>
        /// <remarks>
        ///     Before the rewrite every one of these pairs threw <see cref="ArrayTypeMismatchException"/>. The pairs
        ///     cover widening, narrowing with wrap-around, float truncation toward zero (the samples carry negative
        ///     fractions), float16 both ways, bool both ways, decimal both ways, complex → real (the imaginary part is
        ///     discarded, as astype does) and char.
        /// </remarks>
        [TestMethod]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Int64, NPTypeCode.Double)]
        [DataRow(NPTypeCode.UInt64, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Single, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Half, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Half, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Single, NPTypeCode.Half)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Half)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Int32)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.Int64, NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16, NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Byte, NPTypeCode.Int16)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Decimal, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Decimal, NPTypeCode.Int32)]
        [DataRow(NPTypeCode.Complex, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Complex)]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Complex)]
        [DataRow(NPTypeCode.Char, NPTypeCode.Int32)]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Char)]
        [DataRow(NPTypeCode.UInt16, NPTypeCode.Char)]
        public void Conversion_EveryRoute_MatchesAstype(NPTypeCode from, NPTypeCode to)
        {
            using (var plane = Sample(from, 9, 70))
                foreach (var (name, make) in ConversionLayouts)
                    CheckView(plane, make, to, $"{from} → {to} {name}");
            // A 1-D result of ≥ 1024 elements is allocated uninitialized: the cast must still write every element.
            using (var line = Sample(from, 3000))
            {
                CheckView(line, a => a[":"], to, $"{from} → {to} 1-D contiguous");
                CheckView(line, a => a["::2"], to, $"{from} → {to} 1-D stepped");
            }
        }

        /// <summary>
        ///     decimal → double is bit-identical to the runtime's own <c>(double)decimal</c> on every route: the
        ///     contiguous AVX2 replica (where its start-up probe enabled it), its scalar tail, and the cast core for
        ///     strided views.
        /// </summary>
        /// <exception cref="AssertFailedException">An element's double differs from <c>(double)decimal</c>.</exception>
        /// <remarks>
        ///     The inputs are the ones that separate a faithful replica from an approximation: mantissas at and above
        ///     2⁶³ (where .NET 8's ulong → double rounds twice — the reason the kernel probes before it enables itself),
        ///     full 96-bit mantissas, every scale 0–28, negative zero and the extremes. 4099 elements = whole 4-lane
        ///     blocks plus a ragged tail. On a runtime or CPU where the probe fails, the same assertions pin the
        ///     fallback.
        /// </remarks>
        [TestMethod]
        public void DecimalToDouble_EveryRoute_IsBitIdenticalToTheRuntimeConversion()
        {
            decimal[] edges =
            {
                decimal.MaxValue, decimal.MinValue, -15805.234742681957630m, -1124262054061824103.6m,
                18446744073709551615m, 9223372036854775808m, 9223372036854775809m, -9223372036854775807m,
                0.0000000000000000000000000001m, -7.9228162514264337593543950335m, new decimal(0, 0, 0, true, 5),
                0m, 1m, -1m, 2.5m, 42.00m, -99.75m, 12345678901234567890.123456789m,
            };
            // Fully qualified: NumSharp.Tests.Random is a namespace and would shadow the type.
            var rng = new System.Random(20260923);
            var values = new decimal[4099];
            Array.Copy(edges, values, edges.Length);
            for (int i = edges.Length; i < values.Length; i++)
            {
                // (int) of a random long keeps its low 32 bits: every 32-bit pattern, top bit included.
                int lo = (int)rng.NextInt64(), mid = (int)rng.NextInt64(), hi = (int)rng.NextInt64();
                // A third each of 96-bit, 64-bit (half of them ≥ 2^63) and 32-bit mantissas.
                if (i % 3 == 1)
                    hi = 0;
                else if (i % 3 == 2)
                    hi = mid = 0;
                values[i] = new decimal(lo, mid, hi, rng.Next(2) == 1, (byte)rng.Next(29));
            }

            using var a = np.array(values);
            AssertMatchesRuntimeCast((double[])a.ToMuliDimArray<double>(), i => values[i], "1-D contiguous");

            using var window = a["0:4096"];
            using var m = window.reshape(64, 64);
            AssertMatchesRuntimeCast(Flatten<double>((double[,])m.ToMuliDimArray<double>()), i => values[i], "2-D contiguous");

            using var stepped = a["::3"];
            AssertMatchesRuntimeCast((double[])stepped.ToMuliDimArray<double>(), i => values[3 * i], "1-D stepped");

            using var t = m.T;
            AssertMatchesRuntimeCast(Flatten<double>((double[,])t.ToMuliDimArray<double>()),
                i => values[(i % 64) * 64 + i / 64], "2-D transposed");

            Check(NPTypeCode.Double, a, "1-D contiguous against astype");
        }

        /// <summary>
        ///     A 0-d array yields a ONE-element <c>T[]</c> (.NET has no rank-0 array) holding its value — for a fresh
        ///     scalar, for a 0-d view into a larger buffer, and when the value is converted.
        /// </summary>
        /// <exception cref="AssertFailedException">The result was not a one-element vector of the right value.</exception>
        /// <remarks>Before the rewrite every 0-d call threw ("Must provide at least one rank.").</remarks>
        [TestMethod]
        public void ZeroDimensional_YieldsAOneElementVector()
        {
            using var d = NDArray.Scalar(3.5);
            var r = d.ToMuliDimArray<double>();
            r.Should().BeOfType<double[]>();
            ((double[])r).Should().Equal(3.5);

            using var m = np.arange(12.0).reshape(3, 4);
            using var element = m["1, 2"];
            ((double[])element.ToMuliDimArray<double>()).Should().Equal(6.0);

            using var i = NDArray.Scalar(7);
            ((double[])i.ToMuliDimArray<double>()).Should().Equal(7.0);

            foreach (NPTypeCode dtype in AllDtypes)
            {
                using var line = Sample(dtype, 5);
                CheckView(line, a => a["3"], dtype, $"{dtype} 0-d view");
                CheckView(line, a => a["3"], NPTypeCode.Double, $"{dtype} 0-d view → float64");
            }
        }

        /// <summary>
        ///     An empty array yields an empty .NET array of the same rank and lengths — for every dtype family,
        ///     at ranks 1–6 (rank ≥ 4 through the emitted allocator), and when converting.
        /// </summary>
        /// <exception cref="AssertFailedException">The empty result had the wrong CLR type or lengths.</exception>
        /// <remarks>Before the rewrite an empty DECIMAL array threw: its per-element path read one element unconditionally.</remarks>
        [TestMethod]
        public void EmptyShapes_YieldEmptyArraysOfTheSameShape()
        {
            long[][] shapes =
            {
                new long[] { 0 }, new long[] { 3, 0 }, new long[] { 0, 4 }, new long[] { 2, 0, 5 },
                new long[] { 1, 2, 0, 3 }, new long[] { 0, 1, 1, 2, 1, 1 },
            };
            NPTypeCode[] dtypes =
            {
                NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex, NPTypeCode.Half, NPTypeCode.Boolean,
                NPTypeCode.Int32,
            };
            foreach (long[] shape in shapes)
                foreach (NPTypeCode dtype in dtypes)
                {
                    using var a = np.zeros(new Shape(shape), dtype);
                    string what = $"{dtype} empty ({string.Join(",", shape)})";
                    Check(dtype, a, what);
                    Check(dtype == NPTypeCode.Double ? NPTypeCode.Single : NPTypeCode.Double, a, what + " converted");
                }
        }

        /// <summary>
        ///     Ranks 4 through 32 come back as a <c>T[,…]</c> of exactly that rank and those lengths, holding the
        ///     C-order elements — the IL-emitted <c>newobj</c> allocator, built once per (element type, rank).
        /// </summary>
        /// <exception cref="AssertFailedException">A rank-N result had the wrong type, lengths or elements.</exception>
        /// <remarks>
        ///     Several element types per rank, since the emitted allocator is cached per (T, rank): a table mix-up
        ///     would hand one type's array to another.
        /// </remarks>
        [TestMethod]
        public void RanksFourToThirtyTwo_AllocateTheExactClrRank()
        {
            NPTypeCode[] dtypes =
            {
                NPTypeCode.Double, NPTypeCode.Boolean, NPTypeCode.Decimal, NPTypeCode.Complex, NPTypeCode.Half,
                NPTypeCode.Byte,
            };
            foreach (int rank in new[] { 4, 5, 6, 7, 8, 12, 16, 24, 31, 32 })
            {
                var dims = new long[rank];
                for (int d = 0; d < rank; d++)
                    dims[d] = d % 5 == 0 ? 2 : 1;
                dims[rank - 1] = 3;
                foreach (NPTypeCode dtype in dtypes)
                {
                    using var a = Sample(dtype, dims);
                    Check(dtype, a, $"{dtype} rank {rank}");
                }
                using var ints = Sample(NPTypeCode.Int64, dims);
                Check(NPTypeCode.Double, ints, $"int64 → float64 rank {rank}");
            }
        }

        /// <summary>
        ///     Non-contiguous rank-5 views — a full axis permutation and a stepped / reversed window — are walked in C
        ///     order by the strided copy's outer-axis odometer, with and without a conversion.
        /// </summary>
        /// <exception cref="AssertFailedException">An element landed at the wrong position.</exception>
        [TestMethod]
        public void RankFive_PermutedAndSteppedViews_CopyInCOrder()
        {
            NPTypeCode[] dtypes =
            {
                NPTypeCode.Double, NPTypeCode.Int32, NPTypeCode.Decimal, NPTypeCode.Complex, NPTypeCode.Half,
                NPTypeCode.SByte,
            };
            foreach (NPTypeCode dtype in dtypes)
            {
                using var a = Sample(dtype, 2, 3, 4, 5, 6);
                CheckView(a, v => v.transpose(new[] { 4, 2, 0, 3, 1 }), dtype, $"{dtype} transpose(4, 2, 0, 3, 1)");
                CheckView(a, v => v["::-1, :, 1:4, ::2, ::3"], dtype, $"{dtype} [::-1, :, 1:4, ::2, ::3]");
                CheckView(a, v => v.transpose(new[] { 4, 2, 0, 3, 1 }), NPTypeCode.Double, $"{dtype} permuted → float64");
            }
        }

        /// <summary>
        ///     More than 32 dimensions is the runtime's own limit on array rank: the call reports it with the runtime's
        ///     <see cref="TypeLoadException"/>, directly and through the <c>(Array)nd</c> cast.
        /// </summary>
        /// <exception cref="AssertFailedException">Rank 33 did not throw <see cref="TypeLoadException"/>.</exception>
        [TestMethod]
        public void RankAboveThirtyTwo_ThrowsTypeLoadException()
        {
            var dims = new long[33];
            Array.Fill(dims, 1L);
            dims[0] = 2;
            using var a = np.zeros(new Shape(dims));

            Action direct = () => a.ToMuliDimArray<double>();
            direct.Should().Throw<TypeLoadException>();
            Action cast = () => { _ = (Array)a; };
            cast.Should().Throw<TypeLoadException>();
        }

        /// <summary>
        ///     A dimension above <see cref="int.MaxValue"/> cannot be a .NET array length: the call throws
        ///     <see cref="InvalidOperationException"/> naming the dimension, before allocating anything — including
        ///     through the IL-emitted rank ≥ 4 allocator, which must surface the same exception unwrapped.
        /// </summary>
        /// <param name="rank">Rank of the (broadcast, allocation-free) source; the oversized dimension is the last.</param>
        /// <exception cref="AssertFailedException">The wrong exception type or text surfaced.</exception>
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(4)]
        public void DimensionAboveIntMaxValue_ThrowsInvalidOperationException(int rank)
        {
            var dims = new long[rank];
            Array.Fill(dims, 1L);
            dims[rank - 1] = (long)int.MaxValue + 1;
            using var one = NDArray.Scalar((byte)7);
            using var huge = np.broadcast_to(one, new Shape(dims));

            Action act = () => huge.ToMuliDimArray<byte>();
            act.Should().Throw<InvalidOperationException>().WithMessage("*2147483648 exceeds int.MaxValue*");
        }

        /// <summary>
        ///     An element type that is not a NumSharp dtype has no astype conversion to perform, so the call refuses it
        ///     up front with <see cref="NotSupportedException"/> naming the type.
        /// </summary>
        /// <exception cref="AssertFailedException">A non-dtype element type did not throw as specified.</exception>
        [TestMethod]
        public void NonDtypeElementType_ThrowsNotSupportedException()
        {
            using var a = np.arange(3);

            Action guid = () => a.ToMuliDimArray<Guid>();
            guid.Should().Throw<NotSupportedException>().WithMessage("*Guid is not a NumSharp dtype*");
            Action dateTime = () => a.ToMuliDimArray<DateTime>();
            dateTime.Should().Throw<NotSupportedException>().WithMessage("*DateTime is not a NumSharp dtype*");
            Action pointer = () => a.ToMuliDimArray<nint>();
            pointer.Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     The <c>(Array)nd</c> cast works for float16 and complex128 — it used to throw
        ///     <see cref="ArgumentException"/> because <c>Buffer.BlockCopy</c> accepts CLR primitives only.
        /// </summary>
        /// <exception cref="AssertFailedException">The cast threw or returned the wrong array.</exception>
        [TestMethod]
        public void ExplicitArrayCast_Float16AndComplex_Convert()
        {
            using var h = Sample(NPTypeCode.Half, 3, 4);
            var hr = (Array)h;
            hr.Should().BeOfType<Half[,]>();
            ((Half[,])hr)[2, 3].Should().Be(h.GetValue<Half>(2, 3));

            using var c = Sample(NPTypeCode.Complex, 3, 4);
            using var ct = c.T;
            var cr = (Array)ct;
            cr.Should().BeOfType<Complex[,]>();
            ((Complex[,])cr)[1, 2].Should().Be(c.GetValue<Complex>(2, 1));
        }

        /// <summary>
        ///     The result is a copy: writing it never reaches the source, and writing the source never reaches an
        ///     earlier result — for the memcpy route, the uninitialized pinned 1-D route, a 0-d value and a strided view.
        /// </summary>
        /// <exception cref="AssertFailedException">A write crossed between the result and the source.</exception>
        [TestMethod]
        public void Result_SharesNoMemoryWithTheSource()
        {
            using var a = np.arange(12.0).reshape(3, 4);
            var r = (double[,])a.ToMuliDimArray<double>();
            r[1, 2] = -1.0;
            a.GetDouble(1, 2).Should().Be(6.0, "writing the result must not reach the source");
            a.SetDouble(100.0, 0, 0);
            r[0, 0].Should().Be(0.0, "writing the source must not reach an earlier result");

            using var big = np.arange(2000.0);
            var br = (double[])big.ToMuliDimArray<double>();
            br[5] = -1.0;
            big.GetDouble(5).Should().Be(5.0, "the pinned uninitialized 1-D result is still a private copy");

            using var t = a.T;
            var tr = (double[,])t.ToMuliDimArray<double>();
            tr[0, 1] = -2.0;
            a.GetDouble(1, 0).Should().Be(4.0, "a transposed view's result is a private copy");

            using var s = NDArray.Scalar(9.0);
            var sr = (double[])s.ToMuliDimArray<double>();
            sr[0] = -3.0;
            s.GetDouble().Should().Be(9.0, "a 0-d result is a private copy");
        }

        /// <summary>
        ///     NaN payloads (quiet and signalling, both signs), signed zeros, subnormals and infinities keep their exact
        ///     bits through every strided path — the SIMD transposes, deinterleaves and gathers only MOVE lanes.
        /// </summary>
        /// <param name="dtype">A floating-point dtype: float64, float32 or float16.</param>
        /// <exception cref="AssertFailedException">A special value's bits changed on some path.</exception>
        /// <remarks>
        ///     A signalling NaN is the sharp probe: any path that routed the value through float ARITHMETIC (or an
        ///     x87 load) would quiet it by setting the top mantissa bit. The (40, 72) base puts specials at irregular
        ///     positions between ordinary values, so a misplaced element shows up too.
        /// </remarks>
        [TestMethod]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Half)]
        public void SpecialFloatBits_SurviveEveryStridedPath(NPTypeCode dtype)
        {
            using var plane = SpecialValues(dtype, 40, 72);
            foreach (var (name, make) in PlaneLayouts)
                CheckView(plane, make, dtype, $"{dtype} specials {name}");
        }

        /// <summary>
        ///     1-D results around the 1024-element threshold where the result switches to an uninitialized
        ///     pinned-object-heap allocation are fully written — by the memcpy, the one-axis strided loop and a
        ///     conversion alike.
        /// </summary>
        /// <param name="length">Result length (just below, at, just above, and well above the threshold).</param>
        /// <exception cref="AssertFailedException">An element of the result was left unwritten or wrong.</exception>
        [TestMethod]
        [DataRow(1023)]
        [DataRow(1024)]
        [DataRow(1025)]
        [DataRow(4099)]
        public void OneDimensional_AroundTheUninitializedAllocationThreshold_IsFullyWritten(int length)
        {
            NPTypeCode[] dtypes =
            {
                NPTypeCode.Double, NPTypeCode.Int32, NPTypeCode.Byte, NPTypeCode.Decimal, NPTypeCode.Complex,
                NPTypeCode.Half,
            };
            foreach (NPTypeCode dtype in dtypes)
            {
                using var a = Sample(dtype, length);
                Check(dtype, a, $"{dtype} contiguous length {length}");
                using var twice = Sample(dtype, 2L * length);
                CheckView(twice, v => v["::2"], dtype, $"{dtype} stepped length {length}");
                Check(dtype == NPTypeCode.Double ? NPTypeCode.Single : NPTypeCode.Double, a, $"{dtype} converted length {length}");
            }
        }

        // ------------------------------------------------------------------------------------------------ helpers

        /// <summary>The 15 NumSharp dtypes.</summary>
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        /// <summary>
        ///     Builds a C-contiguous array of <paramref name="dtype"/> whose elements are (nearly) all distinct, so a
        ///     misplaced element changes the bytes.
        /// </summary>
        /// <param name="dtype">Element type of the result.</param>
        /// <param name="shape">Dimensions of the result.</param>
        /// <returns>A fresh owning array; the caller disposes it.</returns>
        private static NDArray Sample(NPTypeCode dtype, params long[] shape)
        {
            long n = 1;
            foreach (long d in shape)
                n *= d;
            using var flat = Pattern(dtype, n);
            using var shaped = flat.reshape(shape);
            // An owning C-contiguous copy, so the caller holds no view onto the disposed temporaries.
            return shaped.copy();
        }

        /// <summary>Produces the 1-D element pattern <see cref="Sample"/> reshapes.</summary>
        /// <param name="dtype">Element type of the pattern.</param>
        /// <param name="n">Element count.</param>
        /// <returns>A fresh 1-D array of <paramref name="n"/> elements.</returns>
        /// <remarks>
        ///     Floats and decimals carry exact binary fractions (multiples of 5/16 around zero, exact in every width
        ///     used), so conversions to integers exercise truncation of negative fractions; float16 cycles through a
        ///     prime period (2039) to stay inside its exact range; bool is a period-3 pattern (a bare arange would be all
        ///     True after element 0); 1-byte integers cycle through the prime 251, so only a misplacement by a multiple of
        ///     251 elements could hide; complex gets a non-zero imaginary part so both halves of each 16-byte element
        ///     matter.
        /// </remarks>
        private static NDArray Pattern(NPTypeCode dtype, long n)
        {
            using var idx = np.arange(n);
            switch (dtype)
            {
                case NPTypeCode.Boolean:
                {
                    using var r = idx % 3;
                    return r == 0;
                }
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                {
                    using var r = idx % 251;
                    return r.astype(dtype);
                }
                case NPTypeCode.Half:
                {
                    using var r = idx % 2039;
                    using var centered = r - 1019;
                    using var f = centered * 0.25;
                    return f.astype(dtype);
                }
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                {
                    using var centered = idx - n / 2;
                    using var f = centered * 0.3125;
                    return f.astype(dtype);
                }
                case NPTypeCode.Complex:
                {
                    var values = new Complex[n];
                    for (long i = 0; i < n; i++)
                        values[i] = new Complex(i * 0.5 - 3, 7 - i * 0.25);
                    return np.array(values);
                }
                default:
                    return idx.astype(dtype);
            }
        }

        /// <summary>
        ///     Builds a (<paramref name="rows"/>, <paramref name="cols"/>) float array that interleaves ordinary values
        ///     with NaN payloads, signalling NaNs, signed zeros, subnormals and infinities.
        /// </summary>
        /// <param name="dtype">Float64, float32 or float16.</param>
        /// <param name="rows">Row count.</param>
        /// <param name="cols">Column count.</param>
        /// <returns>A fresh owning C-contiguous array.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="dtype"/> is not a floating-point dtype.</exception>
        private static NDArray SpecialValues(NPTypeCode dtype, int rows, int cols)
        {
            int n = rows * cols;
            switch (dtype)
            {
                case NPTypeCode.Double:
                {
                    long[] bits =
                    {
                        0x7FF8_0000_0000_0000, 0x7FF8_DEAD_BEEF_0001, 0x7FF0_0000_0000_0001,
                        unchecked((long)0xFFF0_0000_0000_0002), unchecked((long)0xFFF8_0000_0000_0000),
                        unchecked((long)0x8000_0000_0000_0000), 0x0000_0000_0000_0001,
                        unchecked((long)0x800F_FFFF_FFFF_FFFF), 0x7FF0_0000_0000_0000,
                        unchecked((long)0xFFF0_0000_0000_0000), 0x3FF0_0000_0000_0001,
                    };
                    var values = new double[n];
                    for (int i = 0; i < n; i++)
                        values[i] = i % 3 == 0 ? i * 1.5 : BitConverter.Int64BitsToDouble(bits[(i * 7) % bits.Length]);
                    using var flat = np.array(values);
                    return flat.reshape(rows, cols).copy();
                }
                case NPTypeCode.Single:
                {
                    int[] bits =
                    {
                        0x7FC0_0000, 0x7FC0_BEEF, 0x7F80_0001, unchecked((int)0xFF80_0002), unchecked((int)0xFFC0_0000),
                        unchecked((int)0x8000_0000), 0x0000_0001, unchecked((int)0x807F_FFFF), 0x7F80_0000,
                        unchecked((int)0xFF80_0000), 0x3F80_0001,
                    };
                    var values = new float[n];
                    for (int i = 0; i < n; i++)
                        values[i] = i % 3 == 0 ? i * 1.5f : BitConverter.Int32BitsToSingle(bits[(i * 7) % bits.Length]);
                    using var flat = np.array(values);
                    return flat.reshape(rows, cols).copy();
                }
                case NPTypeCode.Half:
                {
                    ushort[] bits = { 0x7E00, 0x7E01, 0x7C01, 0xFC02, 0xFE00, 0x8000, 0x0001, 0x83FF, 0x7C00, 0xFC00, 0x3C01 };
                    var values = new Half[n];
                    for (int i = 0; i < n; i++)
                        values[i] = i % 3 == 0
                            ? (Half)(i % 1024)
                            : BitConverter.Int16BitsToHalf(unchecked((short)bits[(i * 7) % bits.Length]));
                    using var flat = np.array(values);
                    return flat.reshape(rows, cols).copy();
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Only float64, float32 and float16 carry special bit patterns.");
            }
        }

        /// <summary>Builds a view of <paramref name="source"/>, checks it, and disposes the view.</summary>
        /// <param name="source">The base array.</param>
        /// <param name="make">Builds the view (or copy) under test from the base.</param>
        /// <param name="elementType">The requested element type (a conversion when it differs from the dtype).</param>
        /// <param name="what">Description for failure messages.</param>
        /// <exception cref="AssertFailedException">The conversion of the view is wrong.</exception>
        private static void CheckView(NDArray source, Func<NDArray, NDArray> make, NPTypeCode elementType, string what)
        {
            using var view = make(source);
            Check(elementType, view, what);
        }

        /// <summary>
        ///     Runs <see cref="CheckAgainstReference{T}"/> with the CLR type of <paramref name="elementType"/>.
        /// </summary>
        /// <param name="elementType">The requested element type.</param>
        /// <param name="source">The array to convert.</param>
        /// <param name="what">Description for failure messages.</param>
        /// <exception cref="AssertFailedException">The conversion is wrong.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="elementType"/> is not one of the 15 dtypes.</exception>
        private static void Check(NPTypeCode elementType, NDArray source, string what)
        {
            switch (elementType)
            {
                case NPTypeCode.Boolean: CheckAgainstReference<bool>(source, what); break;
                case NPTypeCode.Byte: CheckAgainstReference<byte>(source, what); break;
                case NPTypeCode.SByte: CheckAgainstReference<sbyte>(source, what); break;
                case NPTypeCode.Int16: CheckAgainstReference<short>(source, what); break;
                case NPTypeCode.UInt16: CheckAgainstReference<ushort>(source, what); break;
                case NPTypeCode.Int32: CheckAgainstReference<int>(source, what); break;
                case NPTypeCode.UInt32: CheckAgainstReference<uint>(source, what); break;
                case NPTypeCode.Int64: CheckAgainstReference<long>(source, what); break;
                case NPTypeCode.UInt64: CheckAgainstReference<ulong>(source, what); break;
                case NPTypeCode.Char: CheckAgainstReference<char>(source, what); break;
                case NPTypeCode.Half: CheckAgainstReference<Half>(source, what); break;
                case NPTypeCode.Single: CheckAgainstReference<float>(source, what); break;
                case NPTypeCode.Double: CheckAgainstReference<double>(source, what); break;
                case NPTypeCode.Decimal: CheckAgainstReference<decimal>(source, what); break;
                case NPTypeCode.Complex: CheckAgainstReference<Complex>(source, what); break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "Not a NumSharp dtype.");
            }
        }

        /// <summary>
        ///     Converts <paramref name="source"/> with <c>ToMuliDimArray&lt;T&gt;()</c> and asserts the CLR array type,
        ///     the lengths, and every element against the oracle (the source's own C-order walk, through astype when
        ///     <typeparamref name="T"/> is not its dtype). When no conversion is involved, the <c>(Array)nd</c> cast
        ///     must return the same.
        /// </summary>
        /// <typeparam name="T">The requested element type.</typeparam>
        /// <param name="source">The array to convert (any layout).</param>
        /// <param name="what">Description for failure messages.</param>
        /// <exception cref="AssertFailedException">Type, lengths or elements are wrong.</exception>
        private static void CheckAgainstReference<T>(NDArray source, string what) where T : unmanaged
        {
            Array actual = source.ToMuliDimArray<T>();
            T[] expected = ReferenceElements<T>(source);
            AssertClrShape<T>(actual, source, what);
            AssertElementsEqual(actual, expected, what);

            if (source.typecode == InfoOf<T>.NPTypeCode)
            {
                Array viaCast = (Array)source;
                AssertClrShape<T>(viaCast, source, what + " via (Array) cast");
                AssertElementsEqual(viaCast, expected, what + " via (Array) cast");
            }
        }

        /// <summary>
        ///     The oracle: the elements of <paramref name="source"/> in logical C order, as <typeparamref name="T"/>
        ///     — its own walk when <typeparamref name="T"/> is its dtype, else the walk of <c>astype(T)</c>.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The array to read.</param>
        /// <returns>The expected elements (one for a 0-d array).</returns>
        private static T[] ReferenceElements<T>(NDArray source) where T : unmanaged
        {
            // ToArray<T> requires T to BE the dtype (in Debug a mismatch trips a Debug.Assert), hence astype first.
            if (source.typecode == InfoOf<T>.NPTypeCode)
                return source.ToArray<T>();
            using var converted = source.astype(typeof(T));
            return converted.ToArray<T>();
        }

        /// <summary>
        ///     Asserts the CLR shape of a result: an SZ <c>T[]</c> for rank 0 (one element) and rank 1, a
        ///     <c>T[,…]</c> of the source's rank otherwise, with the source's lengths.
        /// </summary>
        /// <typeparam name="T">The requested element type.</typeparam>
        /// <param name="actual">The returned array.</param>
        /// <param name="source">The converted NDArray.</param>
        /// <param name="what">Description for failure messages.</param>
        /// <exception cref="AssertFailedException">The type or a length is wrong.</exception>
        private static void AssertClrShape<T>(Array actual, NDArray source, string what) where T : unmanaged
        {
            int ndim = source.ndim;
            Type want = ndim <= 1 ? typeof(T[]) : typeof(T).MakeArrayType(ndim);
            Assert.AreEqual(want, actual.GetType(), $"{what}: CLR array type");
            if (ndim == 0)
            {
                Assert.AreEqual(1, actual.Length, $"{what}: a 0-d array yields one element");
                return;
            }
            for (int d = 0; d < ndim; d++)
                Assert.AreEqual((int)source.shape[d], actual.GetLength(d), $"{what}: length of dimension {d}");
        }

        /// <summary>
        ///     Compares a result's elements, in memory (= C) order, with the expected ones: raw bytes for every
        ///     element type except decimal, which is compared by value.
        /// </summary>
        /// <typeparam name="T">The element type (the result's).</typeparam>
        /// <param name="actual">The returned array (any rank).</param>
        /// <param name="expected">The expected elements in C order.</param>
        /// <param name="what">Description for failure messages.</param>
        /// <exception cref="AssertFailedException">The counts differ, or the first differing element (reported with its bytes).</exception>
        private static void AssertElementsEqual<T>(Array actual, T[] expected, string what) where T : unmanaged
        {
            Assert.AreEqual(expected.Length, actual.Length, $"{what}: element count");
            // A rank-N .NET array stores its elements contiguously in row-major order: view them as a flat span.
            ReadOnlySpan<T> got = MemoryMarshal.CreateReadOnlySpan(
                ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(actual)), actual.Length);
            ReadOnlySpan<byte> gotBytes = MemoryMarshal.AsBytes(got);
            ReadOnlySpan<byte> wantBytes = MemoryMarshal.AsBytes(expected.AsSpan());
            if (typeof(T) != typeof(decimal) && gotBytes.SequenceEqual(wantBytes))
                return;

            int size = Unsafe.SizeOf<T>();
            for (int i = 0; i < expected.Length; i++)
            {
                bool same = typeof(T) == typeof(decimal)
                    ? (decimal)(object)got[i] == (decimal)(object)expected[i]
                    : gotBytes.Slice(i * size, size).SequenceEqual(wantBytes.Slice(i * size, size));
                if (!same)
                    Assert.Fail($"{what}: element {i} (C order) is {got[i]} [0x{Convert.ToHexString(gotBytes.Slice(i * size, size))}] " +
                                $"but {expected[i]} [0x{Convert.ToHexString(wantBytes.Slice(i * size, size))}] was expected");
            }
        }

        /// <summary>
        ///     Asserts each double equals — bit for bit — the runtime's <c>(double)decimal</c> of the source decimal
        ///     at the same C-order position.
        /// </summary>
        /// <param name="got">The converted doubles, in C order.</param>
        /// <param name="source">Maps a C-order position to its source decimal.</param>
        /// <param name="what">Description for failure messages.</param>
        /// <exception cref="AssertFailedException">The first element whose bits differ.</exception>
        private static void AssertMatchesRuntimeCast(double[] got, Func<int, decimal> source, string what)
        {
            for (int i = 0; i < got.Length; i++)
            {
                decimal d = source(i);
                long want = BitConverter.DoubleToInt64Bits((double)d);
                long have = BitConverter.DoubleToInt64Bits(got[i]);
                if (want != have)
                    Assert.Fail($"{what}: element {i} = {d}m converted to {got[i]:R} [0x{have:X16}] " +
                                $"but (double)decimal gives {(double)d:R} [0x{want:X16}]");
            }
        }

        /// <summary>Copies a rank-2 array's elements out in row-major order.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="a">The array.</param>
        /// <returns>A new SZ array holding <paramref name="a"/>'s elements in C order.</returns>
        private static T[] Flatten<T>(T[,] a) where T : unmanaged
        {
            var flat = new T[a.Length];
            int k = 0;
            // foreach over a multi-dimensional array enumerates in row-major order.
            foreach (T v in a)
                flat[k++] = v;
            return flat;
        }
    }
}
