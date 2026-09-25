using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;
using static NumSharp.Tests.Casting.ToArrayContractTests;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Edge cases of the three ways to copy an NDArray out to .NET arrays — <see cref="NDArray.ToArray{T}"/> (flat),
    ///     <see cref="NDArray.ToMuliDimArray{T}"/> (<c>T[,…]</c>) and <see cref="NDArray.ToJaggedArray{T}"/> (<c>T[][]…</c>)
    ///     — each case checked through all three against one independent oracle: zero extents in every position, ranks up
    ///     to 64, negative strides on every axis, buffers misaligned for their element type, broadcasts in every axis
    ///     position, bool buffers holding bytes other than 0 and 1, every (source, target) conversion, the IL kernel
    ///     generator switched off, concurrent calls, and the no-dynamic-code fallbacks.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The three APIs share a contract (the view's logical C-order elements, converted exactly as <c>astype</c>
    ///         converts them) but not a line of code, and their edges differ: ToArray returns the shared empty array for an
    ///         empty source and one element for a 0-d one; ToMuliDimArray keeps the rank (0-d → <c>T[1]</c>, rank &gt; 32
    ///         → the runtime's <see cref="TypeLoadException"/>); ToJaggedArray keeps every level (a 0-d source has no
    ///         jagged form). <see cref="ExportDifference{T}"/> holds each to its own form of the same elements.
    ///     </para>
    ///     <para>
    ///         The oracle is <see cref="ToArrayContractTests.LogicalBytes"/> — a plain stride walk over the view's buffer —
    ///         of the view, or of its <c>astype</c> result for a conversion; the sources are the ToArray contract's patterns
    ///         (random bits with planted specials; for conversions, values every conversion defines). The structure checks
    ///         of the jagged form are the ToJaggedArray contract's (<see cref="ToJaggedArrayContractTests.JaggedDifference{T}"/>).
    ///     </para>
    ///     <para>
    ///         Found by these edges, fixed with them: <c>ToJaggedArray</c> read the value of an EMPTY broadcast row
    ///         (stride 0, 0 elements), whose pointer need not be an element — an empty slice of a reversed view keeps its
    ///         parent's offset over a fresh zero-length buffer, so <c>a[::-1][0:0].ToJaggedArray&lt;long&gt;()</c>
    ///         access-violated.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ArrayExportEdgeCaseTests
    {
        // ============================================================================================== tests

        /// <summary>
        ///     Zero extents in every position — owning arrays, empty slices (of reversed, transposed and strided views), an
        ///     empty split child one past its buffer's end, and broadcasts onto zero-length axes — yield ToArray's shared
        ///     empty array, a ToMuliDimArray result of the same lengths, and a jagged result that keeps every level up to
        ///     the empty axis; same dtype and converted.
        /// </summary>
        /// <param name="dtype">The source dtype.</param>
        /// <exception cref="AssertFailedException">A case returned the wrong form, or threw.</exception>
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
        public void ZeroExtents_InEveryPosition_YieldEmptyResultsOfTheRightShape(NPTypeCode dtype)
        {
            var views = new List<(string Name, Func<NDArray, NDArray> Make)>();
            foreach (long[] shape in new[]
                     {
                         new long[] { 0 }, new long[] { 0, 3 }, new long[] { 3, 0 }, new long[] { 0, 3, 4 }, new long[] { 3, 0, 4 },
                         new long[] { 3, 4, 0 }, new long[] { 2, 1, 0, 1, 3 }, new long[] { 0, 0 }, new long[] { 1, 0, 1 },
                     })
                views.Add(($"owning ({string.Join(", ", shape)})", _ => np.empty(new Shape(shape), dtype)));
            views.AddRange(new (string, Func<NDArray, NDArray>)[]
            {
                ("[5:5]", b => b["5:5"]),
                // An empty slice of a reversed view keeps its parent's offset over a fresh zero-length buffer.
                ("[::-1][0:0]", b => b["::-1"]["0:0"]),
                ("(6, 7)[5:5, :]", b => Grid(b, 6, 7)["5:5, :"]),
                ("(6, 7)[:, 3:3]", b => Grid(b, 6, 7)[":, 3:3"]),
                ("(6, 7)[:, 3:3].T", b => Grid(b, 6, 7)[":, 3:3"].T),
                ("(6, 7)[::-1, 3:3]", b => Grid(b, 6, 7)["::-1, 3:3"]),
                ("(3, 4, 5)[:, 0:0, :]", b => Cube(b, 3, 4, 5)[":, 0:0, :"]),
                ("(3, 4, 5)[..., 0:0].transpose(2, 0, 1)", b => np.transpose(Cube(b, 3, 4, 5)["..., 0:0"], new[] { 2, 0, 1 })),
                ("(3, 4, 5)[1:1, ::-1, ::2]", b => Cube(b, 3, 4, 5)["1:1, ::-1, ::2"]),
                // Empty split children at the end: their offset points one past the buffer.
                ("split tail child of (200,)", b => np.split(b, new long[] { 200 })[1]),
                ("split tail rows of (10, 20)", b => np.split(Grid(b, 10, 20), new[] { 10 })[1]),
                // Broadcasts onto zero-length axes (stride-0 axes around an empty one).
                ("broadcast (1,) → (0,)", b => np.broadcast_to(b["0:1"], new Shape(0))),
                ("broadcast (1, 1) → (3, 0)", b => np.broadcast_to(Grid(b, 1, 1), new Shape(3, 0))),
                ("broadcast (1, 1) → (0, 3)", b => np.broadcast_to(Grid(b, 1, 1), new Shape(0, 3))),
                ("broadcast (1, 5) → (0, 5)", b => np.broadcast_to(Grid(b, 1, 5), new Shape(0, 5))),
                ("broadcast (4, 1) → (4, 0)", b => np.broadcast_to(Grid(b, 4, 1), new Shape(4, 0))),
                ("broadcast (1, 1, 1) → (2, 0, 3)", b => np.broadcast_to(Cube(b, 1, 1, 1), new Shape(2, 0, 3))),
                ("broadcast (1, 1, 1) → (2, 3, 0)", b => np.broadcast_to(Cube(b, 1, 1, 1), new Shape(2, 3, 0))),
                ("scalar broadcast → (0,)", b => np.broadcast_to(b["3"], new Shape(0))),
                ("scalar broadcast → (4, 0, 2)", b => np.broadcast_to(b["3"], new Shape(4, 0, 2))),
            });

            var log = new CaseLog();
            using var line = RawPattern(dtype, 200, 0xED6E_0001);
            foreach (NPTypeCode target in Targets(dtype))
                Visit(target, new CatalogVisitor(line, views, log, $"{dtype} empty"));
            AssertNoFailures(log, $"{dtype} zero extents", minimumCases: 3 * 29);
        }

        /// <summary>
        ///     Ranks 7 to 64 — C-contiguous, every axis reversed by a transpose, flipped on every axis, with a zero extent —
        ///     copy their C-order elements through all three APIs; ToMuliDimArray keeps the rank up to 32 and refuses more
        ///     with the runtime's <see cref="TypeLoadException"/>.
        /// </summary>
        /// <param name="dtype">The source dtype.</param>
        /// <exception cref="AssertFailedException">A case returned the wrong elements or form, or threw unexpectedly.</exception>
        /// <remarks>
        ///     <para>
        ///         The shapes spread 2 · 3 · 4 over the rank with every other axis of extent 1: an extent-1 axis adds a level
        ///         (a jagged level, a multi-dimensional rank) but no elements, so the walk's odometer, the jagged builder's
        ///         emitted loops and the rank-N allocator all run at full rank on a 24-element array.
        ///     </para>
        ///     <para>
        ///         The ranks are the ones where a route changes: 7 (past every specialized low rank), 32 (the runtime's
        ///         largest multi-dimensional array) and 33 (the first it refuses), for every dtype; and 64 (NumPy's largest)
        ///         on the float64 row, same dtype and converted.
        ///     </para>
        ///     <para>
        ///         Why 64 runs once: the first ToJaggedArray per (element type, rank) emits and JIT-compiles a
        ///         rank-specialised level builder, and .NET 8's JIT is super-linear in its nesting — measured about 13, 48
        ///         and 315 ms at ranks 16, 32 and 64 (a cold process; .NET 10: 8, 14 and 32 ms), after which a call takes
        ///         3–8 µs. The builder's IL has the same shape for every element type (only its type tokens differ), and
        ///         rank 33 already runs it past 32 levels for each dtype; running 64 for all fifteen would add about three
        ///         seconds to every .NET 8 test run and check nothing more.
        ///     </para>
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
        public void HighRank_UpTo64Dimensions_CopiesInCOrder(NPTypeCode dtype)
        {
            var views = new List<(string Name, Func<NDArray, NDArray> Make)>();
            // NumPy's largest rank on one row only: see the remarks for the measured reason.
            int[] ranks = dtype == NPTypeCode.Double ? new[] { 7, 32, 33, 64 } : new[] { 7, 32, 33 };
            foreach (int rank in ranks)
            {
                long[] dims = ToJaggedArrayContractTests.HighRankDims(rank, 24);
                long[] empty = (long[])dims.Clone();
                empty[rank / 3] = 0;
                int[] reversedAxes = Enumerable.Range(0, rank).Reverse().ToArray();
                views.Add(($"rank {rank} C", b => b["0:24"].reshape(dims)));
                views.Add(($"rank {rank} axes reversed", b => np.transpose(b["0:24"].reshape(dims), reversedAxes)));
                views.Add(($"rank {rank} flipped", b => np.flip(b["0:24"].reshape(dims))));
                views.Add(($"rank {rank} flipped, axes reversed", b => np.transpose(np.flip(b["0:24"].reshape(dims)), reversedAxes)));
                views.Add(($"rank {rank} with a zero extent", b => np.empty(new Shape(empty), b.typecode)));
            }

            var log = new CaseLog();
            using var line = RawPattern(dtype, 24, 0xED6E_0002);
            foreach (NPTypeCode target in Targets(dtype).Take(2))
                Visit(target, new CatalogVisitor(line, views, log, $"{dtype} high rank"));
            AssertNoFailures(log, $"{dtype} high ranks", minimumCases: 2 * 5 * ranks.Length);
        }

        /// <summary>
        ///     Negative strides on every axis — reversed lines, planes reversed on both axes (small and past the tiled
        ///     transposes' sizes), their transposes, negative steps other than −1, flipped 3-D and 4-D views and their
        ///     permutations — copy their C-order elements through all three APIs, same dtype and converted.
        /// </summary>
        /// <param name="dtype">The source dtype.</param>
        /// <exception cref="AssertFailedException">A case returned the wrong elements or form, or threw.</exception>
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
        public void NegativeStridesOnEveryAxis_CopyInCOrder(NPTypeCode dtype)
        {
            var views = new (string Name, Func<NDArray, NDArray> Make)[]
            {
                ("[0:1][::-1]", b => b["0:1"]["::-1"]),
                ("[0:5][::-1]", b => b["0:5"]["::-1"]),
                ("[0:3000][::-1]", b => b["0:3000"]["::-1"]),
                ("[0:2000][::-2]", b => b["0:2000"]["::-2"]),
                ("(37, 41)[::-1, ::-1]", b => Grid(b, 37, 41)["::-1, ::-1"]),
                ("(300, 301)[::-1, ::-1]", b => Grid(b, 300, 301)["::-1, ::-1"]),
                ("(37, 41)[::-2, ::-3]", b => Grid(b, 37, 41)["::-2, ::-3"]),
                ("(37, 41)[::-1, ::-1].T", b => Grid(b, 37, 41)["::-1, ::-1"].T),
                ("(300, 301)[::-1, ::-1].T", b => Grid(b, 300, 301)["::-1, ::-1"].T),
                ("(8, 2000)[:, ::-1].T", b => Grid(b, 8, 2000)[":, ::-1"].T),
                ("(2000, 8)[::-1, :].T", b => Grid(b, 2000, 8)["::-1, :"].T),
                ("(5, 6, 7) flipped", b => np.flip(Cube(b, 5, 6, 7))),
                ("(5, 6, 7)[::-1, ::2, ::-3]", b => Cube(b, 5, 6, 7)["::-1, ::2, ::-3"]),
                ("(5, 6, 7) flipped, transpose(1, 2, 0)", b => np.transpose(np.flip(Cube(b, 5, 6, 7)), new[] { 1, 2, 0 })),
                ("(2, 3, 4, 5) flipped", b => np.flip(b["0:120"].reshape(2, 3, 4, 5))),
                ("(2, 3, 4, 5) flipped, transpose(3, 1, 2, 0)", b => np.transpose(np.flip(b["0:120"].reshape(2, 3, 4, 5)), new[] { 3, 1, 2, 0 })),
            };

            var log = new CaseLog();
            using var line = RawPattern(dtype, 100_000, 0xED6E_0003);
            foreach (NPTypeCode target in Targets(dtype).Take(2))
                Visit(target, new CatalogVisitor(line, views, log, $"{dtype} negative strides"));
            AssertNoFailures(log, $"{dtype} negative strides", minimumCases: 2 * 16);
        }

        /// <summary>
        ///     An array over a MANAGED buffer at a byte offset that misaligns its elements (np.frombuffer at offset 1, and
        ///     at half the element size up to 4 — a 2-byte-aligned int32, a 4-byte-aligned int64) copies and converts
        ///     exactly through all three APIs — lines of every size class, reversed and stepped, planes and their
        ///     transposes (small, and with rows past the jagged big-leaf size for every dtype), and a permuted 3-D view — so
        ///     no route may assume an element-aligned source (the vector loads are unaligned; a dereference of a misaligned
        ///     <c>T*</c> is fine on x64 and arm64).
        /// </summary>
        /// <param name="dtype">A dtype of at least 2 bytes (a 1-byte element cannot be misaligned).</param>
        /// <exception cref="AssertFailedException">A case returned the wrong elements or form, or threw.</exception>
        /// <remarks>
        ///     The routes that stream a large copy align their DESTINATION and load the source unaligned, so a misaligned
        ///     source needs no mebibyte-sized case: the sizes here are the ones where a route is picked (short lines, rows
        ///     of 2 048 bytes and more, the tiled transposes), each under two misalignments and two element types.
        /// </remarks>
        [TestMethod]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public unsafe void MisalignedManagedBuffer_CopiesAndConvertsExactly(NPTypeCode dtype)
        {
            int size = dtype.SizeOf();
            const int Count = 15_000;
            var views = new List<(string Name, Func<NDArray, NDArray> Make)>();
            foreach (long n in new long[] { 1, 7, 64, 1000, 5000 })
            {
                views.Add(($"[0:{n}]", b => b[$"0:{n}"]));
                views.Add(($"[0:{n}][::-1]", b => b[$"0:{n}"]["::-1"]));
                views.Add(($"[0:{2 * n}][::2]", b => b[$"0:{2 * n}"]["::2"]));
            }
            views.Add(("(40, 60)", b => Grid(b, 40, 60)));
            views.Add(("(40, 60).T", b => Grid(b, 40, 60).T));
            views.Add(("(300, 40).T", b => Grid(b, 300, 40).T));
            views.Add(("(4, 30, 20).transpose(2, 0, 1)", b => np.transpose(Cube(b, 4, 30, 20), new[] { 2, 0, 1 })));
            // Rows of 1 100 elements: at least 2 200 bytes for every dtype, past the jagged big-leaf size (2 048).
            views.Add(("(12, 1100)", b => Grid(b, 12, 1100)));
            views.Add(("(12, 1100).T", b => Grid(b, 12, 1100).T));

            var log = new CaseLog();
            using var pattern = RawPattern(dtype, Count, 0xED6E_0004);
            ReadOnlySpan<byte> patternBytes = pattern.Unsafe.ReadOnlyBytes();
            // Offset 1 misaligns every element size; half the element size (up to 4) leaves a coarser alignment that is
            // still not the element's own — the case a route that only checks "is this address even" would get wrong.
            foreach (int offset in new[] { 1, System.Math.Min(size / 2, 4) }.Distinct())
            {
                // The pattern's bytes at a byte offset into a managed array: np.frombuffer wraps it in place (a view).
                var buffer = new byte[offset + patternBytes.Length];
                patternBytes.CopyTo(buffer.AsSpan(offset));
                using NDArray wrapped = np.frombuffer(buffer, dtype, Count, offset);
                ((long)wrapped.Storage.Address % size).Should().NotBe(0, $"offset {offset} must misalign a {size}-byte element");
                foreach (NPTypeCode target in Targets(dtype).Take(2))
                    Visit(target, new CatalogVisitor(wrapped, views, log, $"{dtype} misaligned +{offset}"));
            }
            AssertNoFailures(log, $"{dtype} misaligned buffers", minimumCases: 2 * 21);
        }

        /// <summary>
        ///     Broadcasts in every axis position — one value everywhere (up to a large plane), a repeated row (plain,
        ///     stepped, reversed), a repeated column (and its transpose, and long enough for the tiled transposes and the
        ///     16-byte tiles), a repeated middle axis, a repeated leading axis over a transposed or F-ordered plane — copy
        ///     the repeated values through all three APIs, every jagged row its own array; same dtype and converted.
        /// </summary>
        /// <param name="dtype">The source dtype.</param>
        /// <exception cref="AssertFailedException">A case returned the wrong elements or form, shared a row, or threw.</exception>
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
        public void BroadcastViews_InEveryAxisPosition_CopyTheRepeatedValues(NPTypeCode dtype)
        {
            var views = new (string Name, Func<NDArray, NDArray> Make)[]
            {
                ("scalar → (3,)", b => np.broadcast_to(b["3"], new Shape(3))),
                ("scalar → (5, 7)", b => np.broadcast_to(b["3"], new Shape(5, 7))),
                ("scalar → (2, 3, 4)", b => np.broadcast_to(b["3"], new Shape(2, 3, 4))),
                ("scalar → (70, 150)", b => np.broadcast_to(b["3"], new Shape(70, 150))),
                ("row (300,) → (50, 300)", b => np.broadcast_to(b["0:300"], new Shape(50, 300))),
                ("stepped row [0:300:2] → (7, 150)", b => np.broadcast_to(b["0:300:2"], new Shape(7, 150))),
                ("reversed row [299::-2] → (7, 150)", b => np.broadcast_to(b["299::-2"], new Shape(7, 150))),
                ("column (300, 1) → (300, 50)", b => np.broadcast_to(np.expand_dims(b["0:300"], 1), new Shape(300, 50))),
                ("column (300, 1) → (300, 50), transposed", b => np.broadcast_to(np.expand_dims(b["0:300"], 1), new Shape(300, 50)).T),
                ("column (40, 1) → (40, 3000)", b => np.broadcast_to(np.expand_dims(b["0:40"], 1), new Shape(40, 3000))),
                ("column (35, 1) → (35, 1030)", b => np.broadcast_to(np.expand_dims(b["0:35"], 1), new Shape(35, 1030))),
                ("middle (4, 1, 9) → (4, 6, 9)", b => np.broadcast_to(Cube(b, 4, 1, 9), new Shape(4, 6, 9))),
                ("middle (4, 1, 9) → (4, 6, 9), transpose(1, 2, 0)", b => np.transpose(np.broadcast_to(Cube(b, 4, 1, 9), new Shape(4, 6, 9)), new[] { 1, 2, 0 })),
                ("leading (5, 6) → (3, 5, 6)", b => np.broadcast_to(Grid(b, 5, 6), new Shape(3, 5, 6))),
                ("leading (5, 6).T → (3, 6, 5)", b => np.broadcast_to(Grid(b, 5, 6).T, new Shape(3, 6, 5))),
                ("leading F-contiguous (5, 6) → (2, 5, 6)", b => np.broadcast_to(np.asfortranarray(Grid(b, 5, 6)), new Shape(2, 5, 6))),
            };

            var log = new CaseLog();
            using var line = RawPattern(dtype, 20_000, 0xED6E_0005);
            foreach (NPTypeCode target in Targets(dtype))
                Visit(target, new CatalogVisitor(line, views, log, $"{dtype} broadcast"));
            AssertNoFailures(log, $"{dtype} broadcasts", minimumCases: 3 * 16);
        }

        /// <summary>
        ///     A bool buffer holding bytes other than 0 and 1 (a zero-copy <c>np.frombuffer</c> view, as in NumPy) is
        ///     COPIED verbatim — the same-dtype results keep every byte, as NumPy's <c>copy()</c> does — and CONVERTED with
        ///     NumPy's rule, any non-zero byte → 1 (True), on every route of all three APIs.
        /// </summary>
        /// <exception cref="AssertFailedException">A copy changed a byte, or a conversion read a byte as its value.</exception>
        /// <remarks>
        ///     Probed (numpy 2.4.2): <c>b = np.frombuffer(b'\x00\x01\x02\x80\xff', bool)</c> → <c>[False, True, True, True,
        ///     True]</c>; <c>b.copy().view(np.uint8)</c> → <c>[0, 1, 2, 128, 255]</c>; <c>b.astype(t)</c> → <c>[0, 1, 1, 1,
        ///     1]</c> for every integer, float and complex <c>t</c>. The conversions are compared with astype (the contract)
        ///     and astype itself with that rule, element by element from the raw bytes.
        /// </remarks>
        [TestMethod]
        public void NonCanonicalBooleanBytes_CopyVerbatim_AndConvertAsTrue()
        {
            byte[] codes = { 0, 1, 2, 0x80, 0xFF };
            var raw = new byte[4000];
            for (int i = 0; i < raw.Length; i++)
                raw[i] = codes[(i * 7 + i / 5) % codes.Length];
            using NDArray flags = np.frombuffer(raw, NPTypeCode.Boolean);

            var views = new (string Name, Func<NDArray, NDArray> Make)[]
            {
                ("contiguous (4000,)", b => b[":"]),
                ("[0:13]", b => b["0:13"]),
                ("[::3]", b => b["::3"]),
                ("[::-1]", b => b["::-1"]),
                ("(40, 100)", b => b.reshape(40, 100)),
                ("(40, 100).T", b => b.reshape(40, 100).T),
                ("0-d [2]", b => b["2"]),
                ("[2:3]", b => b["2:3"]),
                ("split child [7:47]", b => SplitChild(b, 7, 40)),
                ("scalar broadcast of [3] → (5, 7)", b => np.broadcast_to(b["3"], new Shape(5, 7))),
                ("(4, 10, 100).transpose(2, 0, 1)", b => np.transpose(b.reshape(4, 10, 100), new[] { 2, 0, 1 })),
            };

            var log = new CaseLog();
            foreach (NPTypeCode target in AllDtypes)
                Visit(target, new CatalogVisitor(flags, views, log, "non-canonical bool"));

            // astype itself follows NumPy's rule: every non-zero byte converts to 1.
            foreach (var (name, make) in views)
            {
                log.Cases++;
                using NDArray view = make(flags);
                byte[] bytes = LogicalBytes(view);
                using NDArray asInt = view.astype(NPTypeCode.Int32);
                int[] got = asInt.ToArray<int>();
                for (int i = 0; i < bytes.Length; i++)
                    if (got[i] != (bytes[i] != 0 ? 1 : 0))
                    {
                        log.Fail($"non-canonical bool {name}: astype(int32) element {i} is {got[i]} for byte 0x{bytes[i]:X2}");
                        break;
                    }
            }
            AssertNoFailures(log, "non-canonical bools", minimumCases: 15 * 11 + 11);
        }

        /// <summary>
        ///     Every (source dtype, target) conversion — all 210 — through all three APIs over the layouts that pick their
        ///     conversion routes (a contiguous run, a strided and a reversed one, a transposed plane, a permuted 3-D view, a
        ///     broadcast, a 0-d and a one-element view, a split child and an empty view), each compared with astype.
        /// </summary>
        /// <param name="from">The source dtype; every other dtype is a target.</param>
        /// <exception cref="AssertFailedException">A case differed from astype or returned the wrong form.</exception>
        /// <remarks>
        ///     ToArray and ToJaggedArray have per-pair catalogs of their own (their contracts); this one gives
        ///     ToMuliDimArray — whose own suite lists 27 pairs — the full matrix, next to the other two on the same views.
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
        public void Conversion_EveryPair_EveryApiMatchesAstype(NPTypeCode from)
        {
            var views = new (string Name, Func<NDArray, NDArray> Make)[]
            {
                ("[0:33]", b => b["0:33"]),
                ("[0:3000]", b => b["0:3000"]),
                ("[0:66][::2]", b => b["0:66"]["::2"]),
                ("[0:40][::-1]", b => b["0:40"]["::-1"]),
                ("(7, 9).T", b => Grid(b, 7, 9).T),
                ("(40, 60).T", b => Grid(b, 40, 60).T),
                ("(3, 4, 5).transpose(2, 0, 1)", b => np.transpose(Cube(b, 3, 4, 5), new[] { 2, 0, 1 })),
                ("row broadcast (5, 7)", b => np.broadcast_to(b["0:7"], new Shape(5, 7))),
                ("scalar broadcast (4, 6)", b => np.broadcast_to(b["3"], new Shape(4, 6))),
                ("0-d [3]", b => b["3"]),
                ("[5:6]", b => b["5:6"]),
                ("split child [3:36]", b => SplitChild(b, 3, 33)),
                ("empty (6, 7)[:, 3:3]", b => Grid(b, 6, 7)[":, 3:3"]),
            };

            var log = new CaseLog();
            foreach (NPTypeCode to in AllDtypes)
            {
                if (to == from)
                    continue;
                using var line = ConversionPattern(from, to, 4_000, 0xED6E_0100UL + (ulong)from * 31 + (ulong)to);
                Visit(to, new CatalogVisitor(line, views, log, $"{from} → {to}"));
            }
            AssertNoFailures(log, $"{from} → every other dtype, all three APIs", minimumCases: 14 * 13);
        }

        /// <summary>
        ///     With NumSharp's IL kernel generator switched off (<see cref="DirectILKernelGenerator.Enabled"/>, a public
        ///     switch), all three APIs still copy and convert exactly what they do with it on — the no-kernel fallbacks of
        ///     every route: the remembered kernel slots must stop answering, the row converters and the iterator fall back
        ///     to per-element conversion.
        /// </summary>
        /// <exception cref="AssertFailedException">A case differed from the results computed with the generator on.</exception>
        /// <remarks>
        ///     The expected bytes are computed first, with the generator ON (astype through the oracle); only then is it
        ///     switched off. The switch is process-wide, so the test is <c>[DoNotParallelize]</c> and restores it in a
        ///     <c>finally</c>.
        /// </remarks>
        [TestMethod]
        [DoNotParallelize]
        public void KernelGeneratorSwitchedOff_EveryApiStillCopiesAndConverts()
        {
            var views = new (string Name, Func<NDArray, NDArray> Make)[]
            {
                ("[0:3000]", b => b["0:3000"]),
                ("[0:66][::2]", b => b["0:66"]["::2"]),
                ("(40, 60).T", b => Grid(b, 40, 60).T),
                ("(3, 4, 5).transpose(2, 0, 1)", b => np.transpose(Cube(b, 3, 4, 5), new[] { 2, 0, 1 })),
                ("row broadcast (5, 7)", b => np.broadcast_to(b["0:7"], new Shape(5, 7))),
                ("0-d [3]", b => b["3"]),
                ("empty [5:5]", b => b["5:5"]),
            };
            NPTypeCode[] sources =
            {
                NPTypeCode.Double, NPTypeCode.Int32, NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Byte, NPTypeCode.Decimal,
                NPTypeCode.Boolean, NPTypeCode.UInt64,
            };

            // Expectations first, with the generator on.
            var cases = new List<FixedCase>();
            var owned = new List<NDArray>();
            try
            {
                foreach (NPTypeCode from in sources)
                    foreach (NPTypeCode to in Targets(from).Append(from == NPTypeCode.Single ? NPTypeCode.Int64 : NPTypeCode.Single))
                    {
                        var line = to == from ? RawPattern(from, 4_000, 0xED6E_0200) : ConversionPattern(from, to, 4_000, 0xED6E_0200UL + (ulong)from * 31 + (ulong)to);
                        owned.Add(line);
                        foreach (var (name, make) in views)
                        {
                            NDArray view = make(line);
                            owned.Add(view);
                            cases.Add(new FixedCase($"{from} → {to} {name}", view, to, ExpectedBytes(view, to), to == NPTypeCode.Decimal && from != to));
                        }
                    }

                var log = new CaseLog();
                DirectILKernelGenerator.Enabled = false;
                try
                {
                    foreach (FixedCase c in cases)
                        Visit(c.Target, new FixedCaseVisitor(c, log));
                }
                finally
                {
                    DirectILKernelGenerator.Enabled = true;
                }
                AssertNoFailures(log, "IL kernel generator switched off", minimumCases: sources.Length * 4 * views.Length);
            }
            finally
            {
                foreach (NDArray a in owned)
                    a.Dispose();
            }
        }

        /// <summary>
        ///     Sixteen threads calling all three APIs at once — on shared sources, into targets and ranks whose kernels,
        ///     remembered kernel slots, jagged-level builders and rank-N allocators are created on first use — all get
        ///     exactly the single-threaded results.
        /// </summary>
        /// <exception cref="AssertFailedException">A concurrent call returned something else, or threw.</exception>
        /// <remarks>
        ///     The first-use caches are written without locks (a race only builds the same delegate twice, and a reference
        ///     store is atomic); this pins that claim. Each thread walks the cases in its own order so first uses collide.
        /// </remarks>
        [TestMethod]
        public void ConcurrentCalls_FromManyThreads_GetTheSingleThreadedResults()
        {
            var owned = new List<NDArray>();
            var cases = new List<FixedCase>();
            try
            {
                // Ranks 1 … 11 (the jagged builders from 3, the rank-N allocator from 4), mixed dtypes and conversions,
                // including element types few other tests request.
                NPTypeCode[] dtypes = { NPTypeCode.UInt16, NPTypeCode.Char, NPTypeCode.SByte, NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal };
                for (int rank = 1; rank <= 11; rank++)
                {
                    long[] dims = new long[rank];
                    Array.Fill(dims, 1L);
                    dims[0] = 3;
                    dims[rank - 1] *= 4;
                    if (rank >= 3)
                        dims[rank / 2] *= 2;
                    long count = dims.Aggregate(1L, (a, d) => a * d);
                    NPTypeCode from = dtypes[rank % dtypes.Length];
                    foreach (NPTypeCode to in new[] { from, dtypes[(rank + 1) % dtypes.Length], NPTypeCode.UInt32 })
                    {
                        var line = to == from ? RawPattern(from, count, 0xED6E_0300) : ConversionPattern(from, to, count, 0xED6E_0300UL + (ulong)rank);
                        owned.Add(line);
                        NDArray c = line.reshape(dims);
                        NDArray flipped = np.flip(c);
                        NDArray permuted = np.transpose(c, Enumerable.Range(0, rank).Reverse().ToArray());
                        owned.AddRange(new[] { c, flipped, permuted });
                        foreach (var (name, view) in new[] { ("C", c), ("flipped", flipped), ("axes reversed", permuted) })
                            cases.Add(new FixedCase($"rank {rank} {from} → {to} {name}", view, to, ExpectedBytes(view, to), to == NPTypeCode.Decimal && from != to));
                    }
                }

                var failures = new ConcurrentQueue<string>();
                Parallel.For(0, 16, new ParallelOptions { MaxDegreeOfParallelism = 16 }, thread =>
                {
                    var order = Enumerable.Range(0, cases.Count).OrderBy(i => (i * 7919 + thread * 104729) % cases.Count).ToArray();
                    for (int round = 0; round < 8; round++)
                        foreach (int i in order)
                        {
                            var log = new CaseLog();
                            Visit(cases[i].Target, new FixedCaseVisitor(cases[i], log));
                            foreach (string f in log.Failures)
                                failures.Enqueue($"thread {thread} round {round}: {f}");
                        }
                });
                if (!failures.IsEmpty)
                    Assert.Fail($"{failures.Count} concurrent failures; first:\n  " + string.Join("\n  ", failures.Take(20)));
                cases.Count.Should().BeGreaterThanOrEqualTo(11 * 3 * 3);
            }
            finally
            {
                foreach (NDArray a in owned)
                    a.Dispose();
            }
        }

        /// <summary>
        ///     The no-dynamic-code fallbacks — ToJaggedArray's recursive descent over reflection-created levels (rank ≥ 3),
        ///     and ToMuliDimArray's <see cref="Array.CreateInstance(Type, int[])"/> allocation (rank ≥ 4) — produce exactly
        ///     what the emitted code produces, same dtype and converted, for every dtype. They are what NativeAOT and the
        ///     interpreter run; a JIT process reaches them only through the thread-static test hook
        ///     <c>NDArray.ForceNoDynamicCodeFallbacks</c>.
        /// </summary>
        /// <exception cref="AssertFailedException">A fallback result differed from the oracle, or threw.</exception>
        [TestMethod]
        public void NoDynamicCodeFallbacks_ProduceWhatTheEmittedCodeProduces()
        {
            var views = new (string Name, Func<NDArray, NDArray> Make)[]
            {
                ("(2, 3, 4) C", b => Cube(b, 2, 3, 4)),
                ("(3, 4, 5) transpose(2, 0, 1)", b => np.transpose(Cube(b, 3, 4, 5), new[] { 2, 0, 1 })),
                ("(3, 4, 5)[::-1, :, ::2]", b => Cube(b, 3, 4, 5)["::-1, :, ::2"]),
                ("(2, 3, 300) C, big leaves", b => Cube(b, 2, 3, 300)),
                ("(2, 3, 4, 5) transpose(3, 1, 0, 2)", b => np.transpose(b["0:120"].reshape(2, 3, 4, 5), new[] { 3, 1, 0, 2 })),
                ("6-D flipped", b => np.flip(b["0:72"].reshape(2, 1, 3, 1, 2, 6))),
                ("(2, 0, 4) empty", b => np.empty(new Shape(2, 0, 4), b.typecode)),
                ("rank 32", b => b["0:24"].reshape(ToJaggedArrayContractTests.HighRankDims(32, 24))),
                ("rank 12 flipped", b => np.flip(b["0:24"].reshape(ToJaggedArrayContractTests.HighRankDims(12, 24)))),
            };

            var log = new CaseLog();
            NDArray.ForceNoDynamicCodeFallbacks = true;
            try
            {
                foreach (NPTypeCode dtype in AllDtypes)
                {
                    using var line = RawPattern(dtype, 2_000, 0xED6E_0400);
                    foreach (NPTypeCode target in Targets(dtype).Take(2))
                        Visit(target, new CatalogVisitor(line, views, log, $"{dtype} no dynamic code"));
                }
            }
            finally
            {
                NDArray.ForceNoDynamicCodeFallbacks = false;
            }
            AssertNoFailures(log, "no-dynamic-code fallbacks", minimumCases: 15 * 2 * 9);
        }

        // ============================================================================================== the check

        /// <summary>
        ///     Checks one view through all three APIs against the expected elements: ToArray (the flat copy; the shared empty
        ///     array for an empty view), ToMuliDimArray (the exact rank and lengths, <c>T[1]</c> for a 0-d view, the
        ///     runtime's <see cref="TypeLoadException"/> above rank 32) and ToJaggedArray (the exact nested types and
        ///     lengths, no shared array; <see cref="InvalidOperationException"/> for a 0-d view).
        /// </summary>
        /// <typeparam name="T">The requested element type.</typeparam>
        /// <param name="view">The source (any layout; not modified).</param>
        /// <param name="want">The expected elements as raw bytes, in C order.</param>
        /// <param name="byValue">Compare decimals by value (a conversion's result) instead of by bits.</param>
        /// <returns>Null when all three match; else the first problem, prefixed with the API.</returns>
        internal static string ExportDifference<T>(NDArray view, byte[] want, bool byValue) where T : unmanaged
        {
            int nd = view.ndim;

            T[] flat = view.ToArray<T>();
            string problem = ElementsDifference<T>(flat, want, byValue);
            if (problem != null)
                return "ToArray: " + problem;
            if (view.size == 0 && !ReferenceEquals(flat, Array.Empty<T>()))
                return $"ToArray: an empty result must be the shared Array.Empty<{typeof(T).Name}>()";

            if (nd > 32)
            {
                try
                {
                    view.ToMuliDimArray<T>();
                    return $"ToMuliDimArray: rank {nd} did not throw TypeLoadException";
                }
                catch (TypeLoadException)
                {
                    // The runtime's own limit on array rank: expected.
                }
            }
            else
            {
                Array md = view.ToMuliDimArray<T>();
                Type expected = nd <= 1 ? typeof(T[]) : typeof(T).MakeArrayType(nd);
                if (md.GetType() != expected)
                    return $"ToMuliDimArray: a {md.GetType()} where a {expected} was expected";
                if (nd == 0 && md.Length != 1)
                    return $"ToMuliDimArray: a 0-d source gave {md.Length} elements, not 1";
                for (int d = 0; d < nd; d++)
                    if (md.GetLength(d) != view.shape[d])
                        return $"ToMuliDimArray: length {md.GetLength(d)} of dimension {d}, not {view.shape[d]}";
                // A rank-N .NET array stores its elements contiguously in row-major order: read them as one span.
                ReadOnlySpan<T> elements = MemoryMarshal.CreateReadOnlySpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(md)), md.Length);
                problem = ElementsDifference(elements, want, byValue);
                if (problem != null)
                    return "ToMuliDimArray: " + problem;
            }

            if (nd == 0)
            {
                try
                {
                    view.ToJaggedArray<T>();
                    return "ToJaggedArray: a 0-d source did not throw";
                }
                catch (InvalidOperationException)
                {
                    // A 0-d array has no jagged form: expected.
                }
            }
            else
            {
                var arrays = new HashSet<object>(ReferenceEqualityComparer.Instance);
                problem = ToJaggedArrayContractTests.JaggedDifference<T>(view.ToJaggedArray<T>(), view.shape, want, byValue, arrays);
                if (problem != null)
                    return "ToJaggedArray: " + problem;
            }
            return null;
        }

        /// <summary>Describes the first element of <paramref name="got"/> that differs from <paramref name="want"/>, or null.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="got">The elements a call returned, in C order.</param>
        /// <param name="want">The expected elements as raw bytes, in C order.</param>
        /// <param name="byValue">Compare decimals by value instead of by bits.</param>
        /// <returns>Null on a match; else the length mismatch or the first differing element with both byte patterns.</returns>
        private static string ElementsDifference<T>(ReadOnlySpan<T> got, byte[] want, bool byValue) where T : unmanaged
        {
            int size = Unsafe.SizeOf<T>();
            if ((long)got.Length * size != want.Length)
                return $"{got.Length} elements but {want.Length / size} were expected";
            ReadOnlySpan<byte> gotBytes = MemoryMarshal.AsBytes(got);
            if (!byValue && gotBytes.SequenceEqual(want))
                return null;
            ReadOnlySpan<T> wantElements = MemoryMarshal.Cast<byte, T>(want.AsSpan());
            for (int i = 0; i < got.Length; i++)
            {
                bool same = byValue && typeof(T) == typeof(decimal)
                    ? (decimal)(object)got[i] == (decimal)(object)wantElements[i]
                    : gotBytes.Slice(i * size, size).SequenceEqual(want.AsSpan(i * size, size));
                if (!same)
                    return $"element {i} (C order) is {got[i]} [0x{Convert.ToHexString(gotBytes.Slice(i * size, size))}] " +
                           $"but {wantElements[i]} [0x{Convert.ToHexString(want.AsSpan(i * size, size))}] was expected";
            }
            return null;
        }

        /// <summary>The expected bytes of <paramref name="view"/> requested as <paramref name="target"/>: its own, or its astype result's.</summary>
        /// <param name="view">The source.</param>
        /// <param name="target">The requested element type.</param>
        /// <returns>The oracle's bytes (C order).</returns>
        private static byte[] ExpectedBytes(NDArray view, NPTypeCode target)
        {
            if (view.typecode == target)
                return LogicalBytes(view);
            using NDArray converted = view.astype(target);
            return LogicalBytes(converted);
        }

        /// <summary>The element types a case is checked as: the dtype itself, a float target and an integer target (never the dtype twice).</summary>
        /// <param name="dtype">The source dtype.</param>
        /// <returns>Three distinct targets, the dtype first.</returns>
        private static NPTypeCode[] Targets(NPTypeCode dtype) => new[]
        {
            dtype,
            dtype == NPTypeCode.Double ? NPTypeCode.Int16 : NPTypeCode.Double,
            dtype == NPTypeCode.SByte ? NPTypeCode.Complex : NPTypeCode.SByte,
        };

        // ============================================================================================== machinery

        /// <summary>
        ///     Runs a catalog of views of one source through <see cref="ExportDifference{T}"/> as the visited element type,
        ///     checking the source's bytes after every case.
        /// </summary>
        private sealed class CatalogVisitor : IElementTypeVisitor
        {
            /// <summary>The array every view reads.</summary>
            private readonly NDArray _source;

            /// <summary>The views to check, each built from the source.</summary>
            private readonly IReadOnlyList<(string Name, Func<NDArray, NDArray> Make)> _views;

            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Prefix of every failure message.</summary>
            private readonly string _prefix;

            /// <summary>Creates the visitor.</summary>
            /// <param name="source">A C-contiguous source (its bytes are snapshotted per case).</param>
            /// <param name="views">The views to check.</param>
            /// <param name="log">The log the cases record into.</param>
            /// <param name="prefix">Prefix of every failure message.</param>
            public CatalogVisitor(NDArray source, IReadOnlyList<(string Name, Func<NDArray, NDArray> Make)> views, CaseLog log, string prefix)
            {
                _source = source;
                _views = views;
                _log = log;
                _prefix = prefix;
            }

            /// <inheritdoc/>
            public void Visit<T>() where T : unmanaged
            {
                byte[] snapshot = _source.Unsafe.ReadOnlyBytes().ToArray();
                foreach (var (name, make) in _views)
                {
                    _log.Cases++;
                    string what = $"{_prefix} as {typeof(T).Name}: {name}";
                    NDArray view = null;
                    try
                    {
                        view = make(_source);
                        what += $" shape ({string.Join(", ", view.shape)})";
                        bool conversion = view.typecode != InfoOf<T>.NPTypeCode;
                        byte[] want = ExpectedBytes(view, InfoOf<T>.NPTypeCode);
                        string problem = ExportDifference<T>(view, want, conversion && typeof(T) == typeof(decimal));
                        if (problem != null)
                            _log.Fail($"{what}: {problem}");
                    }
                    catch (Exception e)
                    {
                        _log.Fail($"{what}: threw {e.GetType().Name}: {e.Message}");
                    }
                    finally
                    {
                        // A view built from nothing (an owning empty array) is its own array: disposed too.
                        if (!ReferenceEquals(view, _source))
                            view?.Dispose();
                    }

                    if (!_source.Unsafe.ReadOnlyBytes().SequenceEqual(snapshot))
                    {
                        _log.Fail($"{what}: a call wrote into its source's buffer");
                        snapshot.CopyTo(_source.Unsafe.Bytes());
                    }
                }
            }
        }

        /// <summary>A view with its target and the bytes computed for it beforehand (under other conditions).</summary>
        /// <param name="Name">The case's description.</param>
        /// <param name="View">The source view (owned by the test).</param>
        /// <param name="Target">The requested element type.</param>
        /// <param name="Want">The expected elements as raw bytes, in C order.</param>
        /// <param name="ByValue">Compare decimals by value (a conversion's result).</param>
        private sealed record FixedCase(string Name, NDArray View, NPTypeCode Target, byte[] Want, bool ByValue);

        /// <summary>Checks one <see cref="FixedCase"/> through <see cref="ExportDifference{T}"/> as the visited element type.</summary>
        private sealed class FixedCaseVisitor : IElementTypeVisitor
        {
            /// <summary>The case.</summary>
            private readonly FixedCase _case;

            /// <summary>The log the case records into.</summary>
            private readonly CaseLog _log;

            /// <summary>Creates the visitor.</summary>
            /// <param name="c">The case.</param>
            /// <param name="log">The log the case records into.</param>
            public FixedCaseVisitor(FixedCase c, CaseLog log)
            {
                _case = c;
                _log = log;
            }

            /// <inheritdoc/>
            public void Visit<T>() where T : unmanaged
            {
                _log.Cases++;
                try
                {
                    string problem = ExportDifference<T>(_case.View, _case.Want, _case.ByValue);
                    if (problem != null)
                        _log.Fail($"{_case.Name}: {problem}");
                }
                catch (Exception e)
                {
                    _log.Fail($"{_case.Name}: threw {e.GetType().Name}: {e.Message}");
                }
            }
        }
    }
}
