using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;
using static NumSharp.Tests.Casting.ToArrayContractTests;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Pins <see cref="NDArray.ToJaggedArray{T}"/> — the copy of an NDArray into a .NET jagged array (<c>T[]</c>,
    ///     <c>T[][]</c>, …, one nesting level per dimension) — for all 15 dtypes over every leaf size and memory layout its
    ///     block filler tells apart, for all 210 (source dtype, T) conversions over every row-converter route, and on its
    ///     contract: the exact nested types, a distinct array for every row, a fresh copy per call, and its refusals.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Why so wide: the jagged copy is not one loop. A 1-D result is one leaf; a 2-D one is a block whose rows are
    ///         allocated zeroed below <c>2048</c> bytes and uninitialized from it (big leaves are allocated all at once,
    ///         then filled), and filled by a block copy (adjacent elements), a broadcast fill, a vectorized reverse copy
    ///         (stride −1; AVX2 for 4- and 8-byte elements, a copy + in-place reverse otherwise), AVX register transposes in
    ///         32-row × 64-column tiles when the rows are adjacent source columns (4×4 blocks for 8-byte elements, inline
    ///         for big leaves; 8×8 for 4-byte; 2×2 for 16-byte ones from 1 024 positions; small leaves from 64 bytes) with
    ///         4- / 8- / 2-row bands and gathered rows for the remainder, or an element gather; rank ≥ 3 adds levels built
    ///         by an IL-emitted, rank-specialised builder. A conversion replaces each row's fill with a row converter: the
    ///         contiguous cast kernel (unit stride), the strided cast kernel, the iterator's per-element cast (a pair with
    ///         no strided kernel), or one converted value replicated (a fully broadcast source). The catalogs below walk
    ///         every one of those, on both sides of every size threshold, per element width.
    ///     </para>
    ///     <para>
    ///         The oracle is the ToArray contract's (<see cref="ToArrayContractTests.LogicalBytes"/>): the view's own
    ///         buffer read through its dimensions, strides and offset by a plain odometer — no NumSharp copy or kernel. A
    ///         same-dtype result is compared as RAW BYTES over random bit patterns with planted specials (signalling NaNs,
    ///         payloads, signed zeros, extremes); a conversion is compared with <c>astype</c> read through the same oracle
    ///         (decimal targets by value), over values for which every conversion is defined
    ///         (<see cref="ToArrayContractTests.ConversionPattern"/>). Every result's structure is checked level by level
    ///         (<see cref="JaggedDifference{T}"/>): the exact CLR type (<c>T[][]</c>, never a covariant <c>object[]</c>),
    ///         each level's length, and that no sub-array appears twice — a broadcast row must still get its own leaf, or
    ///         writing one row would write another.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ToJaggedArrayContractTests
    {
        // ============================================================================================== tests

        /// <summary>
        ///     Every dtype, through every leaf size and memory layout that selects a distinct fill route, copies exactly
        ///     the view's logical C-order elements, bit for bit, into a jagged array of the exact nested type with a
        ///     distinct array per row — and leaves its source untouched.
        /// </summary>
        /// <param name="dtype">The dtype under test (also the requested element type: no conversion).</param>
        /// <exception cref="AssertFailedException">
        ///     A case returned the wrong structure or bytes, shared a sub-array, wrote into its source, or threw; or the
        ///     catalog ran fewer cases than it lists.
        /// </exception>
        /// <remarks>About 136 cases per dtype (see <see cref="RunSameDtypeCatalog{T}"/>).</remarks>
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
        public void SameDtype_EveryLayout_CopiesTheLogicalElementsBitExact(NPTypeCode dtype)
        {
            var log = new CaseLog();
            Visit(dtype, new SameDtypeVisitor(log));
            AssertNoFailures(log, $"{dtype} same-dtype jagged catalog", minimumCases: 130);
        }

        /// <summary>
        ///     From every source dtype into every other dtype, over every row-converter route, ToJaggedArray converts
        ///     exactly as <c>astype</c> does, into the exact nested type — and leaves its source untouched.
        /// </summary>
        /// <param name="from">The source dtype; every other dtype is a target.</param>
        /// <exception cref="AssertFailedException">
        ///     A case differed from astype, returned the wrong structure, shared a sub-array, wrote into its source, or threw;
        ///     or the catalog ran fewer cases than it lists.
        /// </exception>
        /// <remarks>29 cases per target (see <see cref="RunConversionCatalog{T}"/>), 406 per source dtype.</remarks>
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
        public void Conversion_EveryTargetAndLayout_MatchesAstype(NPTypeCode from)
        {
            var log = new CaseLog();
            var visitor = new ConversionVisitor(from, log);
            foreach (NPTypeCode to in AllDtypes)
                if (to != from)
                    Visit(to, visitor);
            AssertNoFailures(log, $"{from} → every other dtype (jagged)", minimumCases: 14 * 29);
        }

        /// <summary>
        ///     A 0-d array has no jagged form: every dtype — a fresh scalar and a 0-d view, same dtype or converted, even a
        ///     non-dtype element type (the 0-d check comes first) — is refused with <see cref="InvalidOperationException"/>.
        /// </summary>
        /// <exception cref="AssertFailedException">A 0-d source did not throw as specified.</exception>
        [TestMethod]
        public void ZeroDimensional_ThrowsInvalidOperationException()
        {
            foreach (NPTypeCode dtype in AllDtypes)
            {
                using var line = RawPattern(dtype, 8, 0x1A66_0100);
                using var view = line["3"];
                using var owning = view.copy();
                foreach (var (name, source) in new[] { ("0-d view", view), ("owning 0-d", owning) })
                {
                    Action same = () => Jagged(source, dtype);
                    same.Should().Throw<InvalidOperationException>($"{dtype} {name}").WithMessage("*0-d array has no jagged form*");
                    NPTypeCode other = dtype == NPTypeCode.Double ? NPTypeCode.Int32 : NPTypeCode.Double;
                    Action converted = () => Jagged(source, other);
                    converted.Should().Throw<InvalidOperationException>($"{dtype} {name} → {other}").WithMessage("*0-d array has no jagged form*");
                    Action foreign = () => source.ToJaggedArray<Guid>();
                    foreign.Should().Throw<InvalidOperationException>($"{dtype} {name} → Guid");
                }
            }
        }

        /// <summary>
        ///     An empty array keeps every level up to its first empty axis: <c>(3, 0)</c> is a <c>T[3][]</c> of three
        ///     distinct empty leaves, <c>(0, 3)</c> an empty <c>T[0][]</c>, <c>(2, 0, 4)</c> a <c>T[2][][]</c> of two empty
        ///     <c>T[0][]</c> — same dtype and converted, owning and views.
        /// </summary>
        /// <exception cref="AssertFailedException">A level had the wrong type or length, or leaves were shared.</exception>
        [TestMethod]
        public void EmptyShapes_KeepEveryLevelUpToTheEmptyAxis()
        {
            var log = new CaseLog();
            foreach (NPTypeCode dtype in AllDtypes)
            {
                using var line = RawPattern(dtype, 60, 0x1A66_0200);
                var sources = new (string Name, Func<NDArray> Make)[]
                {
                    ("(0,)", () => np.empty(new Shape(0), dtype)),
                    ("(3, 0)", () => np.empty(new Shape(3, 0), dtype)),
                    ("(0, 3)", () => np.empty(new Shape(0, 3), dtype)),
                    ("(2, 0, 4)", () => np.empty(new Shape(2, 0, 4), dtype)),
                    ("(2, 3, 0)", () => np.empty(new Shape(2, 3, 0), dtype)),
                    ("(0, 2, 3)", () => np.empty(new Shape(0, 2, 3), dtype)),
                    ("view (5, 6)[:, 2:2]", () => Grid(line, 5, 6)[":, 2:2"]),
                    ("view (5, 6)[::-1, 2:2].T", () => Grid(line, 5, 6)["::-1, 2:2"].T),
                    ("broadcast (1, 1) → (4, 0)", () => np.broadcast_to(Grid(line, 1, 1), new Shape(4, 0))),
                };
                foreach (var (name, make) in sources)
                {
                    log.Cases++;
                    try
                    {
                        using NDArray source = make();
                        long[] dims = source.shape;
                        foreach (NPTypeCode target in new[] { dtype, dtype == NPTypeCode.Double ? NPTypeCode.Int16 : NPTypeCode.Double })
                        {
                            string problem = EmptyStructureDifference(Jagged(source, target), dims, ClrType(target));
                            if (problem != null)
                                log.Fail($"{dtype} {name} as {target}: {problem}");
                        }
                    }
                    catch (Exception e)
                    {
                        log.Fail($"{dtype} {name}: threw {e.GetType().Name}: {e.Message}");
                    }
                }
            }
            AssertNoFailures(log, "empty jagged structures", minimumCases: 15 * 9);
        }

        /// <summary>
        ///     The result is a private copy: every row is its own array (a broadcast source too), writing one row never
        ///     reaches another row or the source, writing the source never reaches an earlier result, and a second call
        ///     shares no array with the first — for a plain block, a broadcast block, a transposed block, a rank-3 result and
        ///     a conversion.
        /// </summary>
        /// <exception cref="AssertFailedException">Two rows or two calls shared an array, or a write crossed.</exception>
        [TestMethod]
        public void Result_IsAFreshCopy_WithADistinctArrayForEveryRow()
        {
            using var a = np.arange(12.0).reshape(3, 4);
            var first = (double[][])a.ToJaggedArray<double>();
            var second = (double[][])a.ToJaggedArray<double>();
            ReferenceEquals(first, second).Should().BeFalse("every call allocates its own result");
            first.Intersect(second, ReferenceEqualityComparer.Instance).Should().BeEmpty("no row is shared between two calls");
            first[1][2] = -1.0;
            a.GetDouble(1, 2).Should().Be(6.0, "writing the result must not reach the source");
            a.SetDouble(100.0, 0, 0);
            second[0][0].Should().Be(0.0, "writing the source must not reach an earlier result");

            // A broadcast block: every row reads the same source row, and still gets its own leaf.
            using var row = np.arange(5.0);
            using var broadcast = np.broadcast_to(row, new Shape(4, 5));
            var rows = (double[][])broadcast.ToJaggedArray<double>();
            rows.Distinct(ReferenceEqualityComparer.Instance).Count().Should().Be(4, "a broadcast row gets a leaf of its own");
            rows[0][1] = -2.0;
            rows[1][1].Should().Be(1.0, "writing one broadcast row must not reach another");
            row.GetDouble(1).Should().Be(1.0, "writing a broadcast row must not reach the source");

            // A big broadcast block (leaves allocated uninitialized, all at once) and a converted one.
            using var longRow = np.arange(3000.0);
            using var longBroadcast = np.broadcast_to(longRow, new Shape(3, 3000));
            var longRows = (double[][])longBroadcast.ToJaggedArray<double>();
            longRows.Distinct(ReferenceEqualityComparer.Instance).Count().Should().Be(3);
            var converted = (float[][])longBroadcast.ToJaggedArray<float>();
            converted.Distinct(ReferenceEqualityComparer.Instance).Count().Should().Be(3);
            converted[2][2999] = -3f;
            converted[1][2999].Should().Be(2999f, "a converted broadcast row gets a leaf of its own");

            // A transposed block and a rank-3 result.
            using var t = a.T;
            var tr = (double[][])t.ToJaggedArray<double>();
            tr[0][1] = -4.0;
            a.GetDouble(1, 0).Should().Be(4.0, "a transposed view's result is a private copy");
            using var cube = np.arange(24).reshape(2, 3, 4);
            var c1 = (long[][][])cube.ToJaggedArray<long>();
            var c2 = (long[][][])cube.ToJaggedArray<long>();
            c1.Concat(c2).Distinct(ReferenceEqualityComparer.Instance).Count().Should().Be(4, "each call builds its own level-1 arrays");
            c1.SelectMany(x => x).Concat(c2.SelectMany(x => x)).Distinct(ReferenceEqualityComparer.Instance).Count()
                .Should().Be(12, "each call builds its own leaves");
        }

        /// <summary>
        ///     An element type that is not a NumSharp dtype has no conversion: every non-0-d source — 1-D, 2-D, rank-3 and
        ///     empty — refuses it with astype's own <see cref="NotSupportedException"/> (the call asks astype, so the
        ///     exception is exactly astype's), and the bits of a foreign type are never reinterpreted.
        /// </summary>
        /// <exception cref="AssertFailedException">A non-dtype element type did not throw as specified.</exception>
        [TestMethod]
        public void NonDtypeElementType_ThrowsAstypesNotSupportedException()
        {
            foreach (NPTypeCode dtype in AllDtypes)
            {
                using var line = RawPattern(dtype, 24, 0x1A66_0300);
                using var grid = line.reshape(4, 6);
                using var cube = line.reshape(2, 3, 4);
                using var transposed = grid.T;
                using var empty = line["7:7"];
                foreach (var (name, source) in new[] { ("1-D", line), ("2-D", grid), ("transposed", transposed), ("3-D", cube), ("empty", empty) })
                {
                    Action guid = () => source.ToJaggedArray<Guid>();
                    guid.Should().Throw<NotSupportedException>($"{dtype} {name}").WithMessage("*System.Guid*");
                    Action dateTime = () => source.ToJaggedArray<DateTime>();
                    dateTime.Should().Throw<NotSupportedException>($"{dtype} {name}").WithMessage("*System.DateTime*");
                    Action pointer = () => source.ToJaggedArray<nint>();
                    pointer.Should().Throw<NotSupportedException>($"{dtype} {name}");
                }
            }
        }

        /// <summary>
        ///     A dimension above <see cref="int.MaxValue"/> cannot be a .NET array length: the call throws
        ///     <see cref="InvalidOperationException"/> naming the dimension, before allocating anything — at the 1-D leaf,
        ///     the 2-D block, and through the emitted rank ≥ 3 builder, same dtype and converted.
        /// </summary>
        /// <param name="rank">Rank of the (broadcast, allocation-free) source; the oversized dimension is the last.</param>
        /// <exception cref="AssertFailedException">The wrong exception type or text surfaced.</exception>
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        public void DimensionAboveIntMaxValue_ThrowsInvalidOperationException(int rank)
        {
            var dims = new long[rank];
            Array.Fill(dims, 1L);
            dims[rank - 1] = (long)int.MaxValue + 1;
            using var one = NDArray.Scalar((byte)7);
            using var huge = np.broadcast_to(one, new Shape(dims));

            Action same = () => huge.ToJaggedArray<byte>();
            same.Should().Throw<InvalidOperationException>().WithMessage($"*Dimension {rank - 1} has length 2147483648*exceeds int.MaxValue*");
            Action converted = () => huge.ToJaggedArray<short>();
            converted.Should().Throw<InvalidOperationException>().WithMessage($"*Dimension {rank - 1} has length 2147483648*exceeds int.MaxValue*");
        }

        // ============================================================================================== catalogs

        /// <summary>
        ///     The same-dtype catalog for one element type: every leaf size and memory layout that selects a distinct fill
        ///     route of the jagged copy, on both sides of every size threshold, per element width.
        /// </summary>
        /// <typeparam name="T">The element type (the dtype's: no conversion).</typeparam>
        /// <param name="log">Collects the cases and their failures.</param>
        /// <remarks>
        ///     The thresholds are functions of the element width, so they are placed per width: a leaf is allocated
        ///     uninitialized from 2 048 bytes (<c>big</c> elements), small transposed leaves take the tiled transposes from
        ///     64 bytes (<c>tile</c>), 16-byte leaves only from 1 024 positions; the tiles are 32 rows × 64 positions, in 4-
        ///     (8-byte), 8- (4-byte) or 2-row (16-byte) sub-bands, so the big transposes use 70 rows (two tiles, a band, and
        ///     gathered rows) for the widths that tile, 9 for the others (which gather, and whose leaves are longest).
        /// </remarks>
        private static void RunSameDtypeCatalog<T>(CaseLog log) where T : unmanaged
        {
            NPTypeCode dtype = InfoOf<T>.NPTypeCode;
            int size = Unsafe.SizeOf<T>();
            string tag = $"{dtype} jagged";
            long big = BigLeaf(size);
            long tile = 64 / size;
            long wide = size >= 4 ? 70 : 9;
            long deep = size >= 4 ? 37 : 5;

            using var line = RawPattern(dtype, 100_000, 0x1A66_0001);
            var c = new JaggedCaseRunner<T>(line, log, tag);

            // ---- 1-D: one leaf — block copy, fill, reverse copy, gather; both sides of the uninitialized-leaf size ------
            foreach (long n in new long[] { 1, 2, 3, 7, tile + 1, big - 1, big, big + 1, 5000 })
                c.Check($"1-D [0:{n}]", () => line[$"0:{n}"]);
            // Reversed runs around every vector width (the AVX2 reverse's tails), gathers of other strides.
            foreach (long step in new long[] { 2, 3, -1, -2, -3 })
                foreach (long n in new long[] { 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, big + 3 })
                    c.Check($"1-D step {step} n={n}", () => Stepped(line, step, n, lead: 2));
            // A non-zero Shape offset (an np.split child keeps the base address; a slice would re-seat it).
            foreach (long n in new long[] { 1, 33, big + 1 })
                c.Check($"1-D split child [5:{5 + n}]", () => SplitChild(line, 5, n));
            c.Check("1-D scalar broadcast (9,)", () => np.broadcast_to(line["3"], new Shape(9)));
            c.Check($"1-D one-element broadcast ({big + 1},)", () => np.broadcast_to(line["3:4"], new Shape(big + 1)));
            c.Check("1-D empty [5:5]", () => line["5:5"]);
            c.Check("1-D empty [::-1][0:0]", () => line["::-1"]["0:0"]);

            // ---- 2-D: one block ---------------------------------------------------------------------------------------
            c.Check("(3, 5) C", () => Grid(line, 3, 5));
            c.Check($"(7, {tile + 1}) C", () => Grid(line, 7, tile + 1));
            c.Check($"(3, {big + 1}) C, big leaves", () => Grid(line, 3, big + 1));
            // Transposed blocks (rows = adjacent source columns): under the small-transpose size (gather), small tiled
            // leaves, both sides of the big-leaf size (the inline 8-byte tile loop / the tile helper), and the 16-byte
            // tiles' 1 024-position threshold; each with a ragged tail of rows and of positions.
            foreach (var (r, cols) in new (long, long)[] { (5, 3), (tile - 1, 9), (tile, 37), (tile + 3, 70), (big - 1, 9), (big, wide), (big + 3, 33), (1023, 5), (1024, 35), (1025, 67) })
                c.Check($"({r}, {cols}).T", () => Grid(line, r, cols).T);
            // Transposed with a NEGATIVE position stride, and F-ordered copies (the same transposes from a fresh owner).
            foreach (var (r, cols) in new (long, long)[] { (tile, 37), (big, wide), (1024, 35) })
            {
                c.Check($"({r}, {cols})[::-1, :].T", () => Grid(line, r, cols)["::-1, :"].T);
                c.Check($"F-contiguous ({cols}, {r})", () => np.asfortranarray(Grid(line, cols, r)));
            }
            // Reversed rows (the reverse copy, per row), small with vector tails and big.
            c.Check("(6, 13)[:, ::-1]", () => Grid(line, 6, 13)[":, ::-1"]);
            c.Check("(3, 17)[:, ::-1]", () => Grid(line, 3, 17)[":, ::-1"]);
            c.Check($"(4, {big + 5})[:, ::-1]", () => Grid(line, 4, big + 5)[":, ::-1"]);
            // Gathers: stepped rows and columns, negative row steps, offset windows.
            c.Check("(9, 11)[::2, ::3]", () => Grid(line, 9, 11)["::2, ::3"]);
            c.Check("(9, 11)[::-2, ::2]", () => Grid(line, 9, 11)["::-2, ::2"]);
            c.Check("(9, 11)[1:-1, 2:-2]", () => Grid(line, 9, 11)["1:-1, 2:-2"]);
            c.Check("(13, 17)[:, ::2]", () => Grid(line, 13, 17)[":, ::2"]);
            // Broadcast rows (row step 0: every leaf copies the same row), small and big.
            c.Check("row broadcast (5, 7)", () => np.broadcast_to(line["0:7"], new Shape(5, 7)));
            c.Check($"row broadcast (3, {big + 1})", () => np.broadcast_to(line[$"0:{big + 1}"], new Shape(3, big + 1)));
            // Broadcast columns (rows are adjacent elements, each repeated along the row: the transposes with a zero
            // position stride), small, big, and at the 16-byte tile length.
            c.Check("column broadcast (40, 30)", () => np.broadcast_to(np.expand_dims(line["0:40"], 1), new Shape(40, 30)));
            c.Check($"column broadcast (40, {big + 1})", () => np.broadcast_to(np.expand_dims(line["0:40"], 1), new Shape(40, big + 1)));
            c.Check("column broadcast (35, 1030)", () => np.broadcast_to(np.expand_dims(line["0:35"], 1), new Shape(35, 1030)));
            // Single-value broadcasts (every stride 0).
            c.Check("scalar broadcast (5, 9)", () => np.broadcast_to(line["3"], new Shape(5, 9)));
            c.Check($"scalar broadcast (3, {big + 1})", () => np.broadcast_to(line["3"], new Shape(3, big + 1)));
            // Blocks at a non-zero Shape offset.
            c.Check("split child rows [4:6] of (6, 5)", () => np.split(Grid(line, 6, 5), new[] { 4 })[1]);
            c.Check("unstack child [1] of (3, 4, 5)", () => np.unstack(Cube(line, 3, 4, 5))[1]);
            // Empty blocks.
            c.Check("empty rows (6, 7)[5:5, :]", () => Grid(line, 6, 7)["5:5, :"]);
            c.Check("empty columns (6, 7)[:, 3:3]", () => Grid(line, 6, 7)[":, 3:3"]);
            c.Check("empty owning (0, 5)", () => np.empty(new Shape(0, 5), dtype));
            c.Check("empty owning (5, 0)", () => np.empty(new Shape(5, 0), dtype));

            // ---- rank ≥ 3: the emitted builder around every block kind -------------------------------------------------
            c.Check("(2, 3, 4) C", () => Cube(line, 2, 3, 4));
            foreach (int[] p in new[] { new[] { 0, 1, 2 }, new[] { 0, 2, 1 }, new[] { 1, 0, 2 }, new[] { 1, 2, 0 }, new[] { 2, 0, 1 }, new[] { 2, 1, 0 } })
                c.Check($"(3, 4, 5) transpose({string.Join(",", p)})", () => np.transpose(Cube(line, 3, 4, 5), p));
            c.Check("(3, 4, 5)[::-1, :, ::2]", () => Cube(line, 3, 4, 5)["::-1, :, ::2"]);
            c.Check("(3, 4, 5) flipped on every axis", () => np.flip(Cube(line, 3, 4, 5)));
            c.Check("middle-axis broadcast (4, 1, 6) → (4, 5, 6)", () => np.broadcast_to(Cube(line, 4, 1, 6), new Shape(4, 5, 6)));
            c.Check("(4, 1, 6) C", () => Cube(line, 4, 1, 6));
            c.Check($"(2, 3, {big + 1}) C, big leaves", () => Cube(line, 2, 3, big + 1));
            c.Check($"(2, {big + 1}, {deep}).transpose(0, 2, 1), big transposed leaves", () => np.transpose(Cube(line, 2, big + 1, deep), new[] { 0, 2, 1 }));
            c.Check("F-contiguous (3, 4, 5)", () => np.asfortranarray(Cube(line, 3, 4, 5)));
            c.Check("4-D transpose(3, 2, 1, 0)", () => np.transpose(line["0:120"].reshape(2, 3, 4, 5), new[] { 3, 2, 1, 0 }));
            c.Check("4-D transpose(1, 3, 0, 2)", () => np.transpose(line["0:120"].reshape(2, 3, 4, 5), new[] { 1, 3, 0, 2 }));
            c.Check("5-D .T", () => line["0:144"].reshape(2, 3, 2, 3, 4).T);
            c.Check("7-D C", () => line["0:72"].reshape(2, 1, 3, 1, 2, 1, 6));
            c.Check("7-D flipped", () => np.flip(line["0:72"].reshape(2, 1, 3, 1, 2, 1, 6)));
            // Past a .NET multi-dimensional array's 32 ranks. Rank 33 is also a rank the cross-API edge tests use: the
            // first call per (element type, rank) emits and compiles a builder — on .NET 8 about 22 ms at this rank, and
            // super-linear in it (≈ 50 ms at 40, ≈ 200 ms at 64) — so sharing the rank shares that one-time cost.
            c.Check("33-D C", () => line["0:24"].reshape(HighRankDims(33, 24)));
            c.Check("33-D flipped", () => np.flip(line["0:24"].reshape(HighRankDims(33, 24))));
            c.Check("empty owning (2, 0, 3)", () => np.empty(new Shape(2, 0, 3), dtype));
            c.Check("empty owning (2, 3, 0)", () => np.empty(new Shape(2, 3, 0), dtype));
            c.Check("empty owning (0, 2, 3)", () => np.empty(new Shape(0, 2, 3), dtype));
            c.Check("empty (3, 4, 5)[:, 2:2, :]", () => Cube(line, 3, 4, 5)[":, 2:2, :"]);
        }

        /// <summary>
        ///     The conversion catalog for one (source dtype, target) pair: every row-converter route, each compared with
        ///     astype.
        /// </summary>
        /// <typeparam name="T">The target element type (a dtype other than <paramref name="from"/>).</typeparam>
        /// <param name="from">The source dtype.</param>
        /// <param name="log">Collects the cases and their failures.</param>
        /// <remarks>
        ///     The routes: unit-stride rows (the contiguous cast kernel), strided / reversed / broadcast-column rows (the
        ///     strided cast kernel, or the iterator's per-element cast for a pair without one — every bool source), a fully
        ///     broadcast source (one converted value, replicated), big leaves (allocated uninitialized: the kernel must
        ///     write every element), non-zero offsets, empties, and the emitted converting builder for rank 3.
        /// </remarks>
        private static void RunConversionCatalog<T>(NPTypeCode from, CaseLog log) where T : unmanaged
        {
            NPTypeCode to = InfoOf<T>.NPTypeCode;
            long big = BigLeaf(Unsafe.SizeOf<T>());
            ulong seed = 0x1A66_0000UL + (ulong)from * 31 + (ulong)to;
            using var line = ConversionPattern(from, to, 30_000, seed);
            var c = new JaggedCaseRunner<T>(line, log, $"{from} → {to} jagged");

            foreach (long n in new long[] { 1, 2, 9, 33, 1000, big + 1 })
                c.Check($"1-D [0:{n}]", () => line[$"0:{n}"]);
            c.Check("1-D step 2 n=33", () => Stepped(line, 2, 33, lead: 1));
            c.Check("1-D step -1 n=100", () => Stepped(line, -1, 100, lead: 1));
            c.Check($"1-D step 3 n={big + 1}", () => Stepped(line, 3, big + 1, lead: 1));
            c.Check("1-D split child [3:36]", () => SplitChild(line, 3, 33));
            c.Check("1-D scalar broadcast (9,)", () => np.broadcast_to(line["3"], new Shape(9)));
            c.Check($"1-D one-element broadcast ({big + 1},)", () => np.broadcast_to(line["3:4"], new Shape(big + 1)));

            c.Check("(5, 7) C", () => Grid(line, 5, 7));
            c.Check($"(3, {big + 1}) C, big leaves", () => Grid(line, 3, big + 1));
            c.Check("(7, 9).T", () => Grid(line, 7, 9).T);
            c.Check("(40, 30).T", () => Grid(line, 40, 30).T);
            c.Check("(6, 13)[:, ::-1]", () => Grid(line, 6, 13)[":, ::-1"]);
            c.Check("(9, 11)[::-2, ::3]", () => Grid(line, 9, 11)["::-2, ::3"]);
            c.Check("row broadcast (5, 7)", () => np.broadcast_to(line["0:7"], new Shape(5, 7)));
            c.Check("column broadcast (40, 30)", () => np.broadcast_to(np.expand_dims(line["0:40"], 1), new Shape(40, 30)));
            c.Check("scalar broadcast (5, 9)", () => np.broadcast_to(line["3"], new Shape(5, 9)));
            c.Check("split child rows [4:6] of (6, 7)", () => np.split(Grid(line, 6, 7), new[] { 4 })[1]);
            c.Check("empty rows (6, 7)[5:5, :]", () => Grid(line, 6, 7)["5:5, :"]);
            c.Check("empty columns (6, 7)[:, 3:3]", () => Grid(line, 6, 7)[":, 3:3"]);

            c.Check("(2, 3, 4) C", () => Cube(line, 2, 3, 4));
            c.Check("(3, 4, 5) transpose(2, 0, 1)", () => np.transpose(Cube(line, 3, 4, 5), new[] { 2, 0, 1 }));
            c.Check("(3, 4, 5)[::-1, :, ::2]", () => Cube(line, 3, 4, 5)["::-1, :, ::2"]);
            c.Check("middle-axis broadcast (4, 1, 6) → (4, 5, 6)", () => np.broadcast_to(Cube(line, 4, 1, 6), new Shape(4, 5, 6)));
            c.Check($"(2, 3, {big + 1}) C, big leaves", () => Cube(line, 2, 3, big + 1));
        }

        // ============================================================================================== helpers

        /// <summary>The first leaf length, in elements of <paramref name="size"/> bytes, that ToJaggedArray allocates uninitialized (2 048 bytes).</summary>
        /// <param name="size">Element size in bytes (1, 2, 4, 8 or 16).</param>
        /// <returns>2 048 / <paramref name="size"/>.</returns>
        private static long BigLeaf(int size) => 2048 / size;

        /// <summary>
        ///     Dimensions of a <paramref name="rank"/>-D shape holding exactly <paramref name="count"/> = 24 elements: the
        ///     factors 2, 3 and 4 spread over the axes, every other axis of extent 1 (an extent-1 axis adds a level but no
        ///     elements).
        /// </summary>
        /// <param name="rank">The rank (at least 3).</param>
        /// <param name="count">The element count (24).</param>
        /// <returns>The dimensions.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is not 24 or <paramref name="rank"/> is below 3.</exception>
        internal static long[] HighRankDims(int rank, long count)
        {
            if (count != 24 || rank < 3)
                throw new ArgumentOutOfRangeException(nameof(count), "HighRankDims spreads exactly 2 · 3 · 4 over at least three axes.");
            var dims = new long[rank];
            Array.Fill(dims, 1L);
            dims[0] = 2;
            dims[rank / 2] = 3;
            dims[rank - 1] = 4;
            return dims;
        }

        /// <summary>The CLR element type of a NumSharp dtype.</summary>
        /// <param name="dtype">One of the 15 dtypes.</param>
        /// <returns>Its CLR type.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="dtype"/> is not one of the 15 dtypes.</exception>
        internal static Type ClrType(NPTypeCode dtype) => dtype switch
        {
            NPTypeCode.Boolean => typeof(bool),
            NPTypeCode.Byte => typeof(byte),
            NPTypeCode.SByte => typeof(sbyte),
            NPTypeCode.Int16 => typeof(short),
            NPTypeCode.UInt16 => typeof(ushort),
            NPTypeCode.Int32 => typeof(int),
            NPTypeCode.UInt32 => typeof(uint),
            NPTypeCode.Int64 => typeof(long),
            NPTypeCode.UInt64 => typeof(ulong),
            NPTypeCode.Char => typeof(char),
            NPTypeCode.Half => typeof(Half),
            NPTypeCode.Single => typeof(float),
            NPTypeCode.Double => typeof(double),
            NPTypeCode.Decimal => typeof(decimal),
            NPTypeCode.Complex => typeof(Complex),
            _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Not a NumSharp dtype."),
        };

        /// <summary>Calls <see cref="NDArray.ToJaggedArray{T}"/> with the CLR type of <paramref name="elementType"/>.</summary>
        /// <param name="source">The array to copy.</param>
        /// <param name="elementType">The requested element type (a conversion when it differs from the dtype).</param>
        /// <returns>The jagged result.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="elementType"/> is not one of the 15 dtypes.</exception>
        /// <exception cref="InvalidOperationException">Whatever ToJaggedArray throws (a 0-d source, an oversized dimension).</exception>
        internal static Array Jagged(NDArray source, NPTypeCode elementType) => elementType switch
        {
            NPTypeCode.Boolean => source.ToJaggedArray<bool>(),
            NPTypeCode.Byte => source.ToJaggedArray<byte>(),
            NPTypeCode.SByte => source.ToJaggedArray<sbyte>(),
            NPTypeCode.Int16 => source.ToJaggedArray<short>(),
            NPTypeCode.UInt16 => source.ToJaggedArray<ushort>(),
            NPTypeCode.Int32 => source.ToJaggedArray<int>(),
            NPTypeCode.UInt32 => source.ToJaggedArray<uint>(),
            NPTypeCode.Int64 => source.ToJaggedArray<long>(),
            NPTypeCode.UInt64 => source.ToJaggedArray<ulong>(),
            NPTypeCode.Char => source.ToJaggedArray<char>(),
            NPTypeCode.Half => source.ToJaggedArray<Half>(),
            NPTypeCode.Single => source.ToJaggedArray<float>(),
            NPTypeCode.Double => source.ToJaggedArray<double>(),
            NPTypeCode.Decimal => source.ToJaggedArray<decimal>(),
            NPTypeCode.Complex => source.ToJaggedArray<Complex>(),
            _ => throw new ArgumentOutOfRangeException(nameof(elementType), elementType, "Not a NumSharp dtype."),
        };

        /// <summary>The exact CLR type of level <paramref name="depth"/> of a rank-<paramref name="rank"/> jagged array of <paramref name="leaf"/>.</summary>
        /// <param name="leaf">The element type.</param>
        /// <param name="rank">The source rank.</param>
        /// <param name="depth">The level (0 = the outermost).</param>
        /// <returns><paramref name="leaf"/> nested <c>rank − depth</c> times (<c>T[]</c> at the last level).</returns>
        internal static Type LevelType(Type leaf, int rank, int depth)
        {
            Type t = leaf;
            for (int d = depth; d < rank; d++)
                t = t.MakeArrayType();
            return t;
        }

        /// <summary>
        ///     Checks a jagged result against the expected elements: every level's exact CLR type and length, that no array
        ///     appears twice (recorded in <paramref name="arrays"/>), and the leaves' elements in C order.
        /// </summary>
        /// <typeparam name="T">The leaf element type.</typeparam>
        /// <param name="result">The jagged result (level 0).</param>
        /// <param name="dims">The source's dimensions (rank ≥ 1).</param>
        /// <param name="want">The expected elements as raw bytes, in C order.</param>
        /// <param name="byValue">Compare decimals by value (a conversion's result) instead of by bits.</param>
        /// <param name="arrays">Receives every array of the result (by reference); an array met twice is a failure.</param>
        /// <returns>Null when the result matches; else a description of the first problem.</returns>
        internal static string JaggedDifference<T>(Array result, long[] dims, byte[] want, bool byValue, HashSet<object> arrays) where T : unmanaged
        {
            int size = Unsafe.SizeOf<T>();
            var got = new byte[want.Length];
            long filled = 0;
            string problem = Walk(result, 0);
            if (problem != null)
                return problem;
            if (filled != want.Length)
                return $"the leaves hold {filled / size} elements but {want.Length / size} were expected";
            if (!byValue && got.AsSpan().SequenceEqual(want))
                return null;

            ReadOnlySpan<T> gotElements = MemoryMarshal.Cast<byte, T>(got.AsSpan());
            ReadOnlySpan<T> wantElements = MemoryMarshal.Cast<byte, T>(want.AsSpan());
            for (int i = 0; i < gotElements.Length; i++)
            {
                bool same = byValue && typeof(T) == typeof(decimal)
                    ? (decimal)(object)gotElements[i] == (decimal)(object)wantElements[i]
                    : got.AsSpan(i * size, size).SequenceEqual(want.AsSpan(i * size, size));
                if (!same)
                    return $"element {i} (C order) is {gotElements[i]} [0x{Convert.ToHexString(got.AsSpan(i * size, size))}] " +
                           $"but {wantElements[i]} [0x{Convert.ToHexString(want.AsSpan(i * size, size))}] was expected";
            }
            return null;

            // Depth-first in C order: the leaves' bytes land in `got` in exactly the order the oracle lists them.
            string Walk(Array level, int depth)
            {
                Type expected = LevelType(typeof(T), dims.Length, depth);
                if (level is null)
                    return $"level {depth}: a null sub-array";
                if (level.GetType() != expected)
                    return $"level {depth}: a {level.GetType()} where a {expected} was expected";
                if (level.Length != dims[depth])
                    return $"level {depth}: length {level.Length} but the dimension is {dims[depth]}";
                if (!arrays.Add(level))
                    return $"level {depth}: an array appears twice in the result (a shared sub-array)";
                if (depth == dims.Length - 1)
                {
                    ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(((T[])level).AsSpan());
                    if (filled + bytes.Length > got.Length)
                        return $"the leaves hold more than the {want.Length / size} expected elements";
                    bytes.CopyTo(got.AsSpan((int)filled));
                    filled += bytes.Length;
                    return null;
                }
                foreach (object child in level)
                {
                    string inner = Walk((Array)child, depth + 1);
                    if (inner != null)
                        return inner;
                }
                return null;
            }
        }

        /// <summary>
        ///     Checks the structure of an EMPTY jagged result: every level down to the first zero-length one has the exact
        ///     type and length, and no array appears twice (each row of a <c>(3, 0)</c> result is its own empty leaf).
        /// </summary>
        /// <param name="result">The jagged result.</param>
        /// <param name="dims">The source's dimensions (at least one of them 0).</param>
        /// <param name="leaf">The requested element type.</param>
        /// <returns>Null when the structure matches; else the first problem.</returns>
        private static string EmptyStructureDifference(Array result, long[] dims, Type leaf)
        {
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
            return Walk(result, 0);

            string Walk(Array level, int depth)
            {
                Type expected = LevelType(leaf, dims.Length, depth);
                if (level is null)
                    return $"level {depth}: a null sub-array";
                if (level.GetType() != expected)
                    return $"level {depth}: a {level.GetType()} where a {expected} was expected";
                if (level.Length != dims[depth])
                    return $"level {depth}: length {level.Length} but the dimension is {dims[depth]}";
                if (!seen.Add(level))
                    return $"level {depth}: an array appears twice in the result";
                if (depth == dims.Length - 1)
                    return null;
                foreach (object child in level)
                {
                    string inner = Walk((Array)child, depth + 1);
                    if (inner != null)
                        return inner;
                }
                return null;
            }
        }

        /// <summary>
        ///     Runs cases over views of ONE C-contiguous source array: each <see cref="Check"/> builds a view, calls
        ///     <see cref="NDArray.ToJaggedArray{T}"/> twice, checks both results against the oracle and each other, checks
        ///     the source's buffer is untouched, and disposes the view — recording failures in the <see cref="CaseLog"/>.
        /// </summary>
        /// <typeparam name="T">The requested element type: the source's dtype (an exact copy) or a conversion target.</typeparam>
        private sealed class JaggedCaseRunner<T> where T : unmanaged
        {
            /// <summary>The array every view reads.</summary>
            private readonly NDArray _source;

            /// <summary>The source's bytes when the runner was made: what they must still be after every case.</summary>
            private readonly byte[] _snapshot;

            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Prefix of every failure message (dtype, pair).</summary>
            private readonly string _prefix;

            /// <summary>Creates a runner over <paramref name="source"/>, snapshotting its bytes.</summary>
            /// <param name="source">A C-contiguous array (its bytes are read whole).</param>
            /// <param name="log">The shared log.</param>
            /// <param name="prefix">Prefix of every failure message.</param>
            public JaggedCaseRunner(NDArray source, CaseLog log, string prefix)
            {
                _source = source;
                _log = log;
                _prefix = prefix;
                _snapshot = source.Unsafe.ReadOnlyBytes().ToArray();
            }

            /// <summary>
            ///     Checks one case: both results' structure and elements against the oracle (the view's own logical bytes,
            ///     or those of its astype result for a conversion), that the two calls share no array, and the source's
            ///     bytes afterwards.
            /// </summary>
            /// <param name="name">The case's description.</param>
            /// <param name="make">Builds the view (disposed here, with any astype result).</param>
            /// <remarks>A source found modified is reported and restored from the snapshot, so later cases start clean.</remarks>
            public void Check(string name, Func<NDArray> make)
            {
                _log.Cases++;
                string what = $"{_prefix} {name}";
                NDArray view = null, converted = null;
                try
                {
                    view = make();
                    what += $" shape ({string.Join(", ", view.shape)})";
                    bool conversion = view.typecode != InfoOf<T>.NPTypeCode;
                    if (conversion)
                        converted = view.astype(InfoOf<T>.NPTypeCode);
                    byte[] want = LogicalBytes(converted ?? view);
                    bool byValue = conversion && typeof(T) == typeof(decimal);
                    var firstArrays = new HashSet<object>(ReferenceEqualityComparer.Instance);
                    var secondArrays = new HashSet<object>(ReferenceEqualityComparer.Instance);
                    Array first = view.ToJaggedArray<T>();
                    Array second = view.ToJaggedArray<T>();
                    string problem = JaggedDifference<T>(first, view.shape, want, byValue, firstArrays)
                                     ?? JaggedDifference<T>(second, view.shape, want, byValue, secondArrays);
                    if (problem == null && firstArrays.Overlaps(secondArrays))
                        problem = "two calls share an array";
                    if (problem != null)
                        _log.Fail($"{what}: {problem}");
                }
                catch (Exception e)
                {
                    _log.Fail($"{what}: threw {e.GetType().Name}: {e.Message}");
                }
                finally
                {
                    converted?.Dispose();
                    view?.Dispose();
                }

                if (!_source.Unsafe.ReadOnlyBytes().SequenceEqual(_snapshot))
                {
                    _log.Fail($"{what}: ToJaggedArray wrote into its source's buffer");
                    _snapshot.CopyTo(_source.Unsafe.Bytes());
                }
            }
        }

        /// <summary>Runs <see cref="RunSameDtypeCatalog{T}"/> for the visited element type.</summary>
        private sealed class SameDtypeVisitor : IElementTypeVisitor
        {
            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Creates the visitor.</summary>
            /// <param name="log">The log the catalog records into.</param>
            public SameDtypeVisitor(CaseLog log) => _log = log;

            /// <inheritdoc/>
            public void Visit<T>() where T : unmanaged => RunSameDtypeCatalog<T>(_log);
        }

        /// <summary>Runs <see cref="RunConversionCatalog{T}"/> from one source dtype into the visited target.</summary>
        private sealed class ConversionVisitor : IElementTypeVisitor
        {
            /// <summary>The source dtype.</summary>
            private readonly NPTypeCode _from;

            /// <summary>The shared log.</summary>
            private readonly CaseLog _log;

            /// <summary>Creates the visitor.</summary>
            /// <param name="from">The source dtype.</param>
            /// <param name="log">The log the catalog records into.</param>
            public ConversionVisitor(NPTypeCode from, CaseLog log)
            {
                _from = from;
                _log = log;
            }

            /// <inheritdoc/>
            public void Visit<T>() where T : unmanaged => RunConversionCatalog<T>(_from, _log);
        }
    }
}
