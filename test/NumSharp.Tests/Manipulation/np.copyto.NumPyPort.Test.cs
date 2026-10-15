using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     <c>np.copyto</c> as NumPy's <c>array_copyto</c> → <c>PyArray_AssignArray</c> / <c>PyArray_AssignRawScalar</c>
    ///     (<c>numpy/_core/src/multiarray/array_assign_array.c</c>, <c>array_assign_scalar.c</c>), pinned with NumPy 2.4.2's
    ///     output for the same calls (plan <c>docs/plans/api-parameter-parity.md</c>, item X16).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Before the port, a masked copy whose source overlapped the destination read values it had already written (silent
    ///     wrong values), a weak C# scalar wrapped where NumPy raises, the checks ran in another order with other texts, a
    ///     source with leading unit axes beyond the destination's rank was refused, and a masked copy ran one element at a
    ///     time through a per-element branch. These tests pin each of those, plus the routes the new kernel takes (every
    ///     element width, strided and F destinations, cross-dtype staging on the stack and in a pooled buffer).
    ///     </para>
    ///     <para>
    ///     The oracle tier <c>copyto_full.jsonl</c> (13,462 cases, zero excuses) covers the same ground across every layout
    ///     and dtype pair.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_copyto_numpy_port_tests
    {
        private static NDArray Mask(params bool[] values) => np.array(values);

        // ======================================================================================================
        //  NumPy's checks, in NumPy's order, with NumPy's texts.
        // ======================================================================================================

        /// <summary>
        ///     The casting name is converted before anything else is read: an unknown name is NumPy's ValueError, a
        ///     <c>null</c> name its TypeError.
        /// </summary>
        [TestMethod]
        public void Casting_Name_IsCheckedFirst()
        {
            Action bogus = () => np.copyto(np.zeros(3), np.ones(3), casting: "bogus");
            bogus.Should().Throw<ValueError>()
                .WithMessage("casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got 'bogus')");

            Action none = () => np.copyto(np.zeros(3), np.ones(3), casting: null);
            none.Should().Throw<TypeError>().WithMessage("casting must be str, not NoneType");
        }

        /// <summary>
        ///     The mask's dtype is checked before the destination's writeability (NumPy converts the mask while parsing the
        ///     arguments), and the read-only destination before the cast, and the cast before the source's shape.
        /// </summary>
        [TestMethod]
        public void Checks_RunInNumPysOrder()
        {
            var readOnly = np.zeros(3);
            readOnly.setflags(write: false);

            Action badMask = () => np.copyto(readOnly, np.ones(3), where: np.array(new long[] { 1, 0, 1 }));
            badMask.Should().Throw<ArgumentException>()
                .WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");

            Action ro = () => np.copyto(readOnly, np.ones(3));
            ro.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");

            Action badCast = () => np.copyto(np.zeros(3, np.int32), np.ones(4));
            badCast.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'same_kind'");

            Action badShape = () => np.copyto(np.zeros(new Shape(2, 3)), np.ones(new Shape(3, 2)));
            badShape.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (3,2) into shape (2,3)");
        }

        /// <summary>
        ///     <c>np.copyto(a, a)</c> is a redundant self-copy (the same data, dtype, shape and strides) and returns before the
        ///     writeability check, so it is legal on a read-only broadcast view.
        /// </summary>
        [TestMethod]
        public void RedundantSelfCopy_OfAReadOnlyView_IsANoOp()
        {
            var b = np.broadcast_to(np.arange(3.0), new Shape(2, 3));
            np.copyto(b, b);
            b.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 0.0, 1.0, 2.0);
        }

        /// <summary>
        ///     Leading extent-1 axes of the source beyond the destination's rank are stripped first (NumPy's backward
        ///     compatibility rule) — never the mask's — and the broadcast errors print the shapes NumPy prints.
        /// </summary>
        [TestMethod]
        public void SourceLeadingUnitAxes_AreStripped_ButNotTheMasks()
        {
            var dst = np.zeros(3);
            np.copyto(dst, np.arange(3.0).reshape(1, 1, 3));
            dst.ToArray<double>().Should().Equal(0.0, 1.0, 2.0);

            Action srcErr = () => np.copyto(np.zeros(3), np.arange(6.0).reshape(1, 2, 3));
            srcErr.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,3) into shape (3,)");

            Action maskErr = () => np.copyto(np.zeros(3), np.ones(3), where: np.array(new[,] { { true, false, true } }));
            maskErr.Should().Throw<ValueError>().WithMessage("could not broadcast where mask from shape (1,3) into shape (3,)");

            Action shortMask = () => np.copyto(np.zeros(3), np.ones(3), where: Mask(true, false));
            shortMask.Should().Throw<ValueError>().WithMessage("could not broadcast where mask from shape (2,) into shape (3,)");

            // A scalar source (PyArray_AssignRawScalar) never strips the mask's axes either, weak or strong.
            Action weakScalar = () => np.copyto(np.zeros(3), 5.0, where: np.array(new[,] { { true, false, true } }));
            weakScalar.Should().Throw<ValueError>().WithMessage("could not broadcast where mask from shape (1,3) into shape (3,)");

            Action strongScalar = () => np.copyto(np.zeros(3), NDArray.Scalar(5.0), where: np.array(new[,] { { true, false, true } }));
            strongScalar.Should().Throw<ValueError>().WithMessage("could not broadcast where mask from shape (1,3) into shape (3,)");
        }

        /// <summary>
        ///     A 0-d mask is NumPy's scalar optimization: the source's shape is still checked first, then false copies
        ///     nothing and true copies everything.
        /// </summary>
        [TestMethod]
        public void ZeroDimensionalMask_ChecksTheSourceShapeFirst()
        {
            Action srcErr = () => np.copyto(np.zeros(3), np.ones(4), where: NDArray.Scalar(false));
            srcErr.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (4,) into shape (3,)");

            var none = np.zeros(3);
            np.copyto(none, np.ones(3), where: NDArray.Scalar(false));
            none.ToArray<double>().Should().Equal(0.0, 0.0, 0.0);

            var all = np.zeros(3);
            np.copyto(all, np.ones(3), where: NDArray.Scalar(true));
            all.ToArray<double>().Should().Equal(1.0, 1.0, 1.0);
        }

        /// <summary>
        ///     An empty destination still has its mask broadcast checked; a valid one copies nothing.
        /// </summary>
        [TestMethod]
        public void EmptyDestination_ChecksTheMask()
        {
            Action bad = () => np.copyto(np.zeros(new Shape(0, 3)), np.ones(3), where: Mask(true, false));
            bad.Should().Throw<ValueError>().WithMessage("could not broadcast where mask from shape (2,) into shape (0,3)");

            var empty = np.zeros(new Shape(0, 3));
            np.copyto(empty, np.ones(3), where: Mask(true, false, true));
            empty.shape.Should().Equal(0L, 3L);
        }

        /// <summary>
        ///     A 0-d destination takes a source whose every axis is a leading unit axis, and a 0-d source broadcasts to every
        ///     selected element.
        /// </summary>
        [TestMethod]
        public void ZeroDimensionalOperands()
        {
            var d0 = NDArray.Scalar(5.0);
            np.copyto(d0, np.array(new[,] { { 7.0 } }));
            d0.GetDouble().Should().Be(7.0);

            var kept = NDArray.Scalar(5.0);
            np.copyto(kept, NDArray.Scalar(7.0), where: NDArray.Scalar(false));
            kept.GetDouble().Should().Be(5.0);

            var grid = np.zeros(new Shape(2, 2));
            np.copyto(grid, NDArray.Scalar((short)3), where: np.array(new[,] { { true, false }, { false, true } }));
            grid.ToArray<double>().Should().Equal(3.0, 0.0, 0.0, 3.0);
        }

        // ======================================================================================================
        //  C# scalars: NumPy's Python scalars (weak) and NumPy scalars (strong).
        // ======================================================================================================

        /// <summary>
        ///     A C# integer is NumPy's Python int: weak (it takes the destination's dtype) and range-checked — it raises
        ///     NumPy's OverflowError text where the old code wrapped, before the mask is read. A <c>BigInteger</c> is a Python
        ///     int of any size.
        /// </summary>
        [TestMethod]
        public void WeakInteger_IsRangeChecked_NeverWrapped()
        {
            Action int8 = () => np.copyto(np.zeros(3, np.int8), 1000);
            int8.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");

            Action masked = () => np.copyto(np.zeros(3, np.int8), 1000, where: Mask(true, false, true));
            masked.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");

            Action negative = () => np.copyto(np.zeros(3, np.uint8), -1);
            negative.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");

            var u8 = np.array(new byte[] { 7, 7, 7 });
            np.copyto(u8, 0);
            u8.ToArray<byte>().Should().Equal((byte)0, (byte)0, (byte)0);

            var i16 = np.arange(4).astype(np.int16);
            np.copyto(i16, 9, where: Mask(true, false, false, true));
            i16.ToArray<short>().Should().Equal((short)9, (short)1, (short)2, (short)9);

            var f64 = np.zeros(2);
            np.copyto(f64, BigInteger.Pow(2, 70));
            f64.ToArray<double>().Should().Equal(1.1805916207174113e+21, 1.1805916207174113e+21);

            var u64 = np.zeros(2, np.uint64);
            np.copyto(u64, (BigInteger)ulong.MaxValue);
            u64.ToArray<ulong>().Should().Equal(ulong.MaxValue, ulong.MaxValue);
        }

        /// <summary>
        ///     A weak float takes a float destination's dtype even under <c>casting="safe"</c>; a float into an integer, an int
        ///     into a bool and a complex into a float are NumPy's cast errors on the resolved dtype ("Cannot cast scalar
        ///     from …"); <c>"equiv"</c> refuses any dtype but the scalar's own default.
        /// </summary>
        [TestMethod]
        public void WeakScalars_ResolveTheirDtype_LikeNumPy()
        {
            var f32 = np.zeros(3, np.float32);
            np.copyto(f32, 0.1, casting: "safe");
            f32.ToArray<float>().Should().Equal(0.1f, 0.1f, 0.1f);

            Action floatToInt = () => np.copyto(np.zeros(3, np.int32), 1.5);
            floatToInt.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast scalar from dtype('float64') to dtype('int32') according to the rule 'same_kind'");

            Action intToBool = () => np.copyto(np.zeros(3, np.@bool), 1);
            intToBool.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast scalar from dtype('int64') to dtype('bool') according to the rule 'same_kind'");

            Action complexToFloat = () => np.copyto(np.zeros(3), new Complex(0, 1));
            complexToFloat.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast scalar from dtype('complex128') to dtype('float64') according to the rule 'same_kind'");

            Action equiv = () => np.copyto(np.zeros(3, np.int32), 5, casting: "equiv");
            equiv.Should().Throw<TypeError>().WithMessage("cannot cast Python int to int32 under the casting rule 'equiv'");

            var i64 = np.zeros(3, np.int64);
            np.copyto(i64, 5, casting: "equiv");
            i64.ToArray<long>().Should().Equal(5L, 5L, 5L);
        }

        /// <summary>
        ///     A 0-d NDArray is NumPy's ndarray (strong): int64 into int8 is a same_kind cast and wraps (1000 → -24), as
        ///     <c>np.copyto(int8, np.array(1000))</c> does. A C# <c>bool</c> is a Python bool: a strong bool, legal into an
        ///     integer.
        /// </summary>
        [TestMethod]
        public void StrongScalars_CastLikeArrays()
        {
            var i8 = np.zeros(3, np.int8);
            np.copyto(i8, NDArray.Scalar(1000L));
            i8.ToArray<sbyte>().Should().Equal((sbyte)-24, (sbyte)-24, (sbyte)-24);

            var i32 = np.zeros(3, np.int32);
            np.copyto(i32, true);
            i32.ToArray<int>().Should().Equal(1, 1, 1);
        }

        // ======================================================================================================
        //  Overlap: NumPy's raw order (1-D, same direction) or a temporary copy first (everything else).
        // ======================================================================================================

        /// <summary>
        ///     A masked 1-D copy whose source overlaps the destination in the same direction walks in NumPy's raw order: from
        ///     the end when the source starts below the destination, so every selected element gets the source's ORIGINAL
        ///     value. The old masked path walked forward and read values it had already written.
        /// </summary>
        [TestMethod]
        public void MaskedOverlap_1D_SameDirection_IsNumPysOrder()
        {
            var m = Mask(true, false, true, true, false, true, true, true, false);

            var a = np.arange(10.0);
            np.copyto(a["1:"], a[":-1"], where: m);
            a.ToArray<double>().Should().Equal(0.0, 0.0, 2.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0, 9.0);

            var b = np.arange(10.0);
            np.copyto(b[":-1"], b["1:"], where: m);
            b.ToArray<double>().Should().Equal(1.0, 1.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0, 8.0, 9.0);

            // Longer than one unrolled SIMD pass (8 doubles): a forward walk would read, in its second pass, an element
            // the first pass already overwrote. NumPy's backward walk gives the copy-first answer for equal strides.
            var c = np.arange(41.0);
            var original = c.copy();
            var most = np.not_equal(np.mod(np.arange(40), 5), 3);
            np.copyto(c["1:"], c[":-1"], where: most);
            var expected = original.copy();
            expected["1:"] = np.where(most, original[":-1"], original["1:"]);
            np.array_equal(c, expected).Should().BeTrue();
        }

        /// <summary>
        ///     Opposite directions (a reversed source over its own array) and every multi-axis overlap copy the source to a
        ///     temporary first, masked or not.
        /// </summary>
        [TestMethod]
        public void MaskedOverlap_OppositeDirectionOr2D_CopiesTheSourceFirst()
        {
            var r = np.arange(10.0);
            var keep = np.not_equal(np.mod(np.arange(10), 3), 0);
            np.copyto(r, r["::-1"], where: keep);
            r.ToArray<double>().Should().Equal(0.0, 8.0, 7.0, 3.0, 5.0, 4.0, 6.0, 2.0, 1.0, 9.0);

            var a = np.arange(12.0).reshape(3, 4);
            np.copyto(a[":, 1:"], a[":, :-1"], where: Mask(true, false, true));
            a.ToArray<double>().Should().Equal(0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0);

            var t = np.arange(12.0).reshape(3, 4);
            var rows = np.array(new[,] { { true, false, true }, { true, false, true }, { true, false, true } });
            np.copyto(t[":3, :3"], t[":3, :3"].T, where: rows);
            t.ToArray<double>().Should().Equal(0.0, 1.0, 8.0, 3.0, 1.0, 5.0, 9.0, 7.0, 2.0, 9.0, 10.0, 11.0);
        }

        /// <summary>
        ///     Overlapping 1-D runs with DIFFERENT strides in the same direction are walked in NumPy's raw order, which reads
        ///     elements an earlier step wrote: <c>np.copyto(a[3:19:2], a[6:14])</c> "smears" in NumPy, masked or not, and the
        ///     same here (a copy-first answer would differ at four elements).
        /// </summary>
        [TestMethod]
        public void Overlap_UnequalStrides_ReadsItsOwnWrites_LikeNumPy()
        {
            var a = np.arange(20.0);
            np.copyto(a["3:19:2"], a["6:14"]);
            a.ToArray<double>().Should().Equal(
                0.0, 1.0, 2.0, 6.0, 4.0, 7.0, 6.0, 8.0, 8.0, 9.0, 10.0, 10.0, 12.0, 10.0, 14.0, 12.0, 16.0, 10.0, 18.0, 19.0);

            var b = np.arange(20.0);
            np.copyto(b["3:19:2"], b["6:14"], where: Mask(true, true, false, true, true, true, false, true));
            b.ToArray<double>().Should().Equal(
                0.0, 1.0, 2.0, 6.0, 4.0, 7.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 12.0, 10.0, 14.0, 15.0, 16.0, 10.0, 18.0, 19.0);
        }

        /// <summary>
        ///     A 1-D destination with a NEGATIVE stride over an overlapping source running the same way: NumPy flips both
        ///     (the walk starts at the destination's lowest address) before its backward rule, so <c>a[8::-1] = a[9:0:-1]</c>
        ///     is the copy-first shift, and unequal strides (<c>a[16::-2] = a[14:5:-1]</c>) read their own writes in that
        ///     flipped order — masked (the prepared raw iteration) and unmasked (<c>AssignRaw1D</c>) alike.
        /// </summary>
        [TestMethod]
        public void Overlap_NegativeStrideDestination_IsFlippedFirst()
        {
            var a = np.arange(10.0);
            np.copyto(a["8::-1"], a["9:0:-1"], where: Mask(true, true, false, true, true, true, false, true, true));
            a.ToArray<double>().Should().Equal(1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 6.0, 8.0, 9.0, 9.0);

            var b = np.arange(10.0);
            np.copyto(b["8::-1"], b["9:0:-1"]);
            b.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0);

            var c = np.arange(20.0);
            np.copyto(c["16::-2"], c["14:5:-1"]);
            c.ToArray<double>().Should().Equal(
                6.0, 1.0, 7.0, 3.0, 8.0, 5.0, 9.0, 7.0, 10.0, 9.0, 11.0, 11.0, 12.0, 13.0, 13.0, 15.0, 13.0, 17.0, 18.0, 19.0);

            var d = np.arange(20.0);
            np.copyto(d["16::-2"], d["14:5:-1"], where: Mask(true, false, true, true, true, false, true, true, true));
            d.ToArray<double>().Should().Equal(
                6.0, 1.0, 7.0, 3.0, 8.0, 5.0, 6.0, 7.0, 10.0, 9.0, 11.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 17.0, 18.0, 19.0);
        }

        /// <summary>
        ///     Slice assignment shares the rule (<c>NDIter.Copy</c> is the assignment path of both): NumPy's
        ///     <c>a[3:19:2] = a[6:14]</c>, <c>a[6:14] = a[3:19:2]</c> and <c>a[0:15:3] = a[0:5]</c> walk in the raw order and
        ///     read their own writes, where the old rule copied the source first (three different answers).
        /// </summary>
        [TestMethod]
        public void SliceAssignment_Overlap_UsesNumPysRawOrder()
        {
            var a = np.arange(20.0);
            a["3:19:2"] = a["6:14"];
            a.ToArray<double>().Should().Equal(
                0.0, 1.0, 2.0, 6.0, 4.0, 7.0, 6.0, 8.0, 8.0, 9.0, 10.0, 10.0, 12.0, 10.0, 14.0, 12.0, 16.0, 10.0, 18.0, 19.0);

            var b = np.arange(20.0);
            b["6:14"] = b["3:19:2"];
            b.ToArray<double>().Should().Equal(
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 5.0, 7.0, 9.0, 17.0, 17.0, 15.0, 17.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0);

            var c = np.arange(15.0);
            c["0:15:3"] = c["0:5"];
            c.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 1.0, 4.0, 5.0, 2.0, 7.0, 8.0, 1.0, 10.0, 11.0, 4.0, 13.0, 14.0);
        }

        /// <summary>
        ///     A 2-D overlapping copy larger than the stack staging (16 KB) stages the source in a pooled buffer: the result
        ///     is the copy-first answer, element for element.
        /// </summary>
        [TestMethod]
        public void MaskedOverlap_LargeStagedSource_IsTheCopyFirstAnswer()
        {
            var a = np.arange(6000.0).reshape(60, 100);
            var original = a.copy();
            var mask = np.equal(np.mod(np.arange(99), 3), 1);
            np.copyto(a[":, 1:"], a[":, :-1"], where: mask);

            var expected = original.copy();
            expected[":, 1:"] = np.where(mask, original[":, :-1"], original[":, 1:"]);
            np.array_equal(a, expected).Should().BeTrue();
        }

        // ======================================================================================================
        //  Cross-dtype masked copies: the source cast into the destination's dtype first.
        // ======================================================================================================

        /// <summary>
        ///     A source of another dtype is cast with NumPy's casts before the masked copy: contiguous, strided, reversed,
        ///     row-broadcast and bool sources, and an unsafe float → int32 cast (NaN and out-of-range → INT_MIN, as on
        ///     x86 NumPy).
        /// </summary>
        [TestMethod]
        public void CrossDtype_Masked_CastsLikeNumPy()
        {
            var m5 = Mask(true, false, true, true, false);

            var f64 = np.array(new[] { -1.0, -1.0, -1.0, -1.0, -1.0 });
            np.copyto(f64, np.array(new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f }), where: m5);
            f64.ToArray<double>().Should().Equal((double)0.1f, -1.0, (double)0.3f, (double)0.4f, -1.0);

            var f16 = np.array(new[] { -1.0, -1.0, -1.0, -1.0, -1.0 }).astype(np.float16);
            np.copyto(f16, np.arange(10).astype(np.int8)["::2"], where: m5);
            f16.ToArray<Half>().Should().Equal((Half)0, (Half)(-1), (Half)4, (Half)6, (Half)(-1));

            var c128 = np.zeros(new Shape(2, 3), np.complex128);
            np.copyto(c128, np.array(new[] { 1, 2, 3 }), where: np.array(new[,] { { true, false, true }, { false, true, true } }));
            c128.ToArray<Complex>().Should().Equal(
                new Complex(1, 0), Complex.Zero, new Complex(3, 0), Complex.Zero, new Complex(2, 0), new Complex(3, 0));

            var f32 = np.array(new[] { -1f, -1f, -1f, -1f, -1f });
            np.copyto(f32, np.arange(5).astype(np.uint16)["::-1"], where: m5);
            f32.ToArray<float>().Should().Equal(4f, -1f, 2f, 1f, -1f);

            var i64 = np.array(new long[] { 9, 9, 9, 9, 9 });
            np.copyto(i64, Mask(true, false, true, false, true), where: m5);
            i64.ToArray<long>().Should().Equal(1L, 9L, 1L, 0L, 9L);

            var i32 = np.array(new[] { -1, -1, -1, -1, -1 });
            np.copyto(i32, np.array(new[] { 1.9, -2.7, 3e10, double.NaN, 5.5 }), casting: "unsafe", where: m5);
            i32.ToArray<int>().Should().Equal(1, -1, int.MinValue, int.MinValue, -1);
        }

        /// <summary>
        ///     A source view whose axes address overlapping elements (<c>np.ndarray</c> with byte strides <c>(8, 8)</c>:
        ///     <c>[[0, 1, 2], [1, 2, 3], [2, 3, 4]]</c> over five doubles) is cast element by element through its own
        ///     strides: two axes of equal stride are never one run.
        /// </summary>
        [TestMethod]
        public void CrossDtype_Masked_SelfOverlappingSourceView()
        {
            var source = np.ndarray(new Shape(3, 3), np.float64, buffer: np.arange(5.0), offset: 0, strides: new long[] { 8, 8 });
            var mask = np.array(new[,] { { true, false, true }, { false, true, true }, { true, true, false } });

            var masked = np.zeros(new Shape(3, 3), np.float32);
            np.copyto(masked, source, where: mask);
            masked.ToArray<float>().Should().Equal(0f, 0f, 2f, 0f, 2f, 3f, 2f, 3f, 0f);

            var all = np.zeros(new Shape(3, 3), np.float32);
            np.copyto(all, source);
            all.ToArray<float>().Should().Equal(0f, 1f, 2f, 1f, 2f, 3f, 2f, 3f, 4f);
        }

        /// <summary>
        ///     A cross-dtype masked copy larger than the stack staging casts into a pooled buffer: the result equals
        ///     <c>np.where</c> over the cast source.
        /// </summary>
        [TestMethod]
        public void CrossDtype_Masked_LargeStagedSource_EqualsWhere()
        {
            var dst = np.full(new long[] { 5000 }, -1.0);
            var src = (np.arange(5000.0) / 7.0).astype(np.float32);
            var mask = np.equal(np.mod(np.arange(5000), 4), 1);
            var expected = np.where(mask, src.astype(np.float64), dst);
            np.copyto(dst, src, where: mask);
            np.array_equal(dst, expected).Should().BeTrue();
        }

        // ======================================================================================================
        //  Every element width, every layout route of the masked kernel.
        // ======================================================================================================

        /// <summary>
        ///     The masked copy is a bit copy of the destination's element width (1, 2, 4, 8 or 16 bytes), so it is pinned for
        ///     all 15 dtypes over an F-ordered destination with a C-ordered source and mask (the kernel reorders the walk to
        ///     keep the most operands contiguous) — NumPy's values for every NumPy dtype, the same pattern for char and
        ///     decimal.
        /// </summary>
        [TestMethod]
        public void EveryDtype_FDestination_CSourceAndMask()
        {
            var mask = np.array(new[,]
            {
                { true, false, true, false },
                { false, true, true, false },
                { true, true, false, true },
            });
            var source = np.array(new double[,] { { 1, 2, 3, 4 }, { 5, 1, 2, 3 }, { 4, 5, 1, 2 } });
            var expected = np.array(new double[,] { { 1, 0, 3, 0 }, { 0, 1, 2, 0 }, { 4, 5, 0, 2 } });

            foreach (var dtype in new[]
                     {
                         np.@bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64,
                         np.float16, np.float32, np.float64, np.complex128, np.@decimal, np.@char,
                     })
            {
                var dst = np.asfortranarray(np.zeros(new Shape(3, 4), dtype));
                np.copyto(dst, source.astype(dtype), where: mask);
                var want = dtype == np.@bool ? mask : expected.astype(dtype);
                np.array_equal(dst, want).Should().BeTrue($"dtype {dtype.name}");
                dst.Shape.IsFContiguous.Should().BeTrue($"dtype {dtype.name}");
            }
        }

        /// <summary>
        ///     A reversed, stepped destination with a stepped mask (NumPy flips the destination's negative stride and walks it
        ///     forward) for 1-, 8- and 16-byte elements.
        /// </summary>
        [TestMethod]
        public void NegativeStrideDestination_SteppedMask()
        {
            var mask = Mask(true, false, true, true, false, true, true, false, true, false, true, true)["::2"];
            foreach (var dtype in new[] { np.int8, np.int64, np.complex128 })
            {
                var view = np.zeros(new Shape(12), dtype)["::-2"];
                np.copyto(view, (np.arange(6) + 1).astype(dtype), where: mask);
                np.array_equal(view, np.array(new double[] { 1, 2, 0, 4, 5, 6 }).astype(dtype)).Should().BeTrue($"dtype {dtype.name}");
            }
        }
    }
}
