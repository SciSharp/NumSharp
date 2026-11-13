using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 8 (the owner approved fourteen of the fifteen picks; the file parameter names, OD3, stay open):
    ///     every expected value, dtype and text is NumPy 2.4.2's (scratchpad batch8_numpy_evidence*.txt,
    ///     batch8_out_evidence.txt, batch8_out13_evidence.txt and batch8_round_wrap_evidence.txt), for the call exactly as
    ///     the C# fix makes it.
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch8Tests
    {
        /// <summary>NumPy's text for a copy=False request it cannot honour (npy_no_copy_err_msg), verbatim.</summary>
        private const string UnableToAvoidCopy =
            "Unable to avoid copy while creating an array as requested.\n" +
            "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).\n" +
            "For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.";

        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="T"/> with NumPy's exact text.</summary>
        /// <typeparam name="T">The exception type NumPy's error maps to.</typeparam>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw <typeparamref name="T"/> with that text.</exception>
        private static void Throws<T>(Action act, string message) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(message);

        /// <summary>A masked array's data in C order, as float64.</summary>
        /// <param name="m">The masked array.</param>
        /// <returns>Its data values (masked slots included).</returns>
        private static double[] Data(NDMaskedArray m) => np.ma.getdata(m).astype(np.float64).ToArray<double>();

        /// <summary>A masked array's full mask in C order (all False for nomask).</summary>
        /// <param name="m">The masked array.</param>
        /// <returns>The mask values.</returns>
        private static bool[] Mask(NDMaskedArray m) => np.ma.getmaskarray(m).ToArray<bool>();

        /// <summary>Whether a masked array holds NumPy's <c>nomask</c> (no mask array at all).</summary>
        /// <param name="m">The masked array.</param>
        /// <returns>True for nomask.</returns>
        private static bool IsNoMask(NDMaskedArray m) => ReferenceEquals(np.ma.getmask(m), np.ma.nomask);

        /// <summary>NumPy's <c>ma.array(np.arange(6.0).reshape(2, 3), mask=[[0, 1, 0], [0, 0, 1]], fill_value=-7.0)</c>.</summary>
        /// <returns>A new masked (2, 3) float64 array.</returns>
        private static NDMaskedArray M23()
            => np.ma.array(np.arange(6.0).reshape(2, 3), mask: new[,] { { false, true, false }, { false, false, true } }, fill_value: -7.0);

        /// <summary>NumPy's <c>ma.array(np.arange(6.0).reshape(2, 3), fill_value=-7.0)</c>: nomask.</summary>
        /// <returns>A new unmasked (2, 3) float64 masked array.</returns>
        private static NDMaskedArray N23() => np.ma.array(np.arange(6.0).reshape(2, 3), fill_value: -7.0);

        /// <summary>A soft masked out of zeros with a mask array (True at the first slot).</summary>
        /// <param name="shape">The out's shape.</param>
        /// <param name="hard">Whether its mask is hard.</param>
        /// <returns>The pre-masked out.</returns>
        private static NDMaskedArray Premasked(Shape shape, bool hard = false)
        {
            var mask = np.zeros(shape, np.@bool);
            mask.SetAtIndex(true, 0);
            return np.ma.array(np.zeros(shape), mask: mask, hard_mask: hard);
        }

        // ---- fix 1: np.zeros / np.ones / np.empty(shape, dtype=None, order='C', *, device=None) ------------------------

        /// <summary>One signature each: NumPy's positional order (shape, dtype, order), the device check, the order letters
        ///     and their refusals in NumPy's argument order, and a view's Shape laid out from its dimensions alone.</summary>
        [TestMethod]
        public void ZerosOnesEmpty_OneSignature_NumPysOrderAndDevice()
        {
            var z = np.zeros((2, 3));
            z.dtype.Should().Be(np.float64);
            z.Shape.IsContiguous.Should().BeTrue();
            np.zeros((2, 3), order: 'F').Shape.IsFContiguous.Should().BeTrue();
            // NumPy: np.zeros((2, 3), None, 'F') — the order is the third positional argument.
            np.zeros(new Shape(2, 3), null, 'F').Shape.IsFContiguous.Should().BeTrue();
            np.zeros(2, device: "cpu").ToArray<double>().Should().Equal(0.0, 0.0);

            Throws<ValueError>(() => np.zeros(2, device: "gpu"), "Device not understood. Only \"cpu\" is allowed, but received: gpu");
            Throws<ValueError>(() => np.zeros(2, order: 'X'), "order must be one of 'C', 'F', 'A', or 'K' (got 'X')");
            Throws<ValueError>(() => np.zeros(2, order: 'A'), "only 'C' or 'F' order is permitted");
            Throws<ValueError>(() => np.zeros(2, order: 'K'), "only 'C' or 'F' order is permitted");
            Throws<ValueError>(() => np.zeros(-1), "negative dimensions are not allowed");
            // NumPy's argument order: the order letter, the device, the C / F switch, then the allocation.
            Throws<ValueError>(() => np.zeros(-1, order: 'A'), "only 'C' or 'F' order is permitted");
            Throws<ValueError>(() => np.zeros(2, order: 'X', device: "gpu"), "order must be one of 'C', 'F', 'A', or 'K' (got 'X')");
            Throws<ValueError>(() => np.zeros(2, order: 'A', device: "gpu"), "Device not understood. Only \"cpu\" is allowed, but received: gpu");

            var ones = np.ones((2, 2), np.int32, 'F');
            ones.dtype.Should().Be(np.int32);
            ones.Shape.IsFContiguous.Should().BeTrue();
            ones.ToArray<int>().Should().Equal(1, 1, 1, 1);
            np.ones(new[] { 2, 3 }, dtype: null).dtype.Should().Be(np.float64);
            np.empty((2, 2), order: 'F').Shape.IsFContiguous.Should().BeTrue();
            var e = np.empty(0, np.int8);
            e.dtype.Should().Be(np.int8);
            e.size.Should().Be(0);

            // A transposed view's Shape: NumPy's np.zeros(t.shape) is C-ordered (gap VS01 kept the view's F layout).
            var t = np.zeros((3, 2)).T;
            np.zeros(t.Shape).strides.Should().Equal(24L, 8L);

            foreach (var name in new[] { "zeros", "ones", "empty" })
                typeof(np).GetMethods().Count(m => m.Name == name).Should().Be(1, name);
        }

        // ---- fix 2: np.full(shape, fill_value, dtype=None, order='C', *, device=None) ---------------------------------

        /// <summary>np.full's one signature: the fill value's own dtype, NumPy's order letters and device, and the order
        ///     refusal before the negative dimension.</summary>
        [TestMethod]
        public void Full_OneSignature_OrderAndDevice()
        {
            // A C# int is int32 (the house mapping; NumPy's Python int 7 is int64, which a C# long spells).
            var f = np.full((2, 2), 7);
            f.dtype.Should().Be(np.int32);
            f.ToArray<int>().Should().Equal(7, 7, 7, 7);
            np.full((2, 2), 7L).dtype.Should().Be(np.int64);
            f.Shape.IsContiguous.Should().BeTrue();
            np.full((2, 2), 7, order: 'F').Shape.IsFContiguous.Should().BeTrue();
            np.full((2, 2), 7, null, 'F').Shape.IsFContiguous.Should().BeTrue();
            var i = np.full(3, 1.5, np.dtype("int32"));
            i.dtype.Should().Be(np.int32);
            i.ToArray<int>().Should().Equal(1, 1, 1);
            np.full(2, 1L, device: "cpu").ToArray<long>().Should().Equal(1L, 1L);
            np.full(new[] { 2, 3 }, 0.5).dtype.Should().Be(np.float64);
            Throws<ValueError>(() => np.full(2, 1, order: 'X'), "order must be one of 'C', 'F', 'A', or 'K' (got 'X')");
            Throws<ValueError>(() => np.full(-1, 1, order: 'A'), "only 'C' or 'F' order is permitted");
            typeof(np).GetMethods().Count(m => m.Name == "full").Should().Be(1);
        }

        // ---- fix 3: the generic creation forms are gone (OD7) ---------------------------------------------------------

        /// <summary>No <c>np.zeros&lt;T&gt;</c> / <c>ones&lt;T&gt;</c> / <c>empty&lt;T&gt;</c> / <c>full&lt;T&gt;</c>: NumPy's
        ///     spelling passes the dtype (<c>np.zeros(shape, dtype=float)</c>).</summary>
        [TestMethod]
        public void CreationGenerics_AreGone_TheDtypeIsAnArgument()
        {
            typeof(np).GetMethods().Where(m => m.Name is "zeros" or "ones" or "empty" or "full" && m.IsGenericMethodDefinition)
                .Should().BeEmpty();
            np.zeros((2, 3), np.float64).dtype.Should().Be(np.float64);
            np.full(2, 7, np.int32).ToArray<int>().Should().Equal(7, 7);
        }

        // ---- fix 4: np.indices(dimensions, dtype=int) / np.ma.indices(..., fill_value, hardmask) ----------------------

        /// <summary>One Shape form: the int64 grid, a dtype, the empty dimensions; numpy.ma's fill_value and hardmask over a
        ///     full all-False mask.</summary>
        [TestMethod]
        public void Indices_OneShape_AndMaIndicesKeywords()
        {
            var g = np.indices((2, 3));
            g.dtype.Should().Be(np.int64);
            g.shape.Should().Equal(2L, 2L, 3L);
            g.ToArray<long>().Should().Equal(0L, 0L, 0L, 1L, 1L, 1L, 0L, 1L, 2L, 0L, 1L, 2L);
            np.indices(new[] { 2, 3 }, np.int32).dtype.Should().Be(np.int32);
            var none = np.indices(Array.Empty<int>());
            none.shape.Should().Equal(0L);
            none.dtype.Should().Be(np.int64);

            var mi = np.ma.indices((2, 2));
            IsNoMask(mi).Should().BeTrue("NumPy's _convert2ma views the plain grid: nomask");
            Convert.ToInt64(mi.fill_value).Should().Be(999999L);
            Convert.ToInt64(np.ma.indices((2, 2), fill_value: -1).fill_value).Should().Be(-1L);
            np.ma.indices((2, 2), hardmask: true).hardmask.Should().BeTrue();
            typeof(np).GetMethods().Count(m => m.Name == "indices").Should().Be(1);
        }

        /// <summary><c>np.indices(3)</c>: an int converts to a one-dimension <see cref="Shape"/>, so NumSharp gives the (1, 3)
        ///     grid where NumPy raises <c>TypeError: 'int' object is not iterable</c> — a Shape cannot tell 3 from (3,).</summary>
        [TestMethod, Misaligned]
        public void Indices_OfAnInt_IsAOneDimensionGrid_WhereNumPyRefuses()
        {
            var g = np.indices(3);
            g.shape.Should().Equal(1L, 3L);
            g.ToArray<long>().Should().Equal(0L, 1L, 2L);
        }

        // ---- fix 5: np.asanyarray(a, dtype=None, order=None, *, device=None, copy=None) -------------------------------

        /// <summary>asanyarray's order (a copy only when the layout lacks it) and copy modes (True always copies, False
        ///     refuses a needed copy with NumPy's text), the subclass kept, the device and the order letter checks.</summary>
        [TestMethod]
        public void Asanyarray_OrderAndCopy()
        {
            var a23 = np.arange(6L).reshape(2, 3);
            var fcopy = np.asanyarray(a23, order: 'F');
            ReferenceEquals(fcopy, a23).Should().BeFalse();
            fcopy.Shape.IsFContiguous.Should().BeTrue();
            fcopy.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            var f23 = np.asfortranarray(a23);
            ReferenceEquals(np.asanyarray(f23, order: 'F'), f23).Should().BeTrue();
            ReferenceEquals(np.asanyarray(a23, copy: true), a23).Should().BeFalse();
            ReferenceEquals(np.asanyarray(a23, copy: false), a23).Should().BeTrue();
            // A 1-D array is C and F at once: order F keeps it.
            var a6 = np.arange(6L);
            ReferenceEquals(np.asanyarray(a6, null, 'F'), a6).Should().BeTrue();

            Throws<ValueError>(() => np.asanyarray(a23, np.float64, copy: false), UnableToAvoidCopy);
            Throws<ValueError>(() => np.asanyarray(a23, order: 'F', copy: false), UnableToAvoidCopy);
            Throws<ValueError>(() => np.asanyarray(new[] { 1, 2 }, order: 'X'), "order must be one of 'C', 'F', 'A', or 'K' (got 'X')");
            np.asanyarray(new[] { 1, 2 }, device: "cpu").size.Should().Be(2);
            np.asanyarray(np.asmatrix(np.array(new long[,] { { 1, 2 } })), order: 'F').Should().BeOfType<NDMatrix>();
        }

        // ---- fix 7: np.tile(A, reps) ----------------------------------------------------------------------------------

        /// <summary>One reps argument (an int, a tuple, an array, the empty tuple), NumPy's negative-repetition text and the
        ///     copies NumPy makes.</summary>
        [TestMethod]
        public void Tile_OneRepsArgument()
        {
            var x12 = np.array(new long[] { 1, 2 });
            np.tile(x12, 2).ToArray<long>().Should().Equal(1L, 2L, 1L, 2L);
            var t22 = np.tile(x12, (2, 2));
            t22.shape.Should().Equal(2L, 4L);
            t22.ToArray<long>().Should().Equal(1L, 2L, 1L, 2L, 1L, 2L, 1L, 2L);
            np.tile(x12, new[] { 2, 1, 2 }).shape.Should().Equal(2L, 1L, 4L);
            Throws<ValueError>(() => np.tile(x12, -1), "negative dimensions are not allowed");
            var zero = np.tile(x12, 0);
            zero.size.Should().Be(0);
            zero.dtype.Should().Be(np.int64);
            var copy = np.tile(x12, Array.Empty<int>());
            copy.ToArray<long>().Should().Equal(1L, 2L);
            np.shares_memory(copy, x12).Should().BeFalse();
            ReferenceEquals(np.tile(x12, new[] { 1 }), x12).Should().BeFalse();
            typeof(np).GetMethods().Count(m => m.Name == "tile").Should().Be(1);
        }

        // ---- fix 8: np.ma.stack / vstack / hstack with dtype and casting ----------------------------------------------

        /// <summary>numpy.ma's joins pass dtype / casting to the data join AND the mask join (the masks read back as
        ///     bool).</summary>
        [TestMethod]
        public void MaJoins_DtypeAndCasting()
        {
            var m12 = np.ma.array(np.array(new long[] { 1, 2 }), mask: new[] { false, true });
            var b = np.array(new long[] { 3, 4 });

            var v = np.ma.vstack(new object[] { m12, b }, np.float32);
            v.dtype.Should().Be(np.float32);
            Data(v).Should().Equal(1.0, 2.0, 3.0, 4.0);
            Mask(v).Should().Equal(false, true, false, false);
            np.ma.getmaskarray(v).dtype.Should().Be(np.@bool);
            var h = np.ma.hstack(new object[] { m12, b }, np.float32);
            h.shape.Should().Equal(4L);
            Mask(h).Should().Equal(false, true, false, false);
            var s = np.ma.stack(new object[] { m12, b }, 1, np.float32);
            Data(s).Should().Equal(1.0, 3.0, 2.0, 4.0);
            Mask(s).Should().Equal(false, false, true, false);

            // NumPy's TypeError texts; the type is np.concatenate's (NumSharp raises its casting refusal as an
            // InvalidCastException, np.concatenate.Test.cs).
            Throws<InvalidCastException>(() => np.ma.stack(new object[] { m12, b }, dtype: np.float32, casting: "no"),
                "Cannot cast array data from dtype('int64') to dtype('float32') according to the rule 'no'");
            Throws<InvalidCastException>(() => np.ma.vstack(new object[] { m12, b }, np.int8, "safe"),
                "Cannot cast array data from dtype('int64') to dtype('int8') according to the rule 'safe'");
        }

        // ---- fix 9: np.ma.reshape(a, new_shape, order='C') / MaskedArray.reshape(shape, order, copy) -------------------

        /// <summary>numpy.ma's reshape takes the shape as one argument and NumPy's order (data and mask alike); the method
        ///     adds NumPy's copy mode.</summary>
        [TestMethod]
        public void MaReshape_OrderAndCopy()
        {
            var m6 = np.ma.array(np.arange(6L), mask: new[] { false, true, false, false, true, false });
            var c = np.ma.reshape(m6, (2, 3));
            Data(c).Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            Mask(c).Should().Equal(false, true, false, false, true, false);
            var f = np.ma.reshape(m6, (2, 3), 'F');
            Data(f).Should().Equal(0.0, 2.0, 4.0, 1.0, 3.0, 5.0);
            Mask(f).Should().Equal(false, false, true, true, false, false);
            np.ma.getdata(f).Shape.IsFContiguous.Should().BeTrue();
            Mask(m6.reshape((2, 3), 'F')).Should().Equal(false, false, true, true, false, false);
            m6.reshape(2, 3).shape.Should().Equal(2L, 3L);
            Throws<ValueError>(() => m6.reshape((3, 2)).T.reshape(6, copy: false), "Unable to avoid creating a copy while reshaping.");
        }

        // ---- fix 10: np.ma.squeeze(a, axis=None, *, fill_value, hardmask) / expand_dims tuple axes ---------------------

        /// <summary>numpy.ma's squeeze takes a tuple of axes and _convert2ma's keywords; expand_dims a tuple; a non-unit
        ///     axis is refused with NumPy's text.</summary>
        [TestMethod]
        public void MaSqueezeAndExpandDims_TupleAxes()
        {
            var m1 = np.ma.array(np.arange(3L).reshape(1, 3, 1), mask: np.array(new[] { false, true, false }).reshape(1, 3, 1));
            var s = np.ma.squeeze(m1, (0, 2));
            s.shape.Should().Equal(3L);
            Mask(s).Should().Equal(false, true, false);
            Convert.ToInt64(np.ma.squeeze(m1, fill_value: 9).fill_value).Should().Be(9L);
            np.ma.squeeze(m1, hardmask: true).hardmask.Should().BeTrue();
            m1.squeeze((0, 2)).shape.Should().Equal(3L);
            var m12 = np.ma.array(np.array(new long[] { 1, 2 }), mask: new[] { false, true });
            var e = np.ma.expand_dims(m12, (0, 2));
            e.shape.Should().Equal(1L, 2L, 1L);
            Mask(e).Should().Equal(false, true);
            Throws<ValueError>(() => np.ma.squeeze(m1, 1), "cannot select an axis to squeeze out which has size not equal to one");
        }

        // ---- fix 11: numpy.ma reductions — tuple axes and out= -------------------------------------------------------

        /// <summary>Tuple axes over every reduction, and NumPy's out= branches: a masked out takes the result mask (sum,
        ///     max), a plain one the data; min / max / ptp refuse a plain integer out; keepdims is the fifth positional.</summary>
        [TestMethod]
        public void MaReductions_TupleAxesAndOut()
        {
            var m23 = M23();
            np.ma.getdata(np.ma.sum(m23, (0, 1))).GetDouble().Should().Be(9.0);
            var s0 = np.ma.sum(m23, new[] { 0 });
            Data(s0).Should().Equal(3.0, 4.0, 2.0);
            np.ma.getdata(np.ma.mean(m23, (0, 1))).GetDouble().Should().Be(2.25);
            np.ma.getdata(np.ma.max(m23, (0, 1))).GetDouble().Should().Be(4.0);
            np.ma.getdata(np.ma.var(m23, (0, 1))).GetDouble().Should().Be(2.1875);
            np.ma.getdata(np.ma.ptp(m23, (0, 1))).GetDouble().Should().Be(4.0);
            np.ma.count(m23, (0, 1)).GetInt64().Should().Be(4L);
            np.ma.getdata(np.ma.all(m23, (0, 1))).GetBoolean().Should().BeFalse();
            Throws<ValueError>(() => np.ma.sum(m23, (0, 0)), "duplicate value in 'axis'");

            var soft = np.ma.zeros(3);
            ReferenceEquals(np.ma.sum(m23, 0, @out: soft), soft).Should().BeTrue();
            Data(soft).Should().Equal(3.0, 4.0, 2.0);
            IsNoMask(soft).Should().BeFalse("outmask.flat = newmask creates the out's mask");
            Mask(soft).Should().Equal(false, false, false);
            var hard = np.ma.zeros(3, hardmask: true);
            np.ma.max(m23, 0, @out: hard);
            Data(hard).Should().Equal(3.0, 4.0, 2.0);
            hard.hardmask.Should().BeTrue();
            var plain = np.zeros(3);
            np.ma.sum(m23, 0, @out: plain);
            plain.ToArray<double>().Should().Equal(3.0, 4.0, 2.0);
            var plainInt = np.zeros(3, np.int64);
            np.ma.mean(m23, 0, @out: plainInt);
            plainInt.ToArray<long>().Should().Equal(1L, 4L, 2L);
            Throws<MaskError>(() => np.ma.min(m23, 0, @out: np.zeros(3, np.int64)), "Masked data information would be lost in one or more location.");
            var v = np.ma.zeros(3);
            np.ma.var(m23, 0, @out: v);
            Data(v).Should().Equal(2.25, 0.0, 0.0);
            IsNoMask(v).Should().BeTrue();
            // NumPy: np.ma.sum(m23, 0, None, None, True) — keepdims is the fifth positional argument.
            np.ma.sum(m23, 0, null, null, true).shape.Should().Equal(1L, 3L);
        }

        // ---- fix 12: np.ma.array(data, dtype, copy, order, mask, ...) / masked_array(data, mask, dtype, ...) ----------

        /// <summary>ma.array's NumPy order (the dtype second, the mask fifth), ndmin, the order letters, masked_array's own
        ///     order and asarray / asanyarray's order.</summary>
        [TestMethod]
        public void MaArray_NumPysParameterOrder()
        {
            var x12 = np.array(new long[] { 1, 2 });
            var f32 = np.ma.array(x12, np.float32);
            f32.dtype.Should().Be(np.float32);
            IsNoMask(f32).Should().BeTrue();
            Mask(np.ma.array(x12, null, false, null, new[] { false, true })).Should().Equal(false, true);
            Mask(np.ma.array(x12, mask: new[] { false, true })).Should().Equal(false, true);
            np.ma.array(x12, ndmin: 2).shape.Should().Equal(1L, 2L);
            var a23 = np.arange(6L).reshape(2, 3);
            np.ma.getdata(np.ma.array(a23, order: 'F')).Shape.IsFContiguous.Should().BeTrue();
            np.ma.getdata(np.ma.array(a23, copy: true, order: 'F')).Shape.IsFContiguous.Should().BeTrue();
            Throws<ValueError>(() => np.ma.array(new[] { 1 }, ndmin: 65), "ndmin must be <= ndmax (64)");
            Throws<ValueError>(() => np.ma.array(new[] { 1 }, order: 'X'), "order must be one of 'C', 'F', 'A', or 'K' (got 'X')");

            Mask(np.ma.masked_array(x12, new[] { false, true })).Should().Equal(false, true);
            var mf = np.ma.masked_array(x12, new[] { false, true }, np.float32);
            mf.dtype.Should().Be(np.float32);
            Mask(mf).Should().Equal(false, true);
            np.ma.getdata(np.ma.asarray(a23, order: 'F')).Shape.IsFContiguous.Should().BeTrue();
            np.ma.getdata(np.ma.asanyarray(a23, order: 'F')).Shape.IsFContiguous.Should().BeTrue();
        }

        // ---- fix 13: numpy.ma out= ------------------------------------------------------------------------------------

        /// <summary>cumsum / cumprod write the filled scan into out, then <c>out.__setmask__(self.mask)</c>: a hard out ORs,
        ///     a soft one takes the values (C order for axis=None), nomask leaves a mask-less out alone and clears a soft
        ///     one; NumPy's shape refusals, including the hard out's broadcast error after the data was written.</summary>
        [TestMethod]
        public void MaCumsumCumprod_Out()
        {
            var m23 = M23();
            var soft = np.ma.zeros((2, 3));
            ReferenceEquals(np.ma.cumsum(m23, 1, null, soft), soft).Should().BeTrue();
            Data(soft).Should().Equal(0.0, 0.0, 2.0, 3.0, 7.0, 7.0);
            Mask(soft).Should().Equal(false, true, false, false, false, true);
            var premaskedHard = Premasked((2, 3), hard: true);
            np.ma.cumsum(m23, 1, null, premaskedHard);
            Mask(premaskedHard).Should().Equal(true, true, false, false, false, true);
            var plainInt = np.zeros((2, 3), np.int64);
            np.ma.cumsum(m23, 1, null, plainInt);
            plainInt.ToArray<long>().Should().Equal(0L, 0L, 2L, 3L, 7L, 7L);
            var flat = np.ma.zeros(6);
            np.ma.cumsum(m23, null, null, flat);
            Data(flat).Should().Equal(0.0, 0.0, 2.0, 5.0, 9.0, 9.0);
            Mask(flat).Should().Equal(false, true, false, false, false, true);
            var flatHard = np.ma.zeros(6, hardmask: true);
            Throws<IncorrectShapeException>(() => np.ma.cumsum(m23, null, null, flatHard), "operands could not be broadcast together with shapes (6,) (2,3) (6,) ");
            Data(flatHard).Should().Equal(0.0, 0.0, 2.0, 5.0, 9.0, 9.0);
            Mask(flatHard).Should().OnlyContain(b => !b);
            var wrong = np.ma.zeros((2, 2));
            Throws<IncorrectShapeException>(() => np.ma.cumsum(m23, 1, null, wrong),
                "operands could not be broadcast together with remapped shapes [original->remapped]: (2,2)->(2,2) (2,3)->(2,3) ");
            Data(wrong).Should().OnlyContain(d => d == 0.0);
            IsNoMask(wrong).Should().BeTrue();

            // An unmasked operand: `__setmask__(nomask)`.
            var bare = np.ma.zeros((2, 3));
            np.ma.cumsum(N23(), 1, null, bare);
            IsNoMask(bare).Should().BeTrue();
            var cleared = Premasked((2, 3));
            np.ma.cumsum(N23(), 1, null, cleared);
            IsNoMask(cleared).Should().BeFalse();
            Mask(cleared).Should().OnlyContain(b => !b);
            var kept = Premasked((2, 3), hard: true);
            np.ma.cumsum(N23(), 1, null, kept);
            Mask(kept).Should().Equal(true, false, false, false, false, false);

            var prod = np.ma.zeros((2, 3));
            ReferenceEquals(m23.cumprod(1, null, prod), prod).Should().BeTrue();
            Data(prod).Should().Equal(0.0, 0.0, 0.0, 3.0, 12.0, 12.0);
            Mask(prod).Should().Equal(false, true, false, false, false, true);
        }

        /// <summary>take writes the gather into out (np.take's checks: nothing written for an out-of-bounds index), sets
        ///     the gathered mask through <c>__setmask__</c>, returns <c>out[()]</c> (masked / a scalar for 0-d), and takes the
        ///     mode after out.</summary>
        [TestMethod]
        public void MaTake_Out()
        {
            var m23 = M23();
            var idx = np.array(new long[] { 0, 2 });
            var soft = np.ma.zeros((2, 2));
            ReferenceEquals(np.ma.take(m23, idx, 1, soft), soft).Should().BeTrue();
            Data(soft).Should().Equal(0.0, 2.0, 3.0, 5.0);
            Mask(soft).Should().Equal(false, false, false, true);
            var premaskedHard = Premasked((2, 2), hard: true);
            np.ma.take(m23, idx, 1, premaskedHard);
            Mask(premaskedHard).Should().Equal(true, false, false, true);
            var bare = np.ma.zeros((2, 2));
            np.ma.take(N23(), idx, 1, bare);
            IsNoMask(bare).Should().BeTrue();
            var cleared = Premasked((2, 2));
            np.ma.take(N23(), idx, 1, cleared);
            Mask(cleared).Should().Equal(false, false, false, false);
            var plain = np.zeros(2);
            np.ma.take(m23, np.array(new long[] { 1, 5 }), null, plain);
            plain.ToArray<double>().Should().Equal(1.0, 5.0);

            var clip = np.ma.zeros((2, 2));
            np.ma.take(m23, np.array(new long[] { 0, 7 }), 1, clip, mode: "clip");
            Data(clip).Should().Equal(0.0, 2.0, 3.0, 5.0);
            var untouched = np.ma.zeros((2, 2));
            Throws<IndexError>(() => np.ma.take(m23, np.array(new long[] { 0, 7 }), 1, untouched), "index 7 is out of bounds for axis 1 with size 3");
            Data(untouched).Should().OnlyContain(d => d == 0.0);
            Throws<ValueError>(() => np.ma.take(m23, idx, 1, np.ma.zeros((2, 3))), "output array does not match result of ndarray.take");

            // `out[()]` of a 0-d out: masked when its element is masked, else the value.
            var o0 = np.ma.zeros(Shape.Scalar);
            ReferenceEquals(np.ma.take(m23, NDArray.Scalar(1L), null, o0), np.ma.NDMasked).Should().BeTrue();
            np.ma.getdata(o0).GetDouble().Should().Be(1.0);
            Mask(o0).Should().Equal(true);
            var scalar = np.ma.take(m23, NDArray.Scalar(0L), null, np.ma.zeros(Shape.Scalar));
            np.ma.getdata(scalar).GetDouble().Should().Be(0.0);
            IsNoMask(scalar).Should().BeTrue();
            ReferenceEquals(np.ma.take(m23, NDArray.Scalar(1L)), np.ma.NDMasked).Should().BeTrue();

            var viaMethod = np.ma.zeros((2, 2));
            ReferenceEquals(m23.take(idx, 1, viaMethod), viaMethod).Should().BeTrue();
            Mask(viaMethod).Should().Equal(false, false, false, true);
        }

        /// <summary>ma.choose picks the masks first, then the choices FILLED with their own fill values (visible through a
        ///     plain out), sets the shrunk mask on a masked out; MaskedArray.choose is ndarray.choose (raw data, the index's
        ///     mask, no mask set on an out).</summary>
        [TestMethod]
        public void MaChoose_Out_AndTheMethodIsNdarrayChoose()
        {
            var idx = np.array(new long[,] { { 0, 1, 0 }, { 1, 0, 1 } });
            var maskedIdx = np.ma.array(idx, mask: new[,] { { false, false, true }, { false, false, false } });
            var choices = new object[]
            {
                np.ma.array(np.array(new double[,] { { 10, 11, 12 }, { 13, 14, 15 } }), mask: new[,] { { true, false, false }, { false, false, false } }),
                np.array(new double[,] { { 20, 21, 22 }, { 23, 24, 25 } }),
            };
            var plainChoices = new object[] { np.array(new double[,] { { 10, 11, 12 }, { 13, 14, 15 } }), choices[1] };

            var soft = np.ma.zeros((2, 3));
            ReferenceEquals(np.ma.choose(idx, choices, soft), soft).Should().BeTrue();
            Data(soft).Should().Equal(1e20, 21.0, 12.0, 23.0, 14.0, 25.0);
            Mask(soft).Should().Equal(true, false, false, false, false, false);
            var plain = np.zeros((2, 3));
            np.ma.choose(idx, choices, plain);
            plain.ToArray<double>().Should().Equal(1e20, 21.0, 12.0, 23.0, 14.0, 25.0);
            var byMaskedIndex = np.ma.zeros((2, 3));
            np.ma.choose(maskedIdx, plainChoices, byMaskedIndex);
            Data(byMaskedIndex).Should().Equal(10.0, 21.0, 12.0, 23.0, 14.0, 25.0);
            Mask(byMaskedIndex).Should().Equal(false, false, true, false, false, false);
            var cleared = Premasked((2, 3));
            np.ma.choose(idx, plainChoices, cleared);
            Mask(cleared).Should().OnlyContain(b => !b);
            var clipped = np.ma.zeros((2, 3));
            np.ma.choose(np.array(new long[,] { { 0, 3, 0 }, { 1, 0, 1 } }), plainChoices, clipped, mode: "clip");
            Data(clipped).Should().Equal(10.0, 21.0, 12.0, 23.0, 14.0, 25.0);
            Throws<TypeError>(() => np.ma.choose(idx, plainChoices, np.ma.zeros((2, 2))), "choose: invalid shape for output array.");

            // MaskedArray.choose = ndarray.choose: the stored index (0 at the masked slot), the raw choices, the index's mask.
            var viaMethod = maskedIdx.choose(choices);
            Data(viaMethod).Should().Equal(10.0, 21.0, 12.0, 23.0, 14.0, 25.0);
            Mask(viaMethod).Should().Equal(false, false, true, false, false, false);
            var methodOut = Premasked((2, 3));
            ReferenceEquals(maskedIdx.choose(choices, methodOut), methodOut).Should().BeTrue();
            Data(methodOut).Should().Equal(10.0, 21.0, 12.0, 23.0, 14.0, 25.0);
            Mask(methodOut).Should().Equal(true, false, false, false, false, false);
        }

        /// <summary>compress writes the data into out and returns a NEW masked array over out's buffer carrying the
        ///     operand's compressed mask and fill value; out's own mask is never touched.</summary>
        [TestMethod]
        public void MaCompress_Out_WritesTheDataOnly()
        {
            var m23 = M23();
            var cond = np.array(new[] { true, false, true });
            var o = np.ma.zeros((2, 2));
            var r = np.ma.compress(cond, m23, 1, o);
            ReferenceEquals(r, o).Should().BeFalse();
            np.shares_memory(np.ma.getdata(r), np.ma.getdata(o)).Should().BeTrue();
            Data(o).Should().Equal(0.0, 2.0, 3.0, 5.0);
            IsNoMask(o).Should().BeTrue();
            Mask(r).Should().Equal(false, false, false, true);
            Convert.ToDouble(r.fill_value).Should().Be(-7.0);
            var premasked = Premasked((2, 2));
            np.ma.compress(cond, m23, 1, premasked);
            Mask(premasked).Should().Equal(true, false, false, false);
            // An unmasked operand: the result shares out's mask.
            var shared = Premasked((2, 2));
            var viaView = np.ma.compress(cond, N23(), 1, shared);
            Mask(viaView).Should().Equal(true, false, false, false);
            Throws<ValueError>(() => np.ma.compress(cond, m23, 1, np.ma.zeros((2, 3))), "output array does not match result of ndarray.take");
            Mask(m23.compress(cond, 1, np.ma.zeros((2, 2)))).Should().Equal(false, false, false, true);
        }

        /// <summary>ma.round / MaskedArray.round / ma.around into a masked out go through NumPy's in-place ufuncs and their
        ///     __array_wrap__: an infinite result becomes 1.0 (10**-d for negative decimals), the wraps leave a mask, then
        ///     round_ shares the operand's mask object, the method calls __setmask__, around restores the masked data and
        ///     returns a new array; integer data with decimals >= 0 is assigned without a wrap; complex data refuses.</summary>
        [TestMethod]
        public void MaRoundAndAround_Out_FollowNumPysUfuncWraps()
        {
            var w = np.ma.array(np.array(new[] { 1.25, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 1e308, 2.5 }),
                                mask: new[] { false, false, false, false, false, true });

            var moduleOut = np.ma.zeros(6);
            ReferenceEquals(np.ma.round(w, 1, moduleOut), moduleOut).Should().BeTrue();
            Data(moduleOut).Should().Equal(1.2, 1.0, 1.0, double.NaN, 1.0, 2.5);
            ReferenceEquals(np.ma.getmask(moduleOut), np.ma.getmask(w)).Should().BeTrue("round_ stores out._mask = getmask(a)");
            var negative = np.ma.zeros(6);
            np.ma.round(w, -1, negative);
            Data(negative).Should().Equal(0.0, 10.0, 10.0, double.NaN, 1e308, 0.0);

            var methodHard = np.ma.zeros(6, hardmask: true);
            w.round(1, methodHard);
            Mask(methodHard).Should().Equal(false, true, true, false, true, true);
            var methodSoft = np.ma.zeros(6);
            w.round(1, methodSoft);
            Mask(methodSoft).Should().Equal(false, false, false, false, false, true);
            Data(methodSoft).Should().Equal(1.2, 1.0, 1.0, double.NaN, 1.0, 2.5);

            var aroundOut = np.ma.zeros(6);
            var r = np.ma.around(w, 1, aroundOut);
            ReferenceEquals(r, aroundOut).Should().BeFalse();
            Data(r).Should().Equal(1.2, 1.0, 1.0, double.NaN, 1.0, 2.5);
            ReferenceEquals(np.ma.getmask(r), np.ma.getmask(w)).Should().BeTrue();
            Mask(aroundOut).Should().Equal(false, true, true, false, true, false);
            var aroundZero = np.ma.zeros(6);
            np.ma.around(w, 0, aroundZero);
            Data(aroundZero).Should().Equal(1.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 1e308, 2.5);
            IsNoMask(aroundZero).Should().BeFalse("rint's wrap stores an all-False mask array");
            Mask(aroundZero).Should().OnlyContain(b => !b);

            // Integer data with decimals >= 0: no ufunc, no wrap — the hard out keeps its own masked slot.
            var ints = np.ma.array(np.array(new long[] { 14, 25, -36, 7, 9, 11 }), mask: new[] { false, false, false, false, false, true });
            var intOut = np.ma.array(np.zeros(6, np.int64), mask: new[] { true, false, false, false, false, false }, hard_mask: true);
            ints.round(0, intOut);
            Mask(intOut).Should().Equal(true, false, false, false, false, true);

            var complexData = np.ma.array(np.array(new[] { new Complex(1.25, 2.5), new Complex(3, 4) }));
            Throws<AttributeError>(() => np.ma.round(complexData, 1, np.ma.zeros(2, np.complex128)), "property 'real' of 'MaskedArray' object has no setter");
            Throws<UFuncOutputCastingError>(() => np.ma.round(w, 1, np.zeros(6, np.int64)),
                "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
            Throws<ValueError>(() => np.ma.round(w, 1, np.ma.zeros(4)), "invalid output shape");
        }

        /// <summary>ma.around is <c>_MaskedUnaryOperation(np.around)</c>, not ma.round: the masked slots keep the operand's
        ///     ORIGINAL data, and the result takes the operand's mask object and fill value (one form since batch 8).</summary>
        [TestMethod]
        public void MaAround_KeepsTheMaskedData_UnlikeMaRound()
        {
            var r23 = np.ma.array(np.arange(6.0).reshape(2, 3) / 3.0, mask: new[,] { { false, true, false }, { false, false, true } }, fill_value: -7.0);
            var around = np.ma.around(r23, 1);
            Data(around).Should().Equal(0.0, 1.0 / 3.0, 0.7, 1.0, 1.3, 5.0 / 3.0);
            ReferenceEquals(np.ma.getmask(around), np.ma.getmask(r23)).Should().BeTrue();
            Convert.ToDouble(around.fill_value).Should().Be(-7.0);
            Data(np.ma.round(r23, 1)).Should().Equal(0.0, 0.3, 0.7, 1.0, 1.3, 1.7);
            var ints = np.ma.around(np.ma.array(np.array(new long[] { 14, 25, 36 }), mask: new[] { false, true, false }), -1);
            np.ma.getdata(ints).ToArray<long>().Should().Equal(10L, 25L, 40L);
            typeof(MaskedArrayModule).GetMethods().Count(m => m.Name == "around").Should().Be(1);
        }

        /// <summary>ma.dot into a MASKED out: the product into its data (np.dot's exact-dtype / C-contiguity / shape rules) and
        ///     the no-valid-pair mask into its mask in place, a hard one included; a plain out has no <c>_data</c>.</summary>
        [TestMethod]
        public void MaDot_Out()
        {
            var a = np.ma.array(np.arange(6.0).reshape(2, 3), mask: new[,] { { false, true, false }, { false, false, true } });
            var b = np.ma.array(np.arange(6.0).reshape(3, 2), mask: new[,] { { true, false }, { false, false }, { false, false } });
            var soft = np.ma.zeros((2, 2));
            ReferenceEquals(np.ma.dot(a, b, @out: soft), soft).Should().BeTrue();
            Data(soft).Should().Equal(8.0, 10.0, 8.0, 15.0);
            IsNoMask(soft).Should().BeFalse();
            Mask(soft).Should().OnlyContain(m => !m);
            var premaskedHard = Premasked((2, 2), hard: true);
            var maskObject = np.ma.getmask(premaskedHard);
            np.ma.dot(a, b, @out: premaskedHard);
            Mask(premaskedHard).Should().OnlyContain(m => !m, "the mask is written in place, hardness ignored");
            ReferenceEquals(np.ma.getmask(premaskedHard), maskObject).Should().BeTrue();
            var strict = np.ma.zeros((2, 2));
            np.ma.dot(a, b, true, strict);
            Data(strict).Should().OnlyContain(d => d == 0.0);
            Mask(strict).Should().OnlyContain(m => m);
            var viaMethod = np.ma.zeros((2, 2));
            ReferenceEquals(a.dot(b, viaMethod), viaMethod).Should().BeTrue();
            Mask(a.dot(b, np.ma.zeros((2, 2)), strict: true)).Should().OnlyContain(m => m);

            Throws<AttributeError>(() => np.ma.dot(a, b, @out: np.zeros((2, 2))), "'numpy.ndarray' object has no attribute '_data'");
            Throws<ValueError>(() => np.ma.dot(a, b, @out: np.ma.zeros((2, 3))), "output array has wrong dimensions");
            Throws<ValueError>(() => np.ma.dot(a, b, @out: np.ma.zeros((2, 2), np.float32)),
                "output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)");
            var scalarOut = np.ma.zeros(Shape.Scalar);
            Throws<TypeError>(() => np.ma.dot(np.ma.array(np.array(new[] { 1.0, 2.0 }), mask: new[] { true, false }), np.array(new[] { 3.0, 4.0 }), @out: scalarOut),
                "'out' must be an array");
            np.ma.getdata(scalarOut).GetDouble().Should().Be(8.0);
        }

        /// <summary>ma.trace with out: a masked operand sums into out and returns it; an unmasked one writes out WITHOUT the
        ///     dtype and returns <c>out.astype(dtype)</c>, a copy.</summary>
        [TestMethod]
        public void MaTrace_Out()
        {
            var s33 = np.ma.array(np.arange(9.0).reshape(3, 3), mask: new[,] { { false, false, false }, { false, true, false }, { false, false, false } });
            var p = np.zeros(Shape.Scalar);
            ReferenceEquals(np.ma.trace(s33, @out: p), p).Should().BeTrue();
            p.GetDouble().Should().Be(8.0);
            ReferenceEquals(s33.trace(@out: p), p).Should().BeTrue();

            var n33 = np.ma.array(np.arange(9.0).reshape(3, 3));
            var q = np.zeros(Shape.Scalar, np.int64);
            var r = np.ma.trace(n33, @out: q);
            ReferenceEquals(r, q).Should().BeFalse();
            q.GetInt64().Should().Be(12L);
            r.dtype.Should().Be(np.float64);
            r.GetDouble().Should().Be(12.0);
            var f = np.zeros(Shape.Scalar);
            var r32 = np.ma.trace(n33, 0, 0, 1, np.int32, f);
            r32.dtype.Should().Be(np.int32);
            f.GetDouble().Should().Be(12.0);
            Throws<ValueError>(() => np.ma.trace(s33, @out: np.zeros(2)),
                "output parameter for reduction operation add has the wrong number of dimensions: Found 1 but expected 0");
        }

        // ---- fix 14: make_mask(m, copy, shrink, dtype) / make_mask_none(newshape, dtype) / mask_or(m1, m2, copy, shrink) --

        /// <summary>NumPy's copy slot (False hands a boolean input back itself), shrink, the dtype keyword (always bool), and
        ///     mask_or's identities.</summary>
        [TestMethod]
        public void MaskHelpers_CopyShrinkDtype()
        {
            var bm = np.array(new[] { true, false });
            np.ma.make_mask(new[] { 0, 1 }).ToArray<bool>().Should().Equal(false, true);
            ReferenceEquals(np.ma.make_mask(bm), bm).Should().BeTrue();
            ReferenceEquals(np.ma.make_mask(bm, false), bm).Should().BeTrue("the second positional is copy");
            ReferenceEquals(np.ma.make_mask(bm, copy: true), bm).Should().BeFalse();
            ReferenceEquals(np.ma.make_mask(new[] { 0, 0 }), np.ma.nomask).Should().BeTrue();
            ReferenceEquals(np.ma.make_mask(new[] { 0, 0 }, false), np.ma.nomask).Should().BeTrue("shrink stays True");
            np.ma.make_mask(new[] { 0, 0 }, shrink: false).ToArray<bool>().Should().Equal(false, false);
            np.ma.make_mask(new[] { 0, 1 }, dtype: np.int32).dtype.Should().Be(np.@bool);
            np.ma.make_mask(new[] { 0, 1 }, false, true, np.@bool).ToArray<bool>().Should().Equal(false, true);
            // An unmasked masked array hands back a view of its data (NumPy's filled returns self._data).
            var unmasked = np.ma.array(np.array(new[] { true, false }));
            var made = np.ma.make_mask(unmasked);
            np.shares_memory(made, np.ma.getdata(unmasked)).Should().BeTrue();
            made.flags.owndata.Should().BeFalse();

            var none = np.ma.make_mask_none(2, np.int32);
            none.dtype.Should().Be(np.@bool);
            none.ToArray<bool>().Should().Equal(false, false);

            Throws<ValueError>(() => np.ma.mask_or(bm, new object[] { 0, 1 }), "Incompatible dtypes 'bool'<>'None'");
            ReferenceEquals(np.ma.mask_or(bm, np.ma.nomask), bm).Should().BeTrue();
            ReferenceEquals(np.ma.mask_or(np.ma.nomask, bm), bm).Should().BeTrue();
            ReferenceEquals(np.ma.mask_or(bm, np.ma.nomask, copy: true), bm).Should().BeFalse();
            ReferenceEquals(np.ma.mask_or(bm, bm, copy: true), bm).Should().BeTrue("m1 is m2 ignores copy");
            np.ma.mask_or(bm, bm, false, false).ToArray<bool>().Should().Equal(true, false);
        }

        // ---- fix 15: ma.where(condition) / atleast_*d(*arys) / MaskedArray.item(*args) ---------------------------------

        /// <summary>The one-argument ma.where is nonzero over the filled condition; atleast_*d of several operands is a
        ///     list; item reads the stored data (ndarray.item), a masked element included.</summary>
        [TestMethod]
        public void MaWhere_Atleast_Item()
        {
            var cond = np.ma.array(np.array(new long[] { 1, 0, 1 }), mask: new[] { false, false, true });
            var w = np.ma.where(cond);
            w.Length.Should().Be(1);
            w[0].ToArray<long>().Should().Equal(0L);
            np.ma.where(np.array(new long[] { 1, 0, 1 }))[0].ToArray<long>().Should().Equal(0L, 2L);

            var m12 = np.ma.array(np.array(new long[] { 1, 2 }), mask: new[] { false, true });
            var pair = np.ma.atleast_1d(m12, 5);
            pair.Length.Should().Be(2);
            Mask(pair[0]).Should().Equal(false, true);
            Data(pair[1]).Should().Equal(5.0);
            IsNoMask(pair[1]).Should().BeFalse("_fromnxfunction gives a full mask");
            np.ma.atleast_2d(m12, np.array(new long[] { 3, 4 })).Select(x => x.shape.Length).Should().Equal(2, 2);
            np.ma.atleast_3d(5, np.arange(2), np.ones((2, 2))).Select(x => string.Join(",", x.shape)).Should().Equal("1,1,1", "1,2,1", "2,2,1");

            var m23 = M23();
            Convert.ToDouble(m23.item(1)).Should().Be(1.0, "element 1 is masked: the stored value");
            Convert.ToDouble(m23.item(0, 1)).Should().Be(1.0);
            Convert.ToDouble(m23.item(1, 2)).Should().Be(5.0);
            Convert.ToDouble(m23.item(4)).Should().Be(4.0);
            Convert.ToDouble(m23.item(-1)).Should().Be(5.0);
            np.ma.array(np.array(new long[] { 5 }), mask: new[] { true }).item().Should().Be(5L);
            Throws<ValueError>(() => m23.item(), "can only convert an array of size 1 to a Python scalar");
            Throws<IndexOutOfRangeException>(() => m23.item(9), "index 9 is out of bounds for size 6");
        }

        /// <summary>A lone <c>object[]</c> is C#'s params array, so <c>np.ma.atleast_1d(new object[] { 1, 2 })</c> gives two
        ///     arrays where NumPy's <c>np.ma.atleast_1d([1, 2])</c> reads the list as ONE array <c>[1, 2]</c>.</summary>
        [TestMethod, Misaligned]
        public void MaAtleast1d_LoneObjectArray_IsTheOperandList()
        {
            var r = np.ma.atleast_1d(new object[] { 1, 2 });
            r.Length.Should().Be(2);
            Data(r[0]).Should().Equal(1.0);
            Data(r[1]).Should().Equal(2.0);
        }
    }
}
